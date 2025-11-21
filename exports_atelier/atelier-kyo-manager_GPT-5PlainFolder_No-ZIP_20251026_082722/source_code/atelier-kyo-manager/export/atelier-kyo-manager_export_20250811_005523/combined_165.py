
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chartsheet\chartsheet.py ===
# Copyright (c) 2010-2024 openpyxl


from openpyxl.descriptors import Typed, Set, Alias
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.drawing.spreadsheet_drawing import (
    AbsoluteAnchor,
    SpreadsheetDrawing,
)
from openpyxl.worksheet.page import (
    PageMargins,
    PrintPageSetup
)
from openpyxl.worksheet.drawing import Drawing
from openpyxl.worksheet.header_footer import HeaderFooter
from openpyxl.workbook.child import _WorkbookChild
from openpyxl.xml.constants import SHEET_MAIN_NS, REL_NS

from .relation import DrawingHF, SheetBackgroundPicture
from .properties import ChartsheetProperties
from .protection import ChartsheetProtection
from .views import ChartsheetViewList
from .custom import CustomChartsheetViews
from .publish import WebPublishItems


class Chartsheet(_WorkbookChild, Serialisable):

    tagname = "chartsheet"
    _default_title = "Chart"
    _rel_type = "chartsheet"
    _path = "/xl/chartsheets/sheet{0}.xml"
    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.chartsheet+xml"

    sheetPr = Typed(expected_type=ChartsheetProperties, allow_none=True)
    sheetViews = Typed(expected_type=ChartsheetViewList)
    sheetProtection = Typed(expected_type=ChartsheetProtection, allow_none=True)
    customSheetViews = Typed(expected_type=CustomChartsheetViews, allow_none=True)
    pageMargins = Typed(expected_type=PageMargins, allow_none=True)
    pageSetup = Typed(expected_type=PrintPageSetup, allow_none=True)
    drawing = Typed(expected_type=Drawing, allow_none=True)
    drawingHF = Typed(expected_type=DrawingHF, allow_none=True)
    picture = Typed(expected_type=SheetBackgroundPicture, allow_none=True)
    webPublishItems = Typed(expected_type=WebPublishItems, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)
    sheet_state = Set(values=('visible', 'hidden', 'veryHidden'))
    headerFooter = Typed(expected_type=HeaderFooter)
    HeaderFooter = Alias('headerFooter')

    __elements__ = (
        'sheetPr', 'sheetViews', 'sheetProtection', 'customSheetViews',
        'pageMargins', 'pageSetup', 'headerFooter', 'drawing', 'drawingHF',
        'picture', 'webPublishItems')

    __attrs__ = ()

    def __init__(self,
                 sheetPr=None,
                 sheetViews=None,
                 sheetProtection=None,
                 customSheetViews=None,
                 pageMargins=None,
                 pageSetup=None,
                 headerFooter=None,
                 drawing=None,
                 drawingHF=None,
                 picture=None,
                 webPublishItems=None,
                 extLst=None,
                 parent=None,
                 title="",
                 sheet_state='visible',
                 ):
        super().__init__(parent, title)
        self._charts = []
        self.sheetPr = sheetPr
        if sheetViews is None:
            sheetViews = ChartsheetViewList()
        self.sheetViews = sheetViews
        self.sheetProtection = sheetProtection
        self.customSheetViews = customSheetViews
        self.pageMargins = pageMargins
        self.pageSetup = pageSetup
        if headerFooter is not None:
            self.headerFooter = headerFooter
        self.drawing = Drawing("rId1")
        self.drawingHF = drawingHF
        self.picture = picture
        self.webPublishItems = webPublishItems
        self.sheet_state = sheet_state


    def add_chart(self, chart):
        chart.anchor = AbsoluteAnchor()
        self._charts.append(chart)


    def to_tree(self):
        self._drawing = SpreadsheetDrawing()
        self._drawing.charts = self._charts
        tree = super().to_tree()
        if not self.headerFooter:
            el = tree.find('headerFooter')
            tree.remove(el)
        tree.set("xmlns", SHEET_MAIN_NS)
        return tree

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\controls.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Bool,
    Integer,
    String,
    Sequence,
)

from openpyxl.descriptors.excel import Relation
from .ole import ObjectAnchor


class ControlProperty(Serialisable):

    tagname = "controlPr"

    anchor = Typed(expected_type=ObjectAnchor, )
    locked = Bool(allow_none=True)
    defaultSize = Bool(allow_none=True)
    _print = Bool(allow_none=True)
    disabled = Bool(allow_none=True)
    recalcAlways = Bool(allow_none=True)
    uiObject = Bool(allow_none=True)
    autoFill = Bool(allow_none=True)
    autoLine = Bool(allow_none=True)
    autoPict = Bool(allow_none=True)
    macro = String(allow_none=True)
    altText = String(allow_none=True)
    linkedCell = String(allow_none=True)
    listFillRange = String(allow_none=True)
    cf = String(allow_none=True)
    id = Relation(allow_none=True)

    __elements__ = ('anchor',)

    def __init__(self,
                 anchor=None,
                 locked=True,
                 defaultSize=True,
                 _print=True,
                 disabled=False,
                 recalcAlways=False,
                 uiObject=False,
                 autoFill=True,
                 autoLine=True,
                 autoPict=True,
                 macro=None,
                 altText=None,
                 linkedCell=None,
                 listFillRange=None,
                 cf='pict',
                 id=None,
                ):
        self.anchor = anchor
        self.locked = locked
        self.defaultSize = defaultSize
        self._print = _print
        self.disabled = disabled
        self.recalcAlways = recalcAlways
        self.uiObject = uiObject
        self.autoFill = autoFill
        self.autoLine = autoLine
        self.autoPict = autoPict
        self.macro = macro
        self.altText = altText
        self.linkedCell = linkedCell
        self.listFillRange = listFillRange
        self.cf = cf
        self.id = id


class Control(Serialisable):

    tagname = "control"

    controlPr = Typed(expected_type=ControlProperty, allow_none=True)
    shapeId = Integer()
    name = String(allow_none=True)

    __elements__ = ('controlPr',)

    def __init__(self,
                 controlPr=None,
                 shapeId=None,
                 name=None,
                ):
        self.controlPr = controlPr
        self.shapeId = shapeId
        self.name = name


class Controls(Serialisable):

    tagname = "controls"

    control = Sequence(expected_type=Control)

    __elements__ = ('control',)

    def __init__(self,
                 control=(),
                ):
        self.control = control


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\response.py ===
from __future__ import absolute_import

from email.errors import MultipartInvariantViolationDefect, StartBoundaryNotFoundDefect

from ..exceptions import HeaderParsingError
from ..packages.six.moves import http_client as httplib


def is_fp_closed(obj):
    """
    Checks whether a given file-like object is closed.

    :param obj:
        The file-like object to check.
    """

    try:
        # Check `isclosed()` first, in case Python3 doesn't set `closed`.
        # GH Issue #928
        return obj.isclosed()
    except AttributeError:
        pass

    try:
        # Check via the official file-like-object way.
        return obj.closed
    except AttributeError:
        pass

    try:
        # Check if the object is a container for another file-like object that
        # gets released on exhaustion (e.g. HTTPResponse).
        return obj.fp is None
    except AttributeError:
        pass

    raise ValueError("Unable to determine whether fp is closed.")


def assert_header_parsing(headers):
    """
    Asserts whether all headers have been successfully parsed.
    Extracts encountered errors from the result of parsing headers.

    Only works on Python 3.

    :param http.client.HTTPMessage headers: Headers to verify.

    :raises urllib3.exceptions.HeaderParsingError:
        If parsing errors are found.
    """

    # This will fail silently if we pass in the wrong kind of parameter.
    # To make debugging easier add an explicit check.
    if not isinstance(headers, httplib.HTTPMessage):
        raise TypeError("expected httplib.Message, got {0}.".format(type(headers)))

    defects = getattr(headers, "defects", None)
    get_payload = getattr(headers, "get_payload", None)

    unparsed_data = None
    if get_payload:
        # get_payload is actually email.message.Message.get_payload;
        # we're only interested in the result if it's not a multipart message
        if not headers.is_multipart():
            payload = get_payload()

            if isinstance(payload, (bytes, str)):
                unparsed_data = payload
    if defects:
        # httplib is assuming a response body is available
        # when parsing headers even when httplib only sends
        # header data to parse_headers() This results in
        # defects on multipart responses in particular.
        # See: https://github.com/urllib3/urllib3/issues/800

        # So we ignore the following defects:
        # - StartBoundaryNotFoundDefect:
        #     The claimed start boundary was never found.
        # - MultipartInvariantViolationDefect:
        #     A message claimed to be a multipart but no subparts were found.
        defects = [
            defect
            for defect in defects
            if not isinstance(
                defect, (StartBoundaryNotFoundDefect, MultipartInvariantViolationDefect)
            )
        ]

    if defects or unparsed_data:
        raise HeaderParsingError(defects=defects, unparsed_data=unparsed_data)


def is_response_to_head(response):
    """
    Checks whether the request of a response has been a HEAD-request.
    Handles the quirks of AppEngine.

    :param http.client.HTTPResponse response:
        Response to check if the originating request
        used 'HEAD' as a method.
    """
    # FIXME: Can we do this somehow without accessing private httplib _method?
    method = response._method
    if isinstance(method, int):  # Platform-specific: Appengine
        return method == 3
    return method.upper() == "HEAD"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\handlers\power.py ===
from sympy.core import Basic, Expr
from sympy.core.function import Lambda
from sympy.core.numbers import oo, Infinity, NegativeInfinity, Zero, Integer
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import (Max, Min)
from sympy.sets.fancysets import ImageSet
from sympy.sets.setexpr import set_div
from sympy.sets.sets import Set, Interval, FiniteSet, Union
from sympy.multipledispatch import Dispatcher


_x, _y = symbols("x y")


_set_pow = Dispatcher('_set_pow')


@_set_pow.register(Basic, Basic)
def _(x, y):
    return None

@_set_pow.register(Set, Set)
def _(x, y):
    return ImageSet(Lambda((_x, _y), (_x ** _y)), x, y)

@_set_pow.register(Expr, Expr)
def _(x, y):
    return x**y

@_set_pow.register(Interval, Zero)
def _(x, z):
    return FiniteSet(S.One)

@_set_pow.register(Interval, Integer)
def _(x, exponent):
    """
    Powers in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    s1 = x.start**exponent
    s2 = x.end**exponent
    if ((s2 > s1) if exponent > 0 else (x.end > -x.start)) == True:
        left_open = x.left_open
        right_open = x.right_open
        # TODO: handle unevaluated condition.
        sleft = s2
    else:
        # TODO: `s2 > s1` could be unevaluated.
        left_open = x.right_open
        right_open = x.left_open
        sleft = s1

    if x.start.is_positive:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)
    elif x.end.is_negative:
        return Interval(
            Min(s1, s2),
            Max(s1, s2), left_open, right_open)

    # Case where x.start < 0 and x.end > 0:
    if exponent.is_odd:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(-oo, s1, True, x.left_open)
            return Union(Interval(-oo, s1, True, x.left_open), Interval(s2, oo, x.right_open))
        else:
            return Interval(s1, s2, x.left_open, x.right_open)
    elif exponent.is_even:
        if exponent.is_negative:
            if x.start.is_zero:
                return Interval(s2, oo, x.right_open)
            if x.end.is_zero:
                return Interval(s1, oo, x.left_open)
            return Interval(0, oo)
        else:
            return Interval(S.Zero, sleft, S.Zero not in x, left_open)

@_set_pow.register(Interval, Infinity)
def _(b, e):
    # TODO: add logic for open intervals?
    if b.start.is_nonnegative:
        if b.end < 1:
            return FiniteSet(S.Zero)
        if b.start > 1:
            return FiniteSet(S.Infinity)
        return Interval(0, oo)
    elif b.end.is_negative:
        if b.start > -1:
            return FiniteSet(S.Zero)
        if b.end < -1:
            return FiniteSet(-oo, oo)
        return Interval(-oo, oo)
    else:
        if b.start > -1:
            if b.end < 1:
                return FiniteSet(S.Zero)
            return Interval(0, oo)
        return Interval(-oo, oo)

@_set_pow.register(Interval, NegativeInfinity)
def _(b, e):
    return _set_pow(set_div(S.One, b), oo)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\cli.py ===
# Command line interface for the coloredlogs package.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: December 15, 2017
# URL: https://coloredlogs.readthedocs.io

"""
Usage: coloredlogs [OPTIONS] [ARGS]

The coloredlogs program provides a simple command line interface for the Python
package by the same name.

Supported options:

  -c, --convert, --to-html

    Capture the output of an external command (given by the positional
    arguments) and convert ANSI escape sequences in the output to HTML.

    If the `coloredlogs' program is attached to an interactive terminal it will
    write the generated HTML to a temporary file and open that file in a web
    browser, otherwise the generated HTML will be written to standard output.

    This requires the `script' program to fake the external command into
    thinking that it's attached to an interactive terminal (in order to enable
    output of ANSI escape sequences).

    If the command didn't produce any output then no HTML will be produced on
    standard output, this is to avoid empty emails from cron jobs.

  -d, --demo

    Perform a simple demonstration of the coloredlogs package to show the
    colored logging on an interactive terminal.

  -h, --help

    Show this message and exit.
"""

# Standard library modules.
import functools
import getopt
import logging
import sys
import tempfile
import webbrowser

# External dependencies.
from humanfriendly.terminal import connected_to_terminal, output, usage, warning

# Modules included in our package.
from coloredlogs.converter import capture, convert
from coloredlogs.demo import demonstrate_colored_logging

# Initialize a logger for this module.
logger = logging.getLogger(__name__)


def main():
    """Command line interface for the ``coloredlogs`` program."""
    actions = []
    try:
        # Parse the command line arguments.
        options, arguments = getopt.getopt(sys.argv[1:], 'cdh', [
            'convert', 'to-html', 'demo', 'help',
        ])
        # Map command line options to actions.
        for option, value in options:
            if option in ('-c', '--convert', '--to-html'):
                actions.append(functools.partial(convert_command_output, *arguments))
                arguments = []
            elif option in ('-d', '--demo'):
                actions.append(demonstrate_colored_logging)
            elif option in ('-h', '--help'):
                usage(__doc__)
                return
            else:
                assert False, "Programming error: Unhandled option!"
        if not actions:
            usage(__doc__)
            return
    except Exception as e:
        warning("Error: %s", e)
        sys.exit(1)
    for function in actions:
        function()


def convert_command_output(*command):
    """
    Command line interface for ``coloredlogs --to-html``.

    Takes a command (and its arguments) and runs the program under ``script``
    (emulating an interactive terminal), intercepts the output of the command
    and converts ANSI escape sequences in the output to HTML.
    """
    captured_output = capture(command)
    converted_output = convert(captured_output)
    if connected_to_terminal():
        fd, temporary_file = tempfile.mkstemp(suffix='.html')
        with open(temporary_file, 'w') as handle:
            handle.write(converted_output)
        webbrowser.open(temporary_file)
    elif captured_output and not captured_output.isspace():
        output(converted_output)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\text_encoding.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Encoding related utilities."""
import re

def _AsciiIsPrint(i):
  return i >= 32 and i < 127

def _MakeStrEscapes():
  ret = {}
  for i in range(0, 128):
    if not _AsciiIsPrint(i):
      ret[i] = r'\%03o' % i
  ret[ord('\t')] = r'\t'  # optional escape
  ret[ord('\n')] = r'\n'  # optional escape
  ret[ord('\r')] = r'\r'  # optional escape
  ret[ord('"')] = r'\"'  # necessary escape
  ret[ord('\'')] = r"\'"  # optional escape
  ret[ord('\\')] = r'\\'  # necessary escape
  return ret

# Maps int -> char, performing string escapes.
_str_escapes = _MakeStrEscapes()

# Maps int -> char, performing byte escaping and string escapes
_byte_escapes = {i: chr(i) for i in range(0, 256)}
_byte_escapes.update(_str_escapes)
_byte_escapes.update({i: r'\%03o' % i for i in range(128, 256)})


def _DecodeUtf8EscapeErrors(text_bytes):
  ret = ''
  while text_bytes:
    try:
      ret += text_bytes.decode('utf-8').translate(_str_escapes)
      text_bytes = ''
    except UnicodeDecodeError as e:
      ret += text_bytes[:e.start].decode('utf-8').translate(_str_escapes)
      ret += _byte_escapes[text_bytes[e.start]]
      text_bytes = text_bytes[e.start+1:]
  return ret


def CEscape(text, as_utf8) -> str:
  """Escape a bytes string for use in an text protocol buffer.

  Args:
    text: A byte string to be escaped.
    as_utf8: Specifies if result may contain non-ASCII characters.
        In Python 3 this allows unescaped non-ASCII Unicode characters.
        In Python 2 the return value will be valid UTF-8 rather than only ASCII.
  Returns:
    Escaped string (str).
  """
  # Python's text.encode() 'string_escape' or 'unicode_escape' codecs do not
  # satisfy our needs; they encodes unprintable characters using two-digit hex
  # escapes whereas our C++ unescaping function allows hex escapes to be any
  # length.  So, "\0011".encode('string_escape') ends up being "\\x011", which
  # will be decoded in C++ as a single-character string with char code 0x11.
  text_is_unicode = isinstance(text, str)
  if as_utf8:
    if text_is_unicode:
      return text.translate(_str_escapes)
    else:
      return _DecodeUtf8EscapeErrors(text)
  else:
    if text_is_unicode:
      text = text.encode('utf-8')
    return ''.join([_byte_escapes[c] for c in text])


_CUNESCAPE_HEX = re.compile(r'(\\+)x([0-9a-fA-F])(?![0-9a-fA-F])')


def CUnescape(text: str) -> bytes:
  """Unescape a text string with C-style escape sequences to UTF-8 bytes.

  Args:
    text: The data to parse in a str.
  Returns:
    A byte string.
  """

  def ReplaceHex(m):
    # Only replace the match if the number of leading back slashes is odd. i.e.
    # the slash itself is not escaped.
    if len(m.group(1)) & 1:
      return m.group(1) + 'x0' + m.group(2)
    return m.group(0)

  # This is required because the 'string_escape' encoding doesn't
  # allow single-digit hex escapes (like '\xf').
  result = _CUNESCAPE_HEX.sub(ReplaceHex, text)

  # Replaces Unicode escape sequences with their character equivalents.
  result = result.encode('raw_unicode_escape').decode('raw_unicode_escape')
  # Encode Unicode characters as UTF-8, then decode to Latin-1 escaping
  # unprintable characters.
  result = result.encode('utf-8').decode('unicode_escape')
  # Convert Latin-1 text back to a byte string (latin-1 codec also works here).
  return result.encode('latin-1')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\itsdangerous\exc.py ===
from __future__ import annotations

import typing as t
from datetime import datetime


class BadData(Exception):
    """Raised if bad data of any sort was encountered. This is the base
    for all exceptions that ItsDangerous defines.

    .. versionadded:: 0.15
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class BadSignature(BadData):
    """Raised if a signature does not match."""

    def __init__(self, message: str, payload: t.Any | None = None):
        super().__init__(message)

        #: The payload that failed the signature test. In some
        #: situations you might still want to inspect this, even if
        #: you know it was tampered with.
        #:
        #: .. versionadded:: 0.14
        self.payload: t.Any | None = payload


class BadTimeSignature(BadSignature):
    """Raised if a time-based signature is invalid. This is a subclass
    of :class:`BadSignature`.
    """

    def __init__(
        self,
        message: str,
        payload: t.Any | None = None,
        date_signed: datetime | None = None,
    ):
        super().__init__(message, payload)

        #: If the signature expired this exposes the date of when the
        #: signature was created. This can be helpful in order to
        #: tell the user how long a link has been gone stale.
        #:
        #: .. versionchanged:: 2.0
        #:     The datetime value is timezone-aware rather than naive.
        #:
        #: .. versionadded:: 0.14
        self.date_signed = date_signed


class SignatureExpired(BadTimeSignature):
    """Raised if a signature timestamp is older than ``max_age``. This
    is a subclass of :exc:`BadTimeSignature`.
    """


class BadHeader(BadSignature):
    """Raised if a signed header is invalid in some form. This only
    happens for serializers that have a header that goes with the
    signature.

    .. versionadded:: 0.24
    """

    def __init__(
        self,
        message: str,
        payload: t.Any | None = None,
        header: t.Any | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, payload)

        #: If the header is actually available but just malformed it
        #: might be stored here.
        self.header: t.Any | None = header

        #: If available, the error that indicates why the payload was
        #: not valid. This might be ``None``.
        self.original_error: Exception | None = original_error


class BadPayload(BadData):
    """Raised if a payload is invalid. This could happen if the payload
    is loaded despite an invalid signature, or if there is a mismatch
    between the serializer and deserializer. The original exception
    that occurred during loading is stored on as :attr:`original_error`.

    .. versionadded:: 0.15
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)

        #: If available, the error that indicates why the payload was
        #: not valid. This might be ``None``.
        self.original_error: Exception | None = original_error

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_array_like.py ===
import sys
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
from numpy import dtype

from ._nbit_base import _32Bit, _64Bit
from ._nested_sequence import _NestedSequence
from ._shape import _AnyShape

if TYPE_CHECKING:
    StringDType = np.dtypes.StringDType
else:
    # at runtime outside of type checking importing this from numpy.dtypes
    # would lead to a circular import
    from numpy._core.multiarray import StringDType

_T = TypeVar("_T")
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_DTypeT = TypeVar("_DTypeT", bound=dtype[Any])
_DTypeT_co = TypeVar("_DTypeT_co", covariant=True, bound=dtype[Any])

NDArray: TypeAlias = np.ndarray[_AnyShape, dtype[_ScalarT]]

# The `_SupportsArray` protocol only cares about the default dtype
# (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
# array.
# Concrete implementations of the protocol are responsible for adding
# any and all remaining overloads
@runtime_checkable
class _SupportsArray(Protocol[_DTypeT_co]):
    def __array__(self) -> np.ndarray[Any, _DTypeT_co]: ...


@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Collection[type[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object: ...


# TODO: Wait until mypy supports recursive objects in combination with typevars
_FiniteNestedSequence: TypeAlias = (
    _T
    | Sequence[_T]
    | Sequence[Sequence[_T]]
    | Sequence[Sequence[Sequence[_T]]]
    | Sequence[Sequence[Sequence[Sequence[_T]]]]
)

# A subset of `npt.ArrayLike` that can be parametrized w.r.t. `np.generic`
_ArrayLike: TypeAlias = (
    _SupportsArray[dtype[_ScalarT]]
    | _NestedSequence[_SupportsArray[dtype[_ScalarT]]]
)

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
_DualArrayLike: TypeAlias = (
    _SupportsArray[_DTypeT]
    | _NestedSequence[_SupportsArray[_DTypeT]]
    | _T
    | _NestedSequence[_T]
)

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _Buffer
else:
    @runtime_checkable
    class _Buffer(Protocol):
        def __buffer__(self, flags: int, /) -> memoryview: ...

ArrayLike: TypeAlias = _Buffer | _DualArrayLike[dtype[Any], complex | bytes | str]

# `ArrayLike<X>_co`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
_ArrayLikeBool_co: TypeAlias = _DualArrayLike[dtype[np.bool], bool]
_ArrayLikeUInt_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.unsignedinteger], bool]
_ArrayLikeInt_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.integer], int]
_ArrayLikeFloat_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.integer | np.floating], float]
_ArrayLikeComplex_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.number], complex]
_ArrayLikeNumber_co: TypeAlias = _ArrayLikeComplex_co
_ArrayLikeTD64_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.integer | np.timedelta64], int]
_ArrayLikeDT64_co: TypeAlias = _ArrayLike[np.datetime64]
_ArrayLikeObject_co: TypeAlias = _ArrayLike[np.object_]

_ArrayLikeVoid_co: TypeAlias = _ArrayLike[np.void]
_ArrayLikeBytes_co: TypeAlias = _DualArrayLike[dtype[np.bytes_], bytes]
_ArrayLikeStr_co: TypeAlias = _DualArrayLike[dtype[np.str_], str]
_ArrayLikeString_co: TypeAlias = _DualArrayLike[StringDType, str]
_ArrayLikeAnyString_co: TypeAlias = _DualArrayLike[dtype[np.character] | StringDType, bytes | str]

__Float64_co: TypeAlias = np.floating[_64Bit] | np.float32 | np.float16 | np.integer | np.bool
__Complex128_co: TypeAlias = np.number[_64Bit] | np.number[_32Bit] | np.float16 | np.integer | np.bool
_ArrayLikeFloat64_co: TypeAlias = _DualArrayLike[dtype[__Float64_co], float]
_ArrayLikeComplex128_co: TypeAlias = _DualArrayLike[dtype[__Complex128_co], complex]

# NOTE: This includes `builtins.bool`, but not `numpy.bool`.
_ArrayLikeInt: TypeAlias = _DualArrayLike[dtype[np.integer], int]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\area_chart.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Set,
    Bool,
    Integer,
    Sequence,
    Alias,
)

from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.nested import (
    NestedMinMax,
    NestedSet,
    NestedBool,
)

from ._chart import ChartBase
from .descriptors import NestedGapAmount
from .axis import TextAxis, NumericAxis, SeriesAxis, ChartLines
from .label import DataLabelList
from .series import Series


class _AreaChartBase(ChartBase):

    grouping = NestedSet(values=(['percentStacked', 'standard', 'stacked']))
    varyColors = NestedBool(nested=True, allow_none=True)
    ser = Sequence(expected_type=Series, allow_none=True)
    dLbls = Typed(expected_type=DataLabelList, allow_none=True)
    dataLabels = Alias("dLbls")
    dropLines = Typed(expected_type=ChartLines, allow_none=True)

    _series_type = "area"

    __elements__ = ('grouping', 'varyColors', 'ser', 'dLbls', 'dropLines')

    def __init__(self,
                 grouping="standard",
                 varyColors=None,
                 ser=(),
                 dLbls=None,
                 dropLines=None,
                ):
        self.grouping = grouping
        self.varyColors = varyColors
        self.ser = ser
        self.dLbls = dLbls
        self.dropLines = dropLines
        super().__init__()


class AreaChart(_AreaChartBase):

    tagname = "areaChart"

    grouping = _AreaChartBase.grouping
    varyColors = _AreaChartBase.varyColors
    ser = _AreaChartBase.ser
    dLbls = _AreaChartBase.dLbls
    dropLines = _AreaChartBase.dropLines

    # chart properties actually used by containing classes
    x_axis = Typed(expected_type=TextAxis)
    y_axis = Typed(expected_type=NumericAxis)

    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _AreaChartBase.__elements__ + ('axId',)

    def __init__(self,
                 axId=None,
                 extLst=None,
                 **kw
                ):
        self.x_axis = TextAxis()
        self.y_axis = NumericAxis()
        super().__init__(**kw)


class AreaChart3D(AreaChart):

    tagname = "area3DChart"

    grouping = _AreaChartBase.grouping
    varyColors = _AreaChartBase.varyColors
    ser = _AreaChartBase.ser
    dLbls = _AreaChartBase.dLbls
    dropLines = _AreaChartBase.dropLines

    gapDepth = NestedGapAmount()

    x_axis = Typed(expected_type=TextAxis)
    y_axis = Typed(expected_type=NumericAxis)
    z_axis = Typed(expected_type=SeriesAxis, allow_none=True)

    __elements__ = AreaChart.__elements__ + ('gapDepth', )

    def __init__(self, gapDepth=None, **kw):
        self.gapDepth = gapDepth
        super(AreaChart3D, self).__init__(**kw)
        self.x_axis = TextAxis()
        self.y_axis = NumericAxis()
        self.z_axis = SeriesAxis()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\compat.py ===
"""
requests.compat
~~~~~~~~~~~~~~~

This module previously handled import compatibility issues
between Python 2 and Python 3. It remains for backwards
compatibility until the next major version.
"""

import importlib
import sys

# -------
# urllib3
# -------
from urllib3 import __version__ as urllib3_version

# Detect which major version of urllib3 is being used.
try:
    is_urllib3_1 = int(urllib3_version.split(".")[0]) == 1
except (TypeError, AttributeError):
    # If we can't discern a version, prefer old functionality.
    is_urllib3_1 = True

# -------------------
# Character Detection
# -------------------


def _resolve_char_detection():
    """Find supported character detection libraries."""
    chardet = None
    for lib in ("chardet", "charset_normalizer"):
        if chardet is None:
            try:
                chardet = importlib.import_module(lib)
            except ImportError:
                pass
    return chardet


chardet = _resolve_char_detection()

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = _ver[0] == 2

#: Python 3.x?
is_py3 = _ver[0] == 3

# json/simplejson module import resolution
has_simplejson = False
try:
    import simplejson as json

    has_simplejson = True
except ImportError:
    import json

if has_simplejson:
    from simplejson import JSONDecodeError
else:
    from json import JSONDecodeError

# Keep OrderedDict for backwards compatibility.
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from http import cookiejar as cookielib
from http.cookies import Morsel
from io import StringIO

# --------------
# Legacy Imports
# --------------
from urllib.parse import (
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urldefrag,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunparse,
)
from urllib.request import (
    getproxies,
    getproxies_environment,
    parse_http_list,
    proxy_bypass,
    proxy_bypass_environment,
)

builtin_str = str
str = str
bytes = bytes
basestring = (str, bytes)
numeric_types = (int, float)
integer_types = (int,)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\managed_window.py ===
from pyglet.window import Window
from pyglet.clock import Clock

from threading import Thread, Lock

gl_lock = Lock()


class ManagedWindow(Window):
    """
    A pyglet window with an event loop which executes automatically
    in a separate thread. Behavior is added by creating a subclass
    which overrides setup, update, and/or draw.
    """
    fps_limit = 30
    default_win_args = {"width": 600,
                            "height": 500,
                            "vsync": False,
                            "resizable": True}

    def __init__(self, **win_args):
        """
        It is best not to override this function in the child
        class, unless you need to take additional arguments.
        Do any OpenGL initialization calls in setup().
        """

        # check if this is run from the doctester
        if win_args.get('runfromdoctester', False):
            return

        self.win_args = dict(self.default_win_args, **win_args)
        self.Thread = Thread(target=self.__event_loop__)
        self.Thread.start()

    def __event_loop__(self, **win_args):
        """
        The event loop thread function. Do not override or call
        directly (it is called by __init__).
        """
        gl_lock.acquire()
        try:
            try:
                super().__init__(**self.win_args)
                self.switch_to()
                self.setup()
            except Exception as e:
                print("Window initialization failed: %s" % (str(e)))
                self.has_exit = True
        finally:
            gl_lock.release()

        clock = Clock()
        clock.fps_limit = self.fps_limit
        while not self.has_exit:
            dt = clock.tick()
            gl_lock.acquire()
            try:
                try:
                    self.switch_to()
                    self.dispatch_events()
                    self.clear()
                    self.update(dt)
                    self.draw()
                    self.flip()
                except Exception as e:
                    print("Uncaught exception in event loop: %s" % str(e))
                    self.has_exit = True
            finally:
                gl_lock.release()
        super().close()

    def close(self):
        """
        Closes the window.
        """
        self.has_exit = True

    def setup(self):
        """
        Called once before the event loop begins.
        Override this method in a child class. This
        is the best place to put things like OpenGL
        initialization calls.
        """
        pass

    def update(self, dt):
        """
        Called before draw during each iteration of
        the event loop. dt is the elapsed time in
        seconds since the last update. OpenGL rendering
        calls are best put in draw() rather than here.
        """
        pass

    def draw(self):
        """
        Called after update during each iteration of
        the event loop. Put OpenGL rendering calls
        here.
        """
        pass

if __name__ == '__main__':
    ManagedWindow()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_logging.py ===
import logging

"""
_logging.py
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

_logger = logging.getLogger("websocket")
try:
    from logging import NullHandler
except ImportError:

    class NullHandler(logging.Handler):
        def emit(self, record) -> None:
            pass


_logger.addHandler(NullHandler())

_traceEnabled = False

__all__ = [
    "enableTrace",
    "dump",
    "error",
    "warning",
    "debug",
    "trace",
    "isEnabledForError",
    "isEnabledForDebug",
    "isEnabledForTrace",
]


def enableTrace(
    traceable: bool,
    handler: logging.StreamHandler = logging.StreamHandler(),
    level: str = "DEBUG",
) -> None:
    """
    Turn on/off the traceability.

    Parameters
    ----------
    traceable: bool
        If set to True, traceability is enabled.
    """
    global _traceEnabled
    _traceEnabled = traceable
    if traceable:
        _logger.addHandler(handler)
        _logger.setLevel(getattr(logging, level))


def dump(title: str, message: str) -> None:
    if _traceEnabled:
        _logger.debug(f"--- {title} ---")
        _logger.debug(message)
        _logger.debug("-----------------------")


def error(msg: str) -> None:
    _logger.error(msg)


def warning(msg: str) -> None:
    _logger.warning(msg)


def debug(msg: str) -> None:
    _logger.debug(msg)


def info(msg: str) -> None:
    _logger.info(msg)


def trace(msg: str) -> None:
    if _traceEnabled:
        _logger.debug(msg)


def isEnabledForError() -> bool:
    return _logger.isEnabledFor(logging.ERROR)


def isEnabledForDebug() -> bool:
    return _logger.isEnabledFor(logging.DEBUG)


def isEnabledForTrace() -> bool:
    return _traceEnabled

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\etag.py ===
from __future__ import annotations

import collections.abc as cabc


class ETags(cabc.Collection[str]):
    """A set that can be used to check if one etag is present in a collection
    of etags.
    """

    def __init__(
        self,
        strong_etags: cabc.Iterable[str] | None = None,
        weak_etags: cabc.Iterable[str] | None = None,
        star_tag: bool = False,
    ):
        if not star_tag and strong_etags:
            self._strong = frozenset(strong_etags)
        else:
            self._strong = frozenset()

        self._weak = frozenset(weak_etags or ())
        self.star_tag = star_tag

    def as_set(self, include_weak: bool = False) -> set[str]:
        """Convert the `ETags` object into a python set.  Per default all the
        weak etags are not part of this set."""
        rv = set(self._strong)
        if include_weak:
            rv.update(self._weak)
        return rv

    def is_weak(self, etag: str) -> bool:
        """Check if an etag is weak."""
        return etag in self._weak

    def is_strong(self, etag: str) -> bool:
        """Check if an etag is strong."""
        return etag in self._strong

    def contains_weak(self, etag: str) -> bool:
        """Check if an etag is part of the set including weak and strong tags."""
        return self.is_weak(etag) or self.contains(etag)

    def contains(self, etag: str) -> bool:
        """Check if an etag is part of the set ignoring weak tags.
        It is also possible to use the ``in`` operator.
        """
        if self.star_tag:
            return True
        return self.is_strong(etag)

    def contains_raw(self, etag: str) -> bool:
        """When passed a quoted tag it will check if this tag is part of the
        set.  If the tag is weak it is checked against weak and strong tags,
        otherwise strong only."""
        from ..http import unquote_etag

        etag, weak = unquote_etag(etag)
        if weak:
            return self.contains_weak(etag)
        return self.contains(etag)

    def to_header(self) -> str:
        """Convert the etags set into a HTTP header string."""
        if self.star_tag:
            return "*"
        return ", ".join(
            [f'"{x}"' for x in self._strong] + [f'W/"{x}"' for x in self._weak]
        )

    def __call__(
        self,
        etag: str | None = None,
        data: bytes | None = None,
        include_weak: bool = False,
    ) -> bool:
        if etag is None:
            if data is None:
                raise TypeError("'data' is required when 'etag' is not given.")

            from ..http import generate_etag

            etag = generate_etag(data)
        if include_weak:
            if etag in self._weak:
                return True
        return etag in self._strong

    def __bool__(self) -> bool:
        return bool(self.star_tag or self._strong or self._weak)

    def __str__(self) -> str:
        return self.to_header()

    def __len__(self) -> int:
        return len(self._strong)

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self._strong)

    def __contains__(self, etag: str) -> bool:  # type: ignore[override]
        return self.contains(etag)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"

# === atelier-kyo-manager/price_optimization_system\main.py ===
# main.py - 価格最適化システム（GBDT + 時系列分析）
import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel, Field
import time
from typing import List
import uvicorn

# FastAPIの日本語タイトル・説明
app = FastAPI(
    title="価格最適化API",
    description="GBDTと時系列分析による高速価格最適化システム",
    version="1.0.0"
)

# 入力データのスキーマ（日本語説明付き）
class PriceOptimizationRequest(BaseModel):
    product_id: int = Field(..., description="商品ID")
    cost: float = Field(..., description="原価")
    current_price: float = Field(..., description="現在価格")
    sales_history: List[float] = Field(..., description="過去7日間の売上データ")
    competitor_prices: List[float] = Field(..., description="競合価格データ")

# 出力データのスキーマ（日本語説明付き）
class PriceOptimizationResponse(BaseModel):
    product_id: int = Field(..., description="商品ID")
    optimal_price: float = Field(..., description="最適価格")
    predicted_demand: float = Field(..., description="予測需要")
    confidence_score: float = Field(..., description="信頼度スコア")
    processing_time_ms: int = Field(..., description="処理時間（ミリ秒）")

# ダミーのLightGBMモデル（実運用時は学習済みモデルを読み込む）
def create_dummy_model():
    X_dummy = np.random.rand(1000, 10)
    y_dummy = np.random.rand(1000)
    model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100
    )
    model.fit(X_dummy, y_dummy)
    return model

model = create_dummy_model()

# 特徴量生成（時系列・競合価格などをまとめて10次元に変換）
def create_features(request: PriceOptimizationRequest):
    features = []
    # 基本特徴量
    features.append(request.current_price / request.cost)  # 価格コスト比
    features.append(request.current_price)                 # 現在価格
    # 時系列特徴量
    if len(request.sales_history) >= 7:
        features.append(np.mean(request.sales_history[-7:]))  # 7日平均
        features.append(np.std(request.sales_history[-7:]))   # 7日標準偏差
        features.append(request.sales_history[-1])            # 直近売上
    else:
        features.extend([0, 0, 0])
    # 競合価格特徴量
    if request.competitor_prices:
        features.append(np.mean(request.competitor_prices))   # 競合平均価格
        features.append(request.current_price / np.mean(request.competitor_prices))  # 相対価格
    else:
        features.extend([0, 1])
    # パディング（10次元に調整）
    while len(features) < 10:
        features.append(0)
    return features[:10]

# 価格最適化エンドポイント
@app.post(
    "/optimize",
    response_model=PriceOptimizationResponse,
    summary="価格最適化",
    description="商品の最適価格を計算します"
)
async def optimize_price(request: PriceOptimizationRequest):
    start_time = time.time()
    features = create_features(request)
    predicted_demand = model.predict([features])[0]
    # 利益最大化のための最適価格計算（例：最低10%マージン＋需要ベース）
    optimal_price = max(
        request.cost * 1.1,
        request.cost + (predicted_demand * 0.1)
    )
    # 信頼度スコア（例：0.75～0.92の範囲で算出）
    confidence_score = min(0.92, max(0.75, 1 - abs(optimal_price - request.current_price) / request.current_price))
    processing_time = int((time.time() - start_time) * 1000)
    return PriceOptimizationResponse(
        product_id=request.product_id,
        optimal_price=round(optimal_price, 2),
        predicted_demand=round(predicted_demand, 2),
        confidence_score=round(confidence_score, 3),
        processing_time_ms=processing_time
    )

# ヘルスチェックエンドポイント
@app.get("/health", summary="ヘルスチェック", description="APIの稼働状況を返します")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\extension\test_arrow.py ===
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

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
from io import (
    BytesIO,
    StringIO,
)
import operator
import pickle
import re

import numpy as np
import pytest

from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
    PY311,
    PY312,
    is_ci_environment,
    is_platform_windows,
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under14p0,
    pa_version_under20p0,
)

from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtypeType,
)

import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_unsigned_integer_dtype,
)
from pandas.tests.extension import base

pa = pytest.importorskip("pyarrow")

from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType


def _require_timezone_database(request):
    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason=(
                "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                "on CI to path to the tzdata for pyarrow."
            ),
        )
        request.applymarker(mark)


@pytest.fixture(params=tm.ALL_PYARROW_DTYPES, ids=str)
def dtype(request):
    return ArrowDtype(pyarrow_dtype=request.param)


@pytest.fixture
def data(dtype):
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data = [True, False] * 4 + [None] + [True, False] * 44 + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 4 + [None] + [-2.0, -1.0] * 44 + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [-2, -1] * 44 + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [2, 1] * 44 + [None] + [1, 99]
    elif pa.types.is_decimal(pa_dtype):
        data = (
            [Decimal("1"), Decimal("0.0")] * 4
            + [None]
            + [Decimal("-2.0"), Decimal("-1.0")] * 44
            + [None]
            + [Decimal("0.5"), Decimal("33.123")]
        )
    elif pa.types.is_date(pa_dtype):
        data = (
            [date(2022, 1, 1), date(1999, 12, 31)] * 4
            + [None]
            + [date(2022, 1, 1), date(2022, 1, 1)] * 44
            + [None]
            + [date(1999, 12, 31), date(1999, 12, 31)]
        )
    elif pa.types.is_timestamp(pa_dtype):
        data = (
            [datetime(2020, 1, 1, 1, 1, 1, 1), datetime(1999, 1, 1, 1, 1, 1, 1)] * 4
            + [None]
            + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)] * 44
            + [None]
            + [datetime(2020, 1, 1), datetime(1999, 1, 1)]
        )
    elif pa.types.is_duration(pa_dtype):
        data = (
            [timedelta(1), timedelta(1, 1)] * 4
            + [None]
            + [timedelta(-1), timedelta(0)] * 44
            + [None]
            + [timedelta(-10), timedelta(10)]
        )
    elif pa.types.is_time(pa_dtype):
        data = (
            [time(12, 0), time(0, 12)] * 4
            + [None]
            + [time(0, 0), time(1, 1)] * 44
            + [None]
            + [time(0, 5), time(5, 0)]
        )
    elif pa.types.is_string(pa_dtype):
        data = ["a", "b"] * 4 + [None] + ["1", "2"] * 44 + [None] + ["!", ">"]
    elif pa.types.is_binary(pa_dtype):
        data = [b"a", b"b"] * 4 + [None] + [b"1", b"2"] * 44 + [None] + [b"!", b">"]
    else:
        raise NotImplementedError
    return pd.array(data, dtype=dtype)


@pytest.fixture
def data_missing(data):
    """Length-2 array with [NA, Valid]"""
    return type(data)._from_sequence([None, data[0]], dtype=data.dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_for_grouping(dtype):
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        A = False
        B = True
        C = True
    elif pa.types.is_floating(pa_dtype):
        A = -1.1
        B = 0.0
        C = 1.1
    elif pa.types.is_signed_integer(pa_dtype):
        A = -1
        B = 0
        C = 1
    elif pa.types.is_unsigned_integer(pa_dtype):
        A = 0
        B = 1
        C = 10
    elif pa.types.is_date(pa_dtype):
        A = date(1999, 12, 31)
        B = date(2010, 1, 1)
        C = date(2022, 1, 1)
    elif pa.types.is_timestamp(pa_dtype):
        A = datetime(1999, 1, 1, 1, 1, 1, 1)
        B = datetime(2020, 1, 1)
        C = datetime(2020, 1, 1, 1)
    elif pa.types.is_duration(pa_dtype):
        A = timedelta(-1)
        B = timedelta(0)
        C = timedelta(1, 4)
    elif pa.types.is_time(pa_dtype):
        A = time(0, 0)
        B = time(0, 12)
        C = time(12, 12)
    elif pa.types.is_string(pa_dtype):
        A = "a"
        B = "b"
        C = "c"
    elif pa.types.is_binary(pa_dtype):
        A = b"a"
        B = b"b"
        C = b"c"
    elif pa.types.is_decimal(pa_dtype):
        A = Decimal("-1.1")
        B = Decimal("0.0")
        C = Decimal("1.1")
    else:
        raise NotImplementedError
    return pd.array([B, B, None, None, A, A, B, C], dtype=dtype)


@pytest.fixture
def data_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_missing_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_for_twos(data):
    """Length-100 array in which all the elements are two."""
    pa_dtype = data.dtype.pyarrow_dtype
    if (
        pa.types.is_integer(pa_dtype)
        or pa.types.is_floating(pa_dtype)
        or pa.types.is_decimal(pa_dtype)
        or pa.types.is_duration(pa_dtype)
    ):
        return pd.array([2] * 100, dtype=data.dtype)
    # tests will be xfailed where 2 is not a valid scalar for pa_dtype
    return data
    # TODO: skip otherwise?


class TestArrowArray(base.ExtensionTests):
    def test_compare_scalar(self, data, comparison_op):
        ser = pd.Series(data)
        self._compare_other(ser, data, comparison_op, data[0])

    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        if data_missing.dtype.kind in "mM":
            result = data_missing.map(lambda x: x, na_action=na_action)
            expected = data_missing.to_numpy(dtype=object)
            tm.assert_numpy_array_equal(result, expected)
        else:
            result = data_missing.map(lambda x: x, na_action=na_action)
            if data_missing.dtype == "float32[pyarrow]":
                # map roundtrips through objects, which converts to float64
                expected = data_missing.to_numpy(dtype="float64", na_value=np.nan)
            else:
                expected = data_missing.to_numpy()
            tm.assert_numpy_array_equal(result, expected)

    def test_astype_str(self, data, request, using_infer_string):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_binary(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"For {pa_dtype} .astype(str) decodes.",
                )
            )
        elif not using_infer_string and (
            (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None)
            or pa.types.is_duration(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason="pd.Timestamp/pd.Timedelta repr different from numpy repr",
                )
            )
        super().test_astype_str(data)

    def test_from_dtype(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype) or pa.types.is_decimal(pa_dtype):
            if pa.types.is_string(pa_dtype):
                reason = "ArrowDtype(pa.string()) != StringDtype('pyarrow')"
            else:
                reason = f"pyarrow.type_for_alias cannot infer {pa_dtype}"

            request.applymarker(
                pytest.mark.xfail(
                    reason=reason,
                )
            )
        super().test_from_dtype(data)

    def test_from_sequence_pa_array(self, data):
        # https://github.com/pandas-dev/pandas/pull/47034#discussion_r955500784
        # data._pa_array = pa.ChunkedArray
        result = type(data)._from_sequence(data._pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)

        result = type(data)._from_sequence(
            data._pa_array.combine_chunks(), dtype=data.dtype
        )
        tm.assert_extension_array_equal(result, data)
        assert isinstance(result._pa_array, pa.ChunkedArray)

    def test_from_sequence_pa_array_notimplemented(self, request):
        with pytest.raises(NotImplementedError, match="Converting strings to"):
            ArrowExtensionArray._from_sequence_of_strings(
                ["12-1"], dtype=pa.month_day_nano_interval()
            )

    def test_from_sequence_of_strings_pa_array(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_time64(pa_dtype) and pa_dtype.equals("time64[ns]") and not PY311:
            request.applymarker(
                pytest.mark.xfail(
                    reason="Nanosecond time parsing not supported.",
                )
            )
        elif pa_version_under11p0 and (
            pa.types.is_duration(pa_dtype) or pa.types.is_decimal(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support parsing {pa_dtype}",
                )
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            _require_timezone_database(request)

        pa_array = data._pa_array.cast(pa.string())
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

        pa_array = pa_array.combine_chunks()
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    def check_accumulate(self, ser, op_name, skipna):
        result = getattr(ser, op_name)(skipna=skipna)

        pa_type = ser.dtype.pyarrow_dtype
        if pa.types.is_temporal(pa_type):
            # Just check that we match the integer behavior.
            if pa_type.bit_width == 32:
                int_type = "int32[pyarrow]"
            else:
                int_type = "int64[pyarrow]"
            ser = ser.astype(int_type)
            result = result.astype(int_type)

        result = result.astype("Float64")
        expected = getattr(ser.astype("Float64"), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)

    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "pyarrow_dtype"
        pa_type = ser.dtype.pyarrow_dtype  # type: ignore[union-attr]

        if pa.types.is_binary(pa_type) or pa.types.is_decimal(pa_type):
            if op_name in ["cumsum", "cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_string(pa_type):
            if op_name == "cumprod":
                return False
        elif pa.types.is_boolean(pa_type):
            if op_name in ["cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_temporal(pa_type):
            if op_name == "cumsum" and not pa.types.is_duration(pa_type):
                return False
            elif op_name == "cumprod":
                return False
        return True

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna, request):
        pa_type = data.dtype.pyarrow_dtype
        op_name = all_numeric_accumulations

        if pa.types.is_string(pa_type) and op_name in ["cumsum", "cummin", "cummax"]:
            # https://github.com/pandas-dev/pandas/pull/60633
            # Doesn't fit test structure, tested in series/test_cumulative.py instead.
            return

        ser = pd.Series(data)

        if not self._supports_accumulation(ser, op_name):
            # The base class test will check that we raise
            return super().test_accumulate_series(
                data, all_numeric_accumulations, skipna
            )

        if pa_version_under13p0 and all_numeric_accumulations != "cumsum":
            # xfailing takes a long time to run because pytest
            # renders the exception messages even when not showing them
            opt = request.config.option
            if opt.markexpr and "not slow" in opt.markexpr:
                pytest.skip(
                    f"{all_numeric_accumulations} not implemented for pyarrow < 9"
                )
            mark = pytest.mark.xfail(
                reason=f"{all_numeric_accumulations} not implemented for pyarrow < 9"
            )
            request.applymarker(mark)

        elif all_numeric_accumulations == "cumsum" and (
            pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{all_numeric_accumulations} not implemented for {pa_type}",
                    raises=TypeError,
                )
            )

        self.check_accumulate(ser, op_name, skipna)

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name == "kurt" or (pa_version_under20p0 and op_name == "skew"):
            return False

        dtype = ser.dtype
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has
        # no attribute "pyarrow_dtype"
        pa_dtype = dtype.pyarrow_dtype  # type: ignore[union-attr]
        if pa.types.is_temporal(pa_dtype) and op_name in [
            "sum",
            "var",
            "skew",
            "kurt",
            "prod",
        ]:
            if pa.types.is_duration(pa_dtype) and op_name in ["sum"]:
                # summing timedeltas is one case that *is* well-defined
                pass
            else:
                return False
        elif pa.types.is_binary(pa_dtype) and op_name in ["sum", "skew"]:
            return False
        elif (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ) and op_name in [
            "mean",
            "median",
            "prod",
            "std",
            "sem",
            "var",
            "skew",
            "kurt",
        ]:
            return False

        if (
            pa.types.is_temporal(pa_dtype)
            and not pa.types.is_duration(pa_dtype)
            and op_name in ["any", "all"]
        ):
            # xref GH#34479 we support this in our non-pyarrow datetime64 dtypes,
            #  but it isn't obvious we _should_.  For now, we keep the pyarrow
            #  behavior which does not support this.
            return False

        return True

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "pyarrow_dtype"
        pa_dtype = ser.dtype.pyarrow_dtype  # type: ignore[union-attr]
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            alt = ser.astype("Float64")
        else:
            # TODO: in the opposite case, aren't we testing... nothing? For
            # e.g. date/time dtypes trying to calculate 'expected' by converting
            # to object will raise for mean, std etc
            alt = ser

        # TODO: in the opposite case, aren't we testing... nothing?
        if op_name == "count":
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        dtype = data.dtype
        pa_dtype = dtype.pyarrow_dtype

        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_numeric_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        if pa.types.is_boolean(pa_dtype) and all_numeric_reductions in {
            "sem",
            "std",
            "var",
            "median",
        }:
            request.applymarker(xfail_mark)
        elif (
            not pa_version_under20p0
            and all_numeric_reductions == "skew"
            and (
                pa.types.is_boolean(pa_dtype)
                or (
                    skipna
                    and (
                        pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype)
                    )
                )
            )
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason="https://github.com/apache/arrow/issues/45733",
                )
            )
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(
        self, data, all_boolean_reductions, skipna, na_value, request
    ):
        pa_dtype = data.dtype.pyarrow_dtype
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_boolean_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype):
            # We *might* want to make this behave like the non-pyarrow cases,
            #  but have not yet decided.
            request.applymarker(xfail_mark)

        return super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        pa_type = arr._pa_array.type
        if op_name in ["max", "min"]:
            cmp_dtype = arr.dtype
        elif arr.dtype.name == "decimal128(7, 3)[pyarrow]":
            if op_name not in ["median", "var", "std", "skew"]:
                cmp_dtype = arr.dtype
            else:
                cmp_dtype = "float64[pyarrow]"
        elif op_name in ["median", "var", "std", "mean", "skew"]:
            cmp_dtype = "float64[pyarrow]"
        elif op_name == "sum" and pa.types.is_string(pa_type):
            cmp_dtype = arr.dtype
        else:
            cmp_dtype = {
                "i": "int64[pyarrow]",
                "u": "uint64[pyarrow]",
                "f": "float64[pyarrow]",
            }[arr.dtype.kind]
        return cmp_dtype

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        op_name = all_numeric_reductions
        if op_name == "skew" and pa_version_under20p0:
            if data.dtype._is_numeric:
                mark = pytest.mark.xfail(reason="skew not implemented")
                request.applymarker(mark)
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    @pytest.mark.parametrize("typ", ["int64", "uint64", "float64"])
    def test_median_not_approximate(self, typ):
        # GH 52679
        result = pd.Series([1, 2], dtype=f"{typ}[pyarrow]").median()
        assert result == 1.5

    def test_construct_from_string_own_name(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )

        if pa.types.is_string(pa_dtype):
            # We still support StringDtype('pyarrow') over ArrowDtype(pa.string())
            msg = r"string\[pyarrow\] should be constructed by StringDtype"
            with pytest.raises(TypeError, match=msg):
                dtype.construct_from_string(dtype.name)

            return

        super().test_construct_from_string_own_name(dtype)

    def test_is_dtype_from_name(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            # We still support StringDtype('pyarrow') over ArrowDtype(pa.string())
            assert not type(dtype).is_dtype(dtype.name)
        else:
            if pa.types.is_decimal(pa_dtype):
                request.applymarker(
                    pytest.mark.xfail(
                        raises=NotImplementedError,
                        reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                    )
                )
            super().test_is_dtype_from_name(dtype)

    def test_construct_from_string_another_type_raises(self, dtype):
        msg = r"'another_type' must end with '\[pyarrow\]'"
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")

    def test_get_common_dtype(self, dtype, request):
        pa_dtype = dtype.pyarrow_dtype
        if (
            pa.types.is_date(pa_dtype)
            or pa.types.is_time(pa_dtype)
            or (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None)
            or pa.types.is_binary(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        f"{pa_dtype} does not have associated numpy "
                        f"dtype findable by find_common_type"
                    )
                )
            )
        super().test_get_common_dtype(dtype)

    def test_is_not_string_type(self, dtype):
        pa_dtype = dtype.pyarrow_dtype
        if pa.types.is_string(pa_dtype):
            assert is_string_dtype(dtype)
        else:
            super().test_is_not_string_type(dtype)

    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views.", run=False
    )
    def test_view(self, data):
        super().test_view(data)

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]

        valid = data[0]
        result = data.fillna(valid)
        assert result is not data
        tm.assert_extension_array_equal(result, data)

        result = data.fillna(method="backfill")
        assert result is not data
        tm.assert_extension_array_equal(result, data)

    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views", run=False
    )
    def test_transpose(self, data):
        super().test_transpose(data)

    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views", run=False
    )
    def test_setitem_preserves_views(self, data):
        super().test_setitem_preserves_views(data)

    @pytest.mark.parametrize("dtype_backend", ["pyarrow", no_default])
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data, dtype_backend, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"Parameterized types {pa_dtype} not supported.",
                )
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.unit in ("us", "ns"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=ValueError,
                    reason="https://github.com/pandas-dev/pandas/issues/49767",
                )
            )
        elif pa.types.is_binary(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(reason="CSV parsers don't correctly handle binary")
            )
        df = pd.DataFrame({"with_dtype": pd.Series(data, dtype=str(data.dtype))})
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        if pa.types.is_binary(pa_dtype):
            csv_output = BytesIO(csv_output)
        else:
            csv_output = StringIO(csv_output)
        result = pd.read_csv(
            csv_output,
            dtype={"with_dtype": str(data.dtype)},
            engine=engine,
            dtype_backend=dtype_backend,
        )
        expected = df
        tm.assert_frame_equal(result, expected)

    def test_invert(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if not (
            pa.types.is_boolean(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_string(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow.compute.invert does support {pa_dtype}",
                )
            )
        if PY312 and pa.types.is_boolean(pa_dtype):
            with tm.assert_produces_warning(
                DeprecationWarning, match="Bitwise inversion", check_stacklevel=False
            ):
                super().test_invert(data)
        else:
            super().test_invert(data)

    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods, request):
        pa_dtype = data.dtype.pyarrow_dtype
        if pa.types.is_unsigned_integer(pa_dtype) and periods == 1:
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"diff with {pa_dtype} and periods={periods} will overflow"
                    ),
                )
            )
        super().test_diff(data, periods)

    def test_value_counts_returns_pyarrow_int64(self, data):
        # GH 51462
        data = data[:10]
        result = data.value_counts()
        assert result.dtype == ArrowDtype(pa.int64())

    _combine_le_expected_dtype = "bool[pyarrow]"

    def get_op_from_name(self, op_name):
        short_opname = op_name.strip("_")
        if short_opname == "rtruediv":
            # use the numpy version that won't raise on division by zero

            def rtruediv(x, y):
                return np.divide(y, x)

            return rtruediv
        elif short_opname == "rfloordiv":
            return lambda x, y: np.floor_divide(y, x)

        return tm.get_op_from_name(op_name)

    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        # BaseOpsUtil._combine can upcast expected dtype
        # (because it generates expected on python scalars)
        # while ArrowExtensionArray maintains original type
        expected = pointwise_result

        if op_name in ["eq", "ne", "lt", "le", "gt", "ge"]:
            return pointwise_result.astype("boolean[pyarrow]")

        was_frame = False
        if isinstance(expected, pd.DataFrame):
            was_frame = True
            expected_data = expected.iloc[:, 0]
            original_dtype = obj.iloc[:, 0].dtype
        else:
            expected_data = expected
            original_dtype = obj.dtype

        orig_pa_type = original_dtype.pyarrow_dtype
        if not was_frame and isinstance(other, pd.Series):
            # i.e. test_arith_series_with_array
            if not (
                pa.types.is_floating(orig_pa_type)
                or (
                    pa.types.is_integer(orig_pa_type)
                    and op_name not in ["__truediv__", "__rtruediv__"]
                )
                or pa.types.is_duration(orig_pa_type)
                or pa.types.is_timestamp(orig_pa_type)
                or pa.types.is_date(orig_pa_type)
                or pa.types.is_decimal(orig_pa_type)
            ):
                # base class _combine always returns int64, while
                #  ArrowExtensionArray does not upcast
                return expected
        elif not (
            (op_name == "__floordiv__" and pa.types.is_integer(orig_pa_type))
            or pa.types.is_duration(orig_pa_type)
            or pa.types.is_timestamp(orig_pa_type)
            or pa.types.is_date(orig_pa_type)
            or pa.types.is_decimal(orig_pa_type)
        ):
            # base class _combine always returns int64, while
            #  ArrowExtensionArray does not upcast
            return expected

        pa_expected = pa.array(expected_data._values)

        if pa.types.is_duration(pa_expected.type):
            if pa.types.is_date(orig_pa_type):
                if pa.types.is_date64(orig_pa_type):
                    # TODO: why is this different vs date32?
                    unit = "ms"
                else:
                    unit = "s"
            else:
                # pyarrow sees sequence of datetime/timedelta objects and defaults
                #  to "us" but the non-pointwise op retains unit
                # timestamp or duration
                unit = orig_pa_type.unit
                if type(other) in [datetime, timedelta] and unit in ["s", "ms"]:
                    # pydatetime/pytimedelta objects have microsecond reso, so we
                    #  take the higher reso of the original and microsecond. Note
                    #  this matches what we would do with DatetimeArray/TimedeltaArray
                    unit = "us"

            pa_expected = pa_expected.cast(f"duration[{unit}]")

        elif pa.types.is_decimal(pa_expected.type) and pa.types.is_decimal(
            orig_pa_type
        ):
            # decimal precision can resize in the result type depending on data
            # just compare the float values
            alt = getattr(obj, op_name)(other)
            alt_dtype = tm.get_dtype(alt)
            assert isinstance(alt_dtype, ArrowDtype)
            if op_name == "__pow__" and isinstance(other, Decimal):
                # TODO: would it make more sense to retain Decimal here?
                alt_dtype = ArrowDtype(pa.float64())
            elif (
                op_name == "__pow__"
                and isinstance(other, pd.Series)
                and other.dtype == original_dtype
            ):
                # TODO: would it make more sense to retain Decimal here?
                alt_dtype = ArrowDtype(pa.float64())
            else:
                assert pa.types.is_decimal(alt_dtype.pyarrow_dtype)
            return expected.astype(alt_dtype)

        else:
            pa_expected = pa_expected.cast(orig_pa_type)

        pd_expected = type(expected_data._values)(pa_expected)
        if was_frame:
            expected = pd.DataFrame(
                pd_expected, index=expected.index, columns=expected.columns
            )
        else:
            expected = pd.Series(pd_expected)
        return expected

    def _is_temporal_supported(self, opname, pa_dtype):
        return (
            (
                opname in ("__add__", "__radd__")
                or (
                    opname
                    in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                    and not pa_version_under14p0
                )
            )
            and pa.types.is_duration(pa_dtype)
            or opname in ("__sub__", "__rsub__")
            and pa.types.is_temporal(pa_dtype)
        )

    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | tuple[type[Exception], ...] | None:
        if op_name in ("__divmod__", "__rdivmod__"):
            return (NotImplementedError, TypeError)

        exc: type[Exception] | tuple[type[Exception], ...] | None
        dtype = tm.get_dtype(obj)
        # error: Item "dtype[Any]" of "dtype[Any] | ExtensionDtype" has no
        # attribute "pyarrow_dtype"
        pa_dtype = dtype.pyarrow_dtype  # type: ignore[union-attr]

        arrow_temporal_supported = self._is_temporal_supported(op_name, pa_dtype)
        if op_name in {
            "__mod__",
            "__rmod__",
        }:
            exc = (NotImplementedError, TypeError)
        elif arrow_temporal_supported:
            exc = None
        elif op_name in ["__add__", "__radd__"] and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            exc = None
        elif not (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            exc = TypeError
        else:
            exc = None
        return exc

    def _get_arith_xfail_marker(self, opname, pa_dtype):
        mark = None

        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)

        if opname == "__rpow__" and (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            mark = pytest.mark.xfail(
                reason=(
                    f"GH#29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                    f"for {pa_dtype}"
                )
            )
        elif arrow_temporal_supported and (
            pa.types.is_time(pa_dtype)
            or (
                opname
                in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                and pa.types.is_duration(pa_dtype)
            )
        ):
            mark = pytest.mark.xfail(
                raises=TypeError,
                reason=(
                    f"{opname} not supported between"
                    f"pd.NA and {pa_dtype} Python scalar"
                ),
            )
        elif opname == "__rfloordiv__" and (
            pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype)
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        elif opname == "__rtruediv__" and pa.types.is_decimal(pa_dtype):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )

        return mark

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        pa_dtype = data.dtype.pyarrow_dtype

        if all_arithmetic_operators == "__rmod__" and pa.types.is_binary(pa_dtype):
            pytest.skip("Skip testing Python string formatting")

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        pa_dtype = data.dtype.pyarrow_dtype

        if all_arithmetic_operators == "__rmod__" and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            pytest.skip("Skip testing Python string formatting")

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    def test_arith_series_with_array(self, data, all_arithmetic_operators, request):
        pa_dtype = data.dtype.pyarrow_dtype

        if all_arithmetic_operators in (
            "__sub__",
            "__rsub__",
        ) and pa.types.is_unsigned_integer(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"Implemented pyarrow.compute.subtract_checked "
                        f"which raises on overflow for {pa_dtype}"
                    ),
                )
            )

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        # pd.Series([ser.iloc[0]] * len(ser)) may not return ArrowExtensionArray
        # since ser.iloc[0] is a python scalar
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))

        self.check_opname(ser, op_name, other)

    def test_add_series_with_extension_array(self, data, request):
        pa_dtype = data.dtype.pyarrow_dtype

        if pa_dtype.equals("int8"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"raises on overflow for {pa_dtype}",
                )
            )
        super().test_add_series_with_extension_array(data)

    def test_invalid_other_comp(self, data, comparison_op):
        # GH 48833
        with pytest.raises(
            NotImplementedError, match=".* not implemented for <class 'object'>"
        ):
            comparison_op(data, object())

    @pytest.mark.parametrize("masked_dtype", ["boolean", "Int64", "Float64"])
    def test_comp_masked_numpy(self, masked_dtype, comparison_op):
        # GH 52625
        data = [1, 0, None]
        ser_masked = pd.Series(data, dtype=masked_dtype)
        ser_pa = pd.Series(data, dtype=f"{masked_dtype.lower()}[pyarrow]")
        result = comparison_op(ser_pa, ser_masked)
        if comparison_op in [operator.lt, operator.gt, operator.ne]:
            exp = [False, False, None]
        else:
            exp = [True, True, None]
        expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)


class TestLogicalOps:
    """Various Series and DataFrame logical ops methods."""

    def test_kleene_or(self):
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a | b
        expected = pd.Series(
            [True, True, True, True, False, None, True, None, None],
            dtype="boolean[pyarrow]",
        )
        tm.assert_series_equal(result, expected)

        result = b | a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [True, None, None]),
            (pd.NA, [True, None, None]),
            (True, [True, True, True]),
            (np.bool_(True), [True, True, True]),
            (False, [True, False, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_or_scalar(self, other, expected):
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a | other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)

        result = other | a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_and(self):
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a & b
        expected = pd.Series(
            [True, False, None, False, False, False, None, False, None],
            dtype="boolean[pyarrow]",
        )
        tm.assert_series_equal(result, expected)

        result = b & a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, False, None]),
            (pd.NA, [None, False, None]),
            (True, [True, False, None]),
            (False, [False, False, False]),
            (np.bool_(True), [True, False, None]),
            (np.bool_(False), [False, False, False]),
        ],
    )
    def test_kleene_and_scalar(self, other, expected):
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a & other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)

        result = other & a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_xor(self):
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        result = a ^ b
        expected = pd.Series(
            [False, True, None, True, False, None, None, None, None],
            dtype="boolean[pyarrow]",
        )
        tm.assert_series_equal(result, expected)

        result = b ^ a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, None, None]),
            (pd.NA, [None, None, None]),
            (True, [False, True, None]),
            (np.bool_(True), [False, True, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_xor_scalar(self, other, expected):
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        result = a ^ other
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        tm.assert_series_equal(result, expected)

        result = other ^ a
        tm.assert_series_equal(result, expected)

        # ensure we haven't mutated anything inplace
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "op, exp",
        [
            ["__and__", True],
            ["__or__", True],
            ["__xor__", False],
        ],
    )
    def test_logical_masked_numpy(self, op, exp):
        # GH 52625
        data = [True, False, None]
        ser_masked = pd.Series(data, dtype="boolean")
        ser_pa = pd.Series(data, dtype="boolean[pyarrow]")
        result = getattr(ser_pa, op)(ser_masked)
        expected = pd.Series([exp, False, None], dtype=ArrowDtype(pa.bool_()))
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
def test_bitwise(pa_type):
    # GH 54495
    dtype = ArrowDtype(pa_type)
    left = pd.Series([1, None, 3, 4], dtype=dtype)
    right = pd.Series([None, 3, 5, 4], dtype=dtype)

    result = left | right
    expected = pd.Series([None, None, 3 | 5, 4 | 4], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = left & right
    expected = pd.Series([None, None, 3 & 5, 4 & 4], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = left ^ right
    expected = pd.Series([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    tm.assert_series_equal(result, expected)

    result = ~left
    expected = ~(left.fillna(0).to_numpy())
    expected = pd.Series(expected, dtype=dtype).mask(left.isnull())
    tm.assert_series_equal(result, expected)


def test_arrowdtype_construct_from_string_type_with_unsupported_parameters():
    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("not_a_real_dype[s, tz=UTC][pyarrow]")

    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("decimal(7, 2)[pyarrow]")


def test_arrowdtype_construct_from_string_supports_dt64tz():
    # as of GH#50689, timestamptz is supported
    dtype = ArrowDtype.construct_from_string("timestamp[s, tz=UTC][pyarrow]")
    expected = ArrowDtype(pa.timestamp("s", "UTC"))
    assert dtype == expected


def test_arrowdtype_construct_from_string_type_only_one_pyarrow():
    # GH#51225
    invalid = "int64[pyarrow]foobar[pyarrow]"
    msg = (
        r"Passing pyarrow type specific parameters \(\[pyarrow\]\) in the "
        r"string is not supported\."
    )
    with pytest.raises(NotImplementedError, match=msg):
        pd.Series(range(3), dtype=invalid)


def test_arrow_string_multiplication():
    # GH 56537
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    repeat = pd.Series([2, -2], dtype="int64[pyarrow]")
    result = binary * repeat
    expected = pd.Series(["abcabc", ""], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)
    reflected_result = repeat * binary
    tm.assert_series_equal(result, reflected_result)


def test_arrow_string_multiplication_scalar_repeat():
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    result = binary * 2
    expected = pd.Series(["abcabc", "defgdefg"], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)
    reflected_result = 2 * binary
    tm.assert_series_equal(reflected_result, expected)


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize("quantile", [0.5, [0.5, 0.5]])
def test_quantile(data, interpolation, quantile, request):
    pa_dtype = data.dtype.pyarrow_dtype

    data = data.take([0, 0, 0])
    ser = pd.Series(data)

    if (
        pa.types.is_string(pa_dtype)
        or pa.types.is_binary(pa_dtype)
        or pa.types.is_boolean(pa_dtype)
    ):
        # For string, bytes, and bool, we don't *expect* to have quantile work
        # Note this matches the non-pyarrow behavior
        msg = r"Function 'quantile' has no kernel matching input types \(.*\)"
        with pytest.raises(pa.ArrowNotImplementedError, match=msg):
            ser.quantile(q=quantile, interpolation=interpolation)
        return

    if (
        pa.types.is_integer(pa_dtype)
        or pa.types.is_floating(pa_dtype)
        or pa.types.is_decimal(pa_dtype)
    ):
        pass
    elif pa.types.is_temporal(data._pa_array.type):
        pass
    else:
        request.applymarker(
            pytest.mark.xfail(
                raises=pa.ArrowNotImplementedError,
                reason=f"quantile not supported by pyarrow for {pa_dtype}",
            )
        )
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    result = ser.quantile(q=quantile, interpolation=interpolation)

    if pa.types.is_timestamp(pa_dtype) and interpolation not in ["lower", "higher"]:
        # rounding error will make the check below fail
        #  (e.g. '2020-01-01 01:01:01.000001' vs '2020-01-01 01:01:01.000001024'),
        #  so we'll check for now that we match the numpy analogue
        if pa_dtype.tz:
            pd_dtype = f"M8[{pa_dtype.unit}, {pa_dtype.tz}]"
        else:
            pd_dtype = f"M8[{pa_dtype.unit}]"
        ser_np = ser.astype(pd_dtype)

        expected = ser_np.quantile(q=quantile, interpolation=interpolation)
        if quantile == 0.5:
            if pa_dtype.unit == "us":
                expected = expected.to_pydatetime(warn=False)
            assert result == expected
        else:
            if pa_dtype.unit == "us":
                expected = expected.dt.floor("us")
            tm.assert_series_equal(result, expected.astype(data.dtype))
        return

    if quantile == 0.5:
        assert result == data[0]
    else:
        # Just check the values
        expected = pd.Series(data.take([0, 0]), index=[0.5, 0.5])
        if (
            pa.types.is_integer(pa_dtype)
            or pa.types.is_floating(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            expected = expected.astype("float64[pyarrow]")
            result = result.astype("float64[pyarrow]")
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "take_idx, exp_idx",
    [[[0, 0, 2, 2, 4, 4], [4, 0]], [[0, 0, 0, 2, 4, 4], [0]]],
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(data_for_grouping, take_idx, exp_idx):
    data = data_for_grouping.take(take_idx)
    ser = pd.Series(data)
    result = ser.mode(dropna=True)
    expected = pd.Series(data_for_grouping.take(exp_idx))
    tm.assert_series_equal(result, expected)


def test_mode_dropna_false_mode_na(data):
    # GH 50982
    more_nans = pd.Series([None, None, data[0]], dtype=data.dtype)
    result = more_nans.mode(dropna=False)
    expected = pd.Series([None], dtype=data.dtype)
    tm.assert_series_equal(result, expected)

    expected = pd.Series([data[0], None], dtype=data.dtype)
    result = expected.mode(dropna=False)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    [
        [pa.binary(), bytes],
        [pa.binary(16), bytes],
        [pa.large_binary(), bytes],
        [pa.large_string(), str],
        [pa.list_(pa.int64()), list],
        [pa.large_list(pa.int64()), list],
        [pa.map_(pa.string(), pa.int64()), list],
        [pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict],
        [pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType],
    ],
)
def test_arrow_dtype_type(arrow_dtype, expected_type):
    # GH 51845
    # TODO: Redundant with test_getitem_scalar once arrow_dtype exists in data fixture
    assert ArrowDtype(arrow_dtype).type == expected_type


def test_is_bool_dtype():
    # GH 22667
    data = ArrowExtensionArray(pa.array([True, False, True]))
    assert is_bool_dtype(data)
    assert pd.core.common.is_bool_indexer(data)
    s = pd.Series(range(len(data)))
    result = s[data]
    expected = s[np.asarray(data)]
    tm.assert_series_equal(result, expected)


def test_is_numeric_dtype(data):
    # GH 50563
    pa_type = data.dtype.pyarrow_dtype
    if (
        pa.types.is_floating(pa_type)
        or pa.types.is_integer(pa_type)
        or pa.types.is_decimal(pa_type)
    ):
        assert is_numeric_dtype(data)
    else:
        assert not is_numeric_dtype(data)


def test_is_integer_dtype(data):
    # GH 50667
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_integer(pa_type):
        assert is_integer_dtype(data)
    else:
        assert not is_integer_dtype(data)


def test_is_signed_integer_dtype(data):
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_signed_integer(pa_type):
        assert is_signed_integer_dtype(data)
    else:
        assert not is_signed_integer_dtype(data)


def test_is_unsigned_integer_dtype(data):
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_unsigned_integer(pa_type):
        assert is_unsigned_integer_dtype(data)
    else:
        assert not is_unsigned_integer_dtype(data)


def test_is_float_dtype(data):
    pa_type = data.dtype.pyarrow_dtype
    if pa.types.is_floating(pa_type):
        assert is_float_dtype(data)
    else:
        assert not is_float_dtype(data)


def test_pickle_roundtrip(data):
    # GH 42600
    expected = pd.Series(data)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)

    assert len(full_pickled) > len(sliced_pickled)

    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)

    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)


def test_astype_from_non_pyarrow(data):
    # GH49795
    pd_array = data._pa_array.to_pandas().array
    result = pd_array.astype(data.dtype)
    assert not isinstance(pd_array.dtype, ArrowDtype)
    assert isinstance(result.dtype, ArrowDtype)
    tm.assert_extension_array_equal(result, data)


def test_astype_float_from_non_pyarrow_str():
    # GH50430
    ser = pd.Series(["1.0"])
    result = ser.astype("float64[pyarrow]")
    expected = pd.Series([1.0], dtype="float64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_astype_errors_ignore():
    # GH 55399
    expected = pd.DataFrame({"col": [17000000]}, dtype="int32[pyarrow]")
    result = expected.astype("float[pyarrow]", errors="ignore")
    tm.assert_frame_equal(result, expected)


def test_to_numpy_with_defaults(data):
    # GH49973
    result = data.to_numpy()

    pa_type = data._pa_array.type
    if pa.types.is_duration(pa_type) or pa.types.is_timestamp(pa_type):
        pytest.skip("Tested in test_to_numpy_temporal")
    elif pa.types.is_date(pa_type):
        expected = np.array(list(data))
    else:
        expected = np.array(data._pa_array)

    if data._hasna and not is_numeric_dtype(data.dtype):
        expected = expected.astype(object)
        expected[pd.isna(data)] = pd.NA

    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_int_with_na():
    # GH51227: ensure to_numpy does not convert int to float
    data = [1, None]
    arr = pd.array(data, dtype="int64[pyarrow]")
    result = arr.to_numpy()
    expected = np.array([1, np.nan])
    assert isinstance(result[0], float)
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("na_val, exp", [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val, exp):
    # GH#52443
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype="float64", na_value=na_val)
    expected = np.array([exp] * 2, dtype="float64")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_null_array_no_dtype():
    # GH#52443
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    result = arr.to_numpy(dtype=None)
    expected = np.array([pd.NA] * 2, dtype="object")
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_without_dtype():
    # GH 54808
    arr = pd.array([True, pd.NA], dtype="boolean[pyarrow]")
    result = arr.to_numpy(na_value=False)
    expected = np.array([True, False], dtype=np.bool_)
    tm.assert_numpy_array_equal(result, expected)

    arr = pd.array([1.0, pd.NA], dtype="float32[pyarrow]")
    result = arr.to_numpy(na_value=0.0)
    expected = np.array([1.0, 0.0], dtype=np.float32)
    tm.assert_numpy_array_equal(result, expected)


def test_setitem_null_slice(data):
    # GH50248
    orig = data.copy()

    result = orig.copy()
    result[:] = data[0]
    expected = ArrowExtensionArray._from_sequence(
        [data[0]] * len(data),
        dtype=data.dtype,
    )
    tm.assert_extension_array_equal(result, expected)

    result = orig.copy()
    result[:] = data[::-1]
    expected = data[::-1]
    tm.assert_extension_array_equal(result, expected)

    result = orig.copy()
    result[:] = data.tolist()
    expected = data
    tm.assert_extension_array_equal(result, expected)


def test_setitem_invalid_dtype(data):
    # GH50248
    pa_type = data._pa_array.type
    if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
        fill_value = 123
        err = TypeError
        msg = "Invalid value '123' for dtype"
    elif (
        pa.types.is_integer(pa_type)
        or pa.types.is_floating(pa_type)
        or pa.types.is_boolean(pa_type)
    ):
        fill_value = "foo"
        err = pa.ArrowInvalid
        msg = "Could not convert"
    else:
        fill_value = "foo"
        err = TypeError
        msg = "Invalid value 'foo' for dtype"
    with pytest.raises(err, match=msg):
        data[:] = fill_value


def test_from_arrow_respecting_given_dtype():
    date_array = pa.array(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")], type=pa.date32()
    )
    result = date_array.to_pandas(
        types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get
    )
    expected = pd.Series(
        [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")],
        dtype=ArrowDtype(pa.date64()),
    )
    tm.assert_series_equal(result, expected)


def test_from_arrow_respecting_given_dtype_unsafe():
    array = pa.array([1.5, 2.5], type=pa.float64())
    with tm.external_error_raised(pa.ArrowInvalid):
        array.to_pandas(types_mapper={pa.float64(): ArrowDtype(pa.int64())}.get)


def test_round():
    dtype = "float64[pyarrow]"

    ser = pd.Series([0.0, 1.23, 2.56, pd.NA], dtype=dtype)
    result = ser.round(1)
    expected = pd.Series([0.0, 1.2, 2.6, pd.NA], dtype=dtype)
    tm.assert_series_equal(result, expected)

    ser = pd.Series([123.4, pd.NA, 56.78], dtype=dtype)
    result = ser.round(-1)
    expected = pd.Series([120.0, pd.NA, 60.0], dtype=dtype)
    tm.assert_series_equal(result, expected)


def test_searchsorted_with_na_raises(data_for_sorting, as_series):
    # GH50447
    b, c, a = data_for_sorting
    arr = data_for_sorting.take([2, 0, 1])  # to get [a, b, c]
    arr[-1] = pd.NA

    if as_series:
        arr = pd.Series(arr)

    msg = (
        "searchsorted requires array to be sorted, "
        "which is impossible with NAs present."
    )
    with pytest.raises(ValueError, match=msg):
        arr.searchsorted(b)


def test_sort_values_dictionary():
    df = pd.DataFrame(
        {
            "a": pd.Series(
                ["x", "y"], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
            ),
            "b": [1, 2],
        },
    )
    expected = df.copy()
    result = df.sort_values(by=["a", "b"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("pat", ["abc", "a[a-z]{2}"])
def test_str_count(pat):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.count(pat)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)


def test_str_count_flags_unsupported():
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="count not"):
        ser.str.count("abc", flags=1)


@pytest.mark.parametrize(
    "side, str_func", [["left", "rjust"], ["right", "ljust"], ["both", "center"]]
)
def test_str_pad(side, str_func):
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.pad(width=3, side=side, fillchar="x")
    expected = pd.Series(
        [getattr("a", str_func)(3, "x"), None], dtype=ArrowDtype(pa.string())
    )
    tm.assert_series_equal(result, expected)


def test_str_pad_invalid_side():
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(ValueError, match="Invalid side: foo"):
        ser.str.pad(3, "foo", "x")


@pytest.mark.parametrize(
    "pat, case, na, regex, exp",
    [
        ["ab", False, None, False, [True, None]],
        ["Ab", True, None, False, [False, None]],
        ["ab", False, True, False, [True, True]],
        ["a[a-z]{1}", False, None, True, [True, None]],
        ["A[a-z]{1}", True, None, True, [False, None]],
    ],
)
def test_str_contains(pat, case, na, regex, exp):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.contains(pat, case=case, na=na, regex=regex)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


def test_str_contains_flags_unsupported():
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="contains not"):
        ser.str.contains("a", flags=1)


@pytest.mark.parametrize(
    "side, pat, na, exp",
    [
        ["startswith", "ab", None, [True, None, False]],
        ["startswith", "b", False, [False, False, False]],
        ["endswith", "b", True, [False, True, False]],
        ["endswith", "bc", None, [True, None, False]],
        ["startswith", ("a", "e", "g"), None, [True, None, True]],
        ["endswith", ("a", "c", "g"), None, [True, None, True]],
        ["startswith", (), None, [False, None, False]],
        ["endswith", (), None, [False, None, False]],
    ],
)
def test_str_start_ends_with(side, pat, na, exp):
    ser = pd.Series(["abc", None, "efg"], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(pat, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("side", ("startswith", "endswith"))
def test_str_starts_ends_with_all_nulls_empty_tuple(side):
    ser = pd.Series([None, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(())

    # bool datatype preserved for all nulls.
    expected = pd.Series([None, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arg_name, arg",
    [["pat", re.compile("b")], ["repl", str], ["case", False], ["flags", 1]],
)
def test_str_replace_unsupported(arg_name, arg):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    kwargs = {"pat": "b", "repl": "x", "regex": True}
    kwargs[arg_name] = arg
    with pytest.raises(NotImplementedError, match="replace is not supported"):
        ser.str.replace(**kwargs)


@pytest.mark.parametrize(
    "pat, repl, n, regex, exp",
    [
        ["a", "x", -1, False, ["xbxc", None]],
        ["a", "x", 1, False, ["xbac", None]],
        ["[a-b]", "x", -1, True, ["xxxc", None]],
    ],
)
def test_str_replace(pat, repl, n, regex, exp):
    ser = pd.Series(["abac", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.replace(pat, repl, n=n, regex=regex)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_replace_negative_n():
    # GH 56404
    ser = pd.Series(["abc", "aaaaaa"], dtype=ArrowDtype(pa.string()))
    actual = ser.str.replace("a", "", -3, True)
    expected = pd.Series(["bc", ""], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(expected, actual)

    # Same bug for pyarrow-backed StringArray GH#59628
    ser2 = ser.astype(pd.StringDtype(storage="pyarrow"))
    actual2 = ser2.str.replace("a", "", -3, True)
    expected2 = expected.astype(ser2.dtype)
    tm.assert_series_equal(expected2, actual2)

    ser3 = ser.astype(pd.StringDtype(storage="pyarrow", na_value=np.nan))
    actual3 = ser3.str.replace("a", "", -3, True)
    expected3 = expected.astype(ser3.dtype)
    tm.assert_series_equal(expected3, actual3)


def test_str_repeat_unsupported():
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    with pytest.raises(NotImplementedError, match="repeat is not"):
        ser.str.repeat([1, 2])


def test_str_repeat():
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.repeat(2)
    expected = pd.Series(["abcabc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    [
        ["ab", False, None, [True, None]],
        ["Ab", True, None, [False, None]],
        ["bc", True, None, [False, None]],
        ["ab", False, True, [True, True]],
        ["a[a-z]{1}", False, None, [True, None]],
        ["A[a-z]{1}", True, None, [False, None]],
    ],
)
def test_str_match(pat, case, na, exp):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.match(pat, case=case, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    [
        ["abc", False, None, [True, True, False, None]],
        ["Abc", True, None, [False, False, False, None]],
        ["bc", True, None, [False, False, False, None]],
        ["ab", False, None, [True, True, False, None]],
        ["a[a-z]{2}", False, None, [True, True, False, None]],
        ["A[a-z]{1}", True, None, [False, False, False, None]],
        # GH Issue: #56652
        ["abc$", False, None, [True, False, False, None]],
        ["abc\\$", False, None, [False, True, False, None]],
        ["Abc$", True, None, [False, False, False, None]],
        ["Abc\\$", True, None, [False, False, False, None]],
    ],
)
def test_str_fullmatch(pat, case, na, exp):
    ser = pd.Series(["abc", "abc$", "$abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.match(pat, case=case, na=na)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "sub, start, end, exp, exp_typ",
    [["ab", 0, None, [0, None], pa.int32()], ["bc", 1, 3, [1, None], pa.int64()]],
)
def test_str_find(sub, start, end, exp, exp_typ):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub, start=start, end=end)
    expected = pd.Series(exp, dtype=ArrowDtype(exp_typ))
    tm.assert_series_equal(result, expected)


def test_str_find_negative_start():
    # GH 56411
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="b", start=-1000, end=3)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_no_end():
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find("ab", start=1)
    expected = pd.Series([-1, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(result, expected)


def test_str_find_negative_start_negative_end():
    # GH 56791
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-6, end=-3)
    expected = pd.Series([3, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_large_start():
    # GH 56791
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=16)
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(
    pa_version_under13p0, reason="https://github.com/apache/arrow/issues/36311"
)
@pytest.mark.parametrize("start", [-15, -3, 0, 1, 15, None])
@pytest.mark.parametrize("end", [-15, -1, 0, 3, 15, None])
@pytest.mark.parametrize("sub", ["", "az", "abce", "a", "caa"])
def test_str_find_e2e(start, end, sub):
    s = pd.Series(
        ["abcaadef", "abc", "abcdeddefgj8292", "ab", "a", ""],
        dtype=ArrowDtype(pa.string()),
    )
    object_series = s.astype(pd.StringDtype(storage="python"))
    result = s.str.find(sub, start, end)
    expected = object_series.str.find(sub, start, end).astype(result.dtype)
    tm.assert_series_equal(result, expected)

    arrow_str_series = s.astype(pd.StringDtype(storage="pyarrow"))
    result2 = arrow_str_series.str.find(sub, start, end).astype(result.dtype)
    tm.assert_series_equal(result2, expected)


def test_str_find_negative_start_negative_end_no_match():
    # GH 56791
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-3, end=-6)
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "i, exp",
    [
        [1, ["b", "e", None]],
        [-1, ["c", "e", None]],
        [2, ["c", None, None]],
        [-3, ["a", None, None]],
        [4, [None, None, None]],
    ],
)
def test_str_get(i, exp):
    ser = pd.Series(["abc", "de", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.get(i)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.xfail(
    reason="TODO: StringMethods._validate should support Arrow list types",
    raises=AttributeError,
)
def test_str_join():
    ser = pd.Series(ArrowExtensionArray(pa.array([list("abc"), list("123"), None])))
    result = ser.str.join("=")
    expected = pd.Series(["a=b=c", "1=2=3", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_join_string_type():
    ser = pd.Series(ArrowExtensionArray(pa.array(["abc", "123", None])))
    result = ser.str.join("=")
    expected = pd.Series(["a=b=c", "1=2=3", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, step, exp",
    [
        [None, 2, None, ["ab", None]],
        [None, 2, 1, ["ab", None]],
        [1, 3, 1, ["bc", None]],
        (None, None, -1, ["dcba", None]),
    ],
)
def test_str_slice(start, stop, step, exp):
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.slice(start, stop, step)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, stop, repl, exp",
    [
        [1, 2, "x", ["axcd", None]],
        [None, 2, "x", ["xcd", None]],
        [None, 2, None, ["cd", None]],
    ],
)
def test_str_slice_replace(start, stop, repl, exp):
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.slice_replace(start, stop, repl)
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "value, method, exp",
    [
        ["a1c", "isalnum", True],
        ["!|,", "isalnum", False],
        ["aaa", "isalpha", True],
        ["!!!", "isalpha", False],
        ["٠", "isdecimal", True],  # noqa: RUF001
        ["~!", "isdecimal", False],
        ["2", "isdigit", True],
        ["~", "isdigit", False],
        ["aaa", "islower", True],
        ["aaA", "islower", False],
        ["123", "isnumeric", True],
        ["11I", "isnumeric", False],
        [" ", "isspace", True],
        ["", "isspace", False],
        ["The That", "istitle", True],
        ["the That", "istitle", False],
        ["AAA", "isupper", True],
        ["AAc", "isupper", False],
    ],
)
def test_str_is_functions(value, method, exp):
    ser = pd.Series([value, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp",
    [
        ["capitalize", "Abc def"],
        ["title", "Abc Def"],
        ["swapcase", "AbC Def"],
        ["lower", "abc def"],
        ["upper", "ABC DEF"],
        ["casefold", "abc def"],
    ],
)
def test_str_transform_functions(method, exp):
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_len():
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.len()
    expected = pd.Series([4, None], dtype=ArrowDtype(pa.int32()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "method, to_strip, val",
    [
        ["strip", None, " abc "],
        ["strip", "x", "xabcx"],
        ["lstrip", None, " abc"],
        ["lstrip", "x", "xabc"],
        ["rstrip", None, "abc "],
        ["rstrip", "x", "abcx"],
    ],
)
def test_str_strip(method, to_strip, val):
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)(to_strip=to_strip)
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("val", ["abc123", "abc"])
def test_str_removesuffix(val):
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removesuffix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("val", ["123abc", "abc"])
def test_str_removeprefix(val):
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    result = ser.str.removeprefix("123")
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("errors", ["ignore", "strict"])
@pytest.mark.parametrize(
    "encoding, exp",
    [
        ["utf8", b"abc"],
        ["utf32", b"\xff\xfe\x00\x00a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00"],
    ],
)
def test_str_encode(errors, encoding, exp):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.encode(encoding, errors)
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.binary()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flags", [0, 2])
def test_str_findall(flags):
    ser = pd.Series(["abc", "efg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.findall("b", flags=flags)
    expected = pd.Series([["b"], [], None], dtype=ArrowDtype(pa.list_(pa.string())))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["index", "rindex"])
@pytest.mark.parametrize(
    "start, end",
    [
        [0, None],
        [1, 4],
    ],
)
def test_str_r_index(method, start, end):
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)("c", start, end)
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)

    with pytest.raises(ValueError, match="substring not found"):
        getattr(ser.str, method)("foo", start, end)


@pytest.mark.parametrize("form", ["NFC", "NFKC"])
def test_str_normalize(form):
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.normalize(form)
    expected = ser.copy()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "start, end",
    [
        [0, None],
        [1, 4],
    ],
)
def test_str_rfind(start, end):
    ser = pd.Series(["abcba", "foo", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.rfind("c", start, end)
    expected = pd.Series([2, -1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_translate():
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.translate({97: "b"})
    expected = pd.Series(["bbcbb", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_str_wrap():
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.wrap(3)
    expected = pd.Series(["abc\nba", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_get_dummies():
    ser = pd.Series(["a|b", None, "a|c"], dtype=ArrowDtype(pa.string()))
    result = ser.str.get_dummies()
    expected = pd.DataFrame(
        [[True, True, False], [False, False, False], [True, False, True]],
        dtype=ArrowDtype(pa.bool_()),
        columns=["a", "b", "c"],
    )
    tm.assert_frame_equal(result, expected)


def test_str_partition():
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.partition("b")
    expected = pd.DataFrame(
        [["a", "b", "cba"], [None, None, None]], dtype=ArrowDtype(pa.string())
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.partition("b", expand=False)
    expected = pd.Series(ArrowExtensionArray(pa.array([["a", "b", "cba"], None])))
    tm.assert_series_equal(result, expected)

    result = ser.str.rpartition("b")
    expected = pd.DataFrame(
        [["abc", "b", "a"], [None, None, None]], dtype=ArrowDtype(pa.string())
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.rpartition("b", expand=False)
    expected = pd.Series(ArrowExtensionArray(pa.array([["abc", "b", "a"], None])))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["rsplit", "split"])
def test_str_split_pat_none(method):
    # GH 56271
    ser = pd.Series(["a1 cbc\nb", None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()
    expected = pd.Series(ArrowExtensionArray(pa.array([["a1", "cbc", "b"], None])))
    tm.assert_series_equal(result, expected)


def test_str_split():
    # GH 52401
    ser = pd.Series(["a1cbcb", "a2cbcb", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.split("c")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "b", "b"], ["a2", "b", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.split("c", n=1)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "bcb"], ["a2", "bcb"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.split("[1-2]", regex=True)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a", "cbcb"], ["a", "cbcb"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.split("[1-2]", regex=True, expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", "cbcb", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.split("1", expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a2cbcb", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", None, None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_rsplit():
    # GH 52401
    ser = pd.Series(["a1cbcb", "a2cbcb", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.rsplit("c")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "b", "b"], ["a2", "b", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.rsplit("c", n=1)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1cb", "b"], ["a2cb", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    result = ser.str.rsplit("c", n=1, expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a1cb", "a2cb", None])),
            1: ArrowExtensionArray(pa.array(["b", "b", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.rsplit("1", expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a2cbcb", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", None, None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_extract_non_symbolic():
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    with pytest.raises(ValueError, match="pat=.* must contain a symbolic group name."):
        ser.str.extract(r"[ab](\d)")


@pytest.mark.parametrize("expand", [True, False])
def test_str_extract(expand):
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    result = ser.str.extract(r"(?P<letter>[ab])(?P<digit>\d)", expand=expand)
    expected = pd.DataFrame(
        {
            "letter": ArrowExtensionArray(pa.array(["a", "b", None])),
            "digit": ArrowExtensionArray(pa.array(["1", "2", None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_extract_expand():
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    result = ser.str.extract(r"[ab](?P<digit>\d)", expand=True)
    expected = pd.DataFrame(
        {
            "digit": ArrowExtensionArray(pa.array(["1", "2", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    result = ser.str.extract(r"[ab](?P<digit>\d)", expand=False)
    expected = pd.Series(ArrowExtensionArray(pa.array(["1", "2", None])), name="digit")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_duration_from_strings_with_nat(unit):
    # GH51175
    strings = ["1000", "NaT"]
    pa_type = pa.duration(unit)
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=pa_type)
    expected = ArrowExtensionArray(pa.array([1000, None], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


def test_unsupported_dt(data):
    pa_dtype = data.dtype.pyarrow_dtype
    if not pa.types.is_temporal(pa_dtype):
        with pytest.raises(
            AttributeError, match="Can only use .dt accessor with datetimelike values"
        ):
            pd.Series(data).dt


@pytest.mark.parametrize(
    "prop, expected",
    [
        ["year", 2023],
        ["day", 2],
        ["day_of_week", 0],
        ["dayofweek", 0],
        ["weekday", 0],
        ["day_of_year", 2],
        ["dayofyear", 2],
        ["hour", 3],
        ["minute", 4],
        ["is_leap_year", False],
        ["microsecond", 5],
        ["month", 1],
        ["nanosecond", 6],
        ["quarter", 1],
        ["second", 7],
        ["date", date(2023, 1, 2)],
        ["time", time(3, 4, 7, 5)],
    ],
)
def test_dt_properties(prop, expected):
    ser = pd.Series(
        [
            pd.Timestamp(
                year=2023,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=7,
                microsecond=5,
                nanosecond=6,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = getattr(ser.dt, prop)
    exp_type = None
    if isinstance(expected, date):
        exp_type = pa.date32()
    elif isinstance(expected, time):
        exp_type = pa.time64("ns")
    expected = pd.Series(ArrowExtensionArray(pa.array([expected, None], type=exp_type)))
    tm.assert_series_equal(result, expected)


def test_dt_is_month_start_end():
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=2, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_month_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

    result = ser.dt.is_month_end
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


def test_dt_is_year_start_end():
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=31, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_year_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

    result = ser.dt.is_year_end
    expected = pd.Series([True, False, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


def test_dt_is_quarter_start_end():
    ser = pd.Series(
        [
            datetime(year=2023, month=11, day=30, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.is_quarter_start
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)

    result = ser.dt.is_quarter_end
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["days_in_month", "daysinmonth"])
def test_dt_days_in_month(method):
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30, hour=3),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = getattr(ser.dt, method)
    expected = pd.Series([31, 30, 28, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_dt_normalize():
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    result = ser.dt.normalize()
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit):
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    assert ser.dt.unit == unit

    result = ser.dt.time
    expected = pd.Series(
        ArrowExtensionArray(pa.array([time(3, 0), None], type=pa.time64(unit)))
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
def test_dt_tz(tz):
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", tz=tz)),
    )
    result = ser.dt.tz
    assert result == timezones.maybe_get_tz(tz)


def test_dt_isocalendar():
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.isocalendar()
    expected = pd.DataFrame(
        [[2023, 1, 1], [0, 0, 0]],
        columns=["year", "week", "day"],
        dtype="int64[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "method, exp", [["day_name", "Sunday"], ["month_name", "January"]]
)
def test_dt_day_month_name(method, exp, request):
    # GH 52388
    _require_timezone_database(request)

    ser = pd.Series([datetime(2023, 1, 1), None], dtype=ArrowDtype(pa.timestamp("ms")))
    result = getattr(ser.dt, method)()
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)


def test_dt_strftime(request):
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.strftime("%Y-%m-%dT%H:%M:%S")
    expected = pd.Series(
        ["2023-01-02T03:00:00.000000000", None], dtype=ArrowDtype(pa.string())
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_tz_options_not_supported(method):
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous is not supported."):
        getattr(ser.dt, method)("1h", ambiguous="NaT")

    with pytest.raises(NotImplementedError, match="nonexistent is not supported."):
        getattr(ser.dt, method)("1h", nonexistent="NaT")


@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_unsupported_freq(method):
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(ValueError, match="freq='1B' is not supported"):
        getattr(ser.dt, method)("1B")

    with pytest.raises(ValueError, match="Must specify a valid frequency: None"):
        getattr(ser.dt, method)(None)


@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_ceil_year_floor(freq, method):
    ser = pd.Series(
        [datetime(year=2023, month=1, day=1), None],
    )
    pa_dtype = ArrowDtype(pa.timestamp("ns"))
    expected = getattr(ser.dt, method)(f"1{freq}").astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f"1{freq}")
    tm.assert_series_equal(result, expected)


def test_dt_to_pydatetime():
    # GH 51859
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp("ns")))

    msg = "The behavior of ArrowTemporalProperties.to_pydatetime is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = ser.dt.to_pydatetime()
    expected = np.array(data, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert all(type(res) is datetime for res in result)

    msg = "The behavior of DatetimeProperties.to_pydatetime is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = ser.astype("datetime64[ns]").dt.to_pydatetime()
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("date_type", [32, 64])
def test_dt_to_pydatetime_date_error(date_type):
    # GH 52812
    ser = pd.Series(
        [date(2022, 12, 31)],
        dtype=ArrowDtype(getattr(pa, f"date{date_type}")()),
    )
    msg = "The behavior of ArrowTemporalProperties.to_pydatetime is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pytest.raises(ValueError, match="to_pydatetime cannot be called with"):
            ser.dt.to_pydatetime()


def test_dt_tz_localize_unsupported_tz_options():
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(NotImplementedError, match="ambiguous='NaT' is not supported"):
        ser.dt.tz_localize("UTC", ambiguous="NaT")

    with pytest.raises(NotImplementedError, match="nonexistent='NaT' is not supported"):
        ser.dt.tz_localize("UTC", nonexistent="NaT")


def test_dt_tz_localize_none():
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", tz="US/Pacific")),
    )
    result = ser.dt.tz_localize(None)
    expected = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_localize(unit, request):
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    result = ser.dt.tz_localize("US/Pacific")
    exp_data = pa.array(
        [datetime(year=2023, month=1, day=2, hour=3), None], type=pa.timestamp(unit)
    )
    exp_data = pa.compute.assume_timezone(exp_data, "US/Pacific")
    expected = pd.Series(ArrowExtensionArray(exp_data))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "nonexistent, exp_date",
    [
        ["shift_forward", datetime(year=2023, month=3, day=12, hour=3)],
        ["shift_backward", pd.Timestamp("2023-03-12 01:59:59.999999999")],
    ],
)
def test_dt_tz_localize_nonexistent(nonexistent, exp_date, request):
    _require_timezone_database(request)

    ser = pd.Series(
        [datetime(year=2023, month=3, day=12, hour=2, minute=30), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    result = ser.dt.tz_localize("US/Pacific", nonexistent=nonexistent)
    exp_data = pa.array([exp_date, None], type=pa.timestamp("ns"))
    exp_data = pa.compute.assume_timezone(exp_data, "US/Pacific")
    expected = pd.Series(ArrowExtensionArray(exp_data))
    tm.assert_series_equal(result, expected)


def test_dt_tz_convert_not_tz_raises():
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        ser.dt.tz_convert("UTC")


def test_dt_tz_convert_none():
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", "US/Pacific")),
    )
    result = ser.dt.tz_convert(None)
    expected = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_convert(unit):
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Pacific")),
    )
    result = ser.dt.tz_convert("US/Eastern")
    expected = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Eastern")),
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["timestamp[ms][pyarrow]", "duration[ms][pyarrow]"])
def test_as_unit(dtype):
    # GH 52284
    ser = pd.Series([1000, None], dtype=dtype)
    result = ser.dt.as_unit("ns")
    expected = ser.astype(dtype.replace("ms", "ns"))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "prop, expected",
    [
        ["days", 1],
        ["seconds", 2],
        ["microseconds", 3],
        ["nanoseconds", 4],
    ],
)
def test_dt_timedelta_properties(prop, expected):
    # GH 52284
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = getattr(ser.dt, prop)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([expected, None], type=pa.int32()))
    )
    tm.assert_series_equal(result, expected)


def test_dt_timedelta_total_seconds():
    # GH 52284
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.total_seconds()
    expected = pd.Series(
        ArrowExtensionArray(pa.array([86402.000003, None], type=pa.float64()))
    )
    tm.assert_series_equal(result, expected)


def test_dt_to_pytimedelta():
    # GH 52284
    data = [timedelta(1, 2, 3), timedelta(1, 2, 4)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.duration("ns")))

    result = ser.dt.to_pytimedelta()
    expected = np.array(data, dtype=object)
    tm.assert_numpy_array_equal(result, expected)
    assert all(type(res) is timedelta for res in result)

    expected = ser.astype("timedelta64[ns]").dt.to_pytimedelta()
    tm.assert_numpy_array_equal(result, expected)


def test_dt_components():
    # GH 52284
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    result = ser.dt.components
    expected = pd.DataFrame(
        [[1, 0, 0, 2, 0, 3, 4], [None, None, None, None, None, None, None]],
        columns=[
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ],
        dtype="int32[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("skipna", [True, False])
def test_boolean_reduce_series_all_null(all_boolean_reductions, skipna):
    # GH51624
    ser = pd.Series([None], dtype="float64[pyarrow]")
    result = getattr(ser, all_boolean_reductions)(skipna=skipna)
    if skipna:
        expected = all_boolean_reductions == "all"
    else:
        expected = pd.NA
    assert result is expected


def test_from_sequence_of_strings_boolean():
    true_strings = ["true", "TRUE", "True", "1", "1.0"]
    false_strings = ["false", "FALSE", "False", "0", "0.0"]
    nulls = [None]
    strings = true_strings + false_strings + nulls
    bools = (
        [True] * len(true_strings) + [False] * len(false_strings) + [None] * len(nulls)
    )

    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=pa.bool_())
    expected = pd.array(bools, dtype="boolean[pyarrow]")
    tm.assert_extension_array_equal(result, expected)

    strings = ["True", "foo"]
    with pytest.raises(pa.ArrowInvalid, match="Failed to parse"):
        ArrowExtensionArray._from_sequence_of_strings(strings, dtype=pa.bool_())


def test_concat_empty_arrow_backed_series(dtype):
    # GH#51734
    ser = pd.Series([], dtype=dtype)
    expected = ser.copy()
    result = pd.concat([ser[np.array([], dtype=np.bool_)]])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]"])
def test_series_from_string_array(dtype):
    arr = pa.array("the quick brown fox".split())
    ser = pd.Series(arr, dtype=dtype)
    expected = pd.Series(ArrowExtensionArray(arr), dtype=dtype)
    tm.assert_series_equal(ser, expected)


# _data was renamed to _pa_data
class OldArrowExtensionArray(ArrowExtensionArray):
    def __getstate__(self):
        state = super().__getstate__()
        state["_data"] = state.pop("_pa_array")
        return state


def test_pickle_old_arrowextensionarray():
    data = pa.array([1])
    expected = OldArrowExtensionArray(data)
    result = pickle.loads(pickle.dumps(expected))
    tm.assert_extension_array_equal(result, expected)
    assert result._pa_array == pa.chunked_array(data)
    assert not hasattr(result, "_data")


def test_setitem_boolean_replace_with_mask_segfault():
    # GH#52059
    N = 145_000
    arr = ArrowExtensionArray(pa.chunked_array([np.ones((N,), dtype=np.bool_)]))
    expected = arr.copy()
    arr[np.zeros((N,), dtype=np.bool_)] = False
    assert arr._pa_array == expected._pa_array


@pytest.mark.parametrize(
    "data, arrow_dtype",
    [
        ([b"a", b"b"], pa.large_binary()),
        (["a", "b"], pa.large_string()),
    ],
)
def test_conversion_large_dtypes_from_numpy_array(data, arrow_dtype):
    dtype = ArrowDtype(arrow_dtype)
    result = pd.array(np.array(data), dtype=dtype)
    expected = pd.array(data, dtype=dtype)
    tm.assert_extension_array_equal(result, expected)


def test_concat_null_array():
    df = pd.DataFrame({"a": [None, None]}, dtype=ArrowDtype(pa.null()))
    df2 = pd.DataFrame({"a": [0, 1]}, dtype="int64[pyarrow]")

    result = pd.concat([df, df2], ignore_index=True)
    expected = pd.DataFrame({"a": [None, None, 0, 1]}, dtype="int64[pyarrow]")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES + tm.FLOAT_PYARROW_DTYPES)
def test_describe_numeric_data(pa_type):
    # GH 52470
    data = pd.Series([1, 2, 3], dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series(
        [3, 2, 1, 1, 1.5, 2.0, 2.5, 3],
        dtype=ArrowDtype(pa.float64()),
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.TIMEDELTA_PYARROW_DTYPES)
def test_describe_timedelta_data(pa_type):
    # GH53001
    data = pd.Series(range(1, 10), dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series(
        [9] + pd.to_timedelta([5, 2, 1, 3, 5, 7, 9], unit=pa_type.unit).tolist(),
        dtype=object,
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.DATETIME_PYARROW_DTYPES)
def test_describe_datetime_data(pa_type):
    # GH53001
    data = pd.Series(range(1, 10), dtype=ArrowDtype(pa_type))
    result = data.describe()
    expected = pd.Series(
        [9]
        + [
            pd.Timestamp(v, tz=pa_type.tz, unit=pa_type.unit)
            for v in [5, 1, 3, 5, 7, 9]
        ],
        dtype=object,
        index=["count", "mean", "min", "25%", "50%", "75%", "max"],
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_quantile_temporal(pa_type):
    # GH52678
    data = [1, 2, 3]
    ser = pd.Series(data, dtype=ArrowDtype(pa_type))
    result = ser.quantile(0.1)
    expected = ser[0]
    assert result == expected


def test_date32_repr():
    # GH48238
    arrow_dt = pa.array([date.fromisoformat("2020-01-01")], type=pa.date32())
    ser = pd.Series(arrow_dt, dtype=ArrowDtype(arrow_dt.type))
    assert repr(ser) == "0    2020-01-01\ndtype: date32[day][pyarrow]"


def test_duration_overflow_from_ndarray_containing_nat():
    # GH52843
    data_ts = pd.to_datetime([1, None])
    data_td = pd.to_timedelta([1, None])
    ser_ts = pd.Series(data_ts, dtype=ArrowDtype(pa.timestamp("ns")))
    ser_td = pd.Series(data_td, dtype=ArrowDtype(pa.duration("ns")))
    result = ser_ts + ser_td
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.timestamp("ns")))
    tm.assert_series_equal(result, expected)


def test_infer_dtype_pyarrow_dtype(data, request):
    res = lib.infer_dtype(data)
    assert res != "unknown-array"

    if data._hasna and res in ["floating", "datetime64", "timedelta64"]:
        mark = pytest.mark.xfail(
            reason="in infer_dtype pd.NA is not ignored in these cases "
            "even with skipna=True in the list(data) check below"
        )
        request.applymarker(mark)

    assert res == lib.infer_dtype(list(data), skipna=True)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_from_sequence_temporal(pa_type):
    # GH 53171
    val = 3
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        seq = [pd.Timedelta(val, unit=unit).as_unit(unit)]
    else:
        seq = [pd.Timestamp(val, unit=unit, tz=pa_type.tz).as_unit(unit)]

    result = ArrowExtensionArray._from_sequence(seq, dtype=pa_type)
    expected = ArrowExtensionArray(pa.array([val], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_setitem_temporal(pa_type):
    # GH 53171
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        val = pd.Timedelta(1, unit=unit).as_unit(unit)
    else:
        val = pd.Timestamp(1, unit=unit, tz=pa_type.tz).as_unit(unit)

    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    result = arr.copy()
    result[:] = val
    expected = ArrowExtensionArray(pa.array([1, 1, 1], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_arithmetic_temporal(pa_type, request):
    # GH 53171
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))
    unit = pa_type.unit
    result = arr - pd.Timedelta(1, unit=unit).as_unit(unit)
    expected = ArrowExtensionArray(pa.array([0, 1, 2], type=pa_type))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_comparison_temporal(pa_type):
    # GH 53171
    unit = pa_type.unit
    if pa.types.is_duration(pa_type):
        val = pd.Timedelta(1, unit=unit).as_unit(unit)
    else:
        val = pd.Timestamp(1, unit=unit, tz=pa_type.tz).as_unit(unit)

    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    result = arr > val
    expected = ArrowExtensionArray(pa.array([False, True, True], type=pa.bool_()))
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_getitem_temporal(pa_type):
    # GH 53326
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))
    result = arr[1]
    if pa.types.is_duration(pa_type):
        expected = pd.Timedelta(2, unit=pa_type.unit).as_unit(pa_type.unit)
        assert isinstance(result, pd.Timedelta)
    else:
        expected = pd.Timestamp(2, unit=pa_type.unit, tz=pa_type.tz).as_unit(
            pa_type.unit
        )
        assert isinstance(result, pd.Timestamp)
    assert result.unit == expected.unit
    assert result == expected


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_iter_temporal(pa_type):
    # GH 53326
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))
    result = list(arr)
    if pa.types.is_duration(pa_type):
        expected = [
            pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit),
            pd.NA,
        ]
        assert isinstance(result[0], pd.Timedelta)
    else:
        expected = [
            pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit),
            pd.NA,
        ]
        assert isinstance(result[0], pd.Timestamp)
    assert result[0].unit == expected[0].unit
    assert result == expected


def test_groupby_series_size_returns_pa_int(data):
    # GH 54132
    ser = pd.Series(data[:3], index=["a", "a", "b"])
    result = ser.groupby(level=0).size()
    expected = pd.Series([2, 1], dtype="int64[pyarrow]", index=["a", "b"])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES, ids=repr
)
@pytest.mark.parametrize("dtype", [None, object])
def test_to_numpy_temporal(pa_type, dtype):
    # GH 53326
    # GH 55997: Return datetime64/timedelta64 types with NaT if possible
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))
    result = arr.to_numpy(dtype=dtype)
    if pa.types.is_duration(pa_type):
        value = pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit)
    else:
        value = pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit)

    if dtype == object or (pa.types.is_timestamp(pa_type) and pa_type.tz is not None):
        if dtype == object:
            na = pd.NA
        else:
            na = pd.NaT
        expected = np.array([value, na], dtype=object)
        assert result[0].unit == value.unit
    else:
        na = pa_type.to_pandas_dtype().type("nat", pa_type.unit)
        value = value.to_numpy()
        expected = np.array([value, na])
        assert np.datetime_data(result[0])[0] == pa_type.unit
    tm.assert_numpy_array_equal(result, expected)


def test_groupby_count_return_arrow_dtype(data_missing):
    df = pd.DataFrame({"A": [1, 1], "B": data_missing, "C": data_missing})
    result = df.groupby("A").count()
    expected = pd.DataFrame(
        [[1, 1]],
        index=pd.Index([1], name="A"),
        columns=["B", "C"],
        dtype="int64[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)


def test_fixed_size_list():
    # GH#55000
    ser = pd.Series(
        [[1, 2], [3, 4]], dtype=ArrowDtype(pa.list_(pa.int64(), list_size=2))
    )
    result = ser.dtype.type
    assert result == list


def test_arrowextensiondtype_dataframe_repr():
    # GH 54062
    df = pd.DataFrame(
        pd.period_range("2012", periods=3),
        columns=["col"],
        dtype=ArrowDtype(ArrowPeriodType("D")),
    )
    result = repr(df)
    # TODO: repr value may not be expected; address how
    # pyarrow.ExtensionType values are displayed
    expected = "     col\n0  15340\n1  15341\n2  15342"
    assert result == expected


def test_pow_missing_operand():
    # GH 55512
    k = pd.Series([2, None], dtype="int64[pyarrow]")
    result = k.pow(None, fill_value=3)
    expected = pd.Series([8, None], dtype="int64[pyarrow]")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.TIMEDELTA_PYARROW_DTYPES)
def test_duration_fillna_numpy(pa_type):
    # GH 54707
    ser1 = pd.Series([None, 2], dtype=ArrowDtype(pa_type))
    ser2 = pd.Series(np.array([1, 3], dtype=f"m8[{pa_type.unit}]"))
    result = ser1.fillna(ser2)
    expected = pd.Series([1, 2], dtype=ArrowDtype(pa_type))
    tm.assert_series_equal(result, expected)


def test_comparison_not_propagating_arrow_error():
    # GH#54944
    a = pd.Series([1 << 63], dtype="uint64[pyarrow]")
    b = pd.Series([None], dtype="int64[pyarrow]")
    with pytest.raises(pa.lib.ArrowInvalid, match="Integer value"):
        a < b


def test_factorize_chunked_dictionary():
    # GH 54844
    pa_array = pa.chunked_array(
        [pa.array(["a"]).dictionary_encode(), pa.array(["b"]).dictionary_encode()]
    )
    ser = pd.Series(ArrowExtensionArray(pa_array))
    res_indices, res_uniques = ser.factorize()
    exp_indicies = np.array([0, 1], dtype=np.intp)
    exp_uniques = pd.Index(ArrowExtensionArray(pa_array.combine_chunks()))
    tm.assert_numpy_array_equal(res_indices, exp_indicies)
    tm.assert_index_equal(res_uniques, exp_uniques)


def test_dictionary_astype_categorical():
    # GH#56672
    arrs = [
        pa.array(np.array(["a", "x", "c", "a"])).dictionary_encode(),
        pa.array(np.array(["a", "d", "c"])).dictionary_encode(),
    ]
    ser = pd.Series(ArrowExtensionArray(pa.chunked_array(arrs)))
    result = ser.astype("category")
    categories = pd.Index(["a", "x", "c", "d"], dtype=ArrowDtype(pa.string()))
    expected = pd.Series(
        ["a", "x", "c", "a", "a", "d", "c"],
        dtype=pd.CategoricalDtype(categories=categories),
    )
    tm.assert_series_equal(result, expected)


def test_arrow_floordiv():
    # GH 55561
    a = pd.Series([-7], dtype="int64[pyarrow]")
    b = pd.Series([4], dtype="int64[pyarrow]")
    expected = pd.Series([-2], dtype="int64[pyarrow]")
    result = a // b
    tm.assert_series_equal(result, expected)


def test_arrow_floordiv_large_values():
    # GH 56645
    a = pd.Series([1425801600000000000], dtype="int64[pyarrow]")
    expected = pd.Series([1425801600000], dtype="int64[pyarrow]")
    result = a // 1_000_000
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_floordiv_large_integral_result(dtype):
    # GH 56676
    a = pd.Series([18014398509481983], dtype=dtype)
    result = a // 1
    tm.assert_series_equal(result, a)


@pytest.mark.parametrize("pa_type", tm.SIGNED_INT_PYARROW_DTYPES)
def test_arrow_floordiv_larger_divisor(pa_type):
    # GH 56676
    dtype = ArrowDtype(pa_type)
    a = pd.Series([-23], dtype=dtype)
    result = a // 24
    expected = pd.Series([-1], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.SIGNED_INT_PYARROW_DTYPES)
def test_arrow_floordiv_integral_invalid(pa_type):
    # GH 56676
    min_value = np.iinfo(pa_type.to_pandas_dtype()).min
    a = pd.Series([min_value], dtype=ArrowDtype(pa_type))
    with pytest.raises(pa.lib.ArrowInvalid, match="overflow|not in range"):
        a // -1
    with pytest.raises(pa.lib.ArrowInvalid, match="divide by zero"):
        a // 0


@pytest.mark.parametrize("dtype", tm.FLOAT_PYARROW_DTYPES_STR_REPR)
def test_arrow_floordiv_floating_0_divisor(dtype):
    # GH 56676
    a = pd.Series([2], dtype=dtype)
    result = a // 0
    expected = pd.Series([float("inf")], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "datetime64[ns]", "timedelta64[ns]"])
def test_astype_int_with_null_to_numpy_dtype(dtype):
    # GH 57093
    ser = pd.Series([1, None], dtype="int64[pyarrow]")
    result = ser.astype(dtype)
    expected = pd.Series([1, None], dtype=dtype)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
def test_arrow_integral_floordiv_large_values(pa_type):
    # GH 56676
    max_value = np.iinfo(pa_type.to_pandas_dtype()).max
    dtype = ArrowDtype(pa_type)
    a = pd.Series([max_value], dtype=dtype)
    b = pd.Series([1], dtype=dtype)
    result = a // b
    tm.assert_series_equal(result, a)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_true_division_large_divisor(dtype):
    # GH 56706
    a = pd.Series([0], dtype=dtype)
    b = pd.Series([18014398509481983], dtype=dtype)
    expected = pd.Series([0], dtype="float64[pyarrow]")
    result = a / b
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_floor_division_large_divisor(dtype):
    # GH 56706
    a = pd.Series([0], dtype=dtype)
    b = pd.Series([18014398509481983], dtype=dtype)
    expected = pd.Series([0], dtype=dtype)
    result = a // b
    tm.assert_series_equal(result, expected)


def test_string_to_datetime_parsing_cast():
    # GH 56266
    string_dates = ["2020-01-01 04:30:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"]
    result = pd.Series(string_dates, dtype="timestamp[ns][pyarrow]")
    expected = pd.Series(
        ArrowExtensionArray(pa.array(pd.to_datetime(string_dates), from_pandas=True))
    )
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(
    pa_version_under13p0, reason="pairwise_diff_checked not implemented in pyarrow"
)
def test_interpolate_not_numeric(data):
    if not data.dtype._is_numeric:
        ser = pd.Series(data)
        msg = re.escape(f"Cannot interpolate with {ser.dtype} dtype")
        with pytest.raises(TypeError, match=msg):
            pd.Series(data).interpolate()


def test_string_to_time_parsing_cast():
    # GH 56463
    string_times = ["11:41:43.076160"]
    result = pd.Series(string_times, dtype="time64[us][pyarrow]")
    expected = pd.Series(
        ArrowExtensionArray(pa.array([time(11, 41, 43, 76160)], from_pandas=True))
    )
    tm.assert_series_equal(result, expected)


def test_to_numpy_float():
    # GH#56267
    ser = pd.Series([32, 40, None], dtype="float[pyarrow]")
    result = ser.astype("float64")
    expected = pd.Series([32, 40, np.nan], dtype="float64")
    tm.assert_series_equal(result, expected)


def test_to_numpy_timestamp_to_int():
    # GH 55997
    ser = pd.Series(["2020-01-01 04:30:00"], dtype="timestamp[ns][pyarrow]")
    result = ser.to_numpy(dtype=np.int64)
    expected = np.array([1577853000000000000])
    tm.assert_numpy_array_equal(result, expected)


def test_map_numeric_na_action():
    ser = pd.Series([32, 40, None], dtype="int64[pyarrow]")
    result = ser.map(lambda x: 42, na_action="ignore")
    expected = pd.Series([42.0, 42.0, np.nan], dtype="float64")
    tm.assert_series_equal(result, expected)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\query.py ===
from __future__ import annotations

import typing as t

import sqlalchemy.exc as sa_exc
import sqlalchemy.orm as sa_orm
from flask import abort

from .pagination import Pagination
from .pagination import QueryPagination


class Query(sa_orm.Query):  # type: ignore[type-arg]
    """SQLAlchemy :class:`~sqlalchemy.orm.query.Query` subclass with some extra methods
    useful for querying in a web application.

    This is the default query class for :attr:`.Model.query`.

    .. versionchanged:: 3.0
        Renamed to ``Query`` from ``BaseQuery``.
    """

    def get_or_404(self, ident: t.Any, description: str | None = None) -> t.Any:
        """Like :meth:`~sqlalchemy.orm.Query.get` but aborts with a ``404 Not Found``
        error instead of returning ``None``.

        :param ident: The primary key to query.
        :param description: A custom message to show on the error page.
        """
        rv = self.get(ident)

        if rv is None:
            abort(404, description=description)

        return rv

    def first_or_404(self, description: str | None = None) -> t.Any:
        """Like :meth:`~sqlalchemy.orm.Query.first` but aborts with a ``404 Not Found``
        error instead of returning ``None``.

        :param description: A custom message to show on the error page.
        """
        rv = self.first()

        if rv is None:
            abort(404, description=description)

        return rv

    def one_or_404(self, description: str | None = None) -> t.Any:
        """Like :meth:`~sqlalchemy.orm.Query.one` but aborts with a ``404 Not Found``
        error instead of raising ``NoResultFound`` or ``MultipleResultsFound``.

        :param description: A custom message to show on the error page.

        .. versionadded:: 3.0
        """
        try:
            return self.one()
        except (sa_exc.NoResultFound, sa_exc.MultipleResultsFound):
            abort(404, description=description)

    def paginate(
        self,
        *,
        page: int | None = None,
        per_page: int | None = None,
        max_per_page: int | None = None,
        error_out: bool = True,
        count: bool = True,
    ) -> Pagination:
        """Apply an offset and limit to the query based on the current page and number
        of items per page, returning a :class:`.Pagination` object.

        :param page: The current page, used to calculate the offset. Defaults to the
            ``page`` query arg during a request, or 1 otherwise.
        :param per_page: The maximum number of items on a page, used to calculate the
            offset and limit. Defaults to the ``per_page`` query arg during a request,
            or 20 otherwise.
        :param max_per_page: The maximum allowed value for ``per_page``, to limit a
            user-provided value. Use ``None`` for no limit. Defaults to 100.
        :param error_out: Abort with a ``404 Not Found`` error if no items are returned
            and ``page`` is not 1, or if ``page`` or ``per_page`` is less than 1, or if
            either are not ints.
        :param count: Calculate the total number of values by issuing an extra count
            query. For very complex queries this may be inaccurate or slow, so it can be
            disabled and set manually if necessary.

        .. versionchanged:: 3.0
            All parameters are keyword-only.

        .. versionchanged:: 3.0
            The ``count`` query is more efficient.

        .. versionchanged:: 3.0
            ``max_per_page`` defaults to 100.
        """
        return QueryPagination(
            query=self,
            page=page,
            per_page=per_page,
            max_per_page=max_per_page,
            error_out=error_out,
            count=count,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\RuntimeOptimizationRecord.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# a single runtime optimization
# see corresponding type in onnxruntime/core/graph/runtime_optimization_record.h
class RuntimeOptimizationRecord(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RuntimeOptimizationRecord()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRuntimeOptimizationRecord(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def RuntimeOptimizationRecordBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # RuntimeOptimizationRecord
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RuntimeOptimizationRecord
    def ActionId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # RuntimeOptimizationRecord
    def NodesToOptimizeIndices(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.NodesToOptimizeIndices import NodesToOptimizeIndices
            obj = NodesToOptimizeIndices()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RuntimeOptimizationRecord
    def ProducedOpIds(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # RuntimeOptimizationRecord
    def ProducedOpIdsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RuntimeOptimizationRecord
    def ProducedOpIdsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

def RuntimeOptimizationRecordStart(builder):
    builder.StartObject(4)

def Start(builder):
    RuntimeOptimizationRecordStart(builder)

def RuntimeOptimizationRecordAddActionId(builder, actionId):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(actionId), 0)

def AddActionId(builder, actionId):
    RuntimeOptimizationRecordAddActionId(builder, actionId)

def RuntimeOptimizationRecordAddNodesToOptimizeIndices(builder, nodesToOptimizeIndices):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(nodesToOptimizeIndices), 0)

def AddNodesToOptimizeIndices(builder, nodesToOptimizeIndices):
    RuntimeOptimizationRecordAddNodesToOptimizeIndices(builder, nodesToOptimizeIndices)

def RuntimeOptimizationRecordAddProducedOpIds(builder, producedOpIds):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(producedOpIds), 0)

def AddProducedOpIds(builder, producedOpIds):
    RuntimeOptimizationRecordAddProducedOpIds(builder, producedOpIds)

def RuntimeOptimizationRecordStartProducedOpIdsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartProducedOpIdsVector(builder, numElems: int) -> int:
    return RuntimeOptimizationRecordStartProducedOpIdsVector(builder, numElems)

def RuntimeOptimizationRecordEnd(builder):
    return builder.EndObject()

def End(builder):
    return RuntimeOptimizationRecordEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\_3d.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import Typed, Alias
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.nested import (
    NestedBool,
    NestedInteger,
    NestedMinMax,
)
from openpyxl.descriptors.excel import ExtensionList
from .marker import PictureOptions
from .shapes import GraphicalProperties


class View3D(Serialisable):

    tagname = "view3D"

    rotX = NestedMinMax(min=-90, max=90, allow_none=True)
    x_rotation = Alias('rotX')
    hPercent = NestedMinMax(min=5, max=500, allow_none=True)
    height_percent = Alias('hPercent')
    rotY = NestedInteger(min=-90, max=90, allow_none=True)
    y_rotation = Alias('rotY')
    depthPercent = NestedInteger(allow_none=True)
    rAngAx = NestedBool(allow_none=True)
    right_angle_axes = Alias('rAngAx')
    perspective = NestedInteger(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('rotX', 'hPercent', 'rotY', 'depthPercent', 'rAngAx',
                    'perspective',)

    def __init__(self,
                 rotX=15,
                 hPercent=None,
                 rotY=20,
                 depthPercent=None,
                 rAngAx=True,
                 perspective=None,
                 extLst=None,
                ):
        self.rotX = rotX
        self.hPercent = hPercent
        self.rotY = rotY
        self.depthPercent = depthPercent
        self.rAngAx = rAngAx
        self.perspective = perspective


class Surface(Serialisable):

    tagname = "surface"

    thickness = NestedInteger(allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')
    pictureOptions = Typed(expected_type=PictureOptions, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('thickness', 'spPr', 'pictureOptions',)

    def __init__(self,
                 thickness=None,
                 spPr=None,
                 pictureOptions=None,
                 extLst=None,
                ):
        self.thickness = thickness
        self.spPr = spPr
        self.pictureOptions = pictureOptions


class _3DBase(Serialisable):

    """
    Base class for 3D charts
    """

    tagname = "ChartBase"

    view3D = Typed(expected_type=View3D, allow_none=True)
    floor = Typed(expected_type=Surface, allow_none=True)
    sideWall = Typed(expected_type=Surface, allow_none=True)
    backWall = Typed(expected_type=Surface, allow_none=True)

    def __init__(self,
                 view3D=None,
                 floor=None,
                 sideWall=None,
                 backWall=None,
                 ):
        if view3D is None:
            view3D = View3D()
        self.view3D = view3D
        if floor is None:
            floor = Surface()
        self.floor = floor
        if sideWall is None:
            sideWall = Surface()
        self.sideWall = sideWall
        if backWall is None:
            backWall = Surface()
        self.backWall = backWall
        super(_3DBase, self).__init__()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\scenario.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    String,
    Integer,
    Bool,
    Sequence,
    Convertible,
)
from .cell_range import MultiCellRange


class InputCells(Serialisable):

    tagname = "inputCells"

    r = String()
    deleted = Bool(allow_none=True)
    undone = Bool(allow_none=True)
    val = String()
    numFmtId = Integer(allow_none=True)

    def __init__(self,
                 r=None,
                 deleted=False,
                 undone=False,
                 val=None,
                 numFmtId=None,
                ):
        self.r = r
        self.deleted = deleted
        self.undone = undone
        self.val = val
        self.numFmtId = numFmtId


class Scenario(Serialisable):

    tagname = "scenario"

    inputCells = Sequence(expected_type=InputCells)
    name = String()
    locked = Bool(allow_none=True)
    hidden = Bool(allow_none=True)
    user = String(allow_none=True)
    comment = String(allow_none=True)

    __elements__ = ('inputCells',)
    __attrs__ = ('name', 'locked', 'hidden', 'user', 'comment', 'count')

    def __init__(self,
                 inputCells=(),
                 name=None,
                 locked=False,
                 hidden=False,
                 count=None,
                 user=None,
                 comment=None,
                ):
        self.inputCells = inputCells
        self.name = name
        self.locked = locked
        self.hidden = hidden
        self.user = user
        self.comment = comment


    @property
    def count(self):
        return len(self.inputCells)


class ScenarioList(Serialisable):

    tagname = "scenarios"

    scenario = Sequence(expected_type=Scenario)
    current = Integer(allow_none=True)
    show = Integer(allow_none=True)
    sqref = Convertible(expected_type=MultiCellRange, allow_none=True)

    __elements__ = ('scenario',)

    def __init__(self,
                 scenario=(),
                 current=None,
                 show=None,
                 sqref=None,
                ):
        self.scenario = scenario
        self.current = current
        self.show = show
        self.sqref = sqref


    def append(self, scenario):
        s = self.scenario
        s.append(scenario)
        self.scenario = s


    def __bool__(self):
        return bool(self.scenario)


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\console.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Console
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class ConsoleMessage:
    '''
    Console message.
    '''
    #: Message source.
    source: str

    #: Message severity.
    level: str

    #: Message text.
    text: str

    #: URL of the message origin.
    url: typing.Optional[str] = None

    #: Line number in the resource that generated this message (1-based).
    line: typing.Optional[int] = None

    #: Column number in the resource that generated this message (1-based).
    column: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['source'] = self.source
        json['level'] = self.level
        json['text'] = self.text
        if self.url is not None:
            json['url'] = self.url
        if self.line is not None:
            json['line'] = self.line
        if self.column is not None:
            json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source=str(json['source']),
            level=str(json['level']),
            text=str(json['text']),
            url=str(json['url']) if 'url' in json else None,
            line=int(json['line']) if 'line' in json else None,
            column=int(json['column']) if 'column' in json else None,
        )


def clear_messages() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Does nothing.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.clearMessages',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables console domain, prevents further console messages from being reported to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables console domain, sends the messages collected so far to the client by means of the
    ``messageAdded`` notification.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.enable',
    }
    json = yield cmd_dict


@event_class('Console.messageAdded')
@dataclass
class MessageAdded:
    '''
    Issued when new console message is added.
    '''
    #: Console message that has been added.
    message: ConsoleMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> MessageAdded:
        return cls(
            message=ConsoleMessage.from_json(json['message'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\console.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Console
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class ConsoleMessage:
    '''
    Console message.
    '''
    #: Message source.
    source: str

    #: Message severity.
    level: str

    #: Message text.
    text: str

    #: URL of the message origin.
    url: typing.Optional[str] = None

    #: Line number in the resource that generated this message (1-based).
    line: typing.Optional[int] = None

    #: Column number in the resource that generated this message (1-based).
    column: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['source'] = self.source
        json['level'] = self.level
        json['text'] = self.text
        if self.url is not None:
            json['url'] = self.url
        if self.line is not None:
            json['line'] = self.line
        if self.column is not None:
            json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source=str(json['source']),
            level=str(json['level']),
            text=str(json['text']),
            url=str(json['url']) if 'url' in json else None,
            line=int(json['line']) if 'line' in json else None,
            column=int(json['column']) if 'column' in json else None,
        )


def clear_messages() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Does nothing.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.clearMessages',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables console domain, prevents further console messages from being reported to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables console domain, sends the messages collected so far to the client by means of the
    ``messageAdded`` notification.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.enable',
    }
    json = yield cmd_dict


@event_class('Console.messageAdded')
@dataclass
class MessageAdded:
    '''
    Issued when new console message is added.
    '''
    #: Console message that has been added.
    message: ConsoleMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> MessageAdded:
        return cls(
            message=ConsoleMessage.from_json(json['message'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\console.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Console
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class ConsoleMessage:
    '''
    Console message.
    '''
    #: Message source.
    source: str

    #: Message severity.
    level: str

    #: Message text.
    text: str

    #: URL of the message origin.
    url: typing.Optional[str] = None

    #: Line number in the resource that generated this message (1-based).
    line: typing.Optional[int] = None

    #: Column number in the resource that generated this message (1-based).
    column: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['source'] = self.source
        json['level'] = self.level
        json['text'] = self.text
        if self.url is not None:
            json['url'] = self.url
        if self.line is not None:
            json['line'] = self.line
        if self.column is not None:
            json['column'] = self.column
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            source=str(json['source']),
            level=str(json['level']),
            text=str(json['text']),
            url=str(json['url']) if 'url' in json else None,
            line=int(json['line']) if 'line' in json else None,
            column=int(json['column']) if 'column' in json else None,
        )


def clear_messages() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Does nothing.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.clearMessages',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables console domain, prevents further console messages from being reported to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables console domain, sends the messages collected so far to the client by means of the
    ``messageAdded`` notification.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Console.enable',
    }
    json = yield cmd_dict


@event_class('Console.messageAdded')
@dataclass
class MessageAdded:
    '''
    Issued when new console message is added.
    '''
    #: Console message that has been added.
    message: ConsoleMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> MessageAdded:
        return cls(
            message=ConsoleMessage.from_json(json['message'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\utils.py ===
from collections import OrderedDict


def expand_tuples(L):
    """
    >>> from sympy.multipledispatch.utils import expand_tuples
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]

    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
    if not L:
        return [()]
    elif not isinstance(L[0], tuple):
        rest = expand_tuples(L[1:])
        return [(L[0],) + t for t in rest]
    else:
        rest = expand_tuples(L[1:])
        return [(item,) + t for t in rest for item in L[0]]


# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)

    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges

    >>> from sympy.multipledispatch.utils import _toposort
    >>> _toposort({1: (2, 3), 2: (3, )})
    [1, 2, 3]

    Closely follows the wikipedia page [2]

    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] https://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_edges)
    L = []

    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S[m] = None
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


def reverse_dict(d):
    """Reverses direction of dependence dict

    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.

    """
    result = {}
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, ()) + (key, )
    return result


# Taken from toolz
# Avoids licensing issues because this version was authored by Matthew Rocklin
def groupby(func, seq):
    """ Group a collection by a key function

    >>> from sympy.multipledispatch.utils import groupby
    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}

    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

    See Also:
        ``countby``
    """

    d = {}
    for item in seq:
        key = func(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\gmpyintegerring.py ===
"""Implementation of :class:`GMPYIntegerRing` class. """


from sympy.polys.domains.groundtypes import (
    GMPYInteger, SymPyInteger,
    factorial as gmpy_factorial,
    gmpy_gcdex, gmpy_gcd, gmpy_lcm, sqrt as gmpy_sqrt,
)
from sympy.core.numbers import int_valued
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

@public
class GMPYIntegerRing(IntegerRing):
    """Integer ring based on GMPY's ``mpz`` type.

    This will be the implementation of :ref:`ZZ` if ``gmpy`` or ``gmpy2`` is
    installed. Elements will be of type ``gmpy.mpz``.
    """

    dtype = GMPYInteger
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)
    alias = 'ZZ_gmpy'

    def __init__(self):
        """Allow instantiation of this domain. """

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return GMPYInteger(a.p)
        elif int_valued(a):
            return GMPYInteger(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return K0.to_int(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return GMPYInteger(a)

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return GMPYInteger(a.numerator)

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return GMPYInteger(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        return K0.to_int(a)

    def from_ZZ_gmpy(K1, a, K0):
        """Convert GMPY's ``mpz`` to GMPY's ``mpz``. """
        return a

    def from_QQ_gmpy(K1, a, K0):
        """Convert GMPY ``mpq`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return a.numerator

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to GMPY's ``mpz``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return GMPYInteger(p)

    def from_GaussianIntegerRing(K1, a, K0):
        if a.y == 0:
            return a.x

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        h, s, t = gmpy_gcdex(a, b)
        return s, t, h

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return gmpy_gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return gmpy_lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return gmpy_sqrt(a)

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return gmpy_factorial(a)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\sampling\sample_numpy.py ===
from functools import singledispatch

from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, FDistributionDistribution, GumbelDistribution, LaplaceDistribution, \
    LogisticDistribution, RayleighDistribution, TriangularDistribution
from sympy.stats.drv_types import GeometricDistribution, PoissonDistribution, ZetaDistribution
from sympy.stats.frv_types import BinomialDistribution, HypergeometricDistribution


numpy = import_module('numpy')


@singledispatch
def do_sample_numpy(dist, size, rand_state):
    return None


# CRV:

@do_sample_numpy.register(BetaDistribution)
def _(dist: BetaDistribution, size, rand_state):
    return rand_state.beta(a=float(dist.alpha), b=float(dist.beta), size=size)


@do_sample_numpy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, rand_state):
    return rand_state.chisquare(df=float(dist.k), size=size)


@do_sample_numpy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, rand_state):
    return rand_state.exponential(1 / float(dist.rate), size=size)

@do_sample_numpy.register(FDistributionDistribution)
def _(dist: FDistributionDistribution, size, rand_state):
    return rand_state.f(dfnum = float(dist.d1), dfden = float(dist.d2), size=size)

@do_sample_numpy.register(GammaDistribution)
def _(dist: GammaDistribution, size, rand_state):
    return rand_state.gamma(shape = float(dist.k), scale = float(dist.theta), size=size)

@do_sample_numpy.register(GumbelDistribution)
def _(dist: GumbelDistribution, size, rand_state):
    return rand_state.gumbel(loc = float(dist.mu), scale = float(dist.beta), size=size)

@do_sample_numpy.register(LaplaceDistribution)
def _(dist: LaplaceDistribution, size, rand_state):
    return rand_state.laplace(loc = float(dist.mu), scale = float(dist.b), size=size)

@do_sample_numpy.register(LogisticDistribution)
def _(dist: LogisticDistribution, size, rand_state):
    return rand_state.logistic(loc = float(dist.mu), scale = float(dist.s), size=size)

@do_sample_numpy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, rand_state):
    return rand_state.lognormal(mean = float(dist.mean), sigma = float(dist.std), size=size)

@do_sample_numpy.register(NormalDistribution)
def _(dist: NormalDistribution, size, rand_state):
    return rand_state.normal(loc = float(dist.mean), scale = float(dist.std), size=size)

@do_sample_numpy.register(RayleighDistribution)
def _(dist: RayleighDistribution, size, rand_state):
    return rand_state.rayleigh(scale = float(dist.sigma), size=size)

@do_sample_numpy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, rand_state):
    return (numpy.random.pareto(a=float(dist.alpha), size=size) + 1) * float(dist.xm)

@do_sample_numpy.register(TriangularDistribution)
def _(dist: TriangularDistribution, size, rand_state):
    return rand_state.triangular(left = float(dist.a), mode = float(dist.b), right = float(dist.c), size=size)

@do_sample_numpy.register(UniformDistribution)
def _(dist: UniformDistribution, size, rand_state):
    return rand_state.uniform(low=float(dist.left), high=float(dist.right), size=size)


# DRV:

@do_sample_numpy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, rand_state):
    return rand_state.geometric(p=float(dist.p), size=size)


@do_sample_numpy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, rand_state):
    return rand_state.poisson(lam=float(dist.lamda), size=size)


@do_sample_numpy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, rand_state):
    return rand_state.zipf(a=float(dist.s), size=size)


# FRV:

@do_sample_numpy.register(BinomialDistribution)
def _(dist: BinomialDistribution, size, rand_state):
    return rand_state.binomial(n=int(dist.n), p=float(dist.p), size=size)

@do_sample_numpy.register(HypergeometricDistribution)
def _(dist: HypergeometricDistribution, size, rand_state):
    return rand_state.hypergeometric(ngood = int(dist.N), nbad = int(dist.m), nsample = int(dist.n), size=size)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\frame\test_constructors.py ===
import array
from collections import (
    OrderedDict,
    abc,
    defaultdict,
    namedtuple,
)
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
    date,
    datetime,
    timedelta,
)
import functools
import re

import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td

from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
    NumpyEADtype,
    PeriodDtype,
)

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    Interval,
    MultiIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    cut,
    date_range,
    isna,
)
import pandas._testing as tm
from pandas.arrays import (
    DatetimeArray,
    IntervalArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)

MIXED_FLOAT_DTYPES = ["float16", "float32", "float64"]
MIXED_INT_DTYPES = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
]


class TestDataFrameConstructors:
    def test_constructor_from_ndarray_with_str_dtype(self):
        # If we don't ravel/reshape around ensure_str_array, we end up
        #  with an array of strings each of which is e.g. "[0 1 2]"
        arr = np.arange(12).reshape(4, 3)
        df = DataFrame(arr, dtype=str)
        expected = DataFrame(arr.astype(str), dtype="str")
        tm.assert_frame_equal(df, expected)

    def test_constructor_from_2d_datetimearray(self, using_array_manager):
        dti = date_range("2016-01-01", periods=6, tz="US/Pacific")
        dta = dti._data.reshape(3, 2)

        df = DataFrame(dta)
        expected = DataFrame({0: dta[:, 0], 1: dta[:, 1]})
        tm.assert_frame_equal(df, expected)
        if not using_array_manager:
            # GH#44724 big performance hit if we de-consolidate
            assert len(df._mgr.blocks) == 1

    def test_constructor_dict_with_tzaware_scalar(self):
        # GH#42505
        dt = Timestamp("2019-11-03 01:00:00-0700").tz_convert("America/Los_Angeles")
        dt = dt.as_unit("ns")

        df = DataFrame({"dt": dt}, index=[0])
        expected = DataFrame({"dt": [dt]})
        tm.assert_frame_equal(df, expected)

        # Non-homogeneous
        df = DataFrame({"dt": dt, "value": [1]})
        expected = DataFrame({"dt": [dt], "value": [1]})
        tm.assert_frame_equal(df, expected)

    def test_construct_ndarray_with_nas_and_int_dtype(self):
        # GH#26919 match Series by not casting np.nan to meaningless int
        arr = np.array([[1, np.nan], [2, 3]])
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype="i8")

        # check this matches Series behavior
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype="i8", name=0)

    def test_construct_from_list_of_datetimes(self):
        df = DataFrame([datetime.now(), datetime.now()])
        assert df[0].dtype == np.dtype("M8[ns]")

    def test_constructor_from_tzaware_datetimeindex(self):
        # don't cast a DatetimeIndex WITH a tz, leave as object
        # GH#6032
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        idx = naive.tz_localize("US/Pacific")

        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        assert expected.dtype == idx.dtype

        # convert index to series
        result = Series(idx)
        tm.assert_series_equal(result, expected)

    def test_columns_with_leading_underscore_work_with_to_dict(self):
        col_underscore = "_b"
        df = DataFrame({"a": [1, 2], col_underscore: [3, 4]})
        d = df.to_dict(orient="records")

        ref_d = [{"a": 1, col_underscore: 3}, {"a": 2, col_underscore: 4}]

        assert ref_d == d

    def test_columns_with_leading_number_and_underscore_work_with_to_dict(self):
        col_with_num = "1_b"
        df = DataFrame({"a": [1, 2], col_with_num: [3, 4]})
        d = df.to_dict(orient="records")

        ref_d = [{"a": 1, col_with_num: 3}, {"a": 2, col_with_num: 4}]

        assert ref_d == d

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series):
        # GH#39462
        nat = np.datetime64("NaT", "ns")
        arr = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            arr = arr.reshape(1, 1)

        msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        with pytest.raises(TypeError, match=msg):
            frame_or_series(arr, dtype="m8[ns]")

    @pytest.mark.parametrize("kind", ["m", "M"])
    def test_datetimelike_values_with_object_dtype(self, kind, frame_or_series):
        # with dtype=object, we should cast dt64 values to Timestamps, not pydatetimes
        if kind == "M":
            dtype = "M8[ns]"
            scalar_type = Timestamp
        else:
            dtype = "m8[ns]"
            scalar_type = Timedelta

        arr = np.arange(6, dtype="i8").view(dtype).reshape(3, 2)
        if frame_or_series is Series:
            arr = arr[:, 0]

        obj = frame_or_series(arr, dtype=object)
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

        # go through a different path in internals.construction
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

        obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

        if frame_or_series is DataFrame:
            # other paths through internals.construction
            sers = [Series(x) for x in arr]
            obj = frame_or_series(sers, dtype=object)
            assert obj._mgr.arrays[0].dtype == object
            assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)

    def test_series_with_name_not_matching_column(self):
        # GH#9232
        x = Series(range(5), name=1)
        y = Series(range(5), name=0)

        result = DataFrame(x, columns=[0])
        expected = DataFrame([], columns=[0])
        tm.assert_frame_equal(result, expected)

        result = DataFrame(y, columns=[1])
        expected = DataFrame([], columns=[1])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: DataFrame(),
            lambda: DataFrame(None),
            lambda: DataFrame(()),
            lambda: DataFrame([]),
            lambda: DataFrame(_ for _ in []),
            lambda: DataFrame(range(0)),
            lambda: DataFrame(data=None),
            lambda: DataFrame(data=()),
            lambda: DataFrame(data=[]),
            lambda: DataFrame(data=(_ for _ in [])),
            lambda: DataFrame(data=range(0)),
        ],
    )
    def test_empty_constructor(self, constructor):
        expected = DataFrame()
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: DataFrame({}),
            lambda: DataFrame(data={}),
        ],
    )
    def test_empty_constructor_object_index(self, constructor):
        expected = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        result = constructor()
        assert len(result.index) == 0
        assert len(result.columns) == 0
        tm.assert_frame_equal(result, expected, check_index_type=True)

    @pytest.mark.parametrize(
        "emptylike,expected_index,expected_columns",
        [
            ([[]], RangeIndex(1), RangeIndex(0)),
            ([[], []], RangeIndex(2), RangeIndex(0)),
            ([(_ for _ in [])], RangeIndex(1), RangeIndex(0)),
        ],
    )
    def test_emptylike_constructor(self, emptylike, expected_index, expected_columns):
        expected = DataFrame(index=expected_index, columns=expected_columns)
        result = DataFrame(emptylike)
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed(self, float_string_frame, using_infer_string):
        dtype = "str" if using_infer_string else np.object_
        assert float_string_frame["foo"].dtype == dtype

    def test_constructor_cast_failure(self):
        # as of 2.0, we raise if we can't respect "dtype", previously we
        #  silently ignored
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": ["a", "b", "c"]}, dtype=np.float64)

        # GH 3010, constructing with odd arrays
        df = DataFrame(np.ones((4, 2)))

        # this is ok
        df["foo"] = np.ones((4, 2)).tolist()

        # this is not ok
        msg = "Expected a 1D array, got an array with shape \\(4, 2\\)"
        with pytest.raises(ValueError, match=msg):
            df["test"] = np.ones((4, 2))

        # this is ok
        df["foo2"] = np.ones((4, 2)).tolist()

    def test_constructor_dtype_copy(self):
        orig_df = DataFrame({"col1": [1.0], "col2": [2.0], "col3": [3.0]})

        new_df = DataFrame(orig_df, dtype=float, copy=True)

        new_df["col1"] = 200.0
        assert orig_df["col1"][0] == 1.0

    def test_constructor_dtype_nocast_view_dataframe(
        self, using_copy_on_write, warn_copy_on_write
    ):
        df = DataFrame([[1, 2]])
        should_be_view = DataFrame(df, dtype=df[0].dtype)
        if using_copy_on_write:
            should_be_view.iloc[0, 0] = 99
            assert df.values[0, 0] == 1
        else:
            with tm.assert_cow_warning(warn_copy_on_write):
                should_be_view.iloc[0, 0] = 99
            assert df.values[0, 0] == 99

    def test_constructor_dtype_nocast_view_2d_array(
        self, using_array_manager, using_copy_on_write, warn_copy_on_write
    ):
        df = DataFrame([[1, 2], [3, 4]], dtype="int64")
        if not using_array_manager and not using_copy_on_write:
            should_be_view = DataFrame(df.values, dtype=df[0].dtype)
            # TODO(CoW-warn) this should warn
            # with tm.assert_cow_warning(warn_copy_on_write):
            should_be_view.iloc[0, 0] = 97
            assert df.values[0, 0] == 97
        else:
            # INFO(ArrayManager) DataFrame(ndarray) doesn't necessarily preserve
            # a view on the array to ensure contiguous 1D arrays
            df2 = DataFrame(df.values, dtype=df[0].dtype)
            assert df2._mgr.arrays[0].flags.c_contiguous

    @td.skip_array_manager_invalid_test
    def test_1d_object_array_does_not_copy(self, using_infer_string):
        # https://github.com/pandas-dev/pandas/issues/39272
        arr = np.array(["a", "b"], dtype="object")
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == "pyarrow":
                # object dtype strings are converted to arrow memory,
                # no numpy arrays to compare
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)

        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    @td.skip_array_manager_invalid_test
    def test_2d_object_array_does_not_copy(self, using_infer_string):
        # https://github.com/pandas-dev/pandas/issues/39272
        arr = np.array([["a", "b"], ["c", "d"]], dtype="object")
        df = DataFrame(arr, copy=False)
        if using_infer_string:
            if df[0].dtype.storage == "pyarrow":
                # object dtype strings are converted to arrow memory,
                # no numpy arrays to compare
                pass
            else:
                assert np.shares_memory(df[0].to_numpy(), arr)
        else:
            assert np.shares_memory(df.values, arr)

        df = DataFrame(arr, dtype=object, copy=False)
        assert np.shares_memory(df.values, arr)

    def test_constructor_dtype_list_data(self):
        df = DataFrame([[1, "2"], [None, "a"]], dtype=object)
        assert df.loc[1, 0] is None
        assert df.loc[0, 1] == "2"

    def test_constructor_list_of_2d_raises(self):
        # https://github.com/pandas-dev/pandas/issues/32289
        a = DataFrame()
        b = np.empty((0, 0))
        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([a])

        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([b])

        a = DataFrame({"A": [1, 2]})
        with pytest.raises(ValueError, match=r"shape=\(2, 2, 1\)"):
            DataFrame([a, a])

    @pytest.mark.parametrize(
        "typ, ad",
        [
            # mixed floating and integer coexist in the same frame
            ["float", {}],
            # add lots of types
            ["float", {"A": 1, "B": "foo", "C": "bar"}],
            # GH 622
            ["int", {}],
        ],
    )
    def test_constructor_mixed_dtypes(self, typ, ad):
        if typ == "int":
            dtypes = MIXED_INT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes
            ]
        elif typ == "float":
            dtypes = MIXED_FLOAT_DTYPES
            arrays = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]

        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        ad.update(dict(zip(dtypes, arrays)))
        df = DataFrame(ad)

        dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        for d in dtypes:
            if d in df:
                assert df.dtypes[d] == d

    def test_constructor_complex_dtypes(self):
        # GH10952
        a = np.random.default_rng(2).random(10).astype(np.complex64)
        b = np.random.default_rng(2).random(10).astype(np.complex128)

        df = DataFrame({"a": a, "b": b})
        assert a.dtype == df.a.dtype
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # https://github.com/pandas-dev/pandas/issues/21083
        df = DataFrame({"A": ["x", None]}, dtype=string_dtype)
        result = df.isna()
        expected = DataFrame({"A": [False, True]})
        tm.assert_frame_equal(result, expected)
        assert df.iloc[1, 0] is None

        df = DataFrame({"A": ["x", np.nan]}, dtype=string_dtype)
        assert np.isnan(df.iloc[1, 0])

    def test_constructor_rec(self, float_frame):
        rec = float_frame.to_records(index=False)
        rec.dtype.names = list(rec.dtype.names)[::-1]

        index = float_frame.index

        df = DataFrame(rec)
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))

        df2 = DataFrame(rec, index=index)
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        tm.assert_index_equal(df2.index, index)

        # case with columns != the ones we would infer from the data
        rng = np.arange(len(rec))[::-1]
        df3 = DataFrame(rec, index=rng, columns=["C", "B"])
        expected = DataFrame(rec, index=rng).reindex(columns=["C", "B"])
        tm.assert_frame_equal(df3, expected)

    def test_constructor_bool(self):
        df = DataFrame({0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)})
        assert df.values.dtype == np.bool_

    def test_constructor_overflow_int64(self):
        # see gh-14881
        values = np.array([2**64 - i for i in range(1, 10)], dtype=np.uint64)

        result = DataFrame({"a": values})
        assert result["a"].dtype == np.uint64

        # see gh-2355
        data_scores = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [("uid", "u8"), ("score", "u8")]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        assert df_crawls["uid"].dtype == np.uint64

    @pytest.mark.parametrize(
        "values",
        [
            np.array([2**64], dtype=object),
            np.array([2**65]),
            [2**64 + 1],
            np.array([-(2**63) - 4], dtype=object),
            np.array([-(2**64) - 1]),
            [-(2**65) - 2],
        ],
    )
    def test_constructor_int_overflow(self, values):
        # see gh-18584
        value = values[0]
        result = DataFrame(values)

        assert result[0].dtype == object
        assert result[0][0] == value

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
        result = DataFrame(values)

        assert result[0].dtype == value.dtype
        assert result[0][0] == value

    def test_constructor_ordereddict(self):
        nitems = 100
        nums = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected = [f"A{i:d}" for i in nums]
        df = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        assert expected == list(df.columns)

    def test_constructor_dict(self):
        datetime_series = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        # test expects index shifted by 5
        datetime_series_short = datetime_series[5:]

        frame = DataFrame({"col1": datetime_series, "col2": datetime_series_short})

        # col2 is padded with NaN
        assert len(datetime_series) == 30
        assert len(datetime_series_short) == 25

        tm.assert_series_equal(frame["col1"], datetime_series.rename("col1"))

        exp = Series(
            np.concatenate([[np.nan] * 5, datetime_series_short.values]),
            index=datetime_series.index,
            name="col2",
        )
        tm.assert_series_equal(exp, frame["col2"])

        frame = DataFrame(
            {"col1": datetime_series, "col2": datetime_series_short},
            columns=["col2", "col3", "col4"],
        )

        assert len(frame) == len(datetime_series_short)
        assert "col1" not in frame
        assert isna(frame["col3"]).all()

        # Corner cases
        assert len(DataFrame()) == 0

        # mix dict and array, wrong size - no spec for which error should raise
        # first
        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

    def test_constructor_dict_length1(self):
        # Length-one dict micro-optimization
        frame = DataFrame({"A": {"1": 1, "2": 2}})
        tm.assert_index_equal(frame.index, Index(["1", "2"]))

    def test_constructor_dict_with_index(self):
        # empty dict plus index
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx)
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self):
        # empty dict with index and columns
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx, columns=idx)
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self):
        # with dict of empty list and Series
        frame = DataFrame({"A": [], "B": []}, columns=["A", "B"])
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self):
        # GH 14381
        # Dict with None value
        frame_none = DataFrame({"a": None}, index=[0])
        frame_none_list = DataFrame({"a": [None]}, index=[0])
        assert frame_none._get_value(0, "a") is None
        assert frame_none_list._get_value(0, "a") is None
        tm.assert_frame_equal(frame_none, frame_none_list)

    def test_constructor_dict_errors(self):
        # GH10856
        # dict with scalar values should raise error, even if columns passed
        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7})

        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7}, columns=["a"])

    @pytest.mark.parametrize("scalar", [2, np.nan, None, "D"])
    def test_constructor_invalid_items_unused(self, scalar):
        # No error if invalid (scalar) value is in fact not used:
        result = DataFrame({"a": scalar}, columns=["b"])
        expected = DataFrame(columns=["b"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # GH 18455
        cols = [1, value, 3]
        idx = ["a", value]
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values(1).sort_values("a", axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values("a", axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [np.nan, None, float("nan")])
    def test_constructor_dict_nan_tuple_key(self, value):
        # GH 18455
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([("a", value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        result = DataFrame(data).sort_values((11, 21)).sort_values(("a", value), axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values(("a", value), axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self):
        datetime_series = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        datetime_series_short = datetime_series[:5]

        # GH19018
        # initialization ordering: by insertion order if python>= 3.6
        d = {"b": datetime_series_short, "a": datetime_series}
        frame = DataFrame(data=d)
        expected = DataFrame(data=d, columns=list("ba"))
        tm.assert_frame_equal(frame, expected)

    def test_constructor_dict_nan_key_and_columns(self):
        # GH 16894
        result = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        expected = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        tm.assert_frame_equal(result, expected)

    def test_constructor_multi_index(self):
        # GH 4078
        # construction error with mi and all-nan frame
        tuples = [(2, 3), (3, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        df = DataFrame(index=mi, columns=mi)
        assert isna(df).values.ravel().all()

    def test_constructor_2d_index(self):
        # GH 25416
        # handling of 2d index in construction
        df = DataFrame([[1]], columns=[[1]], index=[1, 2])
        expected = DataFrame(
            [1, 1],
            index=Index([1, 2], dtype="int64"),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        expected = DataFrame(
            [1, 1],
            index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        tm.assert_frame_equal(df, expected)

    def test_constructor_error_msgs(self):
        msg = "Empty data passed with indices specified."
        # passing an empty array with columns specified.
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])

        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        # mix dict and array, wrong size
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

        # wrong size ndarray, GH 3105
        msg = r"Shape of passed values is \(4, 3\), indices imply \(3, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=["foo", "bar", "baz"],
                index=date_range("2000-01-01", periods=3),
            )

        arr = np.array([[4, 5, 6]])
        msg = r"Shape of passed values is \(1, 3\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)

        arr = np.array([4, 5, 6])
        msg = r"Shape of passed values is \(3, 1\), indices imply \(1, 4\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)

        # higher dim raise exception
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(np.zeros((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # wrong size axis labels
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B", "C"],
                index=[1],
            )

        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B"],
                index=[1, 2],
            )

        # gh-26429
        msg = "2 columns passed, passed data had 10 columns"
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=("ones", "twos"))

        msg = "If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": False, "b": True})

    def test_constructor_subclass_dict(self, dict_subclass):
        # Test for passing dict subclass to constructor
        data = {
            "col1": dict_subclass((x, 10.0 * x) for x in range(10)),
            "col2": dict_subclass((x, 20.0 * x) for x in range(10)),
        }
        df = DataFrame(data)
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
        tm.assert_frame_equal(refdf, df)

        data = dict_subclass(data.items())
        df = DataFrame(data)
        tm.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame):
        # try with defaultdict
        data = {}
        float_frame.loc[: float_frame.index[10], "B"] = np.nan

        for k, v in float_frame.items():
            dct = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        frame = DataFrame(data)
        expected = frame.reindex(index=float_frame.index)
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self):
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        df = DataFrame(
            {"d": [4.0], "c": [3.0], "b": [2.0], "a": [1.0]},
            columns=["d", "c", "b", "a"],
        )
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string):
        # cast float tests
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        frame = DataFrame(test_data, dtype=float)
        assert len(frame) == 3
        assert frame["B"].dtype == np.float64
        assert frame["A"].dtype == np.float64

        frame = DataFrame(test_data)
        assert len(frame) == 3
        assert frame["B"].dtype == np.object_ if not using_infer_string else "str"
        assert frame["A"].dtype == np.float64

    def test_constructor_dict_cast2(self):
        # can't cast to float
        test_data = {
            "A": dict(zip(range(20), [f"word_{i}" for i in range(20)])),
            "B": dict(zip(range(15), np.random.default_rng(2).standard_normal(15))),
        }
        with pytest.raises(ValueError, match="could not convert string"):
            DataFrame(test_data, dtype=float)

    def test_constructor_dict_dont_upcast(self):
        d = {"Col1": {"Row1": "A String", "Row2": np.nan}}
        df = DataFrame(d)
        assert isinstance(df["Col1"]["Row2"], float)

    def test_constructor_dict_dont_upcast2(self):
        dm = DataFrame([[1, 2], ["a", "b"]], index=[1, 2], columns=[1, 2])
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self):
        # GH #1491
        data = {"a": (1, 2, 3), "b": (4, 5, 6)}

        result = DataFrame(data)
        expected = DataFrame({k: list(v) for k, v in data.items()})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_dict_of_ranges(self):
        # GH 26356
        data = {"a": range(3), "b": range(3, 6)}

        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_iterators(self):
        # GH 26349
        data = {"a": iter(range(3)), "b": reversed(range(3))}

        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_of_generators(self):
        # GH 26349
        data = {"a": (i for i in (range(3))), "b": (i for i in reversed(range(3)))}
        result = DataFrame(data)
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_multiindex(self):
        d = {
            ("a", "a"): {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2},
            ("b", "a"): {("i", "i"): 6, ("i", "j"): 5, ("j", "i"): 4},
            ("b", "c"): {("i", "i"): 7, ("i", "j"): 8, ("j", "i"): 9},
        }
        _d = sorted(d.items())
        df = DataFrame(d)
        expected = DataFrame(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        ).T
        expected.index = MultiIndex.from_tuples(expected.index)
        tm.assert_frame_equal(
            df,
            expected,
        )

        d["z"] = {"y": 123.0, ("i", "i"): 111, ("i", "j"): 111, ("j", "i"): 111}
        _d.insert(0, ("z", d["z"]))
        expected = DataFrame(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        ).T
        expected.index = Index(expected.index, tupleize_cols=False)
        df = DataFrame(d)
        df = df.reindex(columns=expected.columns, index=expected.index)
        tm.assert_frame_equal(df, expected)

    def test_constructor_dict_datetime64_index(self):
        # GH 10160
        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]

        def create_data(constructor):
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}

        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timestamp(dt) for dt in dates_as_str],
        )

        result_datetime64 = DataFrame(data_datetime64)
        result_datetime = DataFrame(data_datetime)
        result_Timestamp = DataFrame(data_Timestamp)
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    @pytest.mark.parametrize(
        "klass,name",
        [
            (lambda x: np.timedelta64(x, "D"), "timedelta64"),
            (lambda x: timedelta(days=x), "pytimedelta"),
            (lambda x: Timedelta(x, "D"), "Timedelta[ns]"),
            (lambda x: Timedelta(x, "D").as_unit("s"), "Timedelta[s]"),
        ],
    )
    def test_constructor_dict_timedelta64_index(self, klass, name):
        # GH 10160
        td_as_int = [1, 2, 3, 4]

        data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}

        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, "D") for td in td_as_int],
        )

        result = DataFrame(data)

        tm.assert_frame_equal(result, expected)

    def test_constructor_period_dict(self):
        # PeriodIndex
        a = pd.PeriodIndex(["2012-01", "NaT", "2012-04"], freq="M")
        b = pd.PeriodIndex(["2012-02-01", "2012-03-01", "NaT"], freq="D")
        df = DataFrame({"a": a, "b": b})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

        # list of periods
        df = DataFrame({"a": a.astype(object).tolist(), "b": b.astype(object).tolist()})
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

    def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype):
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        df = DataFrame({"a": ea_scalar}, index=[0])
        assert df["a"].dtype == ea_dtype

        expected = DataFrame(index=[0], columns=["a"], data=ea_scalar)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "data,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),
            (
                Timestamp("2011-01-01", tz="US/Eastern"),
                DatetimeTZDtype(unit="s", tz="US/Eastern"),
            ),
        ],
    )
    def test_constructor_extension_scalar_data(self, data, dtype):
        # GH 34832
        df = DataFrame(index=[0, 1], columns=["a", "b"], data=data)

        assert df["a"].dtype == dtype
        assert df["b"].dtype == dtype

        arr = pd.array([data] * 2, dtype=dtype)
        expected = DataFrame({"a": arr, "b": arr})

        tm.assert_frame_equal(df, expected)

    def test_nested_dict_frame_constructor(self):
        rng = pd.period_range("1/1/2000", periods=5)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)

        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(col, {})[row] = df._get_value(row, col)

        result = DataFrame(data, columns=rng)
        tm.assert_frame_equal(result, df)

        data = {}
        for col in df.columns:
            for row in df.index:
                data.setdefault(row, {})[col] = df._get_value(row, col)

        result = DataFrame(data, index=rng).T
        tm.assert_frame_equal(result, df)

    def _check_basic_constructor(self, empty):
        # mat: 2d matrix with shape (3, 2) to input. empty - makes sized
        # objects
        mat = empty((2, 3), dtype=float)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3

        # 1-D input
        frame = DataFrame(empty((3,)), columns=["A"], index=[1, 2, 3])
        assert len(frame.index) == 3
        assert len(frame.columns) == 1

        if empty is not np.ones:
            msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
            with pytest.raises(IntCastingNaNError, match=msg):
                DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)
            return
        else:
            frame = DataFrame(
                mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64
            )
            assert frame.values.dtype == np.int64

        # wrong size axis labels
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1])
        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, columns=["A", "B"], index=[1, 2])

        # higher dim raise exception
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(empty((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # automatic labeling
        frame = DataFrame(mat)
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        frame = DataFrame(mat, index=[1, 2])
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        frame = DataFrame(mat, columns=["A", "B", "C"])
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)

        # 0-length axis
        frame = DataFrame(empty((0, 3)))
        assert len(frame.index) == 0

        frame = DataFrame(empty((3, 0)))
        assert len(frame.columns) == 0

    def test_constructor_ndarray(self):
        self._check_basic_constructor(np.ones)

        frame = DataFrame(["foo", "bar"], index=[0, 1], columns=["A"])
        assert len(frame) == 2

    def test_constructor_maskedarray(self):
        self._check_basic_constructor(ma.masked_all)

        # Check non-masked values
        mat = ma.masked_all((2, 3), dtype=float)
        mat[0, 0] = 1.0
        mat[1, 2] = 2.0
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert 1.0 == frame["A"][1]
        assert 2.0 == frame["C"][2]

        # what is this even checking??
        mat = ma.masked_all((2, 3), dtype=float)
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])
        assert np.all(~np.asarray(frame == frame))

    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    def test_constructor_maskedarray_nonfloat(self):
        # masked int promoted to float
        mat = ma.masked_all((2, 3), dtype=int)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))

        # cast type
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.float64)
        assert frame.values.dtype == np.float64

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"][1]
        assert 2 == frame["C"][2]

        # masked np.datetime64 stays (use NaT as null)
        mat = ma.masked_all((2, 3), dtype="M8[ns]")
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert isna(frame).values.all()

        # cast type
        msg = r"datetime64\[ns\] values and dtype=int64 is not supported"
        with pytest.raises(TypeError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"].astype("i8")[1]
        assert 2 == frame["C"].astype("i8")[2]

        # masked bool promoted to object
        mat = ma.masked_all((2, 3), dtype=bool)
        # 2-D input
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        assert len(frame.index) == 2
        assert len(frame.columns) == 3
        assert np.all(~np.asarray(frame == frame))

        # cast type
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=object)
        assert frame.values.dtype == object

        # Check non-masked values
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert frame["A"][1] is True
        assert frame["C"][2] is False

    def test_constructor_maskedarray_hardened(self):
        # Check numpy masked arrays with hard masks -- from GH24574
        mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)
        # Check case where mask is hard but no data are masked
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        expected = DataFrame(
            {"A": [1.0, 1.0], "B": [1.0, 1.0]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_maskedrecarray_dtype(self):
        # Ensure constructor honors dtype
        data = np.ma.array(
            np.ma.zeros(5, dtype=[("date", "<f8"), ("price", "<f8")]), mask=[False] * 5
        )
        data = data.view(mrecords.mrecarray)
        with pytest.raises(TypeError, match=r"Pass \{name: data\[name\]"):
            # Support for MaskedRecords deprecated GH#40363
            DataFrame(data, dtype=int)

    def test_constructor_corner_shape(self):
        df = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    @pytest.mark.parametrize(
        "data, index, columns, dtype, expected",
        [
            (None, list(range(10)), ["a", "b"], object, np.object_),
            (None, None, ["a", "b"], "int64", np.dtype("int64")),
            (None, list(range(10)), ["a", "b"], int, np.dtype("float64")),
            ({}, None, ["foo", "bar"], None, np.object_),
            ({"b": 1}, list(range(10)), list("abc"), int, np.dtype("float64")),
        ],
    )
    def test_constructor_dtype(self, data, index, columns, dtype, expected):
        df = DataFrame(data, index, columns, dtype)
        assert df.values.dtype == expected

    @pytest.mark.parametrize(
        "data,input_dtype,expected_dtype",
        (
            ([True, False, None], "boolean", pd.BooleanDtype),
            ([1.0, 2.0, None], "Float64", pd.Float64Dtype),
            ([1, 2, None], "Int64", pd.Int64Dtype),
            (["a", "b", "c"], "string", pd.StringDtype),
        ),
    )
    def test_constructor_dtype_nullable_extension_arrays(
        self, data, input_dtype, expected_dtype
    ):
        df = DataFrame({"a": data}, dtype=input_dtype)
        assert df["a"].dtype == expected_dtype()

    def test_constructor_scalar_inference(self, using_infer_string):
        data = {"int": 1, "bool": True, "float": 3.0, "complex": 4j, "object": "foo"}
        df = DataFrame(data, index=np.arange(10))

        assert df["int"].dtype == np.int64
        assert df["bool"].dtype == np.bool_
        assert df["float"].dtype == np.float64
        assert df["complex"].dtype == np.complex128
        assert df["object"].dtype == np.object_ if not using_infer_string else "str"

    def test_constructor_arrays_and_scalars(self):
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10), "b": True})
        exp = DataFrame({"a": df["a"].values, "b": [True] * 10})

        tm.assert_frame_equal(df, exp)
        with pytest.raises(ValueError, match="must pass an index"):
            DataFrame({"a": False, "b": True})

    def test_constructor_DataFrame(self, float_frame):
        df = DataFrame(float_frame)
        tm.assert_frame_equal(df, float_frame)

        df_casted = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    def test_constructor_empty_dataframe(self):
        # GH 20624
        actual = DataFrame(DataFrame(), dtype="object")
        expected = DataFrame([], dtype="object")
        tm.assert_frame_equal(actual, expected)

    def test_constructor_more(self, float_frame):
        # used to be in test_matrix.py
        arr = np.random.default_rng(2).standard_normal(10)
        dm = DataFrame(arr, columns=["A"], index=np.arange(10))
        assert dm.values.ndim == 2

        arr = np.random.default_rng(2).standard_normal(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        assert dm.values.ndim == 2

        # no data specified
        dm = DataFrame(columns=["A", "B"], index=np.arange(10))
        assert dm.values.shape == (10, 2)

        dm = DataFrame(columns=["A", "B"])
        assert dm.values.shape == (0, 2)

        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)

        # can't cast
        mat = np.array(["foo", "bar"], dtype=object).reshape(2, 1)
        msg = "could not convert string to float: 'foo'"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)

        dm = DataFrame(DataFrame(float_frame._series))
        tm.assert_frame_equal(dm, float_frame)

        # int cast
        dm = DataFrame(
            {"A": np.ones(10, dtype=int), "B": np.ones(10, dtype=np.float64)},
            index=np.arange(10),
        )

        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64

    def test_constructor_empty_list(self):
        df = DataFrame([], index=[])
        expected = DataFrame(index=[])
        tm.assert_frame_equal(df, expected)

        # GH 9939
        df = DataFrame([], columns=["A", "B"])
        expected = DataFrame({}, columns=["A", "B"])
        tm.assert_frame_equal(df, expected)

        # Empty generator: list(empty_gen()) == []
        def empty_gen():
            yield from ()

        df = DataFrame(empty_gen(), columns=["A", "B"])
        tm.assert_frame_equal(df, expected)

    def test_constructor_list_of_lists(self, using_infer_string):
        # GH #484
        df = DataFrame(data=[[1, "a"], [2, "b"]], columns=["num", "str"])
        assert is_integer_dtype(df["num"])
        assert df["str"].dtype == np.object_ if not using_infer_string else "str"

        # GH 4851
        # list of 0-dim ndarrays
        expected = DataFrame({0: np.arange(10)})
        data = [np.array(x) for x in range(10)]
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_nested_pandasarray_matches_nested_ndarray(self):
        # GH#43986
        ser = Series([1, 2])

        arr = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2

        df = DataFrame(arr)
        expected = DataFrame(pd.array(arr))
        tm.assert_frame_equal(df, expected)
        assert df.shape == (2, 1)
        tm.assert_numpy_array_equal(df[0].values, arr)

    def test_constructor_list_like_data_nested_list_column(self):
        # GH 32173
        arrays = [list("abcd"), list("cdef")]
        result = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

        mi = MultiIndex.from_arrays(arrays)
        expected = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)

        tm.assert_frame_equal(result, expected)

    def test_constructor_wrong_length_nested_list_column(self):
        # GH 32173
        arrays = [list("abc"), list("cde")]

        msg = "3 columns passed, passed data had 4"
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    def test_constructor_unequal_length_nested_list_column(self):
        # GH 32173
        arrays = [list("abcd"), list("cde")]

        # exception raised inside MultiIndex constructor
        msg = "all arrays must be same length"
        with pytest.raises(ValueError, match=msg):
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    @pytest.mark.parametrize(
        "data",
        [
            [[Timestamp("2021-01-01")]],
            [{"x": Timestamp("2021-01-01")}],
            {"x": [Timestamp("2021-01-01")]},
            {"x": Timestamp("2021-01-01").as_unit("ns")},
        ],
    )
    def test_constructor_one_element_data_list(self, data):
        # GH#42810
        result = DataFrame(data, index=[0, 1, 2], columns=["x"])
        expected = DataFrame({"x": [Timestamp("2021-01-01")] * 3})
        tm.assert_frame_equal(result, expected)

    def test_constructor_sequence_like(self):
        # GH 3783
        # collections.Sequence like

        class DummyContainer(abc.Sequence):
            def __init__(self, lst) -> None:
                self._lst = lst

            def __getitem__(self, n):
                return self._lst.__getitem__(n)

            def __len__(self) -> int:
                return self._lst.__len__()

        lst_containers = [DummyContainer([1, "a"]), DummyContainer([2, "b"])]
        columns = ["num", "str"]
        result = DataFrame(lst_containers, columns=columns)
        expected = DataFrame([[1, "a"], [2, "b"]], columns=columns)
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self):
        # GH 4297
        # support Array
        result = DataFrame({"A": array.array("i", range(10))})
        expected = DataFrame({"A": list(range(10))})
        tm.assert_frame_equal(result, expected, check_dtype=False)

        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([array.array("i", range(10)), array.array("i", range(10))])
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self):
        # GH26342
        result = DataFrame(range(10))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self):
        result = DataFrame([range(10), range(10)])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterable(self):
        # GH 21987
        class Iter:
            def __iter__(self) -> Iterator:
                for i in range(10):
                    yield [1, 2, 3]

        expected = DataFrame([[1, 2, 3]] * 10)
        result = DataFrame(Iter())
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterator(self):
        result = DataFrame(iter(range(10)))
        expected = DataFrame(list(range(10)))
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self):
        result = DataFrame([iter(range(10)), iter(range(10))])
        expected = DataFrame([list(range(10)), list(range(10))])
        tm.assert_frame_equal(result, expected)

    def test_constructor_generator(self):
        # related #2305

        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))

        expected = DataFrame([list(range(10)), list(range(10))])
        result = DataFrame([gen1, gen2])
        tm.assert_frame_equal(result, expected)

        gen = ([i, "a"] for i in range(10))
        result = DataFrame(gen)
        expected = DataFrame({0: range(10), 1: "a"})
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_list_of_dicts(self):
        result = DataFrame([{}])
        expected = DataFrame(index=RangeIndex(1), columns=[])
        tm.assert_frame_equal(result, expected)

    def test_constructor_ordered_dict_nested_preserve_order(self):
        # see gh-18166
        nested1 = OrderedDict([("b", 1), ("a", 2)])
        nested2 = OrderedDict([("b", 2), ("a", 5)])
        data = OrderedDict([("col2", nested1), ("col1", nested2)])
        result = DataFrame(data)
        data = {"col2": [1, 2], "col1": [2, 5]}
        expected = DataFrame(data=data, index=["b", "a"])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(self, dict_type):
        # see gh-13304
        expected = DataFrame([[2, 1]], columns=["b", "a"])

        data = dict_type()
        data["b"] = [2]
        data["a"] = [1]

        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

        data = dict_type()
        data["b"] = 2
        data["a"] = 1

        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_conflicting_orders(self, dict_type):
        # the first dict element sets the ordering for the DataFrame,
        # even if there are conflicting orders from subsequent ones
        row_one = dict_type()
        row_one["b"] = 2
        row_one["a"] = 1

        row_two = dict_type()
        row_two["a"] = 1
        row_two["b"] = 2

        row_three = {"b": 2, "a": 1}

        expected = DataFrame([[2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_series_aligned_index(self):
        series = [Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)]
        result = DataFrame(series)
        expected = DataFrame(
            {"b": [0, 1, 2], "a": [0, 1, 2], "c": [0, 1, 2]},
            columns=["b", "a", "c"],
            index=["0", "1", "2"],
        )
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_derived_dicts(self):
        class CustomDict(dict):
            pass

        d = {"a": 1.5, "b": 3}

        data_custom = [CustomDict(d)]
        data = [d]

        result_custom = DataFrame(data_custom)
        result = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)

    def test_constructor_ragged(self):
        data = {
            "A": np.random.default_rng(2).standard_normal(10),
            "B": np.random.default_rng(2).standard_normal(8),
        }
        with pytest.raises(ValueError, match="All arrays must be of the same length"):
            DataFrame(data)

    def test_constructor_scalar(self):
        idx = Index(range(3))
        df = DataFrame({"a": 0}, index=idx)
        expected = DataFrame({"a": [0, 0, 0]}, index=idx)
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame):
        df = DataFrame(float_frame["A"], index=float_frame.index, columns=["A"])
        df.copy()

    def test_constructor_mixed_dict_and_Series(self):
        data = {}
        data["A"] = {"foo": 1, "bar": 2, "baz": 3}
        data["B"] = Series([4, 3, 2, 1], index=["bar", "qux", "baz", "foo"])

        result = DataFrame(data)
        assert result.index.is_monotonic_increasing

        # ordering ambiguous, raise exception
        with pytest.raises(ValueError, match="ambiguous ordering"):
            DataFrame({"A": ["a", "b"], "B": {"a": "a", "b": "b"}})

        # this is OK though
        result = DataFrame({"A": ["a", "b"], "B": Series(["a", "b"], index=["a", "b"])})
        expected = DataFrame({"A": ["a", "b"], "B": ["a", "b"]}, index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self):
        # Issue 25075
        data = [[1, 2], (3, 4)]
        result = DataFrame(data)
        expected = DataFrame([[1, 2], [3, 4]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples,lists",
        [
            ((), []),
            ((()), []),
            (((), ()), [(), ()]),
            (((), ()), [[], []]),
            (([], []), [[], []]),
            (([1], [2]), [[1], [2]]),  # GH 32776
            (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_constructor_tuple(self, tuples, lists):
        # GH 25691
        result = DataFrame(tuples)
        expected = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        result = DataFrame({"A": [(1, 2), (3, 4)]})
        expected = DataFrame({"A": Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self):
        # GH11181
        named_tuple = namedtuple("Pandas", list("ab"))
        tuples = [named_tuple(1, 3), named_tuple(2, 4)]
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        result = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)

        # with columns
        expected = DataFrame({"y": [1, 2], "z": [3, 4]})
        result = DataFrame(tuples, columns=["y", "z"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses(self):
        # GH21910
        Point = make_dataclass("Point", [("x", int), ("y", int)])

        data = [Point(0, 3), Point(1, 3)]
        expected = DataFrame({"x": [0, 1], "y": [3, 3]})
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_with_varying_types(self):
        # GH21910
        # varying types
        Point = make_dataclass("Point", [("x", int), ("y", int)])
        HLine = make_dataclass("HLine", [("x0", int), ("x1", int), ("y", int)])

        data = [Point(0, 3), HLine(1, 3, 3)]

        expected = DataFrame(
            {"x": [0, np.nan], "y": [3, 3], "x0": [np.nan, 1], "x1": [np.nan, 3]}
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_dataclasses_error_thrown(self):
        # GH21910
        Point = make_dataclass("Point", [("x", int), ("y", int)])

        # expect TypeError
        msg = "asdict() should be called on dataclass instances"
        with pytest.raises(TypeError, match=re.escape(msg)):
            DataFrame([Point(0, 0), {"x": 1, "y": 0}])

    def test_constructor_list_of_dict_order(self):
        # GH10056
        data = [
            {"First": 1, "Second": 4, "Third": 7, "Fourth": 10},
            {"Second": 5, "First": 2, "Fourth": 11, "Third": 8},
            {"Second": 6, "First": 3, "Fourth": 12, "Third": 9, "YYY": 14, "XXX": 13},
        ]
        expected = DataFrame(
            {
                "First": [1, 2, 3],
                "Second": [4, 5, 6],
                "Third": [7, 8, 9],
                "Fourth": [10, 11, 12],
                "YYY": [None, None, 14],
                "XXX": [None, None, 13],
            }
        )
        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

    def test_constructor_Series_named(self):
        a = Series([1, 2, 3], index=["a", "b", "c"], name="x")
        df = DataFrame(a)
        assert df.columns[0] == "x"
        tm.assert_index_equal(df.index, a.index)

        # ndarray like
        arr = np.random.default_rng(2).standard_normal(10)
        s = Series(arr, name="x")
        df = DataFrame(s)
        expected = DataFrame({"x": s})
        tm.assert_frame_equal(df, expected)

        s = Series(arr, index=range(3, 13))
        df = DataFrame(s)
        expected = DataFrame({0: s})
        tm.assert_frame_equal(df, expected)

        msg = r"Shape of passed values is \(10, 1\), indices imply \(10, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(s, columns=[1, 2])

        # #2234
        a = Series([], name="x", dtype=object)
        df = DataFrame(a)
        assert df.columns[0] == "x"

        # series with name and w/o
        s1 = Series(arr, name="x")
        df = DataFrame([s1, arr]).T
        expected = DataFrame({"x": s1, "Unnamed 0": arr}, columns=["x", "Unnamed 0"])
        tm.assert_frame_equal(df, expected)

        # this is a bit non-intuitive here; the series collapse down to arrays
        df = DataFrame([arr, s1]).T
        expected = DataFrame({1: s1, 0: arr}, columns=[0, 1])
        tm.assert_frame_equal(df, expected)

    def test_constructor_Series_named_and_columns(self):
        # GH 9232 validation

        s0 = Series(range(5), name=0)
        s1 = Series(range(5), name=1)

        # matching name and column gives standard frame
        tm.assert_frame_equal(DataFrame(s0, columns=[0]), s0.to_frame())
        tm.assert_frame_equal(DataFrame(s1, columns=[1]), s1.to_frame())

        # non-matching produces empty frame
        assert DataFrame(s0, columns=[1]).empty
        assert DataFrame(s1, columns=[0]).empty

    def test_constructor_Series_differently_indexed(self):
        # name
        s1 = Series([1, 2, 3], index=["a", "b", "c"], name="x")

        # no name
        s2 = Series([1, 2, 3], index=["a", "b", "c"])

        other_index = Index(["a", "b"])

        df1 = DataFrame(s1, index=other_index)
        exp1 = DataFrame(s1.reindex(other_index))
        assert df1.columns[0] == "x"
        tm.assert_frame_equal(df1, exp1)

        df2 = DataFrame(s2, index=other_index)
        exp2 = DataFrame(s2.reindex(other_index))
        assert df2.columns[0] == 0
        tm.assert_index_equal(df2.index, other_index)
        tm.assert_frame_equal(df2, exp2)

    @pytest.mark.parametrize(
        "name_in1,name_in2,name_in3,name_out",
        [
            ("idx", "idx", "idx", "idx"),
            ("idx", "idx", None, None),
            ("idx", None, None, None),
            ("idx1", "idx2", None, None),
            ("idx1", "idx1", "idx2", None),
            ("idx1", "idx2", "idx3", None),
            (None, None, None, None),
        ],
    )
    def test_constructor_index_names(self, name_in1, name_in2, name_in3, name_out):
        # GH13475
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        series = {
            c: Series([0, 1, 2], index=i) for i, c in zip(indices, ["x", "y", "z"])
        }
        result = DataFrame(series)

        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        tm.assert_frame_equal(result, expected)

    def test_constructor_manager_resize(self, float_frame):
        index = list(float_frame.index[:5])
        columns = list(float_frame.columns[:3])

        msg = "Passing a BlockManager to DataFrame"
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            result = DataFrame(float_frame._mgr, index=index, columns=columns)
        tm.assert_index_equal(result.index, Index(index))
        tm.assert_index_equal(result.columns, Index(columns))

    def test_constructor_mix_series_nonseries(self, float_frame):
        df = DataFrame(
            {"A": float_frame["A"], "B": list(float_frame["B"])}, columns=["A", "B"]
        )
        tm.assert_frame_equal(df, float_frame.loc[:, ["A", "B"]])

        msg = "does not match index length"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": float_frame["A"], "B": list(float_frame["B"])[:-2]})

    def test_constructor_miscast_na_int_dtype(self):
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"

        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)

    def test_constructor_column_duplicates(self):
        # it works! #2079
        df = DataFrame([[8, 5]], columns=["a", "a"])
        edf = DataFrame([[8, 5]])
        edf.columns = ["a", "a"]

        tm.assert_frame_equal(df, edf)

        idf = DataFrame.from_records([(8, 5)], columns=["a", "a"])

        tm.assert_frame_equal(idf, edf)

    def test_constructor_empty_with_string_dtype(self, using_infer_string):
        # GH 9428
        expected = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)
        expected_str = DataFrame(
            index=[0, 1], columns=[0, 1], dtype=pd.StringDtype(na_value=np.nan)
        )

        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        if using_infer_string:
            tm.assert_frame_equal(df, expected_str)
        else:
            tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype="U5")
        tm.assert_frame_equal(df, expected)

    def test_constructor_empty_with_string_extension(self, nullable_string_dtype):
        # GH 34915
        expected = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        df = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        tm.assert_frame_equal(df, expected)

    def test_constructor_single_value(self):
        # expecting single value upcasting here
        df = DataFrame(0.0, index=[1, 2, 3], columns=["a", "b", "c"])
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("float64"), df.index, df.columns)
        )

        df = DataFrame(0, index=[1, 2, 3], columns=["a", "b", "c"])
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("int64"), df.index, df.columns)
        )

        df = DataFrame("a", index=[1, 2], columns=["a", "c"])
        tm.assert_frame_equal(
            df,
            DataFrame(
                np.array([["a", "a"], ["a", "a"]], dtype=object),
                index=[1, 2],
                columns=["a", "c"],
            ),
        )

        msg = "DataFrame constructor not properly called!"
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", columns=["a", "c"])

        msg = "incompatible data and dtype"
        with pytest.raises(TypeError, match=msg):
            DataFrame("a", [1, 2], ["a", "c"], float)

    def test_constructor_with_datetimes(self, using_infer_string):
        intname = np.dtype(int).name
        floatname = np.dtype(np.float64).name
        objectname = np.dtype(np.object_).name

        # single item
        df = DataFrame(
            {
                "A": 1,
                "B": "foo",
                "C": "bar",
                "D": Timestamp("20010101"),
                "E": datetime(2001, 1, 2, 0, 0),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("int64")]
            + [
                np.dtype(objectname)
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            * 2
            + [np.dtype("M8[s]"), np.dtype("M8[us]")],
            index=list("ABCDE"),
        )
        tm.assert_series_equal(result, expected)

        # check with ndarray construction ndim==0 (e.g. we are passing a ndim 0
        # ndarray with a dtype specified)
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array(1.0, dtype=floatname),
                intname: np.array(1, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")]
            + [np.dtype("int64")]
            + [
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            + [np.dtype("float64")]
            + [np.dtype(intname)],
            index=["a", "b", "c", floatname, intname],
        )
        tm.assert_series_equal(result, expected)

        # check with ndarray construction ndim>0
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                floatname: np.array([1.0] * 10, dtype=floatname),
                intname: np.array([1] * 10, dtype=intname),
            },
            index=np.arange(10),
        )
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")]
            + [np.dtype("int64")]
            + [
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan)
            ]
            + [np.dtype("float64")]
            + [np.dtype(intname)],
            index=["a", "b", "c", floatname, intname],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes1(self):
        # GH 2809
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        datetime_s = Series(datetimes)
        assert datetime_s.dtype == "M8[ns]"

    def test_constructor_with_datetimes2(self):
        # GH 2810
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        datetimes = [ts.to_pydatetime() for ts in ind]
        dates = [ts.date() for ts in ind]
        df = DataFrame(datetimes, columns=["datetimes"])
        df["dates"] = dates
        result = df.dtypes
        expected = Series(
            [np.dtype("datetime64[ns]"), np.dtype("object")],
            index=["datetimes", "dates"],
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self):
        # GH 7594
        # don't coerce tz-aware
        tz = pytz.timezone("US/Eastern")
        dt = tz.localize(datetime(2012, 1, 1))

        df = DataFrame({"End Date": dt}, index=[0])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )

        df = DataFrame([{"End Date": dt}])
        assert df.iat[0, 0] == dt
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[ns, US/Eastern]"}, dtype=object)
        )

    def test_constructor_with_datetimes4(self):
        # tz-aware (UTC and other tz's)
        # GH 8411
        dr = date_range("20130101", periods=3)
        df = DataFrame({"value": dr})
        assert df.iat[0, 0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        df = DataFrame({"value": dr})
        assert str(df.iat[0, 0].tz) == "US/Eastern"

    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_constructor_with_datetimes5(self):
        # GH 7822
        # preserver an index with a tz on dict construction
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")

        expected = DataFrame({"a": i.to_series().reset_index(drop=True)})
        df = DataFrame()
        df["a"] = i
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": i})
        tm.assert_frame_equal(df, expected)

    def test_constructor_with_datetimes6(self):
        # multiples
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")
        i_no_tz = date_range("1/1/2011", periods=5, freq="10s")
        df = DataFrame({"a": i, "b": i_no_tz})
        expected = DataFrame({"a": i.to_series().reset_index(drop=True), "b": i_no_tz})
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
            [[np.datetime64("NaT")], [None]],
            [[np.datetime64("NaT")], [pd.NaT]],
            [[None], [np.datetime64("NaT")]],
            [[None], [pd.NaT]],
            [[pd.NaT], [np.datetime64("NaT")]],
            [[pd.NaT], [None]],
        ],
    )
    def test_constructor_datetimes_with_nulls(self, arr):
        # gh-15869, GH#11220
        result = DataFrame(arr).dtypes
        expected = Series([np.dtype("datetime64[ns]")])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        ["M", "D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_constructor_datetimes_non_ns(self, order, unit):
        dtype = f"datetime64[{unit}]"
        na = np.array(
            [
                ["2015-01-01", "2015-01-02", "2015-01-03"],
                ["2017-01-01", "2017-01-02", "2017-02-03"],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        expected = DataFrame(na.astype("M8[ns]"))
        if unit in ["M", "D", "h", "m"]:
            with pytest.raises(TypeError, match="Cannot cast"):
                expected.astype(dtype)

            # instead the constructor casts to the closest supported reso, i.e. "s"
            expected = expected.astype("datetime64[s]")
        else:
            expected = expected.astype(dtype=dtype)

        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        [
            "D",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "ns",
        ],
    )
    def test_constructor_timedelta_non_ns(self, order, unit):
        dtype = f"timedelta64[{unit}]"
        na = np.array(
            [
                [np.timedelta64(1, "D"), np.timedelta64(2, "D")],
                [np.timedelta64(4, "D"), np.timedelta64(5, "D")],
            ],
            dtype=dtype,
            order=order,
        )
        df = DataFrame(na)
        if unit in ["D", "h", "m"]:
            # we get the nearest supported unit, i.e. "s"
            exp_unit = "s"
        else:
            exp_unit = unit
        exp_dtype = np.dtype(f"m8[{exp_unit}]")
        expected = DataFrame(
            [
                [Timedelta(1, "D"), Timedelta(2, "D")],
                [Timedelta(4, "D"), Timedelta(5, "D")],
            ],
            dtype=exp_dtype,
        )
        # TODO(2.0): ideally we should get the same 'expected' without passing
        #  dtype=exp_dtype.
        tm.assert_frame_equal(df, expected)

    def test_constructor_for_list_with_dtypes(self, using_infer_string):
        # test list of lists/ndarrays
        df = DataFrame([np.arange(5) for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int")] * 5)
        tm.assert_series_equal(result, expected)

        df = DataFrame([np.array(np.arange(5), dtype="int32") for x in range(5)])
        result = df.dtypes
        expected = Series([np.dtype("int32")] * 5)
        tm.assert_series_equal(result, expected)

        # overflow issue? (we always expected int64 upcasting here)
        df = DataFrame({"a": [2**31, 2**31 + 1]})
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # GH #2751 (construction with no index specified), make sure we cast to
        # platform values
        df = DataFrame([1, 2])
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame([1.0, 2.0])
        assert df.dtypes.iloc[0] == np.dtype("float64")

        df = DataFrame({"a": [1, 2]})
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame({"a": [1.0, 2.0]})
        assert df.dtypes.iloc[0] == np.dtype("float64")

        df = DataFrame({"a": 1}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("int64")

        df = DataFrame({"a": 1.0}, index=range(3))
        assert df.dtypes.iloc[0] == np.dtype("float64")

        # with object list
        df = DataFrame(
            {
                "a": [1, 2, 4, 7],
                "b": [1.2, 2.3, 5.1, 6.3],
                "c": list("abcd"),
                "d": [datetime(2000, 1, 1) for i in range(4)],
                "e": [1.0, 2, 4.0, 7],
            }
        )
        result = df.dtypes
        expected = Series(
            [
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("object")
                if not using_infer_string
                else pd.StringDtype(na_value=np.nan),
                np.dtype("datetime64[ns]"),
                np.dtype("float64"),
            ],
            index=list("abcde"),
        )
        tm.assert_series_equal(result, expected)

    def test_constructor_frame_copy(self, float_frame):
        cop = DataFrame(float_frame, copy=True)
        cop["A"] = 5
        assert (cop["A"] == 5).all()
        assert not (float_frame["A"] == 5).all()

    def test_constructor_frame_shallow_copy(self, float_frame):
        # constructing a DataFrame from DataFrame with copy=False should still
        # give a "shallow" copy (share data, not attributes)
        # https://github.com/pandas-dev/pandas/issues/49523
        orig = float_frame.copy()
        cop = DataFrame(float_frame)
        assert cop._mgr is not float_frame._mgr
        # Overwriting index of copy doesn't change original
        cop.index = np.arange(len(cop))
        tm.assert_frame_equal(float_frame, orig)

    def test_constructor_ndarray_copy(
        self, float_frame, using_array_manager, using_copy_on_write
    ):
        if not using_array_manager:
            arr = float_frame.values.copy()
            df = DataFrame(arr)

            arr[5] = 5
            if using_copy_on_write:
                assert not (df.values[5] == 5).all()
            else:
                assert (df.values[5] == 5).all()

            df = DataFrame(arr, copy=True)
            arr[6] = 6
            assert not (df.values[6] == 6).all()
        else:
            arr = float_frame.values.copy()
            # default: copy to ensure contiguous arrays
            df = DataFrame(arr)
            assert df._mgr.arrays[0].flags.c_contiguous
            arr[0, 0] = 100
            assert df.iloc[0, 0] != 100

            # manually specify copy=False
            df = DataFrame(arr, copy=False)
            assert not df._mgr.arrays[0].flags.c_contiguous
            arr[0, 0] = 1000
            assert df.iloc[0, 0] == 1000

    def test_constructor_series_copy(self, float_frame):
        series = float_frame._series

        df = DataFrame({"A": series["A"]}, copy=True)
        # TODO can be replaced with `df.loc[:, "A"] = 5` after deprecation about
        # inplace mutation is enforced
        df.loc[df.index[0] : df.index[-1], "A"] = 5

        assert not (series["A"] == 5).all()

    @pytest.mark.parametrize(
        "df",
        [
            DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan]),
            DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan]),
            DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]),
            DataFrame(
                [[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]
            ),
            DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2]),
        ],
    )
    def test_constructor_with_nas(self, df):
        # GH 5016
        # na's in indices
        # GH 21428 (non-unique columns)

        for i in range(len(df.columns)):
            df.iloc[:, i]

        indexer = np.arange(len(df.columns))[isna(df.columns)]

        # No NaN found -> error
        if len(indexer) == 0:
            with pytest.raises(KeyError, match="^nan$"):
                df.loc[:, np.nan]
        # single nan should result in Series
        elif len(indexer) == 1:
            tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
        # multiple nans should result in DataFrame
        else:
            tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

    def test_constructor_lists_to_object_dtype(self):
        # from #1074
        d = DataFrame({"a": [np.nan, False]})
        assert d["a"].dtype == np.object_
        assert not d["a"][1]

    def test_constructor_ndarray_categorical_dtype(self):
        cat = Categorical(["A", "B", "C"])
        arr = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))

        result = DataFrame(arr, dtype=cat.dtype)

        expected = DataFrame({0: cat, 1: cat, 2: cat, 3: cat})
        tm.assert_frame_equal(result, expected)

    def test_constructor_categorical(self):
        # GH8626

        # dict creation
        df = DataFrame({"A": list("abc")}, dtype="category")
        expected = Series(list("abc"), dtype="category", name="A")
        tm.assert_series_equal(df["A"], expected)

        # to_frame
        s = Series(list("abc"), dtype="category")
        result = s.to_frame()
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(result[0], expected)
        result = s.to_frame(name="foo")
        expected = Series(list("abc"), dtype="category", name="foo")
        tm.assert_series_equal(result["foo"], expected)

        # list-like creation
        df = DataFrame(list("abc"), dtype="category")
        expected = Series(list("abc"), dtype="category", name=0)
        tm.assert_series_equal(df[0], expected)

    def test_construct_from_1item_list_of_categorical(self):
        # pre-2.0 this behaved as DataFrame({0: cat}), in 2.0 we remove
        #  Categorical special case
        # ndim != 1
        cat = Categorical(list("abc"))
        df = DataFrame([cat])
        expected = DataFrame([cat.astype(object)])
        tm.assert_frame_equal(df, expected)

    def test_construct_from_list_of_categoricals(self):
        # pre-2.0 this behaved as DataFrame({0: cat}), in 2.0 we remove
        #  Categorical special case

        df = DataFrame([Categorical(list("abc")), Categorical(list("abd"))])
        expected = DataFrame([["a", "b", "c"], ["a", "b", "d"]])
        tm.assert_frame_equal(df, expected)

    def test_from_nested_listlike_mixed_types(self):
        # pre-2.0 this behaved as DataFrame({0: cat}), in 2.0 we remove
        #  Categorical special case
        # mixed
        df = DataFrame([Categorical(list("abc")), list("def")])
        expected = DataFrame([["a", "b", "c"], ["d", "e", "f"]])
        tm.assert_frame_equal(df, expected)

    def test_construct_from_listlikes_mismatched_lengths(self):
        df = DataFrame([Categorical(list("abc")), Categorical(list("abdefg"))])
        expected = DataFrame([list("abc"), list("abdefg")])
        tm.assert_frame_equal(df, expected)

    def test_constructor_categorical_series(self):
        items = [1, 2, 3, 1]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        tm.assert_series_equal(res, exp)

        items = ["a", "b", "c", "a"]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        tm.assert_series_equal(res, exp)

        # insert into frame with different index
        # GH 8076
        index = date_range("20000101", periods=3)
        expected = Series(
            Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
        )
        expected.index = index

        expected = DataFrame({"x": expected})
        df = DataFrame({"x": Series(["a", "b", "c"], dtype="category")}, index=index)
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "dtype",
        tm.ALL_NUMERIC_DTYPES
        + tm.DATETIME64_DTYPES
        + tm.TIMEDELTA64_DTYPES
        + tm.BOOL_DTYPES,
    )
    def test_check_dtype_empty_numeric_column(self, dtype):
        # GH24386: Ensure dtypes are set correctly for an empty DataFrame.
        # Empty DataFrame is generated via dictionary data with non-overlapping columns.
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)

        assert data.b.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", tm.STRING_DTYPES + tm.BYTES_DTYPES + tm.OBJECT_DTYPES
    )
    def test_check_dtype_empty_string_column(self, request, dtype, using_array_manager):
        # GH24386: Ensure dtypes are set correctly for an empty DataFrame.
        # Empty DataFrame is generated via dictionary data with non-overlapping columns.
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)

        if using_array_manager and dtype in tm.BYTES_DTYPES:
            # TODO(ArrayManager) astype to bytes dtypes does not yet give object dtype
            td.mark_array_manager_not_yet_implemented(request)

        assert data.b.dtype.name == "object"

    def test_to_frame_with_falsey_names(self):
        # GH 16114
        result = Series(name=0, dtype=object).to_frame().dtypes
        expected = Series({0: object})
        tm.assert_series_equal(result, expected)

        result = DataFrame(Series(name=0, dtype=object)).dtypes
        tm.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    def test_constructor_range_dtype(self, dtype):
        expected = DataFrame({"A": [0, 1, 2, 3, 4]}, dtype=dtype or "int64")

        # GH 26342
        result = DataFrame(range(5), columns=["A"], dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # GH 16804
        result = DataFrame({"A": range(5)}, dtype=dtype)
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self):
        # GH21226
        class List(list):
            pass

        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        result = DataFrame(List([List([1, 2, 3]), List([4, 5, 6])]))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "extension_arr",
        [
            Categorical(list("aabbc")),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start="1/1/2017", end="1/1/2018", freq="M")),
        ],
    )
    def test_constructor_with_extension_array(self, extension_arr):
        # GH11363
        expected = DataFrame(Series(extension_arr))
        result = DataFrame(extension_arr)
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self):
        # GH 10863
        v = date.today()
        tup = v, v
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        expected = DataFrame([0, 1, 2], columns=Index(Series([tup])))
        tm.assert_frame_equal(result, expected)

    def test_construct_with_two_categoricalindex_series(self):
        # GH 14600
        s1 = Series([39, 6, 4], index=CategoricalIndex(["female", "male", "unknown"]))
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(["f", "female", "m", "male", "unknown"]),
        )
        result = DataFrame([s1, s2])
        expected = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=["female", "male", "unknown", "f", "m"],
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_constructor_series_nonexact_categoricalindex(self):
        # GH 42424
        ser = Series(range(100))
        ser1 = cut(ser, 10).value_counts().head(5)
        ser2 = cut(ser, 10).value_counts().tail(5)
        result = DataFrame({"1": ser1, "2": ser2})
        index = CategoricalIndex(
            [
                Interval(-0.099, 9.9, closed="right"),
                Interval(9.9, 19.8, closed="right"),
                Interval(19.8, 29.7, closed="right"),
                Interval(29.7, 39.6, closed="right"),
                Interval(39.6, 49.5, closed="right"),
                Interval(49.5, 59.4, closed="right"),
                Interval(59.4, 69.3, closed="right"),
                Interval(69.3, 79.2, closed="right"),
                Interval(79.2, 89.1, closed="right"),
                Interval(89.1, 99, closed="right"),
            ],
            ordered=True,
        )
        expected = DataFrame(
            {"1": [10] * 5 + [np.nan] * 5, "2": [np.nan] * 5 + [10] * 5}, index=index
        )
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self):
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        arr = np.array(dates, dtype=[("Date", "M8[us]"), ("Forecasting", "M8[us]")])
        df = DataFrame(arr)

        assert df["Date"][0] == dates[0][0]
        assert df["Forecasting"][0] == dates[0][1]

        s = Series(arr["Date"])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self):
        # GH21142 Verify whether Datetime subclasses are also of dtype datetime
        class DatetimeSubclass(datetime):
            pass

        data = DataFrame({"datetime": [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        assert data.datetime.dtype == "datetime64[ns]"

    def test_with_mismatched_index_length_raises(self):
        # GH#33437
        dti = date_range("2016-01-01", periods=3, tz="US/Pacific")
        msg = "Shape of passed values|Passed arrays should have the same length"
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))

    def test_frame_ctor_datetime64_column(self):
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(len(rng)), "B": dates}
        )
        assert np.issubdtype(df["B"].dtype, np.dtype("M8[ns]"))

    def test_dataframe_constructor_infer_multiindex(self):
        index_lists = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]

        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        assert isinstance(multi.index, MultiIndex)
        assert not isinstance(multi.columns, MultiIndex)

        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=index_lists
        )
        assert isinstance(multi.columns, MultiIndex)

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
        # GH#16605
        # Ensure that data elements are converted to strings when
        # dtype is str, 'str', or 'U'

        result = DataFrame({"A": input_vals}, dtype=string_dtype)
        expected = DataFrame({"A": input_vals}).astype({"A": string_dtype})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_str_na(self, string_dtype):
        result = DataFrame({"A": [1.0, 2.0, None]}, dtype=string_dtype)
        expected = DataFrame({"A": ["1.0", "2.0", None]}, dtype=object)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("copy", [False, True])
    def test_dict_nocopy(
        self,
        request,
        copy,
        any_numeric_ea_dtype,
        any_numpy_dtype,
        using_array_manager,
        using_copy_on_write,
    ):
        if (
            using_array_manager
            and not copy
            and any_numpy_dtype not in tm.STRING_DTYPES + tm.BYTES_DTYPES
        ):
            # TODO(ArrayManager) properly honor copy keyword for dict input
            td.mark_array_manager_not_yet_implemented(request)

        a = np.array([1, 2], dtype=any_numpy_dtype)
        b = np.array([3, 4], dtype=any_numpy_dtype)
        if b.dtype.kind in ["S", "U"]:
            # These get cast, making the checks below more cumbersome
            pytest.skip(f"{b.dtype} get cast, making the checks below more cumbersome")

        c = pd.array([1, 2], dtype=any_numeric_ea_dtype)
        c_orig = c.copy()
        df = DataFrame({"a": a, "b": b, "c": c}, copy=copy)

        def get_base(obj):
            if isinstance(obj, np.ndarray):
                return obj.base
            elif isinstance(obj.dtype, np.dtype):
                # i.e. DatetimeArray, TimedeltaArray
                return obj._ndarray.base
            else:
                raise TypeError

        def check_views(c_only: bool = False):
            # written to work for either BlockManager or ArrayManager

            # Check that the underlying data behind df["c"] is still `c`
            #  after setting with iloc.  Since we don't know which entry in
            #  df._mgr.arrays corresponds to df["c"], we just check that exactly
            #  one of these arrays is `c`.  GH#38939
            assert sum(x is c for x in df._mgr.arrays) == 1
            if c_only:
                # If we ever stop consolidating in setitem_with_indexer,
                #  this will become unnecessary.
                return

            assert (
                sum(
                    get_base(x) is a
                    for x in df._mgr.arrays
                    if isinstance(x.dtype, np.dtype)
                )
                == 1
            )
            assert (
                sum(
                    get_base(x) is b
                    for x in df._mgr.arrays
                    if isinstance(x.dtype, np.dtype)
                )
                == 1
            )

        if not copy:
            # constructor preserves views
            check_views()

        # TODO: most of the rest of this test belongs in indexing tests
        if lib.is_np_dtype(df.dtypes.iloc[0], "fciuO"):
            warn = None
        else:
            warn = FutureWarning
        with tm.assert_produces_warning(warn, match="incompatible dtype"):
            df.iloc[0, 0] = 0
            df.iloc[0, 1] = 0
        if not copy:
            check_views(True)

        # FIXME(GH#35417): until GH#35417, iloc.setitem into EA values does not preserve
        #  view, so we have to check in the other direction
        df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
        assert df.dtypes.iloc[2] == c.dtype
        if not copy and not using_copy_on_write:
            check_views(True)

        if copy:
            if a.dtype.kind == "M":
                assert a[0] == a.dtype.type(1, "ns")
                assert b[0] == b.dtype.type(3, "ns")
            else:
                assert a[0] == a.dtype.type(1)
                assert b[0] == b.dtype.type(3)
            # FIXME(GH#35417): enable after GH#35417
            assert c[0] == c_orig[0]  # i.e. df.iloc[0, 2]=45 did *not* update c
        elif not using_copy_on_write:
            # TODO: we can call check_views if we stop consolidating
            #  in setitem_with_indexer
            assert c[0] == 45  # i.e. df.iloc[0, 2]=45 *did* update c
            # TODO: we can check b[0] == 0 if we stop consolidating in
            #  setitem_with_indexer (except for datetimelike?)

    def test_construct_from_dict_ea_series(self):
        # GH#53744 - default of copy=True should also apply for Series with
        # extension dtype
        ser = Series([1, 2, 3], dtype="Int64")
        df = DataFrame({"a": ser})
        assert not np.shares_memory(ser.values._data, df["a"].values._data)

    def test_from_series_with_name_with_columns(self):
        # GH 7893
        result = DataFrame(Series(1, name="foo"), columns=["bar"])
        expected = DataFrame(columns=["bar"])
        tm.assert_frame_equal(result, expected)

    def test_nested_list_columns(self):
        # GH 14467
        result = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[["A", "A", "A"], ["a", "b", "c"]]
        )
        expected = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_2d_object_array_of_periods_or_intervals(self):
        # Period analogue to GH#26825
        pi = pd.period_range("2016-04-05", periods=3)
        data = pi._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == pi.dtype).all()
        assert (df == pi).all().all()

        ii = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
        data2 = ii._data.astype(object).reshape(1, -1)
        df2 = DataFrame(data2)
        assert df2.shape == (1, 3)
        assert (df2.dtypes == ii.dtype).all()
        assert (df2 == ii).all().all()

        # mixed
        data3 = np.r_[data, data2, data, data2].T
        df3 = DataFrame(data3)
        expected = DataFrame({0: pi, 1: ii, 2: pi, 3: ii})
        tm.assert_frame_equal(df3, expected)

    @pytest.mark.parametrize(
        "col_a, col_b",
        [
            ([[1], [2]], np.array([[1], [2]])),
            (np.array([[1], [2]]), [[1], [2]]),
            (np.array([[1], [2]]), np.array([[1], [2]])),
        ],
    )
    def test_error_from_2darray(self, col_a, col_b):
        msg = "Per-column arrays must each be 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": col_a, "b": col_b})

    def test_from_dict_with_missing_copy_false(self):
        # GH#45369 filled columns should not be views of one another
        df = DataFrame(index=[1, 2, 3], columns=["a", "b", "c"], copy=False)
        assert not np.shares_memory(df["a"]._values, df["b"]._values)

        df.iloc[0, 0] = 0
        expected = DataFrame(
            {
                "a": [0, np.nan, np.nan],
                "b": [np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, np.nan],
            },
            index=[1, 2, 3],
            dtype=object,
        )
        tm.assert_frame_equal(df, expected)

    def test_construction_empty_array_multi_column_raises(self):
        # GH#46822
        msg = r"Shape of passed values is \(0, 1\), indices imply \(0, 2\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(data=np.array([]), columns=["a", "b"])

    def test_construct_with_strings_and_none(self):
        # GH#32218
        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        expected = DataFrame({"a": ["1", "2", None]}, dtype="str")
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference(self):
        # GH#54430
        dtype = pd.StringDtype(na_value=np.nan)
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]})
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", "b"]},
            dtype=dtype,
            columns=Index(["a"], dtype=dtype),
            index=Index(["x", "y"], dtype=dtype),
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, index=["x", "y"])
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", 1]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", 1]})
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", "b"]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, dtype="object")
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_array_string_dtype(self):
        # GH#54496
        dtype = pd.StringDtype(na_value=np.nan)
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": np.array(["a", "b"])})
        tm.assert_frame_equal(df, expected)

        expected = DataFrame({0: ["a", "b"], 1: ["c", "d"]}, dtype=dtype)
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]))
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {"a": ["a", "b"], "b": ["c", "d"]},
            dtype=dtype,
            columns=Index(["a", "b"], dtype=dtype),
        )
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]), columns=["a", "b"])
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_block_dim(self):
        # GH#55363
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))
        assert df._mgr.blocks[0].ndim == 2

    def test_inference_on_pandas_objects(self):
        # GH#56012
        idx = Index([Timestamp("2019-12-31")], dtype=object)
        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            result = DataFrame(idx, columns=["a"])
        assert result.dtypes.iloc[0] != np.object_
        result = DataFrame({"a": idx})
        assert result.dtypes.iloc[0] == np.object_

        ser = Series([Timestamp("2019-12-31")], dtype=object)

        with tm.assert_produces_warning(FutureWarning, match="Dtype inference"):
            result = DataFrame(ser, columns=["a"])
        assert result.dtypes.iloc[0] != np.object_
        result = DataFrame({"a": ser})
        assert result.dtypes.iloc[0] == np.object_


class TestDataFrameConstructorIndexInference:
    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(self):
        rng1 = pd.period_range("1/1/1999", "1/1/2012", freq="M")
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)

        rng2 = pd.period_range("1/1/1980", "12/1/2001", freq="M")
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        df = DataFrame({"s1": s1, "s2": s2})

        exp = pd.period_range("1/1/1980", "1/1/2012", freq="M")
        tm.assert_index_equal(df.index, exp)

    def test_frame_from_dict_with_mixed_tzaware_indexes(self):
        # GH#44091
        dti = date_range("2016-01-01", periods=3)

        ser1 = Series(range(3), index=dti)
        ser2 = Series(range(3), index=dti.tz_localize("UTC"))
        ser3 = Series(range(3), index=dti.tz_localize("US/Central"))
        ser4 = Series(range(3))

        # no tz-naive, but we do have mixed tzs and a non-DTI
        df1 = DataFrame({"A": ser2, "B": ser3, "C": ser4})
        exp_index = Index(
            list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df1.index, exp_index)

        df2 = DataFrame({"A": ser2, "C": ser4, "B": ser3})
        exp_index3 = Index(
            list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object
        )
        tm.assert_index_equal(df2.index, exp_index3)

        df3 = DataFrame({"B": ser3, "A": ser2, "C": ser4})
        exp_index3 = Index(
            list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object
        )
        tm.assert_index_equal(df3.index, exp_index3)

        df4 = DataFrame({"C": ser4, "B": ser3, "A": ser2})
        exp_index4 = Index(
            list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object
        )
        tm.assert_index_equal(df4.index, exp_index4)

        # TODO: not clear if these raising is desired (no extant tests),
        #  but this is de facto behavior 2021-12-22
        msg = "Cannot join tz-naive with tz-aware DatetimeIndex"
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "C": ser4, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"D": ser1, "A": ser2, "B": ser3})

    @pytest.mark.parametrize(
        "key_val, col_vals, col_type",
        [
            ["3", ["3", "4"], "utf8"],
            [3, [3, 4], "int8"],
        ],
    )
    def test_dict_data_arrow_column_expansion(self, key_val, col_vals, col_type):
        # GH 53617
        pa = pytest.importorskip("pyarrow")
        cols = pd.arrays.ArrowExtensionArray(
            pa.array(col_vals, type=pa.dictionary(pa.int8(), getattr(pa, col_type)()))
        )
        result = DataFrame({key_val: [1, 2]}, columns=cols)
        expected = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        tm.assert_frame_equal(result, expected)


class TestDataFrameConstructorWithDtypeCoercion:
    def test_floating_values_integer_dtype(self):
        # GH#40110 make DataFrame behavior with arraylike floating data and
        #  inty dtype match Series behavior

        arr = np.random.default_rng(2).standard_normal((10, 5))

        # GH#49599 in 2.0 we raise instead of either
        #  a) silently ignoring dtype and returningfloat (the old Series behavior) or
        #  b) rounding (the old DataFrame behavior)
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, dtype="i8")

        df = DataFrame(arr.round(), dtype="i8")
        assert (df.dtypes == "i8").all()

        # with NaNs, we go through a different path with a different warning
        arr[0, 0] = np.nan
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype="i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype="i8")
        # The future (raising) behavior matches what we would get via astype:
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr).astype("i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0]).astype("i8")


class TestDataFrameConstructorWithDatetimeTZ:
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_construction_preserves_tzaware_dtypes(self, tz):
        # after GH#7822
        # these retain the timezones on dict construction
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        dr_tz = dr.tz_localize(tz)
        df = DataFrame({"A": "foo", "B": dr_tz}, index=dr)
        tz_expected = DatetimeTZDtype("ns", dr_tz.tzinfo)
        assert df["B"].dtype == tz_expected

        # GH#2810 (with timezones)
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({"dr": dr})
        df["dr_tz"] = dr_tz
        df["datetimes_naive"] = datetimes_naive
        df["datetimes_with_tz"] = datetimes_with_tz
        result = df.dtypes
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
            ],
            index=["dr", "dr_tz", "datetimes_naive", "datetimes_with_tz"],
        )
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
            DataFrame({0: [ts]}, dtype="datetime64[ns]")

        msg2 = "Cannot unbox tzaware Timestamp to tznaive dtype"
        with pytest.raises(TypeError, match=msg2):
            DataFrame({0: ts}, index=[0], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame([ts], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame(np.array([ts], dtype=object), dtype="datetime64[ns]")

        with pytest.raises(TypeError, match=msg2):
            DataFrame(ts, index=[0], columns=[0], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame([Series([ts])], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            DataFrame([[ts]], columns=[0], dtype="datetime64[ns]")

    def test_from_dict(self):
        # 8260
        # support datetime64 with tz

        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        dr = date_range("20130110", periods=3)

        # construction
        df = DataFrame({"A": idx, "B": dr})
        assert df["A"].dtype, "M8[ns, US/Eastern"
        assert df["A"].name == "A"
        tm.assert_series_equal(df["A"], Series(idx, name="A"))
        tm.assert_series_equal(df["B"], Series(dr, name="B"))

    def test_from_index(self):
        # from index
        idx2 = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))

        idx2 = date_range("20130101", periods=3, tz="US/Eastern")
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2[0], Series(idx2, name=0))
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2[0], Series(idx2, name=0))

    def test_frame_dict_constructor_datetime64_1680(self):
        dr = date_range("1/1/2012", periods=10)
        s = Series(dr, index=dr)

        # it works!
        DataFrame({"a": "foo", "b": s}, index=dr)
        DataFrame({"a": "foo", "b": s.values}, index=dr)

    def test_frame_datetime64_mixed_index_ctor_1681(self):
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        ts = Series(dr)

        # it works!
        d = DataFrame({"A": "foo", "B": ts}, index=dr)
        assert d["B"].isna().all()

    def test_frame_timeseries_column(self):
        # GH19157
        dr = date_range(
            start="20130101T10:00:00", periods=3, freq="min", tz="US/Eastern"
        )
        result = DataFrame(dr, columns=["timestamps"])
        expected = DataFrame(
            {
                "timestamps": [
                    Timestamp("20130101T10:00:00", tz="US/Eastern"),
                    Timestamp("20130101T10:01:00", tz="US/Eastern"),
                    Timestamp("20130101T10:02:00", tz="US/Eastern"),
                ]
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_construction(self):
        # GH22227
        columns = ["Nevada", "Ohio"]
        pop = {
            "Nevada": {2001: 2.4, 2002: 2.9},
            "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
        }
        result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        expected = DataFrame(
            [(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)],
            columns=columns,
            index=Index([2001, 2002, 2003]),
        )
        tm.assert_frame_equal(result, expected)

    def test_from_tzaware_object_array(self):
        # GH#26825 2D object array of tzaware timestamps should not raise
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        data = dti._data.astype(object).reshape(1, -1)
        df = DataFrame(data)
        assert df.shape == (1, 3)
        assert (df.dtypes == dti.dtype).all()
        assert (df == dti).all().all()

    def test_from_tzaware_mixed_object_array(self):
        # GH#26825
        arr = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        res = DataFrame(arr, columns=["A", "B", "C"])

        expected_dtypes = [
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
            "datetime64[ns, CET]",
        ]
        assert (res.dtypes == expected_dtypes).all()

    def test_from_2d_ndarray_with_dtype(self):
        # GH#12513
        array_dim2 = np.arange(10).reshape((5, 2))
        df = DataFrame(array_dim2, dtype="datetime64[ns, UTC]")

        expected = DataFrame(array_dim2).astype("datetime64[ns, UTC]")
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("typ", [set, frozenset])
    def test_construction_from_set_raises(self, typ):
        # https://github.com/pandas-dev/pandas/issues/32582
        values = typ({1, 2, 3})
        msg = f"'{typ.__name__}' type is unordered"
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": values})

        with pytest.raises(TypeError, match=msg):
            Series(values)

    def test_construction_from_ndarray_datetimelike(self):
        # ensure the underlying arrays are properly wrapped as EA when
        # constructed from 2D ndarray
        arr = np.arange(0, 12, dtype="datetime64[ns]").reshape(4, 3)
        df = DataFrame(arr)
        assert all(isinstance(arr, DatetimeArray) for arr in df._mgr.arrays)

    def test_construction_from_ndarray_with_eadtype_mismatched_columns(self):
        arr = np.random.default_rng(2).standard_normal((10, 2))
        dtype = pd.array([2.0]).dtype
        msg = r"len\(arrays\) must match len\(columns\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, columns=["foo"], dtype=dtype)

        arr2 = pd.array([2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr2, columns=["foo", "bar"])

    def test_columns_indexes_raise_on_sets(self):
        # GH 47215
        data = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError, match="index cannot be a set"):
            DataFrame(data, index={"a", "b"})
        with pytest.raises(ValueError, match="columns cannot be a set"):
            DataFrame(data, columns={"a", "b", "c"})

    # TODO: make this not cast to object in pandas 3.0
    @pytest.mark.skipif(
        not np_version_gt2, reason="StringDType only available in numpy 2 and above"
    )
    @pytest.mark.parametrize(
        "data",
        [
            {"a": ["a", "b", "c"], "b": [1.0, 2.0, 3.0], "c": ["d", "e", "f"]},
        ],
    )
    def test_np_string_array_object_cast(self, data):
        from numpy.dtypes import StringDType

        data["a"] = np.array(data["a"], dtype=StringDType())
        res = DataFrame(data)
        assert res["a"].dtype == np.object_
        assert (res["a"] == data["a"]).all()


def get1(obj):  # TODO: make a helper in tm?
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        return obj.iloc[0, 0]


class TestFromScalar:
    @pytest.fixture(params=[list, dict, None])
    def box(self, request):
        return request.param

    @pytest.fixture
    def constructor(self, frame_or_series, box):
        extra = {"index": range(2)}
        if frame_or_series is DataFrame:
            extra["columns"] = ["A"]

        if box is None:
            return functools.partial(frame_or_series, **extra)

        elif box is dict:
            if frame_or_series is Series:
                return lambda x, **kwargs: frame_or_series(
                    {0: x, 1: x}, **extra, **kwargs
                )
            else:
                return lambda x, **kwargs: frame_or_series({"A": x}, **extra, **kwargs)
        elif frame_or_series is Series:
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
        else:
            return lambda x, **kwargs: frame_or_series({"A": [x, x]}, **extra, **kwargs)

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_from_nat_scalar(self, dtype, constructor):
        obj = constructor(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    def test_from_timedelta_scalar_preserves_nanos(self, constructor):
        td = Timedelta(1)

        obj = constructor(td, dtype="m8[ns]")
        assert get1(obj) == td

    def test_from_timestamp_scalar_preserves_nanos(self, constructor, fixed_now_ts):
        ts = fixed_now_ts + Timedelta(1)

        obj = constructor(ts, dtype="M8[ns]")
        assert get1(obj) == ts

    def test_from_timedelta64_scalar_object(self, constructor):
        td = Timedelta(1)
        td64 = td.to_timedelta64()

        obj = constructor(td64, dtype=object)
        assert isinstance(get1(obj), np.timedelta64)

    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_from_scalar_datetimelike_mismatched(self, constructor, cls):
        scalar = cls("NaT", "ns")
        dtype = {np.datetime64: "m8[ns]", np.timedelta64: "M8[ns]"}[cls]

        if cls is np.datetime64:
            msg1 = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        else:
            msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
        msg = "|".join(["Cannot cast", msg1])

        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

        scalar = cls(4, "ns")
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

    @pytest.mark.parametrize("cls", [datetime, np.datetime64])
    def test_from_out_of_bounds_ns_datetime(
        self, constructor, cls, request, box, frame_or_series
    ):
        # scalar that won't fit in nanosecond dt64, but will fit in microsecond
        if box is list or (frame_or_series is Series and box is dict):
            mark = pytest.mark.xfail(
                reason="Timestamp constructor has been updated to cast dt64 to "
                "non-nano, but DatetimeArray._from_sequence has not",
                strict=True,
            )
            request.applymarker(mark)

        scalar = datetime(9999, 1, 1)
        exp_dtype = "M8[us]"  # pydatetime objects default to this reso

        if cls is np.datetime64:
            scalar = np.datetime64(scalar, "D")
            exp_dtype = "M8[s]"  # closest reso to input
        result = constructor(scalar)

        item = get1(result)
        dtype = tm.get_dtype(result)

        assert type(item) is Timestamp
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.skip_ubsan
    def test_out_of_s_bounds_datetime64(self, constructor):
        scalar = np.datetime64(np.iinfo(np.int64).max, "D")
        result = constructor(scalar)
        item = get1(result)
        assert type(item) is np.datetime64
        dtype = tm.get_dtype(result)
        assert dtype == object

    @pytest.mark.parametrize("cls", [timedelta, np.timedelta64])
    def test_from_out_of_bounds_ns_timedelta(
        self, constructor, cls, request, box, frame_or_series
    ):
        # scalar that won't fit in nanosecond td64, but will fit in microsecond
        if box is list or (frame_or_series is Series and box is dict):
            mark = pytest.mark.xfail(
                reason="TimedeltaArray constructor has been updated to cast td64 "
                "to non-nano, but TimedeltaArray._from_sequence has not",
                strict=True,
            )
            request.applymarker(mark)

        scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
        exp_dtype = "m8[us]"  # smallest reso that fits
        if cls is np.timedelta64:
            scalar = np.timedelta64(scalar, "D")
            exp_dtype = "m8[s]"  # closest reso to input
        result = constructor(scalar)

        item = get1(result)
        dtype = tm.get_dtype(result)

        assert type(item) is Timedelta
        assert item.asm8.dtype == exp_dtype
        assert dtype == exp_dtype

    @pytest.mark.skip_ubsan
    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_out_of_s_bounds_timedelta64(self, constructor, cls):
        scalar = cls(np.iinfo(np.int64).max, "D")
        result = constructor(scalar)
        item = get1(result)
        assert type(item) is cls
        dtype = tm.get_dtype(result)
        assert dtype == object

    def test_tzaware_data_tznaive_dtype(self, constructor, box, frame_or_series):
        tz = "US/Eastern"
        ts = Timestamp("2019", tz=tz)

        if box is None or (frame_or_series is DataFrame and box is dict):
            msg = "Cannot unbox tzaware Timestamp to tznaive dtype"
            err = TypeError
        else:
            msg = (
                "Cannot convert timezone-aware data to timezone-naive dtype. "
                r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
            )
            err = ValueError

        with pytest.raises(err, match=msg):
            constructor(ts, dtype="M8[ns]")


# TODO: better location for this test?
class TestAllowNonNano:
    # Until 2.0, we do not preserve non-nano dt64/td64 when passed as ndarray,
    #  but do preserve it when passed as DTA/TDA

    @pytest.fixture(params=[True, False])
    def as_td(self, request):
        return request.param

    @pytest.fixture
    def arr(self, as_td):
        values = np.arange(5).astype(np.int64).view("M8[s]")
        if as_td:
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            return DatetimeArray._simple_new(values, dtype=values.dtype)

    def test_index_allow_non_nano(self, arr):
        idx = Index(arr)
        assert idx.dtype == arr.dtype

    def test_dti_tdi_allow_non_nano(self, arr, as_td):
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = DatetimeIndex(arr)
        assert idx.dtype == arr.dtype

    def test_series_allow_non_nano(self, arr):
        ser = Series(arr)
        assert ser.dtype == arr.dtype

    def test_frame_allow_non_nano(self, arr):
        df = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype

    def test_frame_from_dict_allow_non_nano(self, arr):
        df = DataFrame({0: arr})
        assert df.dtypes[0] == arr.dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\indexing\test_loc.py ===
""" test label based indexing with loc """
from collections import namedtuple
import contextlib
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
import re

from dateutil.tz import gettz
import numpy as np
import pytest

from pandas._config import using_string_dtype

from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IndexSlice,
    MultiIndex,
    Period,
    PeriodIndex,
    Series,
    SparseDtype,
    Timedelta,
    Timestamp,
    date_range,
    timedelta_range,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises


@pytest.mark.parametrize(
    "series, new_series, expected_ser",
    [
        [[np.nan, np.nan, "b"], ["a", np.nan, np.nan], [False, True, True]],
        [[np.nan, "b"], ["a", np.nan], [False, True]],
    ],
)
def test_not_change_nan_loc(series, new_series, expected_ser):
    # GH 28403
    df = DataFrame({"A": series})
    df.loc[:, "A"] = new_series
    expected = DataFrame({"A": expected_ser})
    tm.assert_frame_equal(df.isna(), expected)
    tm.assert_frame_equal(df.notna(), ~expected)


class TestLoc:
    def test_none_values_on_string_columns(self, using_infer_string):
        # Issue #32218
        df = DataFrame(["1", "2", None], columns=["a"], dtype=object)
        assert df.loc[2, "a"] is None

        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        if using_infer_string:
            assert np.isnan(df.loc[2, "a"])
        else:
            assert df.loc[2, "a"] is None

    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_int(self, kind, request):
        # int label
        obj = request.getfixturevalue(f"{kind}_labels")
        check_indexing_smoketest_or_raises(obj, "loc", 2, fails=KeyError)

    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_label(self, kind, request):
        # label
        obj = request.getfixturevalue(f"{kind}_empty")
        check_indexing_smoketest_or_raises(obj, "loc", "c", fails=KeyError)

    @pytest.mark.parametrize(
        "key, typs, axes",
        [
            ["f", ["ints", "uints", "labels", "mixed", "ts"], None],
            ["f", ["floats"], None],
            [20, ["ints", "uints", "mixed"], None],
            [20, ["labels"], None],
            [20, ["ts"], 0],
            [20, ["floats"], 0],
        ],
    )
    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_label_out_of_range(self, key, typs, axes, kind, request):
        for typ in typs:
            obj = request.getfixturevalue(f"{kind}_{typ}")
            # out of range label
            check_indexing_smoketest_or_raises(
                obj, "loc", key, axes=axes, fails=KeyError
            )

    @pytest.mark.parametrize(
        "key, typs",
        [
            [[0, 1, 2], ["ints", "uints", "floats"]],
            [[1, 3.0, "A"], ["ints", "uints", "floats"]],
        ],
    )
    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_label_list(self, key, typs, kind, request):
        for typ in typs:
            obj = request.getfixturevalue(f"{kind}_{typ}")
            # list of labels
            check_indexing_smoketest_or_raises(obj, "loc", key, fails=KeyError)

    @pytest.mark.parametrize(
        "key, typs, axes",
        [
            [[0, 1, 2], ["empty"], None],
            [[0, 2, 10], ["ints", "uints", "floats"], 0],
            [[3, 6, 7], ["ints", "uints", "floats"], 1],
            # GH 17758 - MultiIndex and missing keys
            [[(1, 3), (1, 4), (2, 5)], ["multi"], 0],
        ],
    )
    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_label_list_with_missing(self, key, typs, axes, kind, request):
        for typ in typs:
            obj = request.getfixturevalue(f"{kind}_{typ}")
            check_indexing_smoketest_or_raises(
                obj, "loc", key, axes=axes, fails=KeyError
            )

    @pytest.mark.parametrize("typs", ["ints", "uints"])
    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_label_list_fails(self, typs, kind, request):
        # fails
        obj = request.getfixturevalue(f"{kind}_{typs}")
        check_indexing_smoketest_or_raises(
            obj, "loc", [20, 30, 40], axes=1, fails=KeyError
        )

    def test_loc_getitem_label_array_like(self):
        # TODO: test something?
        # array like
        pass

    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_bool(self, kind, request):
        obj = request.getfixturevalue(f"{kind}_empty")
        # boolean indexers
        b = [True, False, True, False]

        check_indexing_smoketest_or_raises(obj, "loc", b, fails=IndexError)

    @pytest.mark.parametrize(
        "slc, typs, axes, fails",
        [
            [
                slice(1, 3),
                ["labels", "mixed", "empty", "ts", "floats"],
                None,
                TypeError,
            ],
            [slice("20130102", "20130104"), ["ts"], 1, TypeError],
            [slice(2, 8), ["mixed"], 0, TypeError],
            [slice(2, 8), ["mixed"], 1, KeyError],
            [slice(2, 4, 2), ["mixed"], 0, TypeError],
        ],
    )
    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_loc_getitem_label_slice(self, slc, typs, axes, fails, kind, request):
        # label slices (with ints)

        # real label slices

        # GH 14316
        for typ in typs:
            obj = request.getfixturevalue(f"{kind}_{typ}")
            check_indexing_smoketest_or_raises(
                obj,
                "loc",
                slc,
                axes=axes,
                fails=fails,
            )

    def test_setitem_from_duplicate_axis(self):
        # GH#34034
        df = DataFrame(
            [[20, "a"], [200, "a"], [200, "a"]],
            columns=["col1", "col2"],
            index=[10, 1, 1],
        )
        df.loc[1, "col1"] = np.arange(2)
        expected = DataFrame(
            [[20, "a"], [0, "a"], [1, "a"]], columns=["col1", "col2"], index=[10, 1, 1]
        )
        tm.assert_frame_equal(df, expected)

    def test_column_types_consistent(self):
        # GH 26779
        df = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", np.nan, "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    pd.NaT,
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        df2 = DataFrame(
            data={"A": ["String 3"], "B": [Timestamp("2019-06-11 12:00:00")]}
        )
        # Change Columns A and B to df2.values wherever Column A is NaN
        df.loc[df["A"].isna(), ["A", "B"]] = df2.values
        expected = DataFrame(
            data={
                "channel": [1, 2, 3],
                "A": ["String 1", "String 3", "String 2"],
                "B": [
                    Timestamp("2019-06-11 11:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                    Timestamp("2019-06-11 12:00:00"),
                ],
            }
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "obj, key, exp",
        [
            (
                DataFrame([[1]], columns=Index([False])),
                IndexSlice[:, False],
                Series([1], name=False),
            ),
            (Series([1], index=Index([False])), False, [1]),
            (DataFrame([[1]], index=Index([False])), False, Series([1], name=False)),
        ],
    )
    def test_loc_getitem_single_boolean_arg(self, obj, key, exp):
        # GH 44322
        res = obj.loc[key]
        if isinstance(exp, (DataFrame, Series)):
            tm.assert_equal(res, exp)
        else:
            assert res == exp


class TestLocBaseIndependent:
    # Tests for loc that do not depend on subclassing Base
    def test_loc_npstr(self):
        # GH#45580
        df = DataFrame(index=date_range("2021", "2022"))
        result = df.loc[np.array(["2021/6/1"])[0] :]
        expected = df.iloc[151:]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "msg, key",
        [
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'bar'", (Period(2019), "foo", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'y1', 'bar'", (Period(2019), "y1", "bar")),
            (r"Period\('2019', 'Y-DEC'\), 'foo', 'z1'", (Period(2019), "foo", "z1")),
            (
                r"Period\('2018', 'Y-DEC'\), Period\('2016', 'Y-DEC'\), 'bar'",
                (Period(2018), Period(2016), "bar"),
            ),
            (r"Period\('2018', 'Y-DEC'\), 'foo', 'y1'", (Period(2018), "foo", "y1")),
            (
                r"Period\('2017', 'Y-DEC'\), 'foo', Period\('2015', 'Y-DEC'\)",
                (Period(2017), "foo", Period(2015)),
            ),
            (r"Period\('2017', 'Y-DEC'\), 'z1', 'bar'", (Period(2017), "z1", "bar")),
        ],
    )
    def test_contains_raise_error_if_period_index_is_in_multi_index(self, msg, key):
        # GH#20684
        """
        parse_datetime_string_with_reso return parameter if type not matched.
        PeriodIndex.get_loc takes returned value from parse_datetime_string_with_reso
        as a tuple.
        If first argument is Period and a tuple has 3 items,
        process go on not raise exception
        """
        df = DataFrame(
            {
                "A": [Period(2019), "x1", "x2"],
                "B": [Period(2018), Period(2016), "y1"],
                "C": [Period(2017), "z1", Period(2015)],
                "V1": [1, 2, 3],
                "V2": [10, 20, 30],
            }
        ).set_index(["A", "B", "C"])
        with pytest.raises(KeyError, match=msg):
            df.loc[key]

    def test_loc_getitem_missing_unicode_key(self):
        df = DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="\u05d0"):
            df.loc[:, "\u05d0"]  # should not raise UnicodeEncodeError

    def test_loc_getitem_dups(self):
        # GH 5678
        # repeated getitems on a dup index returning a ndarray
        df = DataFrame(
            np.random.default_rng(2).random((20, 5)),
            index=["ABCDE"[x % 5] for x in range(20)],
        )
        expected = df.loc["A", 0]
        result = df.loc[:, 0].loc["A"]
        tm.assert_series_equal(result, expected)

    def test_loc_getitem_dups2(self):
        # GH4726
        # dup indexing with iloc/loc
        df = DataFrame(
            [[1, 2, "foo", "bar", Timestamp("20130101")]],
            columns=["a", "a", "a", "a", "a"],
            index=[1],
        )
        expected = Series(
            [1, 2, "foo", "bar", Timestamp("20130101")],
            index=["a", "a", "a", "a", "a"],
            name=1,
        )

        result = df.iloc[0]
        tm.assert_series_equal(result, expected)

        result = df.loc[1]
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_dups(self):
        # GH 6541
        df_orig = DataFrame(
            {
                "me": list("rttti"),
                "foo": list("aaade"),
                "bar": np.arange(5, dtype="float64") * 1.34 + 2,
                "bar2": np.arange(5, dtype="float64") * -0.34 + 2,
            }
        ).set_index("me")

        indexer = (
            "r",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

        indexer = (
            "r",
            "bar",
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]

        indexer = (
            "t",
            ["bar", "bar2"],
        )
        df = df_orig.copy()
        df.loc[indexer] *= 2.0
        tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])

    def test_loc_setitem_slice(self):
        # GH10503

        # assigning the same type should not change the type
        df1 = DataFrame({"a": [0, 1, 1], "b": Series([100, 200, 300], dtype="uint32")})
        ix = df1["a"] == 1
        newb1 = df1.loc[ix, "b"] + 1
        df1.loc[ix, "b"] = newb1
        expected = DataFrame(
            {"a": [0, 1, 1], "b": Series([100, 201, 301], dtype="uint32")}
        )
        tm.assert_frame_equal(df1, expected)

        # assigning a new type should get the inferred type
        df2 = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        ix = df1["a"] == 1
        newb2 = df2.loc[ix, "b"]
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df1.loc[ix, "b"] = newb2
        expected = DataFrame({"a": [0, 1, 1], "b": [100, 200, 300]}, dtype="uint64")
        tm.assert_frame_equal(df2, expected)

    def test_loc_setitem_dtype(self):
        # GH31340
        df = DataFrame({"id": ["A"], "a": [1.2], "b": [0.0], "c": [-2.5]})
        cols = ["a", "b", "c"]
        df.loc[:, cols] = df.loc[:, cols].astype("float32")

        # pre-2.0 this setting would swap in new arrays, in 2.0 it is correctly
        #  in-place, consistent with non-split-path
        expected = DataFrame(
            {
                "id": ["A"],
                "a": np.array([1.2], dtype="float64"),
                "b": np.array([0.0], dtype="float64"),
                "c": np.array([-2.5], dtype="float64"),
            }
        )  # id is inferred as object

        tm.assert_frame_equal(df, expected)

    def test_getitem_label_list_with_missing(self):
        s = Series(range(3), index=["a", "b", "c"])

        # consistency
        with pytest.raises(KeyError, match="not in index"):
            s[["a", "d"]]

        s = Series(range(3))
        with pytest.raises(KeyError, match="not in index"):
            s[[0, 3]]

    @pytest.mark.parametrize("index", [[True, False], [True, False, True, False]])
    def test_loc_getitem_bool_diff_len(self, index):
        # GH26658
        s = Series([1, 2, 3])
        msg = f"Boolean index has wrong length: {len(index)} instead of {len(s)}"
        with pytest.raises(IndexError, match=msg):
            s.loc[index]

    def test_loc_getitem_int_slice(self):
        # TODO: test something here?
        pass

    def test_loc_to_fail(self):
        # GH3449
        df = DataFrame(
            np.random.default_rng(2).random((3, 3)),
            index=["a", "b", "c"],
            columns=["e", "f", "g"],
        )

        msg = (
            rf"\"None of \[Index\(\[1, 2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[1, 2], [1, 2]]

    def test_loc_to_fail2(self):
        # GH  7496
        # loc should not fallback

        s = Series(dtype=object)
        s.loc[1] = 1
        s.loc["a"] = 2

        with pytest.raises(KeyError, match=r"^-1$"):
            s.loc[-1]

        msg = (
            rf"\"None of \[Index\(\[-1, -2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-1, -2]]

        msg = r"\"None of \[Index\(\['4'\], dtype='object'\)\] are in the \[index\]\""
        with pytest.raises(KeyError, match=msg):
            s.loc[Index(["4"], dtype=object)]

        s.loc[-1] = 3
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[-1, -2]]

        s["a"] = 2
        msg = (
            rf"\"None of \[Index\(\[-2\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]]

        del s["a"]

        with pytest.raises(KeyError, match=msg):
            s.loc[[-2]] = 0

    def test_loc_to_fail3(self):
        # inconsistency between .loc[values] and .loc[values,:]
        # GH 7999
        df = DataFrame([["a"], ["b"]], index=[1, 2], columns=["value"])

        msg = (
            rf"\"None of \[Index\(\[3\], dtype='{np.dtype(int)}'\)\] are "
            r"in the \[index\]\""
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[3], :]

        with pytest.raises(KeyError, match=msg):
            df.loc[[3]]

    def test_loc_getitem_list_with_fail(self):
        # 15747
        # should KeyError if *any* missing labels

        s = Series([1, 2, 3])

        s.loc[[2]]

        msg = f"\"None of [Index([3], dtype='{np.dtype(int)}')] are in the [index]"
        with pytest.raises(KeyError, match=re.escape(msg)):
            s.loc[[3]]

        # a non-match and a match
        with pytest.raises(KeyError, match="not in index"):
            s.loc[[2, 3]]

    def test_loc_index(self):
        # gh-17131
        # a boolean index should index like a boolean numpy array

        df = DataFrame(
            np.random.default_rng(2).random(size=(5, 10)),
            index=["alpha_0", "alpha_1", "alpha_2", "beta_0", "beta_1"],
        )

        mask = df.index.map(lambda x: "alpha" in x)
        expected = df.loc[np.array(mask)]

        result = df.loc[mask]
        tm.assert_frame_equal(result, expected)

        result = df.loc[mask.values]
        tm.assert_frame_equal(result, expected)

        result = df.loc[pd.array(mask, dtype="boolean")]
        tm.assert_frame_equal(result, expected)

    def test_loc_general(self):
        df = DataFrame(
            np.random.default_rng(2).random((4, 4)),
            columns=["A", "B", "C", "D"],
            index=["A", "B", "C", "D"],
        )

        # want this to work
        result = df.loc[:, "A":"B"].iloc[0:2, :]
        assert (result.columns == ["A", "B"]).all()
        assert (result.index == ["A", "B"]).all()

        # mixed type
        result = DataFrame({"a": [Timestamp("20130101")], "b": [1]}).iloc[0]
        expected = Series([Timestamp("20130101"), 1], index=["a", "b"], name=0)
        tm.assert_series_equal(result, expected)
        assert result.dtype == object

    @pytest.fixture
    def frame_for_consistency(self):
        return DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5"),
                "val": Series(range(5), dtype=np.int64),
            }
        )

    @pytest.mark.parametrize(
        "val",
        [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)],
    )
    def test_loc_setitem_consistency(self, frame_for_consistency, val):
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice
        expected = DataFrame(
            {
                "date": Series(0, index=range(5), dtype=np.int64),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[:, "date"] = val
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_dt64_to_str(self, frame_for_consistency):
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice

        expected = DataFrame(
            {
                "date": Series("foo", index=range(5)),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[:, "date"] = "foo"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_dt64_to_float(self, frame_for_consistency):
        # GH 6149
        # coerce similarly for setitem and loc when rows have a null-slice
        expected = DataFrame(
            {
                "date": Series(1.0, index=range(5)),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        df = frame_for_consistency.copy()
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[:, "date"] = 1.0
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_single_row(self):
        # GH 15494
        # setting on frame with single row
        df = DataFrame({"date": Series([Timestamp("20180101")])})
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[:, "date"] = "string"
        expected = DataFrame({"date": Series(["string"])})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_consistency_empty(self):
        # empty (essentially noops)
        # before the enforcement of #45333 in 2.0, the loc.setitem here would
        #  change the dtype of df.x to int64
        expected = DataFrame(columns=["x", "y"])
        df = DataFrame(columns=["x", "y"])
        with tm.assert_produces_warning(None):
            df.loc[:, "x"] = 1
        tm.assert_frame_equal(df, expected)

        # setting with setitem swaps in a new array, so changes the dtype
        df = DataFrame(columns=["x", "y"])
        df["x"] = 1
        expected["x"] = expected["x"].astype(np.int64)
        tm.assert_frame_equal(df, expected)

    # incompatible dtype warning
    @pytest.mark.xfail(using_string_dtype(), reason="TODO(infer_string)")
    def test_loc_setitem_consistency_slice_column_len(self, using_infer_string):
        # .loc[:,column] setting with slice == len of the column
        # GH10408
        levels = [
            ["Region_1"] * 4,
            ["Site_1", "Site_1", "Site_2", "Site_2"],
            [3987227376, 3980680971, 3977723249, 3977723089],
        ]
        mi = MultiIndex.from_arrays(levels, names=["Region", "Site", "RespondentID"])

        clevels = [
            ["Respondent", "Respondent", "Respondent", "OtherCat", "OtherCat"],
            ["Something", "StartDate", "EndDate", "Yes/No", "SomethingElse"],
        ]
        cols = MultiIndex.from_arrays(clevels, names=["Level_0", "Level_1"])

        values = [
            ["A", "5/25/2015 10:59", "5/25/2015 11:22", "Yes", np.nan],
            ["A", "5/21/2015 9:40", "5/21/2015 9:52", "Yes", "Yes"],
            ["A", "5/20/2015 8:27", "5/20/2015 8:41", "Yes", np.nan],
            ["A", "5/20/2015 8:33", "5/20/2015 9:09", "Yes", "No"],
        ]
        df = DataFrame(values, index=mi, columns=cols)

        ctx = contextlib.nullcontext()
        if using_infer_string:
            ctx = pytest.raises(TypeError, match="Invalid value")

        with ctx:
            df.loc[:, ("Respondent", "StartDate")] = to_datetime(
                df.loc[:, ("Respondent", "StartDate")]
            )
        with ctx:
            df.loc[:, ("Respondent", "EndDate")] = to_datetime(
                df.loc[:, ("Respondent", "EndDate")]
            )

        if using_infer_string:
            # infer-objects won't infer stuff anymore
            return

        df = df.infer_objects()

        # Adding a new key
        df.loc[:, ("Respondent", "Duration")] = (
            df.loc[:, ("Respondent", "EndDate")]
            - df.loc[:, ("Respondent", "StartDate")]
        )

        # timedelta64[m] -> float, so this cannot be done inplace, so
        #  no warning
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[:, ("Respondent", "Duration")] = df.loc[
                :, ("Respondent", "Duration")
            ] / Timedelta(60_000_000_000)

        expected = Series(
            [23.0, 12.0, 14.0, 36.0], index=df.index, name=("Respondent", "Duration")
        )
        tm.assert_series_equal(df[("Respondent", "Duration")], expected)

    @pytest.mark.parametrize("unit", ["Y", "M", "D", "h", "m", "s", "ms", "us"])
    def test_loc_assign_non_ns_datetime(self, unit):
        # GH 27395, non-ns dtype assignment via .loc should work
        # and return the same result when using simple assignment
        df = DataFrame(
            {
                "timestamp": [
                    np.datetime64("2017-02-11 12:41:29"),
                    np.datetime64("1991-11-07 04:22:37"),
                ]
            }
        )

        df.loc[:, unit] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        df["expected"] = df.loc[:, "timestamp"].values.astype(f"datetime64[{unit}]")
        expected = Series(df.loc[:, "expected"], name=unit)
        tm.assert_series_equal(df.loc[:, unit], expected)

    def test_loc_modify_datetime(self):
        # see gh-28837
        df = DataFrame.from_dict(
            {"date": [1485264372711, 1485265925110, 1540215845888, 1540282121025]}
        )

        df["date_dt"] = to_datetime(df["date"], unit="ms", cache=True)

        df.loc[:, "date_dt_cp"] = df.loc[:, "date_dt"]
        df.loc[[2, 3], "date_dt_cp"] = df.loc[[2, 3], "date_dt"]

        expected = DataFrame(
            [
                [1485264372711, "2017-01-24 13:26:12.711", "2017-01-24 13:26:12.711"],
                [1485265925110, "2017-01-24 13:52:05.110", "2017-01-24 13:52:05.110"],
                [1540215845888, "2018-10-22 13:44:05.888", "2018-10-22 13:44:05.888"],
                [1540282121025, "2018-10-23 08:08:41.025", "2018-10-23 08:08:41.025"],
            ],
            columns=["date", "date_dt", "date_dt_cp"],
        )

        columns = ["date_dt", "date_dt_cp"]
        expected[columns] = expected[columns].apply(to_datetime)

        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex(self):
        # GH#6254 setting issue
        df = DataFrame(index=[3, 5, 4], columns=["A"], dtype=float)
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")

        # setting integer values into a float dataframe with loc is inplace,
        #  so we retain float dtype
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype=float)
        expected = DataFrame({"A": ser})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_reindex_mixed(self):
        # GH#40480
        df = DataFrame(index=[3, 5, 4], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[[4, 3, 5], "A"] = np.array([1, 2, 3], dtype="int64")
        ser = Series([2, 3, 1], index=[3, 5, 4], dtype="int64")
        # pre-2.0 this setting swapped in a new array, now it is inplace
        #  consistent with non-split-path
        expected = DataFrame({"A": ser.astype(float)})
        expected["B"] = "string"
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_with_inverted_slice(self):
        # GH#40480
        df = DataFrame(index=[1, 2, 3], columns=["A", "B"], dtype=float)
        df["B"] = "string"
        df.loc[slice(3, 0, -1), "A"] = np.array([1, 2, 3], dtype="int64")
        # pre-2.0 this setting swapped in a new array, now it is inplace
        #  consistent with non-split-path
        expected = DataFrame({"A": [3.0, 2.0, 1.0], "B": "string"}, index=[1, 2, 3])
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_frame(self):
        # GH#6252 setting with an empty frame
        keys1 = ["@" + str(i) for i in range(5)]
        val1 = np.arange(5, dtype="int64")

        keys2 = ["@" + str(i) for i in range(4)]
        val2 = np.arange(4, dtype="int64")

        index = list(set(keys1).union(keys2))
        df = DataFrame(index=index)
        df["A"] = np.nan
        df.loc[keys1, "A"] = val1

        df["B"] = np.nan
        df.loc[keys2, "B"] = val2

        # Because df["A"] was initialized as float64, setting values into it
        #  is inplace, so that dtype is retained
        sera = Series(val1, index=keys1, dtype=np.float64)
        serb = Series(val2, index=keys2)
        expected = DataFrame(
            {"A": sera, "B": serb}, columns=Index(["A", "B"], dtype=object)
        ).reindex(index=index)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=list("abcd"),
            columns=list("ABCD"),
        )

        result = df.iloc[0, 0]

        df.loc["a", "A"] = 1
        result = df.loc["a", "A"]
        assert result == 1

        result = df.iloc[0, 0]
        assert result == 1

        df.loc[:, "B":"D"] = 0
        expected = df.loc[:, "B":"D"]
        result = df.iloc[:, 1:]
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_frame_nan_int_coercion_invalid(self):
        # GH 8669
        # invalid coercion of nan -> int
        df = DataFrame({"A": [1, 2, 3], "B": np.nan})
        df.loc[df.B > df.A, "B"] = df.A
        expected = DataFrame({"A": [1, 2, 3], "B": np.nan})
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_frame_mixed_labels(self):
        # GH 6546
        # setting with mixed labels
        df = DataFrame({1: [1, 2], 2: [3, 4], "a": ["a", "b"]})

        result = df.loc[0, [1, 2]]
        expected = Series(
            [1, 3], index=Index([1, 2], dtype=object), dtype=object, name=0
        )
        tm.assert_series_equal(result, expected)

        expected = DataFrame({1: [5, 2], 2: [6, 4], "a": ["a", "b"]})
        df.loc[0, [1, 2]] = [5, 6]
        tm.assert_frame_equal(df, expected)

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    def test_loc_setitem_frame_multiples(self, warn_copy_on_write):
        # multiple setting
        df = DataFrame(
            {"A": ["foo", "bar", "baz"], "B": Series(range(3), dtype=np.int64)}
        )
        rhs = df.loc[1:2]
        rhs.index = df.index[0:2]
        df.loc[0:1] = rhs
        expected = DataFrame(
            {"A": ["bar", "baz", "baz"], "B": Series([1, 2, 2], dtype=np.int64)}
        )
        tm.assert_frame_equal(df, expected)

        # multiple setting with frame on rhs (with M8)
        df = DataFrame(
            {
                "date": date_range("2000-01-01", "2000-01-5"),
                "val": Series(range(5), dtype=np.int64),
            }
        )
        expected = DataFrame(
            {
                "date": [
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000101"),
                    Timestamp("20000102"),
                    Timestamp("20000103"),
                ],
                "val": Series([0, 1, 0, 1, 2], dtype=np.int64),
            }
        )
        rhs = df.loc[0:2]
        rhs.index = df.index[2:5]
        df.loc[2:4] = rhs
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "indexer", [["A"], slice(None, "A", None), np.array(["A"])]
    )
    @pytest.mark.parametrize("value", [["Z"], np.array(["Z"])])
    def test_loc_setitem_with_scalar_index(self, indexer, value):
        # GH #19474
        # assigning like "df.loc[0, ['A']] = ['Z']" should be evaluated
        # elementwisely, not using "setter('A', ['Z'])".

        # Set object dtype to avoid upcast when setting 'Z'
        df = DataFrame([[1, 2], [3, 4]], columns=["A", "B"]).astype({"A": object})
        df.loc[0, indexer] = value
        result = df.loc[0, "A"]

        assert is_scalar(result) and result == "Z"

    @pytest.mark.parametrize(
        "index,box,expected",
        [
            (
                ([0, 2], ["A", "B", "C", "D"]),
                7,
                DataFrame(
                    [[7, 7, 7, 7], [3, 4, np.nan, np.nan], [7, 7, 7, 7]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (1, ["C", "D"]),
                [7, 8],
                DataFrame(
                    [[1, 2, np.nan, np.nan], [3, 4, 7, 8], [5, 6, np.nan, np.nan]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (1, ["A", "B", "C"]),
                np.array([7, 8, 9], dtype=np.int64),
                DataFrame(
                    [[1, 2, np.nan], [7, 8, 9], [5, 6, np.nan]], columns=["A", "B", "C"]
                ),
            ),
            (
                (slice(1, 3, None), ["B", "C", "D"]),
                [[7, 8, 9], [10, 11, 12]],
                DataFrame(
                    [[1, 2, np.nan, np.nan], [3, 7, 8, 9], [5, 10, 11, 12]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (slice(1, 3, None), ["C", "A", "D"]),
                np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int64),
                DataFrame(
                    [[1, 2, np.nan, np.nan], [8, 4, 7, 9], [11, 6, 10, 12]],
                    columns=["A", "B", "C", "D"],
                ),
            ),
            (
                (slice(None, None, None), ["A", "C"]),
                DataFrame([[7, 8], [9, 10], [11, 12]], columns=["A", "C"]),
                DataFrame(
                    [[7, 2, 8], [9, 4, 10], [11, 6, 12]], columns=["A", "B", "C"]
                ),
            ),
        ],
    )
    def test_loc_setitem_missing_columns(self, index, box, expected):
        # GH 29334
        df = DataFrame([[1, 2], [3, 4], [5, 6]], columns=["A", "B"])

        df.loc[index] = box
        tm.assert_frame_equal(df, expected)

    def test_loc_coercion(self):
        # GH#12411
        df = DataFrame({"date": [Timestamp("20130101").tz_localize("UTC"), pd.NaT]})
        expected = df.dtypes

        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)

        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion2(self):
        # GH#12045
        df = DataFrame({"date": [datetime(2012, 1, 1), datetime(1012, 1, 2)]})
        expected = df.dtypes

        result = df.iloc[[0]]
        tm.assert_series_equal(result.dtypes, expected)

        result = df.iloc[[1]]
        tm.assert_series_equal(result.dtypes, expected)

    def test_loc_coercion3(self):
        # GH#11594
        df = DataFrame({"text": ["some words"] + [None] * 9})
        expected = df.dtypes

        result = df.iloc[0:2]
        tm.assert_series_equal(result.dtypes, expected)

        result = df.iloc[3:]
        tm.assert_series_equal(result.dtypes, expected)

    def test_setitem_new_key_tz(self, indexer_sl):
        # GH#12862 should not raise on assigning the second value
        vals = [
            to_datetime(42).tz_localize("UTC"),
            to_datetime(666).tz_localize("UTC"),
        ]
        expected = Series(vals, index=Index(["foo", "bar"], dtype=object))

        ser = Series(dtype=object)
        indexer_sl(ser)["foo"] = vals[0]
        indexer_sl(ser)["bar"] = vals[1]

        tm.assert_series_equal(ser, expected)

    def test_loc_non_unique(self):
        # GH3659
        # non-unique indexer with loc slice
        # https://groups.google.com/forum/?fromgroups#!topic/pydata/zTm2No0crYs

        # these are going to raise because the we are non monotonic
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]
        )
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:]
        msg = "'Cannot get left slice bound for non-unique label: 0'"
        with pytest.raises(KeyError, match=msg):
            df.loc[0:]
        msg = "'Cannot get left slice bound for non-unique label: 1'"
        with pytest.raises(KeyError, match=msg):
            df.loc[1:2]

        # monotonic are ok
        df = DataFrame(
            {"A": [1, 2, 3, 4, 5, 6], "B": [3, 4, 5, 6, 7, 8]}, index=[0, 1, 0, 1, 2, 3]
        ).sort_index(axis=0)
        result = df.loc[1:]
        expected = DataFrame({"A": [2, 4, 5, 6], "B": [4, 6, 7, 8]}, index=[1, 1, 2, 3])
        tm.assert_frame_equal(result, expected)

        result = df.loc[0:]
        tm.assert_frame_equal(result, df)

        result = df.loc[1:2]
        expected = DataFrame({"A": [2, 4, 5], "B": [4, 6, 7]}, index=[1, 1, 2])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("length, l2", [[900, 100], [900000, 100000]])
    def test_loc_non_unique_memory_error(self, length, l2):
        # GH 4280
        # non_unique index with a large selection triggers a memory error

        columns = list("ABCDEFG")

        df = pd.concat(
            [
                DataFrame(
                    np.random.default_rng(2).standard_normal((length, len(columns))),
                    index=np.arange(length),
                    columns=columns,
                ),
                DataFrame(np.ones((l2, len(columns))), index=[0] * l2, columns=columns),
            ]
        )

        assert df.index.is_unique is False

        mask = np.arange(l2)
        result = df.loc[mask]
        expected = pd.concat(
            [
                df.take([0]),
                DataFrame(
                    np.ones((len(mask), len(columns))),
                    index=[0] * len(mask),
                    columns=columns,
                ),
                df.take(mask[1:]),
            ]
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_name(self):
        # GH 3880
        df = DataFrame([[1, 1], [1, 1]])
        df.index.name = "index_name"
        result = df.iloc[[0, 1]].index.name
        assert result == "index_name"

        result = df.loc[[0, 1]].index.name
        assert result == "index_name"

    def test_loc_empty_list_indexer_is_ok(self):
        df = DataFrame(
            np.ones((5, 2)),
            index=Index([f"i-{i}" for i in range(5)], name="a"),
            columns=Index([f"i-{i}" for i in range(2)], name="a"),
        )
        # vertical empty
        tm.assert_frame_equal(
            df.loc[:, []], df.iloc[:, :0], check_index_type=True, check_column_type=True
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.loc[[], :], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )
        # horizontal empty
        tm.assert_frame_equal(
            df.loc[[]], df.iloc[:0, :], check_index_type=True, check_column_type=True
        )

    def test_identity_slice_returns_new_object(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # GH13873

        original_df = DataFrame({"a": [1, 2, 3]})
        sliced_df = original_df.loc[:]
        assert sliced_df is not original_df
        assert original_df[:] is not original_df
        assert original_df.loc[:, :] is not original_df

        # should be a shallow copy
        assert np.shares_memory(original_df["a"]._values, sliced_df["a"]._values)

        # Setting using .loc[:, "a"] sets inplace so alters both sliced and orig
        # depending on CoW
        with tm.assert_cow_warning(warn_copy_on_write):
            original_df.loc[:, "a"] = [4, 4, 4]
        if using_copy_on_write:
            assert (sliced_df["a"] == [1, 2, 3]).all()
        else:
            assert (sliced_df["a"] == 4).all()

        # These should not return copies
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        if using_copy_on_write or warn_copy_on_write:
            assert df[0] is not df.loc[:, 0]
        else:
            assert df[0] is df.loc[:, 0]

        # Same tests for Series
        original_series = Series([1, 2, 3, 4, 5, 6])
        sliced_series = original_series.loc[:]
        assert sliced_series is not original_series
        assert original_series[:] is not original_series

        with tm.assert_cow_warning(warn_copy_on_write):
            original_series[:3] = [7, 8, 9]
        if using_copy_on_write:
            assert all(sliced_series[:3] == [1, 2, 3])
        else:
            assert all(sliced_series[:3] == [7, 8, 9])

    def test_loc_copy_vs_view(self, request, using_copy_on_write):
        # GH 15631

        if not using_copy_on_write:
            mark = pytest.mark.xfail(reason="accidental fix reverted - GH37497")
            request.applymarker(mark)
        x = DataFrame(zip(range(3), range(3)), columns=["a", "b"])

        y = x.copy()
        q = y.loc[:, "a"]
        q += 2

        tm.assert_frame_equal(x, y)

        z = x.copy()
        q = z.loc[x.index, "a"]
        q += 2

        tm.assert_frame_equal(x, z)

    def test_loc_uint64(self):
        # GH20722
        # Test whether loc accept uint64 max value as index.
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])

        result = ser.loc[umax - 1]
        expected = ser.iloc[0]
        assert result == expected

        result = ser.loc[[umax - 1]]
        expected = ser.iloc[[0]]
        tm.assert_series_equal(result, expected)

        result = ser.loc[[umax - 1, umax]]
        tm.assert_series_equal(result, ser)

    def test_loc_uint64_disallow_negative(self):
        # GH#41775
        umax = np.iinfo("uint64").max
        ser = Series([1, 2], index=[umax - 1, umax])

        with pytest.raises(KeyError, match="-1"):
            # don't wrap around
            ser.loc[-1]

        with pytest.raises(KeyError, match="-1"):
            # don't wrap around
            ser.loc[[-1]]

    def test_loc_setitem_empty_append_expands_rows(self):
        # GH6173, various appends to an empty dataframe

        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )

        # appends to fit length of data
        df = DataFrame(columns=["x", "y"])
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_expands_rows_mixed_dtype(self):
        # GH#37932 same as test_loc_setitem_empty_append_expands_rows
        #  but with mixed dtype so we go through take_split_path
        data = [1, 2, 3]
        expected = DataFrame(
            {"x": data, "y": np.array([np.nan] * len(data), dtype=object)}
        )

        df = DataFrame(columns=["x", "y"])
        df["x"] = df["x"].astype(np.int64)
        df.loc[:, "x"] = data
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_single_value(self):
        # only appends one value
        expected = DataFrame({"x": [1.0], "y": [np.nan]})
        df = DataFrame(columns=["x", "y"], dtype=float)
        df.loc[0, "x"] = expected.loc[0, "x"]
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_empty_append_raises(self):
        # GH6173, various appends to an empty dataframe

        data = [1, 2]
        df = DataFrame(columns=["x", "y"])
        df.index = df.index.astype(np.int64)
        msg = (
            rf"None of \[Index\(\[0, 1\], dtype='{np.dtype(int)}'\)\] "
            r"are in the \[index\]"
        )
        with pytest.raises(KeyError, match=msg):
            df.loc[[0, 1], "x"] = data

        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            df.loc[0:2, "x"] = data

    def test_indexing_zerodim_np_array(self):
        # GH24924
        df = DataFrame([[1, 2], [3, 4]])
        result = df.loc[np.array(0)]
        s = Series([1, 2], name=0)
        tm.assert_series_equal(result, s)

    def test_series_indexing_zerodim_np_array(self):
        # GH24924
        s = Series([1, 2])
        result = s.loc[np.array(0)]
        assert result == 1

    def test_loc_reverse_assignment(self):
        # GH26939
        data = [1, 2, 3, 4, 5, 6] + [None] * 4
        expected = Series(data, index=range(2010, 2020))

        result = Series(index=range(2010, 2020), dtype=np.float64)
        result.loc[2015:2010:-1] = [6, 5, 4, 3, 2, 1]

        tm.assert_series_equal(result, expected)

    def test_loc_setitem_str_to_small_float_conversion_type(self, using_infer_string):
        # GH#20388

        col_data = [str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)]
        result = DataFrame(col_data, columns=["A"])
        expected = DataFrame(col_data, columns=["A"])
        tm.assert_frame_equal(result, expected)

        # assigning with loc/iloc attempts to set the values inplace, which
        #  in this case is successful
        if using_infer_string:
            with pytest.raises(TypeError, match="Invalid value"):
                result.loc[result.index, "A"] = [float(x) for x in col_data]
        else:
            result.loc[result.index, "A"] = [float(x) for x in col_data]
            expected = DataFrame(col_data, columns=["A"], dtype=float).astype(object)
            tm.assert_frame_equal(result, expected)

        # assigning the entire column using __setitem__ swaps in the new array
        # GH#???
        result["A"] = [float(x) for x in col_data]
        expected = DataFrame(col_data, columns=["A"], dtype=float)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_time_object(self, frame_or_series):
        rng = date_range("1/1/2000", "1/5/2000", freq="5min")
        mask = (rng.hour == 9) & (rng.minute == 30)

        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(rng), 3)), index=rng
        )
        obj = tm.get_obj(obj, frame_or_series)

        result = obj.loc[time(9, 30)]
        exp = obj.loc[mask]
        tm.assert_equal(result, exp)

        chunk = obj.loc["1/4/2000":]
        result = chunk.loc[time(9, 30)]
        expected = result[-1:]

        # Without resetting the freqs, these are 5 min and 1440 min, respectively
        result.index = result.index._with_freq(None)
        expected.index = expected.index._with_freq(None)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("spmatrix_t", ["coo_matrix", "csc_matrix", "csr_matrix"])
    @pytest.mark.parametrize("dtype", [np.int64, np.float64, complex])
    def test_loc_getitem_range_from_spmatrix(self, spmatrix_t, dtype):
        sp_sparse = pytest.importorskip("scipy.sparse")

        spmatrix_t = getattr(sp_sparse, spmatrix_t)

        # The bug is triggered by a sparse matrix with purely sparse columns.  So the
        # recipe below generates a rectangular matrix of dimension (5, 7) where all the
        # diagonal cells are ones, meaning the last two columns are purely sparse.
        rows, cols = 5, 7
        spmatrix = spmatrix_t(np.eye(rows, cols, dtype=dtype), dtype=dtype)
        df = DataFrame.sparse.from_spmatrix(spmatrix)

        # regression test for GH#34526
        itr_idx = range(2, rows)
        result = df.loc[itr_idx].values
        expected = spmatrix.toarray()[itr_idx]
        tm.assert_numpy_array_equal(result, expected)

        # regression test for GH#34540
        result = df.loc[itr_idx].dtypes.values
        expected = np.full(cols, SparseDtype(dtype, fill_value=0))
        tm.assert_numpy_array_equal(result, expected)

    def test_loc_getitem_listlike_all_retains_sparse(self):
        df = DataFrame({"A": pd.array([0, 0], dtype=SparseDtype("int64"))})
        result = df.loc[[0, 1]]
        tm.assert_frame_equal(result, df)

    def test_loc_getitem_sparse_frame(self):
        # GH34687
        sp_sparse = pytest.importorskip("scipy.sparse")

        df = DataFrame.sparse.from_spmatrix(sp_sparse.eye(5))
        result = df.loc[range(2)]
        expected = DataFrame(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=SparseDtype("float64", 0.0),
        )
        tm.assert_frame_equal(result, expected)

        result = df.loc[range(2)].loc[range(1)]
        expected = DataFrame(
            [[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=SparseDtype("float64", 0.0)
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sparse_series(self):
        # GH34687
        s = Series([1.0, 0.0, 0.0, 0.0, 0.0], dtype=SparseDtype("float64", 0.0))

        result = s.loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

        result = s.loc[range(3)].loc[range(2)]
        expected = Series([1.0, 0.0], dtype=SparseDtype("float64", 0.0))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("indexer", ["loc", "iloc"])
    def test_getitem_single_row_sparse_df(self, indexer):
        # GH#46406
        df = DataFrame([[1.0, 0.0, 1.5], [0.0, 2.0, 0.0]], dtype=SparseDtype(float))
        result = getattr(df, indexer)[0]
        expected = Series([1.0, 0.0, 1.5], dtype=SparseDtype(float), name=0)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("key_type", [iter, np.array, Series, Index])
    def test_loc_getitem_iterable(self, float_frame, key_type):
        idx = key_type(["A", "B", "C"])
        result = float_frame.loc[:, idx]
        expected = float_frame.loc[:, ["A", "B", "C"]]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_timedelta_0seconds(self):
        # GH#10583
        df = DataFrame(np.random.default_rng(2).normal(size=(10, 4)))
        df.index = timedelta_range(start="0s", periods=10, freq="s")
        expected = df.loc[Timedelta("0s") :, :]
        result = df.loc["0s":, :]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "val,expected", [(2**63 - 1, Series([1])), (2**63, Series([2]))]
    )
    def test_loc_getitem_uint64_scalar(self, val, expected):
        # see GH#19399
        df = DataFrame([1, 2], index=[2**63 - 1, 2**63])
        result = df.loc[val]

        expected.name = val
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_int_label_with_float_index(self, float_numpy_dtype):
        # note labels are floats
        dtype = float_numpy_dtype
        ser = Series(["a", "b", "c"], index=Index([0, 0.5, 1], dtype=dtype))
        expected = ser.copy()

        ser.loc[1] = "zoo"
        expected.iloc[2] = "zoo"

        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize(
        "indexer, expected",
        [
            # The test name is a misnomer in the 0 case as df.index[indexer]
            #  is a scalar.
            (0, [20, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (slice(4, 8), [0, 1, 2, 3, 20, 20, 20, 20, 8, 9]),
            ([3, 5], [0, 1, 2, 20, 4, 20, 6, 7, 8, 9]),
        ],
    )
    def test_loc_setitem_listlike_with_timedelta64index(self, indexer, expected):
        # GH#16637
        tdi = to_timedelta(range(10), unit="s")
        df = DataFrame({"x": range(10)}, dtype="int64", index=tdi)

        df.loc[df.index[indexer], "x"] = 20

        expected = DataFrame(
            expected,
            index=tdi,
            columns=["x"],
            dtype="int64",
        )

        tm.assert_frame_equal(expected, df)

    def test_loc_setitem_categorical_values_partial_column_slice(self):
        # Assigning a Category to parts of a int/... column uses the values of
        # the Categorical
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": list("aaaaa")})
        exp = DataFrame({"a": [1, "b", "b", 1, 1], "b": list("aabba")})
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df.loc[1:2, "a"] = Categorical(["b", "b"], categories=["a", "b"])
            df.loc[2:3, "b"] = Categorical(["b", "b"], categories=["a", "b"])
        tm.assert_frame_equal(df, exp)

    def test_loc_setitem_single_row_categorical(self, using_infer_string):
        # GH#25495
        df = DataFrame({"Alpha": ["a"], "Numeric": [0]})
        categories = Categorical(df["Alpha"], categories=["a", "b", "c"])

        # pre-2.0 this swapped in a new array, in 2.0 it operates inplace,
        #  consistent with non-split-path
        df.loc[:, "Alpha"] = categories

        result = df["Alpha"]
        expected = Series(categories, index=df.index, name="Alpha").astype(
            object if not using_infer_string else "str"
        )
        tm.assert_series_equal(result, expected)

        # double-check that the non-loc setting retains categoricalness
        df["Alpha"] = categories
        tm.assert_series_equal(df["Alpha"], Series(categories, name="Alpha"))

    def test_loc_setitem_datetime_coercion(self):
        # GH#1048
        df = DataFrame({"c": [Timestamp("2010-10-01")] * 3})
        df.loc[0:1, "c"] = np.datetime64("2008-08-08")
        assert Timestamp("2008-08-08") == df.loc[0, "c"]
        assert Timestamp("2008-08-08") == df.loc[1, "c"]
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            df.loc[2, "c"] = date(2005, 5, 5)
        assert Timestamp("2005-05-05").date() == df.loc[2, "c"]

    @pytest.mark.parametrize("idxer", ["var", ["var"]])
    def test_loc_setitem_datetimeindex_tz(self, idxer, tz_naive_fixture):
        # GH#11365
        tz = tz_naive_fixture
        idx = date_range(start="2015-07-12", periods=3, freq="h", tz=tz)
        expected = DataFrame(1.2, index=idx, columns=["var"])
        # if result started off with object dtype, then the .loc.__setitem__
        #  below would retain object dtype
        result = DataFrame(index=idx, columns=["var"], dtype=np.float64)
        with tm.assert_produces_warning(
            FutureWarning if idxer == "var" else None, match="incompatible dtype"
        ):
            # See https://github.com/pandas-dev/pandas/issues/56223
            result.loc[:, idxer] = expected
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_time_key(self, using_array_manager):
        index = date_range("2012-01-01", "2012-01-05", freq="30min")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), 5)), index=index
        )
        akey = time(12, 0, 0)
        bkey = slice(time(13, 0, 0), time(14, 0, 0))
        ainds = [24, 72, 120, 168]
        binds = [26, 27, 28, 74, 75, 76, 122, 123, 124, 170, 171, 172]

        result = df.copy()
        result.loc[akey] = 0
        result = result.loc[akey]
        expected = df.loc[akey].copy()
        expected.loc[:] = 0
        if using_array_manager:
            # TODO(ArrayManager) we are still overwriting columns
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[akey] = 0
        result.loc[akey] = df.iloc[ainds]
        tm.assert_frame_equal(result, df)

        result = df.copy()
        result.loc[bkey] = 0
        result = result.loc[bkey]
        expected = df.loc[bkey].copy()
        expected.loc[:] = 0
        if using_array_manager:
            # TODO(ArrayManager) we are still overwriting columns
            expected = expected.astype(float)
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        result.loc[bkey] = 0
        result.loc[bkey] = df.iloc[binds]
        tm.assert_frame_equal(result, df)

    @pytest.mark.parametrize("key", ["A", ["A"], ("A", slice(None))])
    def test_loc_setitem_unsorted_multiindex_columns(self, key):
        # GH#38601
        mi = MultiIndex.from_tuples([("A", 4), ("B", "3"), ("A", "2")])
        df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
        obj = df.copy()
        obj.loc[:, key] = np.zeros((2, 2), dtype="int64")
        expected = DataFrame([[0, 2, 0], [0, 5, 0]], columns=mi)
        tm.assert_frame_equal(obj, expected)

        df = df.sort_index(axis=1)
        df.loc[:, key] = np.zeros((2, 2), dtype="int64")
        expected = expected.sort_index(axis=1)
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_uint_drop(self, any_int_numpy_dtype):
        # see GH#18311
        # assigning series.loc[0] = 4 changed series.dtype to int
        series = Series([1, 2, 3], dtype=any_int_numpy_dtype)
        series.loc[0] = 4
        expected = Series([4, 2, 3], dtype=any_int_numpy_dtype)
        tm.assert_series_equal(series, expected)

    def test_loc_setitem_td64_non_nano(self):
        # GH#14155
        ser = Series(10 * [np.timedelta64(10, "m")])
        ser.loc[[1, 2, 3]] = np.timedelta64(20, "m")
        expected = Series(10 * [np.timedelta64(10, "m")])
        expected.loc[[1, 2, 3]] = Timedelta(np.timedelta64(20, "m"))
        tm.assert_series_equal(ser, expected)

    def test_loc_setitem_2d_to_1d_raises(self):
        data = np.random.default_rng(2).standard_normal((2, 2))
        # float64 dtype to avoid upcast when trying to set float data
        ser = Series(range(2), dtype="float64")

        msg = "setting an array element with a sequence."
        with pytest.raises(ValueError, match=msg):
            ser.loc[range(2)] = data

        with pytest.raises(ValueError, match=msg):
            ser.loc[:] = data

    def test_loc_getitem_interval_index(self):
        # GH#19977
        index = pd.interval_range(start=0, periods=3)
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"]
        )

        expected = 1
        result = df.loc[0.5, "A"]
        tm.assert_almost_equal(result, expected)

    def test_loc_getitem_interval_index2(self):
        # GH#19977
        index = pd.interval_range(start=0, periods=3, closed="both")
        df = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index, columns=["A", "B", "C"]
        )

        index_exp = pd.interval_range(start=0, periods=2, freq=1, closed="both")
        expected = Series([1, 4], index=index_exp, name="A")
        result = df.loc[1, "A"]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("tpl", [(1,), (1, 2)])
    def test_loc_getitem_index_single_double_tuples(self, tpl):
        # GH#20991
        idx = Index(
            [(1,), (1, 2)],
            name="A",
            tupleize_cols=False,
        )
        df = DataFrame(index=idx)

        result = df.loc[[tpl]]
        idx = Index([tpl], name="A", tupleize_cols=False)
        expected = DataFrame(index=idx)
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_index_namedtuple(self):
        IndexType = namedtuple("IndexType", ["a", "b"])
        idx1 = IndexType("foo", "bar")
        idx2 = IndexType("baz", "bof")
        index = Index([idx1, idx2], name="composite_index", tupleize_cols=False)
        df = DataFrame([(1, 2), (3, 4)], index=index, columns=["A", "B"])

        result = df.loc[IndexType("foo", "bar")]["A"]
        assert result == 1

    def test_loc_setitem_single_column_mixed(self, using_infer_string):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)),
            index=["a", "b", "c", "d", "e"],
            columns=["foo", "bar", "baz"],
        )
        df["str"] = "qux"
        df.loc[df.index[::2], "str"] = np.nan
        expected = Series(
            [np.nan, "qux", np.nan, "qux", np.nan],
            dtype=object if not using_infer_string else "str",
        ).values
        tm.assert_almost_equal(df["str"].values, expected)

    def test_loc_setitem_cast2(self):
        # GH#7704
        # dtype conversion on setting
        df = DataFrame(np.random.default_rng(2).random((30, 3)), columns=tuple("ABC"))
        df["event"] = np.nan
        with tm.assert_produces_warning(
            FutureWarning, match="item of incompatible dtype"
        ):
            df.loc[10, "event"] = "foo"
        result = df.dtypes
        expected = Series(
            [np.dtype("float64")] * 3 + [np.dtype("object")],
            index=["A", "B", "C", "event"],
        )
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_cast3(self):
        # Test that data type is preserved . GH#5782
        df = DataFrame({"one": np.arange(6, dtype=np.int8)})
        df.loc[1, "one"] = 6
        assert df.dtypes.one == np.dtype(np.int8)
        df.one = np.int8(7)
        assert df.dtypes.one == np.dtype(np.int8)

    def test_loc_setitem_range_key(self, frame_or_series):
        # GH#45479 don't treat range key as positional
        obj = frame_or_series(range(5), index=[3, 4, 1, 0, 2])

        values = [9, 10, 11]
        if obj.ndim == 2:
            values = [[9], [10], [11]]

        obj.loc[range(3)] = values

        expected = frame_or_series([0, 1, 10, 9, 11], index=obj.index)
        tm.assert_equal(obj, expected)

    def test_loc_setitem_numpy_frame_categorical_value(self):
        # GH#52927
        df = DataFrame({"a": [1, 1, 1, 1, 1], "b": ["a", "a", "a", "a", "a"]})
        df.loc[1:2, "a"] = Categorical([2, 2], categories=[1, 2])

        expected = DataFrame({"a": [1, 2, 2, 1, 1], "b": ["a", "a", "a", "a", "a"]})
        tm.assert_frame_equal(df, expected)


class TestLocWithEllipsis:
    @pytest.fixture(params=[tm.loc, tm.iloc])
    def indexer(self, request):
        # Test iloc while we're here
        return request.param

    @pytest.fixture
    def obj(self, series_with_simple_index, frame_or_series):
        obj = series_with_simple_index
        if frame_or_series is not Series:
            obj = obj.to_frame()
        return obj

    def test_loc_iloc_getitem_ellipsis(self, obj, indexer):
        result = indexer(obj)[...]
        tm.assert_equal(result, obj)

    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_loc_iloc_getitem_leading_ellipses(self, series_with_simple_index, indexer):
        obj = series_with_simple_index
        key = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]

        if indexer is tm.loc and obj.index.inferred_type == "boolean":
            # passing [False] will get interpreted as a boolean mask
            # TODO: should it?  unambiguous when lengths dont match?
            return
        if indexer is tm.loc and isinstance(obj.index, MultiIndex):
            msg = "MultiIndex does not support indexing with Ellipsis"
            with pytest.raises(NotImplementedError, match=msg):
                result = indexer(obj)[..., [key]]

        elif len(obj) != 0:
            result = indexer(obj)[..., [key]]
            expected = indexer(obj)[[key]]
            tm.assert_series_equal(result, expected)

        key2 = 0 if indexer is tm.iloc else obj.name
        df = obj.to_frame()
        result = indexer(df)[..., [key2]]
        expected = indexer(df)[:, [key2]]
        tm.assert_frame_equal(result, expected)

    def test_loc_iloc_getitem_ellipses_only_one_ellipsis(self, obj, indexer):
        # GH37750
        key = 0 if (indexer is tm.iloc or len(obj) == 0) else obj.index[0]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ...]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., [key], ...]

        with pytest.raises(IndexingError, match=_one_ellipsis_message):
            indexer(obj)[..., ..., key]

        # one_ellipsis_message takes precedence over "Too many indexers"
        #  only when the first key is Ellipsis
        with pytest.raises(IndexingError, match="Too many indexers"):
            indexer(obj)[key, ..., ...]


class TestLocWithMultiIndex:
    @pytest.mark.parametrize(
        "keys, expected",
        [
            (["b", "a"], [["b", "b", "a", "a"], [1, 2, 1, 2]]),
            (["a", "b"], [["a", "a", "b", "b"], [1, 2, 1, 2]]),
            ((["a", "b"], [1, 2]), [["a", "a", "b", "b"], [1, 2, 1, 2]]),
            ((["a", "b"], [2, 1]), [["a", "a", "b", "b"], [2, 1, 2, 1]]),
            ((["b", "a"], [2, 1]), [["b", "b", "a", "a"], [2, 1, 2, 1]]),
            ((["b", "a"], [1, 2]), [["b", "b", "a", "a"], [1, 2, 1, 2]]),
            ((["c", "a"], [2, 1]), [["c", "a", "a"], [1, 2, 1]]),
        ],
    )
    @pytest.mark.parametrize("dim", ["index", "columns"])
    def test_loc_getitem_multilevel_index_order(self, dim, keys, expected):
        # GH#22797
        # Try to respect order of keys given for MultiIndex.loc
        kwargs = {dim: [["c", "a", "a", "b", "b"], [1, 1, 2, 1, 2]]}
        df = DataFrame(np.arange(25).reshape(5, 5), **kwargs)
        exp_index = MultiIndex.from_arrays(expected)
        if dim == "index":
            res = df.loc[keys, :]
            tm.assert_index_equal(res.index, exp_index)
        elif dim == "columns":
            res = df.loc[:, keys]
            tm.assert_index_equal(res.columns, exp_index)

    def test_loc_preserve_names(self, multiindex_year_month_day_dataframe_random_data):
        ymd = multiindex_year_month_day_dataframe_random_data

        result = ymd.loc[2000]
        result2 = ymd["A"].loc[2000]
        assert result.index.names == ymd.index.names[1:]
        assert result2.index.names == ymd.index.names[1:]

        result = ymd.loc[2000, 2]
        result2 = ymd["A"].loc[2000, 2]
        assert result.index.name == ymd.index.names[2]
        assert result2.index.name == ymd.index.names[2]

    def test_loc_getitem_multiindex_nonunique_len_zero(self):
        # GH#13691
        mi = MultiIndex.from_product([[0], [1, 1]])
        ser = Series(0, index=mi)

        res = ser.loc[[]]

        expected = ser[:0]
        tm.assert_series_equal(res, expected)

        res2 = ser.loc[ser.iloc[0:0]]
        tm.assert_series_equal(res2, expected)

    def test_loc_getitem_access_none_value_in_multiindex(self):
        # GH#34318: test that you can access a None value using .loc
        #  through a Multiindex

        ser = Series([None], MultiIndex.from_arrays([["Level1"], ["Level2"]]))
        result = ser.loc[("Level1", "Level2")]
        assert result is None

        midx = MultiIndex.from_product([["Level1"], ["Level2_a", "Level2_b"]])
        ser = Series([None] * len(midx), dtype=object, index=midx)
        result = ser.loc[("Level1", "Level2_a")]
        assert result is None

        ser = Series([1] * len(midx), dtype=object, index=midx)
        result = ser.loc[("Level1", "Level2_a")]
        assert result == 1

    def test_loc_setitem_multiindex_slice(self):
        # GH 34870

        index = MultiIndex.from_tuples(
            zip(
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ),
            names=["first", "second"],
        )

        result = Series([1, 1, 1, 1, 1, 1, 1, 1], index=index)
        result.loc[("baz", "one"):("foo", "two")] = 100

        expected = Series([1, 1, 100, 100, 100, 100, 1, 1], index=index)

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_datetime_objs_with_datetimeindex(self):
        times = date_range("2000-01-01", freq="10min", periods=100000)
        ser = Series(range(100000), times)
        result = ser.loc[datetime(1900, 1, 1) : datetime(2100, 1, 1)]
        tm.assert_series_equal(result, ser)

    def test_loc_getitem_datetime_string_with_datetimeindex(self):
        # GH 16710
        df = DataFrame(
            {"a": range(10), "b": range(10)},
            index=date_range("2010-01-01", "2010-01-10"),
        )
        result = df.loc[["2010-01-01", "2010-01-05"], ["a", "b"]]
        expected = DataFrame(
            {"a": [0, 4], "b": [0, 4]},
            index=DatetimeIndex(["2010-01-01", "2010-01-05"]),
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_sorted_index_level_with_duplicates(self):
        # GH#4516 sorting a MultiIndex with duplicates and multiple dtypes
        mi = MultiIndex.from_tuples(
            [
                ("foo", "bar"),
                ("foo", "bar"),
                ("bah", "bam"),
                ("bah", "bam"),
                ("foo", "bar"),
                ("bah", "bam"),
            ],
            names=["A", "B"],
        )
        df = DataFrame(
            [
                [1.0, 1],
                [2.0, 2],
                [3.0, 3],
                [4.0, 4],
                [5.0, 5],
                [6.0, 6],
            ],
            index=mi,
            columns=["C", "D"],
        )
        df = df.sort_index(level=0)

        expected = DataFrame(
            [[1.0, 1], [2.0, 2], [5.0, 5]], columns=["C", "D"], index=mi.take([0, 1, 4])
        )

        result = df.loc[("foo", "bar")]
        tm.assert_frame_equal(result, expected)

    def test_additional_element_to_categorical_series_loc(self):
        # GH#47677
        result = Series(["a", "b", "c"], dtype="category")
        result.loc[3] = 0
        expected = Series(["a", "b", "c", 0], dtype="object")
        tm.assert_series_equal(result, expected)

    def test_additional_categorical_element_loc(self):
        # GH#47677
        result = Series(["a", "b", "c"], dtype="category")
        result.loc[3] = "a"
        expected = Series(["a", "b", "c", "a"], dtype="category")
        tm.assert_series_equal(result, expected)

    def test_loc_set_nan_in_categorical_series(self, any_numeric_ea_dtype):
        # GH#47677
        srs = Series(
            [1, 2, 3],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        # enlarge
        srs.loc[3] = np.nan
        expected = Series(
            [1, 2, 3, np.nan],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        tm.assert_series_equal(srs, expected)
        # set into
        srs.loc[1] = np.nan
        expected = Series(
            [1, np.nan, 3, np.nan],
            dtype=CategoricalDtype(Index([1, 2, 3], dtype=any_numeric_ea_dtype)),
        )
        tm.assert_series_equal(srs, expected)

    @pytest.mark.parametrize("na", (np.nan, pd.NA, None, pd.NaT))
    def test_loc_consistency_series_enlarge_set_into(self, na):
        # GH#47677
        srs_enlarge = Series(["a", "b", "c"], dtype="category")
        srs_enlarge.loc[3] = na

        srs_setinto = Series(["a", "b", "c", "a"], dtype="category")
        srs_setinto.loc[3] = na

        tm.assert_series_equal(srs_enlarge, srs_setinto)
        expected = Series(["a", "b", "c", na], dtype="category")
        tm.assert_series_equal(srs_enlarge, expected)

    def test_loc_getitem_preserves_index_level_category_dtype(self):
        # GH#15166
        df = DataFrame(
            data=np.arange(2, 22, 2),
            index=MultiIndex(
                levels=[CategoricalIndex(["a", "b"]), range(10)],
                codes=[[0] * 5 + [1] * 5, range(10)],
                names=["Index1", "Index2"],
            ),
        )

        expected = CategoricalIndex(
            ["a", "b"],
            categories=["a", "b"],
            ordered=False,
            name="Index1",
            dtype="category",
        )

        result = df.index.levels[0]
        tm.assert_index_equal(result, expected)

        result = df.loc[["a"]].index.levels[0]
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("lt_value", [30, 10])
    def test_loc_multiindex_levels_contain_values_not_in_index_anymore(self, lt_value):
        # GH#41170
        df = DataFrame({"a": [12, 23, 34, 45]}, index=[list("aabb"), [0, 1, 2, 3]])
        with pytest.raises(KeyError, match=r"\['b'\] not in index"):
            df.loc[df["a"] < lt_value, :].loc[["b"], :]

    def test_loc_multiindex_null_slice_na_level(self):
        # GH#42055
        lev1 = np.array([np.nan, np.nan])
        lev2 = ["bar", "baz"]
        mi = MultiIndex.from_arrays([lev1, lev2])
        ser = Series([0, 1], index=mi)
        result = ser.loc[:, "bar"]

        # TODO: should we have name="bar"?
        expected = Series([0], index=[np.nan])
        tm.assert_series_equal(result, expected)

    def test_loc_drops_level(self):
        # Based on test_series_varied_multiindex_alignment, where
        #  this used to fail to drop the first level
        mi = MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]], names=["ab", "xy", "num"]
        )
        ser = Series(range(8), index=mi)

        loc_result = ser.loc["a", :, :]
        expected = ser.index.droplevel(0)[:4]
        tm.assert_index_equal(loc_result.index, expected)


class TestLocSetitemWithExpansion:
    def test_loc_setitem_with_expansion_large_dataframe(self, monkeypatch):
        # GH#10692
        size_cutoff = 50
        with monkeypatch.context():
            monkeypatch.setattr(libindex, "_SIZE_CUTOFF", size_cutoff)
            result = DataFrame({"x": range(size_cutoff)}, dtype="int64")
            result.loc[size_cutoff] = size_cutoff
        expected = DataFrame({"x": range(size_cutoff + 1)}, dtype="int64")
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_empty_series(self):
        # GH#5226

        # partially set with an empty object series
        ser = Series(dtype=object)
        ser.loc[1] = 1
        tm.assert_series_equal(ser, Series([1], index=[1]))
        ser.loc[3] = 3
        tm.assert_series_equal(ser, Series([1, 3], index=[1, 3]))

    def test_loc_setitem_empty_series_float(self):
        # GH#5226

        # partially set with an empty object series
        ser = Series(dtype=object)
        ser.loc[1] = 1.0
        tm.assert_series_equal(ser, Series([1.0], index=[1]))
        ser.loc[3] = 3.0
        tm.assert_series_equal(ser, Series([1.0, 3.0], index=[1, 3]))

    def test_loc_setitem_empty_series_str_idx(self):
        # GH#5226

        # partially set with an empty object series
        ser = Series(dtype=object)
        ser.loc["foo"] = 1
        tm.assert_series_equal(ser, Series([1], index=Index(["foo"], dtype=object)))
        ser.loc["bar"] = 3
        tm.assert_series_equal(
            ser, Series([1, 3], index=Index(["foo", "bar"], dtype=object))
        )
        ser.loc[3] = 4
        tm.assert_series_equal(
            ser, Series([1, 3, 4], index=Index(["foo", "bar", 3], dtype=object))
        )

    def test_loc_setitem_incremental_with_dst(self):
        # GH#20724
        base = datetime(2015, 11, 1, tzinfo=gettz("US/Pacific"))
        idxs = [base + timedelta(seconds=i * 900) for i in range(16)]
        result = Series([0], index=[idxs[0]])
        for ts in idxs:
            result.loc[ts] = 1
        expected = Series(1, index=idxs)
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "conv",
        [
            lambda x: x,
            lambda x: x.to_datetime64(),
            lambda x: x.to_pydatetime(),
            lambda x: np.datetime64(x),
        ],
        ids=["self", "to_datetime64", "to_pydatetime", "np.datetime64"],
    )
    def test_loc_setitem_datetime_keys_cast(self, conv):
        # GH#9516
        dt1 = Timestamp("20130101 09:00:00")
        dt2 = Timestamp("20130101 10:00:00")
        df = DataFrame()
        df.loc[conv(dt1), "one"] = 100
        df.loc[conv(dt2), "one"] = 200

        expected = DataFrame(
            {"one": [100.0, 200.0]},
            index=[dt1, dt2],
            columns=Index(["one"], dtype=object),
        )
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_categorical_column_retains_dtype(self, ordered):
        # GH16360
        result = DataFrame({"A": [1]})
        result.loc[:, "B"] = Categorical(["b"], ordered=ordered)
        expected = DataFrame({"A": [1], "B": Categorical(["b"], ordered=ordered)})
        tm.assert_frame_equal(result, expected)

    def test_loc_setitem_with_expansion_and_existing_dst(self):
        # GH#18308
        start = Timestamp("2017-10-29 00:00:00+0200", tz="Europe/Madrid")
        end = Timestamp("2017-10-29 03:00:00+0100", tz="Europe/Madrid")
        ts = Timestamp("2016-10-10 03:00:00", tz="Europe/Madrid")
        idx = date_range(start, end, inclusive="left", freq="h")
        assert ts not in idx  # i.e. result.loc setitem is with-expansion

        result = DataFrame(index=idx, columns=["value"])
        result.loc[ts, "value"] = 12
        expected = DataFrame(
            [np.nan] * len(idx) + [12],
            index=idx.append(DatetimeIndex([ts])),
            columns=["value"],
            dtype=object,
        )
        tm.assert_frame_equal(result, expected)

    def test_setitem_with_expansion(self):
        # indexing - setting an element
        df = DataFrame(
            data=to_datetime(["2015-03-30 20:12:32", "2015-03-12 00:11:11"]),
            columns=["time"],
        )
        df["new_col"] = ["new", "old"]
        df.time = df.set_index("time").index.tz_localize("UTC")
        v = df[df.new_col == "new"].set_index("time").index.tz_convert("US/Pacific")

        # pre-2.0  trying to set a single element on a part of a different
        #  timezone converted to object; in 2.0 it retains dtype
        df2 = df.copy()
        df2.loc[df2.new_col == "new", "time"] = v

        expected = Series([v[0].tz_convert("UTC"), df.loc[1, "time"]], name="time")
        tm.assert_series_equal(df2.time, expected)

        v = df.loc[df.new_col == "new", "time"] + Timedelta("1s")
        df.loc[df.new_col == "new", "time"] = v
        tm.assert_series_equal(df.loc[df.new_col == "new", "time"], v)

    def test_loc_setitem_with_expansion_inf_upcast_empty(self):
        # Test with np.inf in columns
        df = DataFrame()
        df.loc[0, 0] = 1
        df.loc[1, 1] = 2
        df.loc[0, np.inf] = 3

        result = df.columns
        expected = Index([0, 1, np.inf], dtype=np.float64)
        tm.assert_index_equal(result, expected)

    @pytest.mark.filterwarnings("ignore:indexing past lexsort depth")
    def test_loc_setitem_with_expansion_nonunique_index(self, index):
        # GH#40096
        if not len(index):
            pytest.skip("Not relevant for empty Index")

        index = index.repeat(2)  # ensure non-unique
        N = len(index)
        arr = np.arange(N).astype(np.int64)

        orig = DataFrame(arr, index=index, columns=[0])

        # key that will requiring object-dtype casting in the index
        key = "kapow"
        assert key not in index  # otherwise test is invalid
        # TODO: using a tuple key breaks here in many cases

        exp_index = index.insert(len(index), key)
        if isinstance(index, MultiIndex):
            assert exp_index[-1][0] == key
        else:
            assert exp_index[-1] == key
        exp_data = np.arange(N + 1).astype(np.float64)
        expected = DataFrame(exp_data, index=exp_index, columns=[0])

        # Add new row, but no new columns
        df = orig.copy()
        df.loc[key, 0] = N
        tm.assert_frame_equal(df, expected)

        # add new row on a Series
        ser = orig.copy()[0]
        ser.loc[key] = N
        # the series machinery lets us preserve int dtype instead of float
        expected = expected[0].astype(np.int64)
        tm.assert_series_equal(ser, expected)

        # add new row and new column
        df = orig.copy()
        df.loc[key, 1] = N
        expected = DataFrame(
            {0: list(arr) + [np.nan], 1: [np.nan] * N + [float(N)]},
            index=exp_index,
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "dtype", ["Int32", "Int64", "UInt32", "UInt64", "Float32", "Float64"]
    )
    def test_loc_setitem_with_expansion_preserves_nullable_int(self, dtype):
        # GH#42099
        ser = Series([0, 1, 2, 3], dtype=dtype)
        df = DataFrame({"data": ser})

        result = DataFrame(index=df.index)
        result.loc[df.index, "data"] = ser

        tm.assert_frame_equal(result, df, check_column_type=False)

        result = DataFrame(index=df.index)
        result.loc[df.index, "data"] = ser._values
        tm.assert_frame_equal(result, df, check_column_type=False)

    def test_loc_setitem_ea_not_full_column(self):
        # GH#39163
        df = DataFrame({"A": range(5)})

        val = date_range("2016-01-01", periods=3, tz="US/Pacific")

        df.loc[[0, 1, 2], "B"] = val

        bex = val.append(DatetimeIndex([pd.NaT, pd.NaT], dtype=val.dtype))
        expected = DataFrame({"A": range(5), "B": bex})
        assert expected.dtypes["B"] == val.dtype
        tm.assert_frame_equal(df, expected)


class TestLocCallable:
    def test_frame_loc_getitem_callable(self):
        # GH#11485
        df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})
        # iloc cannot use boolean Series (see GH3635)

        # return bool indexer
        res = df.loc[lambda x: x.A > 2]
        tm.assert_frame_equal(res, df.loc[df.A > 2])

        res = df.loc[lambda x: x.B == "b", :]
        tm.assert_frame_equal(res, df.loc[df.B == "b", :])

        res = df.loc[lambda x: x.A > 2, lambda x: x.columns == "B"]
        tm.assert_frame_equal(res, df.loc[df.A > 2, [False, True, False]])

        res = df.loc[lambda x: x.A > 2, lambda x: "B"]
        tm.assert_series_equal(res, df.loc[df.A > 2, "B"])

        res = df.loc[lambda x: x.A > 2, lambda x: ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])

        res = df.loc[lambda x: x.A == 2, lambda x: ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A == 2, ["A", "B"]])

        # scalar
        res = df.loc[lambda x: 1, lambda x: "A"]
        assert res == df.loc[1, "A"]

    def test_frame_loc_getitem_callable_mixture(self):
        # GH#11485
        df = DataFrame({"A": [1, 2, 3, 4], "B": list("aabb"), "C": [1, 2, 3, 4]})

        res = df.loc[lambda x: x.A > 2, ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[df.A > 2, ["A", "B"]])

        res = df.loc[[2, 3], lambda x: ["A", "B"]]
        tm.assert_frame_equal(res, df.loc[[2, 3], ["A", "B"]])

        res = df.loc[3, lambda x: ["A", "B"]]
        tm.assert_series_equal(res, df.loc[3, ["A", "B"]])

    def test_frame_loc_getitem_callable_labels(self):
        # GH#11485
        df = DataFrame({"X": [1, 2, 3, 4], "Y": list("aabb")}, index=list("ABCD"))

        # return label
        res = df.loc[lambda x: ["A", "C"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"]])

        res = df.loc[lambda x: ["A", "C"], :]
        tm.assert_frame_equal(res, df.loc[["A", "C"], :])

        res = df.loc[lambda x: ["A", "C"], lambda x: "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        res = df.loc[lambda x: ["A", "C"], lambda x: ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        # mixture
        res = df.loc[["A", "C"], lambda x: "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        res = df.loc[["A", "C"], lambda x: ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

        res = df.loc[lambda x: ["A", "C"], "X"]
        tm.assert_series_equal(res, df.loc[["A", "C"], "X"])

        res = df.loc[lambda x: ["A", "C"], ["X"]]
        tm.assert_frame_equal(res, df.loc[["A", "C"], ["X"]])

    def test_frame_loc_setitem_callable(self):
        # GH#11485
        df = DataFrame(
            {"X": [1, 2, 3, 4], "Y": Series(list("aabb"), dtype=object)},
            index=list("ABCD"),
        )

        # return label
        res = df.copy()
        res.loc[lambda x: ["A", "C"]] = -20
        exp = df.copy()
        exp.loc[["A", "C"]] = -20
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], :] = 20
        exp = df.copy()
        exp.loc[["A", "C"], :] = 20
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: "X"] = -1
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = -1
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], lambda x: ["X"]] = [5, 10]
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = [5, 10]
        tm.assert_frame_equal(res, exp)

        # mixture
        res = df.copy()
        res.loc[["A", "C"], lambda x: "X"] = np.array([-1, -2])
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = np.array([-1, -2])
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[["A", "C"], lambda x: ["X"]] = 10
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = 10
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], "X"] = -2
        exp = df.copy()
        exp.loc[["A", "C"], "X"] = -2
        tm.assert_frame_equal(res, exp)

        res = df.copy()
        res.loc[lambda x: ["A", "C"], ["X"]] = -4
        exp = df.copy()
        exp.loc[["A", "C"], ["X"]] = -4
        tm.assert_frame_equal(res, exp)


class TestPartialStringSlicing:
    def test_loc_getitem_partial_string_slicing_datetimeindex(self):
        # GH#35509
        df = DataFrame(
            {"col1": ["a", "b", "c"], "col2": [1, 2, 3]},
            index=to_datetime(["2020-08-01", "2020-07-02", "2020-08-05"]),
        )
        expected = DataFrame(
            {"col1": ["a", "c"], "col2": [1, 3]},
            index=to_datetime(["2020-08-01", "2020-08-05"]),
        )
        result = df.loc["2020-08"]
        tm.assert_frame_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_periodindex(self):
        pi = pd.period_range(start="2017-01-01", end="2018-01-01", freq="M")
        ser = pi.to_series()
        result = ser.loc[:"2017-12"]
        expected = ser.iloc[:-1]

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_partial_string_slicing_with_timedeltaindex(self):
        ix = timedelta_range(start="1 day", end="2 days", freq="1h")
        ser = ix.to_series()
        result = ser.loc[:"1 days"]
        expected = ser.iloc[:-1]

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_str_timedeltaindex(self):
        # GH#16896
        df = DataFrame({"x": range(3)}, index=to_timedelta(range(3), unit="days"))
        expected = df.iloc[0]
        sliced = df.loc["0 days"]
        tm.assert_series_equal(sliced, expected)

    @pytest.mark.parametrize("indexer_end", [None, "2020-01-02 23:59:59.999999999"])
    def test_loc_getitem_partial_slice_non_monotonicity(
        self, tz_aware_fixture, indexer_end, frame_or_series
    ):
        # GH#33146
        obj = frame_or_series(
            [1] * 5,
            index=DatetimeIndex(
                [
                    Timestamp("2019-12-30"),
                    Timestamp("2020-01-01"),
                    Timestamp("2019-12-25"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                    Timestamp("2019-12-19"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        expected = frame_or_series(
            [1] * 2,
            index=DatetimeIndex(
                [
                    Timestamp("2020-01-01"),
                    Timestamp("2020-01-02 23:59:59.999999999"),
                ],
                tz=tz_aware_fixture,
            ),
        )
        indexer = slice("2020-01-01", indexer_end)

        result = obj[indexer]
        tm.assert_equal(result, expected)

        result = obj.loc[indexer]
        tm.assert_equal(result, expected)


class TestLabelSlicing:
    def test_loc_getitem_slicing_datetimes_frame(self):
        # GH#7523

        # unique
        df_unique = DataFrame(
            np.arange(4.0, dtype="float64"),
            index=[datetime(2001, 1, i, 10, 00) for i in [1, 2, 3, 4]],
        )

        # duplicates
        df_dups = DataFrame(
            np.arange(5.0, dtype="float64"),
            index=[datetime(2001, 1, i, 10, 00) for i in [1, 2, 2, 3, 4]],
        )

        for df in [df_unique, df_dups]:
            result = df.loc[datetime(2001, 1, 1, 10) :]
            tm.assert_frame_equal(result, df)
            result = df.loc[: datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)
            result = df.loc[datetime(2001, 1, 1, 10) : datetime(2001, 1, 4, 10)]
            tm.assert_frame_equal(result, df)

            result = df.loc[datetime(2001, 1, 1, 11) :]
            expected = df.iloc[1:]
            tm.assert_frame_equal(result, expected)
            result = df.loc["20010101 11":]
            tm.assert_frame_equal(result, expected)

    def test_loc_getitem_label_slice_across_dst(self):
        # GH#21846
        idx = date_range(
            "2017-10-29 01:30:00", tz="Europe/Berlin", periods=5, freq="30 min"
        )
        series2 = Series([0, 1, 2, 3, 4], index=idx)

        t_1 = Timestamp("2017-10-29 02:30:00+02:00", tz="Europe/Berlin")
        t_2 = Timestamp("2017-10-29 02:00:00+01:00", tz="Europe/Berlin")
        result = series2.loc[t_1:t_2]
        expected = Series([2, 3], index=idx[2:4])
        tm.assert_series_equal(result, expected)

        result = series2[t_1]
        expected = 2
        assert result == expected

    @pytest.mark.parametrize(
        "index",
        [
            pd.period_range(start="2017-01-01", end="2018-01-01", freq="M"),
            timedelta_range(start="1 day", end="2 days", freq="1h"),
        ],
    )
    def test_loc_getitem_label_slice_period_timedelta(self, index):
        ser = index.to_series()
        result = ser.loc[: index[-2]]
        expected = ser.iloc[:-1]

        tm.assert_series_equal(result, expected)

    def test_loc_getitem_slice_floats_inexact(self):
        index = [52195.504153, 52196.303147, 52198.369883]
        df = DataFrame(np.random.default_rng(2).random((3, 2)), index=index)

        s1 = df.loc[52195.1:52196.5]
        assert len(s1) == 2

        s1 = df.loc[52195.1:52196.6]
        assert len(s1) == 2

        s1 = df.loc[52195.1:52198.9]
        assert len(s1) == 3

    def test_loc_getitem_float_slice_floatindex(self, float_numpy_dtype):
        dtype = float_numpy_dtype
        ser = Series(
            np.random.default_rng(2).random(10), index=np.arange(10, 20, dtype=dtype)
        )

        assert len(ser.loc[12.0:]) == 8
        assert len(ser.loc[12.5:]) == 7

        idx = np.arange(10, 20, dtype=dtype)
        idx[2] = 12.2
        ser.index = idx
        assert len(ser.loc[12.0:]) == 8
        assert len(ser.loc[12.5:]) == 7

    @pytest.mark.parametrize(
        "start,stop, expected_slice",
        [
            [np.timedelta64(0, "ns"), None, slice(0, 11)],
            [np.timedelta64(1, "D"), np.timedelta64(6, "D"), slice(1, 7)],
            [None, np.timedelta64(4, "D"), slice(0, 5)],
        ],
    )
    def test_loc_getitem_slice_label_td64obj(self, start, stop, expected_slice):
        # GH#20393
        ser = Series(range(11), timedelta_range("0 days", "10 days"))
        result = ser.loc[slice(start, stop)]
        expected = ser.iloc[expected_slice]
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("start", ["2018", "2020"])
    def test_loc_getitem_slice_unordered_dt_index(self, frame_or_series, start):
        obj = frame_or_series(
            [1, 2, 3],
            index=[Timestamp("2016"), Timestamp("2019"), Timestamp("2017")],
        )
        with pytest.raises(
            KeyError, match="Value based partial slicing on non-monotonic"
        ):
            obj.loc[start:"2022"]

    @pytest.mark.parametrize("value", [1, 1.5])
    def test_loc_getitem_slice_labels_int_in_object_index(self, frame_or_series, value):
        # GH: 26491
        obj = frame_or_series(range(4), index=[value, "first", 2, "third"])
        result = obj.loc[value:"third"]
        expected = frame_or_series(range(4), index=[value, "first", 2, "third"])
        tm.assert_equal(result, expected)

    def test_loc_getitem_slice_columns_mixed_dtype(self):
        # GH: 20975
        df = DataFrame({"test": 1, 1: 2, 2: 3}, index=[0])
        expected = DataFrame(
            data=[[2, 3]], index=[0], columns=Index([1, 2], dtype=object)
        )
        tm.assert_frame_equal(df.loc[:, 1:], expected)


class TestLocBooleanLabelsAndSlices:
    @pytest.mark.parametrize("bool_value", [True, False])
    def test_loc_bool_incompatible_index_raises(
        self, index, frame_or_series, bool_value
    ):
        # GH20432
        message = f"{bool_value}: boolean label can not be used without a boolean index"
        if index.inferred_type != "boolean":
            obj = frame_or_series(index=index, dtype="object")
            with pytest.raises(KeyError, match=message):
                obj.loc[bool_value]

    @pytest.mark.parametrize("bool_value", [True, False])
    def test_loc_bool_should_not_raise(self, frame_or_series, bool_value):
        obj = frame_or_series(
            index=Index([True, False], dtype="boolean"), dtype="object"
        )
        obj.loc[bool_value]

    def test_loc_bool_slice_raises(self, index, frame_or_series):
        # GH20432
        message = (
            r"slice\(True, False, None\): boolean values can not be used in a slice"
        )
        obj = frame_or_series(index=index, dtype="object")
        with pytest.raises(TypeError, match=message):
            obj.loc[True:False]


class TestLocBooleanMask:
    def test_loc_setitem_bool_mask_timedeltaindex(self):
        # GH#14946
        df = DataFrame({"x": range(10)})
        df.index = to_timedelta(range(10), unit="s")
        conditions = [df["x"] > 3, df["x"] == 3, df["x"] < 3]
        expected_data = [
            [0, 1, 2, 3, 10, 10, 10, 10, 10, 10],
            [0, 1, 2, 10, 4, 5, 6, 7, 8, 9],
            [10, 10, 10, 3, 4, 5, 6, 7, 8, 9],
        ]
        for cond, data in zip(conditions, expected_data):
            result = df.copy()
            result.loc[cond, "x"] = 10

            expected = DataFrame(
                data,
                index=to_timedelta(range(10), unit="s"),
                columns=["x"],
                dtype="int64",
            )
            tm.assert_frame_equal(expected, result)

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_loc_setitem_mask_with_datetimeindex_tz(self, tz):
        # GH#16889
        # support .loc with alignment and tz-aware DatetimeIndex
        mask = np.array([True, False, True, False])

        idx = date_range("20010101", periods=4, tz=tz)
        df = DataFrame({"a": np.arange(4)}, index=idx).astype("float64")

        result = df.copy()
        result.loc[mask, :] = df.loc[mask, :]
        tm.assert_frame_equal(result, df)

        result = df.copy()
        result.loc[mask] = df.loc[mask]
        tm.assert_frame_equal(result, df)

    def test_loc_setitem_mask_and_label_with_datetimeindex(self):
        # GH#9478
        # a datetimeindex alignment issue with partial setting
        df = DataFrame(
            np.arange(6.0).reshape(3, 2),
            columns=list("AB"),
            index=date_range("1/1/2000", periods=3, freq="1h"),
        )
        expected = df.copy()
        expected["C"] = [expected.index[0]] + [pd.NaT, pd.NaT]

        mask = df.A < 1
        df.loc[mask, "C"] = df.loc[mask].index
        tm.assert_frame_equal(df, expected)

    def test_loc_setitem_mask_td64_series_value(self):
        # GH#23462 key list of bools, value is a Series
        td1 = Timedelta(0)
        td2 = Timedelta(28767471428571405)
        df = DataFrame({"col": Series([td1, td2])})
        df_copy = df.copy()
        ser = Series([td1])

        expected = df["col"].iloc[1]._value
        df.loc[[True, False]] = ser
        result = df["col"].iloc[1]._value

        assert expected == result
        tm.assert_frame_equal(df, df_copy)

    @td.skip_array_manager_invalid_test  # TODO(ArrayManager) rewrite not using .values
    def test_loc_setitem_boolean_and_column(self, float_frame):
        expected = float_frame.copy()
        mask = float_frame["A"] > 0

        float_frame.loc[mask, "B"] = 0

        values = expected.values.copy()
        values[mask.values, 1] = 0
        expected = DataFrame(values, index=expected.index, columns=expected.columns)
        tm.assert_frame_equal(float_frame, expected)

    def test_loc_setitem_ndframe_values_alignment(
        self, using_copy_on_write, warn_copy_on_write
    ):
        # GH#45501
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], ["a"]] = DataFrame(
            {"a": [10, 20, 30]}, index=[2, 1, 0]
        )

        expected = DataFrame({"a": [1, 2, 10], "b": [4, 5, 6]})
        tm.assert_frame_equal(df, expected)

        # same thing with Series RHS
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], ["a"]] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

        # same thing but setting "a" instead of ["a"]
        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.loc[[False, False, True], "a"] = Series([10, 11, 12], index=[2, 1, 0])
        tm.assert_frame_equal(df, expected)

        df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df_orig = df.copy()
        ser = df["a"]
        with tm.assert_cow_warning(warn_copy_on_write):
            ser.loc[[False, False, True]] = Series([10, 11, 12], index=[2, 1, 0])
        if using_copy_on_write:
            tm.assert_frame_equal(df, df_orig)
        else:
            tm.assert_frame_equal(df, expected)

    def test_loc_indexer_empty_broadcast(self):
        # GH#51450
        df = DataFrame({"a": [], "b": []}, dtype=object)
        expected = df.copy()
        df.loc[np.array([], dtype=np.bool_), ["a"]] = df["a"].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_all_false_broadcast(self):
        # GH#51450
        df = DataFrame({"a": ["x"], "b": ["y"]}, dtype=object)
        expected = df.copy()
        df.loc[np.array([False], dtype=np.bool_), ["a"]] = df["b"].copy()
        tm.assert_frame_equal(df, expected)

    def test_loc_indexer_length_one(self):
        # GH#51435
        df = DataFrame({"a": ["x"], "b": ["y"]}, dtype=object)
        expected = DataFrame({"a": ["y"], "b": ["y"]}, dtype=object)
        df.loc[np.array([True], dtype=np.bool_), ["a"]] = df["b"].copy()
        tm.assert_frame_equal(df, expected)


class TestLocListlike:
    @pytest.mark.parametrize("box", [lambda x: x, np.asarray, list])
    def test_loc_getitem_list_of_labels_categoricalindex_with_na(self, box):
        # passing a list can include valid categories _or_ NA values
        ci = CategoricalIndex(["A", "B", np.nan])
        ser = Series(range(3), index=ci)

        result = ser.loc[box(ci)]
        tm.assert_series_equal(result, ser)

        result = ser[box(ci)]
        tm.assert_series_equal(result, ser)

        result = ser.to_frame().loc[box(ci)]
        tm.assert_frame_equal(result, ser.to_frame())

        ser2 = ser[:-1]
        ci2 = ci[1:]
        # but if there are no NAs present, this should raise KeyError
        msg = "not in index"
        with pytest.raises(KeyError, match=msg):
            ser2.loc[box(ci2)]

        with pytest.raises(KeyError, match=msg):
            ser2[box(ci2)]

        with pytest.raises(KeyError, match=msg):
            ser2.to_frame().loc[box(ci2)]

    def test_loc_getitem_series_label_list_missing_values(self):
        # gh-11428
        key = np.array(
            ["2001-01-04", "2001-01-02", "2001-01-04", "2001-01-14"], dtype="datetime64"
        )
        ser = Series([2, 5, 8, 11], date_range("2001-01-01", freq="D", periods=4))
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[key]

    def test_loc_getitem_series_label_list_missing_integer_values(self):
        # GH: 25927
        ser = Series(
            index=np.array([9730701000001104, 10049011000001109]),
            data=np.array([999000011000001104, 999000011000001104]),
        )
        with pytest.raises(KeyError, match="not in index"):
            ser.loc[np.array([9730701000001104, 10047311000001102])]

    @pytest.mark.parametrize("to_period", [True, False])
    def test_loc_getitem_listlike_of_datetimelike_keys(self, to_period):
        # GH#11497

        idx = date_range("2011-01-01", "2011-01-02", freq="D", name="idx")
        if to_period:
            idx = idx.to_period("D")
        ser = Series([0.1, 0.2], index=idx, name="s")

        keys = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        if to_period:
            keys = [x.to_period("D") for x in keys]
        result = ser.loc[keys]
        exp = Series([0.1, 0.2], index=idx, name="s")
        if not to_period:
            exp.index = exp.index._with_freq(None)
        tm.assert_series_equal(result, exp, check_index_type=True)

        keys = [
            Timestamp("2011-01-02"),
            Timestamp("2011-01-02"),
            Timestamp("2011-01-01"),
        ]
        if to_period:
            keys = [x.to_period("D") for x in keys]
        exp = Series(
            [0.2, 0.2, 0.1], index=Index(keys, name="idx", dtype=idx.dtype), name="s"
        )
        result = ser.loc[keys]
        tm.assert_series_equal(result, exp, check_index_type=True)

        keys = [
            Timestamp("2011-01-03"),
            Timestamp("2011-01-02"),
            Timestamp("2011-01-03"),
        ]
        if to_period:
            keys = [x.to_period("D") for x in keys]

        with pytest.raises(KeyError, match="not in index"):
            ser.loc[keys]

    def test_loc_named_index(self):
        # GH 42790
        df = DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        expected = df.iloc[:2]
        expected.index.name = "foo"
        result = df.loc[Index(["cobra", "viper"], name="foo")]
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "columns, column_key, expected_columns",
    [
        ([2011, 2012, 2013], [2011, 2012], [0, 1]),
        ([2011, 2012, "All"], [2011, 2012], [0, 1]),
        ([2011, 2012, "All"], [2011, "All"], [0, 2]),
    ],
)
def test_loc_getitem_label_list_integer_labels(columns, column_key, expected_columns):
    # gh-14836
    df = DataFrame(
        np.random.default_rng(2).random((3, 3)), columns=columns, index=list("ABC")
    )
    expected = df.iloc[:, expected_columns]
    result = df.loc[["A", "B", "C"], column_key]

    tm.assert_frame_equal(result, expected, check_column_type=True)


def test_loc_setitem_float_intindex():
    # GH 8720
    rand_data = np.random.default_rng(2).standard_normal((8, 4))
    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    expected_data = np.hstack((rand_data, np.array([np.nan] * 8).reshape(8, 1)))
    expected = DataFrame(expected_data, columns=[0.0, 1.0, 2.0, 3.0, 0.5])
    tm.assert_frame_equal(result, expected)

    result = DataFrame(rand_data)
    result.loc[:, 0.5] = np.nan
    tm.assert_frame_equal(result, expected)


def test_loc_axis_1_slice():
    # GH 10586
    cols = [(yr, m) for yr in [2014, 2015] for m in [7, 8, 9, 10]]
    df = DataFrame(
        np.ones((10, 8)),
        index=tuple("ABCDEFGHIJ"),
        columns=MultiIndex.from_tuples(cols),
    )
    result = df.loc(axis=1)[(2014, 9):(2015, 8)]
    expected = DataFrame(
        np.ones((10, 4)),
        index=tuple("ABCDEFGHIJ"),
        columns=MultiIndex.from_tuples([(2014, 9), (2014, 10), (2015, 7), (2015, 8)]),
    )
    tm.assert_frame_equal(result, expected)


def test_loc_set_dataframe_multiindex():
    # GH 14592
    expected = DataFrame(
        "a", index=range(2), columns=MultiIndex.from_product([range(2), range(2)])
    )
    result = expected.copy()
    result.loc[0, [(0, 1)]] = result.loc[0, [(0, 1)]]
    tm.assert_frame_equal(result, expected)


def test_loc_mixed_int_float():
    # GH#19456
    ser = Series(range(2), Index([1, 2.0], dtype=object))

    result = ser.loc[1]
    assert result == 0


def test_loc_with_positional_slice_raises():
    # GH#31840
    ser = Series(range(4), index=["A", "B", "C", "D"])

    with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
        ser.loc[:3] = 2


def test_loc_slice_disallows_positional():
    # GH#16121, GH#24612, GH#31810
    dti = date_range("2016-01-01", periods=3)
    df = DataFrame(np.random.default_rng(2).random((3, 2)), index=dti)

    ser = df[0]

    msg = (
        "cannot do slice indexing on DatetimeIndex with these "
        r"indexers \[1\] of type int"
    )

    for obj in [df, ser]:
        with pytest.raises(TypeError, match=msg):
            obj.loc[1:3]

        with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
            # GH#31840 enforce incorrect behavior
            obj.loc[1:3] = 1

    with pytest.raises(TypeError, match=msg):
        df.loc[1:3, 1]

    with pytest.raises(TypeError, match="Slicing a positional slice with .loc"):
        # GH#31840 enforce incorrect behavior
        df.loc[1:3, 1] = 2


def test_loc_datetimelike_mismatched_dtypes():
    # GH#32650 dont mix and match datetime/timedelta/period dtypes

    df = DataFrame(
        np.random.default_rng(2).standard_normal((5, 3)),
        columns=["a", "b", "c"],
        index=date_range("2012", freq="h", periods=5),
    )
    # create dataframe with non-unique DatetimeIndex
    df = df.iloc[[0, 2, 2, 3]].copy()

    dti = df.index
    tdi = pd.TimedeltaIndex(dti.asi8)  # matching i8 values

    msg = r"None of \[TimedeltaIndex.* are in the \[index\]"
    with pytest.raises(KeyError, match=msg):
        df.loc[tdi]

    with pytest.raises(KeyError, match=msg):
        df["a"].loc[tdi]


def test_loc_with_period_index_indexer():
    # GH#4125
    idx = pd.period_range("2002-01", "2003-12", freq="M")
    df = DataFrame(np.random.default_rng(2).standard_normal((24, 10)), index=idx)
    tm.assert_frame_equal(df, df.loc[idx])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df, df.loc[list(idx)])
    tm.assert_frame_equal(df.iloc[0:5], df.loc[idx[0:5]])
    tm.assert_frame_equal(df, df.loc[list(idx)])


def test_loc_setitem_multiindex_timestamp():
    # GH#13831
    vals = np.random.default_rng(2).standard_normal((8, 6))
    idx = date_range("1/1/2000", periods=8)
    cols = ["A", "B", "C", "D", "E", "F"]
    exp = DataFrame(vals, index=idx, columns=cols)
    exp.loc[exp.index[1], ("A", "B")] = np.nan
    vals[1][0:2] = np.nan
    res = DataFrame(vals, index=idx, columns=cols)
    tm.assert_frame_equal(res, exp)


def test_loc_getitem_multiindex_tuple_level():
    # GH#27591
    lev1 = ["a", "b", "c"]
    lev2 = [(0, 1), (1, 0)]
    lev3 = [0, 1]
    cols = MultiIndex.from_product([lev1, lev2, lev3], names=["x", "y", "z"])
    df = DataFrame(6, index=range(5), columns=cols)

    # the lev2[0] here should be treated as a single label, not as a sequence
    #  of labels
    result = df.loc[:, (lev1[0], lev2[0], lev3[0])]

    # TODO: i think this actually should drop levels
    expected = df.iloc[:, :1]
    tm.assert_frame_equal(result, expected)

    alt = df.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=1)
    tm.assert_frame_equal(alt, expected)

    # same thing on a Series
    ser = df.iloc[0]
    expected2 = ser.iloc[:1]

    alt2 = ser.xs((lev1[0], lev2[0], lev3[0]), level=[0, 1, 2], axis=0)
    tm.assert_series_equal(alt2, expected2)

    result2 = ser.loc[lev1[0], lev2[0], lev3[0]]
    assert result2 == 6


def test_loc_getitem_nullable_index_with_duplicates():
    # GH#34497
    df = DataFrame(
        data=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, np.nan, np.nan]]).T,
        columns=["a", "b", "c"],
        dtype="Int64",
    )
    df2 = df.set_index("c")
    assert df2.index.dtype == "Int64"

    res = df2.loc[1]
    expected = Series([1, 5], index=df2.columns, dtype="Int64", name=1)
    tm.assert_series_equal(res, expected)

    # pd.NA and duplicates in an object-dtype Index
    df2.index = df2.index.astype(object)
    res = df2.loc[1]
    tm.assert_series_equal(res, expected)


@pytest.mark.parametrize("value", [300, np.uint16(300), np.int16(300)])
def test_loc_setitem_uint8_upcast(value):
    # GH#26049

    df = DataFrame([1, 2, 3, 4], columns=["col1"], dtype="uint8")
    with tm.assert_produces_warning(FutureWarning, match="item of incompatible dtype"):
        df.loc[2, "col1"] = value  # value that can't be held in uint8

    if np_version_gt2 and isinstance(value, np.int16):
        # Note, result type of uint8 + int16 is int16
        # in numpy < 2, though, numpy would inspect the
        # value and see that it could fit in an uint16, resulting in a uint16
        dtype = "int16"
    else:
        dtype = "uint16"

    expected = DataFrame([1, 2, 300, 4], columns=["col1"], dtype=dtype)
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "fill_val,exp_dtype",
    [
        (Timestamp("2022-01-06"), "datetime64[ns]"),
        (Timestamp("2022-01-07", tz="US/Eastern"), "datetime64[ns, US/Eastern]"),
    ],
)
def test_loc_setitem_using_datetimelike_str_as_index(fill_val, exp_dtype):
    data = ["2022-01-02", "2022-01-03", "2022-01-04", fill_val.date()]
    index = DatetimeIndex(data, tz=fill_val.tz, dtype=exp_dtype)
    df = DataFrame([10, 11, 12, 14], columns=["a"], index=index)
    # adding new row using an unexisting datetime-like str index
    df.loc["2022-01-08", "a"] = 13

    data.append("2022-01-08")
    expected_index = DatetimeIndex(data, dtype=exp_dtype)
    tm.assert_index_equal(df.index, expected_index, exact=True)


def test_loc_set_int_dtype():
    # GH#23326
    df = DataFrame([list("abc")])
    df.loc[:, "col1"] = 5

    expected = DataFrame({0: ["a"], 1: ["b"], 2: ["c"], "col1": [5]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_loc_periodindex_3_levels():
    # GH#24091
    p_index = PeriodIndex(
        ["20181101 1100", "20181101 1200", "20181102 1300", "20181102 1400"],
        name="datetime",
        freq="B",
    )
    mi_series = DataFrame(
        [["A", "B", 1.0], ["A", "C", 2.0], ["Z", "Q", 3.0], ["W", "F", 4.0]],
        index=p_index,
        columns=["ONE", "TWO", "VALUES"],
    )
    mi_series = mi_series.set_index(["ONE", "TWO"], append=True)["VALUES"]
    assert mi_series.loc[(p_index[0], "A", "B")] == 1.0


def test_loc_setitem_pyarrow_strings():
    # GH#52319
    pytest.importorskip("pyarrow")
    df = DataFrame(
        {
            "strings": Series(["A", "B", "C"], dtype="string[pyarrow]"),
            "ids": Series([True, True, False]),
        }
    )
    new_value = Series(["X", "Y"])
    df.loc[df.ids, "strings"] = new_value

    expected_df = DataFrame(
        {
            "strings": Series(["X", "Y", "C"], dtype="string[pyarrow]"),
            "ids": Series([True, True, False]),
        }
    )

    tm.assert_frame_equal(df, expected_df)


class TestLocSeries:
    @pytest.mark.parametrize("val,expected", [(2**63 - 1, 3), (2**63, 4)])
    def test_loc_uint64(self, val, expected):
        # see GH#19399
        ser = Series({2**63 - 1: 3, 2**63: 4})
        assert ser.loc[val] == expected

    def test_loc_getitem(self, string_series, datetime_series):
        inds = string_series.index[[3, 4, 7]]
        tm.assert_series_equal(string_series.loc[inds], string_series.reindex(inds))
        tm.assert_series_equal(string_series.iloc[5::2], string_series[5::2])

        # slice with indices
        d1, d2 = datetime_series.index[[5, 15]]
        result = datetime_series.loc[d1:d2]
        expected = datetime_series.truncate(d1, d2)
        tm.assert_series_equal(result, expected)

        # boolean
        mask = string_series > string_series.median()
        tm.assert_series_equal(string_series.loc[mask], string_series[mask])

        # ask for index value
        assert datetime_series.loc[d1] == datetime_series[d1]
        assert datetime_series.loc[d2] == datetime_series[d2]

    def test_loc_getitem_not_monotonic(self, datetime_series):
        d1, d2 = datetime_series.index[[5, 15]]

        ts2 = datetime_series[::2].iloc[[1, 2, 0]]

        msg = r"Timestamp\('2000-01-10 00:00:00'\)"
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2]
        with pytest.raises(KeyError, match=msg):
            ts2.loc[d1:d2] = 0

    def test_loc_getitem_setitem_integer_slice_keyerrors(self):
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )

        # this is OK
        cp = ser.copy()
        cp.iloc[4:10] = 0
        assert (cp.iloc[4:10] == 0).all()

        # so is this
        cp = ser.copy()
        cp.iloc[3:11] = 0
        assert (cp.iloc[3:11] == 0).values.all()

        result = ser.iloc[2:6]
        result2 = ser.loc[3:11]
        expected = ser.reindex([4, 6, 8, 10])

        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        # non-monotonic, raise KeyError
        s2 = ser.iloc[list(range(5)) + list(range(9, 4, -1))]
        with pytest.raises(KeyError, match=r"^3$"):
            s2.loc[3:11]
        with pytest.raises(KeyError, match=r"^3$"):
            s2.loc[3:11] = 0

    def test_loc_getitem_iterator(self, string_series):
        idx = iter(string_series.index[:10])
        result = string_series.loc[idx]
        tm.assert_series_equal(result, string_series[:10])

    def test_loc_setitem_boolean(self, string_series):
        mask = string_series > string_series.median()

        result = string_series.copy()
        result.loc[mask] = 0
        expected = string_series
        expected[mask] = 0
        tm.assert_series_equal(result, expected)

    def test_loc_setitem_corner(self, string_series):
        inds = list(string_series.index[[5, 8, 12]])
        string_series.loc[inds] = 5
        msg = r"\['foo'\] not in index"
        with pytest.raises(KeyError, match=msg):
            string_series.loc[inds + ["foo"]] = 5

    def test_basic_setitem_with_labels(self, datetime_series):
        indices = datetime_series.index[[5, 10, 15]]

        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices] = 0
        exp.loc[indices] = 0
        tm.assert_series_equal(cp, exp)

        cp = datetime_series.copy()
        exp = datetime_series.copy()
        cp[indices[0] : indices[2]] = 0
        exp.loc[indices[0] : indices[2]] = 0
        tm.assert_series_equal(cp, exp)

    def test_loc_setitem_listlike_of_ints(self):
        # integer indexes, be careful
        ser = Series(
            np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2))
        )
        inds = [0, 4, 6]
        arr_inds = np.array([0, 4, 6])

        cp = ser.copy()
        exp = ser.copy()
        ser[inds] = 0
        ser.loc[inds] = 0
        tm.assert_series_equal(cp, exp)

        cp = ser.copy()
        exp = ser.copy()
        ser[arr_inds] = 0
        ser.loc[arr_inds] = 0
        tm.assert_series_equal(cp, exp)

        inds_notfound = [0, 4, 5, 6]
        arr_inds_notfound = np.array([0, 4, 5, 6])
        msg = r"\[5\] not in index"
        with pytest.raises(KeyError, match=msg):
            ser[inds_notfound] = 0
        with pytest.raises(Exception, match=msg):
            ser[arr_inds_notfound] = 0

    def test_loc_setitem_dt64tz_values(self):
        # GH#12089
        ser = Series(
            date_range("2011-01-01", periods=3, tz="US/Eastern"),
            index=["a", "b", "c"],
        )
        s2 = ser.copy()
        expected = Timestamp("2011-01-03", tz="US/Eastern")
        s2.loc["a"] = expected
        result = s2.loc["a"]
        assert result == expected

        s2 = ser.copy()
        s2.iloc[0] = expected
        result = s2.iloc[0]
        assert result == expected

        s2 = ser.copy()
        s2["a"] = expected
        result = s2["a"]
        assert result == expected

    @pytest.mark.parametrize("array_fn", [np.array, pd.array, list, tuple])
    @pytest.mark.parametrize("size", [0, 4, 5, 6])
    def test_loc_iloc_setitem_with_listlike(self, size, array_fn):
        # GH37748
        # testing insertion, in a Series of size N (here 5), of a listlike object
        # of size  0, N-1, N, N+1

        arr = array_fn([0] * size)
        expected = Series([arr, 0, 0, 0, 0], index=list("abcde"), dtype=object)

        ser = Series(0, index=list("abcde"), dtype=object)
        ser.loc["a"] = arr
        tm.assert_series_equal(ser, expected)

        ser = Series(0, index=list("abcde"), dtype=object)
        ser.iloc[0] = arr
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("indexer", [IndexSlice["A", :], ("A", slice(None))])
    def test_loc_series_getitem_too_many_dimensions(self, indexer):
        # GH#35349
        ser = Series(
            index=MultiIndex.from_tuples([("A", "0"), ("A", "1"), ("B", "0")]),
            data=[21, 22, 23],
        )
        msg = "Too many indexers"
        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :]

        with pytest.raises(IndexingError, match=msg):
            ser.loc[indexer, :] = 1

    def test_loc_setitem(self, string_series):
        inds = string_series.index[[3, 4, 7]]

        result = string_series.copy()
        result.loc[inds] = 5

        expected = string_series.copy()
        expected.iloc[[3, 4, 7]] = 5
        tm.assert_series_equal(result, expected)

        result.iloc[5:10] = 10
        expected[5:10] = 10
        tm.assert_series_equal(result, expected)

        # set slice with indices
        d1, d2 = string_series.index[[5, 15]]
        result.loc[d1:d2] = 6
        expected[5:16] = 6  # because it's inclusive
        tm.assert_series_equal(result, expected)

        # set index value
        string_series.loc[d1] = 4
        string_series.loc[d2] = 6
        assert string_series[d1] == 4
        assert string_series[d2] == 6

    @pytest.mark.parametrize("dtype", ["object", "string"])
    def test_loc_assign_dict_to_row(self, dtype):
        # GH41044
        df = DataFrame({"A": ["abc", "def"], "B": ["ghi", "jkl"]}, dtype=dtype)
        df.loc[0, :] = {"A": "newA", "B": "newB"}

        expected = DataFrame({"A": ["newA", "def"], "B": ["newB", "jkl"]}, dtype=dtype)

        tm.assert_frame_equal(df, expected)

    @td.skip_array_manager_invalid_test
    def test_loc_setitem_dict_timedelta_multiple_set(self):
        # GH 16309
        result = DataFrame(columns=["time", "value"])
        result.loc[1] = {"time": Timedelta(6, unit="s"), "value": "foo"}
        result.loc[1] = {"time": Timedelta(6, unit="s"), "value": "foo"}
        expected = DataFrame(
            [[Timedelta(6, unit="s"), "foo"]], columns=["time", "value"], index=[1]
        )
        tm.assert_frame_equal(result, expected)

    def test_loc_set_multiple_items_in_multiple_new_columns(self):
        # GH 25594
        df = DataFrame(index=[1, 2], columns=["a"])
        df.loc[1, ["b", "c"]] = [6, 7]

        expected = DataFrame(
            {
                "a": Series([np.nan, np.nan], dtype="object"),
                "b": [6, np.nan],
                "c": [7, np.nan],
            },
            index=[1, 2],
        )

        tm.assert_frame_equal(df, expected)

    def test_getitem_loc_str_periodindex(self):
        # GH#33964
        msg = "Period with BDay freq is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            index = pd.period_range(start="2000", periods=20, freq="B")
            series = Series(range(20), index=index)
            assert series.loc["2000-01-14"] == 9

    def test_loc_nonunique_masked_index(self):
        # GH 57027
        ids = list(range(11))
        index = Index(ids * 1000, dtype="Int64")
        df = DataFrame({"val": np.arange(len(index), dtype=np.intp)}, index=index)
        result = df.loc[ids]
        expected = DataFrame(
            {"val": index.argsort(kind="stable").astype(np.intp)},
            index=Index(np.array(ids).repeat(1000), dtype="Int64"),
        )
        tm.assert_frame_equal(result, expected)