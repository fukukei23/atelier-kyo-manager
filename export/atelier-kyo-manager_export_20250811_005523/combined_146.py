
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\timeout.py ===
from __future__ import annotations

import time
import typing
from enum import Enum
from socket import getdefaulttimeout

from ..exceptions import TimeoutStateError

if typing.TYPE_CHECKING:
    from typing import Final


class _TYPE_DEFAULT(Enum):
    # This value should never be passed to socket.settimeout() so for safety we use a -1.
    # socket.settimout() raises a ValueError for negative values.
    token = -1


_DEFAULT_TIMEOUT: Final[_TYPE_DEFAULT] = _TYPE_DEFAULT.token

_TYPE_TIMEOUT = typing.Optional[typing.Union[float, _TYPE_DEFAULT]]


class Timeout:
    """Timeout configuration.

    Timeouts can be defined as a default for a pool:

    .. code-block:: python

        import urllib3

        timeout = urllib3.util.Timeout(connect=2.0, read=7.0)

        http = urllib3.PoolManager(timeout=timeout)

        resp = http.request("GET", "https://example.com/")

        print(resp.status)

    Or per-request (which overrides the default for the pool):

    .. code-block:: python

       response = http.request("GET", "https://example.com/", timeout=Timeout(10))

    Timeouts can be disabled by setting all the parameters to ``None``:

    .. code-block:: python

       no_timeout = Timeout(connect=None, read=None)
       response = http.request("GET", "https://example.com/", timeout=no_timeout)


    :param total:
        This combines the connect and read timeouts into one; the read timeout
        will be set to the time leftover from the connect attempt. In the
        event that both a connect timeout and a total are specified, or a read
        timeout and a total are specified, the shorter timeout will be applied.

        Defaults to None.

    :type total: int, float, or None

    :param connect:
        The maximum amount of time (in seconds) to wait for a connection
        attempt to a server to succeed. Omitting the parameter will default the
        connect timeout to the system default, probably `the global default
        timeout in socket.py
        <http://hg.python.org/cpython/file/603b4d593758/Lib/socket.py#l535>`_.
        None will set an infinite timeout for connection attempts.

    :type connect: int, float, or None

    :param read:
        The maximum amount of time (in seconds) to wait between consecutive
        read operations for a response from the server. Omitting the parameter
        will default the read timeout to the system default, probably `the
        global default timeout in socket.py
        <http://hg.python.org/cpython/file/603b4d593758/Lib/socket.py#l535>`_.
        None will set an infinite timeout.

    :type read: int, float, or None

    .. note::

        Many factors can affect the total amount of time for urllib3 to return
        an HTTP response.

        For example, Python's DNS resolver does not obey the timeout specified
        on the socket. Other factors that can affect total request time include
        high CPU load, high swap, the program running at a low priority level,
        or other behaviors.

        In addition, the read and total timeouts only measure the time between
        read operations on the socket connecting the client and the server,
        not the total amount of time for the request to return a complete
        response. For most requests, the timeout is raised because the server
        has not sent the first byte in the specified time. This is not always
        the case; if a server streams one byte every fifteen seconds, a timeout
        of 20 seconds will not trigger, even though the request will take
        several minutes to complete.
    """

    #: A sentinel object representing the default timeout value
    DEFAULT_TIMEOUT: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT

    def __init__(
        self,
        total: _TYPE_TIMEOUT = None,
        connect: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        read: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
    ) -> None:
        self._connect = self._validate_timeout(connect, "connect")
        self._read = self._validate_timeout(read, "read")
        self.total = self._validate_timeout(total, "total")
        self._start_connect: float | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(connect={self._connect!r}, read={self._read!r}, total={self.total!r})"

    # __str__ provided for backwards compatibility
    __str__ = __repr__

    @staticmethod
    def resolve_default_timeout(timeout: _TYPE_TIMEOUT) -> float | None:
        return getdefaulttimeout() if timeout is _DEFAULT_TIMEOUT else timeout

    @classmethod
    def _validate_timeout(cls, value: _TYPE_TIMEOUT, name: str) -> _TYPE_TIMEOUT:
        """Check that a timeout attribute is valid.

        :param value: The timeout value to validate
        :param name: The name of the timeout attribute to validate. This is
            used to specify in error messages.
        :return: The validated and casted version of the given value.
        :raises ValueError: If it is a numeric value less than or equal to
            zero, or the type is not an integer, float, or None.
        """
        if value is None or value is _DEFAULT_TIMEOUT:
            return value

        if isinstance(value, bool):
            raise ValueError(
                "Timeout cannot be a boolean value. It must "
                "be an int, float or None."
            )
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Timeout value %s was %s, but it must be an "
                "int, float or None." % (name, value)
            ) from None

        try:
            if value <= 0:
                raise ValueError(
                    "Attempted to set %s timeout to %s, but the "
                    "timeout cannot be set to a value less "
                    "than or equal to 0." % (name, value)
                )
        except TypeError:
            raise ValueError(
                "Timeout value %s was %s, but it must be an "
                "int, float or None." % (name, value)
            ) from None

        return value

    @classmethod
    def from_float(cls, timeout: _TYPE_TIMEOUT) -> Timeout:
        """Create a new Timeout from a legacy timeout value.

        The timeout value used by httplib.py sets the same timeout on the
        connect(), and recv() socket requests. This creates a :class:`Timeout`
        object that sets the individual timeouts to the ``timeout`` value
        passed to this function.

        :param timeout: The legacy timeout value.
        :type timeout: integer, float, :attr:`urllib3.util.Timeout.DEFAULT_TIMEOUT`, or None
        :return: Timeout object
        :rtype: :class:`Timeout`
        """
        return Timeout(read=timeout, connect=timeout)

    def clone(self) -> Timeout:
        """Create a copy of the timeout object

        Timeout properties are stored per-pool but each request needs a fresh
        Timeout object to ensure each one has its own start/stop configured.

        :return: a copy of the timeout object
        :rtype: :class:`Timeout`
        """
        # We can't use copy.deepcopy because that will also create a new object
        # for _GLOBAL_DEFAULT_TIMEOUT, which socket.py uses as a sentinel to
        # detect the user default.
        return Timeout(connect=self._connect, read=self._read, total=self.total)

    def start_connect(self) -> float:
        """Start the timeout clock, used during a connect() attempt

        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to start a timer that has been started already.
        """
        if self._start_connect is not None:
            raise TimeoutStateError("Timeout timer has already been started.")
        self._start_connect = time.monotonic()
        return self._start_connect

    def get_connect_duration(self) -> float:
        """Gets the time elapsed since the call to :meth:`start_connect`.

        :return: Elapsed time in seconds.
        :rtype: float
        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to get duration for a timer that hasn't been started.
        """
        if self._start_connect is None:
            raise TimeoutStateError(
                "Can't get connect duration for timer that has not started."
            )
        return time.monotonic() - self._start_connect

    @property
    def connect_timeout(self) -> _TYPE_TIMEOUT:
        """Get the value to use when setting a connection timeout.

        This will be a positive float or integer, the value None
        (never timeout), or the default system timeout.

        :return: Connect timeout.
        :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
        """
        if self.total is None:
            return self._connect

        if self._connect is None or self._connect is _DEFAULT_TIMEOUT:
            return self.total

        return min(self._connect, self.total)  # type: ignore[type-var]

    @property
    def read_timeout(self) -> float | None:
        """Get the value for the read timeout.

        This assumes some time has elapsed in the connection timeout and
        computes the read timeout appropriately.

        If self.total is set, the read timeout is dependent on the amount of
        time taken by the connect timeout. If the connection time has not been
        established, a :exc:`~urllib3.exceptions.TimeoutStateError` will be
        raised.

        :return: Value to use for the read timeout.
        :rtype: int, float or None
        :raises urllib3.exceptions.TimeoutStateError: If :meth:`start_connect`
            has not yet been called on this object.
        """
        if (
            self.total is not None
            and self.total is not _DEFAULT_TIMEOUT
            and self._read is not None
            and self._read is not _DEFAULT_TIMEOUT
        ):
            # In case the connect timeout has not yet been established.
            if self._start_connect is None:
                return self._read
            return max(0, min(self.total - self.get_connect_duration(), self._read))
        elif self.total is not None and self.total is not _DEFAULT_TIMEOUT:
            return max(0, self.total - self.get_connect_duration())
        else:
            return self.resolve_default_timeout(self._read)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\stylesheet.py ===
# Copyright (c) 2010-2024 openpyxl

from warnings import warn

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
)
from openpyxl.descriptors.sequence import NestedSequence
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.utils.indexed_list import IndexedList
from openpyxl.xml.constants import ARC_STYLE, SHEET_MAIN_NS
from openpyxl.xml.functions import fromstring

from .builtins import styles
from .colors import ColorList
from .differential import DifferentialStyle
from .table import TableStyleList
from .borders import Border
from .fills import Fill
from .fonts import Font
from .numbers import (
    NumberFormatList,
    BUILTIN_FORMATS,
    BUILTIN_FORMATS_MAX_SIZE,
    BUILTIN_FORMATS_REVERSE,
    is_date_format,
    is_timedelta_format,
    builtin_format_code
)
from .named_styles import (
    _NamedCellStyleList,
    NamedStyleList,
    NamedStyle,
)
from .cell_style import CellStyle, CellStyleList


class Stylesheet(Serialisable):

    tagname = "styleSheet"

    numFmts = Typed(expected_type=NumberFormatList)
    fonts = NestedSequence(expected_type=Font, count=True)
    fills = NestedSequence(expected_type=Fill, count=True)
    borders = NestedSequence(expected_type=Border, count=True)
    cellStyleXfs = Typed(expected_type=CellStyleList)
    cellXfs = Typed(expected_type=CellStyleList)
    cellStyles = Typed(expected_type=_NamedCellStyleList)
    dxfs = NestedSequence(expected_type=DifferentialStyle, count=True)
    tableStyles = Typed(expected_type=TableStyleList, allow_none=True)
    colors = Typed(expected_type=ColorList, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('numFmts', 'fonts', 'fills', 'borders', 'cellStyleXfs',
                    'cellXfs', 'cellStyles', 'dxfs', 'tableStyles', 'colors')

    def __init__(self,
                 numFmts=None,
                 fonts=(),
                 fills=(),
                 borders=(),
                 cellStyleXfs=None,
                 cellXfs=None,
                 cellStyles=None,
                 dxfs=(),
                 tableStyles=None,
                 colors=None,
                 extLst=None,
                ):
        if numFmts is None:
            numFmts = NumberFormatList()
        self.numFmts = numFmts
        self.number_formats = IndexedList()
        self.fonts = fonts
        self.fills = fills
        self.borders = borders
        if cellStyleXfs is None:
            cellStyleXfs = CellStyleList()
        self.cellStyleXfs = cellStyleXfs
        if cellXfs is None:
            cellXfs = CellStyleList()
        self.cellXfs = cellXfs
        if cellStyles is None:
            cellStyles = _NamedCellStyleList()
        self.cellStyles = cellStyles

        self.dxfs = dxfs
        self.tableStyles = tableStyles
        self.colors = colors

        self.cell_styles = self.cellXfs._to_array()
        self.alignments = self.cellXfs.alignments
        self.protections = self.cellXfs.prots
        self._normalise_numbers()
        self.named_styles = self._merge_named_styles()


    @classmethod
    def from_tree(cls, node):
        # strip all attribs
        attrs = dict(node.attrib)
        for k in attrs:
            del node.attrib[k]
        return super().from_tree(node)


    def _merge_named_styles(self):
        """
        Merge named style names "cellStyles" with their associated styles
        "cellStyleXfs"
        """
        style_refs = self.cellStyles.remove_duplicates()
        from_ref = [self._expand_named_style(style_ref) for style_ref in style_refs]

        return NamedStyleList(from_ref)


    def _expand_named_style(self, style_ref):
        """
        Expand a named style reference element to a
        named style object by binding the relevant
        objects from the stylesheet
        """
        xf = self.cellStyleXfs[style_ref.xfId]
        named_style = NamedStyle(
            name=style_ref.name,
            hidden=style_ref.hidden,
            builtinId=style_ref.builtinId,
        )

        named_style.font = self.fonts[xf.fontId]
        named_style.fill = self.fills[xf.fillId]
        named_style.border = self.borders[xf.borderId]
        if xf.numFmtId < BUILTIN_FORMATS_MAX_SIZE:
            formats = BUILTIN_FORMATS
        else:
            formats = self.custom_formats

        if xf.numFmtId in formats:
            named_style.number_format = formats[xf.numFmtId]
        if xf.alignment:
            named_style.alignment = xf.alignment
        if xf.protection:
            named_style.protection = xf.protection

        return named_style


    def _split_named_styles(self, wb):
        """
        Convert NamedStyle into separate CellStyle and Xf objects

        """
        for  style in wb._named_styles:
            self.cellStyles.cellStyle.append(style.as_name())
            self.cellStyleXfs.xf.append(style.as_xf())


    @property
    def custom_formats(self):
        return dict([(n.numFmtId, n.formatCode) for n in self.numFmts.numFmt])


    def _normalise_numbers(self):
        """
        Rebase custom numFmtIds with a floor of 164 when reading stylesheet
        And index datetime formats
        """
        date_formats = set()
        timedelta_formats = set()
        custom = self.custom_formats
        formats = self.number_formats
        for idx, style in enumerate(self.cell_styles):
            if style.numFmtId in custom:
                fmt = custom[style.numFmtId]
                if fmt in BUILTIN_FORMATS_REVERSE: # remove builtins
                    style.numFmtId = BUILTIN_FORMATS_REVERSE[fmt]
                else:
                    style.numFmtId = formats.add(fmt) + BUILTIN_FORMATS_MAX_SIZE
            else:
                fmt = builtin_format_code(style.numFmtId)
            if is_date_format(fmt):
                # Create an index of which styles refer to datetimes
                date_formats.add(idx)
            if is_timedelta_format(fmt):
                # Create an index of which styles refer to timedeltas
                timedelta_formats.add(idx)
        self.date_formats = date_formats
        self.timedelta_formats = timedelta_formats


    def to_tree(self, tagname=None, idx=None, namespace=None):
        tree = super().to_tree(tagname, idx, namespace)
        tree.set("xmlns", SHEET_MAIN_NS)
        return tree


def apply_stylesheet(archive, wb):
    """
    Add styles to workbook if present
    """
    try:
        src = archive.read(ARC_STYLE)
    except KeyError:
        return wb

    node = fromstring(src)
    stylesheet = Stylesheet.from_tree(node)

    if stylesheet.cell_styles:

        wb._borders = IndexedList(stylesheet.borders)
        wb._fonts = IndexedList(stylesheet.fonts)
        wb._fills = IndexedList(stylesheet.fills)
        wb._differential_styles.styles = stylesheet.dxfs
        wb._number_formats = stylesheet.number_formats
        wb._protections = stylesheet.protections
        wb._alignments = stylesheet.alignments
        wb._table_styles = stylesheet.tableStyles

        # need to overwrite openpyxl defaults in case workbook has different ones
        wb._cell_styles = stylesheet.cell_styles
        wb._named_styles = stylesheet.named_styles
        wb._date_formats = stylesheet.date_formats
        wb._timedelta_formats = stylesheet.timedelta_formats

        for ns in wb._named_styles:
            ns.bind(wb)

    else:
        warn("Workbook contains no stylesheet, using openpyxl's defaults")

    if not wb._named_styles:
        normal = styles['Normal']
        wb.add_named_style(normal)
        warn("Workbook contains no default style, apply openpyxl's default")

    if stylesheet.colors is not None:
        wb._colors = stylesheet.colors.index


def write_stylesheet(wb):
    stylesheet = Stylesheet()
    stylesheet.fonts = wb._fonts
    stylesheet.fills = wb._fills
    stylesheet.borders = wb._borders
    stylesheet.dxfs = wb._differential_styles.styles
    stylesheet.colors = ColorList(indexedColors=wb._colors)

    from .numbers import NumberFormat
    fmts = []
    for idx, code in enumerate(wb._number_formats, BUILTIN_FORMATS_MAX_SIZE):
        fmt = NumberFormat(idx, code)
        fmts.append(fmt)

    stylesheet.numFmts.numFmt = fmts

    xfs = []
    for style in wb._cell_styles:
        xf = CellStyle.from_array(style)

        if style.alignmentId:
            xf.alignment = wb._alignments[style.alignmentId]

        if style.protectionId:
            xf.protection = wb._protections[style.protectionId]
        xfs.append(xf)
    stylesheet.cellXfs = CellStyleList(xf=xfs)

    stylesheet._split_named_styles(wb)
    stylesheet.tableStyles = wb._table_styles

    return stylesheet.to_tree()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\fields.py ===
from __future__ import absolute_import

import email.utils
import mimetypes
import re

from .packages import six


def guess_content_type(filename, default="application/octet-stream"):
    """
    Guess the "Content-Type" of a file.

    :param filename:
        The filename to guess the "Content-Type" of using :mod:`mimetypes`.
    :param default:
        If no "Content-Type" can be guessed, default to `default`.
    """
    if filename:
        return mimetypes.guess_type(filename)[0] or default
    return default


def format_header_param_rfc2231(name, value):
    """
    Helper function to format and quote a single header parameter using the
    strategy defined in RFC 2231.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows
    `RFC 2388 Section 4.4 <https://tools.ietf.org/html/rfc2388#section-4.4>`_.

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as ``bytes`` or `str``.
    :ret:
        An RFC-2231-formatted unicode string.
    """
    if isinstance(value, six.binary_type):
        value = value.decode("utf-8")

    if not any(ch in value for ch in '"\\\r\n'):
        result = u'%s="%s"' % (name, value)
        try:
            result.encode("ascii")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            return result

    if six.PY2:  # Python 2:
        value = value.encode("utf-8")

    # encode_rfc2231 accepts an encoded string and returns an ascii-encoded
    # string in Python 2 but accepts and returns unicode strings in Python 3
    value = email.utils.encode_rfc2231(value, "utf-8")
    value = "%s*=%s" % (name, value)

    if six.PY2:  # Python 2:
        value = value.decode("utf-8")

    return value


_HTML5_REPLACEMENTS = {
    u"\u0022": u"%22",
    # Replace "\" with "\\".
    u"\u005C": u"\u005C\u005C",
}

# All control characters from 0x00 to 0x1F *except* 0x1B.
_HTML5_REPLACEMENTS.update(
    {
        six.unichr(cc): u"%{:02X}".format(cc)
        for cc in range(0x00, 0x1F + 1)
        if cc not in (0x1B,)
    }
)


def _replace_multiple(value, needles_and_replacements):
    def replacer(match):
        return needles_and_replacements[match.group(0)]

    pattern = re.compile(
        r"|".join([re.escape(needle) for needle in needles_and_replacements.keys()])
    )

    result = pattern.sub(replacer, value)

    return result


def format_header_param_html5(name, value):
    """
    Helper function to format and quote a single header parameter using the
    HTML5 strategy.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows the `HTML5 Working Draft
    Section 4.10.22.7`_ and matches the behavior of curl and modern browsers.

    .. _HTML5 Working Draft Section 4.10.22.7:
        https://w3c.github.io/html/sec-forms.html#multipart-form-data

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as ``bytes`` or `str``.
    :ret:
        A unicode string, stripped of troublesome characters.
    """
    if isinstance(value, six.binary_type):
        value = value.decode("utf-8")

    value = _replace_multiple(value, _HTML5_REPLACEMENTS)

    return u'%s="%s"' % (name, value)


# For backwards-compatibility.
format_header_param = format_header_param_html5


class RequestField(object):
    """
    A data container for request body parameters.

    :param name:
        The name of this request field. Must be unicode.
    :param data:
        The data/value body.
    :param filename:
        An optional filename of the request field. Must be unicode.
    :param headers:
        An optional dict-like object of headers to initially use for the field.
    :param header_formatter:
        An optional callable that is used to encode and format the headers. By
        default, this is :func:`format_header_param_html5`.
    """

    def __init__(
        self,
        name,
        data,
        filename=None,
        headers=None,
        header_formatter=format_header_param_html5,
    ):
        self._name = name
        self._filename = filename
        self.data = data
        self.headers = {}
        if headers:
            self.headers = dict(headers)
        self.header_formatter = header_formatter

    @classmethod
    def from_tuples(cls, fieldname, value, header_formatter=format_header_param_html5):
        """
        A :class:`~urllib3.fields.RequestField` factory from old-style tuple parameters.

        Supports constructing :class:`~urllib3.fields.RequestField` from
        parameter of key/value strings AND key/filetuple. A filetuple is a
        (filename, data, MIME type) tuple where the MIME type is optional.
        For example::

            'foo': 'bar',
            'fakefile': ('foofile.txt', 'contents of foofile'),
            'realfile': ('barfile.txt', open('realfile').read()),
            'typedfile': ('bazfile.bin', open('bazfile').read(), 'image/jpeg'),
            'nonamefile': 'contents of nonamefile field',

        Field names and filenames must be unicode.
        """
        if isinstance(value, tuple):
            if len(value) == 3:
                filename, data, content_type = value
            else:
                filename, data = value
                content_type = guess_content_type(filename)
        else:
            filename = None
            content_type = None
            data = value

        request_param = cls(
            fieldname, data, filename=filename, header_formatter=header_formatter
        )
        request_param.make_multipart(content_type=content_type)

        return request_param

    def _render_part(self, name, value):
        """
        Overridable helper function to format a single header parameter. By
        default, this calls ``self.header_formatter``.

        :param name:
            The name of the parameter, a string expected to be ASCII only.
        :param value:
            The value of the parameter, provided as a unicode string.
        """

        return self.header_formatter(name, value)

    def _render_parts(self, header_parts):
        """
        Helper function to format and quote a single header.

        Useful for single headers that are composed of multiple items. E.g.,
        'Content-Disposition' fields.

        :param header_parts:
            A sequence of (k, v) tuples or a :class:`dict` of (k, v) to format
            as `k1="v1"; k2="v2"; ...`.
        """
        parts = []
        iterable = header_parts
        if isinstance(header_parts, dict):
            iterable = header_parts.items()

        for name, value in iterable:
            if value is not None:
                parts.append(self._render_part(name, value))

        return u"; ".join(parts)

    def render_headers(self):
        """
        Renders the headers for this request field.
        """
        lines = []

        sort_keys = ["Content-Disposition", "Content-Type", "Content-Location"]
        for sort_key in sort_keys:
            if self.headers.get(sort_key, False):
                lines.append(u"%s: %s" % (sort_key, self.headers[sort_key]))

        for header_name, header_value in self.headers.items():
            if header_name not in sort_keys:
                if header_value:
                    lines.append(u"%s: %s" % (header_name, header_value))

        lines.append(u"\r\n")
        return u"\r\n".join(lines)

    def make_multipart(
        self, content_disposition=None, content_type=None, content_location=None
    ):
        """
        Makes this request field into a multipart request field.

        This method overrides "Content-Disposition", "Content-Type" and
        "Content-Location" headers to the request parameter.

        :param content_type:
            The 'Content-Type' of the request body.
        :param content_location:
            The 'Content-Location' of the request body.

        """
        self.headers["Content-Disposition"] = content_disposition or u"form-data"
        self.headers["Content-Disposition"] += u"; ".join(
            [
                u"",
                self._render_parts(
                    ((u"name", self._name), (u"filename", self._filename))
                ),
            ]
        )
        self.headers["Content-Type"] = content_type
        self.headers["Content-Location"] = content_location

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\__main__.py ===
import colorsys
import io
from time import process_time

from pip._vendor.rich import box
from pip._vendor.rich.color import Color
from pip._vendor.rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from pip._vendor.rich.markdown import Markdown
from pip._vendor.rich.measure import Measurement
from pip._vendor.rich.pretty import Pretty
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style
from pip._vendor.rich.syntax import Syntax
from pip._vendor.rich.table import Table
from pip._vendor.rich.text import Text


class ColorBox:
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        for y in range(0, 5):
            for x in range(options.max_width):
                h = x / options.max_width
                l = 0.1 + ((y / 5) * 0.7)
                r1, g1, b1 = colorsys.hls_to_rgb(h, l, 1.0)
                r2, g2, b2 = colorsys.hls_to_rgb(h, l + 0.7 / 10, 1.0)
                bgcolor = Color.from_rgb(r1 * 255, g1 * 255, b1 * 255)
                color = Color.from_rgb(r2 * 255, g2 * 255, b2 * 255)
                yield Segment("▄", Style(color=color, bgcolor=bgcolor))
            yield Segment.line()

    def __rich_measure__(
        self, console: "Console", options: ConsoleOptions
    ) -> Measurement:
        return Measurement(1, options.max_width)


def make_test_card() -> Table:
    """Get a renderable that demonstrates a number of features."""
    table = Table.grid(padding=1, pad_edge=True)
    table.title = "Rich features"
    table.add_column("Feature", no_wrap=True, justify="center", style="bold red")
    table.add_column("Demonstration")

    color_table = Table(
        box=None,
        expand=False,
        show_header=False,
        show_edge=False,
        pad_edge=False,
    )
    color_table.add_row(
        (
            "✓ [bold green]4-bit color[/]\n"
            "✓ [bold blue]8-bit color[/]\n"
            "✓ [bold magenta]Truecolor (16.7 million)[/]\n"
            "✓ [bold yellow]Dumb terminals[/]\n"
            "✓ [bold cyan]Automatic color conversion"
        ),
        ColorBox(),
    )

    table.add_row("Colors", color_table)

    table.add_row(
        "Styles",
        "All ansi styles: [bold]bold[/], [dim]dim[/], [italic]italic[/italic], [underline]underline[/], [strike]strikethrough[/], [reverse]reverse[/], and even [blink]blink[/].",
    )

    lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in metus sed sapien ultricies pretium a at justo. Maecenas luctus velit et auctor maximus."
    lorem_table = Table.grid(padding=1, collapse_padding=True)
    lorem_table.pad_edge = False
    lorem_table.add_row(
        Text(lorem, justify="left", style="green"),
        Text(lorem, justify="center", style="yellow"),
        Text(lorem, justify="right", style="blue"),
        Text(lorem, justify="full", style="red"),
    )
    table.add_row(
        "Text",
        Group(
            Text.from_markup(
                """Word wrap text. Justify [green]left[/], [yellow]center[/], [blue]right[/] or [red]full[/].\n"""
            ),
            lorem_table,
        ),
    )

    def comparison(renderable1: RenderableType, renderable2: RenderableType) -> Table:
        table = Table(show_header=False, pad_edge=False, box=None, expand=True)
        table.add_column("1", ratio=1)
        table.add_column("2", ratio=1)
        table.add_row(renderable1, renderable2)
        return table

    table.add_row(
        "Asian\nlanguage\nsupport",
        ":flag_for_china:  该库支持中文，日文和韩文文本！\n:flag_for_japan:  ライブラリは中国語、日本語、韓国語のテキストをサポートしています\n:flag_for_south_korea:  이 라이브러리는 중국어, 일본어 및 한국어 텍스트를 지원합니다",
    )

    markup_example = (
        "[bold magenta]Rich[/] supports a simple [i]bbcode[/i]-like [b]markup[/b] for [yellow]color[/], [underline]style[/], and emoji! "
        ":+1: :apple: :ant: :bear: :baguette_bread: :bus: "
    )
    table.add_row("Markup", markup_example)

    example_table = Table(
        show_edge=False,
        show_header=True,
        expand=False,
        row_styles=["none", "dim"],
        box=box.SIMPLE,
    )
    example_table.add_column("[green]Date", style="green", no_wrap=True)
    example_table.add_column("[blue]Title", style="blue")
    example_table.add_column(
        "[cyan]Production Budget",
        style="cyan",
        justify="right",
        no_wrap=True,
    )
    example_table.add_column(
        "[magenta]Box Office",
        style="magenta",
        justify="right",
        no_wrap=True,
    )
    example_table.add_row(
        "Dec 20, 2019",
        "Star Wars: The Rise of Skywalker",
        "$275,000,000",
        "$375,126,118",
    )
    example_table.add_row(
        "May 25, 2018",
        "[b]Solo[/]: A Star Wars Story",
        "$275,000,000",
        "$393,151,347",
    )
    example_table.add_row(
        "Dec 15, 2017",
        "Star Wars Ep. VIII: The Last Jedi",
        "$262,000,000",
        "[bold]$1,332,539,889[/bold]",
    )
    example_table.add_row(
        "May 19, 1999",
        "Star Wars Ep. [b]I[/b]: [i]The phantom Menace",
        "$115,000,000",
        "$1,027,044,677",
    )

    table.add_row("Tables", example_table)

    code = '''\
def iter_last(values: Iterable[T]) -> Iterable[Tuple[bool, T]]:
    """Iterate and generate a tuple with a flag for last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    for value in iter_values:
        yield False, previous_value
        previous_value = value
    yield True, previous_value'''

    pretty_data = {
        "foo": [
            3.1427,
            (
                "Paul Atreides",
                "Vladimir Harkonnen",
                "Thufir Hawat",
            ),
        ],
        "atomic": (False, True, None),
    }
    table.add_row(
        "Syntax\nhighlighting\n&\npretty\nprinting",
        comparison(
            Syntax(code, "python3", line_numbers=True, indent_guides=True),
            Pretty(pretty_data, indent_guides=True),
        ),
    )

    markdown_example = """\
# Markdown

Supports much of the *markdown* __syntax__!

- Headers
- Basic formatting: **bold**, *italic*, `code`
- Block quotes
- Lists, and more...
    """
    table.add_row(
        "Markdown", comparison("[cyan]" + markdown_example, Markdown(markdown_example))
    )

    table.add_row(
        "+more!",
        """Progress bars, columns, styled logging handler, tracebacks, etc...""",
    )
    return table


if __name__ == "__main__":  # pragma: no cover
    console = Console(
        file=io.StringIO(),
        force_terminal=True,
    )
    test_card = make_test_card()

    # Print once to warm cache
    start = process_time()
    console.print(test_card)
    pre_cache_taken = round((process_time() - start) * 1000.0, 1)

    console.file = io.StringIO()

    start = process_time()
    console.print(test_card)
    taken = round((process_time() - start) * 1000.0, 1)

    c = Console(record=True)
    c.print(test_card)

    print(f"rendered in {pre_cache_taken}ms (cold cache)")
    print(f"rendered in {taken}ms (warm cache)")

    from pip._vendor.rich.panel import Panel

    console = Console()

    sponsor_message = Table.grid(padding=1)
    sponsor_message.add_column(style="green", justify="right")
    sponsor_message.add_column(no_wrap=True)

    sponsor_message.add_row(
        "Textualize",
        "[u blue link=https://github.com/textualize]https://github.com/textualize",
    )
    sponsor_message.add_row(
        "Twitter",
        "[u blue link=https://twitter.com/willmcgugan]https://twitter.com/willmcgugan",
    )

    intro_message = Text.from_markup(
        """\
We hope you enjoy using Rich!

Rich is maintained with [red]:heart:[/] by [link=https://www.textualize.io]Textualize.io[/]

- Will McGugan"""
    )

    message = Table.grid(padding=2)
    message.add_column()
    message.add_column(no_wrap=True)
    message.add_row(intro_message, sponsor_message)

    console.print(
        Panel.fit(
            message,
            box=box.ROUNDED,
            padding=(1, 2),
            title="[b red]Thanks for trying out Rich!",
            border_style="bright_blue",
        ),
        justify="center",
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\handlers\calculus.py ===
"""
This module contains query handlers responsible for calculus queries:
infinitesimal, finite, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Expr, Add, Mul, Pow, Symbol
from sympy.core.numbers import (NegativeInfinity, GoldenRatio,
    Infinity, Exp1, ComplexInfinity, ImaginaryUnit, NaN, Number, Pi, E,
    TribonacciConstant)
from sympy.functions import cos, exp, log, sign, sin
from sympy.logic.boolalg import conjuncts

from ..predicates.calculus import (FinitePredicate, InfinitePredicate,
    PositiveInfinitePredicate, NegativeInfinitePredicate)


# FinitePredicate


@FinitePredicate.register(Symbol)
def _(expr, assumptions):
    """
    Handles Symbol.
    """
    if expr.is_finite is not None:
        return expr.is_finite
    if Q.finite(expr) in conjuncts(assumptions):
        return True
    return None

@FinitePredicate.register(Add)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +-------+-----+-----------+-----------+
    |       |     |           |           |
    |       |  B  |     U     |     ?     |
    |       |     |           |           |
    +-------+-----+---+---+---+---+---+---+
    |       |     |   |   |   |   |   |   |
    |       |     |'+'|'-'|'x'|'+'|'-'|'x'|
    |       |     |   |   |   |   |   |   |
    +-------+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   B   |  B  |     U     |     ?     |
    |       |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    |   |'+'|     | U | ? | ? | U | ? | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |   |   |   |   |   |   |
    | U |'-'|     | ? | U | ? | ? | U | ? |
    |   |   |     |   |   |   |   |   |   |
    |   +---+-----+---+---+---+---+---+---+
    |   |   |     |           |           |
    |   |'x'|     |     ?     |     ?     |
    |   |   |     |           |           |
    +---+---+-----+---+---+---+---+---+---+
    |       |     |           |           |
    |   ?   |     |           |     ?     |
    |       |     |           |           |
    +-------+-----+-----------+---+---+---+

        * 'B' = Bounded

        * 'U' = Unbounded

        * '?' = unknown boundedness

        * '+' = positive sign

        * '-' = negative sign

        * 'x' = sign unknown

        * All Bounded -> True

        * 1 Unbounded and the rest Bounded -> False

        * >1 Unbounded, all with same known sign -> False

        * Any Unknown and unknown sign -> None

        * Else -> None

    When the signs are not the same you can have an undefined
    result as in oo - oo, hence 'bounded' is also undefined.
    """
    sign = -1  # sign of unknown or infinite
    result = True
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            continue
        s = ask(Q.extended_positive(arg), assumptions)
        # if there has been more than one sign or if the sign of this arg
        # is None and Bounded is None or there was already
        # an unknown sign, return None
        if sign != -1 and s != sign or \
                s is None and None in (_bounded, sign):
            return None
        else:
            sign = s
        # once False, do not change
        if result is not False:
            result = _bounded
    return result

@FinitePredicate.register(Mul)
def _(expr, assumptions):
    """
    Return True if expr is bounded, False if not and None if unknown.

    Truth Table:

    +---+---+---+--------+
    |   |   |   |        |
    |   | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    |   |   |   | s | /s |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | B | B | U |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+
    |   |   |   |   |    |
    | U |   | U | U | ?  |
    |   |   |   |   |    |
    +---+---+---+---+----+
    |   |   |   |        |
    | ? |   |   |   ?    |
    |   |   |   |        |
    +---+---+---+---+----+

        * B = Bounded

        * U = Unbounded

        * ? = unknown boundedness

        * s = signed (hence nonzero)

        * /s = not signed
    """
    result = True
    possible_zero = False
    for arg in expr.args:
        _bounded = ask(Q.finite(arg), assumptions)
        if _bounded:
            if ask(Q.zero(arg), assumptions) is not False:
                if result is False:
                    return None
                possible_zero = True
        elif _bounded is None:
            if result is None:
                return None
            if ask(Q.extended_nonzero(arg), assumptions) is None:
                return None
            if result is not False:
                result = None
        else:
            if possible_zero:
                return None
            result = False
    return result

@FinitePredicate.register(Pow)
def _(expr, assumptions):
    """
    * Unbounded ** NonZero -> Unbounded

    * Bounded ** Bounded -> Bounded

    * Abs()<=1 ** Positive -> Bounded

    * Abs()>=1 ** Negative -> Bounded

    * Otherwise unknown
    """
    if expr.base == E:
        return ask(Q.finite(expr.exp), assumptions)

    base_bounded = ask(Q.finite(expr.base), assumptions)
    exp_bounded = ask(Q.finite(expr.exp), assumptions)
    if base_bounded is None and exp_bounded is None:  # Common Case
        return None
    if base_bounded is False and ask(Q.extended_nonzero(expr.exp), assumptions):
        return False
    if base_bounded and exp_bounded:
        is_base_zero = ask(Q.zero(expr.base),assumptions)
        is_exp_negative = ask(Q.negative(expr.exp),assumptions)
        if is_base_zero is True and is_exp_negative is True:
            return False
        if is_base_zero is not False and is_exp_negative is not False:
            return None
        return True
    if (abs(expr.base) <= 1) == True and ask(Q.extended_positive(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and ask(Q.extended_negative(expr.exp), assumptions):
        return True
    if (abs(expr.base) >= 1) == True and exp_bounded is False:
        return False
    return None

@FinitePredicate.register(exp)
def _(expr, assumptions):
    return ask(Q.finite(expr.exp), assumptions)

@FinitePredicate.register(log)
def _(expr, assumptions):
    # After complex -> finite fact is registered to new assumption system,
    # querying Q.infinite may be removed.
    if ask(Q.infinite(expr.args[0]), assumptions):
        return False
    return ask(~Q.zero(expr.args[0]), assumptions)

@FinitePredicate.register_many(cos, sin, Number, Pi, Exp1, GoldenRatio,
    TribonacciConstant, ImaginaryUnit, sign)
def _(expr, assumptions):
    return True

@FinitePredicate.register_many(ComplexInfinity, Infinity, NegativeInfinity)
def _(expr, assumptions):
    return False

@FinitePredicate.register(NaN)
def _(expr, assumptions):
    return None


# InfinitePredicate


@InfinitePredicate.register(Expr)
def _(expr, assumptions):
    is_finite = Q.finite(expr)._eval_ask(assumptions)
    if is_finite is None:
        return None
    return not is_finite


# PositiveInfinitePredicate


@PositiveInfinitePredicate.register(Infinity)
def _(expr, assumptions):
    return True


@PositiveInfinitePredicate.register_many(NegativeInfinity, ComplexInfinity)
def _(expr, assumptions):
    return False


# NegativeInfinitePredicate


@NegativeInfinitePredicate.register(NegativeInfinity)
def _(expr, assumptions):
    return True


@NegativeInfinitePredicate.register_many(Infinity, ComplexInfinity)
def _(expr, assumptions):
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\deutils.py ===
"""Utility functions for classifying and solving
ordinary and partial differential equations.

Contains
========
_preprocess
ode_order
_desolve

"""
from sympy.core import Pow
from sympy.core.function import Derivative, AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Wild

def _preprocess(expr, func=None, hint='_Integral'):
    """Prepare expr for solving by making sure that differentiation
    is done so that only func remains in unevaluated derivatives and
    (if hint does not end with _Integral) that doit is applied to all
    other derivatives. If hint is None, do not do any differentiation.
    (Currently this may cause some simple differential equations to
    fail.)

    In case func is None, an attempt will be made to autodetect the
    function to be solved for.

    >>> from sympy.solvers.deutils import _preprocess
    >>> from sympy import Derivative, Function
    >>> from sympy.abc import x, y, z
    >>> f, g = map(Function, 'fg')

    If f(x)**p == 0 and p>0 then we can solve for f(x)=0
    >>> _preprocess((f(x).diff(x)-4)**5, f(x))
    (Derivative(f(x), x) - 4, f(x))

    Apply doit to derivatives that contain more than the function
    of interest:

    >>> _preprocess(Derivative(f(x) + x, x))
    (Derivative(f(x), x) + 1, f(x))

    Do others if the differentiation variable(s) intersect with those
    of the function of interest or contain the function of interest:

    >>> _preprocess(Derivative(g(x), y, z), f(y))
    (0, f(y))
    >>> _preprocess(Derivative(f(y), z), f(y))
    (0, f(y))

    Do others if the hint does not end in '_Integral' (the default
    assumes that it does):

    >>> _preprocess(Derivative(g(x), y), f(x))
    (Derivative(g(x), y), f(x))
    >>> _preprocess(Derivative(f(x), y), f(x), hint='')
    (0, f(x))

    Do not do any derivatives if hint is None:

    >>> eq = Derivative(f(x) + 1, x) + Derivative(f(x), y)
    >>> _preprocess(eq, f(x), hint=None)
    (Derivative(f(x) + 1, x) + Derivative(f(x), y), f(x))

    If it's not clear what the function of interest is, it must be given:

    >>> eq = Derivative(f(x) + g(x), x)
    >>> _preprocess(eq, g(x))
    (Derivative(f(x), x) + Derivative(g(x), x), g(x))
    >>> try: _preprocess(eq)
    ... except ValueError: print("A ValueError was raised.")
    A ValueError was raised.

    """
    if isinstance(expr, Pow):
        # if f(x)**p=0 then f(x)=0 (p>0)
        if (expr.exp).is_positive:
            expr = expr.base
    derivs = expr.atoms(Derivative)
    if not func:
        funcs = set().union(*[d.atoms(AppliedUndef) for d in derivs])
        if len(funcs) != 1:
            raise ValueError('The function cannot be '
                'automatically detected for %s.' % expr)
        func = funcs.pop()
    fvars = set(func.args)
    if hint is None:
        return expr, func
    reps = [(d, d.doit()) for d in derivs if not hint.endswith('_Integral') or
            d.has(func) or set(d.variables) & fvars]
    eq = expr.subs(reps)
    return eq, func


def ode_order(expr, func):
    """
    Returns the order of a given differential
    equation with respect to func.

    This function is implemented recursively.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.deutils import ode_order
    >>> from sympy.abc import x
    >>> f, g = map(Function, ['f', 'g'])
    >>> ode_order(f(x).diff(x, 2) + f(x).diff(x)**2 +
    ... f(x).diff(x), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), g(x))
    3

    """
    a = Wild('a', exclude=[func])
    if expr.match(a):
        return 0

    if isinstance(expr, Derivative):
        if expr.args[0] == func:
            return len(expr.variables)
        else:
            args = expr.args[0].args
            rv = len(expr.variables)
            if args:
                rv += max(ode_order(_, func) for _ in args)
            return rv
    else:
        return max(ode_order(_, func) for _ in expr.args) if expr.args else 0


def _desolve(eq, func=None, hint="default", ics=None, simplify=True, *, prep=True, **kwargs):
    """This is a helper function to dsolve and pdsolve in the ode
    and pde modules.

    If the hint provided to the function is "default", then a dict with
    the following keys are returned

    'func'    - It provides the function for which the differential equation
                has to be solved. This is useful when the expression has
                more than one function in it.

    'default' - The default key as returned by classifier functions in ode
                and pde.py

    'hint'    - The hint given by the user for which the differential equation
                is to be solved. If the hint given by the user is 'default',
                then the value of 'hint' and 'default' is the same.

    'order'   - The order of the function as returned by ode_order

    'match'   - It returns the match as given by the classifier functions, for
                the default hint.

    If the hint provided to the function is not "default" and is not in
    ('all', 'all_Integral', 'best'), then a dict with the above mentioned keys
    is returned along with the keys which are returned when dict in
    classify_ode or classify_pde is set True

    If the hint given is in ('all', 'all_Integral', 'best'), then this function
    returns a nested dict, with the keys, being the set of classified hints
    returned by classifier functions, and the values being the dict of form
    as mentioned above.

    Key 'eq' is a common key to all the above mentioned hints which returns an
    expression if eq given by user is an Equality.

    See Also
    ========
    classify_ode(ode.py)
    classify_pde(pde.py)
    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # preprocess the equation and find func if not given
    if prep or func is None:
        eq, func = _preprocess(eq, func)
        prep = False

    # type is an argument passed by the solve functions in ode and pde.py
    # that identifies whether the function caller is an ordinary
    # or partial differential equation. Accordingly corresponding
    # changes are made in the function.
    type = kwargs.get('type', None)
    xi = kwargs.get('xi')
    eta = kwargs.get('eta')
    x0 = kwargs.get('x0', 0)
    terms = kwargs.get('n')

    if type == 'ode':
        from sympy.solvers.ode import classify_ode, allhints
        classifier = classify_ode
        string = 'ODE '
        dummy = ''

    elif type == 'pde':
        from sympy.solvers.pde import classify_pde, allhints
        classifier = classify_pde
        string = 'PDE '
        dummy = 'p'

    # Magic that should only be used internally.  Prevents classify_ode from
    # being called more than it needs to be by passing its results through
    # recursive calls.
    if kwargs.get('classify', True):
        hints = classifier(eq, func, dict=True, ics=ics, xi=xi, eta=eta,
        n=terms, x0=x0, hint=hint, prep=prep)

    else:
        # Here is what all this means:
        #
        # hint:    The hint method given to _desolve() by the user.
        # hints:   The dictionary of hints that match the DE, along with other
        #          information (including the internal pass-through magic).
        # default: The default hint to return, the first hint from allhints
        #          that matches the hint; obtained from classify_ode().
        # match:   Dictionary containing the match dictionary for each hint
        #          (the parts of the DE for solving).  When going through the
        #          hints in "all", this holds the match string for the current
        #          hint.
        # order:   The order of the DE, as determined by ode_order().
        hints = kwargs.get('hint',
                           {'default': hint,
                            hint: kwargs['match'],
                            'order': kwargs['order']})
    if not hints['default']:
        # classify_ode will set hints['default'] to None if no hints match.
        if hint not in allhints and hint != 'default':
            raise ValueError("Hint not recognized: " + hint)
        elif hint not in hints['ordered_hints'] and hint != 'default':
            raise ValueError(string + str(eq) + " does not match hint " + hint)
        # If dsolve can't solve the purely algebraic equation then dsolve will raise
        # ValueError
        elif hints['order'] == 0:
            raise ValueError(
                str(eq) + " is not a solvable differential equation in " + str(func))
        else:
            raise NotImplementedError(dummy + "solve" + ": Cannot solve " + str(eq))
    if hint == 'default':
        return _desolve(eq, func, ics=ics, hint=hints['default'], simplify=simplify,
                      prep=prep, x0=x0, classify=False, order=hints['order'],
                      match=hints[hints['default']], xi=xi, eta=eta, n=terms, type=type)
    elif hint in ('all', 'all_Integral', 'best'):
        retdict = {}
        gethints = set(hints) - {'order', 'default', 'ordered_hints'}
        if hint == 'all_Integral':
            for i in hints:
                if i.endswith('_Integral'):
                    gethints.remove(i.removesuffix('_Integral'))
            # special cases
            for k in ["1st_homogeneous_coeff_best", "1st_power_series",
                "lie_group", "2nd_power_series_ordinary", "2nd_power_series_regular"]:
                if k in gethints:
                    gethints.remove(k)
        for i in gethints:
            sol = _desolve(eq, func, ics=ics, hint=i, x0=x0, simplify=simplify, prep=prep,
                classify=False, n=terms, order=hints['order'], match=hints[i], type=type)
            retdict[i] = sol
        retdict['all'] = True
        retdict['eq'] = eq
        return retdict
    elif hint not in allhints:  # and hint not in ('default', 'ordered_hints'):
        raise ValueError("Hint not recognized: " + hint)
    elif hint not in hints:
        raise ValueError(string + str(eq) + " does not match hint " + hint)
    else:
        # Key added to identify the hint needed to solve the equation
        hints['hint'] = hint
    hints.update({'func': func, 'eq': eq})
    return hints

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\cache_control.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t
from inspect import cleandoc

from .mixins import ImmutableDictMixin
from .structures import CallbackDict


def cache_control_property(
    key: str, empty: t.Any, type: type[t.Any] | None, *, doc: str | None = None
) -> t.Any:
    """Return a new property object for a cache header. Useful if you
    want to add support for a cache extension in a subclass.

    :param key: The attribute name present in the parsed cache-control header dict.
    :param empty: The value to use if the key is present without a value.
    :param type: The type to convert the string value to instead of a string. If
        conversion raises a ``ValueError``, the returned value is ``None``.
    :param doc: The docstring for the property. If not given, it is generated
        based on the other params.

    .. versionchanged:: 3.1
        Added the ``doc`` param.

    .. versionchanged:: 2.0
        Renamed from ``cache_property``.
    """
    if doc is None:
        parts = [f"The ``{key}`` attribute."]

        if type is bool:
            parts.append("A ``bool``, either present or not.")
        else:
            if type is None:
                parts.append("A ``str``,")
            else:
                parts.append(f"A ``{type.__name__}``,")

            if empty is not None:
                parts.append(f"``{empty!r}`` if present with no value,")

            parts.append("or ``None`` if not present.")

        doc = " ".join(parts)

    return property(
        lambda x: x._get_cache_value(key, empty, type),
        lambda x, v: x._set_cache_value(key, v, type),
        lambda x: x._del_cache_value(key),
        doc=cleandoc(doc),
    )


class _CacheControl(CallbackDict[str, t.Optional[str]]):
    """Subclass of a dict that stores values for a Cache-Control header.  It
    has accessors for all the cache-control directives specified in RFC 2616.
    The class does not differentiate between request and response directives.

    Because the cache-control directives in the HTTP header use dashes the
    python descriptors use underscores for that.

    To get a header of the :class:`CacheControl` object again you can convert
    the object into a string or call the :meth:`to_header` method.  If you plan
    to subclass it and add your own items have a look at the sourcecode for
    that class.

    .. versionchanged:: 3.1
        Dict values are always ``str | None``. Setting properties will
        convert the value to a string. Setting a non-bool property to
        ``False`` is equivalent to setting it to ``None``. Getting typed
        properties will return ``None`` if conversion raises
        ``ValueError``, rather than the string.

    .. versionchanged:: 2.1
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionchanged:: 0.4
       Setting ``no_cache`` or ``private`` to ``True`` will set the
       implicit value ``"*"``.
    """

    no_store: bool = cache_control_property("no-store", None, bool)
    max_age: int | None = cache_control_property("max-age", None, int)
    no_transform: bool = cache_control_property("no-transform", None, bool)
    stale_if_error: int | None = cache_control_property("stale-if-error", None, int)

    def __init__(
        self,
        values: cabc.Mapping[str, t.Any] | cabc.Iterable[tuple[str, t.Any]] | None = (),
        on_update: cabc.Callable[[_CacheControl], None] | None = None,
    ):
        super().__init__(values, on_update)
        self.provided = values is not None

    def _get_cache_value(
        self, key: str, empty: t.Any, type: type[t.Any] | None
    ) -> t.Any:
        """Used internally by the accessor properties."""
        if type is bool:
            return key in self

        if key not in self:
            return None

        if (value := self[key]) is None:
            return empty

        if type is not None:
            try:
                value = type(value)
            except ValueError:
                return None

        return value

    def _set_cache_value(
        self, key: str, value: t.Any, type: type[t.Any] | None
    ) -> None:
        """Used internally by the accessor properties."""
        if type is bool:
            if value:
                self[key] = None
            else:
                self.pop(key, None)
        elif value is None or value is False:
            self.pop(key, None)
        elif value is True:
            self[key] = None
        else:
            if type is not None:
                value = type(value)

            self[key] = str(value)

    def _del_cache_value(self, key: str) -> None:
        """Used internally by the accessor properties."""
        if key in self:
            del self[key]

    def to_header(self) -> str:
        """Convert the stored values into a cache control header."""
        return http.dump_header(self)

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        kv_str = " ".join(f"{k}={v!r}" for k, v in sorted(self.items()))
        return f"<{type(self).__name__} {kv_str}>"

    cache_property = staticmethod(cache_control_property)


class RequestCacheControl(ImmutableDictMixin[str, t.Optional[str]], _CacheControl):  # type: ignore[misc]
    """A cache control for requests.  This is immutable and gives access
    to all the request-relevant cache control headers.

    To get a header of the :class:`RequestCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionchanged:: 3.1
        Dict values are always ``str | None``. Setting properties will
        convert the value to a string. Setting a non-bool property to
        ``False`` is equivalent to setting it to ``None``. Getting typed
        properties will return ``None`` if conversion raises
        ``ValueError``, rather than the string.

    .. versionchanged:: 3.1
       ``max_age`` is ``None`` if present without a value, rather
       than ``-1``.

    .. versionchanged:: 3.1
        ``no_cache`` is a boolean, it is ``True`` instead of ``"*"``
        when present.

    .. versionchanged:: 3.1
        ``max_stale`` is ``True`` if present without a value, rather
        than ``"*"``.

    .. versionchanged:: 3.1
       ``no_transform`` is a boolean. Previously it was mistakenly
       always ``None``.

    .. versionchanged:: 3.1
       ``min_fresh`` is ``None`` if present without a value, rather
       than ``"*"``.

    .. versionchanged:: 2.1
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionadded:: 0.5
        Response-only properties are not present on this request class.
    """

    no_cache: bool = cache_control_property("no-cache", None, bool)
    max_stale: int | t.Literal[True] | None = cache_control_property(
        "max-stale",
        True,
        int,
    )
    min_fresh: int | None = cache_control_property("min-fresh", None, int)
    only_if_cached: bool = cache_control_property("only-if-cached", None, bool)


class ResponseCacheControl(_CacheControl):
    """A cache control for responses.  Unlike :class:`RequestCacheControl`
    this is mutable and gives access to response-relevant cache control
    headers.

    To get a header of the :class:`ResponseCacheControl` object again you can
    convert the object into a string or call the :meth:`to_header` method.  If
    you plan to subclass it and add your own items have a look at the sourcecode
    for that class.

    .. versionchanged:: 3.1
        Dict values are always ``str | None``. Setting properties will
        convert the value to a string. Setting a non-bool property to
        ``False`` is equivalent to setting it to ``None``. Getting typed
        properties will return ``None`` if conversion raises
        ``ValueError``, rather than the string.

    .. versionchanged:: 3.1
        ``no_cache`` is ``True`` if present without a value, rather than
        ``"*"``.

    .. versionchanged:: 3.1
        ``private`` is ``True`` if present without a value, rather than
        ``"*"``.

    .. versionchanged:: 3.1
       ``no_transform`` is a boolean. Previously it was mistakenly
       always ``None``.

    .. versionchanged:: 3.1
        Added the ``must_understand``, ``stale_while_revalidate``, and
        ``stale_if_error`` properties.

    .. versionchanged:: 2.1.1
        ``s_maxage`` converts the value to an int.

    .. versionchanged:: 2.1
        Setting int properties such as ``max_age`` will convert the
        value to an int.

    .. versionadded:: 0.5
       Request-only properties are not present on this response class.
    """

    no_cache: str | t.Literal[True] | None = cache_control_property(
        "no-cache", True, None
    )
    public: bool = cache_control_property("public", None, bool)
    private: str | t.Literal[True] | None = cache_control_property(
        "private", True, None
    )
    must_revalidate: bool = cache_control_property("must-revalidate", None, bool)
    proxy_revalidate: bool = cache_control_property("proxy-revalidate", None, bool)
    s_maxage: int | None = cache_control_property("s-maxage", None, int)
    immutable: bool = cache_control_property("immutable", None, bool)
    must_understand: bool = cache_control_property("must-understand", None, bool)
    stale_while_revalidate: int | None = cache_control_property(
        "stale-while-revalidate", None, int
    )


# circular dependencies
from .. import http

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\service_reflection.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains metaclasses used to create protocol service and service stub
classes from ServiceDescriptor objects at runtime.

The GeneratedServiceType and GeneratedServiceStubType metaclasses are used to
inject all useful functionality into the classes output by the protocol
compiler at compile-time.
"""

__author__ = 'petar@google.com (Petar Petrov)'


class GeneratedServiceType(type):

  """Metaclass for service classes created at runtime from ServiceDescriptors.

  Implementations for all methods described in the Service class are added here
  by this class. We also create properties to allow getting/setting all fields
  in the protocol message.

  The protocol compiler currently uses this metaclass to create protocol service
  classes at runtime. Clients can also manually create their own classes at
  runtime, as in this example::

    mydescriptor = ServiceDescriptor(.....)
    class MyProtoService(service.Service):
      __metaclass__ = GeneratedServiceType
      DESCRIPTOR = mydescriptor
    myservice_instance = MyProtoService()
    # ...
  """

  _DESCRIPTOR_KEY = 'DESCRIPTOR'

  def __init__(cls, name, bases, dictionary):
    """Creates a message service class.

    Args:
      name: Name of the class (ignored, but required by the metaclass
        protocol).
      bases: Base classes of the class being constructed.
      dictionary: The class dictionary of the class being constructed.
        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object
        describing this protocol service type.
    """
    # Don't do anything if this class doesn't have a descriptor. This happens
    # when a service class is subclassed.
    if GeneratedServiceType._DESCRIPTOR_KEY not in dictionary:
      return

    descriptor = dictionary[GeneratedServiceType._DESCRIPTOR_KEY]
    service_builder = _ServiceBuilder(descriptor)
    service_builder.BuildService(cls)
    cls.DESCRIPTOR = descriptor


class GeneratedServiceStubType(GeneratedServiceType):

  """Metaclass for service stubs created at runtime from ServiceDescriptors.

  This class has similar responsibilities as GeneratedServiceType, except that
  it creates the service stub classes.
  """

  _DESCRIPTOR_KEY = 'DESCRIPTOR'

  def __init__(cls, name, bases, dictionary):
    """Creates a message service stub class.

    Args:
      name: Name of the class (ignored, here).
      bases: Base classes of the class being constructed.
      dictionary: The class dictionary of the class being constructed.
        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object
        describing this protocol service type.
    """
    super(GeneratedServiceStubType, cls).__init__(name, bases, dictionary)
    # Don't do anything if this class doesn't have a descriptor. This happens
    # when a service stub is subclassed.
    if GeneratedServiceStubType._DESCRIPTOR_KEY not in dictionary:
      return

    descriptor = dictionary[GeneratedServiceStubType._DESCRIPTOR_KEY]
    service_stub_builder = _ServiceStubBuilder(descriptor)
    service_stub_builder.BuildServiceStub(cls)


class _ServiceBuilder(object):

  """This class constructs a protocol service class using a service descriptor.

  Given a service descriptor, this class constructs a class that represents
  the specified service descriptor. One service builder instance constructs
  exactly one service class. That means all instances of that class share the
  same builder.
  """

  def __init__(self, service_descriptor):
    """Initializes an instance of the service class builder.

    Args:
      service_descriptor: ServiceDescriptor to use when constructing the
        service class.
    """
    self.descriptor = service_descriptor

  def BuildService(builder, cls):
    """Constructs the service class.

    Args:
      cls: The class that will be constructed.
    """

    # CallMethod needs to operate with an instance of the Service class. This
    # internal wrapper function exists only to be able to pass the service
    # instance to the method that does the real CallMethod work.
    # Making sure to use exact argument names from the abstract interface in
    # service.py to match the type signature
    def _WrapCallMethod(self, method_descriptor, rpc_controller, request, done):
      return builder._CallMethod(self, method_descriptor, rpc_controller,
                                 request, done)

    def _WrapGetRequestClass(self, method_descriptor):
      return builder._GetRequestClass(method_descriptor)

    def _WrapGetResponseClass(self, method_descriptor):
      return builder._GetResponseClass(method_descriptor)

    builder.cls = cls
    cls.CallMethod = _WrapCallMethod
    cls.GetDescriptor = staticmethod(lambda: builder.descriptor)
    cls.GetDescriptor.__doc__ = 'Returns the service descriptor.'
    cls.GetRequestClass = _WrapGetRequestClass
    cls.GetResponseClass = _WrapGetResponseClass
    for method in builder.descriptor.methods:
      setattr(cls, method.name, builder._GenerateNonImplementedMethod(method))

  def _CallMethod(self, srvc, method_descriptor,
                  rpc_controller, request, callback):
    """Calls the method described by a given method descriptor.

    Args:
      srvc: Instance of the service for which this method is called.
      method_descriptor: Descriptor that represent the method to call.
      rpc_controller: RPC controller to use for this method's execution.
      request: Request protocol message.
      callback: A callback to invoke after the method has completed.
    """
    if method_descriptor.containing_service != self.descriptor:
      raise RuntimeError(
          'CallMethod() given method descriptor for wrong service type.')
    method = getattr(srvc, method_descriptor.name)
    return method(rpc_controller, request, callback)

  def _GetRequestClass(self, method_descriptor):
    """Returns the class of the request protocol message.

    Args:
      method_descriptor: Descriptor of the method for which to return the
        request protocol message class.

    Returns:
      A class that represents the input protocol message of the specified
      method.
    """
    if method_descriptor.containing_service != self.descriptor:
      raise RuntimeError(
          'GetRequestClass() given method descriptor for wrong service type.')
    return method_descriptor.input_type._concrete_class

  def _GetResponseClass(self, method_descriptor):
    """Returns the class of the response protocol message.

    Args:
      method_descriptor: Descriptor of the method for which to return the
        response protocol message class.

    Returns:
      A class that represents the output protocol message of the specified
      method.
    """
    if method_descriptor.containing_service != self.descriptor:
      raise RuntimeError(
          'GetResponseClass() given method descriptor for wrong service type.')
    return method_descriptor.output_type._concrete_class

  def _GenerateNonImplementedMethod(self, method):
    """Generates and returns a method that can be set for a service methods.

    Args:
      method: Descriptor of the service method for which a method is to be
        generated.

    Returns:
      A method that can be added to the service class.
    """
    return lambda inst, rpc_controller, request, callback: (
        self._NonImplementedMethod(method.name, rpc_controller, callback))

  def _NonImplementedMethod(self, method_name, rpc_controller, callback):
    """The body of all methods in the generated service class.

    Args:
      method_name: Name of the method being executed.
      rpc_controller: RPC controller used to execute this method.
      callback: A callback which will be invoked when the method finishes.
    """
    rpc_controller.SetFailed('Method %s not implemented.' % method_name)
    callback(None)


class _ServiceStubBuilder(object):

  """Constructs a protocol service stub class using a service descriptor.

  Given a service descriptor, this class constructs a suitable stub class.
  A stub is just a type-safe wrapper around an RpcChannel which emulates a
  local implementation of the service.

  One service stub builder instance constructs exactly one class. It means all
  instances of that class share the same service stub builder.
  """

  def __init__(self, service_descriptor):
    """Initializes an instance of the service stub class builder.

    Args:
      service_descriptor: ServiceDescriptor to use when constructing the
        stub class.
    """
    self.descriptor = service_descriptor

  def BuildServiceStub(self, cls):
    """Constructs the stub class.

    Args:
      cls: The class that will be constructed.
    """

    def _ServiceStubInit(stub, rpc_channel):
      stub.rpc_channel = rpc_channel
    self.cls = cls
    cls.__init__ = _ServiceStubInit
    for method in self.descriptor.methods:
      setattr(cls, method.name, self._GenerateStubMethod(method))

  def _GenerateStubMethod(self, method):
    return (lambda inst, rpc_controller, request, callback=None:
        self._StubMethod(inst, method, rpc_controller, request, callback))

  def _StubMethod(self, stub, method_descriptor,
                  rpc_controller, request, callback):
    """The body of all service methods in the generated stub class.

    Args:
      stub: Stub instance.
      method_descriptor: Descriptor of the invoked method.
      rpc_controller: Rpc controller to execute the method.
      request: Request protocol message.
      callback: A callback to execute when the method finishes.
    Returns:
      Response message (in case of blocking call).
    """
    return stub.rpc_channel.CallMethod(
        method_descriptor, rpc_controller, request,
        method_descriptor.output_type._concrete_class, callback)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\fusions\fusion_gelu.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import onnx

from ..onnx_model import ONNXModel
from .fusion import Fusion


class FusionGelu(Fusion):
    def __init__(self, model: ONNXModel):
        super().__init__(model, "Gelu", "Erf")

    def fuse(
        self,
        erf_node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ):
        """
        Interface function that tries to fuse a node sequence containing an Erf node into a single
        Gelu node.
        """
        if (
            self.fuse_1(erf_node, input_name_to_nodes, output_name_to_node)
            or self.fuse_2(erf_node, input_name_to_nodes, output_name_to_node)
            or self.fuse_3(erf_node, input_name_to_nodes, output_name_to_node)
        ):
            self.model.set_opset_import("com.microsoft", 1)

    def fuse_1(
        self,
        erf_node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ) -> bool:
        """
        This pattern is from PyTorch model
        Fuse Gelu with Erf into one node:
        Pattern 1:
                       +-------Mul(0.5)---------------------+
                       |                                    |
                       |                                    v
                    [root] --> Div -----> Erf  --> Add --> Mul -->
                              (B=1.4142...)       (1)

        Pattern 2:
                       +------------------------------------+
                       |                                    |
                       |                                    v
                    [root] --> Div -----> Erf  --> Add --> Mul -->Mul -->
                              (B=1.4142...)       (1)            (0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if erf_node.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return False
        add_after_erf = children[0]

        if not self.has_constant_input(add_after_erf, 1):
            return False

        if add_after_erf.output[0] not in input_name_to_nodes:
            return False

        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return False

        mul_after_erf = children[0]

        div = self.match_parent(erf_node, "Div", 0, output_name_to_node)
        if div is None:
            return False

        if self.find_constant_input(div, 1.4142, delta=0.001) != 1:
            return False

        subgraph_input = div.input[0]

        another = 1 if mul_after_erf.input[0] == add_after_erf.output[0] else 0
        if subgraph_input == mul_after_erf.input[another]:  # pattern 2
            children = input_name_to_nodes[mul_after_erf.output[0]]
            if len(children) != 1 or children[0].op_type != "Mul":
                return False
            mul_half = children[0]
            if not self.has_constant_input(mul_half, 0.5):
                return False
            subgraph_output = mul_half.output[0]
        else:  # pattern 1
            mul_half = self.match_parent(mul_after_erf, "Mul", another, output_name_to_node)
            if mul_half is None:
                return False

            if not self.has_constant_input(mul_half, 0.5):
                return False

            if subgraph_input not in mul_half.input:
                return False

            subgraph_output = mul_after_erf.output[0]

        subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul_half]
        if not self.is_safe_to_fuse_nodes(subgraph_nodes, [subgraph_output], input_name_to_nodes, output_name_to_node):
            return False

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = onnx.helper.make_node(
            "Gelu", name=self.create_unique_node_name(), inputs=[subgraph_input], outputs=[subgraph_output]
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True

    def fuse_2(
        self,
        erf_node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ) -> bool:
        """
        This pattern is from Keras model
        Fuse Gelu with Erf into one node:
                       +------------------------------------------+
                       |                                          |
                       |                                          v
                    [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                              (B=1.4142...)       (A=1)   (A=0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if erf_node.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return False
        add_after_erf = children[0]

        if not self.has_constant_input(add_after_erf, 1):
            return False

        if add_after_erf.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return False
        mul_after_erf = children[0]

        if not self.has_constant_input(mul_after_erf, 0.5):
            return False

        if mul_after_erf.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[mul_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return False
        mul = children[0]

        div = self.match_parent(erf_node, "Div", 0, output_name_to_node)
        if div is None:
            return False

        sqrt_node = None
        if self.find_constant_input(div, 1.4142, delta=0.001) != 1:
            sqrt_node = self.match_parent(div, "Sqrt", 1, output_name_to_node)
            if sqrt_node is None:
                return False
            if not self.has_constant_input(sqrt_node, 2.0):
                return False

        subgraph_input = div.input[0]

        if subgraph_input not in mul.input:
            return False

        subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul]
        if sqrt_node:
            subgraph_nodes.append(sqrt_node)

        if not self.is_safe_to_fuse_nodes(subgraph_nodes, [mul.output[0]], input_name_to_nodes, output_name_to_node):
            return False

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = onnx.helper.make_node(
            "Gelu", name=self.create_unique_node_name(), inputs=[subgraph_input], outputs=[mul.output[0]]
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True

    def fuse_3(
        self,
        erf_node: onnx.NodeProto,
        input_name_to_nodes: dict[str, list[onnx.NodeProto]],
        output_name_to_node: dict[str, onnx.NodeProto],
    ) -> bool:
        """
        This pattern is from TensorFlow model
        Fuse Gelu with Erf into one node:
                       +----------------------------------------------+
                       |                                              |
                       |                                              v
                    [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                               (A=0.7071067690849304)  (B=1)  (B=0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """

        if erf_node.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return False
        add_after_erf = children[0]

        if not self.has_constant_input(add_after_erf, 1):
            return False

        if add_after_erf.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return False
        mul_half = children[0]

        if not self.has_constant_input(mul_half, 0.5):
            return False

        first_mul = self.match_parent(erf_node, "Mul", 0, output_name_to_node)
        if first_mul is None:
            return False

        i = self.find_constant_input(first_mul, 0.7071067690849304, delta=0.001)
        if i < 0:
            return False

        root_input_index = 1 - i
        subgraph_input = first_mul.input[root_input_index]

        if mul_half.output[0] not in input_name_to_nodes:
            return False
        children = input_name_to_nodes[mul_half.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return False
        last_mul = children[0]

        if not (last_mul.input[0] == subgraph_input or last_mul.input[1] == subgraph_input):
            return False

        subgraph_nodes = [first_mul, erf_node, add_after_erf, mul_half, last_mul]
        if not self.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [last_mul.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return False

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = onnx.helper.make_node(
            "Gelu", name=self.create_unique_node_name(), inputs=[subgraph_input], outputs=[last_mul.output[0]]
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\image_decoder.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import warnings

import torch
import torch.nn.functional as F
from image_encoder import SAM2ImageEncoder, random_sam2_input_image
from mask_decoder import SAM2MaskDecoder
from prompt_encoder import SAM2PromptEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2_utils import compare_tensors_with_tolerance
from torch import nn

logger = logging.getLogger(__name__)


class SAM2ImageDecoder(nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        multimask_output: bool,
        dynamic_multimask_via_stability: bool = True,
        return_logits: bool = False,
        mask_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.prompt_encoder = SAM2PromptEncoder(sam_model)
        self.mask_decoder = SAM2MaskDecoder(sam_model, multimask_output, dynamic_multimask_via_stability)
        self.return_logits = return_logits
        self.mask_threshold = mask_threshold

    @torch.no_grad()
    def forward(
        self,
        image_features_0: torch.Tensor,
        image_features_1: torch.Tensor,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        input_masks: torch.Tensor,
        has_input_masks: torch.Tensor,
        original_image_size: torch.Tensor,
        enable_nvtx_profile: bool = False,
    ):
        """
        Decode masks from image features and prompts. Batched images are not supported. H=W=1024.

        Args:
            image_features_0 (torch.Tensor): [1, 32, H/4, W/4]. high resolution features of level 0 from image encoder.
            image_features_1 (torch.Tensor): [1, 64, H/8, W/8]. high resolution features of level 1 from image encoder.
            image_embeddings (torch.Tensor): [1, 256, H/16, W/16]. image embedding from image encoder.
            point_coords (torch.Tensor): [L, P, 2] shape and float32 dtype and contains the absolute pixel
                                         coordinate in (x, y) format of the P input points in image of size 1024x1024.
            point_labels (torch.Tensor): shape [L, P] and int32 dtype, where 1 means
                                         positive (foreground), 0 means negative (background), -1 means padding,
                                         2 (box left upper corner), 3 (box right bottom corner).
            input_masks (torch.Tensor): [L, 1, H/4, W/4]. Low resolution mask input to the model.
                                        Typically coming from a previous iteration.
            has_input_masks (torch.Tensor): [L]. 1.0 if input_masks is used, 0.0 otherwise.
            original_image_size(torch.Tensor): [2]. original image size H_o, W_o.
            enable_nvtx_profile (bool): enable NVTX profiling.

        Returns:
            masks (torch.Tensor): [1, M, H_o, W_o] where M=3 or 1. Masks of original image size.
            iou_predictions (torch.Tensor): [1, M]. scores for M masks.
            low_res_masks (torch.Tensor, optional): [1, M, H/4, W/4]. low resolution masks.
        """
        nvtx_helper = None
        if enable_nvtx_profile:
            from nvtx_helper import NvtxHelper

            nvtx_helper = NvtxHelper(["prompt_encoder", "mask_decoder", "post_process"])

        if nvtx_helper is not None:
            nvtx_helper.start_profile("prompt_encoder", color="blue")

        sparse_embeddings, dense_embeddings, image_pe = self.prompt_encoder(
            point_coords, point_labels, input_masks, has_input_masks
        )

        if nvtx_helper is not None:
            nvtx_helper.stop_profile("prompt_encoder")
            nvtx_helper.start_profile("mask_decoder", color="red")

        low_res_masks, iou_predictions = self.mask_decoder(
            image_features_0, image_features_1, image_embeddings, image_pe, sparse_embeddings, dense_embeddings
        )

        if nvtx_helper is not None:
            nvtx_helper.stop_profile("mask_decoder")
            nvtx_helper.start_profile("post_process", color="green")

        # Interpolate the low resolution masks back to the original image size.
        masks = F.interpolate(
            low_res_masks,
            (original_image_size[0], original_image_size[1]),
            mode="bilinear",
            align_corners=False,  # Note that align_corners=True has less mismatches during comparing ORT and PyTorch.
        )

        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not self.return_logits:
            masks = masks > self.mask_threshold

        if nvtx_helper is not None:
            nvtx_helper.stop_profile("post_process")
            nvtx_helper.print_latency()

        return masks, iou_predictions, low_res_masks


def export_decoder_onnx(
    sam2_model: SAM2Base,
    onnx_model_path: str,
    multimask_output: bool = False,
    verbose: bool = False,
):
    batch_size = 1
    image = random_sam2_input_image(batch_size)
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    image_features_0, image_features_1, image_embeddings = sam2_encoder(image)

    logger.info("image_features_0.shape: %s", image_features_0.shape)
    logger.info("image_features_1.shape: %s", image_features_1.shape)
    logger.info("image_embeddings.shape: %s", image_embeddings.shape)

    sam2_decoder = SAM2ImageDecoder(
        sam2_model,
        multimask_output=multimask_output,
        dynamic_multimask_via_stability=True,
    ).cpu()

    num_labels = 2
    num_points = 3
    point_coords = torch.randint(low=0, high=1024, size=(num_labels, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(num_labels, num_points), dtype=torch.int32)
    input_masks = torch.zeros(num_labels, 1, 256, 256, dtype=torch.float)
    has_input_masks = torch.ones(1, dtype=torch.float)
    original_image_size = torch.tensor([1200, 1800], dtype=torch.int32)

    example_inputs = (
        image_features_0,
        image_features_1,
        image_embeddings,
        point_coords,
        point_labels,
        input_masks,
        has_input_masks,
        original_image_size,
    )

    logger.info("point_coords.shape: %s", point_coords.shape)
    logger.info("point_labels.shape: %s", point_labels.shape)
    logger.info("input_masks.shape: %s", input_masks.shape)
    logger.info("has_input_masks.shape: %s", has_input_masks.shape)
    logger.info("original_image_size.shape: %s", original_image_size.shape)

    if verbose:
        masks, iou_predictions, low_res_masks = sam2_decoder(*example_inputs)
        logger.info("masks.shape: %s", masks.shape)
        logger.info("iou_predictions.shape: %s", iou_predictions.shape)
        logger.info("low_res_masks.shape: %s", low_res_masks.shape)

    input_names = [
        "image_features_0",
        "image_features_1",
        "image_embeddings",
        "point_coords",
        "point_labels",
        "input_masks",
        "has_input_masks",
        "original_image_size",
    ]

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    dynamic_axes = {
        "point_coords": {0: "num_labels", 1: "num_points"},
        "point_labels": {0: "num_labels", 1: "num_points"},
        "input_masks": {0: "num_labels"},
        "has_input_masks": {0: "num_labels"},
        "masks": {0: "num_labels", 2: "original_image_height", 3: "original_image_width"},
        "low_res_masks": {0: "num_labels"},
        "iou_predictions": {0: "num_labels"},
    }

    with warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

        torch.onnx.export(
            sam2_decoder,
            example_inputs,
            onnx_model_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    logger.info("decoder onnx model saved to %s", onnx_model_path)


def test_decoder_onnx(
    sam2_model: SAM2Base,
    onnx_model_path: str,
    multimask_output=False,
):
    batch_size = 1
    image = random_sam2_input_image(batch_size)
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    image_features_0, image_features_1, image_embeddings = sam2_encoder(image)

    sam2_image_decoder = SAM2ImageDecoder(
        sam2_model,
        multimask_output=multimask_output,
        dynamic_multimask_via_stability=True,
    ).cpu()

    num_labels = 1
    num_points = 5
    point_coords = torch.randint(low=0, high=1024, size=(num_labels, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(num_labels, num_points), dtype=torch.int32)
    input_masks = torch.zeros(num_labels, 1, 256, 256, dtype=torch.float)
    has_input_masks = torch.zeros(1, dtype=torch.float)
    original_image_size = torch.tensor([1500, 1500], dtype=torch.int32)

    example_inputs = (
        image_features_0,
        image_features_1,
        image_embeddings,
        point_coords,
        point_labels,
        input_masks,
        has_input_masks,
        original_image_size,
    )

    masks, iou_predictions, low_res_masks = sam2_image_decoder(*example_inputs)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    logger.info("input_names: %s", input_names)

    model_outputs = ort_session.get_outputs()
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    logger.info("output_names: %s", output_names)
    inputs = {model_inputs[i].name: example_inputs[i].numpy() for i in range(len(model_inputs))}
    outputs = ort_session.run(output_names, inputs)

    for i, output_name in enumerate(output_names):
        logger.info(f"{output_name}.shape: %s", outputs[i].shape)

    ort_masks, ort_iou_predictions, ort_low_res_masks = outputs
    if (
        compare_tensors_with_tolerance("masks", masks.float(), torch.tensor(ort_masks).float())
        and compare_tensors_with_tolerance("iou_predictions", iou_predictions, torch.tensor(ort_iou_predictions))
        and compare_tensors_with_tolerance("low_res_masks", low_res_masks, torch.tensor(ort_low_res_masks))
    ):
        print("onnx model has been verified:", onnx_model_path)
    else:
        print("onnx model verification failed:", onnx_model_path)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\descriptors\base.py ===
# Copyright (c) 2010-2024 openpyxl


"""
Based on Python Cookbook 3rd Edition, 8.13
http://chimera.labs.oreilly.com/books/1230000000393/ch08.html#_discussiuncion_130
"""

import datetime
import re

from openpyxl import DEBUG
from openpyxl.utils.datetime import from_ISO8601

from .namespace import namespaced

class Descriptor:

    def __init__(self, name=None, **kw):
        self.name = name
        for k, v in kw.items():
            setattr(self, k, v)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


class Typed(Descriptor):
    """Values must of a particular type"""

    expected_type = type(None)
    allow_none = False
    nested = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.__doc__ = f"Values must be of type {self.expected_type}"

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            if (not self.allow_none
                or (self.allow_none and value is not None)):
                msg = f"{instance.__class__}.{self.name} should be {self.expected_type} but value is {type(value)}"
                if DEBUG:
                    msg = f"{instance.__class__}.{self.name} should be {self.expected_type} but {value} is {type(value)}"
                raise TypeError(msg)
        super().__set__(instance, value)

    def __repr__(self):
        return  self.__doc__


def _convert(expected_type, value):
    """
    Check value is of or can be converted to expected type.
    """
    if not isinstance(value, expected_type):
        try:
            value = expected_type(value)
        except:
            raise TypeError('expected ' + str(expected_type))
    return value


class Convertible(Typed):
    """Values must be convertible to a particular type"""

    def __set__(self, instance, value):
        if ((self.allow_none and value is not None)
            or not self.allow_none):
            value = _convert(self.expected_type, value)
        super().__set__(instance, value)


class Max(Convertible):
    """Values must be less than a `max` value"""

    expected_type = float
    allow_none = False

    def __init__(self, **kw):
        if 'max' not in kw and not hasattr(self, 'max'):
            raise TypeError('missing max value')
        super().__init__(**kw)

    def __set__(self, instance, value):
        if ((self.allow_none and value is not None)
            or not self.allow_none):
            value = _convert(self.expected_type, value)
            if value > self.max:
                raise ValueError('Max value is {0}'.format(self.max))
        super().__set__(instance, value)


class Min(Convertible):
    """Values must be greater than a `min` value"""

    expected_type = float
    allow_none = False

    def __init__(self, **kw):
        if 'min' not in kw and not hasattr(self, 'min'):
            raise TypeError('missing min value')
        super().__init__(**kw)

    def __set__(self, instance, value):
        if ((self.allow_none and value is not None)
            or not self.allow_none):
            value = _convert(self.expected_type, value)
            if value < self.min:
                raise ValueError('Min value is {0}'.format(self.min))
        super().__set__(instance, value)


class MinMax(Min, Max):
    """Values must be greater than `min` value and less than a `max` one"""
    pass


class Set(Descriptor):
    """Value can only be from a set of know values"""

    def __init__(self, name=None, **kw):
        if not 'values' in kw:
            raise TypeError("missing set of values")
        kw['values'] = set(kw['values'])
        super().__init__(name, **kw)
        self.__doc__ = "Value must be one of {0}".format(self.values)

    def __set__(self, instance, value):
        if value not in self.values:
            raise ValueError(self.__doc__)
        super().__set__(instance, value)


class NoneSet(Set):

    """'none' will be treated as None"""

    def __init__(self, name=None, **kw):
        super().__init__(name, **kw)
        self.values.add(None)

    def __set__(self, instance, value):
        if value == 'none':
            value = None
        super().__set__(instance, value)


class Integer(Convertible):

    expected_type = int


class Float(Convertible):

    expected_type = float


class Bool(Convertible):

    expected_type = bool

    def __set__(self, instance, value):
        if isinstance(value, str):
            if value in ('false', 'f', '0'):
                value = False
        super().__set__(instance, value)


class String(Typed):

    expected_type = str


class Text(String, Convertible):

    pass


class ASCII(Typed):

    expected_type = bytes


class Tuple(Typed):

    expected_type = tuple


class Length(Descriptor):

    def __init__(self, name=None, **kw):
        if "length" not in kw:
            raise TypeError("value length must be supplied")
        super().__init__(**kw)


    def __set__(self, instance, value):
        if len(value) != self.length:
            raise ValueError("Value must be length {0}".format(self.length))
        super().__set__(instance, value)


class Default(Typed):
    """
    When called returns an instance of the expected type.
    Additional default values can be passed in to the descriptor
    """

    def __init__(self, name=None, **kw):
        if "defaults" not in kw:
            kw['defaults'] = {}
        super().__init__(**kw)

    def __call__(self):
        return self.expected_type()


class Alias(Descriptor):
    """
    Aliases can be used when either the desired attribute name is not allowed
    or confusing in Python (eg. "type") or a more descriptive name is desired
    (eg. "underline" for "u")
    """

    def __init__(self, alias):
        self.alias = alias

    def __set__(self, instance, value):
        setattr(instance, self.alias, value)

    def __get__(self, instance, cls):
        return getattr(instance, self.alias)


class MatchPattern(Descriptor):
    """Values must match a regex pattern """
    allow_none = False

    def __init__(self, name=None, **kw):
        if 'pattern' not in kw and not hasattr(self, 'pattern'):
            raise TypeError('missing pattern value')

        super().__init__(name, **kw)
        self.test_pattern = re.compile(self.pattern, re.VERBOSE)


    def __set__(self, instance, value):

        if value is None and not self.allow_none:
            raise ValueError("Value must not be none")

        if ((self.allow_none and value is not None)
            or not self.allow_none):
            if not self.test_pattern.match(value):
                raise ValueError('Value does not match pattern {0}'.format(self.pattern))

        super().__set__(instance, value)


class DateTime(Typed):

    expected_type = datetime.datetime

    def __set__(self, instance, value):
        if value is not None and isinstance(value, str):
            try:
                value = from_ISO8601(value)
            except ValueError:
                raise ValueError("Value must be ISO datetime format")
        super().__set__(instance, value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\integer.py ===
from __future__ import annotations

from typing import ClassVar

import numpy as np

from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_integer_dtype

from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)


class IntegerDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size & kind of integer dtype.

    These specific implementations are subclasses of the non-public
    IntegerDtype. For example, we have Int8Dtype to represent signed int 8s.

    The attributes name & type are set when these subclasses are created.
    """

    _default_np_dtype = np.dtype(np.int64)
    _checker = is_integer_dtype

    @classmethod
    def construct_array_type(cls) -> type[IntegerArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return IntegerArray

    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, IntegerDtype]:
        return NUMPY_INT_TO_DTYPE

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless. e.g. if 'values'
        has a floating dtype, each value must be an integer.
        """
        try:
            return values.astype(dtype, casting="safe", copy=copy)
        except TypeError as err:
            casted = values.astype(dtype, copy=copy)
            if (casted == values).all():
                return casted

            raise TypeError(
                f"cannot safely cast non-equivalent {values.dtype} to {np.dtype(dtype)}"
            ) from err


class IntegerArray(NumericArray):
    """
    Array of integer (optional missing) values.

    Uses :attr:`pandas.NA` as the missing value.

    .. warning::

       IntegerArray is currently experimental, and its API or internal
       implementation may change without warning.

    We represent an IntegerArray with 2 numpy arrays:

    - data: contains a numpy integer array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an IntegerArray from generic array-like input, use
    :func:`pandas.array` with one of the integer dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d integer-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    IntegerArray

    Examples
    --------
    Create an IntegerArray with :func:`pandas.array`.

    >>> int_array = pd.array([1, None, 3], dtype=pd.Int32Dtype())
    >>> int_array
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([1, None, 3], dtype='Int32')
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: Int32

    >>> pd.array([1, None, 3], dtype='UInt16')
    <IntegerArray>
    [1, <NA>, 3]
    Length: 3, dtype: UInt16
    """

    _dtype_cls = IntegerDtype

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = 1
    # Fill values used for any/all
    # Incompatible types in assignment (expression has type "int", base class
    # "BaseMaskedArray" defined the type as "<typing special form>")
    _truthy_value = 1  # type: ignore[assignment]
    _falsey_value = 0  # type: ignore[assignment]


_dtype_docstring = """
An ExtensionDtype for {dtype} integer data.

Uses :attr:`pandas.NA` as its missing value, rather than :attr:`numpy.nan`.

Attributes
----------
None

Methods
-------
None

Examples
--------
For Int8Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int8Dtype())
>>> ser.dtype
Int8Dtype()

For Int16Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int16Dtype())
>>> ser.dtype
Int16Dtype()

For Int32Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int32Dtype())
>>> ser.dtype
Int32Dtype()

For Int64Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.Int64Dtype())
>>> ser.dtype
Int64Dtype()

For UInt8Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt8Dtype())
>>> ser.dtype
UInt8Dtype()

For UInt16Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt16Dtype())
>>> ser.dtype
UInt16Dtype()

For UInt32Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt32Dtype())
>>> ser.dtype
UInt32Dtype()

For UInt64Dtype:

>>> ser = pd.Series([2, pd.NA], dtype=pd.UInt64Dtype())
>>> ser.dtype
UInt64Dtype()
"""

# create the Dtype


@register_extension_dtype
class Int8Dtype(IntegerDtype):
    type = np.int8
    name: ClassVar[str] = "Int8"
    __doc__ = _dtype_docstring.format(dtype="int8")


@register_extension_dtype
class Int16Dtype(IntegerDtype):
    type = np.int16
    name: ClassVar[str] = "Int16"
    __doc__ = _dtype_docstring.format(dtype="int16")


@register_extension_dtype
class Int32Dtype(IntegerDtype):
    type = np.int32
    name: ClassVar[str] = "Int32"
    __doc__ = _dtype_docstring.format(dtype="int32")


@register_extension_dtype
class Int64Dtype(IntegerDtype):
    type = np.int64
    name: ClassVar[str] = "Int64"
    __doc__ = _dtype_docstring.format(dtype="int64")


@register_extension_dtype
class UInt8Dtype(IntegerDtype):
    type = np.uint8
    name: ClassVar[str] = "UInt8"
    __doc__ = _dtype_docstring.format(dtype="uint8")


@register_extension_dtype
class UInt16Dtype(IntegerDtype):
    type = np.uint16
    name: ClassVar[str] = "UInt16"
    __doc__ = _dtype_docstring.format(dtype="uint16")


@register_extension_dtype
class UInt32Dtype(IntegerDtype):
    type = np.uint32
    name: ClassVar[str] = "UInt32"
    __doc__ = _dtype_docstring.format(dtype="uint32")


@register_extension_dtype
class UInt64Dtype(IntegerDtype):
    type = np.uint64
    name: ClassVar[str] = "UInt64"
    __doc__ = _dtype_docstring.format(dtype="uint64")


NUMPY_INT_TO_DTYPE: dict[np.dtype, IntegerDtype] = {
    np.dtype(np.int8): Int8Dtype(),
    np.dtype(np.int16): Int16Dtype(),
    np.dtype(np.int32): Int32Dtype(),
    np.dtype(np.int64): Int64Dtype(),
    np.dtype(np.uint8): UInt8Dtype(),
    np.dtype(np.uint16): UInt16Dtype(),
    np.dtype(np.uint32): UInt32Dtype(),
    np.dtype(np.uint64): UInt64Dtype(),
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\methods\to_dict.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import maybe_box_native
from pandas.core.dtypes.dtypes import (
    BaseMaskedDtype,
    ExtensionDtype,
)

from pandas.core import common as com

if TYPE_CHECKING:
    from pandas._typing import MutableMappingT

    from pandas import DataFrame


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> MutableMappingT:
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[MutableMappingT] | MutableMappingT,
    index: bool = ...,
) -> list[MutableMappingT]:
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> dict:
    ...


@overload
def to_dict(
    df: DataFrame,
    orient: Literal["records"],
    *,
    into: type[dict] = ...,
    index: bool = ...,
) -> list[dict]:
    ...


# error: Incompatible default for argument "into" (default has type "type[dict
# [Any, Any]]", argument has type "type[MutableMappingT] | MutableMappingT")
def to_dict(
    df: DataFrame,
    orient: Literal[
        "dict", "list", "series", "split", "tight", "records", "index"
    ] = "dict",
    *,
    into: type[MutableMappingT] | MutableMappingT = dict,  # type: ignore[assignment]
    index: bool = True,
) -> MutableMappingT | list[MutableMappingT]:
    """
    Convert the DataFrame to a dictionary.

    The type of the key-value pairs can be customized with the parameters
    (see below).

    Parameters
    ----------
    orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        Determines the type of the values of the dictionary.

        - 'dict' (default) : dict like {column -> {index -> value}}
        - 'list' : dict like {column -> [values]}
        - 'series' : dict like {column -> Series(values)}
        - 'split' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
        - 'tight' : dict like
          {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
          'index_names' -> [index.names], 'column_names' -> [column.names]}
        - 'records' : list like
          [{column -> value}, ... , {column -> value}]
        - 'index' : dict like {index -> {column -> value}}

        .. versionadded:: 1.4.0
            'tight' as an allowed value for the ``orient`` argument

    into : class, default dict
        The collections.abc.MutableMapping subclass used for all Mappings
        in the return value.  Can be the actual class or an empty
        instance of the mapping type you want.  If you want a
        collections.defaultdict, you must pass it initialized.

    index : bool, default True
        Whether to include the index item (and index_names item if `orient`
        is 'tight') in the returned dictionary. Can only be ``False``
        when `orient` is 'split' or 'tight'.

        .. versionadded:: 2.0.0

    Returns
    -------
    dict, list or collections.abc.Mapping
        Return a collections.abc.MutableMapping object representing the
        DataFrame. The resulting transformation depends on the `orient` parameter.
    """
    if not df.columns.is_unique:
        warnings.warn(
            "DataFrame columns are not unique, some columns will be omitted.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
    # GH16122
    into_c = com.standardize_mapping(into)

    #  error: Incompatible types in assignment (expression has type "str",
    # variable has type "Literal['dict', 'list', 'series', 'split', 'tight',
    # 'records', 'index']")
    orient = orient.lower()  # type: ignore[assignment]

    if not index and orient not in ["split", "tight"]:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split' or 'tight'"
        )

    if orient == "series":
        # GH46470 Return quickly if orient series to avoid creating dtype objects
        return into_c((k, v) for k, v in df.items())

    box_native_indices = [
        i
        for i, col_dtype in enumerate(df.dtypes.values)
        if col_dtype == np.dtype(object) or isinstance(col_dtype, ExtensionDtype)
    ]
    box_na_values = [
        lib.no_default if not isinstance(col_dtype, BaseMaskedDtype) else libmissing.NA
        for i, col_dtype in enumerate(df.dtypes.values)
    ]
    are_all_object_dtype_cols = len(box_native_indices) == len(df.dtypes)

    if orient == "dict":
        return into_c((k, v.to_dict(into=into)) for k, v in df.items())

    elif orient == "list":
        object_dtype_indices_as_set: set[int] = set(box_native_indices)
        return into_c(
            (
                k,
                list(map(maybe_box_native, v.to_numpy(na_value=box_na_values[i])))
                if i in object_dtype_indices_as_set
                else list(map(maybe_box_native, v.to_numpy())),
            )
            for i, (k, v) in enumerate(df.items())
        )

    elif orient == "split":
        data = df._create_data_for_split_and_tight_to_dict(
            are_all_object_dtype_cols, box_native_indices
        )

        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                ("data", data),
            )
        )

    elif orient == "tight":
        data = df._create_data_for_split_and_tight_to_dict(
            are_all_object_dtype_cols, box_native_indices
        )

        return into_c(
            ((("index", df.index.tolist()),) if index else ())
            + (
                ("columns", df.columns.tolist()),
                (
                    "data",
                    [
                        list(map(maybe_box_native, t))
                        for t in df.itertuples(index=False, name=None)
                    ],
                ),
            )
            + ((("index_names", list(df.index.names)),) if index else ())
            + (("column_names", list(df.columns.names)),)
        )

    elif orient == "records":
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            rows = (
                dict(zip(columns, row)) for row in df.itertuples(index=False, name=None)
            )
            return [
                into_c((k, maybe_box_native(v)) for k, v in row.items()) for row in rows
            ]
        else:
            data = [
                into_c(zip(columns, t)) for t in df.itertuples(index=False, name=None)
            ]
            if box_native_indices:
                object_dtype_indices_as_set = set(box_native_indices)
                object_dtype_cols = {
                    col
                    for i, col in enumerate(df.columns)
                    if i in object_dtype_indices_as_set
                }
                for row in data:
                    for col in object_dtype_cols:
                        row[col] = maybe_box_native(row[col])
            return data

    elif orient == "index":
        if not df.index.is_unique:
            raise ValueError("DataFrame index must be unique for orient='index'.")
        columns = df.columns.tolist()
        if are_all_object_dtype_cols:
            return into_c(
                (t[0], dict(zip(df.columns, map(maybe_box_native, t[1:]))))
                for t in df.itertuples(name=None)
            )
        elif box_native_indices:
            object_dtype_indices_as_set = set(box_native_indices)
            is_object_dtype_by_index = [
                i in object_dtype_indices_as_set for i in range(len(df.columns))
            ]
            return into_c(
                (
                    t[0],
                    {
                        columns[i]: maybe_box_native(v)
                        if is_object_dtype_by_index[i]
                        else v
                        for i, v in enumerate(t[1:])
                    },
                )
                for t in df.itertuples(name=None)
            )
        else:
            return into_c(
                (t[0], dict(zip(df.columns, t[1:]))) for t in df.itertuples(name=None)
            )

    else:
        raise ValueError(f"orient '{orient}' not understood")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\platformdirs\unix.py ===
"""Unix."""

from __future__ import annotations

import os
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from .api import PlatformDirsABC

if TYPE_CHECKING:
    from collections.abc import Iterator

if sys.platform == "win32":

    def getuid() -> NoReturn:
        msg = "should only be used on Unix"
        raise RuntimeError(msg)

else:
    from os import getuid


class Unix(PlatformDirsABC):  # noqa: PLR0904
    """
    On Unix/Linux, we follow the `XDG Basedir Spec <https://specifications.freedesktop.org/basedir-spec/basedir-spec-
    latest.html>`_.

    The spec allows overriding directories with environment variables. The examples shown are the default values,
    alongside the name of the environment variable that overrides them. Makes use of the `appname
    <platformdirs.api.PlatformDirsABC.appname>`, `version <platformdirs.api.PlatformDirsABC.version>`, `multipath
    <platformdirs.api.PlatformDirsABC.multipath>`, `opinion <platformdirs.api.PlatformDirsABC.opinion>`, `ensure_exists
    <platformdirs.api.PlatformDirsABC.ensure_exists>`.

    """

    @property
    def user_data_dir(self) -> str:
        """
        :return: data directory tied to the user, e.g. ``~/.local/share/$appname/$version`` or
         ``$XDG_DATA_HOME/$appname/$version``
        """
        path = os.environ.get("XDG_DATA_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.local/share")  # noqa: PTH111
        return self._append_app_name_and_version(path)

    @property
    def _site_data_dirs(self) -> list[str]:
        path = os.environ.get("XDG_DATA_DIRS", "")
        if not path.strip():
            path = f"/usr/local/share{os.pathsep}/usr/share"
        return [self._append_app_name_and_version(p) for p in path.split(os.pathsep)]

    @property
    def site_data_dir(self) -> str:
        """
        :return: data directories shared by users (if `multipath <platformdirs.api.PlatformDirsABC.multipath>` is
         enabled and ``XDG_DATA_DIRS`` is set and a multi path the response is also a multi path separated by the
         OS path separator), e.g. ``/usr/local/share/$appname/$version`` or ``/usr/share/$appname/$version``
        """
        # XDG default for $XDG_DATA_DIRS; only first, if multipath is False
        dirs = self._site_data_dirs
        if not self.multipath:
            return dirs[0]
        return os.pathsep.join(dirs)

    @property
    def user_config_dir(self) -> str:
        """
        :return: config directory tied to the user, e.g. ``~/.config/$appname/$version`` or
         ``$XDG_CONFIG_HOME/$appname/$version``
        """
        path = os.environ.get("XDG_CONFIG_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.config")  # noqa: PTH111
        return self._append_app_name_and_version(path)

    @property
    def _site_config_dirs(self) -> list[str]:
        path = os.environ.get("XDG_CONFIG_DIRS", "")
        if not path.strip():
            path = "/etc/xdg"
        return [self._append_app_name_and_version(p) for p in path.split(os.pathsep)]

    @property
    def site_config_dir(self) -> str:
        """
        :return: config directories shared by users (if `multipath <platformdirs.api.PlatformDirsABC.multipath>`
         is enabled and ``XDG_CONFIG_DIRS`` is set and a multi path the response is also a multi path separated by
         the OS path separator), e.g. ``/etc/xdg/$appname/$version``
        """
        # XDG default for $XDG_CONFIG_DIRS only first, if multipath is False
        dirs = self._site_config_dirs
        if not self.multipath:
            return dirs[0]
        return os.pathsep.join(dirs)

    @property
    def user_cache_dir(self) -> str:
        """
        :return: cache directory tied to the user, e.g. ``~/.cache/$appname/$version`` or
         ``~/$XDG_CACHE_HOME/$appname/$version``
        """
        path = os.environ.get("XDG_CACHE_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.cache")  # noqa: PTH111
        return self._append_app_name_and_version(path)

    @property
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users, e.g. ``/var/cache/$appname/$version``"""
        return self._append_app_name_and_version("/var/cache")

    @property
    def user_state_dir(self) -> str:
        """
        :return: state directory tied to the user, e.g. ``~/.local/state/$appname/$version`` or
         ``$XDG_STATE_HOME/$appname/$version``
        """
        path = os.environ.get("XDG_STATE_HOME", "")
        if not path.strip():
            path = os.path.expanduser("~/.local/state")  # noqa: PTH111
        return self._append_app_name_and_version(path)

    @property
    def user_log_dir(self) -> str:
        """:return: log directory tied to the user, same as `user_state_dir` if not opinionated else ``log`` in it"""
        path = self.user_state_dir
        if self.opinion:
            path = os.path.join(path, "log")  # noqa: PTH118
            self._optionally_create_directory(path)
        return path

    @property
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user, e.g. ``~/Documents``"""
        return _get_user_media_dir("XDG_DOCUMENTS_DIR", "~/Documents")

    @property
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user, e.g. ``~/Downloads``"""
        return _get_user_media_dir("XDG_DOWNLOAD_DIR", "~/Downloads")

    @property
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user, e.g. ``~/Pictures``"""
        return _get_user_media_dir("XDG_PICTURES_DIR", "~/Pictures")

    @property
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user, e.g. ``~/Videos``"""
        return _get_user_media_dir("XDG_VIDEOS_DIR", "~/Videos")

    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user, e.g. ``~/Music``"""
        return _get_user_media_dir("XDG_MUSIC_DIR", "~/Music")

    @property
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user, e.g. ``~/Desktop``"""
        return _get_user_media_dir("XDG_DESKTOP_DIR", "~/Desktop")

    @property
    def user_runtime_dir(self) -> str:
        """
        :return: runtime directory tied to the user, e.g. ``/run/user/$(id -u)/$appname/$version`` or
         ``$XDG_RUNTIME_DIR/$appname/$version``.

         For FreeBSD/OpenBSD/NetBSD, it would return ``/var/run/user/$(id -u)/$appname/$version`` if
         exists, otherwise ``/tmp/runtime-$(id -u)/$appname/$version``, if``$XDG_RUNTIME_DIR``
         is not set.
        """
        path = os.environ.get("XDG_RUNTIME_DIR", "")
        if not path.strip():
            if sys.platform.startswith(("freebsd", "openbsd", "netbsd")):
                path = f"/var/run/user/{getuid()}"
                if not Path(path).exists():
                    path = f"/tmp/runtime-{getuid()}"  # noqa: S108
            else:
                path = f"/run/user/{getuid()}"
        return self._append_app_name_and_version(path)

    @property
    def site_runtime_dir(self) -> str:
        """
        :return: runtime directory shared by users, e.g. ``/run/$appname/$version`` or \
        ``$XDG_RUNTIME_DIR/$appname/$version``.

        Note that this behaves almost exactly like `user_runtime_dir` if ``$XDG_RUNTIME_DIR`` is set, but will
        fall back to paths associated to the root user instead of a regular logged-in user if it's not set.

        If you wish to ensure that a logged-in root user path is returned e.g. ``/run/user/0``, use `user_runtime_dir`
        instead.

        For FreeBSD/OpenBSD/NetBSD, it would return ``/var/run/$appname/$version`` if ``$XDG_RUNTIME_DIR`` is not set.
        """
        path = os.environ.get("XDG_RUNTIME_DIR", "")
        if not path.strip():
            if sys.platform.startswith(("freebsd", "openbsd", "netbsd")):
                path = "/var/run"
            else:
                path = "/run"
        return self._append_app_name_and_version(path)

    @property
    def site_data_path(self) -> Path:
        """:return: data path shared by users. Only return the first item, even if ``multipath`` is set to ``True``"""
        return self._first_item_as_path_if_multipath(self.site_data_dir)

    @property
    def site_config_path(self) -> Path:
        """:return: config path shared by the users, returns the first item, even if ``multipath`` is set to ``True``"""
        return self._first_item_as_path_if_multipath(self.site_config_dir)

    @property
    def site_cache_path(self) -> Path:
        """:return: cache path shared by users. Only return the first item, even if ``multipath`` is set to ``True``"""
        return self._first_item_as_path_if_multipath(self.site_cache_dir)

    def iter_config_dirs(self) -> Iterator[str]:
        """:yield: all user and site configuration directories."""
        yield self.user_config_dir
        yield from self._site_config_dirs

    def iter_data_dirs(self) -> Iterator[str]:
        """:yield: all user and site data directories."""
        yield self.user_data_dir
        yield from self._site_data_dirs


def _get_user_media_dir(env_var: str, fallback_tilde_path: str) -> str:
    media_dir = _get_user_dirs_folder(env_var)
    if media_dir is None:
        media_dir = os.environ.get(env_var, "").strip()
        if not media_dir:
            media_dir = os.path.expanduser(fallback_tilde_path)  # noqa: PTH111

    return media_dir


def _get_user_dirs_folder(key: str) -> str | None:
    """
    Return directory from user-dirs.dirs config file.

    See https://freedesktop.org/wiki/Software/xdg-user-dirs/.

    """
    user_dirs_config_path = Path(Unix().user_config_dir) / "user-dirs.dirs"
    if user_dirs_config_path.exists():
        parser = ConfigParser()

        with user_dirs_config_path.open() as stream:
            # Add fake section header, so ConfigParser doesn't complain
            parser.read_string(f"[top]\n{stream.read()}")

        if key not in parser["top"]:
            return None

        path = parser["top"][key].strip('"')
        # Handle relative home paths
        return path.replace("$HOME", os.path.expanduser("~"))  # noqa: PTH111

    return None


__all__ = [
    "Unix",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\platformdirs\windows.py ===
"""Windows."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING

from .api import PlatformDirsABC

if TYPE_CHECKING:
    from collections.abc import Callable


class Windows(PlatformDirsABC):
    """
    `MSDN on where to store app data files <https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid>`_.

    Makes use of the `appname <platformdirs.api.PlatformDirsABC.appname>`, `appauthor
    <platformdirs.api.PlatformDirsABC.appauthor>`, `version <platformdirs.api.PlatformDirsABC.version>`, `roaming
    <platformdirs.api.PlatformDirsABC.roaming>`, `opinion <platformdirs.api.PlatformDirsABC.opinion>`, `ensure_exists
    <platformdirs.api.PlatformDirsABC.ensure_exists>`.

    """

    @property
    def user_data_dir(self) -> str:
        """
        :return: data directory tied to the user, e.g.
         ``%USERPROFILE%\\AppData\\Local\\$appauthor\\$appname`` (not roaming) or
         ``%USERPROFILE%\\AppData\\Roaming\\$appauthor\\$appname`` (roaming)
        """
        const = "CSIDL_APPDATA" if self.roaming else "CSIDL_LOCAL_APPDATA"
        path = os.path.normpath(get_win_folder(const))
        return self._append_parts(path)

    def _append_parts(self, path: str, *, opinion_value: str | None = None) -> str:
        params = []
        if self.appname:
            if self.appauthor is not False:
                author = self.appauthor or self.appname
                params.append(author)
            params.append(self.appname)
            if opinion_value is not None and self.opinion:
                params.append(opinion_value)
            if self.version:
                params.append(self.version)
        path = os.path.join(path, *params)  # noqa: PTH118
        self._optionally_create_directory(path)
        return path

    @property
    def site_data_dir(self) -> str:
        """:return: data directory shared by users, e.g. ``C:\\ProgramData\\$appauthor\\$appname``"""
        path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
        return self._append_parts(path)

    @property
    def user_config_dir(self) -> str:
        """:return: config directory tied to the user, same as `user_data_dir`"""
        return self.user_data_dir

    @property
    def site_config_dir(self) -> str:
        """:return: config directory shared by the users, same as `site_data_dir`"""
        return self.site_data_dir

    @property
    def user_cache_dir(self) -> str:
        """
        :return: cache directory tied to the user (if opinionated with ``Cache`` folder within ``$appname``) e.g.
         ``%USERPROFILE%\\AppData\\Local\\$appauthor\\$appname\\Cache\\$version``
        """
        path = os.path.normpath(get_win_folder("CSIDL_LOCAL_APPDATA"))
        return self._append_parts(path, opinion_value="Cache")

    @property
    def site_cache_dir(self) -> str:
        """:return: cache directory shared by users, e.g. ``C:\\ProgramData\\$appauthor\\$appname\\Cache\\$version``"""
        path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
        return self._append_parts(path, opinion_value="Cache")

    @property
    def user_state_dir(self) -> str:
        """:return: state directory tied to the user, same as `user_data_dir`"""
        return self.user_data_dir

    @property
    def user_log_dir(self) -> str:
        """:return: log directory tied to the user, same as `user_data_dir` if not opinionated else ``Logs`` in it"""
        path = self.user_data_dir
        if self.opinion:
            path = os.path.join(path, "Logs")  # noqa: PTH118
            self._optionally_create_directory(path)
        return path

    @property
    def user_documents_dir(self) -> str:
        """:return: documents directory tied to the user e.g. ``%USERPROFILE%\\Documents``"""
        return os.path.normpath(get_win_folder("CSIDL_PERSONAL"))

    @property
    def user_downloads_dir(self) -> str:
        """:return: downloads directory tied to the user e.g. ``%USERPROFILE%\\Downloads``"""
        return os.path.normpath(get_win_folder("CSIDL_DOWNLOADS"))

    @property
    def user_pictures_dir(self) -> str:
        """:return: pictures directory tied to the user e.g. ``%USERPROFILE%\\Pictures``"""
        return os.path.normpath(get_win_folder("CSIDL_MYPICTURES"))

    @property
    def user_videos_dir(self) -> str:
        """:return: videos directory tied to the user e.g. ``%USERPROFILE%\\Videos``"""
        return os.path.normpath(get_win_folder("CSIDL_MYVIDEO"))

    @property
    def user_music_dir(self) -> str:
        """:return: music directory tied to the user e.g. ``%USERPROFILE%\\Music``"""
        return os.path.normpath(get_win_folder("CSIDL_MYMUSIC"))

    @property
    def user_desktop_dir(self) -> str:
        """:return: desktop directory tied to the user, e.g. ``%USERPROFILE%\\Desktop``"""
        return os.path.normpath(get_win_folder("CSIDL_DESKTOPDIRECTORY"))

    @property
    def user_runtime_dir(self) -> str:
        """
        :return: runtime directory tied to the user, e.g.
         ``%USERPROFILE%\\AppData\\Local\\Temp\\$appauthor\\$appname``
        """
        path = os.path.normpath(os.path.join(get_win_folder("CSIDL_LOCAL_APPDATA"), "Temp"))  # noqa: PTH118
        return self._append_parts(path)

    @property
    def site_runtime_dir(self) -> str:
        """:return: runtime directory shared by users, same as `user_runtime_dir`"""
        return self.user_runtime_dir


def get_win_folder_from_env_vars(csidl_name: str) -> str:
    """Get folder from environment variables."""
    result = get_win_folder_if_csidl_name_not_env_var(csidl_name)
    if result is not None:
        return result

    env_var_name = {
        "CSIDL_APPDATA": "APPDATA",
        "CSIDL_COMMON_APPDATA": "ALLUSERSPROFILE",
        "CSIDL_LOCAL_APPDATA": "LOCALAPPDATA",
    }.get(csidl_name)
    if env_var_name is None:
        msg = f"Unknown CSIDL name: {csidl_name}"
        raise ValueError(msg)
    result = os.environ.get(env_var_name)
    if result is None:
        msg = f"Unset environment variable: {env_var_name}"
        raise ValueError(msg)
    return result


def get_win_folder_if_csidl_name_not_env_var(csidl_name: str) -> str | None:
    """Get a folder for a CSIDL name that does not exist as an environment variable."""
    if csidl_name == "CSIDL_PERSONAL":
        return os.path.join(os.path.normpath(os.environ["USERPROFILE"]), "Documents")  # noqa: PTH118

    if csidl_name == "CSIDL_DOWNLOADS":
        return os.path.join(os.path.normpath(os.environ["USERPROFILE"]), "Downloads")  # noqa: PTH118

    if csidl_name == "CSIDL_MYPICTURES":
        return os.path.join(os.path.normpath(os.environ["USERPROFILE"]), "Pictures")  # noqa: PTH118

    if csidl_name == "CSIDL_MYVIDEO":
        return os.path.join(os.path.normpath(os.environ["USERPROFILE"]), "Videos")  # noqa: PTH118

    if csidl_name == "CSIDL_MYMUSIC":
        return os.path.join(os.path.normpath(os.environ["USERPROFILE"]), "Music")  # noqa: PTH118
    return None


def get_win_folder_from_registry(csidl_name: str) -> str:
    """
    Get folder from the registry.

    This is a fallback technique at best. I'm not sure if using the registry for these guarantees us the correct answer
    for all CSIDL_* names.

    """
    shell_folder_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
        "CSIDL_PERSONAL": "Personal",
        "CSIDL_DOWNLOADS": "{374DE290-123F-4565-9164-39C4925E467B}",
        "CSIDL_MYPICTURES": "My Pictures",
        "CSIDL_MYVIDEO": "My Video",
        "CSIDL_MYMUSIC": "My Music",
    }.get(csidl_name)
    if shell_folder_name is None:
        msg = f"Unknown CSIDL name: {csidl_name}"
        raise ValueError(msg)
    if sys.platform != "win32":  # only needed for mypy type checker to know that this code runs only on Windows
        raise NotImplementedError
    import winreg  # noqa: PLC0415

    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
    return str(directory)


def get_win_folder_via_ctypes(csidl_name: str) -> str:
    """Get folder with ctypes."""
    # There is no 'CSIDL_DOWNLOADS'.
    # Use 'CSIDL_PROFILE' (40) and append the default folder 'Downloads' instead.
    # https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid

    import ctypes  # noqa: PLC0415

    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
        "CSIDL_PERSONAL": 5,
        "CSIDL_MYPICTURES": 39,
        "CSIDL_MYVIDEO": 14,
        "CSIDL_MYMUSIC": 13,
        "CSIDL_DOWNLOADS": 40,
        "CSIDL_DESKTOPDIRECTORY": 16,
    }.get(csidl_name)
    if csidl_const is None:
        msg = f"Unknown CSIDL name: {csidl_name}"
        raise ValueError(msg)

    buf = ctypes.create_unicode_buffer(1024)
    windll = getattr(ctypes, "windll")  # noqa: B009 # using getattr to avoid false positive with mypy type checker
    windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # Downgrade to short path name if it has high-bit chars.
    if any(ord(c) > 255 for c in buf):  # noqa: PLR2004
        buf2 = ctypes.create_unicode_buffer(1024)
        if windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2

    if csidl_name == "CSIDL_DOWNLOADS":
        return os.path.join(buf.value, "Downloads")  # noqa: PTH118

    return buf.value


def _pick_get_win_folder() -> Callable[[str], str]:
    try:
        import ctypes  # noqa: PLC0415
    except ImportError:
        pass
    else:
        if hasattr(ctypes, "windll"):
            return get_win_folder_via_ctypes
    try:
        import winreg  # noqa: PLC0415, F401
    except ImportError:
        return get_win_folder_from_env_vars
    else:
        return get_win_folder_from_registry


get_win_folder = lru_cache(maxsize=None)(_pick_get_win_folder())

__all__ = [
    "Windows",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\matrixutils.py ===
"""Utilities to deal with sympy.Matrix, numpy and scipy.sparse."""

from sympy.core.expr import Expr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices import eye, zeros
from sympy.external import import_module

__all__ = [
    'numpy_ndarray',
    'scipy_sparse_matrix',
    'sympy_to_numpy',
    'sympy_to_scipy_sparse',
    'numpy_to_sympy',
    'scipy_sparse_to_sympy',
    'flatten_scalar',
    'matrix_dagger',
    'to_sympy',
    'to_numpy',
    'to_scipy_sparse',
    'matrix_tensor_product',
    'matrix_zeros'
]

# Conditionally define the base classes for numpy and scipy.sparse arrays
# for use in isinstance tests.

np = import_module('numpy')
if not np:
    class numpy_ndarray:
        pass
else:
    numpy_ndarray = np.ndarray  # type: ignore

scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
if not scipy:
    class scipy_sparse_matrix:
        pass
    sparse = None
else:
    sparse = scipy.sparse
    scipy_sparse_matrix = sparse.spmatrix # type: ignore


def sympy_to_numpy(m, **options):
    """Convert a SymPy Matrix/complex number to a numpy matrix or scalar."""
    if not np:
        raise ImportError
    dtype = options.get('dtype', 'complex')
    if isinstance(m, MatrixBase):
        return np.array(m.tolist(), dtype=dtype)
    elif isinstance(m, Expr):
        if m.is_Number or m.is_NumberSymbol or m == I:
            return complex(m)
    raise TypeError('Expected MatrixBase or complex scalar, got: %r' % m)


def sympy_to_scipy_sparse(m, **options):
    """Convert a SymPy Matrix/complex number to a numpy matrix or scalar."""
    if not np or not sparse:
        raise ImportError
    dtype = options.get('dtype', 'complex')
    if isinstance(m, MatrixBase):
        return sparse.csr_matrix(np.array(m.tolist(), dtype=dtype))
    elif isinstance(m, Expr):
        if m.is_Number or m.is_NumberSymbol or m == I:
            return complex(m)
    raise TypeError('Expected MatrixBase or complex scalar, got: %r' % m)


def scipy_sparse_to_sympy(m, **options):
    """Convert a scipy.sparse matrix to a SymPy matrix."""
    return MatrixBase(m.todense())


def numpy_to_sympy(m, **options):
    """Convert a numpy matrix to a SymPy matrix."""
    return MatrixBase(m)


def to_sympy(m, **options):
    """Convert a numpy/scipy.sparse matrix to a SymPy matrix."""
    if isinstance(m, MatrixBase):
        return m
    elif isinstance(m, numpy_ndarray):
        return numpy_to_sympy(m)
    elif isinstance(m, scipy_sparse_matrix):
        return scipy_sparse_to_sympy(m)
    elif isinstance(m, Expr):
        return m
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def to_numpy(m, **options):
    """Convert a sympy/scipy.sparse matrix to a numpy matrix."""
    dtype = options.get('dtype', 'complex')
    if isinstance(m, (MatrixBase, Expr)):
        return sympy_to_numpy(m, dtype=dtype)
    elif isinstance(m, numpy_ndarray):
        return m
    elif isinstance(m, scipy_sparse_matrix):
        return m.todense()
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def to_scipy_sparse(m, **options):
    """Convert a sympy/numpy matrix to a scipy.sparse matrix."""
    dtype = options.get('dtype', 'complex')
    if isinstance(m, (MatrixBase, Expr)):
        return sympy_to_scipy_sparse(m, dtype=dtype)
    elif isinstance(m, numpy_ndarray):
        if not sparse:
            raise ImportError
        return sparse.csr_matrix(m)
    elif isinstance(m, scipy_sparse_matrix):
        return m
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % m)


def flatten_scalar(e):
    """Flatten a 1x1 matrix to a scalar, return larger matrices unchanged."""
    if isinstance(e, MatrixBase):
        if e.shape == (1, 1):
            e = e[0]
    if isinstance(e, (numpy_ndarray, scipy_sparse_matrix)):
        if e.shape == (1, 1):
            e = complex(e[0, 0])
    return e


def matrix_dagger(e):
    """Return the dagger of a sympy/numpy/scipy.sparse matrix."""
    if isinstance(e, MatrixBase):
        return e.H
    elif isinstance(e, (numpy_ndarray, scipy_sparse_matrix)):
        return e.conjugate().transpose()
    raise TypeError('Expected sympy/numpy/scipy.sparse matrix, got: %r' % e)


# TODO: Move this into sympy.matrices.
def _sympy_tensor_product(*matrices):
    """Compute the kronecker product of a sequence of SymPy Matrices.
    """
    from sympy.matrices.expressions.kronecker import matrix_kronecker_product

    return matrix_kronecker_product(*matrices)


def _numpy_tensor_product(*product):
    """numpy version of tensor product of multiple arguments."""
    if not np:
        raise ImportError
    answer = product[0]
    for item in product[1:]:
        answer = np.kron(answer, item)
    return answer


def _scipy_sparse_tensor_product(*product):
    """scipy.sparse version of tensor product of multiple arguments."""
    if not sparse:
        raise ImportError
    answer = product[0]
    for item in product[1:]:
        answer = sparse.kron(answer, item)
    # The final matrices will just be multiplied, so csr is a good final
    # sparse format.
    return sparse.csr_matrix(answer)


def matrix_tensor_product(*product):
    """Compute the matrix tensor product of sympy/numpy/scipy.sparse matrices."""
    if isinstance(product[0], MatrixBase):
        return _sympy_tensor_product(*product)
    elif isinstance(product[0], numpy_ndarray):
        return _numpy_tensor_product(*product)
    elif isinstance(product[0], scipy_sparse_matrix):
        return _scipy_sparse_tensor_product(*product)


def _numpy_eye(n):
    """numpy version of complex eye."""
    if not np:
        raise ImportError
    return np.array(np.eye(n, dtype='complex'))


def _scipy_sparse_eye(n):
    """scipy.sparse version of complex eye."""
    if not sparse:
        raise ImportError
    return sparse.eye(n, n, dtype='complex')


def matrix_eye(n, **options):
    """Get the version of eye and tensor_product for a given format."""
    format = options.get('format', 'sympy')
    if format == 'sympy':
        return eye(n)
    elif format == 'numpy':
        return _numpy_eye(n)
    elif format == 'scipy.sparse':
        return _scipy_sparse_eye(n)
    raise NotImplementedError('Invalid format: %r' % format)


def _numpy_zeros(m, n, **options):
    """numpy version of zeros."""
    dtype = options.get('dtype', 'float64')
    if not np:
        raise ImportError
    return np.zeros((m, n), dtype=dtype)


def _scipy_sparse_zeros(m, n, **options):
    """scipy.sparse version of zeros."""
    spmatrix = options.get('spmatrix', 'csr')
    dtype = options.get('dtype', 'float64')
    if not sparse:
        raise ImportError
    if spmatrix == 'lil':
        return sparse.lil_matrix((m, n), dtype=dtype)
    elif spmatrix == 'csr':
        return sparse.csr_matrix((m, n), dtype=dtype)


def matrix_zeros(m, n, **options):
    """"Get a zeros matrix for a given format."""
    format = options.get('format', 'sympy')
    if format == 'sympy':
        return zeros(m, n)
    elif format == 'numpy':
        return _numpy_zeros(m, n, **options)
    elif format == 'scipy.sparse':
        return _scipy_sparse_zeros(m, n, **options)
    raise NotImplementedError('Invaild format: %r' % format)


def _numpy_matrix_to_zero(e):
    """Convert a numpy zero matrix to the zero scalar."""
    if not np:
        raise ImportError
    test = np.zeros_like(e)
    if np.allclose(e, test):
        return 0.0
    else:
        return e


def _scipy_sparse_matrix_to_zero(e):
    """Convert a scipy.sparse zero matrix to the zero scalar."""
    if not np:
        raise ImportError
    edense = e.todense()
    test = np.zeros_like(edense)
    if np.allclose(edense, test):
        return 0.0
    else:
        return e


def matrix_to_zero(e):
    """Convert a zero matrix to the scalar zero."""
    if isinstance(e, MatrixBase):
        if zeros(*e.shape) == e:
            e = S.Zero
    elif isinstance(e, numpy_ndarray):
        e = _numpy_matrix_to_zero(e)
    elif isinstance(e, scipy_sparse_matrix):
        e = _scipy_sparse_matrix_to_zero(e)
    return e

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\hypergeometric.py ===
r'''
This module contains the implementation of the 2nd_hypergeometric hint for
dsolve. This is an incomplete implementation of the algorithm described in [1].
The algorithm solves 2nd order linear ODEs of the form

.. math:: y'' + A(x) y' + B(x) y = 0\text{,}

where `A` and `B` are rational functions. The algorithm should find any
solution of the form

.. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
Currently only the 2F1 case is implemented in SymPy but the other cases are
described in the paper and could be implemented in future (contributions
welcome!).

References
==========

.. [1] L. Chan, E.S. Cheb-Terrab, Non-Liouvillian solutions for second order
       linear ODEs, (2004).
       https://arxiv.org/abs/math-ph/0402063
'''

from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants


def match_2nd_hypergeometric(eq, func):
    x = func.args[0]
    df = func.diff(x)
    a3 = Wild('a3', exclude=[func, func.diff(x), func.diff(x, 2)])
    b3 = Wild('b3', exclude=[func, func.diff(x), func.diff(x, 2)])
    c3 = Wild('c3', exclude=[func, func.diff(x), func.diff(x, 2)])
    deq = a3*(func.diff(x, 2)) + b3*df + c3*func
    r = collect(eq,
        [func.diff(x, 2), func.diff(x), func]).match(deq)
    if r:
        if not all(val.is_polynomial() for val in r.values()):
            n, d = eq.as_numer_denom()
            eq = expand(n)
            r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)

    if r and r[a3]!=0:
        A = cancel(r[b3]/r[a3])
        B = cancel(r[c3]/r[a3])
        return [A, B]
    else:
        return []


def equivalence_hypergeometric(A, B, func):
    # This method for finding the equivalence is only for 2F1 type.
    # We can extend it for 1F1 and 0F1 type also.
    x = func.args[0]

    # making given equation in normal form
    I1 = factor(cancel(A.diff(x)/2 + A**2/4 - B))

    # computing shifted invariant(J1) of the equation
    J1 = factor(cancel(x**2*I1 + S(1)/4))
    num, dem = J1.as_numer_denom()
    num = powdenest(expand(num))
    dem = powdenest(expand(dem))
    # this function will compute the different powers of variable(x) in J1.
    # then it will help in finding value of k. k is power of x such that we can express
    # J1 = x**k * J0(x**k) then all the powers in J0 become integers.
    def _power_counting(num):
        _pow = {0}
        for val in num:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    _pow.add(val.as_base_exp()[1])
                elif val == x:
                    _pow.add(val.as_base_exp()[1])
                else:
                    _pow.update(_power_counting(val.args))
        return _pow

    pow_num = _power_counting((num, ))
    pow_dem = _power_counting((dem, ))
    pow_dem.update(pow_num)

    _pow = pow_dem
    k = gcd(_pow)

    # computing I0 of the given equation
    I0 = powdenest(simplify(factor(((J1/k**2) - S(1)/4)/((x**k)**2))), force=True)
    I0 = factor(cancel(powdenest(I0.subs(x, x**(S(1)/k)), force=True)))

    # Before this point I0, J1 might be functions of e.g. sqrt(x) but replacing
    # x with x**(1/k) should result in I0 being a rational function of x or
    # otherwise the hypergeometric solver cannot be used. Note that k can be a
    # non-integer rational such as 2/7.
    if not I0.is_rational_function(x):
        return None

    num, dem = I0.as_numer_denom()

    max_num_pow = max(_power_counting((num, )))
    dem_args = dem.args
    sing_point = []
    dem_pow = []
    # calculating singular point of I0.
    for arg in dem_args:
        if arg.has(x):
            if isinstance(arg, Pow):
                # (x-a)**n
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg.as_base_exp()[0], x).keys())[0])
            else:
                # (x-a) type
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg, x).keys())[0])

    dem_pow.sort()
    # checking if equivalence is exists or not.

    if equivalence(max_num_pow, dem_pow) == "2F1":
        return {'I0':I0, 'k':k, 'sing_point':sing_point, 'type':"2F1"}
    else:
        return None


def match_2nd_2F1_hypergeometric(I, k, sing_point, func):
    x = func.args[0]
    a = Wild("a")
    b = Wild("b")
    c = Wild("c")
    t = Wild("t")
    s = Wild("s")
    r = Wild("r")
    alpha = Wild("alpha")
    beta = Wild("beta")
    gamma = Wild("gamma")
    delta = Wild("delta")
    # I0 of the standard 2F1 equation.
    I0 = ((a-b+1)*(a-b-1)*x**2 + 2*((1-a-b)*c + 2*a*b)*x + c*(c-2))/(4*x**2*(x-1)**2)
    if sing_point != [0, 1]:
        # If singular point is [0, 1] then we have standard equation.
        eqs = []
        sing_eqs = [-beta/alpha, -delta/gamma, (delta-beta)/(alpha-gamma)]
        # making equations for the finding the mobius transformation
        for i in range(3):
            if i<len(sing_point):
                eqs.append(Eq(sing_eqs[i], sing_point[i]))
            else:
                eqs.append(Eq(1/sing_eqs[i], 0))
        # solving above equations for the mobius transformation
        _beta = -alpha*sing_point[0]
        _delta = -gamma*sing_point[1]
        _gamma = alpha
        if len(sing_point) == 3:
            _gamma = (_beta + sing_point[2]*alpha)/(sing_point[2] - sing_point[1])
        mob = (alpha*x + beta)/(gamma*x + delta)
        mob = mob.subs(beta, _beta)
        mob = mob.subs(delta, _delta)
        mob = mob.subs(gamma, _gamma)
        mob = cancel(mob)
        t = (beta - delta*x)/(gamma*x - alpha)
        t = cancel(((t.subs(beta, _beta)).subs(delta, _delta)).subs(gamma, _gamma))
    else:
        mob = x
        t = x

    # applying mobius transformation in I to make it into I0.
    I = I.subs(x, t)
    I = I*(t.diff(x))**2
    I = factor(I)
    dict_I = {x**2:0, x:0, 1:0}
    I0_num, I0_dem = I0.as_numer_denom()
    # collecting coeff of (x**2, x), of the standard equation.
    # substituting (a-b) = s, (a+b) = r
    dict_I0 = {x**2:s**2 - 1, x:(2*(1-r)*c + (r+s)*(r-s)), 1:c*(c-2)}
    # collecting coeff of (x**2, x) from I0 of the given equation.
    dict_I.update(collect(expand(cancel(I*I0_dem)), [x**2, x], evaluate=False))
    eqs = []
    # We are comparing the coeff of powers of different x, for finding the values of
    # parameters of standard equation.
    for key in [x**2, x, 1]:
        eqs.append(Eq(dict_I[key], dict_I0[key]))

    # We can have many possible roots for the equation.
    # I am selecting the root on the basis that when we have
    # standard equation eq = x*(x-1)*f(x).diff(x, 2) + ((a+b+1)*x-c)*f(x).diff(x) + a*b*f(x)
    # then root should be a, b, c.

    _c = 1 - factor(sqrt(1+eqs[2].lhs))
    if not _c.has(Symbol):
        _c = min(list(roots(eqs[2], c)))
    _s = factor(sqrt(eqs[0].lhs + 1))
    _r = _c - factor(sqrt(_c**2 + _s**2 + eqs[1].lhs - 2*_c))
    _a = (_r + _s)/2
    _b = (_r - _s)/2

    rn = {'a':simplify(_a), 'b':simplify(_b), 'c':simplify(_c), 'k':k, 'mobius':mob, 'type':"2F1"}
    return rn


def equivalence(max_num_pow, dem_pow):
    # this function is made for checking the equivalence with 2F1 type of equation.
    # max_num_pow is the value of maximum power of x in numerator
    # and dem_pow is list of powers of different factor of form (a*x b).
    # reference from table 1 in paper - "Non-Liouvillian solutions for second order
    # linear ODEs" by L. Chan, E.S. Cheb-Terrab.
    # We can extend it for 1F1 and 0F1 type also.

    if max_num_pow == 2:
        if dem_pow in [[2, 2], [2, 2, 2]]:
            return "2F1"
    elif max_num_pow == 1:
        if dem_pow in [[1, 2, 2], [2, 2, 2], [1, 2], [2, 2]]:
            return "2F1"
    elif max_num_pow == 0:
        if dem_pow in [[1, 1, 2], [2, 2], [1, 2, 2], [1, 1], [2], [1, 2], [2, 2]]:
            return "2F1"

    return None


def get_sol_2F1_hypergeometric(eq, func, match_object):
    x = func.args[0]
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.polys.polytools import factor
    C0, C1 = get_numbered_constants(eq, num=2)
    a = match_object['a']
    b = match_object['b']
    c = match_object['c']
    A = match_object['A']

    sol = None

    if c.is_integer == False:
        sol = C0*hyper([a, b], [c], x) + C1*hyper([a-c+1, b-c+1], [2-c], x)*x**(1-c)
    elif c == 1:
        y2 = Integral(exp(Integral((-(a+b+1)*x + c)/(x**2-x), x))/(hyperexpand(hyper([a, b], [c], x))**2), x)*hyper([a, b], [c], x)
        sol = C0*hyper([a, b], [c], x) + C1*y2
    elif (c-a-b).is_integer == False:
        sol = C0*hyper([a, b], [1+a+b-c], 1-x) + C1*hyper([c-a, c-b], [1+c-a-b], 1-x)*(1-x)**(c-a-b)

    if sol:
        # applying transformation in the solution
        subs = match_object['mobius']
        dtdx = simplify(1/(subs.diff(x)))
        _B = ((a + b + 1)*x - c).subs(x, subs)*dtdx
        _B = factor(_B + ((x**2 -x).subs(x, subs))*(dtdx.diff(x)*dtdx))
        _A = factor((x**2 - x).subs(x, subs)*(dtdx**2))
        e = exp(logcombine(Integral(cancel(_B/(2*_A)), x), force=True))
        sol = sol.subs(x, match_object['mobius'])
        sol = sol.subs(x, x**match_object['k'])
        e = e.subs(x, x**match_object['k'])

        if not A.is_zero:
            e1 = Integral(A/2, x)
            e1 = exp(logcombine(e1, force=True))
            sol = cancel((e/e1)*x**((-match_object['k']+1)/2))*sol
            sol = Eq(func, sol)
            return sol

        sol = cancel((e)*x**((-match_object['k']+1)/2))*sol
        sol = Eq(func, sol)
    return sol

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_ki.py ===
from __future__ import annotations

import signal
import sys
import types
import weakref
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import attrs

from .._util import is_main_thread
from ._run_context import GLOBAL_RUN_CONTEXT

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

    from typing_extensions import Self, TypeGuard
# In ordinary single-threaded Python code, when you hit control-C, it raises
# an exception and automatically does all the regular unwinding stuff.
#
# In Trio code, we would like hitting control-C to raise an exception and
# automatically do all the regular unwinding stuff. In particular, we would
# like to maintain our invariant that all tasks always run to completion (one
# way or another), by unwinding all of them.
#
# But it's basically impossible to write the core task running code in such a
# way that it can maintain this invariant in the face of KeyboardInterrupt
# exceptions arising at arbitrary bytecode positions. Similarly, if a
# KeyboardInterrupt happened at the wrong moment inside pretty much any of our
# inter-task synchronization or I/O primitives, then the system state could
# get corrupted and prevent our being able to clean up properly.
#
# So, we need a way to defer KeyboardInterrupt processing from these critical
# sections.
#
# Things that don't work:
#
# - Listen for SIGINT and process it in a system task: works fine for
#   well-behaved programs that regularly pass through the event loop, but if
#   user-code goes into an infinite loop then it can't be interrupted. Which
#   is unfortunate, since dealing with infinite loops is what
#   KeyboardInterrupt is for!
#
# - Use pthread_sigmask to disable signal delivery during critical section:
#   (a) windows has no pthread_sigmask, (b) python threads start with all
#   signals unblocked, so if there are any threads around they'll receive the
#   signal and then tell the main thread to run the handler, even if the main
#   thread has that signal blocked.
#
# - Install a signal handler which checks a global variable to decide whether
#   to raise the exception immediately (if we're in a non-critical section),
#   or to schedule it on the event loop (if we're in a critical section). The
#   problem here is that it's impossible to transition safely out of user code:
#
#     with keyboard_interrupt_enabled:
#         msg = coro.send(value)
#
#   If this raises a KeyboardInterrupt, it might be because the coroutine got
#   interrupted and has unwound... or it might be the KeyboardInterrupt
#   arrived just *after* 'send' returned, so the coroutine is still running,
#   but we just lost the message it sent. (And worse, in our actual task
#   runner, the send is hidden inside a utility function etc.)
#
# Solution:
#
# Mark *stack frames* as being interrupt-safe or interrupt-unsafe, and from
# the signal handler check which kind of frame we're currently in when
# deciding whether to raise or schedule the exception.
#
# There are still some cases where this can fail, like if someone hits
# control-C while the process is in the event loop, and then it immediately
# enters an infinite loop in user code. In this case the user has to hit
# control-C a second time. And of course if the user code is written so that
# it doesn't actually exit after a task crashes and everything gets cancelled,
# then there's not much to be done. (Hitting control-C repeatedly might help,
# but in general the solution is to kill the process some other way, just like
# for any Python program that's written to catch and ignore
# KeyboardInterrupt.)

_T = TypeVar("_T")


class _IdRef(weakref.ref[_T]):
    __slots__ = ("_hash",)
    _hash: int

    def __new__(
        cls,
        ob: _T,
        callback: Callable[[Self], object] | None = None,
        /,
    ) -> Self:
        self: Self = weakref.ref.__new__(cls, ob, callback)
        self._hash = object.__hash__(ob)
        return self

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True

        if not isinstance(other, _IdRef):
            return NotImplemented

        my_obj = None
        try:
            my_obj = self()
            return my_obj is not None and my_obj is other()
        finally:
            del my_obj

    # we're overriding a builtin so we do need this
    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return self._hash


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


# see also: https://github.com/python/cpython/issues/88306
class WeakKeyIdentityDictionary(Generic[_KT, _VT]):
    def __init__(self) -> None:
        self._data: dict[_IdRef[_KT], _VT] = {}

        def remove(
            k: _IdRef[_KT],
            selfref: weakref.ref[
                WeakKeyIdentityDictionary[_KT, _VT]
            ] = weakref.ref(  # noqa: B008  # function-call-in-default-argument
                self,
            ),
        ) -> None:
            self = selfref()
            if self is not None:
                try:  # noqa: SIM105  # suppressible-exception
                    del self._data[k]
                except KeyError:
                    pass

        self._remove = remove

    def __getitem__(self, k: _KT) -> _VT:
        return self._data[_IdRef(k)]

    def __setitem__(self, k: _KT, v: _VT) -> None:
        self._data[_IdRef(k, self._remove)] = v


_CODE_KI_PROTECTION_STATUS_WMAP: WeakKeyIdentityDictionary[
    types.CodeType,
    bool,
] = WeakKeyIdentityDictionary()


# This is to support the async_generator package necessary for aclosing on <3.10
# functions decorated @async_generator are given this magic property that's a
# reference to the object itself
# see python-trio/async_generator/async_generator/_impl.py
def legacy_isasyncgenfunction(
    obj: object,
) -> TypeGuard[Callable[..., types.AsyncGeneratorType[object, object]]]:
    return getattr(obj, "_async_gen_function", None) == id(obj)


# NB: according to the signal.signal docs, 'frame' can be None on entry to
# this function:
def ki_protection_enabled(frame: types.FrameType | None) -> bool:
    try:
        task = GLOBAL_RUN_CONTEXT.task
    except AttributeError:
        task_ki_protected = False
        task_frame = None
    else:
        task_ki_protected = task._ki_protected
        task_frame = task.coro.cr_frame

    while frame is not None:
        try:
            v = _CODE_KI_PROTECTION_STATUS_WMAP[frame.f_code]
        except KeyError:
            pass
        else:
            return bool(v)
        if frame.f_code.co_name == "__del__":
            return True
        if frame is task_frame:
            return task_ki_protected
        frame = frame.f_back
    return True


def currently_ki_protected() -> bool:
    r"""Check whether the calling code has :exc:`KeyboardInterrupt` protection
    enabled.

    It's surprisingly easy to think that one's :exc:`KeyboardInterrupt`
    protection is enabled when it isn't, or vice-versa. This function tells
    you what Trio thinks of the matter, which makes it useful for ``assert``\s
    and unit tests.

    Returns:
      bool: True if protection is enabled, and False otherwise.

    """
    return ki_protection_enabled(sys._getframe())


class _SupportsCode(Protocol):
    __code__: types.CodeType


_T_supports_code = TypeVar("_T_supports_code", bound=_SupportsCode)


def enable_ki_protection(f: _T_supports_code, /) -> _T_supports_code:
    """Decorator to enable KI protection."""
    orig = f

    if legacy_isasyncgenfunction(f):
        f = f.__wrapped__  # type: ignore

    _CODE_KI_PROTECTION_STATUS_WMAP[f.__code__] = True
    return orig


def disable_ki_protection(f: _T_supports_code, /) -> _T_supports_code:
    """Decorator to disable KI protection."""
    orig = f

    if legacy_isasyncgenfunction(f):
        f = f.__wrapped__  # type: ignore

    _CODE_KI_PROTECTION_STATUS_WMAP[f.__code__] = False
    return orig


@attrs.define(slots=False)
class KIManager:
    handler: Callable[[int, types.FrameType | None], None] | None = None

    def install(
        self,
        deliver_cb: Callable[[], object],
        restrict_keyboard_interrupt_to_checkpoints: bool,
    ) -> None:
        assert self.handler is None
        if (
            not is_main_thread()
            or signal.getsignal(signal.SIGINT) != signal.default_int_handler
        ):
            return

        def handler(signum: int, frame: types.FrameType | None) -> None:
            assert signum == signal.SIGINT
            protection_enabled = ki_protection_enabled(frame)
            if protection_enabled or restrict_keyboard_interrupt_to_checkpoints:
                deliver_cb()
            else:
                raise KeyboardInterrupt

        self.handler = handler
        signal.signal(signal.SIGINT, handler)

    def close(self) -> None:
        if self.handler is not None:
            if signal.getsignal(signal.SIGINT) is self.handler:
                signal.signal(signal.SIGINT, signal.default_int_handler)
            self.handler = None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\timeout.py ===
from __future__ import absolute_import

import time

# The default socket timeout, used by httplib to indicate that no timeout was; specified by the user
from socket import _GLOBAL_DEFAULT_TIMEOUT, getdefaulttimeout

from ..exceptions import TimeoutStateError

# A sentinel value to indicate that no timeout was specified by the user in
# urllib3
_Default = object()


# Use time.monotonic if available.
current_time = getattr(time, "monotonic", time.time)


class Timeout(object):
    """Timeout configuration.

    Timeouts can be defined as a default for a pool:

    .. code-block:: python

       timeout = Timeout(connect=2.0, read=7.0)
       http = PoolManager(timeout=timeout)
       response = http.request('GET', 'http://example.com/')

    Or per-request (which overrides the default for the pool):

    .. code-block:: python

       response = http.request('GET', 'http://example.com/', timeout=Timeout(10))

    Timeouts can be disabled by setting all the parameters to ``None``:

    .. code-block:: python

       no_timeout = Timeout(connect=None, read=None)
       response = http.request('GET', 'http://example.com/, timeout=no_timeout)


    :param total:
        This combines the connect and read timeouts into one; the read timeout
        will be set to the time leftover from the connect attempt. In the
        event that both a connect timeout and a total are specified, or a read
        timeout and a total are specified, the shorter timeout will be applied.

        Defaults to None.

    :type total: int, float, or None

    :param connect:
        The maximum amount of time (in seconds) to wait for a connection
        attempt to a server to succeed. Omitting the parameter will default the
        connect timeout to the system default, probably `the global default
        timeout in socket.py
        <http://hg.python.org/cpython/file/603b4d593758/Lib/socket.py#l535>`_.
        None will set an infinite timeout for connection attempts.

    :type connect: int, float, or None

    :param read:
        The maximum amount of time (in seconds) to wait between consecutive
        read operations for a response from the server. Omitting the parameter
        will default the read timeout to the system default, probably `the
        global default timeout in socket.py
        <http://hg.python.org/cpython/file/603b4d593758/Lib/socket.py#l535>`_.
        None will set an infinite timeout.

    :type read: int, float, or None

    .. note::

        Many factors can affect the total amount of time for urllib3 to return
        an HTTP response.

        For example, Python's DNS resolver does not obey the timeout specified
        on the socket. Other factors that can affect total request time include
        high CPU load, high swap, the program running at a low priority level,
        or other behaviors.

        In addition, the read and total timeouts only measure the time between
        read operations on the socket connecting the client and the server,
        not the total amount of time for the request to return a complete
        response. For most requests, the timeout is raised because the server
        has not sent the first byte in the specified time. This is not always
        the case; if a server streams one byte every fifteen seconds, a timeout
        of 20 seconds will not trigger, even though the request will take
        several minutes to complete.

        If your goal is to cut off any request after a set amount of wall clock
        time, consider having a second "watcher" thread to cut off a slow
        request.
    """

    #: A sentinel object representing the default timeout value
    DEFAULT_TIMEOUT = _GLOBAL_DEFAULT_TIMEOUT

    def __init__(self, total=None, connect=_Default, read=_Default):
        self._connect = self._validate_timeout(connect, "connect")
        self._read = self._validate_timeout(read, "read")
        self.total = self._validate_timeout(total, "total")
        self._start_connect = None

    def __repr__(self):
        return "%s(connect=%r, read=%r, total=%r)" % (
            type(self).__name__,
            self._connect,
            self._read,
            self.total,
        )

    # __str__ provided for backwards compatibility
    __str__ = __repr__

    @classmethod
    def resolve_default_timeout(cls, timeout):
        return getdefaulttimeout() if timeout is cls.DEFAULT_TIMEOUT else timeout

    @classmethod
    def _validate_timeout(cls, value, name):
        """Check that a timeout attribute is valid.

        :param value: The timeout value to validate
        :param name: The name of the timeout attribute to validate. This is
            used to specify in error messages.
        :return: The validated and casted version of the given value.
        :raises ValueError: If it is a numeric value less than or equal to
            zero, or the type is not an integer, float, or None.
        """
        if value is _Default:
            return cls.DEFAULT_TIMEOUT

        if value is None or value is cls.DEFAULT_TIMEOUT:
            return value

        if isinstance(value, bool):
            raise ValueError(
                "Timeout cannot be a boolean value. It must "
                "be an int, float or None."
            )
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Timeout value %s was %s, but it must be an "
                "int, float or None." % (name, value)
            )

        try:
            if value <= 0:
                raise ValueError(
                    "Attempted to set %s timeout to %s, but the "
                    "timeout cannot be set to a value less "
                    "than or equal to 0." % (name, value)
                )
        except TypeError:
            # Python 3
            raise ValueError(
                "Timeout value %s was %s, but it must be an "
                "int, float or None." % (name, value)
            )

        return value

    @classmethod
    def from_float(cls, timeout):
        """Create a new Timeout from a legacy timeout value.

        The timeout value used by httplib.py sets the same timeout on the
        connect(), and recv() socket requests. This creates a :class:`Timeout`
        object that sets the individual timeouts to the ``timeout`` value
        passed to this function.

        :param timeout: The legacy timeout value.
        :type timeout: integer, float, sentinel default object, or None
        :return: Timeout object
        :rtype: :class:`Timeout`
        """
        return Timeout(read=timeout, connect=timeout)

    def clone(self):
        """Create a copy of the timeout object

        Timeout properties are stored per-pool but each request needs a fresh
        Timeout object to ensure each one has its own start/stop configured.

        :return: a copy of the timeout object
        :rtype: :class:`Timeout`
        """
        # We can't use copy.deepcopy because that will also create a new object
        # for _GLOBAL_DEFAULT_TIMEOUT, which socket.py uses as a sentinel to
        # detect the user default.
        return Timeout(connect=self._connect, read=self._read, total=self.total)

    def start_connect(self):
        """Start the timeout clock, used during a connect() attempt

        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to start a timer that has been started already.
        """
        if self._start_connect is not None:
            raise TimeoutStateError("Timeout timer has already been started.")
        self._start_connect = current_time()
        return self._start_connect

    def get_connect_duration(self):
        """Gets the time elapsed since the call to :meth:`start_connect`.

        :return: Elapsed time in seconds.
        :rtype: float
        :raises urllib3.exceptions.TimeoutStateError: if you attempt
            to get duration for a timer that hasn't been started.
        """
        if self._start_connect is None:
            raise TimeoutStateError(
                "Can't get connect duration for timer that has not started."
            )
        return current_time() - self._start_connect

    @property
    def connect_timeout(self):
        """Get the value to use when setting a connection timeout.

        This will be a positive float or integer, the value None
        (never timeout), or the default system timeout.

        :return: Connect timeout.
        :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
        """
        if self.total is None:
            return self._connect

        if self._connect is None or self._connect is self.DEFAULT_TIMEOUT:
            return self.total

        return min(self._connect, self.total)

    @property
    def read_timeout(self):
        """Get the value for the read timeout.

        This assumes some time has elapsed in the connection timeout and
        computes the read timeout appropriately.

        If self.total is set, the read timeout is dependent on the amount of
        time taken by the connect timeout. If the connection time has not been
        established, a :exc:`~urllib3.exceptions.TimeoutStateError` will be
        raised.

        :return: Value to use for the read timeout.
        :rtype: int, float, :attr:`Timeout.DEFAULT_TIMEOUT` or None
        :raises urllib3.exceptions.TimeoutStateError: If :meth:`start_connect`
            has not yet been called on this object.
        """
        if (
            self.total is not None
            and self.total is not self.DEFAULT_TIMEOUT
            and self._read is not None
            and self._read is not self.DEFAULT_TIMEOUT
        ):
            # In case the connect timeout has not yet been established.
            if self._start_connect is None:
                return self._read
            return max(0, min(self.total - self.get_connect_duration(), self._read))
        elif self.total is not None and self.total is not self.DEFAULT_TIMEOUT:
            return max(0, self.total - self.get_connect_duration())
        else:
            return self._read

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\__init__.py ===
r"""
N-dim array module for SymPy.

Four classes are provided to handle N-dim arrays, given by the combinations
dense/sparse (i.e. whether to store all elements or only the non-zero ones in
memory) and mutable/immutable (immutable classes are SymPy objects, but cannot
change after they have been created).

Examples
========

The following examples show the usage of ``Array``. This is an abbreviation for
``ImmutableDenseNDimArray``, that is an immutable and dense N-dim array, the
other classes are analogous. For mutable classes it is also possible to change
element values after the object has been constructed.

Array construction can detect the shape of nested lists and tuples:

>>> from sympy import Array
>>> a1 = Array([[1, 2], [3, 4], [5, 6]])
>>> a1
[[1, 2], [3, 4], [5, 6]]
>>> a1.shape
(3, 2)
>>> a1.rank()
2
>>> from sympy.abc import x, y, z
>>> a2 = Array([[[x, y], [z, x*z]], [[1, x*y], [1/x, x/y]]])
>>> a2
[[[x, y], [z, x*z]], [[1, x*y], [1/x, x/y]]]
>>> a2.shape
(2, 2, 2)
>>> a2.rank()
3

Otherwise one could pass a 1-dim array followed by a shape tuple:

>>> m1 = Array(range(12), (3, 4))
>>> m1
[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
>>> m2 = Array(range(12), (3, 2, 2))
>>> m2
[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
>>> m2[1,1,1]
7
>>> m2.reshape(4, 3)
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

Slice support:

>>> m2[:, 1, 1]
[3, 7, 11]

Elementwise derivative:

>>> from sympy.abc import x, y, z
>>> m3 = Array([x**3, x*y, z])
>>> m3.diff(x)
[3*x**2, y, 0]
>>> m3.diff(z)
[0, 0, 1]

Multiplication with other SymPy expressions is applied elementwisely:

>>> (1+x)*m3
[x**3*(x + 1), x*y*(x + 1), z*(x + 1)]

To apply a function to each element of the N-dim array, use ``applyfunc``:

>>> m3.applyfunc(lambda x: x/2)
[x**3/2, x*y/2, z/2]

N-dim arrays can be converted to nested lists by the ``tolist()`` method:

>>> m2.tolist()
[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]]
>>> isinstance(m2.tolist(), list)
True

If the rank is 2, it is possible to convert them to matrices with ``tomatrix()``:

>>> m1.tomatrix()
Matrix([
[0, 1,  2,  3],
[4, 5,  6,  7],
[8, 9, 10, 11]])

Products and contractions
-------------------------

Tensor product between arrays `A_{i_1,\ldots,i_n}` and `B_{j_1,\ldots,j_m}`
creates the combined array `P = A \otimes B` defined as

`P_{i_1,\ldots,i_n,j_1,\ldots,j_m} := A_{i_1,\ldots,i_n}\cdot B_{j_1,\ldots,j_m}.`

It is available through ``tensorproduct(...)``:

>>> from sympy import Array, tensorproduct
>>> from sympy.abc import x,y,z,t
>>> A = Array([x, y, z, t])
>>> B = Array([1, 2, 3, 4])
>>> tensorproduct(A, B)
[[x, 2*x, 3*x, 4*x], [y, 2*y, 3*y, 4*y], [z, 2*z, 3*z, 4*z], [t, 2*t, 3*t, 4*t]]

In case you don't want to evaluate the tensor product immediately, you can use
``ArrayTensorProduct``, which creates an unevaluated tensor product expression:

>>> from sympy.tensor.array.expressions import ArrayTensorProduct
>>> ArrayTensorProduct(A, B)
ArrayTensorProduct([x, y, z, t], [1, 2, 3, 4])

Calling ``.as_explicit()`` on ``ArrayTensorProduct`` is equivalent to just calling
``tensorproduct(...)``:

>>> ArrayTensorProduct(A, B).as_explicit()
[[x, 2*x, 3*x, 4*x], [y, 2*y, 3*y, 4*y], [z, 2*z, 3*z, 4*z], [t, 2*t, 3*t, 4*t]]

Tensor product between a rank-1 array and a matrix creates a rank-3 array:

>>> from sympy import eye
>>> p1 = tensorproduct(A, eye(4))
>>> p1
[[[x, 0, 0, 0], [0, x, 0, 0], [0, 0, x, 0], [0, 0, 0, x]], [[y, 0, 0, 0], [0, y, 0, 0], [0, 0, y, 0], [0, 0, 0, y]], [[z, 0, 0, 0], [0, z, 0, 0], [0, 0, z, 0], [0, 0, 0, z]], [[t, 0, 0, 0], [0, t, 0, 0], [0, 0, t, 0], [0, 0, 0, t]]]

Now, to get back `A_0 \otimes \mathbf{1}` one can access `p_{0,m,n}` by slicing:

>>> p1[0,:,:]
[[x, 0, 0, 0], [0, x, 0, 0], [0, 0, x, 0], [0, 0, 0, x]]

Tensor contraction sums over the specified axes, for example contracting
positions `a` and `b` means

`A_{i_1,\ldots,i_a,\ldots,i_b,\ldots,i_n} \implies \sum_k A_{i_1,\ldots,k,\ldots,k,\ldots,i_n}`

Remember that Python indexing is zero starting, to contract the a-th and b-th
axes it is therefore necessary to specify `a-1` and `b-1`

>>> from sympy import tensorcontraction
>>> C = Array([[x, y], [z, t]])

The matrix trace is equivalent to the contraction of a rank-2 array:

`A_{m,n} \implies \sum_k A_{k,k}`

>>> tensorcontraction(C, (0, 1))
t + x

To create an expression representing a tensor contraction that does not get
evaluated immediately, use ``ArrayContraction``, which is equivalent to
``tensorcontraction(...)`` if it is followed by ``.as_explicit()``:

>>> from sympy.tensor.array.expressions import ArrayContraction
>>> ArrayContraction(C, (0, 1))
ArrayContraction([[x, y], [z, t]], (0, 1))
>>> ArrayContraction(C, (0, 1)).as_explicit()
t + x

Matrix product is equivalent to a tensor product of two rank-2 arrays, followed
by a contraction of the 2nd and 3rd axes (in Python indexing axes number 1, 2).

`A_{m,n}\cdot B_{i,j} \implies \sum_k A_{m, k}\cdot B_{k, j}`

>>> D = Array([[2, 1], [0, -1]])
>>> tensorcontraction(tensorproduct(C, D), (1, 2))
[[2*x, x - y], [2*z, -t + z]]

One may verify that the matrix product is equivalent:

>>> from sympy import Matrix
>>> Matrix([[x, y], [z, t]])*Matrix([[2, 1], [0, -1]])
Matrix([
[2*x,  x - y],
[2*z, -t + z]])

or equivalently

>>> C.tomatrix()*D.tomatrix()
Matrix([
[2*x,  x - y],
[2*z, -t + z]])

Diagonal operator
-----------------

The ``tensordiagonal`` function acts in a similar manner as ``tensorcontraction``,
but the joined indices are not summed over, for example diagonalizing
positions `a` and `b` means

`A_{i_1,\ldots,i_a,\ldots,i_b,\ldots,i_n} \implies A_{i_1,\ldots,k,\ldots,k,\ldots,i_n}
\implies \tilde{A}_{i_1,\ldots,i_{a-1},i_{a+1},\ldots,i_{b-1},i_{b+1},\ldots,i_n,k}`

where `\tilde{A}` is the array equivalent to the diagonal of `A` at positions
`a` and `b` moved to the last index slot.

Compare the difference between contraction and diagonal operators:

>>> from sympy import tensordiagonal
>>> from sympy.abc import a, b, c, d
>>> m = Matrix([[a, b], [c, d]])
>>> tensorcontraction(m, [0, 1])
a + d
>>> tensordiagonal(m, [0, 1])
[a, d]

In short, no summation occurs with ``tensordiagonal``.


Derivatives by array
--------------------

The usual derivative operation may be extended to support derivation with
respect to arrays, provided that all elements in the that array are symbols or
expressions suitable for derivations.

The definition of a derivative by an array is as follows: given the array
`A_{i_1, \ldots, i_N}` and the array `X_{j_1, \ldots, j_M}`
the derivative of arrays will return a new array `B` defined by

`B_{j_1,\ldots,j_M,i_1,\ldots,i_N} := \frac{\partial A_{i_1,\ldots,i_N}}{\partial X_{j_1,\ldots,j_M}}`

The function ``derive_by_array`` performs such an operation:

>>> from sympy import derive_by_array
>>> from sympy.abc import x, y, z, t
>>> from sympy import sin, exp

With scalars, it behaves exactly as the ordinary derivative:

>>> derive_by_array(sin(x*y), x)
y*cos(x*y)

Scalar derived by an array basis:

>>> derive_by_array(sin(x*y), [x, y, z])
[y*cos(x*y), x*cos(x*y), 0]

Deriving array by an array basis: `B^{nm} := \frac{\partial A^m}{\partial x^n}`

>>> basis = [x, y, z]
>>> ax = derive_by_array([exp(x), sin(y*z), t], basis)
>>> ax
[[exp(x), 0, 0], [0, z*cos(y*z), 0], [0, y*cos(y*z), 0]]

Contraction of the resulting array: `\sum_m \frac{\partial A^m}{\partial x^m}`

>>> tensorcontraction(ax, (0, 1))
z*cos(y*z) + exp(x)

"""

from .dense_ndim_array import MutableDenseNDimArray, ImmutableDenseNDimArray, DenseNDimArray
from .sparse_ndim_array import MutableSparseNDimArray, ImmutableSparseNDimArray, SparseNDimArray
from .ndim_array import NDimArray, ArrayKind
from .arrayop import tensorproduct, tensorcontraction, tensordiagonal, derive_by_array, permutedims
from .array_comprehension import ArrayComprehension, ArrayComprehensionMap

Array = ImmutableDenseNDimArray

__all__ = [
    'MutableDenseNDimArray', 'ImmutableDenseNDimArray', 'DenseNDimArray',

    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'SparseNDimArray',

    'NDimArray', 'ArrayKind',

    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array',

    'permutedims', 'ArrayComprehension', 'ArrayComprehensionMap',

    'Array',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\exceptions.py ===
"""
General SymPy exceptions and warnings.
"""

import warnings
import contextlib

from textwrap import dedent


class SymPyDeprecationWarning(DeprecationWarning):
    r"""
    A warning for deprecated features of SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    Note that simply constructing this class will not cause a warning to be
    issued. To do that, you must call the :func`sympy_deprecation_warning`
    function. For this reason, it is not recommended to ever construct this
    class directly.

    Explanation
    ===========

    The ``SymPyDeprecationWarning`` class is a subclass of
    ``DeprecationWarning`` that is used for all deprecations in SymPy. A
    special subclass is used so that we can automatically augment the warning
    message with additional metadata about the version the deprecation was
    introduced in and a link to the documentation. This also allows users to
    explicitly filter deprecation warnings from SymPy using ``warnings``
    filters (see :ref:`silencing-sympy-deprecation-warnings`).

    Additionally, ``SymPyDeprecationWarning`` is enabled to be shown by
    default, unlike normal ``DeprecationWarning``\s, which are only shown by
    default in interactive sessions. This ensures that deprecation warnings in
    SymPy will actually be seen by users.

    See the documentation of :func:`sympy_deprecation_warning` for a
    description of the parameters to this function.

    To mark a function as deprecated, you can use the :func:`@deprecated
    <sympy.utilities.decorator.deprecated>` decorator.

    See Also
    ========
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    """
    def __init__(self, message, *, deprecated_since_version, active_deprecations_target):

        super().__init__(message, deprecated_since_version,
                     active_deprecations_target)
        self.message = message
        if not isinstance(deprecated_since_version, str):
            raise TypeError(f"'deprecated_since_version' should be a string, got {deprecated_since_version!r}")
        self.deprecated_since_version = deprecated_since_version
        self.active_deprecations_target = active_deprecations_target
        if any(i in active_deprecations_target for i in '()='):
            raise ValueError("active_deprecations_target be the part inside of the '(...)='")

        self.full_message = f"""

{dedent(message).strip()}

See https://docs.sympy.org/latest/explanation/active-deprecations.html#{active_deprecations_target}
for details.

This has been deprecated since SymPy version {deprecated_since_version}. It
will be removed in a future version of SymPy.
"""

    def __str__(self):
        return self.full_message

    def __repr__(self):
        return f"{self.__class__.__name__}({self.message!r}, deprecated_since_version={self.deprecated_since_version!r}, active_deprecations_target={self.active_deprecations_target!r})"

    def __eq__(self, other):
        return isinstance(other, SymPyDeprecationWarning) and self.args == other.args

    # Make pickling work. The by default, it tries to recreate the expression
    # from its args, but this doesn't work because of our keyword-only
    # arguments.
    @classmethod
    def _new(cls, message, deprecated_since_version,
              active_deprecations_target):
        return cls(message, deprecated_since_version=deprecated_since_version, active_deprecations_target=active_deprecations_target)

    def __reduce__(self):
        return (self._new, (self.message, self.deprecated_since_version, self.active_deprecations_target))

# Python by default hides DeprecationWarnings, which we do not want.
warnings.simplefilter("once", SymPyDeprecationWarning)

def sympy_deprecation_warning(message, *, deprecated_since_version,
                              active_deprecations_target, stacklevel=3):
    r'''
    Warn that a feature is deprecated in SymPy.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    To mark an entire function or class as deprecated, you can use the
    :func:`@deprecated <sympy.utilities.decorator.deprecated>` decorator.

    Parameters
    ==========

    message : str
         The deprecation message. This may span multiple lines and contain
         code examples. Messages should be wrapped to 80 characters. The
         message is automatically dedented and leading and trailing whitespace
         stripped. Messages may include dynamic content based on the user
         input, but avoid using ``str(expression)`` if an expression can be
         arbitrary, as it might be huge and make the warning message
         unreadable.

    deprecated_since_version : str
         The version of SymPy the feature has been deprecated since. For new
         deprecations, this should be the version in `sympy/release.py
         <https://github.com/sympy/sympy/blob/master/sympy/release.py>`_
         without the ``.dev``. If the next SymPy version ends up being
         different from this, the release manager will need to update any
         ``SymPyDeprecationWarning``\s using the incorrect version. This
         argument is required and must be passed as a keyword argument.
         (example:  ``deprecated_since_version="1.10"``).

    active_deprecations_target : str
        The Sphinx target corresponding to the section for the deprecation in
        the :ref:`active-deprecations` document (see
        ``doc/src/explanation/active-deprecations.md``). This is used to
        automatically generate a URL to the page in the warning message. This
        argument is required and must be passed as a keyword argument.
        (example: ``active_deprecations_target="deprecated-feature-abc"``)

    stacklevel : int, default: 3
        The ``stacklevel`` parameter that is passed to ``warnings.warn``. If
        you create a wrapper that calls this function, this should be
        increased so that the warning message shows the user line of code that
        produced the warning. Note that in some cases there will be multiple
        possible different user code paths that could result in the warning.
        In that case, just choose the smallest common stacklevel.

    Examples
    ========

    >>> from sympy.utilities.exceptions import sympy_deprecation_warning
    >>> def is_this_zero(x, y=0):
    ...     """
    ...     Determine if x = 0.
    ...
    ...     Parameters
    ...     ==========
    ...
    ...     x : Expr
    ...       The expression to check.
    ...
    ...     y : Expr, optional
    ...       If provided, check if x = y.
    ...
    ...       .. deprecated:: 1.1
    ...
    ...          The ``y`` argument to ``is_this_zero`` is deprecated. Use
    ...          ``is_this_zero(x - y)`` instead.
    ...
    ...     """
    ...     from sympy import simplify
    ...
    ...     if y != 0:
    ...         sympy_deprecation_warning("""
    ...     The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.""",
    ...             deprecated_since_version="1.1",
    ...             active_deprecations_target='is-this-zero-y-deprecation')
    ...     return simplify(x - y) == 0
    >>> is_this_zero(0)
    True
    >>> is_this_zero(1, 1) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The y argument to is_zero() is deprecated. Use is_zero(x - y) instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#is-this-zero-y-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      is_this_zero(1, 1)
    True

    See Also
    ========

    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.ignore_warnings
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    w = SymPyDeprecationWarning(message,
                            deprecated_since_version=deprecated_since_version,
                                active_deprecations_target=active_deprecations_target)
    warnings.warn(w, stacklevel=stacklevel)


@contextlib.contextmanager
def ignore_warnings(warningcls):
    '''
    Context manager to suppress warnings during tests.

    .. note::

       Do not use this with SymPyDeprecationWarning in the tests.
       warns_deprecated_sympy() should be used instead.

    This function is useful for suppressing warnings during tests. The warns
    function should be used to assert that a warning is raised. The
    ignore_warnings function is useful in situation when the warning is not
    guaranteed to be raised (e.g. on importing a module) or if the warning
    comes from third-party code.

    This function is also useful to prevent the same or similar warnings from
    being issue twice due to recursive calls.

    When the warning is coming (reliably) from SymPy the warns function should
    be preferred to ignore_warnings.

    >>> from sympy.utilities.exceptions import ignore_warnings
    >>> import warnings

    Here's a warning:

    >>> with warnings.catch_warnings():  # reset warnings in doctest
    ...     warnings.simplefilter('error')
    ...     warnings.warn('deprecated', UserWarning)
    Traceback (most recent call last):
      ...
    UserWarning: deprecated

    Let's suppress it with ignore_warnings:

    >>> with warnings.catch_warnings():  # reset warnings in doctest
    ...     warnings.simplefilter('error')
    ...     with ignore_warnings(UserWarning):
    ...         warnings.warn('deprecated', UserWarning)

    (No warning emitted)

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.decorator.deprecated
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    # Absorbs all warnings in warnrec
    with warnings.catch_warnings(record=True) as warnrec:
        # Make sure our warning doesn't get filtered
        warnings.simplefilter("always", warningcls)
        # Now run the test
        yield

    # Reissue any warnings that we aren't testing for
    for w in warnrec:
        if not issubclass(w.category, warningcls):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\ssltransport.py ===
from __future__ import annotations

import io
import socket
import ssl
import typing

from ..exceptions import ProxySchemeUnsupported

if typing.TYPE_CHECKING:
    from typing_extensions import Self

    from .ssl_ import _TYPE_PEER_CERT_RET, _TYPE_PEER_CERT_RET_DICT


_WriteBuffer = typing.Union[bytearray, memoryview]
_ReturnValue = typing.TypeVar("_ReturnValue")

SSL_BLOCKSIZE = 16384


class SSLTransport:
    """
    The SSLTransport wraps an existing socket and establishes an SSL connection.

    Contrary to Python's implementation of SSLSocket, it allows you to chain
    multiple TLS connections together. It's particularly useful if you need to
    implement TLS within TLS.

    The class supports most of the socket API operations.
    """

    @staticmethod
    def _validate_ssl_context_for_tls_in_tls(ssl_context: ssl.SSLContext) -> None:
        """
        Raises a ProxySchemeUnsupported if the provided ssl_context can't be used
        for TLS in TLS.

        The only requirement is that the ssl_context provides the 'wrap_bio'
        methods.
        """

        if not hasattr(ssl_context, "wrap_bio"):
            raise ProxySchemeUnsupported(
                "TLS in TLS requires SSLContext.wrap_bio() which isn't "
                "available on non-native SSLContext"
            )

    def __init__(
        self,
        socket: socket.socket,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        suppress_ragged_eofs: bool = True,
    ) -> None:
        """
        Create an SSLTransport around socket using the provided ssl_context.
        """
        self.incoming = ssl.MemoryBIO()
        self.outgoing = ssl.MemoryBIO()

        self.suppress_ragged_eofs = suppress_ragged_eofs
        self.socket = socket

        self.sslobj = ssl_context.wrap_bio(
            self.incoming, self.outgoing, server_hostname=server_hostname
        )

        # Perform initial handshake.
        self._ssl_io_loop(self.sslobj.do_handshake)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: typing.Any) -> None:
        self.close()

    def fileno(self) -> int:
        return self.socket.fileno()

    def read(self, len: int = 1024, buffer: typing.Any | None = None) -> int | bytes:
        return self._wrap_ssl_read(len, buffer)

    def recv(self, buflen: int = 1024, flags: int = 0) -> int | bytes:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to recv")
        return self._wrap_ssl_read(buflen)

    def recv_into(
        self,
        buffer: _WriteBuffer,
        nbytes: int | None = None,
        flags: int = 0,
    ) -> None | int | bytes:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to recv_into")
        if nbytes is None:
            nbytes = len(buffer)
        return self.read(nbytes, buffer)

    def sendall(self, data: bytes, flags: int = 0) -> None:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to sendall")
        count = 0
        with memoryview(data) as view, view.cast("B") as byte_view:
            amount = len(byte_view)
            while count < amount:
                v = self.send(byte_view[count:])
                count += v

    def send(self, data: bytes, flags: int = 0) -> int:
        if flags != 0:
            raise ValueError("non-zero flags not allowed in calls to send")
        return self._ssl_io_loop(self.sslobj.write, data)

    def makefile(
        self,
        mode: str,
        buffering: int | None = None,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> typing.BinaryIO | typing.TextIO | socket.SocketIO:
        """
        Python's httpclient uses makefile and buffered io when reading HTTP
        messages and we need to support it.

        This is unfortunately a copy and paste of socket.py makefile with small
        changes to point to the socket directly.
        """
        if not set(mode) <= {"r", "w", "b"}:
            raise ValueError(f"invalid mode {mode!r} (only r, w, b allowed)")

        writing = "w" in mode
        reading = "r" in mode or not writing
        assert reading or writing
        binary = "b" in mode
        rawmode = ""
        if reading:
            rawmode += "r"
        if writing:
            rawmode += "w"
        raw = socket.SocketIO(self, rawmode)  # type: ignore[arg-type]
        self.socket._io_refs += 1  # type: ignore[attr-defined]
        if buffering is None:
            buffering = -1
        if buffering < 0:
            buffering = io.DEFAULT_BUFFER_SIZE
        if buffering == 0:
            if not binary:
                raise ValueError("unbuffered streams must be binary")
            return raw
        buffer: typing.BinaryIO
        if reading and writing:
            buffer = io.BufferedRWPair(raw, raw, buffering)  # type: ignore[assignment]
        elif reading:
            buffer = io.BufferedReader(raw, buffering)
        else:
            assert writing
            buffer = io.BufferedWriter(raw, buffering)
        if binary:
            return buffer
        text = io.TextIOWrapper(buffer, encoding, errors, newline)
        text.mode = mode  # type: ignore[misc]
        return text

    def unwrap(self) -> None:
        self._ssl_io_loop(self.sslobj.unwrap)

    def close(self) -> None:
        self.socket.close()

    @typing.overload
    def getpeercert(
        self, binary_form: typing.Literal[False] = ...
    ) -> _TYPE_PEER_CERT_RET_DICT | None: ...

    @typing.overload
    def getpeercert(self, binary_form: typing.Literal[True]) -> bytes | None: ...

    def getpeercert(self, binary_form: bool = False) -> _TYPE_PEER_CERT_RET:
        return self.sslobj.getpeercert(binary_form)  # type: ignore[return-value]

    def version(self) -> str | None:
        return self.sslobj.version()

    def cipher(self) -> tuple[str, str, int] | None:
        return self.sslobj.cipher()

    def selected_alpn_protocol(self) -> str | None:
        return self.sslobj.selected_alpn_protocol()

    def shared_ciphers(self) -> list[tuple[str, str, int]] | None:
        return self.sslobj.shared_ciphers()

    def compression(self) -> str | None:
        return self.sslobj.compression()

    def settimeout(self, value: float | None) -> None:
        self.socket.settimeout(value)

    def gettimeout(self) -> float | None:
        return self.socket.gettimeout()

    def _decref_socketios(self) -> None:
        self.socket._decref_socketios()  # type: ignore[attr-defined]

    def _wrap_ssl_read(self, len: int, buffer: bytearray | None = None) -> int | bytes:
        try:
            return self._ssl_io_loop(self.sslobj.read, len, buffer)
        except ssl.SSLError as e:
            if e.errno == ssl.SSL_ERROR_EOF and self.suppress_ragged_eofs:
                return 0  # eof, return 0.
            else:
                raise

    # func is sslobj.do_handshake or sslobj.unwrap
    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[], None]) -> None: ...

    # func is sslobj.write, arg1 is data
    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[bytes], int], arg1: bytes) -> int: ...

    # func is sslobj.read, arg1 is len, arg2 is buffer
    @typing.overload
    def _ssl_io_loop(
        self,
        func: typing.Callable[[int, bytearray | None], bytes],
        arg1: int,
        arg2: bytearray | None,
    ) -> bytes: ...

    def _ssl_io_loop(
        self,
        func: typing.Callable[..., _ReturnValue],
        arg1: None | bytes | int = None,
        arg2: bytearray | None = None,
    ) -> _ReturnValue:
        """Performs an I/O loop between incoming/outgoing and the socket."""
        should_loop = True
        ret = None

        while should_loop:
            errno = None
            try:
                if arg1 is None and arg2 is None:
                    ret = func()
                elif arg2 is None:
                    ret = func(arg1)
                else:
                    ret = func(arg1, arg2)
            except ssl.SSLError as e:
                if e.errno not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE):
                    # WANT_READ, and WANT_WRITE are expected, others are not.
                    raise e
                errno = e.errno

            buf = self.outgoing.read()
            self.socket.sendall(buf)

            if errno is None:
                should_loop = False
            elif errno == ssl.SSL_ERROR_WANT_READ:
                buf = self.socket.recv(SSL_BLOCKSIZE)
                if buf:
                    self.incoming.write(buf)
                else:
                    self.incoming.write_eof()
        return typing.cast(_ReturnValue, ret)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import os
import pathlib
import sys

import torch
from image_decoder import export_decoder_onnx, test_decoder_onnx
from image_encoder import export_image_encoder_onnx, test_image_encoder_onnx
from mask_decoder import export_mask_decoder_onnx, test_mask_decoder_onnx
from prompt_encoder import export_prompt_encoder_onnx, test_prompt_encoder_onnx
from sam2_demo import run_demo, show_all_images
from sam2_utils import load_sam2_model, sam2_onnx_path, setup_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export SAM2 models to ONNX")

    parser.add_argument(
        "--model_type",
        required=False,
        type=str,
        choices=["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"],
        default="sam2_hiera_large",
        help="The model type to export",
    )

    parser.add_argument(
        "--components",
        required=False,
        nargs="+",
        choices=["image_encoder", "mask_decoder", "prompt_encoder", "image_decoder"],
        default=["image_encoder", "image_decoder"],
        help="Type of ONNX models to export. "
        "Note that image_decoder is a combination of prompt_encoder and mask_decoder",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory for the ONNX models",
        default="sam2_onnx_models",
    )

    parser.add_argument(
        "--dynamic_batch_axes",
        required=False,
        default=False,
        action="store_true",
        help="Export image_encoder with dynamic batch axes",
    )

    parser.add_argument(
        "--multimask_output",
        required=False,
        default=False,
        action="store_true",
        help="Export mask_decoder or image_decoder with multimask_output",
    )

    parser.add_argument(
        "--disable_dynamic_multimask_via_stability",
        required=False,
        action="store_true",
        help="Disable mask_decoder dynamic_multimask_via_stability, and output first mask only."
        "This option will be ignored when multimask_output is True",
    )

    parser.add_argument(
        "--sam2_dir",
        required=False,
        type=str,
        default="./segment-anything-2",
        help="The directory of segment-anything-2 git repository",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        default=False,
        action="store_true",
        help="Overwrite onnx model file if exists.",
    )

    parser.add_argument(
        "--demo",
        required=False,
        default=False,
        action="store_true",
        help="Run demo with the exported ONNX models.",
    )

    parser.add_argument(
        "--optimize",
        required=False,
        default=False,
        action="store_true",
        help="Optimize onnx models",
    )

    parser.add_argument(
        "--dtype", required=False, choices=["fp32", "fp16"], default="fp32", help="Data type for inference."
    )

    parser.add_argument(
        "--use_gpu",
        required=False,
        default=False,
        action="store_true",
        help="Optimize onnx models for GPU",
    )

    parser.add_argument(
        "--dynamo",
        required=False,
        default=False,
        action="store_true",
        help="Use dynamo for exporting onnx model. Only image_encoder supports dynamo right now.",
    )

    parser.add_argument(
        "--verbose",
        required=False,
        default=False,
        action="store_true",
        help="Print verbose information",
    )

    args = parser.parse_args()
    return args


def optimize_sam2_model(onnx_model_path, optimized_model_path, float16: bool, use_gpu: bool):
    print(f"Optimizing {onnx_model_path} to {optimized_model_path} with float16={float16} and use_gpu={use_gpu}...")

    # Import from source directory.
    transformers_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if transformers_dir not in sys.path:
        sys.path.insert(0, transformers_dir)
    from optimizer import optimize_model

    optimized_model = optimize_model(onnx_model_path, model_type="sam2", opt_level=1, use_gpu=use_gpu)
    if float16:
        optimized_model.convert_float_to_float16(keep_io_types=False)
    optimized_model.save_model_to_file(optimized_model_path)


def main():
    args = parse_arguments()

    sam2_model = load_sam2_model(args.sam2_dir, args.model_type, device="cpu")

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for component in args.components:
        onnx_model_path = sam2_onnx_path(args.output_dir, args.model_type, component, args.multimask_output)
        if component == "image_encoder":
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_image_encoder_onnx(
                    sam2_model, onnx_model_path, args.dynamic_batch_axes, args.verbose, args.dynamo
                )
                test_image_encoder_onnx(sam2_model, onnx_model_path, dynamic_batch_axes=args.dynamic_batch_axes)

        elif component == "mask_decoder":
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_mask_decoder_onnx(
                    sam2_model,
                    onnx_model_path,
                    args.multimask_output,
                    not args.disable_dynamic_multimask_via_stability,
                    args.verbose,
                )
                test_mask_decoder_onnx(
                    sam2_model,
                    onnx_model_path,
                    args.multimask_output,
                    not args.disable_dynamic_multimask_via_stability,
                )
        elif component == "prompt_encoder":
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_prompt_encoder_onnx(sam2_model, onnx_model_path)
                test_prompt_encoder_onnx(sam2_model, onnx_model_path)
        else:
            assert component == "image_decoder"
            if args.overwrite or not os.path.exists(onnx_model_path):
                export_decoder_onnx(sam2_model, onnx_model_path, args.multimask_output)
                test_decoder_onnx(sam2_model, onnx_model_path, args.multimask_output)

    suffix = ""
    convert_to_fp16 = args.dtype == "fp16"
    if args.optimize:
        suffix = f"_{args.dtype}_" + ("gpu" if args.use_gpu else "cpu")
        for component in args.components:
            onnx_model_path = sam2_onnx_path(args.output_dir, args.model_type, component, args.multimask_output)
            optimized_model_path = sam2_onnx_path(
                args.output_dir, args.model_type, component, args.multimask_output, suffix
            )
            optimize_sam2_model(onnx_model_path, optimized_model_path, convert_to_fp16, args.use_gpu)

    if args.demo:
        # Export required ONNX models for demo if not already exported.
        image_encoder_onnx_path = sam2_onnx_path(
            args.output_dir, args.model_type, "image_encoder", args.multimask_output
        )
        if not os.path.exists(image_encoder_onnx_path):
            export_image_encoder_onnx(sam2_model, image_encoder_onnx_path, args.dynamic_batch_axes, args.verbose)

        image_decoder_onnx_path = sam2_onnx_path(args.output_dir, args.model_type, "image_decoder", False)
        if not os.path.exists(image_decoder_onnx_path):
            export_decoder_onnx(sam2_model, image_decoder_onnx_path, False)

        image_decoder_multi_onnx_path = sam2_onnx_path(args.output_dir, args.model_type, "image_decoder", True)
        if not os.path.exists(image_decoder_multi_onnx_path):
            export_decoder_onnx(sam2_model, image_decoder_multi_onnx_path, True)

        dtype = torch.float32 if args.dtype == "fp32" else torch.float16
        if suffix:
            optimized_image_encoder_onnx_path = image_encoder_onnx_path.replace(".onnx", f"{suffix}.onnx")
            if not os.path.exists(optimized_image_encoder_onnx_path):
                optimize_sam2_model(
                    image_encoder_onnx_path, optimized_image_encoder_onnx_path, convert_to_fp16, args.use_gpu
                )

            optimized_image_decoder_onnx_path = image_decoder_onnx_path.replace(".onnx", f"{suffix}.onnx")
            if not os.path.exists(optimized_image_decoder_onnx_path):
                optimize_sam2_model(
                    image_decoder_onnx_path, optimized_image_decoder_onnx_path, convert_to_fp16, args.use_gpu
                )

            optimized_image_decoder_multi_onnx_path = image_decoder_multi_onnx_path.replace(".onnx", f"{suffix}.onnx")
            if not os.path.exists(optimized_image_decoder_multi_onnx_path):
                optimize_sam2_model(
                    image_decoder_multi_onnx_path,
                    optimized_image_decoder_multi_onnx_path,
                    convert_to_fp16,
                    args.use_gpu,
                )

            # Use optimized models to run demo.
            image_encoder_onnx_path = optimized_image_encoder_onnx_path
            image_decoder_onnx_path = optimized_image_decoder_onnx_path
            image_decoder_multi_onnx_path = optimized_image_decoder_multi_onnx_path

        ort_image_files = run_demo(
            args.sam2_dir,
            args.model_type,
            engine="ort",
            dtype=dtype,
            image_encoder_onnx_path=image_encoder_onnx_path,
            image_decoder_onnx_path=image_decoder_onnx_path,
            image_decoder_multi_onnx_path=image_decoder_multi_onnx_path,
            use_gpu=args.use_gpu,
        )
        print("demo output files for ONNX Runtime:", ort_image_files)

        # Get results from torch engine to compare.
        torch_image_files = run_demo(args.sam2_dir, args.model_type, engine="torch", dtype=dtype, use_gpu=args.use_gpu)
        print("demo output files for PyTorch:", torch_image_files)

        show_all_images(ort_image_files, torch_image_files, suffix)
        print(f"Combined demo output: sam2_demo{suffix}.png")


if __name__ == "__main__":
    setup_logger(verbose=False)
    with torch.no_grad():
        main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\header_footer.py ===
# Copyright (c) 2010-2024 openpyxl

# Simplified implementation of headers and footers: let worksheets have separate items

import re
from warnings import warn

from openpyxl.descriptors import (
    Alias,
    Bool,
    Strict,
    String,
    Integer,
    MatchPattern,
    Typed,
)
from openpyxl.descriptors.serialisable import Serialisable


from openpyxl.xml.functions import Element
from openpyxl.utils.escape import escape, unescape


FONT_PATTERN = '&"(?P<font>.+)"'
COLOR_PATTERN  = "&K(?P<color>[A-F0-9]{6})"
SIZE_REGEX = r"&(?P<size>\d+\s?)"
FORMAT_REGEX = re.compile("{0}|{1}|{2}".format(FONT_PATTERN, COLOR_PATTERN,
                                               SIZE_REGEX)
                          )

def _split_string(text):
    """
    Split the combined (decoded) string into left, center and right parts

    # See http://stackoverflow.com/questions/27711175/regex-with-multiple-optional-groups for discussion
    """

    ITEM_REGEX = re.compile("""
    (&L(?P<left>.+?))?
    (&C(?P<center>.+?))?
    (&R(?P<right>.+?))?
    $""", re.VERBOSE | re.DOTALL)

    m = ITEM_REGEX.match(text)
    try:
        parts = m.groupdict()
    except AttributeError:
        warn("""Cannot parse header or footer so it will be ignored""")
        parts = {'left':'', 'right':'', 'center':''}
    return parts


class _HeaderFooterPart(Strict):

    """
    Individual left/center/right header/footer part

    Do not use directly.

    Header & Footer ampersand codes:

    * &A   Inserts the worksheet name
    * &B   Toggles bold
    * &D or &[Date]   Inserts the current date
    * &E   Toggles double-underline
    * &F or &[File]   Inserts the workbook name
    * &I   Toggles italic
    * &N or &[Pages]   Inserts the total page count
    * &S   Toggles strikethrough
    * &T   Inserts the current time
    * &[Tab]   Inserts the worksheet name
    * &U   Toggles underline
    * &X   Toggles superscript
    * &Y   Toggles subscript
    * &P or &[Page]   Inserts the current page number
    * &P+n   Inserts the page number incremented by n
    * &P-n   Inserts the page number decremented by n
    * &[Path]   Inserts the workbook path
    * &&   Escapes the ampersand character
    * &"fontname"   Selects the named font
    * &nn   Selects the specified 2-digit font point size

    Colours are in RGB Hex
    """

    text = String(allow_none=True)
    font = String(allow_none=True)
    size = Integer(allow_none=True)
    RGB = ("^[A-Fa-f0-9]{6}$")
    color = MatchPattern(allow_none=True, pattern=RGB)


    def __init__(self, text=None, font=None, size=None, color=None):
        self.text = text
        self.font = font
        self.size = size
        self.color = color


    def __str__(self):
        """
        Convert to Excel HeaderFooter miniformat minus position
        """
        fmt = []
        if self.font:
            fmt.append(u'&"{0}"'.format(self.font))
        if self.size:
            fmt.append("&{0} ".format(self.size))
        if self.color:
            fmt.append("&K{0}".format(self.color))
        return u"".join(fmt + [self.text])

    def __bool__(self):
        return bool(self.text)



    @classmethod
    def from_str(cls, text):
        """
        Convert from miniformat to object
        """
        keys = ('font', 'color', 'size')
        kw = dict((k, v) for match in FORMAT_REGEX.findall(text)
                  for k, v in zip(keys, match) if v)

        kw['text'] = FORMAT_REGEX.sub('', text)

        return cls(**kw)


class HeaderFooterItem(Strict):
    """
    Header or footer item

    """

    left = Typed(expected_type=_HeaderFooterPart)
    center = Typed(expected_type=_HeaderFooterPart)
    centre = Alias("center")
    right = Typed(expected_type=_HeaderFooterPart)

    __keys = ('L', 'C', 'R')


    def __init__(self, left=None, right=None, center=None):
        if left is None:
            left = _HeaderFooterPart()
        self.left = left
        if center is None:
            center = _HeaderFooterPart()
        self.center = center
        if right is None:
            right = _HeaderFooterPart()
        self.right = right


    def __str__(self):
        """
        Pack parts into a single string
        """
        TRANSFORM = {'&[Tab]': '&A', '&[Pages]': '&N', '&[Date]': '&D',
                     '&[Path]': '&Z', '&[Page]': '&P', '&[Time]': '&T', '&[File]': '&F',
                     '&[Picture]': '&G'}

        # escape keys and create regex
        SUBS_REGEX = re.compile("|".join(["({0})".format(re.escape(k))
                                          for k in TRANSFORM]))

        def replace(match):
            """
            Callback for re.sub
            Replace expanded control with mini-format equivalent
            """
            sub = match.group(0)
            return TRANSFORM[sub]

        txt = []
        for key, part in zip(
            self.__keys, [self.left, self.center, self.right]):
            if part.text is not None:
                txt.append(u"&{0}{1}".format(key, str(part)))
        txt = "".join(txt)
        txt = SUBS_REGEX.sub(replace, txt)
        return escape(txt)


    def __bool__(self):
        return any([self.left, self.center, self.right])



    def to_tree(self, tagname):
        """
        Return as XML node
        """
        el = Element(tagname)
        el.text = str(self)
        return el


    @classmethod
    def from_tree(cls, node):
        if node.text:
            text = unescape(node.text)
            parts = _split_string(text)
            for k, v in parts.items():
                if v is not None:
                    parts[k] = _HeaderFooterPart.from_str(v)
            self = cls(**parts)
            return self


class HeaderFooter(Serialisable):

    tagname = "headerFooter"

    differentOddEven = Bool(allow_none=True)
    differentFirst = Bool(allow_none=True)
    scaleWithDoc = Bool(allow_none=True)
    alignWithMargins = Bool(allow_none=True)
    oddHeader = Typed(expected_type=HeaderFooterItem, allow_none=True)
    oddFooter = Typed(expected_type=HeaderFooterItem, allow_none=True)
    evenHeader = Typed(expected_type=HeaderFooterItem, allow_none=True)
    evenFooter = Typed(expected_type=HeaderFooterItem, allow_none=True)
    firstHeader = Typed(expected_type=HeaderFooterItem, allow_none=True)
    firstFooter = Typed(expected_type=HeaderFooterItem, allow_none=True)

    __elements__ = ("oddHeader", "oddFooter", "evenHeader", "evenFooter", "firstHeader", "firstFooter")

    def __init__(self,
                 differentOddEven=None,
                 differentFirst=None,
                 scaleWithDoc=None,
                 alignWithMargins=None,
                 oddHeader=None,
                 oddFooter=None,
                 evenHeader=None,
                 evenFooter=None,
                 firstHeader=None,
                 firstFooter=None,
                ):
        self.differentOddEven = differentOddEven
        self.differentFirst = differentFirst
        self.scaleWithDoc = scaleWithDoc
        self.alignWithMargins = alignWithMargins
        if oddHeader is None:
            oddHeader = HeaderFooterItem()
        self.oddHeader = oddHeader
        if oddFooter is None:
            oddFooter = HeaderFooterItem()
        self.oddFooter = oddFooter
        if evenHeader is None:
            evenHeader = HeaderFooterItem()
        self.evenHeader = evenHeader
        if evenFooter is None:
            evenFooter = HeaderFooterItem()
        self.evenFooter = evenFooter
        if firstHeader is None:
            firstHeader = HeaderFooterItem()
        self.firstHeader = firstHeader
        if firstFooter is None:
            firstFooter = HeaderFooterItem()
        self.firstFooter = firstFooter


    def __bool__(self):
        parts = [getattr(self, attr) for attr in self.__attrs__ + self.__elements__]
        return any(parts)


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\facts.py ===
"""
Known facts in assumptions module.

This module defines the facts between unary predicates in ``get_known_facts()``,
and supports functions to generate the contents in
``sympy.assumptions.ask_generated`` file.
"""

from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
    Exclusive,)
from sympy.logic.inference import satisfiable


@cacheit
def get_composite_predicates():
    # To reduce the complexity of sat solver, these predicates are
    # transformed into the combination of primitive predicates.
    return {
        Q.real : Q.negative | Q.zero | Q.positive,
        Q.integer : Q.even | Q.odd,
        Q.nonpositive : Q.negative | Q.zero,
        Q.nonzero : Q.negative | Q.positive,
        Q.nonnegative : Q.zero | Q.positive,
        Q.extended_real : Q.negative_infinite | Q.negative | Q.zero | Q.positive | Q.positive_infinite,
        Q.extended_positive: Q.positive | Q.positive_infinite,
        Q.extended_negative: Q.negative | Q.negative_infinite,
        Q.extended_nonzero: Q.negative_infinite | Q.negative | Q.positive | Q.positive_infinite,
        Q.extended_nonpositive: Q.negative_infinite | Q.negative | Q.zero,
        Q.extended_nonnegative: Q.zero | Q.positive | Q.positive_infinite,
        Q.complex : Q.algebraic | Q.transcendental
    }


@cacheit
def get_known_facts(x=None):
    """
    Facts between unary predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        get_number_facts(x),
        get_matrix_facts(x)
    )
    return fact


@cacheit
def get_number_facts(x = None):
    """
    Facts between unary number predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        # primitive predicates for extended real exclude each other.
        Exclusive(Q.negative_infinite(x), Q.negative(x), Q.zero(x),
            Q.positive(x), Q.positive_infinite(x)),

        # build complex plane
        Exclusive(Q.real(x), Q.imaginary(x)),
        Implies(Q.real(x) | Q.imaginary(x), Q.complex(x)),

        # other subsets of complex
        Exclusive(Q.transcendental(x), Q.algebraic(x)),
        Equivalent(Q.real(x), Q.rational(x) | Q.irrational(x)),
        Exclusive(Q.irrational(x), Q.rational(x)),
        Implies(Q.rational(x), Q.algebraic(x)),

        # integers
        Exclusive(Q.even(x), Q.odd(x)),
        Implies(Q.integer(x), Q.rational(x)),
        Implies(Q.zero(x), Q.even(x)),
        Exclusive(Q.composite(x), Q.prime(x)),
        Implies(Q.composite(x) | Q.prime(x), Q.integer(x) & Q.positive(x)),
        Implies(Q.even(x) & Q.positive(x) & ~Q.prime(x), Q.composite(x)),

        # hermitian and antihermitian
        Implies(Q.real(x), Q.hermitian(x)),
        Implies(Q.imaginary(x), Q.antihermitian(x)),
        Implies(Q.zero(x), Q.hermitian(x) | Q.antihermitian(x)),

        # define finity and infinity, and build extended real line
        Exclusive(Q.infinite(x), Q.finite(x)),
        Implies(Q.complex(x), Q.finite(x)),
        Implies(Q.negative_infinite(x) | Q.positive_infinite(x), Q.infinite(x)),

        # commutativity
        Implies(Q.finite(x) | Q.infinite(x), Q.commutative(x)),
    )
    return fact


@cacheit
def get_matrix_facts(x = None):
    """
    Facts between unary matrix predicates.

    Parameters
    ==========

    x : Symbol, optional
        Placeholder symbol for unary facts. Default is ``Symbol('x')``.

    Returns
    =======

    fact : Known facts in conjugated normal form.

    """
    if x is None:
        x = Symbol('x')

    fact = And(
        # matrices
        Implies(Q.orthogonal(x), Q.positive_definite(x)),
        Implies(Q.orthogonal(x), Q.unitary(x)),
        Implies(Q.unitary(x) & Q.real_elements(x), Q.orthogonal(x)),
        Implies(Q.unitary(x), Q.normal(x)),
        Implies(Q.unitary(x), Q.invertible(x)),
        Implies(Q.normal(x), Q.square(x)),
        Implies(Q.diagonal(x), Q.normal(x)),
        Implies(Q.positive_definite(x), Q.invertible(x)),
        Implies(Q.diagonal(x), Q.upper_triangular(x)),
        Implies(Q.diagonal(x), Q.lower_triangular(x)),
        Implies(Q.lower_triangular(x), Q.triangular(x)),
        Implies(Q.upper_triangular(x), Q.triangular(x)),
        Implies(Q.triangular(x), Q.upper_triangular(x) | Q.lower_triangular(x)),
        Implies(Q.upper_triangular(x) & Q.lower_triangular(x), Q.diagonal(x)),
        Implies(Q.diagonal(x), Q.symmetric(x)),
        Implies(Q.unit_triangular(x), Q.triangular(x)),
        Implies(Q.invertible(x), Q.fullrank(x)),
        Implies(Q.invertible(x), Q.square(x)),
        Implies(Q.symmetric(x), Q.square(x)),
        Implies(Q.fullrank(x) & Q.square(x), Q.invertible(x)),
        Equivalent(Q.invertible(x), ~Q.singular(x)),
        Implies(Q.integer_elements(x), Q.real_elements(x)),
        Implies(Q.real_elements(x), Q.complex_elements(x)),
    )
    return fact



def generate_known_facts_dict(keys, fact):
    """
    Computes and returns a dictionary which contains the relations between
    unary predicates.

    Each key is a predicate, and item is two groups of predicates.
    First group contains the predicates which are implied by the key, and
    second group contains the predicates which are rejected by the key.

    All predicates in *keys* and *fact* must be unary and have same placeholder
    symbol.

    Parameters
    ==========

    keys : list of AppliedPredicate instances.

    fact : Fact between predicates in conjugated normal form.

    Examples
    ========

    >>> from sympy import Q, And, Implies
    >>> from sympy.assumptions.facts import generate_known_facts_dict
    >>> from sympy.abc import x
    >>> keys = [Q.even(x), Q.odd(x), Q.zero(x)]
    >>> fact = And(Implies(Q.even(x), ~Q.odd(x)),
    ...     Implies(Q.zero(x), Q.even(x)))
    >>> generate_known_facts_dict(keys, fact)
    {Q.even: ({Q.even}, {Q.odd}),
     Q.odd: ({Q.odd}, {Q.even, Q.zero}),
     Q.zero: ({Q.even, Q.zero}, {Q.odd})}
    """
    fact_cnf = to_cnf(fact)
    mapping = single_fact_lookup(keys, fact_cnf)

    ret = {}
    for key, value in mapping.items():
        implied = set()
        rejected = set()
        for expr in value:
            if isinstance(expr, AppliedPredicate):
                implied.add(expr.function)
            elif isinstance(expr, Not):
                pred = expr.args[0]
                rejected.add(pred.function)
        ret[key.function] = (implied, rejected)
    return ret


@cacheit
def get_known_facts_keys():
    """
    Return every unary predicates registered to ``Q``.

    This function is used to generate the keys for
    ``generate_known_facts_dict``.

    """
    # exclude polyadic predicates
    exclude = {Q.eq, Q.ne, Q.gt, Q.lt, Q.ge, Q.le}

    result = []
    for attr in Q.__class__.__dict__:
        if attr.startswith('__'):
            continue
        pred = getattr(Q, attr)
        if pred in exclude:
            continue
        result.append(pred)
    return result


def single_fact_lookup(known_facts_keys, known_facts_cnf):
    # Return the dictionary for quick lookup of single fact
    mapping = {}
    for key in known_facts_keys:
        mapping[key] = {key}
        for other_key in known_facts_keys:
            if other_key != key:
                if ask_full_inference(other_key, key, known_facts_cnf):
                    mapping[key].add(other_key)
                if ask_full_inference(~other_key, key, known_facts_cnf):
                    mapping[key].add(~other_key)
    return mapping


def ask_full_inference(proposition, assumptions, known_facts_cnf):
    """
    Method for inferring properties about objects.

    """
    if not satisfiable(And(known_facts_cnf, assumptions, proposition)):
        return False
    if not satisfiable(And(known_facts_cnf, assumptions, Not(proposition))):
        return True
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\f90mod_rules.py ===
"""
Build F90 module support for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.27 $"[10:-1]

f2py_version = 'See `f2py -v`'

import numpy as np

from . import capi_maps, func2subr

# The environment provided by auxfuncs.py is needed for some calls to eval.
# As the needed functions cannot be determined by static inspection of the
# code, it is safest to use import * pending a major refactoring of f2py.
from .auxfuncs import *
from .crackfortran import undo_rmbadname, undo_rmbadname1

options = {}


def findf90modules(m):
    if ismodule(m):
        return [m]
    if not hasbody(m):
        return []
    ret = []
    for b in m['body']:
        if ismodule(b):
            ret.append(b)
        else:
            ret = ret + findf90modules(b)
    return ret


fgetdims1 = """\
      external f2pysetdata
      logical ns
      integer r,i
      integer(%d) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then""" % np.intp().itemsize

fgetdims2 = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))"""

fgetdims2_sa = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
         !s(r) must be equal to len(d(1))
      end if
      flag = 2
      call f2pysetdata(d,allocated(d))"""


def buildhooks(pymod):
    from . import rules
    ret = {'f90modhooks': [], 'initf90modhooks': [], 'body': [],
           'need': ['F_FUNC', 'arrayobject.h'],
           'separatorsfor': {'includes0': '\n', 'includes': '\n'},
           'docs': ['"Fortran 90/95 modules:\\n"'],
           'latexdoc': []}
    fhooks = ['']

    def fadd(line, s=fhooks):
        s[0] = f'{s[0]}\n      {line}'
    doc = ['']

    def dadd(line, s=doc):
        s[0] = f'{s[0]}\n{line}'

    usenames = getuseblocks(pymod)
    for m in findf90modules(pymod):
        sargs, fargs, efargs, modobjs, notvars, onlyvars = [], [], [], [], [
            m['name']], []
        sargsp = []
        ifargs = []
        mfargs = []
        if hasbody(m):
            for b in m['body']:
                notvars.append(b['name'])
        for n in m['vars'].keys():
            var = m['vars'][n]

            if (n not in notvars and isvariable(var)) and (not l_or(isintent_hide, isprivate)(var)):
                onlyvars.append(n)
                mfargs.append(n)
        outmess(f"\t\tConstructing F90 module support for \"{m['name']}\"...\n")
        if len(onlyvars) == 0 and len(notvars) == 1 and m['name'] in notvars:
            outmess(f"\t\t\tSkipping {m['name']} since there are no public vars/func in this module...\n")
            continue

        # gh-25186
        if m['name'] in usenames and containscommon(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a common block...\n")
            continue
        # skip modules with derived types
        if m['name'] in usenames and containsderivedtypes(m):
            outmess(f"\t\t\tSkipping {m['name']} since it is in 'use' and contains a derived type...\n")
            continue
        if onlyvars:
            outmess(f"\t\t  Variables: {' '.join(onlyvars)}\n")
        chooks = ['']

        def cadd(line, s=chooks):
            s[0] = f'{s[0]}\n{line}'
        ihooks = ['']

        def iadd(line, s=ihooks):
            s[0] = f'{s[0]}\n{line}'

        vrd = capi_maps.modsign2map(m)
        cadd('static FortranDataDef f2py_%s_def[] = {' % (m['name']))
        dadd('\\subsection{Fortran 90/95 module \\texttt{%s}}\n' % (m['name']))
        if hasnote(m):
            note = m['note']
            if isinstance(note, list):
                note = '\n'.join(note)
            dadd(note)
        if onlyvars:
            dadd('\\begin{description}')
        for n in onlyvars:
            var = m['vars'][n]
            modobjs.append(n)
            ct = capi_maps.getctype(var)
            at = capi_maps.c2capi_map[ct]
            dm = capi_maps.getarrdims(n, var)
            dms = dm['dims'].replace('*', '-1').strip()
            dms = dms.replace(':', '-1').strip()
            if not dms:
                dms = '-1'
            use_fgetdims2 = fgetdims2
            cadd('\t{"%s",%s,{{%s}},%s, %s},' %
                 (undo_rmbadname1(n), dm['rank'], dms, at,
                  capi_maps.get_elsize(var)))
            dadd('\\item[]{{}\\verb@%s@{}}' %
                 (capi_maps.getarrdocsign(n, var)))
            if hasnote(var):
                note = var['note']
                if isinstance(note, list):
                    note = '\n'.join(note)
                dadd(f'--- {note}')
            if isallocatable(var):
                fargs.append(f"f2py_{m['name']}_getdims_{n}")
                efargs.append(fargs[-1])
                sargs.append(
                    f'void (*{n})(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                sargsp.append('void (*)(int*,npy_intp*,void(*)(char*,npy_intp*),int*)')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].func = {n};")
                fadd(f'subroutine {fargs[-1]}(r,s,f2pysetdata,flag)')
                fadd(f"use {m['name']}, only: d => {undo_rmbadname1(n)}\n")
                fadd('integer flag\n')
                fhooks[0] = fhooks[0] + fgetdims1
                dms = range(1, int(dm['rank']) + 1)
                fadd(' allocate(d(%s))\n' %
                     (','.join(['s(%s)' % i for i in dms])))
                fhooks[0] = fhooks[0] + use_fgetdims2
                fadd(f'end subroutine {fargs[-1]}')
            else:
                fargs.append(n)
                sargs.append(f'char *{n}')
                sargsp.append('char*')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].data = {n};")
        if onlyvars:
            dadd('\\end{description}')
        if hasbody(m):
            for b in m['body']:
                if not isroutine(b):
                    outmess("f90mod_rules.buildhooks:"
                            f" skipping {b['block']} {b['name']}\n")
                    continue
                modobjs.append(f"{b['name']}()")
                b['modulename'] = m['name']
                api, wrap = rules.buildapi(b)
                if isfunction(b):
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append(f"f2pywrap_{m['name']}_{b['name']}")
                    ifargs.append(func2subr.createfuncwrapper(b, signature=1))
                elif wrap:
                    fhooks[0] = fhooks[0] + wrap
                    fargs.append(f"f2pywrap_{m['name']}_{b['name']}")
                    ifargs.append(
                        func2subr.createsubrwrapper(b, signature=1))
                else:
                    fargs.append(b['name'])
                    mfargs.append(fargs[-1])
                api['externroutines'] = []
                ar = applyrules(api, vrd)
                ar['docs'] = []
                ar['docshort'] = []
                ret = dictappend(ret, ar)
                cadd(('\t{"%s",-1,{{-1}},0,0,NULL,(void *)'
                      'f2py_rout_#modulename#_%s_%s,'
                      'doc_f2py_rout_#modulename#_%s_%s},')
                     % (b['name'], m['name'], b['name'], m['name'], b['name']))
                sargs.append(f"char *{b['name']}")
                sargsp.append('char *')
                iadd(f"\tf2py_{m['name']}_def[i_f2py++].data = {b['name']};")
        cadd('\t{NULL}\n};\n')
        iadd('}')
        ihooks[0] = 'static void f2py_setup_%s(%s) {\n\tint i_f2py=0;%s' % (
            m['name'], ','.join(sargs), ihooks[0])
        if '_' in m['name']:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        iadd('extern void %s(f2pyinit%s,F2PYINIT%s)(void (*)(%s));'
             % (F_FUNC, m['name'], m['name'].upper(), ','.join(sargsp)))
        iadd('static void f2py_init_%s(void) {' % (m['name']))
        iadd('\t%s(f2pyinit%s,F2PYINIT%s)(f2py_setup_%s);'
             % (F_FUNC, m['name'], m['name'].upper(), m['name']))
        iadd('}\n')
        ret['f90modhooks'] = ret['f90modhooks'] + chooks + ihooks
        ret['initf90modhooks'] = ['\tPyDict_SetItemString(d, "%s", PyFortranObject_New(f2py_%s_def,f2py_init_%s));' % (
            m['name'], m['name'], m['name'])] + ret['initf90modhooks']
        fadd('')
        fadd(f"subroutine f2pyinit{m['name']}(f2pysetupfunc)")
        if mfargs:
            for a in undo_rmbadname(mfargs):
                fadd(f"use {m['name']}, only : {a}")
        if ifargs:
            fadd(' '.join(['interface'] + ifargs))
            fadd('end interface')
        fadd('external f2pysetupfunc')
        if efargs:
            for a in undo_rmbadname(efargs):
                fadd(f'external {a}')
        fadd(f"call f2pysetupfunc({','.join(undo_rmbadname(fargs))})")
        fadd(f"end subroutine f2pyinit{m['name']}\n")

        dadd('\n'.join(ret['latexdoc']).replace(
            r'\subsection{', r'\subsubsection{'))

        ret['latexdoc'] = []
        ret['docs'].append(f"\"\t{m['name']} --- {','.join(undo_rmbadname(modobjs))}\"")

    ret['routine_defs'] = ''
    ret['doc'] = []
    ret['docshort'] = []
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fhooks[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\methods\selectn.py ===
"""
Implementation of nlargest and nsmallest.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    cast,
    final,
)

import numpy as np

from pandas._libs import algos as libalgos

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import BaseMaskedDtype

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        IndexLabel,
    )

    from pandas import (
        DataFrame,
        Series,
    )


class SelectN:
    def __init__(self, obj, n: int, keep: str) -> None:
        self.obj = obj
        self.n = n
        self.keep = keep

        if self.keep not in ("first", "last", "all"):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> DataFrame | Series:
        raise NotImplementedError

    @final
    def nlargest(self):
        return self.compute("nlargest")

    @final
    def nsmallest(self):
        return self.compute("nsmallest")

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        if is_numeric_dtype(dtype):
            return not is_complex_dtype(dtype)
        return needs_i8_conversion(dtype)


class SelectNSeries(SelectN):
    """
    Implement n largest/smallest for Series

    Parameters
    ----------
    obj : Series
    n : int
    keep : {'first', 'last'}, default 'first'

    Returns
    -------
    nordered : Series
    """

    def compute(self, method: str) -> Series:
        from pandas.core.reshape.concat import concat

        n = self.n
        dtype = self.obj.dtype
        if not self.is_valid_dtype_n_method(dtype):
            raise TypeError(f"Cannot use method '{method}' with dtype {dtype}")

        if n <= 0:
            return self.obj[[]]

        dropped = self.obj.dropna()
        nan_index = self.obj.drop(dropped.index)

        # slow method
        if n >= len(self.obj):
            ascending = method == "nsmallest"
            return self.obj.sort_values(ascending=ascending).head(n)

        # fast method
        new_dtype = dropped.dtype

        # Similar to algorithms._ensure_data
        arr = dropped._values
        if needs_i8_conversion(arr.dtype):
            arr = arr.view("i8")
        elif isinstance(arr.dtype, BaseMaskedDtype):
            arr = arr._data
        else:
            arr = np.asarray(arr)
        if arr.dtype.kind == "b":
            arr = arr.view(np.uint8)

        if method == "nlargest":
            arr = -arr
            if is_integer_dtype(new_dtype):
                # GH 21426: ensure reverse ordering at boundaries
                arr -= 1

            elif is_bool_dtype(new_dtype):
                # GH 26154: ensure False is smaller than True
                arr = 1 - (-arr)

        if self.keep == "last":
            arr = arr[::-1]

        nbase = n
        narr = len(arr)
        n = min(n, narr)

        # arr passed into kth_smallest must be contiguous. We copy
        # here because kth_smallest will modify its input
        # avoid OOB access with kth_smallest_c when n <= 0
        if len(arr) > 0:
            kth_val = libalgos.kth_smallest(arr.copy(order="C"), n - 1)
        else:
            kth_val = np.nan
        (ns,) = np.nonzero(arr <= kth_val)
        inds = ns[arr[ns].argsort(kind="mergesort")]

        if self.keep != "all":
            inds = inds[:n]
            findex = nbase
        else:
            if len(inds) < nbase <= len(nan_index) + len(inds):
                findex = len(nan_index) + len(inds)
            else:
                findex = len(inds)

        if self.keep == "last":
            # reverse indices
            inds = narr - 1 - inds

        return concat([dropped.iloc[inds], nan_index]).iloc[:findex]


class SelectNFrame(SelectN):
    """
    Implement n largest/smallest for DataFrame

    Parameters
    ----------
    obj : DataFrame
    n : int
    keep : {'first', 'last'}, default 'first'
    columns : list or str

    Returns
    -------
    nordered : DataFrame
    """

    def __init__(self, obj: DataFrame, n: int, keep: str, columns: IndexLabel) -> None:
        super().__init__(obj, n, keep)
        if not is_list_like(columns) or isinstance(columns, tuple):
            columns = [columns]

        columns = cast(Sequence[Hashable], columns)
        columns = list(columns)
        self.columns = columns

    def compute(self, method: str) -> DataFrame:
        from pandas.core.api import Index

        n = self.n
        frame = self.obj
        columns = self.columns

        for column in columns:
            dtype = frame[column].dtype
            if not self.is_valid_dtype_n_method(dtype):
                raise TypeError(
                    f"Column {repr(column)} has dtype {dtype}, "
                    f"cannot use method {repr(method)} with this dtype"
                )

        def get_indexer(current_indexer, other_indexer):
            """
            Helper function to concat `current_indexer` and `other_indexer`
            depending on `method`
            """
            if method == "nsmallest":
                return current_indexer.append(other_indexer)
            else:
                return other_indexer.append(current_indexer)

        # Below we save and reset the index in case index contains duplicates
        original_index = frame.index
        cur_frame = frame = frame.reset_index(drop=True)
        cur_n = n
        indexer = Index([], dtype=np.int64)

        for i, column in enumerate(columns):
            # For each column we apply method to cur_frame[column].
            # If it's the last column or if we have the number of
            # results desired we are done.
            # Otherwise there are duplicates of the largest/smallest
            # value and we need to look at the rest of the columns
            # to determine which of the rows with the largest/smallest
            # value in the column to keep.
            series = cur_frame[column]
            is_last_column = len(columns) - 1 == i
            values = getattr(series, method)(
                cur_n, keep=self.keep if is_last_column else "all"
            )

            if is_last_column or len(values) <= cur_n:
                indexer = get_indexer(indexer, values.index)
                break

            # Now find all values which are equal to
            # the (nsmallest: largest)/(nlargest: smallest)
            # from our series.
            border_value = values == values[values.index[-1]]

            # Some of these values are among the top-n
            # some aren't.
            unsafe_values = values[border_value]

            # These values are definitely among the top-n
            safe_values = values[~border_value]
            indexer = get_indexer(indexer, safe_values.index)

            # Go on and separate the unsafe_values on the remaining
            # columns.
            cur_frame = cur_frame.loc[unsafe_values.index]
            cur_n = n - len(indexer)

        frame = frame.take(indexer)

        # Restore the index on frame
        frame.index = original_index.take(indexer)

        # If there is only one column, the frame is already sorted.
        if len(columns) == 1:
            return frame

        ascending = method == "nsmallest"

        return frame.sort_values(columns, ascending=ascending, kind="mergesort")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\mathieu_functions.py ===
""" This module contains the Mathieu functions.
"""

from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos


class MathieuBase(DefinedFunction):
    """
    Abstract base class for Mathieu functions.

    This class is meant to reduce code duplication.

    """

    unbranched = True

    def _eval_conjugate(self):
        a, q, z = self.args
        return self.func(a.conjugate(), q.conjugate(), z.conjugate())


class mathieus(MathieuBase):
    r"""
    The Mathieu Sine function $S(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieus
    >>> from sympy.abc import a, q, z

    >>> mathieus(a, q, z)
    mathieus(a, q, z)

    >>> mathieus(a, 0, z)
    sin(sqrt(a)*z)

    >>> diff(mathieus(a, q, z), z)
    mathieusprime(a, q, z)

    See Also
    ========

    mathieuc: Mathieu cosine function.
    mathieusprime: Derivative of Mathieu sine function.
    mathieucprime: Derivative of Mathieu cosine function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuS/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieusprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sin(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)


class mathieuc(MathieuBase):
    r"""
    The Mathieu Cosine function $C(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieuc
    >>> from sympy.abc import a, q, z

    >>> mathieuc(a, q, z)
    mathieuc(a, q, z)

    >>> mathieuc(a, 0, z)
    cos(sqrt(a)*z)

    >>> diff(mathieuc(a, q, z), z)
    mathieucprime(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieusprime: Derivative of Mathieu sine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuC/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieucprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return cos(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return cls(a, q, -z)


class mathieusprime(MathieuBase):
    r"""
    The derivative $S^{\prime}(a,q,z)$ of the Mathieu Sine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieusprime
    >>> from sympy.abc import a, q, z

    >>> mathieusprime(a, q, z)
    mathieusprime(a, q, z)

    >>> mathieusprime(a, 0, z)
    sqrt(a)*cos(sqrt(a)*z)

    >>> diff(mathieusprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieus(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuSPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2*q*cos(2*z) - a)*mathieus(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sqrt(a)*cos(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return cls(a, q, -z)


class mathieucprime(MathieuBase):
    r"""
    The derivative $C^{\prime}(a,q,z)$ of the Mathieu Cosine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieucprime
    >>> from sympy.abc import a, q, z

    >>> mathieucprime(a, q, z)
    mathieucprime(a, q, z)

    >>> mathieucprime(a, 0, z)
    -sqrt(a)*sin(sqrt(a)*z)

    >>> diff(mathieucprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieuc(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieusprime: Derivative of Mathieu sine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuCPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2*q*cos(2*z) - a)*mathieuc(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return -sqrt(a)*sin(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\appellseqs.py ===
r"""
Efficient functions for generating Appell sequences.

An Appell sequence is a zero-indexed sequence of polynomials `p_i(x)`
satisfying `p_{i+1}'(x)=(i+1)p_i(x)` for all `i`. This definition leads
to the following iterative algorithm:

.. math :: p_0(x) = c_0,\ p_i(x) = i \int_0^x p_{i-1}(t)\,dt + c_i

The constant coefficients `c_i` are usually determined from the
just-evaluated integral and `i`.

Appell sequences satisfy the following identity from umbral calculus:

.. math :: p_n(x+y) = \sum_{k=0}^n \binom{n}{k} p_k(x) y^{n-k}

References
==========

.. [1] https://en.wikipedia.org/wiki/Appell_sequence
.. [2] Peter Luschny, "An introduction to the Bernoulli function",
       https://arxiv.org/abs/2009.06743
"""
from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

def dup_bernoulli(n, K):
    """Low-level implementation of Bernoulli polynomials."""
    if n < 1:
        return [K.one]
    p = [K.one, K(-1,2)]
    for i in range(2, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K(1,2), K) * K(1<<(i-1), (1<<i)-1), K)
    return p

@public
def bernoulli_poly(n, x=None, polys=False):
    r"""Generates the Bernoulli polynomial `\operatorname{B}_n(x)`.

    `\operatorname{B}_n(x)` is the unique polynomial satisfying

    .. math :: \int_{x}^{x+1} \operatorname{B}_n(t) \,dt = x^n.

    Based on this, we have for nonnegative integer `s` and integer
    `a` and `b`

    .. math :: \sum_{k=a}^{b} k^s = \frac{\operatorname{B}_{s+1}(b+1) -
            \operatorname{B}_{s+1}(a)}{s+1}

    which is related to Jakob Bernoulli's original motivation for introducing
    the Bernoulli numbers, the values of these polynomials at `x = 1`.

    Examples
    ========

    >>> from sympy import summation
    >>> from sympy.abc import x
    >>> from sympy.polys import bernoulli_poly
    >>> bernoulli_poly(5, x)
    x**5 - 5*x**4/2 + 5*x**3/3 - x/6

    >>> def psum(p, a, b):
    ...     return (bernoulli_poly(p+1,b+1) - bernoulli_poly(p+1,a)) / (p+1)
    >>> psum(4, -6, 27)
    3144337
    >>> summation(x**4, (x, -6, 27))
    3144337

    >>> psum(1, 1, x).factor()
    x*(x + 1)/2
    >>> psum(2, 1, x).factor()
    x*(x + 1)*(2*x + 1)/6
    >>> psum(3, 1, x).factor()
    x**2*(x + 1)**2/4

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.bernoulli

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_polynomials
    """
    return named_poly(n, dup_bernoulli, QQ, "Bernoulli polynomial", (x,), polys)

def dup_bernoulli_c(n, K):
    """Low-level implementation of central Bernoulli polynomials."""
    p = [K.one]
    for i in range(1, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K) * K((1<<(i-1))-1, (1<<i)-1), K)
    return p

@public
def bernoulli_c_poly(n, x=None, polys=False):
    r"""Generates the central Bernoulli polynomial `\operatorname{B}_n^c(x)`.

    These are scaled and shifted versions of the plain Bernoulli polynomials,
    done in such a way that `\operatorname{B}_n^c(x)` is an even or odd function
    for even or odd `n` respectively:

    .. math :: \operatorname{B}_n^c(x) = 2^n \operatorname{B}_n
            \left(\frac{x+1}{2}\right)

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_bernoulli_c, QQ, "central Bernoulli polynomial", (x,), polys)

def dup_genocchi(n, K):
    """Low-level implementation of Genocchi polynomials."""
    if n < 1:
        return [K.zero]
    p = [-K.one]
    for i in range(2, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K) // K(2), K)
    return p

@public
def genocchi_poly(n, x=None, polys=False):
    r"""Generates the Genocchi polynomial `\operatorname{G}_n(x)`.

    `\operatorname{G}_n(x)` is twice the difference between the plain and
    central Bernoulli polynomials, so has degree `n-1`:

    .. math :: \operatorname{G}_n(x) = 2 (\operatorname{B}_n(x) -
            \operatorname{B}_n^c(x))

    The factor of 2 in the definition endows `\operatorname{G}_n(x)` with
    integer coefficients.

    Parameters
    ==========

    n : int
        Degree of the polynomial plus one.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.genocchi
    """
    return named_poly(n, dup_genocchi, ZZ, "Genocchi polynomial", (x,), polys)

def dup_euler(n, K):
    """Low-level implementation of Euler polynomials."""
    return dup_quo_ground(dup_genocchi(n+1, ZZ), K(-n-1), K)

@public
def euler_poly(n, x=None, polys=False):
    r"""Generates the Euler polynomial `\operatorname{E}_n(x)`.

    These are scaled and reindexed versions of the Genocchi polynomials:

    .. math :: \operatorname{E}_n(x) = -\frac{\operatorname{G}_{n+1}(x)}{n+1}

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.euler
    """
    return named_poly(n, dup_euler, QQ, "Euler polynomial", (x,), polys)

def dup_andre(n, K):
    """Low-level implementation of Andre polynomials."""
    p = [K.one]
    for i in range(1, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K), K)
    return p

@public
def andre_poly(n, x=None, polys=False):
    r"""Generates the Andre polynomial `\mathcal{A}_n(x)`.

    This is the Appell sequence where the constant coefficients form the sequence
    of Euler numbers ``euler(n)``. As such they have integer coefficients
    and parities matching the parity of `n`.

    Luschny calls these the *Swiss-knife polynomials* because their values
    at 0 and 1 can be simply transformed into both the Bernoulli and Euler
    numbers. Here they are called the Andre polynomials because
    `|\mathcal{A}_n(n\bmod 2)|` for `n \ge 0` generates what Luschny calls
    the *Andre numbers*, A000111 in the OEIS.

    Examples
    ========

    >>> from sympy import bernoulli, euler, genocchi
    >>> from sympy.abc import x
    >>> from sympy.polys import andre_poly
    >>> andre_poly(9, x)
    x**9 - 36*x**7 + 630*x**5 - 5124*x**3 + 12465*x

    >>> [andre_poly(n, 0) for n in range(11)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
    >>> [euler(n) for n in range(11)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
    >>> [andre_poly(n-1, 1) * n / (4**n - 2**n) for n in range(1, 11)]
    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> [bernoulli(n) for n in range(1, 11)]
    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> [-andre_poly(n-1, -1) * n / (-2)**(n-1) for n in range(1, 11)]
    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]
    >>> [genocchi(n) for n in range(1, 11)]
    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]

    >>> [abs(andre_poly(n, n%2)) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.andre

    References
    ==========

    .. [1] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """
    return named_poly(n, dup_andre, ZZ, "Andre polynomial", (x,), polys)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_generated_run.py ===
# ***********************************************************
# ******* WARNING: AUTOGENERATED! ALL EDITS WILL BE LOST ******
# *************************************************************
from __future__ import annotations

from typing import TYPE_CHECKING

from ._ki import enable_ki_protection
from ._run import _NO_SEND, GLOBAL_RUN_CONTEXT, RunStatistics, Task

if TYPE_CHECKING:
    import contextvars
    from collections.abc import Awaitable, Callable

    from outcome import Outcome
    from typing_extensions import Unpack

    from .._abc import Clock
    from ._entry_queue import TrioToken
    from ._run import PosArgT


__all__ = [
    "current_clock",
    "current_root_task",
    "current_statistics",
    "current_time",
    "current_trio_token",
    "reschedule",
    "spawn_system_task",
    "wait_all_tasks_blocked",
]


@enable_ki_protection
def current_statistics() -> RunStatistics:
    """Returns ``RunStatistics``, which contains run-loop-level debugging information.

    Currently, the following fields are defined:

    * ``tasks_living`` (int): The number of tasks that have been spawned
      and not yet exited.
    * ``tasks_runnable`` (int): The number of tasks that are currently
      queued on the run queue (as opposed to blocked waiting for something
      to happen).
    * ``seconds_to_next_deadline`` (float): The time until the next
      pending cancel scope deadline. May be negative if the deadline has
      expired but we haven't yet processed cancellations. May be
      :data:`~math.inf` if there are no pending deadlines.
    * ``run_sync_soon_queue_size`` (int): The number of
      unprocessed callbacks queued via
      :meth:`trio.lowlevel.TrioToken.run_sync_soon`.
    * ``io_statistics`` (object): Some statistics from Trio's I/O
      backend. This always has an attribute ``backend`` which is a string
      naming which operating-system-specific I/O backend is in use; the
      other attributes vary between backends.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_statistics()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def current_time() -> float:
    """Returns the current time according to Trio's internal clock.

    Returns:
        float: The current time.

    Raises:
        RuntimeError: if not inside a call to :func:`trio.run`.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_time()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def current_clock() -> Clock:
    """Returns the current :class:`~trio.abc.Clock`."""
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_clock()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def current_root_task() -> Task | None:
    """Returns the current root :class:`Task`.

    This is the task that is the ultimate parent of all other tasks.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_root_task()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def reschedule(task: Task, next_send: Outcome[object] = _NO_SEND) -> None:
    """Reschedule the given task with the given
    :class:`outcome.Outcome`.

    See :func:`wait_task_rescheduled` for the gory details.

    There must be exactly one call to :func:`reschedule` for every call to
    :func:`wait_task_rescheduled`. (And when counting, keep in mind that
    returning :data:`Abort.SUCCEEDED` from an abort callback is equivalent
    to calling :func:`reschedule` once.)

    Args:
      task (trio.lowlevel.Task): the task to be rescheduled. Must be blocked
          in a call to :func:`wait_task_rescheduled`.
      next_send (outcome.Outcome): the value (or error) to return (or
          raise) from :func:`wait_task_rescheduled`.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.reschedule(task, next_send)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def spawn_system_task(
    async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]],
    *args: Unpack[PosArgT],
    name: object = None,
    context: contextvars.Context | None = None,
) -> Task:
    """Spawn a "system" task.

    System tasks have a few differences from regular tasks:

    * They don't need an explicit nursery; instead they go into the
      internal "system nursery".

    * If a system task raises an exception, then it's converted into a
      :exc:`~trio.TrioInternalError` and *all* tasks are cancelled. If you
      write a system task, you should be careful to make sure it doesn't
      crash.

    * System tasks are automatically cancelled when the main task exits.

    * By default, system tasks have :exc:`KeyboardInterrupt` protection
      *enabled*. If you want your task to be interruptible by control-C,
      then you need to use :func:`disable_ki_protection` explicitly (and
      come up with some plan for what to do with a
      :exc:`KeyboardInterrupt`, given that system tasks aren't allowed to
      raise exceptions).

    * System tasks do not inherit context variables from their creator.

    Towards the end of a call to :meth:`trio.run`, after the main
    task and all system tasks have exited, the system nursery
    becomes closed. At this point, new calls to
    :func:`spawn_system_task` will raise ``RuntimeError("Nursery
    is closed to new arrivals")`` instead of creating a system
    task. It's possible to encounter this state either in
    a ``finally`` block in an async generator, or in a callback
    passed to :meth:`TrioToken.run_sync_soon` at the right moment.

    Args:
      async_fn: An async callable.
      args: Positional arguments for ``async_fn``. If you want to pass
          keyword arguments, use :func:`functools.partial`.
      name: The name for this task. Only used for debugging/introspection
          (e.g. ``repr(task_obj)``). If this isn't a string,
          :func:`spawn_system_task` will try to make it one. A common use
          case is if you're wrapping a function before spawning a new
          task, you might pass the original function as the ``name=`` to
          make debugging easier.
      context: An optional ``contextvars.Context`` object with context variables
          to use for this task. You would normally get a copy of the current
          context with ``context = contextvars.copy_context()`` and then you would
          pass that ``context`` object here.

    Returns:
      Task: the newly spawned task

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.spawn_system_task(
            async_fn, *args, name=name, context=context
        )
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def current_trio_token() -> TrioToken:
    """Retrieve the :class:`TrioToken` for the current call to
    :func:`trio.run`.

    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.current_trio_token()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_all_tasks_blocked(cushion: float = 0.0) -> None:
    """Block until there are no runnable tasks.

    This is useful in testing code when you want to give other tasks a
    chance to "settle down". The calling task is blocked, and doesn't wake
    up until all other tasks are also blocked for at least ``cushion``
    seconds. (Setting a non-zero ``cushion`` is intended to handle cases
    like two tasks talking to each other over a local socket, where we
    want to ignore the potential brief moment between a send and receive
    when all tasks are blocked.)

    Note that ``cushion`` is measured in *real* time, not the Trio clock
    time.

    If there are multiple tasks blocked in :func:`wait_all_tasks_blocked`,
    then the one with the shortest ``cushion`` is the one woken (and
    this task becoming unblocked resets the timers for the remaining
    tasks). If there are multiple tasks that have exactly the same
    ``cushion``, then all are woken.

    You should also consider :class:`trio.testing.Sequencer`, which
    provides a more explicit way to control execution ordering within a
    test, and will often produce more readable tests.

    Example:
      Here's an example of one way to test that Trio's locks are fair: we
      take the lock in the parent, start a child, wait for the child to be
      blocked waiting for the lock (!), and then check that we can't
      release and immediately re-acquire the lock::

         async def lock_taker(lock):
             await lock.acquire()
             lock.release()

         async def test_lock_fairness():
             lock = trio.Lock()
             await lock.acquire()
             async with trio.open_nursery() as nursery:
                 nursery.start_soon(lock_taker, lock)
                 # child hasn't run yet, we have the lock
                 assert lock.locked()
                 assert lock._owner is trio.lowlevel.current_task()
                 await trio.testing.wait_all_tasks_blocked()
                 # now the child has run and is blocked on lock.acquire(), we
                 # still have the lock
                 assert lock.locked()
                 assert lock._owner is trio.lowlevel.current_task()
                 lock.release()
                 try:
                     # The child has a prior claim, so we can't have it
                     lock.acquire_nowait()
                 except trio.WouldBlock:
                     assert lock._owner is not trio.lowlevel.current_task()
                     print("PASS")
                 else:
                     print("FAIL")

    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.wait_all_tasks_blocked(cushion)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_file_io.py ===
from __future__ import annotations

import importlib
import io
import os
import re
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import sentinel

import pytest

import trio
from trio import _core, _file_io
from trio._file_io import _FILE_ASYNC_METHODS, _FILE_SYNC_ATTRS, AsyncIOWrapper

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def path(tmp_path: pathlib.Path) -> str:
    return os.fspath(tmp_path / "test")


@pytest.fixture
def wrapped() -> mock.Mock:
    return mock.Mock(spec_set=io.StringIO)


@pytest.fixture
def async_file(wrapped: mock.Mock) -> AsyncIOWrapper[mock.Mock]:
    return trio.wrap_file(wrapped)


def test_wrap_invalid() -> None:
    with pytest.raises(TypeError):
        trio.wrap_file("")


def test_wrap_non_iobase() -> None:
    class FakeFile:
        def close(self) -> None:  # pragma: no cover
            pass

        def write(self) -> None:  # pragma: no cover
            pass

    wrapped = FakeFile()
    assert not isinstance(wrapped, io.IOBase)

    async_file = trio.wrap_file(wrapped)
    assert isinstance(async_file, AsyncIOWrapper)

    del FakeFile.write

    with pytest.raises(TypeError):
        trio.wrap_file(FakeFile())


def test_wrapped_property(
    async_file: AsyncIOWrapper[mock.Mock],
    wrapped: mock.Mock,
) -> None:
    assert async_file.wrapped is wrapped


def test_dir_matches_wrapped(
    async_file: AsyncIOWrapper[mock.Mock],
    wrapped: mock.Mock,
) -> None:
    attrs = _FILE_SYNC_ATTRS.union(_FILE_ASYNC_METHODS)

    # all supported attrs in wrapped should be available in async_file
    assert all(attr in dir(async_file) for attr in attrs if attr in dir(wrapped))
    # all supported attrs not in wrapped should not be available in async_file
    assert not any(
        attr in dir(async_file) for attr in attrs if attr not in dir(wrapped)
    )


def test_unsupported_not_forwarded() -> None:
    class FakeFile(io.RawIOBase):
        def unsupported_attr(self) -> None:  # pragma: no cover
            pass

    async_file = trio.wrap_file(FakeFile())

    assert hasattr(async_file.wrapped, "unsupported_attr")

    with pytest.raises(AttributeError):
        # B018 "useless expression"
        async_file.unsupported_attr  # type: ignore[attr-defined] # noqa: B018


def test_type_stubs_match_lists() -> None:
    """Check the manual stubs match the list of wrapped methods."""
    # Fetch the module's source code.
    assert _file_io.__spec__ is not None
    loader = _file_io.__spec__.loader
    assert isinstance(loader, importlib.abc.SourceLoader)
    source = io.StringIO(loader.get_source("trio._file_io"))

    # Find the class, then find the TYPE_CHECKING block.
    for line in source:
        if "class AsyncIOWrapper" in line:
            break
    else:  # pragma: no cover - should always find this
        pytest.fail("No class definition line?")

    for line in source:
        if "if TYPE_CHECKING" in line:
            break
    else:  # pragma: no cover - should always find this
        pytest.fail("No TYPE CHECKING line?")

    # Now we should be at the type checking block.
    found: list[tuple[str, str]] = []
    for line in source:  # pragma: no branch - expected to break early
        if line.strip() and not line.startswith(" " * 8):
            break  # Dedented out of the if TYPE_CHECKING block.
        match = re.match(r"\s*(async )?def ([a-zA-Z0-9_]+)\(", line)
        if match is not None:
            kind = "async" if match.group(1) is not None else "sync"
            found.append((match.group(2), kind))

    # Compare two lists so that we can easily see duplicates, and see what is different overall.
    expected = [(fname, "async") for fname in _FILE_ASYNC_METHODS]
    expected += [(fname, "sync") for fname in _FILE_SYNC_ATTRS]
    # Ignore order, error if duplicates are present.
    found.sort()
    expected.sort()
    assert found == expected


def test_sync_attrs_forwarded(
    async_file: AsyncIOWrapper[mock.Mock],
    wrapped: mock.Mock,
) -> None:
    for attr_name in _FILE_SYNC_ATTRS:
        if attr_name not in dir(async_file):
            continue

        assert getattr(async_file, attr_name) is getattr(wrapped, attr_name)


def test_sync_attrs_match_wrapper(
    async_file: AsyncIOWrapper[mock.Mock],
    wrapped: mock.Mock,
) -> None:
    for attr_name in _FILE_SYNC_ATTRS:
        if attr_name in dir(async_file):
            continue

        with pytest.raises(AttributeError):
            getattr(async_file, attr_name)

        with pytest.raises(AttributeError):
            getattr(wrapped, attr_name)


def test_async_methods_generated_once(async_file: AsyncIOWrapper[mock.Mock]) -> None:
    for meth_name in _FILE_ASYNC_METHODS:
        if meth_name not in dir(async_file):
            continue

        assert getattr(async_file, meth_name) is getattr(async_file, meth_name)


# I gave up on typing this one
def test_async_methods_signature(async_file: AsyncIOWrapper[mock.Mock]) -> None:
    # use read as a representative of all async methods
    assert async_file.read.__name__ == "read"
    assert async_file.read.__qualname__ == "AsyncIOWrapper.read"

    assert async_file.read.__doc__ is not None
    assert "io.StringIO.read" in async_file.read.__doc__


async def test_async_methods_wrap(
    async_file: AsyncIOWrapper[mock.Mock],
    wrapped: mock.Mock,
) -> None:
    for meth_name in _FILE_ASYNC_METHODS:
        if meth_name not in dir(async_file):
            continue

        meth = getattr(async_file, meth_name)
        wrapped_meth = getattr(wrapped, meth_name)

        value = await meth(sentinel.argument, keyword=sentinel.keyword)

        wrapped_meth.assert_called_once_with(
            sentinel.argument,
            keyword=sentinel.keyword,
        )
        assert value == wrapped_meth()

        wrapped.reset_mock()


def test_async_methods_match_wrapper(
    async_file: AsyncIOWrapper[mock.Mock],
    wrapped: mock.Mock,
) -> None:
    for meth_name in _FILE_ASYNC_METHODS:
        if meth_name in dir(async_file):
            continue

        with pytest.raises(AttributeError):
            getattr(async_file, meth_name)

        with pytest.raises(AttributeError):
            getattr(wrapped, meth_name)


async def test_open(path: pathlib.Path) -> None:
    f = await trio.open_file(path, "w")

    assert isinstance(f, AsyncIOWrapper)

    await f.aclose()


async def test_open_context_manager(path: pathlib.Path) -> None:
    async with await trio.open_file(path, "w") as f:
        assert isinstance(f, AsyncIOWrapper)
        assert not f.closed

    assert f.closed


async def test_async_iter() -> None:
    async_file = trio.wrap_file(io.StringIO("test\nfoo\nbar"))
    expected = list(async_file.wrapped)
    async_file.wrapped.seek(0)

    result = [line async for line in async_file]

    assert result == expected


async def test_aclose_cancelled(path: pathlib.Path) -> None:
    with _core.CancelScope() as cscope:
        f = await trio.open_file(path, "w")
        cscope.cancel()

        with pytest.raises(_core.Cancelled):
            await f.write("a")

        with pytest.raises(_core.Cancelled):
            await f.aclose()

    assert f.closed


async def test_detach_rewraps_asynciobase(tmp_path: pathlib.Path) -> None:
    tmp_file = tmp_path / "filename"
    tmp_file.touch()
    # flake8-async does not like opening files in async mode
    with open(tmp_file, mode="rb", buffering=0) as raw:  # noqa: ASYNC230
        buffered = io.BufferedReader(raw)

        async_file = trio.wrap_file(buffered)

        detached = await async_file.detach()

        assert isinstance(detached, AsyncIOWrapper)
        assert detached.wrapped is raw

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\diagnose.py ===
"""Diagnostic functions, mainly for use when doing tech support."""

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

import cProfile
from io import BytesIO
from html.parser import HTMLParser
import bs4
from bs4 import BeautifulSoup, __version__
from bs4.builder import builder_registry
from typing import (
    Any,
    IO,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from bs4._typing import _IncomingMarkup

import pstats
import random
import tempfile
import time
import traceback
import sys


def diagnose(data: "_IncomingMarkup") -> None:
    """Diagnostic suite for isolating common problems.

    :param data: Some markup that needs to be explained.
    :return: None; diagnostics are printed to standard output.
    """
    print(("Diagnostic running on Beautiful Soup %s" % __version__))
    print(("Python version %s" % sys.version))

    basic_parsers = ["html.parser", "html5lib", "lxml"]
    for name in basic_parsers:
        for builder in builder_registry.builders:
            if name in builder.features:
                break
        else:
            basic_parsers.remove(name)
            print(
                ("I noticed that %s is not installed. Installing it may help." % name)
            )

    if "lxml" in basic_parsers:
        basic_parsers.append("lxml-xml")
        try:
            from lxml import etree

            print(("Found lxml version %s" % ".".join(map(str, etree.LXML_VERSION))))
        except ImportError:
            print("lxml is not installed or couldn't be imported.")

    if "html5lib" in basic_parsers:
        try:
            import html5lib

            print(("Found html5lib version %s" % html5lib.__version__))
        except ImportError:
            print("html5lib is not installed or couldn't be imported.")

    if hasattr(data, "read"):
        data = data.read()

    for parser in basic_parsers:
        print(("Trying to parse your markup with %s" % parser))
        success = False
        try:
            soup = BeautifulSoup(data, features=parser)
            success = True
        except Exception:
            print(("%s could not parse the markup." % parser))
            traceback.print_exc()
        if success:
            print(("Here's what %s did with the markup:" % parser))
            print((soup.prettify()))

        print(("-" * 80))


def lxml_trace(data: "_IncomingMarkup", html: bool = True, **kwargs: Any) -> None:
    """Print out the lxml events that occur during parsing.

    This lets you see how lxml parses a document when no Beautiful
    Soup code is running. You can use this to determine whether
    an lxml-specific problem is in Beautiful Soup's lxml tree builders
    or in lxml itself.

    :param data: Some markup.
    :param html: If True, markup will be parsed with lxml's HTML parser.
       if False, lxml's XML parser will be used.
    """
    from lxml import etree

    recover = kwargs.pop("recover", True)
    if isinstance(data, str):
        data = data.encode("utf8")
    if not isinstance(data, IO):
        reader = BytesIO(data)
    for event, element in etree.iterparse(reader, html=html, recover=recover, **kwargs):
        print(("%s, %4s, %s" % (event, element.tag, element.text)))


class AnnouncingParser(HTMLParser):
    """Subclass of HTMLParser that announces parse events, without doing
    anything else.

    You can use this to get a picture of how html.parser sees a given
    document. The easiest way to do this is to call `htmlparser_trace`.
    """

    def _p(self, s: str) -> None:
        print(s)

    def handle_starttag(
        self,
        name: str,
        attrs: List[Tuple[str, Optional[str]]],
        handle_empty_element: bool = True,
    ) -> None:
        self._p(f"{name} {attrs} START")

    def handle_endtag(self, name: str, check_already_closed: bool = True) -> None:
        self._p("%s END" % name)

    def handle_data(self, data: str) -> None:
        self._p("%s DATA" % data)

    def handle_charref(self, name: str) -> None:
        self._p("%s CHARREF" % name)

    def handle_entityref(self, name: str) -> None:
        self._p("%s ENTITYREF" % name)

    def handle_comment(self, data: str) -> None:
        self._p("%s COMMENT" % data)

    def handle_decl(self, data: str) -> None:
        self._p("%s DECL" % data)

    def unknown_decl(self, data: str) -> None:
        self._p("%s UNKNOWN-DECL" % data)

    def handle_pi(self, data: str) -> None:
        self._p("%s PI" % data)


def htmlparser_trace(data: str) -> None:
    """Print out the HTMLParser events that occur during parsing.

    This lets you see how HTMLParser parses a document when no
    Beautiful Soup code is running.

    :param data: Some markup.
    """
    parser = AnnouncingParser()
    parser.feed(data)


_vowels: str = "aeiou"
_consonants: str = "bcdfghjklmnpqrstvwxyz"


def rword(length: int = 5) -> str:
    """Generate a random word-like string.

    :meta private:
    """
    s = ""
    for i in range(length):
        if i % 2 == 0:
            t = _consonants
        else:
            t = _vowels
        s += random.choice(t)
    return s


def rsentence(length: int = 4) -> str:
    """Generate a random sentence-like string.

    :meta private:
    """
    return " ".join(rword(random.randint(4, 9)) for i in range(length))


def rdoc(num_elements: int = 1000) -> str:
    """Randomly generate an invalid HTML document.

    :meta private:
    """
    tag_names = ["p", "div", "span", "i", "b", "script", "table"]
    elements = []
    for i in range(num_elements):
        choice = random.randint(0, 3)
        if choice == 0:
            # New tag.
            tag_name = random.choice(tag_names)
            elements.append("<%s>" % tag_name)
        elif choice == 1:
            elements.append(rsentence(random.randint(1, 4)))
        elif choice == 2:
            # Close a tag.
            tag_name = random.choice(tag_names)
            elements.append("</%s>" % tag_name)
    return "<html>" + "\n".join(elements) + "</html>"


def benchmark_parsers(num_elements: int = 100000) -> None:
    """Very basic head-to-head performance benchmark."""
    print(("Comparative parser benchmark on Beautiful Soup %s" % __version__))
    data = rdoc(num_elements)
    print(("Generated a large invalid HTML document (%d bytes)." % len(data)))

    for parser_name in ["lxml", ["lxml", "html"], "html5lib", "html.parser"]:
        success = False
        try:
            a = time.time()
            BeautifulSoup(data, parser_name)
            b = time.time()
            success = True
        except Exception:
            print(("%s could not parse the markup." % parser_name))
            traceback.print_exc()
        if success:
            print(("BS4+%s parsed the markup in %.2fs." % (parser_name, b - a)))

    from lxml import etree

    a = time.time()
    etree.HTML(data)
    b = time.time()
    print(("Raw lxml parsed the markup in %.2fs." % (b - a)))

    import html5lib

    parser = html5lib.HTMLParser()
    a = time.time()
    parser.parse(data)
    b = time.time()
    print(("Raw html5lib parsed the markup in %.2fs." % (b - a)))


def profile(num_elements: int = 100000, parser: str = "lxml") -> None:
    """Use Python's profiler on a randomly generated document."""
    filehandle = tempfile.NamedTemporaryFile()
    filename = filehandle.name

    data = rdoc(num_elements)
    vars = dict(bs4=bs4, data=data, parser=parser)
    cProfile.runctx("bs4.BeautifulSoup(data, parser)", vars, vars, filename)

    stats = pstats.Stats(filename)
    # stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats("_html5lib|bs4", 50)


# If this file is run as a script, standard input is diagnosed.
if __name__ == "__main__":
    diagnose(sys.stdin.read())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\demo_txt2img_xl.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import coloredlogs
from cuda import cudart
from demo_utils import (
    add_controlnet_arguments,
    arg_parser,
    get_metadata,
    load_pipelines,
    parse_arguments,
    process_controlnet_arguments,
    repeat_prompt,
)


def run_pipelines(
    args, base, refiner, prompt, negative_prompt, controlnet_image=None, controlnet_scale=None, is_warm_up=False
):
    image_height = args.height
    image_width = args.width
    batch_size = len(prompt)
    base.load_resources(image_height, image_width, batch_size)
    if refiner:
        refiner.load_resources(image_height, image_width, batch_size)

    def run_base_and_refiner(warmup=False):
        images, base_perf = base.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
            controlnet_images=controlnet_image,
            controlnet_scales=controlnet_scale,
            show_latency=not warmup,
            output_type="latent" if refiner else "pil",
        )
        if refiner is None:
            return images, base_perf

        # Use same seed in base and refiner.
        seed = base.get_current_seed()

        images, refiner_perf = refiner.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            denoising_steps=args.refiner_denoising_steps,
            image=images,
            strength=args.strength,
            guidance=args.refiner_guidance,
            seed=seed,
            show_latency=not warmup,
        )

        perf_data = None
        if base_perf and refiner_perf:
            perf_data = {"latency": base_perf["latency"] + refiner_perf["latency"]}
            perf_data.update({"base." + key: val for key, val in base_perf.items()})
            perf_data.update({"refiner." + key: val for key, val in refiner_perf.items()})

        return images, perf_data

    if not args.disable_cuda_graph:
        # inference once to get cuda graph
        _, _ = run_base_and_refiner(warmup=True)

    if args.num_warmup_runs > 0:
        print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _, _ = run_base_and_refiner(warmup=True)

    if is_warm_up:
        return

    print("[I] Running StableDiffusion XL pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images, perf_data = run_base_and_refiner(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    if refiner:
        print("|----------------|--------------|")
        print("| {:^14} | {:>9.2f} ms |".format("e2e", perf_data["latency"]))
        print("|----------------|--------------|")

    metadata = get_metadata(args, True)
    metadata.update({"base." + key: val for key, val in base.metadata().items()})
    if refiner:
        metadata.update({"refiner." + key: val for key, val in refiner.metadata().items()})
    if perf_data:
        metadata.update(perf_data)
    metadata["images"] = len(images)
    print(metadata)
    (refiner or base).save_images(images, prompt, negative_prompt, metadata)


def run_demo(args):
    """Run Stable Diffusion XL Base + Refiner together (known as ensemble of expert denoisers) to generate an image."""
    controlnet_image, controlnet_scale = process_controlnet_arguments(args)
    prompt, negative_prompt = repeat_prompt(args)
    batch_size = len(prompt)
    base, refiner = load_pipelines(args, batch_size)
    run_pipelines(args, base, refiner, prompt, negative_prompt, controlnet_image, controlnet_scale)
    base.teardown()
    if refiner:
        refiner.teardown()


def run_dynamic_shape_demo(args):
    """
    Run demo of generating images with different settings with ORT CUDA provider.
    Try "python demo_txt2img_xl.py --max-cuda-graphs 3 --user-compute-stream" to see the effect of multiple CUDA graphs.
    """
    args.engine = "ORT_CUDA"
    base, refiner = load_pipelines(args, 1)

    prompts = [
        "starry night over Golden Gate Bridge by van gogh",
        "beautiful photograph of Mt. Fuji during cherry blossom",
        "little cute gremlin sitting on a bed, cinematic",
        "cute grey cat with blue eyes, wearing a bowtie, acrylic painting",
        "beautiful Renaissance Revival Estate, Hobbit-House, detailed painting, warm colors, 8k, trending on Artstation",
        "blue owl, big green eyes, portrait, intricate metal design, unreal engine, octane render, realistic",
        "An astronaut riding a rainbow unicorn, cinematic, dramatic",
        "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm",
    ]

    # batch size, height, width, scheduler, steps, prompt, seed, guidance, refiner scheduler, refiner steps, refiner strength
    configs = [
        (1, 832, 1216, "UniPC", 8, prompts[0], None, 5.0, "UniPC", 10, 0.3),
        (1, 1024, 1024, "DDIM", 24, prompts[1], None, 5.0, "DDIM", 30, 0.3),
        (1, 1216, 832, "EulerA", 16, prompts[2], 1716921396712843, 5.0, "EulerA", 10, 0.3),
        (1, 1344, 768, "EulerA", 24, prompts[3], 123698071912362, 5.0, "EulerA", 20, 0.3),
        (2, 640, 1536, "UniPC", 16, prompts[4], 4312973633252712, 5.0, "UniPC", 10, 0.3),
        (2, 1152, 896, "DDIM", 24, prompts[5], 1964684802882906, 5.0, "UniPC", 20, 0.3),
    ]

    # In testing LCM, refiner is disabled so the settings of refiner is not used.
    if args.lcm:
        configs = [
            (1, 1024, 1024, "LCM", 8, prompts[6], None, 1.0, "UniPC", 20, 0.3),
            (1, 1216, 832, "LCM", 6, prompts[7], 1337, 1.0, "UniPC", 20, 0.3),
        ]

    # Warm up each combination of (batch size, height, width) once before serving.
    args.prompt = ["warm up"]
    args.num_warmup_runs = 1
    for batch_size, height, width, _, _, _, _, _, _, _, _ in configs:
        args.batch_size = batch_size
        args.height = height
        args.width = width
        print(f"\nWarm up batch_size={batch_size}, height={height}, width={width}")
        prompt, negative_prompt = repeat_prompt(args)
        run_pipelines(args, base, refiner, prompt, negative_prompt, is_warm_up=True)

    # Run pipeline on a list of prompts.
    args.num_warmup_runs = 0
    for (
        batch_size,
        height,
        width,
        scheduler,
        steps,
        example_prompt,
        seed,
        guidance,
        refiner_scheduler,
        refiner_denoising_steps,
        strength,
    ) in configs:
        args.prompt = [example_prompt]
        args.batch_size = batch_size
        args.height = height
        args.width = width
        args.scheduler = scheduler
        args.denoising_steps = steps
        args.seed = seed
        args.guidance = guidance
        args.refiner_scheduler = refiner_scheduler
        args.refiner_denoising_steps = refiner_denoising_steps
        args.strength = strength
        base.set_scheduler(scheduler)
        if refiner:
            refiner.set_scheduler(refiner_scheduler)
        prompt, negative_prompt = repeat_prompt(args)
        run_pipelines(args, base, refiner, prompt, negative_prompt, is_warm_up=False)

    base.teardown()
    if refiner:
        refiner.teardown()


def run_turbo_demo(args):
    """Run demo of generating images with test prompts with ORT CUDA provider."""
    args.engine = "ORT_CUDA"
    base, refiner = load_pipelines(args, 1)

    from datasets import load_dataset

    dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
    num_rows = dataset["test"].num_rows
    batch_size = args.batch_size
    num_batch = int(num_rows / batch_size)
    args.batch_size = 1
    for i in range(num_batch):
        args.prompt = [dataset["test"][i]["Prompt"] for i in range(i * batch_size, (i + 1) * batch_size)]
        base.set_scheduler(args.scheduler)
        if refiner:
            refiner.set_scheduler(args.refiner_scheduler)
        prompt, negative_prompt = repeat_prompt(args)
        run_pipelines(args, base, refiner, prompt, negative_prompt, is_warm_up=False)

    base.teardown()
    if refiner:
        refiner.teardown()


def main(args):
    no_prompt = isinstance(args.prompt, list) and len(args.prompt) == 1 and not args.prompt[0]
    if no_prompt:
        if args.version == "xl-turbo":
            run_turbo_demo(args)
        else:
            run_dynamic_shape_demo(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    parser = arg_parser("Options for Stable Diffusion XL Demo")
    add_controlnet_arguments(parser)
    args = parse_arguments(is_xl=True, parser=parser)

    if args.user_compute_stream:
        import torch

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            main(args)
    else:
        main(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_inspect.py ===
import inspect
from inspect import cleandoc, getdoc, getfile, isclass, ismodule, signature
from typing import Any, Collection, Iterable, Optional, Tuple, Type, Union

from .console import Group, RenderableType
from .control import escape_control_codes
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin
from .panel import Panel
from .pretty import Pretty
from .table import Table
from .text import Text, TextType


def _first_paragraph(doc: str) -> str:
    """Get the first paragraph from a docstring."""
    paragraph, _, _ = doc.partition("\n\n")
    return paragraph


class Inspect(JupyterMixin):
    """A renderable to inspect any Python Object.

    Args:
        obj (Any): An object to inspect.
        title (str, optional): Title to display over inspect result, or None use type. Defaults to None.
        help (bool, optional): Show full help text rather than just first paragraph. Defaults to False.
        methods (bool, optional): Enable inspection of callables. Defaults to False.
        docs (bool, optional): Also render doc strings. Defaults to True.
        private (bool, optional): Show private attributes (beginning with underscore). Defaults to False.
        dunder (bool, optional): Show attributes starting with double underscore. Defaults to False.
        sort (bool, optional): Sort attributes alphabetically. Defaults to True.
        all (bool, optional): Show all attributes. Defaults to False.
        value (bool, optional): Pretty print value of object. Defaults to True.
    """

    def __init__(
        self,
        obj: Any,
        *,
        title: Optional[TextType] = None,
        help: bool = False,
        methods: bool = False,
        docs: bool = True,
        private: bool = False,
        dunder: bool = False,
        sort: bool = True,
        all: bool = True,
        value: bool = True,
    ) -> None:
        self.highlighter = ReprHighlighter()
        self.obj = obj
        self.title = title or self._make_title(obj)
        if all:
            methods = private = dunder = True
        self.help = help
        self.methods = methods
        self.docs = docs or help
        self.private = private or dunder
        self.dunder = dunder
        self.sort = sort
        self.value = value

    def _make_title(self, obj: Any) -> Text:
        """Make a default title."""
        title_str = (
            str(obj)
            if (isclass(obj) or callable(obj) or ismodule(obj))
            else str(type(obj))
        )
        title_text = self.highlighter(title_str)
        return title_text

    def __rich__(self) -> Panel:
        return Panel.fit(
            Group(*self._render()),
            title=self.title,
            border_style="scope.border",
            padding=(0, 1),
        )

    def _get_signature(self, name: str, obj: Any) -> Optional[Text]:
        """Get a signature for a callable."""
        try:
            _signature = str(signature(obj)) + ":"
        except ValueError:
            _signature = "(...)"
        except TypeError:
            return None

        source_filename: Optional[str] = None
        try:
            source_filename = getfile(obj)
        except (OSError, TypeError):
            # OSError is raised if obj has no source file, e.g. when defined in REPL.
            pass

        callable_name = Text(name, style="inspect.callable")
        if source_filename:
            callable_name.stylize(f"link file://{source_filename}")
        signature_text = self.highlighter(_signature)

        qualname = name or getattr(obj, "__qualname__", name)

        # If obj is a module, there may be classes (which are callable) to display
        if inspect.isclass(obj):
            prefix = "class"
        elif inspect.iscoroutinefunction(obj):
            prefix = "async def"
        else:
            prefix = "def"

        qual_signature = Text.assemble(
            (f"{prefix} ", f"inspect.{prefix.replace(' ', '_')}"),
            (qualname, "inspect.callable"),
            signature_text,
        )

        return qual_signature

    def _render(self) -> Iterable[RenderableType]:
        """Render object."""

        def sort_items(item: Tuple[str, Any]) -> Tuple[bool, str]:
            key, (_error, value) = item
            return (callable(value), key.strip("_").lower())

        def safe_getattr(attr_name: str) -> Tuple[Any, Any]:
            """Get attribute or any exception."""
            try:
                return (None, getattr(obj, attr_name))
            except Exception as error:
                return (error, None)

        obj = self.obj
        keys = dir(obj)
        total_items = len(keys)
        if not self.dunder:
            keys = [key for key in keys if not key.startswith("__")]
        if not self.private:
            keys = [key for key in keys if not key.startswith("_")]
        not_shown_count = total_items - len(keys)
        items = [(key, safe_getattr(key)) for key in keys]
        if self.sort:
            items.sort(key=sort_items)

        items_table = Table.grid(padding=(0, 1), expand=False)
        items_table.add_column(justify="right")
        add_row = items_table.add_row
        highlighter = self.highlighter

        if callable(obj):
            signature = self._get_signature("", obj)
            if signature is not None:
                yield signature
                yield ""

        if self.docs:
            _doc = self._get_formatted_doc(obj)
            if _doc is not None:
                doc_text = Text(_doc, style="inspect.help")
                doc_text = highlighter(doc_text)
                yield doc_text
                yield ""

        if self.value and not (isclass(obj) or callable(obj) or ismodule(obj)):
            yield Panel(
                Pretty(obj, indent_guides=True, max_length=10, max_string=60),
                border_style="inspect.value.border",
            )
            yield ""

        for key, (error, value) in items:
            key_text = Text.assemble(
                (
                    key,
                    "inspect.attr.dunder" if key.startswith("__") else "inspect.attr",
                ),
                (" =", "inspect.equals"),
            )
            if error is not None:
                warning = key_text.copy()
                warning.stylize("inspect.error")
                add_row(warning, highlighter(repr(error)))
                continue

            if callable(value):
                if not self.methods:
                    continue

                _signature_text = self._get_signature(key, value)
                if _signature_text is None:
                    add_row(key_text, Pretty(value, highlighter=highlighter))
                else:
                    if self.docs:
                        docs = self._get_formatted_doc(value)
                        if docs is not None:
                            _signature_text.append("\n" if "\n" in docs else " ")
                            doc = highlighter(docs)
                            doc.stylize("inspect.doc")
                            _signature_text.append(doc)

                    add_row(key_text, _signature_text)
            else:
                add_row(key_text, Pretty(value, highlighter=highlighter))
        if items_table.row_count:
            yield items_table
        elif not_shown_count:
            yield Text.from_markup(
                f"[b cyan]{not_shown_count}[/][i] attribute(s) not shown.[/i] "
                f"Run [b][magenta]inspect[/]([not b]inspect[/])[/b] for options."
            )

    def _get_formatted_doc(self, object_: Any) -> Optional[str]:
        """
        Extract the docstring of an object, process it and returns it.
        The processing consists in cleaning up the doctring's indentation,
        taking only its 1st paragraph if `self.help` is not True,
        and escape its control codes.

        Args:
            object_ (Any): the object to get the docstring from.

        Returns:
            Optional[str]: the processed docstring, or None if no docstring was found.
        """
        docs = getdoc(object_)
        if docs is None:
            return None
        docs = cleandoc(docs).strip()
        if not self.help:
            docs = _first_paragraph(docs)
        return escape_control_codes(docs)


def get_object_types_mro(obj: Union[object, Type[Any]]) -> Tuple[type, ...]:
    """Returns the MRO of an object's class, or of the object itself if it's a class."""
    if not hasattr(obj, "__mro__"):
        # N.B. we cannot use `if type(obj) is type` here because it doesn't work with
        # some types of classes, such as the ones that use abc.ABCMeta.
        obj = type(obj)
    return getattr(obj, "__mro__", ())


def get_object_types_mro_as_strings(obj: object) -> Collection[str]:
    """
    Returns the MRO of an object's class as full qualified names, or of the object itself if it's a class.

    Examples:
        `object_types_mro_as_strings(JSONDecoder)` will return `['json.decoder.JSONDecoder', 'builtins.object']`
    """
    return [
        f'{getattr(type_, "__module__", "")}.{getattr(type_, "__qualname__", "")}'
        for type_ in get_object_types_mro(obj)
    ]


def is_object_one_of_types(
    obj: object, fully_qualified_types_names: Collection[str]
) -> bool:
    """
    Returns `True` if the given object's class (or the object itself, if it's a class) has one of the
    fully qualified names in its MRO.
    """
    for type_name in get_object_types_mro_as_strings(obj):
        if type_name in fully_qualified_types_names:
            return True
    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\itsdangerous\signer.py ===
from __future__ import annotations

import collections.abc as cabc
import hashlib
import hmac
import typing as t

from .encoding import _base64_alphabet
from .encoding import base64_decode
from .encoding import base64_encode
from .encoding import want_bytes
from .exc import BadSignature


class SigningAlgorithm:
    """Subclasses must implement :meth:`get_signature` to provide
    signature generation functionality.
    """

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        """Returns the signature for the given key and value."""
        raise NotImplementedError()

    def verify_signature(self, key: bytes, value: bytes, sig: bytes) -> bool:
        """Verifies the given signature matches the expected
        signature.
        """
        return hmac.compare_digest(sig, self.get_signature(key, value))


class NoneAlgorithm(SigningAlgorithm):
    """Provides an algorithm that does not perform any signing and
    returns an empty signature.
    """

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        return b""


def _lazy_sha1(string: bytes = b"") -> t.Any:
    """Don't access ``hashlib.sha1`` until runtime. FIPS builds may not include
    SHA-1, in which case the import and use as a default would fail before the
    developer can configure something else.
    """
    return hashlib.sha1(string)


class HMACAlgorithm(SigningAlgorithm):
    """Provides signature generation using HMACs."""

    #: The digest method to use with the MAC algorithm. This defaults to
    #: SHA1, but can be changed to any other function in the hashlib
    #: module.
    default_digest_method: t.Any = staticmethod(_lazy_sha1)

    def __init__(self, digest_method: t.Any = None):
        if digest_method is None:
            digest_method = self.default_digest_method

        self.digest_method: t.Any = digest_method

    def get_signature(self, key: bytes, value: bytes) -> bytes:
        mac = hmac.new(key, msg=value, digestmod=self.digest_method)
        return mac.digest()


def _make_keys_list(
    secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
) -> list[bytes]:
    if isinstance(secret_key, (str, bytes)):
        return [want_bytes(secret_key)]

    return [want_bytes(s) for s in secret_key]  # pyright: ignore


class Signer:
    """A signer securely signs bytes, then unsigns them to verify that
    the value hasn't been changed.

    The secret key should be a random string of ``bytes`` and should not
    be saved to code or version control. Different salts should be used
    to distinguish signing in different contexts. See :doc:`/concepts`
    for information about the security of the secret key and salt.

    :param secret_key: The secret key to sign and verify with. Can be a
        list of keys, oldest to newest, to support key rotation.
    :param salt: Extra key to combine with ``secret_key`` to distinguish
        signatures in different contexts.
    :param sep: Separator between the signature and value.
    :param key_derivation: How to derive the signing key from the secret
        key and salt. Possible values are ``concat``, ``django-concat``,
        or ``hmac``. Defaults to :attr:`default_key_derivation`, which
        defaults to ``django-concat``.
    :param digest_method: Hash function to use when generating the HMAC
        signature. Defaults to :attr:`default_digest_method`, which
        defaults to :func:`hashlib.sha1`. Note that the security of the
        hash alone doesn't apply when used intermediately in HMAC.
    :param algorithm: A :class:`SigningAlgorithm` instance to use
        instead of building a default :class:`HMACAlgorithm` with the
        ``digest_method``.

    .. versionchanged:: 2.0
        Added support for key rotation by passing a list to
        ``secret_key``.

    .. versionchanged:: 0.18
        ``algorithm`` was added as an argument to the class constructor.

    .. versionchanged:: 0.14
        ``key_derivation`` and ``digest_method`` were added as arguments
        to the class constructor.
    """

    #: The default digest method to use for the signer. The default is
    #: :func:`hashlib.sha1`, but can be changed to any :mod:`hashlib` or
    #: compatible object. Note that the security of the hash alone
    #: doesn't apply when used intermediately in HMAC.
    #:
    #: .. versionadded:: 0.14
    default_digest_method: t.Any = staticmethod(_lazy_sha1)

    #: The default scheme to use to derive the signing key from the
    #: secret key and salt. The default is ``django-concat``. Possible
    #: values are ``concat``, ``django-concat``, and ``hmac``.
    #:
    #: .. versionadded:: 0.14
    default_key_derivation: str = "django-concat"

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous.Signer",
        sep: str | bytes = b".",
        key_derivation: str | None = None,
        digest_method: t.Any | None = None,
        algorithm: SigningAlgorithm | None = None,
    ):
        #: The list of secret keys to try for verifying signatures, from
        #: oldest to newest. The newest (last) key is used for signing.
        #:
        #: This allows a key rotation system to keep a list of allowed
        #: keys and remove expired ones.
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)
        self.sep: bytes = want_bytes(sep)

        if self.sep in _base64_alphabet:
            raise ValueError(
                "The given separator cannot be used because it may be"
                " contained in the signature itself. ASCII letters,"
                " digits, and '-_=' must not be used."
            )

        if salt is not None:
            salt = want_bytes(salt)
        else:
            salt = b"itsdangerous.Signer"

        self.salt = salt

        if key_derivation is None:
            key_derivation = self.default_key_derivation

        self.key_derivation: str = key_derivation

        if digest_method is None:
            digest_method = self.default_digest_method

        self.digest_method: t.Any = digest_method

        if algorithm is None:
            algorithm = HMACAlgorithm(self.digest_method)

        self.algorithm: SigningAlgorithm = algorithm

    @property
    def secret_key(self) -> bytes:
        """The newest (last) entry in the :attr:`secret_keys` list. This
        is for compatibility from before key rotation support was added.
        """
        return self.secret_keys[-1]

    def derive_key(self, secret_key: str | bytes | None = None) -> bytes:
        """This method is called to derive the key. The default key
        derivation choices can be overridden here. Key derivation is not
        intended to be used as a security method to make a complex key
        out of a short password. Instead you should use large random
        secret keys.

        :param secret_key: A specific secret key to derive from.
            Defaults to the last item in :attr:`secret_keys`.

        .. versionchanged:: 2.0
            Added the ``secret_key`` parameter.
        """
        if secret_key is None:
            secret_key = self.secret_keys[-1]
        else:
            secret_key = want_bytes(secret_key)

        if self.key_derivation == "concat":
            return t.cast(bytes, self.digest_method(self.salt + secret_key).digest())
        elif self.key_derivation == "django-concat":
            return t.cast(
                bytes, self.digest_method(self.salt + b"signer" + secret_key).digest()
            )
        elif self.key_derivation == "hmac":
            mac = hmac.new(secret_key, digestmod=self.digest_method)
            mac.update(self.salt)
            return mac.digest()
        elif self.key_derivation == "none":
            return secret_key
        else:
            raise TypeError("Unknown key derivation method")

    def get_signature(self, value: str | bytes) -> bytes:
        """Returns the signature for the given value."""
        value = want_bytes(value)
        key = self.derive_key()
        sig = self.algorithm.get_signature(key, value)
        return base64_encode(sig)

    def sign(self, value: str | bytes) -> bytes:
        """Signs the given string."""
        value = want_bytes(value)
        return value + self.sep + self.get_signature(value)

    def verify_signature(self, value: str | bytes, sig: str | bytes) -> bool:
        """Verifies the signature for the given value."""
        try:
            sig = base64_decode(sig)
        except Exception:
            return False

        value = want_bytes(value)

        for secret_key in reversed(self.secret_keys):
            key = self.derive_key(secret_key)

            if self.algorithm.verify_signature(key, value, sig):
                return True

        return False

    def unsign(self, signed_value: str | bytes) -> bytes:
        """Unsigns the given string."""
        signed_value = want_bytes(signed_value)

        if self.sep not in signed_value:
            raise BadSignature(f"No {self.sep!r} found in value")

        value, sig = signed_value.rsplit(self.sep, 1)

        if self.verify_signature(value, sig):
            return value

        raise BadSignature(f"Signature {sig!r} does not match", payload=value)

    def validate(self, signed_value: str | bytes) -> bool:
        """Only validates the given signed value. Returns ``True`` if
        the signature exists and is valid.
        """
        try:
            self.unsign(signed_value)
            return True
        except BadSignature:
            return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\strings\base.py ===
from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import Sequence
    import re

    from pandas._typing import Scalar

    from pandas import Series


class BaseStringArrayMethods(abc.ABC):
    """
    Base class for extension arrays implementing string methods.

    This is where our ExtensionArrays can override the implementation of
    Series.str.<method>. We don't expect this to work with
    3rd-party extension arrays.

    * User calls Series.str.<method>
    * pandas extracts the extension array from the Series
    * pandas calls ``extension_array._str_<method>(*args, **kwargs)``
    * pandas wraps the result, to return to the user.

    See :ref:`Series.str` for the docstring of each method.
    """

    def _str_getitem(self, key):
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def _str_count(self, pat, flags: int = 0):
        pass

    @abc.abstractmethod
    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        pass

    @abc.abstractmethod
    def _str_contains(
        self, pat, case: bool = True, flags: int = 0, na=None, regex: bool = True
    ):
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat, na=None):
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat, na=None):
        pass

    @abc.abstractmethod
    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: int | Sequence[int]):
        pass

    @abc.abstractmethod
    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        pass

    @abc.abstractmethod
    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding, errors: str = "strict"):
        pass

    @abc.abstractmethod
    def _str_find(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_findall(self, pat, flags: int = 0):
        pass

    @abc.abstractmethod
    def _str_get(self, i):
        pass

    @abc.abstractmethod
    def _str_index(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_join(self, sep: str):
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand):
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand):
        pass

    @abc.abstractmethod
    def _str_len(self):
        pass

    @abc.abstractmethod
    def _str_slice(self, start=None, stop=None, step=None):
        pass

    @abc.abstractmethod
    def _str_slice_replace(self, start=None, stop=None, repl=None):
        pass

    @abc.abstractmethod
    def _str_translate(self, table):
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs):
        pass

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = "|"):
        pass

    @abc.abstractmethod
    def _str_isalnum(self):
        pass

    @abc.abstractmethod
    def _str_isalpha(self):
        pass

    @abc.abstractmethod
    def _str_isdecimal(self):
        pass

    @abc.abstractmethod
    def _str_isdigit(self):
        pass

    @abc.abstractmethod
    def _str_islower(self):
        pass

    @abc.abstractmethod
    def _str_isnumeric(self):
        pass

    @abc.abstractmethod
    def _str_isspace(self):
        pass

    @abc.abstractmethod
    def _str_istitle(self):
        pass

    @abc.abstractmethod
    def _str_isupper(self):
        pass

    @abc.abstractmethod
    def _str_capitalize(self):
        pass

    @abc.abstractmethod
    def _str_casefold(self):
        pass

    @abc.abstractmethod
    def _str_title(self):
        pass

    @abc.abstractmethod
    def _str_swapcase(self):
        pass

    @abc.abstractmethod
    def _str_lower(self):
        pass

    @abc.abstractmethod
    def _str_upper(self):
        pass

    @abc.abstractmethod
    def _str_normalize(self, form):
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip=None):
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip=None):
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip=None):
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Series:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Series:
        pass

    @abc.abstractmethod
    def _str_split(
        self, pat=None, n=-1, expand: bool = False, regex: bool | None = None
    ):
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat=None, n=-1):
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\vector.py ===
# dialects/oracle/vector.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

import array
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import sqlalchemy.types as types
from sqlalchemy.types import Float


class VectorIndexType(Enum):
    """Enum representing different types of VECTOR index structures.

    See :ref:`oracle_vector_datatype` for background.

    .. versionadded:: 2.0.41

    """

    HNSW = "HNSW"
    """
    The HNSW (Hierarchical Navigable Small World) index type.
    """
    IVF = "IVF"
    """
    The IVF (Inverted File Index) index type
    """


class VectorDistanceType(Enum):
    """Enum representing different types of vector distance metrics.

    See :ref:`oracle_vector_datatype` for background.

    .. versionadded:: 2.0.41

    """

    EUCLIDEAN = "EUCLIDEAN"
    """Euclidean distance (L2 norm).

    Measures the straight-line distance between two vectors in space.
    """
    DOT = "DOT"
    """Dot product similarity.

    Measures the algebraic similarity between two vectors.
    """
    COSINE = "COSINE"
    """Cosine similarity.

    Measures the cosine of the angle between two vectors.
    """
    MANHATTAN = "MANHATTAN"
    """Manhattan distance (L1 norm).

    Calculates the sum of absolute differences across dimensions.
    """


class VectorStorageFormat(Enum):
    """Enum representing the data format used to store vector components.

    See :ref:`oracle_vector_datatype` for background.

    .. versionadded:: 2.0.41

    """

    INT8 = "INT8"
    """
    8-bit integer format.
    """
    BINARY = "BINARY"
    """
    Binary format.
    """
    FLOAT32 = "FLOAT32"
    """
    32-bit floating-point format.
    """
    FLOAT64 = "FLOAT64"
    """
    64-bit floating-point format.
    """


@dataclass
class VectorIndexConfig:
    """Define the configuration for Oracle VECTOR Index.

    See :ref:`oracle_vector_datatype` for background.

    .. versionadded:: 2.0.41

    :param index_type: Enum value from :class:`.VectorIndexType`
     Specifies the indexing method. For HNSW, this must be
     :attr:`.VectorIndexType.HNSW`.

    :param distance: Enum value from :class:`.VectorDistanceType`
     specifies the metric for calculating distance between VECTORS.

    :param accuracy: interger. Should be in the range 0 to 100
     Specifies the accuracy of the nearest neighbor search during
     query execution.

    :param parallel: integer. Specifies degree of parallelism.

    :param hnsw_neighbors: interger. Should be in the range 0 to
     2048. Specifies the number of nearest neighbors considered
     during the search. The attribute :attr:`.VectorIndexConfig.hnsw_neighbors`
     is HNSW index specific.

    :param hnsw_efconstruction: integer. Should be in the range 0
     to 65535. Controls the trade-off between indexing speed and
     recall quality during index construction. The attribute
     :attr:`.VectorIndexConfig.hnsw_efconstruction` is HNSW index
     specific.

    :param ivf_neighbor_partitions: integer. Should be in the range
     0 to 10,000,000. Specifies the number of partitions used to
     divide the dataset. The attribute
     :attr:`.VectorIndexConfig.ivf_neighbor_partitions` is IVF index
     specific.

    :param ivf_sample_per_partition: integer. Should be between 1
     and ``num_vectors / neighbor partitions``. Specifies the
     number of samples used per partition. The attribute
     :attr:`.VectorIndexConfig.ivf_sample_per_partition` is IVF index
     specific.

    :param ivf_min_vectors_per_partition: integer. From 0 (no trimming)
     to the total number of vectors (results in 1 partition). Specifies
     the minimum number of vectors per partition. The attribute
     :attr:`.VectorIndexConfig.ivf_min_vectors_per_partition`
     is IVF index specific.

    """

    index_type: VectorIndexType = VectorIndexType.HNSW
    distance: Optional[VectorDistanceType] = None
    accuracy: Optional[int] = None
    hnsw_neighbors: Optional[int] = None
    hnsw_efconstruction: Optional[int] = None
    ivf_neighbor_partitions: Optional[int] = None
    ivf_sample_per_partition: Optional[int] = None
    ivf_min_vectors_per_partition: Optional[int] = None
    parallel: Optional[int] = None

    def __post_init__(self):
        self.index_type = VectorIndexType(self.index_type)
        for field in [
            "hnsw_neighbors",
            "hnsw_efconstruction",
            "ivf_neighbor_partitions",
            "ivf_sample_per_partition",
            "ivf_min_vectors_per_partition",
            "parallel",
            "accuracy",
        ]:
            value = getattr(self, field)
            if value is not None and not isinstance(value, int):
                raise TypeError(
                    f"{field} must be an integer if"
                    f"provided, got {type(value).__name__}"
                )


class VECTOR(types.TypeEngine):
    """Oracle VECTOR datatype.

    For complete background on using this type, see
    :ref:`oracle_vector_datatype`.

    .. versionadded:: 2.0.41

    """

    cache_ok = True
    __visit_name__ = "VECTOR"

    _typecode_map = {
        VectorStorageFormat.INT8: "b",  # Signed int
        VectorStorageFormat.BINARY: "B",  # Unsigned int
        VectorStorageFormat.FLOAT32: "f",  # Float
        VectorStorageFormat.FLOAT64: "d",  # Double
    }

    def __init__(self, dim=None, storage_format=None):
        """Construct a VECTOR.

        :param dim: integer. The dimension of the VECTOR datatype. This
         should be an integer value.

        :param storage_format: VectorStorageFormat. The VECTOR storage
         type format. This may be Enum values form
         :class:`.VectorStorageFormat` INT8, BINARY, FLOAT32, or FLOAT64.

        """
        if dim is not None and not isinstance(dim, int):
            raise TypeError("dim must be an interger")
        if storage_format is not None and not isinstance(
            storage_format, VectorStorageFormat
        ):
            raise TypeError(
                "storage_format must be an enum of type VectorStorageFormat"
            )
        self.dim = dim
        self.storage_format = storage_format

    def _cached_bind_processor(self, dialect):
        """
        Convert a list to a array.array before binding it to the database.
        """

        def process(value):
            if value is None or isinstance(value, array.array):
                return value

            # Convert list to a array.array
            elif isinstance(value, list):
                typecode = self._array_typecode(self.storage_format)
                value = array.array(typecode, value)
                return value

            else:
                raise TypeError("VECTOR accepts list or array.array()")

        return process

    def _cached_result_processor(self, dialect, coltype):
        """
        Convert a array.array to list before binding it to the database.
        """

        def process(value):
            if isinstance(value, array.array):
                return list(value)

        return process

    def _array_typecode(self, typecode):
        """
        Map storage format to array typecode.
        """
        return self._typecode_map.get(typecode, "d")

    class comparator_factory(types.TypeEngine.Comparator):
        def l2_distance(self, other):
            return self.op("<->", return_type=Float)(other)

        def inner_product(self, other):
            return self.op("<#>", return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op("<=>", return_type=Float)(other)