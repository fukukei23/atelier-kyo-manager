
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\writer\excel.py ===
# Copyright (c) 2010-2024 openpyxl


# Python stdlib imports
import datetime
import re
from zipfile import ZipFile, ZIP_DEFLATED

# package imports
from openpyxl.utils.exceptions import InvalidFileException
from openpyxl.xml.constants import (
    ARC_ROOT_RELS,
    ARC_WORKBOOK_RELS,
    ARC_APP,
    ARC_CORE,
    ARC_CUSTOM,
    CPROPS_TYPE,
    ARC_THEME,
    ARC_STYLE,
    ARC_WORKBOOK,
    )
from openpyxl.drawing.spreadsheet_drawing import SpreadsheetDrawing
from openpyxl.xml.functions import tostring, fromstring
from openpyxl.packaging.manifest import Manifest
from openpyxl.packaging.relationship import (
    get_rels_path,
    RelationshipList,
    Relationship,
)
from openpyxl.comments.comment_sheet import CommentSheet
from openpyxl.styles.stylesheet import write_stylesheet
from openpyxl.worksheet._writer import WorksheetWriter
from openpyxl.workbook._writer import WorkbookWriter
from .theme import theme_xml


class ExcelWriter:
    """Write a workbook object to an Excel file."""

    def __init__(self, workbook, archive):
        self._archive = archive
        self.workbook = workbook
        self.manifest = Manifest()
        self.vba_modified = set()
        self._tables = []
        self._charts = []
        self._images = []
        self._drawings = []
        self._comments = []
        self._pivots = []


    def write_data(self):
        from openpyxl.packaging.extended import ExtendedProperties
        """Write the various xml files into the zip archive."""
        # cleanup all worksheets
        archive = self._archive

        props = ExtendedProperties()
        archive.writestr(ARC_APP, tostring(props.to_tree()))

        archive.writestr(ARC_CORE, tostring(self.workbook.properties.to_tree()))
        if self.workbook.loaded_theme:
            archive.writestr(ARC_THEME, self.workbook.loaded_theme)
        else:
            archive.writestr(ARC_THEME, theme_xml)

        if len(self.workbook.custom_doc_props) >= 1:
            archive.writestr(ARC_CUSTOM, tostring(self.workbook.custom_doc_props.to_tree()))
            class CustomOverride():
                path = "/" + ARC_CUSTOM #PartName
                mime_type = CPROPS_TYPE #ContentType

            custom_override = CustomOverride()
            self.manifest.append(custom_override)

        self._write_worksheets()
        self._write_chartsheets()
        self._write_images()
        self._write_charts()

        self._write_external_links()

        stylesheet = write_stylesheet(self.workbook)
        archive.writestr(ARC_STYLE, tostring(stylesheet))

        writer = WorkbookWriter(self.workbook)
        archive.writestr(ARC_ROOT_RELS, writer.write_root_rels())
        archive.writestr(ARC_WORKBOOK, writer.write())
        archive.writestr(ARC_WORKBOOK_RELS, writer.write_rels())

        self._merge_vba()

        self.manifest._write(archive, self.workbook)

    def _merge_vba(self):
        """
        If workbook contains macros then extract associated files from cache
        of old file and add to archive
        """
        ARC_VBA = re.compile("|".join(
            ('xl/vba', r'xl/drawings/.*vmlDrawing\d\.vml',
             'xl/ctrlProps', 'customUI', 'xl/activeX', r'xl/media/.*\.emf')
        )
                             )

        if self.workbook.vba_archive:
            for name in set(self.workbook.vba_archive.namelist()) - self.vba_modified:
                if ARC_VBA.match(name):
                    self._archive.writestr(name, self.workbook.vba_archive.read(name))


    def _write_images(self):
        # delegate to object
        for img in self._images:
            self._archive.writestr(img.path[1:], img._data())


    def _write_charts(self):
        # delegate to object
        if len(self._charts) != len(set(self._charts)):
            raise InvalidFileException("The same chart cannot be used in more than one worksheet")
        for chart in self._charts:
            self._archive.writestr(chart.path[1:], tostring(chart._write()))
            self.manifest.append(chart)


    def _write_drawing(self, drawing):
        """
        Write a drawing
        """
        self._drawings.append(drawing)
        drawing._id = len(self._drawings)
        for chart in drawing.charts:
            self._charts.append(chart)
            chart._id = len(self._charts)
        for img in drawing.images:
            self._images.append(img)
            img._id = len(self._images)
        rels_path = get_rels_path(drawing.path)[1:]
        self._archive.writestr(drawing.path[1:], tostring(drawing._write()))
        self._archive.writestr(rels_path, tostring(drawing._write_rels()))
        self.manifest.append(drawing)


    def _write_chartsheets(self):
        for idx, sheet in enumerate(self.workbook.chartsheets, 1):

            sheet._id = idx
            xml = tostring(sheet.to_tree())

            self._archive.writestr(sheet.path[1:], xml)
            self.manifest.append(sheet)

            if sheet._drawing:
                self._write_drawing(sheet._drawing)

                rel = Relationship(type="drawing", Target=sheet._drawing.path)
                rels = RelationshipList()
                rels.append(rel)
                tree = rels.to_tree()

                rels_path = get_rels_path(sheet.path[1:])
                self._archive.writestr(rels_path, tostring(tree))


    def _write_comment(self, ws):

        cs = CommentSheet.from_comments(ws._comments)
        self._comments.append(cs)
        cs._id = len(self._comments)
        self._archive.writestr(cs.path[1:], tostring(cs.to_tree()))
        self.manifest.append(cs)

        if ws.legacy_drawing is None or self.workbook.vba_archive is None:
            ws.legacy_drawing = 'xl/drawings/commentsDrawing{0}.vml'.format(cs._id)
            vml = None
        else:
            vml = fromstring(self.workbook.vba_archive.read(ws.legacy_drawing))

        vml = cs.write_shapes(vml)

        self._archive.writestr(ws.legacy_drawing, vml)
        self.vba_modified.add(ws.legacy_drawing)

        comment_rel = Relationship(Id="comments", type=cs._rel_type, Target=cs.path)
        ws._rels.append(comment_rel)


    def write_worksheet(self, ws):
        ws._drawing = SpreadsheetDrawing()
        ws._drawing.charts = ws._charts
        ws._drawing.images = ws._images
        if self.workbook.write_only:
            if not ws.closed:
                ws.close()
            writer = ws._writer
        else:
            writer = WorksheetWriter(ws)
            writer.write()

        ws._rels = writer._rels
        self._archive.write(writer.out, ws.path[1:])
        self.manifest.append(ws)
        writer.cleanup()


    def _write_worksheets(self):

        pivot_caches = set()

        for idx, ws in enumerate(self.workbook.worksheets, 1):

            ws._id = idx
            self.write_worksheet(ws)

            if ws._drawing:
                self._write_drawing(ws._drawing)

                for r in ws._rels:
                    if "drawing" in r.Type:
                        r.Target = ws._drawing.path

            if ws._comments:
                self._write_comment(ws)

            if ws.legacy_drawing is not None:
                shape_rel = Relationship(type="vmlDrawing", Id="anysvml",
                                         Target="/" + ws.legacy_drawing)
                ws._rels.append(shape_rel)

            for t in ws._tables.values():
                self._tables.append(t)
                t.id = len(self._tables)
                t._write(self._archive)
                self.manifest.append(t)
                ws._rels.get(t._rel_id).Target = t.path

            for p in ws._pivots:
                if p.cache not in pivot_caches:
                    pivot_caches.add(p.cache)
                    p.cache._id = len(pivot_caches)

                self._pivots.append(p)
                p._id = len(self._pivots)
                p._write(self._archive, self.manifest)
                self.workbook._pivots.append(p)
                r = Relationship(Type=p.rel_type, Target=p.path)
                ws._rels.append(r)

            if ws._rels:
                tree = ws._rels.to_tree()
                rels_path = get_rels_path(ws.path)[1:]
                self._archive.writestr(rels_path, tostring(tree))


    def _write_external_links(self):
        # delegate to object
        """Write links to external workbooks"""
        wb = self.workbook
        for idx, link in enumerate(wb._external_links, 1):
            link._id = idx
            rels_path = get_rels_path(link.path[1:])

            xml = link.to_tree()
            self._archive.writestr(link.path[1:], tostring(xml))
            rels = RelationshipList()
            rels.append(link.file_link)
            self._archive.writestr(rels_path, tostring(rels.to_tree()))
            self.manifest.append(link)


    def save(self):
        """Write data into the archive."""
        self.write_data()
        self._archive.close()


def save_workbook(workbook, filename):
    """Save the given workbook on the filesystem under the name filename.

    :param workbook: the workbook to save
    :type workbook: :class:`openpyxl.workbook.Workbook`

    :param filename: the path to which save the workbook
    :type filename: string

    :rtype: bool

    """
    archive = ZipFile(filename, 'w', ZIP_DEFLATED, allowZip64=True)
    workbook.properties.modified = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
    writer = ExcelWriter(workbook, archive)
    writer.save()
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\parsers\arrow_parser_wrapper.py ===
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    ParserError,
    ParserWarning,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.inference import is_integer

from pandas.io._util import arrow_table_to_pandas
from pandas.io.parsers.base_parser import ParserBase

if TYPE_CHECKING:
    from pandas._typing import ReadBuffer

    from pandas import DataFrame


class ArrowParserWrapper(ParserBase):
    """
    Wrapper for the pyarrow engine for read_csv()
    """

    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None:
        super().__init__(kwds)
        self.kwds = kwds
        self.src = src

        self._parse_kwds()

    def _parse_kwds(self) -> None:
        """
        Validates keywords before passing to pyarrow.
        """
        encoding: str | None = self.kwds.get("encoding")
        self.encoding = "utf-8" if encoding is None else encoding

        na_values = self.kwds["na_values"]
        if isinstance(na_values, dict):
            raise ValueError(
                "The pyarrow engine doesn't support passing a dict for na_values"
            )
        self.na_values = list(self.kwds["na_values"])

    def _get_pyarrow_options(self) -> None:
        """
        Rename some arguments to pass to pyarrow
        """
        mapping = {
            "usecols": "include_columns",
            "na_values": "null_values",
            "escapechar": "escape_char",
            "skip_blank_lines": "ignore_empty_lines",
            "decimal": "decimal_point",
            "quotechar": "quote_char",
        }
        for pandas_name, pyarrow_name in mapping.items():
            if pandas_name in self.kwds and self.kwds.get(pandas_name) is not None:
                self.kwds[pyarrow_name] = self.kwds.pop(pandas_name)

        # Date format handling
        # If we get a string, we need to convert it into a list for pyarrow
        # If we get a dict, we want to parse those separately
        date_format = self.date_format
        if isinstance(date_format, str):
            date_format = [date_format]
        else:
            # In case of dict, we don't want to propagate through, so
            # just set to pyarrow default of None

            # Ideally, in future we disable pyarrow dtype inference (read in as string)
            # to prevent misreads.
            date_format = None
        self.kwds["timestamp_parsers"] = date_format

        self.parse_options = {
            option_name: option_value
            for option_name, option_value in self.kwds.items()
            if option_value is not None
            and option_name
            in ("delimiter", "quote_char", "escape_char", "ignore_empty_lines")
        }

        on_bad_lines = self.kwds.get("on_bad_lines")
        if on_bad_lines is not None:
            if callable(on_bad_lines):
                self.parse_options["invalid_row_handler"] = on_bad_lines
            elif on_bad_lines == ParserBase.BadLineHandleMethod.ERROR:
                self.parse_options[
                    "invalid_row_handler"
                ] = None  # PyArrow raises an exception by default
            elif on_bad_lines == ParserBase.BadLineHandleMethod.WARN:

                def handle_warning(invalid_row) -> str:
                    warnings.warn(
                        f"Expected {invalid_row.expected_columns} columns, but found "
                        f"{invalid_row.actual_columns}: {invalid_row.text}",
                        ParserWarning,
                        stacklevel=find_stack_level(),
                    )
                    return "skip"

                self.parse_options["invalid_row_handler"] = handle_warning
            elif on_bad_lines == ParserBase.BadLineHandleMethod.SKIP:
                self.parse_options["invalid_row_handler"] = lambda _: "skip"

        self.convert_options = {
            option_name: option_value
            for option_name, option_value in self.kwds.items()
            if option_value is not None
            and option_name
            in (
                "include_columns",
                "null_values",
                "true_values",
                "false_values",
                "decimal_point",
                "timestamp_parsers",
            )
        }
        self.convert_options["strings_can_be_null"] = "" in self.kwds["null_values"]
        # autogenerated column names are prefixed with 'f' in pyarrow.csv
        if self.header is None and "include_columns" in self.convert_options:
            self.convert_options["include_columns"] = [
                f"f{n}" for n in self.convert_options["include_columns"]
            ]

        self.read_options = {
            "autogenerate_column_names": self.header is None,
            "skip_rows": self.header
            if self.header is not None
            else self.kwds["skiprows"],
            "encoding": self.encoding,
        }

    def _finalize_pandas_output(self, frame: DataFrame) -> DataFrame:
        """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame: DataFrame
            The DataFrame to process.

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
        num_cols = len(frame.columns)
        multi_index_named = True
        if self.header is None:
            if self.names is None:
                if self.header is None:
                    self.names = range(num_cols)
            if len(self.names) != num_cols:
                # usecols is passed through to pyarrow, we only handle index col here
                # The only way self.names is not the same length as number of cols is
                # if we have int index_col. We should just pad the names(they will get
                # removed anyways) to expected length then.
                columns_prefix = [str(x) for x in range(num_cols - len(self.names))]
                self.names = columns_prefix + self.names
                multi_index_named = False
            frame.columns = self.names
        # we only need the frame not the names
        _, frame = self._do_date_conversions(frame.columns, frame)
        if self.index_col is not None:
            index_to_set = self.index_col.copy()
            for i, item in enumerate(self.index_col):
                if is_integer(item):
                    index_to_set[i] = frame.columns[item]
                # String case
                elif item not in frame.columns:
                    raise ValueError(f"Index {item} invalid")

                # Process dtype for index_col and drop from dtypes
                if self.dtype is not None:
                    key, new_dtype = (
                        (item, self.dtype.get(item))
                        if self.dtype.get(item) is not None
                        else (frame.columns[item], self.dtype.get(frame.columns[item]))
                    )
                    if new_dtype is not None:
                        frame[key] = frame[key].astype(new_dtype)
                        del self.dtype[key]

            frame.set_index(index_to_set, drop=True, inplace=True)
            # Clear names if headerless and no name given
            if self.header is None and not multi_index_named:
                frame.index.names = [None] * len(frame.index.names)

        if self.dtype is not None:
            # Ignore non-existent columns from dtype mapping
            # like other parsers do
            if isinstance(self.dtype, dict):
                self.dtype = {
                    k: pandas_dtype(v)
                    for k, v in self.dtype.items()
                    if k in frame.columns
                }
            else:
                self.dtype = pandas_dtype(self.dtype)
            try:
                frame = frame.astype(self.dtype)
            except TypeError as e:
                # GH#44901 reraise to keep api consistent
                raise ValueError(e)
        return frame

    def _validate_usecols(self, usecols) -> None:
        if lib.is_list_like(usecols) and not all(isinstance(x, str) for x in usecols):
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be integer "
                "column positions. Pass a list of string column names instead."
            )
        elif callable(usecols):
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be a callable."
            )

    def read(self) -> DataFrame:
        """
        Reads the contents of a CSV file into a DataFrame and
        processes it according to the kwargs passed in the
        constructor.

        Returns
        -------
        DataFrame
            The DataFrame created from the CSV file.
        """
        pa = import_optional_dependency("pyarrow")
        pyarrow_csv = import_optional_dependency("pyarrow.csv")
        self._get_pyarrow_options()

        try:
            convert_options = pyarrow_csv.ConvertOptions(**self.convert_options)
        except TypeError:
            include = self.convert_options.get("include_columns", None)
            if include is not None:
                self._validate_usecols(include)

            nulls = self.convert_options.get("null_values", set())
            if not lib.is_list_like(nulls) or not all(
                isinstance(x, str) for x in nulls
            ):
                raise TypeError(
                    "The 'pyarrow' engine requires all na_values to be strings"
                )

            raise

        try:
            table = pyarrow_csv.read_csv(
                self.src,
                read_options=pyarrow_csv.ReadOptions(**self.read_options),
                parse_options=pyarrow_csv.ParseOptions(**self.parse_options),
                convert_options=convert_options,
            )
        except pa.ArrowInvalid as e:
            raise ParserError(e) from e

        dtype_backend = self.kwds["dtype_backend"]

        # Convert all pa.null() cols -> float64 (non nullable)
        # else Int64 (nullable case, see below)
        if dtype_backend is lib.no_default:
            new_schema = table.schema
            new_type = pa.float64()
            for i, arrow_type in enumerate(table.schema.types):
                if pa.types.is_null(arrow_type):
                    new_schema = new_schema.set(
                        i, new_schema.field(i).with_type(new_type)
                    )

            table = table.cast(new_schema)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "make_block is deprecated",
                DeprecationWarning,
            )
            frame = arrow_table_to_pandas(
                table, dtype_backend=dtype_backend, null_to_int64=True
            )

        return self._finalize_pandas_output(frame)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\events.py ===
"""
wsproto/events
~~~~~~~~~~~~~~

Events that result from processing data on a WebSocket connection.
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Sequence, TypeVar, Union

from .extensions import Extension
from .typing import Headers


class Event(ABC):
    """
    Base class for wsproto events.
    """

    pass  # noqa


@dataclass(frozen=True)
class Request(Event):
    """The beginning of a Websocket connection, the HTTP Upgrade request

    This event is fired when a SERVER connection receives a WebSocket
    handshake request (HTTP with upgrade header).

    Fields:

    .. attribute:: host

       (Required) The hostname, or host header value.

    .. attribute:: target

       (Required) The request target (path and query string)

    .. attribute:: extensions

       The proposed extensions.

    .. attribute:: extra_headers

       The additional request headers, excluding extensions, host, subprotocols,
       and version headers.

    .. attribute:: subprotocols

       A list of the subprotocols proposed in the request, as a list
       of strings.
    """

    host: str
    target: str
    extensions: Union[Sequence[Extension], Sequence[str]] = field(  # type: ignore[assignment]
        default_factory=list
    )
    extra_headers: Headers = field(default_factory=list)
    subprotocols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AcceptConnection(Event):
    """The acceptance of a Websocket upgrade request.

    This event is fired when a CLIENT receives an acceptance response
    from a server. It is also used to accept an upgrade request when
    acting as a SERVER.

    Fields:

    .. attribute:: extra_headers

       Any additional (non websocket related) headers present in the
       acceptance response.

    .. attribute:: subprotocol

       The accepted subprotocol to use.

    """

    subprotocol: Optional[str] = None
    extensions: List[Extension] = field(default_factory=list)
    extra_headers: Headers = field(default_factory=list)


@dataclass(frozen=True)
class RejectConnection(Event):
    """The rejection of a Websocket upgrade request, the HTTP response.

    The ``RejectConnection`` event sends the appropriate HTTP headers to
    communicate to the peer that the handshake has been rejected. You may also
    send an HTTP body by setting the ``has_body`` attribute to ``True`` and then
    sending one or more :class:`RejectData` events after this one. When sending
    a response body, the caller should set the ``Content-Length``,
    ``Content-Type``, and/or ``Transfer-Encoding`` headers as appropriate.

    When receiving a ``RejectConnection`` event, the ``has_body`` attribute will
    in almost all cases be ``True`` (even if the server set it to ``False``) and
    will be followed by at least one ``RejectData`` events, even though the data
    itself might be just ``b""``. (The only scenario in which the caller
    receives a ``RejectConnection`` with ``has_body == False`` is if the peer
    violates sends an informational status code (1xx) other than 101.)

    The ``has_body`` attribute should only be used when receiving the event. (It
    has ) is False the headers must include a
    content-length or transfer encoding.

    Fields:

    .. attribute:: headers (Headers)

       The headers to send with the response.

    .. attribute:: has_body

       This defaults to False, but set to True if there is a body. See
       also :class:`~RejectData`.

    .. attribute:: status_code

       The response status code.

    """

    status_code: int = 400
    headers: Headers = field(default_factory=list)
    has_body: bool = False


@dataclass(frozen=True)
class RejectData(Event):
    """The rejection HTTP response body.

    The caller may send multiple ``RejectData`` events. The final event should
    have the ``body_finished`` attribute set to ``True``.

    Fields:

    .. attribute:: body_finished

       True if this is the final chunk of the body data.

    .. attribute:: data (bytes)

       (Required) The raw body data.

    """

    data: bytes
    body_finished: bool = True


@dataclass(frozen=True)
class CloseConnection(Event):

    """The end of a Websocket connection, represents a closure frame.

    **wsproto does not automatically send a response to a close event.** To
    comply with the RFC you MUST send a close event back to the remote WebSocket
    if you have not already sent one. The :meth:`response` method provides a
    suitable event for this purpose, and you should check if a response needs
    to be sent by checking :func:`wsproto.WSConnection.state`.

    Fields:

    .. attribute:: code

       (Required) The integer close code to indicate why the connection
       has closed.

    .. attribute:: reason

       Additional reasoning for why the connection has closed.

    """

    code: int
    reason: Optional[str] = None

    def response(self) -> "CloseConnection":
        """Generate an RFC-compliant close frame to send back to the peer."""
        return CloseConnection(code=self.code, reason=self.reason)


T = TypeVar("T", bytes, str)


@dataclass(frozen=True)
class Message(Event, Generic[T]):
    """The websocket data message.

    Fields:

    .. attribute:: data

       (Required) The message data as byte string, can be decoded as UTF-8 for
       TEXT messages.  This only represents a single chunk of data and
       not a full WebSocket message.  You need to buffer and
       reassemble these chunks to get the full message.

    .. attribute:: frame_finished

       This has no semantic content, but is provided just in case some
       weird edge case user wants to be able to reconstruct the
       fragmentation pattern of the original stream.

    .. attribute:: message_finished

       True if this frame is the last one of this message, False if
       more frames are expected.

    """

    data: T
    frame_finished: bool = True
    message_finished: bool = True


@dataclass(frozen=True)
class TextMessage(Message[str]):  # pylint: disable=unsubscriptable-object
    """This event is fired when a data frame with TEXT payload is received.

    Fields:

    .. attribute:: data

       The message data as string, This only represents a single chunk
       of data and not a full WebSocket message.  You need to buffer
       and reassemble these chunks to get the full message.

    """

    # https://github.com/python/mypy/issues/5744
    data: str


@dataclass(frozen=True)
class BytesMessage(Message[bytes]):  # pylint: disable=unsubscriptable-object
    """This event is fired when a data frame with BINARY payload is
    received.

    Fields:

    .. attribute:: data

       The message data as byte string, can be decoded as UTF-8 for
       TEXT messages.  This only represents a single chunk of data and
       not a full WebSocket message.  You need to buffer and
       reassemble these chunks to get the full message.
    """

    # https://github.com/python/mypy/issues/5744
    data: bytes


@dataclass(frozen=True)
class Ping(Event):
    """The Ping event can be sent to trigger a ping frame and is fired
    when a Ping is received.

    **wsproto does not automatically send a pong response to a ping event.** To
    comply with the RFC you MUST send a pong even as soon as is practical. The
    :meth:`response` method provides a suitable event for this purpose.

    Fields:

    .. attribute:: payload

       An optional payload to emit with the ping frame.
    """

    payload: bytes = b""

    def response(self) -> "Pong":
        """Generate an RFC-compliant :class:`Pong` response to this ping."""
        return Pong(payload=self.payload)


@dataclass(frozen=True)
class Pong(Event):
    """The Pong event is fired when a Pong is received.

    Fields:

    .. attribute:: payload

       An optional payload to emit with the pong frame.

    """

    payload: bytes = b""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\testutils.py ===
"""Miscellaneous functions for testing masked arrays and subclasses

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
import operator

import numpy as np
import numpy._core.umath as umath
import numpy.testing
from numpy import ndarray
from numpy.testing import (  # noqa: F401
    assert_,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_raises,
    build_err_msg,
)

from .core import filled, getmask, mask_or, masked, masked_array, nomask

__all__masked = [
    'almost', 'approx', 'assert_almost_equal', 'assert_array_almost_equal',
    'assert_array_approx_equal', 'assert_array_compare',
    'assert_array_equal', 'assert_array_less', 'assert_close',
    'assert_equal', 'assert_equal_records', 'assert_mask_equal',
    'assert_not_equal', 'fail_if_array_equal',
    ]

# Include some normal test functions to avoid breaking other projects who
# have mistakenly included them from this file. SciPy is one. That is
# unfortunate, as some of these functions are not intended to work with
# masked arrays. But there was no way to tell before.
from unittest import TestCase  # noqa: F401

__some__from_testing = [
    'TestCase', 'assert_', 'assert_allclose', 'assert_array_almost_equal_nulp',
    'assert_raises'
    ]

__all__ = __all__masked + __some__from_testing  # noqa: PLE0605


def approx(a, b, fill_value=True, rtol=1e-5, atol=1e-8):
    """
    Returns true if all components of a and b are equal to given tolerances.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.  The relative error rtol should
    be positive and << 1.0 The absolute error atol comes into play for
    those elements of b that are very small or zero; it says how small a
    must be also.

    """
    m = mask_or(getmask(a), getmask(b))
    d1 = filled(a)
    d2 = filled(b)
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return np.equal(d1, d2).ravel()
    x = filled(
        masked_array(d1, copy=False, mask=m), fill_value
    ).astype(np.float64)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(np.float64)
    d = np.less_equal(umath.absolute(x - y), atol + rtol * umath.absolute(y))
    return d.ravel()


def almost(a, b, decimal=6, fill_value=True):
    """
    Returns True if a and b are equal up to decimal places.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.

    """
    m = mask_or(getmask(a), getmask(b))
    d1 = filled(a)
    d2 = filled(b)
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return np.equal(d1, d2).ravel()
    x = filled(
        masked_array(d1, copy=False, mask=m), fill_value
    ).astype(np.float64)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(np.float64)
    d = np.around(np.abs(x - y), decimal) <= 10.0 ** (-decimal)
    return d.ravel()


def _assert_equal_on_sequences(actual, desired, err_msg=''):
    """
    Asserts the equality of two non-array sequences.

    """
    assert_equal(len(actual), len(desired), err_msg)
    for k in range(len(desired)):
        assert_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')


def assert_equal_records(a, b):
    """
    Asserts that two records are equal.

    Pretty crude for now.

    """
    assert_equal(a.dtype, b.dtype)
    for f in a.dtype.names:
        (af, bf) = (operator.getitem(a, f), operator.getitem(b, f))
        if not (af is masked) and not (bf is masked):
            assert_equal(operator.getitem(a, f), operator.getitem(b, f))


def assert_equal(actual, desired, err_msg=''):
    """
    Asserts that two items are equal.

    """
    # Case #1: dictionary .....
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(f"{k} not in {actual}")
            assert_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    # Case #2: lists .....
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        return _assert_equal_on_sequences(actual, desired, err_msg='')
    if not (isinstance(actual, ndarray) or isinstance(desired, ndarray)):
        msg = build_err_msg([actual, desired], err_msg,)
        if not desired == actual:
            raise AssertionError(msg)
        return
    # Case #4. arrays or equivalent
    if ((actual is masked) and not (desired is masked)) or \
            ((desired is masked) and not (actual is masked)):
        msg = build_err_msg([actual, desired],
                            err_msg, header='', names=('x', 'y'))
        raise ValueError(msg)
    actual = np.asanyarray(actual)
    desired = np.asanyarray(desired)
    (actual_dtype, desired_dtype) = (actual.dtype, desired.dtype)
    if actual_dtype.char == "S" and desired_dtype.char == "S":
        return _assert_equal_on_sequences(actual.tolist(),
                                          desired.tolist(),
                                          err_msg='')
    return assert_array_equal(actual, desired, err_msg)


def fail_if_equal(actual, desired, err_msg='',):
    """
    Raises an assertion error if two items are equal.

    """
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        fail_if_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            fail_if_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        fail_if_equal(len(actual), len(desired), err_msg)
        for k in range(len(desired)):
            fail_if_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')
        return
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return fail_if_array_equal(actual, desired, err_msg)
    msg = build_err_msg([actual, desired], err_msg)
    if not desired != actual:
        raise AssertionError(msg)


assert_not_equal = fail_if_equal


def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """
    Asserts that two items are almost equal.

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).

    """
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return assert_array_almost_equal(actual, desired, decimal=decimal,
                                         err_msg=err_msg, verbose=verbose)
    msg = build_err_msg([actual, desired],
                        err_msg=err_msg, verbose=verbose)
    if not round(abs(desired - actual), decimal) == 0:
        raise AssertionError(msg)


assert_close = assert_almost_equal


def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         fill_value=True):
    """
    Asserts that comparison between two masked arrays is satisfied.

    The comparison is elementwise.

    """
    # Allocate a common mask and refill
    m = mask_or(getmask(x), getmask(y))
    x = masked_array(x, copy=False, mask=m, keep_mask=False, subok=False)
    y = masked_array(y, copy=False, mask=m, keep_mask=False, subok=False)
    if ((x is masked) and not (y is masked)) or \
            ((y is masked) and not (x is masked)):
        msg = build_err_msg([x, y], err_msg=err_msg, verbose=verbose,
                            header=header, names=('x', 'y'))
        raise ValueError(msg)
    # OK, now run the basic tests on filled versions
    return np.testing.assert_array_compare(comparison,
                                           x.filled(fill_value),
                                           y.filled(fill_value),
                                           err_msg=err_msg,
                                           verbose=verbose, header=header)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """
    Checks the elementwise equality of two masked arrays.

    """
    assert_array_compare(operator.__eq__, x, y,
                         err_msg=err_msg, verbose=verbose,
                         header='Arrays are not equal')


def fail_if_array_equal(x, y, err_msg='', verbose=True):
    """
    Raises an assertion error if two masked arrays are not equal elementwise.

    """
    def compare(x, y):
        return (not np.all(approx(x, y)))
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not equal')


def assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return approx(x, y, rtol=10. ** -decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return almost(x, y, decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')


def assert_array_less(x, y, err_msg='', verbose=True):
    """
    Checks that x is smaller than y elementwise.

    """
    assert_array_compare(operator.__lt__, x, y,
                         err_msg=err_msg, verbose=verbose,
                         header='Arrays are not less-ordered')


def assert_mask_equal(m1, m2, err_msg=''):
    """
    Asserts the equality of two masks.

    """
    if m1 is nomask:
        assert_(m2 is nomask)
    if m2 is nomask:
        assert_(m1 is nomask)
    assert_array_equal(m1, m2, err_msg=err_msg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\parser.py ===
"""Base option parser setup"""

import logging
import optparse
import shutil
import sys
import textwrap
from contextlib import suppress
from typing import Any, Dict, Generator, List, NoReturn, Optional, Tuple

from pip._internal.cli.status_codes import UNKNOWN_ERROR
from pip._internal.configuration import Configuration, ConfigurationError
from pip._internal.utils.misc import redact_auth_from_url, strtobool

logger = logging.getLogger(__name__)


class PrettyHelpFormatter(optparse.IndentedHelpFormatter):
    """A prettier/less verbose help formatter for optparse."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # help position must be aligned with __init__.parseopts.description
        kwargs["max_help_position"] = 30
        kwargs["indent_increment"] = 1
        kwargs["width"] = shutil.get_terminal_size()[0] - 2
        super().__init__(*args, **kwargs)

    def format_option_strings(self, option: optparse.Option) -> str:
        return self._format_option_strings(option)

    def _format_option_strings(
        self, option: optparse.Option, mvarfmt: str = " <{}>", optsep: str = ", "
    ) -> str:
        """
        Return a comma-separated list of option strings and metavars.

        :param option:  tuple of (short opt, long opt), e.g: ('-f', '--format')
        :param mvarfmt: metavar format string
        :param optsep:  separator
        """
        opts = []

        if option._short_opts:
            opts.append(option._short_opts[0])
        if option._long_opts:
            opts.append(option._long_opts[0])
        if len(opts) > 1:
            opts.insert(1, optsep)

        if option.takes_value():
            assert option.dest is not None
            metavar = option.metavar or option.dest.lower()
            opts.append(mvarfmt.format(metavar.lower()))

        return "".join(opts)

    def format_heading(self, heading: str) -> str:
        if heading == "Options":
            return ""
        return heading + ":\n"

    def format_usage(self, usage: str) -> str:
        """
        Ensure there is only one newline between usage and the first heading
        if there is no description.
        """
        msg = "\nUsage: {}\n".format(self.indent_lines(textwrap.dedent(usage), "  "))
        return msg

    def format_description(self, description: Optional[str]) -> str:
        # leave full control over description to us
        if description:
            if hasattr(self.parser, "main"):
                label = "Commands"
            else:
                label = "Description"
            # some doc strings have initial newlines, some don't
            description = description.lstrip("\n")
            # some doc strings have final newlines and spaces, some don't
            description = description.rstrip()
            # dedent, then reindent
            description = self.indent_lines(textwrap.dedent(description), "  ")
            description = f"{label}:\n{description}\n"
            return description
        else:
            return ""

    def format_epilog(self, epilog: Optional[str]) -> str:
        # leave full control over epilog to us
        if epilog:
            return epilog
        else:
            return ""

    def indent_lines(self, text: str, indent: str) -> str:
        new_lines = [indent + line for line in text.split("\n")]
        return "\n".join(new_lines)


class UpdatingDefaultsHelpFormatter(PrettyHelpFormatter):
    """Custom help formatter for use in ConfigOptionParser.

    This is updates the defaults before expanding them, allowing
    them to show up correctly in the help listing.

    Also redact auth from url type options
    """

    def expand_default(self, option: optparse.Option) -> str:
        default_values = None
        if self.parser is not None:
            assert isinstance(self.parser, ConfigOptionParser)
            self.parser._update_defaults(self.parser.defaults)
            assert option.dest is not None
            default_values = self.parser.defaults.get(option.dest)
        help_text = super().expand_default(option)

        if default_values and option.metavar == "URL":
            if isinstance(default_values, str):
                default_values = [default_values]

            # If its not a list, we should abort and just return the help text
            if not isinstance(default_values, list):
                default_values = []

            for val in default_values:
                help_text = help_text.replace(val, redact_auth_from_url(val))

        return help_text


class CustomOptionParser(optparse.OptionParser):
    def insert_option_group(
        self, idx: int, *args: Any, **kwargs: Any
    ) -> optparse.OptionGroup:
        """Insert an OptionGroup at a given position."""
        group = self.add_option_group(*args, **kwargs)

        self.option_groups.pop()
        self.option_groups.insert(idx, group)

        return group

    @property
    def option_list_all(self) -> List[optparse.Option]:
        """Get a list of all options, including those in option groups."""
        res = self.option_list[:]
        for i in self.option_groups:
            res.extend(i.option_list)

        return res


class ConfigOptionParser(CustomOptionParser):
    """Custom option parser which updates its defaults by checking the
    configuration files and environmental variables"""

    def __init__(
        self,
        *args: Any,
        name: str,
        isolated: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.config = Configuration(isolated)

        assert self.name
        super().__init__(*args, **kwargs)

    def check_default(self, option: optparse.Option, key: str, val: Any) -> Any:
        try:
            return option.check_value(key, val)
        except optparse.OptionValueError as exc:
            print(f"An error occurred during configuration: {exc}")
            sys.exit(3)

    def _get_ordered_configuration_items(
        self,
    ) -> Generator[Tuple[str, Any], None, None]:
        # Configuration gives keys in an unordered manner. Order them.
        override_order = ["global", self.name, ":env:"]

        # Pool the options into different groups
        section_items: Dict[str, List[Tuple[str, Any]]] = {
            name: [] for name in override_order
        }
        for section_key, val in self.config.items():
            # ignore empty values
            if not val:
                logger.debug(
                    "Ignoring configuration key '%s' as it's value is empty.",
                    section_key,
                )
                continue

            section, key = section_key.split(".", 1)
            if section in override_order:
                section_items[section].append((key, val))

        # Yield each group in their override order
        for section in override_order:
            for key, val in section_items[section]:
                yield key, val

    def _update_defaults(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the given defaults with values from the config files and
        the environ. Does a little special handling for certain types of
        options (lists)."""

        # Accumulate complex default state.
        self.values = optparse.Values(self.defaults)
        late_eval = set()
        # Then set the options with those values
        for key, val in self._get_ordered_configuration_items():
            # '--' because configuration supports only long names
            option = self.get_option("--" + key)

            # Ignore options not present in this parser. E.g. non-globals put
            # in [global] by users that want them to apply to all applicable
            # commands.
            if option is None:
                continue

            assert option.dest is not None

            if option.action in ("store_true", "store_false"):
                try:
                    val = strtobool(val)
                except ValueError:
                    self.error(
                        f"{val} is not a valid value for {key} option, "
                        "please specify a boolean value like yes/no, "
                        "true/false or 1/0 instead."
                    )
            elif option.action == "count":
                with suppress(ValueError):
                    val = strtobool(val)
                with suppress(ValueError):
                    val = int(val)
                if not isinstance(val, int) or val < 0:
                    self.error(
                        f"{val} is not a valid value for {key} option, "
                        "please instead specify either a non-negative integer "
                        "or a boolean value like yes/no or false/true "
                        "which is equivalent to 1/0."
                    )
            elif option.action == "append":
                val = val.split()
                val = [self.check_default(option, key, v) for v in val]
            elif option.action == "callback":
                assert option.callback is not None
                late_eval.add(option.dest)
                opt_str = option.get_opt_string()
                val = option.convert_value(opt_str, val)
                # From take_action
                args = option.callback_args or ()
                kwargs = option.callback_kwargs or {}
                option.callback(option, opt_str, val, self, *args, **kwargs)
            else:
                val = self.check_default(option, key, val)

            defaults[option.dest] = val

        for key in late_eval:
            defaults[key] = getattr(self.values, key)
        self.values = None
        return defaults

    def get_default_values(self) -> optparse.Values:
        """Overriding to make updating the defaults after instantiation of
        the option parser possible, _update_defaults() does the dirty work."""
        if not self.process_default_values:
            # Old, pre-Optik 1.5 behaviour.
            return optparse.Values(self.defaults)

        # Load the configuration, or error out in case of an error
        try:
            self.config.load()
        except ConfigurationError as err:
            self.exit(UNKNOWN_ERROR, str(err))

        defaults = self._update_defaults(self.defaults.copy())  # ours
        for option in self._get_all_options():
            assert option.dest is not None
            default = defaults.get(option.dest)
            if isinstance(default, str):
                opt_str = option.get_opt_string()
                defaults[option.dest] = option.check_value(opt_str, default)
        return optparse.Values(defaults)

    def error(self, msg: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(UNKNOWN_ERROR, f"{msg}\n")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\group_numbers.py ===
from itertools import chain, combinations

from sympy.external.gmpy import gcd
from sympy.ntheory.factor_ import factorint
from sympy.utilities.misc import as_int


def _is_nilpotent_number(factors: dict) -> bool:
    """ Check whether `n` is a nilpotent number.
    Note that ``factors`` is a prime factorization of `n`.

    This is a low-level helper for ``is_nilpotent_number``, for internal use.
    """
    for p in factors.keys():
        for q, e in factors.items():
            # We want to calculate
            # any(pow(q, k, p) == 1 for k in range(1, e + 1))
            m = 1
            for _ in range(e):
                m = m*q % p
                if m == 1:
                    return False
    return True


def is_nilpotent_number(n) -> bool:
    """
    Check whether `n` is a nilpotent number. A number `n` is said to be
    nilpotent if and only if every finite group of order `n` is nilpotent.
    For more information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_nilpotent_number
    >>> from sympy import randprime
    >>> is_nilpotent_number(21)
    False
    >>> is_nilpotent_number(randprime(1, 30)**12)
    True

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A056867

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    return _is_nilpotent_number(factorint(n))


def is_abelian_number(n) -> bool:
    """
    Check whether `n` is an abelian number. A number `n` is said to be abelian
    if and only if every finite group of order `n` is abelian. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_abelian_number
    >>> from sympy import randprime
    >>> is_abelian_number(4)
    True
    >>> is_abelian_number(randprime(1, 2000)**2)
    True
    >>> is_abelian_number(60)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A051532

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    factors = factorint(n)
    return all(e < 3 for e in factors.values()) and _is_nilpotent_number(factors)


def is_cyclic_number(n) -> bool:
    """
    Check whether `n` is a cyclic number. A number `n` is said to be cyclic
    if and only if every finite group of order `n` is cyclic. For more
    information see [1]_.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import is_cyclic_number
    >>> from sympy import randprime
    >>> is_cyclic_number(15)
    True
    >>> is_cyclic_number(randprime(1, 2000)**2)
    False
    >>> is_cyclic_number(4)
    False

    References
    ==========

    .. [1] Pakianathan, J., Shankar, K., Nilpotent Numbers,
           The American Mathematical Monthly, 107(7), 631-634.
    .. [2] https://oeis.org/A003277

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    factors = factorint(n)
    return all(e == 1 for e in factors.values()) and _is_nilpotent_number(factors)


def _holder_formula(prime_factors):
    r""" Number of groups of order `n`.
    where `n` is squarefree and its prime factors are ``prime_factors``.
    i.e., ``n == math.prod(prime_factors)``

    Explanation
    ===========

    When `n` is squarefree, the number of groups of order `n` is expressed by

    .. math ::
        \sum_{d \mid n} \prod_p \frac{p^{c(p, d)} - 1}{p - 1}

    where `n=de`, `p` is the prime factor of `e`,
    and `c(p, d)` is the number of prime factors `q` of `d` such that `q \equiv 1 \pmod{p}` [2]_.

    The formula is elegant, but can be improved when implemented as an algorithm.
    Since `n` is assumed to be squarefree, the divisor `d` of `n` can be identified with the power set of prime factors.
    We let `N` be the set of prime factors of `n`.
    `F = \{p \in N : \forall q \in N, q \not\equiv 1 \pmod{p} \}, M = N \setminus F`, we have the following.

    .. math ::
        \sum_{d \in 2^{M}} \prod_{p \in M \setminus d} \frac{p^{c(p, F \cup d)} - 1}{p - 1}

    Practically, many prime factors are expected to be members of `F`, thus reducing computation time.

    Parameters
    ==========

    prime_factors : set
        The set of prime factors of ``n``. where `n` is squarefree.

    Returns
    =======

    int : Number of groups of order ``n``

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import _holder_formula
    >>> _holder_formula({2}) # n = 2
    1
    >>> _holder_formula({2, 3}) # n = 2*3 = 6
    2

    See Also
    ========

    groups_count

    References
    ==========

    .. [1] Otto Holder, Die Gruppen der Ordnungen p^3, pq^2, pqr, p^4,
           Math. Ann. 43 pp. 301-412 (1893).
           http://dx.doi.org/10.1007/BF01443651
    .. [2] John H. Conway, Heiko Dietrich and E.A. O'Brien,
           Counting groups: gnus, moas and other exotica
           The Mathematical Intelligencer 30, 6-15 (2008)
           https://doi.org/10.1007/BF02985731

    """
    F = {p for p in prime_factors if all(q % p != 1 for q in prime_factors)}
    M = prime_factors - F

    s = 0
    powerset = chain.from_iterable(combinations(M, r) for r in range(len(M)+1))
    for ps in powerset:
        ps = set(ps)
        prod = 1
        for p in M - ps:
            c = len([q for q in F | ps if q % p == 1])
            prod *= (p**c - 1) // (p - 1)
            if not prod:
                break
        s += prod
    return s


def groups_count(n):
    r""" Number of groups of order `n`.
    In [1]_, ``gnu(n)`` is given, so we follow this notation here as well.

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer

    Returns
    =======

    int : ``gnu(n)``

    Raises
    ======

    ValueError
        Number of groups of order ``n`` is unknown or not implemented.
        For example, gnu(`2^{11}`) is not yet known.
        On the other hand, gnu(99) is known to be 2,
        but this has not yet been implemented in this function.

    Examples
    ========

    >>> from sympy.combinatorics.group_numbers import groups_count
    >>> groups_count(3) # There is only one cyclic group of order 3
    1
    >>> # There are two groups of order 10: the cyclic group and the dihedral group
    >>> groups_count(10)
    2

    See Also
    ========

    is_cyclic_number
        `n` is cyclic iff gnu(n) = 1

    References
    ==========

    .. [1] John H. Conway, Heiko Dietrich and E.A. O'Brien,
           Counting groups: gnus, moas and other exotica
           The Mathematical Intelligencer 30, 6-15 (2008)
           https://doi.org/10.1007/BF02985731
    .. [2] https://oeis.org/A000001

    """
    n = as_int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer, not %i" % n)
    factors = factorint(n)
    if len(factors) == 1:
        (p, e) = list(factors.items())[0]
        if p == 2:
            A000679 = [1, 1, 2, 5, 14, 51, 267, 2328, 56092, 10494213, 49487367289]
            if e < len(A000679):
                return A000679[e]
        if p == 3:
            A090091 = [1, 1, 2, 5, 15, 67, 504, 9310, 1396077, 5937876645]
            if e < len(A090091):
                return A090091[e]
        if e <= 2: # gnu(p) = 1, gnu(p**2) = 2
            return e
        if e == 3: # gnu(p**3) = 5
            return 5
        if e == 4: # if p is an odd prime, gnu(p**4) = 15
            return 15
        if e == 5: # if p >= 5, gnu(p**5) is expressed by the following equation
            return 61 + 2*p + 2*gcd(p-1, 3) + gcd(p-1, 4)
        if e == 6: # if p >= 6, gnu(p**6) is expressed by the following equation
            return 3*p**2 + 39*p + 344 +\
                  24*gcd(p-1, 3) + 11*gcd(p-1, 4) + 2*gcd(p-1, 5)
        if e == 7: # if p >= 7, gnu(p**7) is expressed by the following equation
            if p == 5:
                return 34297
            return 3*p**5 + 12*p**4 + 44*p**3 + 170*p**2 + 707*p + 2455 +\
                  (4*p**2 + 44*p + 291)*gcd(p-1, 3) + (p**2 + 19*p + 135)*gcd(p-1, 4) + \
                  (3*p + 31)*gcd(p-1, 5) + 4*gcd(p-1, 7) + 5*gcd(p-1, 8) + gcd(p-1, 9)
    if any(e > 1 for e in factors.values()): # n is not squarefree
        # some known values for small n that have more than 1 factor and are not square free (https://oeis.org/A000001)
        small = {12: 5, 18: 5, 20: 5, 24: 15, 28: 4, 36: 14, 40: 14, 44: 4, 45: 2, 48: 52,
                50: 5, 52: 5, 54: 15, 56: 13, 60: 13, 63: 4, 68: 5, 72: 50, 75: 3, 76: 4,
                80: 52, 84: 15, 88: 12, 90: 10, 92: 4}
        if n in small:
            return small[n]
        raise ValueError("Number of groups of order n is unknown or not implemented")
    if len(factors) == 2: # n is squarefree semiprime
        p, q = sorted(factors.keys())
        return 2 if q % p == 1 else 1
    return _holder_formula(set(factors.keys()))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\dot.py ===
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.printing.repr import srepr

__all__ = ['dotprint']

default_styles = (
    (Basic, {'color': 'blue', 'shape': 'ellipse'}),
    (Expr,  {'color': 'black'})
)

slotClasses = (Symbol, Integer, Rational, Float)
def purestr(x, with_args=False):
    """A string that follows ```obj = type(obj)(*obj.args)``` exactly.

    Parameters
    ==========

    with_args : boolean, optional
        If ``True``, there will be a second argument for the return
        value, which is a tuple containing ``purestr`` applied to each
        of the subnodes.

        If ``False``, there will not be a second argument for the
        return.

        Default is ``False``

    Examples
    ========

    >>> from sympy import Float, Symbol, MatrixSymbol
    >>> from sympy import Integer # noqa: F401
    >>> from sympy.core.symbol import Str # noqa: F401
    >>> from sympy.printing.dot import purestr

    Applying ``purestr`` for basic symbolic object:
    >>> code = purestr(Symbol('x'))
    >>> code
    "Symbol('x')"
    >>> eval(code) == Symbol('x')
    True

    For basic numeric object:
    >>> purestr(Float(2))
    "Float('2.0', precision=53)"

    For matrix symbol:
    >>> code = purestr(MatrixSymbol('x', 2, 2))
    >>> code
    "MatrixSymbol(Str('x'), Integer(2), Integer(2))"
    >>> eval(code) == MatrixSymbol('x', 2, 2)
    True

    With ``with_args=True``:
    >>> purestr(Float(2), with_args=True)
    ("Float('2.0', precision=53)", ())
    >>> purestr(MatrixSymbol('x', 2, 2), with_args=True)
    ("MatrixSymbol(Str('x'), Integer(2), Integer(2))",
     ("Str('x')", 'Integer(2)', 'Integer(2)'))
    """
    sargs = ()
    if not isinstance(x, Basic):
        rv = str(x)
    elif not x.args:
        rv = srepr(x)
    else:
        args = x.args
        sargs = tuple(map(purestr, args))
        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
    if with_args:
        rv = rv, sargs
    return rv


def styleof(expr, styles=default_styles):
    """ Merge style dictionaries in order

    Examples
    ========

    >>> from sympy import Symbol, Basic, Expr, S
    >>> from sympy.printing.dot import styleof
    >>> styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
    ...           (Expr,  {'color': 'black'})]

    >>> styleof(Basic(S(1)), styles)
    {'color': 'blue', 'shape': 'ellipse'}

    >>> x = Symbol('x')
    >>> styleof(x + 1, styles)  # this is an Expr
    {'color': 'black', 'shape': 'ellipse'}
    """
    style = {}
    for typ, sty in styles:
        if isinstance(expr, typ):
            style.update(sty)
    return style


def attrprint(d, delimiter=', '):
    """ Print a dictionary of attributes

    Examples
    ========

    >>> from sympy.printing.dot import attrprint
    >>> print(attrprint({'color': 'blue', 'shape': 'ellipse'}))
    "color"="blue", "shape"="ellipse"
    """
    return delimiter.join('"%s"="%s"'%item for item in sorted(d.items()))


def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
    """ String defining a node

    Examples
    ========

    >>> from sympy.printing.dot import dotnode
    >>> from sympy.abc import x
    >>> print(dotnode(x))
    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
    """
    style = styleof(expr, styles)

    if isinstance(expr, Basic) and not expr.is_Atom:
        label = str(expr.__class__.__name__)
    else:
        label = labelfunc(expr)
    style['label'] = label
    expr_str = purestr(expr)
    if repeat:
        expr_str += '_%s' % str(pos)
    return '"%s" [%s];' % (expr_str, attrprint(style))


def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True):
    """ List of strings for all expr->expr.arg pairs

    See the docstring of dotprint for explanations of the options.

    Examples
    ========

    >>> from sympy.printing.dot import dotedges
    >>> from sympy.abc import x
    >>> for e in dotedges(x+2):
    ...     print(e)
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    """
    if atom(expr):
        return []
    else:
        expr_str, arg_strs = purestr(expr, with_args=True)
        if repeat:
            expr_str += '_%s' % str(pos)
            arg_strs = ['%s_%s' % (a, str(pos + (i,)))
                for i, a in enumerate(arg_strs)]
        return ['"%s" -> "%s";' % (expr_str, a) for a in arg_strs]

template = \
"""digraph{

# Graph style
%(graphstyle)s

#########
# Nodes #
#########

%(nodes)s

#########
# Edges #
#########

%(edges)s
}"""

_graphstyle = {'rankdir': 'TD', 'ordering': 'out'}

def dotprint(expr,
    styles=default_styles, atom=lambda x: not isinstance(x, Basic),
    maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    """DOT description of a SymPy expression tree

    Parameters
    ==========

    styles : list of lists composed of (Class, mapping), optional
        Styles for different classes.

        The default is

        .. code-block:: python

            (
                (Basic, {'color': 'blue', 'shape': 'ellipse'}),
                (Expr,  {'color': 'black'})
            )

    atom : function, optional
        Function used to determine if an arg is an atom.

        A good choice is ``lambda x: not x.args``.

        The default is ``lambda x: not isinstance(x, Basic)``.

    maxdepth : integer, optional
        The maximum depth.

        The default is ``None``, meaning no limit.

    repeat : boolean, optional
        Whether to use different nodes for common subexpressions.

        The default is ``True``.

        For example, for ``x + x*y`` with ``repeat=True``, it will have
        two nodes for ``x``; with ``repeat=False``, it will have one
        node.

        .. warning::
            Even if a node appears twice in the same object like ``x`` in
            ``Pow(x, x)``, it will still only appear once.
            Hence, with ``repeat=False``, the number of arrows out of an
            object might not equal the number of args it has.

    labelfunc : function, optional
        A function to create a label for a given leaf node.

        The default is ``str``.

        Another good option is ``srepr``.

        For example with ``str``, the leaf nodes of ``x + 1`` are labeled,
        ``x`` and ``1``.  With ``srepr``, they are labeled ``Symbol('x')``
        and ``Integer(1)``.

    **kwargs : optional
        Additional keyword arguments are included as styles for the graph.

    Examples
    ========

    >>> from sympy import dotprint
    >>> from sympy.abc import x
    >>> print(dotprint(x+2)) # doctest: +NORMALIZE_WHITESPACE
    digraph{
    <BLANKLINE>
    # Graph style
    "ordering"="out"
    "rankdir"="TD"
    <BLANKLINE>
    #########
    # Nodes #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" ["color"="black", "label"="Add", "shape"="ellipse"];
    "Integer(2)_(0,)" ["color"="black", "label"="2", "shape"="ellipse"];
    "Symbol('x')_(1,)" ["color"="black", "label"="x", "shape"="ellipse"];
    <BLANKLINE>
    #########
    # Edges #
    #########
    <BLANKLINE>
    "Add(Integer(2), Symbol('x'))_()" -> "Integer(2)_(0,)";
    "Add(Integer(2), Symbol('x'))_()" -> "Symbol('x')_(1,)";
    }

    """
    # repeat works by adding a signature tuple to the end of each node for its
    # position in the graph. For example, for expr = Add(x, Pow(x, 2)), the x in the
    # Pow will have the tuple (1, 0), meaning it is expr.args[1].args[0].
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)

    nodes = []
    edges = []
    def traverse(e, depth, pos=()):
        nodes.append(dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat))
        if maxdepth and depth >= maxdepth:
            return
        edges.extend(dotedges(e, atom=atom, pos=pos, repeat=repeat))
        [traverse(arg, depth+1, pos + (i,)) for i, arg in enumerate(e.args) if not atom(arg)]
    traverse(expr, 0)

    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_io_kqueue.py ===
from __future__ import annotations

import errno
import select
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import attrs
import outcome

from .. import _core
from ._run import _public
from ._wakeup_socketpair import WakeupSocketpair

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from typing_extensions import TypeAlias

    from .._core import Abort, RaiseCancelT, Task, UnboundedQueue
    from .._file_io import _HasFileNo

assert not TYPE_CHECKING or (sys.platform != "linux" and sys.platform != "win32")

EventResult: TypeAlias = "list[select.kevent]"


@attrs.frozen(eq=False)
class _KqueueStatistics:
    tasks_waiting: int
    monitors: int
    backend: Literal["kqueue"] = attrs.field(init=False, default="kqueue")


@attrs.define(eq=False)
class KqueueIOManager:
    _kqueue: select.kqueue = attrs.Factory(select.kqueue)
    # {(ident, filter): Task or UnboundedQueue}
    _registered: dict[tuple[int, int], Task | UnboundedQueue[select.kevent]] = (
        attrs.Factory(dict)
    )
    _force_wakeup: WakeupSocketpair = attrs.Factory(WakeupSocketpair)
    _force_wakeup_fd: int | None = None

    def __attrs_post_init__(self) -> None:
        force_wakeup_event = select.kevent(
            self._force_wakeup.wakeup_sock,
            select.KQ_FILTER_READ,
            select.KQ_EV_ADD,
        )
        self._kqueue.control([force_wakeup_event], 0)
        self._force_wakeup_fd = self._force_wakeup.wakeup_sock.fileno()

    def statistics(self) -> _KqueueStatistics:
        tasks_waiting = 0
        monitors = 0
        for receiver in self._registered.values():
            if type(receiver) is _core.Task:
                tasks_waiting += 1
            else:
                monitors += 1
        return _KqueueStatistics(tasks_waiting=tasks_waiting, monitors=monitors)

    def close(self) -> None:
        self._kqueue.close()
        self._force_wakeup.close()

    def force_wakeup(self) -> None:
        self._force_wakeup.wakeup_thread_and_signal_safe()

    def get_events(self, timeout: float) -> EventResult:
        # max_events must be > 0 or kqueue gets cranky
        # and we generally want this to be strictly larger than the actual
        # number of events we get, so that we can tell that we've gotten
        # all the events in just 1 call.
        max_events = len(self._registered) + 1
        events = []
        while True:
            batch = self._kqueue.control([], max_events, timeout)
            events += batch
            if len(batch) < max_events:
                break
            else:  # TODO: test this line
                timeout = 0
                # and loop back to the start
        return events

    def process_events(self, events: EventResult) -> None:
        for event in events:
            key = (event.ident, event.filter)
            if event.ident == self._force_wakeup_fd:
                self._force_wakeup.drain()
                continue
            receiver = self._registered[key]
            if event.flags & select.KQ_EV_ONESHOT:  # TODO: test this branch
                del self._registered[key]
            if isinstance(receiver, _core.Task):
                _core.reschedule(receiver, outcome.Value(event))
            else:
                receiver.put_nowait(event)  # TODO: test this line

    # kevent registration is complicated -- e.g. aio submission can
    # implicitly perform a EV_ADD, and EVFILT_PROC with NOTE_TRACK will
    # automatically register filters for child processes. So our lowlevel
    # API is *very* low-level: we expose the kqueue itself for adding
    # events or sticking into AIO submission structs, and split waiting
    # off into separate methods. It's your responsibility to make sure
    # that handle_io never receives an event without a corresponding
    # registration! This may be challenging if you want to be careful
    # about e.g. KeyboardInterrupt. Possibly this API could be improved to
    # be more ergonomic...

    @_public
    def current_kqueue(self) -> select.kqueue:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__.
        """
        return self._kqueue

    @contextmanager
    @_public
    def monitor_kevent(
        self,
        ident: int,
        filter: int,
    ) -> Iterator[_core.UnboundedQueue[select.kevent]]:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__.
        """
        key = (ident, filter)
        if key in self._registered:
            raise _core.BusyResourceError(
                "attempt to register multiple listeners for same ident/filter pair",
            )
        q = _core.UnboundedQueue[select.kevent]()
        self._registered[key] = q
        try:
            yield q
        finally:
            del self._registered[key]

    @_public
    async def wait_kevent(
        self,
        ident: int,
        filter: int,
        abort_func: Callable[[RaiseCancelT], Abort],
    ) -> Abort:
        """TODO: these are implemented, but are currently more of a sketch than
        anything real. See `#26
        <https://github.com/python-trio/trio/issues/26>`__.
        """
        key = (ident, filter)
        if key in self._registered:
            raise _core.BusyResourceError(
                "attempt to register multiple listeners for same ident/filter pair",
            )
        self._registered[key] = _core.current_task()

        def abort(raise_cancel: RaiseCancelT) -> Abort:
            r = abort_func(raise_cancel)
            if r is _core.Abort.SUCCEEDED:  # TODO: test this branch
                del self._registered[key]
            return r

        # wait_task_rescheduled does not have its return type typed
        return await _core.wait_task_rescheduled(abort)  # type: ignore[no-any-return]

    async def _wait_common(
        self,
        fd: int | _HasFileNo,
        filter: int,
    ) -> None:
        if not isinstance(fd, int):
            fd = fd.fileno()
        flags = select.KQ_EV_ADD | select.KQ_EV_ONESHOT
        event = select.kevent(fd, filter, flags)
        self._kqueue.control([event], 0)

        def abort(_: RaiseCancelT) -> Abort:
            event = select.kevent(fd, filter, select.KQ_EV_DELETE)
            try:
                self._kqueue.control([event], 0)
            except OSError as exc:
                # kqueue tracks individual fds (*not* the underlying file
                # object, see _io_epoll.py for a long discussion of why this
                # distinction matters), and automatically deregisters an event
                # if the fd is closed. So if kqueue.control says that it
                # doesn't know about this event, then probably it's because
                # the fd was closed behind our backs. (Too bad we can't ask it
                # to wake us up when this happens, versus discovering it after
                # the fact... oh well, you can't have everything.)
                #
                # FreeBSD reports this using EBADF. macOS uses ENOENT.
                if exc.errno in (errno.EBADF, errno.ENOENT):  # pragma: no branch
                    pass
                else:  # pragma: no cover
                    # As far as we know, this branch can't happen.
                    raise
            return _core.Abort.SUCCEEDED

        await self.wait_kevent(fd, filter, abort)

    @_public
    async def wait_readable(self, fd: int | _HasFileNo) -> None:
        """Block until the kernel reports that the given object is readable.

        On Unix systems, ``fd`` must either be an integer file descriptor,
        or else an object with a ``.fileno()`` method which returns an
        integer file descriptor. Any kind of file descriptor can be passed,
        though the exact semantics will depend on your kernel. For example,
        this probably won't do anything useful for on-disk files.

        On Windows systems, ``fd`` must either be an integer ``SOCKET``
        handle, or else an object with a ``.fileno()`` method which returns
        an integer ``SOCKET`` handle. File descriptors aren't supported,
        and neither are handles that refer to anything besides a
        ``SOCKET``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become readable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._wait_common(fd, select.KQ_FILTER_READ)

    @_public
    async def wait_writable(self, fd: int | _HasFileNo) -> None:
        """Block until the kernel reports that the given object is writable.

        See `wait_readable` for the definition of ``fd``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become writable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._wait_common(fd, select.KQ_FILTER_WRITE)

    @_public
    def notify_closing(self, fd: int | _HasFileNo) -> None:
        """Notify waiters of the given object that it will be closed.

        Call this before closing a file descriptor (on Unix) or socket (on
        Windows). This will cause any `wait_readable` or `wait_writable`
        calls on the given object to immediately wake up and raise
        `~trio.ClosedResourceError`.

        This doesn't actually close the object – you still have to do that
        yourself afterwards. Also, you want to be careful to make sure no
        new tasks start waiting on the object in between when you call this
        and when it's actually closed. So to close something properly, you
        usually want to do these steps in order:

        1. Explicitly mark the object as closed, so that any new attempts
           to use it will abort before they start.
        2. Call `notify_closing` to wake up any already-existing users.
        3. Actually close the object.

        It's also possible to do them in a different order if that's more
        convenient, *but only if* you make sure not to have any checkpoints in
        between the steps. This way they all happen in a single atomic
        step, so other tasks won't be able to tell what order they happened
        in anyway.
        """
        if not isinstance(fd, int):
            fd = fd.fileno()

        for filter_ in [select.KQ_FILTER_READ, select.KQ_FILTER_WRITE]:
            key = (fd, filter_)
            receiver = self._registered.get(key)

            if receiver is None:
                continue

            if type(receiver) is _core.Task:
                event = select.kevent(fd, filter_, select.KQ_EV_DELETE)
                self._kqueue.control([event], 0)
                exc = _core.ClosedResourceError("another task closed this fd")
                _core.reschedule(receiver, outcome.Error(exc))
                del self._registered[key]
            else:
                # XX this is an interesting example of a case where being able
                # to close a queue would be useful...
                raise NotImplementedError(
                    "can't close an fd that monitor_kevent is using",
                )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\syslog.py ===
# Easy to use system logging for Python's logging module.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: December 10, 2020
# URL: https://coloredlogs.readthedocs.io

"""
Easy to use UNIX system logging for Python's :mod:`logging` module.

Admittedly system logging has little to do with colored terminal output, however:

- The `coloredlogs` package is my attempt to do Python logging right and system
  logging is an important part of that equation.

- I've seen a surprising number of quirks and mistakes in system logging done
  in Python, for example including ``%(asctime)s`` in a format string (the
  system logging daemon is responsible for adding timestamps and thus you end
  up with duplicate timestamps that make the logs awful to read :-).

- The ``%(programname)s`` filter originated in my system logging code and I
  wanted it in `coloredlogs` so the step to include this module wasn't that big.

- As a bonus this Python module now has a test suite and proper documentation.

So there :-P. Go take a look at :func:`enable_system_logging()`.
"""

# Standard library modules.
import logging
import logging.handlers
import os
import socket
import sys

# External dependencies.
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows

# Modules included in our package.
from coloredlogs import (
    DEFAULT_LOG_LEVEL,
    ProgramNameFilter,
    adjust_level,
    find_program_name,
    level_to_number,
    replace_handler,
)

LOG_DEVICE_MACOSX = '/var/run/syslog'
"""The pathname of the log device on Mac OS X (a string)."""

LOG_DEVICE_UNIX = '/dev/log'
"""The pathname of the log device on Linux and most other UNIX systems (a string)."""

DEFAULT_LOG_FORMAT = '%(programname)s[%(process)d]: %(levelname)s %(message)s'
"""
The default format for log messages sent to the system log (a string).

The ``%(programname)s`` format requires :class:`~coloredlogs.ProgramNameFilter`
but :func:`enable_system_logging()` takes care of this for you.

The ``name[pid]:`` construct (specifically the colon) in the format allows
rsyslogd_ to extract the ``$programname`` from each log message, which in turn
allows configuration files in ``/etc/rsyslog.d/*.conf`` to filter these log
messages to a separate log file (if the need arises).

.. _rsyslogd: https://en.wikipedia.org/wiki/Rsyslog
"""

# Initialize a logger for this module.
logger = logging.getLogger(__name__)


class SystemLogging(object):

    """Context manager to enable system logging."""

    def __init__(self, *args, **kw):
        """
        Initialize a :class:`SystemLogging` object.

        :param args: Positional arguments to :func:`enable_system_logging()`.
        :param kw: Keyword arguments to :func:`enable_system_logging()`.
        """
        self.args = args
        self.kw = kw
        self.handler = None

    def __enter__(self):
        """Enable system logging when entering the context."""
        if self.handler is None:
            self.handler = enable_system_logging(*self.args, **self.kw)
        return self.handler

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """
        Disable system logging when leaving the context.

        .. note:: If an exception is being handled when we leave the context a
                  warning message including traceback is logged *before* system
                  logging is disabled.
        """
        if self.handler is not None:
            if exc_type is not None:
                logger.warning("Disabling system logging due to unhandled exception!", exc_info=True)
            (self.kw.get('logger') or logging.getLogger()).removeHandler(self.handler)
            self.handler = None


def enable_system_logging(programname=None, fmt=None, logger=None, reconfigure=True, **kw):
    """
    Redirect :mod:`logging` messages to the system log (e.g. ``/var/log/syslog``).

    :param programname: The program name to embed in log messages (a string, defaults
                         to the result of :func:`~coloredlogs.find_program_name()`).
    :param fmt: The log format for system log messages (a string, defaults to
                :data:`DEFAULT_LOG_FORMAT`).
    :param logger: The logger to which the :class:`~logging.handlers.SysLogHandler`
                   should be connected (defaults to the root logger).
    :param level: The logging level for the :class:`~logging.handlers.SysLogHandler`
                  (defaults to :data:`.DEFAULT_LOG_LEVEL`). This value is coerced
                  using :func:`~coloredlogs.level_to_number()`.
    :param reconfigure: If :data:`True` (the default) multiple calls to
                        :func:`enable_system_logging()` will each override
                        the previous configuration.
    :param kw: Refer to :func:`connect_to_syslog()`.
    :returns: A :class:`~logging.handlers.SysLogHandler` object or
              :data:`None`. If an existing handler is found and `reconfigure`
              is :data:`False` the existing handler object is returned. If the
              connection to the system logging daemon fails :data:`None` is
              returned.

    As of release 15.0 this function uses :func:`is_syslog_supported()` to
    check whether system logging is supported and appropriate before it's
    enabled.

    .. note:: When the logger's effective level is too restrictive it is
              relaxed (refer to `notes about log levels`_ for details).
    """
    # Check whether system logging is supported / appropriate.
    if not is_syslog_supported():
        return None
    # Provide defaults for omitted arguments.
    programname = programname or find_program_name()
    logger = logger or logging.getLogger()
    fmt = fmt or DEFAULT_LOG_FORMAT
    level = level_to_number(kw.get('level', DEFAULT_LOG_LEVEL))
    # Check whether system logging is already enabled.
    handler, logger = replace_handler(logger, match_syslog_handler, reconfigure)
    # Make sure reconfiguration is allowed or not relevant.
    if not (handler and not reconfigure):
        # Create a system logging handler.
        handler = connect_to_syslog(**kw)
        # Make sure the handler was successfully created.
        if handler:
            # Enable the use of %(programname)s.
            ProgramNameFilter.install(handler=handler, fmt=fmt, programname=programname)
            # Connect the formatter, handler and logger.
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
            # Adjust the level of the selected logger.
            adjust_level(logger, level)
    return handler


def connect_to_syslog(address=None, facility=None, level=None):
    """
    Create a :class:`~logging.handlers.SysLogHandler`.

    :param address: The device file or network address of the system logging
                    daemon (a string or tuple, defaults to the result of
                    :func:`find_syslog_address()`).
    :param facility: Refer to :class:`~logging.handlers.SysLogHandler`.
                     Defaults to ``LOG_USER``.
    :param level: The logging level for the :class:`~logging.handlers.SysLogHandler`
                  (defaults to :data:`.DEFAULT_LOG_LEVEL`). This value is coerced
                  using :func:`~coloredlogs.level_to_number()`.
    :returns: A :class:`~logging.handlers.SysLogHandler` object or :data:`None` (if the
              system logging daemon is unavailable).

    The process of connecting to the system logging daemon goes as follows:

    - The following two socket types are tried (in decreasing preference):

       1. :data:`~socket.SOCK_RAW` avoids truncation of log messages but may
          not be supported.
       2. :data:`~socket.SOCK_STREAM` (TCP) supports longer messages than the
          default (which is UDP).
    """
    if not address:
        address = find_syslog_address()
    if facility is None:
        facility = logging.handlers.SysLogHandler.LOG_USER
    if level is None:
        level = DEFAULT_LOG_LEVEL
    for socktype in socket.SOCK_RAW, socket.SOCK_STREAM, None:
        kw = dict(facility=facility, address=address)
        if socktype is not None:
            kw['socktype'] = socktype
        try:
            handler = logging.handlers.SysLogHandler(**kw)
        except IOError:
            # IOError is a superclass of socket.error which can be raised if the system
            # logging daemon is unavailable.
            pass
        else:
            handler.setLevel(level_to_number(level))
            return handler


def find_syslog_address():
    """
    Find the most suitable destination for system log messages.

    :returns: The pathname of a log device (a string) or an address/port tuple as
              supported by :class:`~logging.handlers.SysLogHandler`.

    On Mac OS X this prefers :data:`LOG_DEVICE_MACOSX`, after that :data:`LOG_DEVICE_UNIX`
    is checked for existence. If both of these device files don't exist the default used
    by :class:`~logging.handlers.SysLogHandler` is returned.
    """
    if sys.platform == 'darwin' and os.path.exists(LOG_DEVICE_MACOSX):
        return LOG_DEVICE_MACOSX
    elif os.path.exists(LOG_DEVICE_UNIX):
        return LOG_DEVICE_UNIX
    else:
        return 'localhost', logging.handlers.SYSLOG_UDP_PORT


def is_syslog_supported():
    """
    Determine whether system logging is supported.

    :returns:

        :data:`True` if system logging is supported and can be enabled,
        :data:`False` if system logging is not supported or there are good
        reasons for not enabling it.

    The decision making process here is as follows:

    Override
     If the environment variable ``$COLOREDLOGS_SYSLOG`` is set it is evaluated
     using :func:`~humanfriendly.coerce_boolean()` and the resulting value
     overrides the platform detection discussed below, this allows users to
     override the decision making process if they disagree / know better.

    Linux / UNIX
     On systems that are not Windows or MacOS (see below) we assume UNIX which
     means either syslog is available or sending a bunch of UDP packets to
     nowhere won't hurt anyone...

    Microsoft Windows
     Over the years I've had multiple reports of :pypi:`coloredlogs` spewing
     extremely verbose errno 10057 warning messages to the console (once for
     each log message I suppose) so I now assume it a default that
     "syslog-style system logging" is not generally available on Windows.

    Apple MacOS
     There's cPython issue `#38780`_ which seems to result in a fatal exception
     when the Python interpreter shuts down. This is (way) worse than not
     having system logging enabled. The error message mentioned in `#38780`_
     has actually been following me around for years now, see for example:

     - https://github.com/xolox/python-rotate-backups/issues/9 mentions Docker
       images implying Linux, so not strictly the same as `#38780`_.

     - https://github.com/xolox/python-npm-accel/issues/4 is definitely related
       to `#38780`_ and is what eventually prompted me to add the
       :func:`is_syslog_supported()` logic.

    .. _#38780: https://bugs.python.org/issue38780
    """
    override = os.environ.get("COLOREDLOGS_SYSLOG")
    if override is not None:
        return coerce_boolean(override)
    else:
        return not (on_windows() or on_macos())


def match_syslog_handler(handler):
    """
    Identify system logging handlers.

    :param handler: The :class:`~logging.Handler` class to check.
    :returns: :data:`True` if the handler is a
              :class:`~logging.handlers.SysLogHandler`,
              :data:`False` otherwise.

    This function can be used as a callback for :func:`.find_handler()`.
    """
    return isinstance(handler, logging.handlers.SysLogHandler)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polymatrix.py ===
from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ

from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar


class MutablePolyDenseMatrix:
    """
    A mutable matrix of objects from poly module or to operate with them.

    Examples
    ========

    >>> from sympy.polys.polymatrix import PolyMatrix
    >>> from sympy import Symbol, Poly
    >>> x = Symbol('x')
    >>> pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    >>> v1 = PolyMatrix([[1, 0], [-1, 0]], x)
    >>> pm1*v1
    PolyMatrix([
    [    x**2 + x, 0],
    [x**3 - x + 1, 0]], ring=QQ[x])

    >>> pm1.ring
    ZZ[x]

    >>> v1*pm1
    PolyMatrix([
    [ x**2, -x],
    [-x**2,  x]], ring=QQ[x])

    >>> pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(1, x, domain='QQ'), \
            Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    >>> v2 = PolyMatrix([1, 0, 0, 0, 0, 0], x)
    >>> v2.ring
    QQ[x]
    >>> pm2*v2
    PolyMatrix([[x**2]], ring=QQ[x])

    """

    def __new__(cls, *args, ring=None):

        if not args:
            # PolyMatrix(ring=QQ[x])
            if ring is None:
                raise TypeError("The ring needs to be specified for an empty PolyMatrix")
            rows, cols, items, gens = 0, 0, [], ()
        elif isinstance(args[0], list):
            elements, gens = args[0], args[1:]
            if not elements:
                # PolyMatrix([])
                rows, cols, items = 0, 0, []
            elif isinstance(elements[0], (list, tuple)):
                # PolyMatrix([[1, 2]], x)
                rows, cols = len(elements), len(elements[0])
                items = [e for row in elements for e in row]
            else:
                # PolyMatrix([1, 2], x)
                rows, cols = len(elements), 1
                items = elements
        elif [type(a) for a in args[:3]] == [int, int, list]:
            # PolyMatrix(2, 2, [1, 2, 3, 4], x)
            rows, cols, items, gens = args[0], args[1], args[2], args[3:]
        elif [type(a) for a in args[:3]] == [int, int, type(lambda: 0)]:
            # PolyMatrix(2, 2, lambda i, j: i+j, x)
            rows, cols, func, gens = args[0], args[1], args[2], args[3:]
            items = [func(i, j) for i in range(rows) for j in range(cols)]
        else:
            raise TypeError("Invalid arguments")

        # PolyMatrix([[1]], x, y) vs PolyMatrix([[1]], (x, y))
        if len(gens) == 1 and isinstance(gens[0], tuple):
            gens = gens[0]
            # gens is now a tuple (x, y)

        return cls.from_list(rows, cols, items, gens, ring)

    @classmethod
    def from_list(cls, rows, cols, items, gens, ring):

        # items can be Expr, Poly, or a mix of Expr and Poly
        items = [_sympify(item) for item in items]
        if items and all(isinstance(item, Poly) for item in items):
            polys = True
        else:
            polys = False

        # Identify the ring for the polys
        if ring is not None:
            # Parse a domain string like 'QQ[x]'
            if isinstance(ring, str):
                ring = Poly(0, Dummy(), domain=ring).domain
        elif polys:
            p = items[0]
            for p2 in items[1:]:
                p, _ = p.unify(p2)
            ring = p.domain[p.gens]
        else:
            items, info = parallel_poly_from_expr(items, gens, field=True)
            ring = info['domain'][info['gens']]
            polys = True

        # Efficiently convert when all elements are Poly
        if polys:
            p_ring = Poly(0, ring.symbols, domain=ring.domain)
            to_ring = ring.ring.from_list
            convert_poly = lambda p: to_ring(p.unify(p_ring)[0].rep.to_list())
            elements = [convert_poly(p) for p in items]
        else:
            convert_expr = ring.from_sympy
            elements = [convert_expr(e.as_expr()) for e in items]

        # Convert to domain elements and construct DomainMatrix
        elements_lol = [[elements[i*cols + j] for j in range(cols)] for i in range(rows)]
        dm = DomainMatrix(elements_lol, (rows, cols), ring)
        return cls.from_dm(dm)

    @classmethod
    def from_dm(cls, dm):
        obj = super().__new__(cls)
        dm = dm.to_sparse()
        R = dm.domain
        obj._dm = dm
        obj.ring = R
        obj.domain = R.domain
        obj.gens = R.symbols
        return obj

    def to_Matrix(self):
        return self._dm.to_Matrix()

    @classmethod
    def from_Matrix(cls, other, *gens, ring=None):
        return cls(*other.shape, other.flat(), *gens, ring=ring)

    def set_gens(self, gens):
        return self.from_Matrix(self.to_Matrix(), gens)

    def __repr__(self):
        if self.rows * self.cols:
            return 'Poly' + repr(self.to_Matrix())[:-1] + f', ring={self.ring})'
        else:
            return f'PolyMatrix({self.rows}, {self.cols}, [], ring={self.ring})'

    @property
    def shape(self):
        return self._dm.shape

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, key):

        def to_poly(v):
            ground = self._dm.domain.domain
            gens = self._dm.domain.symbols
            return Poly(v.to_dict(), gens, domain=ground)

        dm = self._dm

        if isinstance(key, slice):
            items = dm.flat()[key]
            return [to_poly(item) for item in items]
        elif isinstance(key, int):
            i, j = divmod(key, self.cols)
            e = dm[i,j]
            return to_poly(e.element)

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return to_poly(dm[i, j].element)
        else:
            return self.from_dm(dm[i, j])

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self._dm == other._dm

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm + other._dm)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm - other._dm)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm * other._dm)
        elif isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            Kx = self.ring
            try:
                other_ds = DomainScalar(Kx.from_sympy(other), Kx)
            except (CoercionFailed, ValueError):
                other_ds = DomainScalar.from_sympy(other)
            return self.from_dm(self._dm * other_ds)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            other_ds = DomainScalar.from_sympy(other)
            return self.from_dm(other_ds * self._dm)
        return NotImplemented

    def __truediv__(self, other):

        if isinstance(other, Poly):
            other = other.as_expr()
        elif isinstance(other, int):
            other = _sympify(other)
        if not isinstance(other, Expr):
            return NotImplemented

        other = self.domain.from_sympy(other)
        inverse = self.ring.convert_from(1/other, self.domain)
        inverse = DomainScalar(inverse, self.ring)
        dm = self._dm * inverse
        return self.from_dm(dm)

    def __neg__(self):
        return self.from_dm(-self._dm)

    def transpose(self):
        return self.from_dm(self._dm.transpose())

    def row_join(self, other):
        dm = DomainMatrix.hstack(self._dm, other._dm)
        return self.from_dm(dm)

    def col_join(self, other):
        dm = DomainMatrix.vstack(self._dm, other._dm)
        return self.from_dm(dm)

    def applyfunc(self, func):
        M = self.to_Matrix().applyfunc(func)
        return self.from_Matrix(M, self.gens)

    @classmethod
    def eye(cls, n, gens):
        return cls.from_dm(DomainMatrix.eye(n, QQ[gens]))

    @classmethod
    def zeros(cls, m, n, gens):
        return cls.from_dm(DomainMatrix.zeros((m, n), QQ[gens]))

    def rref(self, simplify='ignore', normalize_last='ignore'):
        # If this is K[x] then computes RREF in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix rref is only for ground field elements")
        dm = self._dm
        dm_ground = dm.convert_to(dm.domain.domain)
        dm_rref, pivots = dm_ground.rref()
        dm_rref = dm_rref.convert_to(dm.domain)
        return self.from_dm(dm_rref), pivots

    def nullspace(self):
        # If this is K[x] then computes nullspace in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix nullspace is only for ground field elements")
        dm = self._dm
        K, Kx = self.domain, self.ring
        dm_null_rows = dm.convert_to(K).nullspace(divide_last=True).convert_to(Kx)
        dm_null = dm_null_rows.transpose()
        dm_basis = [dm_null[:,i] for i in range(dm_null.shape[1])]
        return [self.from_dm(dmvec) for dmvec in dm_basis]

    def rank(self):
        return self.cols - len(self.nullspace())

MutablePolyMatrix = PolyMatrix = MutablePolyDenseMatrix

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\cli.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: March 1, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Usage: humanfriendly [OPTIONS]

Human friendly input/output (text formatting) on the command
line based on the Python package with the same name.

Supported options:

  -c, --run-command

    Execute an external command (given as the positional arguments) and render
    a spinner and timer while the command is running. The exit status of the
    command is propagated.

  --format-table

    Read tabular data from standard input (each line is a row and each
    whitespace separated field is a column), format the data as a table and
    print the resulting table to standard output. See also the --delimiter
    option.

  -d, --delimiter=VALUE

    Change the delimiter used by --format-table to VALUE (a string). By default
    all whitespace is treated as a delimiter.

  -l, --format-length=LENGTH

    Convert a length count (given as the integer or float LENGTH) into a human
    readable string and print that string to standard output.

  -n, --format-number=VALUE

    Format a number (given as the integer or floating point number VALUE) with
    thousands separators and two decimal places (if needed) and print the
    formatted number to standard output.

  -s, --format-size=BYTES

    Convert a byte count (given as the integer BYTES) into a human readable
    string and print that string to standard output.

  -b, --binary

    Change the output of -s, --format-size to use binary multiples of bytes
    (base-2) instead of the default decimal multiples of bytes (base-10).

  -t, --format-timespan=SECONDS

    Convert a number of seconds (given as the floating point number SECONDS)
    into a human readable timespan and print that string to standard output.

  --parse-length=VALUE

    Parse a human readable length (given as the string VALUE) and print the
    number of metres to standard output.

  --parse-size=VALUE

    Parse a human readable data size (given as the string VALUE) and print the
    number of bytes to standard output.

  --demo

    Demonstrate changing the style and color of the terminal font using ANSI
    escape sequences.

  -h, --help

    Show this message and exit.
"""

# Standard library modules.
import functools
import getopt
import pipes
import subprocess
import sys

# Modules included in our package.
from humanfriendly import (
    Timer,
    format_length,
    format_number,
    format_size,
    format_timespan,
    parse_length,
    parse_size,
)
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
    ANSI_COLOR_CODES,
    ANSI_TEXT_STYLES,
    HIGHLIGHT_COLOR,
    ansi_strip,
    ansi_wrap,
    enable_ansi_support,
    find_terminal_size,
    output,
    usage,
    warning,
)
from humanfriendly.terminal.spinners import Spinner

# Public identifiers that require documentation.
__all__ = (
    'demonstrate_256_colors',
    'demonstrate_ansi_formatting',
    'main',
    'print_formatted_length',
    'print_formatted_number',
    'print_formatted_size',
    'print_formatted_table',
    'print_formatted_timespan',
    'print_parsed_length',
    'print_parsed_size',
    'run_command',
)


def main():
    """Command line interface for the ``humanfriendly`` program."""
    enable_ansi_support()
    try:
        options, arguments = getopt.getopt(sys.argv[1:], 'cd:l:n:s:bt:h', [
            'run-command', 'format-table', 'delimiter=', 'format-length=',
            'format-number=', 'format-size=', 'binary', 'format-timespan=',
            'parse-length=', 'parse-size=', 'demo', 'help',
        ])
    except Exception as e:
        warning("Error: %s", e)
        sys.exit(1)
    actions = []
    delimiter = None
    should_format_table = False
    binary = any(o in ('-b', '--binary') for o, v in options)
    for option, value in options:
        if option in ('-d', '--delimiter'):
            delimiter = value
        elif option == '--parse-size':
            actions.append(functools.partial(print_parsed_size, value))
        elif option == '--parse-length':
            actions.append(functools.partial(print_parsed_length, value))
        elif option in ('-c', '--run-command'):
            actions.append(functools.partial(run_command, arguments))
        elif option in ('-l', '--format-length'):
            actions.append(functools.partial(print_formatted_length, value))
        elif option in ('-n', '--format-number'):
            actions.append(functools.partial(print_formatted_number, value))
        elif option in ('-s', '--format-size'):
            actions.append(functools.partial(print_formatted_size, value, binary))
        elif option == '--format-table':
            should_format_table = True
        elif option in ('-t', '--format-timespan'):
            actions.append(functools.partial(print_formatted_timespan, value))
        elif option == '--demo':
            actions.append(demonstrate_ansi_formatting)
        elif option in ('-h', '--help'):
            usage(__doc__)
            return
    if should_format_table:
        actions.append(functools.partial(print_formatted_table, delimiter))
    if not actions:
        usage(__doc__)
        return
    for partial in actions:
        partial()


def run_command(command_line):
    """Run an external command and show a spinner while the command is running."""
    timer = Timer()
    spinner_label = "Waiting for command: %s" % " ".join(map(pipes.quote, command_line))
    with Spinner(label=spinner_label, timer=timer) as spinner:
        process = subprocess.Popen(command_line)
        while True:
            spinner.step()
            spinner.sleep()
            if process.poll() is not None:
                break
    sys.exit(process.returncode)


def print_formatted_length(value):
    """Print a human readable length."""
    if '.' in value:
        output(format_length(float(value)))
    else:
        output(format_length(int(value)))


def print_formatted_number(value):
    """Print large numbers in a human readable format."""
    output(format_number(float(value)))


def print_formatted_size(value, binary):
    """Print a human readable size."""
    output(format_size(int(value), binary=binary))


def print_formatted_table(delimiter):
    """Read tabular data from standard input and print a table."""
    data = []
    for line in sys.stdin:
        line = line.rstrip()
        data.append(line.split(delimiter))
    output(format_pretty_table(data))


def print_formatted_timespan(value):
    """Print a human readable timespan."""
    output(format_timespan(float(value)))


def print_parsed_length(value):
    """Parse a human readable length and print the number of metres."""
    output(parse_length(value))


def print_parsed_size(value):
    """Parse a human readable data size and print the number of bytes."""
    output(parse_size(value))


def demonstrate_ansi_formatting():
    """Demonstrate the use of ANSI escape sequences."""
    # First we demonstrate the supported text styles.
    output('%s', ansi_wrap('Text styles:', bold=True))
    styles = ['normal', 'bright']
    styles.extend(ANSI_TEXT_STYLES.keys())
    for style_name in sorted(styles):
        options = dict(color=HIGHLIGHT_COLOR)
        if style_name != 'normal':
            options[style_name] = True
        style_label = style_name.replace('_', ' ').capitalize()
        output(' - %s', ansi_wrap(style_label, **options))
    # Now we demonstrate named foreground and background colors.
    for color_type, color_label in (('color', 'Foreground colors'),
                                    ('background', 'Background colors')):
        intensities = [
            ('normal', dict()),
            ('bright', dict(bright=True)),
        ]
        if color_type != 'background':
            intensities.insert(0, ('faint', dict(faint=True)))
        output('\n%s' % ansi_wrap('%s:' % color_label, bold=True))
        output(format_smart_table([
            [color_name] + [
                ansi_wrap(
                    'XXXXXX' if color_type != 'background' else (' ' * 6),
                    **dict(list(kw.items()) + [(color_type, color_name)])
                ) for label, kw in intensities
            ] for color_name in sorted(ANSI_COLOR_CODES.keys())
        ], column_names=['Color'] + [
            label.capitalize() for label, kw in intensities
        ]))
    # Demonstrate support for 256 colors as well.
    demonstrate_256_colors(0, 7, 'standard colors')
    demonstrate_256_colors(8, 15, 'high-intensity colors')
    demonstrate_256_colors(16, 231, '216 colors')
    demonstrate_256_colors(232, 255, 'gray scale colors')


def demonstrate_256_colors(i, j, group=None):
    """Demonstrate 256 color mode support."""
    # Generate the label.
    label = '256 color mode'
    if group:
        label += ' (%s)' % group
    output('\n' + ansi_wrap('%s:' % label, bold=True))
    # Generate a simple rendering of the colors in the requested range and
    # check if it will fit on a single line (given the terminal's width).
    single_line = ''.join(' ' + ansi_wrap(str(n), color=n) for n in range(i, j + 1))
    lines, columns = find_terminal_size()
    if columns >= len(ansi_strip(single_line)):
        output(single_line)
    else:
        # Generate a more complex rendering of the colors that will nicely wrap
        # over multiple lines without using too many lines.
        width = len(str(j)) + 1
        colors_per_line = int(columns / width)
        colors = [ansi_wrap(str(n).rjust(width), color=n) for n in range(i, j + 1)]
        blocks = [colors[n:n + colors_per_line] for n in range(0, len(colors), colors_per_line)]
        output('\n'.join(''.join(b) for b in blocks))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\formatting\rule.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    String,
    Sequence,
    Bool,
    NoneSet,
    Set,
    Integer,
    Float,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.styles.colors import Color, ColorDescriptor
from openpyxl.styles.differential import DifferentialStyle

from openpyxl.utils.cell import COORD_RE


class ValueDescriptor(Float):
    """
    Expected type depends upon type attribute of parent :-(

    Most values should be numeric BUT they can also be cell references
    """

    def __set__(self, instance, value):
        ref = None
        if value is not None and isinstance(value, str):
            ref = COORD_RE.match(value)
        if instance.type == "formula" or ref:
            self.expected_type = str
        else:
            self.expected_type = float
        super().__set__(instance, value)


class FormatObject(Serialisable):

    tagname = "cfvo"

    type = Set(values=(['num', 'percent', 'max', 'min', 'formula', 'percentile']))
    val = ValueDescriptor(allow_none=True)
    gte = Bool(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 type,
                 val=None,
                 gte=None,
                 extLst=None,
                ):
        self.type = type
        self.val = val
        self.gte = gte


class RuleType(Serialisable):

    cfvo = Sequence(expected_type=FormatObject)


class IconSet(RuleType):

    tagname = "iconSet"

    iconSet = NoneSet(values=(['3Arrows', '3ArrowsGray', '3Flags',
                           '3TrafficLights1', '3TrafficLights2', '3Signs', '3Symbols', '3Symbols2',
                           '4Arrows', '4ArrowsGray', '4RedToBlack', '4Rating', '4TrafficLights',
                           '5Arrows', '5ArrowsGray', '5Rating', '5Quarters']))
    showValue = Bool(allow_none=True)
    percent = Bool(allow_none=True)
    reverse = Bool(allow_none=True)

    __elements__ = ("cfvo",)

    def __init__(self,
                 iconSet=None,
                 showValue=None,
                 percent=None,
                 reverse=None,
                 cfvo=None,
                ):
        self.iconSet = iconSet
        self.showValue = showValue
        self.percent = percent
        self.reverse = reverse
        self.cfvo = cfvo


class DataBar(RuleType):

    tagname = "dataBar"

    minLength = Integer(allow_none=True)
    maxLength = Integer(allow_none=True)
    showValue = Bool(allow_none=True)
    color = ColorDescriptor()

    __elements__ = ('cfvo', 'color')

    def __init__(self,
                 minLength=None,
                 maxLength=None,
                 showValue=None,
                 cfvo=None,
                 color=None,
                ):
        self.minLength = minLength
        self.maxLength = maxLength
        self.showValue = showValue
        self.cfvo = cfvo
        self.color = color


class ColorScale(RuleType):

    tagname = "colorScale"

    color = Sequence(expected_type=Color)

    __elements__ = ('cfvo', 'color')

    def __init__(self,
                 cfvo=None,
                 color=None,
                ):
        self.cfvo = cfvo
        self.color = color


class Rule(Serialisable):

    tagname = "cfRule"

    type = Set(values=(['expression', 'cellIs', 'colorScale', 'dataBar',
                        'iconSet', 'top10', 'uniqueValues', 'duplicateValues', 'containsText',
                        'notContainsText', 'beginsWith', 'endsWith', 'containsBlanks',
                        'notContainsBlanks', 'containsErrors', 'notContainsErrors', 'timePeriod',
                        'aboveAverage']))
    dxfId = Integer(allow_none=True)
    priority = Integer()
    stopIfTrue = Bool(allow_none=True)
    aboveAverage = Bool(allow_none=True)
    percent = Bool(allow_none=True)
    bottom = Bool(allow_none=True)
    operator = NoneSet(values=(['lessThan', 'lessThanOrEqual', 'equal',
                            'notEqual', 'greaterThanOrEqual', 'greaterThan', 'between', 'notBetween',
                            'containsText', 'notContains', 'beginsWith', 'endsWith']))
    text = String(allow_none=True)
    timePeriod = NoneSet(values=(['today', 'yesterday', 'tomorrow', 'last7Days',
                              'thisMonth', 'lastMonth', 'nextMonth', 'thisWeek', 'lastWeek',
                              'nextWeek']))
    rank = Integer(allow_none=True)
    stdDev = Integer(allow_none=True)
    equalAverage = Bool(allow_none=True)
    formula = Sequence(expected_type=str)
    colorScale = Typed(expected_type=ColorScale, allow_none=True)
    dataBar = Typed(expected_type=DataBar, allow_none=True)
    iconSet = Typed(expected_type=IconSet, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)
    dxf = Typed(expected_type=DifferentialStyle, allow_none=True)

    __elements__ = ('colorScale', 'dataBar', 'iconSet', 'formula')
    __attrs__ = ('type', 'rank', 'priority', 'equalAverage', 'operator',
                 'aboveAverage', 'dxfId', 'stdDev', 'stopIfTrue', 'timePeriod', 'text',
                 'percent', 'bottom')


    def __init__(self,
                 type,
                 dxfId=None,
                 priority=0,
                 stopIfTrue=None,
                 aboveAverage=None,
                 percent=None,
                 bottom=None,
                 operator=None,
                 text=None,
                 timePeriod=None,
                 rank=None,
                 stdDev=None,
                 equalAverage=None,
                 formula=(),
                 colorScale=None,
                 dataBar=None,
                 iconSet=None,
                 extLst=None,
                 dxf=None,
                ):
        self.type = type
        self.dxfId = dxfId
        self.priority = priority
        self.stopIfTrue = stopIfTrue
        self.aboveAverage = aboveAverage
        self.percent = percent
        self.bottom = bottom
        self.operator = operator
        self.text = text
        self.timePeriod = timePeriod
        self.rank = rank
        self.stdDev = stdDev
        self.equalAverage = equalAverage
        self.formula = formula
        self.colorScale = colorScale
        self.dataBar = dataBar
        self.iconSet = iconSet
        self.dxf = dxf


def ColorScaleRule(start_type=None,
                 start_value=None,
                 start_color=None,
                 mid_type=None,
                 mid_value=None,
                 mid_color=None,
                 end_type=None,
                 end_value=None,
                 end_color=None):

    """Backwards compatibility"""
    formats = []
    if start_type is not None:
        formats.append(FormatObject(type=start_type, val=start_value))
    if mid_type is not None:
        formats.append(FormatObject(type=mid_type, val=mid_value))
    if end_type is not None:
        formats.append(FormatObject(type=end_type, val=end_value))
    colors = []
    for v in (start_color, mid_color, end_color):
        if v is not None:
            if not isinstance(v, Color):
                v = Color(v)
            colors.append(v)
    cs = ColorScale(cfvo=formats, color=colors)
    rule = Rule(type="colorScale", colorScale=cs)
    return rule


def FormulaRule(formula=None, stopIfTrue=None, font=None, border=None,
                fill=None):
    """
    Conditional formatting with custom differential style
    """
    rule = Rule(type="expression", formula=formula, stopIfTrue=stopIfTrue)
    rule.dxf =  DifferentialStyle(font=font, border=border, fill=fill)
    return rule


def CellIsRule(operator=None, formula=None, stopIfTrue=None, font=None, border=None, fill=None):
    """
    Conditional formatting rule based on cell contents.
    """
    # Excel doesn't use >, >=, etc, but allow for ease of python development
    expand = {">": "greaterThan", ">=": "greaterThanOrEqual", "<": "lessThan", "<=": "lessThanOrEqual",
              "=": "equal", "==": "equal", "!=": "notEqual"}

    operator = expand.get(operator, operator)

    rule = Rule(type='cellIs', operator=operator, formula=formula, stopIfTrue=stopIfTrue)
    rule.dxf = DifferentialStyle(font=font, border=border, fill=fill)

    return rule


def IconSetRule(icon_style=None, type=None, values=None, showValue=None, percent=None, reverse=None):
    """
    Convenience function for creating icon set rules
    """
    cfvo = []
    for val in values:
        cfvo.append(FormatObject(type, val))
    icon_set = IconSet(iconSet=icon_style, cfvo=cfvo, showValue=showValue,
                       percent=percent, reverse=reverse)
    rule = Rule(type='iconSet', iconSet=icon_set)

    return rule


def DataBarRule(start_type=None, start_value=None, end_type=None,
                end_value=None, color=None, showValue=None, minLength=None, maxLength=None):
    start = FormatObject(start_type, start_value)
    end = FormatObject(end_type, end_value)
    data_bar = DataBar(cfvo=[start, end], color=color, showValue=showValue,
                       minLength=minLength, maxLength=maxLength)
    rule = Rule(type='dataBar', dataBar=data_bar)

    return rule

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\writer\theme.py ===
# Copyright (c) 2010-2024 openpyxl

"""Write the theme xml based on a fixed string."""


theme_xml = """<?xml version="1.0"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1>
        <a:sysClr val="windowText" lastClr="000000"/>
      </a:dk1>
      <a:lt1>
        <a:sysClr val="window" lastClr="FFFFFF"/>
      </a:lt1>
      <a:dk2>
        <a:srgbClr val="1F497D"/>
      </a:dk2>
      <a:lt2>
        <a:srgbClr val="EEECE1"/>
      </a:lt2>
      <a:accent1>
        <a:srgbClr val="4F81BD"/>
      </a:accent1>
      <a:accent2>
        <a:srgbClr val="C0504D"/>
      </a:accent2>
      <a:accent3>
        <a:srgbClr val="9BBB59"/>
      </a:accent3>
      <a:accent4>
        <a:srgbClr val="8064A2"/>
      </a:accent4>
      <a:accent5>
        <a:srgbClr val="4BACC6"/>
      </a:accent5>
      <a:accent6>
        <a:srgbClr val="F79646"/>
      </a:accent6>
      <a:hlink>
        <a:srgbClr val="0000FF"/>
      </a:hlink>
      <a:folHlink>
        <a:srgbClr val="800080"/>
      </a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office">
      <a:majorFont>
        <a:latin typeface="Cambria"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
        <a:font script="Jpan" typeface="&#xFF2D;&#xFF33; &#xFF30;&#x30B4;&#x30B7;&#x30C3;&#x30AF;"/>
        <a:font script="Hang" typeface="&#xB9D1;&#xC740; &#xACE0;&#xB515;"/>
        <a:font script="Hans" typeface="&#x5B8B;&#x4F53;"/>
        <a:font script="Hant" typeface="&#x65B0;&#x7D30;&#x660E;&#x9AD4;"/>
        <a:font script="Arab" typeface="Times New Roman"/>
        <a:font script="Hebr" typeface="Times New Roman"/>
        <a:font script="Thai" typeface="Tahoma"/>
        <a:font script="Ethi" typeface="Nyala"/>
        <a:font script="Beng" typeface="Vrinda"/>
        <a:font script="Gujr" typeface="Shruti"/>
        <a:font script="Khmr" typeface="MoolBoran"/>
        <a:font script="Knda" typeface="Tunga"/>
        <a:font script="Guru" typeface="Raavi"/>
        <a:font script="Cans" typeface="Euphemia"/>
        <a:font script="Cher" typeface="Plantagenet Cherokee"/>
        <a:font script="Yiii" typeface="Microsoft Yi Baiti"/>
        <a:font script="Tibt" typeface="Microsoft Himalaya"/>
        <a:font script="Thaa" typeface="MV Boli"/>
        <a:font script="Deva" typeface="Mangal"/>
        <a:font script="Telu" typeface="Gautami"/>
        <a:font script="Taml" typeface="Latha"/>
        <a:font script="Syrc" typeface="Estrangelo Edessa"/>
        <a:font script="Orya" typeface="Kalinga"/>
        <a:font script="Mlym" typeface="Kartika"/>
        <a:font script="Laoo" typeface="DokChampa"/>
        <a:font script="Sinh" typeface="Iskoola Pota"/>
        <a:font script="Mong" typeface="Mongolian Baiti"/>
        <a:font script="Viet" typeface="Times New Roman"/>
        <a:font script="Uigh" typeface="Microsoft Uighur"/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Calibri"/>
        <a:ea typeface=""/>
        <a:cs typeface=""/>
        <a:font script="Jpan" typeface="&#xFF2D;&#xFF33; &#xFF30;&#x30B4;&#x30B7;&#x30C3;&#x30AF;"/>
        <a:font script="Hang" typeface="&#xB9D1;&#xC740; &#xACE0;&#xB515;"/>
        <a:font script="Hans" typeface="&#x5B8B;&#x4F53;"/>
        <a:font script="Hant" typeface="&#x65B0;&#x7D30;&#x660E;&#x9AD4;"/>
        <a:font script="Arab" typeface="Arial"/>
        <a:font script="Hebr" typeface="Arial"/>
        <a:font script="Thai" typeface="Tahoma"/>
        <a:font script="Ethi" typeface="Nyala"/>
        <a:font script="Beng" typeface="Vrinda"/>
        <a:font script="Gujr" typeface="Shruti"/>
        <a:font script="Khmr" typeface="DaunPenh"/>
        <a:font script="Knda" typeface="Tunga"/>
        <a:font script="Guru" typeface="Raavi"/>
        <a:font script="Cans" typeface="Euphemia"/>
        <a:font script="Cher" typeface="Plantagenet Cherokee"/>
        <a:font script="Yiii" typeface="Microsoft Yi Baiti"/>
        <a:font script="Tibt" typeface="Microsoft Himalaya"/>
        <a:font script="Thaa" typeface="MV Boli"/>
        <a:font script="Deva" typeface="Mangal"/>
        <a:font script="Telu" typeface="Gautami"/>
        <a:font script="Taml" typeface="Latha"/>
        <a:font script="Syrc" typeface="Estrangelo Edessa"/>
        <a:font script="Orya" typeface="Kalinga"/>
        <a:font script="Mlym" typeface="Kartika"/>
        <a:font script="Laoo" typeface="DokChampa"/>
        <a:font script="Sinh" typeface="Iskoola Pota"/>
        <a:font script="Mong" typeface="Mongolian Baiti"/>
        <a:font script="Viet" typeface="Arial"/>
        <a:font script="Uigh" typeface="Microsoft Uighur"/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Office">
      <a:fillStyleLst>
        <a:solidFill>
          <a:schemeClr val="phClr"/>
        </a:solidFill>
        <a:gradFill rotWithShape="1">
          <a:gsLst>
            <a:gs pos="0">
              <a:schemeClr val="phClr">
                <a:tint val="50000"/>
                <a:satMod val="300000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="35000">
              <a:schemeClr val="phClr">
                <a:tint val="37000"/>
                <a:satMod val="300000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="100000">
              <a:schemeClr val="phClr">
                <a:tint val="15000"/>
                <a:satMod val="350000"/>
              </a:schemeClr>
            </a:gs>
          </a:gsLst>
          <a:lin ang="16200000" scaled="1"/>
        </a:gradFill>
        <a:gradFill rotWithShape="1">
          <a:gsLst>
            <a:gs pos="0">
              <a:schemeClr val="phClr">
                <a:shade val="51000"/>
                <a:satMod val="130000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="80000">
              <a:schemeClr val="phClr">
                <a:shade val="93000"/>
                <a:satMod val="130000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="100000">
              <a:schemeClr val="phClr">
                <a:shade val="94000"/>
                <a:satMod val="135000"/>
              </a:schemeClr>
            </a:gs>
          </a:gsLst>
          <a:lin ang="16200000" scaled="0"/>
        </a:gradFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill>
            <a:schemeClr val="phClr">
              <a:shade val="95000"/>
              <a:satMod val="105000"/>
            </a:schemeClr>
          </a:solidFill>
          <a:prstDash val="solid"/>
        </a:ln>
        <a:ln w="25400" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill>
            <a:schemeClr val="phClr"/>
          </a:solidFill>
          <a:prstDash val="solid"/>
        </a:ln>
        <a:ln w="38100" cap="flat" cmpd="sng" algn="ctr">
          <a:solidFill>
            <a:schemeClr val="phClr"/>
          </a:solidFill>
          <a:prstDash val="solid"/>
        </a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle>
          <a:effectLst>
            <a:outerShdw blurRad="40000" dist="20000" dir="5400000" rotWithShape="0">
              <a:srgbClr val="000000">
                <a:alpha val="38000"/>
              </a:srgbClr>
            </a:outerShdw>
          </a:effectLst>
        </a:effectStyle>
        <a:effectStyle>
          <a:effectLst>
            <a:outerShdw blurRad="40000" dist="23000" dir="5400000" rotWithShape="0">
              <a:srgbClr val="000000">
                <a:alpha val="35000"/>
              </a:srgbClr>
            </a:outerShdw>
          </a:effectLst>
        </a:effectStyle>
        <a:effectStyle>
          <a:effectLst>
            <a:outerShdw blurRad="40000" dist="23000" dir="5400000" rotWithShape="0">
              <a:srgbClr val="000000">
                <a:alpha val="35000"/>
              </a:srgbClr>
            </a:outerShdw>
          </a:effectLst>
          <a:scene3d>
            <a:camera prst="orthographicFront">
              <a:rot lat="0" lon="0" rev="0"/>
            </a:camera>
            <a:lightRig rig="threePt" dir="t">
              <a:rot lat="0" lon="0" rev="1200000"/>
            </a:lightRig>
          </a:scene3d>
          <a:sp3d>
            <a:bevelT w="63500" h="25400"/>
          </a:sp3d>
        </a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill>
          <a:schemeClr val="phClr"/>
        </a:solidFill>
        <a:gradFill rotWithShape="1">
          <a:gsLst>
            <a:gs pos="0">
              <a:schemeClr val="phClr">
                <a:tint val="40000"/>
                <a:satMod val="350000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="40000">
              <a:schemeClr val="phClr">
                <a:tint val="45000"/>
                <a:shade val="99000"/>
                <a:satMod val="350000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="100000">
              <a:schemeClr val="phClr">
                <a:shade val="20000"/>
                <a:satMod val="255000"/>
              </a:schemeClr>
            </a:gs>
          </a:gsLst>
          <a:path path="circle">
            <a:fillToRect l="50000" t="-80000" r="50000" b="180000"/>
          </a:path>
        </a:gradFill>
        <a:gradFill rotWithShape="1">
          <a:gsLst>
            <a:gs pos="0">
              <a:schemeClr val="phClr">
                <a:tint val="80000"/>
                <a:satMod val="300000"/>
              </a:schemeClr>
            </a:gs>
            <a:gs pos="100000">
              <a:schemeClr val="phClr">
                <a:shade val="30000"/>
                <a:satMod val="200000"/>
              </a:schemeClr>
            </a:gs>
          </a:gsLst>
          <a:path path="circle">
            <a:fillToRect l="50000" t="50000" r="50000" b="50000"/>
          </a:path>
        </a:gradFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""

def write_theme():
    """Write the theme xml."""
    return theme_xml

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\modular.py ===
from math import prod

from sympy.external.gmpy import gcd, gcdext
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int


def symmetric_residue(a, m):
    """Return the residual mod m such that it is within half of the modulus.

    >>> from sympy.ntheory.modular import symmetric_residue
    >>> symmetric_residue(1, 6)
    1
    >>> symmetric_residue(4, 6)
    -2
    """
    if a <= m // 2:
        return a
    return a - m


def crt(m, v, symmetric=False, check=True):
    r"""Chinese Remainder Theorem.

    The moduli in m are assumed to be pairwise coprime.  The output
    is then an integer f, such that f = v_i mod m_i for each pair out
    of v and m. If ``symmetric`` is False a positive integer will be
    returned, else \|f\| will be less than or equal to the LCM of the
    moduli, and thus f may be negative.

    If the moduli are not co-prime the correct result will be returned
    if/when the test of the result is found to be incorrect. This result
    will be None if there is no solution.

    The keyword ``check`` can be set to False if it is known that the moduli
    are coprime.

    Examples
    ========

    As an example consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.ntheory.modular import crt

       >>> crt([99, 97, 95], [49, 76, 65])
       (639985, 912285)

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    If the moduli are not co-prime, you may receive an incorrect result
    if you use ``check=False``:

       >>> crt([12, 6, 17], [3, 4, 2], check=False)
       (954, 1224)
       >>> [954 % m for m in [12, 6, 17]]
       [6, 0, 2]
       >>> crt([12, 6, 17], [3, 4, 2]) is None
       True
       >>> crt([3, 6], [2, 5])
       (5, 6)

    Note: the order of gf_crt's arguments is reversed relative to crt,
    and that solve_congruence takes residue, modulus pairs.

    Programmer's note: rather than checking that all pairs of moduli share
    no GCD (an O(n**2) test) and rather than factoring all moduli and seeing
    that there is no factor in common, a check that the result gives the
    indicated residuals is performed -- an O(n) operation.

    See Also
    ========

    solve_congruence
    sympy.polys.galoistools.gf_crt : low level crt routine used by this routine
    """
    if check:
        m = list(map(as_int, m))
        v = list(map(as_int, v))

    result = gf_crt(v, m, ZZ)
    mm = prod(m)

    if check:
        if not all(v % m == result % m for v, m in zip(v, m)):
            result = solve_congruence(*list(zip(v, m)),
                    check=False, symmetric=symmetric)
            if result is None:
                return result
            result, mm = result

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    return int(result), int(mm)


def crt1(m):
    """First part of Chinese Remainder Theorem, for multiple application.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt, crt1, crt2
    >>> m = [99, 97, 95]
    >>> v = [49, 76, 65]

    The following two codes have the same result.

    >>> crt(m, v)
    (639985, 912285)

    >>> mm, e, s = crt1(m)
    >>> crt2(m, v, mm, e, s)
    (639985, 912285)

    However, it is faster when we want to fix ``m`` and
    compute for multiple ``v``, i.e. the following cases:

    >>> mm, e, s = crt1(m)
    >>> vs = [[52, 21, 37], [19, 46, 76]]
    >>> for v in vs:
    ...     print(crt2(m, v, mm, e, s))
    (397042, 912285)
    (803206, 912285)

    See Also
    ========

    sympy.polys.galoistools.gf_crt1 : low level crt routine used by this routine
    sympy.ntheory.modular.crt
    sympy.ntheory.modular.crt2

    """

    return gf_crt1(m, ZZ)


def crt2(m, v, mm, e, s, symmetric=False):
    """Second part of Chinese Remainder Theorem, for multiple application.

    See ``crt1`` for usage.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt1, crt2
    >>> mm, e, s = crt1([18, 42, 6])
    >>> crt2([18, 42, 6], [0, 0, 0], mm, e, s)
    (0, 4536)

    See Also
    ========

    sympy.polys.galoistools.gf_crt2 : low level crt routine used by this routine
    sympy.ntheory.modular.crt
    sympy.ntheory.modular.crt1

    """

    result = gf_crt2(v, m, mm, e, s, ZZ)

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    return int(result), int(mm)


def solve_congruence(*remainder_modulus_pairs, **hint):
    """Compute the integer ``n`` that has the residual ``ai`` when it is
    divided by ``mi`` where the ``ai`` and ``mi`` are given as pairs to
    this function: ((a1, m1), (a2, m2), ...). If there is no solution,
    return None. Otherwise return ``n`` and its modulus.

    The ``mi`` values need not be co-prime. If it is known that the moduli are
    not co-prime then the hint ``check`` can be set to False (default=True) and
    the check for a quicker solution via crt() (valid when the moduli are
    co-prime) will be skipped.

    If the hint ``symmetric`` is True (default is False), the value of ``n``
    will be within 1/2 of the modulus, possibly negative.

    Examples
    ========

    >>> from sympy.ntheory.modular import solve_congruence

    What number is 2 mod 3, 3 mod 5 and 2 mod 7?

    >>> solve_congruence((2, 3), (3, 5), (2, 7))
    (23, 105)
    >>> [23 % m for m in [3, 5, 7]]
    [2, 3, 2]

    If you prefer to work with all remainder in one list and
    all moduli in another, send the arguments like this:

    >>> solve_congruence(*zip((2, 3, 2), (3, 5, 7)))
    (23, 105)

    The moduli need not be co-prime; in this case there may or
    may not be a solution:

    >>> solve_congruence((2, 3), (4, 6)) is None
    True

    >>> solve_congruence((2, 3), (5, 6))
    (5, 6)

    The symmetric flag will make the result be within 1/2 of the modulus:

    >>> solve_congruence((2, 3), (5, 6), symmetric=True)
    (-1, 6)

    See Also
    ========

    crt : high level routine implementing the Chinese Remainder Theorem

    """
    def combine(c1, c2):
        """Return the tuple (a, m) which satisfies the requirement
        that n = a + i*m satisfy n = a1 + j*m1 and n = a2 = k*m2.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Method_of_successive_substitution
        """
        a1, m1 = c1
        a2, m2 = c2
        a, b, c = m1, a2 - a1, m2
        g = gcd(a, b, c)
        a, b, c = [i//g for i in [a, b, c]]
        if a != 1:
            g, inv_a, _ = gcdext(a, c)
            if g != 1:
                return None
            b *= inv_a
        a, m = a1 + m1*b, m1*c
        return a, m

    rm = remainder_modulus_pairs
    symmetric = hint.get('symmetric', False)

    if hint.get('check', True):
        rm = [(as_int(r), as_int(m)) for r, m in rm]

        # ignore redundant pairs but raise an error otherwise; also
        # make sure that a unique set of bases is sent to gf_crt if
        # they are all prime.
        #
        # The routine will work out less-trivial violations and
        # return None, e.g. for the pairs (1,3) and (14,42) there
        # is no answer because 14 mod 42 (having a gcd of 14) implies
        # (14/2) mod (42/2), (14/7) mod (42/7) and (14/14) mod (42/14)
        # which, being 0 mod 3, is inconsistent with 1 mod 3. But to
        # preprocess the input beyond checking of another pair with 42
        # or 3 as the modulus (for this example) is not necessary.
        uniq = {}
        for r, m in rm:
            r %= m
            if m in uniq:
                if r != uniq[m]:
                    return None
                continue
            uniq[m] = r
        rm = [(r, m) for m, r in uniq.items()]
        del uniq

        # if the moduli are co-prime, the crt will be significantly faster;
        # checking all pairs for being co-prime gets to be slow but a prime
        # test is a good trade-off
        if all(isprime(m) for r, m in rm):
            r, m = list(zip(*rm))
            return crt(m, r, symmetric=symmetric, check=False)

    rv = (0, 1)
    for rmi in rm:
        rv = combine(rv, rmi)
        if rv is None:
            break
        n, m = rv
        n = n % m
    else:
        if symmetric:
            return symmetric_residue(n, m), m
        return n, m

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\transforms.py ===
"""Transforms that are always applied to quantum expressions.

This module uses the kind and _constructor_postprocessor_mapping APIs
to transform different combinations of Operators, Bras, and Kets into
Inner/Outer/TensorProducts. These transformations are registered
with the postprocessing API of core classes like `Mul` and `Pow` and
are always applied to any expression involving Bras, Kets, and
Operators. This API replaces the custom `__mul__` and `__pow__`
methods of the quantum classes, which were found to be inconsistent.

THIS IS EXPERIMENTAL.
"""
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.multipledispatch.dispatcher import (
    Dispatcher, ambiguity_register_error_ignore_dup
)
from sympy.utilities.misc import debug

from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.kind import KetKind, BraKind, OperatorKind
from sympy.physics.quantum.operator import (
    OuterProduct, IdentityOperator, Operator
)
from sympy.physics.quantum.state import BraBase, KetBase, StateBase
from sympy.physics.quantum.tensorproduct import TensorProduct


#-----------------------------------------------------------------------------
# Multipledispatch based transformed for Mul and Pow
#-----------------------------------------------------------------------------

_transform_state_pair = Dispatcher('_transform_state_pair')
"""Transform a pair of expression in a Mul to their canonical form.

All functions that are registered with this dispatcher need to take
two inputs and return either tuple of transformed outputs, or None if no
transform is applied. The output tuple is inserted into the right place
of the ``Mul`` that is being put into canonical form. It works something like
the following:

``Mul(a, b, c, d, e, f) -> Mul(*(_transform_state_pair(a, b) + (c, d, e, f))))``

The transforms here are always applied when quantum objects are multiplied.

THIS IS EXPERIMENTAL.

However, users of ``sympy.physics.quantum`` can import this dispatcher and
register their own transforms to control the canonical form of products
of quantum expressions.
"""

@_transform_state_pair.register(Expr, Expr)
def _transform_expr(a, b):
    """Default transformer that does nothing for base types."""
    return None


# The identity times anything is the anything.
_transform_state_pair.add(
    (IdentityOperator, Expr),
    lambda x, y: (y,),
    on_ambiguity=ambiguity_register_error_ignore_dup
)
_transform_state_pair.add(
    (Expr, IdentityOperator),
    lambda x, y: (x,),
    on_ambiguity=ambiguity_register_error_ignore_dup
)
_transform_state_pair.add(
    (IdentityOperator, IdentityOperator),
    lambda x, y: S.One,
    on_ambiguity=ambiguity_register_error_ignore_dup
)

@_transform_state_pair.register(BraBase, KetBase)
def _transform_bra_ket(a, b):
    """Transform a bra*ket -> InnerProduct(bra, ket)."""
    return (InnerProduct(a, b),)

@_transform_state_pair.register(KetBase, BraBase)
def _transform_ket_bra(a, b):
    """Transform a keT*bra -> OuterProduct(ket, bra)."""
    return (OuterProduct(a, b),)

@_transform_state_pair.register(KetBase, KetBase)
def _transform_ket_ket(a, b):
    """Raise a TypeError if a user tries to multiply two kets.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    raise TypeError(
        'Multiplication of two kets is not allowed. Use TensorProduct instead.'
    )

@_transform_state_pair.register(BraBase, BraBase)
def _transform_bra_bra(a, b):
    """Raise a TypeError if a user tries to multiply two bras.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    raise TypeError(
        'Multiplication of two bras is not allowed. Use TensorProduct instead.'
    )

@_transform_state_pair.register(OuterProduct, KetBase)
def _transform_op_ket(a, b):
    return (InnerProduct(a.bra, b), a.ket)

@_transform_state_pair.register(BraBase, OuterProduct)
def _transform_bra_op(a, b):
    return (InnerProduct(a, b.ket), b.bra)

@_transform_state_pair.register(TensorProduct, KetBase)
def _transform_tp_ket(a, b):
    """Raise a TypeError if a user tries to multiply TensorProduct(*kets)*ket.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if a.kind == KetKind:
        raise TypeError(
            'Multiplication of TensorProduct(*kets)*ket is invalid.'
        )

@_transform_state_pair.register(KetBase, TensorProduct)
def _transform_ket_tp(a, b):
    """Raise a TypeError if a user tries to multiply ket*TensorProduct(*kets).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if b.kind == KetKind:
        raise TypeError(
            'Multiplication of ket*TensorProduct(*kets) is invalid.'
        )

@_transform_state_pair.register(TensorProduct, BraBase)
def _transform_tp_bra(a, b):
    """Raise a TypeError if a user tries to multiply TensorProduct(*bras)*bra.

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if a.kind == BraKind:
        raise TypeError(
            'Multiplication of TensorProduct(*bras)*bra is invalid.'
        )

@_transform_state_pair.register(BraBase, TensorProduct)
def _transform_bra_tp(a, b):
    """Raise a TypeError if a user tries to multiply bra*TensorProduct(*bras).

    Multiplication based on `*` is not a shorthand for tensor products.
    """
    if b.kind == BraKind:
        raise TypeError(
            'Multiplication of bra*TensorProduct(*bras) is invalid.'
        )

@_transform_state_pair.register(TensorProduct, TensorProduct)
def _transform_tp_tp(a, b):
    """Combine a product of tensor products if their number of args matches."""
    debug('_transform_tp_tp', a, b)
    if len(a.args) == len(b.args):
        if a.kind == BraKind and b.kind == KetKind:
            return tuple([InnerProduct(i, j) for (i, j) in zip(a.args, b.args)])
        else:
            return (TensorProduct(*(i*j for (i, j) in zip(a.args, b.args))), )

@_transform_state_pair.register(OuterProduct, OuterProduct)
def _transform_op_op(a, b):
    """Extract an inner produt from a product of outer products."""
    return (InnerProduct(a.bra, b.ket), OuterProduct(a.ket, b.bra))


#-----------------------------------------------------------------------------
# Postprocessing transforms for Mul and Pow
#-----------------------------------------------------------------------------


def _postprocess_state_mul(expr):
    """Transform a ``Mul`` of quantum expressions into canonical form.

    This function is registered ``_constructor_postprocessor_mapping`` as a
    transformer for ``Mul``. This means that every time a quantum expression
    is multiplied, this function will be called to transform it into canonical
    form as defined by the binary functions registered with
    ``_transform_state_pair``.

    The algorithm of this function is as follows. It walks the args
    of the input ``Mul`` from left to right and calls ``_transform_state_pair``
    on every overlapping pair of args. Each time ``_transform_state_pair``
    is called it can return a tuple of items or None. If None, the pair isn't
    transformed. If a tuple, then the last element of the tuple goes back into
    the args to be transformed again and the others are extended onto the result
    args list.

    The algorithm can be visualized in the following table:

    step   result                                 args
    ============================================================================
    #0     []                                     [a, b, c, d, e, f]
    #1     []                                     [T(a,b), c, d, e, f]
    #2     [T(a,b)[:-1]]                          [T(a,b)[-1], c, d, e, f]
    #3     [T(a,b)[:-1]]                          [T(T(a,b)[-1], c), d, e, f]
    #4     [T(a,b)[:-1], T(T(a,b)[-1], c)[:-1]]   [T(T(T(a,b)[-1], c)[-1], d), e, f]
    #5     ...

    One limitation of the current implementation is that we assume that only the
    last item of the transformed tuple goes back into the args to be transformed
    again. These seems to handle the cases needed for Mul. However, we may need
    to extend the algorithm to have the entire tuple go back into the args for
    further transformation.
    """
    args = list(expr.args)
    result = []

    # Continue as long as we have at least 2 elements
    while len(args) > 1:
        # Get first two elements
        first = args.pop(0)
        second = args[0]  # Look at second element without popping yet

        transformed = _transform_state_pair(first, second)

        if transformed is None:
            # If transform returns None, append first element
            result.append(first)
        else:
            # This item was transformed, pop and discard
            args.pop(0)
            # The last item goes back to be transformed again
            args.insert(0, transformed[-1])
            # All other items go directly into the result
            result.extend(transformed[:-1])

    # Append any remaining element
    if args:
        result.append(args[0])

    return Mul._from_args(result, is_commutative=False)


def _postprocess_state_pow(expr):
    """Handle bras and kets raised to powers.

    Under ``*`` multiplication this is invalid. Users should use a
    TensorProduct instead.
    """
    base, exp = expr.as_base_exp()
    if base.kind == KetKind or base.kind == BraKind:
        raise TypeError(
            'A bra or ket to a power is invalid, use TensorProduct instead.'
        )


def _postprocess_tp_pow(expr):
    """Handle TensorProduct(*operators)**(positive integer).

    This handles a tensor product of operators, to an integer power.
    The power here is interpreted as regular multiplication, not
    tensor product exponentiation. The form of exponentiation performed
    here leaves the space and dimension of the object the same.

    This operation does not make sense for tensor product's of states.
    """
    base, exp = expr.as_base_exp()
    debug('_postprocess_tp_pow: ', base, exp, expr.args)
    if isinstance(base, TensorProduct) and exp.is_integer and exp.is_positive and base.kind == OperatorKind:
        new_args = [a**exp for a in base.args]
        return TensorProduct(*new_args)


#-----------------------------------------------------------------------------
# Register the transformers with Basic._constructor_postprocessor_mapping
#-----------------------------------------------------------------------------


Basic._constructor_postprocessor_mapping[StateBase] = {
    "Mul": [_postprocess_state_mul],
    "Pow": [_postprocess_state_pow]
}

Basic._constructor_postprocessor_mapping[TensorProduct] = {
    "Mul": [_postprocess_state_mul],
    "Pow": [_postprocess_tp_pow]
}

Basic._constructor_postprocessor_mapping[Operator] = {
    "Mul": [_postprocess_state_mul]
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\_cse_diff.py ===
"""Module for differentiation using CSE."""

from sympy import cse, Matrix, Derivative, MatrixBase
from sympy.utilities.iterables import iterable


def _remove_cse_from_derivative(replacements, reduced_expressions):
    """
    This function is designed to postprocess the output of a common subexpression
    elimination (CSE) operation. Specifically, it removes any CSE replacement
    symbols from the arguments of ``Derivative`` terms in the expression. This
    is necessary to ensure that the forward Jacobian function correctly handles
    derivative terms.

    Parameters
    ==========

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation.

    reduced_expressions : list of SymPy expressions
        The reduced expressions with all the replacements from the
        replacements list above.

    Returns
    =======

    processed_replacements : list of (Symbol, expression) pairs
        Processed replacement list, in the same format of the
        ``replacements`` input list.

    processed_reduced : list of SymPy expressions
        Processed reduced list, in the same format of the
        ``reduced_expressions`` input list.
    """

    def traverse(node, repl_dict):
        if isinstance(node, Derivative):
            return replace_all(node, repl_dict)
        if not node.args:
            return node
        new_args = [traverse(arg, repl_dict) for arg in node.args]
        return node.func(*new_args)

    def replace_all(node, repl_dict):
        result = node
        while True:
            free_symbols = result.free_symbols
            symbols_dict = {k: repl_dict[k] for k in free_symbols if k in repl_dict}
            if not symbols_dict:
                break
            result = result.xreplace(symbols_dict)
        return result

    repl_dict = dict(replacements)
    processed_replacements = [
        (rep_sym, traverse(sub_exp, repl_dict))
        for rep_sym, sub_exp in replacements
    ]
    processed_reduced = [
        red_exp.__class__([traverse(exp, repl_dict) for exp in red_exp])
        for red_exp in reduced_expressions
    ]

    return processed_replacements, processed_reduced


def _forward_jacobian_cse(replacements, reduced_expr, wrt):
    """
    Core function to compute the Jacobian of an input Matrix of expressions
    through forward accumulation. Takes directly the output of a CSE operation
    (replacements and reduced_expr), and an iterable of variables (wrt) with
    respect to which to differentiate the reduced expression and returns the
    reduced Jacobian matrix and the ``replacements`` list.

    The function also returns a list of precomputed free symbols for each
    subexpression, which are useful in the substitution process.

    Parameters
    ==========

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation.

    reduced_expr : list of SymPy expressions
        The reduced expressions with all the replacements from the
        replacements list above.

    wrt : iterable
        Iterable of expressions with respect to which to compute the
        Jacobian matrix.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation. Compared to the input replacement list,
        the output one doesn't contain replacement symbols inside
        ``Derivative``'s arguments.

    jacobian : list of SymPy expressions
        The list only contains one element, which is the Jacobian matrix with
        elements in reduced form (replacement symbols are present).

    precomputed_fs: list
        List of sets, which store the free symbols present in each sub-expression.
        Useful in the substitution process.
    """

    if not isinstance(reduced_expr[0], MatrixBase):
        raise TypeError("``expr`` must be of matrix type")

    if not (reduced_expr[0].shape[0] == 1 or reduced_expr[0].shape[1] == 1):
        raise TypeError("``expr`` must be a row or a column matrix")

    if not iterable(wrt):
        raise TypeError("``wrt`` must be an iterable of variables")

    elif not isinstance(wrt, MatrixBase):
        wrt = Matrix(wrt)

    if not (wrt.shape[0] == 1 or wrt.shape[1] == 1):
        raise TypeError("``wrt`` must be a row or a column matrix")

    replacements, reduced_expr = _remove_cse_from_derivative(replacements, reduced_expr)

    if replacements:
        rep_sym, sub_expr = map(Matrix, zip(*replacements))
    else:
        rep_sym, sub_expr = Matrix([]), Matrix([])

    l_sub, l_wrt, l_red = len(sub_expr), len(wrt), len(reduced_expr[0])

    f1 = reduced_expr[0].__class__.from_dok(l_red, l_wrt,
        {
            (i, j): diff_value
            for i, r in enumerate(reduced_expr[0])
            for j, w in enumerate(wrt)
            if (diff_value := r.diff(w)) != 0
        },
    )

    if not replacements:
        return [], [f1], []

    f2 = Matrix.from_dok(l_red, l_sub,
        {
            (i, j): diff_value
            for i, (r, fs) in enumerate([(r, r.free_symbols) for r in reduced_expr[0]])
            for j, s in enumerate(rep_sym)
            if s in fs and (diff_value := r.diff(s)) != 0
        },
    )

    rep_sym_set = set(rep_sym)
    precomputed_fs = [s.free_symbols & rep_sym_set for s in sub_expr ]

    c_matrix = Matrix.from_dok(1, l_wrt,
                               {(0, j): diff_value for j, w in enumerate(wrt)
                                if (diff_value := sub_expr[0].diff(w)) != 0})

    for i in range(1, l_sub):

        bi_matrix = Matrix.from_dok(1, i,
                                    {(0, j): diff_value for j in range(i + 1)
                                     if rep_sym[j] in precomputed_fs[i]
                                     and (diff_value := sub_expr[i].diff(rep_sym[j])) != 0})

        ai_matrix = Matrix.from_dok(1, l_wrt,
                                    {(0, j): diff_value for j, w in enumerate(wrt)
                                     if (diff_value := sub_expr[i].diff(w)) != 0})

        if bi_matrix._rep.nnz():
            ci_matrix = bi_matrix.multiply(c_matrix).add(ai_matrix)
            c_matrix = Matrix.vstack(c_matrix, ci_matrix)
        else:
            c_matrix = Matrix.vstack(c_matrix, ai_matrix)

    jacobian = f2.multiply(c_matrix).add(f1)
    jacobian = [reduced_expr[0].__class__(jacobian)]

    return replacements, jacobian, precomputed_fs


def _forward_jacobian_norm_in_cse_out(expr, wrt):
    """
    Function to compute the Jacobian of an input Matrix of expressions through
    forward accumulation. Takes a sympy Matrix of expressions (expr) as input
    and an iterable of variables (wrt) with respect to which to compute the
    Jacobian matrix. The matrix is returned in reduced form (containing
    replacement symbols) along with the ``replacements`` list.

    The function also returns a list of precomputed free symbols for each
    subexpression, which are useful in the substitution process.

    Parameters
    ==========

    expr : Matrix
        The vector to be differentiated.

    wrt : iterable
        The vector with respect to which to perform the differentiation.
        Can be a matrix or an iterable of variables.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation. The output replacement list doesn't
        contain replacement symbols inside ``Derivative``'s arguments.

    jacobian : list of SymPy expressions
        The list only contains one element, which is the Jacobian matrix with
        elements in reduced form (replacement symbols are present).

    precomputed_fs: list
        List of sets, which store the free symbols present in each
        sub-expression. Useful in the substitution process.
    """

    replacements, reduced_expr = cse(expr)
    replacements, jacobian, precomputed_fs = _forward_jacobian_cse(replacements, reduced_expr, wrt)

    return replacements, jacobian, precomputed_fs


def _forward_jacobian(expr, wrt):
    """
    Function to compute the Jacobian of an input Matrix of expressions through
    forward accumulation. Takes a sympy Matrix of expressions (expr) as input
    and an iterable of variables (wrt) with respect to which to compute the
    Jacobian matrix.

    Explanation
    ===========

    Expressions often contain repeated subexpressions. Using a tree structure,
    these subexpressions are duplicated and differentiated multiple times,
    leading to inefficiency.

    Instead, if a data structure called a directed acyclic graph (DAG) is used
    then each of these repeated subexpressions will only exist a single time.
    This function uses a combination of representing the expression as a DAG and
    a forward accumulation algorithm (repeated application of the chain rule
    symbolically) to more efficiently calculate the Jacobian matrix of a target
    expression ``expr`` with respect to an expression or set of expressions
    ``wrt``.

    Note that this function is intended to improve performance when
    differentiating large expressions that contain many common subexpressions.
    For small and simple expressions it is likely less performant than using
    SymPy's standard differentiation functions and methods.

    Parameters
    ==========

    expr : Matrix
        The vector to be differentiated.

    wrt : iterable
        The vector with respect to which to do the differentiation.
        Can be a matrix or an iterable of variables.

    See Also
    ========

    Direct Acyclic Graph : https://en.wikipedia.org/wiki/Directed_acyclic_graph
    """

    replacements, reduced_expr = cse(expr)

    if replacements:
        rep_sym, _ = map(Matrix, zip(*replacements))
    else:
        rep_sym = Matrix([])

    replacements, jacobian, precomputed_fs = _forward_jacobian_cse(replacements, reduced_expr, wrt)

    if not replacements: return jacobian[0]

    sub_rep = dict(replacements)
    for i, ik in enumerate(precomputed_fs):
        sub_dict = {j: sub_rep[j] for j in ik}
        sub_rep[rep_sym[i]] = sub_rep[rep_sym[i]].xreplace(sub_dict)

    return jacobian[0].xreplace(sub_rep)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_unix_pipes.py ===
from __future__ import annotations

import errno
import os
import select
import sys
from typing import TYPE_CHECKING

import pytest

from .. import _core
from .._core._tests.tutil import gc_collect_harder, skip_if_fbsd_pipes_broken
from ..testing import check_one_way_stream, wait_all_tasks_blocked

if TYPE_CHECKING:
    from .._file_io import _HasFileNo

posix = os.name == "posix"
pytestmark = pytest.mark.skipif(not posix, reason="posix only")

assert not TYPE_CHECKING or sys.platform == "unix"

if posix:
    from .._unix_pipes import FdStream
else:
    with pytest.raises(ImportError):
        from .._unix_pipes import FdStream


async def make_pipe() -> tuple[FdStream, FdStream]:
    """Makes a new pair of pipes."""
    (r, w) = os.pipe()
    return FdStream(w), FdStream(r)


async def make_clogged_pipe() -> tuple[FdStream, FdStream]:
    s, r = await make_pipe()
    try:
        while True:
            # We want to totally fill up the pipe buffer.
            # This requires working around a weird feature that POSIX pipes
            # have.
            # If you do a write of <= PIPE_BUF bytes, then it's guaranteed
            # to either complete entirely, or not at all. So if we tried to
            # write PIPE_BUF bytes, and the buffer's free space is only
            # PIPE_BUF/2, then the write will raise BlockingIOError... even
            # though a smaller write could still succeed! To avoid this,
            # make sure to write >PIPE_BUF bytes each time, which disables
            # the special behavior.
            # For details, search for PIPE_BUF here:
            #   http://pubs.opengroup.org/onlinepubs/9699919799/functions/write.html

            # for the getattr:
            # https://bitbucket.org/pypy/pypy/issues/2876/selectpipe_buf-is-missing-on-pypy3
            buf_size = getattr(select, "PIPE_BUF", 8192)
            os.write(s.fileno(), b"x" * buf_size * 2)
    except BlockingIOError:
        pass
    return s, r


async def test_send_pipe() -> None:
    r, w = os.pipe()
    async with FdStream(w) as send:
        assert send.fileno() == w
        await send.send_all(b"123")
        assert (os.read(r, 8)) == b"123"

        os.close(r)


async def test_receive_pipe() -> None:
    r, w = os.pipe()
    async with FdStream(r) as recv:
        assert (recv.fileno()) == r
        os.write(w, b"123")
        assert (await recv.receive_some(8)) == b"123"

        os.close(w)


async def test_pipes_combined() -> None:
    write, read = await make_pipe()
    count = 2**20

    async def sender() -> None:
        big = bytearray(count)
        await write.send_all(big)

    async def reader() -> None:
        await wait_all_tasks_blocked()
        received = 0
        while received < count:
            received += len(await read.receive_some(4096))

        assert received == count

    async with _core.open_nursery() as n:
        n.start_soon(sender)
        n.start_soon(reader)

    await read.aclose()
    await write.aclose()


async def test_pipe_errors() -> None:
    with pytest.raises(TypeError):
        FdStream(None)

    r, w = os.pipe()
    os.close(w)
    async with FdStream(r) as s:
        with pytest.raises(ValueError, match=r"^max_bytes must be integer >= 1$"):
            await s.receive_some(0)


async def test_del() -> None:
    w, r = await make_pipe()
    f1, f2 = w.fileno(), r.fileno()
    del w, r
    gc_collect_harder()

    with pytest.raises(OSError, match=r"Bad file descriptor$") as excinfo:
        os.close(f1)
    assert excinfo.value.errno == errno.EBADF

    with pytest.raises(OSError, match=r"Bad file descriptor$") as excinfo:
        os.close(f2)
    assert excinfo.value.errno == errno.EBADF


async def test_async_with() -> None:
    w, r = await make_pipe()
    async with w, r:
        pass

    assert w.fileno() == -1
    assert r.fileno() == -1

    with pytest.raises(OSError, match=r"Bad file descriptor$") as excinfo:
        os.close(w.fileno())
    assert excinfo.value.errno == errno.EBADF

    with pytest.raises(OSError, match=r"Bad file descriptor$") as excinfo:
        os.close(r.fileno())
    assert excinfo.value.errno == errno.EBADF


async def test_misdirected_aclose_regression() -> None:
    # https://github.com/python-trio/trio/issues/661#issuecomment-456582356
    w, r = await make_pipe()
    old_r_fd = r.fileno()

    # Close the original objects
    await w.aclose()
    await r.aclose()

    # Do a little dance to get a new pipe whose receive handle matches the old
    # receive handle.
    r2_fd, w2_fd = os.pipe()
    if r2_fd != old_r_fd:  # pragma: no cover
        os.dup2(r2_fd, old_r_fd)
        os.close(r2_fd)
    async with FdStream(old_r_fd) as r2:
        assert r2.fileno() == old_r_fd

        # And now set up a background task that's working on the new receive
        # handle
        async def expect_eof() -> None:
            assert await r2.receive_some(10) == b""

        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_eof)
            await wait_all_tasks_blocked()

            # Here's the key test: does calling aclose() again on the *old*
            # handle, cause the task blocked on the *new* handle to raise
            # ClosedResourceError?
            await r.aclose()
            await wait_all_tasks_blocked()

            # Guess we survived! Close the new write handle so that the task
            # gets an EOF and can exit cleanly.
            os.close(w2_fd)


async def test_close_at_bad_time_for_receive_some(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # We used to have race conditions where if one task was using the pipe,
    # and another closed it at *just* the wrong moment, it would give an
    # unexpected error instead of ClosedResourceError:
    # https://github.com/python-trio/trio/issues/661
    #
    # This tests what happens if the pipe gets closed in the moment *between*
    # when receive_some wakes up, and when it tries to call os.read
    async def expect_closedresourceerror() -> None:
        with pytest.raises(_core.ClosedResourceError):
            await r.receive_some(10)

    orig_wait_readable = _core._run.TheIOManager.wait_readable

    async def patched_wait_readable(
        self: _core._run.TheIOManager,
        fd: int | _HasFileNo,
    ) -> None:
        await orig_wait_readable(self, fd)
        await r.aclose()

    monkeypatch.setattr(_core._run.TheIOManager, "wait_readable", patched_wait_readable)
    s, r = await make_pipe()
    async with s, r:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_closedresourceerror)
            await wait_all_tasks_blocked()
            # Trigger everything by waking up the receiver
            await s.send_all(b"x")


async def test_close_at_bad_time_for_send_all(monkeypatch: pytest.MonkeyPatch) -> None:
    # We used to have race conditions where if one task was using the pipe,
    # and another closed it at *just* the wrong moment, it would give an
    # unexpected error instead of ClosedResourceError:
    # https://github.com/python-trio/trio/issues/661
    #
    # This tests what happens if the pipe gets closed in the moment *between*
    # when send_all wakes up, and when it tries to call os.write
    async def expect_closedresourceerror() -> None:
        with pytest.raises(_core.ClosedResourceError):
            await s.send_all(b"x" * 100)

    orig_wait_writable = _core._run.TheIOManager.wait_writable

    async def patched_wait_writable(
        self: _core._run.TheIOManager,
        fd: int | _HasFileNo,
    ) -> None:
        await orig_wait_writable(self, fd)
        await s.aclose()

    monkeypatch.setattr(_core._run.TheIOManager, "wait_writable", patched_wait_writable)
    s, r = await make_clogged_pipe()
    async with s, r:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(expect_closedresourceerror)
            await wait_all_tasks_blocked()
            # Trigger everything by waking up the sender. On ppc64el, PIPE_BUF
            # is 8192 but make_clogged_pipe() ends up writing a total of
            # 1048576 bytes before the pipe is full, and then a subsequent
            # receive_some(10000) isn't sufficient for orig_wait_writable() to
            # return for our subsequent aclose() call. It's necessary to empty
            # the pipe further before this happens. So we loop here until the
            # pipe is empty to make sure that the sender wakes up even in this
            # case. Otherwise patched_wait_writable() never gets to the
            # aclose(), so expect_closedresourceerror() never returns, the
            # nursery never finishes all tasks and this test hangs.
            received_data = await r.receive_some(10000)
            while received_data:
                received_data = await r.receive_some(10000)


# On FreeBSD, directories are readable, and we haven't found any other trick
# for making an unreadable fd, so there's no way to run this test. Fortunately
# the logic this is testing doesn't depend on the platform, so testing on
# other platforms is probably good enough.
@pytest.mark.skipif(
    sys.platform.startswith("freebsd"),
    reason="no way to make read() return a bizarro error on FreeBSD",
)
async def test_bizarro_OSError_from_receive() -> None:
    # Make sure that if the read syscall returns some bizarro error, then we
    # get a BrokenResourceError. This is incredibly unlikely; there's almost
    # no way to trigger a failure here intentionally (except for EBADF, but we
    # exploit that to detect file closure, so it takes a different path). So
    # we set up a strange scenario where the pipe fd somehow transmutes into a
    # directory fd, causing os.read to raise IsADirectoryError (yes, that's a
    # real built-in exception type).
    s, r = await make_pipe()
    async with s, r:
        dir_fd = os.open("/", os.O_DIRECTORY, 0)
        try:
            os.dup2(dir_fd, r.fileno())
            with pytest.raises(_core.BrokenResourceError):
                await r.receive_some(10)
        finally:
            os.close(dir_fd)


@skip_if_fbsd_pipes_broken
async def test_pipe_fully() -> None:
    await check_one_way_stream(make_pipe, make_clogged_pipe)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\common\exceptions.py ===
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
"""Exceptions that may happen in all the webdriver code."""

from typing import Optional, Sequence

SUPPORT_MSG = "For documentation on this error, please visit:"
ERROR_URL = "https://www.selenium.dev/documentation/webdriver/troubleshooting/errors"


class WebDriverException(Exception):
    """Base webdriver exception."""

    def __init__(
        self, msg: Optional[str] = None, screen: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None
    ) -> None:
        super().__init__()
        self.msg = msg
        self.screen = screen
        self.stacktrace = stacktrace

    def __str__(self) -> str:
        exception_msg = f"Message: {self.msg}\n"
        if self.screen:
            exception_msg += "Screenshot: available via screen\n"
        if self.stacktrace:
            stacktrace = "\n".join(self.stacktrace)
            exception_msg += f"Stacktrace:\n{stacktrace}"
        return exception_msg


class InvalidSwitchToTargetException(WebDriverException):
    """Thrown when frame or window target to be switched doesn't exist."""


class NoSuchFrameException(InvalidSwitchToTargetException):
    """Thrown when frame target to be switched doesn't exist."""


class NoSuchWindowException(InvalidSwitchToTargetException):
    """Thrown when window target to be switched doesn't exist.

    To find the current set of active window handles, you can get a list
    of the active window handles in the following way::

        print driver.window_handles
    """


class NoSuchElementException(WebDriverException):
    """Thrown when element could not be found.

    If you encounter this exception, you may want to check the following:
        * Check your selector used in your find_by...
        * Element may not yet be on the screen at the time of the find operation,
          (webpage is still loading) see selenium.webdriver.support.wait.WebDriverWait()
          for how to write a wait wrapper to wait for an element to appear.
    """

    def __init__(
        self, msg: Optional[str] = None, screen: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None
    ) -> None:
        with_support = f"{msg}; {SUPPORT_MSG} {ERROR_URL}#no-such-element-exception"

        super().__init__(with_support, screen, stacktrace)


class NoSuchAttributeException(WebDriverException):
    """Thrown when the attribute of element could not be found.

    You may want to check if the attribute exists in the particular
    browser you are testing against.  Some browsers may have different
    property names for the same property.  (IE8's .innerText vs. Firefox
    .textContent)
    """


class NoSuchShadowRootException(WebDriverException):
    """Thrown when trying to access the shadow root of an element when it does
    not have a shadow root attached."""


class StaleElementReferenceException(WebDriverException):
    """Thrown when a reference to an element is now "stale".

    Stale means the element no longer appears on the DOM of the page.


    Possible causes of StaleElementReferenceException include, but not limited to:
        * You are no longer on the same page, or the page may have refreshed since the element
          was located.
        * The element may have been removed and re-added to the screen, since it was located.
          Such as an element being relocated.
          This can happen typically with a javascript framework when values are updated and the
          node is rebuilt.
        * Element may have been inside an iframe or another context which was refreshed.
    """

    def __init__(
        self, msg: Optional[str] = None, screen: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None
    ) -> None:
        with_support = f"{msg}; {SUPPORT_MSG} {ERROR_URL}#stale-element-reference-exception"

        super().__init__(with_support, screen, stacktrace)


class InvalidElementStateException(WebDriverException):
    """Thrown when a command could not be completed because the element is in
    an invalid state.

    This can be caused by attempting to clear an element that isn't both
    editable and resettable.
    """


class UnexpectedAlertPresentException(WebDriverException):
    """Thrown when an unexpected alert has appeared.

    Usually raised when  an unexpected modal is blocking the webdriver
    from executing commands.
    """

    def __init__(
        self,
        msg: Optional[str] = None,
        screen: Optional[str] = None,
        stacktrace: Optional[Sequence[str]] = None,
        alert_text: Optional[str] = None,
    ) -> None:
        super().__init__(msg, screen, stacktrace)
        self.alert_text = alert_text

    def __str__(self) -> str:
        return f"Alert Text: {self.alert_text}\n{super().__str__()}"


class NoAlertPresentException(WebDriverException):
    """Thrown when switching to no presented alert.

    This can be caused by calling an operation on the Alert() class when
    an alert is not yet on the screen.
    """


class ElementNotVisibleException(InvalidElementStateException):
    """Thrown when an element is present on the DOM, but it is not visible, and
    so is not able to be interacted with.

    Most commonly encountered when trying to click or read text of an
    element that is hidden from view.
    """


class ElementNotInteractableException(InvalidElementStateException):
    """Thrown when an element is present in the DOM but interactions with that
    element will hit another element due to paint order."""


class ElementNotSelectableException(InvalidElementStateException):
    """Thrown when trying to select an unselectable element.

    For example, selecting a 'script' element.
    """


class InvalidCookieDomainException(WebDriverException):
    """Thrown when attempting to add a cookie under a different domain than the
    current URL."""


class UnableToSetCookieException(WebDriverException):
    """Thrown when a driver fails to set a cookie."""


class TimeoutException(WebDriverException):
    """Thrown when a command does not complete in enough time."""


class MoveTargetOutOfBoundsException(WebDriverException):
    """Thrown when the target provided to the `ActionsChains` move() method is
    invalid, i.e. out of document."""


class UnexpectedTagNameException(WebDriverException):
    """Thrown when a support class did not get an expected web element."""


class InvalidSelectorException(WebDriverException):
    """Thrown when the selector which is used to find an element does not
    return a WebElement.

    Currently this only happens when the selector is an xpath expression
    and it is either syntactically invalid (i.e. it is not a xpath
    expression) or the expression does not select WebElements (e.g.
    "count(//input)").
    """

    def __init__(
        self, msg: Optional[str] = None, screen: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None
    ) -> None:
        with_support = f"{msg}; {SUPPORT_MSG} {ERROR_URL}#invalid-selector-exception"

        super().__init__(with_support, screen, stacktrace)


class ImeNotAvailableException(WebDriverException):
    """Thrown when IME support is not available.

    This exception is thrown for every IME-related method call if IME
    support is not available on the machine.
    """


class ImeActivationFailedException(WebDriverException):
    """Thrown when activating an IME engine has failed."""


class InvalidArgumentException(WebDriverException):
    """The arguments passed to a command are either invalid or malformed."""


class JavascriptException(WebDriverException):
    """An error occurred while executing JavaScript supplied by the user."""


class NoSuchCookieException(WebDriverException):
    """No cookie matching the given path name was found amongst the associated
    cookies of the current browsing context's active document."""


class ScreenshotException(WebDriverException):
    """A screen capture was made impossible."""


class ElementClickInterceptedException(WebDriverException):
    """The Element Click command could not be completed because the element
    receiving the events is obscuring the element that was requested to be
    clicked."""


class InsecureCertificateException(WebDriverException):
    """Navigation caused the user agent to hit a certificate warning, which is
    usually the result of an expired or invalid TLS certificate."""


class InvalidCoordinatesException(WebDriverException):
    """The coordinates provided to an interaction's operation are invalid."""


class InvalidSessionIdException(WebDriverException):
    """Occurs if the given session id is not in the list of active sessions,
    meaning the session either does not exist or that it's not active."""


class SessionNotCreatedException(WebDriverException):
    """A new session could not be created."""


class UnknownMethodException(WebDriverException):
    """The requested command matched a known URL but did not match any methods
    for that URL."""


class NoSuchDriverException(WebDriverException):
    """Raised when driver is not specified and cannot be located."""

    def __init__(
        self, msg: Optional[str] = None, screen: Optional[str] = None, stacktrace: Optional[Sequence[str]] = None
    ) -> None:
        with_support = f"{msg}; {SUPPORT_MSG} {ERROR_URL}/driver_location"

        super().__init__(with_support, screen, stacktrace)


class DetachedShadowRootException(WebDriverException):
    """Raised when referenced shadow root is no longer attached to the DOM."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\operatorordering.py ===
"""Functions for reordering operator expressions."""

import warnings

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.physics.quantum import Commutator, AntiCommutator
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp

__all__ = [
    'normal_order',
    'normal_ordered_form'
]


def _expand_powers(factors):
    """
    Helper function for normal_ordered_form and normal_order: Expand a
    power expression to a multiplication expression so that that the
    expression can be handled by the normal ordering functions.
    """

    new_factors = []
    for factor in factors.args:
        if (isinstance(factor, Pow)
                and isinstance(factor.args[1], Integer)
                and factor.args[1] > 0):
            for n in range(factor.args[1]):
                new_factors.append(factor.args[0])
        else:
            new_factors.append(factor)

    return new_factors

def _normal_ordered_form_factor(product, independent=False, recursive_limit=10,
                                _recursive_depth=0):
    """
    Helper function for normal_ordered_form_factor: Write multiplication
    expression with bosonic or fermionic operators on normally ordered form,
    using the bosonic and fermionic commutation relations. The resulting
    operator expression is equivalent to the argument, but will in general be
    a sum of operator products instead of a simple product.
    """

    factors = _expand_powers(product)

    new_factors = []
    n = 0
    while n < len(factors) - 1:
        current, next = factors[n], factors[n + 1]
        if any(not isinstance(f, (FermionOp, BosonOp)) for f in (current, next)):
            new_factors.append(current)
            n += 1
            continue

        key_1 = (current.is_annihilation, str(current.name))
        key_2 = (next.is_annihilation, str(next.name))

        if key_1 <= key_2:
            new_factors.append(current)
            n += 1
            continue

        n += 2
        if current.is_annihilation and not next.is_annihilation:
            if isinstance(current, BosonOp) and isinstance(next, BosonOp):
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = Commutator(current, next)
                    new_factors.append(next * current + c)
                else:
                    new_factors.append(next * current + 1)
            elif isinstance(current, FermionOp) and isinstance(next, FermionOp):
                if current.args[0] != next.args[0]:
                    if independent:
                        c = 0
                    else:
                        c = AntiCommutator(current, next)
                    new_factors.append(-next * current + c)
                else:
                    new_factors.append(-next * current + 1)
        elif (current.is_annihilation == next.is_annihilation and
            isinstance(current, FermionOp) and isinstance(next, FermionOp)):
            new_factors.append(-next * current)
        else:
            new_factors.append(next * current)

    if n == len(factors) - 1:
        new_factors.append(factors[-1])

    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_ordered_form(expr,
                                   recursive_limit=recursive_limit,
                                   _recursive_depth=_recursive_depth + 1,
                                   independent=independent)


def _normal_ordered_form_terms(expr, independent=False, recursive_limit=10,
                               _recursive_depth=0):
    """
    Helper function for normal_ordered_form: loop through each term in an
    addition expression and call _normal_ordered_form_factor to perform the
    factor to an normally ordered expression.
    """

    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            new_term = _normal_ordered_form_factor(
                term, recursive_limit=recursive_limit,
                _recursive_depth=_recursive_depth, independent=independent)
            new_terms.append(new_term)
        else:
            new_terms.append(term)

    return Add(*new_terms)


def normal_ordered_form(expr, independent=False, recursive_limit=10,
                        _recursive_depth=0):
    """Write an expression with bosonic or fermionic operators on normal
    ordered form, where each term is normally ordered. Note that this
    normal ordered form is equivalent to the original expression.

    Parameters
    ==========

    expr : expression
        The expression write on normal ordered form.
    independent : bool (default False)
        Whether to consider operator with different names as operating in
        different Hilbert spaces. If False, the (anti-)commutation is left
        explicit.
    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_ordered_form
    >>> a = BosonOp("a")
    >>> normal_ordered_form(a * Dagger(a))
    1 + Dagger(a)*a
    """

    if _recursive_depth > recursive_limit:
        warnings.warn("Too many recursions, aborting")
        return expr

    if isinstance(expr, Add):
        return _normal_ordered_form_terms(expr,
                                          recursive_limit=recursive_limit,
                                          _recursive_depth=_recursive_depth,
                                          independent=independent)
    elif isinstance(expr, Mul):
        return _normal_ordered_form_factor(expr,
                                           recursive_limit=recursive_limit,
                                           _recursive_depth=_recursive_depth,
                                           independent=independent)
    else:
        return expr


def _normal_order_factor(product, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: Normal order a multiplication expression
    with bosonic or fermionic operators. In general the resulting operator
    expression will not be equivalent to original product.
    """

    factors = _expand_powers(product)

    n = 0
    new_factors = []
    while n < len(factors) - 1:

        if (isinstance(factors[n], BosonOp) and
                factors[n].is_annihilation):
            # boson
            if not isinstance(factors[n + 1], BosonOp):
                new_factors.append(factors[n])
            else:
                if factors[n + 1].is_annihilation:
                    new_factors.append(factors[n])
                else:
                    if factors[n].args[0] != factors[n + 1].args[0]:
                        new_factors.append(factors[n + 1] * factors[n])
                    else:
                        new_factors.append(factors[n + 1] * factors[n])
                    n += 1

        elif (isinstance(factors[n], FermionOp) and
              factors[n].is_annihilation):
            # fermion
            if not isinstance(factors[n + 1], FermionOp):
                new_factors.append(factors[n])
            else:
                if factors[n + 1].is_annihilation:
                    new_factors.append(factors[n])
                else:
                    if factors[n].args[0] != factors[n + 1].args[0]:
                        new_factors.append(-factors[n + 1] * factors[n])
                    else:
                        new_factors.append(-factors[n + 1] * factors[n])
                    n += 1

        else:
            new_factors.append(factors[n])

        n += 1

    if n == len(factors) - 1:
        new_factors.append(factors[-1])

    if new_factors == factors:
        return product
    else:
        expr = Mul(*new_factors).expand()
        return normal_order(expr,
                            recursive_limit=recursive_limit,
                            _recursive_depth=_recursive_depth + 1)


def _normal_order_terms(expr, recursive_limit=10, _recursive_depth=0):
    """
    Helper function for normal_order: look through each term in an addition
    expression and call _normal_order_factor to perform the normal ordering
    on the factors.
    """

    new_terms = []
    for term in expr.args:
        if isinstance(term, Mul):
            new_term = _normal_order_factor(term,
                                            recursive_limit=recursive_limit,
                                            _recursive_depth=_recursive_depth)
            new_terms.append(new_term)
        else:
            new_terms.append(term)

    return Add(*new_terms)


def normal_order(expr, recursive_limit=10, _recursive_depth=0):
    """Normal order an expression with bosonic or fermionic operators. Note
    that this normal order is not equivalent to the original expression, but
    the creation and annihilation operators in each term in expr is reordered
    so that the expression becomes normal ordered.

    Parameters
    ==========

    expr : expression
        The expression to normal order.

    recursive_limit : int (default 10)
        The number of allowed recursive applications of the function.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger
    >>> from sympy.physics.quantum.boson import BosonOp
    >>> from sympy.physics.quantum.operatorordering import normal_order
    >>> a = BosonOp("a")
    >>> normal_order(a * Dagger(a))
    Dagger(a)*a
    """
    if _recursive_depth > recursive_limit:
        warnings.warn("Too many recursions, aborting")
        return expr

    if isinstance(expr, Add):
        return _normal_order_terms(expr, recursive_limit=recursive_limit,
                                   _recursive_depth=_recursive_depth)
    elif isinstance(expr, Mul):
        return _normal_order_factor(expr, recursive_limit=recursive_limit,
                                    _recursive_depth=_recursive_depth)
    else:
        return expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\packaging\custom.py ===
# Copyright (c) 2010-2024 openpyxl

"""Implementation of custom properties see § 22.3 in the specification"""


from warnings import warn

from openpyxl.descriptors import Strict
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.sequence import Sequence
from openpyxl.descriptors import (
    Alias,
    String,
    Integer,
    Float,
    DateTime,
    Bool,
)
from openpyxl.descriptors.nested import (
    NestedText,
)

from openpyxl.xml.constants import (
    CUSTPROPS_NS,
    VTYPES_NS,
    CPROPS_FMTID,
)

from .core import NestedDateTime


class NestedBoolText(Bool, NestedText):
    """
    Descriptor for handling nested elements with the value stored in the text part
    """

    pass


class _CustomDocumentProperty(Serialisable):

    """
    Low-level representation of a Custom Document Property.
    Not used directly
    Must always contain a child element, even if this is empty
    """

    tagname = "property"
    _typ = None

    name = String(allow_none=True)
    lpwstr = NestedText(expected_type=str, allow_none=True, namespace=VTYPES_NS)
    i4 = NestedText(expected_type=int, allow_none=True, namespace=VTYPES_NS)
    r8 = NestedText(expected_type=float, allow_none=True, namespace=VTYPES_NS)
    filetime = NestedDateTime(allow_none=True, namespace=VTYPES_NS)
    bool = NestedBoolText(expected_type=bool, allow_none=True, namespace=VTYPES_NS)
    linkTarget = String(expected_type=str, allow_none=True)
    fmtid = String()
    pid = Integer()

    def __init__(self,
                 name=None,
                 pid=0,
                 fmtid=CPROPS_FMTID,
                 linkTarget=None,
                 **kw):
        self.fmtid = fmtid
        self.pid = pid
        self.name = name
        self._typ = None
        self.linkTarget = linkTarget

        for k, v in kw.items():
            setattr(self, k, v)
            setattr(self, "_typ", k) # ugh!
        for e in self.__elements__:
            if e not in kw:
                setattr(self, e, None)


    @property
    def type(self):
        if self._typ is not None:
            return self._typ
        for a in self.__elements__:
            if getattr(self, a) is not None:
                return a
        if self.linkTarget is not None:
            return "linkTarget"


    def to_tree(self, tagname=None, idx=None, namespace=None):
        child = getattr(self, self._typ, None)
        if child is None:
            setattr(self, self._typ, "")

        return super().to_tree(tagname=None, idx=None, namespace=None)


class _CustomDocumentPropertyList(Serialisable):

    """
    Parses and seriliases property lists but is not used directly
    """

    tagname = "Properties"

    property = Sequence(expected_type=_CustomDocumentProperty, namespace=CUSTPROPS_NS)
    customProps = Alias("property")


    def __init__(self, property=()):
        self.property = property


    def __len__(self):
        return len(self.property)


    def to_tree(self, tagname=None, idx=None, namespace=None):
        for idx, p in enumerate(self.property, 2):
            p.pid = idx
        tree = super().to_tree(tagname, idx, namespace)
        tree.set("xmlns", CUSTPROPS_NS)

        return tree


class _TypedProperty(Strict):

    name = String()

    def __init__(self,
                 name,
                 value):
        self.name = name
        self.value = value


    def __eq__(self, other):
        return self.name == other.name and self.value == other.value


    def __repr__(self):
        return f"{self.__class__.__name__}, name={self.name}, value={self.value}"


class IntProperty(_TypedProperty):

    value = Integer()


class FloatProperty(_TypedProperty):

    value = Float()


class StringProperty(_TypedProperty):

    value = String(allow_none=True)


class DateTimeProperty(_TypedProperty):

    value = DateTime()


class BoolProperty(_TypedProperty):

    value = Bool()


class LinkProperty(_TypedProperty):

    value = String()


# from Python
CLASS_MAPPING = {
    StringProperty: "lpwstr",
    IntProperty: "i4",
    FloatProperty: "r8",
    DateTimeProperty: "filetime",
    BoolProperty: "bool",
    LinkProperty: "linkTarget"
}

XML_MAPPING = {v:k for k,v in CLASS_MAPPING.items()}


class CustomPropertyList(Strict):


    props = Sequence(expected_type=_TypedProperty)

    def __init__(self):
        self.props = []


    @classmethod
    def from_tree(cls, tree):
        """
        Create list from OOXML element
        """
        prop_list = _CustomDocumentPropertyList.from_tree(tree)
        props = []

        for prop in prop_list.property:
            attr = prop.type

            typ = XML_MAPPING.get(attr, None)
            if not typ:
                warn(f"Unknown type for {prop.name}")
                continue
            value = getattr(prop, attr)
            link = prop.linkTarget
            if link is not None:
                typ = LinkProperty
                value = prop.linkTarget

            new_prop = typ(name=prop.name, value=value)
            props.append(new_prop)

        new_prop_list = cls()
        new_prop_list.props = props
        return new_prop_list


    def append(self, prop):
        if prop.name in self.names:
            raise ValueError(f"Property with name {prop.name} already exists")

        self.props.append(prop)


    def to_tree(self):
        props = []

        for p in self.props:
            attr = CLASS_MAPPING.get(p.__class__, None)
            if not attr:
                raise TypeError("Unknown adapter for {p}")
            np = _CustomDocumentProperty(name=p.name, **{attr:p.value})
            if isinstance(p, LinkProperty):
                np._typ = "lpwstr"
                #np.lpwstr = ""
            props.append(np)

        prop_list = _CustomDocumentPropertyList(property=props)
        return prop_list.to_tree()


    def __len__(self):
        return len(self.props)


    @property
    def names(self):
        """List of property names"""
        return [p.name for p in self.props]


    def __getitem__(self, name):
        """
        Get property by name
        """
        for p in self.props:
            if p.name == name:
                return p
        raise KeyError(f"Property with name {name} not found")


    def __delitem__(self, name):
        """
        Delete a propery by name
        """
        for idx, p in enumerate(self.props):
            if p.name == name:
                self.props.pop(idx)
                return
        raise KeyError(f"Property with name {name} not found")


    def __repr__(self):
        return f"{self.__class__.__name__} containing {self.props}"


    def __iter__(self):
        return iter(self.props)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cache.py ===
"""Cache Management"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pip._vendor.packaging.tags import Tag, interpreter_name, interpreter_version
from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.urls import path_to_url

logger = logging.getLogger(__name__)

ORIGIN_JSON_NAME = "origin.json"


def _hash_dict(d: Dict[str, str]) -> str:
    """Return a stable sha224 of a dictionary."""
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha224(s.encode("ascii")).hexdigest()


class Cache:
    """An abstract class - provides cache directories for data from links

    :param cache_dir: The root of the cache.
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        assert not cache_dir or os.path.isabs(cache_dir)
        self.cache_dir = cache_dir or None

    def _get_cache_path_parts(self, link: Link) -> List[str]:
        """Get parts of part that must be os.path.joined with cache_dir"""

        # We want to generate an url to use as our cache key, we don't want to
        # just reuse the URL because it might have other items in the fragment
        # and we don't care about those.
        key_parts = {"url": link.url_without_fragment}
        if link.hash_name is not None and link.hash is not None:
            key_parts[link.hash_name] = link.hash
        if link.subdirectory_fragment:
            key_parts["subdirectory"] = link.subdirectory_fragment

        # Include interpreter name, major and minor version in cache key
        # to cope with ill-behaved sdists that build a different wheel
        # depending on the python version their setup.py is being run on,
        # and don't encode the difference in compatibility tags.
        # https://github.com/pypa/pip/issues/7296
        key_parts["interpreter_name"] = interpreter_name()
        key_parts["interpreter_version"] = interpreter_version()

        # Encode our key url with sha224, we'll use this because it has similar
        # security properties to sha256, but with a shorter total output (and
        # thus less secure). However the differences don't make a lot of
        # difference for our use case here.
        hashed = _hash_dict(key_parts)

        # We want to nest the directories some to prevent having a ton of top
        # level directories where we might run out of sub directories on some
        # FS.
        parts = [hashed[:2], hashed[2:4], hashed[4:6], hashed[6:]]

        return parts

    def _get_candidates(self, link: Link, canonical_package_name: str) -> List[Any]:
        can_not_cache = not self.cache_dir or not canonical_package_name or not link
        if can_not_cache:
            return []

        path = self.get_path_for_link(link)
        if os.path.isdir(path):
            return [(candidate, path) for candidate in os.listdir(path)]
        return []

    def get_path_for_link(self, link: Link) -> str:
        """Return a directory to store cached items in for link."""
        raise NotImplementedError()

    def get(
        self,
        link: Link,
        package_name: Optional[str],
        supported_tags: List[Tag],
    ) -> Link:
        """Returns a link to a cached item if it exists, otherwise returns the
        passed link.
        """
        raise NotImplementedError()


class SimpleWheelCache(Cache):
    """A cache of wheels for future installs."""

    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir)

    def get_path_for_link(self, link: Link) -> str:
        """Return a directory to store cached wheels for link

        Because there are M wheels for any one sdist, we provide a directory
        to cache them in, and then consult that directory when looking up
        cache hits.

        We only insert things into the cache if they have plausible version
        numbers, so that we don't contaminate the cache with things that were
        not unique. E.g. ./package might have dozens of installs done for it
        and build a version of 0.0...and if we built and cached a wheel, we'd
        end up using the same wheel even if the source has been edited.

        :param link: The link of the sdist for which this will cache wheels.
        """
        parts = self._get_cache_path_parts(link)
        assert self.cache_dir
        # Store wheels within the root cache_dir
        return os.path.join(self.cache_dir, "wheels", *parts)

    def get(
        self,
        link: Link,
        package_name: Optional[str],
        supported_tags: List[Tag],
    ) -> Link:
        candidates = []

        if not package_name:
            return link

        canonical_package_name = canonicalize_name(package_name)
        for wheel_name, wheel_dir in self._get_candidates(link, canonical_package_name):
            try:
                wheel = Wheel(wheel_name)
            except InvalidWheelFilename:
                continue
            if canonicalize_name(wheel.name) != canonical_package_name:
                logger.debug(
                    "Ignoring cached wheel %s for %s as it "
                    "does not match the expected distribution name %s.",
                    wheel_name,
                    link,
                    package_name,
                )
                continue
            if not wheel.supported(supported_tags):
                # Built for a different python/arch/etc
                continue
            candidates.append(
                (
                    wheel.support_index_min(supported_tags),
                    wheel_name,
                    wheel_dir,
                )
            )

        if not candidates:
            return link

        _, wheel_name, wheel_dir = min(candidates)
        return Link(path_to_url(os.path.join(wheel_dir, wheel_name)))


class EphemWheelCache(SimpleWheelCache):
    """A SimpleWheelCache that creates it's own temporary cache directory"""

    def __init__(self) -> None:
        self._temp_dir = TempDirectory(
            kind=tempdir_kinds.EPHEM_WHEEL_CACHE,
            globally_managed=True,
        )

        super().__init__(self._temp_dir.path)


class CacheEntry:
    def __init__(
        self,
        link: Link,
        persistent: bool,
    ):
        self.link = link
        self.persistent = persistent
        self.origin: Optional[DirectUrl] = None
        origin_direct_url_path = Path(self.link.file_path).parent / ORIGIN_JSON_NAME
        if origin_direct_url_path.exists():
            try:
                self.origin = DirectUrl.from_json(
                    origin_direct_url_path.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.warning(
                    "Ignoring invalid cache entry origin file %s for %s (%s)",
                    origin_direct_url_path,
                    link.filename,
                    e,
                )


class WheelCache(Cache):
    """Wraps EphemWheelCache and SimpleWheelCache into a single Cache

    This Cache allows for gracefully degradation, using the ephem wheel cache
    when a certain link is not found in the simple wheel cache first.
    """

    def __init__(self, cache_dir: str) -> None:
        super().__init__(cache_dir)
        self._wheel_cache = SimpleWheelCache(cache_dir)
        self._ephem_cache = EphemWheelCache()

    def get_path_for_link(self, link: Link) -> str:
        return self._wheel_cache.get_path_for_link(link)

    def get_ephem_path_for_link(self, link: Link) -> str:
        return self._ephem_cache.get_path_for_link(link)

    def get(
        self,
        link: Link,
        package_name: Optional[str],
        supported_tags: List[Tag],
    ) -> Link:
        cache_entry = self.get_cache_entry(link, package_name, supported_tags)
        if cache_entry is None:
            return link
        return cache_entry.link

    def get_cache_entry(
        self,
        link: Link,
        package_name: Optional[str],
        supported_tags: List[Tag],
    ) -> Optional[CacheEntry]:
        """Returns a CacheEntry with a link to a cached item if it exists or
        None. The cache entry indicates if the item was found in the persistent
        or ephemeral cache.
        """
        retval = self._wheel_cache.get(
            link=link,
            package_name=package_name,
            supported_tags=supported_tags,
        )
        if retval is not link:
            return CacheEntry(retval, persistent=True)

        retval = self._ephem_cache.get(
            link=link,
            package_name=package_name,
            supported_tags=supported_tags,
        )
        if retval is not link:
            return CacheEntry(retval, persistent=False)

        return None

    @staticmethod
    def record_download_origin(cache_dir: str, download_info: DirectUrl) -> None:
        origin_path = Path(cache_dir) / ORIGIN_JSON_NAME
        if origin_path.exists():
            try:
                origin = DirectUrl.from_json(origin_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(
                    "Could not read origin file %s in cache entry (%s). "
                    "Will attempt to overwrite it.",
                    origin_path,
                    e,
                )
            else:
                # TODO: use DirectUrl.equivalent when
                # https://github.com/pypa/pip/pull/10564 is merged.
                if origin.url != download_info.url:
                    logger.warning(
                        "Origin URL %s in cache entry %s does not match download URL "
                        "%s. This is likely a pip bug or a cache corruption issue. "
                        "Will overwrite it with the new value.",
                        origin.url,
                        cache_dir,
                        download_info.url,
                    )
        origin_path.write_text(download_info.to_json(), encoding="utf-8")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_deprecate.py ===
from __future__ import annotations

import inspect
import warnings
from types import ModuleType

import pytest

from .._deprecate import (
    TrioDeprecationWarning,
    deprecated,
    deprecated_alias,
    warn_deprecated,
)
from . import module_with_deprecations


@pytest.fixture
def recwarn_always(recwarn: pytest.WarningsRecorder) -> pytest.WarningsRecorder:
    warnings.simplefilter("always")
    # ResourceWarnings about unclosed sockets can occur nondeterministically
    # (during GC) which throws off the tests in this file
    warnings.simplefilter("ignore", ResourceWarning)
    return recwarn


def _here() -> tuple[str, int]:
    frame = inspect.currentframe()
    assert frame is not None
    assert frame.f_back is not None
    info = inspect.getframeinfo(frame.f_back)
    return (info.filename, info.lineno)


def test_warn_deprecated(recwarn_always: pytest.WarningsRecorder) -> None:
    def deprecated_thing() -> None:
        warn_deprecated("ice", "1.2", issue=1, instead="water")

    deprecated_thing()
    filename, lineno = _here()
    assert len(recwarn_always) == 1
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "ice is deprecated" in got.message.args[0]
    assert "Trio 1.2" in got.message.args[0]
    assert "water instead" in got.message.args[0]
    assert "/issues/1" in got.message.args[0]
    assert got.filename == filename
    assert got.lineno == lineno - 1


def test_warn_deprecated_no_instead_or_issue(
    recwarn_always: pytest.WarningsRecorder,
) -> None:
    # Explicitly no instead or issue
    warn_deprecated("water", "1.3", issue=None, instead=None)
    assert len(recwarn_always) == 1
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "water is deprecated" in got.message.args[0]
    assert "no replacement" in got.message.args[0]
    assert "Trio 1.3" in got.message.args[0]


def test_warn_deprecated_stacklevel(recwarn_always: pytest.WarningsRecorder) -> None:
    def nested1() -> None:
        nested2()

    def nested2() -> None:
        warn_deprecated("x", "1.3", issue=7, instead="y", stacklevel=3)

    filename, lineno = _here()
    nested1()
    got = recwarn_always.pop(DeprecationWarning)
    assert got.filename == filename
    assert got.lineno == lineno + 1


def old() -> None:  # pragma: no cover
    pass


def new() -> None:  # pragma: no cover
    pass


def test_warn_deprecated_formatting(recwarn_always: pytest.WarningsRecorder) -> None:
    warn_deprecated(old, "1.0", issue=1, instead=new)
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "test_deprecate.old is deprecated" in got.message.args[0]
    assert "test_deprecate.new instead" in got.message.args[0]


@deprecated("1.5", issue=123, instead=new)
def deprecated_old() -> int:
    return 3


def test_deprecated_decorator(recwarn_always: pytest.WarningsRecorder) -> None:
    assert deprecated_old() == 3
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "test_deprecate.deprecated_old is deprecated" in got.message.args[0]
    assert "1.5" in got.message.args[0]
    assert "test_deprecate.new" in got.message.args[0]
    assert "issues/123" in got.message.args[0]


class Foo:
    @deprecated("1.0", issue=123, instead="crying")
    def method(self) -> int:
        return 7


def test_deprecated_decorator_method(recwarn_always: pytest.WarningsRecorder) -> None:
    f = Foo()
    assert f.method() == 7
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "test_deprecate.Foo.method is deprecated" in got.message.args[0]


@deprecated("1.2", thing="the thing", issue=None, instead=None)
def deprecated_with_thing() -> int:
    return 72


def test_deprecated_decorator_with_explicit_thing(
    recwarn_always: pytest.WarningsRecorder,
) -> None:
    assert deprecated_with_thing() == 72
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "the thing is deprecated" in got.message.args[0]


def new_hotness() -> str:
    return "new hotness"


old_hotness = deprecated_alias("old_hotness", new_hotness, "1.23", issue=1)


def test_deprecated_alias(recwarn_always: pytest.WarningsRecorder) -> None:
    assert old_hotness() == "new hotness"
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "test_deprecate.old_hotness is deprecated" in got.message.args[0]
    assert "1.23" in got.message.args[0]
    assert "test_deprecate.new_hotness instead" in got.message.args[0]
    assert "issues/1" in got.message.args[0]

    assert isinstance(old_hotness.__doc__, str)
    assert ".. deprecated:: 1.23" in old_hotness.__doc__
    assert "test_deprecate.new_hotness instead" in old_hotness.__doc__
    assert "issues/1>`__" in old_hotness.__doc__


class Alias:
    def new_hotness_method(self) -> str:
        return "new hotness method"

    old_hotness_method = deprecated_alias(
        "Alias.old_hotness_method",
        new_hotness_method,
        "3.21",
        issue=1,
    )


def test_deprecated_alias_method(recwarn_always: pytest.WarningsRecorder) -> None:
    obj = Alias()
    assert obj.old_hotness_method() == "new hotness method"
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    msg = got.message.args[0]
    assert "test_deprecate.Alias.old_hotness_method is deprecated" in msg
    assert "test_deprecate.Alias.new_hotness_method instead" in msg


@deprecated("2.1", issue=1, instead="hi")
def docstring_test1() -> None:  # pragma: no cover
    """Hello!"""


@deprecated("2.1", issue=None, instead="hi")
def docstring_test2() -> None:  # pragma: no cover
    """Hello!"""


@deprecated("2.1", issue=1, instead=None)
def docstring_test3() -> None:  # pragma: no cover
    """Hello!"""


@deprecated("2.1", issue=None, instead=None)
def docstring_test4() -> None:  # pragma: no cover
    """Hello!"""


def test_deprecated_docstring_munging() -> None:
    assert (
        docstring_test1.__doc__
        == """Hello!

.. deprecated:: 2.1
   Use hi instead.
   For details, see `issue #1 <https://github.com/python-trio/trio/issues/1>`__.

"""
    )

    assert (
        docstring_test2.__doc__
        == """Hello!

.. deprecated:: 2.1
   Use hi instead.

"""
    )

    assert (
        docstring_test3.__doc__
        == """Hello!

.. deprecated:: 2.1
   For details, see `issue #1 <https://github.com/python-trio/trio/issues/1>`__.

"""
    )

    assert (
        docstring_test4.__doc__
        == """Hello!

.. deprecated:: 2.1

"""
    )


def test_module_with_deprecations(recwarn_always: pytest.WarningsRecorder) -> None:
    assert module_with_deprecations.regular == "hi"
    assert len(recwarn_always) == 0

    assert type(module_with_deprecations) is ModuleType

    filename, lineno = _here()
    assert module_with_deprecations.dep1 == "value1"  # type: ignore[attr-defined]
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert got.filename == filename
    assert got.lineno == lineno + 1

    assert "module_with_deprecations.dep1" in got.message.args[0]
    assert "Trio 1.1" in got.message.args[0]
    assert "/issues/1" in got.message.args[0]
    assert "value1 instead" in got.message.args[0]

    assert module_with_deprecations.dep2 == "value2"  # type: ignore[attr-defined]
    got = recwarn_always.pop(DeprecationWarning)
    assert isinstance(got.message, Warning)
    assert "instead-string instead" in got.message.args[0]

    with pytest.raises(AttributeError):
        module_with_deprecations.asdf  # type: ignore[attr-defined]  # noqa: B018  # "useless expression"


def test_warning_class() -> None:
    with pytest.deprecated_call():
        warn_deprecated("foo", "bar", issue=None, instead=None)

    # essentially the same as the above check
    with pytest.warns(
        DeprecationWarning,
        match="^foo is deprecated since Trio bar with no replacement$",
    ):
        warn_deprecated("foo", "bar", issue=None, instead=None)

    with pytest.warns(TrioDeprecationWarning):
        warn_deprecated(
            "foo",
            "bar",
            issue=None,
            instead=None,
            use_triodeprecationwarning=True,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\odes.py ===
from bisect import bisect
from ..libmp.backend import xrange

class ODEMethods(object):
    pass

def ode_taylor(ctx, derivs, x0, y0, tol_prec, n):
    h = tol = ctx.ldexp(1, -tol_prec)
    dim = len(y0)
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0
    orig = ctx.prec
    try:
        ctx.prec = orig*(1+n)
        # Use n steps with Euler's method to get
        # evaluation points for derivatives
        for i in range(n):
            fxy = derivs(x, y)
            y = [y[i]+h*fxy[i] for i in xrange(len(y))]
            x += h
            xs.append(x)
            ys.append(y)
        # Compute derivatives
        ser = [[] for d in range(dim)]
        for j in range(n+1):
            s = [0]*dim
            b = (-1) ** (j & 1)
            k = 1
            for i in range(j+1):
                for d in range(dim):
                    s[d] += b * ys[i][d]
                b = (b * (j-k+1)) // (-k)
                k += 1
            scale = h**(-j) / ctx.fac(j)
            for d in range(dim):
                s[d] = s[d] * scale
                ser[d].append(s[d])
    finally:
        ctx.prec = orig
    # Estimate radius for which we can get full accuracy.
    # XXX: do this right for zeros
    radius = ctx.one
    for ts in ser:
        if ts[-1]:
            radius = min(radius, ctx.nthroot(tol/abs(ts[-1]), n))
    radius /= 2  # XXX
    return ser, x0+radius

def odefun(ctx, F, x0, y0, tol=None, degree=None, method='taylor', verbose=False):
    r"""
    Returns a function `y(x) = [y_0(x), y_1(x), \ldots, y_n(x)]`
    that is a numerical solution of the `n+1`-dimensional first-order
    ordinary differential equation (ODE) system

    .. math ::

        y_0'(x) = F_0(x, [y_0(x), y_1(x), \ldots, y_n(x)])

        y_1'(x) = F_1(x, [y_0(x), y_1(x), \ldots, y_n(x)])

        \vdots

        y_n'(x) = F_n(x, [y_0(x), y_1(x), \ldots, y_n(x)])

    The derivatives are specified by the vector-valued function
    *F* that evaluates
    `[y_0', \ldots, y_n'] = F(x, [y_0, \ldots, y_n])`.
    The initial point `x_0` is specified by the scalar argument *x0*,
    and the initial value `y(x_0) =  [y_0(x_0), \ldots, y_n(x_0)]` is
    specified by the vector argument *y0*.

    For convenience, if the system is one-dimensional, you may optionally
    provide just a scalar value for *y0*. In this case, *F* should accept
    a scalar *y* argument and return a scalar. The solution function
    *y* will return scalar values instead of length-1 vectors.

    Evaluation of the solution function `y(x)` is permitted
    for any `x \ge x_0`.

    A high-order ODE can be solved by transforming it into first-order
    vector form. This transformation is described in standard texts
    on ODEs. Examples will also be given below.

    **Options, speed and accuracy**

    By default, :func:`~mpmath.odefun` uses a high-order Taylor series
    method. For reasonably well-behaved problems, the solution will
    be fully accurate to within the working precision. Note that
    *F* must be possible to evaluate to very high precision
    for the generation of Taylor series to work.

    To get a faster but less accurate solution, you can set a large
    value for *tol* (which defaults roughly to *eps*). If you just
    want to plot the solution or perform a basic simulation,
    *tol = 0.01* is likely sufficient.

    The *degree* argument controls the degree of the solver (with
    *method='taylor'*, this is the degree of the Taylor series
    expansion). A higher degree means that a longer step can be taken
    before a new local solution must be generated from *F*,
    meaning that fewer steps are required to get from `x_0` to a given
    `x_1`. On the other hand, a higher degree also means that each
    local solution becomes more expensive (i.e., more evaluations of
    *F* are required per step, and at higher precision).

    The optimal setting therefore involves a tradeoff. Generally,
    decreasing the *degree* for Taylor series is likely to give faster
    solution at low precision, while increasing is likely to be better
    at higher precision.

    The function
    object returned by :func:`~mpmath.odefun` caches the solutions at all step
    points and uses polynomial interpolation between step points.
    Therefore, once `y(x_1)` has been evaluated for some `x_1`,
    `y(x)` can be evaluated very quickly for any `x_0 \le x \le x_1`.
    and continuing the evaluation up to `x_2 > x_1` is also fast.

    **Examples of first-order ODEs**

    We will solve the standard test problem `y'(x) = y(x), y(0) = 1`
    which has explicit solution `y(x) = \exp(x)`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> f = odefun(lambda x, y: y, 0, 1)
        >>> for x in [0, 1, 2.5]:
        ...     print((f(x), exp(x)))
        ...
        (1.0, 1.0)
        (2.71828182845905, 2.71828182845905)
        (12.1824939607035, 12.1824939607035)

    The solution with high precision::

        >>> mp.dps = 50
        >>> f = odefun(lambda x, y: y, 0, 1)
        >>> f(1)
        2.7182818284590452353602874713526624977572470937
        >>> exp(1)
        2.7182818284590452353602874713526624977572470937

    Using the more general vectorized form, the test problem
    can be input as (note that *f* returns a 1-element vector)::

        >>> mp.dps = 15
        >>> f = odefun(lambda x, y: [y[0]], 0, [1])
        >>> f(1)
        [2.71828182845905]

    :func:`~mpmath.odefun` can solve nonlinear ODEs, which are generally
    impossible (and at best difficult) to solve analytically. As
    an example of a nonlinear ODE, we will solve `y'(x) = x \sin(y(x))`
    for `y(0) = \pi/2`. An exact solution happens to be known
    for this problem, and is given by
    `y(x) = 2 \tan^{-1}\left(\exp\left(x^2/2\right)\right)`::

        >>> f = odefun(lambda x, y: x*sin(y), 0, pi/2)
        >>> for x in [2, 5, 10]:
        ...     print((f(x), 2*atan(exp(mpf(x)**2/2))))
        ...
        (2.87255666284091, 2.87255666284091)
        (3.14158520028345, 3.14158520028345)
        (3.14159265358979, 3.14159265358979)

    If `F` is independent of `y`, an ODE can be solved using direct
    integration. We can therefore obtain a reference solution with
    :func:`~mpmath.quad`::

        >>> f = lambda x: (1+x**2)/(1+x**3)
        >>> g = odefun(lambda x, y: f(x), pi, 0)
        >>> g(2*pi)
        0.72128263801696
        >>> quad(f, [pi, 2*pi])
        0.72128263801696

    **Examples of second-order ODEs**

    We will solve the harmonic oscillator equation `y''(x) + y(x) = 0`.
    To do this, we introduce the helper functions `y_0 = y, y_1 = y_0'`
    whereby the original equation can be written as `y_1' + y_0' = 0`. Put
    together, we get the first-order, two-dimensional vector ODE

    .. math ::

        \begin{cases}
        y_0' = y_1 \\
        y_1' = -y_0
        \end{cases}

    To get a well-defined IVP, we need two initial values. With
    `y(0) = y_0(0) = 1` and `-y'(0) = y_1(0) = 0`, the problem will of
    course be solved by `y(x) = y_0(x) = \cos(x)` and
    `-y'(x) = y_1(x) = \sin(x)`. We check this::

        >>> f = odefun(lambda x, y: [-y[1], y[0]], 0, [1, 0])
        >>> for x in [0, 1, 2.5, 10]:
        ...     nprint(f(x), 15)
        ...     nprint([cos(x), sin(x)], 15)
        ...     print("---")
        ...
        [1.0, 0.0]
        [1.0, 0.0]
        ---
        [0.54030230586814, 0.841470984807897]
        [0.54030230586814, 0.841470984807897]
        ---
        [-0.801143615546934, 0.598472144103957]
        [-0.801143615546934, 0.598472144103957]
        ---
        [-0.839071529076452, -0.54402111088937]
        [-0.839071529076452, -0.54402111088937]
        ---

    Note that we get both the sine and the cosine solutions
    simultaneously.

    **TODO**

    * Better automatic choice of degree and step size
    * Make determination of Taylor series convergence radius
      more robust
    * Allow solution for `x < x_0`
    * Allow solution for complex `x`
    * Test for difficult (ill-conditioned) problems
    * Implement Runge-Kutta and other algorithms

    """
    if tol:
        tol_prec = int(-ctx.log(tol, 2))+10
    else:
        tol_prec = ctx.prec+10
    degree = degree or (3 + int(3*ctx.dps/2.))
    workprec = ctx.prec + 40
    try:
        len(y0)
        return_vector = True
    except TypeError:
        F_ = F
        F = lambda x, y: [F_(x, y[0])]
        y0 = [y0]
        return_vector = False
    ser, xb = ode_taylor(ctx, F, x0, y0, tol_prec, degree)
    series_boundaries = [x0, xb]
    series_data = [(ser, x0, xb)]
    # We will be working with vectors of Taylor series
    def mpolyval(ser, a):
        return [ctx.polyval(s[::-1], a) for s in ser]
    # Find nearest expansion point; compute if necessary
    def get_series(x):
        if x < x0:
            raise ValueError
        n = bisect(series_boundaries, x)
        if n < len(series_boundaries):
            return series_data[n-1]
        while 1:
            ser, xa, xb = series_data[-1]
            if verbose:
                print("Computing Taylor series for [%f, %f]" % (xa, xb))
            y = mpolyval(ser, xb-xa)
            xa = xb
            ser, xb = ode_taylor(ctx, F, xb, y, tol_prec, degree)
            series_boundaries.append(xb)
            series_data.append((ser, xa, xb))
            if x <= xb:
                return series_data[-1]
    # Evaluation function
    def interpolant(x):
        x = ctx.convert(x)
        orig = ctx.prec
        try:
            ctx.prec = workprec
            ser, xa, xb = get_series(x)
            y = mpolyval(ser, x-xa)
        finally:
            ctx.prec = orig
        if return_vector:
            return [+yk for yk in y]
        else:
            return +y[0]
    return interpolant

ODEMethods.odefun = odefun

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\engine_builder_ort_trt.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os

import torch
from cuda import cudart
from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType
from packaging import version

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import CudaSession

logger = logging.getLogger(__name__)


class OrtTensorrtEngine(CudaSession):
    def __init__(
        self,
        engine_path,
        device_id,
        onnx_path,
        fp16,
        input_profile,
        workspace_size,
        enable_cuda_graph,
        timing_cache_path=None,
    ):
        self.engine_path = engine_path
        self.ort_trt_provider_options = self.get_tensorrt_provider_options(
            input_profile,
            workspace_size,
            fp16,
            device_id,
            enable_cuda_graph,
            timing_cache_path=timing_cache_path,
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        logger.info("creating TRT EP session for %s", onnx_path)
        ort_session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=[
                ("TensorrtExecutionProvider", self.ort_trt_provider_options),
            ],
        )
        logger.info("created TRT EP session for %s", onnx_path)

        device = torch.device("cuda", device_id)
        super().__init__(ort_session, device, enable_cuda_graph)

    def get_tensorrt_provider_options(
        self, input_profile, workspace_size, fp16, device_id, enable_cuda_graph, timing_cache_path=None
    ):
        trt_ep_options = {
            "device_id": device_id,
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_timing_cache_enable": True,
            "trt_detailed_build_log": True,
            "trt_engine_cache_path": self.engine_path,
        }

        if version.parse(ort.__version__) > version.parse("1.16.2") and timing_cache_path is not None:
            trt_ep_options["trt_timing_cache_path"] = timing_cache_path

        if enable_cuda_graph:
            trt_ep_options["trt_cuda_graph_enable"] = True

        if workspace_size > 0:
            trt_ep_options["trt_max_workspace_size"] = workspace_size

        if input_profile:
            min_shapes = []
            max_shapes = []
            opt_shapes = []
            for name, profile in input_profile.items():
                assert isinstance(profile, list) and len(profile) == 3
                min_shape = profile[0]
                opt_shape = profile[1]
                max_shape = profile[2]
                assert len(min_shape) == len(opt_shape) and len(opt_shape) == len(max_shape)

                min_shapes.append(f"{name}:" + "x".join([str(x) for x in min_shape]))
                opt_shapes.append(f"{name}:" + "x".join([str(x) for x in opt_shape]))
                max_shapes.append(f"{name}:" + "x".join([str(x) for x in max_shape]))

            trt_ep_options["trt_profile_min_shapes"] = ",".join(min_shapes)
            trt_ep_options["trt_profile_max_shapes"] = ",".join(max_shapes)
            trt_ep_options["trt_profile_opt_shapes"] = ",".join(opt_shapes)

        logger.info("trt_ep_options=%s", trt_ep_options)

        return trt_ep_options

    def allocate_buffers(self, shape_dict, device):
        super().allocate_buffers(shape_dict)


class OrtTensorrtEngineBuilder(EngineBuilder):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        device="cuda",
        use_cuda_graph=False,
    ):
        """
        Initializes the ONNX Runtime TensorRT ExecutionProvider Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            device (str):
                device to run.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        super().__init__(
            EngineType.ORT_TRT,
            pipeline_info,
            max_batch_size=max_batch_size,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )

    def has_engine_file(self, engine_path):
        if os.path.isdir(engine_path):
            children = os.scandir(engine_path)
            for entry in children:
                if entry.is_file() and entry.name.endswith(".engine"):
                    return True
        return False

    def get_work_space_size(self, model_name, max_workspace_size):
        gibibyte = 2**30
        workspace_size = 4 * gibibyte if model_name == "clip" else max_workspace_size
        if workspace_size == 0:
            _, free_mem, _ = cudart.cudaMemGetInfo()
            # The following logic are adopted from TensorRT demo diffusion.
            if free_mem > 6 * gibibyte:
                workspace_size = free_mem - 4 * gibibyte
        return workspace_size

    def build_engines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_image_height,
        opt_image_width,
        opt_batch_size=1,
        static_batch=False,
        static_image_shape=True,
        max_workspace_size=0,
        device_id=0,
        timing_cache=None,
    ):
        self.torch_device = torch.device("cuda", device_id)
        self.load_models(framework_model_dir)

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Load lora only when we need export text encoder or UNet to ONNX.
        load_lora = False
        if self.pipeline_info.lora_weights:
            for model_name, model_obj in self.models.items():
                if model_name not in ["clip", "clip2", "unet", "unetxl"]:
                    continue
                profile_id = model_obj.get_profile_id(
                    opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
                )
                engine_path = self.get_engine_path(engine_dir, model_name, profile_id)
                if not self.has_engine_file(engine_path):
                    onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                    onnx_opt_path = self.get_onnx_path(model_name, onnx_dir, opt=True)
                    if not os.path.exists(onnx_opt_path):
                        if not os.path.exists(onnx_path):
                            load_lora = True
                            break

        # Export models to ONNX
        self.disable_torch_spda()
        pipe = self.load_pipeline_with_lora() if load_lora else None

        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            profile_id = model_obj.get_profile_id(
                opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
            )
            engine_path = self.get_engine_path(engine_dir, model_name, profile_id)
            if not self.has_engine_file(engine_path):
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.get_onnx_path(model_name, onnx_dir, opt=True)
                if not os.path.exists(onnx_opt_path):
                    if not os.path.exists(onnx_path):
                        logger.info(f"Exporting model: {onnx_path}")
                        model = self.get_or_load_model(pipe, model_name, model_obj, framework_model_dir)

                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=onnx_opset,
                                do_constant_folding=True,
                                input_names=model_obj.get_input_names(),
                                output_names=model_obj.get_output_names(),
                                dynamic_axes=model_obj.get_dynamic_axes(),
                            )
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        logger.info("Found cached model: %s", onnx_path)

                    # Optimize onnx
                    if not os.path.exists(onnx_opt_path):
                        logger.info("Generating optimizing model: %s", onnx_opt_path)
                        model_obj.optimize_trt(onnx_path, onnx_opt_path)
                    else:
                        logger.info("Found cached optimized model: %s", onnx_opt_path)
        self.enable_torch_spda()

        built_engines = {}
        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            profile_id = model_obj.get_profile_id(
                opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
            )

            engine_path = self.get_engine_path(engine_dir, model_name, profile_id)
            onnx_opt_path = self.get_onnx_path(model_name, onnx_dir, opt=True)
            if not self.has_engine_file(engine_path):
                logger.info(
                    "Building TensorRT engine for %s from %s to %s. It can take a while to complete...",
                    model_name,
                    onnx_opt_path,
                    engine_path,
                )
            else:
                logger.info("Reuse cached TensorRT engine in directory %s", engine_path)

            input_profile = model_obj.get_input_profile(
                opt_batch_size,
                opt_image_height,
                opt_image_width,
                static_batch=static_batch,
                static_image_shape=static_image_shape,
            )

            engine = OrtTensorrtEngine(
                engine_path,
                device_id,
                onnx_opt_path,
                fp16=True,
                input_profile=input_profile,
                workspace_size=self.get_work_space_size(model_name, max_workspace_size),
                enable_cuda_graph=self.use_cuda_graph,
                timing_cache_path=timing_cache,
            )

            built_engines[model_name] = engine

        self.engines = built_engines

    def run_engine(self, model_name, feed_dict):
        return self.engines[model_name].infer(feed_dict)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\media.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Media (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class PlayerId(str):
    '''
    Players will get an ID that is unique within the agent context.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PlayerId:
        return cls(json)

    def __repr__(self):
        return 'PlayerId({})'.format(super().__repr__())


class Timestamp(float):
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> Timestamp:
        return cls(json)

    def __repr__(self):
        return 'Timestamp({})'.format(super().__repr__())


@dataclass
class PlayerMessage:
    '''
    Have one type per entry in MediaLogRecord::Type
    Corresponds to kMessage
    '''
    #: Keep in sync with MediaLogMessageLevel
    #: We are currently keeping the message level 'error' separate from the
    #: PlayerError type because right now they represent different things,
    #: this one being a DVLOG(ERROR) style log message that gets printed
    #: based on what log level is selected in the UI, and the other is a
    #: representation of a media::PipelineStatus object. Soon however we're
    #: going to be moving away from using PipelineStatus for errors and
    #: introducing a new error type which should hopefully let us integrate
    #: the error log level into the PlayerError type.
    level: str

    message: str

    def to_json(self):
        json = dict()
        json['level'] = self.level
        json['message'] = self.message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            level=str(json['level']),
            message=str(json['message']),
        )


@dataclass
class PlayerProperty:
    '''
    Corresponds to kMediaPropertyChange
    '''
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class PlayerEvent:
    '''
    Corresponds to kMediaEventTriggered
    '''
    timestamp: Timestamp

    value: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            timestamp=Timestamp.from_json(json['timestamp']),
            value=str(json['value']),
        )


@dataclass
class PlayerErrorSourceLocation:
    '''
    Represents logged source line numbers reported in an error.
    NOTE: file and line are from chromium c++ implementation code, not js.
    '''
    file: str

    line: int

    def to_json(self):
        json = dict()
        json['file'] = self.file
        json['line'] = self.line
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            file=str(json['file']),
            line=int(json['line']),
        )


@dataclass
class PlayerError:
    '''
    Corresponds to kMediaError
    '''
    error_type: str

    #: Code is the numeric enum entry for a specific set of error codes, such
    #: as PipelineStatusCodes in media/base/pipeline_status.h
    code: int

    #: A trace of where this error was caused / where it passed through.
    stack: typing.List[PlayerErrorSourceLocation]

    #: Errors potentially have a root cause error, ie, a DecoderError might be
    #: caused by an WindowsError
    cause: typing.List[PlayerError]

    #: Extra data attached to an error, such as an HRESULT, Video Codec, etc.
    data: dict

    def to_json(self):
        json = dict()
        json['errorType'] = self.error_type
        json['code'] = self.code
        json['stack'] = [i.to_json() for i in self.stack]
        json['cause'] = [i.to_json() for i in self.cause]
        json['data'] = self.data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_type=str(json['errorType']),
            code=int(json['code']),
            stack=[PlayerErrorSourceLocation.from_json(i) for i in json['stack']],
            cause=[PlayerError.from_json(i) for i in json['cause']],
            data=dict(json['data']),
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the Media domain
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Media.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the Media domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Media.disable',
    }
    json = yield cmd_dict


@event_class('Media.playerPropertiesChanged')
@dataclass
class PlayerPropertiesChanged:
    '''
    This can be called multiple times, and can be used to set / override /
    remove player properties. A null propValue indicates removal.
    '''
    player_id: PlayerId
    properties: typing.List[PlayerProperty]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerPropertiesChanged:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            properties=[PlayerProperty.from_json(i) for i in json['properties']]
        )


@event_class('Media.playerEventsAdded')
@dataclass
class PlayerEventsAdded:
    '''
    Send events as a list, allowing them to be batched on the browser for less
    congestion. If batched, events must ALWAYS be in chronological order.
    '''
    player_id: PlayerId
    events: typing.List[PlayerEvent]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerEventsAdded:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            events=[PlayerEvent.from_json(i) for i in json['events']]
        )


@event_class('Media.playerMessagesLogged')
@dataclass
class PlayerMessagesLogged:
    '''
    Send a list of any messages that need to be delivered.
    '''
    player_id: PlayerId
    messages: typing.List[PlayerMessage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerMessagesLogged:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            messages=[PlayerMessage.from_json(i) for i in json['messages']]
        )


@event_class('Media.playerErrorsRaised')
@dataclass
class PlayerErrorsRaised:
    '''
    Send a list of any errors that need to be delivered.
    '''
    player_id: PlayerId
    errors: typing.List[PlayerError]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerErrorsRaised:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            errors=[PlayerError.from_json(i) for i in json['errors']]
        )


@event_class('Media.playersCreated')
@dataclass
class PlayersCreated:
    '''
    Called whenever a player is created, or when a new agent joins and receives
    a list of active players. If an agent is restored, it will receive the full
    list of player ids and all events again.
    '''
    players: typing.List[PlayerId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayersCreated:
        return cls(
            players=[PlayerId.from_json(i) for i in json['players']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\media.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Media (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class PlayerId(str):
    '''
    Players will get an ID that is unique within the agent context.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PlayerId:
        return cls(json)

    def __repr__(self):
        return 'PlayerId({})'.format(super().__repr__())


class Timestamp(float):
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> Timestamp:
        return cls(json)

    def __repr__(self):
        return 'Timestamp({})'.format(super().__repr__())


@dataclass
class PlayerMessage:
    '''
    Have one type per entry in MediaLogRecord::Type
    Corresponds to kMessage
    '''
    #: Keep in sync with MediaLogMessageLevel
    #: We are currently keeping the message level 'error' separate from the
    #: PlayerError type because right now they represent different things,
    #: this one being a DVLOG(ERROR) style log message that gets printed
    #: based on what log level is selected in the UI, and the other is a
    #: representation of a media::PipelineStatus object. Soon however we're
    #: going to be moving away from using PipelineStatus for errors and
    #: introducing a new error type which should hopefully let us integrate
    #: the error log level into the PlayerError type.
    level: str

    message: str

    def to_json(self):
        json = dict()
        json['level'] = self.level
        json['message'] = self.message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            level=str(json['level']),
            message=str(json['message']),
        )


@dataclass
class PlayerProperty:
    '''
    Corresponds to kMediaPropertyChange
    '''
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class PlayerEvent:
    '''
    Corresponds to kMediaEventTriggered
    '''
    timestamp: Timestamp

    value: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            timestamp=Timestamp.from_json(json['timestamp']),
            value=str(json['value']),
        )


@dataclass
class PlayerErrorSourceLocation:
    '''
    Represents logged source line numbers reported in an error.
    NOTE: file and line are from chromium c++ implementation code, not js.
    '''
    file: str

    line: int

    def to_json(self):
        json = dict()
        json['file'] = self.file
        json['line'] = self.line
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            file=str(json['file']),
            line=int(json['line']),
        )


@dataclass
class PlayerError:
    '''
    Corresponds to kMediaError
    '''
    error_type: str

    #: Code is the numeric enum entry for a specific set of error codes, such
    #: as PipelineStatusCodes in media/base/pipeline_status.h
    code: int

    #: A trace of where this error was caused / where it passed through.
    stack: typing.List[PlayerErrorSourceLocation]

    #: Errors potentially have a root cause error, ie, a DecoderError might be
    #: caused by an WindowsError
    cause: typing.List[PlayerError]

    #: Extra data attached to an error, such as an HRESULT, Video Codec, etc.
    data: dict

    def to_json(self):
        json = dict()
        json['errorType'] = self.error_type
        json['code'] = self.code
        json['stack'] = [i.to_json() for i in self.stack]
        json['cause'] = [i.to_json() for i in self.cause]
        json['data'] = self.data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_type=str(json['errorType']),
            code=int(json['code']),
            stack=[PlayerErrorSourceLocation.from_json(i) for i in json['stack']],
            cause=[PlayerError.from_json(i) for i in json['cause']],
            data=dict(json['data']),
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the Media domain
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Media.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the Media domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Media.disable',
    }
    json = yield cmd_dict


@event_class('Media.playerPropertiesChanged')
@dataclass
class PlayerPropertiesChanged:
    '''
    This can be called multiple times, and can be used to set / override /
    remove player properties. A null propValue indicates removal.
    '''
    player_id: PlayerId
    properties: typing.List[PlayerProperty]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerPropertiesChanged:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            properties=[PlayerProperty.from_json(i) for i in json['properties']]
        )


@event_class('Media.playerEventsAdded')
@dataclass
class PlayerEventsAdded:
    '''
    Send events as a list, allowing them to be batched on the browser for less
    congestion. If batched, events must ALWAYS be in chronological order.
    '''
    player_id: PlayerId
    events: typing.List[PlayerEvent]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerEventsAdded:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            events=[PlayerEvent.from_json(i) for i in json['events']]
        )


@event_class('Media.playerMessagesLogged')
@dataclass
class PlayerMessagesLogged:
    '''
    Send a list of any messages that need to be delivered.
    '''
    player_id: PlayerId
    messages: typing.List[PlayerMessage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerMessagesLogged:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            messages=[PlayerMessage.from_json(i) for i in json['messages']]
        )


@event_class('Media.playerErrorsRaised')
@dataclass
class PlayerErrorsRaised:
    '''
    Send a list of any errors that need to be delivered.
    '''
    player_id: PlayerId
    errors: typing.List[PlayerError]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerErrorsRaised:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            errors=[PlayerError.from_json(i) for i in json['errors']]
        )


@event_class('Media.playersCreated')
@dataclass
class PlayersCreated:
    '''
    Called whenever a player is created, or when a new agent joins and receives
    a list of active players. If an agent is restored, it will receive the full
    list of player ids and all events again.
    '''
    players: typing.List[PlayerId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayersCreated:
        return cls(
            players=[PlayerId.from_json(i) for i in json['players']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\media.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Media (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class PlayerId(str):
    '''
    Players will get an ID that is unique within the agent context.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PlayerId:
        return cls(json)

    def __repr__(self):
        return 'PlayerId({})'.format(super().__repr__())


class Timestamp(float):
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> Timestamp:
        return cls(json)

    def __repr__(self):
        return 'Timestamp({})'.format(super().__repr__())


@dataclass
class PlayerMessage:
    '''
    Have one type per entry in MediaLogRecord::Type
    Corresponds to kMessage
    '''
    #: Keep in sync with MediaLogMessageLevel
    #: We are currently keeping the message level 'error' separate from the
    #: PlayerError type because right now they represent different things,
    #: this one being a DVLOG(ERROR) style log message that gets printed
    #: based on what log level is selected in the UI, and the other is a
    #: representation of a media::PipelineStatus object. Soon however we're
    #: going to be moving away from using PipelineStatus for errors and
    #: introducing a new error type which should hopefully let us integrate
    #: the error log level into the PlayerError type.
    level: str

    message: str

    def to_json(self):
        json = dict()
        json['level'] = self.level
        json['message'] = self.message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            level=str(json['level']),
            message=str(json['message']),
        )


@dataclass
class PlayerProperty:
    '''
    Corresponds to kMediaPropertyChange
    '''
    name: str

    value: str

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
        )


@dataclass
class PlayerEvent:
    '''
    Corresponds to kMediaEventTriggered
    '''
    timestamp: Timestamp

    value: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            timestamp=Timestamp.from_json(json['timestamp']),
            value=str(json['value']),
        )


@dataclass
class PlayerErrorSourceLocation:
    '''
    Represents logged source line numbers reported in an error.
    NOTE: file and line are from chromium c++ implementation code, not js.
    '''
    file: str

    line: int

    def to_json(self):
        json = dict()
        json['file'] = self.file
        json['line'] = self.line
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            file=str(json['file']),
            line=int(json['line']),
        )


@dataclass
class PlayerError:
    '''
    Corresponds to kMediaError
    '''
    error_type: str

    #: Code is the numeric enum entry for a specific set of error codes, such
    #: as PipelineStatusCodes in media/base/pipeline_status.h
    code: int

    #: A trace of where this error was caused / where it passed through.
    stack: typing.List[PlayerErrorSourceLocation]

    #: Errors potentially have a root cause error, ie, a DecoderError might be
    #: caused by an WindowsError
    cause: typing.List[PlayerError]

    #: Extra data attached to an error, such as an HRESULT, Video Codec, etc.
    data: dict

    def to_json(self):
        json = dict()
        json['errorType'] = self.error_type
        json['code'] = self.code
        json['stack'] = [i.to_json() for i in self.stack]
        json['cause'] = [i.to_json() for i in self.cause]
        json['data'] = self.data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_type=str(json['errorType']),
            code=int(json['code']),
            stack=[PlayerErrorSourceLocation.from_json(i) for i in json['stack']],
            cause=[PlayerError.from_json(i) for i in json['cause']],
            data=dict(json['data']),
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables the Media domain
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Media.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables the Media domain.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Media.disable',
    }
    json = yield cmd_dict


@event_class('Media.playerPropertiesChanged')
@dataclass
class PlayerPropertiesChanged:
    '''
    This can be called multiple times, and can be used to set / override /
    remove player properties. A null propValue indicates removal.
    '''
    player_id: PlayerId
    properties: typing.List[PlayerProperty]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerPropertiesChanged:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            properties=[PlayerProperty.from_json(i) for i in json['properties']]
        )


@event_class('Media.playerEventsAdded')
@dataclass
class PlayerEventsAdded:
    '''
    Send events as a list, allowing them to be batched on the browser for less
    congestion. If batched, events must ALWAYS be in chronological order.
    '''
    player_id: PlayerId
    events: typing.List[PlayerEvent]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerEventsAdded:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            events=[PlayerEvent.from_json(i) for i in json['events']]
        )


@event_class('Media.playerMessagesLogged')
@dataclass
class PlayerMessagesLogged:
    '''
    Send a list of any messages that need to be delivered.
    '''
    player_id: PlayerId
    messages: typing.List[PlayerMessage]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerMessagesLogged:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            messages=[PlayerMessage.from_json(i) for i in json['messages']]
        )


@event_class('Media.playerErrorsRaised')
@dataclass
class PlayerErrorsRaised:
    '''
    Send a list of any errors that need to be delivered.
    '''
    player_id: PlayerId
    errors: typing.List[PlayerError]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerErrorsRaised:
        return cls(
            player_id=PlayerId.from_json(json['playerId']),
            errors=[PlayerError.from_json(i) for i in json['errors']]
        )


@event_class('Media.playersCreated')
@dataclass
class PlayersCreated:
    '''
    Called whenever a player is created, or when a new agent joins and receives
    a list of active players. If an agent is restored, it will receive the full
    list of player ids and all events again.
    '''
    players: typing.List[PlayerId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayersCreated:
        return cls(
            players=[PlayerId.from_json(i) for i in json['players']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\log.py ===
# log.py
# Copyright (C) 2006-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
# Includes alterations by Vinay Sajip vinay_sajip@yahoo.co.uk
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Logging control and utilities.

Control of logging for SA can be performed from the regular python logging
module.  The regular dotted module namespace is used, starting at
'sqlalchemy'.  For class-level logging, the class name is appended.

The "echo" keyword parameter, available on SQLA :class:`_engine.Engine`
and :class:`_pool.Pool` objects, corresponds to a logger specific to that
instance only.

"""
from __future__ import annotations

import logging
import sys
from typing import Any
from typing import Optional
from typing import overload
from typing import Set
from typing import Type
from typing import TypeVar
from typing import Union

from .util import py311
from .util import py38
from .util.typing import Literal


if py38:
    STACKLEVEL = True
    # needed as of py3.11.0b1
    # #8019
    STACKLEVEL_OFFSET = 2 if py311 else 1
else:
    STACKLEVEL = False
    STACKLEVEL_OFFSET = 0

_IT = TypeVar("_IT", bound="Identified")

_EchoFlagType = Union[None, bool, Literal["debug"]]

# set initial level to WARN.  This so that
# log statements don't occur in the absence of explicit
# logging being enabled for 'sqlalchemy'.
rootlogger = logging.getLogger("sqlalchemy")
if rootlogger.level == logging.NOTSET:
    rootlogger.setLevel(logging.WARN)


def _add_default_handler(logger: logging.Logger) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)


_logged_classes: Set[Type[Identified]] = set()


def _qual_logger_name_for_cls(cls: Type[Identified]) -> str:
    return (
        getattr(cls, "_sqla_logger_namespace", None)
        or cls.__module__ + "." + cls.__name__
    )


def class_logger(cls: Type[_IT]) -> Type[_IT]:
    logger = logging.getLogger(_qual_logger_name_for_cls(cls))
    cls._should_log_debug = lambda self: logger.isEnabledFor(  # type: ignore[method-assign]  # noqa: E501
        logging.DEBUG
    )
    cls._should_log_info = lambda self: logger.isEnabledFor(  # type: ignore[method-assign]  # noqa: E501
        logging.INFO
    )
    cls.logger = logger
    _logged_classes.add(cls)
    return cls


_IdentifiedLoggerType = Union[logging.Logger, "InstanceLogger"]


class Identified:
    __slots__ = ()

    logging_name: Optional[str] = None

    logger: _IdentifiedLoggerType

    _echo: _EchoFlagType

    def _should_log_debug(self) -> bool:
        return self.logger.isEnabledFor(logging.DEBUG)

    def _should_log_info(self) -> bool:
        return self.logger.isEnabledFor(logging.INFO)


class InstanceLogger:
    """A logger adapter (wrapper) for :class:`.Identified` subclasses.

    This allows multiple instances (e.g. Engine or Pool instances)
    to share a logger, but have its verbosity controlled on a
    per-instance basis.

    The basic functionality is to return a logging level
    which is based on an instance's echo setting.

    Default implementation is:

    'debug' -> logging.DEBUG
    True    -> logging.INFO
    False   -> Effective level of underlying logger (
    logging.WARNING by default)
    None    -> same as False
    """

    # Map echo settings to logger levels
    _echo_map = {
        None: logging.NOTSET,
        False: logging.NOTSET,
        True: logging.INFO,
        "debug": logging.DEBUG,
    }

    _echo: _EchoFlagType

    __slots__ = ("echo", "logger")

    def __init__(self, echo: _EchoFlagType, name: str):
        self.echo = echo
        self.logger = logging.getLogger(name)

        # if echo flag is enabled and no handlers,
        # add a handler to the list
        if self._echo_map[echo] <= logging.INFO and not self.logger.handlers:
            _add_default_handler(self.logger)

    #
    # Boilerplate convenience methods
    #
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a debug call to the underlying logger."""

        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an info call to the underlying logger."""

        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a warning call to the underlying logger."""

        self.log(logging.WARNING, msg, *args, **kwargs)

    warn = warning

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Delegate an error call to the underlying logger.
        """
        self.log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an exception call to the underlying logger."""

        kwargs["exc_info"] = 1
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a critical call to the underlying logger."""

        self.log(logging.CRITICAL, msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a log call to the underlying logger.

        The level here is determined by the echo
        flag as well as that of the underlying logger, and
        logger._log() is called directly.

        """

        # inline the logic from isEnabledFor(),
        # getEffectiveLevel(), to avoid overhead.

        if self.logger.manager.disable >= level:
            return

        selected_level = self._echo_map[self.echo]
        if selected_level == logging.NOTSET:
            selected_level = self.logger.getEffectiveLevel()

        if level >= selected_level:
            if STACKLEVEL:
                kwargs["stacklevel"] = (
                    kwargs.get("stacklevel", 1) + STACKLEVEL_OFFSET
                )

            self.logger._log(level, msg, args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        """Is this logger enabled for level 'level'?"""

        if self.logger.manager.disable >= level:
            return False
        return level >= self.getEffectiveLevel()

    def getEffectiveLevel(self) -> int:
        """What's the effective level for this logger?"""

        level = self._echo_map[self.echo]
        if level == logging.NOTSET:
            level = self.logger.getEffectiveLevel()
        return level


def instance_logger(
    instance: Identified, echoflag: _EchoFlagType = None
) -> None:
    """create a logger for an instance that implements :class:`.Identified`."""

    if instance.logging_name:
        name = "%s.%s" % (
            _qual_logger_name_for_cls(instance.__class__),
            instance.logging_name,
        )
    else:
        name = _qual_logger_name_for_cls(instance.__class__)

    instance._echo = echoflag  # type: ignore

    logger: Union[logging.Logger, InstanceLogger]

    if echoflag in (False, None):
        # if no echo setting or False, return a Logger directly,
        # avoiding overhead of filtering
        logger = logging.getLogger(name)
    else:
        # if a specified echo flag, return an EchoLogger,
        # which checks the flag, overrides normal log
        # levels by calling logger._log()
        logger = InstanceLogger(echoflag, name)

    instance.logger = logger  # type: ignore


class echo_property:
    __doc__ = """\
    When ``True``, enable log output for this element.

    This has the effect of setting the Python logging level for the namespace
    of this element's class and object reference.  A value of boolean ``True``
    indicates that the loglevel ``logging.INFO`` will be set for the logger,
    whereas the string value ``debug`` will set the loglevel to
    ``logging.DEBUG``.
    """

    @overload
    def __get__(
        self, instance: Literal[None], owner: Type[Identified]
    ) -> echo_property: ...

    @overload
    def __get__(
        self, instance: Identified, owner: Type[Identified]
    ) -> _EchoFlagType: ...

    def __get__(
        self, instance: Optional[Identified], owner: Type[Identified]
    ) -> Union[echo_property, _EchoFlagType]:
        if instance is None:
            return self
        else:
            return instance._echo

    def __set__(self, instance: Identified, value: _EchoFlagType) -> None:
        instance_logger(instance, echoflag=value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\_concurrency_py3k.py ===
# util/_concurrency_py3k.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

import asyncio
from contextvars import Context
import sys
import typing
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import Optional
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .langhelpers import memoized_property
from .. import exc
from ..util import py311
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard

_T = TypeVar("_T")

if typing.TYPE_CHECKING:

    class greenlet(Protocol):
        dead: bool
        gr_context: Optional[Context]

        def __init__(self, fn: Callable[..., Any], driver: greenlet): ...

        def throw(self, *arg: Any) -> Any:
            return None

        def switch(self, value: Any) -> Any:
            return None

    def getcurrent() -> greenlet: ...

else:
    from greenlet import getcurrent
    from greenlet import greenlet


# If greenlet.gr_context is present in current version of greenlet,
# it will be set with the current context on creation.
# Refs: https://github.com/python-greenlet/greenlet/pull/198
_has_gr_context = hasattr(getcurrent(), "gr_context")


def is_exit_exception(e: BaseException) -> bool:
    # note asyncio.CancelledError is already BaseException
    # so was an exit exception in any case
    return not isinstance(e, Exception) or isinstance(
        e, (asyncio.TimeoutError, asyncio.CancelledError)
    )


# implementation based on snaury gist at
# https://gist.github.com/snaury/202bf4f22c41ca34e56297bae5f33fef
# Issue for context: https://github.com/python-greenlet/greenlet/issues/173


class _AsyncIoGreenlet(greenlet):
    dead: bool

    __sqlalchemy_greenlet_provider__ = True

    def __init__(self, fn: Callable[..., Any], driver: greenlet):
        greenlet.__init__(self, fn, driver)
        if _has_gr_context:
            self.gr_context = driver.gr_context


_T_co = TypeVar("_T_co", covariant=True)

if TYPE_CHECKING:

    def iscoroutine(
        awaitable: Awaitable[_T_co],
    ) -> TypeGuard[Coroutine[Any, Any, _T_co]]: ...

else:
    iscoroutine = asyncio.iscoroutine


def _safe_cancel_awaitable(awaitable: Awaitable[Any]) -> None:
    # https://docs.python.org/3/reference/datamodel.html#coroutine.close

    if iscoroutine(awaitable):
        awaitable.close()


def in_greenlet() -> bool:
    current = getcurrent()
    return getattr(current, "__sqlalchemy_greenlet_provider__", False)


def await_only(awaitable: Awaitable[_T]) -> _T:
    """Awaits an async function in a sync method.

    The sync method must be inside a :func:`greenlet_spawn` context.
    :func:`await_only` calls cannot be nested.

    :param awaitable: The coroutine to call.

    """
    # this is called in the context greenlet while running fn
    current = getcurrent()
    if not getattr(current, "__sqlalchemy_greenlet_provider__", False):
        _safe_cancel_awaitable(awaitable)

        raise exc.MissingGreenlet(
            "greenlet_spawn has not been called; can't call await_only() "
            "here. Was IO attempted in an unexpected place?"
        )

    # returns the control to the driver greenlet passing it
    # a coroutine to run. Once the awaitable is done, the driver greenlet
    # switches back to this greenlet with the result of awaitable that is
    # then returned to the caller (or raised as error)
    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501


def await_fallback(awaitable: Awaitable[_T]) -> _T:
    """Awaits an async function in a sync method.

    The sync method must be inside a :func:`greenlet_spawn` context.
    :func:`await_fallback` calls cannot be nested.

    :param awaitable: The coroutine to call.

    .. deprecated:: 2.0.24 The ``await_fallback()`` function will be removed
       in SQLAlchemy 2.1.  Use :func:`_util.await_only` instead, running the
       function / program / etc. within a top-level greenlet that is set up
       using :func:`_util.greenlet_spawn`.

    """

    # this is called in the context greenlet while running fn
    current = getcurrent()
    if not getattr(current, "__sqlalchemy_greenlet_provider__", False):
        loop = get_event_loop()
        if loop.is_running():
            _safe_cancel_awaitable(awaitable)

            raise exc.MissingGreenlet(
                "greenlet_spawn has not been called and asyncio event "
                "loop is already running; can't call await_fallback() here. "
                "Was IO attempted in an unexpected place?"
            )
        return loop.run_until_complete(awaitable)

    return current.parent.switch(awaitable)  # type: ignore[no-any-return,attr-defined] # noqa: E501


async def greenlet_spawn(
    fn: Callable[..., _T],
    *args: Any,
    _require_await: bool = False,
    **kwargs: Any,
) -> _T:
    """Runs a sync function ``fn`` in a new greenlet.

    The sync function can then use :func:`await_only` to wait for async
    functions.

    :param fn: The sync callable to call.
    :param \\*args: Positional arguments to pass to the ``fn`` callable.
    :param \\*\\*kwargs: Keyword arguments to pass to the ``fn`` callable.
    """

    result: Any
    context = _AsyncIoGreenlet(fn, getcurrent())
    # runs the function synchronously in gl greenlet. If the execution
    # is interrupted by await_only, context is not dead and result is a
    # coroutine to wait. If the context is dead the function has
    # returned, and its result can be returned.
    switch_occurred = False
    result = context.switch(*args, **kwargs)
    while not context.dead:
        switch_occurred = True
        try:
            # wait for a coroutine from await_only and then return its
            # result back to it.
            value = await result
        except BaseException:
            # this allows an exception to be raised within
            # the moderated greenlet so that it can continue
            # its expected flow.
            result = context.throw(*sys.exc_info())
        else:
            result = context.switch(value)

    if _require_await and not switch_occurred:
        raise exc.AwaitRequired(
            "The current operation required an async execution but none was "
            "detected. This will usually happen when using a non compatible "
            "DBAPI driver. Please ensure that an async DBAPI is used."
        )
    return result  # type: ignore[no-any-return]


class AsyncAdaptedLock:
    @memoized_property
    def mutex(self) -> asyncio.Lock:
        # there should not be a race here for coroutines creating the
        # new lock as we are not using await, so therefore no concurrency
        return asyncio.Lock()

    def __enter__(self) -> bool:
        # await is used to acquire the lock only after the first calling
        # coroutine has created the mutex.
        return await_fallback(self.mutex.acquire())

    def __exit__(self, *arg: Any, **kw: Any) -> None:
        self.mutex.release()


def get_event_loop() -> asyncio.AbstractEventLoop:
    """vendor asyncio.get_event_loop() for python 3.7 and above.

    Python 3.10 deprecates get_event_loop() as a standalone.

    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # avoid "During handling of the above exception, another exception..."
        pass
    return asyncio.get_event_loop_policy().get_event_loop()


if not TYPE_CHECKING and py311:
    _Runner = asyncio.Runner
else:

    class _Runner:
        """Runner implementation for test only"""

        _loop: Union[None, asyncio.AbstractEventLoop, Literal[False]]

        def __init__(self) -> None:
            self._loop = None

        def __enter__(self) -> Self:
            self._lazy_init()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            self.close()

        def close(self) -> None:
            if self._loop:
                try:
                    self._loop.run_until_complete(
                        self._loop.shutdown_asyncgens()
                    )
                finally:
                    self._loop.close()
                    self._loop = False

        def get_loop(self) -> asyncio.AbstractEventLoop:
            """Return embedded event loop."""
            self._lazy_init()
            assert self._loop
            return self._loop

        def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
            self._lazy_init()
            assert self._loop
            return self._loop.run_until_complete(coro)

        def _lazy_init(self) -> None:
            if self._loop is False:
                raise RuntimeError("Runner is closed")
            if self._loop is None:
                self._loop = asyncio.new_event_loop()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\numeric.py ===
from __future__ import annotations

import numbers
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_string_dtype,
    pandas_dtype,
)

from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )


class NumericDtype(BaseMaskedDtype):
    _default_np_dtype: np.dtype
    _checker: Callable[[Any], bool]  # is_foo_dtype

    def __repr__(self) -> str:
        return f"{self.name}Dtype()"

    @cache_readonly
    def is_signed_integer(self) -> bool:
        return self.kind == "i"

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        return self.kind == "u"

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        array_class = self.construct_array_type()

        pyarrow_type = pyarrow.from_numpy_dtype(self.type)
        if not array.type.equals(pyarrow_type) and not pyarrow.types.is_null(
            array.type
        ):
            # test_from_arrow_type_error raise for string, but allow
            #  through itemsize conversion GH#31896
            rt_dtype = pandas_dtype(array.type.to_pandas_dtype())
            if rt_dtype.kind not in "iuf":
                # Could allow "c" or potentially disallow float<->int conversion,
                #  but at the moment we specifically test that uint<->int works
                raise TypeError(
                    f"Expected array of {self} type, got {array.type} instead"
                )

            array = array.cast(pyarrow_type)

        if isinstance(array, pyarrow.ChunkedArray):
            # TODO this "if" can be removed when requiring pyarrow >= 10.0, which fixed
            # combine_chunks for empty arrays https://github.com/apache/arrow/pull/13757
            if array.num_chunks == 0:
                array = pyarrow.array([], type=array.type)
            else:
                array = array.combine_chunks()

        data, mask = pyarrow_array_to_numpy_and_mask(array, dtype=self.numpy_dtype)
        return array_class(data.copy(), ~mask, copy=False)

    @classmethod
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:
        raise AbstractMethodError(cls)

    @classmethod
    def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
        if isinstance(dtype, str) and (dtype.startswith(("Int", "UInt", "Float"))):
            # Avoid DeprecationWarning from NumPy about np.dtype("Int64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not isinstance(dtype, NumericDtype):
            mapping = cls._get_dtype_mapping()
            try:
                dtype = mapping[np.dtype(dtype)]
            except KeyError as err:
                raise ValueError(f"invalid dtype specified {dtype}") from err
        return dtype

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        raise AbstractMethodError(cls)


def _coerce_to_data_and_mask(
    values, dtype, copy: bool, dtype_cls: type[NumericDtype], default_dtype: np.dtype
):
    checker = dtype_cls._checker

    mask = None
    inferred_type = None

    if dtype is None and hasattr(values, "dtype"):
        if checker(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        dtype = dtype_cls._standardize_dtype(dtype)

    cls = dtype_cls.construct_array_type()
    if isinstance(values, cls):
        values, mask = values._data, values._mask
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask, dtype, inferred_type

    original = values
    if not copy:
        values = np.asarray(values)
    else:
        values = np.array(values, copy=copy)
    inferred_type = None
    if values.dtype == object or is_string_dtype(values.dtype):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == "boolean" and dtype is None:
            name = dtype_cls.__name__.strip("_")
            raise TypeError(f"{values.dtype} cannot be converted to {name}")

    elif values.dtype.kind == "b" and checker(dtype):
        if not copy:
            values = np.asarray(values, dtype=default_dtype)
        else:
            values = np.array(values, dtype=default_dtype, copy=copy)

    elif values.dtype.kind not in "iuf":
        name = dtype_cls.__name__.strip("_")
        raise TypeError(f"{values.dtype} cannot be converted to {name}")

    if values.ndim != 1:
        raise TypeError("values must be a 1D list-like")

    if mask is None:
        if values.dtype.kind in "iu":
            # fastpath
            mask = np.zeros(len(values), dtype=np.bool_)
        else:
            mask = libmissing.is_numeric_na(values)
    else:
        assert len(mask) == len(values)

    if mask.ndim != 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = default_dtype
    else:
        dtype = dtype.numpy_dtype

    if is_integer_dtype(dtype) and values.dtype.kind == "f" and len(values) > 0:
        if mask.all():
            values = np.ones(values.shape, dtype=dtype)
        else:
            idx = np.nanargmax(values)
            if int(values[idx]) != original[idx]:
                # We have ints that lost precision during the cast.
                inferred_type = lib.infer_dtype(original, skipna=True)
                if (
                    inferred_type not in ["floating", "mixed-integer-float"]
                    and not mask.any()
                ):
                    values = np.asarray(original, dtype=dtype)
                else:
                    values = np.asarray(original, dtype="object")

    # we copy as need to coerce here
    if mask.any():
        values = values.copy()
        values[mask] = cls._internal_fill_value
    if inferred_type in ("string", "unicode"):
        # casts from str are always safe since they raise
        # a ValueError if the str cannot be parsed into a float
        values = values.astype(dtype, copy=copy)
    else:
        values = dtype_cls._safe_cast(values, dtype, copy=False)

    return values, mask, dtype, inferred_type


class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """

    _dtype_cls: type[NumericDtype]

    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        checker = self._dtype_cls._checker
        if not (isinstance(values, np.ndarray) and checker(values.dtype)):
            descr = (
                "floating"
                if self._dtype_cls.kind == "f"  # type: ignore[comparison-overlap]
                else "integer"
            )
            raise TypeError(
                f"values should be {descr} numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.dtype == np.float16:
            # If we don't raise here, then accessing self.dtype would raise
            raise TypeError("FloatingArray does not support np.float16 dtype.")

        super().__init__(values, mask, copy=copy)

    @cache_readonly
    def dtype(self) -> NumericDtype:
        mapping = self._dtype_cls._get_dtype_mapping()
        return mapping[self._data.dtype]

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        dtype_cls = cls._dtype_cls
        default_dtype = dtype_cls._default_np_dtype
        values, mask, _, _ = _coerce_to_data_and_mask(
            value, dtype, copy, dtype_cls, default_dtype
        )
        return values, mask

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        from pandas.core.tools.numeric import to_numeric

        scalars = to_numeric(strings, errors="raise", dtype_backend="numpy_nullable")
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\expressions.py ===
"""
Expressions
-----------

Offer fast expression evaluation through numexpr

"""
from __future__ import annotations

import operator
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._config import get_option

from pandas.util._exceptions import find_stack_level

from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED

if NUMEXPR_INSTALLED:
    import numexpr as ne

if TYPE_CHECKING:
    from pandas._typing import FuncType

_TEST_MODE: bool | None = None
_TEST_RESULT: list[bool] = []
USE_NUMEXPR = NUMEXPR_INSTALLED
_evaluate: FuncType | None = None
_where: FuncType | None = None

# the set of dtypes that we will allow pass to numexpr
_ALLOWED_DTYPES = {
    "evaluate": {"int64", "int32", "float64", "float32", "bool"},
    "where": {"int64", "float64", "bool"},
}

# the minimum prod shape that we will use numexpr
_MIN_ELEMENTS = 1_000_000


def set_use_numexpr(v: bool = True) -> None:
    # set/unset to use numexpr
    global USE_NUMEXPR
    if NUMEXPR_INSTALLED:
        USE_NUMEXPR = v

    # choose what we are going to do
    global _evaluate, _where

    _evaluate = _evaluate_numexpr if USE_NUMEXPR else _evaluate_standard
    _where = _where_numexpr if USE_NUMEXPR else _where_standard


def set_numexpr_threads(n=None) -> None:
    # if we are using numexpr, set the threads to n
    # otherwise reset
    if NUMEXPR_INSTALLED and USE_NUMEXPR:
        if n is None:
            n = ne.detect_number_of_cores()
        ne.set_num_threads(n)


def _evaluate_standard(op, op_str, a, b):
    """
    Standard evaluation.
    """
    if _TEST_MODE:
        _store_test_result(False)
    return op(a, b)


def _can_use_numexpr(op, op_str, a, b, dtype_check) -> bool:
    """return a boolean if we WILL be using numexpr"""
    if op_str is not None:
        # required min elements (otherwise we are adding overhead)
        if a.size > _MIN_ELEMENTS:
            # check for dtype compatibility
            dtypes: set[str] = set()
            for o in [a, b]:
                # ndarray and Series Case
                if hasattr(o, "dtype"):
                    dtypes |= {o.dtype.name}

            # allowed are a superset
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True

    return False


def _evaluate_numexpr(op, op_str, a, b):
    result = None

    if _can_use_numexpr(op, op_str, a, b, "evaluate"):
        is_reversed = op.__name__.strip("_").startswith("r")
        if is_reversed:
            # we were originally called by a reversed op method
            a, b = b, a

        a_value = a
        b_value = b

        try:
            result = ne.evaluate(
                f"a_value {op_str} b_value",
                local_dict={"a_value": a_value, "b_value": b_value},
                casting="safe",
            )
        except TypeError:
            # numexpr raises eg for array ** array with integers
            # (https://github.com/pydata/numexpr/issues/379)
            pass
        except NotImplementedError:
            if _bool_arith_fallback(op_str, a, b):
                pass
            else:
                raise

        if is_reversed:
            # reverse order to original for fallback
            a, b = b, a

    if _TEST_MODE:
        _store_test_result(result is not None)

    if result is None:
        result = _evaluate_standard(op, op_str, a, b)

    return result


_op_str_mapping = {
    operator.add: "+",
    roperator.radd: "+",
    operator.mul: "*",
    roperator.rmul: "*",
    operator.sub: "-",
    roperator.rsub: "-",
    operator.truediv: "/",
    roperator.rtruediv: "/",
    # floordiv not supported by numexpr 2.x
    operator.floordiv: None,
    roperator.rfloordiv: None,
    # we require Python semantics for mod of negative for backwards compatibility
    # see https://github.com/pydata/numexpr/issues/365
    # so sticking with unaccelerated for now GH#36552
    operator.mod: None,
    roperator.rmod: None,
    operator.pow: "**",
    roperator.rpow: "**",
    operator.eq: "==",
    operator.ne: "!=",
    operator.le: "<=",
    operator.lt: "<",
    operator.ge: ">=",
    operator.gt: ">",
    operator.and_: "&",
    roperator.rand_: "&",
    operator.or_: "|",
    roperator.ror_: "|",
    operator.xor: "^",
    roperator.rxor: "^",
    divmod: None,
    roperator.rdivmod: None,
}


def _where_standard(cond, a, b):
    # Caller is responsible for extracting ndarray if necessary
    return np.where(cond, a, b)


def _where_numexpr(cond, a, b):
    # Caller is responsible for extracting ndarray if necessary
    result = None

    if _can_use_numexpr(None, "where", a, b, "where"):
        result = ne.evaluate(
            "where(cond_value, a_value, b_value)",
            local_dict={"cond_value": cond, "a_value": a, "b_value": b},
            casting="safe",
        )

    if result is None:
        result = _where_standard(cond, a, b)

    return result


# turn myself on
set_use_numexpr(get_option("compute.use_numexpr"))


def _has_bool_dtype(x):
    try:
        return x.dtype == bool
    except AttributeError:
        return isinstance(x, (bool, np.bool_))


_BOOL_OP_UNSUPPORTED = {"+": "|", "*": "&", "-": "^"}


def _bool_arith_fallback(op_str, a, b) -> bool:
    """
    Check if we should fallback to the python `_evaluate_standard` in case
    of an unsupported operation by numexpr, which is the case for some
    boolean ops.
    """
    if _has_bool_dtype(a) and _has_bool_dtype(b):
        if op_str in _BOOL_OP_UNSUPPORTED:
            warnings.warn(
                f"evaluating in Python space because the {repr(op_str)} "
                "operator is not supported by numexpr for the bool dtype, "
                f"use {repr(_BOOL_OP_UNSUPPORTED[op_str])} instead.",
                stacklevel=find_stack_level(),
            )
            return True
    return False


def evaluate(op, a, b, use_numexpr: bool = True):
    """
    Evaluate and return the expression of the op on a and b.

    Parameters
    ----------
    op : the actual operand
    a : left operand
    b : right operand
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    op_str = _op_str_mapping[op]
    if op_str is not None:
        if use_numexpr:
            # error: "None" not callable
            return _evaluate(op, op_str, a, b)  # type: ignore[misc]
    return _evaluate_standard(op, op_str, a, b)


def where(cond, a, b, use_numexpr: bool = True):
    """
    Evaluate the where condition cond on a and b.

    Parameters
    ----------
    cond : np.ndarray[bool]
    a : return if cond is True
    b : return if cond is False
    use_numexpr : bool, default True
        Whether to try to use numexpr.
    """
    assert _where is not None
    return _where(cond, a, b) if use_numexpr else _where_standard(cond, a, b)


def set_test_mode(v: bool = True) -> None:
    """
    Keeps track of whether numexpr was used.

    Stores an additional ``True`` for every successful use of evaluate with
    numexpr since the last ``get_test_result``.
    """
    global _TEST_MODE, _TEST_RESULT
    _TEST_MODE = v
    _TEST_RESULT = []


def _store_test_result(used_numexpr: bool) -> None:
    if used_numexpr:
        _TEST_RESULT.append(used_numexpr)


def get_test_result() -> list[bool]:
    """
    Get test result and reset test_results.
    """
    global _TEST_RESULT
    res = _TEST_RESULT
    _TEST_RESULT = []
    return res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\lra_satask.py ===
from sympy.assumptions.assume import global_assumptions
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.assumptions.ask import Q
from sympy.logic.inference import satisfiable
from sympy.logic.algorithms.lra_theory import UnhandledInput, ALLOWED_PRED
from sympy.matrices.kind import MatrixKind
from sympy.core.kind import NumberKind
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.mul import Mul
from sympy.core.singleton import S


def lra_satask(proposition, assumptions=True, context=global_assumptions):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm
    in conjunction with an Linear Real Arithmetic theory solver.

    Used to handle inequalities. Should eventually be depreciated and combined
    into satask, but infinity handling and other things need to be implemented
    before that can happen.
    """
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)

    cnf = CNF.from_prop(assumptions)
    assumptions = EncodedCNF()
    assumptions.from_cnf(cnf)

    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)

    assumptions.add_from_cnf(context_cnf)

    return check_satisfiability(props, _props, assumptions)

# Some predicates such as Q.prime can't be handled by lra_satask.
# For example, (x > 0) & (x < 1) & Q.prime(x) is unsat but lra_satask would think it was sat.
# WHITE_LIST is a list of predicates that can always be handled.
WHITE_LIST = ALLOWED_PRED | {Q.positive, Q.negative, Q.zero, Q.nonzero, Q.nonpositive, Q.nonnegative,
                                            Q.extended_positive, Q.extended_negative, Q.extended_nonpositive,
                                            Q.extended_negative, Q.extended_nonzero, Q.negative_infinite,
                                            Q.positive_infinite}


def check_satisfiability(prop, _prop, factbase):
    sat_true = factbase.copy()
    sat_false = factbase.copy()
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)

    all_pred, all_exprs = get_all_pred_and_expr_from_enc_cnf(sat_true)

    for pred in all_pred:
        if pred.function not in WHITE_LIST and pred.function != Q.ne:
            raise UnhandledInput(f"LRASolver: {pred} is an unhandled predicate")
    for expr in all_exprs:
        if expr.kind == MatrixKind(NumberKind):
            raise UnhandledInput(f"LRASolver: {expr} is of MatrixKind")
        if expr == S.NaN:
            raise UnhandledInput("LRASolver: nan")

    # convert old assumptions into predicates and add them to sat_true and sat_false
    # also check for unhandled predicates
    for assm in extract_pred_from_old_assum(all_exprs):
        n = len(sat_true.encoding)
        if assm not in sat_true.encoding:
            sat_true.encoding[assm] = n+1
        sat_true.data.append([sat_true.encoding[assm]])

        n = len(sat_false.encoding)
        if assm not in sat_false.encoding:
            sat_false.encoding[assm] = n+1
        sat_false.data.append([sat_false.encoding[assm]])


    sat_true = _preprocess(sat_true)
    sat_false = _preprocess(sat_false)

    can_be_true = satisfiable(sat_true, use_lra_theory=True) is not False
    can_be_false = satisfiable(sat_false, use_lra_theory=True) is not False

    if can_be_true and can_be_false:
        return None

    if can_be_true and not can_be_false:
        return True

    if not can_be_true and can_be_false:
        return False

    if not can_be_true and not can_be_false:
        raise ValueError("Inconsistent assumptions")


def _preprocess(enc_cnf):
    """
    Returns an encoded cnf with only Q.eq, Q.gt, Q.lt,
    Q.ge, and Q.le predicate.

    Converts every unequality into a disjunction of strict
    inequalities. For example, x != 3 would become
    x < 3 OR x > 3.

    Also converts all negated Q.ne predicates into
    equalities.
    """

    # loops through each literal in each clause
    # to construct a new, preprocessed encodedCNF

    enc_cnf = enc_cnf.copy()
    cur_enc = 1
    rev_encoding = {value: key for key, value in enc_cnf.encoding.items()}

    new_encoding = {}
    new_data = []
    for clause in enc_cnf.data:
        new_clause = []
        for lit in clause:
            if lit == 0:
                new_clause.append(lit)
                new_encoding[lit] = False
                continue
            prop = rev_encoding[abs(lit)]
            negated = lit < 0
            sign = (lit > 0) - (lit < 0)

            prop = _pred_to_binrel(prop)

            if not isinstance(prop, AppliedPredicate):
                if prop not in new_encoding:
                    new_encoding[prop] = cur_enc
                    cur_enc += 1
                lit = new_encoding[prop]
                new_clause.append(sign*lit)
                continue


            if negated and prop.function == Q.eq:
                negated = False
                prop = Q.ne(*prop.arguments)

            if prop.function == Q.ne:
                arg1, arg2 = prop.arguments
                if negated:
                    new_prop = Q.eq(arg1, arg2)
                    if new_prop not in new_encoding:
                        new_encoding[new_prop] = cur_enc
                        cur_enc += 1

                    new_enc = new_encoding[new_prop]
                    new_clause.append(new_enc)
                    continue
                else:
                    new_props = (Q.gt(arg1, arg2), Q.lt(arg1, arg2))
                    for new_prop in new_props:
                        if new_prop not in new_encoding:
                            new_encoding[new_prop] = cur_enc
                            cur_enc += 1

                        new_enc = new_encoding[new_prop]
                        new_clause.append(new_enc)
                    continue

            if prop.function == Q.eq and negated:
                assert False

            if prop not in new_encoding:
                new_encoding[prop] = cur_enc
                cur_enc += 1
            new_clause.append(new_encoding[prop]*sign)
        new_data.append(new_clause)

    assert len(new_encoding) >= cur_enc - 1

    enc_cnf = EncodedCNF(new_data, new_encoding)
    return enc_cnf


def _pred_to_binrel(pred):
    if not isinstance(pred, AppliedPredicate):
        return pred

    if pred.function in pred_to_pos_neg_zero:
        f = pred_to_pos_neg_zero[pred.function]
        if f is False:
            return False
        pred = f(pred.arguments[0])

    if pred.function == Q.positive:
        pred = Q.gt(pred.arguments[0], 0)
    elif pred.function == Q.negative:
        pred = Q.lt(pred.arguments[0], 0)
    elif pred.function == Q.zero:
        pred = Q.eq(pred.arguments[0], 0)
    elif pred.function == Q.nonpositive:
        pred = Q.le(pred.arguments[0], 0)
    elif pred.function == Q.nonnegative:
        pred = Q.ge(pred.arguments[0], 0)
    elif pred.function == Q.nonzero:
        pred = Q.ne(pred.arguments[0], 0)

    return pred

pred_to_pos_neg_zero = {
    Q.extended_positive: Q.positive,
    Q.extended_negative: Q.negative,
    Q.extended_nonpositive: Q.nonpositive,
    Q.extended_negative: Q.negative,
    Q.extended_nonzero: Q.nonzero,
    Q.negative_infinite: False,
    Q.positive_infinite: False
}

def get_all_pred_and_expr_from_enc_cnf(enc_cnf):
    all_exprs = set()
    all_pred = set()
    for pred in enc_cnf.encoding.keys():
        if isinstance(pred, AppliedPredicate):
            all_pred.add(pred)
            all_exprs.update(pred.arguments)

    return all_pred, all_exprs

def extract_pred_from_old_assum(all_exprs):
    """
    Returns a list of relevant new assumption predicate
    based on any old assumptions.

    Raises an UnhandledInput exception if any of the assumptions are
    unhandled.

    Ignored predicate:
    - commutative
    - complex
    - algebraic
    - transcendental
    - extended_real
    - real
    - all matrix predicate
    - rational
    - irrational

    Example
    =======
    >>> from sympy.assumptions.lra_satask import extract_pred_from_old_assum
    >>> from sympy import symbols
    >>> x, y = symbols("x y", positive=True)
    >>> extract_pred_from_old_assum([x, y, 2])
    [Q.positive(x), Q.positive(y)]
    """
    ret = []
    for expr in all_exprs:
        if not hasattr(expr, "free_symbols"):
            continue
        if len(expr.free_symbols) == 0:
            continue

        if expr.is_real is not True:
            raise UnhandledInput(f"LRASolver: {expr} must be real")
        # test for I times imaginary variable; such expressions are considered real
        if isinstance(expr, Mul) and any(arg.is_real is not True for arg in expr.args):
            raise UnhandledInput(f"LRASolver: {expr} must be real")

        if expr.is_integer == True and expr.is_zero != True:
            raise UnhandledInput(f"LRASolver: {expr} is an integer")
        if expr.is_integer == False:
            raise UnhandledInput(f"LRASolver: {expr} can't be an integer")
        if expr.is_rational == False:
            raise UnhandledInput(f"LRASolver: {expr} is irational")

        if expr.is_zero:
            ret.append(Q.zero(expr))
        elif expr.is_positive:
            ret.append(Q.positive(expr))
        elif expr.is_negative:
            ret.append(Q.negative(expr))
        elif expr.is_nonzero:
            ret.append(Q.nonzero(expr))
        elif expr.is_nonpositive:
            ret.append(Q.nonpositive(expr))
        elif expr.is_nonnegative:
            ret.append(Q.nonnegative(expr))

    return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\orderings.py ===
"""Definitions of monomial orderings. """

from __future__ import annotations

__all__ = ["lex", "grlex", "grevlex", "ilex", "igrlex", "igrevlex"]

from sympy.core import Symbol
from sympy.utilities.iterables import iterable

class MonomialOrder:
    """Base class for monomial orderings. """

    alias: str | None = None
    is_global: bool | None = None
    is_default = False

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def __str__(self):
        return self.alias

    def __call__(self, monomial):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def __ne__(self, other):
        return not (self == other)

class LexOrder(MonomialOrder):
    """Lexicographic order of monomials. """

    alias = 'lex'
    is_global = True
    is_default = True

    def __call__(self, monomial):
        return monomial

class GradedLexOrder(MonomialOrder):
    """Graded lexicographic order of monomials. """

    alias = 'grlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), monomial)

class ReversedGradedLexOrder(MonomialOrder):
    """Reversed graded lexicographic order of monomials. """

    alias = 'grevlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), tuple(reversed([-m for m in monomial])))

class ProductOrder(MonomialOrder):
    """
    A product order built from other monomial orders.

    Given (not necessarily total) orders O1, O2, ..., On, their product order
    P is defined as M1 > M2 iff there exists i such that O1(M1) = O2(M2),
    ..., Oi(M1) = Oi(M2), O{i+1}(M1) > O{i+1}(M2).

    Product orders are typically built from monomial orders on different sets
    of variables.

    ProductOrder is constructed by passing a list of pairs
    [(O1, L1), (O2, L2), ...] where Oi are MonomialOrders and Li are callables.
    Upon comparison, the Li are passed the total monomial, and should filter
    out the part of the monomial to pass to Oi.

    Examples
    ========

    We can use a lexicographic order on x_1, x_2 and also on
    y_1, y_2, y_3, and their product on {x_i, y_i} as follows:

    >>> from sympy.polys.orderings import lex, grlex, ProductOrder
    >>> P = ProductOrder(
    ...     (lex, lambda m: m[:2]), # lex order on x_1 and x_2 of monomial
    ...     (grlex, lambda m: m[2:]) # grlex on y_1, y_2, y_3
    ... )
    >>> P((2, 1, 1, 0, 0)) > P((1, 10, 0, 2, 0))
    True

    Here the exponent `2` of `x_1` in the first monomial
    (`x_1^2 x_2 y_1`) is bigger than the exponent `1` of `x_1` in the
    second monomial (`x_1 x_2^10 y_2^2`), so the first monomial is greater
    in the product ordering.

    >>> P((2, 1, 1, 0, 0)) < P((2, 1, 0, 2, 0))
    True

    Here the exponents of `x_1` and `x_2` agree, so the grlex order on
    `y_1, y_2, y_3` is used to decide the ordering. In this case the monomial
    `y_2^2` is ordered larger than `y_1`, since for the grlex order the degree
    of the monomial is most important.
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, monomial):
        return tuple(O(lamda(monomial)) for (O, lamda) in self.args)

    def __repr__(self):
        contents = [repr(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    def __str__(self):
        contents = [str(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    def __eq__(self, other):
        if not isinstance(other, ProductOrder):
            return False
        return self.args == other.args

    def __hash__(self):
        return hash((self.__class__, self.args))

    @property
    def is_global(self):
        if all(o.is_global is True for o, _ in self.args):
            return True
        if all(o.is_global is False for o, _ in self.args):
            return False
        return None

class InverseOrder(MonomialOrder):
    """
    The "inverse" of another monomial order.

    If O is any monomial order, we can construct another monomial order iO
    such that `A >_{iO} B` if and only if `B >_O A`. This is useful for
    constructing local orders.

    Note that many algorithms only work with *global* orders.

    For example, in the inverse lexicographic order on a single variable `x`,
    high powers of `x` count as small:

    >>> from sympy.polys.orderings import lex, InverseOrder
    >>> ilex = InverseOrder(lex)
    >>> ilex((5,)) < ilex((0,))
    True
    """

    def __init__(self, O):
        self.O = O

    def __str__(self):
        return "i" + str(self.O)

    def __call__(self, monomial):
        def inv(l):
            if iterable(l):
                return tuple(inv(x) for x in l)
            return -l
        return inv(self.O(monomial))

    @property
    def is_global(self):
        if self.O.is_global is True:
            return False
        if self.O.is_global is False:
            return True
        return None

    def __eq__(self, other):
        return isinstance(other, InverseOrder) and other.O == self.O

    def __hash__(self):
        return hash((self.__class__, self.O))

lex = LexOrder()
grlex = GradedLexOrder()
grevlex = ReversedGradedLexOrder()
ilex = InverseOrder(lex)
igrlex = InverseOrder(grlex)
igrevlex = InverseOrder(grevlex)

_monomial_key = {
    'lex': lex,
    'grlex': grlex,
    'grevlex': grevlex,
    'ilex': ilex,
    'igrlex': igrlex,
    'igrevlex': igrevlex
}

def monomial_key(order=None, gens=None):
    """
    Return a function defining admissible order on monomials.

    The result of a call to :func:`monomial_key` is a function which should
    be used as a key to :func:`sorted` built-in function, to provide order
    in a set of monomials of the same length.

    Currently supported monomial orderings are:

    1. lex       - lexicographic order (default)
    2. grlex     - graded lexicographic order
    3. grevlex   - reversed graded lexicographic order
    4. ilex, igrlex, igrevlex - the corresponding inverse orders

    If the ``order`` input argument is not a string but has ``__call__``
    attribute, then it will pass through with an assumption that the
    callable object defines an admissible order on monomials.

    If the ``gens`` input argument contains a list of generators, the
    resulting key function can be used to sort SymPy ``Expr`` objects.

    """
    if order is None:
        order = lex

    if isinstance(order, Symbol):
        order = str(order)

    if isinstance(order, str):
        try:
            order = _monomial_key[order]
        except KeyError:
            raise ValueError("supported monomial orderings are 'lex', 'grlex' and 'grevlex', got %r" % order)
    if hasattr(order, '__call__'):
        if gens is not None:
            def _order(expr):
                return order(expr.as_poly(*gens).degree_list())
            return _order
        return order
    else:
        raise ValueError("monomial ordering specification must be a string or a callable, got %s" % order)

class _ItemGetter:
    """Helper class to return a subsequence of values."""

    def __init__(self, seq):
        self.seq = tuple(seq)

    def __call__(self, m):
        return tuple(m[idx] for idx in self.seq)

    def __eq__(self, other):
        if not isinstance(other, _ItemGetter):
            return False
        return self.seq == other.seq

def build_product_order(arg, gens):
    """
    Build a monomial order on ``gens``.

    ``arg`` should be a tuple of iterables. The first element of each iterable
    should be a string or monomial order (will be passed to monomial_key),
    the others should be subsets of the generators. This function will build
    the corresponding product order.

    For example, build a product of two grlex orders:

    >>> from sympy.polys.orderings import build_product_order
    >>> from sympy.abc import x, y, z, t

    >>> O = build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
    >>> O((1, 2, 3, 4))
    ((3, (1, 2)), (7, (3, 4)))

    """
    gens2idx = {}
    for i, g in enumerate(gens):
        gens2idx[g] = i
    order = []
    for expr in arg:
        name = expr[0]
        var = expr[1:]

        def makelambda(var):
            return _ItemGetter(gens2idx[g] for g in var)
        order.append((monomial_key(name), makelambda(var)))
    return ProductOrder(*order)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\dyadic.py ===
from __future__ import annotations

from sympy.vector.basisdependent import (BasisDependent, BasisDependentAdd,
                                         BasisDependentMul, BasisDependentZero)
from sympy.core import S, Pow
from sympy.core.expr import AtomicExpr
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
import sympy.vector


class Dyadic(BasisDependent):
    """
    Super class for all Dyadic-classes.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dyadic_tensor
    .. [2] Kane, T., Levinson, D. Dynamics Theory and Applications. 1985
           McGraw-Hill

    """

    _op_priority = 13.0

    _expr_type: type[Dyadic]
    _mul_func: type[Dyadic]
    _add_func: type[Dyadic]
    _zero_func: type[Dyadic]
    _base_func: type[Dyadic]
    zero: DyadicZero

    @property
    def components(self):
        """
        Returns the components of this dyadic in the form of a
        Python dictionary mapping BaseDyadic instances to the
        corresponding measure numbers.

        """
        # The '_components' attribute is defined according to the
        # subclass of Dyadic the instance belongs to.
        return self._components

    def dot(self, other):
        """
        Returns the dot product(also called inner product) of this
        Dyadic, with another Dyadic or Vector.
        If 'other' is a Dyadic, this returns a Dyadic. Else, it returns
        a Vector (unless an error is encountered).

        Parameters
        ==========

        other : Dyadic/Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> D1 = N.i.outer(N.j)
        >>> D2 = N.j.outer(N.j)
        >>> D1.dot(D2)
        (N.i|N.j)
        >>> D1.dot(N.j)
        N.i

        """

        Vector = sympy.vector.Vector
        if isinstance(other, BasisDependentZero):
            return Vector.zero
        elif isinstance(other, Vector):
            outvec = Vector.zero
            for k, v in self.components.items():
                vect_dot = k.args[1].dot(other)
                outvec += vect_dot * v * k.args[0]
            return outvec
        elif isinstance(other, Dyadic):
            outdyad = Dyadic.zero
            for k1, v1 in self.components.items():
                for k2, v2 in other.components.items():
                    vect_dot = k1.args[1].dot(k2.args[0])
                    outer_product = k1.args[0].outer(k2.args[1])
                    outdyad += vect_dot * v1 * v2 * outer_product
            return outdyad
        else:
            raise TypeError("Inner product is not defined for " +
                            str(type(other)) + " and Dyadics.")

    def __and__(self, other):
        return self.dot(other)

    __and__.__doc__ = dot.__doc__

    def cross(self, other):
        """
        Returns the cross product between this Dyadic, and a Vector, as a
        Vector instance.

        Parameters
        ==========

        other : Vector
            The Vector that we are crossing this Dyadic with

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> d = N.i.outer(N.i)
        >>> d.cross(N.j)
        (N.i|N.k)

        """

        Vector = sympy.vector.Vector
        if other == Vector.zero:
            return Dyadic.zero
        elif isinstance(other, Vector):
            outdyad = Dyadic.zero
            for k, v in self.components.items():
                cross_product = k.args[1].cross(other)
                outer = k.args[0].outer(cross_product)
                outdyad += v * outer
            return outdyad
        else:
            raise TypeError(str(type(other)) + " not supported for " +
                            "cross with dyadics")

    def __xor__(self, other):
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__

    def to_matrix(self, system, second_system=None):
        """
        Returns the matrix form of the dyadic with respect to one or two
        coordinate systems.

        Parameters
        ==========

        system : CoordSys3D
            The coordinate system that the rows and columns of the matrix
            correspond to. If a second system is provided, this
            only corresponds to the rows of the matrix.
        second_system : CoordSys3D, optional, default=None
            The coordinate system that the columns of the matrix correspond
            to.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> v = N.i + 2*N.j
        >>> d = v.outer(N.i)
        >>> d.to_matrix(N)
        Matrix([
        [1, 0, 0],
        [2, 0, 0],
        [0, 0, 0]])
        >>> from sympy import Symbol
        >>> q = Symbol('q')
        >>> P = N.orient_new_axis('P', q, N.k)
        >>> d.to_matrix(N, P)
        Matrix([
        [  cos(q),   -sin(q), 0],
        [2*cos(q), -2*sin(q), 0],
        [       0,         0, 0]])

        """

        if second_system is None:
            second_system = system

        return Matrix([i.dot(self).dot(j) for i in system for j in
                       second_system]).reshape(3, 3)

    def _div_helper(one, other):
        """ Helper for division involving dyadics """
        if isinstance(one, Dyadic) and isinstance(other, Dyadic):
            raise TypeError("Cannot divide two dyadics")
        elif isinstance(one, Dyadic):
            return DyadicMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError("Cannot divide by a dyadic")


class BaseDyadic(Dyadic, AtomicExpr):
    """
    Class to denote a base dyadic tensor component.
    """

    def __new__(cls, vector1, vector2):
        Vector = sympy.vector.Vector
        BaseVector = sympy.vector.BaseVector
        VectorZero = sympy.vector.VectorZero
        # Verify arguments
        if not isinstance(vector1, (BaseVector, VectorZero)) or \
                not isinstance(vector2, (BaseVector, VectorZero)):
            raise TypeError("BaseDyadic cannot be composed of non-base " +
                            "vectors")
        # Handle special case of zero vector
        elif vector1 == Vector.zero or vector2 == Vector.zero:
            return Dyadic.zero
        # Initialize instance
        obj = super().__new__(cls, vector1, vector2)
        obj._base_instance = obj
        obj._measure_number = 1
        obj._components = {obj: S.One}
        obj._sys = vector1._sys
        obj._pretty_form = ('(' + vector1._pretty_form + '|' +
                             vector2._pretty_form + ')')
        obj._latex_form = (r'\left(' + vector1._latex_form + r"{\middle|}" +
                           vector2._latex_form + r'\right)')

        return obj

    def _sympystr(self, printer):
        return "({}|{})".format(
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _sympyrepr(self, printer):
        return "BaseDyadic({}, {})".format(
            printer._print(self.args[0]), printer._print(self.args[1]))


class DyadicMul(BasisDependentMul, Dyadic):
    """ Products of scalars and BaseDyadics """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_dyadic(self):
        """ The BaseDyadic involved in the product. """
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this DyadicMul.
        """
        return self._measure_number


class DyadicAdd(BasisDependentAdd, Dyadic):
    """ Class to hold dyadic sums """

    def __new__(cls, *args, **options):
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        items = list(self.components.items())
        items.sort(key=lambda x: x[0].__str__())
        return " + ".join(printer._print(k * v) for k, v in items)


class DyadicZero(BasisDependentZero, Dyadic):
    """
    Class to denote a zero dyadic
    """

    _op_priority = 13.1
    _pretty_form = '(0|0)'
    _latex_form = r'(\mathbf{\hat{0}}|\mathbf{\hat{0}})'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj


Dyadic._expr_type = Dyadic
Dyadic._mul_func = DyadicMul
Dyadic._add_func = DyadicAdd
Dyadic._zero_func = DyadicZero
Dyadic._base_func = BaseDyadic
Dyadic.zero = DyadicZero()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\excel\_xlsxwriter.py ===
from __future__ import annotations

import json
from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        StorageOptions,
        WriteExcelBuffer,
    )


class _XlsxStyler:
    # Map from openpyxl-oriented styles to flatter xlsxwriter representation
    # Ordering necessary for both determinism and because some are keyed by
    # prefixes of others.
    STYLE_MAPPING: dict[str, list[tuple[tuple[str, ...], str]]] = {
        "font": [
            (("name",), "font_name"),
            (("sz",), "font_size"),
            (("size",), "font_size"),
            (("color", "rgb"), "font_color"),
            (("color",), "font_color"),
            (("b",), "bold"),
            (("bold",), "bold"),
            (("i",), "italic"),
            (("italic",), "italic"),
            (("u",), "underline"),
            (("underline",), "underline"),
            (("strike",), "font_strikeout"),
            (("vertAlign",), "font_script"),
            (("vertalign",), "font_script"),
        ],
        "number_format": [(("format_code",), "num_format"), ((), "num_format")],
        "protection": [(("locked",), "locked"), (("hidden",), "hidden")],
        "alignment": [
            (("horizontal",), "align"),
            (("vertical",), "valign"),
            (("text_rotation",), "rotation"),
            (("wrap_text",), "text_wrap"),
            (("indent",), "indent"),
            (("shrink_to_fit",), "shrink"),
        ],
        "fill": [
            (("patternType",), "pattern"),
            (("patterntype",), "pattern"),
            (("fill_type",), "pattern"),
            (("start_color", "rgb"), "fg_color"),
            (("fgColor", "rgb"), "fg_color"),
            (("fgcolor", "rgb"), "fg_color"),
            (("start_color",), "fg_color"),
            (("fgColor",), "fg_color"),
            (("fgcolor",), "fg_color"),
            (("end_color", "rgb"), "bg_color"),
            (("bgColor", "rgb"), "bg_color"),
            (("bgcolor", "rgb"), "bg_color"),
            (("end_color",), "bg_color"),
            (("bgColor",), "bg_color"),
            (("bgcolor",), "bg_color"),
        ],
        "border": [
            (("color", "rgb"), "border_color"),
            (("color",), "border_color"),
            (("style",), "border"),
            (("top", "color", "rgb"), "top_color"),
            (("top", "color"), "top_color"),
            (("top", "style"), "top"),
            (("top",), "top"),
            (("right", "color", "rgb"), "right_color"),
            (("right", "color"), "right_color"),
            (("right", "style"), "right"),
            (("right",), "right"),
            (("bottom", "color", "rgb"), "bottom_color"),
            (("bottom", "color"), "bottom_color"),
            (("bottom", "style"), "bottom"),
            (("bottom",), "bottom"),
            (("left", "color", "rgb"), "left_color"),
            (("left", "color"), "left_color"),
            (("left", "style"), "left"),
            (("left",), "left"),
        ],
    }

    @classmethod
    def convert(cls, style_dict, num_format_str=None):
        """
        converts a style_dict to an xlsxwriter format dict

        Parameters
        ----------
        style_dict : style dictionary to convert
        num_format_str : optional number format string
        """
        # Create a XlsxWriter format object.
        props = {}

        if num_format_str is not None:
            props["num_format"] = num_format_str

        if style_dict is None:
            return props

        if "borders" in style_dict:
            style_dict = style_dict.copy()
            style_dict["border"] = style_dict.pop("borders")

        for style_group_key, style_group in style_dict.items():
            for src, dst in cls.STYLE_MAPPING.get(style_group_key, []):
                # src is a sequence of keys into a nested dict
                # dst is a flat key
                if dst in props:
                    continue
                v = style_group
                for k in src:
                    try:
                        v = v[k]
                    except (KeyError, TypeError):
                        break
                else:
                    props[dst] = v

        if isinstance(props.get("pattern"), str):
            # TODO: support other fill patterns
            props["pattern"] = 0 if props["pattern"] == "none" else 1

        for k in ["border", "top", "right", "bottom", "left"]:
            if isinstance(props.get(k), str):
                try:
                    props[k] = [
                        "none",
                        "thin",
                        "medium",
                        "dashed",
                        "dotted",
                        "thick",
                        "double",
                        "hair",
                        "mediumDashed",
                        "dashDot",
                        "mediumDashDot",
                        "dashDotDot",
                        "mediumDashDotDot",
                        "slantDashDot",
                    ].index(props[k])
                except ValueError:
                    props[k] = 2

        if isinstance(props.get("font_script"), str):
            props["font_script"] = ["baseline", "superscript", "subscript"].index(
                props["font_script"]
            )

        if isinstance(props.get("underline"), str):
            props["underline"] = {
                "none": 0,
                "single": 1,
                "double": 2,
                "singleAccounting": 33,
                "doubleAccounting": 34,
            }[props["underline"]]

        # GH 30107 - xlsxwriter uses different name
        if props.get("valign") == "center":
            props["valign"] = "vcenter"

        return props


class XlsxWriter(ExcelWriter):
    _engine = "xlsxwriter"
    _supported_extensions = (".xlsx",)

    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: str | None = None,
        date_format: str | None = None,
        datetime_format: str | None = None,
        mode: str = "w",
        storage_options: StorageOptions | None = None,
        if_sheet_exists: ExcelWriterIfSheetExists | None = None,
        engine_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # Use the xlsxwriter module as the Excel writer.
        from xlsxwriter import Workbook

        engine_kwargs = combine_kwargs(engine_kwargs, kwargs)

        if mode == "a":
            raise ValueError("Append mode is not supported with xlsxwriter!")

        super().__init__(
            path,
            engine=engine,
            date_format=date_format,
            datetime_format=datetime_format,
            mode=mode,
            storage_options=storage_options,
            if_sheet_exists=if_sheet_exists,
            engine_kwargs=engine_kwargs,
        )

        try:
            self._book = Workbook(self._handles.handle, **engine_kwargs)
        except TypeError:
            self._handles.handle.close()
            raise

    @property
    def book(self):
        """
        Book instance of class xlsxwriter.Workbook.

        This attribute can be used to access engine-specific features.
        """
        return self._book

    @property
    def sheets(self) -> dict[str, Any]:
        result = self.book.sheetnames
        return result

    def _save(self) -> None:
        """
        Save workbook to disk.
        """
        self.book.close()

    def _write_cells(
        self,
        cells,
        sheet_name: str | None = None,
        startrow: int = 0,
        startcol: int = 0,
        freeze_panes: tuple[int, int] | None = None,
    ) -> None:
        # Write the frame cells using xlsxwriter.
        sheet_name = self._get_sheet_name(sheet_name)

        wks = self.book.get_worksheet_by_name(sheet_name)
        if wks is None:
            wks = self.book.add_worksheet(sheet_name)

        style_dict = {"null": None}

        if validate_freeze_panes(freeze_panes):
            wks.freeze_panes(*(freeze_panes))

        for cell in cells:
            val, fmt = self._value_with_fmt(cell.val)

            stylekey = json.dumps(cell.style)
            if fmt:
                stylekey += fmt

            if stylekey in style_dict:
                style = style_dict[stylekey]
            else:
                style = self.book.add_format(_XlsxStyler.convert(cell.style, fmt))
                style_dict[stylekey] = style

            if cell.mergestart is not None and cell.mergeend is not None:
                wks.merge_range(
                    startrow + cell.row,
                    startcol + cell.col,
                    startrow + cell.mergestart,
                    startcol + cell.mergeend,
                    val,
                    style,
                )
            else:
                wks.write(startrow + cell.row, startcol + cell.col, val, style)