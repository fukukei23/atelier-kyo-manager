
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\json\_json.py ===
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from collections import abc
from io import StringIO
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.json import (
    ujson_dumps,
    ujson_loads,
)
from pandas._libs.tslibs import iNaT
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    ensure_str,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import PeriodDtype

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    isna,
    notna,
    to_datetime,
)
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs

from pandas.io._util import arrow_table_to_pandas
from pandas.io.common import (
    IOHandles,
    dedup_names,
    extension_to_compression,
    file_exists,
    get_handle,
    is_fsspec_url,
    is_potential_multi_index,
    is_url,
    stringify_path,
)
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import (
    build_table_schema,
    parse_table_schema,
)
from pandas.io.parsers.readers import validate_integer

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Mapping,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        JSONEngine,
        JSONSerializable,
        ReadBuffer,
        Self,
        StorageOptions,
        WriteBuffer,
    )

    from pandas.core.generic import NDFrame

FrameSeriesStrT = TypeVar("FrameSeriesStrT", bound=Literal["frame", "series"])


# interface to/from
@overload
def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> None:
    ...


@overload
def to_json(
    path_or_buf: None,
    obj: NDFrame,
    orient: str | None = ...,
    date_format: str = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool | None = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
    mode: Literal["a", "w"] = ...,
) -> str:
    ...


def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] | None,
    obj: NDFrame,
    orient: str | None = None,
    date_format: str = "epoch",
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: str = "ms",
    default_handler: Callable[[Any], JSONSerializable] | None = None,
    lines: bool = False,
    compression: CompressionOptions = "infer",
    index: bool | None = None,
    indent: int = 0,
    storage_options: StorageOptions | None = None,
    mode: Literal["a", "w"] = "w",
) -> str | None:
    if orient in ["records", "values"] and index is True:
        raise ValueError(
            "'index=True' is only valid when 'orient' is 'split', 'table', "
            "'index', or 'columns'."
        )
    elif orient in ["index", "columns"] and index is False:
        raise ValueError(
            "'index=False' is only valid when 'orient' is 'split', 'table', "
            "'records', or 'values'."
        )
    elif index is None:
        # will be ignored for orient='records' and 'values'
        index = True

    if lines and orient != "records":
        raise ValueError("'lines' keyword only valid when 'orient' is records")

    if mode not in ["a", "w"]:
        msg = (
            f"mode={mode} is not a valid option."
            "Only 'w' and 'a' are currently supported."
        )
        raise ValueError(msg)

    if mode == "a" and (not lines or orient != "records"):
        msg = (
            "mode='a' (append) is only supported when "
            "lines is True and orient is 'records'"
        )
        raise ValueError(msg)

    if orient == "table" and isinstance(obj, Series):
        obj = obj.to_frame(name=obj.name or "values")

    writer: type[Writer]
    if orient == "table" and isinstance(obj, DataFrame):
        writer = JSONTableWriter
    elif isinstance(obj, Series):
        writer = SeriesWriter
    elif isinstance(obj, DataFrame):
        writer = FrameWriter
    else:
        raise NotImplementedError("'obj' should be a Series or a DataFrame")

    s = writer(
        obj,
        orient=orient,
        date_format=date_format,
        double_precision=double_precision,
        ensure_ascii=force_ascii,
        date_unit=date_unit,
        default_handler=default_handler,
        index=index,
        indent=indent,
    ).write()

    if lines:
        s = convert_to_line_delimits(s)

    if path_or_buf is not None:
        # apply compression and byte/text conversion
        with get_handle(
            path_or_buf, mode, compression=compression, storage_options=storage_options
        ) as handles:
            handles.handle.write(s)
    else:
        return s
    return None


class Writer(ABC):
    _default_orient: str

    def __init__(
        self,
        obj: NDFrame,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ) -> None:
        self.obj = obj

        if orient is None:
            orient = self._default_orient

        self.orient = orient
        self.date_format = date_format
        self.double_precision = double_precision
        self.ensure_ascii = ensure_ascii
        self.date_unit = date_unit
        self.default_handler = default_handler
        self.index = index
        self.indent = indent

        self.is_copy = None
        self._format_axes()

    def _format_axes(self) -> None:
        raise AbstractMethodError(self)

    def write(self) -> str:
        iso_dates = self.date_format == "iso"
        return ujson_dumps(
            self.obj_to_write,
            orient=self.orient,
            double_precision=self.double_precision,
            ensure_ascii=self.ensure_ascii,
            date_unit=self.date_unit,
            iso_dates=iso_dates,
            default_handler=self.default_handler,
            indent=self.indent,
        )

    @property
    @abstractmethod
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        """Object to write in JSON format."""


class SeriesWriter(Writer):
    _default_orient = "index"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        if not self.index and self.orient == "split":
            return {"name": self.obj.name, "data": self.obj.values}
        else:
            return self.obj

    def _format_axes(self) -> None:
        if not self.obj.index.is_unique and self.orient == "index":
            raise ValueError(f"Series index must be unique for orient='{self.orient}'")


class FrameWriter(Writer):
    _default_orient = "columns"

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        if not self.index and self.orient == "split":
            obj_to_write = self.obj.to_dict(orient="split")
            del obj_to_write["index"]
        else:
            obj_to_write = self.obj
        return obj_to_write

    def _format_axes(self) -> None:
        """
        Try to format axes if they are datelike.
        """
        if not self.obj.index.is_unique and self.orient in ("index", "columns"):
            raise ValueError(
                f"DataFrame index must be unique for orient='{self.orient}'."
            )
        if not self.obj.columns.is_unique and self.orient in (
            "index",
            "columns",
            "records",
        ):
            raise ValueError(
                f"DataFrame columns must be unique for orient='{self.orient}'."
            )


class JSONTableWriter(FrameWriter):
    _default_orient = "records"

    def __init__(
        self,
        obj,
        orient: str | None,
        date_format: str,
        double_precision: int,
        ensure_ascii: bool,
        date_unit: str,
        index: bool,
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        indent: int = 0,
    ) -> None:
        """
        Adds a `schema` attribute with the Table Schema, resets
        the index (can't do in caller, because the schema inference needs
        to know what the index is, forces orient to records, and forces
        date_format to 'iso'.
        """
        super().__init__(
            obj,
            orient,
            date_format,
            double_precision,
            ensure_ascii,
            date_unit,
            index,
            default_handler=default_handler,
            indent=indent,
        )

        if date_format != "iso":
            msg = (
                "Trying to write with `orient='table'` and "
                f"`date_format='{date_format}'`. Table Schema requires dates "
                "to be formatted with `date_format='iso'`"
            )
            raise ValueError(msg)

        self.schema = build_table_schema(obj, index=self.index)

        # NotImplemented on a column MultiIndex
        if obj.ndim == 2 and isinstance(obj.columns, MultiIndex):
            raise NotImplementedError(
                "orient='table' is not supported for MultiIndex columns"
            )

        # TODO: Do this timedelta properly in objToJSON.c See GH #15137
        if (
            (obj.ndim == 1)
            and (obj.name in set(obj.index.names))
            or len(obj.columns.intersection(obj.index.names))
        ):
            msg = "Overlapping names between the index and columns"
            raise ValueError(msg)

        obj = obj.copy()
        timedeltas = obj.select_dtypes(include=["timedelta"]).columns
        if len(timedeltas):
            obj[timedeltas] = obj[timedeltas].map(lambda x: x.isoformat())
        # Convert PeriodIndex to datetimes before serializing
        if isinstance(obj.index.dtype, PeriodDtype):
            obj.index = obj.index.to_timestamp()

        # exclude index from obj if index=False
        if not self.index:
            self.obj = obj.reset_index(drop=True)
        else:
            self.obj = obj.reset_index(drop=False)
        self.date_format = "iso"
        self.orient = "records"
        self.index = index

    @property
    def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
        return {"schema": self.schema, "data": self.obj}


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> JsonReader[Literal["frame"]]:
    ...


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> JsonReader[Literal["series"]]:
    ...


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> Series:
    ...


@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    engine: JSONEngine = ...,
) -> DataFrame:
    ...


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "path_or_buf",
)
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = None,
    typ: Literal["frame", "series"] = "frame",
    dtype: DtypeArg | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    lines: bool = False,
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    engine: JSONEngine = "ujson",
) -> DataFrame | Series | JsonReader:
    """
    Convert a JSON string to pandas object.

    Parameters
    ----------
    path_or_buf : a valid JSON str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.json``.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        By file-like object, we refer to objects with a ``read()`` method,
        such as a file handle (e.g. via builtin ``open`` function)
        or ``StringIO``.

        .. deprecated:: 2.1.0
            Passing json literal strings is deprecated.

    orient : str, optional
        Indication of expected JSON string format.
        Compatible JSON strings can be produced by ``to_json()`` with a
        corresponding orient value.
        The set of possible orients is:

        - ``'split'`` : dict like
          ``{{index -> [index], columns -> [columns], data -> [values]}}``
        - ``'records'`` : list like
          ``[{{column -> value}}, ... , {{column -> value}}]``
        - ``'index'`` : dict like ``{{index -> {{column -> value}}}}``
        - ``'columns'`` : dict like ``{{column -> {{index -> value}}}}``
        - ``'values'`` : just the values array
        - ``'table'`` : dict like ``{{'schema': {{schema}}, 'data': {{data}}}}``

        The allowed and default values depend on the value
        of the `typ` parameter.

        * when ``typ == 'series'``,

          - allowed orients are ``{{'split','records','index'}}``
          - default is ``'index'``
          - The Series index must be unique for orient ``'index'``.

        * when ``typ == 'frame'``,

          - allowed orients are ``{{'split','records','index',
            'columns','values', 'table'}}``
          - default is ``'columns'``
          - The DataFrame index must be unique for orients ``'index'`` and
            ``'columns'``.
          - The DataFrame columns must be unique for orients ``'index'``,
            ``'columns'``, and ``'records'``.

    typ : {{'frame', 'series'}}, default 'frame'
        The type of object to recover.

    dtype : bool or dict, default None
        If True, infer dtypes; if a dict of column to dtype, then use those;
        if False, then don't infer dtypes at all, applies only to the data.

        For all ``orient`` values except ``'table'``, default is True.

    convert_axes : bool, default None
        Try to convert the axes to the proper dtypes.

        For all ``orient`` values except ``'table'``, default is True.

    convert_dates : bool or list of str, default True
        If True then default datelike columns may be converted (depending on
        keep_default_dates).
        If False, no dates will be converted.
        If a list of column names, then those columns will be converted and
        default datelike columns may also be converted (depending on
        keep_default_dates).

    keep_default_dates : bool, default True
        If parsing dates (convert_dates is not False), then try to parse the
        default datelike columns.
        A column label is datelike if

        * it ends with ``'_at'``,

        * it ends with ``'_time'``,

        * it begins with ``'timestamp'``,

        * it is ``'modified'``, or

        * it is ``'date'``.

    precise_float : bool, default False
        Set to enable usage of higher precision (strtod) function when
        decoding string to double values. Default (False) is to use fast but
        less precise builtin functionality.

    date_unit : str, default None
        The timestamp unit to detect if converting dates. The default behaviour
        is to try and detect the correct precision, but if this is not desired
        then pass one of 's', 'ms', 'us' or 'ns' to force parsing only seconds,
        milliseconds, microseconds or nanoseconds respectively.

    encoding : str, default is 'utf-8'
        The encoding to use to decode py3 bytes.

    encoding_errors : str, optional, default "strict"
        How encoding errors are treated. `List of possible values
        <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

        .. versionadded:: 1.3.0

    lines : bool, default False
        Read the file as a json object per line.

    chunksize : int, optional
        Return JsonReader object for iteration.
        See the `line-delimited json docs
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json>`_
        for more information on ``chunksize``.
        This can only be passed if `lines=True`.
        If this is None, the file will be read into memory all at once.
    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    nrows : int, optional
        The number of lines from the line-delimited jsonfile that has to be read.
        This can only be passed if `lines=True`.
        If this is None, all the rows will be returned.

    {storage_options}

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    engine : {{"ujson", "pyarrow"}}, default "ujson"
        Parser engine to use. The ``"pyarrow"`` engine is only available when
        ``lines=True``.

        .. versionadded:: 2.0

    Returns
    -------
    Series, DataFrame, or pandas.api.typing.JsonReader
        A JsonReader is returned when ``chunksize`` is not ``0`` or ``None``.
        Otherwise, the type returned depends on the value of ``typ``.

    See Also
    --------
    DataFrame.to_json : Convert a DataFrame to a JSON string.
    Series.to_json : Convert a Series to a JSON string.
    json_normalize : Normalize semi-structured JSON data into a flat table.

    Notes
    -----
    Specific to ``orient='table'``, if a :class:`DataFrame` with a literal
    :class:`Index` name of `index` gets written with :func:`to_json`, the
    subsequent read operation will incorrectly set the :class:`Index` name to
    ``None``. This is because `index` is also used by :func:`DataFrame.to_json`
    to denote a missing :class:`Index` name, and the subsequent
    :func:`read_json` operation cannot distinguish between the two. The same
    limitation is encountered with a :class:`MultiIndex` and any names
    beginning with ``'level_'``.

    Examples
    --------
    >>> from io import StringIO
    >>> df = pd.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   index=['row 1', 'row 2'],
    ...                   columns=['col 1', 'col 2'])

    Encoding/decoding a Dataframe using ``'split'`` formatted JSON:

    >>> df.to_json(orient='split')
        '\
{{\
"columns":["col 1","col 2"],\
"index":["row 1","row 2"],\
"data":[["a","b"],["c","d"]]\
}}\
'
    >>> pd.read_json(StringIO(_), orient='split')
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'index'`` formatted JSON:

    >>> df.to_json(orient='index')
    '{{"row 1":{{"col 1":"a","col 2":"b"}},"row 2":{{"col 1":"c","col 2":"d"}}}}'

    >>> pd.read_json(StringIO(_), orient='index')
          col 1 col 2
    row 1     a     b
    row 2     c     d

    Encoding/decoding a Dataframe using ``'records'`` formatted JSON.
    Note that index labels are not preserved with this encoding.

    >>> df.to_json(orient='records')
    '[{{"col 1":"a","col 2":"b"}},{{"col 1":"c","col 2":"d"}}]'
    >>> pd.read_json(StringIO(_), orient='records')
      col 1 col 2
    0     a     b
    1     c     d

    Encoding with Table Schema

    >>> df.to_json(orient='table')
        '\
{{"schema":{{"fields":[\
{{"name":"index","type":"string"}},\
{{"name":"col 1","type":"string"}},\
{{"name":"col 2","type":"string"}}],\
"primaryKey":["index"],\
"pandas_version":"1.4.0"}},\
"data":[\
{{"index":"row 1","col 1":"a","col 2":"b"}},\
{{"index":"row 2","col 1":"c","col 2":"d"}}]\
}}\
'

    The following example uses ``dtype_backend="numpy_nullable"``

    >>> data = '''{{"index": {{"0": 0, "1": 1}},
    ...        "a": {{"0": 1, "1": null}},
    ...        "b": {{"0": 2.5, "1": 4.5}},
    ...        "c": {{"0": true, "1": false}},
    ...        "d": {{"0": "a", "1": "b"}},
    ...        "e": {{"0": 1577.2, "1": 1577.1}}}}'''
    >>> pd.read_json(StringIO(data), dtype_backend="numpy_nullable")
       index     a    b      c  d       e
    0      0     1  2.5   True  a  1577.2
    1      1  <NA>  4.5  False  b  1577.1
    """
    if orient == "table" and dtype:
        raise ValueError("cannot pass both dtype and orient='table'")
    if orient == "table" and convert_axes:
        raise ValueError("cannot pass both convert_axes and orient='table'")

    check_dtype_backend(dtype_backend)

    if dtype is None and orient != "table":
        # error: Incompatible types in assignment (expression has type "bool", variable
        # has type "Union[ExtensionDtype, str, dtype[Any], Type[str], Type[float],
        # Type[int], Type[complex], Type[bool], Type[object], Dict[Hashable,
        # Union[ExtensionDtype, Union[str, dtype[Any]], Type[str], Type[float],
        # Type[int], Type[complex], Type[bool], Type[object]]], None]")
        dtype = True  # type: ignore[assignment]
    if convert_axes is None and orient != "table":
        convert_axes = True

    json_reader = JsonReader(
        path_or_buf,
        orient=orient,
        typ=typ,
        dtype=dtype,
        convert_axes=convert_axes,
        convert_dates=convert_dates,
        keep_default_dates=keep_default_dates,
        precise_float=precise_float,
        date_unit=date_unit,
        encoding=encoding,
        lines=lines,
        chunksize=chunksize,
        compression=compression,
        nrows=nrows,
        storage_options=storage_options,
        encoding_errors=encoding_errors,
        dtype_backend=dtype_backend,
        engine=engine,
    )

    if chunksize:
        return json_reader
    else:
        return json_reader.read()


class JsonReader(abc.Iterator, Generic[FrameSeriesStrT]):
    """
    JsonReader provides an interface for reading in a JSON file.

    If initialized with ``lines=True`` and ``chunksize``, can be iterated over
    ``chunksize`` lines at a time. Otherwise, calling ``read`` reads in the
    whole document.
    """

    def __init__(
        self,
        filepath_or_buffer,
        orient,
        typ: FrameSeriesStrT,
        dtype,
        convert_axes: bool | None,
        convert_dates,
        keep_default_dates: bool,
        precise_float: bool,
        date_unit,
        encoding,
        lines: bool,
        chunksize: int | None,
        compression: CompressionOptions,
        nrows: int | None,
        storage_options: StorageOptions | None = None,
        encoding_errors: str | None = "strict",
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        engine: JSONEngine = "ujson",
    ) -> None:
        self.orient = orient
        self.typ = typ
        self.dtype = dtype
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.keep_default_dates = keep_default_dates
        self.precise_float = precise_float
        self.date_unit = date_unit
        self.encoding = encoding
        self.engine = engine
        self.compression = compression
        self.storage_options = storage_options
        self.lines = lines
        self.chunksize = chunksize
        self.nrows_seen = 0
        self.nrows = nrows
        self.encoding_errors = encoding_errors
        self.handles: IOHandles[str] | None = None
        self.dtype_backend = dtype_backend

        if self.engine not in {"pyarrow", "ujson"}:
            raise ValueError(
                f"The engine type {self.engine} is currently not supported."
            )
        if self.chunksize is not None:
            self.chunksize = validate_integer("chunksize", self.chunksize, 1)
            if not self.lines:
                raise ValueError("chunksize can only be passed if lines=True")
            if self.engine == "pyarrow":
                raise ValueError(
                    "currently pyarrow engine doesn't support chunksize parameter"
                )
        if self.nrows is not None:
            self.nrows = validate_integer("nrows", self.nrows, 0)
            if not self.lines:
                raise ValueError("nrows can only be passed if lines=True")
        if (
            isinstance(filepath_or_buffer, str)
            and not self.lines
            and "\n" in filepath_or_buffer
        ):
            warnings.warn(
                "Passing literal json to 'read_json' is deprecated and "
                "will be removed in a future version. To read from a "
                "literal string, wrap it in a 'StringIO' object.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if self.engine == "pyarrow":
            if not self.lines:
                raise ValueError(
                    "currently pyarrow engine only supports "
                    "the line-delimited JSON format"
                )
            self.data = filepath_or_buffer
        elif self.engine == "ujson":
            data = self._get_data_from_filepath(filepath_or_buffer)
            self.data = self._preprocess_data(data)

    def _preprocess_data(self, data):
        """
        At this point, the data either has a `read` attribute (e.g. a file
        object or a StringIO) or is a string that is a JSON document.

        If self.chunksize, we prepare the data for the `__next__` method.
        Otherwise, we read it into memory for the `read` method.
        """
        if hasattr(data, "read") and not (self.chunksize or self.nrows):
            with self:
                data = data.read()
        if not hasattr(data, "read") and (self.chunksize or self.nrows):
            data = StringIO(data)

        return data

    def _get_data_from_filepath(self, filepath_or_buffer):
        """
        The function read_json accepts three input types:
            1. filepath (string-like)
            2. file-like object (e.g. open file object, StringIO)
            3. JSON string

        This method turns (1) into (2) to simplify the rest of the processing.
        It returns input types (2) and (3) unchanged.

        It raises FileNotFoundError if the input is a string ending in
        one of .json, .json.gz, .json.bz2, etc. but no such file exists.
        """
        # if it is a string but the file does not exist, it might be a JSON string
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        if (
            not isinstance(filepath_or_buffer, str)
            or is_url(filepath_or_buffer)
            or is_fsspec_url(filepath_or_buffer)
            or file_exists(filepath_or_buffer)
        ):
            self.handles = get_handle(
                filepath_or_buffer,
                "r",
                encoding=self.encoding,
                compression=self.compression,
                storage_options=self.storage_options,
                errors=self.encoding_errors,
            )
            filepath_or_buffer = self.handles.handle
        elif (
            isinstance(filepath_or_buffer, str)
            and filepath_or_buffer.lower().endswith(
                (".json",) + tuple(f".json{c}" for c in extension_to_compression)
            )
            and not file_exists(filepath_or_buffer)
        ):
            raise FileNotFoundError(f"File {filepath_or_buffer} does not exist")
        else:
            warnings.warn(
                "Passing literal json to 'read_json' is deprecated and "
                "will be removed in a future version. To read from a "
                "literal string, wrap it in a 'StringIO' object.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return filepath_or_buffer

    def _combine_lines(self, lines) -> str:
        """
        Combines a list of JSON objects into one JSON object.
        """
        return (
            f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'
        )

    @overload
    def read(self: JsonReader[Literal["frame"]]) -> DataFrame:
        ...

    @overload
    def read(self: JsonReader[Literal["series"]]) -> Series:
        ...

    @overload
    def read(self: JsonReader[Literal["frame", "series"]]) -> DataFrame | Series:
        ...

    def read(self) -> DataFrame | Series:
        """
        Read the whole JSON input into a pandas object.
        """
        obj: DataFrame | Series
        with self:
            if self.engine == "pyarrow":
                pyarrow_json = import_optional_dependency("pyarrow.json")
                pa_table = pyarrow_json.read_json(self.data)
                return arrow_table_to_pandas(pa_table, dtype_backend=self.dtype_backend)
            elif self.engine == "ujson":
                if self.lines:
                    if self.chunksize:
                        obj = concat(self)
                    elif self.nrows:
                        lines = list(islice(self.data, self.nrows))
                        lines_json = self._combine_lines(lines)
                        obj = self._get_object_parser(lines_json)
                    else:
                        data = ensure_str(self.data)
                        data_lines = data.split("\n")
                        obj = self._get_object_parser(self._combine_lines(data_lines))
                else:
                    obj = self._get_object_parser(self.data)
                if self.dtype_backend is not lib.no_default:
                    return obj.convert_dtypes(
                        infer_objects=False, dtype_backend=self.dtype_backend
                    )
                else:
                    return obj

    def _get_object_parser(self, json) -> DataFrame | Series:
        """
        Parses a json document into a pandas object.
        """
        typ = self.typ
        dtype = self.dtype
        kwargs = {
            "orient": self.orient,
            "dtype": self.dtype,
            "convert_axes": self.convert_axes,
            "convert_dates": self.convert_dates,
            "keep_default_dates": self.keep_default_dates,
            "precise_float": self.precise_float,
            "date_unit": self.date_unit,
            "dtype_backend": self.dtype_backend,
        }
        obj = None
        if typ == "frame":
            obj = FrameParser(json, **kwargs).parse()

        if typ == "series" or obj is None:
            if not isinstance(dtype, bool):
                kwargs["dtype"] = dtype
            obj = SeriesParser(json, **kwargs).parse()

        return obj

    def close(self) -> None:
        """
        If we opened a stream earlier, in _get_data_from_filepath, we should
        close it.

        If an open stream or file was passed, we leave it open.
        """
        if self.handles is not None:
            self.handles.close()

    def __iter__(self) -> Self:
        return self

    @overload
    def __next__(self: JsonReader[Literal["frame"]]) -> DataFrame:
        ...

    @overload
    def __next__(self: JsonReader[Literal["series"]]) -> Series:
        ...

    @overload
    def __next__(self: JsonReader[Literal["frame", "series"]]) -> DataFrame | Series:
        ...

    def __next__(self) -> DataFrame | Series:
        if self.nrows and self.nrows_seen >= self.nrows:
            self.close()
            raise StopIteration

        lines = list(islice(self.data, self.chunksize))
        if not lines:
            self.close()
            raise StopIteration

        try:
            lines_json = self._combine_lines(lines)
            obj = self._get_object_parser(lines_json)

            # Make sure that the returned objects have the right index.
            obj.index = range(self.nrows_seen, self.nrows_seen + len(obj))
            self.nrows_seen += len(obj)
        except Exception as ex:
            self.close()
            raise ex

        if self.dtype_backend is not lib.no_default:
            return obj.convert_dtypes(
                infer_objects=False, dtype_backend=self.dtype_backend
            )
        else:
            return obj

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class Parser:
    _split_keys: tuple[str, ...]
    _default_orient: str

    _STAMP_UNITS = ("s", "ms", "us", "ns")
    _MIN_STAMPS = {
        "s": 31536000,
        "ms": 31536000000,
        "us": 31536000000000,
        "ns": 31536000000000000,
    }
    json: str

    def __init__(
        self,
        json: str,
        orient,
        dtype: DtypeArg | None = None,
        convert_axes: bool = True,
        convert_dates: bool | list[str] = True,
        keep_default_dates: bool = False,
        precise_float: bool = False,
        date_unit=None,
        dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    ) -> None:
        self.json = json

        if orient is None:
            orient = self._default_orient

        self.orient = orient

        self.dtype = dtype

        if date_unit is not None:
            date_unit = date_unit.lower()
            if date_unit not in self._STAMP_UNITS:
                raise ValueError(f"date_unit must be one of {self._STAMP_UNITS}")
            self.min_stamp = self._MIN_STAMPS[date_unit]
        else:
            self.min_stamp = self._MIN_STAMPS["s"]

        self.precise_float = precise_float
        self.convert_axes = convert_axes
        self.convert_dates = convert_dates
        self.date_unit = date_unit
        self.keep_default_dates = keep_default_dates
        self.obj: DataFrame | Series | None = None
        self.dtype_backend = dtype_backend

    @final
    def check_keys_split(self, decoded: dict) -> None:
        """
        Checks that dict has only the appropriate keys for orient='split'.
        """
        bad_keys = set(decoded.keys()).difference(set(self._split_keys))
        if bad_keys:
            bad_keys_joined = ", ".join(bad_keys)
            raise ValueError(f"JSON data had unexpected key(s): {bad_keys_joined}")

    @final
    def parse(self):
        self._parse()

        if self.obj is None:
            return None
        if self.convert_axes:
            self._convert_axes()
        self._try_convert_types()
        return self.obj

    def _parse(self) -> None:
        raise AbstractMethodError(self)

    @final
    def _convert_axes(self) -> None:
        """
        Try to convert axes.
        """
        obj = self.obj
        assert obj is not None  # for mypy
        for axis_name in obj._AXIS_ORDERS:
            ax = obj._get_axis(axis_name)
            ser = Series(ax, dtype=ax.dtype, copy=False)
            new_ser, result = self._try_convert_data(
                name=axis_name,
                data=ser,
                use_dtypes=False,
                convert_dates=True,
                is_axis=True,
            )
            if result:
                new_axis = Index(new_ser, dtype=new_ser.dtype, copy=False)
                setattr(self.obj, axis_name, new_axis)

    def _try_convert_types(self) -> None:
        raise AbstractMethodError(self)

    @final
    def _try_convert_data(
        self,
        name: Hashable,
        data: Series,
        use_dtypes: bool = True,
        convert_dates: bool | list[str] = True,
        is_axis: bool = False,
    ) -> tuple[Series, bool]:
        """
        Try to parse a Series into a column by inferring dtype.
        """
        # don't try to coerce, unless a force conversion
        if use_dtypes:
            if not self.dtype:
                if all(notna(data)):
                    return data, False

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        "Downcasting object dtype arrays",
                        category=FutureWarning,
                    )
                    filled = data.fillna(np.nan)

                return filled, True

            elif self.dtype is True:
                pass
            else:
                # dtype to force
                dtype = (
                    self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype
                )
                if dtype is not None:
                    try:
                        return data.astype(dtype), True
                    except (TypeError, ValueError):
                        return data, False

        if convert_dates:
            new_data, result = self._try_convert_to_date(data)
            if result:
                return new_data, True

        converted = False
        if self.dtype_backend is not lib.no_default and not is_axis:
            # Fall through for conversion later on
            return data, True
        elif is_string_dtype(data.dtype):
            # try float
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        if data.dtype.kind == "f" and data.dtype != "float64":
            # coerce floats to 64
            try:
                data = data.astype("float64")
                converted = True
            except (TypeError, ValueError):
                pass

        # don't coerce 0-len data
        if len(data) and data.dtype in ("float", "object"):
            # coerce ints if we can
            try:
                new_data = data.astype("int64")
                if (new_data == data).all():
                    data = new_data
                    converted = True
            except (TypeError, ValueError, OverflowError):
                pass

        if data.dtype == "int" and data.dtype != "int64":
            # coerce ints to 64
            try:
                data = data.astype("int64")
                converted = True
            except (TypeError, ValueError):
                pass

        # if we have an index, we want to preserve dtypes
        if name == "index" and len(data):
            if self.orient == "split":
                return data, False

        return data, converted

    @final
    def _try_convert_to_date(self, data: Series) -> tuple[Series, bool]:
        """
        Try to parse a ndarray like into a date column.

        Try to coerce object in epoch/iso formats and integer/float in epoch
        formats. Return a boolean if parsing was successful.
        """
        # no conversion on empty
        if not len(data):
            return data, False

        new_data = data

        if new_data.dtype == "string":
            new_data = new_data.astype(object)

        if new_data.dtype == "object":
            try:
                new_data = data.astype("int64")
            except OverflowError:
                return data, False
            except (TypeError, ValueError):
                pass

        # ignore numbers that are out of range
        if issubclass(new_data.dtype.type, np.number):
            in_range = (
                isna(new_data._values)
                | (new_data > self.min_stamp)
                | (new_data._values == iNaT)
            )
            if not in_range.all():
                return data, False

        date_units = (self.date_unit,) if self.date_unit else self._STAMP_UNITS
        for date_unit in date_units:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*parsing datetimes with mixed time "
                        "zones will raise an error",
                        category=FutureWarning,
                    )
                    new_data = to_datetime(new_data, errors="raise", unit=date_unit)
            except (ValueError, OverflowError, TypeError):
                continue
            return new_data, True
        return data, False


class SeriesParser(Parser):
    _default_orient = "index"
    _split_keys = ("name", "index", "data")
    obj: Series | None

    def _parse(self) -> None:
        data = ujson_loads(self.json, precise_float=self.precise_float)

        if self.orient == "split":
            decoded = {str(k): v for k, v in data.items()}
            self.check_keys_split(decoded)
            self.obj = Series(**decoded)
        else:
            self.obj = Series(data)

    def _try_convert_types(self) -> None:
        if self.obj is None:
            return
        obj, result = self._try_convert_data(
            "data", self.obj, convert_dates=self.convert_dates
        )
        if result:
            self.obj = obj


class FrameParser(Parser):
    _default_orient = "columns"
    _split_keys = ("columns", "index", "data")
    obj: DataFrame | None

    def _parse(self) -> None:
        json = self.json
        orient = self.orient

        if orient == "columns":
            self.obj = DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )
        elif orient == "split":
            decoded = {
                str(k): v
                for k, v in ujson_loads(json, precise_float=self.precise_float).items()
            }
            self.check_keys_split(decoded)
            orig_names = [
                (tuple(col) if isinstance(col, list) else col)
                for col in decoded["columns"]
            ]
            decoded["columns"] = dedup_names(
                orig_names,
                is_potential_multi_index(orig_names, None),
            )
            self.obj = DataFrame(dtype=None, **decoded)
        elif orient == "index":
            self.obj = DataFrame.from_dict(
                ujson_loads(json, precise_float=self.precise_float),
                dtype=None,
                orient="index",
            )
        elif orient == "table":
            self.obj = parse_table_schema(json, precise_float=self.precise_float)
        else:
            self.obj = DataFrame(
                ujson_loads(json, precise_float=self.precise_float), dtype=None
            )

    def _process_converter(
        self,
        f: Callable[[Hashable, Series], tuple[Series, bool]],
        filt: Callable[[Hashable], bool] | None = None,
    ) -> None:
        """
        Take a conversion function and possibly recreate the frame.
        """
        if filt is None:
            filt = lambda col: True

        obj = self.obj
        assert obj is not None  # for mypy

        needs_new_obj = False
        new_obj = {}
        for i, (col, c) in enumerate(obj.items()):
            if filt(col):
                new_data, result = f(col, c)
                if result:
                    c = new_data
                    needs_new_obj = True
            new_obj[i] = c

        if needs_new_obj:
            # possibly handle dup columns
            new_frame = DataFrame(new_obj, index=obj.index)
            new_frame.columns = obj.columns
            self.obj = new_frame

    def _try_convert_types(self) -> None:
        if self.obj is None:
            return
        if self.convert_dates:
            self._try_convert_dates()

        self._process_converter(
            lambda col, c: self._try_convert_data(col, c, convert_dates=False)
        )

    def _try_convert_dates(self) -> None:
        if self.obj is None:
            return

        # our columns to parse
        convert_dates_list_bool = self.convert_dates
        if isinstance(convert_dates_list_bool, bool):
            convert_dates_list_bool = []
        convert_dates = set(convert_dates_list_bool)

        def is_ok(col) -> bool:
            """
            Return if this col is ok to try for a date parse.
            """
            if col in convert_dates:
                return True
            if not self.keep_default_dates:
                return False
            if not isinstance(col, str):
                return False

            col_lower = col.lower()
            if (
                col_lower.endswith(("_at", "_time"))
                or col_lower == "modified"
                or col_lower == "date"
                or col_lower == "datetime"
                or col_lower.startswith("timestamp")
            ):
                return True
            return False

        self._process_converter(lambda col, c: self._try_convert_to_date(c), filt=is_ok)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\complexes.py ===
from __future__ import annotations

from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (DefinedFunction, Derivative, ArgumentIndexError,
    AppliedUndef, expand_mul, PoleError)
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise

###############################################################################
######################### REAL and IMAGINARY PARTS ############################
###############################################################################


class re(DefinedFunction):
    """
    Returns real part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly
    more complicated expressions. If completely simplified result
    is needed then use ``Basic.as_real_imag()`` or perform complex
    expansion on instance of this function.

    Examples
    ========

    >>> from sympy import re, im, I, E, symbols
    >>> x, y = symbols('x y', real=True)
    >>> re(2*E)
    2*E
    >>> re(2*I + 17)
    17
    >>> re(2*I)
    0
    >>> re(im(x) + x*I + 2)
    2
    >>> re(5 + I + 2)
    7

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Real part of expression.

    See Also
    ========

    im
    """

    args: tuple[Expr]

    is_extended_real = True
    unbranched = True  # implicitly works on the projection to C
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return arg
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return S.Zero
        elif arg.is_Matrix:
            return arg.as_real_imag()[0]
        elif arg.is_Function and isinstance(arg, conjugate):
            return re(arg.args[0])
        else:

            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                elif not term.has(I) and term.is_extended_real:
                    excluded.append(term)
                else:
                    # Try to do some advanced expansion.  If
                    # impossible, don't try to do re(arg) again
                    # (because this is what we are trying to do now).
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[0])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) - im(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Returns the real number with a zero imaginary part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return re(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * im(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_im(self, arg, **kwargs):
        return self.args[0] - I*im(self.args[0])

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        # is_imaginary implies nonzero
        return fuzzy_or([self.args[0].is_imaginary, self.args[0].is_zero])

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True


class im(DefinedFunction):
    """
    Returns imaginary part of expression. This function performs only
    elementary analysis and so it will fail to decompose properly more
    complicated expressions. If completely simplified result is needed then
    use ``Basic.as_real_imag()`` or perform complex expansion on instance of
    this function.

    Examples
    ========

    >>> from sympy import re, im, E, I
    >>> from sympy.abc import x, y
    >>> im(2*E)
    0
    >>> im(2*I + 17)
    2
    >>> im(x*I)
    re(x)
    >>> im(re(x) + y)
    im(y)
    >>> im(2 + 3*I)
    3

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Imaginary part of expression.

    See Also
    ========

    re
    """

    args: tuple[Expr]

    is_extended_real = True
    unbranched = True  # implicitly works on the projection to C
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return S.NaN
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif arg.is_extended_real:
            return S.Zero
        elif arg.is_imaginary or (I*arg).is_extended_real:
            return -I * arg
        elif arg.is_Matrix:
            return arg.as_real_imag()[1]
        elif arg.is_Function and isinstance(arg, conjugate):
            return -im(arg.args[0])
        else:
            included, reverted, excluded = [], [], []
            args = Add.make_args(arg)
            for term in args:
                coeff = term.as_coefficient(I)

                if coeff is not None:
                    if not coeff.is_extended_real:
                        reverted.append(coeff)
                    else:
                        excluded.append(coeff)
                elif term.has(I) or not term.is_extended_real:
                    # Try to do some advanced expansion.  If
                    # impossible, don't try to do im(arg) again
                    # (because this is what we are trying to do now).
                    real_imag = term.as_real_imag(ignore=arg)
                    if real_imag:
                        excluded.append(real_imag[1])
                    else:
                        included.append(term)

            if len(args) != len(included):
                a, b, c = (Add(*xs) for xs in [included, reverted, excluded])

                return cls(a) + re(b) + c

    def as_real_imag(self, deep=True, **hints):
        """
        Return the imaginary part with a zero real part.

        """
        return (self, S.Zero)

    def _eval_derivative(self, x):
        if x.is_extended_real or self.args[0].is_extended_real:
            return im(Derivative(self.args[0], x, evaluate=True))
        if x.is_imaginary or self.args[0].is_imaginary:
            return -I \
                * re(Derivative(self.args[0], x, evaluate=True))

    def _eval_rewrite_as_re(self, arg, **kwargs):
        return -I*(self.args[0] - re(self.args[0]))

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_is_zero(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_finite:
            return True

###############################################################################
############### SIGN, ABSOLUTE VALUE, ARGUMENT and CONJUGATION ################
###############################################################################

class sign(DefinedFunction):
    """
    Returns the complex sign of an expression:

    Explanation
    ===========

    If the expression is real the sign will be:

        * $1$ if expression is positive
        * $0$ if expression is equal to zero
        * $-1$ if expression is negative

    If the expression is imaginary the sign will be:

        * $I$ if im(expression) is positive
        * $-I$ if im(expression) is negative

    Otherwise an unevaluated expression will be returned. When evaluated, the
    result (in general) will be ``cos(arg(expr)) + I*sin(arg(expr))``.

    Examples
    ========

    >>> from sympy import sign, I

    >>> sign(-1)
    -1
    >>> sign(0)
    0
    >>> sign(-3*I)
    -I
    >>> sign(1 + I)
    sign(1 + I)
    >>> _.evalf()
    0.707106781186548 + 0.707106781186548*I

    Parameters
    ==========

    arg : Expr
        Real or imaginary expression.

    Returns
    =======

    expr : Expr
        Complex sign of expression.

    See Also
    ========

    Abs, conjugate
    """

    is_complex = True
    _singularities = True

    def doit(self, **hints):
        s = super().doit()
        if s == self and self.args[0].is_zero is False:
            return self.args[0] / Abs(self.args[0])
        return s

    @classmethod
    def eval(cls, arg):
        # handle what we can
        if arg.is_Mul:
            c, args = arg.as_coeff_mul()
            unk = []
            s = sign(c)
            for a in args:
                if a.is_extended_negative:
                    s = -s
                elif a.is_extended_positive:
                    pass
                else:
                    if a.is_imaginary:
                        ai = im(a)
                        if ai.is_comparable:  # i.e. a = I*real
                            s *= I
                            if ai.is_extended_negative:
                                # can't use sign(ai) here since ai might not be
                                # a Number
                                s = -s
                        else:
                            unk.append(a)
                    else:
                        unk.append(a)
            if c is S.One and len(unk) == len(args):
                return None
            return s * cls(arg._new_rawargs(*unk))
        if arg is S.NaN:
            return S.NaN
        if arg.is_zero:  # it may be an Expr that is zero
            return S.Zero
        if arg.is_extended_positive:
            return S.One
        if arg.is_extended_negative:
            return S.NegativeOne
        if arg.is_Function:
            if isinstance(arg, sign):
                return arg
        if arg.is_imaginary:
            if arg.is_Pow and arg.exp is S.Half:
                # we catch this because non-trivial sqrt args are not expanded
                # e.g. sqrt(1-sqrt(2)) --x-->  to I*sqrt(sqrt(2) - 1)
                return I
            arg2 = -I * arg
            if arg2.is_extended_positive:
                return I
            if arg2.is_extended_negative:
                return -I

    def _eval_Abs(self):
        if fuzzy_not(self.args[0].is_zero):
            return S.One

    def _eval_conjugate(self):
        return sign(conjugate(self.args[0]))

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(self.args[0])
        elif self.args[0].is_imaginary:
            from sympy.functions.special.delta_functions import DiracDelta
            return 2 * Derivative(self.args[0], x, evaluate=True) \
                * DiracDelta(-I * self.args[0])

    def _eval_is_nonnegative(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_nonpositive(self):
        if self.args[0].is_nonpositive:
            return True

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_extended_real

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_power(self, other):
        if (
            fuzzy_not(self.args[0].is_zero) and
            other.is_integer and
            other.is_even
        ):
            return S.One

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg0 = self.args[0]
        x0 = arg0.subs(x, 0)
        if x0 != 0:
            return self.func(x0)
        if cdir != 0:
            cdir = arg0.dir(x, cdir)
        return -S.One if re(cdir) < 0 else S.One

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((1, arg > 0), (-1, arg < 0), (0, True))

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return Heaviside(arg) * 2 - 1

    def _eval_rewrite_as_Abs(self, arg, **kwargs):
        return Piecewise((0, Eq(arg, 0)), (arg / Abs(arg), True))

    def _eval_simplify(self, **kwargs):
        return self.func(factor_terms(self.args[0]))  # XXX include doit?


class Abs(DefinedFunction):
    """
    Return the absolute value of the argument.

    Explanation
    ===========

    This is an extension of the built-in function ``abs()`` to accept symbolic
    values.  If you pass a SymPy expression to the built-in ``abs()``, it will
    pass it automatically to ``Abs()``.

    Examples
    ========

    >>> from sympy import Abs, Symbol, S, I
    >>> Abs(-1)
    1
    >>> x = Symbol('x', real=True)
    >>> Abs(-x)
    Abs(x)
    >>> Abs(x**2)
    x**2
    >>> abs(-x) # The Python built-in
    Abs(x)
    >>> Abs(3*x + 2*I)
    sqrt(9*x**2 + 4)
    >>> Abs(8*I)
    8

    Note that the Python built-in will return either an Expr or int depending on
    the argument::

        >>> type(abs(-1))
        <... 'int'>
        >>> type(abs(S.NegativeOne))
        <class 'sympy.core.numbers.One'>

    Abs will always return a SymPy object.

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    expr : Expr
        Absolute value returned can be an expression or integer depending on
        input arg.

    See Also
    ========

    sign, conjugate
    """

    args: tuple[Expr]

    is_extended_real = True
    is_extended_negative = False
    is_extended_nonnegative = True
    unbranched = True
    _singularities = True  # non-holomorphic

    def fdiff(self, argindex=1):
        """
        Get the first derivative of the argument to Abs().

        """
        if argindex == 1:
            return sign(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.simplify.simplify import signsimp

        if hasattr(arg, '_eval_Abs'):
            obj = arg._eval_Abs()
            if obj is not None:
                return obj
        if not isinstance(arg, Expr):
            raise TypeError("Bad argument type for Abs(): %s" % type(arg))

        # handle what we can
        arg = signsimp(arg, evaluate=False)
        n, d = arg.as_numer_denom()
        if d.free_symbols and not n.free_symbols:
            return cls(n)/cls(d)

        if arg.is_Mul:
            known = []
            unk = []
            for t in arg.args:
                if t.is_Pow and t.exp.is_integer and t.exp.is_negative:
                    bnew = cls(t.base)
                    if isinstance(bnew, cls):
                        unk.append(t)
                    else:
                        known.append(Pow(bnew, t.exp))
                else:
                    tnew = cls(t)
                    if isinstance(tnew, cls):
                        unk.append(t)
                    else:
                        known.append(tnew)
            known = Mul(*known)
            unk = cls(Mul(*unk), evaluate=False) if unk else S.One
            return known*unk
        if arg is S.NaN:
            return S.NaN
        if arg is S.ComplexInfinity:
            return oo
        from sympy.functions.elementary.exponential import exp, log

        if arg.is_Pow:
            base, exponent = arg.as_base_exp()
            if base.is_extended_real:
                if exponent.is_integer:
                    if exponent.is_even:
                        return arg
                    if base is S.NegativeOne:
                        return S.One
                    return Abs(base)**exponent
                if base.is_extended_nonnegative:
                    return base**re(exponent)
                if base.is_extended_negative:
                    return (-base)**re(exponent)*exp(-pi*im(exponent))
                return
            elif not base.has(Symbol): # complex base
                # express base**exponent as exp(exponent*log(base))
                a, b = log(base).as_real_imag()
                z = a + I*b
                return exp(re(exponent*z))
        if isinstance(arg, exp):
            return exp(re(arg.args[0]))
        if isinstance(arg, AppliedUndef):
            if arg.is_positive:
                return arg
            elif arg.is_negative:
                return -arg
            return
        if arg.is_Add and arg.has(oo, S.NegativeInfinity):
            if any(a.is_infinite for a in arg.as_real_imag()):
                return oo
        if arg.is_zero:
            return S.Zero
        if arg.is_extended_nonnegative:
            return arg
        if arg.is_extended_nonpositive:
            return -arg
        if arg.is_imaginary:
            arg2 = -I * arg
            if arg2.is_extended_nonnegative:
                return arg2
        if arg.is_extended_real:
            return
        # reject result if all new conjugates are just wrappers around
        # an expression that was already in the arg
        conj = signsimp(arg.conjugate(), evaluate=False)
        new_conj = conj.atoms(conjugate) - arg.atoms(conjugate)
        if new_conj and all(arg.has(i.args[0]) for i in new_conj):
            return
        if arg != conj and arg != -conj:
            ignore = arg.atoms(Abs)
            abs_free_arg = arg.xreplace({i: Dummy(real=True) for i in ignore})
            unk = [a for a in abs_free_arg.free_symbols if a.is_extended_real is None]
            if not unk or not all(conj.has(conjugate(u)) for u in unk):
                return sqrt(expand_mul(arg*conj))

    def _eval_is_real(self):
        if self.args[0].is_finite:
            return True

    def _eval_is_integer(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_integer

    def _eval_is_extended_nonzero(self):
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_zero(self):
        return self._args[0].is_zero

    def _eval_is_extended_positive(self):
        return fuzzy_not(self._args[0].is_zero)

    def _eval_is_rational(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_rational

    def _eval_is_even(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_even

    def _eval_is_odd(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_odd

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic

    def _eval_power(self, exponent):
        if self.args[0].is_extended_real and exponent.is_integer:
            if exponent.is_even:
                return self.args[0]**exponent
            elif exponent is not S.NegativeOne and exponent.is_Integer:
                return self.args[0]**(exponent - 1)*self
        return

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.exponential import log
        direction = self.args[0].leadterm(x)[0]
        if direction.has(log(x)):
            direction = direction.subs(log(x), logx)
        s = self.args[0]._eval_nseries(x, n=n, logx=logx)
        return (sign(direction)*s).expand()

    def _eval_derivative(self, x):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return Derivative(self.args[0], x, evaluate=True) \
                * sign(conjugate(self.args[0]))
        rv = (re(self.args[0]) * Derivative(re(self.args[0]), x,
            evaluate=True) + im(self.args[0]) * Derivative(im(self.args[0]),
                x, evaluate=True)) / Abs(self.args[0])
        return rv.rewrite(sign)

    def _eval_rewrite_as_Heaviside(self, arg, **kwargs):
        # Note this only holds for real arg (since Heaviside is not defined
        # for complex arguments).
        from sympy.functions.special.delta_functions import Heaviside
        if arg.is_extended_real:
            return arg*(Heaviside(arg) - Heaviside(-arg))

    def _eval_rewrite_as_Piecewise(self, arg, **kwargs):
        if arg.is_extended_real:
            return Piecewise((arg, arg >= 0), (-arg, True))
        elif arg.is_imaginary:
            return Piecewise((I*arg, I*arg >= 0), (-I*arg, True))

    def _eval_rewrite_as_sign(self, arg, **kwargs):
        return arg/sign(arg)

    def _eval_rewrite_as_conjugate(self, arg, **kwargs):
        return sqrt(arg*conjugate(arg))


class arg(DefinedFunction):
    r"""
    Returns the argument (in radians) of a complex number. The argument is
    evaluated in consistent convention with ``atan2`` where the branch-cut is
    taken along the negative real axis and ``arg(z)`` is in the interval
    $(-\pi,\pi]$. For a positive number, the argument is always 0; the
    argument of a negative number is $\pi$; and the argument of 0
    is undefined and returns ``nan``. So the ``arg`` function will never nest
    greater than 3 levels since at the 4th application, the result must be
    nan; for a real number, nan is returned on the 3rd application.

    Examples
    ========

    >>> from sympy import arg, I, sqrt, Dummy
    >>> from sympy.abc import x
    >>> arg(2.0)
    0
    >>> arg(I)
    pi/2
    >>> arg(sqrt(2) + I*sqrt(2))
    pi/4
    >>> arg(sqrt(3)/2 + I/2)
    pi/6
    >>> arg(4 + 3*I)
    atan(3/4)
    >>> arg(0.8 + 0.6*I)
    0.643501108793284
    >>> arg(arg(arg(arg(x))))
    nan
    >>> real = Dummy(real=True)
    >>> arg(arg(arg(real)))
    nan

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    value : Expr
        Returns arc tangent of arg measured in radians.

    """

    is_extended_real = True
    is_real = True
    is_finite = True
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        a = arg
        for i in range(3):
            if isinstance(a, cls):
                a = a.args[0]
            else:
                if i == 2 and a.is_extended_real:
                    return S.NaN
                break
        else:
            return S.NaN
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(arg, exp_polar):
            return periodic_argument(arg, oo)
        elif isinstance(arg, exp):
            i_ = im(arg.args[0])
            if i_.is_comparable:
                i_ %= 2*S.Pi
                if i_ > S.Pi:
                    i_ -= 2*S.Pi
                return i_

        if not arg.is_Atom:
            c, arg_ = factor_terms(arg).as_coeff_Mul()
            if arg_.is_Mul:
                arg_ = Mul(*[a if (sign(a) not in (-1, 1)) else
                    sign(a) for a in arg_.args])
            arg_ = sign(c)*arg_
        else:
            arg_ = arg
        if any(i.is_extended_positive is None for i in arg_.atoms(AppliedUndef)):
            return
        from sympy.functions.elementary.trigonometric import atan2
        x, y = arg_.as_real_imag()
        rv = atan2(y, x)
        if rv.is_number:
            return rv
        if arg_ != arg:
            return cls(arg_, evaluate=False)

    def _eval_derivative(self, t):
        x, y = self.args[0].as_real_imag()
        return (x * Derivative(y, t, evaluate=True) - y *
                    Derivative(x, t, evaluate=True)) / (x**2 + y**2)

    def _eval_rewrite_as_atan2(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import atan2
        x, y = self.args[0].as_real_imag()
        return atan2(y, x)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg0 = self.args[0]
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg0.subs(x, cdir*t)
        if z.is_positive:
            return S.Zero
        elif z.is_negative:
            return S.Pi
        else:
            raise PoleError("Cannot expand %s around 0" % (self))

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        if n <= 0:
            return Order(1)
        return self._eval_as_leading_term(x, logx=logx, cdir=cdir)


class conjugate(DefinedFunction):
    """
    Returns the *complex conjugate* [1]_ of an argument.
    In mathematics, the complex conjugate of a complex number
    is given by changing the sign of the imaginary part.

    Thus, the conjugate of the complex number
    :math:`a + ib` (where $a$ and $b$ are real numbers) is :math:`a - ib`

    Examples
    ========

    >>> from sympy import conjugate, I
    >>> conjugate(2)
    2
    >>> conjugate(I)
    -I
    >>> conjugate(3 + 2*I)
    3 - 2*I
    >>> conjugate(5 - I)
    5 + I

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    Returns
    =======

    arg : Expr
        Complex conjugate of arg as real, imaginary or mixed expression.

    See Also
    ========

    sign, Abs

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_conjugation
    """
    _singularities = True  # non-holomorphic

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_conjugate()
        if obj is not None:
            return obj

    def inverse(self):
        return conjugate

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)

    def _eval_adjoint(self):
        return transpose(self.args[0])

    def _eval_conjugate(self):
        return self.args[0]

    def _eval_derivative(self, x):
        if x.is_real:
            return conjugate(Derivative(self.args[0], x, evaluate=True))
        elif x.is_imaginary:
            return -conjugate(Derivative(self.args[0], x, evaluate=True))

    def _eval_transpose(self):
        return adjoint(self.args[0])

    def _eval_is_algebraic(self):
        return self.args[0].is_algebraic


class transpose(DefinedFunction):
    """
    Linear map transposition.

    Examples
    ========

    >>> from sympy import transpose, Matrix, MatrixSymbol
    >>> A = MatrixSymbol('A', 25, 9)
    >>> transpose(A)
    A.T
    >>> B = MatrixSymbol('B', 9, 22)
    >>> transpose(B)
    B.T
    >>> transpose(A*B)
    B.T*A.T
    >>> M = Matrix([[4, 5], [2, 1], [90, 12]])
    >>> M
    Matrix([
    [ 4,  5],
    [ 2,  1],
    [90, 12]])
    >>> transpose(M)
    Matrix([
    [4, 2, 90],
    [5, 1, 12]])

    Parameters
    ==========

    arg : Matrix
         Matrix or matrix expression to take the transpose of.

    Returns
    =======

    value : Matrix
        Transpose of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_transpose()
        if obj is not None:
            return obj

    def _eval_adjoint(self):
        return conjugate(self.args[0])

    def _eval_conjugate(self):
        return adjoint(self.args[0])

    def _eval_transpose(self):
        return self.args[0]


class adjoint(DefinedFunction):
    """
    Conjugate transpose or Hermite conjugation.

    Examples
    ========

    >>> from sympy import adjoint, MatrixSymbol
    >>> A = MatrixSymbol('A', 10, 5)
    >>> adjoint(A)
    Adjoint(A)

    Parameters
    ==========

    arg : Matrix
        Matrix or matrix expression to take the adjoint of.

    Returns
    =======

    value : Matrix
        Represents the conjugate transpose or Hermite
        conjugation of arg.

    """

    @classmethod
    def eval(cls, arg):
        obj = arg._eval_adjoint()
        if obj is not None:
            return obj
        obj = arg._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def _eval_adjoint(self):
        return self.args[0]

    def _eval_conjugate(self):
        return transpose(self.args[0])

    def _eval_transpose(self):
        return conjugate(self.args[0])

    def _latex(self, printer, exp=None, *args):
        arg = printer._print(self.args[0])
        tex = r'%s^{\dagger}' % arg
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = printer._print(self.args[0], *args)
        if printer._use_unicode:
            pform = pform**prettyForm('\N{DAGGER}')
        else:
            pform = pform**prettyForm('+')
        return pform

###############################################################################
############### HANDLING OF POLAR NUMBERS #####################################
###############################################################################


class polar_lift(DefinedFunction):
    """
    Lift argument to the Riemann surface of the logarithm, using the
    standard branch.

    Examples
    ========

    >>> from sympy import Symbol, polar_lift, I
    >>> p = Symbol('p', polar=True)
    >>> x = Symbol('x')
    >>> polar_lift(4)
    4*exp_polar(0)
    >>> polar_lift(-4)
    4*exp_polar(I*pi)
    >>> polar_lift(-I)
    exp_polar(-I*pi/2)
    >>> polar_lift(I + 2)
    polar_lift(2 + I)

    >>> polar_lift(4*x)
    4*polar_lift(x)
    >>> polar_lift(4*p)
    4*p

    Parameters
    ==========

    arg : Expr
        Real or complex expression.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    periodic_argument
    """

    is_polar = True
    is_comparable = False  # Cannot be evalf'd.

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.complexes import arg as argument
        if arg.is_number:
            ar = argument(arg)
            # In general we want to affirm that something is known,
            # e.g. `not ar.has(argument) and not ar.has(atan)`
            # but for now we will just be more restrictive and
            # see that it has evaluated to one of the known values.
            if ar in (0, pi/2, -pi/2, pi):
                from sympy.functions.elementary.exponential import exp_polar
                return exp_polar(I*ar)*abs(arg)

        if arg.is_Mul:
            args = arg.args
        else:
            args = [arg]
        included = []
        excluded = []
        positive = []
        for arg in args:
            if arg.is_polar:
                included += [arg]
            elif arg.is_positive:
                positive += [arg]
            else:
                excluded += [arg]
        if len(excluded) < len(args):
            if excluded:
                return Mul(*(included + positive))*polar_lift(Mul(*excluded))
            elif included:
                return Mul(*(included + positive))
            else:
                from sympy.functions.elementary.exponential import exp_polar
                return Mul(*positive)*exp_polar(0)

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        return self.args[0]._eval_evalf(prec)

    def _eval_Abs(self):
        return Abs(self.args[0], evaluate=True)


class periodic_argument(DefinedFunction):
    r"""
    Represent the argument on a quotient of the Riemann surface of the
    logarithm. That is, given a period $P$, always return a value in
    $(-P/2, P/2]$, by using $\exp(PI) = 1$.

    Examples
    ========

    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(10*I*pi), 2*pi)
    0
    >>> periodic_argument(exp_polar(5*I*pi), 4*pi)
    pi
    >>> from sympy import exp_polar, periodic_argument
    >>> from sympy import I, pi
    >>> periodic_argument(exp_polar(5*I*pi), 2*pi)
    pi
    >>> periodic_argument(exp_polar(5*I*pi), 3*pi)
    -pi
    >>> periodic_argument(exp_polar(5*I*pi), pi)
    0

    Parameters
    ==========

    ar : Expr
        A polar number.

    period : Expr
        The period $P$.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    principal_branch
    """

    @classmethod
    def _getunbranched(cls, ar):
        from sympy.functions.elementary.exponential import exp_polar, log
        if ar.is_Mul:
            args = ar.args
        else:
            args = [ar]
        unbranched = 0
        for a in args:
            if not a.is_polar:
                unbranched += arg(a)
            elif isinstance(a, exp_polar):
                unbranched += a.exp.as_real_imag()[1]
            elif a.is_Pow:
                re, im = a.exp.as_real_imag()
                unbranched += re*unbranched_argument(
                    a.base) + im*log(abs(a.base))
            elif isinstance(a, polar_lift):
                unbranched += arg(a.args[0])
            else:
                return None
        return unbranched

    @classmethod
    def eval(cls, ar, period):
        # Our strategy is to evaluate the argument on the Riemann surface of the
        # logarithm, and then reduce.
        # NOTE evidently this means it is a rather bad idea to use this with
        # period != 2*pi and non-polar numbers.
        if not period.is_extended_positive:
            return None
        if period == oo and isinstance(ar, principal_branch):
            return periodic_argument(*ar.args)
        if isinstance(ar, polar_lift) and period >= 2*pi:
            return periodic_argument(ar.args[0], period)
        if ar.is_Mul:
            newargs = [x for x in ar.args if not x.is_positive]
            if len(newargs) != len(ar.args):
                return periodic_argument(Mul(*newargs), period)
        unbranched = cls._getunbranched(ar)
        if unbranched is None:
            return None
        from sympy.functions.elementary.trigonometric import atan, atan2
        if unbranched.has(periodic_argument, atan2, atan):
            return None
        if period == oo:
            return unbranched
        if period != oo:
            from sympy.functions.elementary.integers import ceiling
            n = ceiling(unbranched/period - S.Half)*period
            if not n.has(ceiling):
                return unbranched - n

    def _eval_evalf(self, prec):
        z, period = self.args
        if period == oo:
            unbranched = periodic_argument._getunbranched(z)
            if unbranched is None:
                return self
            return unbranched._eval_evalf(prec)
        ub = periodic_argument(z, oo)._eval_evalf(prec)
        from sympy.functions.elementary.integers import ceiling
        return (ub - ceiling(ub/period - S.Half)*period)._eval_evalf(prec)


def unbranched_argument(arg):
    '''
    Returns periodic argument of arg with period as infinity.

    Examples
    ========

    >>> from sympy import exp_polar, unbranched_argument
    >>> from sympy import I, pi
    >>> unbranched_argument(exp_polar(15*I*pi))
    15*pi
    >>> unbranched_argument(exp_polar(7*I*pi))
    7*pi

    See also
    ========

    periodic_argument
    '''
    return periodic_argument(arg, oo)


class principal_branch(DefinedFunction):
    """
    Represent a polar number reduced to its principal branch on a quotient
    of the Riemann surface of the logarithm.

    Explanation
    ===========

    This is a function of two arguments. The first argument is a polar
    number `z`, and the second one a positive real number or infinity, `p`.
    The result is ``z mod exp_polar(I*p)``.

    Examples
    ========

    >>> from sympy import exp_polar, principal_branch, oo, I, pi
    >>> from sympy.abc import z
    >>> principal_branch(z, oo)
    z
    >>> principal_branch(exp_polar(2*pi*I)*3, 2*pi)
    3*exp_polar(0)
    >>> principal_branch(exp_polar(2*pi*I)*3*z, 2*pi)
    3*principal_branch(z, 2*pi)

    Parameters
    ==========

    x : Expr
        A polar number.

    period : Expr
        Positive real number or infinity.

    See Also
    ========

    sympy.functions.elementary.exponential.exp_polar
    polar_lift : Lift argument to the Riemann surface of the logarithm
    periodic_argument
    """

    is_polar = True
    is_comparable = False  # cannot always be evalf'd

    @classmethod
    def eval(self, x, period):
        from sympy.functions.elementary.exponential import exp_polar
        if isinstance(x, polar_lift):
            return principal_branch(x.args[0], period)
        if period == oo:
            return x
        ub = periodic_argument(x, oo)
        barg = periodic_argument(x, period)
        if ub != barg and not ub.has(periodic_argument) \
                and not barg.has(periodic_argument):
            pl = polar_lift(x)

            def mr(expr):
                if not isinstance(expr, Symbol):
                    return polar_lift(expr)
                return expr
            pl = pl.replace(polar_lift, mr)
            # Recompute unbranched argument
            ub = periodic_argument(pl, oo)
            if not pl.has(polar_lift):
                if ub != barg:
                    res = exp_polar(I*(barg - ub))*pl
                else:
                    res = pl
                if not res.is_polar and not res.has(exp_polar):
                    res *= exp_polar(0)
                return res

        if not x.free_symbols:
            c, m = x, ()
        else:
            c, m = x.as_coeff_mul(*x.free_symbols)
        others = []
        for y in m:
            if y.is_positive:
                c *= y
            else:
                others += [y]
        m = tuple(others)
        arg = periodic_argument(c, period)
        if arg.has(periodic_argument):
            return None
        if arg.is_number and (unbranched_argument(c) != arg or
                              (arg == 0 and m != () and c != 1)):
            if arg == 0:
                return abs(c)*principal_branch(Mul(*m), period)
            return principal_branch(exp_polar(I*arg)*Mul(*m), period)*abs(c)
        if arg.is_number and ((abs(arg) < period/2) == True or arg == period/2) \
                and m == ():
            return exp_polar(arg*I)*abs(c)

    def _eval_evalf(self, prec):
        z, period = self.args
        p = periodic_argument(z, period)._eval_evalf(prec)
        if abs(p) > pi or p == -pi:
            return self  # Cannot evalf for this argument.
        from sympy.functions.elementary.exponential import exp
        return (abs(z)*exp(I*p))._eval_evalf(prec)


def _polarify(eq, lift, pause=False):
    from sympy.integrals.integrals import Integral
    if eq.is_polar:
        return eq
    if eq.is_number and not pause:
        return polar_lift(eq)
    if isinstance(eq, Symbol) and not pause and lift:
        return polar_lift(eq)
    elif eq.is_Atom:
        return eq
    elif eq.is_Add:
        r = eq.func(*[_polarify(arg, lift, pause=True) for arg in eq.args])
        if lift:
            return polar_lift(r)
        return r
    elif eq.is_Pow and eq.base == S.Exp1:
        return eq.func(S.Exp1, _polarify(eq.exp, lift, pause=False))
    elif eq.is_Function:
        return eq.func(*[_polarify(arg, lift, pause=False) for arg in eq.args])
    elif isinstance(eq, Integral):
        # Don't lift the integration variable
        func = _polarify(eq.function, lift, pause=pause)
        limits = []
        for limit in eq.args[1:]:
            var = _polarify(limit[0], lift=False, pause=pause)
            rest = _polarify(limit[1:], lift=lift, pause=pause)
            limits.append((var,) + rest)
        return Integral(*((func,) + tuple(limits)))
    else:
        return eq.func(*[_polarify(arg, lift, pause=pause)
                         if isinstance(arg, Expr) else arg for arg in eq.args])


def polarify(eq, subs=True, lift=False):
    """
    Turn all numbers in eq into their polar equivalents (under the standard
    choice of argument).

    Note that no attempt is made to guess a formal convention of adding
    polar numbers, expressions like $1 + x$ will generally not be altered.

    Note also that this function does not promote ``exp(x)`` to ``exp_polar(x)``.

    If ``subs`` is ``True``, all symbols which are not already polar will be
    substituted for polar dummies; in this case the function behaves much
    like :func:`~.posify`.

    If ``lift`` is ``True``, both addition statements and non-polar symbols are
    changed to their ``polar_lift()``ed versions.
    Note that ``lift=True`` implies ``subs=False``.

    Examples
    ========

    >>> from sympy import polarify, sin, I
    >>> from sympy.abc import x, y
    >>> expr = (-x)**y
    >>> expr.expand()
    (-x)**y
    >>> polarify(expr)
    ((_x*exp_polar(I*pi))**_y, {_x: x, _y: y})
    >>> polarify(expr)[0].expand()
    _x**_y*exp_polar(_y*I*pi)
    >>> polarify(x, lift=True)
    polar_lift(x)
    >>> polarify(x*(1+y), lift=True)
    polar_lift(x)*polar_lift(y + 1)

    Adds are treated carefully:

    >>> polarify(1 + sin((1 + I)*x))
    (sin(_x*polar_lift(1 + I)) + 1, {_x: x})
    """
    if lift:
        subs = False
    eq = _polarify(sympify(eq), lift)
    if not subs:
        return eq
    reps = {s: Dummy(s.name, polar=True) for s in eq.free_symbols}
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}


def _unpolarify(eq, exponents_only, pause=False):
    if not isinstance(eq, Basic) or eq.is_Atom:
        return eq

    if not pause:
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(eq, exp_polar):
            return exp(_unpolarify(eq.exp, exponents_only))
        if isinstance(eq, principal_branch) and eq.args[1] == 2*pi:
            return _unpolarify(eq.args[0], exponents_only)
        if (
            eq.is_Add or eq.is_Mul or eq.is_Boolean or
            eq.is_Relational and (
                eq.rel_op in ('==', '!=') and 0 in eq.args or
                eq.rel_op not in ('==', '!='))
        ):
            return eq.func(*[_unpolarify(x, exponents_only) for x in eq.args])
        if isinstance(eq, polar_lift):
            return _unpolarify(eq.args[0], exponents_only)

    if eq.is_Pow:
        expo = _unpolarify(eq.exp, exponents_only)
        base = _unpolarify(eq.base, exponents_only,
            not (expo.is_integer and not pause))
        return base**expo

    if eq.is_Function and getattr(eq.func, 'unbranched', False):
        return eq.func(*[_unpolarify(x, exponents_only, exponents_only)
            for x in eq.args])

    return eq.func(*[_unpolarify(x, exponents_only, True) for x in eq.args])


def unpolarify(eq, subs=None, exponents_only=False):
    """
    If `p` denotes the projection from the Riemann surface of the logarithm to
    the complex line, return a simplified version `eq'` of `eq` such that
    `p(eq') = p(eq)`.
    Also apply the substitution subs in the end. (This is a convenience, since
    ``unpolarify``, in a certain sense, undoes :func:`polarify`.)

    Examples
    ========

    >>> from sympy import unpolarify, polar_lift, sin, I
    >>> unpolarify(polar_lift(I + 2))
    2 + I
    >>> unpolarify(sin(polar_lift(I + 7)))
    sin(7 + I)
    """
    if isinstance(eq, bool):
        return eq

    eq = sympify(eq)
    if subs is not None:
        return unpolarify(eq.subs(subs))
    changed = True
    pause = False
    if exponents_only:
        pause = True
    while changed:
        changed = False
        res = _unpolarify(eq, exponents_only, pause)
        if res != eq:
            changed = True
            eq = res
        if isinstance(res, bool):
            return res
    # Finally, replacing Exp(0) by 1 is always correct.
    # So is polar_lift(0) -> 0.
    from sympy.functions.elementary.exponential import exp_polar
    return res.subs({exp_polar(0): 1, polar_lift(0): 0})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\interfaces.py ===
# orm/interfaces.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""

Contains various base classes used throughout the ORM.

Defines some key base classes prominent within the internals.

This module and the classes within are mostly private, though some attributes
are exposed when inspecting mappings.

"""

from __future__ import annotations

import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict

if typing.TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _IdentityKeyType
    from ._typing import _InstanceDict
    from ._typing import _InternalEntityType
    from ._typing import _ORMAdapterProto
    from .attributes import InstrumentedAttribute
    from .base import Mapped
    from .context import _MapperEntity
    from .context import ORMCompileState
    from .context import QueryContext
    from .decl_api import RegistryType
    from .decl_base import _ClassScanMapperConfig
    from .loading import _PopulatorDict
    from .mapper import Mapper
    from .path_registry import AbstractEntityRegistry
    from .query import Query
    from .session import Session
    from .state import InstanceState
    from .strategy_options import _LoadElement
    from .util import AliasedInsp
    from .util import ORMAdapter
    from ..engine.result import Result
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _ColumnsClauseArgument
    from ..sql._typing import _DMLColumnArgument
    from ..sql._typing import _InfoType
    from ..sql.operators import OperatorType
    from ..sql.visitors import _TraverseInternalsType
    from ..util.typing import _AnnotationScanType

_StrategyKey = Tuple[Any, ...]

_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)

_TLS = TypeVar("_TLS", bound="Type[LoaderStrategy]")


class ORMStatementRole(roles.StatementRole):
    __slots__ = ()
    _role_name = (
        "Executable SQL or text() construct, including ORM aware objects"
    )


class ORMColumnsClauseRole(
    roles.ColumnsClauseRole, roles.TypedColumnsClauseRole[_T]
):
    __slots__ = ()
    _role_name = "ORM mapped entity, aliased entity, or Column expression"


class ORMEntityColumnsClauseRole(ORMColumnsClauseRole[_T]):
    __slots__ = ()
    _role_name = "ORM mapped or aliased entity"


class ORMFromClauseRole(roles.StrictFromClauseRole):
    __slots__ = ()
    _role_name = "ORM mapped entity, aliased entity, or FROM expression"


class ORMColumnDescription(TypedDict):
    name: str
    # TODO: add python_type and sql_type here; combining them
    # into "type" is a bad idea
    type: Union[Type[Any], TypeEngine[Any]]
    aliased: bool
    expr: _ColumnsClauseArgument[Any]
    entity: Optional[_ColumnsClauseArgument[Any]]


class _IntrospectsAnnotations:
    __slots__ = ()

    @classmethod
    def _mapper_property_name(cls) -> str:
        return cls.__name__

    def found_in_pep593_annotated(self) -> Any:
        """return a copy of this object to use in declarative when the
        object is found inside of an Annotated object."""

        raise NotImplementedError(
            f"Use of the {self._mapper_property_name()!r} "
            "construct inside of an Annotated object is not yet supported."
        )

    def declarative_scan(
        self,
        decl_scan: _ClassScanMapperConfig,
        registry: RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
        mapped_container: Optional[Type[Mapped[Any]]],
        annotation: Optional[_AnnotationScanType],
        extracted_mapped_annotation: Optional[_AnnotationScanType],
        is_dataclass_field: bool,
    ) -> None:
        """Perform class-specific initializaton at early declarative scanning
        time.

        .. versionadded:: 2.0

        """

    def _raise_for_required(self, key: str, cls: Type[Any]) -> NoReturn:
        raise sa_exc.ArgumentError(
            f"Python typing annotation is required for attribute "
            f'"{cls.__name__}.{key}" when primary argument(s) for '
            f'"{self._mapper_property_name()}" '
            "construct are None or not present"
        )


class _AttributeOptions(NamedTuple):
    """define Python-local attribute behavior options common to all
    :class:`.MapperProperty` objects.

    Currently this includes dataclass-generation arguments.

    .. versionadded:: 2.0

    """

    dataclasses_init: Union[_NoArg, bool]
    dataclasses_repr: Union[_NoArg, bool]
    dataclasses_default: Union[_NoArg, Any]
    dataclasses_default_factory: Union[_NoArg, Callable[[], Any]]
    dataclasses_compare: Union[_NoArg, bool]
    dataclasses_kw_only: Union[_NoArg, bool]
    dataclasses_hash: Union[_NoArg, bool, None]

    def _as_dataclass_field(self, key: str) -> Any:
        """Return a ``dataclasses.Field`` object given these arguments."""

        kw: Dict[str, Any] = {}
        if self.dataclasses_default_factory is not _NoArg.NO_ARG:
            kw["default_factory"] = self.dataclasses_default_factory
        if self.dataclasses_default is not _NoArg.NO_ARG:
            kw["default"] = self.dataclasses_default
        if self.dataclasses_init is not _NoArg.NO_ARG:
            kw["init"] = self.dataclasses_init
        if self.dataclasses_repr is not _NoArg.NO_ARG:
            kw["repr"] = self.dataclasses_repr
        if self.dataclasses_compare is not _NoArg.NO_ARG:
            kw["compare"] = self.dataclasses_compare
        if self.dataclasses_kw_only is not _NoArg.NO_ARG:
            kw["kw_only"] = self.dataclasses_kw_only
        if self.dataclasses_hash is not _NoArg.NO_ARG:
            kw["hash"] = self.dataclasses_hash

        if "default" in kw and callable(kw["default"]):
            # callable defaults are ambiguous. deprecate them in favour of
            # insert_default or default_factory. #9936
            warn_deprecated(
                f"Callable object passed to the ``default`` parameter for "
                f"attribute {key!r} in a ORM-mapped Dataclasses context is "
                "ambiguous, "
                "and this use will raise an error in a future release.  "
                "If this callable is intended to produce Core level INSERT "
                "default values for an underlying ``Column``, use "
                "the ``mapped_column.insert_default`` parameter instead.  "
                "To establish this callable as providing a default value "
                "for instances of the dataclass itself, use the "
                "``default_factory`` dataclasses parameter.",
                "2.0",
            )

        if (
            "init" in kw
            and not kw["init"]
            and "default" in kw
            and not callable(kw["default"])  # ignore callable defaults. #9936
            and "default_factory" not in kw  # illegal but let dc.field raise
        ):
            # fix for #9879
            default = kw.pop("default")
            kw["default_factory"] = lambda: default

        return dataclasses.field(**kw)

    @classmethod
    def _get_arguments_for_make_dataclass(
        cls,
        key: str,
        annotation: _AnnotationScanType,
        mapped_container: Optional[Any],
        elem: _T,
    ) -> Union[
        Tuple[str, _AnnotationScanType],
        Tuple[str, _AnnotationScanType, dataclasses.Field[Any]],
    ]:
        """given attribute key, annotation, and value from a class, return
        the argument tuple we would pass to dataclasses.make_dataclass()
        for this attribute.

        """
        if isinstance(elem, _DCAttributeOptions):
            dc_field = elem._attribute_options._as_dataclass_field(key)

            return (key, annotation, dc_field)
        elif elem is not _NoArg.NO_ARG:
            # why is typing not erroring on this?
            return (key, annotation, elem)
        elif mapped_container is not None:
            # it's Mapped[], but there's no "element", which means declarative
            # did not actually do anything for this field.  this shouldn't
            # happen.
            # previously, this would occur because _scan_attributes would
            # skip a field that's on an already mapped superclass, but it
            # would still include it in the annotations, leading
            # to issue #8718

            assert False, "Mapped[] received without a mapping declaration"

        else:
            # plain dataclass field, not mapped.  Is only possible
            # if __allow_unmapped__ is set up.  I can see this mode causing
            # problems...
            return (key, annotation)


_DEFAULT_ATTRIBUTE_OPTIONS = _AttributeOptions(
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
)

_DEFAULT_READONLY_ATTRIBUTE_OPTIONS = _AttributeOptions(
    False,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
    _NoArg.NO_ARG,
)


class _DCAttributeOptions:
    """mixin for descriptors or configurational objects that include dataclass
    field options.

    This includes :class:`.MapperProperty`, :class:`._MapsColumn` within
    the ORM, but also includes :class:`.AssociationProxy` within ext.
    Can in theory be used for other descriptors that serve a similar role
    as association proxy.   (*maybe* hybrids, not sure yet.)

    """

    __slots__ = ()

    _attribute_options: _AttributeOptions
    """behavioral options for ORM-enabled Python attributes

    .. versionadded:: 2.0

    """

    _has_dataclass_arguments: bool


class _MapsColumns(_DCAttributeOptions, _MappedAttribute[_T]):
    """interface for declarative-capable construct that delivers one or more
    Column objects to the declarative process to be part of a Table.
    """

    __slots__ = ()

    @property
    def mapper_property_to_assign(self) -> Optional[MapperProperty[_T]]:
        """return a MapperProperty to be assigned to the declarative mapping"""
        raise NotImplementedError()

    @property
    def columns_to_assign(self) -> List[Tuple[Column[_T], int]]:
        """A list of Column objects that should be declaratively added to the
        new Table object.

        """
        raise NotImplementedError()


# NOTE: MapperProperty needs to extend _MappedAttribute so that declarative
# typing works, i.e. "Mapped[A] = relationship()".   This introduces an
# inconvenience which is that all the MapperProperty objects are treated
# as descriptors by typing tools, which are misled by this as assignment /
# access to a descriptor attribute wants to move through __get__.
# Therefore, references to MapperProperty as an instance variable, such
# as in PropComparator, may have some special typing workarounds such as the
# use of sqlalchemy.util.typing.DescriptorReference to avoid mis-interpretation
# by typing tools
@inspection._self_inspects
class MapperProperty(
    HasCacheKey,
    _DCAttributeOptions,
    _MappedAttribute[_T],
    InspectionAttrInfo,
    util.MemoizedSlots,
):
    """Represent a particular class attribute mapped by :class:`_orm.Mapper`.

    The most common occurrences of :class:`.MapperProperty` are the
    mapped :class:`_schema.Column`, which is represented in a mapping as
    an instance of :class:`.ColumnProperty`,
    and a reference to another class produced by :func:`_orm.relationship`,
    represented in the mapping as an instance of
    :class:`.Relationship`.

    """

    __slots__ = (
        "_configure_started",
        "_configure_finished",
        "_attribute_options",
        "_has_dataclass_arguments",
        "parent",
        "key",
        "info",
        "doc",
    )

    _cache_key_traversal: _TraverseInternalsType = [
        ("parent", visitors.ExtendedInternalTraversal.dp_has_cache_key),
        ("key", visitors.ExtendedInternalTraversal.dp_string),
    ]

    if not TYPE_CHECKING:
        cascade = None

    is_property = True
    """Part of the InspectionAttr interface; states this object is a
    mapper property.

    """

    comparator: PropComparator[_T]
    """The :class:`_orm.PropComparator` instance that implements SQL
    expression construction on behalf of this mapped attribute."""

    key: str
    """name of class attribute"""

    parent: Mapper[Any]
    """the :class:`.Mapper` managing this property."""

    _is_relationship = False

    _links_to_entity: bool
    """True if this MapperProperty refers to a mapped entity.

    Should only be True for Relationship, False for all others.

    """

    doc: Optional[str]
    """optional documentation string"""

    info: _InfoType
    """Info dictionary associated with the object, allowing user-defined
    data to be associated with this :class:`.InspectionAttr`.

    The dictionary is generated when first accessed.  Alternatively,
    it can be specified as a constructor argument to the
    :func:`.column_property`, :func:`_orm.relationship`, or :func:`.composite`
    functions.

    .. seealso::

        :attr:`.QueryableAttribute.info`

        :attr:`.SchemaItem.info`

    """

    def _memoized_attr_info(self) -> _InfoType:
        """Info dictionary associated with the object, allowing user-defined
        data to be associated with this :class:`.InspectionAttr`.

        The dictionary is generated when first accessed.  Alternatively,
        it can be specified as a constructor argument to the
        :func:`.column_property`, :func:`_orm.relationship`, or
        :func:`.composite`
        functions.

        .. seealso::

            :attr:`.QueryableAttribute.info`

            :attr:`.SchemaItem.info`

        """
        return {}

    def setup(
        self,
        context: ORMCompileState,
        query_entity: _MapperEntity,
        path: AbstractEntityRegistry,
        adapter: Optional[ORMAdapter],
        **kwargs: Any,
    ) -> None:
        """Called by Query for the purposes of constructing a SQL statement.

        Each MapperProperty associated with the target mapper processes the
        statement referenced by the query context, adding columns and/or
        criterion as appropriate.

        """

    def create_row_processor(
        self,
        context: ORMCompileState,
        query_entity: _MapperEntity,
        path: AbstractEntityRegistry,
        mapper: Mapper[Any],
        result: Result[Any],
        adapter: Optional[ORMAdapter],
        populators: _PopulatorDict,
    ) -> None:
        """Produce row processing functions and append to the given
        set of populators lists.

        """

    def cascade_iterator(
        self,
        type_: str,
        state: InstanceState[Any],
        dict_: _InstanceDict,
        visited_states: Set[InstanceState[Any]],
        halt_on: Optional[Callable[[InstanceState[Any]], bool]] = None,
    ) -> Iterator[
        Tuple[object, Mapper[Any], InstanceState[Any], _InstanceDict]
    ]:
        """Iterate through instances related to the given instance for
        a particular 'cascade', starting with this MapperProperty.

        Return an iterator3-tuples (instance, mapper, state).

        Note that the 'cascade' collection on this MapperProperty is
        checked first for the given type before cascade_iterator is called.

        This method typically only applies to Relationship.

        """

        return iter(())

    def set_parent(self, parent: Mapper[Any], init: bool) -> None:
        """Set the parent mapper that references this MapperProperty.

        This method is overridden by some subclasses to perform extra
        setup when the mapper is first known.

        """
        self.parent = parent

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        """Hook called by the Mapper to the property to initiate
        instrumentation of the class attribute managed by this
        MapperProperty.

        The MapperProperty here will typically call out to the
        attributes module to set up an InstrumentedAttribute.

        This step is the first of two steps to set up an InstrumentedAttribute,
        and is called early in the mapper setup process.

        The second step is typically the init_class_attribute step,
        called from StrategizedProperty via the post_instrument_class()
        hook.  This step assigns additional state to the InstrumentedAttribute
        (specifically the "impl") which has been determined after the
        MapperProperty has determined what kind of persistence
        management it needs to do (e.g. scalar, object, collection, etc).

        """

    def __init__(
        self,
        attribute_options: Optional[_AttributeOptions] = None,
        _assume_readonly_dc_attributes: bool = False,
    ) -> None:
        self._configure_started = False
        self._configure_finished = False

        if _assume_readonly_dc_attributes:
            default_attrs = _DEFAULT_READONLY_ATTRIBUTE_OPTIONS
        else:
            default_attrs = _DEFAULT_ATTRIBUTE_OPTIONS

        if attribute_options and attribute_options != default_attrs:
            self._has_dataclass_arguments = True
            self._attribute_options = attribute_options
        else:
            self._has_dataclass_arguments = False
            self._attribute_options = default_attrs

    def init(self) -> None:
        """Called after all mappers are created to assemble
        relationships between mappers and perform other post-mapper-creation
        initialization steps.


        """
        self._configure_started = True
        self.do_init()
        self._configure_finished = True

    @property
    def class_attribute(self) -> InstrumentedAttribute[_T]:
        """Return the class-bound descriptor corresponding to this
        :class:`.MapperProperty`.

        This is basically a ``getattr()`` call::

            return getattr(self.parent.class_, self.key)

        I.e. if this :class:`.MapperProperty` were named ``addresses``,
        and the class to which it is mapped is ``User``, this sequence
        is possible::

            >>> from sqlalchemy import inspect
            >>> mapper = inspect(User)
            >>> addresses_property = mapper.attrs.addresses
            >>> addresses_property.class_attribute is User.addresses
            True
            >>> User.addresses.property is addresses_property
            True


        """

        return getattr(self.parent.class_, self.key)  # type: ignore

    def do_init(self) -> None:
        """Perform subclass-specific initialization post-mapper-creation
        steps.

        This is a template method called by the ``MapperProperty``
        object's init() method.

        """

    def post_instrument_class(self, mapper: Mapper[Any]) -> None:
        """Perform instrumentation adjustments that need to occur
        after init() has completed.

        The given Mapper is the Mapper invoking the operation, which
        may not be the same Mapper as self.parent in an inheritance
        scenario; however, Mapper will always at least be a sub-mapper of
        self.parent.

        This method is typically used by StrategizedProperty, which delegates
        it to LoaderStrategy.init_class_attribute() to perform final setup
        on the class-bound InstrumentedAttribute.

        """

    def merge(
        self,
        session: Session,
        source_state: InstanceState[Any],
        source_dict: _InstanceDict,
        dest_state: InstanceState[Any],
        dest_dict: _InstanceDict,
        load: bool,
        _recursive: Dict[Any, object],
        _resolve_conflict_map: Dict[_IdentityKeyType[Any], object],
    ) -> None:
        """Merge the attribute represented by this ``MapperProperty``
        from source to destination object.

        """

    def __repr__(self) -> str:
        return "<%s at 0x%x; %s>" % (
            self.__class__.__name__,
            id(self),
            getattr(self, "key", "no key"),
        )


@inspection._self_inspects
class PropComparator(SQLORMOperations[_T_co], Generic[_T_co], ColumnOperators):
    r"""Defines SQL operations for ORM mapped attributes.

    SQLAlchemy allows for operators to
    be redefined at both the Core and ORM level.  :class:`.PropComparator`
    is the base class of operator redefinition for ORM-level operations,
    including those of :class:`.ColumnProperty`,
    :class:`.Relationship`, and :class:`.Composite`.

    User-defined subclasses of :class:`.PropComparator` may be created. The
    built-in Python comparison and math operator methods, such as
    :meth:`.operators.ColumnOperators.__eq__`,
    :meth:`.operators.ColumnOperators.__lt__`, and
    :meth:`.operators.ColumnOperators.__add__`, can be overridden to provide
    new operator behavior. The custom :class:`.PropComparator` is passed to
    the :class:`.MapperProperty` instance via the ``comparator_factory``
    argument. In each case,
    the appropriate subclass of :class:`.PropComparator` should be used::

        # definition of custom PropComparator subclasses

        from sqlalchemy.orm.properties import (
            ColumnProperty,
            Composite,
            Relationship,
        )


        class MyColumnComparator(ColumnProperty.Comparator):
            def __eq__(self, other):
                return self.__clause_element__() == other


        class MyRelationshipComparator(Relationship.Comparator):
            def any(self, expression):
                "define the 'any' operation"
                # ...


        class MyCompositeComparator(Composite.Comparator):
            def __gt__(self, other):
                "redefine the 'greater than' operation"

                return sql.and_(
                    *[
                        a > b
                        for a, b in zip(
                            self.__clause_element__().clauses,
                            other.__composite_values__(),
                        )
                    ]
                )


        # application of custom PropComparator subclasses

        from sqlalchemy.orm import column_property, relationship, composite
        from sqlalchemy import Column, String


        class SomeMappedClass(Base):
            some_column = column_property(
                Column("some_column", String),
                comparator_factory=MyColumnComparator,
            )

            some_relationship = relationship(
                SomeOtherClass, comparator_factory=MyRelationshipComparator
            )

            some_composite = composite(
                Column("a", String),
                Column("b", String),
                comparator_factory=MyCompositeComparator,
            )

    Note that for column-level operator redefinition, it's usually
    simpler to define the operators at the Core level, using the
    :attr:`.TypeEngine.comparator_factory` attribute.  See
    :ref:`types_operators` for more detail.

    .. seealso::

        :class:`.ColumnProperty.Comparator`

        :class:`.Relationship.Comparator`

        :class:`.Composite.Comparator`

        :class:`.ColumnOperators`

        :ref:`types_operators`

        :attr:`.TypeEngine.comparator_factory`

    """

    __slots__ = "prop", "_parententity", "_adapt_to_entity"

    __visit_name__ = "orm_prop_comparator"

    _parententity: _InternalEntityType[Any]
    _adapt_to_entity: Optional[AliasedInsp[Any]]
    prop: RODescriptorReference[MapperProperty[_T_co]]

    def __init__(
        self,
        prop: MapperProperty[_T],
        parentmapper: _InternalEntityType[Any],
        adapt_to_entity: Optional[AliasedInsp[Any]] = None,
    ):
        self.prop = prop
        self._parententity = adapt_to_entity or parentmapper
        self._adapt_to_entity = adapt_to_entity

    @util.non_memoized_property
    def property(self) -> MapperProperty[_T_co]:
        """Return the :class:`.MapperProperty` associated with this
        :class:`.PropComparator`.


        Return values here will commonly be instances of
        :class:`.ColumnProperty` or :class:`.Relationship`.


        """
        return self.prop

    def __clause_element__(self) -> roles.ColumnsClauseRole:
        raise NotImplementedError("%r" % self)

    def _bulk_update_tuples(
        self, value: Any
    ) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
        """Receive a SQL expression that represents a value in the SET
        clause of an UPDATE statement.

        Return a tuple that can be passed to a :class:`_expression.Update`
        construct.

        """

        return [(cast("_DMLColumnArgument", self.__clause_element__()), value)]

    def adapt_to_entity(
        self, adapt_to_entity: AliasedInsp[Any]
    ) -> PropComparator[_T_co]:
        """Return a copy of this PropComparator which will use the given
        :class:`.AliasedInsp` to produce corresponding expressions.
        """
        return self.__class__(self.prop, self._parententity, adapt_to_entity)

    @util.ro_non_memoized_property
    def _parentmapper(self) -> Mapper[Any]:
        """legacy; this is renamed to _parententity to be
        compatible with QueryableAttribute."""
        return self._parententity.mapper

    def _criterion_exists(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> ColumnElement[Any]:
        return self.prop.comparator._criterion_exists(criterion, **kwargs)

    @util.ro_non_memoized_property
    def adapter(self) -> Optional[_ORMAdapterProto]:
        """Produce a callable that adapts column expressions
        to suit an aliased version of this comparator.

        """
        if self._adapt_to_entity is None:
            return None
        else:
            return self._adapt_to_entity._orm_adapt_element

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        return self.prop.info

    @staticmethod
    def _any_op(a: Any, b: Any, **kwargs: Any) -> Any:
        return a.any(b, **kwargs)

    @staticmethod
    def _has_op(left: Any, other: Any, **kwargs: Any) -> Any:
        return left.has(other, **kwargs)

    @staticmethod
    def _of_type_op(a: Any, class_: Any) -> Any:
        return a.of_type(class_)

    any_op = cast(operators.OperatorType, _any_op)
    has_op = cast(operators.OperatorType, _has_op)
    of_type_op = cast(operators.OperatorType, _of_type_op)

    if typing.TYPE_CHECKING:

        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnElement[Any]: ...

        def reverse_operate(
            self, op: OperatorType, other: Any, **kwargs: Any
        ) -> ColumnElement[Any]: ...

    def of_type(self, class_: _EntityType[Any]) -> PropComparator[_T_co]:
        r"""Redefine this object in terms of a polymorphic subclass,
        :func:`_orm.with_polymorphic` construct, or :func:`_orm.aliased`
        construct.

        Returns a new PropComparator from which further criterion can be
        evaluated.

        e.g.::

            query.join(Company.employees.of_type(Engineer)).filter(
                Engineer.name == "foo"
            )

        :param \class_: a class or mapper indicating that criterion will be
            against this specific subclass.

        .. seealso::

            :ref:`orm_queryguide_joining_relationships_aliased` - in the
            :ref:`queryguide_toplevel`

            :ref:`inheritance_of_type`

        """

        return self.operate(PropComparator.of_type_op, class_)  # type: ignore

    def and_(
        self, *criteria: _ColumnExpressionArgument[bool]
    ) -> PropComparator[bool]:
        """Add additional criteria to the ON clause that's represented by this
        relationship attribute.

        E.g.::


            stmt = select(User).join(
                User.addresses.and_(Address.email_address != "foo")
            )

            stmt = select(User).options(
                joinedload(User.addresses.and_(Address.email_address != "foo"))
            )

        .. versionadded:: 1.4

        .. seealso::

            :ref:`orm_queryguide_join_on_augmented`

            :ref:`loader_option_criteria`

            :func:`.with_loader_criteria`

        """
        return self.operate(operators.and_, *criteria)  # type: ignore

    def any(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> ColumnElement[bool]:
        r"""Return a SQL expression representing true if this element
        references a member which meets the given criterion.

        The usual implementation of ``any()`` is
        :meth:`.Relationship.Comparator.any`.

        :param criterion: an optional ClauseElement formulated against the
          member class' table or attributes.

        :param \**kwargs: key/value pairs corresponding to member class
          attribute names which will be compared via equality to the
          corresponding values.

        """

        return self.operate(PropComparator.any_op, criterion, **kwargs)

    def has(
        self,
        criterion: Optional[_ColumnExpressionArgument[bool]] = None,
        **kwargs: Any,
    ) -> ColumnElement[bool]:
        r"""Return a SQL expression representing true if this element
        references a member which meets the given criterion.

        The usual implementation of ``has()`` is
        :meth:`.Relationship.Comparator.has`.

        :param criterion: an optional ClauseElement formulated against the
          member class' table or attributes.

        :param \**kwargs: key/value pairs corresponding to member class
          attribute names which will be compared via equality to the
          corresponding values.

        """

        return self.operate(PropComparator.has_op, criterion, **kwargs)


class StrategizedProperty(MapperProperty[_T]):
    """A MapperProperty which uses selectable strategies to affect
    loading behavior.

    There is a single strategy selected by default.  Alternate
    strategies can be selected at Query time through the usage of
    ``StrategizedOption`` objects via the Query.options() method.

    The mechanics of StrategizedProperty are used for every Query
    invocation for every mapped attribute participating in that Query,
    to determine first how the attribute will be rendered in SQL
    and secondly how the attribute will retrieve a value from a result
    row and apply it to a mapped object.  The routines here are very
    performance-critical.

    """

    __slots__ = (
        "_strategies",
        "strategy",
        "_wildcard_token",
        "_default_path_loader_key",
        "strategy_key",
    )
    inherit_cache = True
    strategy_wildcard_key: ClassVar[str]

    strategy_key: _StrategyKey

    _strategies: Dict[_StrategyKey, LoaderStrategy]

    def _memoized_attr__wildcard_token(self) -> Tuple[str]:
        return (
            f"{self.strategy_wildcard_key}:{path_registry._WILDCARD_TOKEN}",
        )

    def _memoized_attr__default_path_loader_key(
        self,
    ) -> Tuple[str, Tuple[str]]:
        return (
            "loader",
            (f"{self.strategy_wildcard_key}:{path_registry._DEFAULT_TOKEN}",),
        )

    def _get_context_loader(
        self, context: ORMCompileState, path: AbstractEntityRegistry
    ) -> Optional[_LoadElement]:
        load: Optional[_LoadElement] = None

        search_path = path[self]

        # search among: exact match, "attr.*", "default" strategy
        # if any.
        for path_key in (
            search_path._loader_key,
            search_path._wildcard_path_loader_key,
            search_path._default_path_loader_key,
        ):
            if path_key in context.attributes:
                load = context.attributes[path_key]
                break

                # note that if strategy_options.Load is placing non-actionable
                # objects in the context like defaultload(), we would
                # need to continue the loop here if we got such an
                # option as below.
                # if load.strategy or load.local_opts:
                #    break

        return load

    def _get_strategy(self, key: _StrategyKey) -> LoaderStrategy:
        try:
            return self._strategies[key]
        except KeyError:
            pass

        # run outside to prevent transfer of exception context
        cls = self._strategy_lookup(self, *key)
        # this previously was setting self._strategies[cls], that's
        # a bad idea; should use strategy key at all times because every
        # strategy has multiple keys at this point
        self._strategies[key] = strategy = cls(self, key)
        return strategy

    def setup(
        self,
        context: ORMCompileState,
        query_entity: _MapperEntity,
        path: AbstractEntityRegistry,
        adapter: Optional[ORMAdapter],
        **kwargs: Any,
    ) -> None:
        loader = self._get_context_loader(context, path)
        if loader and loader.strategy:
            strat = self._get_strategy(loader.strategy)
        else:
            strat = self.strategy
        strat.setup_query(
            context, query_entity, path, loader, adapter, **kwargs
        )

    def create_row_processor(
        self,
        context: ORMCompileState,
        query_entity: _MapperEntity,
        path: AbstractEntityRegistry,
        mapper: Mapper[Any],
        result: Result[Any],
        adapter: Optional[ORMAdapter],
        populators: _PopulatorDict,
    ) -> None:
        loader = self._get_context_loader(context, path)
        if loader and loader.strategy:
            strat = self._get_strategy(loader.strategy)
        else:
            strat = self.strategy
        strat.create_row_processor(
            context,
            query_entity,
            path,
            loader,
            mapper,
            result,
            adapter,
            populators,
        )

    def do_init(self) -> None:
        self._strategies = {}
        self.strategy = self._get_strategy(self.strategy_key)

    def post_instrument_class(self, mapper: Mapper[Any]) -> None:
        if (
            not self.parent.non_primary
            and not mapper.class_manager._attr_has_impl(self.key)
        ):
            self.strategy.init_class_attribute(mapper)

    _all_strategies: collections.defaultdict[
        Type[MapperProperty[Any]], Dict[_StrategyKey, Type[LoaderStrategy]]
    ] = collections.defaultdict(dict)

    @classmethod
    def strategy_for(cls, **kw: Any) -> Callable[[_TLS], _TLS]:
        def decorate(dec_cls: _TLS) -> _TLS:
            # ensure each subclass of the strategy has its
            # own _strategy_keys collection
            if "_strategy_keys" not in dec_cls.__dict__:
                dec_cls._strategy_keys = []
            key = tuple(sorted(kw.items()))
            cls._all_strategies[cls][key] = dec_cls
            dec_cls._strategy_keys.append(key)
            return dec_cls

        return decorate

    @classmethod
    def _strategy_lookup(
        cls, requesting_property: MapperProperty[Any], *key: Any
    ) -> Type[LoaderStrategy]:
        requesting_property.parent._with_polymorphic_mappers

        for prop_cls in cls.__mro__:
            if prop_cls in cls._all_strategies:
                if TYPE_CHECKING:
                    assert issubclass(prop_cls, MapperProperty)
                strategies = cls._all_strategies[prop_cls]
                try:
                    return strategies[key]
                except KeyError:
                    pass

        for property_type, strats in cls._all_strategies.items():
            if key in strats:
                intended_property_type = property_type
                actual_strategy = strats[key]
                break
        else:
            intended_property_type = None
            actual_strategy = None

        raise orm_exc.LoaderStrategyException(
            cls,
            requesting_property,
            intended_property_type,
            actual_strategy,
            key,
        )


class ORMOption(ExecutableOption):
    """Base class for option objects that are passed to ORM queries.

    These options may be consumed by :meth:`.Query.options`,
    :meth:`.Select.options`, or in a more general sense by any
    :meth:`.Executable.options` method.   They are interpreted at
    statement compile time or execution time in modern use.  The
    deprecated :class:`.MapperOption` is consumed at ORM query construction
    time.

    .. versionadded:: 1.4

    """

    __slots__ = ()

    _is_legacy_option = False

    propagate_to_loaders = False
    """if True, indicate this option should be carried along
    to "secondary" SELECT statements that occur for relationship
    lazy loaders as well as attribute load / refresh operations.

    """

    _is_core = False

    _is_user_defined = False

    _is_compile_state = False

    _is_criteria_option = False

    _is_strategy_option = False

    def _adapt_cached_option_to_uncached_option(
        self, context: QueryContext, uncached_opt: ORMOption
    ) -> ORMOption:
        """adapt this option to the "uncached" version of itself in a
        loader strategy context.

        given "self" which is an option from a cached query, as well as the
        corresponding option from the uncached version of the same query,
        return the option we should use in a new query, in the context of a
        loader strategy being asked to load related rows on behalf of that
        cached query, which is assumed to be building a new query based on
        entities passed to us from the cached query.

        Currently this routine chooses between "self" and "uncached" without
        manufacturing anything new. If the option is itself a loader strategy
        option which has a path, that path needs to match to the entities being
        passed to us by the cached query, so the :class:`_orm.Load` subclass
        overrides this to return "self". For all other options, we return the
        uncached form which may have changing state, such as a
        with_loader_criteria() option which will very often have new state.

        This routine could in the future involve
        generating a new option based on both inputs if use cases arise,
        such as if with_loader_criteria() needed to match up to
        ``AliasedClass`` instances given in the parent query.

        However, longer term it might be better to restructure things such that
        ``AliasedClass`` entities are always matched up on their cache key,
        instead of identity, in things like paths and such, so that this whole
        issue of "the uncached option does not match the entities" goes away.
        However this would make ``PathRegistry`` more complicated and difficult
        to debug as well as potentially less performant in that it would be
        hashing enormous cache keys rather than a simple AliasedInsp. UNLESS,
        we could get cache keys overall to be reliably hashed into something
        like an md5 key.

        .. versionadded:: 1.4.41

        """
        if uncached_opt is not None:
            return uncached_opt
        else:
            return self


class CompileStateOption(HasCacheKey, ORMOption):
    """base for :class:`.ORMOption` classes that affect the compilation of
    a SQL query and therefore need to be part of the cache key.

    .. note::  :class:`.CompileStateOption` is generally non-public and
       should not be used as a base class for user-defined options; instead,
       use :class:`.UserDefinedOption`, which is easier to use as it does not
       interact with ORM compilation internals or caching.

    :class:`.CompileStateOption` defines an internal attribute
    ``_is_compile_state=True`` which has the effect of the ORM compilation
    routines for SELECT and other statements will call upon these options when
    a SQL string is being compiled. As such, these classes implement
    :class:`.HasCacheKey` and need to provide robust ``_cache_key_traversal``
    structures.

    The :class:`.CompileStateOption` class is used to implement the ORM
    :class:`.LoaderOption` and :class:`.CriteriaOption` classes.

    .. versionadded:: 1.4.28


    """

    __slots__ = ()

    _is_compile_state = True

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        """Apply a modification to a given :class:`.ORMCompileState`.

        This method is part of the implementation of a particular
        :class:`.CompileStateOption` and is only invoked internally
        when an ORM query is compiled.

        """

    def process_compile_state_replaced_entities(
        self,
        compile_state: ORMCompileState,
        mapper_entities: Sequence[_MapperEntity],
    ) -> None:
        """Apply a modification to a given :class:`.ORMCompileState`,
        given entities that were replaced by with_only_columns() or
        with_entities().

        This method is part of the implementation of a particular
        :class:`.CompileStateOption` and is only invoked internally
        when an ORM query is compiled.

        .. versionadded:: 1.4.19

        """


class LoaderOption(CompileStateOption):
    """Describe a loader modification to an ORM statement at compilation time.

    .. versionadded:: 1.4

    """

    __slots__ = ()

    def process_compile_state_replaced_entities(
        self,
        compile_state: ORMCompileState,
        mapper_entities: Sequence[_MapperEntity],
    ) -> None:
        self.process_compile_state(compile_state)


class CriteriaOption(CompileStateOption):
    """Describe a WHERE criteria modification to an ORM statement at
    compilation time.

    .. versionadded:: 1.4

    """

    __slots__ = ()

    _is_criteria_option = True

    def get_global_criteria(self, attributes: Dict[str, Any]) -> None:
        """update additional entity criteria options in the given
        attributes dictionary.

        """


class UserDefinedOption(ORMOption):
    """Base class for a user-defined option that can be consumed from the
    :meth:`.SessionEvents.do_orm_execute` event hook.

    """

    __slots__ = ("payload",)

    _is_legacy_option = False

    _is_user_defined = True

    propagate_to_loaders = False
    """if True, indicate this option should be carried along
    to "secondary" Query objects produced during lazy loads
    or refresh operations.

    """

    def __init__(self, payload: Optional[Any] = None):
        self.payload = payload


@util.deprecated_cls(
    "1.4",
    "The :class:`.MapperOption class is deprecated and will be removed "
    "in a future release.   For "
    "modifications to queries on a per-execution basis, use the "
    ":class:`.UserDefinedOption` class to establish state within a "
    ":class:`.Query` or other Core statement, then use the "
    ":meth:`.SessionEvents.before_orm_execute` hook to consume them.",
    constructor=None,
)
class MapperOption(ORMOption):
    """Describe a modification to a Query"""

    __slots__ = ()

    _is_legacy_option = True

    propagate_to_loaders = False
    """if True, indicate this option should be carried along
    to "secondary" Query objects produced during lazy loads
    or refresh operations.

    """

    def process_query(self, query: Query[Any]) -> None:
        """Apply a modification to the given :class:`_query.Query`."""

    def process_query_conditionally(self, query: Query[Any]) -> None:
        """same as process_query(), except that this option may not
        apply to the given query.

        This is typically applied during a lazy load or scalar refresh
        operation to propagate options stated in the original Query to the
        new Query being used for the load.  It occurs for those options that
        specify propagate_to_loaders=True.

        """

        self.process_query(query)


class LoaderStrategy:
    """Describe the loading behavior of a StrategizedProperty object.

    The ``LoaderStrategy`` interacts with the querying process in three
    ways:

    * it controls the configuration of the ``InstrumentedAttribute``
      placed on a class to handle the behavior of the attribute.  this
      may involve setting up class-level callable functions to fire
      off a select operation when the attribute is first accessed
      (i.e. a lazy load)

    * it processes the ``QueryContext`` at statement construction time,
      where it can modify the SQL statement that is being produced.
      For example, simple column attributes will add their represented
      column to the list of selected columns, a joined eager loader
      may establish join clauses to add to the statement.

    * It produces "row processor" functions at result fetching time.
      These "row processor" functions populate a particular attribute
      on a particular mapped instance.

    """

    __slots__ = (
        "parent_property",
        "is_class_level",
        "parent",
        "key",
        "strategy_key",
        "strategy_opts",
    )

    _strategy_keys: ClassVar[List[_StrategyKey]]

    def __init__(
        self, parent: MapperProperty[Any], strategy_key: _StrategyKey
    ):
        self.parent_property = parent
        self.is_class_level = False
        self.parent = self.parent_property.parent
        self.key = self.parent_property.key
        self.strategy_key = strategy_key
        self.strategy_opts = dict(strategy_key)

    def init_class_attribute(self, mapper: Mapper[Any]) -> None:
        pass

    def setup_query(
        self,
        compile_state: ORMCompileState,
        query_entity: _MapperEntity,
        path: AbstractEntityRegistry,
        loadopt: Optional[_LoadElement],
        adapter: Optional[ORMAdapter],
        **kwargs: Any,
    ) -> None:
        """Establish column and other state for a given QueryContext.

        This method fulfills the contract specified by MapperProperty.setup().

        StrategizedProperty delegates its setup() method
        directly to this method.

        """

    def create_row_processor(
        self,
        context: ORMCompileState,
        query_entity: _MapperEntity,
        path: AbstractEntityRegistry,
        loadopt: Optional[_LoadElement],
        mapper: Mapper[Any],
        result: Result[Any],
        adapter: Optional[ORMAdapter],
        populators: _PopulatorDict,
    ) -> None:
        """Establish row processing functions for a given QueryContext.

        This method fulfills the contract specified by
        MapperProperty.create_row_processor().

        StrategizedProperty delegates its create_row_processor() method
        directly to this method.

        """

    def __str__(self) -> str:
        return str(self.parent_property)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\agca\modules.py ===
"""
Computations with modules over polynomial rings.

This module implements various classes that encapsulate groebner basis
computations for modules. Most of them should not be instantiated by hand.
Instead, use the constructing routines on objects you already have.

For example, to construct a free module over ``QQ[x, y]``, call
``QQ[x, y].free_module(rank)`` instead of the ``FreeModule`` constructor.
In fact ``FreeModule`` is an abstract base class that should not be
instantiated, the ``free_module`` method instead returns the implementing class
``FreeModulePolyRing``.

In general, the abstract base classes implement most functionality in terms of
a few non-implemented methods. The concrete base classes supply only these
non-implemented methods. They may also supply new implementations of the
convenience methods, for example if there are faster algorithms available.
"""


from copy import copy
from functools import reduce

from sympy.polys.agca.ideals import Ideal
from sympy.polys.domains.field import Field
from sympy.polys.orderings import ProductOrder, monomial_key
from sympy.polys.polyclasses import DMP
from sympy.polys.polyerrors import CoercionFailed
from sympy.core.basic import _aresame
from sympy.utilities.iterables import iterable

# TODO
# - module saturation
# - module quotient/intersection for quotient rings
# - free resoltutions / syzygies
# - finding small/minimal generating sets
# - ...

##########################################################################
## Abstract base classes #################################################
##########################################################################


class Module:
    """
    Abstract base class for modules.

    Do not instantiate - use ring explicit constructors instead:

    >>> from sympy import QQ
    >>> from sympy.abc import x
    >>> QQ.old_poly_ring(x).free_module(2)
    QQ[x]**2

    Attributes:

    - dtype - type of elements
    - ring - containing ring

    Non-implemented methods:

    - submodule
    - quotient_module
    - is_zero
    - is_submodule
    - multiply_ideal

    The method convert likely needs to be changed in subclasses.
    """

    def __init__(self, ring):
        self.ring = ring

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into internal representation of this module.

        If ``M`` is not None, it should be a module containing it.
        """
        if not isinstance(elem, self.dtype):
            raise CoercionFailed
        return elem

    def submodule(self, *gens):
        """Generate a submodule."""
        raise NotImplementedError

    def quotient_module(self, other):
        """Generate a quotient module."""
        raise NotImplementedError

    def __truediv__(self, e):
        if not isinstance(e, Module):
            e = self.submodule(*e)
        return self.quotient_module(e)

    def contains(self, elem):
        """Return True if ``elem`` is an element of this module."""
        try:
            self.convert(elem)
            return True
        except CoercionFailed:
            return False

    def __contains__(self, elem):
        return self.contains(elem)

    def subset(self, other):
        """
        Returns True if ``other`` is is a subset of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.subset([(1, x), (x, 2)])
        True
        >>> F.subset([(1/x, x), (x, 2)])
        False
        """
        return all(self.contains(x) for x in other)

    def __eq__(self, other):
        return self.is_submodule(other) and other.is_submodule(self)

    def __ne__(self, other):
        return not (self == other)

    def is_zero(self):
        """Returns True if ``self`` is a zero module."""
        raise NotImplementedError

    def is_submodule(self, other):
        """Returns True if ``other`` is a submodule of ``self``."""
        raise NotImplementedError

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.
        """
        raise NotImplementedError

    def __mul__(self, e):
        if not isinstance(e, Ideal):
            try:
                e = self.ring.ideal(e)
            except (CoercionFailed, NotImplementedError):
                return NotImplemented
        return self.multiply_ideal(e)

    __rmul__ = __mul__

    def identity_hom(self):
        """Return the identity homomorphism on ``self``."""
        raise NotImplementedError


class ModuleElement:
    """
    Base class for module element wrappers.

    Use this class to wrap primitive data types as module elements. It stores
    a reference to the containing module, and implements all the arithmetic
    operators.

    Attributes:

    - module - containing module
    - data - internal data

    Methods that likely need change in subclasses:

    - add
    - mul
    - div
    - eq
    """

    def __init__(self, module, data):
        self.module = module
        self.data = data

    def add(self, d1, d2):
        """Add data ``d1`` and ``d2``."""
        return d1 + d2

    def mul(self, m, d):
        """Multiply module data ``m`` by coefficient d."""
        return m * d

    def div(self, m, d):
        """Divide module data ``m`` by coefficient d."""
        return m / d

    def eq(self, d1, d2):
        """Return true if d1 and d2 represent the same element."""
        return d1 == d2

    def __add__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.add(self.data, om.data))

    __radd__ = __add__

    def __neg__(self):
        return self.__class__(self.module, self.mul(self.data,
                       self.module.ring.convert(-1)))

    def __sub__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return NotImplemented
        return self.__add__(-om)

    def __rsub__(self, om):
        return (-self).__add__(om)

    def __mul__(self, o):
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.mul(self.data, o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if not isinstance(o, self.module.ring.dtype):
            try:
                o = self.module.ring.convert(o)
            except CoercionFailed:
                return NotImplemented
        return self.__class__(self.module, self.div(self.data, o))

    def __eq__(self, om):
        if not isinstance(om, self.__class__) or om.module != self.module:
            try:
                om = self.module.convert(om)
            except CoercionFailed:
                return False
        return self.eq(self.data, om.data)

    def __ne__(self, om):
        return not self == om

##########################################################################
## Free Modules ##########################################################
##########################################################################


class FreeModuleElement(ModuleElement):
    """Element of a free module. Data stored as a tuple."""

    def add(self, d1, d2):
        return tuple(x + y for x, y in zip(d1, d2))

    def mul(self, d, p):
        return tuple(x * p for x in d)

    def div(self, d, p):
        return tuple(x / p for x in d)

    def __repr__(self):
        from sympy.printing.str import sstr
        data = self.data
        if any(isinstance(x, DMP) for x in data):
            data = [self.module.ring.to_sympy(x) for x in data]
        return '[' + ', '.join(sstr(x) for x in data) + ']'

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, idx):
        return self.data[idx]


class FreeModule(Module):
    """
    Abstract base class for free modules.

    Additional attributes:

    - rank - rank of the free module

    Non-implemented methods:

    - submodule
    """

    dtype = FreeModuleElement

    def __init__(self, ring, rank):
        Module.__init__(self, ring)
        self.rank = rank

    def __repr__(self):
        return repr(self.ring) + "**" + repr(self.rank)

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> F.is_submodule(F)
        True
        >>> F.is_submodule(M)
        True
        >>> M.is_submodule(F)
        False
        """
        if isinstance(other, SubModule):
            return other.container == self
        if isinstance(other, FreeModule):
            return other.ring == self.ring and other.rank == self.rank
        return False

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.convert([1, 0])
        [1, 0]
        """
        if isinstance(elem, FreeModuleElement):
            if elem.module is self:
                return elem
            if elem.module.rank != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self,
                     tuple(self.ring.convert(x, elem.module.ring) for x in elem.data))
        elif iterable(elem):
            tpl = tuple(self.ring.convert(x) for x in elem)
            if len(tpl) != self.rank:
                raise CoercionFailed
            return FreeModuleElement(self, tpl)
        elif _aresame(elem, 0):
            return FreeModuleElement(self, (self.ring.convert(0),)*self.rank)
        else:
            raise CoercionFailed

    def is_zero(self):
        """
        Returns True if ``self`` is a zero module.

        (If, as this implementation assumes, the coefficient ring is not the
        zero ring, then this is equivalent to the rank being zero.)

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(0).is_zero()
        True
        >>> QQ.old_poly_ring(x).free_module(1).is_zero()
        False
        """
        return self.rank == 0

    def basis(self):
        """
        Return a set of basis elements.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(3).basis()
        ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        """
        from sympy.matrices import eye
        M = eye(self.rank)
        return tuple(self.convert(M.row(i)) for i in range(self.rank))

    def quotient_module(self, submodule):
        """
        Return a quotient module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2)
        >>> M.quotient_module(M.submodule([1, x], [x, 2]))
        QQ[x]**2/<[1, x], [x, 2]>

        Or more conicisely, using the overloaded division operator:

        >>> QQ.old_poly_ring(x).free_module(2) / [[1, x], [x, 2]]
        QQ[x]**2/<[1, x], [x, 2]>
        """
        return QuotientModule(self.ring, self, submodule)

    def multiply_ideal(self, other):
        """
        Multiply ``self`` by the ideal ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x)
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.multiply_ideal(I)
        <[x, 0], [0, x]>
        """
        return self.submodule(*self.basis()).multiply_ideal(other)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2
        [0, 1]])
        """
        from sympy.polys.agca.homomorphisms import homomorphism
        return homomorphism(self, self, self.basis())


class FreeModulePolyRing(FreeModule):
    """
    Free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(3)
    >>> F
    QQ[x]**3
    >>> F.contains([x, 1, 0])
    True
    >>> F.contains([1/x, 0, 1])
    False
    """

    def __init__(self, ring, rank):
        from sympy.polys.domains.old_polynomialring import PolynomialRingBase
        FreeModule.__init__(self, ring, rank)
        if not isinstance(ring, PolynomialRingBase):
            raise NotImplementedError('This implementation only works over '
                                      + 'polynomial rings, got %s' % ring)
        if not isinstance(ring.dom, Field):
            raise NotImplementedError('Ground domain must be a field, '
                                      + 'got %s' % ring.dom)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, x + y])
        >>> M
        <[x, x + y]>
        >>> M.contains([2*x, 2*x + 2*y])
        True
        >>> M.contains([x, y])
        False
        """
        return SubModulePolyRing(gens, self, **opts)


class FreeModuleQuotientRing(FreeModule):
    """
    Free module over a quotient ring.

    Do not instantiate this, use the constructor method of the ring instead:

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(3)
    >>> F
    (QQ[x]/<x**2 + 1>)**3

    Attributes

    - quot - the quotient module `R^n / IR^n`, where `R/I` is our ring
    """

    def __init__(self, ring, rank):
        from sympy.polys.domains.quotientring import QuotientRing
        FreeModule.__init__(self, ring, rank)
        if not isinstance(ring, QuotientRing):
            raise NotImplementedError('This implementation only works over '
                             + 'quotient rings, got %s' % ring)
        F = self.ring.ring.free_module(self.rank)
        self.quot = F / (self.ring.base_ideal*F)

    def __repr__(self):
        return "(" + repr(self.ring) + ")" + "**" + repr(self.rank)

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
        >>> M
        <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
        >>> M.contains([y**2, x**2 + x*y])
        True
        >>> M.contains([x, y])
        False
        """
        return SubModuleQuotientRing(gens, self, **opts)

    def lift(self, elem):
        """
        Lift the element ``elem`` of self to the module self.quot.

        Note that self.quot is the same set as self, just as an R-module
        and not as an R/I-module, so this makes sense.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> e
        [1 + <x**2 + 1>, 0 + <x**2 + 1>]
        >>> L = F.quot
        >>> l = F.lift(e)
        >>> l
        [1, 0] + <[x**2 + 1, 0], [0, x**2 + 1]>
        >>> L.contains(l)
        True
        """
        return self.quot.convert([x.data for x in elem])

    def unlift(self, elem):
        """
        Push down an element of self.quot to self.

        This undoes ``lift``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = (QQ.old_poly_ring(x)/[x**2 + 1]).free_module(2)
        >>> e = F.convert([1, 0])
        >>> l = F.lift(e)
        >>> e == l
        False
        >>> e == F.unlift(l)
        True
        """
        return self.convert(elem.data)

##########################################################################
## Submodules and subquotients ###########################################
##########################################################################


class SubModule(Module):
    """
    Base class for submodules.

    Attributes:

    - container - containing module
    - gens - generators (subset of containing module)
    - rank - rank of containing module

    Non-implemented methods:

    - _contains
    - _syzygies
    - _in_terms_of_generators
    - _intersect
    - _module_quotient

    Methods that likely need change in subclasses:

    - reduce_element
    """

    def __init__(self, gens, container):
        Module.__init__(self, container.ring)
        self.gens = tuple(container.convert(x) for x in gens)
        self.container = container
        self.rank = container.rank
        self.ring = container.ring
        self.dtype = container.dtype

    def __repr__(self):
        return "<" + ", ".join(repr(x) for x in self.gens) + ">"

    def _contains(self, other):
        """Implementation of containment.
           Other is guaranteed to be FreeModuleElement."""
        raise NotImplementedError

    def _syzygies(self):
        """Implementation of syzygy computation wrt self generators."""
        raise NotImplementedError

    def _in_terms_of_generators(self, e):
        """Implementation of expression in terms of generators."""
        raise NotImplementedError

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal represantition.

        Mostly called implicitly.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, x])
        >>> M.convert([2, 2*x])
        [2, 2*x]
        """
        if isinstance(elem, self.container.dtype) and elem.module is self:
            return elem
        r = copy(self.container.convert(elem, M))
        r.module = self
        if not self._contains(r):
            raise CoercionFailed
        return r

    def _intersect(self, other):
        """Implementation of intersection.
           Other is guaranteed to be a submodule of same free module."""
        raise NotImplementedError

    def _module_quotient(self, other):
        """Implementation of quotient.
           Other is guaranteed to be a submodule of same free module."""
        raise NotImplementedError

    def intersect(self, other, **options):
        """
        Returns the intersection of ``self`` with submodule ``other``.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> F.submodule([x, x]).intersect(F.submodule([y, y]))
        <[x*y, x*y]>

        Some implementation allow further options to be passed. Currently, to
        only one implemented is ``relations=True``, in which case the function
        will return a triple ``(res, rela, relb)``, where ``res`` is the
        intersection module, and ``rela`` and ``relb`` are lists of coefficient
        vectors, expressing the generators of ``res`` in terms of the
        generators of ``self`` (``rela``) and ``other`` (``relb``).

        >>> F.submodule([x, x]).intersect(F.submodule([y, y]), relations=True)
        (<[x*y, x*y]>, [(DMP_Python([[1, 0]], QQ),)], [(DMP_Python([[1], []], QQ),)])

        The above result says: the intersection module is generated by the
        single element `(-xy, -xy) = -y (x, x) = -x (y, y)`, where
        `(x, x)` and `(y, y)` respectively are the unique generators of
        the two modules being intersected.
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self._intersect(other, **options)

    def module_quotient(self, other, **options):
        r"""
        Returns the module quotient of ``self`` by submodule ``other``.

        That is, if ``self`` is the module `M` and ``other`` is `N`, then
        return the ideal `\{f \in R | fN \subset M\}`.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.abc import x, y
        >>> F = QQ.old_poly_ring(x, y).free_module(2)
        >>> S = F.submodule([x*y, x*y])
        >>> T = F.submodule([x, x])
        >>> S.module_quotient(T)
        <y>

        Some implementations allow further options to be passed. Currently, the
        only one implemented is ``relations=True``, which may only be passed
        if ``other`` is principal. In this case the function
        will return a pair ``(res, rel)`` where ``res`` is the ideal, and
        ``rel`` is a list of coefficient vectors, expressing the generators of
        the ideal, multiplied by the generator of ``other`` in terms of
        generators of ``self``.

        >>> S.module_quotient(T, relations=True)
        (<y>, [[DMP_Python([[1]], QQ)]])

        This means that the quotient ideal is generated by the single element
        `y`, and that `y (x, x) = 1 (xy, xy)`, `(x, x)` and `(xy, xy)` being
        the generators of `T` and `S`, respectively.
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self._module_quotient(other, **options)

    def union(self, other):
        """
        Returns the module generated by the union of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(1)
        >>> M = F.submodule([x**2 + x]) # <x(x+1)>
        >>> N = F.submodule([x**2 - 1]) # <(x-1)(x+1)>
        >>> M.union(N) == F.submodule([x+1])
        True
        """
        if not isinstance(other, SubModule):
            raise TypeError('%s is not a SubModule' % other)
        if other.container != self.container:
            raise ValueError(
                '%s is contained in a different free module' % other)
        return self.__class__(self.gens + other.gens, self.container)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_zero()
        False
        >>> F.submodule([0, 0]).is_zero()
        True
        """
        return all(x == 0 for x in self.gens)

    def submodule(self, *gens):
        """
        Generate a submodule.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([x, 1])
        >>> M.submodule([x**2, x])
        <[x**2, x]>
        """
        if not self.subset(gens):
            raise ValueError('%s not a subset of %s' % (gens, self))
        return self.__class__(gens, self.container)

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return all(self.contains(x) for x in self.container.basis())

    def is_submodule(self, other):
        """
        Returns True if ``other`` is a submodule of ``self``.

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([2, x])
        >>> N = M.submodule([2*x, x**2])
        >>> M.is_submodule(M)
        True
        >>> M.is_submodule(N)
        True
        >>> N.is_submodule(M)
        False
        """
        if isinstance(other, SubModule):
            return self.container == other.container and \
                all(self.contains(x) for x in other.gens)
        if isinstance(other, (FreeModule, QuotientModule)):
            return self.container == other and self.is_full_module()
        return False

    def syzygy_module(self, **opts):
        r"""
        Compute the syzygy module of the generators of ``self``.

        Suppose `M` is generated by `f_1, \ldots, f_n` over the ring
        `R`. Consider the homomorphism `\phi: R^n \to M`, given by
        sending `(r_1, \ldots, r_n) \to r_1 f_1 + \cdots + r_n f_n`.
        The syzygy module is defined to be the kernel of `\phi`.

        Examples
        ========

        The syzygy module is zero iff the generators generate freely a free
        submodule:

        >>> from sympy.abc import x, y
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([1, 0], [1, 1]).syzygy_module().is_zero()
        True

        A slightly more interesting example:

        >>> M = QQ.old_poly_ring(x, y).free_module(2).submodule([x, 2*x], [y, 2*y])
        >>> S = QQ.old_poly_ring(x, y).free_module(2).submodule([y, -x])
        >>> M.syzygy_module() == S
        True
        """
        F = self.ring.free_module(len(self.gens))
        # NOTE we filter out zero syzygies. This is for convenience of the
        # _syzygies function and not meant to replace any real "generating set
        # reduction" algorithm
        return F.submodule(*[x for x in self._syzygies() if F.convert(x) != 0],
                           **opts)

    def in_terms_of_generators(self, e):
        """
        Express element ``e`` of ``self`` in terms of the generators.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> M = F.submodule([1, 0], [1, 1])
        >>> M.in_terms_of_generators([x, x**2])  # doctest: +SKIP
        [DMP_Python([-1, 1, 0], QQ), DMP_Python([1, 0, 0], QQ)]
        """
        try:
            e = self.convert(e)
        except CoercionFailed:
            raise ValueError('%s is not an element of %s' % (e, self))
        return self._in_terms_of_generators(e)

    def reduce_element(self, x):
        """
        Reduce the element ``x`` of our ring modulo the ideal ``self``.

        Here "reduce" has no specific meaning, it could return a unique normal
        form, simplify the expression a bit, or just do nothing.
        """
        return x

    def quotient_module(self, other, **opts):
        """
        Return a quotient module.

        This is the same as taking a submodule of a quotient of the containing
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> S1 = F.submodule([x, 1])
        >>> S2 = F.submodule([x**2, x])
        >>> S1.quotient_module(S2)
        <[x, 1] + <[x**2, x]>>

        Or more coincisely, using the overloaded division operator:

        >>> F.submodule([x, 1]) / [(x**2, x)]
        <[x, 1] + <[x**2, x]>>
        """
        if not self.is_submodule(other):
            raise ValueError('%s not a submodule of %s' % (other, self))
        return SubQuotientModule(self.gens,
                self.container.quotient_module(other), **opts)

    def __add__(self, oth):
        return self.container.quotient_module(self).convert(oth)

    __radd__ = __add__

    def multiply_ideal(self, I):
        """
        Multiply ``self`` by the ideal ``I``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> I = QQ.old_poly_ring(x).ideal(x**2)
        >>> M = QQ.old_poly_ring(x).free_module(2).submodule([1, 1])
        >>> I*M
        <[x**2, x**2]>
        """
        return self.submodule(*[x*g for [x] in I._module.gens for g in self.gens])

    def inclusion_hom(self):
        """
        Return a homomorphism representing the inclusion map of ``self``.

        That is, the natural map from ``self`` to ``self.container``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).inclusion_hom()
        Matrix([
        [1, 0], : <[x, x]> -> QQ[x]**2
        [0, 1]])
        """
        return self.container.identity_hom().restrict_domain(self)

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2).submodule([x, x]).identity_hom()
        Matrix([
        [1, 0], : <[x, x]> -> <[x, x]>
        [0, 1]])
        """
        return self.container.identity_hom().restrict_domain(
            self).restrict_codomain(self)


class SubQuotientModule(SubModule):
    """
    Submodule of a quotient module.

    Equivalently, quotient module of a submodule.

    Do not instantiate this, instead use the submodule or quotient_module
    constructing methods:

    >>> from sympy.abc import x
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x).free_module(2)
    >>> S = F.submodule([1, 0], [1, x])
    >>> Q = F/[(1, 0)]
    >>> S/[(1, 0)] == Q.submodule([5, x])
    True

    Attributes:

    - base - base module we are quotient of
    - killed_module - submodule used to form the quotient
    """
    def __init__(self, gens, container, **opts):
        SubModule.__init__(self, gens, container)
        self.killed_module = self.container.killed_module
        # XXX it is important for some code below that the generators of base
        #     are in this particular order!
        self.base = self.container.base.submodule(
            *[x.data for x in self.gens], **opts).union(self.killed_module)

    def _contains(self, elem):
        return self.base.contains(elem.data)

    def _syzygies(self):
        # let N = self.killed_module be generated by e_1, ..., e_r
        # let F = self.base be generated by f_1, ..., f_s and e_1, ..., e_r
        # Then self = F/N.
        # Let phi: R**s --> self be the evident surjection.
        # Similarly psi: R**(s + r) --> F.
        # We need to find generators for ker(phi). Let chi: R**s --> F be the
        # evident lift of phi. For X in R**s, phi(X) = 0 iff chi(X) is
        # contained in N, iff there exists Y in R**r such that
        # psi(X, Y) = 0.
        # Hence if alpha: R**(s + r) --> R**s is the projection map, then
        # ker(phi) = alpha ker(psi).
        return [X[:len(self.gens)] for X in self.base._syzygies()]

    def _in_terms_of_generators(self, e):
        return self.base._in_terms_of_generators(e.data)[:len(self.gens)]

    def is_full_module(self):
        """
        Return True if ``self`` is the entire free module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> F.submodule([x, 1]).is_full_module()
        False
        >>> F.submodule([1, 1], [1, 2]).is_full_module()
        True
        """
        return self.base.is_full_module()

    def quotient_hom(self):
        """
        Return the quotient homomorphism to self.

        That is, return the natural map from ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = (QQ.old_poly_ring(x).free_module(2) / [(1, x)]).submodule([1, 0])
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : <[1, 0], [1, x]> -> <[1, 0] + <[1, x]>, [1, x] + <[1, x]>>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(self.killed_module)


_subs0 = lambda x: x[0]
_subs1 = lambda x: x[1:]


class ModuleOrder(ProductOrder):
    """A product monomial order with a zeroth term as module index."""

    def __init__(self, o1, o2, TOP):
        if TOP:
            ProductOrder.__init__(self, (o2, _subs1), (o1, _subs0))
        else:
            ProductOrder.__init__(self, (o1, _subs0), (o2, _subs1))


class SubModulePolyRing(SubModule):
    """
    Submodule of a free module over a generalized polynomial ring.

    Do not instantiate this, use the constructor method of FreeModule instead:

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> F = QQ.old_poly_ring(x, y).free_module(2)
    >>> F.submodule([x, y], [1, 0])
    <[x, y], [1, 0]>

    Attributes:

    - order - monomial order used
    """

    #self._gb - cached groebner basis
    #self._gbe - cached groebner basis relations

    def __init__(self, gens, container, order="lex", TOP=True):
        SubModule.__init__(self, gens, container)
        if not isinstance(container, FreeModulePolyRing):
            raise NotImplementedError('This implementation is for submodules of '
                             + 'FreeModulePolyRing, got %s' % container)
        self.order = ModuleOrder(monomial_key(order), self.ring.order, TOP)
        self._gb = None
        self._gbe = None

    def __eq__(self, other):
        if isinstance(other, SubModulePolyRing) and self.order != other.order:
            return False
        return SubModule.__eq__(self, other)

    def _groebner(self, extended=False):
        """Returns a standard basis in sdm form."""
        from sympy.polys.distributedmodules import sdm_groebner, sdm_nf_mora
        if self._gbe is None and extended:
            gb, gbe = sdm_groebner(
                [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
                sdm_nf_mora, self.order, self.ring.dom, extended=True)
            self._gb, self._gbe = tuple(gb), tuple(gbe)
        if self._gb is None:
            self._gb = tuple(sdm_groebner(
                             [self.ring._vector_to_sdm(x, self.order) for x in self.gens],
               sdm_nf_mora, self.order, self.ring.dom))
        if extended:
            return self._gb, self._gbe
        else:
            return self._gb

    def _groebner_vec(self, extended=False):
        """Returns a standard basis in element form."""
        if not extended:
            return [FreeModuleElement(self,
                        tuple(self.ring._sdm_to_vector(x, self.rank)))
                    for x in self._groebner()]
        gb, gbe = self._groebner(extended=True)
        return ([self.convert(self.ring._sdm_to_vector(x, self.rank))
                 for x in gb],
                [self.ring._sdm_to_vector(x, len(self.gens)) for x in gbe])

    def _contains(self, x):
        from sympy.polys.distributedmodules import sdm_zero, sdm_nf_mora
        return sdm_nf_mora(self.ring._vector_to_sdm(x, self.order),
                           self._groebner(), self.order, self.ring.dom) == \
            sdm_zero()

    def _syzygies(self):
        """Compute syzygies. See [SCA, algorithm 2.5.4]."""
        # NOTE if self.gens is a standard basis, this can be done more
        #      efficiently using Schreyer's theorem

        # First bullet point
        k = len(self.gens)
        r = self.rank
        zero = self.ring.convert(0)
        one = self.ring.convert(1)
        Rkr = self.ring.free_module(r + k)
        newgens = []
        for j, f in enumerate(self.gens):
            m = [0]*(r + k)
            for i, v in enumerate(f):
                m[i] = v
            for i in range(k):
                m[r + i] = one if j == i else zero
            m = FreeModuleElement(Rkr, tuple(m))
            newgens.append(m)
        # Note: we need *descending* order on module index, and TOP=False to
        #       get an elimination order
        F = Rkr.submodule(*newgens, order='ilex', TOP=False)

        # Second bullet point: standard basis of F
        G = F._groebner_vec()

        # Third bullet point: G0 = G intersect the new k components
        G0 = [x[r:] for x in G if all(y == zero for y in x[:r])]

        # Fourth and fifth bullet points: we are done
        return G0

    def _in_terms_of_generators(self, e):
        """Expression in terms of generators. See [SCA, 2.8.1]."""
        # NOTE: if gens is a standard basis, this can be done more efficiently
        M = self.ring.free_module(self.rank).submodule(*((e,) + self.gens))
        S = M.syzygy_module(
            order="ilex", TOP=False)  # We want decreasing order!
        G = S._groebner_vec()
        # This list cannot not be empty since e is an element
        e = [x for x in G if self.ring.is_unit(x[0])][0]
        return [-x/e[0] for x in e[1:]]

    def reduce_element(self, x, NF=None):
        """
        Reduce the element ``x`` of our container modulo ``self``.

        This applies the normal form ``NF`` to ``x``. If ``NF`` is passed
        as none, the default Mora normal form is used (which is not unique!).
        """
        from sympy.polys.distributedmodules import sdm_nf_mora
        if NF is None:
            NF = sdm_nf_mora
        return self.container.convert(self.ring._sdm_to_vector(NF(
            self.ring._vector_to_sdm(x, self.order), self._groebner(),
            self.order, self.ring.dom),
            self.rank))

    def _intersect(self, other, relations=False):
        # See: [SCA, section 2.8.2]
        fi = self.gens
        hi = other.gens
        r = self.rank
        ci = [[0]*(2*r) for _ in range(r)]
        for k in range(r):
            ci[k][k] = 1
            ci[k][r + k] = 1
        di = [list(f) + [0]*r for f in fi]
        ei = [[0]*r + list(h) for h in hi]
        syz = self.ring.free_module(2*r).submodule(*(ci + di + ei))._syzygies()
        nonzero = [x for x in syz if any(y != self.ring.zero for y in x[:r])]
        res = self.container.submodule(*([-y for y in x[:r]] for x in nonzero))
        reln1 = [x[r:r + len(fi)] for x in nonzero]
        reln2 = [x[r + len(fi):] for x in nonzero]
        if relations:
            return res, reln1, reln2
        return res

    def _module_quotient(self, other, relations=False):
        # See: [SCA, section 2.8.4]
        if relations and len(other.gens) != 1:
            raise NotImplementedError
        if len(other.gens) == 0:
            return self.ring.ideal(1)
        elif len(other.gens) == 1:
            # We do some trickery. Let f be the (vector!) generating ``other``
            # and f1, .., fn be the (vectors) generating self.
            # Consider the submodule of R^{r+1} generated by (f, 1) and
            # {(fi, 0) | i}. Then the intersection with the last module
            # component yields the quotient.
            g1 = list(other.gens[0]) + [1]
            gi = [list(x) + [0] for x in self.gens]
            # NOTE: We *need* to use an elimination order
            M = self.ring.free_module(self.rank + 1).submodule(*([g1] + gi),
                                            order='ilex', TOP=False)
            if not relations:
                return self.ring.ideal(*[x[-1] for x in M._groebner_vec() if
                                         all(y == self.ring.zero for y in x[:-1])])
            else:
                G, R = M._groebner_vec(extended=True)
                indices = [i for i, x in enumerate(G) if
                           all(y == self.ring.zero for y in x[:-1])]
                return (self.ring.ideal(*[G[i][-1] for i in indices]),
                        [[-x for x in R[i][1:]] for i in indices])
        # For more generators, we use I : <h1, .., hn> = intersection of
        #                                    {I : <hi> | i}
        # TODO this can be done more efficiently
        return reduce(lambda x, y: x.intersect(y),
            (self._module_quotient(self.container.submodule(x)) for x in other.gens))


class SubModuleQuotientRing(SubModule):
    """
    Class for submodules of free modules over quotient rings.

    Do not instantiate this. Instead use the submodule methods.

    >>> from sympy.abc import x, y
    >>> from sympy import QQ
    >>> M = (QQ.old_poly_ring(x, y)/[x**2 - y**2]).free_module(2).submodule([x, x + y])
    >>> M
    <[x + <x**2 - y**2>, x + y + <x**2 - y**2>]>
    >>> M.contains([y**2, x**2 + x*y])
    True
    >>> M.contains([x, y])
    False

    Attributes:

    - quot - the subquotient of `R^n/IR^n` generated by lifts of our generators
    """

    def __init__(self, gens, container):
        SubModule.__init__(self, gens, container)
        self.quot = self.container.quot.submodule(
            *[self.container.lift(x) for x in self.gens])

    def _contains(self, elem):
        return self.quot._contains(self.container.lift(elem))

    def _syzygies(self):
        return [tuple(self.ring.convert(y, self.quot.ring) for y in x)
                for x in self.quot._syzygies()]

    def _in_terms_of_generators(self, elem):
        return [self.ring.convert(x, self.quot.ring) for x in
            self.quot._in_terms_of_generators(self.container.lift(elem))]

##########################################################################
## Quotient Modules ######################################################
##########################################################################


class QuotientModuleElement(ModuleElement):
    """Element of a quotient module."""

    def eq(self, d1, d2):
        """Equality comparison."""
        return self.module.killed_module.contains(d1 - d2)

    def __repr__(self):
        return repr(self.data) + " + " + repr(self.module.killed_module)


class QuotientModule(Module):
    """
    Class for quotient modules.

    Do not instantiate this directly. For subquotients, see the
    SubQuotientModule class.

    Attributes:

    - base - the base module we are a quotient of
    - killed_module - the submodule used to form the quotient
    - rank of the base
    """

    dtype = QuotientModuleElement

    def __init__(self, ring, base, submodule):
        Module.__init__(self, ring)
        if not base.is_submodule(submodule):
            raise ValueError('%s is not a submodule of %s' % (submodule, base))
        self.base = base
        self.killed_module = submodule
        self.rank = base.rank

    def __repr__(self):
        return repr(self.base) + "/" + repr(self.killed_module)

    def is_zero(self):
        """
        Return True if ``self`` is a zero module.

        This happens if and only if the base module is the same as the
        submodule being killed.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2)
        >>> (F/[(1, 0)]).is_zero()
        False
        >>> (F/[(1, 0), (0, 1)]).is_zero()
        True
        """
        return self.base == self.killed_module

    def is_submodule(self, other):
        """
        Return True if ``other`` is a submodule of ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> S = Q.submodule([1, 0])
        >>> Q.is_submodule(S)
        True
        >>> S.is_submodule(Q)
        False
        """
        if isinstance(other, QuotientModule):
            return self.killed_module == other.killed_module and \
                self.base.is_submodule(other.base)
        if isinstance(other, SubQuotientModule):
            return other.container == self
        return False

    def submodule(self, *gens, **opts):
        """
        Generate a submodule.

        This is the same as taking a quotient of a submodule of the base
        module.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> Q = QQ.old_poly_ring(x).free_module(2) / [(x, x)]
        >>> Q.submodule([x, 0])
        <[x, 0] + <[x, x]>>
        """
        return SubQuotientModule(gens, self, **opts)

    def convert(self, elem, M=None):
        """
        Convert ``elem`` into the internal representation.

        This method is called implicitly whenever computations involve elements
        not in the internal representation.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> F = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> F.convert([1, 0])
        [1, 0] + <[1, 2], [1, x]>
        """
        if isinstance(elem, QuotientModuleElement):
            if elem.module is self:
                return elem
            if self.killed_module.is_submodule(elem.module.killed_module):
                return QuotientModuleElement(self, self.base.convert(elem.data))
            raise CoercionFailed
        return QuotientModuleElement(self, self.base.convert(elem))

    def identity_hom(self):
        """
        Return the identity homomorphism on ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.identity_hom()
        Matrix([
        [1, 0], : QQ[x]**2/<[1, 2], [1, x]> -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(
            self.killed_module).quotient_domain(self.killed_module)

    def quotient_hom(self):
        """
        Return the quotient homomorphism to ``self``.

        That is, return a homomorphism representing the natural map from
        ``self.base`` to ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> M = QQ.old_poly_ring(x).free_module(2) / [(1, 2), (1, x)]
        >>> M.quotient_hom()
        Matrix([
        [1, 0], : QQ[x]**2 -> QQ[x]**2/<[1, 2], [1, x]>
        [0, 1]])
        """
        return self.base.identity_hom().quotient_codomain(
            self.killed_module)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\util.py ===
# sql/util.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""High level utilities which build upon other modules here.

"""
from __future__ import annotations

from collections import deque
import copy
from itertools import chain
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import coercions
from . import operators
from . import roles
from . import visitors
from ._typing import is_text_clause
from .annotation import _deep_annotate as _deep_annotate  # noqa: F401
from .annotation import _deep_deannotate as _deep_deannotate  # noqa: F401
from .annotation import _shallow_annotate as _shallow_annotate  # noqa: F401
from .base import _expand_cloned
from .base import _from_objects
from .cache_key import HasCacheKey as HasCacheKey  # noqa: F401
from .ddl import sort_tables as sort_tables  # noqa: F401
from .elements import _find_columns as _find_columns
from .elements import _label_reference
from .elements import _textual_label_reference
from .elements import BindParameter
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Grouping
from .elements import KeyedColumnElement
from .elements import Label
from .elements import NamedColumn
from .elements import Null
from .elements import UnaryExpression
from .schema import Column
from .selectable import Alias
from .selectable import FromClause
from .selectable import FromGrouping
from .selectable import Join
from .selectable import ScalarSelect
from .selectable import SelectBase
from .selectable import TableClause
from .visitors import _ET
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol

if typing.TYPE_CHECKING:
    from ._typing import _EquivalentColumnMap
    from ._typing import _LimitOffsetType
    from ._typing import _TypeEngineArgument
    from .elements import BinaryExpression
    from .elements import TextClause
    from .selectable import _JoinTargetElement
    from .selectable import _SelectIterable
    from .selectable import Selectable
    from .visitors import _TraverseCallableType
    from .visitors import ExternallyTraversible
    from .visitors import ExternalTraversal
    from ..engine.interfaces import _AnyExecuteParams
    from ..engine.interfaces import _AnyMultiExecuteParams
    from ..engine.interfaces import _AnySingleExecuteParams
    from ..engine.interfaces import _CoreSingleExecuteParams
    from ..engine.row import Row

_CE = TypeVar("_CE", bound="ColumnElement[Any]")


def join_condition(
    a: FromClause,
    b: FromClause,
    a_subset: Optional[FromClause] = None,
    consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]] = None,
) -> ColumnElement[bool]:
    """Create a join condition between two tables or selectables.

    e.g.::

        join_condition(tablea, tableb)

    would produce an expression along the lines of::

        tablea.c.id == tableb.c.tablea_id

    The join is determined based on the foreign key relationships
    between the two selectables.   If there are multiple ways
    to join, or no way to join, an error is raised.

    :param a_subset: An optional expression that is a sub-component
        of ``a``.  An attempt will be made to join to just this sub-component
        first before looking at the full ``a`` construct, and if found
        will be successful even if there are other ways to join to ``a``.
        This allows the "right side" of a join to be passed thereby
        providing a "natural join".

    """
    return Join._join_condition(
        a,
        b,
        a_subset=a_subset,
        consider_as_foreign_keys=consider_as_foreign_keys,
    )


def find_join_source(
    clauses: List[FromClause], join_to: FromClause
) -> List[int]:
    """Given a list of FROM clauses and a selectable,
    return the first index and element from the list of
    clauses which can be joined against the selectable.  returns
    None, None if no match is found.

    e.g.::

        clause1 = table1.join(table2)
        clause2 = table4.join(table5)

        join_to = table2.join(table3)

        find_join_source([clause1, clause2], join_to) == clause1

    """

    selectables = list(_from_objects(join_to))
    idx = []
    for i, f in enumerate(clauses):
        for s in selectables:
            if f.is_derived_from(s):
                idx.append(i)
    return idx


def find_left_clause_that_matches_given(
    clauses: Sequence[FromClause], join_from: FromClause
) -> List[int]:
    """Given a list of FROM clauses and a selectable,
    return the indexes from the list of
    clauses which is derived from the selectable.

    """

    selectables = list(_from_objects(join_from))
    liberal_idx = []
    for i, f in enumerate(clauses):
        for s in selectables:
            # basic check, if f is derived from s.
            # this can be joins containing a table, or an aliased table
            # or select statement matching to a table.  This check
            # will match a table to a selectable that is adapted from
            # that table.  With Query, this suits the case where a join
            # is being made to an adapted entity
            if f.is_derived_from(s):
                liberal_idx.append(i)
                break

    # in an extremely small set of use cases, a join is being made where
    # there are multiple FROM clauses where our target table is represented
    # in more than one, such as embedded or similar.   in this case, do
    # another pass where we try to get a more exact match where we aren't
    # looking at adaption relationships.
    if len(liberal_idx) > 1:
        conservative_idx = []
        for idx in liberal_idx:
            f = clauses[idx]
            for s in selectables:
                if set(surface_selectables(f)).intersection(
                    surface_selectables(s)
                ):
                    conservative_idx.append(idx)
                    break
        if conservative_idx:
            return conservative_idx

    return liberal_idx


def find_left_clause_to_join_from(
    clauses: Sequence[FromClause],
    join_to: _JoinTargetElement,
    onclause: Optional[ColumnElement[Any]],
) -> List[int]:
    """Given a list of FROM clauses, a selectable,
    and optional ON clause, return a list of integer indexes from the
    clauses list indicating the clauses that can be joined from.

    The presence of an "onclause" indicates that at least one clause can
    definitely be joined from; if the list of clauses is of length one
    and the onclause is given, returns that index.   If the list of clauses
    is more than length one, and the onclause is given, attempts to locate
    which clauses contain the same columns.

    """
    idx = []
    selectables = set(_from_objects(join_to))

    # if we are given more than one target clause to join
    # from, use the onclause to provide a more specific answer.
    # otherwise, don't try to limit, after all, "ON TRUE" is a valid
    # on clause
    if len(clauses) > 1 and onclause is not None:
        resolve_ambiguity = True
        cols_in_onclause = _find_columns(onclause)
    else:
        resolve_ambiguity = False
        cols_in_onclause = None

    for i, f in enumerate(clauses):
        for s in selectables.difference([f]):
            if resolve_ambiguity:
                assert cols_in_onclause is not None
                if set(f.c).union(s.c).issuperset(cols_in_onclause):
                    idx.append(i)
                    break
            elif onclause is not None or Join._can_join(f, s):
                idx.append(i)
                break

    if len(idx) > 1:
        # this is the same "hide froms" logic from
        # Selectable._get_display_froms
        toremove = set(
            chain(*[_expand_cloned(f._hide_froms) for f in clauses])
        )
        idx = [i for i in idx if clauses[i] not in toremove]

    # onclause was given and none of them resolved, so assume
    # all indexes can match
    if not idx and onclause is not None:
        return list(range(len(clauses)))
    else:
        return idx


def visit_binary_product(
    fn: Callable[
        [BinaryExpression[Any], ColumnElement[Any], ColumnElement[Any]], None
    ],
    expr: ColumnElement[Any],
) -> None:
    """Produce a traversal of the given expression, delivering
    column comparisons to the given function.

    The function is of the form::

        def my_fn(binary, left, right): ...

    For each binary expression located which has a
    comparison operator, the product of "left" and
    "right" will be delivered to that function,
    in terms of that binary.

    Hence an expression like::

        and_((a + b) == q + func.sum(e + f), j == r)

    would have the traversal:

    .. sourcecode:: text

        a <eq> q
        a <eq> e
        a <eq> f
        b <eq> q
        b <eq> e
        b <eq> f
        j <eq> r

    That is, every combination of "left" and
    "right" that doesn't further contain
    a binary comparison is passed as pairs.

    """
    stack: List[BinaryExpression[Any]] = []

    def visit(element: ClauseElement) -> Iterator[ColumnElement[Any]]:
        if isinstance(element, ScalarSelect):
            # we don't want to dig into correlated subqueries,
            # those are just column elements by themselves
            yield element
        elif element.__visit_name__ == "binary" and operators.is_comparison(
            element.operator  # type: ignore
        ):
            stack.insert(0, element)  # type: ignore
            for l in visit(element.left):  # type: ignore
                for r in visit(element.right):  # type: ignore
                    fn(stack[0], l, r)
            stack.pop(0)
            for elem in element.get_children():
                visit(elem)
        else:
            if isinstance(element, ColumnClause):
                yield element
            for elem in element.get_children():
                yield from visit(elem)

    list(visit(expr))
    visit = None  # type: ignore  # remove gc cycles


def find_tables(
    clause: ClauseElement,
    *,
    check_columns: bool = False,
    include_aliases: bool = False,
    include_joins: bool = False,
    include_selects: bool = False,
    include_crud: bool = False,
) -> List[TableClause]:
    """locate Table objects within the given expression."""

    tables: List[TableClause] = []
    _visitors: Dict[str, _TraverseCallableType[Any]] = {}

    if include_selects:
        _visitors["select"] = _visitors["compound_select"] = tables.append

    if include_joins:
        _visitors["join"] = tables.append

    if include_aliases:
        _visitors["alias"] = _visitors["subquery"] = _visitors[
            "tablesample"
        ] = _visitors["lateral"] = tables.append

    if include_crud:
        _visitors["insert"] = _visitors["update"] = _visitors["delete"] = (
            lambda ent: tables.append(ent.table)
        )

    if check_columns:

        def visit_column(column):
            tables.append(column.table)

        _visitors["column"] = visit_column

    _visitors["table"] = tables.append

    visitors.traverse(clause, {}, _visitors)
    return tables


def unwrap_order_by(clause: Any) -> Any:
    """Break up an 'order by' expression into individual column-expressions,
    without DESC/ASC/NULLS FIRST/NULLS LAST"""

    cols = util.column_set()
    result = []
    stack = deque([clause])

    # examples
    # column -> ASC/DESC == column
    # column -> ASC/DESC -> label == column
    # column -> label -> ASC/DESC -> label == column
    # scalar_select -> label -> ASC/DESC == scalar_select -> label

    while stack:
        t = stack.popleft()
        if isinstance(t, ColumnElement) and (
            not isinstance(t, UnaryExpression)
            or not operators.is_ordering_modifier(t.modifier)  # type: ignore
        ):
            if isinstance(t, Label) and not isinstance(
                t.element, ScalarSelect
            ):
                t = t.element

                if isinstance(t, Grouping):
                    t = t.element

                stack.append(t)
                continue
            elif isinstance(t, _label_reference):
                t = t.element

                stack.append(t)
                continue
            if isinstance(t, (_textual_label_reference)):
                continue
            if t not in cols:
                cols.add(t)
                result.append(t)

        else:
            for c in t.get_children():
                stack.append(c)
    return result


def unwrap_label_reference(element):
    def replace(
        element: ExternallyTraversible, **kw: Any
    ) -> Optional[ExternallyTraversible]:
        if isinstance(element, _label_reference):
            return element.element
        elif isinstance(element, _textual_label_reference):
            assert False, "can't unwrap a textual label reference"
        return None

    return visitors.replacement_traverse(element, {}, replace)


def expand_column_list_from_order_by(collist, order_by):
    """Given the columns clause and ORDER BY of a selectable,
    return a list of column expressions that can be added to the collist
    corresponding to the ORDER BY, without repeating those already
    in the collist.

    """
    cols_already_present = {
        col.element if col._order_by_label_element is not None else col
        for col in collist
    }

    to_look_for = list(chain(*[unwrap_order_by(o) for o in order_by]))

    return [col for col in to_look_for if col not in cols_already_present]


def clause_is_present(clause, search):
    """Given a target clause and a second to search within, return True
    if the target is plainly present in the search without any
    subqueries or aliases involved.

    Basically descends through Joins.

    """

    for elem in surface_selectables(search):
        if clause == elem:  # use == here so that Annotated's compare
            return True
    else:
        return False


def tables_from_leftmost(clause: FromClause) -> Iterator[FromClause]:
    if isinstance(clause, Join):
        yield from tables_from_leftmost(clause.left)
        yield from tables_from_leftmost(clause.right)
    elif isinstance(clause, FromGrouping):
        yield from tables_from_leftmost(clause.element)
    else:
        yield clause


def surface_selectables(clause):
    stack = [clause]
    while stack:
        elem = stack.pop()
        yield elem
        if isinstance(elem, Join):
            stack.extend((elem.left, elem.right))
        elif isinstance(elem, FromGrouping):
            stack.append(elem.element)


def surface_selectables_only(clause: ClauseElement) -> Iterator[ClauseElement]:
    stack = [clause]
    while stack:
        elem = stack.pop()
        if isinstance(elem, (TableClause, Alias)):
            yield elem
        if isinstance(elem, Join):
            stack.extend((elem.left, elem.right))
        elif isinstance(elem, FromGrouping):
            stack.append(elem.element)
        elif isinstance(elem, ColumnClause):
            if elem.table is not None:
                stack.append(elem.table)
            else:
                yield elem
        elif elem is not None:
            yield elem


def extract_first_column_annotation(column, annotation_name):
    filter_ = (FromGrouping, SelectBase)

    stack = deque([column])
    while stack:
        elem = stack.popleft()
        if annotation_name in elem._annotations:
            return elem._annotations[annotation_name]
        for sub in elem.get_children():
            if isinstance(sub, filter_):
                continue
            stack.append(sub)
    return None


def selectables_overlap(left: FromClause, right: FromClause) -> bool:
    """Return True if left/right have some overlapping selectable"""

    return bool(
        set(surface_selectables(left)).intersection(surface_selectables(right))
    )


def bind_values(clause):
    """Return an ordered list of "bound" values in the given clause.

    E.g.::

        >>> expr = and_(table.c.foo == 5, table.c.foo == 7)
        >>> bind_values(expr)
        [5, 7]
    """

    v = []

    def visit_bindparam(bind):
        v.append(bind.effective_value)

    visitors.traverse(clause, {}, {"bindparam": visit_bindparam})
    return v


def _quote_ddl_expr(element):
    if isinstance(element, str):
        element = element.replace("'", "''")
        return "'%s'" % element
    else:
        return repr(element)


class _repr_base:
    _LIST: int = 0
    _TUPLE: int = 1
    _DICT: int = 2

    __slots__ = ("max_chars",)

    max_chars: int

    def trunc(self, value: Any) -> str:
        rep = repr(value)
        lenrep = len(rep)
        if lenrep > self.max_chars:
            segment_length = self.max_chars // 2
            rep = (
                rep[0:segment_length]
                + (
                    " ... (%d characters truncated) ... "
                    % (lenrep - self.max_chars)
                )
                + rep[-segment_length:]
            )
        return rep


def _repr_single_value(value):
    rp = _repr_base()
    rp.max_chars = 300
    return rp.trunc(value)


class _repr_row(_repr_base):
    """Provide a string view of a row."""

    __slots__ = ("row",)

    def __init__(self, row: Row[Any], max_chars: int = 300):
        self.row = row
        self.max_chars = max_chars

    def __repr__(self) -> str:
        trunc = self.trunc
        return "(%s%s)" % (
            ", ".join(trunc(value) for value in self.row),
            "," if len(self.row) == 1 else "",
        )


class _long_statement(str):
    def __str__(self) -> str:
        lself = len(self)
        if lself > 500:
            lleft = 250
            lright = 100
            trunc = lself - lleft - lright
            return (
                f"{self[0:lleft]} ... {trunc} "
                f"characters truncated ... {self[-lright:]}"
            )
        else:
            return str.__str__(self)


class _repr_params(_repr_base):
    """Provide a string view of bound parameters.

    Truncates display to a given number of 'multi' parameter sets,
    as well as long values to a given number of characters.

    """

    __slots__ = "params", "batches", "ismulti", "max_params"

    def __init__(
        self,
        params: Optional[_AnyExecuteParams],
        batches: int,
        max_params: int = 100,
        max_chars: int = 300,
        ismulti: Optional[bool] = None,
    ):
        self.params = params
        self.ismulti = ismulti
        self.batches = batches
        self.max_chars = max_chars
        self.max_params = max_params

    def __repr__(self) -> str:
        if self.ismulti is None:
            return self.trunc(self.params)

        if isinstance(self.params, list):
            typ = self._LIST

        elif isinstance(self.params, tuple):
            typ = self._TUPLE
        elif isinstance(self.params, dict):
            typ = self._DICT
        else:
            return self.trunc(self.params)

        if self.ismulti:
            multi_params = cast(
                "_AnyMultiExecuteParams",
                self.params,
            )

            if len(self.params) > self.batches:
                msg = (
                    " ... displaying %i of %i total bound parameter sets ... "
                )
                return " ".join(
                    (
                        self._repr_multi(
                            multi_params[: self.batches - 2],
                            typ,
                        )[0:-1],
                        msg % (self.batches, len(self.params)),
                        self._repr_multi(multi_params[-2:], typ)[1:],
                    )
                )
            else:
                return self._repr_multi(multi_params, typ)
        else:
            return self._repr_params(
                cast(
                    "_AnySingleExecuteParams",
                    self.params,
                ),
                typ,
            )

    def _repr_multi(
        self,
        multi_params: _AnyMultiExecuteParams,
        typ: int,
    ) -> str:
        if multi_params:
            if isinstance(multi_params[0], list):
                elem_type = self._LIST
            elif isinstance(multi_params[0], tuple):
                elem_type = self._TUPLE
            elif isinstance(multi_params[0], dict):
                elem_type = self._DICT
            else:
                assert False, "Unknown parameter type %s" % (
                    type(multi_params[0])
                )

            elements = ", ".join(
                self._repr_params(params, elem_type) for params in multi_params
            )
        else:
            elements = ""

        if typ == self._LIST:
            return "[%s]" % elements
        else:
            return "(%s)" % elements

    def _get_batches(self, params: Iterable[Any]) -> Any:
        lparams = list(params)
        lenparams = len(lparams)
        if lenparams > self.max_params:
            lleft = self.max_params // 2
            return (
                lparams[0:lleft],
                lparams[-lleft:],
                lenparams - self.max_params,
            )
        else:
            return lparams, None, None

    def _repr_params(
        self,
        params: _AnySingleExecuteParams,
        typ: int,
    ) -> str:
        if typ is self._DICT:
            return self._repr_param_dict(
                cast("_CoreSingleExecuteParams", params)
            )
        elif typ is self._TUPLE:
            return self._repr_param_tuple(cast("Sequence[Any]", params))
        else:
            return self._repr_param_list(params)

    def _repr_param_dict(self, params: _CoreSingleExecuteParams) -> str:
        trunc = self.trunc
        (
            items_first_batch,
            items_second_batch,
            trunclen,
        ) = self._get_batches(params.items())

        if items_second_batch:
            text = "{%s" % (
                ", ".join(
                    f"{key!r}: {trunc(value)}"
                    for key, value in items_first_batch
                )
            )
            text += f" ... {trunclen} parameters truncated ... "
            text += "%s}" % (
                ", ".join(
                    f"{key!r}: {trunc(value)}"
                    for key, value in items_second_batch
                )
            )
        else:
            text = "{%s}" % (
                ", ".join(
                    f"{key!r}: {trunc(value)}"
                    for key, value in items_first_batch
                )
            )
        return text

    def _repr_param_tuple(self, params: Sequence[Any]) -> str:
        trunc = self.trunc

        (
            items_first_batch,
            items_second_batch,
            trunclen,
        ) = self._get_batches(params)

        if items_second_batch:
            text = "(%s" % (
                ", ".join(trunc(value) for value in items_first_batch)
            )
            text += f" ... {trunclen} parameters truncated ... "
            text += "%s)" % (
                ", ".join(trunc(value) for value in items_second_batch),
            )
        else:
            text = "(%s%s)" % (
                ", ".join(trunc(value) for value in items_first_batch),
                "," if len(items_first_batch) == 1 else "",
            )
        return text

    def _repr_param_list(self, params: _AnySingleExecuteParams) -> str:
        trunc = self.trunc
        (
            items_first_batch,
            items_second_batch,
            trunclen,
        ) = self._get_batches(params)

        if items_second_batch:
            text = "[%s" % (
                ", ".join(trunc(value) for value in items_first_batch)
            )
            text += f" ... {trunclen} parameters truncated ... "
            text += "%s]" % (
                ", ".join(trunc(value) for value in items_second_batch)
            )
        else:
            text = "[%s]" % (
                ", ".join(trunc(value) for value in items_first_batch)
            )
        return text


def adapt_criterion_to_null(crit: _CE, nulls: Collection[Any]) -> _CE:
    """given criterion containing bind params, convert selected elements
    to IS NULL.

    """

    def visit_binary(binary):
        if (
            isinstance(binary.left, BindParameter)
            and binary.left._identifying_key in nulls
        ):
            # reverse order if the NULL is on the left side
            binary.left = binary.right
            binary.right = Null()
            binary.operator = operators.is_
            binary.negate = operators.is_not
        elif (
            isinstance(binary.right, BindParameter)
            and binary.right._identifying_key in nulls
        ):
            binary.right = Null()
            binary.operator = operators.is_
            binary.negate = operators.is_not

    return visitors.cloned_traverse(crit, {}, {"binary": visit_binary})


def splice_joins(
    left: Optional[FromClause],
    right: Optional[FromClause],
    stop_on: Optional[FromClause] = None,
) -> Optional[FromClause]:
    if left is None:
        return right

    stack: List[Tuple[Optional[FromClause], Optional[Join]]] = [(right, None)]

    adapter = ClauseAdapter(left)
    ret = None
    while stack:
        (right, prevright) = stack.pop()
        if isinstance(right, Join) and right is not stop_on:
            right = right._clone()
            right.onclause = adapter.traverse(right.onclause)
            stack.append((right.left, right))
        else:
            right = adapter.traverse(right)
        if prevright is not None:
            assert right is not None
            prevright.left = right
        if ret is None:
            ret = right

    return ret


@overload
def reduce_columns(
    columns: Iterable[ColumnElement[Any]],
    *clauses: Optional[ClauseElement],
    **kw: bool,
) -> Sequence[ColumnElement[Any]]: ...


@overload
def reduce_columns(
    columns: _SelectIterable,
    *clauses: Optional[ClauseElement],
    **kw: bool,
) -> Sequence[Union[ColumnElement[Any], TextClause]]: ...


def reduce_columns(
    columns: _SelectIterable,
    *clauses: Optional[ClauseElement],
    **kw: bool,
) -> Collection[Union[ColumnElement[Any], TextClause]]:
    r"""given a list of columns, return a 'reduced' set based on natural
    equivalents.

    the set is reduced to the smallest list of columns which have no natural
    equivalent present in the list.  A "natural equivalent" means that two
    columns will ultimately represent the same value because they are related
    by a foreign key.

    \*clauses is an optional list of join clauses which will be traversed
    to further identify columns that are "equivalent".

    \**kw may specify 'ignore_nonexistent_tables' to ignore foreign keys
    whose tables are not yet configured, or columns that aren't yet present.

    This function is primarily used to determine the most minimal "primary
    key" from a selectable, by reducing the set of primary key columns present
    in the selectable to just those that are not repeated.

    """
    ignore_nonexistent_tables = kw.pop("ignore_nonexistent_tables", False)
    only_synonyms = kw.pop("only_synonyms", False)

    column_set = util.OrderedSet(columns)
    cset_no_text: util.OrderedSet[ColumnElement[Any]] = column_set.difference(
        c for c in column_set if is_text_clause(c)  # type: ignore
    )

    omit = util.column_set()
    for col in cset_no_text:
        for fk in chain(*[c.foreign_keys for c in col.proxy_set]):
            for c in cset_no_text:
                if c is col:
                    continue
                try:
                    fk_col = fk.column
                except exc.NoReferencedColumnError:
                    # TODO: add specific coverage here
                    # to test/sql/test_selectable ReduceTest
                    if ignore_nonexistent_tables:
                        continue
                    else:
                        raise
                except exc.NoReferencedTableError:
                    # TODO: add specific coverage here
                    # to test/sql/test_selectable ReduceTest
                    if ignore_nonexistent_tables:
                        continue
                    else:
                        raise
                if fk_col.shares_lineage(c) and (
                    not only_synonyms or c.name == col.name
                ):
                    omit.add(col)
                    break

    if clauses:

        def visit_binary(binary):
            if binary.operator == operators.eq:
                cols = util.column_set(
                    chain(
                        *[c.proxy_set for c in cset_no_text.difference(omit)]
                    )
                )
                if binary.left in cols and binary.right in cols:
                    for c in reversed(cset_no_text):
                        if c.shares_lineage(binary.right) and (
                            not only_synonyms or c.name == binary.left.name
                        ):
                            omit.add(c)
                            break

        for clause in clauses:
            if clause is not None:
                visitors.traverse(clause, {}, {"binary": visit_binary})

    return column_set.difference(omit)


def criterion_as_pairs(
    expression,
    consider_as_foreign_keys=None,
    consider_as_referenced_keys=None,
    any_operator=False,
):
    """traverse an expression and locate binary criterion pairs."""

    if consider_as_foreign_keys and consider_as_referenced_keys:
        raise exc.ArgumentError(
            "Can only specify one of "
            "'consider_as_foreign_keys' or "
            "'consider_as_referenced_keys'"
        )

    def col_is(a, b):
        # return a is b
        return a.compare(b)

    def visit_binary(binary):
        if not any_operator and binary.operator is not operators.eq:
            return
        if not isinstance(binary.left, ColumnElement) or not isinstance(
            binary.right, ColumnElement
        ):
            return

        if consider_as_foreign_keys:
            if binary.left in consider_as_foreign_keys and (
                col_is(binary.right, binary.left)
                or binary.right not in consider_as_foreign_keys
            ):
                pairs.append((binary.right, binary.left))
            elif binary.right in consider_as_foreign_keys and (
                col_is(binary.left, binary.right)
                or binary.left not in consider_as_foreign_keys
            ):
                pairs.append((binary.left, binary.right))
        elif consider_as_referenced_keys:
            if binary.left in consider_as_referenced_keys and (
                col_is(binary.right, binary.left)
                or binary.right not in consider_as_referenced_keys
            ):
                pairs.append((binary.left, binary.right))
            elif binary.right in consider_as_referenced_keys and (
                col_is(binary.left, binary.right)
                or binary.left not in consider_as_referenced_keys
            ):
                pairs.append((binary.right, binary.left))
        else:
            if isinstance(binary.left, Column) and isinstance(
                binary.right, Column
            ):
                if binary.left.references(binary.right):
                    pairs.append((binary.right, binary.left))
                elif binary.right.references(binary.left):
                    pairs.append((binary.left, binary.right))

    pairs: List[Tuple[ColumnElement[Any], ColumnElement[Any]]] = []
    visitors.traverse(expression, {}, {"binary": visit_binary})
    return pairs


class ClauseAdapter(visitors.ReplacingExternalTraversal):
    """Clones and modifies clauses based on column correspondence.

    E.g.::

      table1 = Table(
          "sometable",
          metadata,
          Column("col1", Integer),
          Column("col2", Integer),
      )
      table2 = Table(
          "someothertable",
          metadata,
          Column("col1", Integer),
          Column("col2", Integer),
      )

      condition = table1.c.col1 == table2.c.col1

    make an alias of table1::

      s = table1.alias("foo")

    calling ``ClauseAdapter(s).traverse(condition)`` converts
    condition to read::

      s.c.col1 == table2.c.col1

    """

    __slots__ = (
        "__traverse_options__",
        "selectable",
        "include_fn",
        "exclude_fn",
        "equivalents",
        "adapt_on_names",
        "adapt_from_selectables",
    )

    def __init__(
        self,
        selectable: Selectable,
        equivalents: Optional[_EquivalentColumnMap] = None,
        include_fn: Optional[Callable[[ClauseElement], bool]] = None,
        exclude_fn: Optional[Callable[[ClauseElement], bool]] = None,
        adapt_on_names: bool = False,
        anonymize_labels: bool = False,
        adapt_from_selectables: Optional[AbstractSet[FromClause]] = None,
    ):
        self.__traverse_options__ = {
            "stop_on": [selectable],
            "anonymize_labels": anonymize_labels,
        }
        self.selectable = selectable
        self.include_fn = include_fn
        self.exclude_fn = exclude_fn
        self.equivalents = util.column_dict(equivalents or {})
        self.adapt_on_names = adapt_on_names
        self.adapt_from_selectables = adapt_from_selectables

    if TYPE_CHECKING:

        @overload
        def traverse(self, obj: Literal[None]) -> None: ...

        # note this specializes the ReplacingExternalTraversal.traverse()
        # method to state
        # that we will return the same kind of ExternalTraversal object as
        # we were given.  This is probably not 100% true, such as it's
        # possible for us to swap out Alias for Table at the top level.
        # Ideally there could be overloads specific to ColumnElement and
        # FromClause but Mypy is not accepting those as compatible with
        # the base ReplacingExternalTraversal
        @overload
        def traverse(self, obj: _ET) -> _ET: ...

        def traverse(
            self, obj: Optional[ExternallyTraversible]
        ) -> Optional[ExternallyTraversible]: ...

    def _corresponding_column(
        self, col, require_embedded, _seen=util.EMPTY_SET
    ):
        newcol = self.selectable.corresponding_column(
            col, require_embedded=require_embedded
        )
        if newcol is None and col in self.equivalents and col not in _seen:
            for equiv in self.equivalents[col]:
                newcol = self._corresponding_column(
                    equiv,
                    require_embedded=require_embedded,
                    _seen=_seen.union([col]),
                )
                if newcol is not None:
                    return newcol

        if (
            self.adapt_on_names
            and newcol is None
            and isinstance(col, NamedColumn)
        ):
            newcol = self.selectable.exported_columns.get(col.name)
        return newcol

    @util.preload_module("sqlalchemy.sql.functions")
    def replace(
        self, col: _ET, _include_singleton_constants: bool = False
    ) -> Optional[_ET]:
        functions = util.preloaded.sql_functions

        # TODO: cython candidate

        if self.include_fn and not self.include_fn(col):  # type: ignore
            return None
        elif self.exclude_fn and self.exclude_fn(col):  # type: ignore
            return None

        if isinstance(col, FromClause) and not isinstance(
            col, functions.FunctionElement
        ):
            if self.selectable.is_derived_from(col):
                if self.adapt_from_selectables:
                    for adp in self.adapt_from_selectables:
                        if adp.is_derived_from(col):
                            break
                    else:
                        return None
                return self.selectable  # type: ignore
            elif isinstance(col, Alias) and isinstance(
                col.element, TableClause
            ):
                # we are a SELECT statement and not derived from an alias of a
                # table (which nonetheless may be a table our SELECT derives
                # from), so return the alias to prevent further traversal
                # or
                # we are an alias of a table and we are not derived from an
                # alias of a table (which nonetheless may be the same table
                # as ours) so, same thing
                return col  # type: ignore
            else:
                # other cases where we are a selectable and the element
                # is another join or selectable that contains a table which our
                # selectable derives from, that we want to process
                return None

        elif not isinstance(col, ColumnElement):
            return None
        elif not _include_singleton_constants and col._is_singleton_constant:
            # dont swap out NULL, TRUE, FALSE for a label name
            # in a SQL statement that's being rewritten,
            # leave them as the constant.  This is first noted in #6259,
            # however the logic to check this moved here as of #7154 so that
            # it is made specific to SQL rewriting and not all column
            # correspondence

            return None

        if "adapt_column" in col._annotations:
            col = col._annotations["adapt_column"]

        if TYPE_CHECKING:
            assert isinstance(col, KeyedColumnElement)

        if self.adapt_from_selectables and col not in self.equivalents:
            for adp in self.adapt_from_selectables:
                if adp.c.corresponding_column(col, False) is not None:
                    break
            else:
                return None

        if TYPE_CHECKING:
            assert isinstance(col, KeyedColumnElement)

        return self._corresponding_column(  # type: ignore
            col, require_embedded=True
        )


class _ColumnLookup(Protocol):
    @overload
    def __getitem__(self, key: None) -> None: ...

    @overload
    def __getitem__(self, key: ColumnClause[Any]) -> ColumnClause[Any]: ...

    @overload
    def __getitem__(self, key: ColumnElement[Any]) -> ColumnElement[Any]: ...

    @overload
    def __getitem__(self, key: _ET) -> _ET: ...

    def __getitem__(self, key: Any) -> Any: ...


class ColumnAdapter(ClauseAdapter):
    """Extends ClauseAdapter with extra utility functions.

    Key aspects of ColumnAdapter include:

    * Expressions that are adapted are stored in a persistent
      .columns collection; so that an expression E adapted into
      an expression E1, will return the same object E1 when adapted
      a second time.   This is important in particular for things like
      Label objects that are anonymized, so that the ColumnAdapter can
      be used to present a consistent "adapted" view of things.

    * Exclusion of items from the persistent collection based on
      include/exclude rules, but also independent of hash identity.
      This because "annotated" items all have the same hash identity as their
      parent.

    * "wrapping" capability is added, so that the replacement of an expression
      E can proceed through a series of adapters.  This differs from the
      visitor's "chaining" feature in that the resulting object is passed
      through all replacing functions unconditionally, rather than stopping
      at the first one that returns non-None.

    * An adapt_required option, used by eager loading to indicate that
      We don't trust a result row column that is not translated.
      This is to prevent a column from being interpreted as that
      of the child row in a self-referential scenario, see
      inheritance/test_basic.py->EagerTargetingTest.test_adapt_stringency

    """

    __slots__ = (
        "columns",
        "adapt_required",
        "allow_label_resolve",
        "_wrap",
        "__weakref__",
    )

    columns: _ColumnLookup

    def __init__(
        self,
        selectable: Selectable,
        equivalents: Optional[_EquivalentColumnMap] = None,
        adapt_required: bool = False,
        include_fn: Optional[Callable[[ClauseElement], bool]] = None,
        exclude_fn: Optional[Callable[[ClauseElement], bool]] = None,
        adapt_on_names: bool = False,
        allow_label_resolve: bool = True,
        anonymize_labels: bool = False,
        adapt_from_selectables: Optional[AbstractSet[FromClause]] = None,
    ):
        super().__init__(
            selectable,
            equivalents,
            include_fn=include_fn,
            exclude_fn=exclude_fn,
            adapt_on_names=adapt_on_names,
            anonymize_labels=anonymize_labels,
            adapt_from_selectables=adapt_from_selectables,
        )

        self.columns = util.WeakPopulateDict(self._locate_col)  # type: ignore
        if self.include_fn or self.exclude_fn:
            self.columns = self._IncludeExcludeMapping(self, self.columns)
        self.adapt_required = adapt_required
        self.allow_label_resolve = allow_label_resolve
        self._wrap = None

    class _IncludeExcludeMapping:
        def __init__(self, parent, columns):
            self.parent = parent
            self.columns = columns

        def __getitem__(self, key):
            if (
                self.parent.include_fn and not self.parent.include_fn(key)
            ) or (self.parent.exclude_fn and self.parent.exclude_fn(key)):
                if self.parent._wrap:
                    return self.parent._wrap.columns[key]
                else:
                    return key
            return self.columns[key]

    def wrap(self, adapter):
        ac = copy.copy(self)
        ac._wrap = adapter
        ac.columns = util.WeakPopulateDict(ac._locate_col)  # type: ignore
        if ac.include_fn or ac.exclude_fn:
            ac.columns = self._IncludeExcludeMapping(ac, ac.columns)

        return ac

    @overload
    def traverse(self, obj: Literal[None]) -> None: ...

    @overload
    def traverse(self, obj: _ET) -> _ET: ...

    def traverse(
        self, obj: Optional[ExternallyTraversible]
    ) -> Optional[ExternallyTraversible]:
        return self.columns[obj]

    def chain(self, visitor: ExternalTraversal) -> ColumnAdapter:
        assert isinstance(visitor, ColumnAdapter)

        return super().chain(visitor)

    if TYPE_CHECKING:

        @property
        def visitor_iterator(self) -> Iterator[ColumnAdapter]: ...

    adapt_clause = traverse
    adapt_list = ClauseAdapter.copy_and_process

    def adapt_check_present(
        self, col: ColumnElement[Any]
    ) -> Optional[ColumnElement[Any]]:
        newcol = self.columns[col]

        if newcol is col and self._corresponding_column(col, True) is None:
            return None

        return newcol

    def _locate_col(
        self, col: ColumnElement[Any]
    ) -> Optional[ColumnElement[Any]]:
        # both replace and traverse() are overly complicated for what
        # we are doing here and we would do better to have an inlined
        # version that doesn't build up as much overhead.  the issue is that
        # sometimes the lookup does in fact have to adapt the insides of
        # say a labeled scalar subquery.   However, if the object is an
        # Immutable, i.e. Column objects, we can skip the "clone" /
        # "copy internals" part since those will be no-ops in any case.
        # additionally we want to catch singleton objects null/true/false
        # and make sure they are adapted as well here.

        if col._is_immutable:
            for vis in self.visitor_iterator:
                c = vis.replace(col, _include_singleton_constants=True)
                if c is not None:
                    break
            else:
                c = col
        else:
            c = ClauseAdapter.traverse(self, col)

        if self._wrap:
            c2 = self._wrap._locate_col(c)
            if c2 is not None:
                c = c2

        if self.adapt_required and c is col:
            return None

        # allow_label_resolve is consumed by one case for joined eager loading
        # as part of its logic to prevent its own columns from being affected
        # by .order_by().  Before full typing were applied to the ORM, this
        # logic would set this attribute on the incoming object (which is
        # typically a column, but we have a test for it being a non-column
        # object) if no column were found.  While this seemed to
        # have no negative effects, this adjustment should only occur on the
        # new column which is assumed to be local to an adapted selectable.
        if c is not col:
            c._allow_label_resolve = self.allow_label_resolve

        return c


def _offset_or_limit_clause(
    element: _LimitOffsetType,
    name: Optional[str] = None,
    type_: Optional[_TypeEngineArgument[int]] = None,
) -> ColumnElement[int]:
    """Convert the given value to an "offset or limit" clause.

    This handles incoming integers and converts to an expression; if
    an expression is already given, it is passed through.

    """
    return coercions.expect(
        roles.LimitOffsetRole, element, name=name, type_=type_
    )


def _offset_or_limit_clause_asint_if_possible(
    clause: _LimitOffsetType,
) -> _LimitOffsetType:
    """Return the offset or limit clause as a simple integer if possible,
    else return the clause.

    """
    if clause is None:
        return None
    if hasattr(clause, "_limit_offset_value"):
        value = clause._limit_offset_value
        return util.asint(value)
    else:
        return clause


def _make_slice(
    limit_clause: _LimitOffsetType,
    offset_clause: _LimitOffsetType,
    start: int,
    stop: int,
) -> Tuple[Optional[ColumnElement[int]], Optional[ColumnElement[int]]]:
    """Compute LIMIT/OFFSET in terms of slice start/end"""

    # for calculated limit/offset, try to do the addition of
    # values to offset in Python, however if a SQL clause is present
    # then the addition has to be on the SQL side.

    # TODO: typing is finding a few gaps in here, see if they can be
    # closed up

    if start is not None and stop is not None:
        offset_clause = _offset_or_limit_clause_asint_if_possible(
            offset_clause
        )
        if offset_clause is None:
            offset_clause = 0

        if start != 0:
            offset_clause = offset_clause + start  # type: ignore

        if offset_clause == 0:
            offset_clause = None
        else:
            assert offset_clause is not None
            offset_clause = _offset_or_limit_clause(offset_clause)

        limit_clause = _offset_or_limit_clause(stop - start)

    elif start is None and stop is not None:
        limit_clause = _offset_or_limit_clause(stop)
    elif start is not None and stop is None:
        offset_clause = _offset_or_limit_clause_asint_if_possible(
            offset_clause
        )
        if offset_clause is None:
            offset_clause = 0

        if start != 0:
            offset_clause = offset_clause + start

        if offset_clause == 0:
            offset_clause = None
        else:
            offset_clause = _offset_or_limit_clause(offset_clause)

    return limit_clause, offset_clause

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\qdq_quantizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import TensorProto
from onnx import onnx_pb as onnx_proto

from .base_quantizer import BaseQuantizer, QuantizationParams
from .calibrate import TensorData
from .quant_utils import (
    DEQUANT_OP_NAME,
    ONNX_TYPE_TO_NP_TYPE,
    QUANT_OP_NAME,
    QuantizedValue,
    QuantizedValueType,
    __producer__,
    __version__,
    add_dequant_output_suffix,
    add_dequant_suffix,
    add_quant_input_suffix,
    add_quant_output_suffix,
    add_quant_suffix,
    compute_data_quant_params,
    compute_scale_zp,
    compute_scale_zp_float8,
    find_by_name,
    get_qmin_qmax_for_qType,
    ms_domain,
    normalize_axis,
    quantize_onnx_initializer,
    tensor_proto_to_array,
)
from .registry import CreateQDQQuantizer


class QDQQuantTensorType(Enum):
    ACTIVATION = 0
    WEIGHT = 1
    BIAS = 2


# Holds the name of the node input from which a node output will share the
# same quantization param initializers (zero-point and scale initializers).
# Ex: A Transpose node's output will use the same quant param initializers used at the input.
@dataclass
class QDQQuantParamProvider:
    input_name: str
    node_name: str


# Holds information for tensors that have been marked for quantization by operator quantizers.
# Does not hold information for bias tensors.
class QDQTensorQuantInfo:
    def __init__(self, tensor_type=QDQQuantTensorType.ACTIVATION, quant_para_provider=None, axis=None, data_type=None):
        self.tensor_type = tensor_type
        self.quant_para_provider = quant_para_provider
        self.axis = axis
        self.is_shared = quant_para_provider is not None
        assert data_type is not None
        self.data_type = data_type


# Holds information for bias tensors that have been marked for quantization by operator quantizers.
@dataclass
class QDQBiasQuantInfo:
    node_name: str
    input_name: str
    weight_name: str
    beta: float


# Holds quantization parameter values (scale, zp) for a tensor.
# A tensor typically has a one set of quantization parameters, unless the tensor is
# at a "mixed-precision" boundary where the activation quantization type changes (e.g., from uint8 to uint16).
@dataclass
class QDQTensorQuantParams:
    original: QuantizationParams  # Generated by producer node.
    converted: QuantizationParams | None  # Converted type consumed by some (or all/none) consumer nodes.
    converted_recv_nodes: set[str] | None  # The name of nodes that consume the converted type.

    def get_for_consumer(self, consumer_node_name) -> QuantizationParams:
        if self.converted is None:  # Quantized value is not converted, return original
            return self.original

        if self.converted_recv_nodes is None:  # All consumers receive the converted value
            return self.converted

        # Check if consumer node name is in the list of nodes that
        # receive the converted quantization value. If not, return the original value generated
        # by the tensor's producer.
        return self.converted if (consumer_node_name in self.converted_recv_nodes) else self.original


# Holds scale and zero_point initializer TensorProtos.
@dataclass
class QDQScaleZpInitializers:
    scale: TensorProto
    zero_point: TensorProto


# Holds all scale and zero-point initializers for a tensor.
# A tensor typically has a one set of quantization parameters, unless the tensor is
# at a "mixed-precision" boundary where the activation quantization type changes (e.g., from uint8 to uint16).
@dataclass
class QDQTensorScaleZpInitializers:
    original: QDQScaleZpInitializers
    converted: QDQScaleZpInitializers | None
    converted_recv_nodes: set[str] | None


# Holds cached information of a tensor's quantized values (types, zp/scale initializer names, etc.).
# A tensor typically has a one set of quantization parameters, unless the tensor is
# at a "mixed-precision" boundary where the activation quantization type changes (e.g., from uint8 to uint16).
@dataclass
class QDQTensorQuantizedValue:
    original: QuantizedValue
    converted: QuantizedValue | None
    converted_recv_nodes: set[str] | None

    def get_for_consumer(self, consumer_node_name) -> QuantizedValue:
        if self.converted is None:  # Quantized value is not converted, return original
            return self.original

        if self.converted_recv_nodes is None:  # All consumers receive the converted value
            return self.converted

        # Check if consumer node name is in the list of nodes that
        # receive the converted quantization value. If not, return the original value generated
        # by the tensor's producer.
        return self.converted if (consumer_node_name in self.converted_recv_nodes) else self.original


class QDQQuantizer(BaseQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        BaseQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize: dict[str, QDQTensorQuantInfo] = {}
        self.bias_to_quantize: dict[str, QDQBiasQuantInfo] = {}

        self.nodes_to_remove = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = extra_options.get("OpTypesToExcludeOutputQuantization", [])

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = extra_options.get("AddQDQPairToWeight", False)

        # Some scenarios do not need the bias quantized. For example, in the case of Quantization Aware Training,
        # quantizing the bias is not needed. This is because in QAT, all model parameters are expected to be in
        # floating point format. To that end, we can use the FakeQuant operator for weights and activations that
        # can always have QDQ pairs (by using AddQDQPairToWeight). But for biases in a quantized model, we can't use
        # FakeQuant because it only ever appears before a DQ (since it is quantized as int32).
        self.quantize_bias = extra_options.get("QuantizeBias", True)

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can`t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = extra_options.get("DedicatedQDQPair", False)
        self.tensor_to_its_receiving_nodes: dict[str, list[onnx.NodeProto]] = {}

        # Maps a tensor to the DequantizeLinear node (in the original input model) that outputs the tensor.
        # Populated for input models with some pre-quantized weights (typically via a different tool).
        self.tensor_to_producing_dq: dict[str, onnx.NodeProto] = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = extra_options.get("QDQOpTypePerChannelSupportToAxis", {})

        self.qdq_op_domain = ms_domain if extra_options.get("UseQDQContribOps", False) else None

        # User can specify if removable activations, like Clip/Relu, should be kept in the graph.
        # Used in the QDQRemovableActivation class.
        self.qdq_keep_removable_activations = extra_options.get("QDQKeepRemovableActivations", False)

        # Let user disable adjustment of weight scales for bias inputs that are quantized to int32.
        self.qdq_disable_weight_adjust_for_int32_bias = extra_options.get("QDQDisableWeightAdjustForInt32Bias", False)

        # The ONNX spec did not support 16-bit Q/DQ ops before opset 21.
        # So, may have to override the Q/DQ op domain to 'com.microsoft' if the activation or weight types
        # are 16-bit or 4-bit integers.
        if self.opset_version < 21:
            opset21_types = (TensorProto.UINT16, TensorProto.INT16, TensorProto.UINT4, TensorProto.INT4)
            overrides_have_opset21_types = any(
                t.tensor_type in opset21_types for t in self.tensor_quant_override_qtypes
            )
            if not self.qdq_op_domain and (
                self.activation_qType in opset21_types
                or self.weight_qType in opset21_types
                or overrides_have_opset21_types
            ):
                logging.warning(
                    "ONNX QuantizeLinear and DequantizeLinear operators do not support "
                    "16-bit/4-bit integer quantization types prior to opset 21. "
                    f"The domain of QuantizeLinear and DequantizeLinear operators will be set to '{ms_domain}' to "
                    "enable support."
                )
                self.qdq_op_domain = ms_domain

        self.quantization_params = self.calc_graph_quant_params()
        self.initializer_quant_params: dict[str, QuantizationParams] = {}

        # Map of all original value names to quantized value names
        self.quantized_value_map = {}

    def _get_tensor_type(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            return weight.data_type
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type"):
                return vi.type.tensor_type.elem_type
        return None

    def _is_tensor_quantizable(self, tensor_name):
        """
        Check if tensor can be quantized
        """
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight is not None:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                return True
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                TensorProto.FLOAT,
                TensorProto.FLOAT16,
            ):
                return True
        else:
            logging.warning(
                f"failed to infer the type of tensor: {tensor_name}. Skip to quantize it. Please check if it is expected."
            )

        return False

    def __quantize_tensor(self, tensor_name, quant_sharing_provider=None, tensor_type=QDQQuantTensorType.ACTIVATION):
        """
        Adds a tensor to the list (actually a dict) of tensors to quantize. Called indirectly by op quantizers that
        want to quantize a tensor (i.e., "mark" a tensor for quantization).

        If quant_sharing_provider is not None, tensor with name tensor_name will be quantized with the same
        quantization parameters as the node input specified in quant_sharing_provider. Ex: A Tranpose node's output
        will typically use the same quantization parameter initializers used at the Transpose node's input.

        Args:
            tensor_name: name of the tensor to quantize
            quant_sharing_provider: name of the tensor and node that provides quantization parameter
            tensor_type: QDQQuantTensorType default ACTIVATION
        """
        if self._is_tensor_quantizable(tensor_name):
            if quant_sharing_provider:
                if not isinstance(quant_sharing_provider, QDQQuantParamProvider):
                    raise TypeError(
                        f"quant_sharing_provider must be of type QDQQuantParamProvider, not {type(quant_sharing_provider)}."
                    )

                data_type = self._get_tensor_type(tensor_name)
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=tensor_type, quant_para_provider=quant_sharing_provider, data_type=data_type
                )
            elif tensor_name not in self.tensors_to_quantize:
                data_type = self._get_tensor_type(tensor_name)
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(tensor_type=tensor_type, data_type=data_type)

    def quantize_activation_tensor(self, tensor_name: str):
        """
        Adds a tensor to the list of tensors to quantize. Called by op quantizers that
        want to quantize a tensor (i.e., "mark" a tensor for quantization).

        Args:
            tensor_name: name of the tensor to quantize
        """
        return self.__quantize_tensor(tensor_name, None, QDQQuantTensorType.ACTIVATION)

    def quantize_output_same_as_input(self, output_name: str, input_name: str, node_name: str):
        """
        Adds a tensor to the list of tensors to quantize. Called by op quantizers that
        want to quantize an output tensor using the same quantization parameters as one of the node's inputs.

        Ex: A Tranpose node's output will typically use the same quantization parameter initializers used at
        the Transpose node's input.

        Args:
            output_name: name of the node output to quantize so that it uses the same quantization params as an input.
            input_name: name of the node input from which the output tensor will get its quantization params.
            node_name: name of the node that consumes `input_name`.
        """
        return self.__quantize_tensor(
            output_name, QDQQuantParamProvider(input_name, node_name), QDQQuantTensorType.ACTIVATION
        )

    def quantize_weight_tensor(self, tensor_name: str):
        """
        Adds a tensor to the list of weight tensors to quantize. Called by op quantizers that
        want to quantize a weight (i.e., "mark" a weight for quantization).

        Args:
            tensor_name: name of the weight to quantize
        """
        return self.__quantize_tensor(tensor_name, None, QDQQuantTensorType.WEIGHT)

    def quantize_weight_tensor_per_channel(self, tensor_name, axis):
        weight = find_by_name(tensor_name, self.model.initializer())
        if weight:
            if weight.data_type in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
                self.tensors_to_quantize[tensor_name] = QDQTensorQuantInfo(
                    tensor_type=QDQQuantTensorType.WEIGHT, axis=axis, data_type=weight.data_type
                )
        else:
            logging.warning(f"only support per-channel quantization on weight. Tensor: {tensor_name} is not quantized.")

    def _dup_initializer(self, initializer: onnx.TensorProto) -> onnx.TensorProto:
        """
        Duplicates an existing initializer and adds it to the model. Returns the new initializer.
        """
        name_suffix: int = self.model.get_largest_initializer_name_suffix(initializer.name) + 1
        new_initializer_name = f"{initializer.name}{name_suffix}"
        new_initializer = onnx.TensorProto()
        new_initializer.CopyFrom(initializer)
        new_initializer.name = new_initializer_name
        self.model.add_initializer(new_initializer)
        return new_initializer

    def quantize_bias_tensor(self, node_name, bias_name, input_name, weight_name, beta=1.0):
        """
        Adds a bias tensor to the list of bias tensors to quantize. Called by op quantizers that
        want to quantize a bias with bias_zero_point = 0 and bias_scale = input_scale * weight_scale * beta.
        TODO: Explain the reasoning for using this formula.

        Args:
            node_name: name of the node that consumes the bias, input, and weight tensors.
            bias_name: name of the bias tensor to quantize.
            input_name: name of the input tensor whose scale is used to compute the bias's scale.
            weight_name: name of the weight tensor whose scale is used to compute the bias's scale.
            beta: Multiplier used to compute the bias's scale.
        """
        # If the user provided quantization overrides for this tensor, treat it as a regular weight.
        if self.tensor_quant_overrides.get(bias_name):
            logging.info(
                f"Quantizing bias tensor '{bias_name}' as a weight due to the presence of user-specified overrides"
            )
            is_per_channel, axis = self.is_tensor_per_channel(bias_name, default_axis=0)
            if is_per_channel:
                self.quantize_weight_tensor_per_channel(bias_name, axis)
            else:
                self.quantize_weight_tensor(bias_name)
            return

        bias_initializer = find_by_name(bias_name, self.model.initializer())
        if bias_initializer is None:
            logging.warning(f"Expected bias '{bias_name}' to be an initializer")
            return

        if bias_initializer.data_type not in (onnx_proto.TensorProto.FLOAT, onnx_proto.TensorProto.FLOAT16):
            logging.info(f"Expected bias '{bias_name}' to be an floating-point initializer")
            return

        actual_bias_name = bias_name
        if bias_name in self.bias_to_quantize:
            # This bias input is consumed by two different nodes. We need to duplicate the bias so that
            # each node has its own bias input. This is necessary because the bias's scale is computed
            # from the node's other input scales.
            new_bias_initializer = self._dup_initializer(bias_initializer)
            actual_bias_name = new_bias_initializer.name

            # Replace this node's bias input
            self.model.replace_input_of_nodes(bias_name, actual_bias_name, {node_name})
            logging.info(f"Created a copy of bias input '{bias_name}' called '{actual_bias_name}'")

        # Add this to our list of biases to quantize.
        self.bias_to_quantize[actual_bias_name] = QDQBiasQuantInfo(node_name, input_name, weight_name, beta)

    def _adjust_weight_scale_for_int32_bias(
        self,
        input_scale: np.ndarray,
        weight_scale: np.ndarray,
        weight_name: str,
        bias_tp: onnx.TensorProto,
        is_per_channel: bool,
    ) -> tuple[bool, np.ndarray | None]:
        """
        Checks if the bias scale (input_scale * weight_scale) that we intend to use is too small.
        A bias scale that is too small leads to quantized bias values that fall outside the range of a int32 and have to
        be clipped, which decreases accuracy. If this function detects such a scenario, the weight_scale value will be
        increased to prevent this from happening.

        Although the adjustment method and amount differs, the idea to adjust the weight's scale came from the following
        reference:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/optimize/quantization_utils.cc#L252

        :param input_scale: The input's scale.
        :param weight_scale: The weight scale to potentially adjust.
        :param weight_name: The weight initializer's name. Used for logging.
        :param bias_tp: The bias ONNX initializer.
        :param is_per_channel: True if the bias and weight are quantized per-channel.
        :return: A tuple with a bool indicating if the weight's scale was adjusted and the new weight scale.
        """
        if not weight_scale.size:
            return False, None

        bias_float_data = tensor_proto_to_array(bias_tp)

        int32_info = np.iinfo(np.int32)
        multiplicative_epsilon = 1.0001
        qrange = np.array(int32_info.max, dtype=np.float64) - np.array(int32_info.min + 1, dtype=np.float64)
        weight_scale_dtype = weight_scale.dtype
        updated_an_elem = False

        if not is_per_channel:
            rmin = np.minimum(bias_float_data.min(), np.array(0, dtype=np.float64))
            rmax = np.maximum(bias_float_data.max(), np.array(0, dtype=np.float64))
            absmax = np.maximum(np.abs(rmin), np.abs(rmax))
            bias_smallest_valid_scale = multiplicative_epsilon * (2.0 * absmax) / qrange

            input_scale_fp64 = np.array(input_scale.item(), dtype=np.float64)
            weight_scale_fp64 = np.array(weight_scale.item(), dtype=np.float64)
            bias_candidate_scale = input_scale_fp64 * weight_scale_fp64

            if (bias_candidate_scale < bias_smallest_valid_scale) and (bias_candidate_scale > 0.0):
                # The candidate bias scale would be too small, so increase the weight_scale by the necessary ratio.
                ratio = bias_smallest_valid_scale / bias_candidate_scale
                logging.info(
                    f"Increasing scale for weight `{weight_name}` by the ratio {ratio} to "
                    f"ensure bias input `{bias_tp.name}` has a valid scale."
                )
                new_scale = weight_scale_fp64 * ratio
                weight_scale = new_scale.astype(weight_scale_dtype)
                updated_an_elem = True
        elif weight_scale.shape and len(weight_scale.shape) == 1:
            # per-channel case
            num_elems = weight_scale.shape[0]

            for i in range(num_elems):
                bias_rmax = np.abs(bias_float_data[i])
                bias_smallest_valid_scale = multiplicative_epsilon * (2.0 * bias_rmax) / qrange

                input_scale_fp64 = np.array(input_scale.item(), dtype=np.float64)
                weight_scale_fp64 = np.array(weight_scale[i].item(), dtype=np.float64)
                bias_candidate_scale = input_scale_fp64 * weight_scale_fp64
                if (bias_candidate_scale < bias_smallest_valid_scale) and (bias_candidate_scale > 0.0):
                    # The candidate bias scale would be too small, so increase the weight_scale by the necessary ratio.
                    ratio = bias_smallest_valid_scale / bias_candidate_scale
                    logging.info(
                        f"Increased scale[{i}] for weight `{weight_name}` by ratio {ratio} "
                        f"to ensure bias input `{bias_tp.name}` has a valid scale."
                    )
                    new_scale = weight_scale_fp64 * ratio
                    weight_scale[i] = new_scale.astype(weight_scale_dtype)
                    updated_an_elem = True

        return updated_an_elem, weight_scale

    def _adjust_weight_quant_params_for_bias_tensors(self):
        """
        Iterates through all bias inputs that should be quantized to int32. If the intended
        bias scale (equal to input_scale * weight_scale) is too small, this function will increase
        the associated weight's scale to ensure the bias does not overflow the int32 range when quantized.
        """

        if self.qdq_disable_weight_adjust_for_int32_bias:
            # User passed an extra_option to disable this adjustment.
            return

        for bias_name, bias_info in self.bias_to_quantize.items():
            if (
                bias_info.input_name not in self.quantization_params
                or bias_info.input_name not in self.tensors_to_quantize
                or bias_info.weight_name not in self.initializer_quant_params
            ):
                continue

            # Get the associated input's scale.
            input_qparams = self.quantization_params[bias_info.input_name].get_for_consumer(bias_info.node_name)
            input_info = self.tensors_to_quantize[bias_info.input_name]
            input_scale = np.asarray(
                input_qparams["scale"], dtype=onnx.helper.tensor_dtype_to_np_dtype(input_info.data_type)
            )

            weight_quant_params = self.initializer_quant_params[bias_info.weight_name]
            weight_quant_type = weight_quant_params["quant_type"]
            if weight_quant_type not in (onnx.TensorProto.INT8, onnx.TensorProto.INT16):
                continue

            weight_zero_point: np.ndarray = weight_quant_params["zero_point"]
            if weight_zero_point.any():
                # Skip if zero_point(s) are not all zero (i.e., symmetric quant)
                continue

            weight_scale: np.ndarray = weight_quant_params["scale"]
            is_per_channel = weight_quant_params.get("axis", None) is not None

            # Get adjusted weight scales.
            did_update_weight_scale, new_weight_scale = self._adjust_weight_scale_for_int32_bias(
                input_scale,
                weight_scale,
                bias_info.weight_name,
                find_by_name(bias_name, self.model.initializer()),
                is_per_channel,
            )

            if did_update_weight_scale:
                weight_quant_params["scale"] = new_weight_scale

    def remove_node(self, node):
        self.nodes_to_remove.append(node)

    def remove_nodes(self):
        self.model.remove_nodes(self.nodes_to_remove)

    def quantize_model(self):
        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                for tensor_name in node.input:
                    if tensor_name not in self.tensor_to_its_receiving_nodes:
                        self.tensor_to_its_receiving_nodes[tensor_name] = []
                    self.tensor_to_its_receiving_nodes[tensor_name].append(node)
            if node.op_type == DEQUANT_OP_NAME:
                for tensor_name in node.output:
                    self.tensor_to_producing_dq[tensor_name] = node

        self.initializer_quant_params = self._calc_initializer_quant_params()
        self._adjust_weight_quant_params_for_bias_tensors()
        self._quantize_normal_tensors()
        self._quantize_sharing_param_tensors()
        if self.quantize_bias:
            self._quantize_bias_tensors()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        self.model.model.producer_name = __producer__
        self.model.model.producer_version = __version__
        if self.qdq_op_domain == ms_domain:
            self.model.set_opset_import(ms_domain, 1)

        return self.model.model

    def try_replacing_upstream_output(self, upstream_output_name, output_name):
        if (
            output_name in self.quantization_params
            and self.quantization_params[output_name].converted is None
            and self.quantization_params[upstream_output_name].converted is None
            and len(self.model.input_name_to_nodes()[upstream_output_name]) == 1
            and not self.model.is_graph_output(upstream_output_name)
            and not self.model.is_graph_input(upstream_output_name)
        ):
            self.model.replace_output_of_all_nodes(upstream_output_name, output_name)
            if upstream_output_name in self.tensors_to_quantize:
                del self.tensors_to_quantize[upstream_output_name]
            return True
        return False

    def _create_q_node(
        self,
        q_input: str,
        q_output: str,
        quant_node_name: str,
        scale_name: str,
        zp_name: str,
        axis: int | None = None,
    ):
        """
        Creates a QuantizeLinear node and adds it to the model.
        """
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=self.qdq_op_domain,
        )
        self.model.add_nodes([qlinear_node])

    def _create_dq_node(
        self,
        dq_input: str,
        dq_output: str,
        dequant_node_name: str,
        scale_name: str,
        zp_name: str,
        axis: int | None = None,
    ):
        """
        Creates a DequantizeLinear node and adds it to the model.
        """
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=self.qdq_op_domain,
        )
        self.model.add_nodes([dequant_node])

    def _create_qdq_nodes(
        self, q_input, q_output, quant_node_name, dq_input, dq_output, dequant_node_name, scale_name, zp_name, axis=None
    ):
        qlinear_node = onnx.helper.make_node(
            QUANT_OP_NAME,
            [q_input, scale_name, zp_name],
            [q_output],
            quant_node_name,
            axis=axis,
            domain=self.qdq_op_domain,
        )
        dequant_node = onnx.helper.make_node(
            DEQUANT_OP_NAME,
            [dq_input, scale_name, zp_name],
            [dq_output],
            dequant_node_name,
            axis=axis,
            domain=self.qdq_op_domain,
        )
        self.model.add_nodes([qlinear_node, dequant_node])

    def _add_qdq_nodes_for_initializer(self, weight_proto: onnx.TensorProto):
        """
        Adds Q/DQ nodes for an initializer. If `self.add_qdq_pair_to_weight` is true, creates
        the sequence (weight_f32 -> Q -> DQ -> ). Otherwise, this function quantizes the initializer
        and adds the sequence (weight_quant -> DQ ->).
        """
        weight_name = weight_proto.name
        if weight_name in self.quantized_value_map:
            return

        quant_params: QuantizationParams = self.initializer_quant_params[weight_name]
        axis: int = quant_params.get("axis")
        scale_zp_initializers = self._make_scale_zp_initializers(weight_name, quant_params)
        q_weight_name: str | None = None
        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)

        if self.add_qdq_pair_to_weight:
            # Don't actually quantize the weight. Instead, keep floating-point weight and create the node
            # sequence (weight_f32 -> Q -> DQ -> weight_dequant)
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_zp_initializers.scale.name,
                scale_zp_initializers.zero_point.name,
                axis,
            )
        else:
            # Quantize the weight and create the node sequence:
            # (weight_quantized -> DQ -> weight_dequant)
            quant_weight = quantize_onnx_initializer(
                weight_proto,
                quant_params["quant_type"],
                quant_params["zero_point"],
                quant_params["scale"],
                axis,
            )
            self.model.add_initializer(quant_weight)

            q_weight_name = quant_weight.name
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [quant_weight.name, scale_zp_initializers.scale.name, scale_zp_initializers.zero_point.name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
                domain=self.qdq_op_domain,
            )
            self.model.add_node(dequant_node)

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight_name,
            q_weight_name,
            scale_zp_initializers.scale.name,
            scale_zp_initializers.zero_point.name,
            QuantizedValueType.Initializer,
            axis=axis,
        )
        self.quantized_value_map[weight_name] = QDQTensorQuantizedValue(quantized_value, None, None)

    def _add_qdq_pair_for_activation(self, tensor_name, scale_name, zp_name, data_type=None):
        if (
            self.dedicated_qdq_pair
            and tensor_name in self.tensor_to_its_receiving_nodes
            and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1
        ):
            num_dedicated_qdq_pair = len(self.tensor_to_its_receiving_nodes[tensor_name])
            for i in range(num_dedicated_qdq_pair):
                postfix = f"_{i + 1}"
                tensor_name_quant_output_postfix = add_quant_output_suffix(tensor_name) + postfix
                tensor_name_dequant_output_postfix = add_dequant_output_suffix(tensor_name) + postfix
                quant_node_name_postfix = add_quant_suffix(tensor_name) + postfix
                dequant_node_name_postfix = add_dequant_suffix(tensor_name) + postfix
                self._create_qdq_nodes(
                    tensor_name,
                    tensor_name_quant_output_postfix,
                    quant_node_name_postfix,
                    tensor_name_quant_output_postfix,
                    tensor_name_dequant_output_postfix,
                    dequant_node_name_postfix,
                    scale_name,
                    zp_name,
                )

                node = self.tensor_to_its_receiving_nodes[tensor_name][i]
                self.model.replace_node_input(node, tensor_name, tensor_name_dequant_output_postfix)
                if i == 0:
                    quantized_value = QuantizedValue(
                        tensor_name,
                        tensor_name_dequant_output_postfix,
                        scale_name,
                        zp_name,
                        QuantizedValueType.Input,
                        scale_type=data_type,
                    )
                    self.quantized_value_map[tensor_name] = QDQTensorQuantizedValue(quantized_value, None, None)
        else:
            q_input = tensor_name
            dq_output = add_dequant_output_suffix(tensor_name)
            if self.model.is_graph_output(tensor_name):
                q_input = add_quant_input_suffix(tensor_name)
                dq_output = tensor_name
                self.model.replace_output_of_all_nodes(tensor_name, q_input)
            else:
                self.model.replace_input_of_all_nodes(tensor_name, dq_output)

            self._create_qdq_nodes(
                q_input,
                add_quant_output_suffix(tensor_name),
                add_quant_suffix(tensor_name),
                add_quant_output_suffix(tensor_name),
                dq_output,
                add_dequant_suffix(tensor_name),
                scale_name,
                zp_name,
            )

            quantized_value = QuantizedValue(
                tensor_name,
                dq_output,
                scale_name,
                zp_name,
                QuantizedValueType.Input,
                scale_type=data_type,
            )
            self.quantized_value_map[tensor_name] = QDQTensorQuantizedValue(quantized_value, None, None)

    def _add_qdq_ops_for_converted_activation(
        self,
        tensor_name,
        first_scale_name,
        first_zp_name,
        scale_data_type,
        convert_scale_name,
        convert_zp_name,
        convert_recv_nodes,
    ):
        """
        Adds Q and DQ ops to a tensor whose quantized data type is converted. That is, some consumers may use the
        original data type from the producer, while other consumers use the converted data type.
        This is generally done by adding a sequence of ops that convert from one data type (e.g., uint8) to another (e.g., uint16).

        T_float ---> Quant(to u8) ---> Convert(to u16) ---> Dequant(to float) ---> T_float'
        where Convert(to u16) is equivalent to: ---> Dequant(to float) ---> Quant(to u16) --->

        This function handles the following scenarios:

        1) Tensor T is not a graph output; all consumers use the converted type

            <Producer> ---> Q1 ---> DQ1 ---> Q2 ---> DQ2 ---> <Consumers>

        2) Tensor T is not a graph output; some consumers use the original type, others use the converted type

            <Producer> ---> Q1 -+-> DQ1 ---> <Consumers of original type>
                                |
                                +-> DQ1' ---> Q2 ---> DQ2 ---> <Consumers of converted type>

        3) Tensor T is a graph output; all consumers use the converted type

            <Producer> ---> Q1 ---> DQ1 ---> Q2 ---> DQ2 -+-> <Consumers>
                                                          |
                                                          +-> <Graph output>

        4) Tensor T is a graph output; some consumers use the original type, others use the converted type

            <Producer> ---> Q1 -+-> DQ1 -+-> <Consumers of original type>
                                |        |
                                |        +-> <Graph output>
                                |
                                +-> DQ1' ---> Q2 ---> DQ2 ---> <Consumers of converted type>

        5) Tensor T is a graph output that is not consumed by any other nodes.

            <Producer> ---> Q1 ---> DQ1 ---> Q2 ---> DQ2 ---> <Graph output>
        """
        tensor_recv_nodes = {node.name for node in self.tensor_to_its_receiving_nodes.get(tensor_name, [])}

        if (
            self.dedicated_qdq_pair
            and tensor_name in self.tensor_to_its_receiving_nodes
            and len(self.tensor_to_its_receiving_nodes[tensor_name]) > 1
        ):
            # TODO: Add support for dedicated_qdq_pair if/when needed.
            raise ValueError(
                "Do not currently support converted quant_types in TensorQuantOverrides when the `dedicated_qdq_pair` extra_option is enabled"
            )

        # Determine which nodes consume the original quantized type and which nodes
        # consume the converted quantized type.
        original_recv_nodes = tensor_recv_nodes
        if convert_recv_nodes is None:  # In this case, all consumers receive the converted type.
            convert_recv_nodes = tensor_recv_nodes
            original_recv_nodes = set()
        else:
            original_recv_nodes = original_recv_nodes - convert_recv_nodes

        all_use_converted = len(convert_recv_nodes) == len(tensor_recv_nodes)
        is_graph_output = self.model.is_graph_output(tensor_name)

        # Create first Q op.
        first_q_input = tensor_name
        if is_graph_output:
            first_q_input = add_quant_input_suffix(tensor_name)
            self.model.replace_output_of_all_nodes(tensor_name, first_q_input)

        first_q_output = add_quant_output_suffix(tensor_name)
        self._create_q_node(
            first_q_input, first_q_output, add_quant_suffix(tensor_name), first_scale_name, first_zp_name
        )

        # Create first DQ op.
        first_dq_output = add_dequant_output_suffix(tensor_name)
        if is_graph_output and not all_use_converted:
            first_dq_output = tensor_name
        if original_recv_nodes and first_dq_output != tensor_name:
            self.model.replace_input_of_nodes(tensor_name, first_dq_output, original_recv_nodes)

        self._create_dq_node(
            first_q_output, first_dq_output, add_dequant_suffix(tensor_name), first_scale_name, first_zp_name
        )

        # Create parallel clone of first DQ op if _not all_ consumers use the converted type.
        # --> DQ1' --> Q2 --> DQ2 --> <Consumers of converted type>
        #
        # This DQ clone would only have one consumer Q node (Q2) and could be potentially fused with
        # it by some EPs (e.g., QNN) without breaking other "node units".
        # Ex QNN fusion:
        # --> Convert (fused) --> DQ2 --> <Consumers of converted type>
        second_q_input = first_dq_output
        if not all_use_converted:
            second_q_input = add_quant_input_suffix(f"{tensor_name}_convert")
            self._create_dq_node(
                first_q_output,
                second_q_input,
                add_dequant_suffix(f"{tensor_name}_convert_clone"),
                first_scale_name,
                first_zp_name,
            )

        # Create second Q op.
        second_q_output = add_quant_output_suffix(f"{tensor_name}_convert")
        self._create_q_node(
            second_q_input,
            second_q_output,
            add_quant_suffix(f"{tensor_name}_convert"),
            convert_scale_name,
            convert_zp_name,
        )

        # Create second DQ op.
        second_dq_output = add_dequant_output_suffix(f"{tensor_name}_convert")
        if is_graph_output and all_use_converted:
            second_dq_output = tensor_name
        if convert_recv_nodes and second_dq_output != tensor_name:
            self.model.replace_input_of_nodes(tensor_name, second_dq_output, convert_recv_nodes)
        self._create_dq_node(
            second_q_output,
            second_dq_output,
            add_dequant_suffix(f"{tensor_name}_convert"),
            convert_scale_name,
            convert_zp_name,
        )

        # Store in quantized_value_map
        original_quantized_value = QuantizedValue(
            tensor_name,
            first_dq_output,
            first_scale_name,
            first_zp_name,
            QuantizedValueType.Input,
            scale_type=scale_data_type,
        )
        converted_quantized_value = QuantizedValue(
            tensor_name,
            second_dq_output,
            convert_scale_name,
            convert_zp_name,
            QuantizedValueType.Input,
            scale_type=scale_data_type,
        )
        self.quantized_value_map[tensor_name] = QDQTensorQuantizedValue(
            original_quantized_value, converted_quantized_value, convert_recv_nodes
        )

    def _quantize_normal_tensors(self):
        """
        Adds Q/DQ ops to tensors (activations and weights) that have been marked for quantization by op quantizers.
        """
        for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
            if tensor_name in self.quantized_value_map:
                continue

            if not tensor_info.is_shared:
                # Quantize the input
                initializer = find_by_name(tensor_name, self.model.initializer())
                if initializer:
                    self._add_qdq_nodes_for_initializer(initializer)
                else:
                    # Check if this tensor is already a dequantized value. If so, skip it.
                    # This happens if the original input model already has some pre-quantized weights
                    # generated by a different tool.
                    # Ex: (quantized_weight -> DequantizeLinear -> this_tensor)
                    if tensor_name in self.tensor_to_producing_dq:
                        del self.tensors_to_quantize[tensor_name]
                        continue

                    tensor_qparam_initializers = self._make_tensor_scale_zp_initializers(tensor_name)
                    if not tensor_qparam_initializers:
                        raise ValueError(
                            f"Quantization parameters are not specified for param {tensor_name}. "
                            "In static mode quantization params for inputs and outputs of nodes to be quantized are required."
                        )

                    if tensor_qparam_initializers.converted is None:
                        # Normal case: <producer> --> Q --> DQ --> <consumers>
                        self._add_qdq_pair_for_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            data_type=tensor_info.data_type,
                        )
                    else:
                        # Conversion case: <producer> ---> Q1 -+-> DQ1 --> <consumers of original type>
                        #                                      |
                        #                                      +-> DQ1' --> Q2 --> DQ2 --> <consumers of converted type>
                        assert tensor_info.data_type == tensor_qparam_initializers.original.scale.data_type
                        self._add_qdq_ops_for_converted_activation(
                            tensor_name,
                            tensor_qparam_initializers.original.scale.name,
                            tensor_qparam_initializers.original.zero_point.name,
                            tensor_info.data_type,
                            tensor_qparam_initializers.converted.scale.name,
                            tensor_qparam_initializers.converted.zero_point.name,
                            tensor_qparam_initializers.converted_recv_nodes,
                        )

                del self.tensors_to_quantize[tensor_name]

    def _quantize_sharing_param_tensors(self):
        """
        Adds Q/DQ ops to tensors that have been marked for quantization by op quantizers.
        Only operates on tensors that want to use the quantization parameter initializers from an upstream tensor.
        For example, a Transpose node's output tensor will typically want to use the same quantization parameter
        initializers as the Transpose node's input.
        """
        while self.tensors_to_quantize:
            for tensor_name, tensor_info in self.tensors_to_quantize.copy().items():
                quant_provider = tensor_info.quant_para_provider
                if quant_provider and quant_provider.input_name in self.quantized_value_map:
                    del self.tensors_to_quantize[tensor_name]

                    quantized_value = self.quantized_value_map[quant_provider.input_name].get_for_consumer(
                        quant_provider.node_name
                    )
                    if self.is_input_a_initializer(tensor_name):
                        raise ValueError("Quantization parameter shared mode is not supported for weight yet")

                    if tensor_name in self.tensor_to_producing_dq:
                        raise ValueError(
                            f"Quantization parameter sharing is invalid for tensor {tensor_name} "
                            "because it has already been quantized"
                        )

                    # Need to check if this tensor's quant_type is converted for some consumers.
                    # If so, create new scale/zp initializers for these consumers.
                    converted_qparam_inits = None
                    converted_recv_nodes = None
                    if tensor_name in self.quantization_params:
                        tensor_params = self.quantization_params[tensor_name]
                        if tensor_params.converted:
                            converted_qparam_inits = self._make_scale_zp_initializers(
                                tensor_name, tensor_params.converted, "_convert"
                            )
                            converted_recv_nodes = tensor_params.converted_recv_nodes

                    if converted_qparam_inits is None:
                        # Normal case: <producer> --> Q_shared --> DQ_shared --> <consumers>
                        self._add_qdq_pair_for_activation(
                            tensor_name, quantized_value.scale_name, quantized_value.zp_name
                        )
                    else:
                        # Conversion case: <producer> ---> Q_shared -+-> DQ_shared --> <consumers of original type>
                        #                                            |
                        #                                            +-> DQ_shared' --> Q2 --> DQ2 --> <consumers of converted type>
                        self._add_qdq_ops_for_converted_activation(
                            tensor_name,
                            quantized_value.scale_name,
                            quantized_value.zp_name,
                            converted_qparam_inits.scale.data_type,
                            converted_qparam_inits.scale.name,
                            converted_qparam_inits.zero_point.name,
                            converted_recv_nodes,
                        )

    def _quantize_bias_tensors(self):
        """
        Adds DQ ops (or Cast) for bias tensors that have been marked for quantization by op quantizers.
        """
        for bias_name, bias_info in self.bias_to_quantize.items():
            if bias_name in self.quantized_value_map:
                continue
            # Quantize the input
            self.quantize_bias_static(bias_name, bias_info)
            init = find_by_name(bias_name, self.model.initializer())
            self.model.remove_initializer(init)
            quant_value = self.quantized_value_map[bias_name].original
            if quant_value.node_type == "Cast":
                # simple cast to float 16 and not DequantizeLinear
                # cublasLtMatmul only supports (b)float16, float bias.
                if not isinstance(init.data_type, int):
                    raise TypeError(f"Unexpected type {type(init.data_type)} for input={bias_info.input_name!r}")
                node_name = add_dequant_suffix(bias_name)
                dequant_node = onnx.helper.make_node(
                    "Cast",
                    [quant_value.q_name],
                    [bias_name],
                    name=node_name,
                    to=init.data_type,
                )
            elif quant_value.node_type in (None, "DequantizeLinear"):
                if quant_value.node_qtype in {
                    onnx.TensorProto.FLOAT16,
                    onnx.TensorProto.BFLOAT16,
                    onnx.TensorProto.FLOAT,
                }:
                    raise RuntimeError(f"Unexpected quantize type {quant_value.node_qtype} for DequantizeLinear.")
                inputs = [quant_value.q_name, quant_value.scale_name, quant_value.zp_name]
                node_name = add_dequant_suffix(bias_name)
                if quant_value.axis is not None:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        axis=quant_value.axis,
                        domain=self.qdq_op_domain,
                    )
                else:
                    dequant_node = onnx.helper.make_node(
                        "DequantizeLinear",
                        inputs,
                        [bias_name],
                        node_name,
                        domain=self.qdq_op_domain,
                    )
            else:
                raise RuntimeError(f"Unexpected operator type {quant_value.node_type!r}.")
            self.model.add_node(dequant_node)

    def is_tensor_quantized(self, tensor_name: str):
        return tensor_name in self.tensors_to_quantize or tensor_name in self.bias_to_quantize

    def is_tensor_per_channel(
        self,
        tensor_name: str,
        default_axis: int,
        op_type: str | None = None,
    ) -> tuple[bool, int | None]:
        """
        Checks if a given tensor is configured to be quantized per-channel. If so, also returns the channel axis.

        ORT only supports per-channel quantization on static weights (i.e., ONNX initializers). If the user did not provide
        tensor quantization overrides for this tensor, then the value of self.per_channel determines if the weight
        is to be quantized per-channel.

        Params:
            tensor_name: The name of the tensor to check.
            default_axis: The default channel axis. This method checks if the normalized axis is within bounds.
                          Can be overridden via the extra_options 'QDQOpTypePerChannelSupportToAxis'
                          and 'TensorQuantOverrides'.
            op_type: Optional, defaults to None. The operator type that is the only consumer of this weight.
                     Used to access the extra option 'QDQOpTypePerChannelSupportToAxis'.
        Returns:
            A tuple (is_per_channel, axis) in which the first element indicates whether the tensor is
            quantized per-channel and the second element is the channel axis.
            The returned axis is only None if the tensor is not per-channel or the axis is out of bounds.
        """
        weight_initializer = self.initializers.get(tensor_name)
        if weight_initializer is None:
            return False, None  # Only support per-channel weights

        if self.tensor_quant_overrides.has_per_tensor_overrides(tensor_name):
            return False, None  # User provided per-tensor overrides for this initializer

        has_per_chan_overrides = self.tensor_quant_overrides.has_per_channel_overrides(tensor_name)
        if not self.per_channel and not has_per_chan_overrides:
            return False, None  # global self.per_channel is off and user did not provide per-channel overrides.

        axis = self.qdq_op_type_per_channel_support_to_axis.get(op_type, default_axis) if op_type else default_axis
        if has_per_chan_overrides:
            per_chan_overrides = self.tensor_quant_overrides.get_per_channel_overrides(tensor_name)
            axis = per_chan_overrides[0]["axis"]  # Prefer axis from user-specified tensor-level overrides if available

        weight_rank = len(weight_initializer.dims)
        axis_valid, axis = normalize_axis(axis, weight_rank)
        if not axis_valid:
            logging.warning(f"Axis {axis} is out-of-range for weight '{tensor_name}' with rank {weight_rank}")
            return False, None

        return True, axis

    def _get_tensor_quantization_scale(self, tensor_name: str, consumer_node_name: str) -> np.ndarray | None:
        """
        Returns the quantization scale of a tensor that is consumed by the given node.
        :parameter tensor_name: The name of the tensor.
        :parameter consumer_node_name: The name of the node that consumes the tensor as input. Necessary in case
                                       the quantization type of the tensor was converted.
                                       Refer: QDQQuantizer::_add_qdq_ops_for_converted_activation.
        :returns: The quantization scale or None.
        """
        initializers = self.model.initializer()
        scale_initializer: onnx.TensorProto | None = None

        if tensor_name in self.quantized_value_map:
            # Tensor was quantized by this tool, so get scale from initializer created by this tool run.
            scale_name = self.quantized_value_map[tensor_name].get_for_consumer(consumer_node_name).scale_name
            scale_initializer = find_by_name(scale_name, initializers)
        else:
            # Tensor was already quantized in original model, so get scale from DQ node that outputs the tensor.
            dq_node = self.tensor_to_producing_dq.get(tensor_name, None)
            if dq_node:
                scale_initializer = find_by_name(dq_node.input[1], initializers)

        return tensor_proto_to_array(scale_initializer) if scale_initializer is not None else None

    def quantize_bias_static(self, bias_name: str, bias_info: QDQBiasQuantInfo) -> str:
        """
        Quantized the bias. Zero Point == 0 and Scale == Input_Scale * Weight_Scale
        """

        # Handle case where bias already in quantization map
        if bias_name in self.quantized_value_map:
            return self.quantized_value_map[bias_name].original.q_name

        # get scale for weight.
        weight_scale = self._get_tensor_quantization_scale(bias_info.weight_name, bias_info.node_name)
        if weight_scale is None:
            raise ValueError(
                f"Unable to get valid quantization scale for weight input '{bias_info.weight_name}' "
                f"when quantizing bias '{bias_name}' to int32."
            )

        # get scale for input.
        input_scale = self._get_tensor_quantization_scale(bias_info.input_name, bias_info.node_name)
        if input_scale is None:
            raise ValueError(
                f"Unable to get valid quantization scale for input '{bias_info.input_name}' "
                f"when quantizing bias '{bias_name}' to int32."
            )

        (
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            bias_scale_data,
            node_type,
            node_qtype,
        ) = self.quantize_bias_static_impl(bias_name, input_scale, weight_scale, bias_info.beta)

        quantized_value = QuantizedValue(
            bias_name,
            quantized_bias_name,
            quantized_bias_scale_name,
            quantized_bias_zp_name,
            QuantizedValueType.Initializer,
            0 if bias_scale_data.size > 1 else None,
            node_type=node_type,
            node_qtype=node_qtype,
        )
        self.quantized_value_map[bias_name] = QDQTensorQuantizedValue(quantized_value, None, None)

        return quantized_bias_name

    def _make_scale_zp_initializers(
        self, param_name: str, quant_params: QuantizationParams, init_name_suffix: str = ""
    ) -> QDQScaleZpInitializers:
        """
        Creates and returns scale and zero-point initializers for the given quantization params. The initializers are
        named:
            - {param_name}_zero_point{init_name_suffix}
            - {param_name}_scale{init_name_suffix}
        """
        zero_point = quant_params["zero_point"]
        scale = quant_params["scale"]
        zero_point_type = quant_params["quant_type"]
        axis: int | None = quant_params.get("axis")
        assert (axis is not None and len(scale.shape) == 1) or (axis is None and len(scale.shape) == 0), (
            "Wrong scale/zp shapes"
        )
        assert len(scale.shape) == len(zero_point.shape), "Scale and zero-point must have the same rank"

        zero_point_name = param_name + "_zero_point" + init_name_suffix
        scale_name = param_name + "_scale" + init_name_suffix

        # Add initializers to model
        init_zp = onnx.helper.make_tensor(
            zero_point_name, zero_point_type, zero_point.shape, zero_point.ravel().tolist()
        )
        self.model.add_initializer(init_zp)

        if scale.dtype == np.float32:
            scale_type = onnx_proto.TensorProto.FLOAT
        elif scale.dtype == np.float16:
            scale_type = onnx_proto.TensorProto.FLOAT16
        else:
            raise ValueError(f"Unexpected dtype={scale.dtype} for param_name={param_name!r}")
        init_scale = onnx.helper.make_tensor(scale_name, scale_type, scale.shape, scale.ravel().tolist())
        self.model.add_initializer(init_scale)

        return QDQScaleZpInitializers(init_scale, init_zp)

    def _make_tensor_scale_zp_initializers(self, tensor_name: str) -> QDQTensorScaleZpInitializers | None:
        """
        Create and returns all scale/zero_point initializers for a given tensor. If the tensor is converted
        to a different quantization type, this function creates two pairs of zp/scale initializers. Otherwise,
        only one pair of zp/scale initializers is created.
        """
        if self.quantization_params is None or tensor_name not in self.quantization_params:
            logging.info(f'Quantization parameters for tensor:"{tensor_name}" not specified')
            return None

        tensor_params = self.quantization_params[tensor_name]
        if not isinstance(tensor_params, QDQTensorQuantParams):
            raise TypeError(f"Unexpected type {type(tensor_params)} for {tensor_name!r}.")

        original_inits = self._make_scale_zp_initializers(tensor_name, tensor_params.original)
        converted_inits = (
            self._make_scale_zp_initializers(tensor_name, tensor_params.converted, "_convert")
            if tensor_params.converted
            else None
        )

        return QDQTensorScaleZpInitializers(original_inits, converted_inits, tensor_params.converted_recv_nodes)

    def calc_quant_params(self, tensor_data: TensorData, quant_overrides: dict[str, Any]) -> QuantizationParams:
        """
        Calculates quantization parameters (scale/zero-point) given a tensor's min/max range and optional
        user-provided overrides.
        """
        quant_type = self.activation_qType
        if "quant_type" in quant_overrides:
            quant_type = quant_overrides["quant_type"].tensor_type

        if "scale" in quant_overrides and "zero_point" in quant_overrides:
            zero, scale = quant_overrides["zero_point"], quant_overrides["scale"]
        elif quant_type == onnx.TensorProto.FLOAT8E4M3FN:
            zero, scale = compute_scale_zp_float8(quant_type, tensor_data.avg_std[1])
        else:
            rmin = quant_overrides.get("rmin", tensor_data.range_value[0])
            rmax = quant_overrides.get("rmax", tensor_data.range_value[1])
            symmetric = quant_overrides.get("symmetric", self.is_activation_symmetric)
            reduce_range = quant_overrides.get("reduce_range", False)
            qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=reduce_range, symmetric=symmetric)
            zero, scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric, self.min_real_range)

        return QuantizationParams(zero_point=zero.squeeze(), scale=scale.squeeze(), quant_type=quant_type)

    def calc_graph_quant_params(self) -> dict[str, QDQTensorQuantParams]:
        """
        Calculates quantization parameters (scale/zero-point) for all tensors in the graph using each tensor's min/max range
        and optional user-provided overrides.
        """
        if self.tensors_range is None:
            return {}

        self.adjust_tensor_ranges()

        quantization_params = {}
        for tensor_name in self.tensors_range:
            td = self.tensors_range[tensor_name]
            if not isinstance(td, TensorData):
                raise TypeError(f"Unexpected type {type(td)} for {tensor_name!r}.")

            quant_overrides = self.tensor_quant_overrides.get_per_tensor_overrides(tensor_name, default_val={})
            original = self.calc_quant_params(td, quant_overrides)
            converted = None
            converted_recv_nodes = None

            if "convert" in quant_overrides:
                converted = self.calc_quant_params(td, quant_overrides["convert"])
                converted_recv_nodes = quant_overrides["convert"].get("recv_nodes")

            quantization_params[tensor_name] = QDQTensorQuantParams(original, converted, converted_recv_nodes)

        return quantization_params

    def _calc_initializer_quant_params(self) -> dict[str, QuantizationParams]:
        """
        Returns quantization parameters (scale/zero_point/quant_type) for all initializers.
        """

        quantization_params: dict[str, QuantizationParams] = {}
        for tensor_name, tensor_info in self.tensors_to_quantize.items():
            initializer = find_by_name(tensor_name, self.model.initializer())
            if not initializer:
                continue

            initializer_data = tensor_proto_to_array(initializer)
            initializer_rank = len(initializer_data.shape)

            # initializers for elementwise ops use the quant_type for activations.
            is_weight = tensor_info.tensor_type is QDQQuantTensorType.WEIGHT
            quant_type = self.weight_qType if is_weight else self.activation_qType

            # Try to get scale/zp directly from user's overrides and avoid computation.
            if self.tensor_quant_overrides.overrides_scale_zp(tensor_name):
                overrides = self.tensor_quant_overrides[tensor_name]
                if "quant_type" in overrides[0]:
                    quant_type = overrides[0]["quant_type"].tensor_type

                zp_dtype = ONNX_TYPE_TO_NP_TYPE[quant_type]
                is_per_channel = "axis" in overrides[0]
                if not is_per_channel:
                    quantization_params[tensor_name] = QuantizationParams(
                        zero_point=np.array(overrides[0]["zero_point"], dtype=zp_dtype),
                        scale=np.array(overrides[0]["scale"], initializer_data.dtype),
                        quant_type=quant_type,
                    )
                else:
                    zero_points_list = []
                    scales_list = []
                    for chan_overrides in overrides:
                        zero_points_list.append(np.array(chan_overrides["zero_point"], zp_dtype))
                        scales_list.append(np.array(chan_overrides["scale"], dtype=initializer_data.dtype))

                    channel_axis = overrides[0]["axis"]
                    is_axis_valid, norm_channel_axis = normalize_axis(channel_axis, initializer_rank)
                    if not is_axis_valid:
                        raise ValueError(
                            f"Weight {initializer.name} has a per-channel axis with value {channel_axis} that is "
                            f"out-of-bounds for rank {initializer_rank}"
                        )

                    quantization_params[tensor_name] = QuantizationParams(
                        zero_point=np.array(zero_points_list),
                        scale=np.array(scales_list),
                        quant_type=quant_type,
                        axis=norm_channel_axis,
                    )

                continue

            # Compute scale/zp normally. User's overrides may still override parameters
            # used to compute the scale/zp (e.g., rmin, rmax, symmetric, etc.)
            overrides = self.tensor_quant_overrides.get(tensor_name, [{}])
            if "quant_type" in overrides[0]:
                quant_type = overrides[0]["quant_type"].tensor_type

            channel_axis = overrides[0].get("axis", tensor_info.axis)
            is_per_channel = channel_axis is not None

            # Note: always quantize per-channel initializers as symmetric because QLinear* ops require the
            # same zero-point in every channel, which is necessarily the case for symmetric quantization.
            is_symmetric_default = is_per_channel or (
                self.is_weight_symmetric(quant_type) if is_weight else self.is_activation_symmetric
            )
            is_symmetric = overrides[0].get("symmetric", is_symmetric_default)
            reduce_range = overrides[0].get("reduce_range", self.reduce_range)
            zero_point: np.ndarray | None = None
            scale: np.ndarray | None = None

            if not is_per_channel:
                zero_point, scale = compute_data_quant_params(
                    initializer_data.flatten(),
                    quant_type,
                    is_symmetric,
                    reduce_range=reduce_range,
                    min_real_range=self.min_real_range,
                    rmin_override=overrides[0].get("rmin"),
                    rmax_override=overrides[0].get("rmax"),
                )
            else:
                is_axis_valid, norm_channel_axis = normalize_axis(channel_axis, initializer_rank)
                if not is_axis_valid:
                    raise ValueError(
                        f"Weight {initializer.name} has a per-channel axis with value {channel_axis} that is "
                        f"out-of-bounds for rank {initializer_rank}"
                    )

                channel_axis = norm_channel_axis
                channel_count = initializer_data.shape[channel_axis]
                zero_points_list = []
                scales_list = []
                for i in range(channel_count):
                    per_channel_data = initializer_data.take(i, channel_axis)
                    channel_overrides = overrides[i] if overrides and i < len(overrides) else {}
                    channel_zero_point, channel_scale = compute_data_quant_params(
                        per_channel_data.ravel(),
                        quant_type,
                        is_symmetric,
                        reduce_range=reduce_range,
                        min_real_range=self.min_real_range,
                        rmin_override=channel_overrides.get("rmin"),
                        rmax_override=channel_overrides.get("rmax"),
                    )
                    zero_points_list.append(channel_zero_point)
                    scales_list.append(channel_scale)

                zero_point = np.asarray(zero_points_list)
                scale = np.asarray(scales_list)

            quantization_params[tensor_name] = QuantizationParams(
                zero_point=zero_point,
                scale=scale,
                quant_type=quant_type,
                axis=channel_axis,
            )

        return quantization_params

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\asyncio\engine.py ===
# ext/asyncio/engine.py
# Copyright (C) 2020-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Dict
from typing import Generator
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import exc as async_exc
from .base import asyncstartablecontext
from .base import GeneratorStartableContext
from .base import ProxyComparable
from .base import StartableContext
from .result import _ensure_sync_result
from .result import AsyncResult
from .result import AsyncScalarResult
from ... import exc
from ... import inspection
from ... import util
from ...engine import Connection
from ...engine import create_engine as _create_engine
from ...engine import create_pool_from_url as _create_pool_from_url
from ...engine import Engine
from ...engine.base import NestedTransaction
from ...engine.base import Transaction
from ...exc import ArgumentError
from ...util.concurrency import greenlet_spawn
from ...util.typing import Concatenate
from ...util.typing import ParamSpec

if TYPE_CHECKING:
    from ...engine.cursor import CursorResult
    from ...engine.interfaces import _CoreAnyExecuteParams
    from ...engine.interfaces import _CoreSingleExecuteParams
    from ...engine.interfaces import _DBAPIAnyExecuteParams
    from ...engine.interfaces import _ExecuteOptions
    from ...engine.interfaces import CompiledCacheType
    from ...engine.interfaces import CoreExecuteOptionsParameter
    from ...engine.interfaces import Dialect
    from ...engine.interfaces import IsolationLevel
    from ...engine.interfaces import SchemaTranslateMapType
    from ...engine.result import ScalarResult
    from ...engine.url import URL
    from ...pool import Pool
    from ...pool import PoolProxiedConnection
    from ...sql._typing import _InfoType
    from ...sql.base import Executable
    from ...sql.selectable import TypedReturnsRows

_P = ParamSpec("_P")
_T = TypeVar("_T", bound=Any)


def create_async_engine(url: Union[str, URL], **kw: Any) -> AsyncEngine:
    """Create a new async engine instance.

    Arguments passed to :func:`_asyncio.create_async_engine` are mostly
    identical to those passed to the :func:`_sa.create_engine` function.
    The specified dialect must be an asyncio-compatible dialect
    such as :ref:`dialect-postgresql-asyncpg`.

    .. versionadded:: 1.4

    :param async_creator: an async callable which returns a driver-level
        asyncio connection. If given, the function should take no arguments,
        and return a new asyncio connection from the underlying asyncio
        database driver; the connection will be wrapped in the appropriate
        structures to be used with the :class:`.AsyncEngine`.   Note that the
        parameters specified in the URL are not applied here, and the creator
        function should use its own connection parameters.

        This parameter is the asyncio equivalent of the
        :paramref:`_sa.create_engine.creator` parameter of the
        :func:`_sa.create_engine` function.

        .. versionadded:: 2.0.16

    """

    if kw.get("server_side_cursors", False):
        raise async_exc.AsyncMethodRequired(
            "Can't set server_side_cursors for async engine globally; "
            "use the connection.stream() method for an async "
            "streaming result set"
        )
    kw["_is_async"] = True
    async_creator = kw.pop("async_creator", None)
    if async_creator:
        if kw.get("creator", None):
            raise ArgumentError(
                "Can only specify one of 'async_creator' or 'creator', "
                "not both."
            )

        def creator() -> Any:
            # note that to send adapted arguments like
            # prepared_statement_cache_size, user would use
            # "creator" and emulate this form here
            return sync_engine.dialect.dbapi.connect(  # type: ignore
                async_creator_fn=async_creator
            )

        kw["creator"] = creator
    sync_engine = _create_engine(url, **kw)
    return AsyncEngine(sync_engine)


def async_engine_from_config(
    configuration: Dict[str, Any], prefix: str = "sqlalchemy.", **kwargs: Any
) -> AsyncEngine:
    """Create a new AsyncEngine instance using a configuration dictionary.

    This function is analogous to the :func:`_sa.engine_from_config` function
    in SQLAlchemy Core, except that the requested dialect must be an
    asyncio-compatible dialect such as :ref:`dialect-postgresql-asyncpg`.
    The argument signature of the function is identical to that
    of :func:`_sa.engine_from_config`.

    .. versionadded:: 1.4.29

    """
    options = {
        key[len(prefix) :]: value
        for key, value in configuration.items()
        if key.startswith(prefix)
    }
    options["_coerce_config"] = True
    options.update(kwargs)
    url = options.pop("url")
    return create_async_engine(url, **options)


def create_async_pool_from_url(url: Union[str, URL], **kwargs: Any) -> Pool:
    """Create a new async engine instance.

    Arguments passed to :func:`_asyncio.create_async_pool_from_url` are mostly
    identical to those passed to the :func:`_sa.create_pool_from_url` function.
    The specified dialect must be an asyncio-compatible dialect
    such as :ref:`dialect-postgresql-asyncpg`.

    .. versionadded:: 2.0.10

    """
    kwargs["_is_async"] = True
    return _create_pool_from_url(url, **kwargs)


class AsyncConnectable:
    __slots__ = "_slots_dispatch", "__weakref__"

    @classmethod
    def _no_async_engine_events(cls) -> NoReturn:
        raise NotImplementedError(
            "asynchronous events are not implemented at this time.  Apply "
            "synchronous listeners to the AsyncEngine.sync_engine or "
            "AsyncConnection.sync_connection attributes."
        )


@util.create_proxy_methods(
    Connection,
    ":class:`_engine.Connection`",
    ":class:`_asyncio.AsyncConnection`",
    classmethods=[],
    methods=[],
    attributes=[
        "closed",
        "invalidated",
        "dialect",
        "default_isolation_level",
    ],
)
class AsyncConnection(
    ProxyComparable[Connection],
    StartableContext["AsyncConnection"],
    AsyncConnectable,
):
    """An asyncio proxy for a :class:`_engine.Connection`.

    :class:`_asyncio.AsyncConnection` is acquired using the
    :meth:`_asyncio.AsyncEngine.connect`
    method of :class:`_asyncio.AsyncEngine`::

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("postgresql+asyncpg://user:pass@host/dbname")

        async with engine.connect() as conn:
            result = await conn.execute(select(table))

    .. versionadded:: 1.4

    """  # noqa

    # AsyncConnection is a thin proxy; no state should be added here
    # that is not retrievable from the "sync" engine / connection, e.g.
    # current transaction, info, etc.   It should be possible to
    # create a new AsyncConnection that matches this one given only the
    # "sync" elements.
    __slots__ = (
        "engine",
        "sync_engine",
        "sync_connection",
    )

    def __init__(
        self,
        async_engine: AsyncEngine,
        sync_connection: Optional[Connection] = None,
    ):
        self.engine = async_engine
        self.sync_engine = async_engine.sync_engine
        self.sync_connection = self._assign_proxied(sync_connection)

    sync_connection: Optional[Connection]
    """Reference to the sync-style :class:`_engine.Connection` this
    :class:`_asyncio.AsyncConnection` proxies requests towards.

    This instance can be used as an event target.

    .. seealso::

        :ref:`asyncio_events`

    """

    sync_engine: Engine
    """Reference to the sync-style :class:`_engine.Engine` this
    :class:`_asyncio.AsyncConnection` is associated with via its underlying
    :class:`_engine.Connection`.

    This instance can be used as an event target.

    .. seealso::

        :ref:`asyncio_events`

    """

    @classmethod
    def _regenerate_proxy_for_target(
        cls, target: Connection, **additional_kw: Any  # noqa: U100
    ) -> AsyncConnection:
        return AsyncConnection(
            AsyncEngine._retrieve_proxy_for_target(target.engine), target
        )

    async def start(
        self, is_ctxmanager: bool = False  # noqa: U100
    ) -> AsyncConnection:
        """Start this :class:`_asyncio.AsyncConnection` object's context
        outside of using a Python ``with:`` block.

        """
        if self.sync_connection:
            raise exc.InvalidRequestError("connection is already started")
        self.sync_connection = self._assign_proxied(
            await greenlet_spawn(self.sync_engine.connect)
        )
        return self

    @property
    def connection(self) -> NoReturn:
        """Not implemented for async; call
        :meth:`_asyncio.AsyncConnection.get_raw_connection`.
        """
        raise exc.InvalidRequestError(
            "AsyncConnection.connection accessor is not implemented as the "
            "attribute may need to reconnect on an invalidated connection.  "
            "Use the get_raw_connection() method."
        )

    async def get_raw_connection(self) -> PoolProxiedConnection:
        """Return the pooled DBAPI-level connection in use by this
        :class:`_asyncio.AsyncConnection`.

        This is a SQLAlchemy connection-pool proxied connection
        which then has the attribute
        :attr:`_pool._ConnectionFairy.driver_connection` that refers to the
        actual driver connection. Its
        :attr:`_pool._ConnectionFairy.dbapi_connection` refers instead
        to an :class:`_engine.AdaptedConnection` instance that
        adapts the driver connection to the DBAPI protocol.

        """

        return await greenlet_spawn(getattr, self._proxied, "connection")

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        """Return the :attr:`_engine.Connection.info` dictionary of the
        underlying :class:`_engine.Connection`.

        This dictionary is freely writable for user-defined state to be
        associated with the database connection.

        This attribute is only available if the :class:`.AsyncConnection` is
        currently connected.   If the :attr:`.AsyncConnection.closed` attribute
        is ``True``, then accessing this attribute will raise
        :class:`.ResourceClosedError`.

        .. versionadded:: 1.4.0b2

        """
        return self._proxied.info

    @util.ro_non_memoized_property
    def _proxied(self) -> Connection:
        if not self.sync_connection:
            self._raise_for_not_started()
        return self.sync_connection

    def begin(self) -> AsyncTransaction:
        """Begin a transaction prior to autobegin occurring."""
        assert self._proxied
        return AsyncTransaction(self)

    def begin_nested(self) -> AsyncTransaction:
        """Begin a nested transaction and return a transaction handle."""
        assert self._proxied
        return AsyncTransaction(self, nested=True)

    async def invalidate(
        self, exception: Optional[BaseException] = None
    ) -> None:
        """Invalidate the underlying DBAPI connection associated with
        this :class:`_engine.Connection`.

        See the method :meth:`_engine.Connection.invalidate` for full
        detail on this method.

        """

        return await greenlet_spawn(
            self._proxied.invalidate, exception=exception
        )

    async def get_isolation_level(self) -> IsolationLevel:
        return await greenlet_spawn(self._proxied.get_isolation_level)

    def in_transaction(self) -> bool:
        """Return True if a transaction is in progress."""

        return self._proxied.in_transaction()

    def in_nested_transaction(self) -> bool:
        """Return True if a transaction is in progress.

        .. versionadded:: 1.4.0b2

        """
        return self._proxied.in_nested_transaction()

    def get_transaction(self) -> Optional[AsyncTransaction]:
        """Return an :class:`.AsyncTransaction` representing the current
        transaction, if any.

        This makes use of the underlying synchronous connection's
        :meth:`_engine.Connection.get_transaction` method to get the current
        :class:`_engine.Transaction`, which is then proxied in a new
        :class:`.AsyncTransaction` object.

        .. versionadded:: 1.4.0b2

        """

        trans = self._proxied.get_transaction()
        if trans is not None:
            return AsyncTransaction._retrieve_proxy_for_target(trans)
        else:
            return None

    def get_nested_transaction(self) -> Optional[AsyncTransaction]:
        """Return an :class:`.AsyncTransaction` representing the current
        nested (savepoint) transaction, if any.

        This makes use of the underlying synchronous connection's
        :meth:`_engine.Connection.get_nested_transaction` method to get the
        current :class:`_engine.Transaction`, which is then proxied in a new
        :class:`.AsyncTransaction` object.

        .. versionadded:: 1.4.0b2

        """

        trans = self._proxied.get_nested_transaction()
        if trans is not None:
            return AsyncTransaction._retrieve_proxy_for_target(trans)
        else:
            return None

    @overload
    async def execution_options(
        self,
        *,
        compiled_cache: Optional[CompiledCacheType] = ...,
        logging_token: str = ...,
        isolation_level: IsolationLevel = ...,
        no_parameters: bool = False,
        stream_results: bool = False,
        max_row_buffer: int = ...,
        yield_per: int = ...,
        insertmanyvalues_page_size: int = ...,
        schema_translate_map: Optional[SchemaTranslateMapType] = ...,
        preserve_rowcount: bool = False,
        **opt: Any,
    ) -> AsyncConnection: ...

    @overload
    async def execution_options(self, **opt: Any) -> AsyncConnection: ...

    async def execution_options(self, **opt: Any) -> AsyncConnection:
        r"""Set non-SQL options for the connection which take effect
        during execution.

        This returns this :class:`_asyncio.AsyncConnection` object with
        the new options added.

        See :meth:`_engine.Connection.execution_options` for full details
        on this method.

        """

        conn = self._proxied
        c2 = await greenlet_spawn(conn.execution_options, **opt)
        assert c2 is conn
        return self

    async def commit(self) -> None:
        """Commit the transaction that is currently in progress.

        This method commits the current transaction if one has been started.
        If no transaction was started, the method has no effect, assuming
        the connection is in a non-invalidated state.

        A transaction is begun on a :class:`_engine.Connection` automatically
        whenever a statement is first executed, or when the
        :meth:`_engine.Connection.begin` method is called.

        """
        await greenlet_spawn(self._proxied.commit)

    async def rollback(self) -> None:
        """Roll back the transaction that is currently in progress.

        This method rolls back the current transaction if one has been started.
        If no transaction was started, the method has no effect.  If a
        transaction was started and the connection is in an invalidated state,
        the transaction is cleared using this method.

        A transaction is begun on a :class:`_engine.Connection` automatically
        whenever a statement is first executed, or when the
        :meth:`_engine.Connection.begin` method is called.


        """
        await greenlet_spawn(self._proxied.rollback)

    async def close(self) -> None:
        """Close this :class:`_asyncio.AsyncConnection`.

        This has the effect of also rolling back the transaction if one
        is in place.

        """
        await greenlet_spawn(self._proxied.close)

    async def aclose(self) -> None:
        """A synonym for :meth:`_asyncio.AsyncConnection.close`.

        The :meth:`_asyncio.AsyncConnection.aclose` name is specifically
        to support the Python standard library ``@contextlib.aclosing``
        context manager function.

        .. versionadded:: 2.0.20

        """
        await self.close()

    async def exec_driver_sql(
        self,
        statement: str,
        parameters: Optional[_DBAPIAnyExecuteParams] = None,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[Any]:
        r"""Executes a driver-level SQL string and return buffered
        :class:`_engine.Result`.

        """

        result = await greenlet_spawn(
            self._proxied.exec_driver_sql,
            statement,
            parameters,
            execution_options,
            _require_await=True,
        )

        return await _ensure_sync_result(result, self.exec_driver_sql)

    @overload
    def stream(
        self,
        statement: TypedReturnsRows[_T],
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> GeneratorStartableContext[AsyncResult[_T]]: ...

    @overload
    def stream(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> GeneratorStartableContext[AsyncResult[Any]]: ...

    @asyncstartablecontext
    async def stream(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> AsyncIterator[AsyncResult[Any]]:
        """Execute a statement and return an awaitable yielding a
        :class:`_asyncio.AsyncResult` object.

        E.g.::

            result = await conn.stream(stmt)
            async for row in result:
                print(f"{row}")

        The :meth:`.AsyncConnection.stream`
        method supports optional context manager use against the
        :class:`.AsyncResult` object, as in::

            async with conn.stream(stmt) as result:
                async for row in result:
                    print(f"{row}")

        In the above pattern, the :meth:`.AsyncResult.close` method is
        invoked unconditionally, even if the iterator is interrupted by an
        exception throw.   Context manager use remains optional, however,
        and the function may be called in either an ``async with fn():`` or
        ``await fn()`` style.

        .. versionadded:: 2.0.0b3 added context manager support


        :return: an awaitable object that will yield an
         :class:`_asyncio.AsyncResult` object.

        .. seealso::

            :meth:`.AsyncConnection.stream_scalars`

        """
        if not self.dialect.supports_server_side_cursors:
            raise exc.InvalidRequestError(
                "Cant use `stream` or `stream_scalars` with the current "
                "dialect since it does not support server side cursors."
            )

        result = await greenlet_spawn(
            self._proxied.execute,
            statement,
            parameters,
            execution_options=util.EMPTY_DICT.merge_with(
                execution_options, {"stream_results": True}
            ),
            _require_await=True,
        )
        assert result.context._is_server_side
        ar = AsyncResult(result)
        try:
            yield ar
        except GeneratorExit:
            pass
        else:
            task = asyncio.create_task(ar.close())
            await asyncio.shield(task)

    @overload
    async def execute(
        self,
        statement: TypedReturnsRows[_T],
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[_T]: ...

    @overload
    async def execute(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[Any]: ...

    async def execute(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[Any]:
        r"""Executes a SQL statement construct and return a buffered
        :class:`_engine.Result`.

        :param object: The statement to be executed.  This is always
         an object that is in both the :class:`_expression.ClauseElement` and
         :class:`_expression.Executable` hierarchies, including:

         * :class:`_expression.Select`
         * :class:`_expression.Insert`, :class:`_expression.Update`,
           :class:`_expression.Delete`
         * :class:`_expression.TextClause` and
           :class:`_expression.TextualSelect`
         * :class:`_schema.DDL` and objects which inherit from
           :class:`_schema.ExecutableDDLElement`

        :param parameters: parameters which will be bound into the statement.
         This may be either a dictionary of parameter names to values,
         or a mutable sequence (e.g. a list) of dictionaries.  When a
         list of dictionaries is passed, the underlying statement execution
         will make use of the DBAPI ``cursor.executemany()`` method.
         When a single dictionary is passed, the DBAPI ``cursor.execute()``
         method will be used.

        :param execution_options: optional dictionary of execution options,
         which will be associated with the statement execution.  This
         dictionary can provide a subset of the options that are accepted
         by :meth:`_engine.Connection.execution_options`.

        :return: a :class:`_engine.Result` object.

        """
        result = await greenlet_spawn(
            self._proxied.execute,
            statement,
            parameters,
            execution_options=execution_options,
            _require_await=True,
        )
        return await _ensure_sync_result(result, self.execute)

    @overload
    async def scalar(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Optional[_T]: ...

    @overload
    async def scalar(
        self,
        statement: Executable,
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Any: ...

    async def scalar(
        self,
        statement: Executable,
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Any:
        r"""Executes a SQL statement construct and returns a scalar object.

        This method is shorthand for invoking the
        :meth:`_engine.Result.scalar` method after invoking the
        :meth:`_engine.Connection.execute` method.  Parameters are equivalent.

        :return: a scalar Python value representing the first column of the
         first row returned.

        """
        result = await self.execute(
            statement, parameters, execution_options=execution_options
        )
        return result.scalar()

    @overload
    async def scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> ScalarResult[_T]: ...

    @overload
    async def scalars(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> ScalarResult[Any]: ...

    async def scalars(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> ScalarResult[Any]:
        r"""Executes a SQL statement construct and returns a scalar objects.

        This method is shorthand for invoking the
        :meth:`_engine.Result.scalars` method after invoking the
        :meth:`_engine.Connection.execute` method.  Parameters are equivalent.

        :return: a :class:`_engine.ScalarResult` object.

        .. versionadded:: 1.4.24

        """
        result = await self.execute(
            statement, parameters, execution_options=execution_options
        )
        return result.scalars()

    @overload
    def stream_scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> GeneratorStartableContext[AsyncScalarResult[_T]]: ...

    @overload
    def stream_scalars(
        self,
        statement: Executable,
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> GeneratorStartableContext[AsyncScalarResult[Any]]: ...

    @asyncstartablecontext
    async def stream_scalars(
        self,
        statement: Executable,
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> AsyncIterator[AsyncScalarResult[Any]]:
        r"""Execute a statement and return an awaitable yielding a
        :class:`_asyncio.AsyncScalarResult` object.

        E.g.::

            result = await conn.stream_scalars(stmt)
            async for scalar in result:
                print(f"{scalar}")

        This method is shorthand for invoking the
        :meth:`_engine.AsyncResult.scalars` method after invoking the
        :meth:`_engine.Connection.stream` method.  Parameters are equivalent.

        The :meth:`.AsyncConnection.stream_scalars`
        method supports optional context manager use against the
        :class:`.AsyncScalarResult` object, as in::

            async with conn.stream_scalars(stmt) as result:
                async for scalar in result:
                    print(f"{scalar}")

        In the above pattern, the :meth:`.AsyncScalarResult.close` method is
        invoked unconditionally, even if the iterator is interrupted by an
        exception throw.  Context manager use remains optional, however,
        and the function may be called in either an ``async with fn():`` or
        ``await fn()`` style.

        .. versionadded:: 2.0.0b3 added context manager support

        :return: an awaitable object that will yield an
         :class:`_asyncio.AsyncScalarResult` object.

        .. versionadded:: 1.4.24

        .. seealso::

            :meth:`.AsyncConnection.stream`

        """

        async with self.stream(
            statement, parameters, execution_options=execution_options
        ) as result:
            yield result.scalars()

    async def run_sync(
        self,
        fn: Callable[Concatenate[Connection, _P], _T],
        *arg: _P.args,
        **kw: _P.kwargs,
    ) -> _T:
        '''Invoke the given synchronous (i.e. not async) callable,
        passing a synchronous-style :class:`_engine.Connection` as the first
        argument.

        This method allows traditional synchronous SQLAlchemy functions to
        run within the context of an asyncio application.

        E.g.::

            def do_something_with_core(conn: Connection, arg1: int, arg2: str) -> str:
                """A synchronous function that does not require awaiting

                :param conn: a Core SQLAlchemy Connection, used synchronously

                :return: an optional return value is supported

                """
                conn.execute(some_table.insert().values(int_col=arg1, str_col=arg2))
                return "success"


            async def do_something_async(async_engine: AsyncEngine) -> None:
                """an async function that uses awaiting"""

                async with async_engine.begin() as async_conn:
                    # run do_something_with_core() with a sync-style
                    # Connection, proxied into an awaitable
                    return_code = await async_conn.run_sync(
                        do_something_with_core, 5, "strval"
                    )
                    print(return_code)

        This method maintains the asyncio event loop all the way through
        to the database connection by running the given callable in a
        specially instrumented greenlet.

        The most rudimentary use of :meth:`.AsyncConnection.run_sync` is to
        invoke methods such as :meth:`_schema.MetaData.create_all`, given
        an :class:`.AsyncConnection` that needs to be provided to
        :meth:`_schema.MetaData.create_all` as a :class:`_engine.Connection`
        object::

            # run metadata.create_all(conn) with a sync-style Connection,
            # proxied into an awaitable
            with async_engine.begin() as conn:
                await conn.run_sync(metadata.create_all)

        .. note::

            The provided callable is invoked inline within the asyncio event
            loop, and will block on traditional IO calls.  IO within this
            callable should only call into SQLAlchemy's asyncio database
            APIs which will be properly adapted to the greenlet context.

        .. seealso::

            :meth:`.AsyncSession.run_sync`

            :ref:`session_run_sync`

        '''  # noqa: E501

        return await greenlet_spawn(
            fn, self._proxied, *arg, _require_await=False, **kw
        )

    def __await__(self) -> Generator[Any, None, AsyncConnection]:
        return self.start().__await__()

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        task = asyncio.create_task(self.close())
        await asyncio.shield(task)

    # START PROXY METHODS AsyncConnection

    # code within this block is **programmatically,
    # statically generated** by tools/generate_proxy_methods.py

    @property
    def closed(self) -> Any:
        r"""Return True if this connection is closed.

        .. container:: class_bases

            Proxied for the :class:`_engine.Connection` class
            on behalf of the :class:`_asyncio.AsyncConnection` class.

        """  # noqa: E501

        return self._proxied.closed

    @property
    def invalidated(self) -> Any:
        r"""Return True if this connection was invalidated.

        .. container:: class_bases

            Proxied for the :class:`_engine.Connection` class
            on behalf of the :class:`_asyncio.AsyncConnection` class.

        This does not indicate whether or not the connection was
        invalidated at the pool level, however


        """  # noqa: E501

        return self._proxied.invalidated

    @property
    def dialect(self) -> Dialect:
        r"""Proxy for the :attr:`_engine.Connection.dialect` attribute
        on behalf of the :class:`_asyncio.AsyncConnection` class.

        """  # noqa: E501

        return self._proxied.dialect

    @dialect.setter
    def dialect(self, attr: Dialect) -> None:
        self._proxied.dialect = attr

    @property
    def default_isolation_level(self) -> Any:
        r"""The initial-connection time isolation level associated with the
        :class:`_engine.Dialect` in use.

        .. container:: class_bases

            Proxied for the :class:`_engine.Connection` class
            on behalf of the :class:`_asyncio.AsyncConnection` class.

        This value is independent of the
        :paramref:`.Connection.execution_options.isolation_level` and
        :paramref:`.Engine.execution_options.isolation_level` execution
        options, and is determined by the :class:`_engine.Dialect` when the
        first connection is created, by performing a SQL query against the
        database for the current isolation level before any additional commands
        have been emitted.

        Calling this accessor does not invoke any new SQL queries.

        .. seealso::

            :meth:`_engine.Connection.get_isolation_level`
            - view current actual isolation level

            :paramref:`_sa.create_engine.isolation_level`
            - set per :class:`_engine.Engine` isolation level

            :paramref:`.Connection.execution_options.isolation_level`
            - set per :class:`_engine.Connection` isolation level


        """  # noqa: E501

        return self._proxied.default_isolation_level

    # END PROXY METHODS AsyncConnection


@util.create_proxy_methods(
    Engine,
    ":class:`_engine.Engine`",
    ":class:`_asyncio.AsyncEngine`",
    classmethods=[],
    methods=[
        "clear_compiled_cache",
        "update_execution_options",
        "get_execution_options",
    ],
    attributes=["url", "pool", "dialect", "engine", "name", "driver", "echo"],
)
class AsyncEngine(ProxyComparable[Engine], AsyncConnectable):
    """An asyncio proxy for a :class:`_engine.Engine`.

    :class:`_asyncio.AsyncEngine` is acquired using the
    :func:`_asyncio.create_async_engine` function::

        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine("postgresql+asyncpg://user:pass@host/dbname")

    .. versionadded:: 1.4

    """  # noqa

    # AsyncEngine is a thin proxy; no state should be added here
    # that is not retrievable from the "sync" engine / connection, e.g.
    # current transaction, info, etc.   It should be possible to
    # create a new AsyncEngine that matches this one given only the
    # "sync" elements.
    __slots__ = "sync_engine"

    _connection_cls: Type[AsyncConnection] = AsyncConnection

    sync_engine: Engine
    """Reference to the sync-style :class:`_engine.Engine` this
    :class:`_asyncio.AsyncEngine` proxies requests towards.

    This instance can be used as an event target.

    .. seealso::

        :ref:`asyncio_events`
    """

    def __init__(self, sync_engine: Engine):
        if not sync_engine.dialect.is_async:
            raise exc.InvalidRequestError(
                "The asyncio extension requires an async driver to be used. "
                f"The loaded {sync_engine.dialect.driver!r} is not async."
            )
        self.sync_engine = self._assign_proxied(sync_engine)

    @util.ro_non_memoized_property
    def _proxied(self) -> Engine:
        return self.sync_engine

    @classmethod
    def _regenerate_proxy_for_target(
        cls, target: Engine, **additional_kw: Any  # noqa: U100
    ) -> AsyncEngine:
        return AsyncEngine(target)

    @contextlib.asynccontextmanager
    async def begin(self) -> AsyncIterator[AsyncConnection]:
        """Return a context manager which when entered will deliver an
        :class:`_asyncio.AsyncConnection` with an
        :class:`_asyncio.AsyncTransaction` established.

        E.g.::

            async with async_engine.begin() as conn:
                await conn.execute(
                    text("insert into table (x, y, z) values (1, 2, 3)")
                )
                await conn.execute(text("my_special_procedure(5)"))

        """
        conn = self.connect()

        async with conn:
            async with conn.begin():
                yield conn

    def connect(self) -> AsyncConnection:
        """Return an :class:`_asyncio.AsyncConnection` object.

        The :class:`_asyncio.AsyncConnection` will procure a database
        connection from the underlying connection pool when it is entered
        as an async context manager::

            async with async_engine.connect() as conn:
                result = await conn.execute(select(user_table))

        The :class:`_asyncio.AsyncConnection` may also be started outside of a
        context manager by invoking its :meth:`_asyncio.AsyncConnection.start`
        method.

        """

        return self._connection_cls(self)

    async def raw_connection(self) -> PoolProxiedConnection:
        """Return a "raw" DBAPI connection from the connection pool.

        .. seealso::

            :ref:`dbapi_connections`

        """
        return await greenlet_spawn(self.sync_engine.raw_connection)

    @overload
    def execution_options(
        self,
        *,
        compiled_cache: Optional[CompiledCacheType] = ...,
        logging_token: str = ...,
        isolation_level: IsolationLevel = ...,
        insertmanyvalues_page_size: int = ...,
        schema_translate_map: Optional[SchemaTranslateMapType] = ...,
        **opt: Any,
    ) -> AsyncEngine: ...

    @overload
    def execution_options(self, **opt: Any) -> AsyncEngine: ...

    def execution_options(self, **opt: Any) -> AsyncEngine:
        """Return a new :class:`_asyncio.AsyncEngine` that will provide
        :class:`_asyncio.AsyncConnection` objects with the given execution
        options.

        Proxied from :meth:`_engine.Engine.execution_options`.  See that
        method for details.

        """

        return AsyncEngine(self.sync_engine.execution_options(**opt))

    async def dispose(self, close: bool = True) -> None:
        """Dispose of the connection pool used by this
        :class:`_asyncio.AsyncEngine`.

        :param close: if left at its default of ``True``, has the
         effect of fully closing all **currently checked in**
         database connections.  Connections that are still checked out
         will **not** be closed, however they will no longer be associated
         with this :class:`_engine.Engine`,
         so when they are closed individually, eventually the
         :class:`_pool.Pool` which they are associated with will
         be garbage collected and they will be closed out fully, if
         not already closed on checkin.

         If set to ``False``, the previous connection pool is de-referenced,
         and otherwise not touched in any way.

        .. seealso::

            :meth:`_engine.Engine.dispose`

        """

        await greenlet_spawn(self.sync_engine.dispose, close=close)

    # START PROXY METHODS AsyncEngine

    # code within this block is **programmatically,
    # statically generated** by tools/generate_proxy_methods.py

    def clear_compiled_cache(self) -> None:
        r"""Clear the compiled cache associated with the dialect.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class on
            behalf of the :class:`_asyncio.AsyncEngine` class.

        This applies **only** to the built-in cache that is established
        via the :paramref:`_engine.create_engine.query_cache_size` parameter.
        It will not impact any dictionary caches that were passed via the
        :paramref:`.Connection.execution_options.compiled_cache` parameter.

        .. versionadded:: 1.4


        """  # noqa: E501

        return self._proxied.clear_compiled_cache()

    def update_execution_options(self, **opt: Any) -> None:
        r"""Update the default execution_options dictionary
        of this :class:`_engine.Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class on
            behalf of the :class:`_asyncio.AsyncEngine` class.

        The given keys/values in \**opt are added to the
        default execution options that will be used for
        all connections.  The initial contents of this dictionary
        can be sent via the ``execution_options`` parameter
        to :func:`_sa.create_engine`.

        .. seealso::

            :meth:`_engine.Connection.execution_options`

            :meth:`_engine.Engine.execution_options`


        """  # noqa: E501

        return self._proxied.update_execution_options(**opt)

    def get_execution_options(self) -> _ExecuteOptions:
        r"""Get the non-SQL options which will take effect during execution.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class on
            behalf of the :class:`_asyncio.AsyncEngine` class.

        .. versionadded: 1.3

        .. seealso::

            :meth:`_engine.Engine.execution_options`

        """  # noqa: E501

        return self._proxied.get_execution_options()

    @property
    def url(self) -> URL:
        r"""Proxy for the :attr:`_engine.Engine.url` attribute
        on behalf of the :class:`_asyncio.AsyncEngine` class.

        """  # noqa: E501

        return self._proxied.url

    @url.setter
    def url(self, attr: URL) -> None:
        self._proxied.url = attr

    @property
    def pool(self) -> Pool:
        r"""Proxy for the :attr:`_engine.Engine.pool` attribute
        on behalf of the :class:`_asyncio.AsyncEngine` class.

        """  # noqa: E501

        return self._proxied.pool

    @pool.setter
    def pool(self, attr: Pool) -> None:
        self._proxied.pool = attr

    @property
    def dialect(self) -> Dialect:
        r"""Proxy for the :attr:`_engine.Engine.dialect` attribute
        on behalf of the :class:`_asyncio.AsyncEngine` class.

        """  # noqa: E501

        return self._proxied.dialect

    @dialect.setter
    def dialect(self, attr: Dialect) -> None:
        self._proxied.dialect = attr

    @property
    def engine(self) -> Any:
        r"""Returns this :class:`.Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.

        Used for legacy schemes that accept :class:`.Connection` /
        :class:`.Engine` objects within the same variable.


        """  # noqa: E501

        return self._proxied.engine

    @property
    def name(self) -> Any:
        r"""String name of the :class:`~sqlalchemy.engine.interfaces.Dialect`
        in use by this :class:`Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.


        """  # noqa: E501

        return self._proxied.name

    @property
    def driver(self) -> Any:
        r"""Driver name of the :class:`~sqlalchemy.engine.interfaces.Dialect`
        in use by this :class:`Engine`.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.


        """  # noqa: E501

        return self._proxied.driver

    @property
    def echo(self) -> Any:
        r"""When ``True``, enable log output for this element.

        .. container:: class_bases

            Proxied for the :class:`_engine.Engine` class
            on behalf of the :class:`_asyncio.AsyncEngine` class.

        This has the effect of setting the Python logging level for the namespace
        of this element's class and object reference.  A value of boolean ``True``
        indicates that the loglevel ``logging.INFO`` will be set for the logger,
        whereas the string value ``debug`` will set the loglevel to
        ``logging.DEBUG``.

        """  # noqa: E501

        return self._proxied.echo

    @echo.setter
    def echo(self, attr: Any) -> None:
        self._proxied.echo = attr

    # END PROXY METHODS AsyncEngine


class AsyncTransaction(
    ProxyComparable[Transaction], StartableContext["AsyncTransaction"]
):
    """An asyncio proxy for a :class:`_engine.Transaction`."""

    __slots__ = ("connection", "sync_transaction", "nested")

    sync_transaction: Optional[Transaction]
    connection: AsyncConnection
    nested: bool

    def __init__(self, connection: AsyncConnection, nested: bool = False):
        self.connection = connection
        self.sync_transaction = None
        self.nested = nested

    @classmethod
    def _regenerate_proxy_for_target(
        cls, target: Transaction, **additional_kw: Any  # noqa: U100
    ) -> AsyncTransaction:
        sync_connection = target.connection
        sync_transaction = target
        nested = isinstance(target, NestedTransaction)

        async_connection = AsyncConnection._retrieve_proxy_for_target(
            sync_connection
        )
        assert async_connection is not None

        obj = cls.__new__(cls)
        obj.connection = async_connection
        obj.sync_transaction = obj._assign_proxied(sync_transaction)
        obj.nested = nested
        return obj

    @util.ro_non_memoized_property
    def _proxied(self) -> Transaction:
        if not self.sync_transaction:
            self._raise_for_not_started()
        return self.sync_transaction

    @property
    def is_valid(self) -> bool:
        return self._proxied.is_valid

    @property
    def is_active(self) -> bool:
        return self._proxied.is_active

    async def close(self) -> None:
        """Close this :class:`.AsyncTransaction`.

        If this transaction is the base transaction in a begin/commit
        nesting, the transaction will rollback().  Otherwise, the
        method returns.

        This is used to cancel a Transaction without affecting the scope of
        an enclosing transaction.

        """
        await greenlet_spawn(self._proxied.close)

    async def rollback(self) -> None:
        """Roll back this :class:`.AsyncTransaction`."""
        await greenlet_spawn(self._proxied.rollback)

    async def commit(self) -> None:
        """Commit this :class:`.AsyncTransaction`."""

        await greenlet_spawn(self._proxied.commit)

    async def start(self, is_ctxmanager: bool = False) -> AsyncTransaction:
        """Start this :class:`_asyncio.AsyncTransaction` object's context
        outside of using a Python ``with:`` block.

        """

        self.sync_transaction = self._assign_proxied(
            await greenlet_spawn(
                self.connection._proxied.begin_nested
                if self.nested
                else self.connection._proxied.begin
            )
        )
        if is_ctxmanager:
            self.sync_transaction.__enter__()
        return self

    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> None:
        await greenlet_spawn(self._proxied.__exit__, type_, value, traceback)


@overload
def _get_sync_engine_or_connection(async_engine: AsyncEngine) -> Engine: ...


@overload
def _get_sync_engine_or_connection(
    async_engine: AsyncConnection,
) -> Connection: ...


def _get_sync_engine_or_connection(
    async_engine: Union[AsyncEngine, AsyncConnection]
) -> Union[Engine, Connection]:
    if isinstance(async_engine, AsyncConnection):
        return async_engine._proxied

    try:
        return async_engine.sync_engine
    except AttributeError as e:
        raise exc.ArgumentError(
            "AsyncEngine expected, got %r" % async_engine
        ) from e


@inspection._inspects(AsyncConnection)
def _no_insp_for_async_conn_yet(
    subject: AsyncConnection,  # noqa: U100
) -> NoReturn:
    raise exc.NoInspectionAvailable(
        "Inspection on an AsyncConnection is currently not supported. "
        "Please use ``run_sync`` to pass a callable where it's possible "
        "to call ``inspect`` on the passed connection.",
        code="xd3s",
    )


@inspection._inspects(AsyncEngine)
def _no_insp_for_async_engine_xyet(
    subject: AsyncEngine,  # noqa: U100
) -> NoReturn:
    raise exc.NoInspectionAvailable(
        "Inspection on an AsyncEngine is currently not supported. "
        "Please obtain a connection then use ``conn.run_sync`` to pass a "
        "callable where it's possible to call ``inspect`` on the "
        "passed connection.",
        code="xd3s",
    )