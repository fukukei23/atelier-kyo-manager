
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\pytables.py ===
"""
High level interface to PyTables for reading and writing pandas data structures
to disk
"""
from __future__ import annotations

from contextlib import suppress
import copy
from datetime import (
    date,
    tzinfo,
)
import itertools
import os
import re
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import (
    config,
    get_option,
    using_copy_on_write,
    using_string_dtype,
)

from pandas._libs import (
    lib,
    writers as libwriters,
)
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
    AttributeConflictWarning,
    ClosedFileError,
    IncompatibilityWarning,
    PerformanceWarning,
    PossibleDataLossError,
)
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_complex_dtype,
    is_list_like,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import array_equivalent

from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    StringDtype,
    TimedeltaIndex,
    concat,
    isna,
)
from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    PeriodArray,
)
from pandas.core.arrays.string_ import BaseStringArray
import pandas.core.common as com
from pandas.core.computation.pytables import (
    PyTablesExpr,
    maybe_expression,
)
from pandas.core.construction import (
    array as pd_array,
    extract_array,
)
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
    ArrayManager,
    BlockManager,
)

from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
    adjoin,
    pprint_thing,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Sequence,
    )
    from types import TracebackType

    from tables import (
        Col,
        File,
        Node,
    )

    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        AxisInt,
        DtypeArg,
        FilePath,
        Self,
        Shape,
        npt,
    )

    from pandas.core.internals import Block

# versioning attribute
_version = "0.15.2"

# encoding
_default_encoding = "UTF-8"


def _ensure_decoded(s):
    """if we have bytes, decode them to unicode"""
    if isinstance(s, np.bytes_):
        s = s.decode("UTF-8")
    return s


def _ensure_encoding(encoding: str | None) -> str:
    # set the encoding if we need
    if encoding is None:
        encoding = _default_encoding

    return encoding


def _ensure_str(name):
    """
    Ensure that an index / column name is a str (python 3); otherwise they
    may be np.string dtype. Non-string dtypes are passed through unchanged.

    https://github.com/pandas-dev/pandas/issues/13492
    """
    if isinstance(name, str):
        name = str(name)
    return name


Term = PyTablesExpr


def _ensure_term(where, scope_level: int):
    """
    Ensure that the where is a Term or a list of Term.

    This makes sure that we are capturing the scope of variables that are
    passed create the terms here with a frame_level=2 (we are 2 levels down)
    """
    # only consider list/tuple here as an ndarray is automatically a coordinate
    # list
    level = scope_level + 1
    if isinstance(where, (list, tuple)):
        where = [
            Term(term, scope_level=level + 1) if maybe_expression(term) else term
            for term in where
            if term is not None
        ]
    elif maybe_expression(where):
        where = Term(where, scope_level=level)
    return where if where is None or len(where) else None


incompatibility_doc: Final = """
where criteria is being ignored as this version [%s] is too old (or
not-defined), read the file in and write it out to a new file to upgrade (with
the copy_to method)
"""

attribute_conflict_doc: Final = """
the [%s] attribute of the existing index is [%s] which conflicts with the new
[%s], resetting the attribute to None
"""

performance_doc: Final = """
your performance may suffer as PyTables will pickle object types that it cannot
map directly to c-types [inferred_type->%s,key->%s] [items->%s]
"""

# formats
_FORMAT_MAP = {"f": "fixed", "fixed": "fixed", "t": "table", "table": "table"}

# axes map
_AXES_MAP = {DataFrame: [0]}

# register our configuration options
dropna_doc: Final = """
: boolean
    drop ALL nan rows when appending to a table
"""
format_doc: Final = """
: format
    default format writing format, if None, then
    put will default to 'fixed' and append will default to 'table'
"""

with config.config_prefix("io.hdf"):
    config.register_option("dropna_table", False, dropna_doc, validator=config.is_bool)
    config.register_option(
        "default_format",
        None,
        format_doc,
        validator=config.is_one_of_factory(["fixed", "table", None]),
    )

# oh the troubles to reduce import time
_table_mod = None
_table_file_open_policy_is_strict = False


def _tables():
    global _table_mod
    global _table_file_open_policy_is_strict
    if _table_mod is None:
        import tables

        _table_mod = tables

        # set the file open policy
        # return the file open policy; this changes as of pytables 3.1
        # depending on the HDF5 version
        with suppress(AttributeError):
            _table_file_open_policy_is_strict = (
                tables.file._FILE_OPEN_POLICY == "strict"
            )

    return _table_mod


# interface to/from ###


def to_hdf(
    path_or_buf: FilePath | HDFStore,
    key: str,
    value: DataFrame | Series,
    mode: str = "a",
    complevel: int | None = None,
    complib: str | None = None,
    append: bool = False,
    format: str | None = None,
    index: bool = True,
    min_itemsize: int | dict[str, int] | None = None,
    nan_rep=None,
    dropna: bool | None = None,
    data_columns: Literal[True] | list[str] | None = None,
    errors: str = "strict",
    encoding: str = "UTF-8",
) -> None:
    """store this object, close it if we opened it"""
    if append:
        f = lambda store: store.append(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )
    else:
        # NB: dropna is not passed to `put`
        f = lambda store: store.put(
            key,
            value,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
            dropna=dropna,
        )

    path_or_buf = stringify_path(path_or_buf)
    if isinstance(path_or_buf, str):
        with HDFStore(
            path_or_buf, mode=mode, complevel=complevel, complib=complib
        ) as store:
            f(store)
    else:
        f(path_or_buf)


def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key=None,
    mode: str = "r",
    errors: str = "strict",
    where: str | list | None = None,
    start: int | None = None,
    stop: int | None = None,
    columns: list[str] | None = None,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwargs,
):
    """
    Read from the store, close it if we opened it.

    Retrieve pandas object stored in file, optionally based on where
    criteria.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path_or_buf : str, path object, pandas.HDFStore
        Any valid string path is acceptable. Only supports the local file system,
        remote URLs and file-like objects are not supported.

        If you want to pass in a path object, pandas accepts any
        ``os.PathLike``.

        Alternatively, pandas accepts an open :class:`pandas.HDFStore` object.

    key : object, optional
        The group identifier in the store. Can be omitted if the HDF file
        contains a single pandas object.
    mode : {'r', 'r+', 'a'}, default 'r'
        Mode to use when opening the file. Ignored if path_or_buf is a
        :class:`pandas.HDFStore`. Default is 'r'.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    where : list, optional
        A list of Term (or convertible) objects.
    start : int, optional
        Row number to start selection.
    stop  : int, optional
        Row number to stop selection.
    columns : list, optional
        A list of columns names to return.
    iterator : bool, optional
        Return an iterator object.
    chunksize : int, optional
        Number of rows to include in an iteration when using an iterator.
    **kwargs
        Additional keyword arguments passed to HDFStore.

    Returns
    -------
    object
        The selected object. Return type depends on the object stored.

    See Also
    --------
    DataFrame.to_hdf : Write a HDF file from a DataFrame.
    HDFStore : Low-level access to HDF files.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 1.0, 'a']], columns=['x', 'y', 'z'])  # doctest: +SKIP
    >>> df.to_hdf('./store.h5', 'data')  # doctest: +SKIP
    >>> reread = pd.read_hdf('./store.h5')  # doctest: +SKIP
    """
    if mode not in ["r", "r+", "a"]:
        raise ValueError(
            f"mode {mode} is not allowed while performing a read. "
            f"Allowed modes are r, r+ and a."
        )
    # grab the scope
    if where is not None:
        where = _ensure_term(where, scope_level=1)

    if isinstance(path_or_buf, HDFStore):
        if not path_or_buf.is_open:
            raise OSError("The HDFStore must be open for reading.")

        store = path_or_buf
        auto_close = False
    else:
        path_or_buf = stringify_path(path_or_buf)
        if not isinstance(path_or_buf, str):
            raise NotImplementedError(
                "Support for generic buffers has not been implemented."
            )
        try:
            exists = os.path.exists(path_or_buf)

        # if filepath is too long
        except (TypeError, ValueError):
            exists = False

        if not exists:
            raise FileNotFoundError(f"File {path_or_buf} does not exist")

        store = HDFStore(path_or_buf, mode=mode, errors=errors, **kwargs)
        # can't auto open/close if we are using an iterator
        # so delegate to the iterator
        auto_close = True

    try:
        if key is None:
            groups = store.groups()
            if len(groups) == 0:
                raise ValueError(
                    "Dataset(s) incompatible with Pandas data types, "
                    "not table, or no datasets found in HDF5 file."
                )
            candidate_only_group = groups[0]

            # For the HDF file to have only one dataset, all other groups
            # should then be metadata groups for that candidate group. (This
            # assumes that the groups() method enumerates parent groups
            # before their children.)
            for group_to_check in groups[1:]:
                if not _is_metadata_of(group_to_check, candidate_only_group):
                    raise ValueError(
                        "key must be provided when HDF5 "
                        "file contains multiple datasets."
                    )
            key = candidate_only_group._v_pathname
        return store.select(
            key,
            where=where,
            start=start,
            stop=stop,
            columns=columns,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )
    except (ValueError, TypeError, LookupError):
        if not isinstance(path_or_buf, HDFStore):
            # if there is an error, close the store if we opened it.
            with suppress(AttributeError):
                store.close()

        raise


def _is_metadata_of(group: Node, parent_group: Node) -> bool:
    """Check if a given group is a metadata group for a given parent_group."""
    if group._v_depth <= parent_group._v_depth:
        return False

    current = group
    while current._v_depth > 1:
        parent = current._v_parent
        if parent == parent_group and current._v_name == "meta":
            return True
        current = current._v_parent
    return False


class HDFStore:
    """
    Dict-like IO interface for storing pandas objects in PyTables.

    Either Fixed or Table format.

    .. warning::

       Pandas uses PyTables for reading and writing HDF5 files, which allows
       serializing object-dtype data with pickle when using the "fixed" format.
       Loading pickled data received from untrusted sources can be unsafe.

       See: https://docs.python.org/3/library/pickle.html for more.

    Parameters
    ----------
    path : str
        File path to HDF5 file.
    mode : {'a', 'w', 'r', 'r+'}, default 'a'

        ``'r'``
            Read-only; no data can be modified.
        ``'w'``
            Write; a new file is created (an existing file with the same
            name would be deleted).
        ``'a'``
            Append; an existing file is opened for reading and writing,
            and if the file does not exist it is created.
        ``'r+'``
            It is similar to ``'a'``, but the file must already exist.
    complevel : int, 0-9, default None
        Specifies a compression level for data.
        A value of 0 or None disables compression.
    complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
        Specifies the compression library to be used.
        These additional compressors for Blosc are supported
        (default if no compressor specified: 'blosc:blosclz'):
        {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
         'blosc:zlib', 'blosc:zstd'}.
        Specifying a compression library which is not available issues
        a ValueError.
    fletcher32 : bool, default False
        If applying compression use the fletcher32 checksum.
    **kwargs
        These parameters will be passed to the PyTables open_file method.

    Examples
    --------
    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore('test.h5')
    >>> store['foo'] = bar   # write to HDF5
    >>> bar = store['foo']   # retrieve
    >>> store.close()

    **Create or load HDF5 file in-memory**

    When passing the `driver` option to the PyTables open_file method through
    **kwargs, the HDF5 file is loaded or created in-memory and will only be
    written when closed:

    >>> bar = pd.DataFrame(np.random.randn(10, 4))
    >>> store = pd.HDFStore('test.h5', driver='H5FD_CORE')
    >>> store['foo'] = bar
    >>> store.close()   # only now, data is written to disk
    """

    _handle: File | None
    _mode: str

    def __init__(
        self,
        path,
        mode: str = "a",
        complevel: int | None = None,
        complib=None,
        fletcher32: bool = False,
        **kwargs,
    ) -> None:
        if "format" in kwargs:
            raise ValueError("format is not a defined argument for HDFStore")

        tables = import_optional_dependency("tables")

        if complib is not None and complib not in tables.filters.all_complibs:
            raise ValueError(
                f"complib only supports {tables.filters.all_complibs} compression."
            )

        if complib is None and complevel is not None:
            complib = tables.filters.default_complib

        self._path = stringify_path(path)
        if mode is None:
            mode = "a"
        self._mode = mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        self.open(mode=mode, **kwargs)

    def __fspath__(self) -> str:
        return self._path

    @property
    def root(self):
        """return the root node"""
        self._check_if_open()
        assert self._handle is not None  # for mypy
        return self._handle.root

    @property
    def filename(self) -> str:
        return self._path

    def __getitem__(self, key: str):
        return self.get(key)

    def __setitem__(self, key: str, value) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        return self.remove(key)

    def __getattr__(self, name: str):
        """allow attribute access to get stores"""
        try:
            return self.get(name)
        except (KeyError, ClosedFileError):
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __contains__(self, key: str) -> bool:
        """
        check for existence of this key
        can match the exact pathname or the pathnm w/o the leading '/'
        """
        node = self.get_node(key)
        if node is not None:
            name = node._v_pathname
            if key in (name, name[1:]):
                return True
        return False

    def __len__(self) -> int:
        return len(self.groups())

    def __repr__(self) -> str:
        pstr = pprint_thing(self._path)
        return f"{type(self)}\nFile path: {pstr}\n"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def keys(self, include: str = "pandas") -> list[str]:
        """
        Return a list of keys corresponding to objects stored in HDFStore.

        Parameters
        ----------

        include : str, default 'pandas'
                When kind equals 'pandas' return pandas objects.
                When kind equals 'native' return native HDF5 Table objects.

        Returns
        -------
        list
            List of ABSOLUTE path-names (e.g. have the leading '/').

        Raises
        ------
        raises ValueError if kind has an illegal value

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df)  # doctest: +SKIP
        >>> store.get('data')  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.close()  # doctest: +SKIP
        """
        if include == "pandas":
            return [n._v_pathname for n in self.groups()]

        elif include == "native":
            assert self._handle is not None  # mypy
            return [
                n._v_pathname for n in self._handle.walk_nodes("/", classname="Table")
            ]
        raise ValueError(
            f"`include` should be either 'pandas' or 'native' but is '{include}'"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def items(self) -> Iterator[tuple[str, list]]:
        """
        iterate on key->group
        """
        for g in self.groups():
            yield g._v_pathname, g

    def open(self, mode: str = "a", **kwargs) -> None:
        """
        Open the file in the specified mode

        Parameters
        ----------
        mode : {'a', 'w', 'r', 'r+'}, default 'a'
            See HDFStore docstring or tables.open_file for info about modes
        **kwargs
            These parameters will be passed to the PyTables open_file method.
        """
        tables = _tables()

        if self._mode != mode:
            # if we are changing a write mode to read, ok
            if self._mode in ["a", "w"] and mode in ["r", "r+"]:
                pass
            elif mode in ["w"]:
                # this would truncate, raise here
                if self.is_open:
                    raise PossibleDataLossError(
                        f"Re-opening the file [{self._path}] with mode [{self._mode}] "
                        "will delete the current file!"
                    )

            self._mode = mode

        # close and reopen the handle
        if self.is_open:
            self.close()

        if self._complevel and self._complevel > 0:
            self._filters = _tables().Filters(
                self._complevel, self._complib, fletcher32=self._fletcher32
            )

        if _table_file_open_policy_is_strict and self.is_open:
            msg = (
                "Cannot open HDF5 file, which is already opened, "
                "even in read-only mode."
            )
            raise ValueError(msg)

        self._handle = tables.open_file(self._path, self._mode, **kwargs)

    def close(self) -> None:
        """
        Close the PyTables file handle
        """
        if self._handle is not None:
            self._handle.close()
        self._handle = None

    @property
    def is_open(self) -> bool:
        """
        return a boolean indicating whether the file is open
        """
        if self._handle is None:
            return False
        return bool(self._handle.isopen)

    def flush(self, fsync: bool = False) -> None:
        """
        Force all buffered modifications to be written to disk.

        Parameters
        ----------
        fsync : bool (default False)
          call ``os.fsync()`` on the file handle to force writing to disk.

        Notes
        -----
        Without ``fsync=True``, flushing may not guarantee that the OS writes
        to disk. With fsync, the operation will block until the OS claims the
        file has been written; however, other caching layers may still
        interfere.
        """
        if self._handle is not None:
            self._handle.flush()
            if fsync:
                with suppress(OSError):
                    os.fsync(self._handle.fileno())

    def get(self, key: str):
        """
        Retrieve pandas object stored in file.

        Parameters
        ----------
        key : str

        Returns
        -------
        object
            Same type as object stored in file.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df)  # doctest: +SKIP
        >>> store.get('data')  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        with patch_pickle():
            # GH#31167 Without this patch, pickle doesn't know how to unpickle
            #  old DateOffset objects now that they are cdef classes.
            group = self.get_node(key)
            if group is None:
                raise KeyError(f"No object named {key} in the file")
            return self._read_group(group)

    def select(
        self,
        key: str,
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ):
        """
        Retrieve pandas object stored in file, optionally based on where criteria.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
            Object being retrieved from file.
        where : list or None
            List of Term (or convertible) objects, optional.
        start : int or None
            Row number to start selection.
        stop : int, default None
            Row number to stop selection.
        columns : list or None
            A list of columns that if not None, will limit the return columns.
        iterator : bool or False
            Returns an iterator.
        chunksize : int or None
            Number or rows to include in iteration, return an iterator.
        auto_close : bool or False
            Should automatically close the store when finished.

        Returns
        -------
        object
            Retrieved object from file.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df)  # doctest: +SKIP
        >>> store.get('data')  # doctest: +SKIP
        >>> print(store.keys())  # doctest: +SKIP
        ['/data1', '/data2']
        >>> store.select('/data1')  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        >>> store.select('/data1', where='columns == A')  # doctest: +SKIP
           A
        0  1
        1  3
        >>> store.close()  # doctest: +SKIP
        """
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        # create the storer and axes
        where = _ensure_term(where, scope_level=1)
        s = self._create_storer(group)
        s.infer_axes()

        # function to call on iteration
        def func(_start, _stop, _where):
            return s.read(start=_start, stop=_stop, where=_where, columns=columns)

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=s.nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result()

    def select_as_coordinates(
        self,
        key: str,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        return the selection as an Index

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.


        Parameters
        ----------
        key : str
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        """
        where = _ensure_term(where, scope_level=1)
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_coordinates with a table")
        return tbl.read_coordinates(where=where, start=start, stop=stop)

    def select_column(
        self,
        key: str,
        column: str,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        return a single column from the table. This is generally only useful to
        select an indexable

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        key : str
        column : str
            The column of interest.
        start : int or None, default None
        stop : int or None, default None

        Raises
        ------
        raises KeyError if the column is not found (or key is not a valid
            store)
        raises ValueError if the column can not be extracted individually (it
            is part of a data block)

        """
        tbl = self.get_storer(key)
        if not isinstance(tbl, Table):
            raise TypeError("can only read_column with a table")
        return tbl.read_column(column=column, start=start, stop=stop)

    def select_as_multiple(
        self,
        keys,
        where=None,
        selector=None,
        columns=None,
        start=None,
        stop=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ):
        """
        Retrieve pandas objects from multiple tables.

        .. warning::

           Pandas uses PyTables for reading and writing HDF5 files, which allows
           serializing object-dtype data with pickle when using the "fixed" format.
           Loading pickled data received from untrusted sources can be unsafe.

           See: https://docs.python.org/3/library/pickle.html for more.

        Parameters
        ----------
        keys : a list of the tables
        selector : the table to apply the where criteria (defaults to keys[0]
            if not supplied)
        columns : the columns I want back
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection
        iterator : bool, return an iterator, default False
        chunksize : nrows to include in iteration, return an iterator
        auto_close : bool, default False
            Should automatically close the store when finished.

        Raises
        ------
        raises KeyError if keys or selector is not found or keys is empty
        raises TypeError if keys is not a list or tuple
        raises ValueError if the tables are not ALL THE SAME DIMENSIONS
        """
        # default to single select
        where = _ensure_term(where, scope_level=1)
        if isinstance(keys, (list, tuple)) and len(keys) == 1:
            keys = keys[0]
        if isinstance(keys, str):
            return self.select(
                key=keys,
                where=where,
                columns=columns,
                start=start,
                stop=stop,
                iterator=iterator,
                chunksize=chunksize,
                auto_close=auto_close,
            )

        if not isinstance(keys, (list, tuple)):
            raise TypeError("keys must be a list/tuple")

        if not len(keys):
            raise ValueError("keys must have a non-zero length")

        if selector is None:
            selector = keys[0]

        # collect the tables
        tbls = [self.get_storer(k) for k in keys]
        s = self.get_storer(selector)

        # validate rows
        nrows = None
        for t, k in itertools.chain([(s, selector)], zip(tbls, keys)):
            if t is None:
                raise KeyError(f"Invalid table [{k}]")
            if not t.is_table:
                raise TypeError(
                    f"object [{t.pathname}] is not a table, and cannot be used in all "
                    "select as multiple"
                )

            if nrows is None:
                nrows = t.nrows
            elif t.nrows != nrows:
                raise ValueError("all tables must have exactly the same nrows!")

        # The isinstance checks here are redundant with the check above,
        #  but necessary for mypy; see GH#29757
        _tbls = [x for x in tbls if isinstance(x, Table)]

        # axis is the concentration axes
        axis = {t.non_index_axes[0][0] for t in _tbls}.pop()

        def func(_start, _stop, _where):
            # retrieve the objs, _where is always passed as a set of
            # coordinates here
            objs = [
                t.read(where=_where, columns=columns, start=_start, stop=_stop)
                for t in tbls
            ]

            # concat and return
            return concat(objs, axis=axis, verify_integrity=False)._consolidate()

        # create the iterator
        it = TableIterator(
            self,
            s,
            func,
            where=where,
            nrows=nrows,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            auto_close=auto_close,
        )

        return it.get_result(coordinates=True)

    def put(
        self,
        key: str,
        value: DataFrame | Series,
        format=None,
        index: bool = True,
        append: bool = False,
        complib=None,
        complevel: int | None = None,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        data_columns: Literal[True] | list[str] | None = None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
        dropna: bool = False,
    ) -> None:
        """
        Store object in HDFStore.

        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : 'fixed(f)|table(t)', default is 'fixed'
            Format to use when storing object in HDFStore. Value can be one of:

            ``'fixed'``
                Fixed format.  Fast writing/reading. Not-appendable, nor searchable.
            ``'table'``
                Table format.  Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        index : bool, default True
            Write DataFrame index as a column.
        append : bool, default False
            This will force Table format, append the input data to the existing.
        data_columns : list of columns or True, default None
            List of columns to create as data columns, or True to use all columns.
            See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        encoding : str, default None
            Provide an encoding for strings.
        track_times : bool, default True
            Parameter is propagated to 'create_table' method of 'PyTables'.
            If set to False it enables to have the same h5 files (same hashes)
            independent on creation time.
        dropna : bool, default False, optional
            Remove missing values.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df)  # doctest: +SKIP
        """
        if format is None:
            format = get_option("io.hdf.default_format") or "fixed"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
            track_times=track_times,
            dropna=dropna,
        )

    def remove(self, key: str, where=None, start=None, stop=None) -> None:
        """
        Remove pandas object partially by specifying the where condition

        Parameters
        ----------
        key : str
            Node to remove or delete rows from
        where : list of Term (or convertible) objects, optional
        start : integer (defaults to None), row number to start selection
        stop  : integer (defaults to None), row number to stop selection

        Returns
        -------
        number of rows removed (or None if not a Table)

        Raises
        ------
        raises KeyError if key is not a valid store

        """
        where = _ensure_term(where, scope_level=1)
        try:
            s = self.get_storer(key)
        except KeyError:
            # the key is not a valid store, re-raising KeyError
            raise
        except AssertionError:
            # surface any assertion errors for e.g. debugging
            raise
        except Exception as err:
            # In tests we get here with ClosedFileError, TypeError, and
            #  _table_mod.NoSuchNodeError.  TODO: Catch only these?

            if where is not None:
                raise ValueError(
                    "trying to remove a node with a non-None where clause!"
                ) from err

            # we are actually trying to remove a node (with children)
            node = self.get_node(key)
            if node is not None:
                node._f_remove(recursive=True)
                return None

        # remove the node
        if com.all_none(where, start, stop):
            s.group._f_remove(recursive=True)

        # delete from the table
        else:
            if not s.is_table:
                raise ValueError(
                    "can only remove with where on objects written as tables"
                )
            return s.delete(where=where, start=start, stop=stop)

    def append(
        self,
        key: str,
        value: DataFrame | Series,
        format=None,
        axes=None,
        index: bool | list[str] = True,
        append: bool = True,
        complib=None,
        complevel: int | None = None,
        columns=None,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool | None = None,
        data_columns: Literal[True] | list[str] | None = None,
        encoding=None,
        errors: str = "strict",
    ) -> None:
        """
        Append to Table in file.

        Node must already exist and be Table format.

        Parameters
        ----------
        key : str
        value : {Series, DataFrame}
        format : 'table' is the default
            Format to use when storing object in HDFStore.  Value can be one of:

            ``'table'``
                Table format. Write as a PyTables Table structure which may perform
                worse but allow more flexible operations like searching / selecting
                subsets of the data.
        index : bool, default True
            Write DataFrame index as a column.
        append       : bool, default True
            Append the input data to the existing.
        data_columns : list of columns, or True, default None
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. See `here
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#query-via-data-columns>`__.
        min_itemsize : dict of columns that specify minimum str sizes
        nan_rep      : str to use as str nan representation
        chunksize    : size to chunk the writing
        expectedrows : expected TOTAL row size of this table
        encoding     : default None, provide an encoding for str
        dropna : bool, default False, optional
            Do not write an ALL nan row to the store settable
            by the option 'io.hdf.dropna_table'.

        Notes
        -----
        Does *not* check if data being appended overlaps with existing
        data in the table, so be careful

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df1, format='table')  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
        >>> store.append('data', df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8
        """
        if columns is not None:
            raise TypeError(
                "columns is not a supported keyword in append, try data_columns"
            )

        if dropna is None:
            dropna = get_option("io.hdf.dropna_table")
        if format is None:
            format = get_option("io.hdf.default_format") or "table"
        format = self._validate_format(format)
        self._write_to_group(
            key,
            value,
            format=format,
            axes=axes,
            index=index,
            append=append,
            complib=complib,
            complevel=complevel,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            data_columns=data_columns,
            encoding=encoding,
            errors=errors,
        )

    def append_to_multiple(
        self,
        d: dict,
        value,
        selector,
        data_columns=None,
        axes=None,
        dropna: bool = False,
        **kwargs,
    ) -> None:
        """
        Append to multiple tables

        Parameters
        ----------
        d : a dict of table_name to table_columns, None is acceptable as the
            values of one node (this will get all the remaining columns)
        value : a pandas object
        selector : a string that designates the indexable table; all of its
            columns will be designed as data_columns, unless data_columns is
            passed, in which case these are used
        data_columns : list of columns to create as data columns, or True to
            use all columns
        dropna : if evaluates to True, drop rows from all tables if any single
                 row in each table has all NaN. Default False.

        Notes
        -----
        axes parameter is currently not accepted

        """
        if axes is not None:
            raise TypeError(
                "axes is currently not accepted as a parameter to append_to_multiple; "
                "you can create the tables independently instead"
            )

        if not isinstance(d, dict):
            raise ValueError(
                "append_to_multiple must have a dictionary specified as the "
                "way to split the value"
            )

        if selector not in d:
            raise ValueError(
                "append_to_multiple requires a selector that is in passed dict"
            )

        # figure out the splitting axis (the non_index_axis)
        axis = next(iter(set(range(value.ndim)) - set(_AXES_MAP[type(value)])))

        # figure out how to split the value
        remain_key = None
        remain_values: list = []
        for k, v in d.items():
            if v is None:
                if remain_key is not None:
                    raise ValueError(
                        "append_to_multiple can only have one value in d that is None"
                    )
                remain_key = k
            else:
                remain_values.extend(v)
        if remain_key is not None:
            ordered = value.axes[axis]
            ordd = ordered.difference(Index(remain_values))
            ordd = sorted(ordered.get_indexer(ordd))
            d[remain_key] = ordered.take(ordd)

        # data_columns
        if data_columns is None:
            data_columns = d[selector]

        # ensure rows are synchronized across the tables
        if dropna:
            idxs = (value[cols].dropna(how="all").index for cols in d.values())
            valid_index = next(idxs)
            for index in idxs:
                valid_index = valid_index.intersection(index)
            value = value.loc[valid_index]

        min_itemsize = kwargs.pop("min_itemsize", None)

        # append
        for k, v in d.items():
            dc = data_columns if k == selector else None

            # compute the val
            val = value.reindex(v, axis=axis)

            filtered = (
                {key: value for (key, value) in min_itemsize.items() if key in v}
                if min_itemsize is not None
                else None
            )
            self.append(k, val, data_columns=dc, min_itemsize=filtered, **kwargs)

    def create_table_index(
        self,
        key: str,
        columns=None,
        optlevel: int | None = None,
        kind: str | None = None,
    ) -> None:
        """
        Create a pytables index on the table.

        Parameters
        ----------
        key : str
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError: raises if the node is not a table
        """
        # version requirements
        _tables()
        s = self.get_storer(key)
        if s is None:
            return

        if not isinstance(s, Table):
            raise TypeError("cannot create table index on a Fixed format store")
        s.create_index(columns=columns, optlevel=optlevel, kind=kind)

    def groups(self) -> list:
        """
        Return a list of all the top-level nodes.

        Each node returned is not a pandas storage object.

        Returns
        -------
        list
            List of objects.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df)  # doctest: +SKIP
        >>> print(store.groups())  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        [/data (Group) ''
          children := ['axis0' (Array), 'axis1' (Array), 'block0_values' (Array),
          'block0_items' (Array)]]
        """
        _tables()
        self._check_if_open()
        assert self._handle is not None  # for mypy
        assert _table_mod is not None  # for mypy
        return [
            g
            for g in self._handle.walk_groups()
            if (
                not isinstance(g, _table_mod.link.Link)
                and (
                    getattr(g._v_attrs, "pandas_type", None)
                    or getattr(g, "table", None)
                    or (isinstance(g, _table_mod.table.Table) and g._v_name != "table")
                )
            )
        ]

    def walk(self, where: str = "/") -> Iterator[tuple[str, list[str], list[str]]]:
        """
        Walk the pytables group hierarchy for pandas objects.

        This generator will yield the group path, subgroups and pandas object
        names for each group.

        Any non-pandas PyTables objects that are not a group will be ignored.

        The `where` group itself is listed first (preorder), then each of its
        child groups (following an alphanumerical order) is also traversed,
        following the same procedure.

        Parameters
        ----------
        where : str, default "/"
            Group where to start walking.

        Yields
        ------
        path : str
            Full path to a group (without trailing '/').
        groups : list
            Names (strings) of the groups contained in `path`.
        leaves : list
            Names (strings) of the pandas objects contained in `path`.

        Examples
        --------
        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df1, format='table')  # doctest: +SKIP
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
        >>> store.append('data', df2)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        >>> for group in store.walk():  # doctest: +SKIP
        ...     print(group)  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        """
        _tables()
        self._check_if_open()
        assert self._handle is not None  # for mypy
        assert _table_mod is not None  # for mypy

        for g in self._handle.walk_groups(where):
            if getattr(g._v_attrs, "pandas_type", None) is not None:
                continue

            groups = []
            leaves = []
            for child in g._v_children.values():
                pandas_type = getattr(child._v_attrs, "pandas_type", None)
                if pandas_type is None:
                    if isinstance(child, _table_mod.group.Group):
                        groups.append(child._v_name)
                else:
                    leaves.append(child._v_name)

            yield (g._v_pathname.rstrip("/"), groups, leaves)

    def get_node(self, key: str) -> Node | None:
        """return the node with the key or None if it does not exist"""
        self._check_if_open()
        if not key.startswith("/"):
            key = "/" + key

        assert self._handle is not None
        assert _table_mod is not None  # for mypy
        try:
            node = self._handle.get_node(self.root, key)
        except _table_mod.exceptions.NoSuchNodeError:
            return None

        assert isinstance(node, _table_mod.Node), type(node)
        return node

    def get_storer(self, key: str) -> GenericFixed | Table:
        """return the storer object for a key, raise if not in the file"""
        group = self.get_node(key)
        if group is None:
            raise KeyError(f"No object named {key} in the file")

        s = self._create_storer(group)
        s.infer_axes()
        return s

    def copy(
        self,
        file,
        mode: str = "w",
        propindexes: bool = True,
        keys=None,
        complib=None,
        complevel: int | None = None,
        fletcher32: bool = False,
        overwrite: bool = True,
    ) -> HDFStore:
        """
        Copy the existing store to a new file, updating in place.

        Parameters
        ----------
        propindexes : bool, default True
            Restore indexes in copied file.
        keys : list, optional
            List of keys to include in the copy (defaults to all).
        overwrite : bool, default True
            Whether to overwrite (remove and replace) existing nodes in the new store.
        mode, complib, complevel, fletcher32 same as in HDFStore.__init__

        Returns
        -------
        open file handle of the new store
        """
        new_store = HDFStore(
            file, mode=mode, complib=complib, complevel=complevel, fletcher32=fletcher32
        )
        if keys is None:
            keys = list(self.keys())
        if not isinstance(keys, (tuple, list)):
            keys = [keys]
        for k in keys:
            s = self.get_storer(k)
            if s is not None:
                if k in new_store:
                    if overwrite:
                        new_store.remove(k)

                data = self.select(k)
                if isinstance(s, Table):
                    index: bool | list[str] = False
                    if propindexes:
                        index = [a.name for a in s.axes if a.is_indexed]
                    new_store.append(
                        k,
                        data,
                        index=index,
                        data_columns=getattr(s, "data_columns", None),
                        encoding=s.encoding,
                    )
                else:
                    new_store.put(k, data, encoding=s.encoding)

        return new_store

    def info(self) -> str:
        """
        Print detailed information on the store.

        Returns
        -------
        str

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
        >>> store = pd.HDFStore("store.h5", 'w')  # doctest: +SKIP
        >>> store.put('data', df)  # doctest: +SKIP
        >>> print(store.info())  # doctest: +SKIP
        >>> store.close()  # doctest: +SKIP
        <class 'pandas.io.pytables.HDFStore'>
        File path: store.h5
        /data    frame    (shape->[2,2])
        """
        path = pprint_thing(self._path)
        output = f"{type(self)}\nFile path: {path}\n"

        if self.is_open:
            lkeys = sorted(self.keys())
            if len(lkeys):
                keys = []
                values = []

                for k in lkeys:
                    try:
                        s = self.get_storer(k)
                        if s is not None:
                            keys.append(pprint_thing(s.pathname or k))
                            values.append(pprint_thing(s or "invalid_HDFStore node"))
                    except AssertionError:
                        # surface any assertion errors for e.g. debugging
                        raise
                    except Exception as detail:
                        keys.append(k)
                        dstr = pprint_thing(detail)
                        values.append(f"[invalid_HDFStore node: {dstr}]")

                output += adjoin(12, keys, values)
            else:
                output += "Empty"
        else:
            output += "File is CLOSED"

        return output

    # ------------------------------------------------------------------------
    # private methods

    def _check_if_open(self) -> None:
        if not self.is_open:
            raise ClosedFileError(f"{self._path} file is not open!")

    def _validate_format(self, format: str) -> str:
        """validate / deprecate formats"""
        # validate
        try:
            format = _FORMAT_MAP[format.lower()]
        except KeyError as err:
            raise TypeError(f"invalid HDFStore format specified [{format}]") from err

        return format

    def _create_storer(
        self,
        group,
        format=None,
        value: DataFrame | Series | None = None,
        encoding: str = "UTF-8",
        errors: str = "strict",
    ) -> GenericFixed | Table:
        """return a suitable class to operate"""
        cls: type[GenericFixed | Table]

        if value is not None and not isinstance(value, (Series, DataFrame)):
            raise TypeError("value must be None, Series, or DataFrame")

        pt = _ensure_decoded(getattr(group._v_attrs, "pandas_type", None))
        tt = _ensure_decoded(getattr(group._v_attrs, "table_type", None))

        # infer the pt from the passed value
        if pt is None:
            if value is None:
                _tables()
                assert _table_mod is not None  # for mypy
                if getattr(group, "table", None) or isinstance(
                    group, _table_mod.table.Table
                ):
                    pt = "frame_table"
                    tt = "generic_table"
                else:
                    raise TypeError(
                        "cannot create a storer if the object is not existing "
                        "nor a value are passed"
                    )
            else:
                if isinstance(value, Series):
                    pt = "series"
                else:
                    pt = "frame"

                # we are actually a table
                if format == "table":
                    pt += "_table"

        # a storer node
        if "table" not in pt:
            _STORER_MAP = {"series": SeriesFixed, "frame": FrameFixed}
            try:
                cls = _STORER_MAP[pt]
            except KeyError as err:
                raise TypeError(
                    f"cannot properly create the storer for: [_STORER_MAP] [group->"
                    f"{group},value->{type(value)},format->{format}"
                ) from err
            return cls(self, group, encoding=encoding, errors=errors)

        # existing node (and must be a table)
        if tt is None:
            # if we are a writer, determine the tt
            if value is not None:
                if pt == "series_table":
                    index = getattr(value, "index", None)
                    if index is not None:
                        if index.nlevels == 1:
                            tt = "appendable_series"
                        elif index.nlevels > 1:
                            tt = "appendable_multiseries"
                elif pt == "frame_table":
                    index = getattr(value, "index", None)
                    if index is not None:
                        if index.nlevels == 1:
                            tt = "appendable_frame"
                        elif index.nlevels > 1:
                            tt = "appendable_multiframe"

        _TABLE_MAP = {
            "generic_table": GenericTable,
            "appendable_series": AppendableSeriesTable,
            "appendable_multiseries": AppendableMultiSeriesTable,
            "appendable_frame": AppendableFrameTable,
            "appendable_multiframe": AppendableMultiFrameTable,
            "worm": WORMTable,
        }
        try:
            cls = _TABLE_MAP[tt]
        except KeyError as err:
            raise TypeError(
                f"cannot properly create the storer for: [_TABLE_MAP] [group->"
                f"{group},value->{type(value)},format->{format}"
            ) from err

        return cls(self, group, encoding=encoding, errors=errors)

    def _write_to_group(
        self,
        key: str,
        value: DataFrame | Series,
        format,
        axes=None,
        index: bool | list[str] = True,
        append: bool = False,
        complib=None,
        complevel: int | None = None,
        fletcher32=None,
        min_itemsize: int | dict[str, int] | None = None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool = False,
        nan_rep=None,
        data_columns=None,
        encoding=None,
        errors: str = "strict",
        track_times: bool = True,
    ) -> None:
        # we don't want to store a table node at all if our object is 0-len
        # as there are not dtypes
        if getattr(value, "empty", None) and (format == "table" or append):
            return

        group = self._identify_group(key, append)

        s = self._create_storer(group, format, value, encoding=encoding, errors=errors)
        if append:
            # raise if we are trying to append to a Fixed format,
            #       or a table that exists (and we are putting)
            if not s.is_table or (s.is_table and format == "fixed" and s.is_exists):
                raise ValueError("Can only append to Tables")
            if not s.is_exists:
                s.set_object_info()
        else:
            s.set_object_info()

        if not s.is_table and complib:
            raise ValueError("Compression not supported on Fixed format stores")

        # write the object
        s.write(
            obj=value,
            axes=axes,
            append=append,
            complib=complib,
            complevel=complevel,
            fletcher32=fletcher32,
            min_itemsize=min_itemsize,
            chunksize=chunksize,
            expectedrows=expectedrows,
            dropna=dropna,
            nan_rep=nan_rep,
            data_columns=data_columns,
            track_times=track_times,
        )

        if isinstance(s, Table) and index:
            s.create_index(columns=index)

    def _read_group(self, group: Node):
        s = self._create_storer(group)
        s.infer_axes()
        return s.read()

    def _identify_group(self, key: str, append: bool) -> Node:
        """Identify HDF5 group based on key, delete/create group if needed."""
        group = self.get_node(key)

        # we make this assertion for mypy; the get_node call will already
        # have raised if this is incorrect
        assert self._handle is not None

        # remove the node if we are not appending
        if group is not None and not append:
            self._handle.remove_node(group, recursive=True)
            group = None

        if group is None:
            group = self._create_nodes_and_group(key)

        return group

    def _create_nodes_and_group(self, key: str) -> Node:
        """Create nodes from key and return group name."""
        # assertion for mypy
        assert self._handle is not None

        paths = key.split("/")
        # recursively create the groups
        path = "/"
        for p in paths:
            if not len(p):
                continue
            new_path = path
            if not path.endswith("/"):
                new_path += "/"
            new_path += p
            group = self.get_node(new_path)
            if group is None:
                group = self._handle.create_group(path, p)
            path = new_path
        return group


class TableIterator:
    """
    Define the iteration interface on a table

    Parameters
    ----------
    store : HDFStore
    s     : the referred storer
    func  : the function to execute the query
    where : the where of the query
    nrows : the rows to iterate on
    start : the passed start value (default is None)
    stop  : the passed stop value (default is None)
    iterator : bool, default False
        Whether to use the default iterator.
    chunksize : the passed chunking value (default is 100000)
    auto_close : bool, default False
        Whether to automatically close the store at the end of iteration.
    """

    chunksize: int | None
    store: HDFStore
    s: GenericFixed | Table

    def __init__(
        self,
        store: HDFStore,
        s: GenericFixed | Table,
        func,
        where,
        nrows,
        start=None,
        stop=None,
        iterator: bool = False,
        chunksize: int | None = None,
        auto_close: bool = False,
    ) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where

        # set start/stop if they are not set if we are a table
        if self.s.is_table:
            if nrows is None:
                nrows = 0
            if start is None:
                start = 0
            if stop is None:
                stop = nrows
            stop = min(nrows, stop)

        self.nrows = nrows
        self.start = start
        self.stop = stop

        self.coordinates = None
        if iterator or chunksize is not None:
            if chunksize is None:
                chunksize = 100000
            self.chunksize = int(chunksize)
        else:
            self.chunksize = None

        self.auto_close = auto_close

    def __iter__(self) -> Iterator:
        # iterate
        current = self.start
        if self.coordinates is None:
            raise ValueError("Cannot iterate until get_result is called.")
        while current < self.stop:
            stop = min(current + self.chunksize, self.stop)
            value = self.func(None, None, self.coordinates[current:stop])
            current = stop
            if value is None or not len(value):
                continue

            yield value

        self.close()

    def close(self) -> None:
        if self.auto_close:
            self.store.close()

    def get_result(self, coordinates: bool = False):
        #  return the actual iterator
        if self.chunksize is not None:
            if not isinstance(self.s, Table):
                raise TypeError("can only use an iterator or chunksize on a table")

            self.coordinates = self.s.read_coordinates(where=self.where)

            return self

        # if specified read via coordinates (necessary for multiple selections
        if coordinates:
            if not isinstance(self.s, Table):
                raise TypeError("can only read_coordinates on a table")
            where = self.s.read_coordinates(
                where=self.where, start=self.start, stop=self.stop
            )
        else:
            where = self.where

        # directly return the result
        results = self.func(self.start, self.stop, where)
        self.close()
        return results


class IndexCol:
    """
    an index column description class

    Parameters
    ----------
    axis   : axis which I reference
    values : the ndarray like converted values
    kind   : a string description of this type
    typ    : the pytables type
    pos    : the position in the pytables

    """

    is_an_indexable: bool = True
    is_data_indexable: bool = True
    _info_fields = ["freq", "tz", "index_name"]

    def __init__(
        self,
        name: str,
        values=None,
        kind=None,
        typ=None,
        cname: str | None = None,
        axis=None,
        pos=None,
        freq=None,
        tz=None,
        index_name=None,
        ordered=None,
        table=None,
        meta=None,
        metadata=None,
    ) -> None:
        if not isinstance(name, str):
            raise ValueError("`name` must be a str.")

        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata

        if pos is not None:
            self.set_pos(pos)

        # These are ensured as long as the passed arguments match the
        #  constructor annotations.
        assert isinstance(self.name, str)
        assert isinstance(self.cname, str)

    @property
    def itemsize(self) -> int:
        # Assumes self.typ has already been initialized
        return self.typ.itemsize

    @property
    def kind_attr(self) -> str:
        return f"{self.name}_kind"

    def set_pos(self, pos: int) -> None:
        """set the position of this column in the Table"""
        self.pos = pos
        if pos is not None and self.typ is not None:
            self.typ._v_pos = pos

    def __repr__(self) -> str:
        temp = tuple(
            map(pprint_thing, (self.name, self.cname, self.axis, self.pos, self.kind))
        )
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(["name", "cname", "axis", "pos", "kind"], temp)
            ]
        )

    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "axis", "pos"]
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    @property
    def is_indexed(self) -> bool:
        """return whether I am an indexed column"""
        if not hasattr(self.table, "cols"):
            # e.g. if infer hasn't been called yet, self.table will be None.
            return False
        return getattr(self.table.cols, self.cname).is_indexed

    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ) -> tuple[np.ndarray, np.ndarray] | tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.
        """
        assert isinstance(values, np.ndarray), type(values)

        # values is a recarray
        if values.dtype.fields is not None:
            # Copy, otherwise values will be a view
            # preventing the original recarry from being free'ed
            values = values[self.cname].copy()

        val_kind = _ensure_decoded(self.kind)
        values = _maybe_convert(values, val_kind, encoding, errors)
        kwargs = {}
        kwargs["name"] = _ensure_decoded(self.index_name)

        if self.freq is not None:
            kwargs["freq"] = _ensure_decoded(self.freq)

        factory: type[Index | DatetimeIndex] = Index
        if lib.is_np_dtype(values.dtype, "M") or isinstance(
            values.dtype, DatetimeTZDtype
        ):
            factory = DatetimeIndex
        elif values.dtype == "i8" and "freq" in kwargs:
            # PeriodIndex data is stored as i8
            # error: Incompatible types in assignment (expression has type
            # "Callable[[Any, KwArg(Any)], PeriodIndex]", variable has type
            # "Union[Type[Index], Type[DatetimeIndex]]")
            factory = lambda x, **kwds: PeriodIndex.from_ordinals(  # type: ignore[assignment]
                x, freq=kwds.get("freq", None)
            )._rename(
                kwds["name"]
            )

        # making an Index instance could throw a number of different errors
        try:
            new_pd_index = factory(values, **kwargs)
        except ValueError:
            # if the output freq is different that what we recorded,
            # it should be None (see also 'doc example part 2')
            if "freq" in kwargs:
                kwargs["freq"] = None
            new_pd_index = factory(values, **kwargs)
        final_pd_index = _set_tz(new_pd_index, self.tz)
        return final_pd_index, final_pd_index

    def take_data(self):
        """return the values"""
        return self.values

    @property
    def attrs(self):
        return self.table._v_attrs

    @property
    def description(self):
        return self.table.description

    @property
    def col(self):
        """return my current col description"""
        return getattr(self.description, self.cname, None)

    @property
    def cvalues(self):
        """return my cython values"""
        return self.values

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def maybe_set_size(self, min_itemsize=None) -> None:
        """
        maybe set a string col itemsize:
            min_itemsize can be an integer or a dict with this columns name
            with an integer size
        """
        if _ensure_decoded(self.kind) == "string":
            if isinstance(min_itemsize, dict):
                min_itemsize = min_itemsize.get(self.name)

            if min_itemsize is not None and self.typ.itemsize < min_itemsize:
                self.typ = _tables().StringCol(itemsize=min_itemsize, pos=self.pos)

    def validate_names(self) -> None:
        pass

    def validate_and_set(self, handler: AppendableTable, append: bool) -> None:
        self.table = handler.table
        self.validate_col()
        self.validate_attr(append)
        self.validate_metadata(handler)
        self.write_metadata(handler)
        self.set_attr()

    def validate_col(self, itemsize=None):
        """validate this column: return the compared against itemsize"""
        # validate this column for string truncation (or reset to the max size)
        if _ensure_decoded(self.kind) == "string":
            c = self.col
            if c is not None:
                if itemsize is None:
                    itemsize = self.itemsize
                if c.itemsize < itemsize:
                    raise ValueError(
                        f"Trying to store a string with len [{itemsize}] in "
                        f"[{self.cname}] column but\nthis column has a limit of "
                        f"[{c.itemsize}]!\nConsider using min_itemsize to "
                        "preset the sizes on these columns"
                    )
                return c.itemsize

        return None

    def validate_attr(self, append: bool) -> None:
        # check for backwards incompatibility
        if append:
            existing_kind = getattr(self.attrs, self.kind_attr, None)
            if existing_kind is not None and existing_kind != self.kind:
                raise TypeError(
                    f"incompatible kind in col [{existing_kind} - {self.kind}]"
                )

    def update_info(self, info) -> None:
        """
        set/update the info for this indexable with the key/value
        if there is a conflict raise/warn as needed
        """
        for key in self._info_fields:
            value = getattr(self, key, None)
            idx = info.setdefault(self.name, {})

            existing_value = idx.get(key)
            if key in idx and value is not None and existing_value != value:
                # frequency/name just warn
                if key in ["freq", "index_name"]:
                    ws = attribute_conflict_doc % (key, existing_value, value)
                    warnings.warn(
                        ws, AttributeConflictWarning, stacklevel=find_stack_level()
                    )

                    # reset
                    idx[key] = None
                    setattr(self, key, None)

                else:
                    raise ValueError(
                        f"invalid info for [{self.name}] for [{key}], "
                        f"existing_value [{existing_value}] conflicts with "
                        f"new value [{value}]"
                    )
            elif value is not None or existing_value is not None:
                idx[key] = value

    def set_info(self, info) -> None:
        """set my state from the passed info"""
        idx = info.get(self.name)
        if idx is not None:
            self.__dict__.update(idx)

    def set_attr(self) -> None:
        """set the kind for this column"""
        setattr(self.attrs, self.kind_attr, self.kind)

    def validate_metadata(self, handler: AppendableTable) -> None:
        """validate that kind=category does not change the categories"""
        if self.meta == "category":
            new_metadata = self.metadata
            cur_metadata = handler.read_metadata(self.cname)
            if (
                new_metadata is not None
                and cur_metadata is not None
                and not array_equivalent(
                    new_metadata, cur_metadata, strict_nan=True, dtype_equal=True
                )
            ):
                raise ValueError(
                    "cannot append a categorical with "
                    "different categories to the existing"
                )

    def write_metadata(self, handler: AppendableTable) -> None:
        """set the meta data"""
        if self.metadata is not None:
            handler.write_metadata(self.cname, self.metadata)


class GenericIndexCol(IndexCol):
    """an index which is not represented in the data of the table"""

    @property
    def is_indexed(self) -> bool:
        return False

    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ) -> tuple[Index, Index]:
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep : str
        encoding : str
        errors : str
        """
        assert isinstance(values, np.ndarray), type(values)

        index = RangeIndex(len(values))
        return index, index

    def set_attr(self) -> None:
        pass


class DataCol(IndexCol):
    """
    a data holding column, by definition this is not indexable

    Parameters
    ----------
    data   : the actual data
    cname  : the column name in the table to hold the data (typically
                values)
    meta   : a string description of the metadata
    metadata : the actual metadata
    """

    is_an_indexable = False
    is_data_indexable = False
    _info_fields = ["tz", "ordered"]

    def __init__(
        self,
        name: str,
        values=None,
        kind=None,
        typ=None,
        cname: str | None = None,
        pos=None,
        tz=None,
        ordered=None,
        table=None,
        meta=None,
        metadata=None,
        dtype: DtypeArg | None = None,
        data=None,
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            pos=pos,
            cname=cname,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )
        self.dtype = dtype
        self.data = data

    @property
    def dtype_attr(self) -> str:
        return f"{self.name}_dtype"

    @property
    def meta_attr(self) -> str:
        return f"{self.name}_meta"

    def __repr__(self) -> str:
        temp = tuple(
            map(
                pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)
            )
        )
        return ",".join(
            [
                f"{key}->{value}"
                for key, value in zip(["name", "cname", "dtype", "kind", "shape"], temp)
            ]
        )

    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
        return all(
            getattr(self, a, None) == getattr(other, a, None)
            for a in ["name", "cname", "dtype", "pos"]
        )

    def set_data(self, data: ArrayLike) -> None:
        assert data is not None
        assert self.dtype is None

        data, dtype_name = _get_data_and_dtype_name(data)

        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def take_data(self):
        """return the data"""
        return self.data

    @classmethod
    def _get_atom(cls, values: ArrayLike) -> Col:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        dtype = values.dtype
        # error: Item "ExtensionDtype" of "Union[ExtensionDtype, dtype[Any]]" has no
        # attribute "itemsize"
        itemsize = dtype.itemsize  # type: ignore[union-attr]

        shape = values.shape
        if values.ndim == 1:
            # EA, use block shape pretending it is 2D
            # TODO(EA2D): not necessary with 2D EAs
            shape = (1, values.size)

        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            atom = cls.get_atom_datetime64(shape)
        elif lib.is_np_dtype(dtype, "m"):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)

        return atom

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]:
        """return the PyTables column class for this column"""
        if kind.startswith("uint"):
            k4 = kind[4:]
            col_name = f"UInt{k4}Col"
        elif kind.startswith("period"):
            # we store as integer
            col_name = "Int64Col"
        else:
            kcap = kind.capitalize()
            col_name = f"{kcap}Col"

        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape):
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape):
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self):
        return getattr(self.data, "shape", None)

    @property
    def cvalues(self):
        """return my cython values"""
        return self.data

    def validate_attr(self, append) -> None:
        """validate that we have the same order as the existing & same dtype"""
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError("appended items do not match existing items in table!")

            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError(
                    "appended items dtype do not match existing items dtype in table!"
                )

    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str):
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep :
        encoding : str
        errors : str

        Returns
        -------
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
        assert isinstance(values, np.ndarray), type(values)

        # values is a recarray
        if values.dtype.fields is not None:
            values = values[self.cname]

        assert self.typ is not None
        if self.dtype is None:
            # Note: in tests we never have timedelta64 or datetime64,
            #  so the _get_data_and_dtype_name may be unnecessary
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype
            kind = self.kind

        assert isinstance(converted, np.ndarray)  # for mypy

        # use the meta if needed
        meta = _ensure_decoded(self.meta)
        metadata = self.metadata
        ordered = self.ordered
        tz = self.tz

        assert dtype_name is not None
        # convert to the correct dtype
        dtype = _ensure_decoded(dtype_name)

        # reverse converts
        if dtype.startswith("datetime64"):
            # recreate with tz if indicated
            converted = _set_tz(converted, tz, coerce=True)

        elif dtype == "timedelta64":
            converted = np.asarray(converted, dtype="m8[ns]")
        elif dtype == "date":
            try:
                converted = np.asarray(
                    [date.fromordinal(v) for v in converted], dtype=object
                )
            except ValueError:
                converted = np.asarray(
                    [date.fromtimestamp(v) for v in converted], dtype=object
                )

        elif meta == "category":
            # we have a categorical
            categories = metadata
            codes = converted.ravel()

            # if we have stored a NaN in the categories
            # then strip it; in theory we could have BOTH
            # -1s in the codes and nulls :<
            if categories is None:
                # Handle case of NaN-only categorical columns in which case
                # the categories are an empty array; when this is stored,
                # pytables cannot write a zero-len array, so on readback
                # the categories would be None and `read_hdf()` would fail.
                categories = Index([], dtype=np.float64)
            else:
                mask = isna(categories)
                if mask.any():
                    categories = categories[~mask]
                    codes[codes != -1] -= mask.astype(int).cumsum()._values

            converted = Categorical.from_codes(
                codes, categories=categories, ordered=ordered, validate=False
            )

        else:
            try:
                converted = converted.astype(dtype, copy=False)
            except TypeError:
                converted = converted.astype("O", copy=False)

        # convert nans / decode
        if _ensure_decoded(kind) == "string":
            converted = _unconvert_string_array(
                converted, nan_rep=nan_rep, encoding=encoding, errors=errors
            )

        return self.values, converted

    def set_attr(self) -> None:
        """set the data for this column"""
        setattr(self.attrs, self.kind_attr, self.values)
        setattr(self.attrs, self.meta_attr, self.meta)
        assert self.dtype is not None
        setattr(self.attrs, self.dtype_attr, self.dtype)


class DataIndexableCol(DataCol):
    """represent a data column that can be indexed"""

    is_data_indexable = True

    def validate_names(self) -> None:
        if not is_string_dtype(Index(self.values).dtype):
            # TODO: should the message here be more specifically non-str?
            raise ValueError("cannot have non-object label DataIndexableCol")

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        return _tables().StringCol(itemsize=itemsize)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        return cls.get_atom_coltype(kind=kind)()

    @classmethod
    def get_atom_datetime64(cls, shape):
        return _tables().Int64Col()

    @classmethod
    def get_atom_timedelta64(cls, shape):
        return _tables().Int64Col()


class GenericDataIndexableCol(DataIndexableCol):
    """represent a generic pytables data column"""


class Fixed:
    """
    represent an object in my store
    facilitate read/write of various types of objects
    this is an abstract base class

    Parameters
    ----------
    parent : HDFStore
    group : Node
        The group node where the table resides.
    """

    pandas_kind: str
    format_type: str = "fixed"  # GH#30962 needed by dask
    obj_type: type[DataFrame | Series]
    ndim: int
    parent: HDFStore
    is_table: bool = False

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: str | None = "UTF-8",
        errors: str = "strict",
    ) -> None:
        assert isinstance(parent, HDFStore), type(parent)
        assert _table_mod is not None  # needed for mypy
        assert isinstance(group, _table_mod.Node), type(group)
        self.parent = parent
        self.group = group
        self.encoding = _ensure_encoding(encoding)
        self.errors = errors

    @property
    def is_old_version(self) -> bool:
        return self.version[0] <= 0 and self.version[1] <= 10 and self.version[2] < 1

    @property
    def version(self) -> tuple[int, int, int]:
        """compute and set our version"""
        version = _ensure_decoded(getattr(self.group._v_attrs, "pandas_version", None))
        try:
            version = tuple(int(x) for x in version.split("."))
            if len(version) == 2:
                version = version + (0,)
        except AttributeError:
            version = (0, 0, 0)
        return version

    @property
    def pandas_type(self):
        return _ensure_decoded(getattr(self.group._v_attrs, "pandas_type", None))

    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        self.infer_axes()
        s = self.shape
        if s is not None:
            if isinstance(s, (list, tuple)):
                jshape = ",".join([pprint_thing(x) for x in s])
                s = f"[{jshape}]"
            return f"{self.pandas_type:12.12} (shape->{s})"
        return self.pandas_type

    def set_object_info(self) -> None:
        """set my pandas type & version"""
        self.attrs.pandas_type = str(self.pandas_kind)
        self.attrs.pandas_version = str(_version)

    def copy(self) -> Fixed:
        new_self = copy.copy(self)
        return new_self

    @property
    def shape(self):
        return self.nrows

    @property
    def pathname(self):
        return self.group._v_pathname

    @property
    def _handle(self):
        return self.parent._handle

    @property
    def _filters(self):
        return self.parent._filters

    @property
    def _complevel(self) -> int:
        return self.parent._complevel

    @property
    def _fletcher32(self) -> bool:
        return self.parent._fletcher32

    @property
    def attrs(self):
        return self.group._v_attrs

    def set_attrs(self) -> None:
        """set our object attributes"""

    def get_attrs(self) -> None:
        """get our object attributes"""

    @property
    def storable(self):
        """return my storable"""
        return self.group

    @property
    def is_exists(self) -> bool:
        return False

    @property
    def nrows(self):
        return getattr(self.storable, "nrows", None)

    def validate(self, other) -> Literal[True] | None:
        """validate against an existing storable"""
        if other is None:
            return None
        return True

    def validate_version(self, where=None) -> None:
        """are we trying to operate on an old version?"""

    def infer_axes(self) -> bool:
        """
        infer the axes of my storer
        return a boolean indicating if we have a valid storer or not
        """
        s = self.storable
        if s is None:
            return False
        self.get_attrs()
        return True

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        raise NotImplementedError(
            "cannot read on an abstract storer: subclasses should implement"
        )

    def write(self, obj, **kwargs) -> None:
        raise NotImplementedError(
            "cannot write on an abstract storer: subclasses should implement"
        )

    def delete(
        self, where=None, start: int | None = None, stop: int | None = None
    ) -> None:
        """
        support fully deleting the node in its entirety (only) - where
        specification must be None
        """
        if com.all_none(where, start, stop):
            self._handle.remove_node(self.group, recursive=True)
            return None

        raise TypeError("cannot delete on an abstract storer")


class GenericFixed(Fixed):
    """a generified fixed version"""

    _index_type_map = {DatetimeIndex: "datetime", PeriodIndex: "period"}
    _reverse_index_map = {v: k for k, v in _index_type_map.items()}
    attributes: list[str] = []

    # indexer helpers
    def _class_to_alias(self, cls) -> str:
        return self._index_type_map.get(cls, "")

    def _alias_to_class(self, alias):
        if isinstance(alias, type):  # pragma: no cover
            # compat: for a short period of time master stored types
            return alias
        return self._reverse_index_map.get(alias, Index)

    def _get_index_factory(self, attrs):
        index_class = self._alias_to_class(
            _ensure_decoded(getattr(attrs, "index_class", ""))
        )

        factory: Callable

        if index_class == DatetimeIndex:

            def f(values, freq=None, tz=None):
                # data are already in UTC, localize and convert if tz present
                dta = DatetimeArray._simple_new(
                    values.values, dtype=values.dtype, freq=freq
                )
                result = DatetimeIndex._simple_new(dta, name=None)
                if tz is not None:
                    result = result.tz_localize("UTC").tz_convert(tz)
                return result

            factory = f
        elif index_class == PeriodIndex:

            def f(values, freq=None, tz=None):
                dtype = PeriodDtype(freq)
                parr = PeriodArray._simple_new(values, dtype=dtype)
                return PeriodIndex._simple_new(parr, name=None)

            factory = f
        else:
            factory = index_class

        kwargs = {}
        if "freq" in attrs:
            kwargs["freq"] = attrs["freq"]
            if index_class is Index:
                # DTI/PI would be gotten by _alias_to_class
                factory = TimedeltaIndex

        if "tz" in attrs:
            if isinstance(attrs["tz"], bytes):
                # created by python2
                kwargs["tz"] = attrs["tz"].decode("utf-8")
            else:
                # created by python3
                kwargs["tz"] = attrs["tz"]
            assert index_class is DatetimeIndex  # just checking

        return factory, kwargs

    def validate_read(self, columns, where) -> None:
        """
        raise if any keywords are passed which are not-None
        """
        if columns is not None:
            raise TypeError(
                "cannot pass a column specification when reading "
                "a Fixed format store. this store must be selected in its entirety"
            )
        if where is not None:
            raise TypeError(
                "cannot pass a where specification when reading "
                "from a Fixed format store. this store must be selected in its entirety"
            )

    @property
    def is_exists(self) -> bool:
        return True

    def set_attrs(self) -> None:
        """set our object attributes"""
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors

    def get_attrs(self) -> None:
        """retrieve our attributes"""
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        self.errors = _ensure_decoded(getattr(self.attrs, "errors", "strict"))
        for n in self.attributes:
            setattr(self, n, _ensure_decoded(getattr(self.attrs, n, None)))

    def write(self, obj, **kwargs) -> None:
        self.set_attrs()

    def read_array(self, key: str, start: int | None = None, stop: int | None = None):
        """read an array for the specified node (off of group"""
        import tables

        node = getattr(self.group, key)
        attrs = node._v_attrs

        transposed = getattr(attrs, "transposed", False)

        if isinstance(node, tables.VLArray):
            ret = node[0][start:stop]
            dtype = getattr(attrs, "value_type", None)
            if dtype is not None:
                ret = pd_array(ret, dtype=dtype)
        else:
            dtype = _ensure_decoded(getattr(attrs, "value_type", None))
            shape = getattr(attrs, "shape", None)

            if shape is not None:
                # length 0 axis
                ret = np.empty(shape, dtype=dtype)
            else:
                ret = node[start:stop]

            if dtype and dtype.startswith("datetime64"):
                # reconstruct a timezone if indicated
                tz = getattr(attrs, "tz", None)
                ret = _set_tz(ret, tz, coerce=True)

            elif dtype == "timedelta64":
                ret = np.asarray(ret, dtype="m8[ns]")

        if transposed:
            return ret.T
        else:
            return ret

    def read_index(
        self, key: str, start: int | None = None, stop: int | None = None
    ) -> Index:
        variety = _ensure_decoded(getattr(self.attrs, f"{key}_variety"))

        if variety == "multi":
            return self.read_multi_index(key, start=start, stop=stop)
        elif variety == "regular":
            node = getattr(self.group, key)
            index = self.read_index_node(node, start=start, stop=stop)
            return index
        else:  # pragma: no cover
            raise TypeError(f"unrecognized index variety: {variety}")

    def write_index(self, key: str, index: Index) -> None:
        if isinstance(index, MultiIndex):
            setattr(self.attrs, f"{key}_variety", "multi")
            self.write_multi_index(key, index)
        else:
            setattr(self.attrs, f"{key}_variety", "regular")
            converted = _convert_index("index", index, self.encoding, self.errors)

            self.write_array(key, converted.values)

            node = getattr(self.group, key)
            node._v_attrs.kind = converted.kind
            node._v_attrs.name = index.name

            if isinstance(index, (DatetimeIndex, PeriodIndex)):
                node._v_attrs.index_class = self._class_to_alias(type(index))

            if isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex)):
                node._v_attrs.freq = index.freq

            if isinstance(index, DatetimeIndex) and index.tz is not None:
                node._v_attrs.tz = _get_tz(index.tz)

    def write_multi_index(self, key: str, index: MultiIndex) -> None:
        setattr(self.attrs, f"{key}_nlevels", index.nlevels)

        for i, (lev, level_codes, name) in enumerate(
            zip(index.levels, index.codes, index.names)
        ):
            # write the level
            if isinstance(lev.dtype, ExtensionDtype):
                raise NotImplementedError(
                    "Saving a MultiIndex with an extension dtype is not supported."
                )
            level_key = f"{key}_level{i}"
            conv_level = _convert_index(level_key, lev, self.encoding, self.errors)
            self.write_array(level_key, conv_level.values)
            node = getattr(self.group, level_key)
            node._v_attrs.kind = conv_level.kind
            node._v_attrs.name = name

            # write the name
            setattr(node._v_attrs, f"{key}_name{name}", name)

            # write the labels
            label_key = f"{key}_label{i}"
            self.write_array(label_key, level_codes)

    def read_multi_index(
        self, key: str, start: int | None = None, stop: int | None = None
    ) -> MultiIndex:
        nlevels = getattr(self.attrs, f"{key}_nlevels")

        levels = []
        codes = []
        names: list[Hashable] = []
        for i in range(nlevels):
            level_key = f"{key}_level{i}"
            node = getattr(self.group, level_key)
            lev = self.read_index_node(node, start=start, stop=stop)
            levels.append(lev)
            names.append(lev.name)

            label_key = f"{key}_label{i}"
            level_codes = self.read_array(label_key, start=start, stop=stop)
            codes.append(level_codes)

        return MultiIndex(
            levels=levels, codes=codes, names=names, verify_integrity=True
        )

    def read_index_node(
        self, node: Node, start: int | None = None, stop: int | None = None
    ) -> Index:
        data = node[start:stop]
        # If the index was an empty array write_array_empty() will
        # have written a sentinel. Here we replace it with the original.
        if "shape" in node._v_attrs and np.prod(node._v_attrs.shape) == 0:
            data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
        kind = _ensure_decoded(node._v_attrs.kind)
        name = None

        if "name" in node._v_attrs:
            name = _ensure_str(node._v_attrs.name)
            name = _ensure_decoded(name)

        attrs = node._v_attrs
        factory, kwargs = self._get_index_factory(attrs)

        if kind in ("date", "object"):
            index = factory(
                _unconvert_index(
                    data, kind, encoding=self.encoding, errors=self.errors
                ),
                dtype=object,
                **kwargs,
            )
        else:
            index = factory(
                _unconvert_index(
                    data, kind, encoding=self.encoding, errors=self.errors
                ),
                **kwargs,
            )

        index.name = name

        return index

    def write_array_empty(self, key: str, value: ArrayLike) -> None:
        """write a 0-len array"""
        # ugly hack for length 0 axes
        arr = np.empty((1,) * value.ndim)
        self._handle.create_array(self.group, key, arr)
        node = getattr(self.group, key)
        node._v_attrs.value_type = str(value.dtype)
        node._v_attrs.shape = value.shape

    def write_array(
        self, key: str, obj: AnyArrayLike, items: Index | None = None
    ) -> None:
        # TODO: we only have a few tests that get here, the only EA
        #  that gets passed is DatetimeArray, and we never have
        #  both self._filters and EA

        value = extract_array(obj, extract_numpy=True)

        if key in self.group:
            self._handle.remove_node(self.group, key)

        # Transform needed to interface with pytables row/col notation
        empty_array = value.size == 0
        transposed = False

        if isinstance(value.dtype, CategoricalDtype):
            raise NotImplementedError(
                "Cannot store a category dtype in a HDF5 dataset that uses format="
                '"fixed". Use format="table".'
            )
        if not empty_array:
            if hasattr(value, "T"):
                # ExtensionArrays (1d) may not have transpose.
                value = value.T
                transposed = True

        atom = None
        if self._filters is not None:
            with suppress(ValueError):
                # get the atom for this datatype
                atom = _tables().Atom.from_dtype(value.dtype)

        if atom is not None:
            # We only get here if self._filters is non-None and
            #  the Atom.from_dtype call succeeded

            # create an empty chunked array and fill it from value
            if not empty_array:
                ca = self._handle.create_carray(
                    self.group, key, atom, value.shape, filters=self._filters
                )
                ca[:] = value

            else:
                self.write_array_empty(key, value)

        elif value.dtype.type == np.object_:
            # infer the type, warn if we have a non-string type here (for
            # performance)
            inferred_type = lib.infer_dtype(value, skipna=False)
            if empty_array:
                pass
            elif inferred_type == "string":
                pass
            else:
                ws = performance_doc % (inferred_type, key, items)
                warnings.warn(ws, PerformanceWarning, stacklevel=find_stack_level())

            vlarr = self._handle.create_vlarray(self.group, key, _tables().ObjectAtom())
            vlarr.append(value)

        elif lib.is_np_dtype(value.dtype, "M"):
            self._handle.create_array(self.group, key, value.view("i8"))
            getattr(self.group, key)._v_attrs.value_type = str(value.dtype)
        elif isinstance(value.dtype, DatetimeTZDtype):
            # store as UTC
            # with a zone

            # error: Item "ExtensionArray" of "Union[Any, ExtensionArray]" has no
            # attribute "asi8"
            self._handle.create_array(
                self.group, key, value.asi8  # type: ignore[union-attr]
            )

            node = getattr(self.group, key)
            # error: Item "ExtensionArray" of "Union[Any, ExtensionArray]" has no
            # attribute "tz"
            node._v_attrs.tz = _get_tz(value.tz)  # type: ignore[union-attr]
            node._v_attrs.value_type = f"datetime64[{value.dtype.unit}]"
        elif lib.is_np_dtype(value.dtype, "m"):
            self._handle.create_array(self.group, key, value.view("i8"))
            getattr(self.group, key)._v_attrs.value_type = "timedelta64"
        elif isinstance(value, BaseStringArray):
            vlarr = self._handle.create_vlarray(self.group, key, _tables().ObjectAtom())
            vlarr.append(value.to_numpy())
            node = getattr(self.group, key)
            node._v_attrs.value_type = str(value.dtype)
        elif empty_array:
            self.write_array_empty(key, value)
        else:
            self._handle.create_array(self.group, key, value)

        getattr(self.group, key)._v_attrs.transposed = transposed


class SeriesFixed(GenericFixed):
    pandas_kind = "series"
    attributes = ["name"]

    name: Hashable

    @property
    def shape(self):
        try:
            return (len(self.group.values),)
        except (TypeError, AttributeError):
            return None

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        self.validate_read(columns, where)
        index = self.read_index("index", start=start, stop=stop)
        values = self.read_array("values", start=start, stop=stop)
        result = Series(values, index=index, name=self.name, copy=False)
        if (
            using_string_dtype()
            and isinstance(values, np.ndarray)
            and is_string_array(values, skipna=True)
        ):
            result = result.astype(StringDtype(na_value=np.nan))
        return result

    def write(self, obj, **kwargs) -> None:
        super().write(obj, **kwargs)
        self.write_index("index", obj.index)
        self.write_array("values", obj)
        self.attrs.name = obj.name


class BlockManagerFixed(GenericFixed):
    attributes = ["ndim", "nblocks"]

    nblocks: int

    @property
    def shape(self) -> Shape | None:
        try:
            ndim = self.ndim

            # items
            items = 0
            for i in range(self.nblocks):
                node = getattr(self.group, f"block{i}_items")
                shape = getattr(node, "shape", None)
                if shape is not None:
                    items += shape[0]

            # data shape
            node = self.group.block0_values
            shape = getattr(node, "shape", None)
            if shape is not None:
                shape = list(shape[0 : (ndim - 1)])
            else:
                shape = []

            shape.append(items)

            return shape
        except AttributeError:
            return None

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> DataFrame:
        # start, stop applied to rows, so 0th axis only
        self.validate_read(columns, where)
        select_axis = self.obj_type()._get_block_manager_axis(0)

        axes = []
        for i in range(self.ndim):
            _start, _stop = (start, stop) if i == select_axis else (None, None)
            ax = self.read_index(f"axis{i}", start=_start, stop=_stop)
            axes.append(ax)

        items = axes[0]
        dfs = []

        for i in range(self.nblocks):
            blk_items = self.read_index(f"block{i}_items")
            values = self.read_array(f"block{i}_values", start=_start, stop=_stop)

            columns = items[items.get_indexer(blk_items)]
            df = DataFrame(values.T, columns=columns, index=axes[1], copy=False)
            if (
                using_string_dtype()
                and isinstance(values, np.ndarray)
                and is_string_array(values, skipna=True)
            ):
                df = df.astype(StringDtype(na_value=np.nan))
            dfs.append(df)

        if len(dfs) > 0:
            out = concat(dfs, axis=1, copy=True)
            if using_copy_on_write():
                # with CoW, concat ignores the copy keyword. Here, we still want
                # to copy to enforce optimized column-major layout
                out = out.copy()
            out = out.reindex(columns=items, copy=False)
            return out

        return DataFrame(columns=axes[0], index=axes[1])

    def write(self, obj, **kwargs) -> None:
        super().write(obj, **kwargs)

        # TODO(ArrayManager) HDFStore relies on accessing the blocks
        if isinstance(obj._mgr, ArrayManager):
            obj = obj._as_manager("block")

        data = obj._mgr
        if not data.is_consolidated():
            data = data.consolidate()

        self.attrs.ndim = data.ndim
        for i, ax in enumerate(data.axes):
            if i == 0 and (not ax.is_unique):
                raise ValueError("Columns index has to be unique for fixed format")
            self.write_index(f"axis{i}", ax)

        # Supporting mixed-type DataFrame objects...nontrivial
        self.attrs.nblocks = len(data.blocks)
        for i, blk in enumerate(data.blocks):
            # I have no idea why, but writing values before items fixed #2299
            blk_items = data.items.take(blk.mgr_locs)
            self.write_array(f"block{i}_values", blk.values, items=blk_items)
            self.write_index(f"block{i}_items", blk_items)


class FrameFixed(BlockManagerFixed):
    pandas_kind = "frame"
    obj_type = DataFrame


class Table(Fixed):
    """
    represent a table:
        facilitate read/write of various types of tables

    Attrs in Table Node
    -------------------
    These are attributes that are store in the main table node, they are
    necessary to recreate these tables when read back in.

    index_axes    : a list of tuples of the (original indexing axis and
        index column)
    non_index_axes: a list of tuples of the (original index axis and
        columns on a non-indexing axis)
    values_axes   : a list of the columns which comprise the data of this
        table
    data_columns  : a list of the columns that we are allowing indexing
        (these become single columns in values_axes)
    nan_rep       : the string to use for nan representations for string
        objects
    levels        : the names of levels
    metadata      : the names of the metadata columns
    """

    pandas_kind = "wide_table"
    format_type: str = "table"  # GH#30962 needed by dask
    table_type: str
    levels: int | list[Hashable] = 1
    is_table = True

    metadata: list

    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: str | None = None,
        errors: str = "strict",
        index_axes: list[IndexCol] | None = None,
        non_index_axes: list[tuple[AxisInt, Any]] | None = None,
        values_axes: list[DataCol] | None = None,
        data_columns: list | None = None,
        info: dict | None = None,
        nan_rep=None,
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes = index_axes or []
        self.non_index_axes = non_index_axes or []
        self.values_axes = values_axes or []
        self.data_columns = data_columns or []
        self.info = info or {}
        self.nan_rep = nan_rep

    @property
    def table_type_short(self) -> str:
        return self.table_type.split("_")[0]

    def __repr__(self) -> str:
        """return a pretty representation of myself"""
        self.infer_axes()
        jdc = ",".join(self.data_columns) if len(self.data_columns) else ""
        dc = f",dc->[{jdc}]"

        ver = ""
        if self.is_old_version:
            jver = ".".join([str(x) for x in self.version])
            ver = f"[{jver}]"

        jindex_axes = ",".join([a.name for a in self.index_axes])
        return (
            f"{self.pandas_type:12.12}{ver} "
            f"(typ->{self.table_type_short},nrows->{self.nrows},"
            f"ncols->{self.ncols},indexers->[{jindex_axes}]{dc})"
        )

    def __getitem__(self, c: str):
        """return the axis for c"""
        for a in self.axes:
            if c == a.name:
                return a
        return None

    def validate(self, other) -> None:
        """validate against an existing table"""
        if other is None:
            return

        if other.table_type != self.table_type:
            raise TypeError(
                "incompatible table_type with existing "
                f"[{other.table_type} - {self.table_type}]"
            )

        for c in ["index_axes", "non_index_axes", "values_axes"]:
            sv = getattr(self, c, None)
            ov = getattr(other, c, None)
            if sv != ov:
                # show the error for the specific axes
                # Argument 1 to "enumerate" has incompatible type
                # "Optional[Any]"; expected "Iterable[Any]"  [arg-type]
                for i, sax in enumerate(sv):  # type: ignore[arg-type]
                    # Value of type "Optional[Any]" is not indexable  [index]
                    oax = ov[i]  # type: ignore[index]
                    if sax != oax:
                        if c == "values_axes" and sax.kind != oax.kind:
                            raise ValueError(
                                f"Cannot serialize the column [{oax.values[0]}] "
                                f"because its data contents are not [{sax.kind}] "
                                f"but [{oax.kind}] object dtype"
                            )
                        raise ValueError(
                            f"invalid combination of [{c}] on appending data "
                            f"[{sax}] vs current table [{oax}]"
                        )

                # should never get here
                raise Exception(
                    f"invalid combination of [{c}] on appending data [{sv}] vs "
                    f"current table [{ov}]"
                )

    @property
    def is_multi_index(self) -> bool:
        """the levels attribute is 1 or a list in the case of a multi-index"""
        return isinstance(self.levels, list)

    def validate_multiindex(
        self, obj: DataFrame | Series
    ) -> tuple[DataFrame, list[Hashable]]:
        """
        validate that we can store the multi-index; reset and return the
        new object
        """
        levels = com.fill_missing_names(obj.index.names)
        try:
            reset_obj = obj.reset_index()
        except ValueError as err:
            raise ValueError(
                "duplicate names/columns in the multi-index when storing as a table"
            ) from err
        assert isinstance(reset_obj, DataFrame)  # for mypy
        return reset_obj, levels

    @property
    def nrows_expected(self) -> int:
        """based on our axes, compute the expected nrows"""
        return np.prod([i.cvalues.shape[0] for i in self.index_axes])

    @property
    def is_exists(self) -> bool:
        """has this table been created"""
        return "table" in self.group

    @property
    def storable(self):
        return getattr(self.group, "table", None)

    @property
    def table(self):
        """return the table group (this is my storable)"""
        return self.storable

    @property
    def dtype(self):
        return self.table.dtype

    @property
    def description(self):
        return self.table.description

    @property
    def axes(self) -> itertools.chain[IndexCol]:
        return itertools.chain(self.index_axes, self.values_axes)

    @property
    def ncols(self) -> int:
        """the number of total columns in the values axes"""
        return sum(len(a.values) for a in self.values_axes)

    @property
    def is_transposed(self) -> bool:
        return False

    @property
    def data_orientation(self) -> tuple[int, ...]:
        """return a tuple of my permutated axes, non_indexable at the front"""
        return tuple(
            itertools.chain(
                [int(a[0]) for a in self.non_index_axes],
                [int(a.axis) for a in self.index_axes],
            )
        )

    def queryables(self) -> dict[str, Any]:
        """return a dict of the kinds allowable columns for this object"""
        # mypy doesn't recognize DataFrame._AXIS_NAMES, so we re-write it here
        axis_names = {0: "index", 1: "columns"}

        # compute the values_axes queryables
        d1 = [(a.cname, a) for a in self.index_axes]
        d2 = [(axis_names[axis], None) for axis, values in self.non_index_axes]
        d3 = [
            (v.cname, v) for v in self.values_axes if v.name in set(self.data_columns)
        ]

        return dict(d1 + d2 + d3)

    def index_cols(self):
        """return a list of my index cols"""
        # Note: each `i.cname` below is assured to be a str.
        return [(i.axis, i.cname) for i in self.index_axes]

    def values_cols(self) -> list[str]:
        """return a list of my values cols"""
        return [i.cname for i in self.values_axes]

    def _get_metadata_path(self, key: str) -> str:
        """return the metadata pathname for this key"""
        group = self.group._v_pathname
        return f"{group}/meta/{key}/meta"

    def write_metadata(self, key: str, values: np.ndarray) -> None:
        """
        Write out a metadata array to the key as a fixed-format Series.

        Parameters
        ----------
        key : str
        values : ndarray
        """
        self.parent.put(
            self._get_metadata_path(key),
            Series(values, copy=False),
            format="table",
            encoding=self.encoding,
            errors=self.errors,
            nan_rep=self.nan_rep,
        )

    def read_metadata(self, key: str):
        """return the meta data array for this key"""
        if getattr(getattr(self.group, "meta", None), key, None) is not None:
            return self.parent.select(self._get_metadata_path(key))
        return None

    def set_attrs(self) -> None:
        """set our table type & indexables"""
        self.attrs.table_type = str(self.table_type)
        self.attrs.index_cols = self.index_cols()
        self.attrs.values_cols = self.values_cols()
        self.attrs.non_index_axes = self.non_index_axes
        self.attrs.data_columns = self.data_columns
        self.attrs.nan_rep = self.nan_rep
        self.attrs.encoding = self.encoding
        self.attrs.errors = self.errors
        self.attrs.levels = self.levels
        self.attrs.info = self.info

    def get_attrs(self) -> None:
        """retrieve our attributes"""
        self.non_index_axes = getattr(self.attrs, "non_index_axes", None) or []
        self.data_columns = getattr(self.attrs, "data_columns", None) or []
        self.info = getattr(self.attrs, "info", None) or {}
        self.nan_rep = getattr(self.attrs, "nan_rep", None)
        self.encoding = _ensure_encoding(getattr(self.attrs, "encoding", None))
        self.errors = _ensure_decoded(getattr(self.attrs, "errors", "strict"))
        self.levels: list[Hashable] = getattr(self.attrs, "levels", None) or []
        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]

    def validate_version(self, where=None) -> None:
        """are we trying to operate on an old version?"""
        if where is not None:
            if self.is_old_version:
                ws = incompatibility_doc % ".".join([str(x) for x in self.version])
                warnings.warn(
                    ws,
                    IncompatibilityWarning,
                    stacklevel=find_stack_level(),
                )

    def validate_min_itemsize(self, min_itemsize) -> None:
        """
        validate the min_itemsize doesn't contain items that are not in the
        axes this needs data_columns to be defined
        """
        if min_itemsize is None:
            return
        if not isinstance(min_itemsize, dict):
            return

        q = self.queryables()
        for k in min_itemsize:
            # ok, apply generally
            if k == "values":
                continue
            if k not in q:
                raise ValueError(
                    f"min_itemsize has the key [{k}] which is not an axis or "
                    "data_column"
                )

    @cache_readonly
    def indexables(self):
        """create/cache the indexables if they don't exist"""
        _indexables = []

        desc = self.description
        table_attrs = self.table.attrs

        # Note: each of the `name` kwargs below are str, ensured
        #  by the definition in index_cols.
        # index columns
        for i, (axis, name) in enumerate(self.attrs.index_cols):
            atom = getattr(desc, name)
            md = self.read_metadata(name)
            meta = "category" if md is not None else None

            kind_attr = f"{name}_kind"
            kind = getattr(table_attrs, kind_attr, None)

            index_col = IndexCol(
                name=name,
                axis=axis,
                pos=i,
                kind=kind,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            _indexables.append(index_col)

        # values columns
        dc = set(self.data_columns)
        base_pos = len(_indexables)

        def f(i, c):
            assert isinstance(c, str)
            klass = DataCol
            if c in dc:
                klass = DataIndexableCol

            atom = getattr(desc, c)
            adj_name = _maybe_adjust_name(c, self.version)

            # TODO: why kind_attr here?
            values = getattr(table_attrs, f"{adj_name}_kind", None)
            dtype = getattr(table_attrs, f"{adj_name}_dtype", None)
            # Argument 1 to "_dtype_to_kind" has incompatible type
            # "Optional[Any]"; expected "str"  [arg-type]
            kind = _dtype_to_kind(dtype)  # type: ignore[arg-type]

            md = self.read_metadata(c)
            # TODO: figure out why these two versions of `meta` dont always match.
            #  meta = "category" if md is not None else None
            meta = getattr(table_attrs, f"{adj_name}_meta", None)

            obj = klass(
                name=adj_name,
                cname=c,
                values=values,
                kind=kind,
                pos=base_pos + i,
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
                dtype=dtype,
            )
            return obj

        # Note: the definition of `values_cols` ensures that each
        #  `c` below is a str.
        _indexables.extend([f(i, c) for i, c in enumerate(self.attrs.values_cols)])

        return _indexables

    def create_index(
        self, columns=None, optlevel=None, kind: str | None = None
    ) -> None:
        """
        Create a pytables index on the specified columns.

        Parameters
        ----------
        columns : None, bool, or listlike[str]
            Indicate which columns to create an index on.

            * False : Do not create any indexes.
            * True : Create indexes on all columns.
            * None : Create indexes on all columns.
            * listlike : Create indexes on the given columns.

        optlevel : int or None, default None
            Optimization level, if None, pytables defaults to 6.
        kind : str or None, default None
            Kind of index, if None, pytables defaults to "medium".

        Raises
        ------
        TypeError if trying to create an index on a complex-type column.

        Notes
        -----
        Cannot index Time64Col or ComplexCol.
        Pytables must be >= 3.0.
        """
        if not self.infer_axes():
            return
        if columns is False:
            return

        # index all indexables and data_columns
        if columns is None or columns is True:
            columns = [a.cname for a in self.axes if a.is_data_indexable]
        if not isinstance(columns, (tuple, list)):
            columns = [columns]

        kw = {}
        if optlevel is not None:
            kw["optlevel"] = optlevel
        if kind is not None:
            kw["kind"] = kind

        table = self.table
        for c in columns:
            v = getattr(table.cols, c, None)
            if v is not None:
                # remove the index if the kind/optlevel have changed
                if v.is_indexed:
                    index = v.index
                    cur_optlevel = index.optlevel
                    cur_kind = index.kind

                    if kind is not None and cur_kind != kind:
                        v.remove_index()
                    else:
                        kw["kind"] = cur_kind

                    if optlevel is not None and cur_optlevel != optlevel:
                        v.remove_index()
                    else:
                        kw["optlevel"] = cur_optlevel

                # create the index
                if not v.is_indexed:
                    if v.type.startswith("complex"):
                        raise TypeError(
                            "Columns containing complex values can be stored but "
                            "cannot be indexed when using table format. Either use "
                            "fixed format, set index=False, or do not include "
                            "the columns containing complex values to "
                            "data_columns when initializing the table."
                        )
                    v.create_index(**kw)
            elif c in self.non_index_axes[0][1]:
                # GH 28156
                raise AttributeError(
                    f"column {c} is not a data_column.\n"
                    f"In order to read column {c} you must reload the dataframe \n"
                    f"into HDFStore and include {c} with the data_columns argument."
                )

    def _read_axes(
        self, where, start: int | None = None, stop: int | None = None
    ) -> list[tuple[np.ndarray, np.ndarray] | tuple[Index, Index]]:
        """
        Create the axes sniffed from the table.

        Parameters
        ----------
        where : ???
        start : int or None, default None
        stop : int or None, default None

        Returns
        -------
        List[Tuple[index_values, column_values]]
        """
        # create the selection
        selection = Selection(self, where=where, start=start, stop=stop)
        values = selection.select()

        results = []
        # convert the data
        for a in self.axes:
            a.set_info(self.info)
            res = a.convert(
                values,
                nan_rep=self.nan_rep,
                encoding=self.encoding,
                errors=self.errors,
            )
            results.append(res)

        return results

    @classmethod
    def get_object(cls, obj, transposed: bool):
        """return the data for this obj"""
        return obj

    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes):
        """
        take the input data_columns and min_itemize and create a data
        columns spec
        """
        if not len(non_index_axes):
            return []

        axis, axis_labels = non_index_axes[0]
        info = self.info.get(axis, {})
        if info.get("type") == "MultiIndex" and data_columns:
            raise ValueError(
                f"cannot use a multi-index on axis [{axis}] with "
                f"data_columns {data_columns}"
            )

        # evaluate the passed data_columns, True == use all columns
        # take only valid axis labels
        if data_columns is True:
            data_columns = list(axis_labels)
        elif data_columns is None:
            data_columns = []

        # if min_itemsize is a dict, add the keys (exclude 'values')
        if isinstance(min_itemsize, dict):
            existing_data_columns = set(data_columns)
            data_columns = list(data_columns)  # ensure we do not modify
            data_columns.extend(
                [
                    k
                    for k in min_itemsize.keys()
                    if k != "values" and k not in existing_data_columns
                ]
            )

        # return valid columns in the order of our axis
        return [c for c in data_columns if c in axis_labels]

    def _create_axes(
        self,
        axes,
        obj: DataFrame,
        validate: bool = True,
        nan_rep=None,
        data_columns=None,
        min_itemsize=None,
    ):
        """
        Create and return the axes.

        Parameters
        ----------
        axes: list or None
            The names or numbers of the axes to create.
        obj : DataFrame
            The object to create axes on.
        validate: bool, default True
            Whether to validate the obj against an existing object already written.
        nan_rep :
            A value to use for string column nan_rep.
        data_columns : List[str], True, or None, default None
            Specify the columns that we want to create to allow indexing on.

            * True : Use all available columns.
            * None : Use no columns.
            * List[str] : Use the specified columns.

        min_itemsize: Dict[str, int] or None, default None
            The min itemsize for a column in bytes.
        """
        if not isinstance(obj, DataFrame):
            group = self.group._v_name
            raise TypeError(
                f"cannot properly create the storer for: [group->{group},"
                f"value->{type(obj)}]"
            )

        # set the default axes if needed
        if axes is None:
            axes = [0]

        # map axes to numbers
        axes = [obj._get_axis_number(a) for a in axes]

        # do we have an existing table (if so, use its axes & data_columns)
        if self.infer_axes():
            table_exists = True
            axes = [a.axis for a in self.index_axes]
            data_columns = list(self.data_columns)
            nan_rep = self.nan_rep
            # TODO: do we always have validate=True here?
        else:
            table_exists = False

        new_info = self.info

        assert self.ndim == 2  # with next check, we must have len(axes) == 1
        # currently support on ndim-1 axes
        if len(axes) != self.ndim - 1:
            raise ValueError(
                "currently only support ndim-1 indexers in an AppendableTable"
            )

        # create according to the new data
        new_non_index_axes: list = []

        # nan_representation
        if nan_rep is None:
            nan_rep = "nan"

        # We construct the non-index-axis first, since that alters new_info
        idx = next(x for x in [0, 1] if x not in axes)

        a = obj.axes[idx]
        # we might be able to change the axes on the appending data if necessary
        append_axis = list(a)
        if table_exists:
            indexer = len(new_non_index_axes)  # i.e. 0
            exist_axis = self.non_index_axes[indexer][1]
            if not array_equivalent(
                np.array(append_axis),
                np.array(exist_axis),
                strict_nan=True,
                dtype_equal=True,
            ):
                # ahah! -> reindex
                if array_equivalent(
                    np.array(sorted(append_axis)),
                    np.array(sorted(exist_axis)),
                    strict_nan=True,
                    dtype_equal=True,
                ):
                    append_axis = exist_axis

        # the non_index_axes info
        info = new_info.setdefault(idx, {})
        info["names"] = list(a.names)
        info["type"] = type(a).__name__

        new_non_index_axes.append((idx, append_axis))

        # Now we can construct our new index axis
        idx = axes[0]
        a = obj.axes[idx]
        axis_name = obj._get_axis_name(idx)
        new_index = _convert_index(axis_name, a, self.encoding, self.errors)
        new_index.axis = idx

        # Because we are always 2D, there is only one new_index, so
        #  we know it will have pos=0
        new_index.set_pos(0)
        new_index.update_info(new_info)
        new_index.maybe_set_size(min_itemsize)  # check for column conflicts

        new_index_axes = [new_index]
        j = len(new_index_axes)  # i.e. 1
        assert j == 1

        # reindex by our non_index_axes & compute data_columns
        assert len(new_non_index_axes) == 1
        for a in new_non_index_axes:
            obj = _reindex_axis(obj, a[0], a[1])

        transposed = new_index.axis == 1

        # figure out data_columns and get out blocks
        data_columns = self.validate_data_columns(
            data_columns, min_itemsize, new_non_index_axes
        )

        frame = self.get_object(obj, transposed)._consolidate()

        blocks, blk_items = self._get_blocks_and_items(
            frame, table_exists, new_non_index_axes, self.values_axes, data_columns
        )

        # add my values
        vaxes = []
        for i, (blk, b_items) in enumerate(zip(blocks, blk_items)):
            # shape of the data column are the indexable axes
            klass = DataCol
            name = None

            # we have a data_column
            if data_columns and len(b_items) == 1 and b_items[0] in data_columns:
                klass = DataIndexableCol
                name = b_items[0]
                if not (name is None or isinstance(name, str)):
                    # TODO: should the message here be more specifically non-str?
                    raise ValueError("cannot have non-object label DataIndexableCol")

            # make sure that we match up the existing columns
            # if we have an existing table
            existing_col: DataCol | None

            if table_exists and validate:
                try:
                    existing_col = self.values_axes[i]
                except (IndexError, KeyError) as err:
                    raise ValueError(
                        f"Incompatible appended table [{blocks}]"
                        f"with existing table [{self.values_axes}]"
                    ) from err
            else:
                existing_col = None

            new_name = name or f"values_block_{i}"
            data_converted = _maybe_convert_for_string_atom(
                new_name,
                blk.values,
                existing_col=existing_col,
                min_itemsize=min_itemsize,
                nan_rep=nan_rep,
                encoding=self.encoding,
                errors=self.errors,
                columns=b_items,
            )
            adj_name = _maybe_adjust_name(new_name, self.version)

            typ = klass._get_atom(data_converted)
            kind = _dtype_to_kind(data_converted.dtype.name)
            tz = None
            if getattr(data_converted, "tz", None) is not None:
                tz = _get_tz(data_converted.tz)

            meta = metadata = ordered = None
            if isinstance(data_converted.dtype, CategoricalDtype):
                ordered = data_converted.ordered
                meta = "category"
                metadata = np.asarray(data_converted.categories).ravel()
            elif isinstance(blk.dtype, StringDtype):
                meta = str(blk.dtype)

            data, dtype_name = _get_data_and_dtype_name(data_converted)

            col = klass(
                name=adj_name,
                cname=new_name,
                values=list(b_items),
                typ=typ,
                pos=j,
                kind=kind,
                tz=tz,
                ordered=ordered,
                meta=meta,
                metadata=metadata,
                dtype=dtype_name,
                data=data,
            )
            col.update_info(new_info)

            vaxes.append(col)

            j += 1

        dcs = [col.name for col in vaxes if col.is_data_indexable]

        new_table = type(self)(
            parent=self.parent,
            group=self.group,
            encoding=self.encoding,
            errors=self.errors,
            index_axes=new_index_axes,
            non_index_axes=new_non_index_axes,
            values_axes=vaxes,
            data_columns=dcs,
            info=new_info,
            nan_rep=nan_rep,
        )
        if hasattr(self, "levels"):
            # TODO: get this into constructor, only for appropriate subclass
            new_table.levels = self.levels

        new_table.validate_min_itemsize(min_itemsize)

        if validate and table_exists:
            new_table.validate(self)

        return new_table

    @staticmethod
    def _get_blocks_and_items(
        frame: DataFrame,
        table_exists: bool,
        new_non_index_axes,
        values_axes,
        data_columns,
    ):
        # Helper to clarify non-state-altering parts of _create_axes

        # TODO(ArrayManager) HDFStore relies on accessing the blocks
        if isinstance(frame._mgr, ArrayManager):
            frame = frame._as_manager("block")

        def get_blk_items(mgr):
            return [mgr.items.take(blk.mgr_locs) for blk in mgr.blocks]

        mgr = frame._mgr
        mgr = cast(BlockManager, mgr)
        blocks: list[Block] = list(mgr.blocks)
        blk_items: list[Index] = get_blk_items(mgr)

        if len(data_columns):
            # TODO: prove that we only get here with axis == 1?
            #  It is the case in all extant tests, but NOT the case
            #  outside this `if len(data_columns)` check.

            axis, axis_labels = new_non_index_axes[0]
            new_labels = Index(axis_labels).difference(Index(data_columns))
            mgr = frame.reindex(new_labels, axis=axis)._mgr
            mgr = cast(BlockManager, mgr)

            blocks = list(mgr.blocks)
            blk_items = get_blk_items(mgr)
            for c in data_columns:
                # This reindex would raise ValueError if we had a duplicate
                #  index, so we can infer that (as long as axis==1) we
                #  get a single column back, so a single block.
                mgr = frame.reindex([c], axis=axis)._mgr
                mgr = cast(BlockManager, mgr)
                blocks.extend(mgr.blocks)
                blk_items.extend(get_blk_items(mgr))

        # reorder the blocks in the same order as the existing table if we can
        if table_exists:
            by_items = {
                tuple(b_items.tolist()): (b, b_items)
                for b, b_items in zip(blocks, blk_items)
            }
            new_blocks: list[Block] = []
            new_blk_items = []
            for ea in values_axes:
                items = tuple(ea.values)
                try:
                    b, b_items = by_items.pop(items)
                    new_blocks.append(b)
                    new_blk_items.append(b_items)
                except (IndexError, KeyError) as err:
                    jitems = ",".join([pprint_thing(item) for item in items])
                    raise ValueError(
                        f"cannot match existing table structure for [{jitems}] "
                        "on appending data"
                    ) from err
            blocks = new_blocks
            blk_items = new_blk_items

        return blocks, blk_items

    def process_axes(self, obj, selection: Selection, columns=None) -> DataFrame:
        """process axes filters"""
        # make a copy to avoid side effects
        if columns is not None:
            columns = list(columns)

        # make sure to include levels if we have them
        if columns is not None and self.is_multi_index:
            assert isinstance(self.levels, list)  # assured by is_multi_index
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)

        # reorder by any non_index_axes & limit to the select columns
        for axis, labels in self.non_index_axes:
            obj = _reindex_axis(obj, axis, labels, columns)

            def process_filter(field, filt, op):
                for axis_name in obj._AXIS_ORDERS:
                    axis_number = obj._get_axis_number(axis_name)
                    axis_values = obj._get_axis(axis_name)
                    assert axis_number is not None

                    # see if the field is the name of an axis
                    if field == axis_name:
                        # if we have a multi-index, then need to include
                        # the levels
                        if self.is_multi_index:
                            filt = filt.union(Index(self.levels))

                        takers = op(axis_values, filt)
                        return obj.loc(axis=axis_number)[takers]

                    # this might be the name of a file IN an axis
                    elif field in axis_values:
                        # we need to filter on this dimension
                        values = ensure_index(getattr(obj, field).values)
                        filt = ensure_index(filt)

                        # hack until we support reversed dim flags
                        if isinstance(obj, DataFrame):
                            axis_number = 1 - axis_number

                        takers = op(values, filt)
                        return obj.loc(axis=axis_number)[takers]

                raise ValueError(f"cannot find the field [{field}] for filtering!")

        # apply the selection filters (but keep in the same order)
        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                obj = process_filter(field, filt, op)

        return obj

    def create_description(
        self,
        complib,
        complevel: int | None,
        fletcher32: bool,
        expectedrows: int | None,
    ) -> dict[str, Any]:
        """create the description of the table from the axes & values"""
        # provided expected rows if its passed
        if expectedrows is None:
            expectedrows = max(self.nrows_expected, 10000)

        d = {"name": "table", "expectedrows": expectedrows}

        # description from the axes & values
        d["description"] = {a.cname: a.typ for a in self.axes}

        if complib:
            if complevel is None:
                complevel = self._complevel or 9
            filters = _tables().Filters(
                complevel=complevel,
                complib=complib,
                fletcher32=fletcher32 or self._fletcher32,
            )
            d["filters"] = filters
        elif self._filters is not None:
            d["filters"] = self._filters

        return d

    def read_coordinates(
        self, where=None, start: int | None = None, stop: int | None = None
    ):
        """
        select coordinates (row numbers) from a table; return the
        coordinates object
        """
        # validate the version
        self.validate_version(where)

        # infer the data kind
        if not self.infer_axes():
            return False

        # create the selection
        selection = Selection(self, where=where, start=start, stop=stop)
        coords = selection.select_coords()
        if selection.filter is not None:
            for field, op, filt in selection.filter.format():
                data = self.read_column(
                    field, start=coords.min(), stop=coords.max() + 1
                )
                coords = coords[op(data.iloc[coords - coords.min()], filt).values]

        return Index(coords)

    def read_column(
        self,
        column: str,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        return a single column from the table, generally only indexables
        are interesting
        """
        # validate the version
        self.validate_version()

        # infer the data kind
        if not self.infer_axes():
            return False

        if where is not None:
            raise TypeError("read_column does not currently accept a where clause")

        # find the axes
        for a in self.axes:
            if column == a.name:
                if not a.is_data_indexable:
                    raise ValueError(
                        f"column [{column}] can not be extracted individually; "
                        "it is not data indexable"
                    )

                # column must be an indexable or a data column
                c = getattr(self.table.cols, column)
                a.set_info(self.info)
                col_values = a.convert(
                    c[start:stop],
                    nan_rep=self.nan_rep,
                    encoding=self.encoding,
                    errors=self.errors,
                )
                cvs = _set_tz(col_values[1], a.tz)
                dtype = getattr(self.table.attrs, f"{column}_meta", None)
                return Series(cvs, name=column, copy=False, dtype=dtype)

        raise KeyError(f"column [{column}] not found in the table")


class WORMTable(Table):
    """
    a write-once read-many table: this format DOES NOT ALLOW appending to a
    table. writing is a one-time operation the data are stored in a format
    that allows for searching the data on disk
    """

    table_type = "worm"

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        """
        read the indices and the indexing array, calculate offset rows and return
        """
        raise NotImplementedError("WORMTable needs to implement read")

    def write(self, obj, **kwargs) -> None:
        """
        write in a format that we can search later on (but cannot append
        to): write out the indices and the values using _write_array
        (e.g. a CArray) create an indexing table so that we can search
        """
        raise NotImplementedError("WORMTable needs to implement write")


class AppendableTable(Table):
    """support the new appendable table formats"""

    table_type = "appendable"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(  # type: ignore[override]
        self,
        obj,
        axes=None,
        append: bool = False,
        complib=None,
        complevel=None,
        fletcher32=None,
        min_itemsize=None,
        chunksize: int | None = None,
        expectedrows=None,
        dropna: bool = False,
        nan_rep=None,
        data_columns=None,
        track_times: bool = True,
    ) -> None:
        if not append and self.is_exists:
            self._handle.remove_node(self.group, "table")

        # create the axes
        table = self._create_axes(
            axes=axes,
            obj=obj,
            validate=append,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            data_columns=data_columns,
        )

        for a in table.axes:
            a.validate_names()

        if not table.is_exists:
            # create the table
            options = table.create_description(
                complib=complib,
                complevel=complevel,
                fletcher32=fletcher32,
                expectedrows=expectedrows,
            )

            # set the table attributes
            table.set_attrs()

            options["track_times"] = track_times

            # create the table
            table._handle.create_table(table.group, **options)

        # update my info
        table.attrs.info = table.info

        # validate the axes and set the kinds
        for a in table.axes:
            a.validate_and_set(table, append)

        # add the rows
        table.write_data(chunksize, dropna=dropna)

    def write_data(self, chunksize: int | None, dropna: bool = False) -> None:
        """
        we form the data into a 2-d including indexes,values,mask write chunk-by-chunk
        """
        names = self.dtype.names
        nrows = self.nrows_expected

        # if dropna==True, then drop ALL nan rows
        masks = []
        if dropna:
            for a in self.values_axes:
                # figure the mask: only do if we can successfully process this
                # column, otherwise ignore the mask
                mask = isna(a.data).all(axis=0)
                if isinstance(mask, np.ndarray):
                    masks.append(mask.astype("u1", copy=False))

        # consolidate masks
        if len(masks):
            mask = masks[0]
            for m in masks[1:]:
                mask = mask & m
            mask = mask.ravel()
        else:
            mask = None

        # broadcast the indexes if needed
        indexes = [a.cvalues for a in self.index_axes]
        nindexes = len(indexes)
        assert nindexes == 1, nindexes  # ensures we dont need to broadcast

        # transpose the values so first dimension is last
        # reshape the values if needed
        values = [a.take_data() for a in self.values_axes]
        values = [v.transpose(np.roll(np.arange(v.ndim), v.ndim - 1)) for v in values]
        bvalues = []
        for i, v in enumerate(values):
            new_shape = (nrows,) + self.dtype[names[nindexes + i]].shape
            bvalues.append(v.reshape(new_shape))

        # write the chunks
        if chunksize is None:
            chunksize = 100000

        rows = np.empty(min(chunksize, nrows), dtype=self.dtype)
        chunks = nrows // chunksize + 1
        for i in range(chunks):
            start_i = i * chunksize
            end_i = min((i + 1) * chunksize, nrows)
            if start_i >= end_i:
                break

            self.write_data_chunk(
                rows,
                indexes=[a[start_i:end_i] for a in indexes],
                mask=mask[start_i:end_i] if mask is not None else None,
                values=[v[start_i:end_i] for v in bvalues],
            )

    def write_data_chunk(
        self,
        rows: np.ndarray,
        indexes: list[np.ndarray],
        mask: npt.NDArray[np.bool_] | None,
        values: list[np.ndarray],
    ) -> None:
        """
        Parameters
        ----------
        rows : an empty memory space where we are putting the chunk
        indexes : an array of the indexes
        mask : an array of the masks
        values : an array of the values
        """
        # 0 len
        for v in values:
            if not np.prod(v.shape):
                return

        nrows = indexes[0].shape[0]
        if nrows != len(rows):
            rows = np.empty(nrows, dtype=self.dtype)
        names = self.dtype.names
        nindexes = len(indexes)

        # indexes
        for i, idx in enumerate(indexes):
            rows[names[i]] = idx

        # values
        for i, v in enumerate(values):
            rows[names[i + nindexes]] = v

        # mask
        if mask is not None:
            m = ~mask.ravel().astype(bool, copy=False)
            if not m.all():
                rows = rows[m]

        if len(rows):
            self.table.append(rows)
            self.table.flush()

    def delete(self, where=None, start: int | None = None, stop: int | None = None):
        # delete all rows (and return the nrows)
        if where is None or not len(where):
            if start is None and stop is None:
                nrows = self.nrows
                self._handle.remove_node(self.group, recursive=True)
            else:
                # pytables<3.0 would remove a single row with stop=None
                if stop is None:
                    stop = self.nrows
                nrows = self.table.remove_rows(start=start, stop=stop)
                self.table.flush()
            return nrows

        # infer the data kind
        if not self.infer_axes():
            return None

        # create the selection
        table = self.table
        selection = Selection(self, where, start=start, stop=stop)
        values = selection.select_coords()

        # delete the rows in reverse order
        sorted_series = Series(values, copy=False).sort_values()
        ln = len(sorted_series)

        if ln:
            # construct groups of consecutive rows
            diff = sorted_series.diff()
            groups = list(diff[diff > 1].index)

            # 1 group
            if not len(groups):
                groups = [0]

            # final element
            if groups[-1] != ln:
                groups.append(ln)

            # initial element
            if groups[0] != 0:
                groups.insert(0, 0)

            # we must remove in reverse order!
            pg = groups.pop()
            for g in reversed(groups):
                rows = sorted_series.take(range(g, pg))
                table.remove_rows(
                    start=rows[rows.index[0]], stop=rows[rows.index[-1]] + 1
                )
                pg = g

            self.table.flush()

        # return the number of rows removed
        return ln


class AppendableFrameTable(AppendableTable):
    """support the new appendable table formats"""

    pandas_kind = "frame_table"
    table_type = "appendable_frame"
    ndim = 2
    obj_type: type[DataFrame | Series] = DataFrame

    @property
    def is_transposed(self) -> bool:
        return self.index_axes[0].axis == 1

    @classmethod
    def get_object(cls, obj, transposed: bool):
        """these are written transposed"""
        if transposed:
            obj = obj.T
        return obj

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        # validate the version
        self.validate_version(where)

        # infer the data kind
        if not self.infer_axes():
            return None

        result = self._read_axes(where=where, start=start, stop=stop)

        info = (
            self.info.get(self.non_index_axes[0][0], {})
            if len(self.non_index_axes)
            else {}
        )

        inds = [i for i, ax in enumerate(self.axes) if ax is self.index_axes[0]]
        assert len(inds) == 1
        ind = inds[0]

        index = result[ind][0]

        frames = []
        for i, a in enumerate(self.axes):
            if a not in self.values_axes:
                continue
            index_vals, cvalues = result[i]

            # we could have a multi-index constructor here
            # ensure_index doesn't recognized our list-of-tuples here
            if info.get("type") != "MultiIndex":
                cols = Index(index_vals)
            else:
                cols = MultiIndex.from_tuples(index_vals)

            names = info.get("names")
            if names is not None:
                cols.set_names(names, inplace=True)

            if self.is_transposed:
                values = cvalues
                index_ = cols
                cols_ = Index(index, name=getattr(index, "name", None))
            else:
                values = cvalues.T
                index_ = Index(index, name=getattr(index, "name", None))
                cols_ = cols

            # if we have a DataIndexableCol, its shape will only be 1 dim
            if values.ndim == 1 and isinstance(values, np.ndarray):
                values = values.reshape((1, values.shape[0]))

            if isinstance(values, np.ndarray):
                df = DataFrame(values.T, columns=cols_, index=index_, copy=False)
            elif isinstance(values, Index):
                df = DataFrame(values, columns=cols_, index=index_)
            else:
                # Categorical
                df = DataFrame._from_arrays([values], columns=cols_, index=index_)
            if not (using_string_dtype() and values.dtype.kind == "O"):
                assert (df.dtypes == values.dtype).all(), (df.dtypes, values.dtype)

            # If str / string dtype is stored in meta, use that.
            converted = False
            for column in cols_:
                dtype = getattr(self.table.attrs, f"{column}_meta", None)
                if dtype in ["str", "string"]:
                    df[column] = df[column].astype(dtype)
                    converted = True
            # Otherwise try inference.
            if (
                not converted
                and using_string_dtype()
                and isinstance(values, np.ndarray)
                and is_string_array(
                    values,
                    skipna=True,
                )
            ):
                df = df.astype(StringDtype(na_value=np.nan))
            frames.append(df)

        if len(frames) == 1:
            df = frames[0]
        else:
            df = concat(frames, axis=1)

        selection = Selection(self, where=where, start=start, stop=stop)
        # apply the selection filters & axis orderings
        df = self.process_axes(df, selection=selection, columns=columns)
        return df


class AppendableSeriesTable(AppendableFrameTable):
    """support the new appendable table formats"""

    pandas_kind = "series_table"
    table_type = "appendable_series"
    ndim = 2
    obj_type = Series

    @property
    def is_transposed(self) -> bool:
        return False

    @classmethod
    def get_object(cls, obj, transposed: bool):
        return obj

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        """we are going to write this as a frame table"""
        if not isinstance(obj, DataFrame):
            name = obj.name or "values"
            obj = obj.to_frame(name)
        super().write(obj=obj, data_columns=obj.columns.tolist(), **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> Series:
        is_multi_index = self.is_multi_index
        if columns is not None and is_multi_index:
            assert isinstance(self.levels, list)  # needed for mypy
            for n in self.levels:
                if n not in columns:
                    columns.insert(0, n)
        s = super().read(where=where, columns=columns, start=start, stop=stop)
        if is_multi_index:
            s.set_index(self.levels, inplace=True)

        s = s.iloc[:, 0]

        # remove the default name
        if s.name == "values":
            s.name = None
        return s


class AppendableMultiSeriesTable(AppendableSeriesTable):
    """support the new appendable table formats"""

    pandas_kind = "series_table"
    table_type = "appendable_multiseries"

    #  error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, **kwargs) -> None:  # type: ignore[override]
        """we are going to write this as a frame table"""
        name = obj.name or "values"
        newobj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        cols = list(self.levels)
        cols.append(name)
        newobj.columns = Index(cols)
        super().write(obj=newobj, **kwargs)


class GenericTable(AppendableFrameTable):
    """a table that read/writes the generic pytables table format"""

    pandas_kind = "frame_table"
    table_type = "generic_table"
    ndim = 2
    obj_type = DataFrame
    levels: list[Hashable]

    @property
    def pandas_type(self) -> str:
        return self.pandas_kind

    @property
    def storable(self):
        return getattr(self.group, "table", None) or self.group

    def get_attrs(self) -> None:
        """retrieve our attributes"""
        self.non_index_axes = []
        self.nan_rep = None
        self.levels = []

        self.index_axes = [a for a in self.indexables if a.is_an_indexable]
        self.values_axes = [a for a in self.indexables if not a.is_an_indexable]
        self.data_columns = [a.name for a in self.values_axes]

    @cache_readonly
    def indexables(self):
        """create the indexables from the table description"""
        d = self.description

        # TODO: can we get a typ for this?  AFAICT it is the only place
        #  where we aren't passing one
        # the index columns is just a simple index
        md = self.read_metadata("index")
        meta = "category" if md is not None else None
        index_col = GenericIndexCol(
            name="index", axis=0, table=self.table, meta=meta, metadata=md
        )

        _indexables: list[GenericIndexCol | GenericDataIndexableCol] = [index_col]

        for i, n in enumerate(d._v_names):
            assert isinstance(n, str)

            atom = getattr(d, n)
            md = self.read_metadata(n)
            meta = "category" if md is not None else None
            dc = GenericDataIndexableCol(
                name=n,
                pos=i,
                values=[n],
                typ=atom,
                table=self.table,
                meta=meta,
                metadata=md,
            )
            _indexables.append(dc)

        return _indexables

    # error: Signature of "write" incompatible with supertype "AppendableTable"
    def write(self, **kwargs) -> None:  # type: ignore[override]
        raise NotImplementedError("cannot write on an generic table")


class AppendableMultiFrameTable(AppendableFrameTable):
    """a frame with a multi-index"""

    table_type = "appendable_multiframe"
    obj_type = DataFrame
    ndim = 2
    _re_levels = re.compile(r"^level_\d+$")

    @property
    def table_type_short(self) -> str:
        return "appendable_multi"

    # error: Signature of "write" incompatible with supertype "Fixed"
    def write(self, obj, data_columns=None, **kwargs) -> None:  # type: ignore[override]
        if data_columns is None:
            data_columns = []
        elif data_columns is True:
            data_columns = obj.columns.tolist()
        obj, self.levels = self.validate_multiindex(obj)
        assert isinstance(self.levels, list)  # for mypy
        for n in self.levels:
            if n not in data_columns:
                data_columns.insert(0, n)
        super().write(obj=obj, data_columns=data_columns, **kwargs)

    def read(
        self,
        where=None,
        columns=None,
        start: int | None = None,
        stop: int | None = None,
    ):
        df = super().read(where=where, columns=columns, start=start, stop=stop)
        df = df.set_index(self.levels)

        # remove names for 'level_%d'
        df.index = df.index.set_names(
            [None if self._re_levels.search(name) else name for name in df.index.names]
        )

        return df


def _reindex_axis(
    obj: DataFrame, axis: AxisInt, labels: Index, other=None
) -> DataFrame:
    ax = obj._get_axis(axis)
    labels = ensure_index(labels)

    # try not to reindex even if other is provided
    # if it equals our current index
    if other is not None:
        other = ensure_index(other)
    if (other is None or labels.equals(other)) and labels.equals(ax):
        return obj

    labels = ensure_index(labels.unique())
    if other is not None:
        labels = ensure_index(other.unique()).intersection(labels, sort=False)
    if not labels.equals(ax):
        slicer: list[slice | Index] = [slice(None, None)] * obj.ndim
        slicer[axis] = labels
        obj = obj.loc[tuple(slicer)]
    return obj


# tz to/from coercion


def _get_tz(tz: tzinfo) -> str | tzinfo:
    """for a tz-aware type, return an encoded zone"""
    zone = timezones.get_timezone(tz)
    return zone


@overload
def _set_tz(
    values: np.ndarray | Index, tz: str | tzinfo, coerce: bool = False
) -> DatetimeIndex:
    ...


@overload
def _set_tz(values: np.ndarray | Index, tz: None, coerce: bool = False) -> np.ndarray:
    ...


def _set_tz(
    values: np.ndarray | Index, tz: str | tzinfo | None, coerce: bool = False
) -> np.ndarray | DatetimeIndex:
    """
    coerce the values to a DatetimeIndex if tz is set
    preserve the input shape if possible

    Parameters
    ----------
    values : ndarray or Index
    tz : str or tzinfo
    coerce : if we do not have a passed timezone, coerce to M8[ns] ndarray
    """
    if isinstance(values, DatetimeIndex):
        # If values is tzaware, the tz gets dropped in the values.ravel()
        #  call below (which returns an ndarray).  So we are only non-lossy
        #  if `tz` matches `values.tz`.
        assert values.tz is None or values.tz == tz
        if values.tz is not None:
            return values

    if tz is not None:
        if isinstance(values, DatetimeIndex):
            name = values.name
        else:
            name = None
            values = values.ravel()

        tz = _ensure_decoded(tz)
        values = DatetimeIndex(values, name=name)
        values = values.tz_localize("UTC").tz_convert(tz)
    elif coerce:
        values = np.asarray(values, dtype="M8[ns]")

    # error: Incompatible return value type (got "Union[ndarray, Index]",
    # expected "Union[ndarray, DatetimeIndex]")
    return values  # type: ignore[return-value]


def _convert_index(name: str, index: Index, encoding: str, errors: str) -> IndexCol:
    assert isinstance(name, str)

    index_name = index.name
    # error: Argument 1 to "_get_data_and_dtype_name" has incompatible type "Index";
    # expected "Union[ExtensionArray, ndarray]"
    converted, dtype_name = _get_data_and_dtype_name(index)  # type: ignore[arg-type]
    kind = _dtype_to_kind(dtype_name)
    atom = DataIndexableCol._get_atom(converted)

    if (
        lib.is_np_dtype(index.dtype, "iu")
        or needs_i8_conversion(index.dtype)
        or is_bool_dtype(index.dtype)
    ):
        # Includes Index, RangeIndex, DatetimeIndex, TimedeltaIndex, PeriodIndex,
        #  in which case "kind" is "integer", "integer", "datetime64",
        #  "timedelta64", and "integer", respectively.
        return IndexCol(
            name,
            values=converted,
            kind=kind,
            typ=atom,
            freq=getattr(index, "freq", None),
            tz=getattr(index, "tz", None),
            index_name=index_name,
        )

    if isinstance(index, MultiIndex):
        raise TypeError("MultiIndex not supported here!")

    inferred_type = lib.infer_dtype(index, skipna=False)
    # we won't get inferred_type of "datetime64" or "timedelta64" as these
    #  would go through the DatetimeIndex/TimedeltaIndex paths above

    values = np.asarray(index)

    if inferred_type == "date":
        converted = np.asarray([v.toordinal() for v in values], dtype=np.int32)
        return IndexCol(
            name, converted, "date", _tables().Time32Col(), index_name=index_name
        )
    elif inferred_type == "string":
        converted = _convert_string_array(values, encoding, errors)
        itemsize = converted.dtype.itemsize
        return IndexCol(
            name,
            converted,
            "string",
            _tables().StringCol(itemsize),
            index_name=index_name,
        )

    elif inferred_type in ["integer", "floating"]:
        return IndexCol(
            name, values=converted, kind=kind, typ=atom, index_name=index_name
        )
    else:
        assert isinstance(converted, np.ndarray) and converted.dtype == object
        assert kind == "object", kind
        atom = _tables().ObjectAtom()
        return IndexCol(name, converted, kind, atom, index_name=index_name)


def _unconvert_index(data, kind: str, encoding: str, errors: str) -> np.ndarray | Index:
    index: Index | np.ndarray

    if kind.startswith("datetime64"):
        if kind == "datetime64":
            # created before we stored resolution information
            index = DatetimeIndex(data)
        else:
            index = DatetimeIndex(data.view(kind))
    elif kind == "timedelta64":
        index = TimedeltaIndex(data)
    elif kind == "date":
        try:
            index = np.asarray([date.fromordinal(v) for v in data], dtype=object)
        except ValueError:
            index = np.asarray([date.fromtimestamp(v) for v in data], dtype=object)
    elif kind in ("integer", "float", "bool"):
        index = np.asarray(data)
    elif kind in ("string"):
        index = _unconvert_string_array(
            data, nan_rep=None, encoding=encoding, errors=errors
        )
    elif kind == "object":
        index = np.asarray(data[0])
    else:  # pragma: no cover
        raise ValueError(f"unrecognized index type {kind}")
    return index


def _maybe_convert_for_string_atom(
    name: str,
    bvalues: ArrayLike,
    existing_col,
    min_itemsize,
    nan_rep,
    encoding,
    errors,
    columns: list[str],
):
    if isinstance(bvalues.dtype, StringDtype):
        # "ndarray[Any, Any]" has no attribute "to_numpy"
        bvalues = bvalues.to_numpy()  # type: ignore[union-attr]
    if bvalues.dtype != object:
        return bvalues

    bvalues = cast(np.ndarray, bvalues)

    dtype_name = bvalues.dtype.name
    inferred_type = lib.infer_dtype(bvalues, skipna=False)

    if inferred_type == "date":
        raise TypeError("[date] is not implemented as a table column")
    if inferred_type == "datetime":
        # after GH#8260
        # this only would be hit for a multi-timezone dtype which is an error
        raise TypeError(
            "too many timezones in this block, create separate data columns"
        )

    if not (inferred_type == "string" or dtype_name == "object"):
        return bvalues

    mask = isna(bvalues)
    data = bvalues.copy()
    data[mask] = nan_rep

    if existing_col and mask.any() and len(nan_rep) > existing_col.itemsize:
        raise ValueError("NaN representation is too large for existing column size")

    # see if we have a valid string type
    inferred_type = lib.infer_dtype(data, skipna=False)
    if inferred_type != "string":
        # we cannot serialize this data, so report an exception on a column
        # by column basis

        # expected behaviour:
        # search block for a non-string object column by column
        for i in range(data.shape[0]):
            col = data[i]
            inferred_type = lib.infer_dtype(col, skipna=False)
            if inferred_type != "string":
                error_column_label = columns[i] if len(columns) > i else f"No.{i}"
                raise TypeError(
                    f"Cannot serialize the column [{error_column_label}]\n"
                    f"because its data contents are not [string] but "
                    f"[{inferred_type}] object dtype"
                )

    # itemsize is the maximum length of a string (along any dimension)

    data_converted = _convert_string_array(data, encoding, errors).reshape(data.shape)
    itemsize = data_converted.itemsize

    # specified min_itemsize?
    if isinstance(min_itemsize, dict):
        min_itemsize = int(min_itemsize.get(name) or min_itemsize.get("values") or 0)
    itemsize = max(min_itemsize or 0, itemsize)

    # check for column in the values conflicts
    if existing_col is not None:
        eci = existing_col.validate_col(itemsize)
        if eci is not None and eci > itemsize:
            itemsize = eci

    data_converted = data_converted.astype(f"|S{itemsize}", copy=False)
    return data_converted


def _convert_string_array(data: np.ndarray, encoding: str, errors: str) -> np.ndarray:
    """
    Take a string-like that is object dtype and coerce to a fixed size string type.

    Parameters
    ----------
    data : np.ndarray[object]
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[fixed-length-string]
    """
    # encode if needed
    if len(data):
        data = (
            Series(data.ravel(), copy=False)
            .str.encode(encoding, errors)
            ._values.reshape(data.shape)
        )

    # create the sized dtype
    ensured = ensure_object(data.ravel())
    itemsize = max(1, libwriters.max_len_string_array(ensured))

    data = np.asarray(data, dtype=f"S{itemsize}")
    return data


def _unconvert_string_array(
    data: np.ndarray, nan_rep, encoding: str, errors: str
) -> np.ndarray:
    """
    Inverse of _convert_string_array.

    Parameters
    ----------
    data : np.ndarray[fixed-length-string]
    nan_rep : the storage repr of NaN
    encoding : str
    errors : str
        Handler for encoding errors.

    Returns
    -------
    np.ndarray[object]
        Decoded data.
    """
    shape = data.shape
    data = np.asarray(data.ravel(), dtype=object)

    if len(data):
        itemsize = libwriters.max_len_string_array(ensure_object(data))
        dtype = f"U{itemsize}"

        if isinstance(data[0], bytes):
            ser = Series(data, copy=False).str.decode(encoding, errors=errors)
            data = ser.to_numpy()
            data.flags.writeable = True
        else:
            data = data.astype(dtype, copy=False).astype(object, copy=False)

    if nan_rep is None:
        nan_rep = "nan"

    libwriters.string_array_replace_from_nan_rep(data, nan_rep)
    return data.reshape(shape)


def _maybe_convert(values: np.ndarray, val_kind: str, encoding: str, errors: str):
    assert isinstance(val_kind, str), type(val_kind)
    if _need_convert(val_kind):
        conv = _get_converter(val_kind, encoding, errors)
        values = conv(values)
    return values


def _get_converter(kind: str, encoding: str, errors: str):
    if kind == "datetime64":
        return lambda x: np.asarray(x, dtype="M8[ns]")
    elif "datetime64" in kind:
        return lambda x: np.asarray(x, dtype=kind)
    elif kind == "string":
        return lambda x: _unconvert_string_array(
            x, nan_rep=None, encoding=encoding, errors=errors
        )
    else:  # pragma: no cover
        raise ValueError(f"invalid kind {kind}")


def _need_convert(kind: str) -> bool:
    if kind in ("datetime64", "string") or "datetime64" in kind:
        return True
    return False


def _maybe_adjust_name(name: str, version: Sequence[int]) -> str:
    """
    Prior to 0.10.1, we named values blocks like: values_block_0 an the
    name values_0, adjust the given name if necessary.

    Parameters
    ----------
    name : str
    version : Tuple[int, int, int]

    Returns
    -------
    str
    """
    if isinstance(version, str) or len(version) < 3:
        raise ValueError("Version is incorrect, expected sequence of 3 integers.")

    if version[0] == 0 and version[1] <= 10 and version[2] == 0:
        m = re.search(r"values_block_(\d+)", name)
        if m:
            grp = m.groups()[0]
            name = f"values_{grp}"
    return name


def _dtype_to_kind(dtype_str: str) -> str:
    """
    Find the "kind" string describing the given dtype name.
    """
    dtype_str = _ensure_decoded(dtype_str)

    if dtype_str.startswith(("string", "bytes")):
        kind = "string"
    elif dtype_str.startswith("float"):
        kind = "float"
    elif dtype_str.startswith("complex"):
        kind = "complex"
    elif dtype_str.startswith(("int", "uint")):
        kind = "integer"
    elif dtype_str.startswith("datetime64"):
        kind = dtype_str
    elif dtype_str.startswith("timedelta"):
        kind = "timedelta64"
    elif dtype_str.startswith("bool"):
        kind = "bool"
    elif dtype_str.startswith("category"):
        kind = "category"
    elif dtype_str.startswith("period"):
        # We store the `freq` attr so we can restore from integers
        kind = "integer"
    elif dtype_str == "object":
        kind = "object"
    elif dtype_str == "str":
        kind = "str"
    else:
        raise ValueError(f"cannot interpret dtype of [{dtype_str}]")

    return kind


def _get_data_and_dtype_name(data: ArrayLike):
    """
    Convert the passed data into a storable form and a dtype string.
    """
    if isinstance(data, Categorical):
        data = data.codes

    if isinstance(data.dtype, DatetimeTZDtype):
        # For datetime64tz we need to drop the TZ in tests TODO: why?
        dtype_name = f"datetime64[{data.dtype.unit}]"
    else:
        dtype_name = data.dtype.name

    if data.dtype.kind in "mM":
        data = np.asarray(data.view("i8"))
        # TODO: we used to reshape for the dt64tz case, but no longer
        #  doing that doesn't seem to break anything.  why?

    elif isinstance(data, PeriodIndex):
        data = data.asi8

    data = np.asarray(data)
    return data, dtype_name


class Selection:
    """
    Carries out a selection operation on a tables.Table object.

    Parameters
    ----------
    table : a Table object
    where : list of Terms (or convertible to)
    start, stop: indices to start and/or stop selection

    """

    def __init__(
        self,
        table: Table,
        where=None,
        start: int | None = None,
        stop: int | None = None,
    ) -> None:
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None

        if is_list_like(where):
            # see if we have a passed coordinate like
            with suppress(ValueError):
                inferred = lib.infer_dtype(where, skipna=False)
                if inferred in ("integer", "boolean"):
                    where = np.asarray(where)
                    if where.dtype == np.bool_:
                        start, stop = self.start, self.stop
                        if start is None:
                            start = 0
                        if stop is None:
                            stop = self.table.nrows
                        self.coordinates = np.arange(start, stop)[where]
                    elif issubclass(where.dtype.type, np.integer):
                        if (self.start is not None and (where < self.start).any()) or (
                            self.stop is not None and (where >= self.stop).any()
                        ):
                            raise ValueError(
                                "where must have index locations >= start and < stop"
                            )
                        self.coordinates = where

        if self.coordinates is None:
            self.terms = self.generate(where)

            # create the numexpr & the filter
            if self.terms is not None:
                self.condition, self.filter = self.terms.evaluate()

    def generate(self, where):
        """where can be a : dict,list,tuple,string"""
        if where is None:
            return None

        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            # raise a nice message, suggesting that the user should use
            # data_columns
            qkeys = ",".join(q.keys())
            msg = dedent(
                f"""\
                The passed where expression: {where}
                            contains an invalid variable reference
                            all of the variable references must be a reference to
                            an axis (e.g. 'index' or 'columns'), or a data_column
                            The currently defined references are: {qkeys}
                """
            )
            raise ValueError(msg) from err

    def select(self):
        """
        generate the selection
        """
        if self.condition is not None:
            return self.table.table.read_where(
                self.condition.format(), start=self.start, stop=self.stop
            )
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self):
        """
        generate the selection
        """
        start, stop = self.start, self.stop
        nrows = self.table.nrows
        if start is None:
            start = 0
        elif start < 0:
            start += nrows
        if stop is None:
            stop = nrows
        elif stop < 0:
            stop += nrows

        if self.condition is not None:
            return self.table.table.get_where_list(
                self.condition.format(), start=start, stop=stop, sort=True
            )
        elif self.coordinates is not None:
            return self.coordinates

        return np.arange(start, stop)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\perm_groups.py ===
from math import factorial as _factorial, log, prod
from itertools import chain, product


from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
    _af_rmul, _af_rmuln, _af_pow, Cycle)
from sympy.combinatorics.util import (_check_cycles_alt_sym,
    _distribute_gens_by_base, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs, _base_ordering, _strong_gens_from_distr,
    _strip, _strip_af)
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq

rmul = Permutation.rmul_with_af
_af_new = Permutation._af_new


class PermutationGroup(Basic):
    r"""The class defining a Permutation group.

    Explanation
    ===========

    ``PermutationGroup([p1, p2, ..., pn])`` returns the permutation group
    generated by the list of permutations. This group can be supplied
    to Polyhedron if one desires to decorate the elements to which the
    indices of the permutation refer.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> from sympy.combinatorics import Polyhedron

    The permutations corresponding to motion of the front, right and
    bottom face of a $2 \times 2$ Rubik's cube are defined:

    >>> F = Permutation(2, 19, 21, 8)(3, 17, 20, 10)(4, 6, 7, 5)
    >>> R = Permutation(1, 5, 21, 14)(3, 7, 23, 12)(8, 10, 11, 9)
    >>> D = Permutation(6, 18, 14, 10)(7, 19, 15, 11)(20, 22, 23, 21)

    These are passed as permutations to PermutationGroup:

    >>> G = PermutationGroup(F, R, D)
    >>> G.order()
    3674160

    The group can be supplied to a Polyhedron in order to track the
    objects being moved. An example involving the $2 \times 2$ Rubik's cube is
    given there, but here is a simple demonstration:

    >>> a = Permutation(2, 1)
    >>> b = Permutation(1, 0)
    >>> G = PermutationGroup(a, b)
    >>> P = Polyhedron(list('ABC'), pgroup=G)
    >>> P.corners
    (A, B, C)
    >>> P.rotate(0) # apply permutation 0
    >>> P.corners
    (A, C, B)
    >>> P.reset()
    >>> P.corners
    (A, B, C)

    Or one can make a permutation as a product of selected permutations
    and apply them to an iterable directly:

    >>> P10 = G.make_perm([0, 1])
    >>> P10('ABC')
    ['C', 'A', 'B']

    See Also
    ========

    sympy.combinatorics.polyhedron.Polyhedron,
    sympy.combinatorics.permutations.Permutation

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"

    .. [2] Seress, A.
           "Permutation Group Algorithms"

    .. [3] https://en.wikipedia.org/wiki/Schreier_vector

    .. [4] https://en.wikipedia.org/wiki/Nielsen_transformation#Product_replacement_algorithm

    .. [5] Frank Celler, Charles R.Leedham-Green, Scott H.Murray,
           Alice C.Niemeyer, and E.A.O'Brien. "Generating Random
           Elements of a Finite Group"

    .. [6] https://en.wikipedia.org/wiki/Block_%28permutation_group_theory%29

    .. [7] https://algorithmist.com/wiki/Union_find

    .. [8] https://en.wikipedia.org/wiki/Multiply_transitive_group#Multiply_transitive_groups

    .. [9] https://en.wikipedia.org/wiki/Center_%28group_theory%29

    .. [10] https://en.wikipedia.org/wiki/Centralizer_and_normalizer

    .. [11] https://groupprops.subwiki.org/wiki/Derived_subgroup

    .. [12] https://en.wikipedia.org/wiki/Nilpotent_group

    .. [13] https://www.math.colostate.edu/~hulpke/CGT/cgtnotes.pdf

    .. [14] https://docs.gap-system.org/doc/ref/manual.pdf

    """
    is_group = True

    def __new__(cls, *args, dups=True, **kwargs):
        """The default constructor. Accepts Cycle and Permutation forms.
        Removes duplicates unless ``dups`` keyword is ``False``.
        """
        if not args:
            args = [Permutation()]
        else:
            args = list(args[0] if is_sequence(args[0]) else args)
            if not args:
                args = [Permutation()]
        if any(isinstance(a, Cycle) for a in args):
            args = [Permutation(a) for a in args]
        if has_variety(a.size for a in args):
            degree = kwargs.pop('degree', None)
            if degree is None:
                degree = max(a.size for a in args)
            for i in range(len(args)):
                if args[i].size != degree:
                    args[i] = Permutation(args[i], size=degree)
        if dups:
            args = list(uniq([_af_new(list(a)) for a in args]))
        if len(args) > 1:
            args = [g for g in args if not g.is_identity]
        return Basic.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self._generators = list(self.args)
        self._order = None
        self._elements = []
        self._center = None
        self._is_abelian = None
        self._is_transitive = None
        self._is_sym = None
        self._is_alt = None
        self._is_primitive = None
        self._is_nilpotent = None
        self._is_solvable = None
        self._is_trivial = None
        self._transitivity_degree = None
        self._max_div = None
        self._is_perfect = None
        self._is_cyclic = None
        self._is_dihedral = None
        self._r = len(self._generators)
        self._degree = self._generators[0].size

        # these attributes are assigned after running schreier_sims
        self._base = []
        self._strong_gens = []
        self._strong_gens_slp = []
        self._basic_orbits = []
        self._transversals = []
        self._transversal_slp = []

        # these attributes are assigned after running _random_pr_init
        self._random_gens = []

        # finite presentation of the group as an instance of `FpGroup`
        self._fp_presentation = None

    def __getitem__(self, i):
        return self._generators[i]

    def __contains__(self, i):
        """Return ``True`` if *i* is contained in PermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = Permutation(1, 2, 3)
        >>> Permutation(3) in PermutationGroup(p)
        True

        """
        if not isinstance(i, Permutation):
            raise TypeError("A PermutationGroup contains only Permutations as "
                            "elements, not elements of type %s" % type(i))
        return self.contains(i)

    def __len__(self):
        return len(self._generators)

    def equals(self, other):
        """Return ``True`` if PermutationGroup generated by elements in the
        group are same i.e they represent the same PermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = Permutation(0, 1, 2, 3, 4, 5)
        >>> G = PermutationGroup([p, p**2])
        >>> H = PermutationGroup([p**2, p])
        >>> G.generators == H.generators
        False
        >>> G.equals(H)
        True

        """
        if not isinstance(other, PermutationGroup):
            return False

        set_self_gens = set(self.generators)
        set_other_gens = set(other.generators)

        # before reaching the general case there are also certain
        # optimisation and obvious cases requiring less or no actual
        # computation.
        if set_self_gens == set_other_gens:
            return True

        # in the most general case it will check that each generator of
        # one group belongs to the other PermutationGroup and vice-versa
        for gen1 in set_self_gens:
            if not other.contains(gen1):
                return False
        for gen2 in set_other_gens:
            if not self.contains(gen2):
                return False
        return True

    def __mul__(self, other):
        """
        Return the direct product of two permutation groups as a permutation
        group.

        Explanation
        ===========

        This implementation realizes the direct product by shifting the index
        set for the generators of the second group: so if we have ``G`` acting
        on ``n1`` points and ``H`` acting on ``n2`` points, ``G*H`` acts on
        ``n1 + n2`` points.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import CyclicGroup
        >>> G = CyclicGroup(5)
        >>> H = G*G
        >>> H
        PermutationGroup([
            (9)(0 1 2 3 4),
            (5 6 7 8 9)])
        >>> H.order()
        25

        """
        if isinstance(other, Permutation):
            return Coset(other, self, dir='+')
        gens1 = [perm._array_form for perm in self.generators]
        gens2 = [perm._array_form for perm in other.generators]
        n1 = self._degree
        n2 = other._degree
        start = list(range(n1))
        end = list(range(n1, n1 + n2))
        for i in range(len(gens2)):
            gens2[i] = [x + n1 for x in gens2[i]]
        gens2 = [start + gen for gen in gens2]
        gens1 = [gen + end for gen in gens1]
        together = gens1 + gens2
        gens = [_af_new(x) for x in together]
        return PermutationGroup(gens)

    def _random_pr_init(self, r, n, _random_prec_n=None):
        r"""Initialize random generators for the product replacement algorithm.

        Explanation
        ===========

        The implementation uses a modification of the original product
        replacement algorithm due to Leedham-Green, as described in [1],
        pp. 69-71; also, see [2], pp. 27-29 for a detailed theoretical
        analysis of the original product replacement algorithm, and [4].

        The product replacement algorithm is used for producing random,
        uniformly distributed elements of a group `G` with a set of generators
        `S`. For the initialization ``_random_pr_init``, a list ``R`` of
        `\max\{r, |S|\}` group generators is created as the attribute
        ``G._random_gens``, repeating elements of `S` if necessary, and the
        identity element of `G` is appended to ``R`` - we shall refer to this
        last element as the accumulator. Then the function ``random_pr()``
        is called ``n`` times, randomizing the list ``R`` while preserving
        the generation of `G` by ``R``. The function ``random_pr()`` itself
        takes two random elements ``g, h`` among all elements of ``R`` but
        the accumulator and replaces ``g`` with a randomly chosen element
        from `\{gh, g(~h), hg, (~h)g\}`. Then the accumulator is multiplied
        by whatever ``g`` was replaced by. The new value of the accumulator is
        then returned by ``random_pr()``.

        The elements returned will eventually (for ``n`` large enough) become
        uniformly distributed across `G` ([5]). For practical purposes however,
        the values ``n = 50, r = 11`` are suggested in [1].

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: it changes the attribute
        self._random_gens

        See Also
        ========

        random_pr

        """
        deg = self.degree
        random_gens = [x._array_form for x in self.generators]
        k = len(random_gens)
        if k < r:
            for i in range(k, r):
                random_gens.append(random_gens[i - k])
        acc = list(range(deg))
        random_gens.append(acc)
        self._random_gens = random_gens

        # handle randomized input for testing purposes
        if _random_prec_n is None:
            for i in range(n):
                self.random_pr()
        else:
            for i in range(n):
                self.random_pr(_random_prec=_random_prec_n[i])

    def _union_find_merge(self, first, second, ranks, parents, not_rep):
        """Merges two classes in a union-find data structure.

        Explanation
        ===========

        Used in the implementation of Atkinson's algorithm as suggested in [1],
        pp. 83-87. The class merging process uses union by rank as an
        optimization. ([7])

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: the list of class representatives,
        ``parents``, the list of class sizes, ``ranks``, and the list of
        elements that are not representatives, ``not_rep``, are changed due to
        class merging.

        See Also
        ========

        minimal_block, _union_find_rep

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of computational group theory"

        .. [7] https://algorithmist.com/wiki/Union_find

        """
        rep_first = self._union_find_rep(first, parents)
        rep_second = self._union_find_rep(second, parents)
        if rep_first != rep_second:
            # union by rank
            if ranks[rep_first] >= ranks[rep_second]:
                new_1, new_2 = rep_first, rep_second
            else:
                new_1, new_2 = rep_second, rep_first
            total_rank = ranks[new_1] + ranks[new_2]
            if total_rank > self.max_div:
                return -1
            parents[new_2] = new_1
            ranks[new_1] = total_rank
            not_rep.append(new_2)
            return 1
        return 0

    def _union_find_rep(self, num, parents):
        """Find representative of a class in a union-find data structure.

        Explanation
        ===========

        Used in the implementation of Atkinson's algorithm as suggested in [1],
        pp. 83-87. After the representative of the class to which ``num``
        belongs is found, path compression is performed as an optimization
        ([7]).

        Notes
        =====

        THIS FUNCTION HAS SIDE EFFECTS: the list of class representatives,
        ``parents``, is altered due to path compression.

        See Also
        ========

        minimal_block, _union_find_merge

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of computational group theory"

        .. [7] https://algorithmist.com/wiki/Union_find

        """
        rep, parent = num, parents[num]
        while parent != rep:
            rep = parent
            parent = parents[rep]
        # path compression
        temp, parent = num, parents[num]
        while parent != rep:
            parents[temp] = rep
            temp = parent
            parent = parents[temp]
        return rep

    @property
    def base(self):
        r"""Return a base from the Schreier-Sims algorithm.

        Explanation
        ===========

        For a permutation group `G`, a base is a sequence of points
        `B = (b_1, b_2, \dots, b_k)` such that no element of `G` apart
        from the identity fixes all the points in `B`. The concepts of
        a base and strong generating set and their applications are
        discussed in depth in [1], pp. 87-89 and [2], pp. 55-57.

        An alternative way to think of `B` is that it gives the
        indices of the stabilizer cosets that contain more than the
        identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation(0, 1, 3)(2, 4)])
        >>> G.base
        [0, 2]

        See Also
        ========

        strong_gens, basic_transversals, basic_orbits, basic_stabilizers

        """
        if self._base == []:
            self.schreier_sims()
        return self._base

    def baseswap(self, base, strong_gens, pos, randomized=False,
                 transversals=None, basic_orbits=None, strong_gens_distr=None):
        r"""Swap two consecutive base points in base and strong generating set.

        Explanation
        ===========

        If a base for a group `G` is given by `(b_1, b_2, \dots, b_k)`, this
        function returns a base `(b_1, b_2, \dots, b_{i+1}, b_i, \dots, b_k)`,
        where `i` is given by ``pos``, and a strong generating set relative
        to that base. The original base and strong generating set are not
        modified.

        The randomized version (default) is of Las Vegas type.

        Parameters
        ==========

        base, strong_gens
            The base and strong generating set.
        pos
            The position at which swapping is performed.
        randomized
            A switch between randomized and deterministic version.
        transversals
            The transversals for the basic orbits, if known.
        basic_orbits
            The basic orbits, if known.
        strong_gens_distr
            The strong generators distributed by basic stabilizers, if known.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the new base, and ``strong_gens`` is a generating set
            relative to it.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> S = SymmetricGroup(4)
        >>> S.schreier_sims()
        >>> S.base
        [0, 1, 2]
        >>> base, gens = S.baseswap(S.base, S.strong_gens, 1, randomized=False)
        >>> base, gens
        ([0, 2, 1],
        [(0 1 2 3), (3)(0 1), (1 3 2),
         (2 3), (1 3)])

        check that base, gens is a BSGS

        >>> S1 = PermutationGroup(gens)
        >>> _verify_bsgs(S1, base, gens)
        True

        See Also
        ========

        schreier_sims

        Notes
        =====

        The deterministic version of the algorithm is discussed in
        [1], pp. 102-103; the randomized version is discussed in [1], p.103, and
        [2], p.98. It is of Las Vegas type.
        Notice that [1] contains a mistake in the pseudocode and
        discussion of BASESWAP: on line 3 of the pseudocode,
        `|\beta_{i+1}^{\left\langle T\right\rangle}|` should be replaced by
        `|\beta_{i}^{\left\langle T\right\rangle}|`, and the same for the
        discussion of the algorithm.

        """
        # construct the basic orbits, generators for the stabilizer chain
        # and transversal elements from whatever was provided
        transversals, basic_orbits, strong_gens_distr = \
            _handle_precomputed_bsgs(base, strong_gens, transversals,
                                 basic_orbits, strong_gens_distr)
        base_len = len(base)
        degree = self.degree
        # size of orbit of base[pos] under the stabilizer we seek to insert
        # in the stabilizer chain at position pos + 1
        size = len(basic_orbits[pos])*len(basic_orbits[pos + 1]) \
            //len(_orbit(degree, strong_gens_distr[pos], base[pos + 1]))
        # initialize the wanted stabilizer by a subgroup
        if pos + 2 > base_len - 1:
            T = []
        else:
            T = strong_gens_distr[pos + 2][:]
        # randomized version
        if randomized is True:
            stab_pos = PermutationGroup(strong_gens_distr[pos])
            schreier_vector = stab_pos.schreier_vector(base[pos + 1])
            # add random elements of the stabilizer until they generate it
            while len(_orbit(degree, T, base[pos])) != size:
                new = stab_pos.random_stab(base[pos + 1],
                                           schreier_vector=schreier_vector)
                T.append(new)
        # deterministic version
        else:
            Gamma = set(basic_orbits[pos])
            Gamma.remove(base[pos])
            if base[pos + 1] in Gamma:
                Gamma.remove(base[pos + 1])
            # add elements of the stabilizer until they generate it by
            # ruling out member of the basic orbit of base[pos] along the way
            while len(_orbit(degree, T, base[pos])) != size:
                gamma = next(iter(Gamma))
                x = transversals[pos][gamma]
                temp = x._array_form.index(base[pos + 1]) # (~x)(base[pos + 1])
                if temp not in basic_orbits[pos + 1]:
                    Gamma = Gamma - _orbit(degree, T, gamma)
                else:
                    y = transversals[pos + 1][temp]
                    el = rmul(x, y)
                    if el(base[pos]) not in _orbit(degree, T, base[pos]):
                        T.append(el)
                        Gamma = Gamma - _orbit(degree, T, base[pos])
        # build the new base and strong generating set
        strong_gens_new_distr = strong_gens_distr[:]
        strong_gens_new_distr[pos + 1] = T
        base_new = base[:]
        base_new[pos], base_new[pos + 1] = base_new[pos + 1], base_new[pos]
        strong_gens_new = _strong_gens_from_distr(strong_gens_new_distr)
        for gen in T:
            if gen not in strong_gens_new:
                strong_gens_new.append(gen)
        return base_new, strong_gens_new

    @property
    def basic_orbits(self):
        r"""
        Return the basic orbits relative to a base and strong generating set.

        Explanation
        ===========

        If `(b_1, b_2, \dots, b_k)` is a base for a group `G`, and
        `G^{(i)} = G_{b_1, b_2, \dots, b_{i-1}}` is the ``i``-th basic stabilizer
        (so that `G^{(1)} = G`), the ``i``-th basic orbit relative to this base
        is the orbit of `b_i` under `G^{(i)}`. See [1], pp. 87-89 for more
        information.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(4)
        >>> S.basic_orbits
        [[0, 1, 2, 3], [1, 2, 3], [2, 3]]

        See Also
        ========

        base, strong_gens, basic_transversals, basic_stabilizers

        """
        if self._basic_orbits == []:
            self.schreier_sims()
        return self._basic_orbits

    @property
    def basic_stabilizers(self):
        r"""
        Return a chain of stabilizers relative to a base and strong generating
        set.

        Explanation
        ===========

        The ``i``-th basic stabilizer `G^{(i)}` relative to a base
        `(b_1, b_2, \dots, b_k)` is `G_{b_1, b_2, \dots, b_{i-1}}`. For more
        information, see [1], pp. 87-89.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> A = AlternatingGroup(4)
        >>> A.schreier_sims()
        >>> A.base
        [0, 1]
        >>> for g in A.basic_stabilizers:
        ...     print(g)
        ...
        PermutationGroup([
            (3)(0 1 2),
            (1 2 3)])
        PermutationGroup([
            (1 2 3)])

        See Also
        ========

        base, strong_gens, basic_orbits, basic_transversals

        """

        if self._transversals == []:
            self.schreier_sims()
        strong_gens = self._strong_gens
        base = self._base
        if not base: # e.g. if self is trivial
            return []
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_stabilizers = []
        for gens in strong_gens_distr:
            basic_stabilizers.append(PermutationGroup(gens))
        return basic_stabilizers

    @property
    def basic_transversals(self):
        """
        Return basic transversals relative to a base and strong generating set.

        Explanation
        ===========

        The basic transversals are transversals of the basic orbits. They
        are provided as a list of dictionaries, each dictionary having
        keys - the elements of one of the basic orbits, and values - the
        corresponding transversal elements. See [1], pp. 87-89 for more
        information.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> A = AlternatingGroup(4)
        >>> A.basic_transversals
        [{0: (3), 1: (3)(0 1 2), 2: (3)(0 2 1), 3: (0 3 1)}, {1: (3), 2: (1 2 3), 3: (1 3 2)}]

        See Also
        ========

        strong_gens, base, basic_orbits, basic_stabilizers

        """

        if self._transversals == []:
            self.schreier_sims()
        return self._transversals

    def composition_series(self):
        r"""
        Return the composition series for a group as a list
        of permutation groups.

        Explanation
        ===========

        The composition series for a group `G` is defined as a
        subnormal series `G = H_0 > H_1 > H_2 \ldots` A composition
        series is a subnormal series such that each factor group
        `H(i+1) / H(i)` is simple.
        A subnormal series is a composition series only if it is of
        maximum length.

        The algorithm works as follows:
        Starting with the derived series the idea is to fill
        the gap between `G = der[i]` and `H = der[i+1]` for each
        `i` independently. Since, all subgroups of the abelian group
        `G/H` are normal so, first step is to take the generators
        `g` of `G` and add them to generators of `H` one by one.

        The factor groups formed are not simple in general. Each
        group is obtained from the previous one by adding one
        generator `g`, if the previous group is denoted by `H`
        then the next group `K` is generated by `g` and `H`.
        The factor group `K/H` is cyclic and it's order is
        `K.order()//G.order()`. The series is then extended between
        `K` and `H` by groups generated by powers of `g` and `H`.
        The series formed is then prepended to the already existing
        series.

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import CyclicGroup
        >>> S = SymmetricGroup(12)
        >>> G = S.sylow_subgroup(2)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        >>> G = S.sylow_subgroup(3)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [243, 81, 27, 9, 3, 1]
        >>> G = CyclicGroup(12)
        >>> C = G.composition_series()
        >>> [H.order() for H in C]
        [12, 6, 3, 1]

        """
        der = self.derived_series()
        if not all(g.is_identity for g in der[-1].generators):
            raise NotImplementedError('Group should be solvable')
        series = []

        for i in range(len(der)-1):
            H = der[i+1]
            up_seg = []
            for g in der[i].generators:
                K = PermutationGroup([g] + H.generators)
                order = K.order() // H.order()
                down_seg = []
                for p, e in factorint(order).items():
                    for _ in range(e):
                        down_seg.append(PermutationGroup([g] + H.generators))
                        g = g**p
                up_seg = down_seg + up_seg
                H = K
            up_seg[0] = der[i]
            series.extend(up_seg)
        series.append(der[-1])
        return series

    def coset_transversal(self, H):
        """Return a transversal of the right cosets of self by its subgroup H
        using the second method described in [1], Subsection 4.6.7

        """

        if not H.is_subgroup(self):
            raise ValueError("The argument must be a subgroup")

        if H.order() == 1:
            return self.elements

        self._schreier_sims(base=H.base) # make G.base an extension of H.base

        base = self.base
        base_ordering = _base_ordering(base, self.degree)
        identity = Permutation(self.degree - 1)

        transversals = self.basic_transversals[:]
        # transversals is a list of dictionaries. Get rid of the keys
        # so that it is a list of lists and sort each list in
        # the increasing order of base[l]^x
        for l, t in enumerate(transversals):
            transversals[l] = sorted(t.values(),
                                key = lambda x: base_ordering[base[l]^x])

        orbits = H.basic_orbits
        h_stabs = H.basic_stabilizers
        g_stabs = self.basic_stabilizers

        indices = [x.order()//y.order() for x, y in zip(g_stabs, h_stabs)]

        # T^(l) should be a right transversal of H^(l) in G^(l) for
        # 1<=l<=len(base). While H^(l) is the trivial group, T^(l)
        # contains all the elements of G^(l) so we might just as well
        # start with l = len(h_stabs)-1
        if len(g_stabs) > len(h_stabs):
            T = g_stabs[len(h_stabs)].elements
        else:
            T = [identity]
        l = len(h_stabs)-1
        t_len = len(T)
        while l > -1:
            T_next = []
            for u in transversals[l]:
                if u == identity:
                    continue
                b = base_ordering[base[l]^u]
                for t in T:
                    p = t*u
                    if all(base_ordering[h^p] >= b for h in orbits[l]):
                        T_next.append(p)
                    if t_len + len(T_next) == indices[l]:
                        break
                if t_len + len(T_next) == indices[l]:
                    break
            T += T_next
            t_len += len(T_next)
            l -= 1
        T.remove(identity)
        T = [identity] + T
        return T

    def _coset_representative(self, g, H):
        """Return the representative of Hg from the transversal that
        would be computed by ``self.coset_transversal(H)``.

        """
        if H.order() == 1:
            return g
        # The base of self must be an extension of H.base.
        if not(self.base[:len(H.base)] == H.base):
            self._schreier_sims(base=H.base)
        orbits = H.basic_orbits[:]
        h_transversals = [list(_.values()) for _ in H.basic_transversals]
        transversals = [list(_.values()) for _ in self.basic_transversals]
        base = self.base
        base_ordering = _base_ordering(base, self.degree)
        def step(l, x):
            gamma = min(orbits[l], key = lambda y: base_ordering[y^x])
            i = [base[l]^h for h in h_transversals[l]].index(gamma)
            x = h_transversals[l][i]*x
            if l < len(orbits)-1:
                for u in transversals[l]:
                    if base[l]^u == base[l]^x:
                        break
                x = step(l+1, x*u**-1)*u
            return x
        return step(0, g)

    def coset_table(self, H):
        """Return the standardised (right) coset table of self in H as
        a list of lists.
        """
        # Maybe this should be made to return an instance of CosetTable
        # from fp_groups.py but the class would need to be changed first
        # to be compatible with PermutationGroups

        if not H.is_subgroup(self):
            raise ValueError("The argument must be a subgroup")
        T = self.coset_transversal(H)
        n = len(T)

        A = list(chain.from_iterable((gen, gen**-1)
                    for gen in self.generators))

        table = []
        for i in range(n):
            row = [self._coset_representative(T[i]*x, H) for x in A]
            row = [T.index(r) for r in row]
            table.append(row)

        # standardize (this is the same as the algorithm used in coset_table)
        # If CosetTable is made compatible with PermutationGroups, this
        # should be replaced by table.standardize()
        A = range(len(A))
        gamma = 1
        for alpha, a in product(range(n), A):
            beta = table[alpha][a]
            if beta >= gamma:
                if beta > gamma:
                    for x in A:
                        z = table[gamma][x]
                        table[gamma][x] = table[beta][x]
                        table[beta][x] = z
                        for i in range(n):
                            if table[i][x] == beta:
                                table[i][x] = gamma
                            elif table[i][x] == gamma:
                                table[i][x] = beta
                gamma += 1
            if gamma >= n-1:
                return table

    def center(self):
        r"""
        Return the center of a permutation group.

        Explanation
        ===========

        The center for a group `G` is defined as
        `Z(G) = \{z\in G | \forall g\in G, zg = gz \}`,
        the set of elements of `G` that commute with all elements of `G`.
        It is equal to the centralizer of `G` inside `G`, and is naturally a
        subgroup of `G` ([9]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(4)
        >>> G = D.center()
        >>> G.order()
        2

        See Also
        ========

        centralizer

        Notes
        =====

        This is a naive implementation that is a straightforward application
        of ``.centralizer()``

        """
        if not self._center:
            self._center = self.centralizer(self)
        return self._center

    def centralizer(self, other):
        r"""
        Return the centralizer of a group/set/element.

        Explanation
        ===========

        The centralizer of a set of permutations ``S`` inside
        a group ``G`` is the set of elements of ``G`` that commute with all
        elements of ``S``::

            `C_G(S) = \{ g \in G | gs = sg \forall s \in S\}` ([10])

        Usually, ``S`` is a subset of ``G``, but if ``G`` is a proper subgroup of
        the full symmetric group, we allow for ``S`` to have elements outside
        ``G``.

        It is naturally a subgroup of ``G``; the centralizer of a permutation
        group is equal to the centralizer of any set of generators for that
        group, since any element commuting with the generators commutes with
        any product of the  generators.

        Parameters
        ==========

        other
            a permutation group/list of permutations/single permutation

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> S = SymmetricGroup(6)
        >>> C = CyclicGroup(6)
        >>> H = S.centralizer(C)
        >>> H.is_subgroup(C)
        True

        See Also
        ========

        subgroup_search

        Notes
        =====

        The implementation is an application of ``.subgroup_search()`` with
        tests using a specific base for the group ``G``.

        """
        if hasattr(other, 'generators'):
            if other.is_trivial or self.is_trivial:
                return self
            degree = self.degree
            identity = _af_new(list(range(degree)))
            orbits = other.orbits()
            num_orbits = len(orbits)
            orbits.sort(key=lambda x: -len(x))
            long_base = []
            orbit_reps = [None]*num_orbits
            orbit_reps_indices = [None]*num_orbits
            orbit_descr = [None]*degree
            for i in range(num_orbits):
                orbit = list(orbits[i])
                orbit_reps[i] = orbit[0]
                orbit_reps_indices[i] = len(long_base)
                for point in orbit:
                    orbit_descr[point] = i
                long_base = long_base + orbit
            base, strong_gens = self.schreier_sims_incremental(base=long_base)
            strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
            i = 0
            for i in range(len(base)):
                if strong_gens_distr[i] == [identity]:
                    break
            base = base[:i]
            base_len = i
            for j in range(num_orbits):
                if base[base_len - 1] in orbits[j]:
                    break
            rel_orbits = orbits[: j + 1]
            num_rel_orbits = len(rel_orbits)
            transversals = [None]*num_rel_orbits
            for j in range(num_rel_orbits):
                rep = orbit_reps[j]
                transversals[j] = dict(
                    other.orbit_transversal(rep, pairs=True))
            trivial_test = lambda x: True
            tests = [None]*base_len
            for l in range(base_len):
                if base[l] in orbit_reps:
                    tests[l] = trivial_test
                else:
                    def test(computed_words, l=l):
                        g = computed_words[l]
                        rep_orb_index = orbit_descr[base[l]]
                        rep = orbit_reps[rep_orb_index]
                        im = g._array_form[base[l]]
                        im_rep = g._array_form[rep]
                        tr_el = transversals[rep_orb_index][base[l]]
                        # using the definition of transversal,
                        # base[l]^g = rep^(tr_el*g);
                        # if g belongs to the centralizer, then
                        # base[l]^g = (rep^g)^tr_el
                        return im == tr_el._array_form[im_rep]
                    tests[l] = test

            def prop(g):
                return [rmul(g, gen) for gen in other.generators] == \
                       [rmul(gen, g) for gen in other.generators]
            return self.subgroup_search(prop, base=base,
                                        strong_gens=strong_gens, tests=tests)
        elif hasattr(other, '__getitem__'):
            gens = list(other)
            return self.centralizer(PermutationGroup(gens))
        elif hasattr(other, 'array_form'):
            return self.centralizer(PermutationGroup([other]))

    def commutator(self, G, H):
        """
        Return the commutator of two subgroups.

        Explanation
        ===========

        For a permutation group ``K`` and subgroups ``G``, ``H``, the
        commutator of ``G`` and ``H`` is defined as the group generated
        by all the commutators `[g, h] = hgh^{-1}g^{-1}` for ``g`` in ``G`` and
        ``h`` in ``H``. It is naturally a subgroup of ``K`` ([1], p.27).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> A = AlternatingGroup(5)
        >>> G = S.commutator(S, A)
        >>> G.is_subgroup(A)
        True

        See Also
        ========

        derived_subgroup

        Notes
        =====

        The commutator of two subgroups `H, G` is equal to the normal closure
        of the commutators of all the generators, i.e. `hgh^{-1}g^{-1}` for `h`
        a generator of `H` and `g` a generator of `G` ([1], p.28)

        """
        ggens = G.generators
        hgens = H.generators
        commutators = []
        for ggen in ggens:
            for hgen in hgens:
                commutator = rmul(hgen, ggen, ~hgen, ~ggen)
                if commutator not in commutators:
                    commutators.append(commutator)
        res = self.normal_closure(commutators)
        return res

    def coset_factor(self, g, factor_index=False):
        """Return ``G``'s (self's) coset factorization of ``g``

        Explanation
        ===========

        If ``g`` is an element of ``G`` then it can be written as the product
        of permutations drawn from the Schreier-Sims coset decomposition,

        The permutations returned in ``f`` are those for which
        the product gives ``g``: ``g = f[n]*...f[1]*f[0]`` where ``n = len(B)``
        and ``B = G.base``. f[i] is one of the permutations in
        ``self._basic_orbits[i]``.

        If factor_index==True,
        returns a tuple ``[b[0],..,b[n]]``, where ``b[i]``
        belongs to ``self._basic_orbits[i]``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])

        Define g:

        >>> g = Permutation(7)(1, 2, 4)(3, 6, 5)

        Confirm that it is an element of G:

        >>> G.contains(g)
        True

        Thus, it can be written as a product of factors (up to
        3) drawn from u. See below that a factor from u1 and u2
        and the Identity permutation have been used:

        >>> f = G.coset_factor(g)
        >>> f[2]*f[1]*f[0] == g
        True
        >>> f1 = G.coset_factor(g, True); f1
        [0, 4, 4]
        >>> tr = G.basic_transversals
        >>> f[0] == tr[0][f1[0]]
        True

        If g is not an element of G then [] is returned:

        >>> c = Permutation(5, 6, 7)
        >>> G.coset_factor(c)
        []

        See Also
        ========

        sympy.combinatorics.util._strip

        """
        if isinstance(g, (Cycle, Permutation)):
            g = g.list()
        if len(g) != self._degree:
            # this could either adjust the size or return [] immediately
            # but we don't choose between the two and just signal a possible
            # error
            raise ValueError('g should be the same size as permutations of G')
        I = list(range(self._degree))
        basic_orbits = self.basic_orbits
        transversals = self._transversals
        factors = []
        base = self.base
        h = g
        for i in range(len(base)):
            beta = h[base[i]]
            if beta == base[i]:
                factors.append(beta)
                continue
            if beta not in basic_orbits[i]:
                return []
            u = transversals[i][beta]._array_form
            h = _af_rmul(_af_invert(u), h)
            factors.append(beta)
        if h != I:
            return []
        if factor_index:
            return factors
        tr = self.basic_transversals
        factors = [tr[i][factors[i]] for i in range(len(base))]
        return factors

    def generator_product(self, g, original=False):
        r'''
        Return a list of strong generators `[s1, \dots, sn]`
        s.t `g = sn \times \dots \times s1`. If ``original=True``, make the
        list contain only the original group generators

        '''
        product = []
        if g.is_identity:
            return []
        if g in self.strong_gens:
            if not original or g in self.generators:
                return [g]
            else:
                slp = self._strong_gens_slp[g]
                for s in slp:
                    product.extend(self.generator_product(s, original=True))
                return product
        elif g**-1 in self.strong_gens:
            g = g**-1
            if not original or g in self.generators:
                return [g**-1]
            else:
                slp = self._strong_gens_slp[g]
                for s in slp:
                    product.extend(self.generator_product(s, original=True))
                l = len(product)
                product = [product[l-i-1]**-1 for i in range(l)]
                return product

        f = self.coset_factor(g, True)
        for i, j in enumerate(f):
            slp = self._transversal_slp[i][j]
            for s in slp:
                if not original:
                    product.append(self.strong_gens[s])
                else:
                    s = self.strong_gens[s]
                    product.extend(self.generator_product(s, original=True))
        return product

    def coset_rank(self, g):
        """rank using Schreier-Sims representation.

        Explanation
        ===========

        The coset rank of ``g`` is the ordering number in which
        it appears in the lexicographic listing according to the
        coset decomposition

        The ordering is the same as in G.generate(method='coset').
        If ``g`` does not belong to the group it returns None.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(0, 1, 3, 7, 6, 4)(2, 5)
        >>> b = Permutation(0, 1, 3, 2)(4, 5, 7, 6)
        >>> G = PermutationGroup([a, b])
        >>> c = Permutation(7)(2, 4)(3, 5)
        >>> G.coset_rank(c)
        16
        >>> G.coset_unrank(16)
        (7)(2 4)(3 5)

        See Also
        ========

        coset_factor

        """
        factors = self.coset_factor(g, True)
        if not factors:
            return None
        rank = 0
        b = 1
        transversals = self._transversals
        base = self._base
        basic_orbits = self._basic_orbits
        for i in range(len(base)):
            k = factors[i]
            j = basic_orbits[i].index(k)
            rank += b*j
            b = b*len(transversals[i])
        return rank

    def coset_unrank(self, rank, af=False):
        """unrank using Schreier-Sims representation

        coset_unrank is the inverse operation of coset_rank
        if 0 <= rank < order; otherwise it returns None.

        """
        if rank < 0 or rank >= self.order():
            return None
        base = self.base
        transversals = self.basic_transversals
        basic_orbits = self.basic_orbits
        m = len(base)
        v = [0]*m
        for i in range(m):
            rank, c = divmod(rank, len(transversals[i]))
            v[i] = basic_orbits[i][c]
        a = [transversals[i][v[i]]._array_form for i in range(m)]
        h = _af_rmuln(*a)
        if af:
            return h
        else:
            return _af_new(h)

    @property
    def degree(self):
        """Returns the size of the permutations in the group.

        Explanation
        ===========

        The number of permutations comprising the group is given by
        ``len(group)``; the number of permutations that can be generated
        by the group is given by ``group.order()``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a])
        >>> G.degree
        3
        >>> len(G)
        1
        >>> G.order()
        2
        >>> list(G.generate())
        [(2), (2)(0 1)]

        See Also
        ========

        order
        """
        return self._degree

    @property
    def identity(self):
        '''
        Return the identity element of the permutation group.

        '''
        return _af_new(list(range(self.degree)))

    @property
    def elements(self):
        """Returns all the elements of the permutation group as a list

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> p = PermutationGroup(Permutation(1, 3), Permutation(1, 2))
        >>> p.elements
        [(3), (3)(1 2), (1 3), (2 3), (1 2 3), (1 3 2)]

        """
        if not self._elements:
            self._elements = list(self.generate())

        return self._elements

    def derived_series(self):
        r"""Return the derived series for the group.

        Explanation
        ===========

        The derived series for a group `G` is defined as
        `G = G_0 > G_1 > G_2 > \ldots` where `G_i = [G_{i-1}, G_{i-1}]`,
        i.e. `G_i` is the derived subgroup of `G_{i-1}`, for
        `i\in\mathbb{N}`. When we have `G_k = G_{k-1}` for some
        `k\in\mathbb{N}`, the series terminates.

        Returns
        =======

        A list of permutation groups containing the members of the derived
        series in the order `G = G_0, G_1, G_2, \ldots`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup, DihedralGroup)
        >>> A = AlternatingGroup(5)
        >>> len(A.derived_series())
        1
        >>> S = SymmetricGroup(4)
        >>> len(S.derived_series())
        4
        >>> S.derived_series()[1].is_subgroup(AlternatingGroup(4))
        True
        >>> S.derived_series()[2].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        derived_subgroup

        """
        res = [self]
        current = self
        nxt = self.derived_subgroup()
        while not current.is_subgroup(nxt):
            res.append(nxt)
            current = nxt
            nxt = nxt.derived_subgroup()
        return res

    def derived_subgroup(self):
        r"""Compute the derived subgroup.

        Explanation
        ===========

        The derived subgroup, or commutator subgroup is the subgroup generated
        by all commutators `[g, h] = hgh^{-1}g^{-1}` for `g, h\in G` ; it is
        equal to the normal closure of the set of commutators of the generators
        ([1], p.28, [11]).

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 0, 2, 4, 3])
        >>> b = Permutation([0, 1, 3, 2, 4])
        >>> G = PermutationGroup([a, b])
        >>> C = G.derived_subgroup()
        >>> list(C.generate(af=True))
        [[0, 1, 2, 3, 4], [0, 1, 3, 4, 2], [0, 1, 4, 2, 3]]

        See Also
        ========

        derived_series

        """
        r = self._r
        gens = [p._array_form for p in self.generators]
        set_commutators = set()
        degree = self._degree
        rng = list(range(degree))
        for i in range(r):
            for j in range(r):
                p1 = gens[i]
                p2 = gens[j]
                c = list(range(degree))
                for k in rng:
                    c[p2[p1[k]]] = p1[p2[k]]
                ct = tuple(c)
                if ct not in set_commutators:
                    set_commutators.add(ct)
        cms = [_af_new(p) for p in set_commutators]
        G2 = self.normal_closure(cms)
        return G2

    def generate(self, method="coset", af=False):
        """Return iterator to generate the elements of the group.

        Explanation
        ===========

        Iteration is done with one of these methods::

          method='coset'  using the Schreier-Sims coset representation
          method='dimino' using the Dimino method

        If ``af = True`` it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import PermutationGroup
        >>> from sympy.combinatorics.polyhedron import tetrahedron

        The permutation group given in the tetrahedron object is also
        true groups:

        >>> G = tetrahedron.pgroup
        >>> G.is_group
        True

        Also the group generated by the permutations in the tetrahedron
        pgroup -- even the first two -- is a proper group:

        >>> H = PermutationGroup(G[0], G[1])
        >>> J = PermutationGroup(list(H.generate())); J
        PermutationGroup([
            (0 1)(2 3),
            (1 2 3),
            (1 3 2),
            (0 3 1),
            (0 2 3),
            (0 3)(1 2),
            (0 1 3),
            (3)(0 2 1),
            (0 3 2),
            (3)(0 1 2),
            (0 2)(1 3)])
        >>> _.is_group
        True
        """
        if method == "coset":
            return self.generate_schreier_sims(af)
        elif method == "dimino":
            return self.generate_dimino(af)
        else:
            raise NotImplementedError('No generation defined for %s' % method)

    def generate_dimino(self, af=False):
        """Yield group elements using Dimino's algorithm.

        If ``af == True`` it yields the array form of the permutations.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_dimino(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 3, 1],
         [0, 1, 3, 2], [0, 3, 2, 1], [0, 3, 1, 2]]

        References
        ==========

        .. [1] The Implementation of Various Algorithms for Permutation Groups in
               the Computer Algebra System: AXIOM, N.J. Doye, M.Sc. Thesis

        """
        idn = list(range(self.degree))
        order = 0
        element_list = [idn]
        set_element_list = {tuple(idn)}
        if af:
            yield idn
        else:
            yield _af_new(idn)
        gens = [p._array_form for p in self.generators]

        for i in range(len(gens)):
            # D elements of the subgroup G_i generated by gens[:i]
            D = element_list.copy()
            N = [idn]
            while N:
                A = N
                N = []
                for a in A:
                    for g in gens[:i + 1]:
                        ag = _af_rmul(a, g)
                        if tuple(ag) not in set_element_list:
                            # produce G_i*g
                            for d in D:
                                order += 1
                                ap = _af_rmul(d, ag)
                                if af:
                                    yield ap
                                else:
                                    p = _af_new(ap)
                                    yield p
                                element_list.append(ap)
                                set_element_list.add(tuple(ap))
                                N.append(ap)
        self._order = len(element_list)

    def generate_schreier_sims(self, af=False):
        """Yield group elements using the Schreier-Sims representation
        in coset_rank order

        If ``af = True`` it yields the array form of the permutations

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([0, 2, 3, 1])
        >>> g = PermutationGroup([a, b])
        >>> list(g.generate_schreier_sims(af=True))
        [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 2, 1],
         [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 1, 2]]
        """

        n = self._degree
        u = self.basic_transversals
        basic_orbits = self._basic_orbits
        if len(u) == 0:
            for x in self.generators:
                if af:
                    yield x._array_form
                else:
                    yield x
            return
        if len(u) == 1:
            for i in basic_orbits[0]:
                if af:
                    yield u[0][i]._array_form
                else:
                    yield u[0][i]
            return

        u = list(reversed(u))
        basic_orbits = basic_orbits[::-1]
        # stg stack of group elements
        stg = [list(range(n))]
        posmax = [len(x) for x in u]
        n1 = len(posmax) - 1
        pos = [0]*n1
        h = 0
        while 1:
            # backtrack when finished iterating over coset
            if pos[h] >= posmax[h]:
                if h == 0:
                    return
                pos[h] = 0
                h -= 1
                stg.pop()
                continue
            p = _af_rmul(u[h][basic_orbits[h][pos[h]]]._array_form, stg[-1])
            pos[h] += 1
            stg.append(p)
            h += 1
            if h == n1:
                if af:
                    for i in basic_orbits[-1]:
                        p = _af_rmul(u[-1][i]._array_form, stg[-1])
                        yield p
                else:
                    for i in basic_orbits[-1]:
                        p = _af_rmul(u[-1][i]._array_form, stg[-1])
                        p1 = _af_new(p)
                        yield p1
                stg.pop()
                h -= 1

    @property
    def generators(self):
        """Returns the generators of the group.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.generators
        [(1 2), (2)(0 1)]

        """
        return self._generators

    def contains(self, g, strict=True):
        """Test if permutation ``g`` belong to self, ``G``.

        Explanation
        ===========

        If ``g`` is an element of ``G`` it can be written as a product
        of factors drawn from the cosets of ``G``'s stabilizers. To see
        if ``g`` is one of the actual generators defining the group use
        ``G.has(g)``.

        If ``strict`` is not ``True``, ``g`` will be resized, if necessary,
        to match the size of permutations in ``self``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup

        >>> a = Permutation(1, 2)
        >>> b = Permutation(2, 3, 1)
        >>> G = PermutationGroup(a, b, degree=5)
        >>> G.contains(G[0]) # trivial check
        True
        >>> elem = Permutation([[2, 3]], size=5)
        >>> G.contains(elem)
        True
        >>> G.contains(Permutation(4)(0, 1, 2, 3))
        False

        If strict is False, a permutation will be resized, if
        necessary:

        >>> H = PermutationGroup(Permutation(5))
        >>> H.contains(Permutation(3))
        False
        >>> H.contains(Permutation(3), strict=False)
        True

        To test if a given permutation is present in the group:

        >>> elem in G.generators
        False
        >>> G.has(elem)
        False

        See Also
        ========

        coset_factor, sympy.core.basic.Basic.has, __contains__

        """
        if not isinstance(g, Permutation):
            return False
        if g.size != self.degree:
            if strict:
                return False
            g = Permutation(g, size=self.degree)
        if g in self.generators:
            return True
        return bool(self.coset_factor(g.array_form, True))

    @property
    def is_perfect(self):
        """Return ``True`` if the group is perfect.
        A group is perfect if it equals to its derived subgroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1,2,3)(4,5)
        >>> b = Permutation(1,2,3,4,5)
        >>> G = PermutationGroup([a, b])
        >>> G.is_perfect
        False

        """
        if self._is_perfect is None:
            self._is_perfect = self.equals(self.derived_subgroup())
        return self._is_perfect

    @property
    def is_abelian(self):
        """Test if the group is Abelian.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.is_abelian
        False
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_abelian
        True

        """
        if self._is_abelian is not None:
            return self._is_abelian

        self._is_abelian = True
        gens = [p._array_form for p in self.generators]
        for x in gens:
            for y in gens:
                if y <= x:
                    continue
                if not _af_commutes_with(x, y):
                    self._is_abelian = False
                    return False
        return True

    def abelian_invariants(self):
        """
        Returns the abelian invariants for the given group.
        Let ``G`` be a nontrivial finite abelian group. Then G is isomorphic to
        the direct product of finitely many nontrivial cyclic groups of
        prime-power order.

        Explanation
        ===========

        The prime-powers that occur as the orders of the factors are uniquely
        determined by G. More precisely, the primes that occur in the orders of the
        factors in any such decomposition of ``G`` are exactly the primes that divide
        ``|G|`` and for any such prime ``p``, if the orders of the factors that are
        p-groups in one such decomposition of ``G`` are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``,
        then the orders of the factors that are p-groups in any such decomposition of ``G``
        are ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``.

        The uniquely determined integers ``p^{t_1} >= p^{t_2} >= ... p^{t_r}``, taken
        for all primes that divide ``|G|`` are called the invariants of the nontrivial
        group ``G`` as suggested in ([14], p. 542).

        Notes
        =====

        We adopt the convention that the invariants of a trivial group are [].

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.abelian_invariants()
        [2]
        >>> from sympy.combinatorics import CyclicGroup
        >>> G = CyclicGroup(7)
        >>> G.abelian_invariants()
        [7]

        """
        if self.is_trivial:
            return []
        gns = self.generators
        inv = []
        G = self
        H = G.derived_subgroup()
        Hgens = H.generators
        for p in primefactors(G.order()):
            ranks = []
            while True:
                pows = []
                for g in gns:
                    elm = g**p
                    if not H.contains(elm):
                        pows.append(elm)
                K = PermutationGroup(Hgens + pows) if pows else H
                r = G.order()//K.order()
                G = K
                gns = pows
                if r == 1:
                    break
                ranks.append(multiplicity(p, r))

            if ranks:
                pows = [1]*ranks[0]
                for i in ranks:
                    for j in range(i):
                        pows[j] = pows[j]*p
                inv.extend(pows)
        inv.sort()
        return inv

    def is_elementary(self, p):
        """Return ``True`` if the group is elementary abelian. An elementary
        abelian group is a finite abelian group, where every nontrivial
        element has order `p`, where `p` is a prime.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_elementary(2)
        True
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([3, 1, 2, 0])
        >>> G = PermutationGroup([a, b])
        >>> G.is_elementary(2)
        True
        >>> G.is_elementary(3)
        False

        """
        return self.is_abelian and all(g.order() == p for g in self.generators)

    def _eval_is_alt_sym_naive(self, only_sym=False, only_alt=False):
        """A naive test using the group order."""
        if only_sym and only_alt:
            raise ValueError(
                "Both {} and {} cannot be set to True"
                .format(only_sym, only_alt))

        n = self.degree
        sym_order = _factorial(n)
        order = self.order()

        if order == sym_order:
            self._is_sym = True
            self._is_alt = False
            return not only_alt

        if 2*order == sym_order:
            self._is_sym = False
            self._is_alt = True
            return not only_sym

        return False

    def _eval_is_alt_sym_monte_carlo(self, eps=0.05, perms=None):
        """A test using monte-carlo algorithm.

        Parameters
        ==========

        eps : float, optional
            The criterion for the incorrect ``False`` return.

        perms : list[Permutation], optional
            If explicitly given, it tests over the given candidates
            for testing.

            If ``None``, it randomly computes ``N_eps`` and chooses
            ``N_eps`` sample of the permutation from the group.

        See Also
        ========

        _check_cycles_alt_sym
        """
        if perms is None:
            n = self.degree
            if n < 17:
                c_n = 0.34
            else:
                c_n = 0.57
            d_n = (c_n*log(2))/log(n)
            N_eps = int(-log(eps)/d_n)

            perms = (self.random_pr() for i in range(N_eps))
            return self._eval_is_alt_sym_monte_carlo(perms=perms)

        for perm in perms:
            if _check_cycles_alt_sym(perm):
                return True
        return False

    def is_alt_sym(self, eps=0.05, _random_prec=None):
        r"""Monte Carlo test for the symmetric/alternating group for degrees
        >= 8.

        Explanation
        ===========

        More specifically, it is one-sided Monte Carlo with the
        answer True (i.e., G is symmetric/alternating) guaranteed to be
        correct, and the answer False being incorrect with probability eps.

        For degree < 8, the order of the group is checked so the test
        is deterministic.

        Notes
        =====

        The algorithm itself uses some nontrivial results from group theory and
        number theory:
        1) If a transitive group ``G`` of degree ``n`` contains an element
        with a cycle of length ``n/2 < p < n-2`` for ``p`` a prime, ``G`` is the
        symmetric or alternating group ([1], pp. 81-82)
        2) The proportion of elements in the symmetric/alternating group having
        the property described in 1) is approximately `\log(2)/\log(n)`
        ([1], p.82; [2], pp. 226-227).
        The helper function ``_check_cycles_alt_sym`` is used to
        go over the cycles in a permutation and look for ones satisfying 1).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_alt_sym()
        False

        See Also
        ========

        _check_cycles_alt_sym

        """
        if _random_prec is not None:
            N_eps = _random_prec['N_eps']
            perms= (_random_prec[i] for i in range(N_eps))
            return self._eval_is_alt_sym_monte_carlo(perms=perms)

        if self._is_sym or self._is_alt:
            return True
        if self._is_sym is False and self._is_alt is False:
            return False

        n = self.degree
        if n < 8:
            return self._eval_is_alt_sym_naive()
        elif self.is_transitive():
            return self._eval_is_alt_sym_monte_carlo(eps=eps)

        self._is_sym, self._is_alt = False, False
        return False

    @property
    def is_nilpotent(self):
        """Test if the group is nilpotent.

        Explanation
        ===========

        A group `G` is nilpotent if it has a central series of finite length.
        Alternatively, `G` is nilpotent if its lower central series terminates
        with the trivial group. Every nilpotent group is also solvable
        ([1], p.29, [12]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> C = CyclicGroup(6)
        >>> C.is_nilpotent
        True
        >>> S = SymmetricGroup(5)
        >>> S.is_nilpotent
        False

        See Also
        ========

        lower_central_series, is_solvable

        """
        if self._is_nilpotent is None:
            lcs = self.lower_central_series()
            terminator = lcs[len(lcs) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                self._is_nilpotent = True
                return True
            else:
                self._is_nilpotent = False
                return False
        else:
            return self._is_nilpotent

    def is_normal(self, gr, strict=True):
        """Test if ``G=self`` is a normal subgroup of ``gr``.

        Explanation
        ===========

        G is normal in gr if
        for each g2 in G, g1 in gr, ``g = g1*g2*g1**-1`` belongs to G
        It is sufficient to check this for each g1 in gr.generators and
        g2 in G.generators.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G1 = PermutationGroup([a, Permutation([2, 0, 1])])
        >>> G1.is_normal(G)
        True

        """
        if not self.is_subgroup(gr, strict=strict):
            return False
        d_self = self.degree
        d_gr = gr.degree
        if self.is_trivial and (d_self == d_gr or not strict):
            return True
        if self._is_abelian:
            return True
        new_self = self.copy()
        if not strict and d_self != d_gr:
            if d_self < d_gr:
                new_self = PermGroup(new_self.generators + [Permutation(d_gr - 1)])
            else:
                gr = PermGroup(gr.generators + [Permutation(d_self - 1)])
        gens2 = [p._array_form for p in new_self.generators]
        gens1 = [p._array_form for p in gr.generators]
        for g1 in gens1:
            for g2 in gens2:
                p = _af_rmuln(g1, g2, _af_invert(g1))
                if not new_self.coset_factor(p, True):
                    return False
        return True

    def is_primitive(self, randomized=True):
        r"""Test if a group is primitive.

        Explanation
        ===========

        A permutation group ``G`` acting on a set ``S`` is called primitive if
        ``S`` contains no nontrivial block under the action of ``G``
        (a block is nontrivial if its cardinality is more than ``1``).

        Notes
        =====

        The algorithm is described in [1], p.83, and uses the function
        minimal_block to search for blocks of the form `\{0, k\}` for ``k``
        ranging over representatives for the orbits of `G_0`, the stabilizer of
        ``0``. This algorithm has complexity `O(n^2)` where ``n`` is the degree
        of the group, and will perform badly if `G_0` is small.

        There are two implementations offered: one finds `G_0`
        deterministically using the function ``stabilizer``, and the other
        (default) produces random elements of `G_0` using ``random_stab``,
        hoping that they generate a subgroup of `G_0` with not too many more
        orbits than `G_0` (this is suggested in [1], p.83). Behavior is changed
        by the ``randomized`` flag.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.is_primitive()
        False

        See Also
        ========

        minimal_block, random_stab

        """
        if self._is_primitive is not None:
            return self._is_primitive

        if self.is_transitive() is False:
            return False

        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for _ in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0 and any(e != 0 for e in self.minimal_block([0, x])):
                self._is_primitive = False
                return False
        self._is_primitive = True
        return True

    def minimal_blocks(self, randomized=True):
        '''
        For a transitive group, return the list of all minimal
        block systems. If a group is intransitive, return `False`.

        Examples
        ========
        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> DihedralGroup(6).minimal_blocks()
        [[0, 1, 0, 1, 0, 1], [0, 1, 2, 0, 1, 2]]
        >>> G = PermutationGroup(Permutation(1,2,5))
        >>> G.minimal_blocks()
        False

        See Also
        ========

        minimal_block, is_transitive, is_primitive

        '''
        def _number_blocks(blocks):
            # number the blocks of a block system
            # in order and return the number of
            # blocks and the tuple with the
            # reordering
            n = len(blocks)
            appeared = {}
            m = 0
            b = [None]*n
            for i in range(n):
                if blocks[i] not in appeared:
                    appeared[blocks[i]] = m
                    b[i] = m
                    m += 1
                else:
                    b[i] = appeared[blocks[i]]
            return tuple(b), m

        if not self.is_transitive():
            return False
        blocks = []
        num_blocks = []
        rep_blocks = []
        if randomized:
            random_stab_gens = []
            v = self.schreier_vector(0)
            for i in range(len(self)):
                random_stab_gens.append(self.random_stab(0, v))
            stab = PermutationGroup(random_stab_gens)
        else:
            stab = self.stabilizer(0)
        orbits = stab.orbits()
        for orb in orbits:
            x = orb.pop()
            if x != 0:
                block = self.minimal_block([0, x])
                num_block, _ = _number_blocks(block)
                # a representative block (containing 0)
                rep = {j for j in range(self.degree) if num_block[j] == 0}
                # check if the system is minimal with
                # respect to the already discovere ones
                minimal = True
                blocks_remove_mask = [False] * len(blocks)
                for i, r in enumerate(rep_blocks):
                    if len(r) > len(rep) and rep.issubset(r):
                        # i-th block system is not minimal
                        blocks_remove_mask[i] = True
                    elif len(r) < len(rep) and r.issubset(rep):
                        # the system being checked is not minimal
                        minimal = False
                        break
                # remove non-minimal representative blocks
                blocks = [b for i, b in enumerate(blocks) if not blocks_remove_mask[i]]
                num_blocks = [n for i, n in enumerate(num_blocks) if not blocks_remove_mask[i]]
                rep_blocks = [r for i, r in enumerate(rep_blocks) if not blocks_remove_mask[i]]

                if minimal and num_block not in num_blocks:
                    blocks.append(block)
                    num_blocks.append(num_block)
                    rep_blocks.append(rep)
        return blocks

    @property
    def is_solvable(self):
        """Test if the group is solvable.

        ``G`` is solvable if its derived series terminates with the trivial
        group ([1], p.29).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(3)
        >>> S.is_solvable
        True

        See Also
        ========

        is_nilpotent, derived_series

        """
        if self._is_solvable is None:
            if self.order() % 2 != 0:
                return True
            ds = self.derived_series()
            terminator = ds[len(ds) - 1]
            gens = terminator.generators
            degree = self.degree
            identity = _af_new(list(range(degree)))
            if all(g == identity for g in gens):
                self._is_solvable = True
                return True
            else:
                self._is_solvable = False
                return False
        else:
            return self._is_solvable

    def is_subgroup(self, G, strict=True):
        """Return ``True`` if all elements of ``self`` belong to ``G``.

        If ``strict`` is ``False`` then if ``self``'s degree is smaller
        than ``G``'s, the elements will be resized to have the same degree.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.combinatorics import SymmetricGroup, CyclicGroup

        Testing is strict by default: the degree of each group must be the
        same:

        >>> p = Permutation(0, 1, 2, 3, 4, 5)
        >>> G1 = PermutationGroup([Permutation(0, 1, 2), Permutation(0, 1)])
        >>> G2 = PermutationGroup([Permutation(0, 2), Permutation(0, 1, 2)])
        >>> G3 = PermutationGroup([p, p**2])
        >>> assert G1.order() == G2.order() == G3.order() == 6
        >>> G1.is_subgroup(G2)
        True
        >>> G1.is_subgroup(G3)
        False
        >>> G3.is_subgroup(PermutationGroup(G3[1]))
        False
        >>> G3.is_subgroup(PermutationGroup(G3[0]))
        True

        To ignore the size, set ``strict`` to ``False``:

        >>> S3 = SymmetricGroup(3)
        >>> S5 = SymmetricGroup(5)
        >>> S3.is_subgroup(S5, strict=False)
        True
        >>> C7 = CyclicGroup(7)
        >>> G = S5*C7
        >>> S5.is_subgroup(G, False)
        True
        >>> C7.is_subgroup(G, 0)
        False

        """
        if isinstance(G, SymmetricPermutationGroup):
            if self.degree != G.degree:
                return False
            return True
        if not isinstance(G, PermutationGroup):
            return False
        if self == G or self.generators[0]==Permutation():
            return True
        if G.order() % self.order() != 0:
            return False
        if self.degree == G.degree or \
                (self.degree < G.degree and not strict):
            gens = self.generators
        else:
            return False
        return all(G.contains(g, strict=strict) for g in gens)

    @property
    def is_polycyclic(self):
        """Return ``True`` if a group is polycyclic. A group is polycyclic if
        it has a subnormal series with cyclic factors. For finite groups,
        this is the same as if the group is solvable.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G = PermutationGroup([a, b])
        >>> G.is_polycyclic
        True

        """
        return self.is_solvable

    def is_transitive(self, strict=True):
        """Test if the group is transitive.

        Explanation
        ===========

        A group is transitive if it has a single orbit.

        If ``strict`` is ``False`` the group is transitive if it has
        a single orbit of length different from 1.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G1 = PermutationGroup([a, b])
        >>> G1.is_transitive()
        False
        >>> G1.is_transitive(strict=False)
        True
        >>> c = Permutation([2, 3, 0, 1])
        >>> G2 = PermutationGroup([a, c])
        >>> G2.is_transitive()
        True
        >>> d = Permutation([1, 0, 2, 3])
        >>> e = Permutation([0, 1, 3, 2])
        >>> G3 = PermutationGroup([d, e])
        >>> G3.is_transitive() or G3.is_transitive(strict=False)
        False

        """
        if self._is_transitive:  # strict or not, if True then True
            return self._is_transitive
        if strict:
            if self._is_transitive is not None:  # we only store strict=True
                return self._is_transitive

            ans = len(self.orbit(0)) == self.degree
            self._is_transitive = ans
            return ans

        got_orb = False
        for x in self.orbits():
            if len(x) > 1:
                if got_orb:
                    return False
                got_orb = True
        return got_orb

    @property
    def is_trivial(self):
        """Test if the group is the trivial group.

        This is true if the group contains only the identity permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 1, 2])])
        >>> G.is_trivial
        True

        """
        if self._is_trivial is None:
            self._is_trivial = len(self) == 1 and self[0].is_Identity
        return self._is_trivial

    def lower_central_series(self):
        r"""Return the lower central series for the group.

        The lower central series for a group `G` is the series
        `G = G_0 > G_1 > G_2 > \ldots` where
        `G_k = [G, G_{k-1}]`, i.e. every term after the first is equal to the
        commutator of `G` and the previous term in `G1` ([1], p.29).

        Returns
        =======

        A list of permutation groups in the order `G = G_0, G_1, G_2, \ldots`

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (AlternatingGroup,
        ... DihedralGroup)
        >>> A = AlternatingGroup(4)
        >>> len(A.lower_central_series())
        2
        >>> A.lower_central_series()[1].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        commutator, derived_series

        """
        res = [self]
        current = self
        nxt = self.commutator(self, current)
        while not current.is_subgroup(nxt):
            res.append(nxt)
            current = nxt
            nxt = self.commutator(self, current)
        return res

    @property
    def max_div(self):
        """Maximum proper divisor of the degree of a permutation group.

        Explanation
        ===========

        Obviously, this is the degree divided by its minimal proper divisor
        (larger than ``1``, if one exists). As it is guaranteed to be prime,
        the ``sieve`` from ``sympy.ntheory`` is used.
        This function is also used as an optimization tool for the functions
        ``minimal_block`` and ``_union_find_merge``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> G = PermutationGroup([Permutation([0, 2, 1, 3])])
        >>> G.max_div
        2

        See Also
        ========

        minimal_block, _union_find_merge

        """
        if self._max_div is not None:
            return self._max_div
        n = self.degree
        if n == 1:
            return 1
        for x in sieve:
            if n % x == 0:
                d = n//x
                self._max_div = d
                return d

    def minimal_block(self, points):
        r"""For a transitive group, finds the block system generated by
        ``points``.

        Explanation
        ===========

        If a group ``G`` acts on a set ``S``, a nonempty subset ``B`` of ``S``
        is called a block under the action of ``G`` if for all ``g`` in ``G``
        we have ``gB = B`` (``g`` fixes ``B``) or ``gB`` and ``B`` have no
        common points (``g`` moves ``B`` entirely). ([1], p.23; [6]).

        The distinct translates ``gB`` of a block ``B`` for ``g`` in ``G``
        partition the set ``S`` and this set of translates is known as a block
        system. Moreover, we obviously have that all blocks in the partition
        have the same size, hence the block size divides ``|S|`` ([1], p.23).
        A ``G``-congruence is an equivalence relation ``~`` on the set ``S``
        such that ``a ~ b`` implies ``g(a) ~ g(b)`` for all ``g`` in ``G``.
        For a transitive group, the equivalence classes of a ``G``-congruence
        and the blocks of a block system are the same thing ([1], p.23).

        The algorithm below checks the group for transitivity, and then finds
        the ``G``-congruence generated by the pairs ``(p_0, p_1), (p_0, p_2),
        ..., (p_0,p_{k-1})`` which is the same as finding the maximal block
        system (i.e., the one with minimum block size) such that
        ``p_0, ..., p_{k-1}`` are in the same block ([1], p.83).

        It is an implementation of Atkinson's algorithm, as suggested in [1],
        and manipulates an equivalence relation on the set ``S`` using a
        union-find data structure. The running time is just above
        `O(|points||S|)`. ([1], pp. 83-87; [7]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(10)
        >>> D.minimal_block([0, 5])
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        >>> D.minimal_block([0, 1])
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        See Also
        ========

        _union_find_rep, _union_find_merge, is_transitive, is_primitive

        """
        if not self.is_transitive():
            return False
        n = self.degree
        gens = self.generators
        # initialize the list of equivalence class representatives
        parents = list(range(n))
        ranks = [1]*n
        not_rep = []
        k = len(points)
        # the block size must divide the degree of the group
        if k > self.max_div:
            return [0]*n
        for i in range(k - 1):
            parents[points[i + 1]] = points[0]
            not_rep.append(points[i + 1])
        ranks[points[0]] = k
        i = 0
        len_not_rep = k - 1
        while i < len_not_rep:
            gamma = not_rep[i]
            i += 1
            for gen in gens:
                # find has side effects: performs path compression on the list
                # of representatives
                delta = self._union_find_rep(gamma, parents)
                # union has side effects: performs union by rank on the list
                # of representatives
                temp = self._union_find_merge(gen(gamma), gen(delta), ranks,
                                              parents, not_rep)
                if temp == -1:
                    return [0]*n
                len_not_rep += temp
        for i in range(n):
            # force path compression to get the final state of the equivalence
            # relation
            self._union_find_rep(i, parents)

        # rewrite result so that block representatives are minimal
        new_reps = {}
        return [new_reps.setdefault(r, i) for i, r in enumerate(parents)]

    def conjugacy_class(self, x):
        r"""Return the conjugacy class of an element in the group.

        Explanation
        ===========

        The conjugacy class of an element ``g`` in a group ``G`` is the set of
        elements ``x`` in ``G`` that are conjugate with ``g``, i.e. for which

            ``g = xax^{-1}``

        for some ``a`` in ``G``.

        Note that conjugacy is an equivalence relation, and therefore that
        conjugacy classes are partitions of ``G``. For a list of all the
        conjugacy classes of the group, use the conjugacy_classes() method.

        In a permutation group, each conjugacy class corresponds to a particular
        `cycle structure': for example, in ``S_3``, the conjugacy classes are:

            * the identity class, ``{()}``
            * all transpositions, ``{(1 2), (1 3), (2 3)}``
            * all 3-cycles, ``{(1 2 3), (1 3 2)}``

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricGroup
        >>> S3 = SymmetricGroup(3)
        >>> S3.conjugacy_class(Permutation(0, 1, 2))
        {(0 1 2), (0 2 1)}

        Notes
        =====

        This procedure computes the conjugacy class directly by finding the
        orbit of the element under conjugation in G. This algorithm is only
        feasible for permutation groups of relatively small order, but is like
        the orbit() function itself in that respect.
        """
        # Ref: "Computing the conjugacy classes of finite groups"; Butler, G.
        # Groups '93 Galway/St Andrews; edited by Campbell, C. M.
        new_class = {x}
        last_iteration = new_class

        while len(last_iteration) > 0:
            this_iteration = set()

            for y in last_iteration:
                for s in self.generators:
                    conjugated = s * y * (~s)
                    if conjugated not in new_class:
                        this_iteration.add(conjugated)

            new_class.update(last_iteration)
            last_iteration = this_iteration

        return new_class


    def conjugacy_classes(self):
        r"""Return the conjugacy classes of the group.

        Explanation
        ===========

        As described in the documentation for the .conjugacy_class() function,
        conjugacy is an equivalence relation on a group G which partitions the
        set of elements. This method returns a list of all these conjugacy
        classes of G.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricGroup
        >>> SymmetricGroup(3).conjugacy_classes()
        [{(2)}, {(0 1 2), (0 2 1)}, {(0 2), (1 2), (2)(0 1)}]

        """
        identity = _af_new(list(range(self.degree)))
        known_elements = {identity}
        classes = [known_elements.copy()]

        for x in self.generate():
            if x not in known_elements:
                new_class = self.conjugacy_class(x)
                classes.append(new_class)
                known_elements.update(new_class)

        return classes

    def normal_closure(self, other, k=10):
        r"""Return the normal closure of a subgroup/set of permutations.

        Explanation
        ===========

        If ``S`` is a subset of a group ``G``, the normal closure of ``A`` in ``G``
        is defined as the intersection of all normal subgroups of ``G`` that
        contain ``A`` ([1], p.14). Alternatively, it is the group generated by
        the conjugates ``x^{-1}yx`` for ``x`` a generator of ``G`` and ``y`` a
        generator of the subgroup ``\left\langle S\right\rangle`` generated by
        ``S`` (for some chosen generating set for ``\left\langle S\right\rangle``)
        ([1], p.73).

        Parameters
        ==========

        other
            a subgroup/list of permutations/single permutation
        k
            an implementation-specific parameter that determines the number
            of conjugates that are adjoined to ``other`` at once

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup, AlternatingGroup)
        >>> S = SymmetricGroup(5)
        >>> C = CyclicGroup(5)
        >>> G = S.normal_closure(C)
        >>> G.order()
        60
        >>> G.is_subgroup(AlternatingGroup(5))
        True

        See Also
        ========

        commutator, derived_subgroup, random_pr

        Notes
        =====

        The algorithm is described in [1], pp. 73-74; it makes use of the
        generation of random elements for permutation groups by the product
        replacement algorithm.

        """
        if hasattr(other, 'generators'):
            degree = self.degree
            identity = _af_new(list(range(degree)))

            if all(g == identity for g in other.generators):
                return other
            Z = PermutationGroup(other.generators[:])
            base, strong_gens = Z.schreier_sims_incremental()
            strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
            basic_orbits, basic_transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr)

            self._random_pr_init(r=10, n=20)

            _loop = True
            while _loop:
                Z._random_pr_init(r=10, n=10)
                for _ in range(k):
                    g = self.random_pr()
                    h = Z.random_pr()
                    conj = h^g
                    res = _strip(conj, base, basic_orbits, basic_transversals)
                    if res[0] != identity or res[1] != len(base) + 1:
                        gens = Z.generators
                        gens.append(conj)
                        Z = PermutationGroup(gens)
                        strong_gens.append(conj)
                        temp_base, temp_strong_gens = \
                            Z.schreier_sims_incremental(base, strong_gens)
                        base, strong_gens = temp_base, temp_strong_gens
                        strong_gens_distr = \
                            _distribute_gens_by_base(base, strong_gens)
                        basic_orbits, basic_transversals = \
                            _orbits_transversals_from_bsgs(base,
                                strong_gens_distr)
                _loop = False
                for g in self.generators:
                    for h in Z.generators:
                        conj = h^g
                        res = _strip(conj, base, basic_orbits,
                                     basic_transversals)
                        if res[0] != identity or res[1] != len(base) + 1:
                            _loop = True
                            break
                    if _loop:
                        break
            return Z
        elif hasattr(other, '__getitem__'):
            return self.normal_closure(PermutationGroup(other))
        elif hasattr(other, 'array_form'):
            return self.normal_closure(PermutationGroup([other]))

    def orbit(self, alpha, action='tuples'):
        r"""Compute the orbit of alpha `\{g(\alpha) | g \in G\}` as a set.

        Explanation
        ===========

        The time complexity of the algorithm used here is `O(|Orb|*r)` where
        `|Orb|` is the size of the orbit and ``r`` is the number of generators of
        the group. For a more detailed analysis, see [1], p.78, [2], pp. 19-21.
        Here alpha can be a single point, or a list of points.

        If alpha is a single point, the ordinary orbit is computed.
        if alpha is a list of points, there are three available options:

        'union' - computes the union of the orbits of the points in the list
        'tuples' - computes the orbit of the list interpreted as an ordered
        tuple under the group action ( i.e., g((1,2,3)) = (g(1), g(2), g(3)) )
        'sets' - computes the orbit of the list interpreted as a sets

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0, 4, 5, 6, 3])
        >>> G = PermutationGroup([a])
        >>> G.orbit(0)
        {0, 1, 2}
        >>> G.orbit([0, 4], 'union')
        {0, 1, 2, 3, 4, 5, 6}

        See Also
        ========

        orbit_transversal

        """
        return _orbit(self.degree, self.generators, alpha, action)

    def orbit_rep(self, alpha, beta, schreier_vector=None):
        """Return a group element which sends ``alpha`` to ``beta``.

        Explanation
        ===========

        If ``beta`` is not in the orbit of ``alpha``, the function returns
        ``False``. This implementation makes use of the schreier vector.
        For a proof of correctness, see [1], p.80

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> G = AlternatingGroup(5)
        >>> G.orbit_rep(0, 4)
        (0 4 1 2 3)

        See Also
        ========

        schreier_vector

        """
        if schreier_vector is None:
            schreier_vector = self.schreier_vector(alpha)
        if schreier_vector[beta] is None:
            return False
        k = schreier_vector[beta]
        gens = [x._array_form for x in self.generators]
        a = []
        while k != -1:
            a.append(gens[k])
            beta = gens[k].index(beta) # beta = (~gens[k])(beta)
            k = schreier_vector[beta]
        if a:
            return _af_new(_af_rmuln(*a))
        else:
            return _af_new(list(range(self._degree)))

    def orbit_transversal(self, alpha, pairs=False):
        r"""Computes a transversal for the orbit of ``alpha`` as a set.

        Explanation
        ===========

        For a permutation group `G`, a transversal for the orbit
        `Orb = \{g(\alpha) | g \in G\}` is a set
        `\{g_\beta | g_\beta(\alpha) = \beta\}` for `\beta \in Orb`.
        Note that there may be more than one possible transversal.
        If ``pairs`` is set to ``True``, it returns the list of pairs
        `(\beta, g_\beta)`. For a proof of correctness, see [1], p.79

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.orbit_transversal(0)
        [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]

        See Also
        ========

        orbit

        """
        return _orbit_transversal(self._degree, self.generators, alpha, pairs)

    def orbits(self, rep=False):
        """Return the orbits of ``self``, ordered according to lowest element
        in each orbit.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1, 5)(2, 3)(4, 0, 6)
        >>> b = Permutation(1, 5)(3, 4)(2, 6, 0)
        >>> G = PermutationGroup([a, b])
        >>> G.orbits()
        [{0, 2, 3, 4, 6}, {1, 5}]
        """
        return _orbits(self._degree, self._generators)

    def order(self):
        """Return the order of the group: the number of permutations that
        can be generated from elements of the group.

        The number of permutations comprising the group is given by
        ``len(group)``; the length of each permutation in the group is
        given by ``group.size``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup

        >>> a = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a])
        >>> G.degree
        3
        >>> len(G)
        1
        >>> G.order()
        2
        >>> list(G.generate())
        [(2), (2)(0 1)]

        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.order()
        6

        See Also
        ========

        degree

        """
        if self._order is not None:
            return self._order
        if self._is_sym:
            n = self._degree
            self._order = factorial(n)
            return self._order
        if self._is_alt:
            n = self._degree
            self._order = factorial(n)/2
            return self._order

        m = prod([len(x) for x in self.basic_transversals])
        self._order = m
        return m

    def index(self, H):
        """
        Returns the index of a permutation group.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation(1,2,3)
        >>> b =Permutation(3)
        >>> G = PermutationGroup([a])
        >>> H = PermutationGroup([b])
        >>> G.index(H)
        3

        """
        if H.is_subgroup(self):
            return self.order()//H.order()

    @property
    def is_symmetric(self):
        """Return ``True`` if the group is symmetric.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricGroup
        >>> g = SymmetricGroup(5)
        >>> g.is_symmetric
        True

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> g = PermutationGroup(
        ...     Permutation(0, 1, 2, 3, 4),
        ...     Permutation(2, 3))
        >>> g.is_symmetric
        True

        Notes
        =====

        This uses a naive test involving the computation of the full
        group order.
        If you need more quicker taxonomy for large groups, you can use
        :meth:`PermutationGroup.is_alt_sym`.
        However, :meth:`PermutationGroup.is_alt_sym` may not be accurate
        and is not able to distinguish between an alternating group and
        a symmetric group.

        See Also
        ========

        is_alt_sym
        """
        _is_sym = self._is_sym
        if _is_sym is not None:
            return _is_sym

        n = self.degree
        if n >= 8:
            if self.is_transitive():
                _is_alt_sym = self._eval_is_alt_sym_monte_carlo()
                if _is_alt_sym:
                    if any(g.is_odd for g in self.generators):
                        self._is_sym, self._is_alt = True, False
                        return True

                    self._is_sym, self._is_alt = False, True
                    return False

                return self._eval_is_alt_sym_naive(only_sym=True)

            self._is_sym, self._is_alt = False, False
            return False

        return self._eval_is_alt_sym_naive(only_sym=True)


    @property
    def is_alternating(self):
        """Return ``True`` if the group is alternating.

        Examples
        ========

        >>> from sympy.combinatorics import AlternatingGroup
        >>> g = AlternatingGroup(5)
        >>> g.is_alternating
        True

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> g = PermutationGroup(
        ...     Permutation(0, 1, 2, 3, 4),
        ...     Permutation(2, 3, 4))
        >>> g.is_alternating
        True

        Notes
        =====

        This uses a naive test involving the computation of the full
        group order.
        If you need more quicker taxonomy for large groups, you can use
        :meth:`PermutationGroup.is_alt_sym`.
        However, :meth:`PermutationGroup.is_alt_sym` may not be accurate
        and is not able to distinguish between an alternating group and
        a symmetric group.

        See Also
        ========

        is_alt_sym
        """
        _is_alt = self._is_alt
        if _is_alt is not None:
            return _is_alt

        n = self.degree
        if n >= 8:
            if self.is_transitive():
                _is_alt_sym = self._eval_is_alt_sym_monte_carlo()
                if _is_alt_sym:
                    if all(g.is_even for g in self.generators):
                        self._is_sym, self._is_alt = False, True
                        return True

                    self._is_sym, self._is_alt = True, False
                    return False

                return self._eval_is_alt_sym_naive(only_alt=True)

            self._is_sym, self._is_alt = False, False
            return False

        return self._eval_is_alt_sym_naive(only_alt=True)

    @classmethod
    def _distinct_primes_lemma(cls, primes):
        """Subroutine to test if there is only one cyclic group for the
        order."""
        primes = sorted(primes)
        l = len(primes)
        for i in range(l):
            for j in range(i+1, l):
                if primes[j] % primes[i] == 1:
                    return None
        return True

    @property
    def is_cyclic(self):
        r"""
        Return ``True`` if the group is Cyclic.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AbelianGroup
        >>> G = AbelianGroup(3, 4)
        >>> G.is_cyclic
        True
        >>> G = AbelianGroup(4, 4)
        >>> G.is_cyclic
        False

        Notes
        =====

        If the order of a group $n$ can be factored into the distinct
        primes $p_1, p_2, \dots , p_s$ and if

        .. math::
            \forall i, j \in \{1, 2, \dots, s \}:
            p_i \not \equiv 1 \pmod {p_j}

        holds true, there is only one group of the order $n$ which
        is a cyclic group [1]_. This is a generalization of the lemma
        that the group of order $15, 35, \dots$ are cyclic.

        And also, these additional lemmas can be used to test if a
        group is cyclic if the order of the group is already found.

        - If the group is abelian and the order of the group is
          square-free, the group is cyclic.
        - If the order of the group is less than $6$ and is not $4$, the
          group is cyclic.
        - If the order of the group is prime, the group is cyclic.

        References
        ==========

        .. [1] 1978: John S. Rose: A Course on Group Theory,
            Introduction to Finite Group Theory: 1.4
        """
        if self._is_cyclic is not None:
            return self._is_cyclic

        if len(self.generators) == 1:
            self._is_cyclic = True
            self._is_abelian = True
            return True

        if self._is_abelian is False:
            self._is_cyclic = False
            return False

        order = self.order()

        if order < 6:
            self._is_abelian = True
            if order != 4:
                self._is_cyclic = True
                return True

        factors = factorint(order)
        if all(v == 1 for v in factors.values()):
            if self._is_abelian:
                self._is_cyclic = True
                return True

            primes = list(factors.keys())
            if PermutationGroup._distinct_primes_lemma(primes) is True:
                self._is_cyclic = True
                self._is_abelian = True
                return True

        if not self.is_abelian:
            self._is_cyclic = False
            return False

        self._is_cyclic = all(
            any(g**(order//p) != self.identity for g in self.generators)
            for p, e in factors.items() if e > 1
        )
        return self._is_cyclic

    @property
    def is_dihedral(self):
        r"""
        Return ``True`` if the group is dihedral.

        Examples
        ========

        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup
        >>> G = PermutationGroup(Permutation(1, 6)(2, 5)(3, 4), Permutation(0, 1, 2, 3, 4, 5, 6))
        >>> G.is_dihedral
        True
        >>> G = SymmetricGroup(3)
        >>> G.is_dihedral
        True
        >>> G = CyclicGroup(6)
        >>> G.is_dihedral
        False

        References
        ==========

        .. [Di1] https://math.stackexchange.com/questions/827230/given-a-cayley-table-is-there-an-algorithm-to-determine-if-it-is-a-dihedral-gro/827273#827273
        .. [Di2] https://kconrad.math.uconn.edu/blurbs/grouptheory/dihedral.pdf
        .. [Di3] https://kconrad.math.uconn.edu/blurbs/grouptheory/dihedral2.pdf
        .. [Di4] https://en.wikipedia.org/wiki/Dihedral_group
        """
        if self._is_dihedral is not None:
            return self._is_dihedral

        order = self.order()

        if order % 2 == 1:
            self._is_dihedral = False
            return False
        if order == 2:
            self._is_dihedral = True
            return True
        if order == 4:
            # The dihedral group of order 4 is the Klein 4-group.
            self._is_dihedral = not self.is_cyclic
            return self._is_dihedral
        if self.is_abelian:
            # The only abelian dihedral groups are the ones of orders 2 and 4.
            self._is_dihedral = False
            return False

        # Now we know the group is of even order >= 6, and nonabelian.
        n = order // 2

        # Handle special cases where there are exactly two generators.
        gens = self.generators
        if len(gens) == 2:
            x, y = gens
            a, b = x.order(), y.order()
            # Make a >= b
            if a < b:
                x, y, a, b = y, x, b, a
            # Using Theorem 2.1 of [Di3]:
            if a == 2 == b:
                self._is_dihedral = True
                return True
            # Using Theorem 1.1 of [Di3]:
            if a == n and b == 2 and y*x*y == ~x:
                self._is_dihedral = True
                return True

        # Proceed with algorithm of [Di1]
        # Find elements of orders 2 and n
        order_2, order_n = [], []
        for p in self.elements:
            k = p.order()
            if k == 2:
                order_2.append(p)
            elif k == n:
                order_n.append(p)

        if len(order_2) != n + 1 - (n % 2):
            self._is_dihedral = False
            return False

        if not order_n:
            self._is_dihedral = False
            return False

        x = order_n[0]
        # Want an element y of order 2 that is not a power of x
        # (i.e. that is not the 180-deg rotation, when n is even).
        y = order_2[0]
        if n % 2 == 0 and y == x**(n//2):
            y = order_2[1]

        self._is_dihedral = (y*x*y == ~x)
        return self._is_dihedral

    def pointwise_stabilizer(self, points, incremental=True):
        r"""Return the pointwise stabilizer for a set of points.

        Explanation
        ===========

        For a permutation group `G` and a set of points
        `\{p_1, p_2,\ldots, p_k\}`, the pointwise stabilizer of
        `p_1, p_2, \ldots, p_k` is defined as
        `G_{p_1,\ldots, p_k} =
        \{g\in G | g(p_i) = p_i \forall i\in\{1, 2,\ldots,k\}\}` ([1],p20).
        It is a subgroup of `G`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(7)
        >>> Stab = S.pointwise_stabilizer([2, 3, 5])
        >>> Stab.is_subgroup(S.stabilizer(2).stabilizer(3).stabilizer(5))
        True

        See Also
        ========

        stabilizer, schreier_sims_incremental

        Notes
        =====

        When incremental == True,
        rather than the obvious implementation using successive calls to
        ``.stabilizer()``, this uses the incremental Schreier-Sims algorithm
        to obtain a base with starting segment - the given points.

        """
        if incremental:
            base, strong_gens = self.schreier_sims_incremental(base=points)
            stab_gens = []
            degree = self.degree
            for gen in strong_gens:
                if [gen(point) for point in points] == points:
                    stab_gens.append(gen)
            if not stab_gens:
                stab_gens = _af_new(list(range(degree)))
            return PermutationGroup(stab_gens)
        else:
            gens = self._generators
            degree = self.degree
            for x in points:
                gens = _stabilizer(degree, gens, x)
        return PermutationGroup(gens)

    def make_perm(self, n, seed=None):
        """
        Multiply ``n`` randomly selected permutations from
        pgroup together, starting with the identity
        permutation. If ``n`` is a list of integers, those
        integers will be used to select the permutations and they
        will be applied in L to R order: make_perm((A, B, C)) will
        give CBA(I) where I is the identity permutation.

        ``seed`` is used to set the seed for the random selection
        of permutations from pgroup. If this is a list of integers,
        the corresponding permutations from pgroup will be selected
        in the order give. This is mainly used for testing purposes.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a, b = [Permutation([1, 0, 3, 2]), Permutation([1, 3, 0, 2])]
        >>> G = PermutationGroup([a, b])
        >>> G.make_perm(1, [0])
        (0 1)(2 3)
        >>> G.make_perm(3, [0, 1, 0])
        (0 2 3 1)
        >>> G.make_perm([0, 1, 0])
        (0 2 3 1)

        See Also
        ========

        random
        """
        if is_sequence(n):
            if seed is not None:
                raise ValueError('If n is a sequence, seed should be None')
            n, seed = len(n), n
        else:
            try:
                n = int(n)
            except TypeError:
                raise ValueError('n must be an integer or a sequence.')
        randomrange = _randrange(seed)

        # start with the identity permutation
        result = Permutation(list(range(self.degree)))
        m = len(self)
        for _ in range(n):
            p = self[randomrange(m)]
            result = rmul(result, p)
        return result

    def random(self, af=False):
        """Return a random group element
        """
        rank = randrange(self.order())
        return self.coset_unrank(rank, af)

    def random_pr(self, gen_count=11, iterations=50, _random_prec=None):
        """Return a random group element using product replacement.

        Explanation
        ===========

        For the details of the product replacement algorithm, see
        ``_random_pr_init`` In ``random_pr`` the actual 'product replacement'
        is performed. Notice that if the attribute ``_random_gens``
        is empty, it needs to be initialized by ``_random_pr_init``.

        See Also
        ========

        _random_pr_init

        """
        if self._random_gens == []:
            self._random_pr_init(gen_count, iterations)
        random_gens = self._random_gens
        r = len(random_gens) - 1

        # handle randomized input for testing purposes
        if _random_prec is None:
            s = randrange(r)
            t = randrange(r - 1)
            if t == s:
                t = r - 1
            x = choice([1, 2])
            e = choice([-1, 1])
        else:
            s = _random_prec['s']
            t = _random_prec['t']
            if t == s:
                t = r - 1
            x = _random_prec['x']
            e = _random_prec['e']

        if x == 1:
            random_gens[s] = _af_rmul(random_gens[s], _af_pow(random_gens[t], e))
            random_gens[r] = _af_rmul(random_gens[r], random_gens[s])
        else:
            random_gens[s] = _af_rmul(_af_pow(random_gens[t], e), random_gens[s])
            random_gens[r] = _af_rmul(random_gens[s], random_gens[r])
        return _af_new(random_gens[r])

    def random_stab(self, alpha, schreier_vector=None, _random_prec=None):
        """Random element from the stabilizer of ``alpha``.

        The schreier vector for ``alpha`` is an optional argument used
        for speeding up repeated calls. The algorithm is described in [1], p.81

        See Also
        ========

        random_pr, orbit_rep

        """
        if schreier_vector is None:
            schreier_vector = self.schreier_vector(alpha)
        if _random_prec is None:
            rand = self.random_pr()
        else:
            rand = _random_prec['rand']
        beta = rand(alpha)
        h = self.orbit_rep(alpha, beta, schreier_vector)
        return rmul(~h, rand)

    def schreier_sims(self):
        """Schreier-Sims algorithm.

        Explanation
        ===========

        It computes the generators of the chain of stabilizers
        `G > G_{b_1} > .. > G_{b1,..,b_r} > 1`
        in which `G_{b_1,..,b_i}` stabilizes `b_1,..,b_i`,
        and the corresponding ``s`` cosets.
        An element of the group can be written as the product
        `h_1*..*h_s`.

        We use the incremental Schreier-Sims algorithm.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_sims()
        >>> G.basic_transversals
        [{0: (2)(0 1), 1: (2), 2: (1 2)},
         {0: (2), 2: (0 2)}]
        """
        if self._transversals:
            return
        self._schreier_sims()
        return

    def _schreier_sims(self, base=None):
        schreier = self.schreier_sims_incremental(base=base, slp_dict=True)
        base, strong_gens = schreier[:2]
        self._base = base
        self._strong_gens = strong_gens
        self._strong_gens_slp = schreier[2]
        if not base:
            self._transversals = []
            self._basic_orbits = []
            return

        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_orbits, transversals, slps = _orbits_transversals_from_bsgs(base,\
                strong_gens_distr, slp=True)

        # rewrite the indices stored in slps in terms of strong_gens
        for i, slp in enumerate(slps):
            gens = strong_gens_distr[i]
            for k in slp:
                slp[k] = [strong_gens.index(gens[s]) for s in slp[k]]

        self._transversals = transversals
        self._basic_orbits = [sorted(x) for x in basic_orbits]
        self._transversal_slp = slps

    def schreier_sims_incremental(self, base=None, gens=None, slp_dict=False):
        """Extend a sequence of points and generating set to a base and strong
        generating set.

        Parameters
        ==========

        base
            The sequence of points to be extended to a base. Optional
            parameter with default value ``[]``.
        gens
            The generating set to be extended to a strong generating set
            relative to the base obtained. Optional parameter with default
            value ``self.generators``.

        slp_dict
            If `True`, return a dictionary `{g: gens}` for each strong
            generator `g` where `gens` is a list of strong generators
            coming before `g` in `strong_gens`, such that the product
            of the elements of `gens` is equal to `g`.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the base obtained, and ``strong_gens`` is the strong
            generating set relative to it. The original parameters ``base``,
            ``gens`` remain unchanged.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> A = AlternatingGroup(7)
        >>> base = [2, 3]
        >>> seq = [2, 3]
        >>> base, strong_gens = A.schreier_sims_incremental(base=seq)
        >>> _verify_bsgs(A, base, strong_gens)
        True
        >>> base[:2]
        [2, 3]

        Notes
        =====

        This version of the Schreier-Sims algorithm runs in polynomial time.
        There are certain assumptions in the implementation - if the trivial
        group is provided, ``base`` and ``gens`` are returned immediately,
        as any sequence of points is a base for the trivial group. If the
        identity is present in the generators ``gens``, it is removed as
        it is a redundant generator.
        The implementation is described in [1], pp. 90-93.

        See Also
        ========

        schreier_sims, schreier_sims_random

        """
        if base is None:
            base = []
        if gens is None:
            gens = self.generators[:]
        degree = self.degree
        id_af = list(range(degree))
        # handle the trivial group
        if len(gens) == 1 and gens[0].is_Identity:
            if slp_dict:
                return base, gens, {gens[0]: [gens[0]]}
            return base, gens
        # prevent side effects
        _base, _gens = base[:], gens[:]
        # remove the identity as a generator
        _gens = [x for x in _gens if not x.is_Identity]
        # make sure no generator fixes all base points
        for gen in _gens:
            if all(x == gen._array_form[x] for x in _base):
                for new in id_af:
                    if gen._array_form[new] != new:
                        break
                else:
                    assert None  # can this ever happen?
                _base.append(new)
        # distribute generators according to basic stabilizers
        strong_gens_distr = _distribute_gens_by_base(_base, _gens)
        strong_gens_slp = []
        # initialize the basic stabilizers, basic orbits and basic transversals
        orbs = {}
        transversals = {}
        slps = {}
        base_len = len(_base)
        for i in range(base_len):
            transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i],
                _base[i], pairs=True, af=True, slp=True)
            transversals[i] = dict(transversals[i])
            orbs[i] = list(transversals[i].keys())
        # main loop: amend the stabilizer chain until we have generators
        # for all stabilizers
        i = base_len - 1
        while i >= 0:
            # this flag is used to continue with the main loop from inside
            # a nested loop
            continue_i = False
            # test the generators for being a strong generating set
            db = {}
            for beta, u_beta in list(transversals[i].items()):
                for j, gen in enumerate(strong_gens_distr[i]):
                    gb = gen._array_form[beta]
                    u1 = transversals[i][gb]
                    g1 = _af_rmul(gen._array_form, u_beta)
                    slp = [(i, g) for g in slps[i][beta]]
                    slp = [(i, j)] + slp
                    if g1 != u1:
                        # test if the schreier generator is in the i+1-th
                        # would-be basic stabilizer
                        y = True
                        try:
                            u1_inv = db[gb]
                        except KeyError:
                            u1_inv = db[gb] = _af_invert(u1)
                        schreier_gen = _af_rmul(u1_inv, g1)
                        u1_inv_slp = slps[i][gb][:]
                        u1_inv_slp.reverse()
                        u1_inv_slp = [(i, (g,)) for g in u1_inv_slp]
                        slp = u1_inv_slp + slp
                        h, j, slp = _strip_af(schreier_gen, _base, orbs, transversals, i, slp=slp, slps=slps)
                        if j <= base_len:
                            # new strong generator h at level j
                            y = False
                        elif h:
                            # h fixes all base points
                            y = False
                            moved = 0
                            while h[moved] == moved:
                                moved += 1
                            _base.append(moved)
                            base_len += 1
                            strong_gens_distr.append([])
                        if y is False:
                            # if a new strong generator is found, update the
                            # data structures and start over
                            h = _af_new(h)
                            strong_gens_slp.append((h, slp))
                            for l in range(i + 1, j):
                                strong_gens_distr[l].append(h)
                                transversals[l], slps[l] =\
                                _orbit_transversal(degree, strong_gens_distr[l],
                                    _base[l], pairs=True, af=True, slp=True)
                                transversals[l] = dict(transversals[l])
                                orbs[l] = list(transversals[l].keys())
                            i = j - 1
                            # continue main loop using the flag
                            continue_i = True
                    if continue_i is True:
                        break
                if continue_i is True:
                    break
            if continue_i is True:
                continue
            i -= 1

        strong_gens = _gens[:]

        if slp_dict:
            # create the list of the strong generators strong_gens and
            # rewrite the indices of strong_gens_slp in terms of the
            # elements of strong_gens
            for k, slp in strong_gens_slp:
                strong_gens.append(k)
                for i in range(len(slp)):
                    s = slp[i]
                    if isinstance(s[1], tuple):
                        slp[i] = strong_gens_distr[s[0]][s[1][0]]**-1
                    else:
                        slp[i] = strong_gens_distr[s[0]][s[1]]
            strong_gens_slp = dict(strong_gens_slp)
            # add the original generators
            for g in _gens:
                strong_gens_slp[g] = [g]
            return (_base, strong_gens, strong_gens_slp)

        strong_gens.extend([k for k, _ in strong_gens_slp])
        return _base, strong_gens

    def schreier_sims_random(self, base=None, gens=None, consec_succ=10,
                             _random_prec=None):
        r"""Randomized Schreier-Sims algorithm.

        Explanation
        ===========

        The randomized Schreier-Sims algorithm takes the sequence ``base``
        and the generating set ``gens``, and extends ``base`` to a base, and
        ``gens`` to a strong generating set relative to that base with
        probability of a wrong answer at most `2^{-consec\_succ}`,
        provided the random generators are sufficiently random.

        Parameters
        ==========

        base
            The sequence to be extended to a base.
        gens
            The generating set to be extended to a strong generating set.
        consec_succ
            The parameter defining the probability of a wrong answer.
        _random_prec
            An internal parameter used for testing purposes.

        Returns
        =======

        (base, strong_gens)
            ``base`` is the base and ``strong_gens`` is the strong generating
            set relative to it.

        Examples
        ========

        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(5)
        >>> base, strong_gens = S.schreier_sims_random(consec_succ=5)
        >>> _verify_bsgs(S, base, strong_gens) #doctest: +SKIP
        True

        Notes
        =====

        The algorithm is described in detail in [1], pp. 97-98. It extends
        the orbits ``orbs`` and the permutation groups ``stabs`` to
        basic orbits and basic stabilizers for the base and strong generating
        set produced in the end.
        The idea of the extension process
        is to "sift" random group elements through the stabilizer chain
        and amend the stabilizers/orbits along the way when a sift
        is not successful.
        The helper function ``_strip`` is used to attempt
        to decompose a random group element according to the current
        state of the stabilizer chain and report whether the element was
        fully decomposed (successful sift) or not (unsuccessful sift). In
        the latter case, the level at which the sift failed is reported and
        used to amend ``stabs``, ``base``, ``gens`` and ``orbs`` accordingly.
        The halting condition is for ``consec_succ`` consecutive successful
        sifts to pass. This makes sure that the current ``base`` and ``gens``
        form a BSGS with probability at least `1 - 1/\text{consec\_succ}`.

        See Also
        ========

        schreier_sims

        """
        if base is None:
            base = []
        if gens is None:
            gens = self.generators
        base_len = len(base)
        n = self.degree
        # make sure no generator fixes all base points
        for gen in gens:
            if all(gen(x) == x for x in base):
                new = 0
                while gen._array_form[new] == new:
                    new += 1
                base.append(new)
                base_len += 1
        # distribute generators according to basic stabilizers
        strong_gens_distr = _distribute_gens_by_base(base, gens)
        # initialize the basic stabilizers, basic transversals and basic orbits
        transversals = {}
        orbs = {}
        for i in range(base_len):
            transversals[i] = dict(_orbit_transversal(n, strong_gens_distr[i],
                base[i], pairs=True))
            orbs[i] = list(transversals[i].keys())
        # initialize the number of consecutive elements sifted
        c = 0
        # start sifting random elements while the number of consecutive sifts
        # is less than consec_succ
        while c < consec_succ:
            if _random_prec is None:
                g = self.random_pr()
            else:
                g = _random_prec['g'].pop()
            h, j = _strip(g, base, orbs, transversals)
            y = True
            # determine whether a new base point is needed
            if j <= base_len:
                y = False
            elif not h.is_Identity:
                y = False
                moved = 0
                while h(moved) == moved:
                    moved += 1
                base.append(moved)
                base_len += 1
                strong_gens_distr.append([])
            # if the element doesn't sift, amend the strong generators and
            # associated stabilizers and orbits
            if y is False:
                for l in range(1, j):
                    strong_gens_distr[l].append(h)
                    transversals[l] = dict(_orbit_transversal(n,
                        strong_gens_distr[l], base[l], pairs=True))
                    orbs[l] = list(transversals[l].keys())
                c = 0
            else:
                c += 1
        # build the strong generating set
        strong_gens = strong_gens_distr[0][:]
        for gen in strong_gens_distr[1]:
            if gen not in strong_gens:
                strong_gens.append(gen)
        return base, strong_gens

    def schreier_vector(self, alpha):
        """Computes the schreier vector for ``alpha``.

        Explanation
        ===========

        The Schreier vector efficiently stores information
        about the orbit of ``alpha``. It can later be used to quickly obtain
        elements of the group that send ``alpha`` to a particular element
        in the orbit. Notice that the Schreier vector depends on the order
        in which the group generators are listed. For a definition, see [3].
        Since list indices start from zero, we adopt the convention to use
        "None" instead of 0 to signify that an element does not belong
        to the orbit.
        For the algorithm and its correctness, see [2], pp.78-80.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([2, 4, 6, 3, 1, 5, 0])
        >>> b = Permutation([0, 1, 3, 5, 4, 6, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_vector(0)
        [-1, None, 0, 1, None, 1, 0]

        See Also
        ========

        orbit

        """
        n = self.degree
        v = [None]*n
        v[alpha] = -1
        orb = [alpha]
        used = [False]*n
        used[alpha] = True
        gens = self.generators
        r = len(gens)
        for b in orb:
            for i in range(r):
                temp = gens[i]._array_form[b]
                if used[temp] is False:
                    orb.append(temp)
                    used[temp] = True
                    v[temp] = i
        return v

    def stabilizer(self, alpha):
        r"""Return the stabilizer subgroup of ``alpha``.

        Explanation
        ===========

        The stabilizer of `\alpha` is the group `G_\alpha =
        \{g \in G | g(\alpha) = \alpha\}`.
        For a proof of correctness, see [1], p.79.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> G = DihedralGroup(6)
        >>> G.stabilizer(5)
        PermutationGroup([
            (5)(0 4)(1 3)])

        See Also
        ========

        orbit

        """
        return PermGroup(_stabilizer(self._degree, self._generators, alpha))

    @property
    def strong_gens(self):
        r"""Return a strong generating set from the Schreier-Sims algorithm.

        Explanation
        ===========

        A generating set `S = \{g_1, g_2, \dots, g_t\}` for a permutation group
        `G` is a strong generating set relative to the sequence of points
        (referred to as a "base") `(b_1, b_2, \dots, b_k)` if, for
        `1 \leq i \leq k` we have that the intersection of the pointwise
        stabilizer `G^{(i+1)} := G_{b_1, b_2, \dots, b_i}` with `S` generates
        the pointwise stabilizer `G^{(i+1)}`. The concepts of a base and
        strong generating set and their applications are discussed in depth
        in [1], pp. 87-89 and [2], pp. 55-57.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> D = DihedralGroup(4)
        >>> D.strong_gens
        [(0 1 2 3), (0 3)(1 2), (1 3)]
        >>> D.base
        [0, 1]

        See Also
        ========

        base, basic_transversals, basic_orbits, basic_stabilizers

        """
        if self._strong_gens == []:
            self.schreier_sims()
        return self._strong_gens

    def subgroup(self, gens):
        """
           Return the subgroup generated by `gens` which is a list of
           elements of the group
        """

        if not all(g in self for g in gens):
            raise ValueError("The group does not contain the supplied generators")

        G = PermutationGroup(gens)
        return G

    def subgroup_search(self, prop, base=None, strong_gens=None, tests=None,
                        init_subgroup=None):
        """Find the subgroup of all elements satisfying the property ``prop``.

        Explanation
        ===========

        This is done by a depth-first search with respect to base images that
        uses several tests to prune the search tree.

        Parameters
        ==========

        prop
            The property to be used. Has to be callable on group elements
            and always return ``True`` or ``False``. It is assumed that
            all group elements satisfying ``prop`` indeed form a subgroup.
        base
            A base for the supergroup.
        strong_gens
            A strong generating set for the supergroup.
        tests
            A list of callables of length equal to the length of ``base``.
            These are used to rule out group elements by partial base images,
            so that ``tests[l](g)`` returns False if the element ``g`` is known
            not to satisfy prop base on where g sends the first ``l + 1`` base
            points.
        init_subgroup
            if a subgroup of the sought group is
            known in advance, it can be passed to the function as this
            parameter.

        Returns
        =======

        res
            The subgroup of all elements satisfying ``prop``. The generating
            set for this group is guaranteed to be a strong generating set
            relative to the base ``base``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup)
        >>> from sympy.combinatorics.testutil import _verify_bsgs
        >>> S = SymmetricGroup(7)
        >>> prop_even = lambda x: x.is_even
        >>> base, strong_gens = S.schreier_sims_incremental()
        >>> G = S.subgroup_search(prop_even, base=base, strong_gens=strong_gens)
        >>> G.is_subgroup(AlternatingGroup(7))
        True
        >>> _verify_bsgs(G, base, G.generators)
        True

        Notes
        =====

        This function is extremely lengthy and complicated and will require
        some careful attention. The implementation is described in
        [1], pp. 114-117, and the comments for the code here follow the lines
        of the pseudocode in the book for clarity.

        The complexity is exponential in general, since the search process by
        itself visits all members of the supergroup. However, there are a lot
        of tests which are used to prune the search tree, and users can define
        their own tests via the ``tests`` parameter, so in practice, and for
        some computations, it's not terrible.

        A crucial part in the procedure is the frequent base change performed
        (this is line 11 in the pseudocode) in order to obtain a new basic
        stabilizer. The book mentiones that this can be done by using
        ``.baseswap(...)``, however the current implementation uses a more
        straightforward way to find the next basic stabilizer - calling the
        function ``.stabilizer(...)`` on the previous basic stabilizer.

        """
        # initialize BSGS and basic group properties
        def get_reps(orbits):
            # get the minimal element in the base ordering
            return [min(orbit, key = lambda x: base_ordering[x]) \
              for orbit in orbits]

        def update_nu(l):
            temp_index = len(basic_orbits[l]) + 1 -\
                         len(res_basic_orbits_init_base[l])
            # this corresponds to the element larger than all points
            if temp_index >= len(sorted_orbits[l]):
                nu[l] = base_ordering[degree]
            else:
                nu[l] = sorted_orbits[l][temp_index]

        if base is None:
            base, strong_gens = self.schreier_sims_incremental()
        base_len = len(base)
        degree = self.degree
        identity = _af_new(list(range(degree)))
        base_ordering = _base_ordering(base, degree)
        # add an element larger than all points
        base_ordering.append(degree)
        # add an element smaller than all points
        base_ordering.append(-1)
        # compute BSGS-related structures
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
        basic_orbits, transversals = _orbits_transversals_from_bsgs(base,
                                     strong_gens_distr)
        # handle subgroup initialization and tests
        if init_subgroup is None:
            init_subgroup = PermutationGroup([identity])
        if tests is None:
            trivial_test = lambda x: True
            tests = []
            for i in range(base_len):
                tests.append(trivial_test)
        # line 1: more initializations.
        res = init_subgroup
        f = base_len - 1
        l = base_len - 1
        # line 2: set the base for K to the base for G
        res_base = base[:]
        # line 3: compute BSGS and related structures for K
        res_base, res_strong_gens = res.schreier_sims_incremental(
            base=res_base)
        res_strong_gens_distr = _distribute_gens_by_base(res_base,
                                res_strong_gens)
        res_generators = res.generators
        res_basic_orbits_init_base = \
        [_orbit(degree, res_strong_gens_distr[i], res_base[i])\
         for i in range(base_len)]
        # initialize orbit representatives
        orbit_reps = [None]*base_len
        # line 4: orbit representatives for f-th basic stabilizer of K
        orbits = _orbits(degree, res_strong_gens_distr[f])
        orbit_reps[f] = get_reps(orbits)
        # line 5: remove the base point from the representatives to avoid
        # getting the identity element as a generator for K
        orbit_reps[f].remove(base[f])
        # line 6: more initializations
        c = [0]*base_len
        u = [identity]*base_len
        sorted_orbits = [None]*base_len
        for i in range(base_len):
            sorted_orbits[i] = basic_orbits[i][:]
            sorted_orbits[i].sort(key=lambda point: base_ordering[point])
        # line 7: initializations
        mu = [None]*base_len
        nu = [None]*base_len
        # this corresponds to the element smaller than all points
        mu[l] = degree + 1
        update_nu(l)
        # initialize computed words
        computed_words = [identity]*base_len
        # line 8: main loop
        while True:
            # apply all the tests
            while l < base_len - 1 and \
                computed_words[l](base[l]) in orbit_reps[l] and \
                base_ordering[mu[l]] < \
                base_ordering[computed_words[l](base[l])] < \
                base_ordering[nu[l]] and \
                    tests[l](computed_words):
                # line 11: change the (partial) base of K
                new_point = computed_words[l](base[l])
                res_base[l] = new_point
                new_stab_gens = _stabilizer(degree, res_strong_gens_distr[l],
                        new_point)
                res_strong_gens_distr[l + 1] = new_stab_gens
                # line 12: calculate minimal orbit representatives for the
                # l+1-th basic stabilizer
                orbits = _orbits(degree, new_stab_gens)
                orbit_reps[l + 1] = get_reps(orbits)
                # line 13: amend sorted orbits
                l += 1
                temp_orbit = [computed_words[l - 1](point) for point
                             in basic_orbits[l]]
                temp_orbit.sort(key=lambda point: base_ordering[point])
                sorted_orbits[l] = temp_orbit
                # lines 14 and 15: update variables used minimality tests
                new_mu = degree + 1
                for i in range(l):
                    if base[l] in res_basic_orbits_init_base[i]:
                        candidate = computed_words[i](base[i])
                        if base_ordering[candidate] > base_ordering[new_mu]:
                            new_mu = candidate
                mu[l] = new_mu
                update_nu(l)
                # line 16: determine the new transversal element
                c[l] = 0
                temp_point = sorted_orbits[l][c[l]]
                gamma = computed_words[l - 1]._array_form.index(temp_point)
                u[l] = transversals[l][gamma]
                # update computed words
                computed_words[l] = rmul(computed_words[l - 1], u[l])
            # lines 17 & 18: apply the tests to the group element found
            g = computed_words[l]
            temp_point = g(base[l])
            if l == base_len - 1 and \
                base_ordering[mu[l]] < \
                base_ordering[temp_point] < base_ordering[nu[l]] and \
                temp_point in orbit_reps[l] and \
                tests[l](computed_words) and \
                    prop(g):
                # line 19: reset the base of K
                res_generators.append(g)
                res_base = base[:]
                # line 20: recalculate basic orbits (and transversals)
                res_strong_gens.append(g)
                res_strong_gens_distr = _distribute_gens_by_base(res_base,
                                                          res_strong_gens)
                res_basic_orbits_init_base = \
                [_orbit(degree, res_strong_gens_distr[i], res_base[i]) \
                 for i in range(base_len)]
                # line 21: recalculate orbit representatives
                # line 22: reset the search depth
                orbit_reps[f] = get_reps(orbits)
                l = f
            # line 23: go up the tree until in the first branch not fully
            # searched
            while l >= 0 and c[l] == len(basic_orbits[l]) - 1:
                l = l - 1
            # line 24: if the entire tree is traversed, return K
            if l == -1:
                return PermutationGroup(res_generators)
            # lines 25-27: update orbit representatives
            if l < f:
                # line 26
                f = l
                c[l] = 0
                # line 27
                temp_orbits = _orbits(degree, res_strong_gens_distr[f])
                orbit_reps[f] = get_reps(temp_orbits)
                # line 28: update variables used for minimality testing
                mu[l] = degree + 1
                temp_index = len(basic_orbits[l]) + 1 - \
                    len(res_basic_orbits_init_base[l])
                if temp_index >= len(sorted_orbits[l]):
                    nu[l] = base_ordering[degree]
                else:
                    nu[l] = sorted_orbits[l][temp_index]
            # line 29: set the next element from the current branch and update
            # accordingly
            c[l] += 1
            if l == 0:
                gamma  = sorted_orbits[l][c[l]]
            else:
                gamma = computed_words[l - 1]._array_form.index(sorted_orbits[l][c[l]])

            u[l] = transversals[l][gamma]
            if l == 0:
                computed_words[l] = u[l]
            else:
                computed_words[l] = rmul(computed_words[l - 1], u[l])

    @property
    def transitivity_degree(self):
        r"""Compute the degree of transitivity of the group.

        Explanation
        ===========

        A permutation group `G` acting on `\Omega = \{0, 1, \dots, n-1\}` is
        ``k``-fold transitive, if, for any `k` points
        `(a_1, a_2, \dots, a_k) \in \Omega` and any `k` points
        `(b_1, b_2, \dots, b_k) \in \Omega` there exists `g \in  G` such that
        `g(a_1) = b_1, g(a_2) = b_2, \dots, g(a_k) = b_k`
        The degree of transitivity of `G` is the maximum ``k`` such that
        `G` is ``k``-fold transitive. ([8])

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([1, 2, 0])
        >>> b = Permutation([1, 0, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.transitivity_degree
        3

        See Also
        ========

        is_transitive, orbit

        """
        if self._transitivity_degree is None:
            n = self.degree
            G = self
            # if G is k-transitive, a tuple (a_0,..,a_k)
            # can be brought to (b_0,...,b_(k-1), b_k)
            # where b_0,...,b_(k-1) are fixed points;
            # consider the group G_k which stabilizes b_0,...,b_(k-1)
            # if G_k is transitive on the subset excluding b_0,...,b_(k-1)
            # then G is (k+1)-transitive
            for i in range(n):
                orb = G.orbit(i)
                if len(orb) != n - i:
                    self._transitivity_degree = i
                    return i
                G = G.stabilizer(i)
            self._transitivity_degree = n
            return n
        else:
            return self._transitivity_degree

    def _p_elements_group(self, p):
        '''
        For an abelian p-group, return the subgroup consisting of
        all elements of order p (and the identity)

        '''
        gens = self.generators[:]
        gens = sorted(gens, key=lambda x: x.order(), reverse=True)
        gens_p = [g**(g.order()/p) for g in gens]
        gens_r = []
        for i in range(len(gens)):
            x = gens[i]
            x_order = x.order()
            # x_p has order p
            x_p = x**(x_order/p)
            if i > 0:
                P = PermutationGroup(gens_p[:i])
            else:
                P = PermutationGroup(self.identity)
            if x**(x_order/p) not in P:
                gens_r.append(x**(x_order/p))
            else:
                # replace x by an element of order (x.order()/p)
                # so that gens still generates G
                g = P.generator_product(x_p, original=True)
                for s in g:
                    x = x*s**-1
                x_order = x_order/p
                # insert x to gens so that the sorting is preserved
                del gens[i]
                del gens_p[i]
                j = i - 1
                while j < len(gens) and gens[j].order() >= x_order:
                    j += 1
                gens = gens[:j] + [x] + gens[j:]
                gens_p = gens_p[:j] + [x] + gens_p[j:]
        return PermutationGroup(gens_r)

    def _sylow_alt_sym(self, p):
        '''
        Return a p-Sylow subgroup of a symmetric or an
        alternating group.

        Explanation
        ===========

        The algorithm for this is hinted at in [1], Chapter 4,
        Exercise 4.

        For Sym(n) with n = p^i, the idea is as follows. Partition
        the interval [0..n-1] into p equal parts, each of length p^(i-1):
        [0..p^(i-1)-1], [p^(i-1)..2*p^(i-1)-1]...[(p-1)*p^(i-1)..p^i-1].
        Find a p-Sylow subgroup of Sym(p^(i-1)) (treated as a subgroup
        of ``self``) acting on each of the parts. Call the subgroups
        P_1, P_2...P_p. The generators for the subgroups P_2...P_p
        can be obtained from those of P_1 by applying a "shifting"
        permutation to them, that is, a permutation mapping [0..p^(i-1)-1]
        to the second part (the other parts are obtained by using the shift
        multiple times). The union of this permutation and the generators
        of P_1 is a p-Sylow subgroup of ``self``.

        For n not equal to a power of p, partition
        [0..n-1] in accordance with how n would be written in base p.
        E.g. for p=2 and n=11, 11 = 2^3 + 2^2 + 1 so the partition
        is [[0..7], [8..9], {10}]. To generate a p-Sylow subgroup,
        take the union of the generators for each of the parts.
        For the above example, {(0 1), (0 2)(1 3), (0 4), (1 5)(2 7)}
        from the first part, {(8 9)} from the second part and
        nothing from the third. This gives 4 generators in total, and
        the subgroup they generate is p-Sylow.

        Alternating groups are treated the same except when p=2. In this
        case, (0 1)(s s+1) should be added for an appropriate s (the start
        of a part) for each part in the partitions.

        See Also
        ========

        sylow_subgroup, is_alt_sym

        '''
        n = self.degree
        gens = []
        identity = Permutation(n-1)
        # the case of 2-sylow subgroups of alternating groups
        # needs special treatment
        alt = p == 2 and all(g.is_even for g in self.generators)

        # find the presentation of n in base p
        coeffs = []
        m = n
        while m > 0:
            coeffs.append(m % p)
            m = m // p

        power = len(coeffs)-1
        # for a symmetric group, gens[:i] is the generating
        # set for a p-Sylow subgroup on [0..p**(i-1)-1]. For
        # alternating groups, the same is given by gens[:2*(i-1)]
        for i in range(1, power+1):
            if i == 1 and alt:
                # (0 1) shouldn't be added for alternating groups
                continue
            gen = Permutation([(j + p**(i-1)) % p**i for j in range(p**i)])
            gens.append(identity*gen)
            if alt:
                gen = Permutation(0, 1)*gen*Permutation(0, 1)*gen
                gens.append(gen)

        # the first point in the current part (see the algorithm
        # description in the docstring)
        start = 0

        while power > 0:
            a = coeffs[power]

            # make the permutation shifting the start of the first
            # part ([0..p^i-1] for some i) to the current one
            for _ in range(a):
                shift = Permutation()
                if start > 0:
                    for i in range(p**power):
                        shift = shift(i, start + i)

                    if alt:
                        gen = Permutation(0, 1)*shift*Permutation(0, 1)*shift
                        gens.append(gen)
                        j = 2*(power - 1)
                    else:
                        j = power

                    for i, gen in enumerate(gens[:j]):
                        if alt and i % 2 == 1:
                            continue
                        # shift the generator to the start of the
                        # partition part
                        gen = shift*gen*shift
                        gens.append(gen)

                start += p**power
            power = power-1

        return gens

    def sylow_subgroup(self, p):
        '''
        Return a p-Sylow subgroup of the group.

        The algorithm is described in [1], Chapter 4, Section 7

        Examples
        ========
        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.named_groups import AlternatingGroup

        >>> D = DihedralGroup(6)
        >>> S = D.sylow_subgroup(2)
        >>> S.order()
        4
        >>> G = SymmetricGroup(6)
        >>> S = G.sylow_subgroup(5)
        >>> S.order()
        5

        >>> G1 = AlternatingGroup(3)
        >>> G2 = AlternatingGroup(5)
        >>> G3 = AlternatingGroup(9)

        >>> S1 = G1.sylow_subgroup(3)
        >>> S2 = G2.sylow_subgroup(3)
        >>> S3 = G3.sylow_subgroup(3)

        >>> len1 = len(S1.lower_central_series())
        >>> len2 = len(S2.lower_central_series())
        >>> len3 = len(S3.lower_central_series())

        >>> len1 == len2
        True
        >>> len1 < len3
        True

        '''
        from sympy.combinatorics.homomorphisms import (
                orbit_homomorphism, block_homomorphism)

        if not isprime(p):
            raise ValueError("p must be a prime")

        def is_p_group(G):
            # check if the order of G is a power of p
            # and return the power
            m = G.order()
            n = 0
            while m % p == 0:
                m = m/p
                n += 1
                if m == 1:
                    return True, n
            return False, n

        def _sylow_reduce(mu, nu):
            # reduction based on two homomorphisms
            # mu and nu with trivially intersecting
            # kernels
            Q = mu.image().sylow_subgroup(p)
            Q = mu.invert_subgroup(Q)
            nu = nu.restrict_to(Q)
            R = nu.image().sylow_subgroup(p)
            return nu.invert_subgroup(R)

        order = self.order()
        if order % p != 0:
            return PermutationGroup([self.identity])
        p_group, n = is_p_group(self)
        if p_group:
            return self

        if self.is_alt_sym():
            return PermutationGroup(self._sylow_alt_sym(p))

        # if there is a non-trivial orbit with size not divisible
        # by p, the sylow subgroup is contained in its stabilizer
        # (by orbit-stabilizer theorem)
        orbits = self.orbits()
        non_p_orbits = [o for o in orbits if len(o) % p != 0 and len(o) != 1]
        if non_p_orbits:
            G = self.stabilizer(list(non_p_orbits[0]).pop())
            return G.sylow_subgroup(p)

        if not self.is_transitive():
            # apply _sylow_reduce to orbit actions
            orbits = sorted(orbits, key=len)
            omega1 = orbits.pop()
            omega2 = orbits[0].union(*orbits)
            mu = orbit_homomorphism(self, omega1)
            nu = orbit_homomorphism(self, omega2)
            return _sylow_reduce(mu, nu)

        blocks = self.minimal_blocks()
        if len(blocks) > 1:
            # apply _sylow_reduce to block system actions
            mu = block_homomorphism(self, blocks[0])
            nu = block_homomorphism(self, blocks[1])
            return _sylow_reduce(mu, nu)
        elif len(blocks) == 1:
            block = list(blocks)[0]
            if any(e != 0 for e in block):
                # self is imprimitive
                mu = block_homomorphism(self, block)
                if not is_p_group(mu.image())[0]:
                    S = mu.image().sylow_subgroup(p)
                    return mu.invert_subgroup(S).sylow_subgroup(p)

        # find an element of order p
        g = self.random()
        g_order = g.order()
        while g_order % p != 0 or g_order == 0:
            g = self.random()
            g_order = g.order()
        g = g**(g_order // p)
        if order % p**2 != 0:
            return PermutationGroup(g)

        C = self.centralizer(g)
        while C.order() % p**n != 0:
            S = C.sylow_subgroup(p)
            s_order = S.order()
            Z = S.center()
            P = Z._p_elements_group(p)
            h = P.random()
            C_h = self.centralizer(h)
            while C_h.order() % p*s_order != 0:
                h = P.random()
                C_h = self.centralizer(h)
            C = C_h

        return C.sylow_subgroup(p)

    def _block_verify(self, L, alpha):
        delta = sorted(self.orbit(alpha))
        # p[i] will be the number of the block
        # delta[i] belongs to
        p = [-1]*len(delta)
        blocks = [-1]*len(delta)

        B = [[]] # future list of blocks
        u = [0]*len(delta) # u[i] in L s.t. alpha^u[i] = B[0][i]

        t = L.orbit_transversal(alpha, pairs=True)
        for a, beta in t:
            B[0].append(a)
            i_a = delta.index(a)
            p[i_a] = 0
            blocks[i_a] = alpha
            u[i_a] = beta

        rho = 0
        m = 0 # number of blocks - 1

        while rho <= m:
            beta = B[rho][0]
            for g in self.generators:
                d = beta^g
                i_d = delta.index(d)
                sigma = p[i_d]
                if sigma < 0:
                    # define a new block
                    m += 1
                    sigma = m
                    u[i_d] = u[delta.index(beta)]*g
                    p[i_d] = sigma
                    rep = d
                    blocks[i_d] = rep
                    newb = [rep]
                    for gamma in B[rho][1:]:
                        i_gamma = delta.index(gamma)
                        d = gamma^g
                        i_d = delta.index(d)
                        if p[i_d] < 0:
                            u[i_d] = u[i_gamma]*g
                            p[i_d] = sigma
                            blocks[i_d] = rep
                            newb.append(d)
                        else:
                            # B[rho] is not a block
                            s = u[i_gamma]*g*u[i_d]**(-1)
                            return False, s

                    B.append(newb)
                else:
                    for h in B[rho][1:]:
                        if h^g not in B[sigma]:
                            # B[rho] is not a block
                            s = u[delta.index(beta)]*g*u[i_d]**(-1)
                            return False, s
            rho += 1

        return True, blocks

    def _verify(H, K, phi, z, alpha):
        '''
        Return a list of relators ``rels`` in generators ``gens`_h` that
        are mapped to ``H.generators`` by ``phi`` so that given a finite
        presentation <gens_k | rels_k> of ``K`` on a subset of ``gens_h``
        <gens_h | rels_k + rels> is a finite presentation of ``H``.

        Explanation
        ===========

        ``H`` should be generated by the union of ``K.generators`` and ``z``
        (a single generator), and ``H.stabilizer(alpha) == K``; ``phi`` is a
        canonical injection from a free group into a permutation group
        containing ``H``.

        The algorithm is described in [1], Chapter 6.

        Examples
        ========

        >>> from sympy.combinatorics import free_group, Permutation, PermutationGroup
        >>> from sympy.combinatorics.homomorphisms import homomorphism
        >>> from sympy.combinatorics.fp_groups import FpGroup

        >>> H = PermutationGroup(Permutation(0, 2), Permutation (1, 5))
        >>> K = PermutationGroup(Permutation(5)(0, 2))
        >>> F = free_group("x_0 x_1")[0]
        >>> gens = F.generators
        >>> phi = homomorphism(F, H, F.generators, H.generators)
        >>> rels_k = [gens[0]**2] # relators for presentation of K
        >>> z= Permutation(1, 5)
        >>> check, rels_h = H._verify(K, phi, z, 1)
        >>> check
        True
        >>> rels = rels_k + rels_h
        >>> G = FpGroup(F, rels) # presentation of H
        >>> G.order() == H.order()
        True

        See also
        ========

        strong_presentation, presentation, stabilizer

        '''

        orbit = H.orbit(alpha)
        beta = alpha^(z**-1)

        K_beta = K.stabilizer(beta)

        # orbit representatives of K_beta
        gammas = [alpha, beta]
        orbits = list({tuple(K_beta.orbit(o)) for o in orbit})
        orbit_reps = [orb[0] for orb in orbits]
        for rep in orbit_reps:
            if rep not in gammas:
                gammas.append(rep)

        # orbit transversal of K
        betas = [alpha, beta]
        transversal = {alpha: phi.invert(H.identity), beta: phi.invert(z**-1)}

        for s, g in K.orbit_transversal(beta, pairs=True):
            if s not in transversal:
                transversal[s] = transversal[beta]*phi.invert(g)


        union = K.orbit(alpha).union(K.orbit(beta))
        while (len(union) < len(orbit)):
            for gamma in gammas:
                if gamma in union:
                    r = gamma^z
                    if r not in union:
                        betas.append(r)
                        transversal[r] = transversal[gamma]*phi.invert(z)
                        for s, g in K.orbit_transversal(r, pairs=True):
                            if s not in transversal:
                                transversal[s] = transversal[r]*phi.invert(g)
                        union = union.union(K.orbit(r))
                        break

        # compute relators
        rels = []

        for b in betas:
            k_gens = K.stabilizer(b).generators
            for y in k_gens:
                new_rel = transversal[b]
                gens = K.generator_product(y, original=True)
                for g in gens[::-1]:
                    new_rel = new_rel*phi.invert(g)
                new_rel = new_rel*transversal[b]**-1

                perm = phi(new_rel)
                try:
                    gens = K.generator_product(perm, original=True)
                except ValueError:
                    return False, perm
                for g in gens:
                    new_rel = new_rel*phi.invert(g)**-1
                if new_rel not in rels:
                    rels.append(new_rel)

        for gamma in gammas:
            new_rel = transversal[gamma]*phi.invert(z)*transversal[gamma^z]**-1
            perm = phi(new_rel)
            try:
                gens = K.generator_product(perm, original=True)
            except ValueError:
                return False, perm
            for g in gens:
                new_rel = new_rel*phi.invert(g)**-1
            if new_rel not in rels:
                rels.append(new_rel)

        return True, rels

    def strong_presentation(self):
        '''
        Return a strong finite presentation of group. The generators
        of the returned group are in the same order as the strong
        generators of group.

        The algorithm is based on Sims' Verify algorithm described
        in [1], Chapter 6.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import DihedralGroup
        >>> P = DihedralGroup(4)
        >>> G = P.strong_presentation()
        >>> P.order() == G.order()
        True

        See Also
        ========

        presentation, _verify

        '''
        from sympy.combinatorics.fp_groups import (FpGroup,
                                            simplify_presentation)
        from sympy.combinatorics.free_groups import free_group
        from sympy.combinatorics.homomorphisms import (block_homomorphism,
                                           homomorphism, GroupHomomorphism)

        strong_gens = self.strong_gens[:]
        stabs = self.basic_stabilizers[:]
        base = self.base[:]

        # injection from a free group on len(strong_gens)
        # generators into G
        gen_syms = [('x_%d'%i) for i in range(len(strong_gens))]
        F = free_group(', '.join(gen_syms))[0]
        phi = homomorphism(F, self, F.generators, strong_gens)

        H = PermutationGroup(self.identity)
        while stabs:
            alpha = base.pop()
            K = H
            H = stabs.pop()
            new_gens = [g for g in H.generators if g not in K]

            if K.order() == 1:
                z = new_gens.pop()
                rels = [F.generators[-1]**z.order()]
                intermediate_gens = [z]
                K = PermutationGroup(intermediate_gens)

            # add generators one at a time building up from K to H
            while new_gens:
                z = new_gens.pop()
                intermediate_gens = [z] + intermediate_gens
                K_s = PermutationGroup(intermediate_gens)
                orbit = K_s.orbit(alpha)
                orbit_k = K.orbit(alpha)

                # split into cases based on the orbit of K_s
                if orbit_k == orbit:
                    if z in K:
                        rel = phi.invert(z)
                        perm = z
                    else:
                        t = K.orbit_rep(alpha, alpha^z)
                        rel = phi.invert(z)*phi.invert(t)**-1
                        perm = z*t**-1
                    for g in K.generator_product(perm, original=True):
                        rel = rel*phi.invert(g)**-1
                    new_rels = [rel]
                elif len(orbit_k) == 1:
                    # `success` is always true because `strong_gens`
                    # and `base` are already a verified BSGS. Later
                    # this could be changed to start with a randomly
                    # generated (potential) BSGS, and then new elements
                    # would have to be appended to it when `success`
                    # is false.
                    success, new_rels = K_s._verify(K, phi, z, alpha)
                else:
                    # K.orbit(alpha) should be a block
                    # under the action of K_s on K_s.orbit(alpha)
                    check, block = K_s._block_verify(K, alpha)
                    if check:
                        # apply _verify to the action of K_s
                        # on the block system; for convenience,
                        # add the blocks as additional points
                        # that K_s should act on
                        t = block_homomorphism(K_s, block)
                        m = t.codomain.degree # number of blocks
                        d = K_s.degree

                        # conjugating with p will shift
                        # permutations in t.image() to
                        # higher numbers, e.g.
                        # p*(0 1)*p = (m m+1)
                        p = Permutation()
                        for i in range(m):
                            p *= Permutation(i, i+d)

                        t_img = t.images
                        # combine generators of K_s with their
                        # action on the block system
                        images = {g: g*p*t_img[g]*p for g in t_img}
                        for g in self.strong_gens[:-len(K_s.generators)]:
                            images[g] = g
                        K_s_act = PermutationGroup(list(images.values()))
                        f = GroupHomomorphism(self, K_s_act, images)

                        K_act = PermutationGroup([f(g) for g in K.generators])
                        success, new_rels = K_s_act._verify(K_act, f.compose(phi), f(z), d)

                for n in new_rels:
                    if n not in rels:
                        rels.append(n)
                K = K_s

        group = FpGroup(F, rels)
        return simplify_presentation(group)

    def presentation(self, eliminate_gens=True):
        '''
        Return an `FpGroup` presentation of the group.

        The algorithm is described in [1], Chapter 6.1.

        '''
        from sympy.combinatorics.fp_groups import (FpGroup,
                                            simplify_presentation)
        from sympy.combinatorics.coset_table import CosetTable
        from sympy.combinatorics.free_groups import free_group
        from sympy.combinatorics.homomorphisms import homomorphism

        if self._fp_presentation:
            return self._fp_presentation

        def _factor_group_by_rels(G, rels):
            if isinstance(G, FpGroup):
                rels.extend(G.relators)
                return FpGroup(G.free_group, list(set(rels)))
            return FpGroup(G, rels)

        gens = self.generators
        len_g = len(gens)

        if len_g == 1:
            order = gens[0].order()
            # handle the trivial group
            if order == 1:
                return free_group([])[0]
            F, x = free_group('x')
            return FpGroup(F, [x**order])

        if self.order() > 20:
            half_gens = self.generators[0:(len_g+1)//2]
        else:
            half_gens = []
        H = PermutationGroup(half_gens)
        H_p = H.presentation()

        len_h = len(H_p.generators)

        C = self.coset_table(H)
        n = len(C) # subgroup index

        gen_syms = [('x_%d'%i) for i in range(len(gens))]
        F = free_group(', '.join(gen_syms))[0]

        # mapping generators of H_p to those of F
        images = [F.generators[i] for i in range(len_h)]
        R = homomorphism(H_p, F, H_p.generators, images, check=False)

        # rewrite relators
        rels = R(H_p.relators)
        G_p = FpGroup(F, rels)

        # injective homomorphism from G_p into self
        T = homomorphism(G_p, self, G_p.generators, gens)

        C_p = CosetTable(G_p, [])

        C_p.table = [[None]*(2*len_g) for i in range(n)]

        # initiate the coset transversal
        transversal = [None]*n
        transversal[0] = G_p.identity

        # fill in the coset table as much as possible
        for i in range(2*len_h):
            C_p.table[0][i] = 0

        gamma = 1
        for alpha, x in product(range(n), range(2*len_g)):
            beta = C[alpha][x]
            if beta == gamma:
                gen = G_p.generators[x//2]**((-1)**(x % 2))
                transversal[beta] = transversal[alpha]*gen
                C_p.table[alpha][x] = beta
                C_p.table[beta][x + (-1)**(x % 2)] = alpha
                gamma += 1
                if gamma == n:
                    break

        C_p.p = list(range(n))
        beta = x = 0

        while not C_p.is_complete():
            # find the first undefined entry
            while C_p.table[beta][x] == C[beta][x]:
                x = (x + 1) % (2*len_g)
                if x == 0:
                    beta = (beta + 1) % n

            # define a new relator
            gen = G_p.generators[x//2]**((-1)**(x % 2))
            new_rel = transversal[beta]*gen*transversal[C[beta][x]]**-1
            perm = T(new_rel)
            nxt = G_p.identity
            for s in H.generator_product(perm, original=True):
                nxt = nxt*T.invert(s)**-1
            new_rel = new_rel*nxt

            # continue coset enumeration
            G_p = _factor_group_by_rels(G_p, [new_rel])
            C_p.scan_and_fill(0, new_rel)
            C_p = G_p.coset_enumeration([], strategy="coset_table",
                                draft=C_p, max_cosets=n, incomplete=True)

        self._fp_presentation = simplify_presentation(G_p)
        return self._fp_presentation

    def polycyclic_group(self):
        """
        Return the PolycyclicGroup instance with below parameters:

        Explanation
        ===========

        * pc_sequence : Polycyclic sequence is formed by collecting all
          the missing generators between the adjacent groups in the
          derived series of given permutation group.

        * pc_series : Polycyclic series is formed by adding all the missing
          generators of ``der[i+1]`` in ``der[i]``, where ``der`` represents
          the derived series.

        * relative_order : A list, computed by the ratio of adjacent groups in
          pc_series.

        """
        from sympy.combinatorics.pc_groups import PolycyclicGroup
        if not self.is_polycyclic:
            raise ValueError("The group must be solvable")

        der = self.derived_series()
        pc_series = []
        pc_sequence = []
        relative_order = []
        pc_series.append(der[-1])
        der.reverse()

        for i in range(len(der)-1):
            H = der[i]
            for g in der[i+1].generators:
                if g not in H:
                    H = PermutationGroup([g] + H.generators)
                    pc_series.insert(0, H)
                    pc_sequence.insert(0, g)

                    G1 = pc_series[0].order()
                    G2 = pc_series[1].order()
                    relative_order.insert(0, G1 // G2)

        return PolycyclicGroup(pc_sequence, pc_series, relative_order, collector=None)


def _orbit(degree, generators, alpha, action='tuples'):
    r"""Compute the orbit of alpha `\{g(\alpha) | g \in G\}` as a set.

    Explanation
    ===========

    The time complexity of the algorithm used here is `O(|Orb|*r)` where
    `|Orb|` is the size of the orbit and ``r`` is the number of generators of
    the group. For a more detailed analysis, see [1], p.78, [2], pp. 19-21.
    Here alpha can be a single point, or a list of points.

    If alpha is a single point, the ordinary orbit is computed.
    if alpha is a list of points, there are three available options:

    'union' - computes the union of the orbits of the points in the list
    'tuples' - computes the orbit of the list interpreted as an ordered
    tuple under the group action ( i.e., g((1, 2, 3)) = (g(1), g(2), g(3)) )
    'sets' - computes the orbit of the list interpreted as a sets

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, PermutationGroup
    >>> from sympy.combinatorics.perm_groups import _orbit
    >>> a = Permutation([1, 2, 0, 4, 5, 6, 3])
    >>> G = PermutationGroup([a])
    >>> _orbit(G.degree, G.generators, 0)
    {0, 1, 2}
    >>> _orbit(G.degree, G.generators, [0, 4], 'union')
    {0, 1, 2, 3, 4, 5, 6}

    See Also
    ========

    orbit, orbit_transversal

    """
    if not hasattr(alpha, '__getitem__'):
        alpha = [alpha]

    gens = [x._array_form for x in generators]
    if len(alpha) == 1 or action == 'union':
        orb = alpha
        used = [False]*degree
        for el in alpha:
            used[el] = True
        for b in orb:
            for gen in gens:
                temp = gen[b]
                if used[temp] == False:
                    orb.append(temp)
                    used[temp] = True
        return set(orb)
    elif action == 'tuples':
        alpha = tuple(alpha)
        orb = [alpha]
        used = {alpha}
        for b in orb:
            for gen in gens:
                temp = tuple([gen[x] for x in b])
                if temp not in used:
                    orb.append(temp)
                    used.add(temp)
        return set(orb)
    elif action == 'sets':
        alpha = frozenset(alpha)
        orb = [alpha]
        used = {alpha}
        for b in orb:
            for gen in gens:
                temp = frozenset([gen[x] for x in b])
                if temp not in used:
                    orb.append(temp)
                    used.add(temp)
        return {tuple(x) for x in orb}


def _orbits(degree, generators):
    """Compute the orbits of G.

    If ``rep=False`` it returns a list of sets else it returns a list of
    representatives of the orbits

    Examples
    ========

    >>> from sympy.combinatorics import Permutation
    >>> from sympy.combinatorics.perm_groups import _orbits
    >>> a = Permutation([0, 2, 1])
    >>> b = Permutation([1, 0, 2])
    >>> _orbits(a.size, [a, b])
    [{0, 1, 2}]
    """

    orbs = []
    sorted_I = list(range(degree))
    I = set(sorted_I)
    while I:
        i = sorted_I[0]
        orb = _orbit(degree, generators, i)
        orbs.append(orb)
        # remove all indices that are in this orbit
        I -= orb
        sorted_I = [i for i in sorted_I if i not in orb]
    return orbs


def _orbit_transversal(degree, generators, alpha, pairs, af=False, slp=False):
    r"""Computes a transversal for the orbit of ``alpha`` as a set.

    Explanation
    ===========

    generators   generators of the group ``G``

    For a permutation group ``G``, a transversal for the orbit
    `Orb = \{g(\alpha) | g \in G\}` is a set
    `\{g_\beta | g_\beta(\alpha) = \beta\}` for `\beta \in Orb`.
    Note that there may be more than one possible transversal.
    If ``pairs`` is set to ``True``, it returns the list of pairs
    `(\beta, g_\beta)`. For a proof of correctness, see [1], p.79

    if ``af`` is ``True``, the transversal elements are given in
    array form.

    If `slp` is `True`, a dictionary `{beta: slp_beta}` is returned
    for `\beta \in Orb` where `slp_beta` is a list of indices of the
    generators in `generators` s.t. if `slp_beta = [i_1 \dots i_n]`
    `g_\beta = generators[i_n] \times \dots \times generators[i_1]`.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.perm_groups import _orbit_transversal
    >>> G = DihedralGroup(6)
    >>> _orbit_transversal(G.degree, G.generators, 0, False)
    [(5), (0 1 2 3 4 5), (0 5)(1 4)(2 3), (0 2 4)(1 3 5), (5)(0 4)(1 3), (0 3)(1 4)(2 5)]
    """

    tr = [(alpha, list(range(degree)))]
    slp_dict = {alpha: []}
    used = [False]*degree
    used[alpha] = True
    gens = [x._array_form for x in generators]
    for x, px in tr:
        px_slp = slp_dict[x]
        for gen in gens:
            temp = gen[x]
            if used[temp] == False:
                slp_dict[temp] = [gens.index(gen)] + px_slp
                tr.append((temp, _af_rmul(gen, px)))
                used[temp] = True
    if pairs:
        if not af:
            tr = [(x, _af_new(y)) for x, y in tr]
        if not slp:
            return tr
        return tr, slp_dict

    if af:
        tr = [y for _, y in tr]
        if not slp:
            return tr
        return tr, slp_dict

    tr = [_af_new(y) for _, y in tr]
    if not slp:
        return tr
    return tr, slp_dict


def _stabilizer(degree, generators, alpha):
    r"""Return the stabilizer subgroup of ``alpha``.

    Explanation
    ===========

    The stabilizer of `\alpha` is the group `G_\alpha =
    \{g \in G | g(\alpha) = \alpha\}`.
    For a proof of correctness, see [1], p.79.

    degree :       degree of G
    generators :   generators of G

    Examples
    ========

    >>> from sympy.combinatorics.perm_groups import _stabilizer
    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> G = DihedralGroup(6)
    >>> _stabilizer(G.degree, G.generators, 5)
    [(5)(0 4)(1 3), (5)]

    See Also
    ========

    orbit

    """
    orb = [alpha]
    table = {alpha: list(range(degree))}
    table_inv = {alpha: list(range(degree))}
    used = [False]*degree
    used[alpha] = True
    gens = [x._array_form for x in generators]
    stab_gens = []
    for b in orb:
        for gen in gens:
            temp = gen[b]
            if used[temp] is False:
                gen_temp = _af_rmul(gen, table[b])
                orb.append(temp)
                table[temp] = gen_temp
                table_inv[temp] = _af_invert(gen_temp)
                used[temp] = True
            else:
                schreier_gen = _af_rmuln(table_inv[temp], gen, table[b])
                if schreier_gen not in stab_gens:
                    stab_gens.append(schreier_gen)
    return [_af_new(x) for x in stab_gens]


PermGroup = PermutationGroup


class SymmetricPermutationGroup(Basic):
    """
    The class defining the lazy form of SymmetricGroup.

    deg : int

    """
    def __new__(cls, deg):
        deg = _sympify(deg)
        obj = Basic.__new__(cls, deg)
        return obj

    def __init__(self, *args, **kwargs):
        self._deg = self.args[0]
        self._order = None

    def __contains__(self, i):
        """Return ``True`` if *i* is contained in SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> Permutation(1, 2, 3) in G
        True

        """
        if not isinstance(i, Permutation):
            raise TypeError("A SymmetricPermutationGroup contains only Permutations as "
                            "elements, not elements of type %s" % type(i))
        return i.size == self.degree

    def order(self):
        """
        Return the order of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.order()
        24
        """
        if self._order is not None:
            return self._order
        n = self._deg
        self._order = factorial(n)
        return self._order

    @property
    def degree(self):
        """
        Return the degree of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.degree
        4

        """
        return self._deg

    @property
    def identity(self):
        '''
        Return the identity element of the SymmetricPermutationGroup.

        Examples
        ========

        >>> from sympy.combinatorics import SymmetricPermutationGroup
        >>> G = SymmetricPermutationGroup(4)
        >>> G.identity()
        (3)

        '''
        return _af_new(list(range(self._deg)))


class Coset(Basic):
    """A left coset of a permutation group with respect to an element.

    Parameters
    ==========

    g : Permutation

    H : PermutationGroup

    dir : "+" or "-", If not specified by default it will be "+"
        here ``dir`` specified the type of coset "+" represent the
        right coset and "-" represent the left coset.

    G : PermutationGroup, optional
        The group which contains *H* as its subgroup and *g* as its
        element.

        If not specified, it would automatically become a symmetric
        group ``SymmetricPermutationGroup(g.size)`` and
        ``SymmetricPermutationGroup(H.degree)`` if ``g.size`` and ``H.degree``
        are matching.``SymmetricPermutationGroup`` is a lazy form of SymmetricGroup
        used for representation purpose.

    """

    def __new__(cls, g, H, G=None, dir="+"):
        g = _sympify(g)
        if not isinstance(g, Permutation):
            raise NotImplementedError

        H = _sympify(H)
        if not isinstance(H, PermutationGroup):
            raise NotImplementedError

        if G is not None:
            G = _sympify(G)
            if not isinstance(G, (PermutationGroup, SymmetricPermutationGroup)):
                raise NotImplementedError
            if not H.is_subgroup(G):
                raise ValueError("{} must be a subgroup of {}.".format(H, G))
            if g not in G:
                raise ValueError("{} must be an element of {}.".format(g, G))
        else:
            g_size = g.size
            h_degree = H.degree
            if g_size != h_degree:
                raise ValueError(
                    "The size of the permutation {} and the degree of "
                    "the permutation group {} should be matching "
                    .format(g, H))
            G = SymmetricPermutationGroup(g.size)

        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("dir must be of type basestring or "
                    "Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-'):
            raise ValueError("dir must be one of '+' or '-' not %s" % dir)
        obj = Basic.__new__(cls, g, H, G, dir)
        return obj

    def __init__(self, *args, **kwargs):
        self._dir = self.args[3]

    @property
    def is_left_coset(self):
        """
        Check if the coset is left coset that is ``gH``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="-")
        >>> cst.is_left_coset
        True

        """
        return str(self._dir) == '-'

    @property
    def is_right_coset(self):
        """
        Check if the coset is right coset that is ``Hg``.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup, Coset
        >>> a = Permutation(1, 2)
        >>> b = Permutation(0, 1)
        >>> G = PermutationGroup([a, b])
        >>> cst = Coset(a, G, dir="+")
        >>> cst.is_right_coset
        True

        """
        return str(self._dir) == '+'

    def as_list(self):
        """
        Return all the elements of coset in the form of list.
        """
        g = self.args[0]
        H = self.args[1]
        cst = []
        if str(self._dir) == '+':
            for h in H.elements:
                cst.append(h*g)
        else:
            for h in H.elements:
                cst.append(g*h)
        return cst