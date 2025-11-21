
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\sql.py ===
"""
Collection of query wrappers / abstractions to both facilitate data
retrieval and to reduce dependency on DB-specific API.
"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from contextlib import (
    ExitStack,
    contextmanager,
)
from datetime import (
    date,
    datetime,
    time,
)
from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    DatabaseError,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas import get_option
from pandas.core.api import (
    DataFrame,
    Series,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.common import maybe_make_list
from pandas.core.internals.construction import convert_object_array
from pandas.core.tools.datetimes import to_datetime

from pandas.io._util import arrow_table_to_pandas

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Mapping,
    )

    from sqlalchemy import Table
    from sqlalchemy.sql.expression import (
        Select,
        TextClause,
    )

    from pandas._typing import (
        DateTimeErrorChoices,
        DtypeArg,
        DtypeBackend,
        IndexLabel,
        Self,
    )

    from pandas import Index

# -----------------------------------------------------------------------------
# -- Helper functions


def _process_parse_dates_argument(parse_dates):
    """Process parse_dates argument for read_sql functions"""
    # handle non-list entries for parse_dates gracefully
    if parse_dates is True or parse_dates is None or parse_dates is False:
        parse_dates = []

    elif not hasattr(parse_dates, "__iter__"):
        parse_dates = [parse_dates]
    return parse_dates


def _handle_date_column(
    col, utc: bool = False, format: str | dict[str, Any] | None = None
):
    if isinstance(format, dict):
        # GH35185 Allow custom error values in parse_dates argument of
        # read_sql like functions.
        # Format can take on custom to_datetime argument values such as
        # {"errors": "coerce"} or {"dayfirst": True}
        error: DateTimeErrorChoices = format.pop("errors", None) or "ignore"
        if error == "ignore":
            try:
                return to_datetime(col, **format)
            except (TypeError, ValueError):
                # TODO: not reached 2023-10-27; needed?
                return col
        return to_datetime(col, errors=error, **format)
    else:
        # Allow passing of formatting string for integers
        # GH17855
        if format is None and (
            issubclass(col.dtype.type, np.floating)
            or issubclass(col.dtype.type, np.integer)
        ):
            format = "s"
        if format in ["D", "d", "h", "m", "s", "ms", "us", "ns"]:
            return to_datetime(col, errors="coerce", unit=format, utc=utc)
        elif isinstance(col.dtype, DatetimeTZDtype):
            # coerce to UTC timezone
            # GH11216
            return to_datetime(col, utc=True)
        else:
            return to_datetime(col, errors="coerce", format=format, utc=utc)


def _parse_date_columns(data_frame, parse_dates):
    """
    Force non-datetime columns to be read as such.
    Supports both string formatted and integer timestamp columns.
    """
    parse_dates = _process_parse_dates_argument(parse_dates)

    # we want to coerce datetime64_tz dtypes for now to UTC
    # we could in theory do a 'nice' conversion from a FixedOffset tz
    # GH11216
    for i, (col_name, df_col) in enumerate(data_frame.items()):
        if isinstance(df_col.dtype, DatetimeTZDtype) or col_name in parse_dates:
            try:
                fmt = parse_dates[col_name]
            except (KeyError, TypeError):
                fmt = None
            data_frame.isetitem(i, _handle_date_column(df_col, format=fmt))

    return data_frame


def _convert_arrays_to_dataframe(
    data,
    columns,
    coerce_float: bool = True,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    content = lib.to_object_array_tuples(data)
    arrays = convert_object_array(
        list(content.T),
        dtype=None,
        coerce_float=coerce_float,
        dtype_backend=dtype_backend,
    )
    if dtype_backend == "pyarrow":
        pa = import_optional_dependency("pyarrow")

        result_arrays = []
        for arr in arrays:
            pa_array = pa.array(arr, from_pandas=True)
            if arr.dtype == "string":
                # TODO: Arrow still infers strings arrays as regular strings instead
                # of large_string, which is what we preserver everywhere else for
                # dtype_backend="pyarrow". We may want to reconsider this
                pa_array = pa_array.cast(pa.string())
            result_arrays.append(ArrowExtensionArray(pa_array))
        arrays = result_arrays  # type: ignore[assignment]
    if arrays:
        df = DataFrame(dict(zip(list(range(len(columns))), arrays)))
        df.columns = columns
        return df
    else:
        return DataFrame(columns=columns)


def _wrap_result(
    data,
    columns,
    index_col=None,
    coerce_float: bool = True,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
):
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    frame = _convert_arrays_to_dataframe(data, columns, coerce_float, dtype_backend)

    if dtype:
        frame = frame.astype(dtype)

    frame = _parse_date_columns(frame, parse_dates)

    if index_col is not None:
        frame = frame.set_index(index_col)

    return frame


def _wrap_result_adbc(
    df: DataFrame,
    *,
    index_col=None,
    parse_dates=None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
) -> DataFrame:
    """Wrap result set of a SQLAlchemy query in a DataFrame."""
    if dtype:
        df = df.astype(dtype)

    df = _parse_date_columns(df, parse_dates)

    if index_col is not None:
        df = df.set_index(index_col)

    return df


def execute(sql, con, params=None):
    """
    Execute the given SQL query using the provided connection object.

    Parameters
    ----------
    sql : string
        SQL query to be executed.
    con : SQLAlchemy connection or sqlite3 connection
        If a DBAPI2 object, only sqlite3 is supported.
    params : list or tuple, optional, default: None
        List of parameters to pass to execute method.

    Returns
    -------
    Results Iterable
    """
    warnings.warn(
        "`pandas.io.sql.execute` is deprecated and "
        "will be removed in the future version.",
        FutureWarning,
        stacklevel=find_stack_level(),
    )  # GH50185
    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")

    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Engine)):
        raise TypeError("pandas.io.sql.execute requires a connection")  # GH50185
    with pandasSQL_builder(con, need_transaction=True) as pandas_sql:
        return pandas_sql.execute(sql, params)


# -----------------------------------------------------------------------------
# -- Read and write to DataFrames


@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    columns: list[str] | None = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


@overload
def read_sql_table(
    table_name: str,
    con,
    schema=...,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    columns: list[str] | None = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]:
    ...


def read_sql_table(
    table_name: str,
    con,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list[str] | dict[str, str] | None = None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.
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
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table('table_name', 'postgres:///db_name')  # doctest:+SKIP
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        if not pandas_sql.has_table(table_name):
            raise ValueError(f"Table {table_name} not found")

        table = pandas_sql.read_table(
            table_name,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    if table is not None:
        return table
    else:
        raise ValueError(f"Table {table_name} not found", con)


@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


@overload
def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params: list[Any] | Mapping[str, Any] | None = ...,
    parse_dates: list[str] | dict[str, str] | None = ...,
    chunksize: int = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> Iterator[DataFrame]:
    ...


def read_sql_query(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list[Any] | Mapping[str, Any] | None = None,
    parse_dates: list[str] | dict[str, str] | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or mapping, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.

        .. versionadded:: 1.3.0
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
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.

    Examples
    --------
    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("sqlite:///database.db")  # doctest: +SKIP
    >>> with engine.connect() as conn, conn.begin():  # doctest: +SKIP
    ...     data = pd.read_sql_table("data", conn)  # doctest: +SKIP
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        return pandas_sql.read_query(
            sql,
            index_col=index_col,
            params=params,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            chunksize=chunksize,
            dtype=dtype,
            dtype_backend=dtype_backend,
        )


@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> DataFrame:
    ...


@overload
def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = ...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns: list[str] = ...,
    chunksize: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    dtype: DtypeArg | None = None,
) -> Iterator[DataFrame]:
    ...


def read_sql(
    sql,
    con,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params=None,
    parse_dates=None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    dtype: DtypeArg | None = None,
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the ADBC connection and
        SQLAlchemy connectable; str connections are closed automatically. See
        `here <https://docs.sqlalchemy.org/en/20/core/connections.html>`_.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {'a': np.float64, 'b': np.int32, 'c': 'Int64'}.
        The argument is ignored if a table is passed instead of a query.

        .. versionadded:: 2.0.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Examples
    --------
    Read data from SQL via either a SQL query or a SQL tablename.
    When using a SQLite database only SQL queries are accepted,
    providing only the SQL tablename will result in an error.

    >>> from sqlite3 import connect
    >>> conn = connect(':memory:')
    >>> df = pd.DataFrame(data=[[0, '10/11/12'], [1, '12/11/10']],
    ...                   columns=['int_column', 'date_column'])
    >>> df.to_sql(name='test_data', con=conn)
    2

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data', conn)
       int_column date_column
    0           0    10/11/12
    1           1    12/11/10

    >>> pd.read_sql('test_data', 'postgres:///db_name')  # doctest:+SKIP

    Apply date parsing to columns through the ``parse_dates`` argument
    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.
    Custom argument values for applying ``pd.to_datetime`` on a column are specified
    via a dictionary format:

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data',
    ...             conn,
    ...             parse_dates={"date_column": {"format": "%d/%m/%y"}})
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12

    .. versionadded:: 2.2.0

       pandas now supports reading via ADBC drivers

    >>> from adbc_driver_postgresql import dbapi  # doctest:+SKIP
    >>> with dbapi.connect('postgres:///db_name') as conn:  # doctest:+SKIP
    ...     pd.read_sql('SELECT int_column FROM test_data', conn)
       int_column
    0           0
    1           1
    """

    check_dtype_backend(dtype_backend)
    if dtype_backend is lib.no_default:
        dtype_backend = "numpy"  # type: ignore[assignment]
    assert dtype_backend is not lib.no_default

    with pandasSQL_builder(con) as pandas_sql:
        if isinstance(pandas_sql, SQLiteDatabase):
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )

        try:
            _is_table_name = pandas_sql.has_table(sql)
        except Exception:
            # using generic exception to catch errors from sql drivers (GH24988)
            _is_table_name = False

        if _is_table_name:
            return pandas_sql.read_table(
                sql,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
            )
        else:
            return pandas_sql.read_query(
                sql,
                index_col=index_col,
                params=params,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
                dtype=dtype,
            )


def to_sql(
    frame,
    name: str,
    con,
    schema: str | None = None,
    if_exists: Literal["fail", "replace", "append"] = "fail",
    index: bool = True,
    index_label: IndexLabel | None = None,
    chunksize: int | None = None,
    dtype: DtypeArg | None = None,
    method: Literal["multi"] | Callable | None = None,
    engine: str = "auto",
    **engine_kwargs,
) -> int | None:
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    frame : DataFrame, Series
    name : str
        Name of SQL table.
    con : ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        or sqlite3 DBAPI2 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : str, optional
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
    index : bool, default True
        Write DataFrame index as a column.
    index_label : str or sequence, optional
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.
    chunksize : int, optional
        Specify the number of rows in each batch to be written at a time.
        By default, all rows will be written at once.
    dtype : dict or scalar, optional
        Specifying the datatype for columns. If a dictionary is used, the
        keys should be the column names and the values should be the
        SQLAlchemy types or strings for the sqlite3 fallback mode. If a
        scalar is provided, it will be applied to all columns.
    method : {None, 'multi', callable}, optional
        Controls the SQL insertion clause used:

        - None : Uses standard SQL ``INSERT`` clause (one per row).
        - ``'multi'``: Pass multiple values in a single ``INSERT`` clause.
        - callable with signature ``(pd_table, conn, keys, data_iter) -> int | None``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.
    engine : {'auto', 'sqlalchemy'}, default 'auto'
        SQL engine library to use. If 'auto', then the option
        ``io.sql.engine`` is used. The default ``io.sql.engine``
        behavior is 'sqlalchemy'

        .. versionadded:: 1.3.0

    **engine_kwargs
        Any additional kwargs are passed to the engine.

    Returns
    -------
    None or int
        Number of rows affected by to_sql. None is returned if the callable
        passed into ``method`` does not return an integer number of rows.

        .. versionadded:: 1.4.0

    Notes
    -----
    The returned rows affected is the sum of the ``rowcount`` attribute of ``sqlite3.Cursor``
    or SQLAlchemy connectable. If using ADBC the returned rows are the result
    of ``Cursor.adbc_ingest``. The returned value may not reflect the exact number of written
    rows as stipulated in the
    `sqlite3 <https://docs.python.org/3/library/sqlite3.html#sqlite3.Cursor.rowcount>`__ or
    `SQLAlchemy <https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.BaseCursorResult.rowcount>`__
    """  # noqa: E501
    if if_exists not in ("fail", "replace", "append"):
        raise ValueError(f"'{if_exists}' is not valid for if_exists")

    if isinstance(frame, Series):
        frame = frame.to_frame()
    elif not isinstance(frame, DataFrame):
        raise NotImplementedError(
            "'frame' argument should be either a Series or a DataFrame"
        )

    with pandasSQL_builder(con, schema=schema, need_transaction=True) as pandas_sql:
        return pandas_sql.to_sql(
            frame,
            name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
            engine=engine,
            **engine_kwargs,
        )


def has_table(table_name: str, con, schema: str | None = None) -> bool:
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: ADBC Connection, SQLAlchemy connectable, str, or sqlite3 connection
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
    with pandasSQL_builder(con, schema=schema) as pandas_sql:
        return pandas_sql.has_table(table_name)


table_exists = has_table


def pandasSQL_builder(
    con,
    schema: str | None = None,
    need_transaction: bool = False,
) -> PandasSQL:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.  Also creates a sqlalchemy connection and transaction
    if necessary.
    """
    import sqlite3

    if isinstance(con, sqlite3.Connection) or con is None:
        return SQLiteDatabase(con)

    sqlalchemy = import_optional_dependency("sqlalchemy", errors="ignore")

    if isinstance(con, str) and sqlalchemy is None:
        raise ImportError("Using URI string without sqlalchemy installed.")

    if sqlalchemy is not None and isinstance(con, (str, sqlalchemy.engine.Connectable)):
        return SQLDatabase(con, schema, need_transaction)

    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    if adbc and isinstance(con, adbc.Connection):
        return ADBCDatabase(con)

    warnings.warn(
        "pandas only supports SQLAlchemy connectable (engine/connection) or "
        "database string URI or sqlite3 DBAPI2 connection. Other DBAPI2 "
        "objects are not tested. Please consider using SQLAlchemy.",
        UserWarning,
        stacklevel=find_stack_level(),
    )
    return SQLiteDatabase(con)


class SQLTable(PandasObject):
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """

    # TODO: support for multiIndex

    def __init__(
        self,
        name: str,
        pandas_sql_engine,
        frame=None,
        index: bool | str | list[str] | None = True,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        prefix: str = "pandas",
        index_label=None,
        schema=None,
        keys=None,
        dtype: DtypeArg | None = None,
    ) -> None:
        self.name = name
        self.pd_sql = pandas_sql_engine
        self.prefix = prefix
        self.frame = frame
        self.index = self._index_name(index, index_label)
        self.schema = schema
        self.if_exists = if_exists
        self.keys = keys
        self.dtype = dtype

        if frame is not None:
            # We want to initialize based on a dataframe
            self.table = self._create_table_setup()
        else:
            # no data provided, read-only mode
            self.table = self.pd_sql.get_table(self.name, self.schema)

        if self.table is None:
            raise ValueError(f"Could not init table '{name}'")

        if not len(self.name):
            raise ValueError("Empty table name specified")

    def exists(self):
        return self.pd_sql.has_table(self.name, self.schema)

    def sql_schema(self) -> str:
        from sqlalchemy.schema import CreateTable

        return str(CreateTable(self.table).compile(self.pd_sql.con))

    def _execute_create(self) -> None:
        # Inserting table into database, add to MetaData object
        self.table = self.table.to_metadata(self.pd_sql.meta)
        with self.pd_sql.run_transaction():
            self.table.create(bind=self.pd_sql.con)

    def create(self) -> None:
        if self.exists():
            if self.if_exists == "fail":
                raise ValueError(f"Table '{self.name}' already exists.")
            if self.if_exists == "replace":
                self.pd_sql.drop_table(self.name, self.schema)
                self._execute_create()
            elif self.if_exists == "append":
                pass
            else:
                raise ValueError(f"'{self.if_exists}' is not valid for if_exists")
        else:
            self._execute_create()

    def _execute_insert(self, conn, keys: list[str], data_iter) -> int:
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
           Column names
        data_iter : generator of list
           Each item contains a list of values to be inserted
        """
        data = [dict(zip(keys, row)) for row in data_iter]
        result = conn.execute(self.table.insert(), data)
        return result.rowcount

    def _execute_insert_multi(self, conn, keys: list[str], data_iter) -> int:
        """
        Alternative to _execute_insert for DBs support multi-value INSERT.

        Note: multi-value insert is usually faster for analytics DBs
        and tables containing a few columns
        but performance degrades quickly with increase of columns.

        """

        from sqlalchemy import insert

        data = [dict(zip(keys, row)) for row in data_iter]
        stmt = insert(self.table).values(data)
        result = conn.execute(stmt)
        return result.rowcount

    def insert_data(self) -> tuple[list[str], list[np.ndarray]]:
        if self.index is not None:
            temp = self.frame.copy()
            temp.index.names = self.index
            try:
                temp.reset_index(inplace=True)
            except ValueError as err:
                raise ValueError(f"duplicate name in index/columns: {err}") from err
        else:
            temp = self.frame

        column_names = list(map(str, temp.columns))
        ncols = len(column_names)
        # this just pre-allocates the list: None's will be replaced with ndarrays
        # error: List item 0 has incompatible type "None"; expected "ndarray"
        data_list: list[np.ndarray] = [None] * ncols  # type: ignore[list-item]

        for i, (_, ser) in enumerate(temp.items()):
            if ser.dtype.kind == "M":
                if isinstance(ser._values, ArrowExtensionArray):
                    import pyarrow as pa

                    if pa.types.is_date(ser.dtype.pyarrow_dtype):
                        # GH#53854 to_pydatetime not supported for pyarrow date dtypes
                        d = ser._values.to_numpy(dtype=object)
                    else:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            # GH#52459 to_pydatetime will return Index[object]
                            d = np.asarray(ser.dt.to_pydatetime(), dtype=object)
                else:
                    d = ser._values.to_pydatetime()
            elif ser.dtype.kind == "m":
                vals = ser._values
                if isinstance(vals, ArrowExtensionArray):
                    vals = vals.to_numpy(dtype=np.dtype("m8[ns]"))
                # store as integers, see GH#6921, GH#7076
                d = vals.view("i8").astype(object)
            else:
                d = ser._values.astype(object)

            assert isinstance(d, np.ndarray), type(d)

            if ser._can_hold_na:
                # Note: this will miss timedeltas since they are converted to int
                mask = isna(d)
                d[mask] = None

            data_list[i] = d

        return column_names, data_list

    def insert(
        self,
        chunksize: int | None = None,
        method: Literal["multi"] | Callable | None = None,
    ) -> int | None:
        # set insert method
        if method is None:
            exec_insert = self._execute_insert
        elif method == "multi":
            exec_insert = self._execute_insert_multi
        elif callable(method):
            exec_insert = partial(method, self)
        else:
            raise ValueError(f"Invalid parameter `method`: {method}")

        keys, data_list = self.insert_data()

        nrows = len(self.frame)

        if nrows == 0:
            return 0

        if chunksize is None:
            chunksize = nrows
        elif chunksize == 0:
            raise ValueError("chunksize argument should be non-zero")

        chunks = (nrows // chunksize) + 1
        total_inserted = None
        with self.pd_sql.run_transaction() as conn:
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, nrows)
                if start_i >= end_i:
                    break

                chunk_iter = zip(*(arr[start_i:end_i] for arr in data_list))
                num_inserted = exec_insert(conn, keys, chunk_iter)
                # GH 46891
                if num_inserted is not None:
                    if total_inserted is None:
                        total_inserted = num_inserted
                    else:
                        total_inserted += num_inserted
        return total_inserted

    def _query_iterator(
        self,
        result,
        exit_stack: ExitStack,
        chunksize: int | None,
        columns,
        coerce_float: bool = True,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Return generator through chunked result set."""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield DataFrame.from_records(
                            [], columns=columns, coerce_float=coerce_float
                        )
                    break

                has_read_data = True
                self.frame = _convert_arrays_to_dataframe(
                    data, columns, coerce_float, dtype_backend
                )

                self._harmonize_columns(
                    parse_dates=parse_dates, dtype_backend=dtype_backend
                )

                if self.index is not None:
                    self.frame.set_index(self.index, inplace=True)

                yield self.frame

    def read(
        self,
        exit_stack: ExitStack,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        from sqlalchemy import select

        if columns is not None and len(columns) > 0:
            cols = [self.table.c[n] for n in columns]
            if self.index is not None:
                for idx in self.index[::-1]:
                    cols.insert(0, self.table.c[idx])
            sql_select = select(*cols)
        else:
            sql_select = select(self.table)
        result = self.pd_sql.execute(sql_select)
        column_names = result.keys()

        if chunksize is not None:
            return self._query_iterator(
                result,
                exit_stack,
                chunksize,
                column_names,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            self.frame = _convert_arrays_to_dataframe(
                data, column_names, coerce_float, dtype_backend
            )

            self._harmonize_columns(
                parse_dates=parse_dates, dtype_backend=dtype_backend
            )

            if self.index is not None:
                self.frame.set_index(self.index, inplace=True)

            return self.frame

    def _index_name(self, index, index_label):
        # for writing: index=True to include index in sql table
        if index is True:
            nlevels = self.frame.index.nlevels
            # if index_label is specified, set this as index name(s)
            if index_label is not None:
                if not isinstance(index_label, list):
                    index_label = [index_label]
                if len(index_label) != nlevels:
                    raise ValueError(
                        "Length of 'index_label' should match number of "
                        f"levels, which is {nlevels}"
                    )
                return index_label
            # return the used column labels for the index columns
            if (
                nlevels == 1
                and "index" not in self.frame.columns
                and self.frame.index.name is None
            ):
                return ["index"]
            else:
                return com.fill_missing_names(self.frame.index.names)

        # for reading: index=(list of) string to specify column to set as index
        elif isinstance(index, str):
            return [index]
        elif isinstance(index, list):
            return index
        else:
            return None

    def _get_column_names_and_types(self, dtype_mapper):
        column_names_and_types = []
        if self.index is not None:
            for i, idx_label in enumerate(self.index):
                idx_type = dtype_mapper(self.frame.index._get_level_values(i))
                column_names_and_types.append((str(idx_label), idx_type, True))

        column_names_and_types += [
            (str(self.frame.columns[i]), dtype_mapper(self.frame.iloc[:, i]), False)
            for i in range(len(self.frame.columns))
        ]

        return column_names_and_types

    def _create_table_setup(self):
        from sqlalchemy import (
            Column,
            PrimaryKeyConstraint,
            Table,
        )
        from sqlalchemy.schema import MetaData

        column_names_and_types = self._get_column_names_and_types(self._sqlalchemy_type)

        columns: list[Any] = [
            Column(name, typ, index=is_index)
            for name, typ, is_index in column_names_and_types
        ]

        if self.keys is not None:
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            pkc = PrimaryKeyConstraint(*keys, name=self.name + "_pk")
            columns.append(pkc)

        schema = self.schema or self.pd_sql.meta.schema

        # At this point, attach to new metadata, only attach to self.meta
        # once table is created.
        meta = MetaData()
        return Table(self.name, meta, *columns, schema=schema)

    def _harmonize_columns(
        self,
        parse_dates=None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> None:
        """
        Make the DataFrame's column types align with the SQL table
        column types.
        Need to work around limited NA value support. Floats are always
        fine, ints must always be floats if there are Null values.
        Booleans are hard because converting bool column with None replaces
        all Nones with false. Therefore only convert bool if there are no
        NA values.
        Datetimes should already be converted to np.datetime64 if supported,
        but here we also force conversion if required.
        """
        parse_dates = _process_parse_dates_argument(parse_dates)

        for sql_col in self.table.columns:
            col_name = sql_col.name
            try:
                df_col = self.frame[col_name]

                # Handle date parsing upfront; don't try to convert columns
                # twice
                if col_name in parse_dates:
                    try:
                        fmt = parse_dates[col_name]
                    except TypeError:
                        fmt = None
                    self.frame[col_name] = _handle_date_column(df_col, format=fmt)
                    continue

                # the type the dataframe column should have
                col_type = self._get_dtype(sql_col.type)

                if (
                    col_type is datetime
                    or col_type is date
                    or col_type is DatetimeTZDtype
                ):
                    # Convert tz-aware Datetime SQL columns to UTC
                    utc = col_type is DatetimeTZDtype
                    self.frame[col_name] = _handle_date_column(df_col, utc=utc)
                elif dtype_backend == "numpy" and col_type is float:
                    # floats support NA, can always convert!
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
                elif (
                    using_string_dtype()
                    and is_string_dtype(col_type)
                    and is_object_dtype(self.frame[col_name])
                ):
                    self.frame[col_name] = df_col.astype(col_type, copy=False)
                elif dtype_backend == "numpy" and len(df_col) == df_col.count():
                    # No NA values, can convert ints and bools
                    if col_type is np.dtype("int64") or col_type is bool:
                        self.frame[col_name] = df_col.astype(col_type, copy=False)
            except KeyError:
                pass  # this column not in results

    def _sqlalchemy_type(self, col: Index | Series):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        from sqlalchemy.types import (
            TIMESTAMP,
            BigInteger,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            SmallInteger,
            Text,
            Time,
        )

        if col_type in ("datetime64", "datetime"):
            # GH 9086: TIMESTAMP is the suggested type if the column contains
            # timezone information
            try:
                # error: Item "Index" of "Union[Index, Series]" has no attribute "dt"
                if col.dt.tz is not None:  # type: ignore[union-attr]
                    return TIMESTAMP(timezone=True)
            except AttributeError:
                # The column is actually a DatetimeIndex
                # GH 26761 or an Index with date-like data e.g. 9999-01-01
                if getattr(col, "tz", None) is not None:
                    return TIMESTAMP(timezone=True)
            return DateTime
        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            return BigInteger
        elif col_type == "floating":
            if col.dtype == "float32":
                return Float(precision=23)
            else:
                return Float(precision=53)
        elif col_type == "integer":
            # GH35076 Map pandas integer to optimal SQLAlchemy integer type
            if col.dtype.name.lower() in ("int8", "uint8", "int16"):
                return SmallInteger
            elif col.dtype.name.lower() in ("uint16", "int32"):
                return Integer
            elif col.dtype.name.lower() == "uint64":
                raise ValueError("Unsigned 64 bit integer datatype is not supported")
            else:
                return BigInteger
        elif col_type == "boolean":
            return Boolean
        elif col_type == "date":
            return Date
        elif col_type == "time":
            return Time
        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        return Text

    def _get_dtype(self, sqltype):
        from sqlalchemy.types import (
            TIMESTAMP,
            Boolean,
            Date,
            DateTime,
            Float,
            Integer,
            String,
        )

        if isinstance(sqltype, Float):
            return float
        elif isinstance(sqltype, Integer):
            # TODO: Refine integer size.
            return np.dtype("int64")
        elif isinstance(sqltype, TIMESTAMP):
            # we have a timezone capable type
            if not sqltype.timezone:
                return datetime
            return DatetimeTZDtype
        elif isinstance(sqltype, DateTime):
            # Caution: np.datetime64 is also a subclass of np.number.
            return datetime
        elif isinstance(sqltype, Date):
            return date
        elif isinstance(sqltype, Boolean):
            return bool
        elif isinstance(sqltype, String):
            if using_string_dtype():
                return StringDtype(na_value=np.nan)

        return object


class PandasSQL(PandasObject, ABC):
    """
    Subclasses Should define read_query and to_sql.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        pass

    @abstractmethod
    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        pass

    @abstractmethod
    def execute(self, sql: str | Select | TextClause, params=None):
        pass

    @abstractmethod
    def has_table(self, name: str, schema: str | None = None) -> bool:
        pass

    @abstractmethod
    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        pass


class BaseEngine:
    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        """
        Inserts data into already-prepared table
        """
        raise AbstractMethodError(self)


class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None:
        import_optional_dependency(
            "sqlalchemy", extra="sqlalchemy is required for SQL support."
        )

    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name: str,
        index: bool | str | list[str] | None = True,
        schema=None,
        chunksize: int | None = None,
        method=None,
        **engine_kwargs,
    ) -> int | None:
        from sqlalchemy import exc

        try:
            return table.insert(chunksize=chunksize, method=method)
        except exc.StatementError as err:
            # GH34431
            # https://stackoverflow.com/a/67358288/6067848
            msg = r"""(\(1054, "Unknown column 'inf(e0)?' in 'field list'"\))(?#
            )|inf can not be used with MySQL"""
            err_text = str(err.orig)
            if re.search(msg, err_text):
                raise ValueError("inf cannot be used with MySQL") from err
            raise err


def get_engine(engine: str) -> BaseEngine:
    """return our implementation"""
    if engine == "auto":
        engine = get_option("io.sql.engine")

    if engine == "auto":
        # try engines in this order
        engine_classes = [SQLAlchemyEngine]

        error_msgs = ""
        for engine_class in engine_classes:
            try:
                return engine_class()
            except ImportError as err:
                error_msgs += "\n - " + str(err)

        raise ImportError(
            "Unable to find a usable engine; "
            "tried using: 'sqlalchemy'.\n"
            "A suitable version of "
            "sqlalchemy is required for sql I/O "
            "support.\n"
            "Trying to import the above resulted in these errors:"
            f"{error_msgs}"
        )

    if engine == "sqlalchemy":
        return SQLAlchemyEngine()

    raise ValueError("engine must be one of 'auto', 'sqlalchemy'")


class SQLDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    con : SQLAlchemy Connectable or URI string.
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    need_transaction : bool, default False
        If True, SQLDatabase will create a transaction.

    """

    def __init__(
        self, con, schema: str | None = None, need_transaction: bool = False
    ) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        from sqlalchemy.schema import MetaData

        # self.exit_stack cleans up the Engine and Connection and commits the
        # transaction if any of those objects was created below.
        # Cleanup happens either in self.__exit__ or at the end of the iterator
        # returned by read_sql when chunksize is not None.
        self.exit_stack = ExitStack()
        if isinstance(con, str):
            con = create_engine(con)
            self.exit_stack.callback(con.dispose)
        if isinstance(con, Engine):
            con = self.exit_stack.enter_context(con.connect())
        if need_transaction and not con.in_transaction():
            self.exit_stack.enter_context(con.begin())
        self.con = con
        self.meta = MetaData(schema=schema)
        self.returns_generator = False

    def __exit__(self, *args) -> None:
        if not self.returns_generator:
            self.exit_stack.close()

    @contextmanager
    def run_transaction(self):
        if not self.con.in_transaction():
            with self.con.begin():
                yield self.con
        else:
            yield self.con

    def execute(self, sql: str | Select | TextClause, params=None):
        """Simple passthrough to SQLAlchemy connectable"""
        args = [] if params is None else [params]
        if isinstance(sql, str):
            return self.con.exec_driver_sql(sql, *args)
        return self.con.execute(sql, *args)

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        index_col : string, optional, default: None
            Column to set as index.
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
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

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        self.meta.reflect(bind=self.con, only=[table_name], views=True)
        table = SQLTable(table_name, self, index=index_col, schema=schema)
        if chunksize is not None:
            self.returns_generator = True
        return table.read(
            self.exit_stack,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
        )

    @staticmethod
    def _query_iterator(
        result,
        exit_stack: ExitStack,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Return generator through chunked result set"""
        has_read_data = False
        with exit_stack:
            while True:
                data = result.fetchmany(chunksize)
                if not data:
                    if not has_read_data:
                        yield _wrap_result(
                            [],
                            columns,
                            index_col=index_col,
                            coerce_float=coerce_float,
                            parse_dates=parse_dates,
                            dtype=dtype,
                            dtype_backend=dtype_backend,
                        )
                    break

                has_read_data = True
                yield _wrap_result(
                    data,
                    columns,
                    index_col=index_col,
                    coerce_float=coerce_float,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    dtype_backend=dtype_backend,
                )

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        result = self.execute(sql, params)
        columns = result.keys()

        if chunksize is not None:
            self.returns_generator = True
            return self._query_iterator(
                result,
                self.exit_stack,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = result.fetchall()
            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    read_sql = read_query

    def prep_table(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool | str | list[str] | None = True,
        index_label=None,
        schema=None,
        dtype: DtypeArg | None = None,
    ) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)

            from sqlalchemy.types import TypeEngine

            for col, my_type in dtype.items():
                if isinstance(my_type, type) and issubclass(my_type, TypeEngine):
                    pass
                elif isinstance(my_type, TypeEngine):
                    pass
                else:
                    raise ValueError(f"The type of {col} is not a SQLAlchemy type")

        table = SQLTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )
        table.create()
        return table

    def check_case_sensitive(
        self,
        name: str,
        schema: str | None,
    ) -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
        if not name.isdigit() and not name.islower():
            # check for potentially case sensitivity issues (GH7815)
            # Only check when name is not a number and name is not lower case
            from sqlalchemy import inspect as sqlalchemy_inspect

            insp = sqlalchemy_inspect(self.con)
            table_names = insp.get_table_names(schema=schema or self.meta.schema)
            if name not in table_names:
                msg = (
                    f"The provided table name '{name}' is not found exactly as "
                    "such in the database after writing the table, possibly "
                    "due to case sensitivity issues. Consider using lower "
                    "case table names."
                )
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.
        method : {None', 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            SQL engine library to use. If 'auto', then the option
            ``io.sql.engine`` is used. The default ``io.sql.engine``
            behavior is 'sqlalchemy'

            .. versionadded:: 1.3.0

        **engine_kwargs
            Any additional kwargs are passed to the engine.
        """
        sql_engine = get_engine(engine)

        table = self.prep_table(
            frame=frame,
            name=name,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            schema=schema,
            dtype=dtype,
        )

        total_inserted = sql_engine.insert_records(
            table=table,
            con=self.con,
            frame=frame,
            name=name,
            index=index,
            schema=schema,
            chunksize=chunksize,
            method=method,
            **engine_kwargs,
        )

        self.check_case_sensitive(name=name, schema=schema)
        return total_inserted

    @property
    def tables(self):
        return self.meta.tables

    def has_table(self, name: str, schema: str | None = None) -> bool:
        from sqlalchemy import inspect as sqlalchemy_inspect

        insp = sqlalchemy_inspect(self.con)
        return insp.has_table(name, schema or self.meta.schema)

    def get_table(self, table_name: str, schema: str | None = None) -> Table:
        from sqlalchemy import (
            Numeric,
            Table,
        )

        schema = schema or self.meta.schema
        tbl = Table(table_name, self.meta, autoload_with=self.con, schema=schema)
        for column in tbl.columns:
            if isinstance(column.type, Numeric):
                column.type.asdecimal = False
        return tbl

    def drop_table(self, table_name: str, schema: str | None = None) -> None:
        schema = schema or self.meta.schema
        if self.has_table(table_name, schema):
            self.meta.reflect(
                bind=self.con, only=[table_name], schema=schema, views=True
            )
            with self.run_transaction():
                self.get_table(table_name, schema).drop(bind=self.con)
            self.meta.clear()

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        table = SQLTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


# ---- SQL without SQLAlchemy ---


class ADBCDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using ADBC to handle DataBase abstraction.

    Parameters
    ----------
    con : adbc_driver_manager.dbapi.Connection
    """

    def __init__(self, con) -> None:
        self.con = con

    @contextmanager
    def run_transaction(self):
        with self.con.cursor() as cur:
            try:
                yield cur
            except Exception:
                self.con.rollback()
                raise
            self.con.commit()

    def execute(self, sql: str | Select | TextClause, params=None):
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Exception as exc:
            try:
                self.con.rollback()
            except Exception as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    def read_table(
        self,
        table_name: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        columns=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        coerce_float : bool, default True
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            Raises NotImplementedError
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

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")

        if columns:
            if index_col:
                index_select = maybe_make_list(index_col)
            else:
                index_select = []
            to_select = index_select + columns
            select_list = ", ".join(f'"{x}"' for x in to_select)
        else:
            select_list = "*"
        if schema:
            stmt = f"SELECT {select_list} FROM {schema}.{table_name}"
        else:
            stmt = f"SELECT {select_list} FROM {table_name}"

        with self.con.cursor() as cur:
            cur.execute(stmt)
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)

        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
        )

    def read_query(
        self,
        sql: str,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Raises NotImplementedError
        params : list, tuple or dict, optional, default: None
            Raises NotImplementedError
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {'a': np.float64, 'b': np.int32, 'c': 'Int64'}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        if coerce_float is not True:
            raise NotImplementedError(
                "'coerce_float' is not implemented for ADBC drivers"
            )
        if params:
            raise NotImplementedError("'params' is not implemented for ADBC drivers")
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")

        with self.con.cursor() as cur:
            cur.execute(sql)
            pa_table = cur.fetch_arrow_table()
            df = arrow_table_to_pandas(pa_table, dtype_backend=dtype_backend)

        return _wrap_result_adbc(
            df,
            index_col=index_col,
            parse_dates=parse_dates,
            dtype=dtype,
        )

    read_sql = read_query

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = True,
        index_label=None,
        schema: str | None = None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Raises NotImplementedError
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            Raises NotImplementedError
        dtype : single type or dict of column name to SQL type, default None
            Raises NotImplementedError
        method : {None', 'multi', callable}, default None
            Raises NotImplementedError
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            Raises NotImplementedError if not set to 'auto'
        """
        if index_label:
            raise NotImplementedError(
                "'index_label' is not implemented for ADBC drivers"
            )
        if chunksize:
            raise NotImplementedError("'chunksize' is not implemented for ADBC drivers")
        if dtype:
            raise NotImplementedError("'dtype' is not implemented for ADBC drivers")
        if method:
            raise NotImplementedError("'method' is not implemented for ADBC drivers")
        if engine != "auto":
            raise NotImplementedError(
                "engine != 'auto' not implemented for ADBC drivers"
            )

        if schema:
            table_name = f"{schema}.{name}"
        else:
            table_name = name

        # pandas if_exists="append" will still create the
        # table if it does not exist; ADBC is more explicit with append/create
        # as applicable modes, so the semantics get blurred across
        # the libraries
        mode = "create"
        if self.has_table(name, schema):
            if if_exists == "fail":
                raise ValueError(f"Table '{table_name}' already exists.")
            elif if_exists == "replace":
                with self.con.cursor() as cur:
                    cur.execute(f"DROP TABLE {table_name}")
            elif if_exists == "append":
                mode = "append"

        import pyarrow as pa

        try:
            tbl = pa.Table.from_pandas(frame, preserve_index=index)
        except pa.ArrowNotImplementedError as exc:
            raise ValueError("datatypes not supported") from exc

        with self.con.cursor() as cur:
            total_inserted = cur.adbc_ingest(
                table_name=name, data=tbl, mode=mode, db_schema_name=schema
            )

        self.con.commit()
        return total_inserted

    def has_table(self, name: str, schema: str | None = None) -> bool:
        meta = self.con.adbc_get_objects(
            db_schema_filter=schema, table_name_filter=name
        ).read_all()

        for catalog_schema in meta["catalog_db_schemas"].to_pylist():
            if not catalog_schema:
                continue
            for schema_record in catalog_schema:
                if not schema_record:
                    continue

                for table_record in schema_record["db_schema_tables"]:
                    if table_record["table_name"] == name:
                        return True

        return False

    def _create_sql_schema(
        self,
        frame: DataFrame,
        table_name: str,
        keys: list[str] | None = None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        raise NotImplementedError("not implemented for adbc")


# sqlite-specific sql strings and handler class
# dictionary used for readability purposes
_SQL_TYPES = {
    "string": "TEXT",
    "floating": "REAL",
    "integer": "INTEGER",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "INTEGER",
}


def _get_unicode_name(name: object):
    try:
        uname = str(name).encode("utf-8", "strict").decode("utf-8")
    except UnicodeError as err:
        raise ValueError(f"Cannot convert identifier to UTF-8: '{name}'") from err
    return uname


def _get_valid_sqlite_name(name: object):
    # See https://stackoverflow.com/questions/6514274/how-do-you-escape-strings\
    # -for-sqlite-table-column-names-in-python
    # Ensure the string can be encoded as UTF-8.
    # Ensure the string does not include any NUL characters.
    # Replace all " with "".
    # Wrap the entire thing in double quotes.

    uname = _get_unicode_name(name)
    if not len(uname):
        raise ValueError("Empty table or column name specified")

    nul_index = uname.find("\x00")
    if nul_index >= 0:
        raise ValueError("SQLite identifier cannot contain NULs")
    return '"' + uname.replace('"', '""') + '"'


class SQLiteTable(SQLTable):
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._register_date_adapters()

    def _register_date_adapters(self) -> None:
        # GH 8341
        # register an adapter callable for datetime.time object
        import sqlite3

        # this will transform time(12,34,56,789) into '12:34:56.000789'
        # (this is what sqlalchemy does)
        def _adapt_time(t) -> str:
            # This is faster than strftime
            return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.{t.microsecond:06d}"

        # Also register adapters for date/datetime and co
        # xref https://docs.python.org/3.12/library/sqlite3.html#adapter-and-converter-recipes
        # Python 3.12+ doesn't auto-register adapters for us anymore

        adapt_date_iso = lambda val: val.isoformat()
        adapt_datetime_iso = lambda val: val.isoformat(" ")

        sqlite3.register_adapter(time, _adapt_time)

        sqlite3.register_adapter(date, adapt_date_iso)
        sqlite3.register_adapter(datetime, adapt_datetime_iso)

        convert_date = lambda val: date.fromisoformat(val.decode())
        convert_timestamp = lambda val: datetime.fromisoformat(val.decode())

        sqlite3.register_converter("date", convert_date)
        sqlite3.register_converter("timestamp", convert_timestamp)

    def sql_schema(self) -> str:
        return str(";\n".join(self.table))

    def _execute_create(self) -> None:
        with self.pd_sql.run_transaction() as conn:
            for stmt in self.table:
                conn.execute(stmt)

    def insert_statement(self, *, num_rows: int) -> str:
        names = list(map(str, self.frame.columns))
        wld = "?"  # wildcard char
        escape = _get_valid_sqlite_name

        if self.index is not None:
            for idx in self.index[::-1]:
                names.insert(0, idx)

        bracketed_names = [escape(column) for column in names]
        col_names = ",".join(bracketed_names)

        row_wildcards = ",".join([wld] * len(names))
        wildcards = ",".join([f"({row_wildcards})" for _ in range(num_rows)])
        insert_statement = (
            f"INSERT INTO {escape(self.name)} ({col_names}) VALUES {wildcards}"
        )
        return insert_statement

    def _execute_insert(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        conn.executemany(self.insert_statement(num_rows=1), data_list)
        return conn.rowcount

    def _execute_insert_multi(self, conn, keys, data_iter) -> int:
        data_list = list(data_iter)
        flattened_data = [x for row in data_list for x in row]
        conn.execute(self.insert_statement(num_rows=len(data_list)), flattened_data)
        return conn.rowcount

    def _create_table_setup(self):
        """
        Return a list of SQL statements that creates a table reflecting the
        structure of a DataFrame.  The first entry will be a CREATE TABLE
        statement while the rest will be CREATE INDEX statements.
        """
        column_names_and_types = self._get_column_names_and_types(self._sql_type_name)
        escape = _get_valid_sqlite_name

        create_tbl_stmts = [
            escape(cname) + " " + ctype for cname, ctype, _ in column_names_and_types
        ]

        if self.keys is not None and len(self.keys):
            if not is_list_like(self.keys):
                keys = [self.keys]
            else:
                keys = self.keys
            cnames_br = ", ".join([escape(c) for c in keys])
            create_tbl_stmts.append(
                f"CONSTRAINT {self.name}_pk PRIMARY KEY ({cnames_br})"
            )
        if self.schema:
            schema_name = self.schema + "."
        else:
            schema_name = ""
        create_stmts = [
            "CREATE TABLE "
            + schema_name
            + escape(self.name)
            + " (\n"
            + ",\n  ".join(create_tbl_stmts)
            + "\n)"
        ]

        ix_cols = [cname for cname, _, is_index in column_names_and_types if is_index]
        if len(ix_cols):
            cnames = "_".join(ix_cols)
            cnames_br = ",".join([escape(c) for c in ix_cols])
            create_stmts.append(
                "CREATE INDEX "
                + escape("ix_" + self.name + "_" + cnames)
                + "ON "
                + escape(self.name)
                + " ("
                + cnames_br
                + ")"
            )

        return create_stmts

    def _sql_type_name(self, col):
        dtype: DtypeArg = self.dtype or {}
        if is_dict_like(dtype):
            dtype = cast(dict, dtype)
            if col.name in dtype:
                return dtype[col.name]

        # Infer type of column, while ignoring missing values.
        # Needed for inserting typed data containing NULLs, GH 8778.
        col_type = lib.infer_dtype(col, skipna=True)

        if col_type == "timedelta64":
            warnings.warn(
                "the 'timedelta' type is not supported, and will be "
                "written as integer values (ns frequency) to the database.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            col_type = "integer"

        elif col_type == "datetime64":
            col_type = "datetime"

        elif col_type == "empty":
            col_type = "string"

        elif col_type == "complex":
            raise ValueError("Complex datatypes not supported")

        if col_type not in _SQL_TYPES:
            col_type = "string"

        return _SQL_TYPES[col_type]


class SQLiteDatabase(PandasSQL):
    """
    Version of SQLDatabase to support SQLite connections (fallback without
    SQLAlchemy). This should only be used internally.

    Parameters
    ----------
    con : sqlite connection object

    """

    def __init__(self, con) -> None:
        self.con = con

    @contextmanager
    def run_transaction(self):
        cur = self.con.cursor()
        try:
            yield cur
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        finally:
            cur.close()

    def execute(self, sql: str | Select | TextClause, params=None):
        if not isinstance(sql, str):
            raise TypeError("Query must be a string unless using sqlalchemy.")
        args = [] if params is None else [params]
        cur = self.con.cursor()
        try:
            cur.execute(sql, *args)
            return cur
        except Exception as exc:
            try:
                self.con.rollback()
            except Exception as inner_exc:  # pragma: no cover
                ex = DatabaseError(
                    f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
                )
                raise ex from inner_exc

            ex = DatabaseError(f"Execution failed on sql '{sql}': {exc}")
            raise ex from exc

    @staticmethod
    def _query_iterator(
        cursor,
        chunksize: int,
        columns,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ):
        """Return generator through chunked result set"""
        has_read_data = False
        while True:
            data = cursor.fetchmany(chunksize)
            if type(data) == tuple:
                data = list(data)
            if not data:
                cursor.close()
                if not has_read_data:
                    result = DataFrame.from_records(
                        [], columns=columns, coerce_float=coerce_float
                    )
                    if dtype:
                        result = result.astype(dtype)
                    yield result
                break

            has_read_data = True
            yield _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )

    def read_query(
        self,
        sql,
        index_col=None,
        coerce_float: bool = True,
        parse_dates=None,
        params=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        dtype_backend: DtypeBackend | Literal["numpy"] = "numpy",
    ) -> DataFrame | Iterator[DataFrame]:
        cursor = self.execute(sql, params)
        columns = [col_desc[0] for col_desc in cursor.description]

        if chunksize is not None:
            return self._query_iterator(
                cursor,
                chunksize,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
        else:
            data = self._fetchall_as_list(cursor)
            cursor.close()

            frame = _wrap_result(
                data,
                columns,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                dtype=dtype,
                dtype_backend=dtype_backend,
            )
            return frame

    def _fetchall_as_list(self, cur):
        result = cur.fetchall()
        if not isinstance(result, list):
            result = list(result)
        return result

    def to_sql(
        self,
        frame,
        name: str,
        if_exists: str = "fail",
        index: bool = True,
        index_label=None,
        schema=None,
        chunksize: int | None = None,
        dtype: DtypeArg | None = None,
        method: Literal["multi"] | Callable | None = None,
        engine: str = "auto",
        **engine_kwargs,
    ) -> int | None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame: DataFrame
        name: string
            Name of SQL table.
        if_exists: {'fail', 'replace', 'append'}, default 'fail'
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if it does not exist.
        index : bool, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Ignored parameter included for compatibility with SQLAlchemy
            version of ``to_sql``.
        chunksize : int, default None
            If not None, then rows will be written in batches of this
            size at a time. If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a string. If all columns are of the same type, one single value
            can be used.
        method : {None, 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        """
        if dtype:
            if not is_dict_like(dtype):
                # error: Value expression in dictionary comprehension has incompatible
                # type "Union[ExtensionDtype, str, dtype[Any], Type[object],
                # Dict[Hashable, Union[ExtensionDtype, Union[str, dtype[Any]],
                # Type[str], Type[float], Type[int], Type[complex], Type[bool],
                # Type[object]]]]"; expected type "Union[ExtensionDtype, str,
                # dtype[Any], Type[object]]"
                dtype = {col_name: dtype for col_name in frame}  # type: ignore[misc]
            else:
                dtype = cast(dict, dtype)

            for col, my_type in dtype.items():
                if not isinstance(my_type, str):
                    raise ValueError(f"{col} ({my_type}) not a string")

        table = SQLiteTable(
            name,
            self,
            frame=frame,
            index=index,
            if_exists=if_exists,
            index_label=index_label,
            dtype=dtype,
        )
        table.create()
        return table.insert(chunksize, method)

    def has_table(self, name: str, schema: str | None = None) -> bool:
        wld = "?"
        query = f"""
        SELECT
            name
        FROM
            sqlite_master
        WHERE
            type IN ('table', 'view')
            AND name={wld};
        """

        return len(self.execute(query, [name]).fetchall()) > 0

    def get_table(self, table_name: str, schema: str | None = None) -> None:
        return None  # not supported in fallback mode

    def drop_table(self, name: str, schema: str | None = None) -> None:
        drop_sql = f"DROP TABLE {_get_valid_sqlite_name(name)}"
        self.execute(drop_sql)

    def _create_sql_schema(
        self,
        frame,
        table_name: str,
        keys=None,
        dtype: DtypeArg | None = None,
        schema: str | None = None,
    ) -> str:
        table = SQLiteTable(
            table_name,
            self,
            frame=frame,
            index=False,
            keys=keys,
            dtype=dtype,
            schema=schema,
        )
        return str(table.sql_schema())


def get_schema(
    frame,
    name: str,
    keys=None,
    con=None,
    dtype: DtypeArg | None = None,
    schema: str | None = None,
) -> str:
    """
    Get the SQL db table schema for the given frame.

    Parameters
    ----------
    frame : DataFrame
    name : str
        name of SQL table
    keys : string or sequence, default: None
        columns to use a primary key
    con: ADBC Connection, SQLAlchemy connectable, sqlite3 connection, default: None
        ADBC provides high performance I/O with native type support, where available.
        Using SQLAlchemy makes it possible to use any DB supported by that
        library
        If a DBAPI2 object, only sqlite3 is supported.
    dtype : dict of column name to SQL type, default None
        Optional specifying the datatype for columns. The SQL type should
        be a SQLAlchemy type, or a string for sqlite3 fallback connection.
    schema: str, default: None
        Optional specifying the schema to be used in creating the table.
    """
    with pandasSQL_builder(con=con) as pandas_sql:
        return pandas_sql._create_sql_schema(
            frame, name, keys=keys, dtype=dtype, schema=schema
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\element.py ===
from __future__ import annotations

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

import re
import warnings

from bs4.css import CSS
from bs4._deprecation import (
    _deprecated,
    _deprecated_alias,
    _deprecated_function_alias,
)
from bs4.formatter import (
    Formatter,
    HTMLFormatter,
    XMLFormatter,
)
from bs4._warnings import AttributeResemblesVariableWarning

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Pattern,
    Set,
    TYPE_CHECKING,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from typing_extensions import (
    Self,
    TypeAlias,
)

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.builder import TreeBuilder
    from bs4.filter import ElementFilter
    from bs4.formatter import (
        _EntitySubstitutionFunction,
        _FormatterOrName,
    )
    from bs4._typing import (
        _AtMostOneElement,
        _AttributeValue,
        _AttributeValues,
        _Encoding,
        _InsertableElement,
        _OneElement,
        _QueryResults,
        _RawOrProcessedAttributeValues,
        _StrainableElement,
        _StrainableAttribute,
        _StrainableAttributes,
        _StrainableString,
    )

_OneOrMoreStringTypes: TypeAlias = Union[
    Type["NavigableString"], Iterable[Type["NavigableString"]]
]

_FindMethodName: TypeAlias = Optional[Union["_StrainableElement", "ElementFilter"]]

# Deprecated module-level attributes.
# See https://peps.python.org/pep-0562/
_deprecated_names = dict(
    whitespace_re="The {name} attribute was deprecated in version 4.7.0. If you need it, make your own copy."
)
#: :meta private:
_deprecated_whitespace_re: Pattern[str] = re.compile(r"\s+")


def __getattr__(name: str) -> Any:
    if name in _deprecated_names:
        message = _deprecated_names[name]
        warnings.warn(message.format(name=name), DeprecationWarning, stacklevel=2)

        return globals()[f"_deprecated_{name}"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: Documents output by Beautiful Soup will be encoded with
#: this encoding unless you specify otherwise.
DEFAULT_OUTPUT_ENCODING: str = "utf-8"

#: A regular expression that can be used to split on whitespace.
nonwhitespace_re: Pattern[str] = re.compile(r"\S+")

#: These encodings are recognized by Python (so `Tag.encode`
#: could theoretically support them) but XML and HTML don't recognize
#: them (so they should not show up in an XML or HTML document as that
#: document's encoding).
#:
#: If an XML document is encoded in one of these encodings, no encoding
#: will be mentioned in the XML declaration. If an HTML document is
#: encoded in one of these encodings, and the HTML document has a
#: <meta> tag that mentions an encoding, the encoding will be given as
#: the empty string.
#:
#: Source:
#: Python documentation, `Python Specific Encodings <https://docs.python.org/3/library/codecs.html#python-specific-encodings>`_
PYTHON_SPECIFIC_ENCODINGS: Set[_Encoding] = set(
    [
        "idna",
        "mbcs",
        "oem",
        "palmos",
        "punycode",
        "raw_unicode_escape",
        "undefined",
        "unicode_escape",
        "raw-unicode-escape",
        "unicode-escape",
        "string-escape",
        "string_escape",
    ]
)


class NamespacedAttribute(str):
    """A namespaced attribute (e.g. the 'xml:lang' in 'xml:lang="en"')
    which remembers the namespace prefix ('xml') and the name ('lang')
    that were used to create it.
    """

    prefix: Optional[str]
    name: Optional[str]
    namespace: Optional[str]

    def __new__(
        cls,
        prefix: Optional[str],
        name: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Self:
        if not name:
            # This is the default namespace. Its name "has no value"
            # per https://www.w3.org/TR/xml-names/#defaulting
            name = None

        if not name:
            obj = str.__new__(cls, prefix)
        elif not prefix:
            # Not really namespaced.
            obj = str.__new__(cls, name)
        else:
            obj = str.__new__(cls, prefix + ":" + name)
        obj.prefix = prefix
        obj.name = name
        obj.namespace = namespace
        return obj


class AttributeValueWithCharsetSubstitution(str):
    """An abstract class standing in for a character encoding specified
    inside an HTML ``<meta>`` tag.

    Subclasses exist for each place such a character encoding might be
    found: either inside the ``charset`` attribute
    (`CharsetMetaAttributeValue`) or inside the ``content`` attribute
    (`ContentMetaAttributeValue`)

    This allows Beautiful Soup to replace that part of the HTML file
    with a different encoding when ouputting a tree as a string.
    """

    # The original, un-encoded value of the ``content`` attribute.
    #: :meta private:
    original_value: str

    def substitute_encoding(self, eventual_encoding: str) -> str:
        """Do whatever's necessary in this implementation-specific
        portion an HTML document to substitute in a specific encoding.
        """
        raise NotImplementedError()


class CharsetMetaAttributeValue(AttributeValueWithCharsetSubstitution):
    """A generic stand-in for the value of a ``<meta>`` tag's ``charset``
    attribute.

    When Beautiful Soup parses the markup ``<meta charset="utf8">``, the
    value of the ``charset`` attribute will become one of these objects.

    If the document is later encoded to an encoding other than UTF-8, its
    ``<meta>`` tag will mention the new encoding instead of ``utf8``.
    """

    def __new__(cls, original_value: str) -> Self:
        # We don't need to use the original value for anything, but
        # it might be useful for the user to know.
        obj = str.__new__(cls, original_value)
        obj.original_value = original_value
        return obj

    def substitute_encoding(self, eventual_encoding: _Encoding = "utf-8") -> str:
        """When an HTML document is being encoded to a given encoding, the
        value of a ``<meta>`` tag's ``charset`` becomes the name of
        the encoding.
        """
        if eventual_encoding in PYTHON_SPECIFIC_ENCODINGS:
            return ""
        return eventual_encoding


class AttributeValueList(List[str]):
    """Class for the list used to hold the values of attributes which
    have multiple values (such as HTML's 'class'). It's just a regular
    list, but you can subclass it and pass it in to the TreeBuilder
    constructor as attribute_value_list_class, to have your subclass
    instantiated instead.
    """


class AttributeDict(Dict[Any,Any]):
    """Superclass for the dictionary used to hold a tag's
    attributes. You can use this, but it's just a regular dict with no
    special logic.
    """


class XMLAttributeDict(AttributeDict):
    """A dictionary for holding a Tag's attributes, which processes
    incoming values for consistency with the HTML spec.
    """

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute value, possibly modifying it to comply with
        the XML spec.

        This just means converting common non-string values to
        strings: XML attributes may have "any literal string as a
        value."
        """
        if value is None:
            value = ""
        if isinstance(value, bool):
            # XML does not define any rules for boolean attributes.
            # Preserve the old Beautiful Soup behavior (a bool that
            # gets converted to a string on output) rather than
            # guessing what the value should be.
            pass
        elif isinstance(value, (int, float)):
            # It's dangerous to convert _every_ attribute value into a
            # plain string, since an attribute value may be a more
            # sophisticated string-like object
            # (e.g. CharsetMetaAttributeValue). But we can definitely
            # convert numeric values and booleans, which are the most common.
            value = str(value)

        super().__setitem__(key, value)


class HTMLAttributeDict(AttributeDict):
    """A dictionary for holding a Tag's attributes, which processes
    incoming values for consistency with the HTML spec, which says
    'Attribute values are a mixture of text and character
    references...'

    Basically, this means converting common non-string values into
    strings, like XMLAttributeDict, though HTML also has some rules
    around boolean attributes that XML doesn't have.
    """

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute value, possibly modifying it to comply
        with the HTML spec,
        """
        if value in (False, None):
            # 'The values "true" and "false" are not allowed on
            # boolean attributes. To represent a false value, the
            # attribute has to be omitted altogether.'
            if key in self:
                del self[key]
            return
        if isinstance(value, bool):
            # 'If the [boolean] attribute is present, its value must
            # either be the empty string or a value that is an ASCII
            # case-insensitive match for the attribute's canonical
            # name, with no leading or trailing whitespace.'
            #
            # [fixme] It's not clear to me whether "canonical name"
            # means fully-qualified name, unqualified name, or
            # (probably not) name with namespace prefix. For now I'm
            # going with unqualified name.
            if isinstance(key, NamespacedAttribute):
                value = key.name
            else:
                value = key
        elif isinstance(value, (int, float)):
            # See note in XMLAttributeDict for the reasoning why we
            # only do this to numbers.
            value = str(value)
        super().__setitem__(key, value)


class ContentMetaAttributeValue(AttributeValueWithCharsetSubstitution):
    """A generic stand-in for the value of a ``<meta>`` tag's ``content``
    attribute.

    When Beautiful Soup parses the markup:
     ``<meta http-equiv="content-type" content="text/html; charset=utf8">``

    The value of the ``content`` attribute will become one of these objects.

    If the document is later encoded to an encoding other than UTF-8, its
    ``<meta>`` tag will mention the new encoding instead of ``utf8``.
    """

    #: Match the 'charset' argument inside the 'content' attribute
    #: of a <meta> tag.
    #: :meta private:
    CHARSET_RE: Pattern[str] = re.compile(r"((^|;)\s*charset=)([^;]*)", re.M)

    def __new__(cls, original_value: str) -> Self:
        cls.CHARSET_RE.search(original_value)
        obj = str.__new__(cls, original_value)
        obj.original_value = original_value
        return obj

    def substitute_encoding(self, eventual_encoding: _Encoding = "utf-8") -> str:
        """When an HTML document is being encoded to a given encoding, the
        value of the ``charset=`` in a ``<meta>`` tag's ``content`` becomes
        the name of the encoding.
        """
        if eventual_encoding in PYTHON_SPECIFIC_ENCODINGS:
            return self.CHARSET_RE.sub("", self.original_value)

        def rewrite(match: re.Match[str]) -> str:
            return match.group(1) + eventual_encoding

        return self.CHARSET_RE.sub(rewrite, self.original_value)


class PageElement(object):
    """An abstract class representing a single element in the parse tree.

    `NavigableString`, `Tag`, etc. are all subclasses of
    `PageElement`. For this reason you'll see a lot of methods that
    return `PageElement`, but you'll never see an actual `PageElement`
    object. For the most part you can think of `PageElement` as
    meaning "a `Tag` or a `NavigableString`."
    """

    #: In general, we can't tell just by looking at an element whether
    #: it's contained in an XML document or an HTML document. But for
    #: `Tag` objects (q.v.) we can store this information at parse time.
    #: :meta private:
    known_xml: Optional[bool] = None

    #: Whether or not this element has been decomposed from the tree
    #: it was created in.
    _decomposed: bool

    parent: Optional[Tag]
    next_element: _AtMostOneElement
    previous_element: _AtMostOneElement
    next_sibling: _AtMostOneElement
    previous_sibling: _AtMostOneElement

    #: Whether or not this element is hidden from generated output.
    #: Only the `BeautifulSoup` object itself is hidden.
    hidden: bool = False

    def setup(
        self,
        parent: Optional[Tag] = None,
        previous_element: _AtMostOneElement = None,
        next_element: _AtMostOneElement = None,
        previous_sibling: _AtMostOneElement = None,
        next_sibling: _AtMostOneElement = None,
    ) -> None:
        """Sets up the initial relations between this element and
        other elements.

        :param parent: The parent of this element.

        :param previous_element: The element parsed immediately before
            this one.

        :param next_element: The element parsed immediately after
            this one.

        :param previous_sibling: The most recently encountered element
            on the same level of the parse tree as this one.

        :param previous_sibling: The next element to be encountered
            on the same level of the parse tree as this one.
        """
        self.parent = parent

        self.previous_element = previous_element
        if self.previous_element is not None:
            self.previous_element.next_element = self

        self.next_element = next_element
        if self.next_element is not None:
            self.next_element.previous_element = self

        self.next_sibling = next_sibling
        if self.next_sibling is not None:
            self.next_sibling.previous_sibling = self

        if (
            previous_sibling is None
            and self.parent is not None
            and self.parent.contents
        ):
            previous_sibling = self.parent.contents[-1]

        self.previous_sibling = previous_sibling
        if self.previous_sibling is not None:
            self.previous_sibling.next_sibling = self

    def format_string(self, s: str, formatter: Optional[_FormatterOrName]) -> str:
        """Format the given string using the given formatter.

        :param s: A string.
        :param formatter: A Formatter object, or a string naming one of the standard formatters.
        """
        if formatter is None:
            return s
        if not isinstance(formatter, Formatter):
            formatter = self.formatter_for_name(formatter)
        output = formatter.substitute(s)
        return output

    def formatter_for_name(
        self, formatter_name: Union[_FormatterOrName, _EntitySubstitutionFunction]
    ) -> Formatter:
        """Look up or create a Formatter for the given identifier,
        if necessary.

        :param formatter: Can be a `Formatter` object (used as-is), a
            function (used as the entity substitution hook for an
            `bs4.formatter.XMLFormatter` or
            `bs4.formatter.HTMLFormatter`), or a string (used to look
            up an `bs4.formatter.XMLFormatter` or
            `bs4.formatter.HTMLFormatter` in the appropriate registry.

        """
        if isinstance(formatter_name, Formatter):
            return formatter_name
        c: type[Formatter]
        registry: Mapping[Optional[str], Formatter]
        if self._is_xml:
            c = XMLFormatter
            registry = XMLFormatter.REGISTRY
        else:
            c = HTMLFormatter
            registry = HTMLFormatter.REGISTRY
        if callable(formatter_name):
            return c(entity_substitution=formatter_name)
        return registry[formatter_name]

    @property
    def _is_xml(self) -> bool:
        """Is this element part of an XML tree or an HTML tree?

        This is used in formatter_for_name, when deciding whether an
        XMLFormatter or HTMLFormatter is more appropriate. It can be
        inefficient, but it should be called very rarely.
        """
        if self.known_xml is not None:
            # Most of the time we will have determined this when the
            # document is parsed.
            return self.known_xml

        # Otherwise, it's likely that this element was created by
        # direct invocation of the constructor from within the user's
        # Python code.
        if self.parent is None:
            # This is the top-level object. It should have .known_xml set
            # from tree creation. If not, take a guess--BS is usually
            # used on HTML markup.
            return getattr(self, "is_xml", False)
        return self.parent._is_xml

    nextSibling = _deprecated_alias("nextSibling", "next_sibling", "4.0.0")
    previousSibling = _deprecated_alias("previousSibling", "previous_sibling", "4.0.0")

    def __deepcopy__(self, memo: Dict[Any, Any], recursive: bool = False) -> Self:
        raise NotImplementedError()

    def __copy__(self) -> Self:
        """A copy of a PageElement can only be a deep copy, because
        only one PageElement can occupy a given place in a parse tree.
        """
        return self.__deepcopy__({})

    default: Iterable[type[NavigableString]] = tuple()  #: :meta private:

    def _all_strings(
        self, strip: bool = False, types: Iterable[type[NavigableString]] = default
    ) -> Iterator[str]:
        """Yield all strings of certain classes, possibly stripping them.

        This is implemented differently in `Tag` and `NavigableString`.
        """
        raise NotImplementedError()

    @property
    def stripped_strings(self) -> Iterator[str]:
        """Yield all interesting strings in this PageElement, stripping them
        first.

        See `Tag` for information on which strings are considered
        interesting in a given context.
        """
        for string in self._all_strings(True):
            yield string

    def get_text(
        self,
        separator: str = "",
        strip: bool = False,
        types: Iterable[Type[NavigableString]] = default,
    ) -> str:
        """Get all child strings of this PageElement, concatenated using the
        given separator.

        :param separator: Strings will be concatenated using this separator.

        :param strip: If True, strings will be stripped before being
            concatenated.

        :param types: A tuple of NavigableString subclasses. Any
            strings of a subclass not found in this list will be
            ignored. Although there are exceptions, the default
            behavior in most cases is to consider only NavigableString
            and CData objects. That means no comments, processing
            instructions, etc.

        :return: A string.
        """
        return separator.join([s for s in self._all_strings(strip, types=types)])

    getText = get_text
    text = property(get_text)

    def replace_with(self, *args: PageElement) -> Self:
        """Replace this `PageElement` with one or more other `PageElement`,
        objects, keeping the rest of the tree the same.

        :return: This `PageElement`, no longer part of the tree.
        """
        if self.parent is None:
            raise ValueError(
                "Cannot replace one element with another when the "
                "element to be replaced is not part of a tree."
            )
        if len(args) == 1 and args[0] is self:
            # Replacing an element with itself is a no-op.
            return self
        if any(x is self.parent for x in args):
            raise ValueError("Cannot replace a Tag with its parent.")
        old_parent = self.parent
        my_index = self.parent.index(self)
        self.extract(_self_index=my_index)
        for idx, replace_with in enumerate(args, start=my_index):
            old_parent.insert(idx, replace_with)
        return self

    replaceWith = _deprecated_function_alias("replaceWith", "replace_with", "4.0.0")

    def wrap(self, wrap_inside: Tag) -> Tag:
        """Wrap this `PageElement` inside a `Tag`.

        :return: ``wrap_inside``, occupying the position in the tree that used
           to be occupied by this object, and with this object now inside it.
        """
        me = self.replace_with(wrap_inside)
        wrap_inside.append(me)
        return wrap_inside

    def extract(self, _self_index: Optional[int] = None) -> Self:
        """Destructively rips this element out of the tree.

        :param _self_index: The location of this element in its parent's
           .contents, if known. Passing this in allows for a performance
           optimization.

        :return: this `PageElement`, no longer part of the tree.
        """
        if self.parent is not None:
            if _self_index is None:
                _self_index = self.parent.index(self)
            del self.parent.contents[_self_index]

        # Find the two elements that would be next to each other if
        # this element (and any children) hadn't been parsed. Connect
        # the two.
        last_child = self._last_descendant()

        # last_child can't be None because we passed accept_self=True
        # into _last_descendant. Worst case, last_child will be
        # self. Making this cast removes several mypy complaints later
        # on as we manipulate last_child.
        last_child = cast(PageElement, last_child)
        next_element = last_child.next_element

        if self.previous_element is not None:
            if self.previous_element is not next_element:
                self.previous_element.next_element = next_element
        if next_element is not None and next_element is not self.previous_element:
            next_element.previous_element = self.previous_element
        self.previous_element = None
        last_child.next_element = None

        self.parent = None
        if (
            self.previous_sibling is not None
            and self.previous_sibling is not self.next_sibling
        ):
            self.previous_sibling.next_sibling = self.next_sibling
        if (
            self.next_sibling is not None
            and self.next_sibling is not self.previous_sibling
        ):
            self.next_sibling.previous_sibling = self.previous_sibling
        self.previous_sibling = self.next_sibling = None
        return self

    def decompose(self) -> None:
        """Recursively destroys this `PageElement` and its children.

        The element will be removed from the tree and wiped out; so
        will everything beneath it.

        The behavior of a decomposed `PageElement` is undefined and you
        should never use one for anything, but if you need to *check*
        whether an element has been decomposed, you can use the
        `PageElement.decomposed` property.
        """
        self.extract()
        e: _AtMostOneElement = self
        next_up: _AtMostOneElement = None
        while e is not None:
            next_up = e.next_element
            e.__dict__.clear()
            if isinstance(e, Tag):
                e.contents = []
            e._decomposed = True
            e = next_up

    def _last_descendant(
        self, is_initialized: bool = True, accept_self: bool = True
    ) -> _AtMostOneElement:
        """Finds the last element beneath this object to be parsed.

        Special note to help you figure things out if your type
        checking is tripped up by the fact that this method returns
        _AtMostOneElement instead of PageElement: the only time
        this method returns None is if `accept_self` is False and the
        `PageElement` has no children--either it's a NavigableString
        or an empty Tag.

        :param is_initialized: Has `PageElement.setup` been called on
            this `PageElement` yet?

        :param accept_self: Is ``self`` an acceptable answer to the
            question?
        """
        if is_initialized and self.next_sibling is not None:
            last_child = self.next_sibling.previous_element
        else:
            last_child = self
            while isinstance(last_child, Tag) and last_child.contents:
                last_child = last_child.contents[-1]
        if not accept_self and last_child is self:
            last_child = None
        return last_child

    _lastRecursiveChild = _deprecated_alias(
        "_lastRecursiveChild", "_last_descendant", "4.0.0"
    )

    def insert_before(self, *args: _InsertableElement) -> List[PageElement]:
        """Makes the given element(s) the immediate predecessor of this one.

        All the elements will have the same `PageElement.parent` as
        this one, and the given elements will occur immediately before
        this one.

        :param args: One or more PageElements.

        :return The list of PageElements that were inserted.
        """
        parent = self.parent
        if parent is None:
            raise ValueError("Element has no parent, so 'before' has no meaning.")
        if any(x is self for x in args):
            raise ValueError("Can't insert an element before itself.")
        results: List[PageElement] = []
        for predecessor in args:
            # Extract first so that the index won't be screwed up if they
            # are siblings.
            if isinstance(predecessor, PageElement):
                predecessor.extract()
            index = parent.index(self)
            results.extend(parent.insert(index, predecessor))

        return results

    def insert_after(self, *args: _InsertableElement) -> List[PageElement]:
        """Makes the given element(s) the immediate successor of this one.

        The elements will have the same `PageElement.parent` as this
        one, and the given elements will occur immediately after this
        one.

        :param args: One or more PageElements.

        :return The list of PageElements that were inserted.
        """
        # Do all error checking before modifying the tree.
        parent = self.parent
        if parent is None:
            raise ValueError("Element has no parent, so 'after' has no meaning.")
        if any(x is self for x in args):
            raise ValueError("Can't insert an element after itself.")

        offset = 0
        results: List[PageElement] = []
        for successor in args:
            # Extract first so that the index won't be screwed up if they
            # are siblings.
            if isinstance(successor, PageElement):
                successor.extract()
            index = parent.index(self)
            results.extend(parent.insert(index + 1 + offset, successor))
            offset += 1

        return results

    def find_next(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        """Find the first PageElement that matches the given criteria and
        appears later in the document than this PageElement.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_one(self.find_all_next, name, attrs, string, **kwargs)

    findNext = _deprecated_function_alias("findNext", "find_next", "4.0.0")

    def find_all_next(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Find all `PageElement` objects that match the given criteria and
        appear later in the document than this `PageElement`.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :param limit: Stop looking after finding this many results.
        :param _stacklevel: Used internally to improve warning messages.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_all(
            name,
            attrs,
            string,
            limit,
            self.next_elements,
            _stacklevel=_stacklevel + 1,
            **kwargs,
        )

    findAllNext = _deprecated_function_alias("findAllNext", "find_all_next", "4.0.0")

    def find_next_sibling(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        """Find the closest sibling to this PageElement that matches the
        given criteria and appears later in the document.

        All find_* methods take a common set of arguments. See the
        online documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a `NavigableString` with specific text.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_one(self.find_next_siblings, name, attrs, string, **kwargs)

    findNextSibling = _deprecated_function_alias(
        "findNextSibling", "find_next_sibling", "4.0.0"
    )

    def find_next_siblings(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Find all siblings of this `PageElement` that match the given criteria
        and appear later in the document.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a `NavigableString` with specific text.
        :param limit: Stop looking after finding this many results.
        :param _stacklevel: Used internally to improve warning messages.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_all(
            name,
            attrs,
            string,
            limit,
            self.next_siblings,
            _stacklevel=_stacklevel + 1,
            **kwargs,
        )

    findNextSiblings = _deprecated_function_alias(
        "findNextSiblings", "find_next_siblings", "4.0.0"
    )
    fetchNextSiblings = _deprecated_function_alias(
        "fetchNextSiblings", "find_next_siblings", "3.0.0"
    )

    def find_previous(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        """Look backwards in the document from this `PageElement` and find the
        first `PageElement` that matches the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a `NavigableString` with specific text.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_one(self.find_all_previous, name, attrs, string, **kwargs)

    findPrevious = _deprecated_function_alias("findPrevious", "find_previous", "3.0.0")

    def find_all_previous(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Look backwards in the document from this `PageElement` and find all
        `PageElement` that match the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a `NavigableString` with specific text.
        :param limit: Stop looking after finding this many results.
        :param _stacklevel: Used internally to improve warning messages.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_all(
            name,
            attrs,
            string,
            limit,
            self.previous_elements,
            _stacklevel=_stacklevel + 1,
            **kwargs,
        )

    findAllPrevious = _deprecated_function_alias(
        "findAllPrevious", "find_all_previous", "4.0.0"
    )
    fetchAllPrevious = _deprecated_function_alias(
        "fetchAllPrevious", "find_all_previous", "3.0.0"
    )

    def find_previous_sibling(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        """Returns the closest sibling to this `PageElement` that matches the
        given criteria and appears earlier in the document.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a `NavigableString` with specific text.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_one(
            self.find_previous_siblings, name, attrs, string, **kwargs
        )

    findPreviousSibling = _deprecated_function_alias(
        "findPreviousSibling", "find_previous_sibling", "4.0.0"
    )

    def find_previous_siblings(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        string: Optional[_StrainableString] = None,
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Returns all siblings to this PageElement that match the
        given criteria and appear earlier in the document.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param string: A filter for a NavigableString with specific text.
        :param limit: Stop looking after finding this many results.
        :param _stacklevel: Used internally to improve warning messages.
        :kwargs: Additional filters on attribute values.
        """
        return self._find_all(
            name,
            attrs,
            string,
            limit,
            self.previous_siblings,
            _stacklevel=_stacklevel + 1,
            **kwargs,
        )

    findPreviousSiblings = _deprecated_function_alias(
        "findPreviousSiblings", "find_previous_siblings", "4.0.0"
    )
    fetchPreviousSiblings = _deprecated_function_alias(
        "fetchPreviousSiblings", "find_previous_siblings", "3.0.0"
    )

    def find_parent(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        """Find the closest parent of this PageElement that matches the given
        criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param self: Whether the PageElement itself should be considered
           as one of its 'parents'.
        :kwargs: Additional filters on attribute values.
        """
        # NOTE: We can't use _find_one because findParents takes a different
        # set of arguments.
        r = None
        results = self.find_parents(
            name, attrs, 1, _stacklevel=3, **kwargs
        )
        if results:
            r = results[0]
        return r

    findParent = _deprecated_function_alias("findParent", "find_parent", "4.0.0")

    def find_parents(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Find all parents of this `PageElement` that match the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param limit: Stop looking after finding this many results.
        :param _stacklevel: Used internally to improve warning messages.
        :kwargs: Additional filters on attribute values.
        """
        iterator = self.parents
        return self._find_all(
            name, attrs, None, limit, iterator, _stacklevel=_stacklevel + 1, **kwargs
        )

    findParents = _deprecated_function_alias("findParents", "find_parents", "4.0.0")
    fetchParents = _deprecated_function_alias("fetchParents", "find_parents", "3.0.0")

    @property
    def next(self) -> _AtMostOneElement:
        """The `PageElement`, if any, that was parsed just after this one."""
        return self.next_element

    @property
    def previous(self) -> _AtMostOneElement:
        """The `PageElement`, if any, that was parsed just before this one."""
        return self.previous_element

    # These methods do the real heavy lifting.

    def _find_one(
        self,
        # TODO-TYPING: "There is no syntax to indicate optional or
        # keyword arguments; such function types are rarely used
        # as callback types." - So, not sure how to get more
        # specific here.
        method: Callable,
        name: _FindMethodName,
        attrs: _StrainableAttributes,
        string: Optional[_StrainableString],
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        r: _AtMostOneElement = None
        results: _QueryResults = method(name, attrs, string, 1, _stacklevel=4, **kwargs)
        if results:
            r = results[0]
        return r

    def _find_all(
        self,
        name: _FindMethodName,
        attrs: _StrainableAttributes,
        string: Optional[_StrainableString],
        limit: Optional[int],
        generator: Iterator[PageElement],
        _stacklevel: int = 3,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Iterates over a generator looking for things that match."""

        if string is None and "text" in kwargs:
            string = kwargs.pop("text")
            warnings.warn(
                "The 'text' argument to find()-type methods is deprecated. Use 'string' instead.",
                DeprecationWarning,
                stacklevel=_stacklevel,
            )

        if "_class" in kwargs:
            warnings.warn(
                AttributeResemblesVariableWarning.MESSAGE
                % dict(
                    original="_class",
                    autocorrect="class_",
                ),
                AttributeResemblesVariableWarning,
                stacklevel=_stacklevel,
            )

        from bs4.filter import ElementFilter

        if isinstance(name, ElementFilter):
            matcher = name
        else:
            matcher = SoupStrainer(name, attrs, string, **kwargs)

        result: Iterable[_OneElement]
        if string is None and not limit and not attrs and not kwargs:
            if name is True or name is None:
                # Optimization to find all tags.
                result = (element for element in generator if isinstance(element, Tag))
                return ResultSet(matcher, result)
            elif isinstance(name, str):
                # Optimization to find all tags with a given name.
                if name.count(":") == 1:
                    # This is a name with a prefix. If this is a namespace-aware document,
                    # we need to match the local name against tag.name. If not,
                    # we need to match the fully-qualified name against tag.name.
                    prefix, local_name = name.split(":", 1)
                else:
                    prefix = None
                    local_name = name
                result = []
                for element in generator:
                    if not isinstance(element, Tag):
                        continue
                    if element.name == name or (
                        element.name == local_name
                        and (prefix is None or element.prefix == prefix)
                    ):
                        result.append(element)
                return ResultSet(matcher, result)
        return matcher.find_all(generator, limit)

    # These generators can be used to navigate starting from both
    # NavigableStrings and Tags.
    @property
    def next_elements(self) -> Iterator[PageElement]:
        """All PageElements that were parsed after this one."""
        i = self.next_element
        while i is not None:
            successor = i.next_element
            yield i
            i = successor

    @property
    def self_and_next_elements(self) -> Iterator[PageElement]:
        """This PageElement, then all PageElements that were parsed after it."""
        return self._self_and(self.next_elements)

    @property
    def next_siblings(self) -> Iterator[PageElement]:
        """All PageElements that are siblings of this one but were parsed
        later.
        """
        i = self.next_sibling
        while i is not None:
            successor = i.next_sibling
            yield i
            i = successor

    @property
    def self_and_next_siblings(self) -> Iterator[PageElement]:
        """This PageElement, then all of its siblings."""
        return self._self_and(self.next_siblings)

    @property
    def previous_elements(self) -> Iterator[PageElement]:
        """All PageElements that were parsed before this one.

        :yield: A sequence of PageElements.
        """
        i = self.previous_element
        while i is not None:
            successor = i.previous_element
            yield i
            i = successor

    @property
    def self_and_previous_elements(self) -> Iterator[PageElement]:
        """This PageElement, then all elements that were parsed
        earlier."""
        return self._self_and(self.previous_elements)

    @property
    def previous_siblings(self) -> Iterator[PageElement]:
        """All PageElements that are siblings of this one but were parsed
        earlier.

        :yield: A sequence of PageElements.
        """
        i = self.previous_sibling
        while i is not None:
            successor = i.previous_sibling
            yield i
            i = successor

    @property
    def self_and_previous_siblings(self) -> Iterator[PageElement]:
        """This PageElement, then all of its siblings that were parsed
        earlier."""
        return self._self_and(self.previous_siblings)

    @property
    def parents(self) -> Iterator[Tag]:
        """All elements that are parents of this PageElement.

        :yield: A sequence of Tags, ending with a BeautifulSoup object.
        """
        i = self.parent
        while i is not None:
            successor = i.parent
            yield i
            i = successor

    @property
    def self_and_parents(self) -> Iterator[PageElement]:
        """This element, then all of its parents.

        :yield: A sequence of PageElements, ending with a BeautifulSoup object.
        """
        return self._self_and(self.parents)

    def _self_and(self, other_generator:Iterator[PageElement]) -> Iterator[PageElement]:
        """Modify a generator by yielding this element, then everything
        yielded by the other generator.
        """
        if not self.hidden:
            yield self
        for i in other_generator:
            yield i

    @property
    def decomposed(self) -> bool:
        """Check whether a PageElement has been decomposed."""
        return getattr(self, "_decomposed", False) or False

    @_deprecated("next_elements", "4.0.0")
    def nextGenerator(self) -> Iterator[PageElement]:
        ":meta private:"
        return self.next_elements

    @_deprecated("next_siblings", "4.0.0")
    def nextSiblingGenerator(self) -> Iterator[PageElement]:
        ":meta private:"
        return self.next_siblings

    @_deprecated("previous_elements", "4.0.0")
    def previousGenerator(self) -> Iterator[PageElement]:
        ":meta private:"
        return self.previous_elements

    @_deprecated("previous_siblings", "4.0.0")
    def previousSiblingGenerator(self) -> Iterator[PageElement]:
        ":meta private:"
        return self.previous_siblings

    @_deprecated("parents", "4.0.0")
    def parentGenerator(self) -> Iterator[PageElement]:
        ":meta private:"
        return self.parents


class NavigableString(str, PageElement):
    """A Python string that is part of a parse tree.

    When Beautiful Soup parses the markup ``<b>penguin</b>``, it will
    create a `NavigableString` for the string "penguin".
    """

    #: A string prepended to the body of the 'real' string
    #: when formatting it as part of a document, such as the '<!--'
    #: in an HTML comment.
    PREFIX: str = ""

    #: A string appended to the body of the 'real' string
    #: when formatting it as part of a document, such as the '-->'
    #: in an HTML comment.
    SUFFIX: str = ""

    def __new__(cls, value: Union[str, bytes]) -> Self:
        """Create a new NavigableString.

        When unpickling a NavigableString, this method is called with
        the string in DEFAULT_OUTPUT_ENCODING. That encoding needs to be
        passed in to the superclass's __new__ or the superclass won't know
        how to handle non-ASCII characters.
        """
        if isinstance(value, str):
            u = str.__new__(cls, value)
        else:
            u = str.__new__(cls, value, DEFAULT_OUTPUT_ENCODING)
        u.hidden = False
        u.setup()
        return u

    def __deepcopy__(self, memo: Dict[Any, Any], recursive: bool = False) -> Self:
        """A copy of a NavigableString has the same contents and class
        as the original, but it is not connected to the parse tree.

        :param recursive: This parameter is ignored; it's only defined
           so that NavigableString.__deepcopy__ implements the same
           signature as Tag.__deepcopy__.
        """
        return type(self)(self)

    def __getnewargs__(self) -> Tuple[str]:
        return (str(self),)

    # TODO-TYPING This should be SupportsIndex|slice but SupportsIndex
    # is introduced in 3.8.
    def __getitem__(self, key: Union[int|slice]) -> str:
        """Raise an exception """
        if isinstance(key, str):
            raise TypeError("string indices must be integers, not '{0}'. Are you treating a NavigableString like a Tag?".format(key.__class__.__name__))
        return super(NavigableString, self).__getitem__(key)

    @property
    def string(self) -> str:
        """Convenience property defined to match `Tag.string`.

        :return: This property always returns the `NavigableString` it was
           called on.

        :meta private:
        """
        return self

    def output_ready(self, formatter: _FormatterOrName = "minimal") -> str:
        """Run the string through the provided formatter, making it
        ready for output as part of an HTML or XML document.

        :param formatter: A `Formatter` object, or a string naming one
            of the standard formatters.
        """
        output = self.format_string(self, formatter)
        return self.PREFIX + output + self.SUFFIX

    @property
    def name(self) -> None:
        """Since a NavigableString is not a Tag, it has no .name.

        This property is implemented so that code like this doesn't crash
        when run on a mixture of Tag and NavigableString objects:
            [x.name for x in tag.children]

        :meta private:
        """
        return None

    @name.setter
    def name(self, name: str) -> None:
        """Prevent NavigableString.name from ever being set.

        :meta private:
        """
        raise AttributeError("A NavigableString cannot be given a name.")

    def _all_strings(
        self, strip: bool = False, types: _OneOrMoreStringTypes = PageElement.default
    ) -> Iterator[str]:
        """Yield all strings of certain classes, possibly stripping them.

        This makes it easy for NavigableString to implement methods
        like get_text() as conveniences, creating a consistent
        text-extraction API across all PageElements.

        :param strip: If True, all strings will be stripped before being
            yielded.

        :param types: A tuple of NavigableString subclasses. If this
            NavigableString isn't one of those subclasses, the
            sequence will be empty. By default, the subclasses
            considered are NavigableString and CData objects. That
            means no comments, processing instructions, etc.

        :yield: A sequence that either contains this string, or is empty.
        """
        if types is self.default:
            # This is kept in Tag because it's full of subclasses of
            # this class, which aren't defined until later in the file.
            types = Tag.MAIN_CONTENT_STRING_TYPES

        # Do nothing if the caller is looking for specific types of
        # string, and we're of a different type.
        #
        # We check specific types instead of using isinstance(self,
        # types) because all of these classes subclass
        # NavigableString. Anyone who's using this feature probably
        # wants generic NavigableStrings but not other stuff.
        my_type = type(self)
        if types is not None:
            if isinstance(types, type):
                # Looking for a single type.
                if my_type is not types:
                    return
            elif my_type not in types:
                # Looking for one of a list of types.
                return

        value = self
        if strip:
            final_value = value.strip()
        else:
            final_value = self
        if len(final_value) > 0:
            yield final_value

    @property
    def strings(self) -> Iterator[str]:
        """Yield this string, but only if it is interesting.

        This is defined the way it is for compatibility with
        `Tag.strings`. See `Tag` for information on which strings are
        interesting in a given context.

        :yield: A sequence that either contains this string, or is empty.
        """
        return self._all_strings()


class PreformattedString(NavigableString):
    """A `NavigableString` not subject to the normal formatting rules.

    This is an abstract class used for special kinds of strings such
    as comments (`Comment`) and CDATA blocks (`CData`).
    """

    PREFIX: str = ""
    SUFFIX: str = ""

    def output_ready(self, formatter: Optional[_FormatterOrName] = None) -> str:
        """Make this string ready for output by adding any subclass-specific
            prefix or suffix.

        :param formatter: A `Formatter` object, or a string naming one
            of the standard formatters. The string will be passed into the
            `Formatter`, but only to trigger any side effects: the return
            value is ignored.

        :return: The string, with any subclass-specific prefix and
           suffix added on.
        """
        if formatter is not None:
            self.format_string(self, formatter)
        return self.PREFIX + self + self.SUFFIX


class CData(PreformattedString):
    """A `CDATA section <https://dev.w3.org/html5/spec-LC/syntax.html#cdata-sections>`_."""

    PREFIX: str = "<![CDATA["
    SUFFIX: str = "]]>"


class ProcessingInstruction(PreformattedString):
    """A SGML processing instruction."""

    PREFIX: str = "<?"
    SUFFIX: str = ">"


class XMLProcessingInstruction(ProcessingInstruction):
    """An `XML processing instruction <https://www.w3.org/TR/REC-xml/#sec-pi>`_."""

    PREFIX: str = "<?"
    SUFFIX: str = "?>"


class Comment(PreformattedString):
    """An `HTML comment <https://dev.w3.org/html5/spec-LC/syntax.html#comments>`_ or `XML comment <https://www.w3.org/TR/REC-xml/#sec-comments>`_."""

    PREFIX: str = "<!--"
    SUFFIX: str = "-->"


class Declaration(PreformattedString):
    """An `XML declaration <https://www.w3.org/TR/REC-xml/#sec-prolog-dtd>`_."""

    PREFIX: str = "<?"
    SUFFIX: str = "?>"


class Doctype(PreformattedString):
    """A `document type declaration <https://www.w3.org/TR/REC-xml/#dt-doctype>`_."""

    @classmethod
    def for_name_and_ids(
        cls, name: str, pub_id: Optional[str], system_id: Optional[str]
    ) -> Doctype:
        """Generate an appropriate document type declaration for a given
        public ID and system ID.

        :param name: The name of the document's root element, e.g. 'html'.
        :param pub_id: The Formal Public Identifier for this document type,
            e.g. '-//W3C//DTD XHTML 1.1//EN'
        :param system_id: The system identifier for this document type,
            e.g. 'http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd'
        """
        return Doctype(cls._string_for_name_and_ids(name, pub_id, system_id))

    @classmethod
    def _string_for_name_and_ids(
        self, name: str, pub_id: Optional[str], system_id: Optional[str]
    ) -> str:
        """Generate a string to be used as the basis of a Doctype object.

        This is a separate method from for_name_and_ids() because the lxml
        TreeBuilder needs to call it.
        """
        value = name or ""
        if pub_id is not None:
            value += ' PUBLIC "%s"' % pub_id
            if system_id is not None:
                value += ' "%s"' % system_id
        elif system_id is not None:
            value += ' SYSTEM "%s"' % system_id
        return value

    PREFIX: str = "<!DOCTYPE "
    SUFFIX: str = ">\n"


class Stylesheet(NavigableString):
    """A `NavigableString` representing the contents of a `<style> HTML
    tag <https://dev.w3.org/html5/spec-LC/Overview.html#the-style-element>`_
    (probably CSS).

    Used to distinguish embedded stylesheets from textual content.
    """


class Script(NavigableString):
    """A `NavigableString` representing the contents of a `<script>
    HTML tag
    <https://dev.w3.org/html5/spec-LC/Overview.html#the-script-element>`_
    (probably Javascript).

    Used to distinguish executable code from textual content.
    """


class TemplateString(NavigableString):
    """A `NavigableString` representing a string found inside an `HTML
    <template> tag <https://html.spec.whatwg.org/multipage/scripting.html#the-template-element>`_
    embedded in a larger document.

    Used to distinguish such strings from the main body of the document.
    """


class RubyTextString(NavigableString):
    """A NavigableString representing the contents of an `<rt> HTML
    tag <https://dev.w3.org/html5/spec-LC/text-level-semantics.html#the-rt-element>`_.

    Can be used to distinguish such strings from the strings they're
    annotating.
    """


class RubyParenthesisString(NavigableString):
    """A NavigableString representing the contents of an `<rp> HTML
    tag <https://dev.w3.org/html5/spec-LC/text-level-semantics.html#the-rp-element>`_.
    """


class Tag(PageElement):
    """An HTML or XML tag that is part of a parse tree, along with its
    attributes, contents, and relationships to other parts of the tree.

    When Beautiful Soup parses the markup ``<b>penguin</b>``, it will
    create a `Tag` object representing the ``<b>`` tag. You can
    instantiate `Tag` objects directly, but it's not necessary unless
    you're adding entirely new markup to a parsed document. Most of
    the constructor arguments are intended for use by the `TreeBuilder`
    that's parsing a document.

    :param parser: A `BeautifulSoup` object representing the parse tree this
        `Tag` will be part of.
    :param builder: The `TreeBuilder` being used to build the tree.
    :param name: The name of the tag.
    :param namespace: The URI of this tag's XML namespace, if any.
    :param prefix: The prefix for this tag's XML namespace, if any.
    :param attrs: A dictionary of attribute values.
    :param parent: The `Tag` to use as the parent of this `Tag`. May be
       the `BeautifulSoup` object itself.
    :param previous: The `PageElement` that was parsed immediately before
        parsing this tag.
    :param is_xml: If True, this is an XML tag. Otherwise, this is an
        HTML tag.
    :param sourceline: The line number where this tag was found in its
        source document.
    :param sourcepos: The character position within ``sourceline`` where this
        tag was found.
    :param can_be_empty_element: If True, this tag should be
        represented as <tag/>. If False, this tag should be represented
        as <tag></tag>.
    :param cdata_list_attributes: A dictionary of attributes whose values should
        be parsed as lists of strings if they ever show up on this tag.
    :param preserve_whitespace_tags: Names of tags whose contents
        should have their whitespace preserved if they are encountered inside
        this tag.
    :param interesting_string_types: When iterating over this tag's
        string contents in methods like `Tag.strings` or
        `PageElement.get_text`, these are the types of strings that are
        interesting enough to be considered. By default,
        `NavigableString` (normal strings) and `CData` (CDATA
        sections) are the only interesting string subtypes.
    :param namespaces: A dictionary mapping currently active
        namespace prefixes to URIs, as of the point in the parsing process when
        this tag was encountered. This can be used later to
        construct CSS selectors.

    """

    def __init__(
        self,
        parser: Optional[BeautifulSoup] = None,
        builder: Optional[TreeBuilder] = None,
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        prefix: Optional[str] = None,
        attrs: Optional[_RawOrProcessedAttributeValues] = None,
        parent: Optional[Union[BeautifulSoup, Tag]] = None,
        previous: _AtMostOneElement = None,
        is_xml: Optional[bool] = None,
        sourceline: Optional[int] = None,
        sourcepos: Optional[int] = None,
        can_be_empty_element: Optional[bool] = None,
        cdata_list_attributes: Optional[Dict[str, Set[str]]] = None,
        preserve_whitespace_tags: Optional[Set[str]] = None,
        interesting_string_types: Optional[Set[Type[NavigableString]]] = None,
        namespaces: Optional[Dict[str, str]] = None,
        # NOTE: Any new arguments here need to be mirrored in
        # Tag.copy_self, and potentially BeautifulSoup.new_tag
        # as well.
    ):
        if parser is None:
            self.parser_class = None
        else:
            # We don't actually store the parser object: that lets extracted
            # chunks be garbage-collected.
            self.parser_class = parser.__class__
        if name is None:
            raise ValueError("No value provided for new tag's name.")
        self.name = name
        self.namespace = namespace
        self._namespaces = namespaces or {}
        self.prefix = prefix
        if (not builder or builder.store_line_numbers) and (
            sourceline is not None or sourcepos is not None
        ):
            self.sourceline = sourceline
            self.sourcepos = sourcepos
        else:
            self.sourceline = sourceline
            self.sourcepos = sourcepos

        attr_dict_class: type[AttributeDict]
        attribute_value_list_class: type[AttributeValueList]
        if builder is None:
            if is_xml:
                attr_dict_class = XMLAttributeDict
            else:
                attr_dict_class = HTMLAttributeDict
            attribute_value_list_class = AttributeValueList
        else:
            attr_dict_class = builder.attribute_dict_class
            attribute_value_list_class = builder.attribute_value_list_class
        self.attribute_value_list_class = attribute_value_list_class

        if attrs is None:
            self.attrs = attr_dict_class()
        else:
            if builder is not None and builder.cdata_list_attributes:
                self.attrs = builder._replace_cdata_list_attribute_values(
                    self.name, attrs
                )
            else:
                self.attrs = attr_dict_class()
                # Make sure that the values of any multi-valued
                # attributes (e.g. when a Tag is copied) are stored in
                # new lists.
                for k, v in attrs.items():
                    if isinstance(v, list):
                        v = v.__class__(v)
                    self.attrs[k] = v

        # If possible, determine ahead of time whether this tag is an
        # XML tag.
        if builder:
            self.known_xml = builder.is_xml
        else:
            self.known_xml = is_xml
        self.contents: List[PageElement] = []
        self.setup(parent, previous)
        self.hidden = False

        if builder is None:
            # In the absence of a TreeBuilder, use whatever values were
            # passed in here. They're probably None, unless this is a copy of some
            # other tag.
            self.can_be_empty_element = can_be_empty_element
            self.cdata_list_attributes = cdata_list_attributes
            self.preserve_whitespace_tags = preserve_whitespace_tags
            self.interesting_string_types = interesting_string_types
        else:
            # Set up any substitutions for this tag, such as the charset in a META tag.
            self.attribute_value_list_class = builder.attribute_value_list_class
            builder.set_up_substitutions(self)

            # Ask the TreeBuilder whether this tag might be an empty-element tag.
            self.can_be_empty_element = builder.can_be_empty_element(name)

            # Keep track of the list of attributes of this tag that
            # might need to be treated as a list.
            #
            # For performance reasons, we store the whole data structure
            # rather than asking the question of every tag. Asking would
            # require building a new data structure every time, and
            # (unlike can_be_empty_element), we almost never need
            # to check this.
            self.cdata_list_attributes = builder.cdata_list_attributes

            # Keep track of the names that might cause this tag to be treated as a
            # whitespace-preserved tag.
            self.preserve_whitespace_tags = builder.preserve_whitespace_tags

            if self.name in builder.string_containers:
                # This sort of tag uses a special string container
                # subclass for most of its strings. We need to be able
                # to look up the proper container subclass.
                self.interesting_string_types = {builder.string_containers[self.name]}
            else:
                self.interesting_string_types = self.MAIN_CONTENT_STRING_TYPES

    parser_class: Optional[type[BeautifulSoup]]
    name: str
    namespace: Optional[str]
    prefix: Optional[str]
    attrs: _AttributeValues
    sourceline: Optional[int]
    sourcepos: Optional[int]
    known_xml: Optional[bool]
    contents: List[PageElement]
    hidden: bool
    interesting_string_types: Optional[Set[Type[NavigableString]]]

    can_be_empty_element: Optional[bool]
    cdata_list_attributes: Optional[Dict[str, Set[str]]]
    preserve_whitespace_tags: Optional[Set[str]]

    #: :meta private:
    parserClass = _deprecated_alias("parserClass", "parser_class", "4.0.0")

    def __deepcopy__(self, memo: Dict[Any, Any], recursive: bool = True) -> Self:
        """A deepcopy of a Tag is a new Tag, unconnected to the parse tree.
        Its contents are a copy of the old Tag's contents.
        """
        clone = self.copy_self()

        if recursive:
            # Clone this tag's descendants recursively, but without
            # making any recursive function calls.
            tag_stack: List[Tag] = [clone]
            for event, element in self._event_stream(self.descendants):
                if event is Tag.END_ELEMENT_EVENT:
                    # Stop appending incoming Tags to the Tag that was
                    # just closed.
                    tag_stack.pop()
                else:
                    descendant_clone = element.__deepcopy__(memo, recursive=False)
                    # Add to its parent's .contents
                    tag_stack[-1].append(descendant_clone)

                    if event is Tag.START_ELEMENT_EVENT:
                        # Add the Tag itself to the stack so that its
                        # children will be .appended to it.
                        tag_stack.append(cast(Tag, descendant_clone))
        return clone

    def copy_self(self) -> Self:
        """Create a new Tag just like this one, but with no
        contents and unattached to any parse tree.

        This is the first step in the deepcopy process, but you can
        call it on its own to create a copy of a Tag without copying its
        contents.
        """
        clone = type(self)(
            None,
            None,
            self.name,
            self.namespace,
            self.prefix,
            self.attrs,
            is_xml=self._is_xml,
            sourceline=self.sourceline,
            sourcepos=self.sourcepos,
            can_be_empty_element=self.can_be_empty_element,
            cdata_list_attributes=self.cdata_list_attributes,
            preserve_whitespace_tags=self.preserve_whitespace_tags,
            interesting_string_types=self.interesting_string_types,
            namespaces=self._namespaces,
        )
        for attr in ("can_be_empty_element", "hidden"):
            setattr(clone, attr, getattr(self, attr))
        return clone

    @property
    def is_empty_element(self) -> bool:
        """Is this tag an empty-element tag? (aka a self-closing tag)

        A tag that has contents is never an empty-element tag.

        A tag that has no contents may or may not be an empty-element
        tag. It depends on the `TreeBuilder` used to create the
        tag. If the builder has a designated list of empty-element
        tags, then only a tag whose name shows up in that list is
        considered an empty-element tag. This is usually the case
        for HTML documents.

        If the builder has no designated list of empty-element, then
        any tag with no contents is an empty-element tag. This is usually
        the case for XML documents.
        """
        return len(self.contents) == 0 and self.can_be_empty_element is True

    @_deprecated("is_empty_element", "4.0.0")
    def isSelfClosing(self) -> bool:
        ": :meta private:"
        return self.is_empty_element

    @property
    def string(self) -> Optional[str]:
        """Convenience property to get the single string within this
        `Tag`, assuming there is just one.

        :return: If this `Tag` has a single child that's a
         `NavigableString`, the return value is that string. If this
         element has one child `Tag`, the return value is that child's
         `Tag.string`, recursively. If this `Tag` has no children,
         or has more than one child, the return value is ``None``.

         If this property is unexpectedly returning ``None`` for you,
         it's probably because your `Tag` has more than one thing
         inside it.
        """
        if len(self.contents) != 1:
            return None
        child = self.contents[0]
        if isinstance(child, NavigableString):
            return child
        elif isinstance(child, Tag):
            return child.string
        return None

    @string.setter
    def string(self, string: str) -> None:
        """Replace the `Tag.contents` of this `Tag` with a single string."""
        self.clear()
        if isinstance(string, NavigableString):
            new_class = string.__class__
        else:
            new_class = NavigableString
        self.append(new_class(string))

    #: :meta private:
    MAIN_CONTENT_STRING_TYPES = {NavigableString, CData}

    def _all_strings(
        self, strip: bool = False, types: _OneOrMoreStringTypes = PageElement.default
    ) -> Iterator[str]:
        """Yield all strings of certain classes, possibly stripping them.

        :param strip: If True, all strings will be stripped before being
            yielded.

        :param types: A tuple of NavigableString subclasses. Any strings of
            a subclass not found in this list will be ignored. By
            default, the subclasses considered are the ones found in
            self.interesting_string_types. If that's not specified,
            only NavigableString and CData objects will be
            considered. That means no comments, processing
            instructions, etc.
        """
        if types is self.default:
            if self.interesting_string_types is None:
                types = self.MAIN_CONTENT_STRING_TYPES
            else:
                types = self.interesting_string_types

        for descendant in self.descendants:
            if not isinstance(descendant, NavigableString):
                continue
            descendant_type = type(descendant)
            if isinstance(types, type):
                if descendant_type is not types:
                    # We're not interested in strings of this type.
                    continue
            elif types is not None and descendant_type not in types:
                # We're not interested in strings of this type.
                continue
            if strip:
                stripped = descendant.strip()
                if len(stripped) == 0:
                    continue
                yield stripped
            else:
                yield descendant

    strings = property(_all_strings)

    def insert(self, position: int, *new_children: _InsertableElement) -> List[PageElement]:
        """Insert one or more new PageElements as a child of this `Tag`.

        This works similarly to :py:meth:`list.insert`, except you can insert
        multiple elements at once.

        :param position: The numeric position that should be occupied
           in this Tag's `Tag.children` by the first new `PageElement`.

        :param new_children: The PageElements to insert.

        :return The newly inserted PageElements.
        """
        inserted: List[PageElement] = []
        for new_child in new_children:
            inserted.extend(self._insert(position, new_child))
            position += 1
        return inserted

    def _insert(self, position: int, new_child: _InsertableElement) -> List[PageElement]:
        if new_child is None:
            raise ValueError("Cannot insert None into a tag.")
        if new_child is self:
            raise ValueError("Cannot insert a tag into itself.")
        if isinstance(new_child, str) and not isinstance(new_child, NavigableString):
            new_child = NavigableString(new_child)

        from bs4 import BeautifulSoup
        if isinstance(new_child, BeautifulSoup):
            # We don't want to end up with a situation where one BeautifulSoup
            # object contains another. Insert the BeautifulSoup's children and
            # return them.
            return self.insert(position, *list(new_child.contents))
        position = min(position, len(self.contents))
        if hasattr(new_child, "parent") and new_child.parent is not None:
            # We're 'inserting' an element that's already one
            # of this object's children.
            if new_child.parent is self:
                current_index = self.index(new_child)
                if current_index < position:
                    # We're moving this element further down the list
                    # of this object's children. That means that when
                    # we extract this element, our target index will
                    # jump down one.
                    position -= 1
                elif current_index == position:
                    # We're 'inserting' an element into its current location.
                    # This is a no-op.
                    return [new_child]
            new_child.extract()

        new_child.parent = self
        previous_child = None
        if position == 0:
            new_child.previous_sibling = None
            new_child.previous_element = self
        else:
            previous_child = self.contents[position - 1]
            new_child.previous_sibling = previous_child
            new_child.previous_sibling.next_sibling = new_child
            new_child.previous_element = previous_child._last_descendant(False)
        if new_child.previous_element is not None:
            new_child.previous_element.next_element = new_child

        new_childs_last_element = new_child._last_descendant(
            is_initialized=False, accept_self=True
        )
        # new_childs_last_element can't be None because we passed
        # accept_self=True into _last_descendant. Worst case,
        # new_childs_last_element will be new_child itself. Making
        # this cast removes several mypy complaints later on as we
        # manipulate new_childs_last_element.
        new_childs_last_element = cast(PageElement, new_childs_last_element)

        if position >= len(self.contents):
            new_child.next_sibling = None

            parent: Optional[Tag] = self
            parents_next_sibling = None
            while parents_next_sibling is None and parent is not None:
                parents_next_sibling = parent.next_sibling
                parent = parent.parent
                if parents_next_sibling is not None:
                    # We found the element that comes next in the document.
                    break
            if parents_next_sibling is not None:
                new_childs_last_element.next_element = parents_next_sibling
            else:
                # The last element of this tag is the last element in
                # the document.
                new_childs_last_element.next_element = None
        else:
            next_child = self.contents[position]
            new_child.next_sibling = next_child
            if new_child.next_sibling is not None:
                new_child.next_sibling.previous_sibling = new_child
            new_childs_last_element.next_element = next_child

        if new_childs_last_element.next_element is not None:
            new_childs_last_element.next_element.previous_element = (
                new_childs_last_element
            )
        self.contents.insert(position, new_child)

        return [new_child]

    def unwrap(self) -> Self:
        """Replace this `PageElement` with its contents.

        :return: This object, no longer part of the tree.
        """
        my_parent = self.parent
        if my_parent is None:
            raise ValueError(
                "Cannot replace an element with its contents when that "
                "element is not part of a tree."
            )
        my_index = my_parent.index(self)
        self.extract(_self_index=my_index)
        for child in reversed(self.contents[:]):
            my_parent.insert(my_index, child)
        return self

    replace_with_children = unwrap

    @_deprecated("unwrap", "4.0.0")
    def replaceWithChildren(self) -> _OneElement:
        ": :meta private:"
        return self.unwrap()

    def append(self, tag: _InsertableElement) -> PageElement:
        """
        Appends the given `PageElement` to the contents of this `Tag`.

        :param tag: A PageElement.

        :return The newly appended PageElement.
        """
        return self.insert(len(self.contents), tag)[0]

    def extend(self, tags: Union[Iterable[_InsertableElement], Tag]) -> List[PageElement]:
        """Appends one or more objects to the contents of this
        `Tag`.

        :param tags: If a list of `PageElement` objects is provided,
            they will be appended to this tag's contents, one at a time.
            If a single `Tag` is provided, its `Tag.contents` will be
            used to extend this object's `Tag.contents`.

        :return The list of PageElements that were appended.
        """
        tag_list: Iterable[_InsertableElement]

        if isinstance(tags, Tag):
            tag_list = list(tags.contents)
        elif isinstance(tags, (PageElement, str)):
            # The caller should really be using append() instead,
            # but we can make it work.
            warnings.warn(
                "A single non-Tag item was passed into Tag.extend. Use Tag.append instead.",
                UserWarning,
                stacklevel=2,
            )
            if isinstance(tags, str) and not isinstance(tags, PageElement):
                tags = NavigableString(tags)
            tag_list = [tags]
        elif isinstance(tags, Iterable):
            # Moving items around the tree may change their position in
            # the original list. Make a list that won't change.
            tag_list = list(tags)

        results: List[PageElement] = []
        for tag in tag_list:
            results.append(self.append(tag))

        return results

    def clear(self, decompose: bool = False) -> None:
        """Destroy all children of this `Tag` by calling
           `PageElement.extract` on them.

        :param decompose: If this is True, `PageElement.decompose` (a
            more destructive method) will be called instead of
            `PageElement.extract`.
        """
        for element in self.contents[:]:
            if decompose:
                element.decompose()
            else:
                element.extract()

    def smooth(self) -> None:
        """Smooth out the children of this `Tag` by consolidating consecutive
        strings.

        If you perform a lot of operations that modify the tree,
        calling this method afterwards can make pretty-printed output
        look more natural.
        """
        # Mark the first position of every pair of children that need
        # to be consolidated.  Do this rather than making a copy of
        # self.contents, since in most cases very few strings will be
        # affected.
        marked = []
        for i, a in enumerate(self.contents):
            if isinstance(a, Tag):
                # Recursively smooth children.
                a.smooth()
            if i == len(self.contents) - 1:
                # This is the last item in .contents, and it's not a
                # tag. There's no chance it needs any work.
                continue
            b = self.contents[i + 1]
            if (
                isinstance(a, NavigableString)
                and isinstance(b, NavigableString)
                and not isinstance(a, PreformattedString)
                and not isinstance(b, PreformattedString)
            ):
                marked.append(i)

        # Go over the marked positions in reverse order, so that
        # removing items from .contents won't affect the remaining
        # positions.
        for i in reversed(marked):
            a = cast(NavigableString, self.contents[i])
            b = cast(NavigableString, self.contents[i + 1])
            b.extract()
            n = NavigableString(a + b)
            a.replace_with(n)

    def index(self, element: PageElement) -> int:
        """Find the index of a child of this `Tag` (by identity, not value).

        Doing this by identity avoids issues when a `Tag` contains two
        children that have string equality.

        :param element: Look for this `PageElement` in this object's contents.
        """
        for i, child in enumerate(self.contents):
            if child is element:
                return i
        raise ValueError("Tag.index: element not in tag")

    def get(
        self, key: str, default: Optional[_AttributeValue] = None
    ) -> Optional[_AttributeValue]:
        """Returns the value of the 'key' attribute for the tag, or
        the value given for 'default' if it doesn't have that
        attribute.

        :param key: The attribute to look for.
        :param default: Use this value if the attribute is not present
            on this `Tag`.
        """
        return self.attrs.get(key, default)

    def get_attribute_list(
        self, key: str, default: Optional[AttributeValueList] = None
    ) -> AttributeValueList:
        """The same as get(), but always returns a (possibly empty) list.

        :param key: The attribute to look for.
        :param default: Use this value if the attribute is not present
            on this `Tag`.
        :return: A list of strings, usually empty or containing only a single
            value.
        """
        list_value: AttributeValueList
        value = self.get(key, default)
        if value is None:
            list_value = self.attribute_value_list_class()
        elif isinstance(value, list):
            list_value = value
        else:
            if not isinstance(value, str):
                value = cast(str, value)
            list_value = self.attribute_value_list_class([value])
        return list_value

    def has_attr(self, key: str) -> bool:
        """Does this `Tag` have an attribute with the given name?"""
        return key in self.attrs

    def __hash__(self) -> int:
        return str(self).__hash__()

    def __getitem__(self, key: str) -> _AttributeValue:
        """tag[key] returns the value of the 'key' attribute for the Tag,
        and throws an exception if it's not there."""
        return self.attrs[key]

    def __iter__(self) -> Iterator[PageElement]:
        "Iterating over a Tag iterates over its contents."
        return iter(self.contents)

    def __len__(self) -> int:
        "The length of a Tag is the length of its list of contents."
        return len(self.contents)

    def __contains__(self, x: Any) -> bool:
        return x in self.contents

    def __bool__(self) -> bool:
        "A tag is non-None even if it has no contents."
        return True

    def __setitem__(self, key: str, value: _AttributeValue) -> None:
        """Setting tag[key] sets the value of the 'key' attribute for the
        tag."""
        self.attrs[key] = value

    def __delitem__(self, key: str) -> None:
        "Deleting tag[key] deletes all 'key' attributes for the tag."
        self.attrs.pop(key, None)

    def __call__(
        self,
        name: Optional[_StrainableElement] = None,
        attrs: _StrainableAttributes = {},
        recursive: bool = True,
        string: Optional[_StrainableString] = None,
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Calling a Tag like a function is the same as calling its
        find_all() method. Eg. tag('a') returns a list of all the A tags
        found within this tag."""
        return self.find_all(
            name, attrs, recursive, string, limit, _stacklevel, **kwargs
        )

    def __getattr__(self, subtag: str) -> Optional[Tag]:
        """Calling tag.subtag is the same as calling tag.find(name="subtag")"""
        # print("Getattr %s.%s" % (self.__class__, tag))
        result: _AtMostOneElement
        if len(subtag) > 3 and subtag.endswith("Tag"):
            # BS3: soup.aTag -> "soup.find("a")
            tag_name = subtag[:-3]
            warnings.warn(
                '.%(name)sTag is deprecated, use .find("%(name)s") instead. If you really were looking for a tag called %(name)sTag, use .find("%(name)sTag")'
                % dict(name=tag_name),
                DeprecationWarning,
                stacklevel=2,
            )
            result = self.find(tag_name)
        # We special case contents to avoid recursion.
        elif not subtag.startswith("__") and not subtag == "contents":
            result = self.find(subtag)
        else:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (self.__class__, subtag)
            )
        return cast(Optional[Tag], result)

    def __eq__(self, other: Any) -> bool:
        """Returns true iff this Tag has the same name, the same attributes,
        and the same contents (recursively) as `other`."""
        if self is other:
            return True
        if not isinstance(other, Tag):
            return False
        if (
            not hasattr(other, "name")
            or not hasattr(other, "attrs")
            or not hasattr(other, "contents")
            or self.name != other.name
            or self.attrs != other.attrs
            or len(self) != len(other)
        ):
            return False
        for i, my_child in enumerate(self.contents):
            if my_child != other.contents[i]:
                return False
        return True

    def __ne__(self, other: Any) -> bool:
        """Returns true iff this Tag is not identical to `other`,
        as defined in __eq__."""
        return not self == other

    def __repr__(self) -> str:
        """Renders this `Tag` as a string."""
        return self.decode()

    __str__ = __unicode__ = __repr__

    def encode(
        self,
        encoding: _Encoding = DEFAULT_OUTPUT_ENCODING,
        indent_level: Optional[int] = None,
        formatter: _FormatterOrName = "minimal",
        errors: str = "xmlcharrefreplace",
    ) -> bytes:
        """Render this `Tag` and its contents as a bytestring.

        :param encoding: The encoding to use when converting to
           a bytestring. This may also affect the text of the document,
           specifically any encoding declarations within the document.
        :param indent_level: Each line of the rendering will be
           indented this many levels. (The ``formatter`` decides what a
           'level' means, in terms of spaces or other characters
           output.) This is used internally in recursive calls while
           pretty-printing.
        :param formatter: Either a `Formatter` object, or a string naming one of
            the standard formatters.
        :param errors: An error handling strategy such as
            'xmlcharrefreplace'. This value is passed along into
            :py:meth:`str.encode` and its value should be one of the `error
            handling constants defined by Python's codecs module
            <https://docs.python.org/3/library/codecs.html#error-handlers>`_.
        """
        # Turn the data structure into Unicode, then encode the
        # Unicode.
        u = self.decode(indent_level, encoding, formatter)
        return u.encode(encoding, errors)

    def decode(
        self,
        indent_level: Optional[int] = None,
        eventual_encoding: _Encoding = DEFAULT_OUTPUT_ENCODING,
        formatter: _FormatterOrName = "minimal",
        iterator: Optional[Iterator[PageElement]] = None,
    ) -> str:
        """Render this `Tag` and its contents as a Unicode string.

        :param indent_level: Each line of the rendering will be
           indented this many levels. (The ``formatter`` decides what a
           'level' means, in terms of spaces or other characters
           output.) This is used internally in recursive calls while
           pretty-printing.
        :param encoding: The encoding you intend to use when
           converting the string to a bytestring. decode() is *not*
           responsible for performing that encoding. This information
           is needed so that a real encoding can be substituted in if
           the document contains an encoding declaration (e.g. in a
           <meta> tag).
        :param formatter: Either a `Formatter` object, or a string
            naming one of the standard formatters.
        :param iterator: The iterator to use when navigating over the
            parse tree. This is only used by `Tag.decode_contents` and
            you probably won't need to use it.
        """
        pieces = []
        # First off, turn a non-Formatter `formatter` into a Formatter
        # object. This will stop the lookup from happening over and
        # over again.
        if not isinstance(formatter, Formatter):
            formatter = self.formatter_for_name(formatter)

        if indent_level is True:
            indent_level = 0

        # The currently active tag that put us into string literal
        # mode. Until this element is closed, children will be treated
        # as string literals and not pretty-printed. String literal
        # mode is turned on immediately after this tag begins, and
        # turned off immediately before it's closed. This means there
        # will be whitespace before and after the tag itself.
        string_literal_tag = None

        for event, element in self._event_stream(iterator):
            if event in (Tag.START_ELEMENT_EVENT, Tag.EMPTY_ELEMENT_EVENT):
                element = cast(Tag, element)
                piece = element._format_tag(eventual_encoding, formatter, opening=True)
            elif event is Tag.END_ELEMENT_EVENT:
                element = cast(Tag, element)
                piece = element._format_tag(eventual_encoding, formatter, opening=False)
                if indent_level is not None:
                    indent_level -= 1
            else:
                element = cast(NavigableString, element)
                piece = element.output_ready(formatter)

            # Now we need to apply the 'prettiness' -- extra
            # whitespace before and/or after this tag. This can get
            # complicated because certain tags, like <pre> and
            # <script>, can't be prettified, since adding whitespace would
            # change the meaning of the content.

            # The default behavior is to add whitespace before and
            # after an element when string literal mode is off, and to
            # leave things as they are when string literal mode is on.
            if string_literal_tag:
                indent_before = indent_after = False
            else:
                indent_before = indent_after = True

            # The only time the behavior is more complex than that is
            # when we encounter an opening or closing tag that might
            # put us into or out of string literal mode.
            if (
                event is Tag.START_ELEMENT_EVENT
                and not string_literal_tag
                and not cast(Tag, element)._should_pretty_print()
            ):
                # We are about to enter string literal mode. Add
                # whitespace before this tag, but not after. We
                # will stay in string literal mode until this tag
                # is closed.
                indent_before = True
                indent_after = False
                string_literal_tag = element
            elif event is Tag.END_ELEMENT_EVENT and element is string_literal_tag:
                # We are about to exit string literal mode by closing
                # the tag that sent us into that mode. Add whitespace
                # after this tag, but not before.
                indent_before = False
                indent_after = True
                string_literal_tag = None

            # Now we know whether to add whitespace before and/or
            # after this element.
            if indent_level is not None:
                if indent_before or indent_after:
                    if isinstance(element, NavigableString):
                        piece = piece.strip()
                    if piece:
                        piece = self._indent_string(
                            piece, indent_level, formatter, indent_before, indent_after
                        )
                if event == Tag.START_ELEMENT_EVENT:
                    indent_level += 1
            pieces.append(piece)
        return "".join(pieces)

    class _TreeTraversalEvent(object):
        """An internal class representing an event in the process
        of traversing a parse tree.

        :meta private:
        """

    # Stand-ins for the different events yielded by _event_stream
    START_ELEMENT_EVENT = _TreeTraversalEvent()  #: :meta private:
    END_ELEMENT_EVENT = _TreeTraversalEvent()  #: :meta private:
    EMPTY_ELEMENT_EVENT = _TreeTraversalEvent()  #: :meta private:
    STRING_ELEMENT_EVENT = _TreeTraversalEvent()  #: :meta private:

    def _event_stream(
        self, iterator: Optional[Iterator[PageElement]] = None
    ) -> Iterator[Tuple[_TreeTraversalEvent, PageElement]]:
        """Yield a sequence of events that can be used to reconstruct the DOM
        for this element.

        This lets us recreate the nested structure of this element
        (e.g. when formatting it as a string) without using recursive
        method calls.

        This is similar in concept to the SAX API, but it's a simpler
        interface designed for internal use. The events are different
        from SAX and the arguments associated with the events are Tags
        and other Beautiful Soup objects.

        :param iterator: An alternate iterator to use when traversing
         the tree.
        """
        tag_stack: List[Tag] = []

        iterator = iterator or self.self_and_descendants

        for c in iterator:
            # If the parent of the element we're about to yield is not
            # the tag currently on the stack, it means that the tag on
            # the stack closed before this element appeared.
            while tag_stack and c.parent != tag_stack[-1]:
                now_closed_tag = tag_stack.pop()
                yield Tag.END_ELEMENT_EVENT, now_closed_tag

            if isinstance(c, Tag):
                if c.is_empty_element:
                    yield Tag.EMPTY_ELEMENT_EVENT, c
                else:
                    yield Tag.START_ELEMENT_EVENT, c
                    tag_stack.append(c)
                    continue
            else:
                yield Tag.STRING_ELEMENT_EVENT, c

        while tag_stack:
            now_closed_tag = tag_stack.pop()
            yield Tag.END_ELEMENT_EVENT, now_closed_tag

    def _indent_string(
        self,
        s: str,
        indent_level: int,
        formatter: Formatter,
        indent_before: bool,
        indent_after: bool,
    ) -> str:
        """Add indentation whitespace before and/or after a string.

        :param s: The string to amend with whitespace.
        :param indent_level: The indentation level; affects how much
           whitespace goes before the string.
        :param indent_before: Whether or not to add whitespace
           before the string.
        :param indent_after: Whether or not to add whitespace
           (a newline) after the string.
        """
        space_before = ""
        if indent_before and indent_level:
            space_before = formatter.indent * indent_level

        space_after = ""
        if indent_after:
            space_after = "\n"

        return space_before + s + space_after

    def _format_tag(
        self, eventual_encoding: str, formatter: Formatter, opening: bool
    ) -> str:
        if self.hidden:
            # A hidden tag is invisible, although its contents
            # are visible.
            return ""

        # A tag starts with the < character (see below).

        # Then the / character, if this is a closing tag.
        closing_slash = ""
        if not opening:
            closing_slash = "/"

        # Then an optional namespace prefix.
        prefix = ""
        if self.prefix:
            prefix = self.prefix + ":"

        # Then a list of attribute values, if this is an opening tag.
        attribute_string = ""
        if opening:
            attributes = formatter.attributes(self)
            attrs = []
            for key, val in attributes:
                if val is None:
                    decoded = key
                else:
                    if isinstance(val, list) or isinstance(val, tuple):
                        val = " ".join(val)
                    elif not isinstance(val, str):
                        val = str(val)
                    elif (
                        isinstance(val, AttributeValueWithCharsetSubstitution)
                        and eventual_encoding is not None
                    ):
                        val = val.substitute_encoding(eventual_encoding)

                    text = formatter.attribute_value(val)
                    decoded = str(key) + "=" + formatter.quoted_attribute_value(text)
                attrs.append(decoded)
            if attrs:
                attribute_string = " " + " ".join(attrs)

        # Then an optional closing slash (for a void element in an
        # XML document).
        void_element_closing_slash = ""
        if self.is_empty_element:
            void_element_closing_slash = formatter.void_element_close_prefix or ""

        # Put it all together.
        return (
            "<"
            + closing_slash
            + prefix
            + self.name
            + attribute_string
            + void_element_closing_slash
            + ">"
        )

    def _should_pretty_print(self, indent_level: int = 1) -> bool:
        """Should this tag be pretty-printed?

        Most of them should, but some (such as <pre> in HTML
        documents) should not.
        """
        return indent_level is not None and (
            not self.preserve_whitespace_tags
            or self.name not in self.preserve_whitespace_tags
        )

    def prettify(
        self,
        encoding: Optional[_Encoding] = None,
        formatter: _FormatterOrName = "minimal",
    ) -> Union[str, bytes]:
        """Pretty-print this `Tag` as a string or bytestring.

        :param encoding: The encoding of the bytestring, or None if you want Unicode.
        :param formatter: A Formatter object, or a string naming one of
            the standard formatters.
        :return: A string (if no ``encoding`` is provided) or a bytestring
            (otherwise).
        """
        if encoding is None:
            return self.decode(indent_level=0, formatter=formatter)
        else:
            return self.encode(encoding=encoding, indent_level=0, formatter=formatter)

    def decode_contents(
        self,
        indent_level: Optional[int] = None,
        eventual_encoding: _Encoding = DEFAULT_OUTPUT_ENCODING,
        formatter: _FormatterOrName = "minimal",
    ) -> str:
        """Renders the contents of this tag as a Unicode string.

        :param indent_level: Each line of the rendering will be
           indented this many levels. (The formatter decides what a
           'level' means in terms of spaces or other characters
           output.) Used internally in recursive calls while
           pretty-printing.

        :param eventual_encoding: The tag is destined to be
           encoded into this encoding. decode_contents() is *not*
           responsible for performing that encoding. This information
           is needed so that a real encoding can be substituted in if
           the document contains an encoding declaration (e.g. in a
           <meta> tag).

        :param formatter: A `Formatter` object, or a string naming one of
            the standard Formatters.
        """
        return self.decode(
            indent_level, eventual_encoding, formatter, iterator=self.descendants
        )

    def encode_contents(
        self,
        indent_level: Optional[int] = None,
        encoding: _Encoding = DEFAULT_OUTPUT_ENCODING,
        formatter: _FormatterOrName = "minimal",
    ) -> bytes:
        """Renders the contents of this PageElement as a bytestring.

        :param indent_level: Each line of the rendering will be
           indented this many levels. (The ``formatter`` decides what a
           'level' means, in terms of spaces or other characters
           output.) This is used internally in recursive calls while
           pretty-printing.
        :param formatter: Either a `Formatter` object, or a string naming one of
            the standard formatters.
        :param encoding: The bytestring will be in this encoding.
        """
        contents = self.decode_contents(indent_level, encoding, formatter)
        return contents.encode(encoding)

    @_deprecated("encode_contents", "4.0.0")
    def renderContents(
        self,
        encoding: _Encoding = DEFAULT_OUTPUT_ENCODING,
        prettyPrint: bool = False,
        indentLevel: Optional[int] = 0,
    ) -> bytes:
        """Deprecated method for BS3 compatibility.

        :meta private:
        """
        if not prettyPrint:
            indentLevel = None
        return self.encode_contents(indent_level=indentLevel, encoding=encoding)

    # Soup methods

    def find(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        recursive: bool = True,
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ) -> _AtMostOneElement:
        """Look in the children of this PageElement and find the first
        PageElement that matches the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param recursive: If this is True, find() will perform a
            recursive search of this Tag's children. Otherwise,
            only the direct children will be considered.
        :param string: A filter on the `Tag.string` attribute.
        :param limit: Stop looking after finding this many results.
        :kwargs: Additional filters on attribute values.
        """
        r = None
        results = self.find_all(name, attrs, recursive, string, 1, _stacklevel=3, **kwargs)
        if results:
            r = results[0]
        return r

    findChild = _deprecated_function_alias("findChild", "find", "3.0.0")

    def find_all(
        self,
        name: _FindMethodName = None,
        attrs: _StrainableAttributes = {},
        recursive: bool = True,
        string: Optional[_StrainableString] = None,
        limit: Optional[int] = None,
        _stacklevel: int = 2,
        **kwargs: _StrainableAttribute,
    ) -> _QueryResults:
        """Look in the children of this `PageElement` and find all
        `PageElement` objects that match the given criteria.

        All find_* methods take a common set of arguments. See the online
        documentation for detailed explanations.

        :param name: A filter on tag name.
        :param attrs: Additional filters on attribute values.
        :param recursive: If this is True, find_all() will perform a
            recursive search of this PageElement's children. Otherwise,
            only the direct children will be considered.
        :param limit: Stop looking after finding this many results.
        :param _stacklevel: Used internally to improve warning messages.
        :kwargs: Additional filters on attribute values.
        """
        generator = self.descendants
        if not recursive:
            generator = self.children
        return self._find_all(
            name, attrs, string, limit, generator, _stacklevel=_stacklevel + 1, **kwargs
        )

    findAll = _deprecated_function_alias("findAll", "find_all", "4.0.0")
    findChildren = _deprecated_function_alias("findChildren", "find_all", "3.0.0")

    # Generator methods
    @property
    def children(self) -> Iterator[PageElement]:
        """Iterate over all direct children of this `PageElement`."""
        return (x for x in self.contents)

    @property
    def self_and_descendants(self) -> Iterator[PageElement]:
        """Iterate over this `Tag` and its children in a
        breadth-first sequence.
        """
        return self._self_and(self.descendants)

    @property
    def descendants(self) -> Iterator[PageElement]:
        """Iterate over all children of this `Tag` in a
        breadth-first sequence.
        """
        if not len(self.contents):
            return
        # _last_descendant() can't return None here because
        # accept_self is True. Worst case, last_descendant will end up
        # as self.
        last_descendant = cast(PageElement, self._last_descendant(accept_self=True))
        stopNode = last_descendant.next_element
        current: _AtMostOneElement = self.contents[0]
        while current is not stopNode and current is not None:
            successor = current.next_element
            yield current
            current = successor

    # CSS selector code
    def select_one(
        self, selector: str, namespaces: Optional[Dict[str, str]] = None, **kwargs: Any
    ) -> Optional[Tag]:
        """Perform a CSS selection operation on the current element.

        :param selector: A CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
           used in the CSS selector to namespace URIs. By default,
           Beautiful Soup will use the prefixes it encountered while
           parsing the document.

        :param kwargs: Keyword arguments to be passed into Soup Sieve's
           soupsieve.select() method.
        """
        return self.css.select_one(selector, namespaces, **kwargs)

    def select(
        self,
        selector: str,
        namespaces: Optional[Dict[str, str]] = None,
        limit: int = 0,
        **kwargs: Any,
    ) -> ResultSet[Tag]:
        """Perform a CSS selection operation on the current element.

        This uses the SoupSieve library.

        :param selector: A string containing a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
           used in the CSS selector to namespace URIs. By default,
           Beautiful Soup will use the prefixes it encountered while
           parsing the document.

        :param limit: After finding this number of results, stop looking.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
           soupsieve.select() method.
        """
        return self.css.select(selector, namespaces, limit, **kwargs)

    @property
    def css(self) -> CSS:
        """Return an interface to the CSS selector API."""
        return CSS(self)

    # Old names for backwards compatibility
    @_deprecated("children", "4.0.0")
    def childGenerator(self) -> Iterator[PageElement]:
        """Deprecated generator.

        :meta private:
        """
        return self.children

    @_deprecated("descendants", "4.0.0")
    def recursiveChildGenerator(self) -> Iterator[PageElement]:
        """Deprecated generator.

        :meta private:
        """
        return self.descendants

    @_deprecated("has_attr", "4.0.0")
    def has_key(self, key: str) -> bool:
        """Deprecated method. This was kind of misleading because has_key()
        (attributes) was different from __in__ (contents).

        has_key() is gone in Python 3, anyway.

        :meta private:
        """
        return self.has_attr(key)


_PageElementT = TypeVar("_PageElementT", bound=PageElement)


class ResultSet(List[_PageElementT], Generic[_PageElementT]):
    """A ResultSet is a list of `PageElement` objects, gathered as the result
    of matching an :py:class:`ElementFilter` against a parse tree. Basically, a list of
    search results.
    """

    source: Optional[ElementFilter]

    def __init__(
        self, source: Optional[ElementFilter], result: Iterable[_PageElementT] = ()
    ) -> None:
        super(ResultSet, self).__init__(result)
        self.source = source

    def __getattr__(self, key: str) -> None:
        """Raise a helpful exception to explain a common code fix."""
        raise AttributeError(
            f"""ResultSet object has no attribute "{key}". You're probably treating a list of elements like a single element. Did you call find_all() when you meant to call find()?"""
        )


# Now that all the classes used by SoupStrainer have been defined,
# import SoupStrainer itself into this module to preserve the
# backwards compatibility of anyone who imports
# bs4.element.SoupStrainer.
from bs4.filter import SoupStrainer # noqa: E402

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\polygon.py ===
from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray
from .point import Point
from sympy.logic import And
from sympy.matrices import Matrix
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import has_dups, has_variety, uniq, rotate_left, least_rotation
from sympy.utilities.misc import as_int, func_name

from mpmath.libmp.libmpf import prec_to_dps

import warnings


x, y, T = [Dummy('polygon_dummy', real=True) for i in range(3)]


class Polygon(GeometrySet):
    """A two-dimensional polygon.

    A simple polygon in space. Can be constructed from a sequence of points
    or from a center, radius, number of sides and rotation angle.

    Parameters
    ==========

    vertices
        A sequence of points.

    n : int, optional
        If $> 0$, an n-sided RegularPolygon is created.
        Default value is $0$.

    Attributes
    ==========

    area
    angles
    perimeter
    vertices
    centroid
    sides

    Raises
    ======

    GeometryError
        If all parameters are not Points.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment, Triangle

    Notes
    =====

    Polygons are treated as closed paths rather than 2D areas so
    some calculations can be be negative or positive (e.g., area)
    based on the orientation of the points.

    Any consecutive identical points are reduced to a single point
    and any points collinear and between two points will be removed
    unless they are needed to define an explicit intersection (see examples).

    A Triangle, Segment or Point will be returned when there are 3 or
    fewer points provided.

    Examples
    ========

    >>> from sympy import Polygon, pi
    >>> p1, p2, p3, p4, p5 = [(0, 0), (1, 0), (5, 1), (0, 1), (3, 0)]
    >>> Polygon(p1, p2, p3, p4)
    Polygon(Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1))
    >>> Polygon(p1, p2)
    Segment2D(Point2D(0, 0), Point2D(1, 0))
    >>> Polygon(p1, p2, p5)
    Segment2D(Point2D(0, 0), Point2D(3, 0))

    The area of a polygon is calculated as positive when vertices are
    traversed in a ccw direction. When the sides of a polygon cross the
    area will have positive and negative contributions. The following
    defines a Z shape where the bottom right connects back to the top
    left.

    >>> Polygon((0, 2), (2, 2), (0, 0), (2, 0)).area
    0

    When the keyword `n` is used to define the number of sides of the
    Polygon then a RegularPolygon is created and the other arguments are
    interpreted as center, radius and rotation. The unrotated RegularPolygon
    will always have a vertex at Point(r, 0) where `r` is the radius of the
    circle that circumscribes the RegularPolygon. Its method `spin` can be
    used to increment that angle.

    >>> p = Polygon((0,0), 1, n=3)
    >>> p
    RegularPolygon(Point2D(0, 0), 1, 3, 0)
    >>> p.vertices[0]
    Point2D(1, 0)
    >>> p.args[0]
    Point2D(0, 0)
    >>> p.spin(pi/2)
    >>> p.vertices[0]
    Point2D(0, 1)

    """

    __slots__ = ()

    def __new__(cls, *args, n = 0, **kwargs):
        if n:
            args = list(args)
            # return a virtual polygon with n sides
            if len(args) == 2:  # center, radius
                args.append(n)
            elif len(args) == 3:  # center, radius, rotation
                args.insert(2, n)
            return RegularPolygon(*args, **kwargs)

        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # remove consecutive duplicates
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = nodup[i], nodup[i + 1], nodup[i + 2]
            if Point.is_collinear(a, b, c):
                nodup.pop(i + 1)
                if a == c:
                    nodup.pop(i)
            else:
                i += 1

        vertices = list(nodup)

        if len(vertices) > 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 3:
            return Triangle(*vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def area(self):
        """
        The area of the polygon.

        Notes
        =====

        The area calculation can be positive or negative based on the
        orientation of the points. If any side of the polygon crosses
        any other side, there will be areas having opposite signs.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.area

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.area
        3

        In the Z shaped polygon (with the lower right connecting back
        to the upper left) the areas cancel out:

        >>> Z = Polygon((0, 1), (1, 1), (0, 0), (1, 0))
        >>> Z.area
        0

        In the M shaped polygon, areas do not cancel because no side
        crosses any other (though there is a point of contact).

        >>> M = Polygon((0, 0), (0, 1), (2, 0), (3, 1), (3, 0))
        >>> M.area
        -3/2

        """
        area = 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            area += x1*y2 - x2*y1
        return simplify(area) / 2

    @staticmethod
    def _is_clockwise(a, b, c):
        """Return True/False for cw/ccw orientation.

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b, c = [Point(i) for i in [(0, 0), (1, 1), (1, 0)]]
        >>> Polygon._is_clockwise(a, b, c)
        True
        >>> Polygon._is_clockwise(a, c, b)
        False
        """
        ba = b - a
        ca = c - a
        t_area = simplify(ba.x*ca.y - ca.x*ba.y)
        res = t_area.is_nonpositive
        if res is None:
            raise ValueError("Can't determine orientation")
        return res

    @property
    def angles(self):
        """The internal angle at each vertex.

        Returns
        =======

        angles : dict
            A dictionary where each key is a vertex and each value is the
            internal angle at that vertex. The vertices are represented as
            Points.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.angles[p1]
        pi/2
        >>> poly.angles[p2]
        acos(-4*sqrt(17)/17)

        """

        args = self.vertices
        n = len(args)
        ret = {}
        for i in range(n):
            a, b, c = args[i - 2], args[i - 1], args[i]
            reflex_ang = Ray(b, a).angle_between(Ray(b, c))
            if self._is_clockwise(a, b, c):
                ret[b] = 2*S.Pi - reflex_ang
            else:
                ret[b] = reflex_ang

        # internal sum should be pi*(n - 2), not pi*(n+2)
        # so if ratio is (n+2)/(n-2) > 1 it is wrong
        wrong = ((sum(ret.values())/S.Pi-1)/(n - 2) - 1).is_positive
        if wrong:
            two_pi = 2*S.Pi
            for b in ret:
                ret[b] = two_pi - ret[b]
        elif wrong is None:
            raise ValueError("could not determine Polygon orientation.")
        return ret

    @property
    def ambient_dimension(self):
        return self.vertices[0].ambient_dimension

    @property
    def perimeter(self):
        """The perimeter of the polygon.

        Returns
        =======

        perimeter : number or Basic instance

        See Also
        ========

        sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.perimeter
        sqrt(17) + 7
        """
        p = 0
        args = self.vertices
        for i in range(len(args)):
            p += args[i - 1].distance(args[i])
        return simplify(p)

    @property
    def vertices(self):
        """The vertices of the polygon.

        Returns
        =======

        vertices : list of Points

        Notes
        =====

        When iterating over the vertices, it is more efficient to index self
        rather than to request the vertices and index them. Only use the
        vertices when you want to process all of them at once. This is even
        more important with RegularPolygons that calculate each vertex.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.vertices
        [Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1)]
        >>> poly.vertices[0]
        Point2D(0, 0)

        """
        return list(self.args)

    @property
    def centroid(self):
        """The centroid of the polygon.

        Returns
        =======

        centroid : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.util.centroid

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.centroid
        Point2D(31/18, 11/18)

        """
        A = 1/(6*self.area)
        cx, cy = 0, 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            cx += v*(x1 + x2)
            cy += v*(y1 + y2)
        return Point(simplify(A*cx), simplify(A*cy))


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment of area of a two dimensional polygon.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the polygon.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
                           I_xx, I_yy are second moment of area of a two dimensional polygon.
                           I_xy is product moment of area of a two dimensional polygon.

        Examples
        ========

        >>> from sympy import Polygon, symbols
        >>> a, b = symbols('a, b')
        >>> p1, p2, p3, p4, p5 = [(0, 0), (a, 0), (a, b), (0, b), (a/3, b/3)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> rectangle.second_moment_of_area()
        (a*b**3/12, a**3*b/12, 0)
        >>> rectangle.second_moment_of_area(p5)
        (a*b**3/9, a**3*b/9, a**2*b**2/36)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Second_moment_of_area

        """

        I_xx, I_yy, I_xy = 0, 0, 0
        args = self.vertices
        for i in range(len(args)):
            x1, y1 = args[i-1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            I_xx += (y1**2 + y1*y2 + y2**2)*v
            I_yy += (x1**2 + x1*x2 + x2**2)*v
            I_xy += (x1*y2 + 2*x1*y1 + 2*x2*y2 + x2*y1)*v
        A = self.area
        c_x = self.centroid[0]
        c_y = self.centroid[1]
        # parallel axis theorem
        I_xx_c = (I_xx/12) - (A*(c_y**2))
        I_yy_c = (I_yy/12) - (A*(c_x**2))
        I_xy_c = (I_xy/24) - (A*(c_x*c_y))
        if point is None:
            return I_xx_c, I_yy_c, I_xy_c

        I_xx = (I_xx_c + A*((point[1]-c_y)**2))
        I_yy = (I_yy_c + A*((point[0]-c_x)**2))
        I_xy = (I_xy_c + A*((point[0]-c_x)*(point[1]-c_y)))

        return I_xx, I_yy, I_xy


    def first_moment_of_area(self, point=None):
        """
        Returns the first moment of area of a two-dimensional polygon with
        respect to a certain point of interest.

        First moment of area is a measure of the distribution of the area
        of a polygon in relation to an axis. The first moment of area of
        the entire polygon about its own centroid is always zero. Therefore,
        here it is calculated for an area, above or below a certain point
        of interest, that makes up a smaller portion of the polygon. This
        area is bounded by the point of interest and the extreme end
        (top or bottom) of the polygon. The first moment for this area is
        is then determined about the centroidal axis of the initial polygon.

        References
        ==========

        .. [1] https://skyciv.com/docs/tutorials/section-tutorials/calculating-the-statical-or-first-moment-of-area-of-beam-sections/?cc=BMD
        .. [2] https://mechanicalc.com/reference/cross-sections

        Parameters
        ==========

        point: Point, two-tuple of sympifyable objects, or None (default=None)
            point is the point above or below which the area of interest lies
            If ``point=None`` then the centroid acts as the point of interest.

        Returns
        =======

        Q_x, Q_y: number or SymPy expressions
            Q_x is the first moment of area about the x-axis
            Q_y is the first moment of area about the y-axis
            A negative sign indicates that the section modulus is
            determined for a section below (or left of) the centroidal axis

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b = 50, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> p = Polygon(p1, p2, p3, p4)
        >>> p.first_moment_of_area()
        (625, 3125)
        >>> p.first_moment_of_area(point=Point(30, 7))
        (525, 3000)
        """
        if point:
            xc, yc = self.centroid
        else:
            point = self.centroid
            xc, yc = point

        h_line = Line(point, slope=0)
        v_line = Line(point, slope=S.Infinity)

        h_poly = self.cut_section(h_line)
        v_poly = self.cut_section(v_line)

        poly_1 = h_poly[0] if h_poly[0].area <= h_poly[1].area else h_poly[1]
        poly_2 = v_poly[0] if v_poly[0].area <= v_poly[1].area else v_poly[1]

        Q_x = (poly_1.centroid.y - yc)*poly_1.area
        Q_y = (poly_2.centroid.x - xc)*poly_2.area

        return Q_x, Q_y


    def polar_second_moment_of_area(self):
        """Returns the polar modulus of a two-dimensional polygon

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import Polygon, symbols
        >>> a, b = symbols('a, b')
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.polar_second_moment_of_area()
        a**3*b/12 + a*b**3/12

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]


    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of a two-dimensional
        polygon.

        Section modulus is a geometric property of a polygon defined as the
        ratio of second moment of area to the distance of the extreme end of
        the polygon from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" it will be calculated for the point farthest from the
            centroidal axis of the polygon.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis

        Examples
        ========

        >>> from sympy import symbols, Polygon, Point
        >>> a, b = symbols('a, b', positive=True)
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.section_modulus()
        (a*b**2/6, a**2*b/6)
        >>> rectangle.section_modulus(Point(a/4, b/4))
        (-a*b**2/3, -a**2*b/3)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.centroid
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the centroid
            y = point.y - y_c
            x = point.x - x_c

        second_moment= self.second_moment_of_area()
        S_x = second_moment[0]/y
        S_y = second_moment[1]/x

        return S_x, S_y


    @property
    def sides(self):
        """The directed line segments that form the sides of the polygon.

        Returns
        =======

        sides : list of sides
            Each side is a directed Segment.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.sides
        [Segment2D(Point2D(0, 0), Point2D(1, 0)),
        Segment2D(Point2D(1, 0), Point2D(5, 1)),
        Segment2D(Point2D(5, 1), Point2D(0, 1)), Segment2D(Point2D(0, 1), Point2D(0, 0))]

        """
        res = []
        args = self.vertices
        for i in range(-len(args), 0):
            res.append(Segment(args[i], args[i + 1]))
        return res

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        verts = self.vertices
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    def is_convex(self):
        """Is the polygon convex?

        A polygon is convex if all its interior angles are less than 180
        degrees and there are no intersections between sides.

        Returns
        =======

        is_convex : boolean
            True if this polygon is convex, False otherwise.

        See Also
        ========

        sympy.geometry.util.convex_hull

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.is_convex()
        True

        """
        # Determine orientation of points
        args = self.vertices
        cw = self._is_clockwise(args[-2], args[-1], args[0])
        for i in range(1, len(args)):
            if cw ^ self._is_clockwise(args[i - 2], args[i - 1], args[i]):
                return False
        # check for intersecting sides
        sides = self.sides
        for i, si in enumerate(sides):
            pts = si.args
            # exclude the sides connected to si
            for j in range(1 if i == len(sides) - 1 else 0, i - 1):
                sj = sides[j]
                if sj.p1 not in pts and sj.p2 not in pts:
                    hit = si.intersection(sj)
                    if hit:
                        return False
        return True

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon((0, 0), (4, 0), (4, 4))
        >>> p.encloses_point(Point(2, 1))
        True
        >>> p.encloses_point(Point(2, 2))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        References
        ==========

        .. [1] https://paulbourke.net/geometry/polygonmesh/#insidepoly

        """
        p = Point(p, dim=2)
        if p in self.vertices or any(p in s for s in self.sides):
            return False

        # move to p, checking that the result is numeric
        lit = []
        for v in self.vertices:
            lit.append(v - p)  # the difference is simplified
            if lit[-1].free_symbols:
                return None

        poly = Polygon(*lit)

        # polygon closure is assumed in the following test but Polygon removes duplicate pts so
        # the last point has to be added so all sides are computed. Using Polygon.sides is
        # not good since Segments are unordered.
        args = poly.args
        indices = list(range(-len(args), 1))

        if poly.is_convex():
            orientation = None
            for i in indices:
                a = args[i]
                b = args[i + 1]
                test = ((-a.y)*(b.x - a.x) - (-a.x)*(b.y - a.y)).is_negative
                if orientation is None:
                    orientation = test
                elif test is not orientation:
                    return False
            return True

        hit_odd = False
        p1x, p1y = args[0].args
        for i in indices[1:]:
            p2x, p2y = args[i].args
            if 0 > min(p1y, p2y):
                if 0 <= max(p1y, p2y):
                    if 0 <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (-p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                            if p1x == p2x or 0 <= xinters:
                                hit_odd = not hit_odd
            p1x, p1y = p2x, p2y
        return hit_odd

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the polygon.

        The parameter, varying from 0 to 1, assigns points to the position on
        the perimeter that is that fraction of the total perimeter. So the
        point evaluated at t=1/2 would return the point from the first vertex
        that is 1/2 way around the polygon.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        arbitrary_point : Point

        Raises
        ======

        ValueError
            When `parameter` already appears in the Polygon's definition.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Polygon, Symbol
        >>> t = Symbol('t', real=True)
        >>> tri = Polygon((0, 0), (1, 0), (1, 1))
        >>> p = tri.arbitrary_point('t')
        >>> perimeter = tri.perimeter
        >>> s1, s2 = [s.length for s in tri.sides[:2]]
        >>> p.subs(t, (s1 + s2/2)/perimeter)
        Point2D(1, 1/2)

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError('Symbol %s already appears in object and cannot be used as a parameter.' % t.name)
        sides = []
        perimeter = self.perimeter
        perim_fraction_start = 0
        for s in self.sides:
            side_perim_fraction = s.length/perimeter
            perim_fraction_end = perim_fraction_start + side_perim_fraction
            pt = s.arbitrary_point(parameter).subs(
                t, (t - perim_fraction_start)/side_perim_fraction)
            sides.append(
                (pt, (And(perim_fraction_start <= t, t < perim_fraction_end))))
            perim_fraction_start = perim_fraction_end
        return Piecewise(*sides)

    def parameter_value(self, other, t):
        if not isinstance(other,GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other,Point):
            raise ValueError("other must be a point")
        if other.free_symbols:
            raise NotImplementedError('non-numeric coordinates')
        unknown = False
        p = self.arbitrary_point(T)
        for pt, cond in p.args:
            sol = solve(pt - other, T, dict=True)
            if not sol:
                continue
            value = sol[0][T]
            if simplify(cond.subs(T, value)) == True:
                return {t: value}
            unknown = True
        if unknown:
            raise ValueError("Given point may not be on %s" % func_name(self))
        raise ValueError("Given point is not on %s" % func_name(self))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the polygon.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list (plot interval)
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Polygon
        >>> p = Polygon((0, 0), (1, 0), (1, 1))
        >>> p.plot_interval()
        [t, 0, 1]

        """
        t = Symbol(parameter, real=True)
        return [t, 0, 1]

    def intersection(self, o):
        """The intersection of polygon and geometry entity.

        The intersection may be empty and can contain individual Points and
        complete Line Segments.

        Parameters
        ==========

        other: GeometryEntity

        Returns
        =======

        intersection : list
            The list of Segments and Points

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon, Line
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly1 = Polygon(p1, p2, p3, p4)
        >>> p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
        >>> poly2 = Polygon(p5, p6, p7)
        >>> poly1.intersection(poly2)
        [Point2D(1/3, 1), Point2D(2/3, 0), Point2D(9/5, 1/5), Point2D(7/3, 1)]
        >>> poly1.intersection(Line(p1, p2))
        [Segment2D(Point2D(0, 0), Point2D(1, 0))]
        >>> poly1.intersection(p1)
        [Point2D(0, 0)]
        """
        intersection_result = []
        k = o.sides if isinstance(o, Polygon) else [o]
        for side in self.sides:
            for side1 in k:
                intersection_result.extend(side.intersection(side1))

        intersection_result = list(uniq(intersection_result))
        points = [entity for entity in intersection_result if isinstance(entity, Point)]
        segments = [entity for entity in intersection_result if isinstance(entity, Segment)]

        if points and segments:
            points_in_segments = list(uniq([point for point in points for segment in segments if point in segment]))
            if points_in_segments:
                for i in points_in_segments:
                    points.remove(i)
            return list(ordered(segments + points))
        else:
            return list(ordered(intersection_result))


    def cut_section(self, line):
        """
        Returns a tuple of two polygon segments that lie above and below
        the intersecting line respectively.

        Parameters
        ==========

        line: Line object of geometry module
            line which cuts the Polygon. The part of the Polygon that lies
            above and below this line is returned.

        Returns
        =======

        upper_polygon, lower_polygon: Polygon objects or None
            upper_polygon is the polygon that lies above the given line.
            lower_polygon is the polygon that lies below the given line.
            upper_polygon and lower polygon are ``None`` when no polygon
            exists above the line or below the line.

        Raises
        ======

        ValueError: When the line does not intersect the polygon

        Examples
        ========

        >>> from sympy import Polygon, Line
        >>> a, b = 20, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> t = rectangle.cut_section(Line((0, 5), slope=0))
        >>> t
        (Polygon(Point2D(0, 10), Point2D(0, 5), Point2D(20, 5), Point2D(20, 10)),
        Polygon(Point2D(0, 5), Point2D(0, 0), Point2D(20, 0), Point2D(20, 5)))
        >>> upper_segment, lower_segment = t
        >>> upper_segment.area
        100
        >>> upper_segment.centroid
        Point2D(10, 15/2)
        >>> lower_segment.centroid
        Point2D(10, 5/2)

        References
        ==========

        .. [1] https://github.com/sympy/sympy/wiki/A-method-to-return-a-cut-section-of-any-polygon-geometry

        """
        intersection_points = self.intersection(line)
        if not intersection_points:
            raise ValueError("This line does not intersect the polygon")

        points = list(self.vertices)
        points.append(points[0])

        eq = line.equation(x, y)

        # considering equation of line to be `ax +by + c`
        a = eq.coeff(x)
        b = eq.coeff(y)

        upper_vertices = []
        lower_vertices = []
        # prev is true when previous point is above the line
        prev = True
        prev_point = None
        for point in points:
            # when coefficient of y is 0, right side of the line is
            # considered
            compare = eq.subs({x: point.x, y: point.y})/b if b \
                    else eq.subs(x, point.x)/a

            # if point lies above line
            if compare > 0:
                if not prev:
                    # if previous point lies below the line, the intersection
                    # point of the polygon edge and the line has to be included
                    edge = Line(point, prev_point)
                    new_point = edge.intersection(line)
                    upper_vertices.append(new_point[0])
                    lower_vertices.append(new_point[0])

                upper_vertices.append(point)
                prev = True
            else:
                if prev and prev_point:
                    edge = Line(point, prev_point)
                    new_point = edge.intersection(line)
                    upper_vertices.append(new_point[0])
                    lower_vertices.append(new_point[0])
                lower_vertices.append(point)
                prev = False
            prev_point = point

        upper_polygon, lower_polygon = None, None
        if upper_vertices and isinstance(Polygon(*upper_vertices), Polygon):
            upper_polygon = Polygon(*upper_vertices)
        if lower_vertices and isinstance(Polygon(*lower_vertices), Polygon):
            lower_polygon = Polygon(*lower_vertices)

        return upper_polygon, lower_polygon


    def distance(self, o):
        """
        Returns the shortest distance between self and o.

        If o is a point, then self does not need to be convex.
        If o is another polygon self and o must be convex.

        Examples
        ========

        >>> from sympy import Point, Polygon, RegularPolygon
        >>> p1, p2 = map(Point, [(0, 0), (7, 5)])
        >>> poly = Polygon(*RegularPolygon(p1, 1, 3).vertices)
        >>> poly.distance(p2)
        sqrt(61)
        """
        if isinstance(o, Point):
            dist = oo
            for side in self.sides:
                current = side.distance(o)
                if current == 0:
                    return S.Zero
                elif current < dist:
                    dist = current
            return dist
        elif isinstance(o, Polygon) and self.is_convex() and o.is_convex():
            return self._do_poly_distance(o)
        raise NotImplementedError()

    def _do_poly_distance(self, e2):
        """
        Calculates the least distance between the exteriors of two
        convex polygons e1 and e2. Does not check for the convexity
        of the polygons as this is checked by Polygon.distance.

        Notes
        =====

            - Prints a warning if the two polygons possibly intersect as the return
              value will not be valid in such a case. For a more through test of
              intersection use intersection().

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> square = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
        >>> triangle = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
        >>> square._do_poly_distance(triangle)
        sqrt(2)/2

        Description of method used
        ==========================

        Method:
        [1] https://web.archive.org/web/20150509035744/http://cgm.cs.mcgill.ca/~orm/mind2p.html
        Uses rotating calipers:
        [2] https://en.wikipedia.org/wiki/Rotating_calipers
        and antipodal points:
        [3] https://en.wikipedia.org/wiki/Antipodal_point
        """
        e1 = self

        '''Tests for a possible intersection between the polygons and outputs a warning'''
        e1_center = e1.centroid
        e2_center = e2.centroid
        e1_max_radius = S.Zero
        e2_max_radius = S.Zero
        for vertex in e1.vertices:
            r = Point.distance(e1_center, vertex)
            if e1_max_radius < r:
                e1_max_radius = r
        for vertex in e2.vertices:
            r = Point.distance(e2_center, vertex)
            if e2_max_radius < r:
                e2_max_radius = r
        center_dist = Point.distance(e1_center, e2_center)
        if center_dist <= e1_max_radius + e2_max_radius:
            warnings.warn("Polygons may intersect producing erroneous output",
                          stacklevel=3)

        '''
        Find the upper rightmost vertex of e1 and the lowest leftmost vertex of e2
        '''
        e1_ymax = Point(0, -oo)
        e2_ymin = Point(0, oo)

        for vertex in e1.vertices:
            if vertex.y > e1_ymax.y or (vertex.y == e1_ymax.y and vertex.x > e1_ymax.x):
                e1_ymax = vertex
        for vertex in e2.vertices:
            if vertex.y < e2_ymin.y or (vertex.y == e2_ymin.y and vertex.x < e2_ymin.x):
                e2_ymin = vertex
        min_dist = Point.distance(e1_ymax, e2_ymin)

        '''
        Produce a dictionary with vertices of e1 as the keys and, for each vertex, the points
        to which the vertex is connected as its value. The same is then done for e2.
        '''
        e1_connections = {}
        e2_connections = {}

        for side in e1.sides:
            if side.p1 in e1_connections:
                e1_connections[side.p1].append(side.p2)
            else:
                e1_connections[side.p1] = [side.p2]

            if side.p2 in e1_connections:
                e1_connections[side.p2].append(side.p1)
            else:
                e1_connections[side.p2] = [side.p1]

        for side in e2.sides:
            if side.p1 in e2_connections:
                e2_connections[side.p1].append(side.p2)
            else:
                e2_connections[side.p1] = [side.p2]

            if side.p2 in e2_connections:
                e2_connections[side.p2].append(side.p1)
            else:
                e2_connections[side.p2] = [side.p1]

        e1_current = e1_ymax
        e2_current = e2_ymin
        support_line = Line(Point(S.Zero, S.Zero), Point(S.One, S.Zero))

        '''
        Determine which point in e1 and e2 will be selected after e2_ymin and e1_ymax,
        this information combined with the above produced dictionaries determines the
        path that will be taken around the polygons
        '''
        point1 = e1_connections[e1_ymax][0]
        point2 = e1_connections[e1_ymax][1]
        angle1 = support_line.angle_between(Line(e1_ymax, point1))
        angle2 = support_line.angle_between(Line(e1_ymax, point2))
        if angle1 < angle2:
            e1_next = point1
        elif angle2 < angle1:
            e1_next = point2
        elif Point.distance(e1_ymax, point1) > Point.distance(e1_ymax, point2):
            e1_next = point2
        else:
            e1_next = point1

        point1 = e2_connections[e2_ymin][0]
        point2 = e2_connections[e2_ymin][1]
        angle1 = support_line.angle_between(Line(e2_ymin, point1))
        angle2 = support_line.angle_between(Line(e2_ymin, point2))
        if angle1 > angle2:
            e2_next = point1
        elif angle2 > angle1:
            e2_next = point2
        elif Point.distance(e2_ymin, point1) > Point.distance(e2_ymin, point2):
            e2_next = point2
        else:
            e2_next = point1

        '''
        Loop which determines the distance between anti-podal pairs and updates the
        minimum distance accordingly. It repeats until it reaches the starting position.
        '''
        while True:
            e1_angle = support_line.angle_between(Line(e1_current, e1_next))
            e2_angle = pi - support_line.angle_between(Line(
                e2_current, e2_next))

            if (e1_angle < e2_angle) is True:
                support_line = Line(e1_current, e1_next)
                e1_segment = Segment(e1_current, e1_next)
                min_dist_current = e1_segment.distance(e2_current)

                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e1_connections[e1_next][0] != e1_current:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][0]
                else:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][1]
            elif (e1_angle > e2_angle) is True:
                support_line = Line(e2_next, e2_current)
                e2_segment = Segment(e2_current, e2_next)
                min_dist_current = e2_segment.distance(e1_current)

                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e2_connections[e2_next][0] != e2_current:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][0]
                else:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][1]
            else:
                support_line = Line(e1_current, e1_next)
                e1_segment = Segment(e1_current, e1_next)
                e2_segment = Segment(e2_current, e2_next)
                min1 = e1_segment.distance(e2_next)
                min2 = e2_segment.distance(e1_next)

                min_dist_current = min(min1, min2)
                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e1_connections[e1_next][0] != e1_current:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][0]
                else:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][1]

                if e2_connections[e2_next][0] != e2_current:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][0]
                else:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][1]
            if e1_current == e1_ymax and e2_current == e2_ymin:
                break
        return min_dist

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the Polygon.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = map(N, self.vertices)
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color)

    def _hashable_content(self):

        D = {}
        def ref_list(point_list):
            kee = {}
            for i, p in enumerate(ordered(set(point_list))):
                kee[p] = i
                D[i] = p
            return [kee[p] for p in point_list]

        S1 = ref_list(self.args)
        r_nor = rotate_left(S1, least_rotation(S1))
        S2 = ref_list(list(reversed(self.args)))
        r_rev = rotate_left(S2, least_rotation(S2))
        if r_nor < r_rev:
            r = r_nor
        else:
            r = r_rev
        canonical_args = [ D[order] for order in r ]
        return tuple(canonical_args)

    def __contains__(self, o):
        """
        Return True if o is contained within the boundary lines of self.altitudes

        Parameters
        ==========

        other : GeometryEntity

        Returns
        =======

        contained in : bool
            The points (and sides, if applicable) are contained in self.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.encloses

        Examples
        ========

        >>> from sympy import Line, Segment, Point
        >>> p = Point(0, 0)
        >>> q = Point(1, 1)
        >>> s = Segment(p, q*2)
        >>> l = Line(p, q)
        >>> p in q
        False
        >>> p in s
        True
        >>> q*3 in s
        False
        >>> s in l
        True

        """

        if isinstance(o, Polygon):
            return self == o
        elif isinstance(o, Segment):
            return any(o in s for s in self.sides)
        elif isinstance(o, Point):
            if o in self.vertices:
                return True
            for side in self.sides:
                if o in side:
                    return True

        return False

    def bisectors(p, prec=None):
        """Returns angle bisectors of a polygon. If prec is given
        then approximate the point defining the ray to that precision.

        The distance between the points defining the bisector ray is 1.

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
        >>> p.bisectors(2)
        {Point2D(0, 0): Ray2D(Point2D(0, 0), Point2D(0.71, 0.71)),
         Point2D(0, 3): Ray2D(Point2D(0, 3), Point2D(0.23, 2.0)),
         Point2D(1, 1): Ray2D(Point2D(1, 1), Point2D(0.19, 0.42)),
         Point2D(2, 0): Ray2D(Point2D(2, 0), Point2D(1.1, 0.38))}
        """
        b = {}
        pts = list(p.args)
        pts.append(pts[0])  # close it
        cw = Polygon._is_clockwise(*pts[:3])
        if cw:
            pts = list(reversed(pts))
        for v, a in p.angles.items():
            i = pts.index(v)
            p1, p2 = Point._normalize_dimension(pts[i], pts[i + 1])
            ray = Ray(p1, p2).rotate(a/2, v)
            dir = ray.direction
            ray = Ray(ray.p1, ray.p1 + dir/dir.distance((0, 0)))
            if prec is not None:
                ray = Ray(ray.p1, ray.p2.n(prec))
            b[v] = ray
        return b


class RegularPolygon(Polygon):
    """
    A regular polygon.

    Such a polygon has all internal angles equal and all sides the same length.

    Parameters
    ==========

    center : Point
    radius : number or Basic instance
        The distance from the center to a vertex
    n : int
        The number of sides

    Attributes
    ==========

    vertices
    center
    radius
    rotation
    apothem
    interior_angle
    exterior_angle
    circumcircle
    incircle
    angles

    Raises
    ======

    GeometryError
        If the `center` is not a Point, or the `radius` is not a number or Basic
        instance, or the number of sides, `n`, is less than three.

    Notes
    =====

    A RegularPolygon can be instantiated with Polygon with the kwarg n.

    Regular polygons are instantiated with a center, radius, number of sides
    and a rotation angle. Whereas the arguments of a Polygon are vertices, the
    vertices of the RegularPolygon must be obtained with the vertices method.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import RegularPolygon, Point
    >>> r = RegularPolygon(Point(0, 0), 5, 3)
    >>> r
    RegularPolygon(Point2D(0, 0), 5, 3, 0)
    >>> r.vertices[0]
    Point2D(5, 0)

    """

    __slots__ = ('_n', '_center', '_radius', '_rot')

    def __new__(self, c, r, n, rot=0, **kwargs):
        r, n, rot = map(sympify, (r, n, rot))
        c = Point(c, dim=2, **kwargs)
        if not isinstance(r, Expr):
            raise GeometryError("r must be an Expr object, not %s" % r)
        if n.is_Number:
            as_int(n)  # let an error raise if necessary
            if n < 3:
                raise GeometryError("n must be a >= 3, not %s" % n)

        obj = GeometryEntity.__new__(self, c, r, n, **kwargs)
        obj._n = n
        obj._center = c
        obj._radius = r
        obj._rot = rot % (2*S.Pi/n) if rot.is_number else rot
        return obj

    def _eval_evalf(self, prec=15, **options):
        c, r, n, a = self.args
        dps = prec_to_dps(prec)
        c, r, a = [i.evalf(n=dps, **options) for i in (c, r, a)]
        return self.func(c, r, n, a)

    @property
    def args(self):
        """
        Returns the center point, the radius,
        the number of sides, and the orientation angle.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.args
        (Point2D(0, 0), 5, 3, 0)
        """
        return self._center, self._radius, self._n, self._rot

    def __str__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    def __repr__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    @property
    def area(self):
        """Returns the area.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> square = RegularPolygon((0, 0), 1, 4)
        >>> square.area
        2
        >>> _ == square.length**2
        True
        """
        c, r, n, rot = self.args
        return sign(r)*n*self.length**2/(4*tan(pi/n))

    @property
    def length(self):
        """Returns the length of the sides.

        The half-length of the side and the apothem form two legs
        of a right triangle whose hypotenuse is the radius of the
        regular polygon.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> from sympy import sqrt
        >>> s = square_in_unit_circle = RegularPolygon((0, 0), 1, 4)
        >>> s.length
        sqrt(2)
        >>> sqrt((_/2)**2 + s.apothem**2) == s.radius
        True

        """
        return self.radius*2*sin(pi/self._n)

    @property
    def center(self):
        """The center of the RegularPolygon

        This is also the center of the circumscribing circle.

        Returns
        =======

        center : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.center

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.center
        Point2D(0, 0)
        """
        return self._center

    centroid = center

    @property
    def circumcenter(self):
        """
        Alias for center.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.circumcenter
        Point2D(0, 0)
        """
        return self.center

    @property
    def radius(self):
        """Radius of the RegularPolygon

        This is also the radius of the circumscribing circle.

        Returns
        =======

        radius : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.radius
        r

        """
        return self._radius

    @property
    def circumradius(self):
        """
        Alias for radius.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.circumradius
        r
        """
        return self.radius

    @property
    def rotation(self):
        """CCW angle by which the RegularPolygon is rotated

        Returns
        =======

        rotation : number or instance of Basic

        Examples
        ========

        >>> from sympy import pi
        >>> from sympy.abc import a
        >>> from sympy import RegularPolygon, Point
        >>> RegularPolygon(Point(0, 0), 3, 4, pi/4).rotation
        pi/4

        Numerical rotation angles are made canonical:

        >>> RegularPolygon(Point(0, 0), 3, 4, a).rotation
        a
        >>> RegularPolygon(Point(0, 0), 3, 4, pi).rotation
        0

        """
        return self._rot

    @property
    def apothem(self):
        """The inradius of the RegularPolygon.

        The apothem/inradius is the radius of the inscribed circle.

        Returns
        =======

        apothem : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.apothem
        sqrt(2)*r/2

        """
        return self.radius * cos(S.Pi/self._n)

    @property
    def inradius(self):
        """
        Alias for apothem.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.inradius
        sqrt(2)*r/2
        """
        return self.apothem

    @property
    def interior_angle(self):
        """Measure of the interior angles.

        Returns
        =======

        interior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.interior_angle
        3*pi/4

        """
        return (self._n - 2)*S.Pi/self._n

    @property
    def exterior_angle(self):
        """Measure of the exterior angles.

        Returns
        =======

        exterior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.exterior_angle
        pi/4

        """
        return 2*S.Pi/self._n

    @property
    def circumcircle(self):
        """The circumcircle of the RegularPolygon.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        circumcenter, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.circumcircle
        Circle(Point2D(0, 0), 4)

        """
        return Circle(self.center, self.radius)

    @property
    def incircle(self):
        """The incircle of the RegularPolygon.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        inradius, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 7)
        >>> rp.incircle
        Circle(Point2D(0, 0), 4*cos(pi/7))

        """
        return Circle(self.center, self.apothem)

    @property
    def angles(self):
        """
        Returns a dictionary with keys, the vertices of the Polygon,
        and values, the interior angle at each vertex.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.angles
        {Point2D(-5/2, -5*sqrt(3)/2): pi/3,
         Point2D(-5/2, 5*sqrt(3)/2): pi/3,
         Point2D(5, 0): pi/3}
        """
        ret = {}
        ang = self.interior_angle
        for v in self.vertices:
            ret[v] = ang
        return ret

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        The general Polygon.encloses_point method is called only if
        a point is not within or beyond the incircle or circumcircle,
        respectively.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, S, Point, Symbol
        >>> p = RegularPolygon((0, 0), 3, 4)
        >>> p.encloses_point(Point(0, 0))
        True
        >>> r, R = p.inradius, p.circumradius
        >>> p.encloses_point(Point((r + R)/2, 0))
        True
        >>> p.encloses_point(Point(R/2, R/2 + (R - r)/10))
        False
        >>> t = Symbol('t', real=True)
        >>> p.encloses_point(p.arbitrary_point().subs(t, S.Half))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        """

        c = self.center
        d = Segment(c, p).length
        if d >= self.radius:
            return False
        elif d < self.inradius:
            return True
        else:
            # now enumerate the RegularPolygon like a general polygon.
            return Polygon.encloses_point(self, p)

    def spin(self, angle):
        """Increment *in place* the virtual Polygon's rotation by ccw angle.

        See also: rotate method which moves the center.

        >>> from sympy import Polygon, Point, pi
        >>> r = Polygon(Point(0,0), 1, n=3)
        >>> r.vertices[0]
        Point2D(1, 0)
        >>> r.spin(pi/6)
        >>> r.vertices[0]
        Point2D(sqrt(3)/2, 1/2)

        See Also
        ========

        rotation
        rotate : Creates a copy of the RegularPolygon rotated about a Point

        """
        self._rot += angle

    def rotate(self, angle, pt=None):
        """Override GeometryEntity.rotate to first rotate the RegularPolygon
        about its center.

        >>> from sympy import Point, RegularPolygon, pi
        >>> t = RegularPolygon(Point(1, 0), 1, 3)
        >>> t.vertices[0] # vertex on x-axis
        Point2D(2, 0)
        >>> t.rotate(pi/2).vertices[0] # vertex on y axis now
        Point2D(0, 2)

        See Also
        ========

        rotation
        spin : Rotates a RegularPolygon in place

        """

        r = type(self)(*self.args)  # need a copy or else changes are in-place
        r._rot += angle
        return GeometryEntity.rotate(r, angle, pt)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the radius that must be
        scaled (if x == y) or else a new Polygon must be returned.

        >>> from sympy import RegularPolygon

        Symmetric scaling returns a RegularPolygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 2)
        RegularPolygon(Point2D(0, 0), 2, 4, 0)

        Asymmetric scaling returns a kite as a Polygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 1)
        Polygon(Point2D(2, 0), Point2D(0, 1), Point2D(-2, 0), Point2D(0, -1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        if x != y:
            return Polygon(*self.vertices).scale(x, y)
        c, r, n, rot = self.args
        r *= x
        return self.func(c, r, n, rot)

    def reflect(self, line):
        """Override GeometryEntity.reflect since this is not made of only
        points.

        Examples
        ========

        >>> from sympy import RegularPolygon, Line

        >>> RegularPolygon((0, 0), 1, 4).reflect(Line((0, 1), slope=-2))
        RegularPolygon(Point2D(4/5, 2/5), -1, 4, atan(4/3))

        """
        c, r, n, rot = self.args
        v = self.vertices[0]
        d = v - c
        cc = c.reflect(line)
        vv = v.reflect(line)
        dd = vv - cc
        # calculate rotation about the new center
        # which will align the vertices
        l1 = Ray((0, 0), dd)
        l2 = Ray((0, 0), d)
        ang = l1.closing_angle(l2)
        rot += ang
        # change sign of radius as point traversal is reversed
        return self.func(cc, -r, n, rot)

    @property
    def vertices(self):
        """The vertices of the RegularPolygon.

        Returns
        =======

        vertices : list
            Each vertex is a Point.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.vertices
        [Point2D(5, 0), Point2D(0, 5), Point2D(-5, 0), Point2D(0, -5)]

        """
        c = self._center
        r = abs(self._radius)
        rot = self._rot
        v = 2*S.Pi/self._n

        return [Point(c.x + r*cos(k*v + rot), c.y + r*sin(k*v + rot))
                for k in range(self._n)]

    def __eq__(self, o):
        if not isinstance(o, Polygon):
            return False
        elif not isinstance(o, RegularPolygon):
            return Polygon.__eq__(o, self)
        return self.args == o.args

    def __hash__(self):
        return super().__hash__()


class Triangle(Polygon):
    """
    A polygon with three vertices and three sides.

    Parameters
    ==========

    points : sequence of Points
    keyword: asa, sas, or sss to specify sides/angles of the triangle

    Attributes
    ==========

    vertices
    altitudes
    orthocenter
    circumcenter
    circumradius
    circumcircle
    inradius
    incircle
    exradii
    medians
    medial
    nine_point_circle

    Raises
    ======

    GeometryError
        If the number of vertices is not equal to three, or one of the vertices
        is not a Point, or a valid keyword is not given.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import Triangle, Point
    >>> Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
    Triangle(Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

    Keywords sss, sas, or asa can be used to give the desired
    side lengths (in order) and interior angles (in degrees) that
    define the triangle:

    >>> Triangle(sss=(3, 4, 5))
    Triangle(Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> Triangle(asa=(30, 1, 30))
    Triangle(Point2D(0, 0), Point2D(1, 0), Point2D(1/2, sqrt(3)/6))
    >>> Triangle(sas=(1, 45, 2))
    Triangle(Point2D(0, 0), Point2D(2, 0), Point2D(sqrt(2)/2, sqrt(2)/2))

    """

    def __new__(cls, *args, **kwargs):
        if len(args) != 3:
            if 'sss' in kwargs:
                return _sss(*[simplify(a) for a in kwargs['sss']])
            if 'asa' in kwargs:
                return _asa(*[simplify(a) for a in kwargs['asa']])
            if 'sas' in kwargs:
                return _sas(*[simplify(a) for a in kwargs['sas']])
            msg = "Triangle instantiates with three points or a valid keyword."
            raise GeometryError(msg)

        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # remove consecutive duplicates
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = sorted(
                [nodup[i], nodup[i + 1], nodup[i + 2]], key=default_sort_key)
            if Point.is_collinear(a, b, c):
                nodup[i] = a
                nodup[i + 1] = None
                nodup.pop(i + 1)
            i += 1

        vertices = list(filter(lambda x: x is not None, nodup))

        if len(vertices) == 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def vertices(self):
        """The triangle's vertices

        Returns
        =======

        vertices : tuple
            Each element in the tuple is a Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t.vertices
        (Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

        """
        return self.args

    def is_similar(t1, t2):
        """Is another triangle similar to this one.

        Two triangles are similar if one can be uniformly scaled to the other.

        Parameters
        ==========

        other: Triangle

        Returns
        =======

        is_similar : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -3))
        >>> t1.is_similar(t2)
        True

        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -4))
        >>> t1.is_similar(t2)
        False

        """
        if not isinstance(t2, Polygon):
            return False

        s1_1, s1_2, s1_3 = [side.length for side in t1.sides]
        s2 = [side.length for side in t2.sides]

        def _are_similar(u1, u2, u3, v1, v2, v3):
            e1 = simplify(u1/v1)
            e2 = simplify(u2/v2)
            e3 = simplify(u3/v3)
            return bool(e1 == e2) and bool(e2 == e3)

        # There's only 6 permutations, so write them out
        return _are_similar(s1_1, s1_2, s1_3, *s2) or \
            _are_similar(s1_1, s1_3, s1_2, *s2) or \
            _are_similar(s1_2, s1_1, s1_3, *s2) or \
            _are_similar(s1_2, s1_3, s1_1, *s2) or \
            _are_similar(s1_3, s1_1, s1_2, *s2) or \
            _are_similar(s1_3, s1_2, s1_1, *s2)

    def is_equilateral(self):
        """Are all the sides the same length?

        Returns
        =======

        is_equilateral : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar, RegularPolygon
        is_isosceles, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_equilateral()
        False

        >>> from sympy import sqrt
        >>> t2 = Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3)))
        >>> t2.is_equilateral()
        True

        """
        return not has_variety(s.length for s in self.sides)

    def is_isosceles(self):
        """Are two or more of the sides the same length?

        Returns
        =======

        is_isosceles : boolean

        See Also
        ========

        is_equilateral, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(2, 4))
        >>> t1.is_isosceles()
        True

        """
        return has_dups(s.length for s in self.sides)

    def is_scalene(self):
        """Are all the sides of the triangle of different lengths?

        Returns
        =======

        is_scalene : boolean

        See Also
        ========

        is_equilateral, is_isosceles, is_right

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(1, 4))
        >>> t1.is_scalene()
        True

        """
        return not has_dups(s.length for s in self.sides)

    def is_right(self):
        """Is the triangle right-angled.

        Returns
        =======

        is_right : boolean

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular
        is_equilateral, is_isosceles, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_right()
        True

        """
        s = self.sides
        return Segment.is_perpendicular(s[0], s[1]) or \
            Segment.is_perpendicular(s[1], s[2]) or \
            Segment.is_perpendicular(s[0], s[2])

    @property
    def altitudes(self):
        """The altitudes of the triangle.

        An altitude of a triangle is a segment through a vertex,
        perpendicular to the opposite side, with length being the
        height of the vertex measured from the line containing the side.

        Returns
        =======

        altitudes : dict
            The dictionary consists of keys which are vertices and values
            which are Segments.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.altitudes[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: s[1].perpendicular_segment(v[0]),
                v[1]: s[2].perpendicular_segment(v[1]),
                v[2]: s[0].perpendicular_segment(v[2])}

    @property
    def orthocenter(self):
        """The orthocenter of the triangle.

        The orthocenter is the intersection of the altitudes of a triangle.
        It may lie inside, outside or on the triangle.

        Returns
        =======

        orthocenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.orthocenter
        Point2D(0, 0)

        """
        a = self.altitudes
        v = self.vertices
        return Line(a[v[0]]).intersection(Line(a[v[1]]))[0]

    @property
    def circumcenter(self):
        """The circumcenter of the triangle

        The circumcenter is the center of the circumcircle.

        Returns
        =======

        circumcenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcenter
        Point2D(1/2, 1/2)
        """
        a, b, c = [x.perpendicular_bisector() for x in self.sides]
        return a.intersection(b)[0]

    @property
    def circumradius(self):
        """The radius of the circumcircle of the triangle.

        Returns
        =======

        circumradius : number of Basic instance

        See Also
        ========

        sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import Point, Triangle
        >>> a = Symbol('a')
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, a)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumradius
        sqrt(a**2/4 + 1/4)
        """
        return Point.distance(self.circumcenter, self.vertices[0])

    @property
    def circumcircle(self):
        """The circle which passes through the three vertices of the triangle.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcircle
        Circle(Point2D(1/2, 1/2), sqrt(2)/2)

        """
        return Circle(self.circumcenter, self.circumradius)

    def bisectors(self):
        """The angle bisectors of the triangle.

        An angle bisector of a triangle is a straight line through a vertex
        which cuts the corresponding angle in half.

        Returns
        =======

        bisectors : dict
            Each key is a vertex (Point) and each value is the corresponding
            bisector (Segment).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Triangle, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> from sympy import sqrt
        >>> t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
        True

        """
        # use lines containing sides so containment check during
        # intersection calculation can be avoided, thus reducing
        # the processing time for calculating the bisectors
        s = [Line(l) for l in self.sides]
        v = self.vertices
        c = self.incenter
        l1 = Segment(v[0], Line(v[0], c).intersection(s[1])[0])
        l2 = Segment(v[1], Line(v[1], c).intersection(s[2])[0])
        l3 = Segment(v[2], Line(v[2], c).intersection(s[0])[0])
        return {v[0]: l1, v[1]: l2, v[2]: l3}

    @property
    def incenter(self):
        """The center of the incircle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incenter : Point

        See Also
        ========

        incircle, sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incenter
        Point2D(1 - sqrt(2)/2, 1 - sqrt(2)/2)

        """
        s = self.sides
        l = Matrix([s[i].length for i in [1, 2, 0]])
        p = sum(l)
        v = self.vertices
        x = simplify(l.dot(Matrix([vi.x for vi in v]))/p)
        y = simplify(l.dot(Matrix([vi.y for vi in v]))/p)
        return Point(x, y)

    @property
    def inradius(self):
        """The radius of the incircle.

        Returns
        =======

        inradius : number of Basic instance

        See Also
        ========

        incircle, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(4, 0), Point(0, 3)
        >>> t = Triangle(p1, p2, p3)
        >>> t.inradius
        1

        """
        return simplify(2 * self.area / self.perimeter)

    @property
    def incircle(self):
        """The incircle of the triangle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(2, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incircle
        Circle(Point2D(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

        """
        return Circle(self.incenter, self.inradius)

    @property
    def exradii(self):
        """The radius of excircles of a triangle.

        An excircle of the triangle is a circle lying outside the triangle,
        tangent to one of its sides and tangent to the extensions of the
        other two.

        Returns
        =======

        exradii : dict

        See Also
        ========

        sympy.geometry.polygon.Triangle.inradius

        Examples
        ========

        The exradius touches the side of the triangle to which it is keyed, e.g.
        the exradius touching side 2 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.exradii[t.sides[2]]
        -2 + sqrt(10)

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Exradius.html
        .. [2] https://mathworld.wolfram.com/Excircles.html

        """

        side = self.sides
        a = side[0].length
        b = side[1].length
        c = side[2].length
        s = (a+b+c)/2
        area = self.area
        exradii = {self.sides[0]: simplify(area/(s-a)),
                   self.sides[1]: simplify(area/(s-b)),
                   self.sides[2]: simplify(area/(s-c))}

        return exradii

    @property
    def excenters(self):
        """Excenters of the triangle.

        An excenter is the center of a circle that is tangent to a side of the
        triangle and the extensions of the other two sides.

        Returns
        =======

        excenters : dict


        Examples
        ========

        The excenters are keyed to the side of the triangle to which their corresponding
        excircle is tangent: The center is keyed, e.g. the excenter of a circle touching
        side 0 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.excenters[t.sides[0]]
        Point2D(12*sqrt(10), 2/3 + sqrt(10)/3)

        See Also
        ========

        sympy.geometry.polygon.Triangle.exradii

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Excircles.html

        """

        s = self.sides
        v = self.vertices
        a = s[0].length
        b = s[1].length
        c = s[2].length
        x = [v[0].x, v[1].x, v[2].x]
        y = [v[0].y, v[1].y, v[2].y]

        exc_coords = {
            "x1": simplify(-a*x[0]+b*x[1]+c*x[2]/(-a+b+c)),
            "x2": simplify(a*x[0]-b*x[1]+c*x[2]/(a-b+c)),
            "x3": simplify(a*x[0]+b*x[1]-c*x[2]/(a+b-c)),
            "y1": simplify(-a*y[0]+b*y[1]+c*y[2]/(-a+b+c)),
            "y2": simplify(a*y[0]-b*y[1]+c*y[2]/(a-b+c)),
            "y3": simplify(a*y[0]+b*y[1]-c*y[2]/(a+b-c))
        }

        excenters = {
            s[0]: Point(exc_coords["x1"], exc_coords["y1"]),
            s[1]: Point(exc_coords["x2"], exc_coords["y2"]),
            s[2]: Point(exc_coords["x3"], exc_coords["y3"])
        }

        return excenters

    @property
    def medians(self):
        """The medians of the triangle.

        A median of a triangle is a straight line through a vertex and the
        midpoint of the opposite side, and divides the triangle into two
        equal areas.

        Returns
        =======

        medians : dict
            Each key is a vertex (Point) and each value is the median (Segment)
            at that point.

        See Also
        ========

        sympy.geometry.point.Point.midpoint, sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medians[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: Segment(v[0], s[1].midpoint),
                v[1]: Segment(v[1], s[2].midpoint),
                v[2]: Segment(v[2], s[0].midpoint)}

    @property
    def medial(self):
        """The medial triangle of the triangle.

        The triangle which is formed from the midpoints of the three sides.

        Returns
        =======

        medial : Triangle

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medial
        Triangle(Point2D(1/2, 0), Point2D(1/2, 1/2), Point2D(0, 1/2))

        """
        s = self.sides
        return Triangle(s[0].midpoint, s[1].midpoint, s[2].midpoint)

    @property
    def nine_point_circle(self):
        """The nine-point circle of the triangle.

        Nine-point circle is the circumcircle of the medial triangle, which
        passes through the feet of altitudes and the middle points of segments
        connecting the vertices and the orthocenter.

        Returns
        =======

        nine_point_circle : Circle

        See also
        ========

        sympy.geometry.line.Segment.midpoint
        sympy.geometry.polygon.Triangle.medial
        sympy.geometry.polygon.Triangle.orthocenter

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.nine_point_circle
        Circle(Point2D(1/4, 1/4), sqrt(2)/4)

        """
        return Circle(*self.medial.vertices)

    @property
    def eulerline(self):
        """The Euler line of the triangle.

        The line which passes through circumcenter, centroid and orthocenter.

        Returns
        =======

        eulerline : Line (or Point for equilateral triangles in which case all
                    centers coincide)

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.eulerline
        Line2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        if self.is_equilateral():
            return self.orthocenter
        return Line(self.orthocenter, self.circumcenter)

def rad(d):
    """Return the radian value for the given degrees (pi = 180 degrees)."""
    return d*pi/180


def deg(r):
    """Return the degree value for the given radians (pi = 180 degrees)."""
    return r/pi*180


def _slope(d):
    rv = tan(rad(d))
    return rv


def _asa(d1, l, d2):
    """Return triangle having side with length l on the x-axis."""
    xy = Line((0, 0), slope=_slope(d1)).intersection(
        Line((l, 0), slope=_slope(180 - d2)))[0]
    return Triangle((0, 0), (l, 0), xy)


def _sss(l1, l2, l3):
    """Return triangle having side of length l1 on the x-axis."""
    c1 = Circle((0, 0), l3)
    c2 = Circle((l1, 0), l2)
    inter = [a for a in c1.intersection(c2) if a.y.is_nonnegative]
    if not inter:
        return None
    pt = inter[0]
    return Triangle((0, 0), (l1, 0), pt)


def _sas(l1, d, l2):
    """Return triangle having side with length l2 on the x-axis."""
    p1 = Point(0, 0)
    p2 = Point(l2, 0)
    p3 = Point(cos(rad(d))*l1, sin(rad(d))*l1)
    return Triangle(p1, p2, p3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_run.py ===
from __future__ import annotations

import contextvars
import functools
import gc
import sys
import threading
import time
import types
import weakref
from contextlib import ExitStack, contextmanager, suppress
from math import inf, nan
from typing import TYPE_CHECKING, NoReturn, TypeVar
from unittest import mock

import outcome
import pytest
import sniffio

from ... import _core
from ..._threads import to_thread_run_sync
from ..._timeouts import fail_after, sleep
from ...testing import (
    Matcher,
    RaisesGroup,
    Sequencer,
    assert_checkpoints,
    wait_all_tasks_blocked,
)
from .._run import DEADLINE_HEAP_MIN_PRUNE_THRESHOLD, _count_context_run_tb_frames
from .tutil import (
    check_sequence_matches,
    create_asyncio_future_in_new_loop,
    gc_collect_harder,
    ignore_coroutine_never_awaited_warnings,
    restore_unraisablehook,
    slow,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Awaitable,
        Callable,
        Generator,
    )

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup


T = TypeVar("T")


# slightly different from _timeouts.sleep_forever because it returns the value
# its rescheduled with, which is really only useful for tests of
# rescheduling...
async def sleep_forever() -> object:
    return await _core.wait_task_rescheduled(lambda _: _core.Abort.SUCCEEDED)


def not_none(x: T | None) -> T:
    """Assert that this object is not None.

    This is just to satisfy type checkers, if this ever fails the test is broken.
    """
    assert x is not None
    return x


def test_basic() -> None:
    async def trivial(x: T) -> T:
        return x

    assert _core.run(trivial, 8) == 8

    with pytest.raises(TypeError):
        # Missing an argument
        _core.run(trivial)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        # Not an async function
        _core.run(lambda: None)  # type: ignore

    async def trivial2(x: T) -> T:
        await _core.checkpoint()
        return x

    assert _core.run(trivial2, 1) == 1


def test_initial_task_error() -> None:
    async def main(x: object) -> NoReturn:
        raise ValueError(x)

    with pytest.raises(ValueError, match=r"^17$") as excinfo:
        _core.run(main, 17)
    assert excinfo.value.args == (17,)


def test_run_nesting() -> None:
    async def inception() -> None:
        async def main() -> None:  # pragma: no cover
            pass

        return _core.run(main)

    with pytest.raises(RuntimeError) as excinfo:
        _core.run(inception)
    assert "from inside" in str(excinfo.value)


async def test_nursery_warn_use_async_with() -> None:
    on = _core.open_nursery()
    with pytest.raises(RuntimeError) as excinfo:
        with on:  # type: ignore
            pass  # pragma: no cover
    excinfo.match(
        r"use 'async with open_nursery\(...\)', not 'with open_nursery\(...\)'",
    )

    # avoid unawaited coro.
    async with on:
        pass


async def test_nursery_main_block_error_basic() -> None:
    exc = ValueError("whoops")

    with RaisesGroup(Matcher(check=lambda e: e is exc)):
        async with _core.open_nursery():
            raise exc


async def test_child_crash_basic() -> None:
    my_exc = ValueError("uh oh")

    async def erroring() -> NoReturn:
        raise my_exc

    with RaisesGroup(Matcher(check=lambda e: e is my_exc)):
        # nursery.__aexit__ propagates exception from child back to parent
        async with _core.open_nursery() as nursery:
            nursery.start_soon(erroring)


async def test_basic_interleave() -> None:
    async def looper(whoami: str, record: list[tuple[str, int]]) -> None:
        for i in range(3):
            record.append((whoami, i))
            await _core.checkpoint()

    record: list[tuple[str, int]] = []
    async with _core.open_nursery() as nursery:
        nursery.start_soon(looper, "a", record)
        nursery.start_soon(looper, "b", record)

    check_sequence_matches(
        record,
        [{("a", 0), ("b", 0)}, {("a", 1), ("b", 1)}, {("a", 2), ("b", 2)}],
    )


def test_task_crash_propagation() -> None:
    looper_record: list[str] = []

    async def looper() -> None:
        try:
            while True:
                await _core.checkpoint()
        except _core.Cancelled:
            print("looper cancelled")
            looper_record.append("cancelled")

    async def crasher() -> NoReturn:
        raise ValueError("argh")

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(looper)
            nursery.start_soon(crasher)

    with RaisesGroup(Matcher(ValueError, "^argh$")):
        _core.run(main)

    assert looper_record == ["cancelled"]


def test_main_and_task_both_crash() -> None:
    # If main crashes and there's also a task crash, then we get both in an
    # ExceptionGroup
    async def crasher() -> NoReturn:
        raise ValueError

    async def main() -> NoReturn:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher)
            raise KeyError

    with RaisesGroup(ValueError, KeyError):
        _core.run(main)


def test_two_child_crashes() -> None:
    async def crasher(etype: type[Exception]) -> NoReturn:
        raise etype

    async def main() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher, KeyError)
            nursery.start_soon(crasher, ValueError)

    with RaisesGroup(ValueError, KeyError):
        _core.run(main)


async def test_child_crash_wakes_parent() -> None:
    async def crasher() -> NoReturn:
        raise ValueError("this is a crash")

    with RaisesGroup(Matcher(ValueError, "^this is a crash$")):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher)
            await sleep_forever()


async def test_reschedule() -> None:
    t1: _core.Task | None = None
    t2: _core.Task | None = None

    async def child1() -> None:
        nonlocal t1, t2
        t1 = _core.current_task()
        print("child1 start")
        x = await sleep_forever()
        print("child1 woke")
        assert x == 0
        print("child1 rescheduling t2")
        _core.reschedule(not_none(t2), outcome.Error(ValueError("error message")))
        print("child1 exit")

    async def child2() -> None:
        nonlocal t1, t2
        print("child2 start")
        t2 = _core.current_task()
        _core.reschedule(not_none(t1), outcome.Value(0))
        print("child2 sleep")
        with pytest.raises(ValueError, match=r"^error message$"):
            await sleep_forever()
        print("child2 successful exit")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child1)
        # let t1 run and fall asleep
        await _core.checkpoint()
        nursery.start_soon(child2)


async def test_current_time() -> None:
    t1 = _core.current_time()
    # Windows clock is pretty low-resolution -- appveyor tests fail unless we
    # sleep for a bit here.
    time.sleep(time.get_clock_info("perf_counter").resolution)  # noqa: ASYNC251
    t2 = _core.current_time()
    assert t1 < t2


async def test_current_time_with_mock_clock(mock_clock: _core.MockClock) -> None:
    start = mock_clock.current_time()
    assert mock_clock.current_time() == _core.current_time()
    assert mock_clock.current_time() == _core.current_time()
    mock_clock.jump(3.15)
    assert start + 3.15 == mock_clock.current_time() == _core.current_time()


async def test_current_clock(mock_clock: _core.MockClock) -> None:
    assert mock_clock is _core.current_clock()


async def test_current_task() -> None:
    parent_task = _core.current_task()

    async def child() -> None:
        assert not_none(_core.current_task().parent_nursery).parent_task is parent_task

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child)


async def test_root_task() -> None:
    root = not_none(_core.current_root_task())
    assert root.parent_nursery is root.eventual_parent_nursery is None


def test_out_of_context() -> None:
    with pytest.raises(RuntimeError):
        _core.current_task()
    with pytest.raises(RuntimeError):
        _core.current_time()


async def test_current_statistics(mock_clock: _core.MockClock) -> None:
    # Make sure all the early startup stuff has settled down
    await wait_all_tasks_blocked()

    # A child that sticks around to make some interesting stats:
    async def child() -> None:
        with suppress(_core.Cancelled):
            await sleep_forever()

    stats = _core.current_statistics()
    print(stats)
    # 2 system tasks + us
    assert stats.tasks_living == 3
    assert stats.run_sync_soon_queue_size == 0

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child)
        await wait_all_tasks_blocked()
        token = _core.current_trio_token()
        token.run_sync_soon(lambda: None)
        token.run_sync_soon(lambda: None, idempotent=True)
        stats = _core.current_statistics()
        print(stats)
        # 2 system tasks + us + child
        assert stats.tasks_living == 4
        # the exact value here might shift if we change how we do accounting
        # (currently it only counts tasks that we already know will be
        # runnable on the next pass), but still useful to at least test the
        # difference between now and after we wake up the child:
        assert stats.tasks_runnable == 0
        assert stats.run_sync_soon_queue_size == 2

        nursery.cancel_scope.cancel()
        stats = _core.current_statistics()
        print(stats)
        assert stats.tasks_runnable == 1

    # Give the child a chance to die and the run_sync_soon a chance to clear
    await _core.checkpoint()
    await _core.checkpoint()

    with _core.CancelScope(deadline=_core.current_time() + 5):
        stats = _core.current_statistics()
        print(stats)
        assert stats.seconds_to_next_deadline == 5
    stats = _core.current_statistics()
    print(stats)
    assert stats.seconds_to_next_deadline == inf


async def test_cancel_scope_repr(mock_clock: _core.MockClock) -> None:
    scope = _core.CancelScope()
    assert "unbound" in repr(scope)
    with scope:
        assert "active" in repr(scope)
        scope.deadline = _core.current_time() - 1
        assert "deadline is 1.00 seconds ago" in repr(scope)
        scope.deadline = _core.current_time() + 10
        assert "deadline is 10.00 seconds from now" in repr(scope)
        # when not in async context, can't get the current time
        assert "deadline" not in await to_thread_run_sync(repr, scope)
        scope.cancel()
        assert "cancelled" in repr(scope)
    assert "exited" in repr(scope)


async def test_cancel_scope_validation() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Cannot specify both a deadline and a relative deadline$",
    ):
        _core.CancelScope(deadline=7, relative_deadline=3)

    with pytest.raises(ValueError, match=r"^deadline must not be NaN$"):
        _core.CancelScope(deadline=nan)
    with pytest.raises(ValueError, match=r"^relative deadline must not be NaN$"):
        _core.CancelScope(relative_deadline=nan)

    with pytest.raises(ValueError, match=r"^timeout must be non-negative$"):
        _core.CancelScope(relative_deadline=-3)

    scope = _core.CancelScope()

    with pytest.raises(ValueError, match=r"^deadline must not be NaN$"):
        scope.deadline = nan
    with pytest.raises(ValueError, match=r"^relative deadline must not be NaN$"):
        scope.relative_deadline = nan

    with pytest.raises(ValueError, match=r"^relative deadline must be non-negative$"):
        scope.relative_deadline = -3
    scope.relative_deadline = 5
    assert scope.relative_deadline == 5

    # several related tests of CancelScope are implicitly handled by test_timeouts.py


def test_cancel_points() -> None:
    async def main1() -> None:
        with _core.CancelScope() as scope:
            await _core.checkpoint_if_cancelled()
            scope.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint_if_cancelled()

    _core.run(main1)

    async def main2() -> None:
        with _core.CancelScope() as scope:
            await _core.checkpoint()
            scope.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()

    _core.run(main2)

    async def main3() -> None:
        with _core.CancelScope() as scope:
            scope.cancel()
            with pytest.raises(_core.Cancelled):
                await sleep_forever()

    _core.run(main3)

    async def main4() -> None:
        with _core.CancelScope() as scope:
            scope.cancel()
            await _core.cancel_shielded_checkpoint()
            await _core.cancel_shielded_checkpoint()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()

    _core.run(main4)


async def test_cancel_edge_cases() -> None:
    with _core.CancelScope() as scope:
        # Two cancels in a row -- idempotent
        scope.cancel()
        scope.cancel()
        await _core.checkpoint()
    assert scope.cancel_called
    assert scope.cancelled_caught

    with _core.CancelScope() as scope:
        # Check level-triggering
        scope.cancel()
        with pytest.raises(_core.Cancelled):
            await sleep_forever()
        with pytest.raises(_core.Cancelled):
            await sleep_forever()


async def test_cancel_scope_exceptiongroup_filtering() -> None:
    async def crasher() -> NoReturn:
        raise KeyError

    # This is outside the outer scope, so all the Cancelled
    # exceptions should have been absorbed, leaving just a regular
    # KeyError from crasher(), wrapped in an ExceptionGroup
    with RaisesGroup(KeyError):
        with _core.CancelScope() as outer:
            # Since the outer scope became cancelled before the
            # nursery block exited, all cancellations inside the
            # nursery block continue propagating to reach the
            # outer scope.
            with RaisesGroup(
                _core.Cancelled,
                _core.Cancelled,
                _core.Cancelled,
                KeyError,
            ) as excinfo:
                async with _core.open_nursery() as nursery:
                    # Two children that get cancelled by the nursery scope
                    nursery.start_soon(sleep_forever)  # t1
                    nursery.start_soon(sleep_forever)  # t2
                    nursery.cancel_scope.cancel()
                    with _core.CancelScope(shield=True):
                        await wait_all_tasks_blocked()
                    # One child that gets cancelled by the outer scope
                    nursery.start_soon(sleep_forever)  # t3
                    outer.cancel()
                    # And one that raises a different error
                    nursery.start_soon(crasher)  # t4
                # and then our __aexit__ also receives an outer Cancelled
            # reraise the exception caught by RaisesGroup for the
            # CancelScope to handle
            raise excinfo.value


async def test_precancelled_task() -> None:
    # a task that gets spawned into an already-cancelled nursery should begin
    # execution (https://github.com/python-trio/trio/issues/41), but get a
    # cancelled error at its first blocking call.
    record: list[str] = []

    async def blocker() -> None:
        record.append("started")
        await sleep_forever()

    async with _core.open_nursery() as nursery:
        nursery.cancel_scope.cancel()
        nursery.start_soon(blocker)
    assert record == ["started"]


async def test_cancel_shielding() -> None:
    with _core.CancelScope() as outer:
        with _core.CancelScope() as inner:
            await _core.checkpoint()
            outer.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()

            assert inner.shield is False
            with pytest.raises(TypeError):
                inner.shield = "hello"  # type: ignore
            assert inner.shield is False

            inner.shield = True
            assert inner.shield is True
            # shield protects us from 'outer'
            await _core.checkpoint()

            with _core.CancelScope() as innerest:
                innerest.cancel()
                # but it doesn't protect us from scope inside inner
                with pytest.raises(_core.Cancelled):
                    await _core.checkpoint()
            await _core.checkpoint()

            inner.shield = False
            # can disable shield again
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()

            # re-enable shield
            inner.shield = True
            await _core.checkpoint()
            # shield doesn't protect us from inner itself
            inner.cancel()
            # This should now raise, but be absorbed by the inner scope
            await _core.checkpoint()
        assert inner.cancelled_caught


# make sure that cancellation propagates immediately to all children
async def test_cancel_inheritance() -> None:
    record: set[str] = set()

    async def leaf(ident: str) -> None:
        try:
            await sleep_forever()
        except _core.Cancelled:
            record.add(ident)

    async def worker(ident: str) -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(leaf, ident + "-l1")
            nursery.start_soon(leaf, ident + "-l2")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(worker, "w1")
        nursery.start_soon(worker, "w2")
        nursery.cancel_scope.cancel()

    assert record == {"w1-l1", "w1-l2", "w2-l1", "w2-l2"}


async def test_cancel_shield_abort() -> None:
    with _core.CancelScope() as outer:
        async with _core.open_nursery() as nursery:
            outer.cancel()
            nursery.cancel_scope.shield = True
            # The outer scope is cancelled, but this task is protected by the
            # shield, so it manages to get to sleep
            record = []

            async def sleeper() -> None:
                record.append("sleeping")
                try:
                    await sleep_forever()
                except _core.Cancelled:
                    record.append("cancelled")

            nursery.start_soon(sleeper)
            await wait_all_tasks_blocked()
            assert record == ["sleeping"]
            # now when we unshield, it should abort the sleep.
            nursery.cancel_scope.shield = False
            # wait for the task to finish before entering the nursery
            # __aexit__, because __aexit__ could make it spuriously look like
            # this worked by cancelling the nursery scope. (When originally
            # written, without these last few lines, the test spuriously
            # passed, even though shield assignment was buggy.)
            with _core.CancelScope(shield=True):
                await wait_all_tasks_blocked()
                assert record == ["sleeping", "cancelled"]


async def test_basic_timeout(mock_clock: _core.MockClock) -> None:
    start = _core.current_time()
    with _core.CancelScope() as scope:
        assert scope.deadline == inf
        scope.deadline = start + 1
        assert scope.deadline == start + 1
    assert not scope.cancel_called
    mock_clock.jump(2)
    await _core.checkpoint()
    await _core.checkpoint()
    await _core.checkpoint()
    assert not scope.cancel_called

    start = _core.current_time()
    with _core.CancelScope(deadline=start + 1) as scope:
        mock_clock.jump(2)
        await sleep_forever()
    # But then the scope swallowed the exception... but we can still see it
    # here:
    assert scope.cancel_called
    assert scope.cancelled_caught

    # changing deadline
    start = _core.current_time()
    with _core.CancelScope() as scope:
        await _core.checkpoint()
        scope.deadline = start + 10
        await _core.checkpoint()
        mock_clock.jump(5)
        await _core.checkpoint()
        scope.deadline = start + 1
        with pytest.raises(_core.Cancelled):
            await _core.checkpoint()
        with pytest.raises(_core.Cancelled):
            await _core.checkpoint()


async def test_cancel_scope_nesting() -> None:
    # Nested scopes: if two triggering at once, the outer one wins
    with _core.CancelScope() as scope1:
        with _core.CancelScope() as scope2:
            with _core.CancelScope() as scope3:
                scope3.cancel()
                scope2.cancel()
                await sleep_forever()
    assert scope3.cancel_called
    assert not scope3.cancelled_caught
    assert scope2.cancel_called
    assert scope2.cancelled_caught
    assert not scope1.cancel_called
    assert not scope1.cancelled_caught

    # shielding
    with _core.CancelScope() as scope1:
        with _core.CancelScope() as scope2:
            scope1.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()
            scope2.shield = True
            await _core.checkpoint()
            scope2.cancel()
            with pytest.raises(_core.Cancelled):
                await _core.checkpoint()

    # if a scope is pending, but then gets popped off the stack, then it
    # isn't delivered
    with _core.CancelScope() as scope:
        scope.cancel()
        await _core.cancel_shielded_checkpoint()
    await _core.checkpoint()
    assert not scope.cancelled_caught


# Regression test for https://github.com/python-trio/trio/issues/1175
async def test_unshield_while_cancel_propagating() -> None:
    with _core.CancelScope() as outer:
        with _core.CancelScope() as inner:
            outer.cancel()
            try:
                await _core.checkpoint()
            finally:
                inner.shield = True
    assert outer.cancelled_caught
    assert not inner.cancelled_caught


async def test_cancel_unbound() -> None:
    async def sleep_until_cancelled(scope: _core.CancelScope) -> None:
        with scope, fail_after(1):
            await sleep_forever()

    # Cancel before entry
    scope = _core.CancelScope()
    scope.cancel()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(sleep_until_cancelled, scope)

    # Cancel after entry
    scope = _core.CancelScope()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(sleep_until_cancelled, scope)
        await wait_all_tasks_blocked()
        scope.cancel()

    # Shield before entry
    scope = _core.CancelScope()
    scope.shield = True
    with _core.CancelScope() as outer, scope:
        outer.cancel()
        await _core.checkpoint()
        scope.shield = False
        with pytest.raises(_core.Cancelled):
            await _core.checkpoint()

    # Can't reuse
    with _core.CancelScope() as scope:
        await _core.checkpoint()
    scope.cancel()
    await _core.checkpoint()
    assert scope.cancel_called
    assert not scope.cancelled_caught
    with pytest.raises(RuntimeError) as exc_info:
        with scope:
            pass  # pragma: no cover
    assert "single 'with' block" in str(exc_info.value)

    # Can't reenter
    with _core.CancelScope() as scope:
        with pytest.raises(RuntimeError) as exc_info:
            with scope:
                pass  # pragma: no cover
        assert "single 'with' block" in str(exc_info.value)

    # Can't enter from multiple tasks simultaneously
    scope = _core.CancelScope()

    async def enter_scope() -> None:
        with scope:
            await sleep_forever()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(enter_scope, name="this one")
        await wait_all_tasks_blocked()

        with pytest.raises(RuntimeError) as exc_info:
            with scope:
                pass  # pragma: no cover
        assert "single 'with' block" in str(exc_info.value)
        nursery.cancel_scope.cancel()

    # If not yet entered, cancel_called is true when the deadline has passed
    # even if cancel() hasn't been called yet
    scope = _core.CancelScope(deadline=_core.current_time() + 1)
    assert not scope.cancel_called
    scope.deadline -= 1
    assert scope.cancel_called
    scope.deadline += 1
    assert scope.cancel_called  # never become un-cancelled


async def test_cancel_scope_misnesting() -> None:
    outer = _core.CancelScope()
    inner = _core.CancelScope()
    with ExitStack() as stack:
        stack.enter_context(outer)
        with inner:
            with pytest.raises(RuntimeError, match="still within its child"):
                stack.close()
        # No further error is raised when exiting the inner context

    # If there are other tasks inside the abandoned part of the cancel tree,
    # they get cancelled when the misnesting is detected
    async def task1() -> None:
        with pytest.raises(_core.Cancelled):
            await sleep_forever()

    # Even if inside another cancel scope
    async def task2() -> None:
        with _core.CancelScope():
            with pytest.raises(_core.Cancelled):
                await sleep_forever()

    with ExitStack() as stack:
        stack.enter_context(_core.CancelScope())
        async with _core.open_nursery() as nursery:
            nursery.start_soon(task1)
            nursery.start_soon(task2)
            await wait_all_tasks_blocked()
            with pytest.raises(RuntimeError, match="still within its child"):
                stack.close()

    # Variant that makes the child tasks direct children of the scope
    # that noticed the misnesting:
    nursery_mgr = _core.open_nursery()
    nursery = await nursery_mgr.__aenter__()
    try:
        nursery.start_soon(task1)
        nursery.start_soon(task2)
        nursery.start_soon(sleep_forever)
        await wait_all_tasks_blocked()
        nursery.cancel_scope.__exit__(None, None, None)
    finally:
        with pytest.raises(
            RuntimeError,
            match="which had already been exited",
        ) as exc_info:
            await nursery_mgr.__aexit__(*sys.exc_info())

    def no_context(exc: RuntimeError) -> bool:
        return exc.__context__ is None

    msg = "closed before the task exited"
    group = RaisesGroup(
        Matcher(RuntimeError, match=msg, check=no_context),
        Matcher(RuntimeError, match=msg, check=no_context),
        # sleep_forever
        Matcher(
            RuntimeError,
            match=msg,
            check=lambda x: isinstance(x.__context__, _core.Cancelled),
        ),
    )
    assert group.matches(exc_info.value.__context__)

    # Trying to exit a cancel scope from an unrelated task raises an error
    # without affecting any state
    async def task3(task_status: _core.TaskStatus[_core.CancelScope]) -> None:
        with _core.CancelScope() as scope:
            task_status.started(scope)
            await sleep_forever()

    async with _core.open_nursery() as nursery:
        value = await nursery.start(task3)
        assert isinstance(value, _core.CancelScope)
        scope: _core.CancelScope = value
        with pytest.raises(RuntimeError, match="from unrelated"):
            scope.__exit__(None, None, None)
        scope.cancel()


@slow
async def test_timekeeping() -> None:
    # probably a good idea to use a real clock for *one* test anyway...
    TARGET = 1.0
    # give it a few tries in case of random CI server flakiness
    for _ in range(4):
        real_start = time.perf_counter()
        with _core.CancelScope() as scope:
            scope.deadline = _core.current_time() + TARGET
            await sleep_forever()
        real_duration = time.perf_counter() - real_start
        accuracy = real_duration / TARGET
        print(accuracy)
        # Actual time elapsed should always be >= target time
        # (== is possible depending on system behavior for time.perf_counter resolution
        if 1.0 <= accuracy < 2:  # pragma: no branch
            break
    else:  # pragma: no cover
        raise AssertionError()


async def test_failed_abort() -> None:
    stubborn_task: _core.Task | None = None
    stubborn_scope: _core.CancelScope | None = None
    record: list[str] = []

    async def stubborn_sleeper() -> None:
        nonlocal stubborn_task, stubborn_scope
        stubborn_task = _core.current_task()
        with _core.CancelScope() as scope:
            stubborn_scope = scope
            record.append("sleep")
            x = await _core.wait_task_rescheduled(lambda _: _core.Abort.FAILED)
            assert x == 1
            record.append("woke")
            try:
                await _core.checkpoint_if_cancelled()
            except _core.Cancelled:
                record.append("cancelled")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(stubborn_sleeper)
        await wait_all_tasks_blocked()
        assert record == ["sleep"]
        not_none(stubborn_scope).cancel()
        await wait_all_tasks_blocked()
        # cancel didn't wake it up
        assert record == ["sleep"]
        # wake it up again by hand
        _core.reschedule(not_none(stubborn_task), outcome.Value(1))
    assert record == ["sleep", "woke", "cancelled"]


@restore_unraisablehook()
def test_broken_abort() -> None:
    async def main() -> None:
        # These yields are here to work around an annoying warning -- we're
        # going to crash the main loop, and if we (by chance) do this before
        # the run_sync_soon task runs for the first time, then Python gives us
        # a spurious warning about it not being awaited. (I mean, the warning
        # is correct, but here we're testing our ability to deliver a
        # semi-meaningful error after things have gone totally pear-shaped, so
        # it's not relevant.) By letting the run_sync_soon_task run first, we
        # avoid the warning.
        await _core.checkpoint()
        await _core.checkpoint()
        with _core.CancelScope() as scope:
            scope.cancel()
            # None is not a legal return value here
            await _core.wait_task_rescheduled(lambda _: None)  # type: ignore

    with pytest.raises(_core.TrioInternalError):
        _core.run(main)

    # Because this crashes, various __del__ methods print complaints on
    # stderr. Make sure that they get run now, so the output is attached to
    # this test.
    gc_collect_harder()


@restore_unraisablehook()
def test_error_in_run_loop() -> None:
    # Blow stuff up real good to check we at least get a TrioInternalError
    async def main() -> None:
        task = _core.current_task()
        task._schedule_points = "hello!"  # type: ignore
        await _core.checkpoint()

    with ignore_coroutine_never_awaited_warnings():
        with pytest.raises(_core.TrioInternalError):
            _core.run(main)


async def test_spawn_system_task() -> None:
    record: list[tuple[str, int]] = []

    async def system_task(x: int) -> None:
        record.append(("x", x))
        record.append(("ki", _core.currently_ki_protected()))
        await _core.checkpoint()

    _core.spawn_system_task(system_task, 1)
    await wait_all_tasks_blocked()
    assert record == [("x", 1), ("ki", True)]


# intentionally make a system task crash
def test_system_task_crash() -> None:
    async def crasher() -> NoReturn:
        raise KeyError

    async def main() -> None:
        _core.spawn_system_task(crasher)
        await sleep_forever()

    with pytest.raises(_core.TrioInternalError):
        _core.run(main)


def test_system_task_crash_ExceptionGroup() -> None:
    async def crasher1() -> NoReturn:
        raise KeyError

    async def crasher2() -> NoReturn:
        raise ValueError

    async def system_task() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher1)
            nursery.start_soon(crasher2)

    async def main() -> None:
        _core.spawn_system_task(system_task)
        await sleep_forever()

    # TrioInternalError is not wrapped
    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)

    # the first exceptiongroup is from the first nursery opened in Runner.init()
    # the second exceptiongroup is from the second nursery opened in Runner.init()
    # the third exceptongroup is from the nursery defined in `system_task` above
    assert RaisesGroup(RaisesGroup(RaisesGroup(KeyError, ValueError))).matches(
        excinfo.value.__cause__,
    )


def test_system_task_crash_plus_Cancelled() -> None:
    # Set up a situation where a system task crashes with a
    # ExceptionGroup([Cancelled, ValueError])
    async def crasher() -> None:
        try:
            await sleep_forever()
        except _core.Cancelled:
            raise ValueError from None

    async def cancelme() -> None:
        await sleep_forever()

    async def system_task() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher)
            nursery.start_soon(cancelme)

    async def main() -> None:
        _core.spawn_system_task(system_task)
        # then we exit, triggering a cancellation

    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)

    # See explanation for triple-wrap in test_system_task_crash_ExceptionGroup
    assert RaisesGroup(RaisesGroup(RaisesGroup(ValueError))).matches(
        excinfo.value.__cause__,
    )


def test_system_task_crash_KeyboardInterrupt() -> None:
    async def ki() -> NoReturn:
        raise KeyboardInterrupt

    async def main() -> None:
        _core.spawn_system_task(ki)
        await sleep_forever()

    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)
    # "Only" double-wrapped since ki() doesn't create an exceptiongroup
    assert RaisesGroup(RaisesGroup(KeyboardInterrupt)).matches(excinfo.value.__cause__)


# This used to fail because checkpoint was a yield followed by an immediate
# reschedule. So we had:
# 1) this task yields
# 2) this task is rescheduled
# ...
# 3) next iteration of event loop starts, runs timeouts
# 4) this task has timed out
# 5) ...but it's on the run queue, so the timeout is queued to be delivered
#    the next time that it's blocked.
async def test_yield_briefly_checks_for_timeout(mock_clock: _core.MockClock) -> None:
    with _core.CancelScope(deadline=_core.current_time() + 5):
        await _core.checkpoint()
        mock_clock.jump(10)
        with pytest.raises(_core.Cancelled):
            await _core.checkpoint()


# This tests that sys.exc_info is properly saved/restored as we swap between
# tasks. It turns out that the interpreter automagically handles this for us
# so there's no special code in Trio required to pass this test, but it's
# still nice to know that it works :-).
#
# Update: it turns out I was right to be nervous! see the next test...
async def test_exc_info() -> None:
    record: list[str] = []
    seq = Sequencer()

    async def child1() -> None:
        async with seq(0):
            pass  # we don't yield until seq(2) below
        record.append("child1 raise")
        with pytest.raises(ValueError, match=r"^child1$") as excinfo:
            try:
                raise ValueError("child1")
            except ValueError:
                record.append("child1 sleep")
                async with seq(2):
                    pass
                assert "child2 wake" in record
                record.append("child1 re-raise")
                raise
        assert excinfo.value.__context__ is None
        record.append("child1 success")

    async def child2() -> None:
        async with seq(1):
            pass  # we don't yield until seq(3) below
        assert "child1 sleep" in record
        record.append("child2 wake")
        assert sys.exc_info() == (None, None, None)
        with pytest.raises(KeyError) as excinfo:
            try:
                raise KeyError("child2")
            except KeyError:
                record.append("child2 sleep again")
                async with seq(3):
                    pass
                assert "child1 re-raise" in record
                record.append("child2 re-raise")
                raise
        assert excinfo.value.__context__ is None
        record.append("child2 success")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child1)
        nursery.start_soon(child2)

    assert record == [
        "child1 raise",
        "child1 sleep",
        "child2 wake",
        "child2 sleep again",
        "child1 re-raise",
        "child1 success",
        "child2 re-raise",
        "child2 success",
    ]


# On all CPython versions (at time of writing), using .throw() to raise an
# exception inside a coroutine/generator can cause the original `exc_info` state
# to be lost, so things like re-raising and exception chaining are broken unless
# Trio implements its workaround. At time of writing, CPython main (3.13-dev)
# and every CPython release (excluding releases for old Python versions not
# supported by Trio) is affected (albeit in differing ways).
#
# If the `ValueError()` gets sent in via `throw` and is suppressed, then CPython
# loses track of the original `exc_info`:
#   https://bugs.python.org/issue25612 (Example 1)
#   https://bugs.python.org/issue29587 (Example 2)
# This is fixed in CPython >= 3.7.
async def test_exc_info_after_throw_suppressed() -> None:
    child_task: _core.Task | None = None

    async def child() -> None:
        nonlocal child_task
        child_task = _core.current_task()

        try:
            raise KeyError
        except KeyError:
            with suppress(ValueError):
                await sleep_forever()
            raise

    with RaisesGroup(Matcher(KeyError, check=lambda e: e.__context__ is None)):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(child)
            await wait_all_tasks_blocked()
            _core.reschedule(not_none(child_task), outcome.Error(ValueError()))


# Similar to previous test -- if the `ValueError()` gets sent in via 'throw' and
# propagates out, then CPython doesn't set its `__context__` so normal implicit
# exception chaining is broken:
#   https://bugs.python.org/issue25612 (Example 2)
#   https://bugs.python.org/issue25683
#   https://bugs.python.org/issue29587 (Example 1)
# This is fixed in CPython >= 3.9.
async def test_exception_chaining_after_throw() -> None:
    child_task: _core.Task | None = None

    async def child() -> None:
        nonlocal child_task
        child_task = _core.current_task()

        try:
            raise KeyError
        except KeyError:
            await sleep_forever()

    with RaisesGroup(
        Matcher(
            ValueError,
            "error text",
            lambda e: isinstance(e.__context__, KeyError),
        ),
    ):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(child)
            await wait_all_tasks_blocked()
            _core.reschedule(
                not_none(child_task),
                outcome.Error(ValueError("error text")),
            )


# Similar to previous tests -- if the `ValueError()` gets sent into an inner
# `await` via 'throw' and is suppressed there, then CPython loses track of
# `exc_info` in the inner coroutine:
#   https://github.com/python/cpython/issues/108668
# This is unfixed in CPython at time of writing.
async def test_exc_info_after_throw_to_inner_suppressed() -> None:
    child_task: _core.Task | None = None

    async def child() -> None:
        nonlocal child_task
        child_task = _core.current_task()

        try:
            raise KeyError
        except KeyError as exc:
            await inner(exc)
            raise

    async def inner(exc: BaseException) -> None:
        with suppress(ValueError):
            await sleep_forever()
        assert not_none(sys.exc_info()[1]) is exc

    with RaisesGroup(Matcher(KeyError, check=lambda e: e.__context__ is None)):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(child)
            await wait_all_tasks_blocked()
            _core.reschedule(not_none(child_task), outcome.Error(ValueError()))


# Similar to previous tests -- if the `ValueError()` gets sent into an inner
# `await` via `throw` and propagates out, then CPython incorrectly sets its
# `__context__` so normal implicit exception chaining is broken:
#   https://bugs.python.org/issue40694
# This is unfixed in CPython at time of writing.
async def test_exception_chaining_after_throw_to_inner() -> None:
    child_task: _core.Task | None = None

    async def child() -> None:
        nonlocal child_task
        child_task = _core.current_task()

        try:
            raise KeyError
        except KeyError:
            await inner()

    async def inner() -> None:
        try:
            raise IndexError
        except IndexError:
            await sleep_forever()

    with RaisesGroup(
        Matcher(
            ValueError,
            "^Unique Text$",
            lambda e: isinstance(e.__context__, IndexError)
            and isinstance(e.__context__.__context__, KeyError),
        ),
    ):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(child)
            await wait_all_tasks_blocked()
            _core.reschedule(
                not_none(child_task),
                outcome.Error(ValueError("Unique Text")),
            )


async def test_nursery_exception_chaining_doesnt_make_context_loops() -> None:
    async def crasher() -> NoReturn:
        raise KeyError

    # the ExceptionGroup should not have the KeyError or ValueError as context
    with RaisesGroup(ValueError, KeyError, check=lambda x: x.__context__ is None):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(crasher)
            raise ValueError


def test_TrioToken_identity() -> None:
    async def get_and_check_token() -> _core.TrioToken:
        token = _core.current_trio_token()
        # Two calls in the same run give the same object
        assert token is _core.current_trio_token()
        return token

    t1 = _core.run(get_and_check_token)
    t2 = _core.run(get_and_check_token)
    assert t1 is not t2
    assert t1 != t2
    assert hash(t1) != hash(t2)


async def test_TrioToken_run_sync_soon_basic() -> None:
    record: list[tuple[str, int]] = []

    def cb(x: int) -> None:
        record.append(("cb", x))

    token = _core.current_trio_token()
    token.run_sync_soon(cb, 1)
    assert not record
    await wait_all_tasks_blocked()
    assert record == [("cb", 1)]


def test_TrioToken_run_sync_soon_too_late() -> None:
    token: _core.TrioToken | None = None

    async def main() -> None:
        nonlocal token
        token = _core.current_trio_token()

    _core.run(main)
    with pytest.raises(_core.RunFinishedError):
        not_none(token).run_sync_soon(lambda: None)  # pragma: no branch


async def test_TrioToken_run_sync_soon_idempotent() -> None:
    record: list[int] = []

    def cb(x: int) -> None:
        record.append(x)

    token = _core.current_trio_token()
    token.run_sync_soon(cb, 1)
    token.run_sync_soon(cb, 1, idempotent=True)
    token.run_sync_soon(cb, 1, idempotent=True)
    token.run_sync_soon(cb, 1, idempotent=True)
    token.run_sync_soon(cb, 2, idempotent=True)
    token.run_sync_soon(cb, 2, idempotent=True)
    await wait_all_tasks_blocked()
    assert len(record) == 3
    assert sorted(record) == [1, 1, 2]

    # ordering test
    record = []
    for _ in range(3):
        for i in range(100):
            token.run_sync_soon(cb, i, idempotent=True)
    await wait_all_tasks_blocked()
    # We guarantee FIFO
    assert record == list(range(100))


def test_TrioToken_run_sync_soon_idempotent_requeue() -> None:
    # We guarantee that if a call has finished, queueing it again will call it
    # again. Due to the lack of synchronization, this effectively means that
    # we have to guarantee that once a call has *started*, queueing it again
    # will call it again. Also this is much easier to test :-)
    record: list[None] = []

    def redo(token: _core.TrioToken) -> None:
        record.append(None)
        with suppress(_core.RunFinishedError):
            token.run_sync_soon(redo, token, idempotent=True)

    async def main() -> None:
        token = _core.current_trio_token()
        token.run_sync_soon(redo, token, idempotent=True)
        await _core.checkpoint()
        await _core.checkpoint()
        await _core.checkpoint()

    _core.run(main)

    assert len(record) >= 2


def test_TrioToken_run_sync_soon_after_main_crash() -> None:
    record: list[str] = []

    async def main() -> None:
        token = _core.current_trio_token()
        # After main exits but before finally cleaning up, callback processed
        # normally
        token.run_sync_soon(lambda: record.append("sync-cb"))
        raise ValueError("error text")

    with pytest.raises(ValueError, match=r"^error text$"):
        _core.run(main)

    assert record == ["sync-cb"]


def test_TrioToken_run_sync_soon_crashes() -> None:
    record: set[str] = set()

    async def main() -> None:
        token = _core.current_trio_token()
        token.run_sync_soon(lambda: {}["nope"])  # type: ignore[index]
        # check that a crashing run_sync_soon callback doesn't stop further
        # calls to run_sync_soon
        token.run_sync_soon(lambda: record.add("2nd run_sync_soon ran"))
        try:
            await sleep_forever()
        except _core.Cancelled:
            record.add("cancelled!")

    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)
    # the first exceptiongroup is from the first nursery opened in Runner.init()
    # the second exceptiongroup is from the second nursery opened in Runner.init()
    assert RaisesGroup(RaisesGroup(KeyError)).matches(excinfo.value.__cause__)
    assert record == {"2nd run_sync_soon ran", "cancelled!"}


async def test_TrioToken_run_sync_soon_FIFO() -> None:
    N = 100
    record = []
    token = _core.current_trio_token()
    for i in range(N):
        token.run_sync_soon(lambda j: record.append(j), i)
    await wait_all_tasks_blocked()
    assert record == list(range(N))


def test_TrioToken_run_sync_soon_starvation_resistance() -> None:
    # Even if we push callbacks in from callbacks, so that the callback queue
    # never empties out, then we still can't starve out other tasks from
    # running.
    token: _core.TrioToken | None = None
    record: list[tuple[str, int]] = []

    def naughty_cb(i: int) -> None:
        try:
            not_none(token).run_sync_soon(naughty_cb, i + 1)
        except _core.RunFinishedError:
            record.append(("run finished", i))

    async def main() -> None:
        nonlocal token
        token = _core.current_trio_token()
        token.run_sync_soon(naughty_cb, 0)
        record.append(("starting", 0))
        for _ in range(20):
            await _core.checkpoint()

    _core.run(main)
    assert len(record) == 2
    assert record[0] == ("starting", 0)
    assert record[1][0] == "run finished"
    assert record[1][1] >= 19


def test_TrioToken_run_sync_soon_threaded_stress_test() -> None:
    cb_counter = 0

    def cb() -> None:
        nonlocal cb_counter
        cb_counter += 1

    def stress_thread(token: _core.TrioToken) -> None:
        try:
            while True:
                token.run_sync_soon(cb)
                time.sleep(0)
        except _core.RunFinishedError:
            pass

    async def main() -> None:
        token = _core.current_trio_token()
        thread = threading.Thread(target=stress_thread, args=(token,))
        thread.start()
        for _ in range(10):
            start_value = cb_counter
            while cb_counter == start_value:
                await sleep(0.01)

    _core.run(main)
    print(cb_counter)


async def test_TrioToken_run_sync_soon_massive_queue() -> None:
    # There are edge cases in the wakeup fd code when the wakeup fd overflows,
    # so let's try to make that happen. This is also just a good stress test
    # in general. (With the current-as-of-2017-02-14 code using a socketpair
    # with minimal buffer, Linux takes 6 wakeups to fill the buffer and macOS
    # takes 1 wakeup. So 1000 is overkill if anything. Windows OTOH takes
    # ~600,000 wakeups, but has the same code paths...)
    COUNT = 1000
    token = _core.current_trio_token()
    counter = [0]

    def cb(i: int) -> None:
        # This also tests FIFO ordering of callbacks
        assert counter[0] == i
        counter[0] += 1

    for i in range(COUNT):
        token.run_sync_soon(cb, i)
    await wait_all_tasks_blocked()
    assert counter[0] == COUNT


def test_TrioToken_run_sync_soon_late_crash() -> None:
    # Crash after system nursery is closed -- easiest way to do that is
    # from an async generator finalizer.
    record: list[str] = []
    saved: list[AsyncGenerator[int, None]] = []

    async def agen() -> AsyncGenerator[int, None]:
        token = _core.current_trio_token()
        try:
            yield 1
        finally:
            token.run_sync_soon(lambda: {}["nope"])  # type: ignore[index]
            token.run_sync_soon(lambda: record.append("2nd ran"))

    async def main() -> None:
        saved.append(agen())
        await saved[-1].asend(None)
        record.append("main exiting")

    with pytest.raises(_core.TrioInternalError) as excinfo:
        _core.run(main)

    assert RaisesGroup(KeyError).matches(excinfo.value.__cause__)
    assert record == ["main exiting", "2nd ran"]


async def test_slow_abort_basic() -> None:
    with _core.CancelScope() as scope:
        scope.cancel()

        task = _core.current_task()
        token = _core.current_trio_token()

        def slow_abort(raise_cancel: _core.RaiseCancelT) -> _core.Abort:
            result = outcome.capture(raise_cancel)
            token.run_sync_soon(_core.reschedule, task, result)
            return _core.Abort.FAILED

        with pytest.raises(_core.Cancelled):
            await _core.wait_task_rescheduled(slow_abort)


async def test_slow_abort_edge_cases() -> None:
    record: list[str] = []

    async def slow_aborter() -> None:
        task = _core.current_task()
        token = _core.current_trio_token()

        def slow_abort(raise_cancel: _core.RaiseCancelT) -> _core.Abort:
            record.append("abort-called")
            result = outcome.capture(raise_cancel)
            token.run_sync_soon(_core.reschedule, task, result)
            return _core.Abort.FAILED

        record.append("sleeping")
        with pytest.raises(_core.Cancelled):
            await _core.wait_task_rescheduled(slow_abort)
        record.append("cancelled")
        # blocking again, this time it's okay, because we're shielded
        await _core.checkpoint()
        record.append("done")

    with _core.CancelScope() as outer1:
        with _core.CancelScope() as outer2:
            async with _core.open_nursery() as nursery:
                # So we have a task blocked on an operation that can't be
                # aborted immediately
                nursery.start_soon(slow_aborter)
                await wait_all_tasks_blocked()
                assert record == ["sleeping"]
                # And then we cancel it, so the abort callback gets run
                outer1.cancel()
                assert record == ["sleeping", "abort-called"]
                # In fact that happens twice! (This used to cause the abort
                # callback to be run twice)
                outer2.cancel()
                assert record == ["sleeping", "abort-called"]
                # But then before the abort finishes, the task gets shielded!
                nursery.cancel_scope.shield = True
                # Now we wait for the task to finish...
            # The cancellation was delivered, even though it was shielded
            assert record == ["sleeping", "abort-called", "cancelled", "done"]


async def test_task_tree_introspection() -> None:
    tasks: dict[str, _core.Task] = {}
    nurseries: dict[str, _core.Nursery] = {}

    async def parent(
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        tasks["parent"] = _core.current_task()

        assert tasks["parent"].child_nurseries == []

        async with _core.open_nursery() as nursery1:
            async with _core.open_nursery() as nursery2:
                assert tasks["parent"].child_nurseries == [nursery1, nursery2]

        assert tasks["parent"].child_nurseries == []

        nursery: _core.Nursery | None
        async with _core.open_nursery() as nursery:
            nurseries["parent"] = nursery
            await nursery.start(child1)

        # Upward links survive after tasks/nurseries exit
        assert nurseries["parent"].parent_task is tasks["parent"]
        assert tasks["child1"].parent_nursery is nurseries["parent"]
        assert nurseries["child1"].parent_task is tasks["child1"]
        assert tasks["child2"].parent_nursery is nurseries["child1"]

        nursery = _core.current_task().parent_nursery
        # Make sure that chaining eventually gives a nursery of None (and not,
        # for example, an error)
        while nursery is not None:
            t = nursery.parent_task
            nursery = t.parent_nursery

    async def child2() -> None:
        tasks["child2"] = _core.current_task()
        assert tasks["parent"].child_nurseries == [nurseries["parent"]]
        assert nurseries["parent"].child_tasks == frozenset({tasks["child1"]})
        assert tasks["child1"].child_nurseries == [nurseries["child1"]]
        assert nurseries["child1"].child_tasks == frozenset({tasks["child2"]})
        assert tasks["child2"].child_nurseries == []

    async def child1(
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        me = tasks["child1"] = _core.current_task()
        assert not_none(me.parent_nursery).parent_task is tasks["parent"]
        assert me.parent_nursery is not nurseries["parent"]
        assert me.eventual_parent_nursery is nurseries["parent"]
        task_status.started()
        assert me.parent_nursery is nurseries["parent"]
        assert me.eventual_parent_nursery is None

        # Wait for the start() call to return and close its internal nursery, to
        # ensure consistent results in child2:
        await _core.wait_all_tasks_blocked()

        async with _core.open_nursery() as nursery:
            nurseries["child1"] = nursery
            nursery.start_soon(child2)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(parent)

    # There are no pending starts, so no one should have a non-None
    # eventual_parent_nursery
    for task in tasks.values():
        assert task.eventual_parent_nursery is None


async def test_nursery_closure() -> None:
    async def child1(nursery: _core.Nursery) -> None:
        # We can add new tasks to the nursery even after entering __aexit__,
        # so long as there are still tasks running
        nursery.start_soon(child2)

    async def child2() -> None:
        pass

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child1, nursery)

    # But once we've left __aexit__, the nursery is closed
    with pytest.raises(RuntimeError):
        nursery.start_soon(child2)


async def test_spawn_name() -> None:
    async def func1(expected: str) -> None:
        task = _core.current_task()
        assert expected in task.name

    async def func2() -> None:  # pragma: no cover
        pass

    async def check(  # type: ignore[explicit-any]
        spawn_fn: Callable[..., object],
    ) -> None:
        spawn_fn(func1, "func1")
        spawn_fn(func1, "func2", name=func2)
        spawn_fn(func1, "func3", name="func3")
        spawn_fn(functools.partial(func1, "func1"))
        spawn_fn(func1, "object", name=object())

    async with _core.open_nursery() as nursery:
        await check(nursery.start_soon)
    await check(_core.spawn_system_task)


async def test_current_effective_deadline(mock_clock: _core.MockClock) -> None:
    assert _core.current_effective_deadline() == inf

    with _core.CancelScope(deadline=5) as scope1:
        with _core.CancelScope(deadline=10) as scope2:
            assert _core.current_effective_deadline() == 5
            scope2.deadline = 3
            assert _core.current_effective_deadline() == 3
            scope2.deadline = 10
            assert _core.current_effective_deadline() == 5
            scope2.shield = True
            assert _core.current_effective_deadline() == 10
            scope2.shield = False
            assert _core.current_effective_deadline() == 5
            scope1.cancel()
            assert _core.current_effective_deadline() == -inf
            scope2.shield = True
            assert _core.current_effective_deadline() == 10
        assert _core.current_effective_deadline() == -inf
    assert _core.current_effective_deadline() == inf


def test_nice_error_on_bad_calls_to_run_or_spawn() -> None:
    def bad_call_run(  # type: ignore[explicit-any]
        func: Callable[..., Awaitable[object]],
        *args: tuple[object, ...],
    ) -> None:
        _core.run(func, *args)

    def bad_call_spawn(  # type: ignore[explicit-any]
        func: Callable[..., Awaitable[object]],
        *args: tuple[object, ...],
    ) -> None:
        async def main() -> None:
            async with _core.open_nursery() as nursery:
                nursery.start_soon(func, *args)

        _core.run(main)

    async def f() -> None:  # pragma: no cover
        pass

    async def async_gen(arg: T) -> AsyncGenerator[T, None]:  # pragma: no cover
        yield arg

    # If/when RaisesGroup/Matcher is added to pytest in some form this test can be
    # rewritten to use a loop again, and avoid specifying the exceptions twice in
    # different ways
    with pytest.raises(
        TypeError,
        match=r"^Trio was expecting an async function, but instead it got a coroutine object <.*>",
    ):
        bad_call_run(f())  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match=r"expected an async function but got an async generator",
    ):
        bad_call_run(async_gen, 0)  # type: ignore

    # bad_call_spawn calls the function inside a nursery, so the exception will be
    # wrapped in an exceptiongroup
    with RaisesGroup(Matcher(TypeError, "expecting an async function")):
        bad_call_spawn(f())  # type: ignore[arg-type]

    with RaisesGroup(
        Matcher(TypeError, "expected an async function but got an async generator"),
    ):
        bad_call_spawn(async_gen, 0)  # type: ignore


def test_calling_asyncio_function_gives_nice_error() -> None:
    async def child_xyzzy() -> None:
        await create_asyncio_future_in_new_loop()

    async def misguided() -> None:
        await child_xyzzy()

    with pytest.raises(TypeError, match="asyncio") as excinfo:
        _core.run(misguided)

    # The traceback should point to the location of the foreign await
    assert any(  # pragma: no branch
        entry.name == "child_xyzzy" for entry in excinfo.traceback
    )


async def test_asyncio_function_inside_nursery_does_not_explode() -> None:
    # Regression test for https://github.com/python-trio/trio/issues/552
    with RaisesGroup(Matcher(TypeError, "asyncio")):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleep_forever)
            await create_asyncio_future_in_new_loop()


async def test_trivial_yields() -> None:
    with assert_checkpoints():
        await _core.checkpoint()

    with assert_checkpoints():
        await _core.checkpoint_if_cancelled()
        await _core.cancel_shielded_checkpoint()

    # Weird case: opening and closing a nursery schedules, but doesn't check
    # for cancellation (unless something inside the nursery does)
    task = _core.current_task()
    before_schedule_points = task._schedule_points
    with _core.CancelScope() as cs:
        cs.cancel()
        async with _core.open_nursery():
            pass
    assert not cs.cancelled_caught
    assert task._schedule_points > before_schedule_points

    before_schedule_points = task._schedule_points

    async def noop_with_no_checkpoint() -> None:
        pass

    with _core.CancelScope() as cs:
        cs.cancel()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(noop_with_no_checkpoint)
    assert not cs.cancelled_caught

    assert task._schedule_points > before_schedule_points

    with _core.CancelScope() as cancel_scope:
        cancel_scope.cancel()
        with RaisesGroup(KeyError):
            async with _core.open_nursery():
                raise KeyError


async def test_nursery_start(autojump_clock: _core.MockClock) -> None:
    async def no_args() -> None:  # pragma: no cover
        pass

    # Errors in calling convention get raised immediately from start
    async with _core.open_nursery() as nursery:
        with pytest.raises(TypeError):
            await nursery.start(no_args)

    async def sleep_then_start(
        seconds: int,
        *,
        task_status: _core.TaskStatus[int] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        repr(task_status)  # smoke test
        await sleep(seconds)
        task_status.started(seconds)
        await sleep(seconds)

    # Basic happy-path check: start waits for the task to call started(), then
    # returns, passes back the value, and the given nursery then waits for it
    # to exit.
    for seconds in [1, 2]:
        async with _core.open_nursery() as nursery:
            assert len(nursery.child_tasks) == 0
            t0 = _core.current_time()
            assert await nursery.start(sleep_then_start, seconds) == seconds
            assert _core.current_time() - t0 == seconds
            assert len(nursery.child_tasks) == 1
        assert _core.current_time() - t0 == 2 * seconds

    # Make sure TASK_STATUS_IGNORED works so task function can be called
    # directly
    t0 = _core.current_time()
    await sleep_then_start(3)
    assert _core.current_time() - t0 == 2 * 3

    # calling started twice
    async def double_started(
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started()
        with pytest.raises(RuntimeError):
            task_status.started()

    async with _core.open_nursery() as nursery:
        await nursery.start(double_started)

    # child crashes before calling started -> error comes out of .start()
    async def raise_keyerror(
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        raise KeyError("oops")

    async with _core.open_nursery() as nursery:
        with pytest.raises(KeyError):
            await nursery.start(raise_keyerror)

    # child exiting cleanly before calling started -> triggers a RuntimeError
    async def nothing(
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        return

    async with _core.open_nursery() as nursery:
        with pytest.raises(RuntimeError) as excinfo1:
            await nursery.start(nothing)
        assert "exited without calling" in str(excinfo1.value)

    # if the call to start() is cancelled, then the call to started() does
    # nothing -- the child keeps executing under start(). The value it passed
    # is ignored; start() raises Cancelled.
    async def just_started(
        *,
        task_status: _core.TaskStatus[str] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started("hi")
        await _core.checkpoint()

    async with _core.open_nursery() as nursery:
        with _core.CancelScope() as cs:
            cs.cancel()
            with pytest.raises(_core.Cancelled):
                await nursery.start(just_started)

    # but if the task does not execute any checkpoints, and exits, then start()
    # doesn't raise Cancelled, since the task completed successfully.
    async def started_with_no_checkpoint(
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started(None)

    async with _core.open_nursery() as nursery:
        with _core.CancelScope() as cs:
            cs.cancel()
            await nursery.start(started_with_no_checkpoint)
        assert not cs.cancelled_caught

    # and since starting in a cancelled context makes started() a no-op, if
    # the child crashes after calling started(), the error can *still* come
    # out of start()
    async def raise_keyerror_after_started(
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        task_status.started()
        raise KeyError("whoopsiedaisy")

    async with _core.open_nursery() as nursery:
        with _core.CancelScope() as cs:
            cs.cancel()
            with pytest.raises(KeyError):
                await nursery.start(raise_keyerror_after_started)

    # trying to start in a closed nursery raises an error immediately
    async with _core.open_nursery() as closed_nursery:
        pass
    t0 = _core.current_time()
    with pytest.raises(RuntimeError):
        await closed_nursery.start(sleep_then_start, 7)
    # sub-second delays can be caused by unrelated multitasking by an OS
    assert int(_core.current_time()) == int(t0)


async def test_task_nursery_stack() -> None:
    task = _core.current_task()
    assert task._child_nurseries == []
    async with _core.open_nursery() as nursery1:
        assert task._child_nurseries == [nursery1]
        with RaisesGroup(KeyError):
            async with _core.open_nursery() as nursery2:
                assert task._child_nurseries == [nursery1, nursery2]
                raise KeyError
        assert task._child_nurseries == [nursery1]
    assert task._child_nurseries == []


async def test_nursery_start_with_cancelled_nursery() -> None:
    # This function isn't testing task_status, it's using task_status as a
    # convenient way to get a nursery that we can test spawning stuff into.
    async def setup_nursery(
        task_status: _core.TaskStatus[_core.Nursery] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        async with _core.open_nursery() as nursery:
            task_status.started(nursery)
            await sleep_forever()

    # Calls started() while children are asleep, so we can make sure
    # that the cancellation machinery notices and aborts when a sleeping task
    # is moved into a cancelled scope.
    async def sleeping_children(
        fn: Callable[[], object],
        *,
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleep_forever)
            nursery.start_soon(sleep_forever)
            await wait_all_tasks_blocked()
            fn()
            task_status.started()

    # Cancelling the setup_nursery just *before* calling started()
    async with _core.open_nursery() as nursery:
        value = await nursery.start(setup_nursery)
        assert isinstance(value, _core.Nursery)
        target_nursery: _core.Nursery = value
        await target_nursery.start(
            sleeping_children,
            target_nursery.cancel_scope.cancel,
        )

    # Cancelling the setup_nursery just *after* calling started()
    async with _core.open_nursery() as nursery:
        value = await nursery.start(setup_nursery)
        assert isinstance(value, _core.Nursery)
        target_nursery = value
        await target_nursery.start(sleeping_children, lambda: None)
        target_nursery.cancel_scope.cancel()


async def test_nursery_start_keeps_nursery_open(
    autojump_clock: _core.MockClock,
) -> None:
    async def sleep_a_bit(
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        await sleep(2)
        task_status.started()
        await sleep(3)

    async with _core.open_nursery() as nursery1:
        t0 = _core.current_time()
        async with _core.open_nursery() as nursery2:
            # Start the 'start' call running in the background
            nursery1.start_soon(nursery2.start, sleep_a_bit)
            # Sleep a bit
            await sleep(1)
            # Start another one.
            nursery1.start_soon(nursery2.start, sleep_a_bit)
            # Then exit this nursery. At this point, there are no tasks
            # present in this nursery -- the only thing keeping it open is
            # that the tasks will be placed into it soon, when they call
            # started().
        assert _core.current_time() - t0 == 6

    # Check that it still works even if the task that the nursery is waiting
    # for ends up crashing, and never actually enters the nursery.
    async def sleep_then_crash(
        task_status: _core.TaskStatus[None] = _core.TASK_STATUS_IGNORED,
    ) -> None:
        await sleep(7)
        raise KeyError

    async def start_sleep_then_crash(nursery: _core.Nursery) -> None:
        with pytest.raises(KeyError):
            await nursery.start(sleep_then_crash)

    async with _core.open_nursery() as nursery1:
        t0 = _core.current_time()
        async with _core.open_nursery() as nursery2:
            nursery1.start_soon(start_sleep_then_crash, nursery2)
            await wait_all_tasks_blocked()
        assert _core.current_time() - t0 == 7


async def test_nursery_explicit_exception() -> None:
    with RaisesGroup(KeyError):
        async with _core.open_nursery():
            raise KeyError()


async def test_nursery_stop_iteration() -> None:
    async def fail() -> NoReturn:
        raise ValueError

    with RaisesGroup(StopIteration, ValueError):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(fail)
            raise StopIteration


async def test_nursery_stop_async_iteration() -> None:
    class it:
        def __init__(self, count: int) -> None:
            self.count = count
            self.val = 0

        async def __anext__(self) -> int:
            await sleep(0)
            val = self.val
            if val >= self.count:
                raise StopAsyncIteration
            self.val += 1
            return val

    class async_zip:
        def __init__(self, *largs: it) -> None:
            self.nexts = [obj.__anext__ for obj in largs]

        async def _accumulate(
            self,
            f: Callable[[], Awaitable[int]],
            items: list[int],
            i: int,
        ) -> None:
            items[i] = await f()

        def __aiter__(self) -> async_zip:
            return self

        async def __anext__(self) -> list[int]:
            nexts = self.nexts
            items: list[int] = [-1] * len(nexts)

            try:
                async with _core.open_nursery() as nursery:
                    for i, f in enumerate(nexts):
                        nursery.start_soon(self._accumulate, f, items, i)
            except ExceptionGroup as e:
                # With strict_exception_groups enabled, users now need to unwrap
                # StopAsyncIteration and re-raise it.
                # This would be relatively clean on python3.11+ with except*.
                # We could also use RaisesGroup, but that's primarily meant as
                # test infra, not as a runtime tool.
                if len(e.exceptions) == 1 and isinstance(
                    e.exceptions[0],
                    StopAsyncIteration,
                ):
                    raise e.exceptions[0] from None
                else:  # pragma: no cover
                    raise AssertionError("unknown error in _accumulate") from e

            return items

    result: list[list[int]] = [vals async for vals in async_zip(it(4), it(2))]
    assert result == [[0, 0], [1, 1]]


async def test_traceback_frame_removal() -> None:
    async def my_child_task() -> NoReturn:
        raise KeyError()

    def check_traceback(exc: KeyError) -> bool:
        # The top frame in the exception traceback should be inside the child
        # task, not trio/contextvars internals. And there's only one frame
        # inside the child task, so this will also detect if our frame-removal
        # is too eager.
        tb = exc.__traceback__
        assert tb is not None
        return tb.tb_frame.f_code is my_child_task.__code__

    expected_exception = Matcher(KeyError, check=check_traceback)

    with RaisesGroup(expected_exception, expected_exception):
        # Trick: For now cancel/nursery scopes still leave a bunch of tb gunk
        # behind. But if there's an ExceptionGroup, they leave it on the group,
        # which lets us get a clean look at the KeyError itself. Someday I
        # guess this will always be an ExceptionGroup (#611), but for now we can
        # force it by raising two exceptions.
        async with _core.open_nursery() as nursery:
            nursery.start_soon(my_child_task)
            nursery.start_soon(my_child_task)


def test_contextvar_support() -> None:
    var: contextvars.ContextVar[str] = contextvars.ContextVar("test")
    var.set("before")

    assert var.get() == "before"

    async def inner() -> None:
        task = _core.current_task()
        assert task.context.get(var) == "before"
        assert var.get() == "before"
        var.set("after")
        assert var.get() == "after"
        assert var in task.context
        assert task.context.get(var) == "after"

    _core.run(inner)
    assert var.get() == "before"


async def test_contextvar_multitask() -> None:
    var = contextvars.ContextVar("test", default="hmmm")

    async def t1() -> None:
        assert var.get() == "hmmm"
        var.set("hmmmm")
        assert var.get() == "hmmmm"

    async def t2() -> None:
        assert var.get() == "hmmmm"

    async with _core.open_nursery() as n:
        n.start_soon(t1)
        await wait_all_tasks_blocked()
        assert var.get() == "hmmm"
        var.set("hmmmm")
        n.start_soon(t2)
        await wait_all_tasks_blocked()


def test_system_task_contexts() -> None:
    cvar: contextvars.ContextVar[str] = contextvars.ContextVar("qwilfish")
    cvar.set("water")

    async def system_task() -> None:
        assert cvar.get() == "water"

    async def regular_task() -> None:
        assert cvar.get() == "poison"

    async def inner() -> None:
        async with _core.open_nursery() as nursery:
            cvar.set("poison")
            nursery.start_soon(regular_task)
            _core.spawn_system_task(system_task)
            await wait_all_tasks_blocked()

    _core.run(inner)


async def test_Nursery_init() -> None:
    """Test that nurseries cannot be constructed directly."""
    # This function is async so that we have access to a task object we can
    # pass in. It should never be accessed though.
    task = _core.current_task()
    scope = _core.CancelScope()
    with pytest.raises(TypeError):
        _core._run.Nursery(task, scope, True)


async def test_Nursery_private_init() -> None:
    # context manager creation should not raise
    async with _core.open_nursery() as nursery:
        assert not nursery._closed


def test_Nursery_subclass() -> None:
    with pytest.raises(TypeError):
        type("Subclass", (_core._run.Nursery,), {})


def test_Cancelled_init() -> None:
    with pytest.raises(TypeError):
        raise _core.Cancelled

    with pytest.raises(TypeError):
        _core.Cancelled()

    # private constructor should not raise
    _core.Cancelled._create()


def test_Cancelled_str() -> None:
    cancelled = _core.Cancelled._create()
    assert str(cancelled) == "Cancelled"


def test_Cancelled_subclass() -> None:
    with pytest.raises(TypeError):
        type("Subclass", (_core.Cancelled,), {})


def test_CancelScope_subclass() -> None:
    with pytest.raises(TypeError):
        type("Subclass", (_core.CancelScope,), {})


def test_sniffio_integration() -> None:
    with pytest.raises(sniffio.AsyncLibraryNotFoundError):
        sniffio.current_async_library()

    async def check_inside_trio() -> None:
        assert sniffio.current_async_library() == "trio"

    def check_function_returning_coroutine() -> Awaitable[object]:
        assert sniffio.current_async_library() == "trio"
        return check_inside_trio()

    _core.run(check_inside_trio)

    with pytest.raises(sniffio.AsyncLibraryNotFoundError):
        sniffio.current_async_library()

    @contextmanager
    def alternate_sniffio_library() -> Generator[None, None, None]:
        prev_token = sniffio.current_async_library_cvar.set("nullio")
        prev_library, sniffio.thread_local.name = sniffio.thread_local.name, "nullio"
        try:
            yield
            assert sniffio.current_async_library() == "nullio"
        finally:
            sniffio.thread_local.name = prev_library
            sniffio.current_async_library_cvar.reset(prev_token)

    async def check_new_task_resets_sniffio_library() -> None:
        with alternate_sniffio_library():
            _core.spawn_system_task(check_inside_trio)
        async with _core.open_nursery() as nursery:
            with alternate_sniffio_library():
                nursery.start_soon(check_inside_trio)
                nursery.start_soon(check_function_returning_coroutine)

    _core.run(check_new_task_resets_sniffio_library)


async def test_Task_custom_sleep_data() -> None:
    task = _core.current_task()
    assert task.custom_sleep_data is None
    task.custom_sleep_data = 1
    assert task.custom_sleep_data == 1
    await _core.checkpoint()
    assert task.custom_sleep_data is None


@types.coroutine
def async_yield(value: T) -> Generator[T, None, None]:
    yield value


async def test_permanently_detach_coroutine_object() -> None:
    task: _core.Task | None = None
    pdco_outcome: outcome.Outcome[str] | None = None

    async def detachable_coroutine(
        task_outcome: outcome.Outcome[None],
        yield_value: object,
    ) -> None:
        await sleep(0)
        nonlocal task, pdco_outcome
        task = _core.current_task()
        # `No overload variant of "acapture" matches argument types "Callable[[Outcome[object]], Coroutine[Any, Any, object]]", "Outcome[None]"`
        pdco_outcome = await outcome.acapture(  # type: ignore[call-overload]
            _core.permanently_detach_coroutine_object,
            task_outcome,
        )
        await async_yield(yield_value)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(detachable_coroutine, outcome.Value(None), "I'm free!")

    # If we get here then Trio thinks the task has exited... but the coroutine
    # is still iterable. At that point anything can be sent into the coroutine, so the .coro type
    # is wrong.
    assert pdco_outcome is None
    # `Argument 1 to "send" of "Coroutine" has incompatible type "str"; expected "Outcome[object]"`
    assert not_none(task).coro.send("be free!") == "I'm free!"  # type: ignore[arg-type]
    assert pdco_outcome == outcome.Value("be free!")
    with pytest.raises(StopIteration):
        not_none(task).coro.send(None)  # type: ignore[arg-type]

    # Check the exception paths too
    task = None
    pdco_outcome = None
    with RaisesGroup(KeyError):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(detachable_coroutine, outcome.Error(KeyError()), "uh oh")
    throw_in = ValueError()
    assert isinstance(task, _core.Task)  # For type checkers.
    assert not_none(task).coro.throw(throw_in) == "uh oh"
    assert pdco_outcome == outcome.Error(throw_in)
    with pytest.raises(StopIteration):
        task.coro.send(None)

    async def bad_detach() -> None:
        async with _core.open_nursery():
            with pytest.raises(RuntimeError) as excinfo:
                await _core.permanently_detach_coroutine_object(outcome.Value(None))
            assert "open nurser" in str(excinfo.value)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(bad_detach)


async def test_detach_and_reattach_coroutine_object() -> None:
    unrelated_task: _core.Task | None = None
    task: _core.Task | None = None

    async def unrelated_coroutine() -> None:
        nonlocal unrelated_task
        unrelated_task = _core.current_task()

    async def reattachable_coroutine() -> None:
        nonlocal task
        await sleep(0)

        task = _core.current_task()

        def abort_fn(_: _core.RaiseCancelT) -> _core.Abort:  # pragma: no cover
            return _core.Abort.FAILED

        got = await _core.temporarily_detach_coroutine_object(abort_fn)
        assert got == "not trio!"

        await async_yield(1)
        await async_yield(2)

        with pytest.raises(RuntimeError) as excinfo:
            await _core.reattach_detached_coroutine_object(
                not_none(unrelated_task),
                None,
            )
        assert "does not match" in str(excinfo.value)

        await _core.reattach_detached_coroutine_object(task, "byebye")

        await sleep(0)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(unrelated_coroutine)
        nursery.start_soon(reattachable_coroutine)
        await wait_all_tasks_blocked()

        # Okay, it's detached. Here's our coroutine runner:
        # `Argument 1 to "send" of "Coroutine" has incompatible type "str"; expected "Outcome[object]"`
        assert not_none(task).coro.send("not trio!") == 1  # type: ignore[arg-type]
        assert not_none(task).coro.send(None) == 2  # type: ignore[arg-type]
        assert not_none(task).coro.send(None) == "byebye"  # type: ignore[arg-type]

        # Now it's been reattached, and we can leave the nursery


async def test_detached_coroutine_cancellation() -> None:
    abort_fn_called = False
    task: _core.Task | None = None

    async def reattachable_coroutine() -> None:
        await sleep(0)

        nonlocal task
        task = _core.current_task()

        def abort_fn(_: _core.RaiseCancelT) -> _core.Abort:
            nonlocal abort_fn_called
            abort_fn_called = True
            return _core.Abort.FAILED

        await _core.temporarily_detach_coroutine_object(abort_fn)
        await _core.reattach_detached_coroutine_object(task, None)
        with pytest.raises(_core.Cancelled):
            await sleep(0)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(reattachable_coroutine)
        await wait_all_tasks_blocked()
        assert task is not None
        nursery.cancel_scope.cancel()
        # `Argument 1 to "send" of "Coroutine" has incompatible type "None"; expected "Outcome[object]"`
        task.coro.send(None)  # type: ignore[arg-type]

    assert abort_fn_called


@restore_unraisablehook()
def test_async_function_implemented_in_C() -> None:
    # These used to crash because we'd try to mutate the coroutine object's
    # cr_frame, but C functions don't have Python frames.

    async def agen_fn(record: list[str]) -> AsyncIterator[None]:
        assert not _core.currently_ki_protected()
        record.append("the generator ran")
        yield

    run_record: list[str] = []
    agen = agen_fn(run_record)
    _core.run(agen.__anext__)
    assert run_record == ["the generator ran"]

    async def main() -> None:
        start_soon_record: list[str] = []
        agen = agen_fn(start_soon_record)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(agen.__anext__)
        assert start_soon_record == ["the generator ran"]

    _core.run(main)


async def test_very_deep_cancel_scope_nesting() -> None:
    # This used to crash with a RecursionError in CancelStatus.recalculate
    with ExitStack() as exit_stack:
        outermost_scope = _core.CancelScope()
        exit_stack.enter_context(outermost_scope)
        for _ in range(5000):
            exit_stack.enter_context(_core.CancelScope())
        outermost_scope.cancel()


async def test_cancel_scope_deadline_duplicates() -> None:
    # This exercises an assert in Deadlines._prune, by intentionally creating
    # duplicate entries in the deadline heap.
    now = _core.current_time()
    with _core.CancelScope() as cscope:
        for _ in range(DEADLINE_HEAP_MIN_PRUNE_THRESHOLD * 2):
            cscope.deadline = now + 9998
            cscope.deadline = now + 9999
        await sleep(0.01)


# I don't know if this one can fail anymore, the `del` next to the comment that used to
# refer to this only seems to break test_cancel_scope_exit_doesnt_create_cyclic_garbage
# We're keeping it for now to cover Outcome and potential future refactoring
@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="Only makes sense with refcounting GC",
)
async def test_simple_cancel_scope_usage_doesnt_create_cyclic_garbage() -> None:
    # https://github.com/python-trio/trio/issues/1770
    gc.collect()

    async def do_a_cancel() -> None:
        with _core.CancelScope() as cscope:
            cscope.cancel()
            await sleep_forever()

    async def crasher() -> NoReturn:
        raise ValueError("this is a crash")

    old_flags = gc.get_debug()
    try:
        gc.collect()
        gc.set_debug(gc.DEBUG_SAVEALL)

        # cover outcome.Error.unwrap
        # (See https://github.com/python-trio/outcome/pull/29)
        await do_a_cancel()
        # cover outcome.Error.unwrap if unrolled_run hangs on to exception refs
        # (See https://github.com/python-trio/trio/pull/1864)
        await do_a_cancel()

        with RaisesGroup(Matcher(ValueError, "^this is a crash$")):
            async with _core.open_nursery() as nursery:
                # cover NurseryManager.__aexit__
                nursery.start_soon(crasher)

        gc.collect()
        assert not gc.garbage
    finally:
        gc.set_debug(old_flags)
        gc.garbage.clear()


@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="Only makes sense with refcounting GC",
)
async def test_cancel_scope_exit_doesnt_create_cyclic_garbage() -> None:
    # https://github.com/python-trio/trio/pull/2063
    gc.collect()

    async def crasher() -> NoReturn:
        raise ValueError("this is a crash")

    old_flags = gc.get_debug()
    try:
        # fmt: off
        # Remove after 3.9 unsupported, black formats in a way that breaks if
        # you do `-X oldparser`
        with RaisesGroup(
            Matcher(ValueError, "^this is a crash$"),
        ), _core.CancelScope() as outer:
            # fmt: on
            async with _core.open_nursery() as nursery:
                gc.collect()
                gc.set_debug(gc.DEBUG_SAVEALL)
                # One child that gets cancelled by the outer scope
                nursery.start_soon(sleep_forever)
                outer.cancel()
                # And one that raises a different error
                nursery.start_soon(crasher)
                # so that outer filters a Cancelled from the ExceptionGroup and
                # covers CancelScope.__exit__ (and NurseryManager.__aexit__)
                # (See https://github.com/python-trio/trio/pull/2063)

        gc.collect()
        assert not gc.garbage
    finally:
        gc.set_debug(old_flags)
        gc.garbage.clear()


@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="Only makes sense with refcounting GC",
)
async def test_nursery_cancel_doesnt_create_cyclic_garbage() -> None:
    collected = False

    # https://github.com/python-trio/trio/issues/1770#issuecomment-730229423
    def toggle_collected() -> None:
        nonlocal collected
        collected = True

    gc.collect()
    old_flags = gc.get_debug()
    try:
        gc.set_debug(0)
        gc.collect()
        gc.set_debug(gc.DEBUG_SAVEALL)

        # cover Nursery._nested_child_finished
        async with _core.open_nursery() as nursery:
            nursery.cancel_scope.cancel()

        weakref.finalize(nursery, toggle_collected)
        del nursery
        # a checkpoint clears the nursery from the internals, apparently
        # TODO: stop event loop from hanging on to the nursery at this point
        await _core.checkpoint()

        assert collected
        gc.collect()
        assert not gc.garbage
    finally:
        gc.set_debug(old_flags)
        gc.garbage.clear()


@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="Only makes sense with refcounting GC",
)
async def test_locals_destroyed_promptly_on_cancel() -> None:
    destroyed = False

    def finalizer() -> None:
        nonlocal destroyed
        destroyed = True

    class A:
        pass

    async def task() -> None:
        a = A()
        weakref.finalize(a, finalizer)
        await _core.checkpoint()

    async with _core.open_nursery() as nursery:
        nursery.start_soon(task)
        nursery.cancel_scope.cancel()
    assert destroyed


def _create_kwargs(strictness: bool | None) -> dict[str, bool]:
    """Turn a bool|None into a kwarg dict that can be passed to `run` or `open_nursery`"""

    if strictness is None:
        return {}
    return {"strict_exception_groups": strictness}


@pytest.mark.filterwarnings(
    "ignore:.*strict_exception_groups=False:trio.TrioDeprecationWarning",
)
@pytest.mark.parametrize("run_strict", [True, False, None])
@pytest.mark.parametrize("open_nursery_strict", [True, False, None])
@pytest.mark.parametrize("multiple_exceptions", [True, False])
def test_setting_strict_exception_groups(
    run_strict: bool | None,
    open_nursery_strict: bool | None,
    multiple_exceptions: bool,
) -> None:
    """
    Test default values and that nurseries can both inherit and override the global context
    setting of strict_exception_groups.
    """

    async def raise_error() -> NoReturn:
        raise RuntimeError("test error")

    async def main() -> None:
        """Open a nursery, and raise one or two errors inside"""
        async with _core.open_nursery(**_create_kwargs(open_nursery_strict)) as nursery:
            nursery.start_soon(raise_error)
            if multiple_exceptions:
                nursery.start_soon(raise_error)

    def run_main() -> None:
        # mypy doesn't like kwarg magic
        _core.run(main, **_create_kwargs(run_strict))  # type: ignore[arg-type]

    matcher = Matcher(RuntimeError, r"^test error$")

    if multiple_exceptions:
        with RaisesGroup(matcher, matcher):
            run_main()
    elif open_nursery_strict or (
        open_nursery_strict is None and run_strict is not False
    ):
        with RaisesGroup(matcher):
            run_main()
    else:
        with pytest.raises(RuntimeError, match=r"^test error$"):
            run_main()


@pytest.mark.filterwarnings(
    "ignore:.*strict_exception_groups=False:trio.TrioDeprecationWarning",
)
@pytest.mark.parametrize("strict", [True, False, None])
async def test_nursery_collapse(strict: bool | None) -> None:
    """
    Test that a single exception from a nested nursery gets collapsed correctly
    depending on strict_exception_groups value when CancelledErrors are stripped from it.
    """

    async def raise_error() -> NoReturn:
        raise RuntimeError("test error")

    # mypy requires explicit type for conditional expression
    maybe_wrapped_runtime_error: type[RuntimeError] | RaisesGroup[RuntimeError] = (
        RuntimeError if strict is False else RaisesGroup(RuntimeError)
    )

    with RaisesGroup(RuntimeError, maybe_wrapped_runtime_error):
        async with _core.open_nursery() as nursery:
            nursery.start_soon(sleep_forever)
            nursery.start_soon(raise_error)
            async with _core.open_nursery(**_create_kwargs(strict)) as nursery2:
                nursery2.start_soon(sleep_forever)
                nursery2.start_soon(raise_error)
                nursery.cancel_scope.cancel()


async def test_cancel_scope_no_cancellederror() -> None:
    """
    Test that when a cancel scope encounters an exception group that does NOT contain
    a Cancelled exception, it will NOT set the ``cancelled_caught`` flag.
    """

    with RaisesGroup(RuntimeError, RuntimeError, match="test"):
        with _core.CancelScope() as scope:
            scope.cancel()
            raise ExceptionGroup("test", [RuntimeError(), RuntimeError()])

    assert not scope.cancelled_caught


@pytest.mark.filterwarnings(
    "ignore:.*strict_exception_groups=False:trio.TrioDeprecationWarning",
)
@pytest.mark.parametrize("run_strict", [False, True])
@pytest.mark.parametrize("start_raiser_strict", [False, True, None])
@pytest.mark.parametrize("raise_after_started", [False, True])
@pytest.mark.parametrize("raise_custom_exc_grp", [False, True])
def test_trio_run_strict_before_started(
    run_strict: bool,
    start_raiser_strict: bool | None,
    raise_after_started: bool,
    raise_custom_exc_grp: bool,
) -> None:
    """
    Regression tests for #2611, where exceptions raised before
    `TaskStatus.started()` caused `Nursery.start()` to wrap them in an
    ExceptionGroup when using `run(..., strict_exception_groups=True)`.

    Regression tests for #2844, where #2611 was initially fixed in a way that
    had unintended side effects.
    """

    raiser_exc: ValueError | ExceptionGroup[ValueError]
    if raise_custom_exc_grp:
        raiser_exc = ExceptionGroup("my group", [ValueError()])
    else:
        raiser_exc = ValueError()

    async def raiser(*, task_status: _core.TaskStatus[None]) -> None:
        if raise_after_started:
            task_status.started()
        raise raiser_exc

    async def start_raiser() -> None:
        try:
            async with _core.open_nursery(
                strict_exception_groups=start_raiser_strict,
            ) as nursery:
                await nursery.start(raiser)
        except BaseExceptionGroup as exc_group:
            if start_raiser_strict:
                # Iff the code using the nursery *forced* it to be strict
                # (overriding the runner setting) then it may replace the bland
                # exception group raised by trio with a more specific one (subtype,
                # different message, etc.).
                raise BaseExceptionGroup(
                    "start_raiser nursery custom message",
                    exc_group.exceptions,
                ) from None
            raise

    with pytest.raises(BaseException) as exc_info:  # noqa: PT011  # no `match`
        _core.run(start_raiser, strict_exception_groups=run_strict)

    if start_raiser_strict or (run_strict and start_raiser_strict is None):
        # start_raiser's nursery was strict.
        assert isinstance(exc_info.value, BaseExceptionGroup)
        if start_raiser_strict:
            # start_raiser didn't unknowingly inherit its nursery strictness
            # from `run`---it explicitly chose for its nursery to be strict.
            assert exc_info.value.message == "start_raiser nursery custom message"
        assert len(exc_info.value.exceptions) == 1
        should_be_raiser_exc = exc_info.value.exceptions[0]
    else:
        # start_raiser's nursery was not strict.
        should_be_raiser_exc = exc_info.value
    if isinstance(raiser_exc, ValueError):
        assert should_be_raiser_exc is raiser_exc
    else:
        # Check attributes, not identity, because should_be_raiser_exc may be a
        # copy of raiser_exc rather than raiser_exc by identity.
        assert type(should_be_raiser_exc) is type(raiser_exc)
        assert should_be_raiser_exc.message == raiser_exc.message
        assert should_be_raiser_exc.exceptions == raiser_exc.exceptions


async def test_internal_error_old_nursery_multiple_tasks() -> None:
    async def error_func() -> None:
        raise ValueError

    async def spawn_tasks_in_old_nursery(task_status: _core.TaskStatus[None]) -> None:
        old_nursery = _core.current_task().parent_nursery
        assert old_nursery is not None
        old_nursery.start_soon(error_func)
        old_nursery.start_soon(error_func)

    async with _core.open_nursery() as nursery:
        with pytest.raises(_core.TrioInternalError) as excinfo:
            await nursery.start(spawn_tasks_in_old_nursery)
    assert RaisesGroup(ValueError, ValueError).matches(excinfo.value.__cause__)


if sys.version_info >= (3, 11):

    def no_other_refs() -> list[object]:
        return []

else:

    def no_other_refs() -> list[object]:
        return [sys._getframe(1)]


@pytest.mark.skipif(
    sys.implementation.name != "cpython",
    reason="Only makes sense with refcounting GC",
)
@pytest.mark.xfail(
    sys.version_info >= (3, 14),
    reason="https://github.com/python/cpython/issues/125603",
)
async def test_ki_protection_doesnt_leave_cyclic_garbage() -> None:
    class MyException(Exception):
        pass

    async def demo() -> None:
        async def handle_error() -> None:
            try:
                raise MyException
            except MyException as e:
                exceptions.append(e)

        exceptions: list[MyException] = []
        try:
            async with _core.open_nursery() as n:
                n.start_soon(handle_error)
            raise ExceptionGroup("errors", exceptions)
        finally:
            exceptions = []

    exc: Exception | None = None
    try:
        await demo()
    except ExceptionGroup as excs:
        exc = excs.exceptions[0]

    assert isinstance(exc, MyException)
    assert gc.get_referrers(exc) == no_other_refs()


def test_context_run_tb_frames() -> None:
    class Context:
        def run(self, fn: Callable[[], object]) -> object:
            return fn()

    with mock.patch("trio._core._run.copy_context", return_value=Context()):
        assert _count_context_run_tb_frames() == 1


@restore_unraisablehook()
def test_trio_context_detection() -> None:
    assert not _core.in_trio_run()
    assert not _core.in_trio_task()

    def inner() -> None:
        assert _core.in_trio_run()
        assert _core.in_trio_task()

    def sync_inner() -> None:
        assert not _core.in_trio_run()
        assert not _core.in_trio_task()

    def inner_abort(_: object) -> _core.Abort:
        assert _core.in_trio_run()
        assert _core.in_trio_task()
        return _core.Abort.SUCCEEDED

    async def main() -> None:
        assert _core.in_trio_run()
        assert _core.in_trio_task()

        inner()

        await to_thread_run_sync(sync_inner)
        with _core.CancelScope(deadline=_core.current_time() - 1):
            await _core.wait_task_rescheduled(inner_abort)

    _core.run(main)