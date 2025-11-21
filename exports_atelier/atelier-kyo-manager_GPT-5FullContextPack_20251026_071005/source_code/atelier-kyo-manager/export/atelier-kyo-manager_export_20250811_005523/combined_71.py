
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\cx_oracle.py ===
# dialects/oracle/cx_oracle.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


r""".. dialect:: oracle+cx_oracle
    :name: cx-Oracle
    :dbapi: cx_oracle
    :connectstring: oracle+cx_oracle://user:pass@hostname:port[/dbname][?service_name=<service>[&key=value&key=value...]]
    :url: https://oracle.github.io/python-cx_Oracle/

Description
-----------

cx_Oracle was the original driver for Oracle Database. It was superseded by
python-oracledb which should be used instead.

DSN vs. Hostname connections
-----------------------------

cx_Oracle provides several methods of indicating the target database.  The
dialect translates from a series of different URL forms.

Hostname Connections with Easy Connect Syntax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a hostname, port and service name of the target database, for example
from Oracle Database's Easy Connect syntax then connect in SQLAlchemy using the
``service_name`` query string parameter::

    engine = create_engine(
        "oracle+cx_oracle://scott:tiger@hostname:port?service_name=myservice&encoding=UTF-8&nencoding=UTF-8"
    )

Note that the default driver value for encoding and nencoding was changed to
“UTF-8” in cx_Oracle 8.0 so these parameters can be omitted when using that
version, or later.

To use a full Easy Connect string, pass it as the ``dsn`` key value in a
:paramref:`_sa.create_engine.connect_args` dictionary::

    import cx_Oracle

    e = create_engine(
        "oracle+cx_oracle://@",
        connect_args={
            "user": "scott",
            "password": "tiger",
            "dsn": "hostname:port/myservice?transport_connect_timeout=30&expire_time=60",
        },
    )

Connections with tnsnames.ora or to Oracle Autonomous Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, if no port, database name, or service name is provided, the
dialect will use an Oracle Database DSN "connection string".  This takes the
"hostname" portion of the URL as the data source name.  For example, if the
``tnsnames.ora`` file contains a TNS Alias of ``myalias`` as below:

.. sourcecode:: text

    myalias =
      (DESCRIPTION =
        (ADDRESS = (PROTOCOL = TCP)(HOST = mymachine.example.com)(PORT = 1521))
        (CONNECT_DATA =
          (SERVER = DEDICATED)
          (SERVICE_NAME = orclpdb1)
        )
      )

The cx_Oracle dialect connects to this database service when ``myalias`` is the
hostname portion of the URL, without specifying a port, database name or
``service_name``::

    engine = create_engine("oracle+cx_oracle://scott:tiger@myalias")

Users of Oracle Autonomous Database should use this syntax. If the database is
configured for mutural TLS ("mTLS"), then you must also configure the cloud
wallet as shown in cx_Oracle documentation `Connecting to Autononmous Databases
<https://cx-oracle.readthedocs.io/en/latest/user_guide/connection_handling.html#autonomousdb>`_.

SID Connections
^^^^^^^^^^^^^^^

To use Oracle Database's obsolete System Identifier connection syntax, the SID
can be passed in a "database name" portion of the URL::

    engine = create_engine(
        "oracle+cx_oracle://scott:tiger@hostname:port/dbname"
    )

Above, the DSN passed to cx_Oracle is created by ``cx_Oracle.makedsn()`` as
follows::

    >>> import cx_Oracle
    >>> cx_Oracle.makedsn("hostname", 1521, sid="dbname")
    '(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=hostname)(PORT=1521))(CONNECT_DATA=(SID=dbname)))'

Note that although the SQLAlchemy syntax ``hostname:port/dbname`` looks like
Oracle's Easy Connect syntax it is different. It uses a SID in place of the
service name required by Easy Connect.  The Easy Connect syntax does not
support SIDs.

Passing cx_Oracle connect arguments
-----------------------------------

Additional connection arguments can usually be passed via the URL query string;
particular symbols like ``SYSDBA`` are intercepted and converted to the correct
symbol::

    e = create_engine(
        "oracle+cx_oracle://user:pass@dsn?encoding=UTF-8&nencoding=UTF-8&mode=SYSDBA&events=true"
    )

.. versionchanged:: 1.3 the cx_Oracle dialect now accepts all argument names
   within the URL string itself, to be passed to the cx_Oracle DBAPI.   As
   was the case earlier but not correctly documented, the
   :paramref:`_sa.create_engine.connect_args` parameter also accepts all
   cx_Oracle DBAPI connect arguments.

To pass arguments directly to ``.connect()`` without using the query
string, use the :paramref:`_sa.create_engine.connect_args` dictionary.
Any cx_Oracle parameter value and/or constant may be passed, such as::

    import cx_Oracle

    e = create_engine(
        "oracle+cx_oracle://user:pass@dsn",
        connect_args={
            "encoding": "UTF-8",
            "nencoding": "UTF-8",
            "mode": cx_Oracle.SYSDBA,
            "events": True,
        },
    )

Note that the default driver value for ``encoding`` and ``nencoding`` was
changed to "UTF-8" in cx_Oracle 8.0 so these parameters can be omitted when
using that version, or later.

Options consumed by the SQLAlchemy cx_Oracle dialect outside of the driver
--------------------------------------------------------------------------

There are also options that are consumed by the SQLAlchemy cx_oracle dialect
itself.  These options are always passed directly to :func:`_sa.create_engine`
, such as::

    e = create_engine(
        "oracle+cx_oracle://user:pass@dsn", coerce_to_decimal=False
    )

The parameters accepted by the cx_oracle dialect are as follows:

* ``arraysize`` - set the cx_oracle.arraysize value on cursors; defaults
  to ``None``, indicating that the driver default should be used (typically
  the value is 100).  This setting controls how many rows are buffered when
  fetching rows, and can have a significant effect on performance when
  modified.

  .. versionchanged:: 2.0.26 - changed the default value from 50 to None,
    to use the default value of the driver itself.

* ``auto_convert_lobs`` - defaults to True; See :ref:`cx_oracle_lob`.

* ``coerce_to_decimal`` - see :ref:`cx_oracle_numeric` for detail.

* ``encoding_errors`` - see :ref:`cx_oracle_unicode_encoding_errors` for detail.

.. _cx_oracle_sessionpool:

Using cx_Oracle SessionPool
---------------------------

The cx_Oracle driver provides its own connection pool implementation that may
be used in place of SQLAlchemy's pooling functionality. The driver pool
supports Oracle Database features such dead connection detection, connection
draining for planned database downtime, support for Oracle Application
Continuity and Transparent Application Continuity, and gives support for
Database Resident Connection Pooling (DRCP).

Using the driver pool can be achieved by using the
:paramref:`_sa.create_engine.creator` parameter to provide a function that
returns a new connection, along with setting
:paramref:`_sa.create_engine.pool_class` to ``NullPool`` to disable
SQLAlchemy's pooling::

    import cx_Oracle
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    pool = cx_Oracle.SessionPool(
        user="scott",
        password="tiger",
        dsn="orclpdb",
        min=1,
        max=4,
        increment=1,
        threaded=True,
        encoding="UTF-8",
        nencoding="UTF-8",
    )

    engine = create_engine(
        "oracle+cx_oracle://", creator=pool.acquire, poolclass=NullPool
    )

The above engine may then be used normally where cx_Oracle's pool handles
connection pooling::

    with engine.connect() as conn:
        print(conn.scalar("select 1 from dual"))

As well as providing a scalable solution for multi-user applications, the
cx_Oracle session pool supports some Oracle features such as DRCP and
`Application Continuity
<https://cx-oracle.readthedocs.io/en/latest/user_guide/ha.html#application-continuity-ac>`_.

Note that the pool creation parameters ``threaded``, ``encoding`` and
``nencoding`` were deprecated in later cx_Oracle releases.

Using Oracle Database Resident Connection Pooling (DRCP)
--------------------------------------------------------

When using Oracle Database's DRCP, the best practice is to pass a connection
class and "purity" when acquiring a connection from the SessionPool.  Refer to
the `cx_Oracle DRCP documentation
<https://cx-oracle.readthedocs.io/en/latest/user_guide/connection_handling.html#database-resident-connection-pooling-drcp>`_.

This can be achieved by wrapping ``pool.acquire()``::

    import cx_Oracle
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    pool = cx_Oracle.SessionPool(
        user="scott",
        password="tiger",
        dsn="orclpdb",
        min=2,
        max=5,
        increment=1,
        threaded=True,
        encoding="UTF-8",
        nencoding="UTF-8",
    )


    def creator():
        return pool.acquire(
            cclass="MYCLASS", purity=cx_Oracle.ATTR_PURITY_SELF
        )


    engine = create_engine(
        "oracle+cx_oracle://", creator=creator, poolclass=NullPool
    )

The above engine may then be used normally where cx_Oracle handles session
pooling and Oracle Database additionally uses DRCP::

    with engine.connect() as conn:
        print(conn.scalar("select 1 from dual"))

.. _cx_oracle_unicode:

Unicode
-------

As is the case for all DBAPIs under Python 3, all strings are inherently
Unicode strings. In all cases however, the driver requires an explicit
encoding configuration.

Ensuring the Correct Client Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The long accepted standard for establishing client encoding for nearly all
Oracle Database related software is via the `NLS_LANG
<https://www.oracle.com/database/technologies/faq-nls-lang.html>`_ environment
variable.  Older versions of cx_Oracle use this environment variable as the
source of its encoding configuration.  The format of this variable is
Territory_Country.CharacterSet; a typical value would be
``AMERICAN_AMERICA.AL32UTF8``.  cx_Oracle version 8 and later use the character
set "UTF-8" by default, and ignore the character set component of NLS_LANG.

The cx_Oracle driver also supported a programmatic alternative which is to pass
the ``encoding`` and ``nencoding`` parameters directly to its ``.connect()``
function.  These can be present in the URL as follows::

    engine = create_engine(
        "oracle+cx_oracle://scott:tiger@tnsalias?encoding=UTF-8&nencoding=UTF-8"
    )

For the meaning of the ``encoding`` and ``nencoding`` parameters, please
consult
`Characters Sets and National Language Support (NLS) <https://cx-oracle.readthedocs.io/en/latest/user_guide/globalization.html#globalization>`_.

.. seealso::

    `Characters Sets and National Language Support (NLS) <https://cx-oracle.readthedocs.io/en/latest/user_guide/globalization.html#globalization>`_
    - in the cx_Oracle documentation.


Unicode-specific Column datatypes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Core expression language handles unicode data by use of the
:class:`.Unicode` and :class:`.UnicodeText` datatypes.  These types correspond
to the VARCHAR2 and CLOB Oracle Database datatypes by default.  When using
these datatypes with Unicode data, it is expected that the database is
configured with a Unicode-aware character set, as well as that the ``NLS_LANG``
environment variable is set appropriately (this applies to older versions of
cx_Oracle), so that the VARCHAR2 and CLOB datatypes can accommodate the data.

In the case that Oracle Database is not configured with a Unicode character
set, the two options are to use the :class:`_types.NCHAR` and
:class:`_oracle.NCLOB` datatypes explicitly, or to pass the flag
``use_nchar_for_unicode=True`` to :func:`_sa.create_engine`, which will cause
the SQLAlchemy dialect to use NCHAR/NCLOB for the :class:`.Unicode` /
:class:`.UnicodeText` datatypes instead of VARCHAR/CLOB.

.. versionchanged:: 1.3 The :class:`.Unicode` and :class:`.UnicodeText`
   datatypes now correspond to the ``VARCHAR2`` and ``CLOB`` Oracle Database
   datatypes unless the ``use_nchar_for_unicode=True`` is passed to the dialect
   when :func:`_sa.create_engine` is called.


.. _cx_oracle_unicode_encoding_errors:

Encoding Errors
^^^^^^^^^^^^^^^

For the unusual case that data in Oracle Database is present with a broken
encoding, the dialect accepts a parameter ``encoding_errors`` which will be
passed to Unicode decoding functions in order to affect how decoding errors are
handled.  The value is ultimately consumed by the Python `decode
<https://docs.python.org/3/library/stdtypes.html#bytes.decode>`_ function, and
is passed both via cx_Oracle's ``encodingErrors`` parameter consumed by
``Cursor.var()``, as well as SQLAlchemy's own decoding function, as the
cx_Oracle dialect makes use of both under different circumstances.

.. versionadded:: 1.3.11


.. _cx_oracle_setinputsizes:

Fine grained control over cx_Oracle data binding performance with setinputsizes
-------------------------------------------------------------------------------

The cx_Oracle DBAPI has a deep and fundamental reliance upon the usage of the
DBAPI ``setinputsizes()`` call.  The purpose of this call is to establish the
datatypes that are bound to a SQL statement for Python values being passed as
parameters.  While virtually no other DBAPI assigns any use to the
``setinputsizes()`` call, the cx_Oracle DBAPI relies upon it heavily in its
interactions with the Oracle Database client interface, and in some scenarios
it is not possible for SQLAlchemy to know exactly how data should be bound, as
some settings can cause profoundly different performance characteristics, while
altering the type coercion behavior at the same time.

Users of the cx_Oracle dialect are **strongly encouraged** to read through
cx_Oracle's list of built-in datatype symbols at
https://cx-oracle.readthedocs.io/en/latest/api_manual/module.html#database-types.
Note that in some cases, significant performance degradation can occur when
using these types vs. not, in particular when specifying ``cx_Oracle.CLOB``.

On the SQLAlchemy side, the :meth:`.DialectEvents.do_setinputsizes` event can
be used both for runtime visibility (e.g. logging) of the setinputsizes step as
well as to fully control how ``setinputsizes()`` is used on a per-statement
basis.

.. versionadded:: 1.2.9 Added :meth:`.DialectEvents.setinputsizes`


Example 1 - logging all setinputsizes calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example illustrates how to log the intermediary values from a
SQLAlchemy perspective before they are converted to the raw ``setinputsizes()``
parameter dictionary.  The keys of the dictionary are :class:`.BindParameter`
objects which have a ``.key`` and a ``.type`` attribute::

    from sqlalchemy import create_engine, event

    engine = create_engine("oracle+cx_oracle://scott:tiger@host/xe")


    @event.listens_for(engine, "do_setinputsizes")
    def _log_setinputsizes(inputsizes, cursor, statement, parameters, context):
        for bindparam, dbapitype in inputsizes.items():
            log.info(
                "Bound parameter name: %s  SQLAlchemy type: %r DBAPI object: %s",
                bindparam.key,
                bindparam.type,
                dbapitype,
            )

Example 2 - remove all bindings to CLOB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``CLOB`` datatype in cx_Oracle incurs a significant performance overhead,
however is set by default for the ``Text`` type within the SQLAlchemy 1.2
series.   This setting can be modified as follows::

    from sqlalchemy import create_engine, event
    from cx_Oracle import CLOB

    engine = create_engine("oracle+cx_oracle://scott:tiger@host/xe")


    @event.listens_for(engine, "do_setinputsizes")
    def _remove_clob(inputsizes, cursor, statement, parameters, context):
        for bindparam, dbapitype in list(inputsizes.items()):
            if dbapitype is CLOB:
                del inputsizes[bindparam]

.. _cx_oracle_lob:

LOB Datatypes
--------------

LOB datatypes refer to the "large object" datatypes such as CLOB, NCLOB and
BLOB. Modern versions of cx_Oracle is optimized for these datatypes to be
delivered as a single buffer. As such, SQLAlchemy makes use of these newer type
handlers by default.

To disable the use of newer type handlers and deliver LOB objects as classic
buffered objects with a ``read()`` method, the parameter
``auto_convert_lobs=False`` may be passed to :func:`_sa.create_engine`,
which takes place only engine-wide.

.. _cx_oracle_returning:

RETURNING Support
-----------------

The cx_Oracle dialect implements RETURNING using OUT parameters.
The dialect supports RETURNING fully.

Two Phase Transactions Not Supported
------------------------------------

Two phase transactions are **not supported** under cx_Oracle due to poor driver
support. The newer :ref:`oracledb` dialect however **does** support two phase
transactions.

.. _cx_oracle_numeric:

Precision Numerics
------------------

SQLAlchemy's numeric types can handle receiving and returning values as Python
``Decimal`` objects or float objects.  When a :class:`.Numeric` object, or a
subclass such as :class:`.Float`, :class:`_oracle.DOUBLE_PRECISION` etc. is in
use, the :paramref:`.Numeric.asdecimal` flag determines if values should be
coerced to ``Decimal`` upon return, or returned as float objects.  To make
matters more complicated under Oracle Database, the ``NUMBER`` type can also
represent integer values if the "scale" is zero, so the Oracle
Database-specific :class:`_oracle.NUMBER` type takes this into account as well.

The cx_Oracle dialect makes extensive use of connection- and cursor-level
"outputtypehandler" callables in order to coerce numeric values as requested.
These callables are specific to the specific flavor of :class:`.Numeric` in
use, as well as if no SQLAlchemy typing objects are present.  There are
observed scenarios where Oracle Database may send incomplete or ambiguous
information about the numeric types being returned, such as a query where the
numeric types are buried under multiple levels of subquery.  The type handlers
do their best to make the right decision in all cases, deferring to the
underlying cx_Oracle DBAPI for all those cases where the driver can make the
best decision.

When no typing objects are present, as when executing plain SQL strings, a
default "outputtypehandler" is present which will generally return numeric
values which specify precision and scale as Python ``Decimal`` objects.  To
disable this coercion to decimal for performance reasons, pass the flag
``coerce_to_decimal=False`` to :func:`_sa.create_engine`::

    engine = create_engine("oracle+cx_oracle://dsn", coerce_to_decimal=False)

The ``coerce_to_decimal`` flag only impacts the results of plain string
SQL statements that are not otherwise associated with a :class:`.Numeric`
SQLAlchemy type (or a subclass of such).

.. versionchanged:: 1.2  The numeric handling system for cx_Oracle has been
   reworked to take advantage of newer cx_Oracle features as well
   as better integration of outputtypehandlers.

"""  # noqa
from __future__ import annotations

import decimal
import random
import re

from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler

# source:
# https://github.com/oracle/python-cx_Oracle/issues/596#issuecomment-999243649
_CX_ORACLE_MAGIC_LOB_SIZE = 131072


class _OracleInteger(sqltypes.Integer):
    def get_dbapi_type(self, dbapi):
        # see https://github.com/oracle/python-cx_Oracle/issues/
        # 208#issuecomment-409715955
        return int

    def _cx_oracle_var(self, dialect, cursor, arraysize=None):
        cx_Oracle = dialect.dbapi
        return cursor.var(
            cx_Oracle.STRING,
            255,
            arraysize=arraysize if arraysize is not None else cursor.arraysize,
            outconverter=int,
        )

    def _cx_oracle_outputtypehandler(self, dialect):
        def handler(cursor, name, default_type, size, precision, scale):
            return self._cx_oracle_var(dialect, cursor)

        return handler


class _OracleNumeric(sqltypes.Numeric):
    is_number = False

    def bind_processor(self, dialect):
        if self.scale == 0:
            return None
        elif self.asdecimal:
            processor = processors.to_decimal_processor_factory(
                decimal.Decimal, self._effective_decimal_return_scale
            )

            def process(value):
                if isinstance(value, (int, float)):
                    return processor(value)
                elif value is not None and value.is_infinite():
                    return float(value)
                else:
                    return value

            return process
        else:
            return processors.to_float

    def result_processor(self, dialect, coltype):
        return None

    def _cx_oracle_outputtypehandler(self, dialect):
        cx_Oracle = dialect.dbapi

        def handler(cursor, name, default_type, size, precision, scale):
            outconverter = None

            if precision:
                if self.asdecimal:
                    if default_type == cx_Oracle.NATIVE_FLOAT:
                        # receiving float and doing Decimal after the fact
                        # allows for float("inf") to be handled
                        type_ = default_type
                        outconverter = decimal.Decimal
                    else:
                        type_ = decimal.Decimal
                else:
                    if self.is_number and scale == 0:
                        # integer. cx_Oracle is observed to handle the widest
                        # variety of ints when no directives are passed,
                        # from 5.2 to 7.0.  See [ticket:4457]
                        return None
                    else:
                        type_ = cx_Oracle.NATIVE_FLOAT

            else:
                if self.asdecimal:
                    if default_type == cx_Oracle.NATIVE_FLOAT:
                        type_ = default_type
                        outconverter = decimal.Decimal
                    else:
                        type_ = decimal.Decimal
                else:
                    if self.is_number and scale == 0:
                        # integer. cx_Oracle is observed to handle the widest
                        # variety of ints when no directives are passed,
                        # from 5.2 to 7.0.  See [ticket:4457]
                        return None
                    else:
                        type_ = cx_Oracle.NATIVE_FLOAT

            return cursor.var(
                type_,
                255,
                arraysize=cursor.arraysize,
                outconverter=outconverter,
            )

        return handler


class _OracleUUID(sqltypes.Uuid):
    def get_dbapi_type(self, dbapi):
        return dbapi.STRING


class _OracleBinaryFloat(_OracleNumeric):
    def get_dbapi_type(self, dbapi):
        return dbapi.NATIVE_FLOAT


class _OracleBINARY_FLOAT(_OracleBinaryFloat, oracle.BINARY_FLOAT):
    pass


class _OracleBINARY_DOUBLE(_OracleBinaryFloat, oracle.BINARY_DOUBLE):
    pass


class _OracleNUMBER(_OracleNumeric):
    is_number = True


class _CXOracleDate(oracle._OracleDate):
    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is not None:
                return value.date()
            else:
                return value

        return process


class _CXOracleTIMESTAMP(_OracleDateLiteralRender, sqltypes.TIMESTAMP):
    def literal_processor(self, dialect):
        return self._literal_processor_datetime(dialect)


class _LOBDataType:
    pass


# TODO: the names used across CHAR / VARCHAR / NCHAR / NVARCHAR
# here are inconsistent and not very good
class _OracleChar(sqltypes.CHAR):
    def get_dbapi_type(self, dbapi):
        return dbapi.FIXED_CHAR


class _OracleNChar(sqltypes.NCHAR):
    def get_dbapi_type(self, dbapi):
        return dbapi.FIXED_NCHAR


class _OracleUnicodeStringNCHAR(oracle.NVARCHAR2):
    def get_dbapi_type(self, dbapi):
        return dbapi.NCHAR


class _OracleUnicodeStringCHAR(sqltypes.Unicode):
    def get_dbapi_type(self, dbapi):
        return dbapi.LONG_STRING


class _OracleUnicodeTextNCLOB(_LOBDataType, oracle.NCLOB):
    def get_dbapi_type(self, dbapi):
        # previously, this was dbapi.NCLOB.
        # DB_TYPE_NVARCHAR will instead be passed to setinputsizes()
        # when this datatype is used.
        return dbapi.DB_TYPE_NVARCHAR


class _OracleUnicodeTextCLOB(_LOBDataType, sqltypes.UnicodeText):
    def get_dbapi_type(self, dbapi):
        # previously, this was dbapi.CLOB.
        # DB_TYPE_NVARCHAR will instead be passed to setinputsizes()
        # when this datatype is used.
        return dbapi.DB_TYPE_NVARCHAR


class _OracleText(_LOBDataType, sqltypes.Text):
    def get_dbapi_type(self, dbapi):
        # previously, this was dbapi.CLOB.
        # DB_TYPE_NVARCHAR will instead be passed to setinputsizes()
        # when this datatype is used.
        return dbapi.DB_TYPE_NVARCHAR


class _OracleLong(_LOBDataType, oracle.LONG):
    def get_dbapi_type(self, dbapi):
        return dbapi.LONG_STRING


class _OracleString(sqltypes.String):
    pass


class _OracleEnum(sqltypes.Enum):
    def bind_processor(self, dialect):
        enum_proc = sqltypes.Enum.bind_processor(self, dialect)

        def process(value):
            raw_str = enum_proc(value)
            return raw_str

        return process


class _OracleBinary(_LOBDataType, sqltypes.LargeBinary):
    def get_dbapi_type(self, dbapi):
        # previously, this was dbapi.BLOB.
        # DB_TYPE_RAW will instead be passed to setinputsizes()
        # when this datatype is used.
        return dbapi.DB_TYPE_RAW

    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):
        if not dialect.auto_convert_lobs:
            return None
        else:
            return super().result_processor(dialect, coltype)


class _OracleInterval(oracle.INTERVAL):
    def get_dbapi_type(self, dbapi):
        return dbapi.INTERVAL


class _OracleRaw(oracle.RAW):
    pass


class _OracleRowid(oracle.ROWID):
    def get_dbapi_type(self, dbapi):
        return dbapi.ROWID


class OracleCompiler_cx_oracle(OracleCompiler):
    _oracle_cx_sql_compiler = True

    _oracle_returning = False

    # Oracle bind names can't start with digits or underscores.
    # currently we rely upon Oracle-specific quoting of bind names in most
    # cases.  however for expanding params, the escape chars are used.
    # see #8708
    bindname_escape_characters = util.immutabledict(
        {
            "%": "P",
            "(": "A",
            ")": "Z",
            ":": "C",
            ".": "C",
            "[": "C",
            "]": "C",
            " ": "C",
            "\\": "C",
            "/": "C",
            "?": "C",
        }
    )

    def bindparam_string(self, name, **kw):
        quote = getattr(name, "quote", None)
        if (
            quote is True
            or quote is not False
            and self.preparer._bindparam_requires_quotes(name)
            # bind param quoting for Oracle doesn't work with post_compile
            # params.  For those, the default bindparam_string will escape
            # special chars, and the appending of a number "_1" etc. will
            # take care of reserved words
            and not kw.get("post_compile", False)
        ):
            # interesting to note about expanding parameters - since the
            # new parameters take the form <paramname>_<int>, at least if
            # they are originally formed from reserved words, they no longer
            # need quoting :).    names that include illegal characters
            # won't work however.
            quoted_name = '"%s"' % name
            kw["escaped_from"] = name
            name = quoted_name
            return OracleCompiler.bindparam_string(self, name, **kw)

        # TODO: we could likely do away with quoting altogether for
        # Oracle parameters and use the custom escaping here
        escaped_from = kw.get("escaped_from", None)
        if not escaped_from:
            if self._bind_translate_re.search(name):
                # not quite the translate use case as we want to
                # also get a quick boolean if we even found
                # unusual characters in the name
                new_name = self._bind_translate_re.sub(
                    lambda m: self._bind_translate_chars[m.group(0)],
                    name,
                )
                if new_name[0].isdigit() or new_name[0] == "_":
                    new_name = "D" + new_name
                kw["escaped_from"] = name
                name = new_name
            elif name[0].isdigit() or name[0] == "_":
                new_name = "D" + name
                kw["escaped_from"] = name
                name = new_name

        return OracleCompiler.bindparam_string(self, name, **kw)


class OracleExecutionContext_cx_oracle(OracleExecutionContext):
    out_parameters = None

    def _generate_out_parameter_vars(self):
        # check for has_out_parameters or RETURNING, create cx_Oracle.var
        # objects if so
        if self.compiled.has_out_parameters or self.compiled._oracle_returning:
            out_parameters = self.out_parameters
            assert out_parameters is not None

            len_params = len(self.parameters)

            quoted_bind_names = self.compiled.escaped_bind_names
            for bindparam in self.compiled.binds.values():
                if bindparam.isoutparam:
                    name = self.compiled.bind_names[bindparam]
                    type_impl = bindparam.type.dialect_impl(self.dialect)

                    if hasattr(type_impl, "_cx_oracle_var"):
                        out_parameters[name] = type_impl._cx_oracle_var(
                            self.dialect, self.cursor, arraysize=len_params
                        )
                    else:
                        dbtype = type_impl.get_dbapi_type(self.dialect.dbapi)

                        cx_Oracle = self.dialect.dbapi

                        assert cx_Oracle is not None

                        if dbtype is None:
                            raise exc.InvalidRequestError(
                                "Cannot create out parameter for "
                                "parameter "
                                "%r - its type %r is not supported by"
                                " cx_oracle" % (bindparam.key, bindparam.type)
                            )

                        # note this is an OUT parameter.   Using
                        # non-LOB datavalues with large unicode-holding
                        # values causes the failure (both cx_Oracle and
                        # oracledb):
                        # ORA-22835: Buffer too small for CLOB to CHAR or
                        # BLOB to RAW conversion (actual: 16507,
                        # maximum: 4000)
                        # [SQL: INSERT INTO long_text (x, y, z) VALUES
                        # (:x, :y, :z) RETURNING long_text.x, long_text.y,
                        # long_text.z INTO :ret_0, :ret_1, :ret_2]
                        # so even for DB_TYPE_NVARCHAR we convert to a LOB

                        if isinstance(type_impl, _LOBDataType):
                            if dbtype == cx_Oracle.DB_TYPE_NVARCHAR:
                                dbtype = cx_Oracle.NCLOB
                            elif dbtype == cx_Oracle.DB_TYPE_RAW:
                                dbtype = cx_Oracle.BLOB
                            # other LOB types go in directly

                            out_parameters[name] = self.cursor.var(
                                dbtype,
                                # this is fine also in oracledb_async since
                                # the driver will await the read coroutine
                                outconverter=lambda value: value.read(),
                                arraysize=len_params,
                            )
                        elif (
                            isinstance(type_impl, _OracleNumeric)
                            and type_impl.asdecimal
                        ):
                            out_parameters[name] = self.cursor.var(
                                decimal.Decimal,
                                arraysize=len_params,
                            )

                        else:
                            out_parameters[name] = self.cursor.var(
                                dbtype, arraysize=len_params
                            )

                    for param in self.parameters:
                        param[quoted_bind_names.get(name, name)] = (
                            out_parameters[name]
                        )

    def _generate_cursor_outputtype_handler(self):
        output_handlers = {}

        for keyname, name, objects, type_ in self.compiled._result_columns:
            handler = type_._cached_custom_processor(
                self.dialect,
                "cx_oracle_outputtypehandler",
                self._get_cx_oracle_type_handler,
            )

            if handler:
                denormalized_name = self.dialect.denormalize_name(keyname)
                output_handlers[denormalized_name] = handler

        if output_handlers:
            default_handler = self._dbapi_connection.outputtypehandler

            def output_type_handler(
                cursor, name, default_type, size, precision, scale
            ):
                if name in output_handlers:
                    return output_handlers[name](
                        cursor, name, default_type, size, precision, scale
                    )
                else:
                    return default_handler(
                        cursor, name, default_type, size, precision, scale
                    )

            self.cursor.outputtypehandler = output_type_handler

    def _get_cx_oracle_type_handler(self, impl):
        if hasattr(impl, "_cx_oracle_outputtypehandler"):
            return impl._cx_oracle_outputtypehandler(self.dialect)
        else:
            return None

    def pre_exec(self):
        super().pre_exec()
        if not getattr(self.compiled, "_oracle_cx_sql_compiler", False):
            return

        self.out_parameters = {}

        self._generate_out_parameter_vars()

        self._generate_cursor_outputtype_handler()

    def post_exec(self):
        if (
            self.compiled
            and is_sql_compiler(self.compiled)
            and self.compiled._oracle_returning
        ):
            initial_buffer = self.fetchall_for_returning(
                self.cursor, _internal=True
            )

            fetch_strategy = _cursor.FullyBufferedCursorFetchStrategy(
                self.cursor,
                [
                    (entry.keyname, None)
                    for entry in self.compiled._result_columns
                ],
                initial_buffer=initial_buffer,
            )

            self.cursor_fetch_strategy = fetch_strategy

    def create_cursor(self):
        c = self._dbapi_connection.cursor()
        if self.dialect.arraysize:
            c.arraysize = self.dialect.arraysize

        return c

    def fetchall_for_returning(self, cursor, *, _internal=False):
        compiled = self.compiled
        if (
            not _internal
            and compiled is None
            or not is_sql_compiler(compiled)
            or not compiled._oracle_returning
        ):
            raise NotImplementedError(
                "execution context was not prepared for Oracle RETURNING"
            )

        # create a fake cursor result from the out parameters. unlike
        # get_out_parameter_values(), the result-row handlers here will be
        # applied at the Result level

        numcols = len(self.out_parameters)

        # [stmt_result for stmt_result in outparam.values] == each
        # statement in executemany
        # [val for val in stmt_result] == each row for a particular
        # statement
        return list(
            zip(
                *[
                    [
                        val
                        for stmt_result in self.out_parameters[
                            f"ret_{j}"
                        ].values
                        for val in (stmt_result or ())
                    ]
                    for j in range(numcols)
                ]
            )
        )

    def get_out_parameter_values(self, out_param_names):
        # this method should not be called when the compiler has
        # RETURNING as we've turned the has_out_parameters flag set to
        # False.
        assert not self.compiled.returning

        return [
            self.dialect._paramval(self.out_parameters[name])
            for name in out_param_names
        ]


class OracleDialect_cx_oracle(OracleDialect):
    supports_statement_cache = True
    execution_ctx_cls = OracleExecutionContext_cx_oracle
    statement_compiler = OracleCompiler_cx_oracle

    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True

    insert_executemany_returning = True
    insert_executemany_returning_sort_by_parameter_order = True
    update_executemany_returning = True
    delete_executemany_returning = True

    bind_typing = interfaces.BindTyping.SETINPUTSIZES

    driver = "cx_oracle"

    colspecs = util.update_copy(
        OracleDialect.colspecs,
        {
            sqltypes.TIMESTAMP: _CXOracleTIMESTAMP,
            sqltypes.Numeric: _OracleNumeric,
            sqltypes.Float: _OracleNumeric,
            oracle.BINARY_FLOAT: _OracleBINARY_FLOAT,
            oracle.BINARY_DOUBLE: _OracleBINARY_DOUBLE,
            sqltypes.Integer: _OracleInteger,
            oracle.NUMBER: _OracleNUMBER,
            sqltypes.Date: _CXOracleDate,
            sqltypes.LargeBinary: _OracleBinary,
            sqltypes.Boolean: oracle._OracleBoolean,
            sqltypes.Interval: _OracleInterval,
            oracle.INTERVAL: _OracleInterval,
            sqltypes.Text: _OracleText,
            sqltypes.String: _OracleString,
            sqltypes.UnicodeText: _OracleUnicodeTextCLOB,
            sqltypes.CHAR: _OracleChar,
            sqltypes.NCHAR: _OracleNChar,
            sqltypes.Enum: _OracleEnum,
            oracle.LONG: _OracleLong,
            oracle.RAW: _OracleRaw,
            sqltypes.Unicode: _OracleUnicodeStringCHAR,
            sqltypes.NVARCHAR: _OracleUnicodeStringNCHAR,
            sqltypes.Uuid: _OracleUUID,
            oracle.NCLOB: _OracleUnicodeTextNCLOB,
            oracle.ROWID: _OracleRowid,
        },
    )

    execute_sequence_format = list

    _cx_oracle_threaded = None

    _cursor_var_unicode_kwargs = util.immutabledict()

    @util.deprecated_params(
        threaded=(
            "1.3",
            "The 'threaded' parameter to the cx_oracle/oracledb dialect "
            "is deprecated as a dialect-level argument, and will be removed "
            "in a future release.  As of version 1.3, it defaults to False "
            "rather than True.  The 'threaded' option can be passed to "
            "cx_Oracle directly in the URL query string passed to "
            ":func:`_sa.create_engine`.",
        )
    )
    def __init__(
        self,
        auto_convert_lobs=True,
        coerce_to_decimal=True,
        arraysize=None,
        encoding_errors=None,
        threaded=None,
        **kwargs,
    ):
        OracleDialect.__init__(self, **kwargs)
        self.arraysize = arraysize
        self.encoding_errors = encoding_errors
        if encoding_errors:
            self._cursor_var_unicode_kwargs = {
                "encodingErrors": encoding_errors
            }
        if threaded is not None:
            self._cx_oracle_threaded = threaded
        self.auto_convert_lobs = auto_convert_lobs
        self.coerce_to_decimal = coerce_to_decimal
        if self._use_nchar_for_unicode:
            self.colspecs = self.colspecs.copy()
            self.colspecs[sqltypes.Unicode] = _OracleUnicodeStringNCHAR
            self.colspecs[sqltypes.UnicodeText] = _OracleUnicodeTextNCLOB

        dbapi_module = self.dbapi
        self._load_version(dbapi_module)

        if dbapi_module is not None:
            # these constants will first be seen in SQLAlchemy datatypes
            # coming from the get_dbapi_type() method.   We then
            # will place the following types into setinputsizes() calls
            # on each statement.  Oracle constants that are not in this
            # list will not be put into setinputsizes().
            self.include_set_input_sizes = {
                dbapi_module.DATETIME,
                dbapi_module.DB_TYPE_NVARCHAR,  # used for CLOB, NCLOB
                dbapi_module.DB_TYPE_RAW,  # used for BLOB
                dbapi_module.NCLOB,  # not currently used except for OUT param
                dbapi_module.CLOB,  # not currently used except for OUT param
                dbapi_module.LOB,  # not currently used
                dbapi_module.BLOB,  # not currently used except for OUT param
                dbapi_module.NCHAR,
                dbapi_module.FIXED_NCHAR,
                dbapi_module.FIXED_CHAR,
                dbapi_module.TIMESTAMP,
                int,  # _OracleInteger,
                # _OracleBINARY_FLOAT, _OracleBINARY_DOUBLE,
                dbapi_module.NATIVE_FLOAT,
            }

            self._paramval = lambda value: value.getvalue()

    def _load_version(self, dbapi_module):
        version = (0, 0, 0)
        if dbapi_module is not None:
            m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", dbapi_module.version)
            if m:
                version = tuple(
                    int(x) for x in m.group(1, 2, 3) if x is not None
                )
        self.cx_oracle_ver = version
        if self.cx_oracle_ver < (8,) and self.cx_oracle_ver > (0, 0, 0):
            raise exc.InvalidRequestError(
                "cx_Oracle version 8 and above are supported"
            )

    @classmethod
    def import_dbapi(cls):
        import cx_Oracle

        return cx_Oracle

    def initialize(self, connection):
        super().initialize(connection)
        self._detect_decimal_char(connection)

    def get_isolation_level(self, dbapi_connection):
        # sources:

        # general idea of transaction id, have to start one, etc.
        # https://stackoverflow.com/questions/10711204/how-to-check-isoloation-level

        # how to decode xid cols from v$transaction to match
        # https://asktom.oracle.com/pls/apex/f?p=100:11:0::::P11_QUESTION_ID:9532779900346079444

        # Oracle tuple comparison without using IN:
        # https://www.sql-workbench.eu/comparison/tuple_comparison.html

        with dbapi_connection.cursor() as cursor:
            # this is the only way to ensure a transaction is started without
            # actually running DML.   There's no way to see the configured
            # isolation level without getting it from v$transaction which
            # means transaction has to be started.
            outval = cursor.var(str)
            cursor.execute(
                """
                begin
                   :trans_id := dbms_transaction.local_transaction_id( TRUE );
                end;
                """,
                {"trans_id": outval},
            )
            trans_id = outval.getvalue()
            xidusn, xidslot, xidsqn = trans_id.split(".", 2)

            cursor.execute(
                "SELECT CASE BITAND(t.flag, POWER(2, 28)) "
                "WHEN 0 THEN 'READ COMMITTED' "
                "ELSE 'SERIALIZABLE' END AS isolation_level "
                "FROM v$transaction t WHERE "
                "(t.xidusn, t.xidslot, t.xidsqn) = "
                "((:xidusn, :xidslot, :xidsqn))",
                {"xidusn": xidusn, "xidslot": xidslot, "xidsqn": xidsqn},
            )
            row = cursor.fetchone()
            if row is None:
                raise exc.InvalidRequestError(
                    "could not retrieve isolation level"
                )
            result = row[0]

        return result

    def get_isolation_level_values(self, dbapi_connection):
        return super().get_isolation_level_values(dbapi_connection) + [
            "AUTOCOMMIT"
        ]

    def set_isolation_level(self, dbapi_connection, level):
        if level == "AUTOCOMMIT":
            dbapi_connection.autocommit = True
        else:
            dbapi_connection.autocommit = False
            dbapi_connection.rollback()
            with dbapi_connection.cursor() as cursor:
                cursor.execute(f"ALTER SESSION SET ISOLATION_LEVEL={level}")

    def _detect_decimal_char(self, connection):
        # we have the option to change this setting upon connect,
        # or just look at what it is upon connect and convert.
        # to minimize the chance of interference with changes to
        # NLS_TERRITORY or formatting behavior of the DB, we opt
        # to just look at it

        dbapi_connection = connection.connection

        with dbapi_connection.cursor() as cursor:
            # issue #8744
            # nls_session_parameters is not available in some Oracle
            # modes like "mount mode".  But then, v$nls_parameters is not
            # available if the connection doesn't have SYSDBA priv.
            #
            # simplify the whole thing and just use the method that we were
            # doing in the test suite already, selecting a number

            def output_type_handler(
                cursor, name, defaultType, size, precision, scale
            ):
                return cursor.var(
                    self.dbapi.STRING, 255, arraysize=cursor.arraysize
                )

            cursor.outputtypehandler = output_type_handler
            cursor.execute("SELECT 1.1 FROM DUAL")
            value = cursor.fetchone()[0]

            decimal_char = value.lstrip("0")[1]
            assert not decimal_char[0].isdigit()

        self._decimal_char = decimal_char

        if self._decimal_char != ".":
            _detect_decimal = self._detect_decimal
            _to_decimal = self._to_decimal

            self._detect_decimal = lambda value: _detect_decimal(
                value.replace(self._decimal_char, ".")
            )
            self._to_decimal = lambda value: _to_decimal(
                value.replace(self._decimal_char, ".")
            )

    def _detect_decimal(self, value):
        if "." in value:
            return self._to_decimal(value)
        else:
            return int(value)

    _to_decimal = decimal.Decimal

    def _generate_connection_outputtype_handler(self):
        """establish the default outputtypehandler established at the
        connection level.

        """

        dialect = self
        cx_Oracle = dialect.dbapi

        number_handler = _OracleNUMBER(
            asdecimal=True
        )._cx_oracle_outputtypehandler(dialect)
        float_handler = _OracleNUMBER(
            asdecimal=False
        )._cx_oracle_outputtypehandler(dialect)

        def output_type_handler(
            cursor, name, default_type, size, precision, scale
        ):
            if (
                default_type == cx_Oracle.NUMBER
                and default_type is not cx_Oracle.NATIVE_FLOAT
            ):
                if not dialect.coerce_to_decimal:
                    return None
                elif precision == 0 and scale in (0, -127):
                    # ambiguous type, this occurs when selecting
                    # numbers from deep subqueries
                    return cursor.var(
                        cx_Oracle.STRING,
                        255,
                        outconverter=dialect._detect_decimal,
                        arraysize=cursor.arraysize,
                    )
                elif precision and scale > 0:
                    return number_handler(
                        cursor, name, default_type, size, precision, scale
                    )
                else:
                    return float_handler(
                        cursor, name, default_type, size, precision, scale
                    )

            # if unicode options were specified, add a decoder, otherwise
            # cx_Oracle should return Unicode
            elif (
                dialect._cursor_var_unicode_kwargs
                and default_type
                in (
                    cx_Oracle.STRING,
                    cx_Oracle.FIXED_CHAR,
                )
                and default_type is not cx_Oracle.CLOB
                and default_type is not cx_Oracle.NCLOB
            ):
                return cursor.var(
                    str,
                    size,
                    cursor.arraysize,
                    **dialect._cursor_var_unicode_kwargs,
                )

            elif dialect.auto_convert_lobs and default_type in (
                cx_Oracle.CLOB,
                cx_Oracle.NCLOB,
            ):
                typ = (
                    cx_Oracle.DB_TYPE_VARCHAR
                    if default_type is cx_Oracle.CLOB
                    else cx_Oracle.DB_TYPE_NVARCHAR
                )
                return cursor.var(
                    typ,
                    _CX_ORACLE_MAGIC_LOB_SIZE,
                    cursor.arraysize,
                    **dialect._cursor_var_unicode_kwargs,
                )

            elif dialect.auto_convert_lobs and default_type in (
                cx_Oracle.BLOB,
            ):
                return cursor.var(
                    cx_Oracle.DB_TYPE_RAW,
                    _CX_ORACLE_MAGIC_LOB_SIZE,
                    cursor.arraysize,
                )

        return output_type_handler

    def on_connect(self):
        output_type_handler = self._generate_connection_outputtype_handler()

        def on_connect(conn):
            conn.outputtypehandler = output_type_handler

        return on_connect

    def create_connect_args(self, url):
        opts = dict(url.query)

        for opt in ("use_ansi", "auto_convert_lobs"):
            if opt in opts:
                util.warn_deprecated(
                    f"{self.driver} dialect option {opt!r} should only be "
                    "passed to create_engine directly, not within the URL "
                    "string",
                    version="1.3",
                )
                util.coerce_kw_type(opts, opt, bool)
                setattr(self, opt, opts.pop(opt))

        database = url.database
        service_name = opts.pop("service_name", None)
        if database or service_name:
            # if we have a database, then we have a remote host
            port = url.port
            if port:
                port = int(port)
            else:
                port = 1521

            if database and service_name:
                raise exc.InvalidRequestError(
                    '"service_name" option shouldn\'t '
                    'be used with a "database" part of the url'
                )
            if database:
                makedsn_kwargs = {"sid": database}
            if service_name:
                makedsn_kwargs = {"service_name": service_name}

            dsn = self.dbapi.makedsn(url.host, port, **makedsn_kwargs)
        else:
            # we have a local tnsname
            dsn = url.host

        if dsn is not None:
            opts["dsn"] = dsn
        if url.password is not None:
            opts["password"] = url.password
        if url.username is not None:
            opts["user"] = url.username

        if self._cx_oracle_threaded is not None:
            opts.setdefault("threaded", self._cx_oracle_threaded)

        def convert_cx_oracle_constant(value):
            if isinstance(value, str):
                try:
                    int_val = int(value)
                except ValueError:
                    value = value.upper()
                    return getattr(self.dbapi, value)
                else:
                    return int_val
            else:
                return value

        util.coerce_kw_type(opts, "mode", convert_cx_oracle_constant)
        util.coerce_kw_type(opts, "threaded", bool)
        util.coerce_kw_type(opts, "events", bool)
        util.coerce_kw_type(opts, "purity", convert_cx_oracle_constant)
        return ([], opts)

    def _get_server_version_info(self, connection):
        return tuple(int(x) for x in connection.connection.version.split("."))

    def is_disconnect(self, e, connection, cursor):
        (error,) = e.args
        if isinstance(
            e, (self.dbapi.InterfaceError, self.dbapi.DatabaseError)
        ) and "not connected" in str(e):
            return True

        if hasattr(error, "code") and error.code in {
            28,
            3114,
            3113,
            3135,
            1033,
            2396,
        }:
            # ORA-00028: your session has been killed
            # ORA-03114: not connected to ORACLE
            # ORA-03113: end-of-file on communication channel
            # ORA-03135: connection lost contact
            # ORA-01033: ORACLE initialization or shutdown in progress
            # ORA-02396: exceeded maximum idle time, please connect again
            # TODO: Others ?
            return True

        if re.match(r"^(?:DPI-1010|DPI-1080|DPY-1001|DPY-4011)", str(e)):
            # DPI-1010: not connected
            # DPI-1080: connection was closed by ORA-3113
            # python-oracledb's DPY-1001: not connected to database
            # python-oracledb's DPY-4011: the database or network closed the
            # connection
            # TODO: others?
            return True

        return False

    def create_xid(self):
        id_ = random.randint(0, 2**128)
        return (0x1234, "%032x" % id_, "%032x" % 9)

    def do_executemany(self, cursor, statement, parameters, context=None):
        if isinstance(parameters, tuple):
            parameters = list(parameters)
        cursor.executemany(statement, parameters)

    def do_begin_twophase(self, connection, xid):
        connection.connection.begin(*xid)
        connection.connection.info["cx_oracle_xid"] = xid

    def do_prepare_twophase(self, connection, xid):
        result = connection.connection.prepare()
        connection.info["cx_oracle_prepared"] = result

    def do_rollback_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        self.do_rollback(connection.connection)
        # TODO: need to end XA state here

    def do_commit_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        if not is_prepared:
            self.do_commit(connection.connection)
        else:
            if recover:
                raise NotImplementedError(
                    "2pc recovery not implemented for cx_Oracle"
                )
            oci_prepared = connection.info["cx_oracle_prepared"]
            if oci_prepared:
                self.do_commit(connection.connection)
        # TODO: need to end XA state here

    def do_set_input_sizes(self, cursor, list_of_tuples, context):
        if self.positional:
            # not usually used, here to support if someone is modifying
            # the dialect to use positional style
            cursor.setinputsizes(
                *[dbtype for key, dbtype, sqltype in list_of_tuples]
            )
        else:
            collection = (
                (key, dbtype)
                for key, dbtype, sqltype in list_of_tuples
                if dbtype
            )

            cursor.setinputsizes(**{key: dbtype for key, dbtype in collection})

    def do_recover_twophase(self, connection):
        raise NotImplementedError(
            "recover two phase query for cx_Oracle not implemented"
        )


dialect = OracleDialect_cx_oracle

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\app.py ===
from __future__ import annotations

import collections.abc as cabc
import os
import sys
import typing as t
import weakref
from datetime import timedelta
from inspect import iscoroutinefunction
from itertools import chain
from types import TracebackType
from urllib.parse import quote as _url_quote

import click
from werkzeug.datastructures import Headers
from werkzeug.datastructures import ImmutableDict
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import InternalServerError
from werkzeug.routing import BuildError
from werkzeug.routing import MapAdapter
from werkzeug.routing import RequestRedirect
from werkzeug.routing import RoutingException
from werkzeug.routing import Rule
from werkzeug.serving import is_running_from_reloader
from werkzeug.wrappers import Response as BaseResponse
from werkzeug.wsgi import get_host

from . import cli
from . import typing as ft
from .ctx import AppContext
from .ctx import RequestContext
from .globals import _cv_app
from .globals import _cv_request
from .globals import current_app
from .globals import g
from .globals import request
from .globals import request_ctx
from .globals import session
from .helpers import get_debug_flag
from .helpers import get_flashed_messages
from .helpers import get_load_dotenv
from .helpers import send_from_directory
from .sansio.app import App
from .sansio.scaffold import _sentinel
from .sessions import SecureCookieSessionInterface
from .sessions import SessionInterface
from .signals import appcontext_tearing_down
from .signals import got_request_exception
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .templating import Environment
from .wrappers import Request
from .wrappers import Response

if t.TYPE_CHECKING:  # pragma: no cover
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIEnvironment

    from .testing import FlaskClient
    from .testing import FlaskCliRunner
    from .typing import HeadersValue

T_shell_context_processor = t.TypeVar(
    "T_shell_context_processor", bound=ft.ShellContextProcessorCallable
)
T_teardown = t.TypeVar("T_teardown", bound=ft.TeardownCallable)
T_template_filter = t.TypeVar("T_template_filter", bound=ft.TemplateFilterCallable)
T_template_global = t.TypeVar("T_template_global", bound=ft.TemplateGlobalCallable)
T_template_test = t.TypeVar("T_template_test", bound=ft.TemplateTestCallable)


def _make_timedelta(value: timedelta | int | None) -> timedelta | None:
    if value is None or isinstance(value, timedelta):
        return value

    return timedelta(seconds=value)


class Flask(App):
    """The flask object implements a WSGI application and acts as the central
    object.  It is passed the name of the module or package of the
    application.  Once it is created it will act as a central registry for
    the view functions, the URL rules, template configuration and much more.

    The name of the package is used to resolve resources from inside the
    package or the folder the module is contained in depending on if the
    package parameter resolves to an actual python package (a folder with
    an :file:`__init__.py` file inside) or a standard module (just a ``.py`` file).

    For more information about resource loading, see :func:`open_resource`.

    Usually you create a :class:`Flask` instance in your main module or
    in the :file:`__init__.py` file of your package like this::

        from flask import Flask
        app = Flask(__name__)

    .. admonition:: About the First Parameter

        The idea of the first parameter is to give Flask an idea of what
        belongs to your application.  This name is used to find resources
        on the filesystem, can be used by extensions to improve debugging
        information and a lot more.

        So it's important what you provide there.  If you are using a single
        module, `__name__` is always the correct value.  If you however are
        using a package, it's usually recommended to hardcode the name of
        your package there.

        For example if your application is defined in :file:`yourapplication/app.py`
        you should create it with one of the two versions below::

            app = Flask('yourapplication')
            app = Flask(__name__.split('.')[0])

        Why is that?  The application will work even with `__name__`, thanks
        to how resources are looked up.  However it will make debugging more
        painful.  Certain extensions can make assumptions based on the
        import name of your application.  For example the Flask-SQLAlchemy
        extension will look for the code in your application that triggered
        an SQL query in debug mode.  If the import name is not properly set
        up, that debugging information is lost.  (For example it would only
        pick up SQL queries in `yourapplication.app` and not
        `yourapplication.views.frontend`)

    .. versionadded:: 0.7
       The `static_url_path`, `static_folder`, and `template_folder`
       parameters were added.

    .. versionadded:: 0.8
       The `instance_path` and `instance_relative_config` parameters were
       added.

    .. versionadded:: 0.11
       The `root_path` parameter was added.

    .. versionadded:: 1.0
       The ``host_matching`` and ``static_host`` parameters were added.

    .. versionadded:: 1.0
       The ``subdomain_matching`` parameter was added. Subdomain
       matching needs to be enabled manually now. Setting
       :data:`SERVER_NAME` does not implicitly enable it.

    :param import_name: the name of the application package
    :param static_url_path: can be used to specify a different path for the
                            static files on the web.  Defaults to the name
                            of the `static_folder` folder.
    :param static_folder: The folder with static files that is served at
        ``static_url_path``. Relative to the application ``root_path``
        or an absolute path. Defaults to ``'static'``.
    :param static_host: the host to use when adding the static route.
        Defaults to None. Required when using ``host_matching=True``
        with a ``static_folder`` configured.
    :param host_matching: set ``url_map.host_matching`` attribute.
        Defaults to False.
    :param subdomain_matching: consider the subdomain relative to
        :data:`SERVER_NAME` when matching routes. Defaults to False.
    :param template_folder: the folder that contains the templates that should
                            be used by the application.  Defaults to
                            ``'templates'`` folder in the root path of the
                            application.
    :param instance_path: An alternative instance path for the application.
                          By default the folder ``'instance'`` next to the
                          package or module is assumed to be the instance
                          path.
    :param instance_relative_config: if set to ``True`` relative filenames
                                     for loading the config are assumed to
                                     be relative to the instance path instead
                                     of the application root.
    :param root_path: The path to the root of the application files.
        This should only be set manually when it can't be detected
        automatically, such as for namespace packages.
    """

    default_config = ImmutableDict(
        {
            "DEBUG": None,
            "TESTING": False,
            "PROPAGATE_EXCEPTIONS": None,
            "SECRET_KEY": None,
            "SECRET_KEY_FALLBACKS": None,
            "PERMANENT_SESSION_LIFETIME": timedelta(days=31),
            "USE_X_SENDFILE": False,
            "TRUSTED_HOSTS": None,
            "SERVER_NAME": None,
            "APPLICATION_ROOT": "/",
            "SESSION_COOKIE_NAME": "session",
            "SESSION_COOKIE_DOMAIN": None,
            "SESSION_COOKIE_PATH": None,
            "SESSION_COOKIE_HTTPONLY": True,
            "SESSION_COOKIE_SECURE": False,
            "SESSION_COOKIE_PARTITIONED": False,
            "SESSION_COOKIE_SAMESITE": None,
            "SESSION_REFRESH_EACH_REQUEST": True,
            "MAX_CONTENT_LENGTH": None,
            "MAX_FORM_MEMORY_SIZE": 500_000,
            "MAX_FORM_PARTS": 1_000,
            "SEND_FILE_MAX_AGE_DEFAULT": None,
            "TRAP_BAD_REQUEST_ERRORS": None,
            "TRAP_HTTP_EXCEPTIONS": False,
            "EXPLAIN_TEMPLATE_LOADING": False,
            "PREFERRED_URL_SCHEME": "http",
            "TEMPLATES_AUTO_RELOAD": None,
            "MAX_COOKIE_SIZE": 4093,
            "PROVIDE_AUTOMATIC_OPTIONS": True,
        }
    )

    #: The class that is used for request objects.  See :class:`~flask.Request`
    #: for more information.
    request_class: type[Request] = Request

    #: The class that is used for response objects.  See
    #: :class:`~flask.Response` for more information.
    response_class: type[Response] = Response

    #: the session interface to use.  By default an instance of
    #: :class:`~flask.sessions.SecureCookieSessionInterface` is used here.
    #:
    #: .. versionadded:: 0.8
    session_interface: SessionInterface = SecureCookieSessionInterface()

    def __init__(
        self,
        import_name: str,
        static_url_path: str | None = None,
        static_folder: str | os.PathLike[str] | None = "static",
        static_host: str | None = None,
        host_matching: bool = False,
        subdomain_matching: bool = False,
        template_folder: str | os.PathLike[str] | None = "templates",
        instance_path: str | None = None,
        instance_relative_config: bool = False,
        root_path: str | None = None,
    ):
        super().__init__(
            import_name=import_name,
            static_url_path=static_url_path,
            static_folder=static_folder,
            static_host=static_host,
            host_matching=host_matching,
            subdomain_matching=subdomain_matching,
            template_folder=template_folder,
            instance_path=instance_path,
            instance_relative_config=instance_relative_config,
            root_path=root_path,
        )

        #: The Click command group for registering CLI commands for this
        #: object. The commands are available from the ``flask`` command
        #: once the application has been discovered and blueprints have
        #: been registered.
        self.cli = cli.AppGroup()

        # Set the name of the Click group in case someone wants to add
        # the app's commands to another CLI tool.
        self.cli.name = self.name

        # Add a static route using the provided static_url_path, static_host,
        # and static_folder if there is a configured static_folder.
        # Note we do this without checking if static_folder exists.
        # For one, it might be created while the server is running (e.g. during
        # development). Also, Google App Engine stores static files somewhere
        if self.has_static_folder:
            assert bool(static_host) == host_matching, (
                "Invalid static_host/host_matching combination"
            )
            # Use a weakref to avoid creating a reference cycle between the app
            # and the view function (see #3761).
            self_ref = weakref.ref(self)
            self.add_url_rule(
                f"{self.static_url_path}/<path:filename>",
                endpoint="static",
                host=static_host,
                view_func=lambda **kw: self_ref().send_static_file(**kw),  # type: ignore # noqa: B950
            )

    def get_send_file_max_age(self, filename: str | None) -> int | None:
        """Used by :func:`send_file` to determine the ``max_age`` cache
        value for a given file path if it wasn't passed.

        By default, this returns :data:`SEND_FILE_MAX_AGE_DEFAULT` from
        the configuration of :data:`~flask.current_app`. This defaults
        to ``None``, which tells the browser to use conditional requests
        instead of a timed cache, which is usually preferable.

        Note this is a duplicate of the same method in the Flask
        class.

        .. versionchanged:: 2.0
            The default configuration is ``None`` instead of 12 hours.

        .. versionadded:: 0.9
        """
        value = current_app.config["SEND_FILE_MAX_AGE_DEFAULT"]

        if value is None:
            return None

        if isinstance(value, timedelta):
            return int(value.total_seconds())

        return value  # type: ignore[no-any-return]

    def send_static_file(self, filename: str) -> Response:
        """The view function used to serve files from
        :attr:`static_folder`. A route is automatically registered for
        this view at :attr:`static_url_path` if :attr:`static_folder` is
        set.

        Note this is a duplicate of the same method in the Flask
        class.

        .. versionadded:: 0.5

        """
        if not self.has_static_folder:
            raise RuntimeError("'static_folder' must be set to serve static_files.")

        # send_file only knows to call get_send_file_max_age on the app,
        # call it here so it works for blueprints too.
        max_age = self.get_send_file_max_age(filename)
        return send_from_directory(
            t.cast(str, self.static_folder), filename, max_age=max_age
        )

    def open_resource(
        self, resource: str, mode: str = "rb", encoding: str | None = None
    ) -> t.IO[t.AnyStr]:
        """Open a resource file relative to :attr:`root_path` for reading.

        For example, if the file ``schema.sql`` is next to the file
        ``app.py`` where the ``Flask`` app is defined, it can be opened
        with:

        .. code-block:: python

            with app.open_resource("schema.sql") as f:
                conn.executescript(f.read())

        :param resource: Path to the resource relative to :attr:`root_path`.
        :param mode: Open the file in this mode. Only reading is supported,
            valid values are ``"r"`` (or ``"rt"``) and ``"rb"``.
        :param encoding: Open the file with this encoding when opening in text
            mode. This is ignored when opening in binary mode.

        .. versionchanged:: 3.1
            Added the ``encoding`` parameter.
        """
        if mode not in {"r", "rt", "rb"}:
            raise ValueError("Resources can only be opened for reading.")

        path = os.path.join(self.root_path, resource)

        if mode == "rb":
            return open(path, mode)  # pyright: ignore

        return open(path, mode, encoding=encoding)

    def open_instance_resource(
        self, resource: str, mode: str = "rb", encoding: str | None = "utf-8"
    ) -> t.IO[t.AnyStr]:
        """Open a resource file relative to the application's instance folder
        :attr:`instance_path`. Unlike :meth:`open_resource`, files in the
        instance folder can be opened for writing.

        :param resource: Path to the resource relative to :attr:`instance_path`.
        :param mode: Open the file in this mode.
        :param encoding: Open the file with this encoding when opening in text
            mode. This is ignored when opening in binary mode.

        .. versionchanged:: 3.1
            Added the ``encoding`` parameter.
        """
        path = os.path.join(self.instance_path, resource)

        if "b" in mode:
            return open(path, mode)

        return open(path, mode, encoding=encoding)

    def create_jinja_environment(self) -> Environment:
        """Create the Jinja environment based on :attr:`jinja_options`
        and the various Jinja-related methods of the app. Changing
        :attr:`jinja_options` after this will have no effect. Also adds
        Flask-related globals and filters to the environment.

        .. versionchanged:: 0.11
           ``Environment.auto_reload`` set in accordance with
           ``TEMPLATES_AUTO_RELOAD`` configuration option.

        .. versionadded:: 0.5
        """
        options = dict(self.jinja_options)

        if "autoescape" not in options:
            options["autoescape"] = self.select_jinja_autoescape

        if "auto_reload" not in options:
            auto_reload = self.config["TEMPLATES_AUTO_RELOAD"]

            if auto_reload is None:
                auto_reload = self.debug

            options["auto_reload"] = auto_reload

        rv = self.jinja_environment(self, **options)
        rv.globals.update(
            url_for=self.url_for,
            get_flashed_messages=get_flashed_messages,
            config=self.config,
            # request, session and g are normally added with the
            # context processor for efficiency reasons but for imported
            # templates we also want the proxies in there.
            request=request,
            session=session,
            g=g,
        )
        rv.policies["json.dumps_function"] = self.json.dumps
        return rv

    def create_url_adapter(self, request: Request | None) -> MapAdapter | None:
        """Creates a URL adapter for the given request. The URL adapter
        is created at a point where the request context is not yet set
        up so the request is passed explicitly.

        .. versionchanged:: 3.1
            If :data:`SERVER_NAME` is set, it does not restrict requests to
            only that domain, for both ``subdomain_matching`` and
            ``host_matching``.

        .. versionchanged:: 1.0
            :data:`SERVER_NAME` no longer implicitly enables subdomain
            matching. Use :attr:`subdomain_matching` instead.

        .. versionchanged:: 0.9
           This can be called outside a request when the URL adapter is created
           for an application context.

        .. versionadded:: 0.6
        """
        if request is not None:
            if (trusted_hosts := self.config["TRUSTED_HOSTS"]) is not None:
                request.trusted_hosts = trusted_hosts

            # Check trusted_hosts here until bind_to_environ does.
            request.host = get_host(request.environ, request.trusted_hosts)  # pyright: ignore
            subdomain = None
            server_name = self.config["SERVER_NAME"]

            if self.url_map.host_matching:
                # Don't pass SERVER_NAME, otherwise it's used and the actual
                # host is ignored, which breaks host matching.
                server_name = None
            elif not self.subdomain_matching:
                # Werkzeug doesn't implement subdomain matching yet. Until then,
                # disable it by forcing the current subdomain to the default, or
                # the empty string.
                subdomain = self.url_map.default_subdomain or ""

            return self.url_map.bind_to_environ(
                request.environ, server_name=server_name, subdomain=subdomain
            )

        # Need at least SERVER_NAME to match/build outside a request.
        if self.config["SERVER_NAME"] is not None:
            return self.url_map.bind(
                self.config["SERVER_NAME"],
                script_name=self.config["APPLICATION_ROOT"],
                url_scheme=self.config["PREFERRED_URL_SCHEME"],
            )

        return None

    def raise_routing_exception(self, request: Request) -> t.NoReturn:
        """Intercept routing exceptions and possibly do something else.

        In debug mode, intercept a routing redirect and replace it with
        an error if the body will be discarded.

        With modern Werkzeug this shouldn't occur, since it now uses a
        308 status which tells the browser to resend the method and
        body.

        .. versionchanged:: 2.1
            Don't intercept 307 and 308 redirects.

        :meta private:
        :internal:
        """
        if (
            not self.debug
            or not isinstance(request.routing_exception, RequestRedirect)
            or request.routing_exception.code in {307, 308}
            or request.method in {"GET", "HEAD", "OPTIONS"}
        ):
            raise request.routing_exception  # type: ignore[misc]

        from .debughelpers import FormDataRoutingRedirect

        raise FormDataRoutingRedirect(request)

    def update_template_context(self, context: dict[str, t.Any]) -> None:
        """Update the template context with some commonly used variables.
        This injects request, session, config and g into the template
        context as well as everything template context processors want
        to inject.  Note that the as of Flask 0.6, the original values
        in the context will not be overridden if a context processor
        decides to return a value with the same key.

        :param context: the context as a dictionary that is updated in place
                        to add extra variables.
        """
        names: t.Iterable[str | None] = (None,)

        # A template may be rendered outside a request context.
        if request:
            names = chain(names, reversed(request.blueprints))

        # The values passed to render_template take precedence. Keep a
        # copy to re-apply after all context functions.
        orig_ctx = context.copy()

        for name in names:
            if name in self.template_context_processors:
                for func in self.template_context_processors[name]:
                    context.update(self.ensure_sync(func)())

        context.update(orig_ctx)

    def make_shell_context(self) -> dict[str, t.Any]:
        """Returns the shell context for an interactive shell for this
        application.  This runs all the registered shell context
        processors.

        .. versionadded:: 0.11
        """
        rv = {"app": self, "g": g}
        for processor in self.shell_context_processors:
            rv.update(processor())
        return rv

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
        load_dotenv: bool = True,
        **options: t.Any,
    ) -> None:
        """Runs the application on a local development server.

        Do not use ``run()`` in a production setting. It is not intended to
        meet security and performance requirements for a production server.
        Instead, see :doc:`/deploying/index` for WSGI server recommendations.

        If the :attr:`debug` flag is set the server will automatically reload
        for code changes and show a debugger in case an exception happened.

        If you want to run the application in debug mode, but disable the
        code execution on the interactive debugger, you can pass
        ``use_evalex=False`` as parameter.  This will keep the debugger's
        traceback screen active, but disable code execution.

        It is not recommended to use this function for development with
        automatic reloading as this is badly supported.  Instead you should
        be using the :command:`flask` command line script's ``run`` support.

        .. admonition:: Keep in Mind

           Flask will suppress any server error with a generic error page
           unless it is in debug mode.  As such to enable just the
           interactive debugger without the code reloading, you have to
           invoke :meth:`run` with ``debug=True`` and ``use_reloader=False``.
           Setting ``use_debugger`` to ``True`` without being in debug mode
           won't catch any exceptions because there won't be any to
           catch.

        :param host: the hostname to listen on. Set this to ``'0.0.0.0'`` to
            have the server available externally as well. Defaults to
            ``'127.0.0.1'`` or the host in the ``SERVER_NAME`` config variable
            if present.
        :param port: the port of the webserver. Defaults to ``5000`` or the
            port defined in the ``SERVER_NAME`` config variable if present.
        :param debug: if given, enable or disable debug mode. See
            :attr:`debug`.
        :param load_dotenv: Load the nearest :file:`.env` and :file:`.flaskenv`
            files to set environment variables. Will also change the working
            directory to the directory containing the first file found.
        :param options: the options to be forwarded to the underlying Werkzeug
            server. See :func:`werkzeug.serving.run_simple` for more
            information.

        .. versionchanged:: 1.0
            If installed, python-dotenv will be used to load environment
            variables from :file:`.env` and :file:`.flaskenv` files.

            The :envvar:`FLASK_DEBUG` environment variable will override :attr:`debug`.

            Threaded mode is enabled by default.

        .. versionchanged:: 0.10
            The default port is now picked from the ``SERVER_NAME``
            variable.
        """
        # Ignore this call so that it doesn't start another server if
        # the 'flask run' command is used.
        if os.environ.get("FLASK_RUN_FROM_CLI") == "true":
            if not is_running_from_reloader():
                click.secho(
                    " * Ignoring a call to 'app.run()' that would block"
                    " the current 'flask' CLI command.\n"
                    "   Only call 'app.run()' in an 'if __name__ =="
                    ' "__main__"\' guard.',
                    fg="red",
                )

            return

        if get_load_dotenv(load_dotenv):
            cli.load_dotenv()

            # if set, env var overrides existing value
            if "FLASK_DEBUG" in os.environ:
                self.debug = get_debug_flag()

        # debug passed to method overrides all other sources
        if debug is not None:
            self.debug = bool(debug)

        server_name = self.config.get("SERVER_NAME")
        sn_host = sn_port = None

        if server_name:
            sn_host, _, sn_port = server_name.partition(":")

        if not host:
            if sn_host:
                host = sn_host
            else:
                host = "127.0.0.1"

        if port or port == 0:
            port = int(port)
        elif sn_port:
            port = int(sn_port)
        else:
            port = 5000

        options.setdefault("use_reloader", self.debug)
        options.setdefault("use_debugger", self.debug)
        options.setdefault("threaded", True)

        cli.show_server_banner(self.debug, self.name)

        from werkzeug.serving import run_simple

        try:
            run_simple(t.cast(str, host), port, self, **options)
        finally:
            # reset the first request information if the development server
            # reset normally.  This makes it possible to restart the server
            # without reloader and that stuff from an interactive shell.
            self._got_first_request = False

    def test_client(self, use_cookies: bool = True, **kwargs: t.Any) -> FlaskClient:
        """Creates a test client for this application.  For information
        about unit testing head over to :doc:`/testing`.

        Note that if you are testing for assertions or exceptions in your
        application code, you must set ``app.testing = True`` in order for the
        exceptions to propagate to the test client.  Otherwise, the exception
        will be handled by the application (not visible to the test client) and
        the only indication of an AssertionError or other exception will be a
        500 status code response to the test client.  See the :attr:`testing`
        attribute.  For example::

            app.testing = True
            client = app.test_client()

        The test client can be used in a ``with`` block to defer the closing down
        of the context until the end of the ``with`` block.  This is useful if
        you want to access the context locals for testing::

            with app.test_client() as c:
                rv = c.get('/?vodka=42')
                assert request.args['vodka'] == '42'

        Additionally, you may pass optional keyword arguments that will then
        be passed to the application's :attr:`test_client_class` constructor.
        For example::

            from flask.testing import FlaskClient

            class CustomClient(FlaskClient):
                def __init__(self, *args, **kwargs):
                    self._authentication = kwargs.pop("authentication")
                    super(CustomClient,self).__init__( *args, **kwargs)

            app.test_client_class = CustomClient
            client = app.test_client(authentication='Basic ....')

        See :class:`~flask.testing.FlaskClient` for more information.

        .. versionchanged:: 0.4
           added support for ``with`` block usage for the client.

        .. versionadded:: 0.7
           The `use_cookies` parameter was added as well as the ability
           to override the client to be used by setting the
           :attr:`test_client_class` attribute.

        .. versionchanged:: 0.11
           Added `**kwargs` to support passing additional keyword arguments to
           the constructor of :attr:`test_client_class`.
        """
        cls = self.test_client_class
        if cls is None:
            from .testing import FlaskClient as cls
        return cls(  # type: ignore
            self, self.response_class, use_cookies=use_cookies, **kwargs
        )

    def test_cli_runner(self, **kwargs: t.Any) -> FlaskCliRunner:
        """Create a CLI runner for testing CLI commands.
        See :ref:`testing-cli`.

        Returns an instance of :attr:`test_cli_runner_class`, by default
        :class:`~flask.testing.FlaskCliRunner`. The Flask app object is
        passed as the first argument.

        .. versionadded:: 1.0
        """
        cls = self.test_cli_runner_class

        if cls is None:
            from .testing import FlaskCliRunner as cls

        return cls(self, **kwargs)  # type: ignore

    def handle_http_exception(
        self, e: HTTPException
    ) -> HTTPException | ft.ResponseReturnValue:
        """Handles an HTTP exception.  By default this will invoke the
        registered error handlers and fall back to returning the
        exception as response.

        .. versionchanged:: 1.0.3
            ``RoutingException``, used internally for actions such as
             slash redirects during routing, is not passed to error
             handlers.

        .. versionchanged:: 1.0
            Exceptions are looked up by code *and* by MRO, so
            ``HTTPException`` subclasses can be handled with a catch-all
            handler for the base ``HTTPException``.

        .. versionadded:: 0.3
        """
        # Proxy exceptions don't have error codes.  We want to always return
        # those unchanged as errors
        if e.code is None:
            return e

        # RoutingExceptions are used internally to trigger routing
        # actions, such as slash redirects raising RequestRedirect. They
        # are not raised or handled in user code.
        if isinstance(e, RoutingException):
            return e

        handler = self._find_error_handler(e, request.blueprints)
        if handler is None:
            return e
        return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]

    def handle_user_exception(
        self, e: Exception
    ) -> HTTPException | ft.ResponseReturnValue:
        """This method is called whenever an exception occurs that
        should be handled. A special case is :class:`~werkzeug
        .exceptions.HTTPException` which is forwarded to the
        :meth:`handle_http_exception` method. This function will either
        return a response value or reraise the exception with the same
        traceback.

        .. versionchanged:: 1.0
            Key errors raised from request data like ``form`` show the
            bad key in debug mode rather than a generic bad request
            message.

        .. versionadded:: 0.7
        """
        if isinstance(e, BadRequestKeyError) and (
            self.debug or self.config["TRAP_BAD_REQUEST_ERRORS"]
        ):
            e.show_exception = True

        if isinstance(e, HTTPException) and not self.trap_http_exception(e):
            return self.handle_http_exception(e)

        handler = self._find_error_handler(e, request.blueprints)

        if handler is None:
            raise

        return self.ensure_sync(handler)(e)  # type: ignore[no-any-return]

    def handle_exception(self, e: Exception) -> Response:
        """Handle an exception that did not have an error handler
        associated with it, or that was raised from an error handler.
        This always causes a 500 ``InternalServerError``.

        Always sends the :data:`got_request_exception` signal.

        If :data:`PROPAGATE_EXCEPTIONS` is ``True``, such as in debug
        mode, the error will be re-raised so that the debugger can
        display it. Otherwise, the original exception is logged, and
        an :exc:`~werkzeug.exceptions.InternalServerError` is returned.

        If an error handler is registered for ``InternalServerError`` or
        ``500``, it will be used. For consistency, the handler will
        always receive the ``InternalServerError``. The original
        unhandled exception is available as ``e.original_exception``.

        .. versionchanged:: 1.1.0
            Always passes the ``InternalServerError`` instance to the
            handler, setting ``original_exception`` to the unhandled
            error.

        .. versionchanged:: 1.1.0
            ``after_request`` functions and other finalization is done
            even for the default 500 response when there is no handler.

        .. versionadded:: 0.3
        """
        exc_info = sys.exc_info()
        got_request_exception.send(self, _async_wrapper=self.ensure_sync, exception=e)
        propagate = self.config["PROPAGATE_EXCEPTIONS"]

        if propagate is None:
            propagate = self.testing or self.debug

        if propagate:
            # Re-raise if called with an active exception, otherwise
            # raise the passed in exception.
            if exc_info[1] is e:
                raise

            raise e

        self.log_exception(exc_info)
        server_error: InternalServerError | ft.ResponseReturnValue
        server_error = InternalServerError(original_exception=e)
        handler = self._find_error_handler(server_error, request.blueprints)

        if handler is not None:
            server_error = self.ensure_sync(handler)(server_error)

        return self.finalize_request(server_error, from_error_handler=True)

    def log_exception(
        self,
        exc_info: (tuple[type, BaseException, TracebackType] | tuple[None, None, None]),
    ) -> None:
        """Logs an exception.  This is called by :meth:`handle_exception`
        if debugging is disabled and right before the handler is called.
        The default implementation logs the exception as error on the
        :attr:`logger`.

        .. versionadded:: 0.8
        """
        self.logger.error(
            f"Exception on {request.path} [{request.method}]", exc_info=exc_info
        )

    def dispatch_request(self) -> ft.ResponseReturnValue:
        """Does the request dispatching.  Matches the URL and returns the
        return value of the view or error handler.  This does not have to
        be a response object.  In order to convert the return value to a
        proper response object, call :func:`make_response`.

        .. versionchanged:: 0.7
           This no longer does the exception handling, this code was
           moved to the new :meth:`full_dispatch_request`.
        """
        req = request_ctx.request
        if req.routing_exception is not None:
            self.raise_routing_exception(req)
        rule: Rule = req.url_rule  # type: ignore[assignment]
        # if we provide automatic options for this URL and the
        # request came with the OPTIONS method, reply automatically
        if (
            getattr(rule, "provide_automatic_options", False)
            and req.method == "OPTIONS"
        ):
            return self.make_default_options_response()
        # otherwise dispatch to the handler for that endpoint
        view_args: dict[str, t.Any] = req.view_args  # type: ignore[assignment]
        return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)  # type: ignore[no-any-return]

    def full_dispatch_request(self) -> Response:
        """Dispatches the request and on top of that performs request
        pre and postprocessing as well as HTTP exception catching and
        error handling.

        .. versionadded:: 0.7
        """
        self._got_first_request = True

        try:
            request_started.send(self, _async_wrapper=self.ensure_sync)
            rv = self.preprocess_request()
            if rv is None:
                rv = self.dispatch_request()
        except Exception as e:
            rv = self.handle_user_exception(e)
        return self.finalize_request(rv)

    def finalize_request(
        self,
        rv: ft.ResponseReturnValue | HTTPException,
        from_error_handler: bool = False,
    ) -> Response:
        """Given the return value from a view function this finalizes
        the request by converting it into a response and invoking the
        postprocessing functions.  This is invoked for both normal
        request dispatching as well as error handlers.

        Because this means that it might be called as a result of a
        failure a special safe mode is available which can be enabled
        with the `from_error_handler` flag.  If enabled, failures in
        response processing will be logged and otherwise ignored.

        :internal:
        """
        response = self.make_response(rv)
        try:
            response = self.process_response(response)
            request_finished.send(
                self, _async_wrapper=self.ensure_sync, response=response
            )
        except Exception:
            if not from_error_handler:
                raise
            self.logger.exception(
                "Request finalizing failed with an error while handling an error"
            )
        return response

    def make_default_options_response(self) -> Response:
        """This method is called to create the default ``OPTIONS`` response.
        This can be changed through subclassing to change the default
        behavior of ``OPTIONS`` responses.

        .. versionadded:: 0.7
        """
        adapter = request_ctx.url_adapter
        methods = adapter.allowed_methods()  # type: ignore[union-attr]
        rv = self.response_class()
        rv.allow.update(methods)
        return rv

    def ensure_sync(self, func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """Ensure that the function is synchronous for WSGI workers.
        Plain ``def`` functions are returned as-is. ``async def``
        functions are wrapped to run and wait for the response.

        Override this method to change how the app runs async views.

        .. versionadded:: 2.0
        """
        if iscoroutinefunction(func):
            return self.async_to_sync(func)

        return func

    def async_to_sync(
        self, func: t.Callable[..., t.Coroutine[t.Any, t.Any, t.Any]]
    ) -> t.Callable[..., t.Any]:
        """Return a sync function that will run the coroutine function.

        .. code-block:: python

            result = app.async_to_sync(func)(*args, **kwargs)

        Override this method to change how the app converts async code
        to be synchronously callable.

        .. versionadded:: 2.0
        """
        try:
            from asgiref.sync import async_to_sync as asgiref_async_to_sync
        except ImportError:
            raise RuntimeError(
                "Install Flask with the 'async' extra in order to use async views."
            ) from None

        return asgiref_async_to_sync(func)

    def url_for(
        self,
        /,
        endpoint: str,
        *,
        _anchor: str | None = None,
        _method: str | None = None,
        _scheme: str | None = None,
        _external: bool | None = None,
        **values: t.Any,
    ) -> str:
        """Generate a URL to the given endpoint with the given values.

        This is called by :func:`flask.url_for`, and can be called
        directly as well.

        An *endpoint* is the name of a URL rule, usually added with
        :meth:`@app.route() <route>`, and usually the same name as the
        view function. A route defined in a :class:`~flask.Blueprint`
        will prepend the blueprint's name separated by a ``.`` to the
        endpoint.

        In some cases, such as email messages, you want URLs to include
        the scheme and domain, like ``https://example.com/hello``. When
        not in an active request, URLs will be external by default, but
        this requires setting :data:`SERVER_NAME` so Flask knows what
        domain to use. :data:`APPLICATION_ROOT` and
        :data:`PREFERRED_URL_SCHEME` should also be configured as
        needed. This config is only used when not in an active request.

        Functions can be decorated with :meth:`url_defaults` to modify
        keyword arguments before the URL is built.

        If building fails for some reason, such as an unknown endpoint
        or incorrect values, the app's :meth:`handle_url_build_error`
        method is called. If that returns a string, that is returned,
        otherwise a :exc:`~werkzeug.routing.BuildError` is raised.

        :param endpoint: The endpoint name associated with the URL to
            generate. If this starts with a ``.``, the current blueprint
            name (if any) will be used.
        :param _anchor: If given, append this as ``#anchor`` to the URL.
        :param _method: If given, generate the URL associated with this
            method for the endpoint.
        :param _scheme: If given, the URL will have this scheme if it
            is external.
        :param _external: If given, prefer the URL to be internal
            (False) or require it to be external (True). External URLs
            include the scheme and domain. When not in an active
            request, URLs are external by default.
        :param values: Values to use for the variable parts of the URL
            rule. Unknown keys are appended as query string arguments,
            like ``?a=b&c=d``.

        .. versionadded:: 2.2
            Moved from ``flask.url_for``, which calls this method.
        """
        req_ctx = _cv_request.get(None)

        if req_ctx is not None:
            url_adapter = req_ctx.url_adapter
            blueprint_name = req_ctx.request.blueprint

            # If the endpoint starts with "." and the request matches a
            # blueprint, the endpoint is relative to the blueprint.
            if endpoint[:1] == ".":
                if blueprint_name is not None:
                    endpoint = f"{blueprint_name}{endpoint}"
                else:
                    endpoint = endpoint[1:]

            # When in a request, generate a URL without scheme and
            # domain by default, unless a scheme is given.
            if _external is None:
                _external = _scheme is not None
        else:
            app_ctx = _cv_app.get(None)

            # If called by helpers.url_for, an app context is active,
            # use its url_adapter. Otherwise, app.url_for was called
            # directly, build an adapter.
            if app_ctx is not None:
                url_adapter = app_ctx.url_adapter
            else:
                url_adapter = self.create_url_adapter(None)

            if url_adapter is None:
                raise RuntimeError(
                    "Unable to build URLs outside an active request"
                    " without 'SERVER_NAME' configured. Also configure"
                    " 'APPLICATION_ROOT' and 'PREFERRED_URL_SCHEME' as"
                    " needed."
                )

            # When outside a request, generate a URL with scheme and
            # domain by default.
            if _external is None:
                _external = True

        # It is an error to set _scheme when _external=False, in order
        # to avoid accidental insecure URLs.
        if _scheme is not None and not _external:
            raise ValueError("When specifying '_scheme', '_external' must be True.")

        self.inject_url_defaults(endpoint, values)

        try:
            rv = url_adapter.build(  # type: ignore[union-attr]
                endpoint,
                values,
                method=_method,
                url_scheme=_scheme,
                force_external=_external,
            )
        except BuildError as error:
            values.update(
                _anchor=_anchor, _method=_method, _scheme=_scheme, _external=_external
            )
            return self.handle_url_build_error(error, endpoint, values)

        if _anchor is not None:
            _anchor = _url_quote(_anchor, safe="%!#$&'()*+,/:;=?@")
            rv = f"{rv}#{_anchor}"

        return rv

    def make_response(self, rv: ft.ResponseReturnValue) -> Response:
        """Convert the return value from a view function to an instance of
        :attr:`response_class`.

        :param rv: the return value from the view function. The view function
            must return a response. Returning ``None``, or the view ending
            without returning, is not allowed. The following types are allowed
            for ``view_rv``:

            ``str``
                A response object is created with the string encoded to UTF-8
                as the body.

            ``bytes``
                A response object is created with the bytes as the body.

            ``dict``
                A dictionary that will be jsonify'd before being returned.

            ``list``
                A list that will be jsonify'd before being returned.

            ``generator`` or ``iterator``
                A generator that returns ``str`` or ``bytes`` to be
                streamed as the response.

            ``tuple``
                Either ``(body, status, headers)``, ``(body, status)``, or
                ``(body, headers)``, where ``body`` is any of the other types
                allowed here, ``status`` is a string or an integer, and
                ``headers`` is a dictionary or a list of ``(key, value)``
                tuples. If ``body`` is a :attr:`response_class` instance,
                ``status`` overwrites the exiting value and ``headers`` are
                extended.

            :attr:`response_class`
                The object is returned unchanged.

            other :class:`~werkzeug.wrappers.Response` class
                The object is coerced to :attr:`response_class`.

            :func:`callable`
                The function is called as a WSGI application. The result is
                used to create a response object.

        .. versionchanged:: 2.2
            A generator will be converted to a streaming response.
            A list will be converted to a JSON response.

        .. versionchanged:: 1.1
            A dict will be converted to a JSON response.

        .. versionchanged:: 0.9
           Previously a tuple was interpreted as the arguments for the
           response object.
        """

        status: int | None = None
        headers: HeadersValue | None = None

        # unpack tuple returns
        if isinstance(rv, tuple):
            len_rv = len(rv)

            # a 3-tuple is unpacked directly
            if len_rv == 3:
                rv, status, headers = rv  # type: ignore[misc]
            # decide if a 2-tuple has status or headers
            elif len_rv == 2:
                if isinstance(rv[1], (Headers, dict, tuple, list)):
                    rv, headers = rv  # pyright: ignore
                else:
                    rv, status = rv  # type: ignore[assignment,misc]
            # other sized tuples are not allowed
            else:
                raise TypeError(
                    "The view function did not return a valid response tuple."
                    " The tuple must have the form (body, status, headers),"
                    " (body, status), or (body, headers)."
                )

        # the body must not be None
        if rv is None:
            raise TypeError(
                f"The view function for {request.endpoint!r} did not"
                " return a valid response. The function either returned"
                " None or ended without a return statement."
            )

        # make sure the body is an instance of the response class
        if not isinstance(rv, self.response_class):
            if isinstance(rv, (str, bytes, bytearray)) or isinstance(rv, cabc.Iterator):
                # let the response class set the status and headers instead of
                # waiting to do it manually, so that the class can handle any
                # special logic
                rv = self.response_class(
                    rv,  # pyright: ignore
                    status=status,
                    headers=headers,  # type: ignore[arg-type]
                )
                status = headers = None
            elif isinstance(rv, (dict, list)):
                rv = self.json.response(rv)
            elif isinstance(rv, BaseResponse) or callable(rv):
                # evaluate a WSGI callable, or coerce a different response
                # class to the correct type
                try:
                    rv = self.response_class.force_type(
                        rv,  # type: ignore[arg-type]
                        request.environ,
                    )
                except TypeError as e:
                    raise TypeError(
                        f"{e}\nThe view function did not return a valid"
                        " response. The return type must be a string,"
                        " dict, list, tuple with headers or status,"
                        " Response instance, or WSGI callable, but it"
                        f" was a {type(rv).__name__}."
                    ).with_traceback(sys.exc_info()[2]) from None
            else:
                raise TypeError(
                    "The view function did not return a valid"
                    " response. The return type must be a string,"
                    " dict, list, tuple with headers or status,"
                    " Response instance, or WSGI callable, but it was a"
                    f" {type(rv).__name__}."
                )

        rv = t.cast(Response, rv)
        # prefer the status if it was provided
        if status is not None:
            if isinstance(status, (str, bytes, bytearray)):
                rv.status = status
            else:
                rv.status_code = status

        # extend existing headers with provided headers
        if headers:
            rv.headers.update(headers)

        return rv

    def preprocess_request(self) -> ft.ResponseReturnValue | None:
        """Called before the request is dispatched. Calls
        :attr:`url_value_preprocessors` registered with the app and the
        current blueprint (if any). Then calls :attr:`before_request_funcs`
        registered with the app and the blueprint.

        If any :meth:`before_request` handler returns a non-None value, the
        value is handled as if it was the return value from the view, and
        further request handling is stopped.
        """
        names = (None, *reversed(request.blueprints))

        for name in names:
            if name in self.url_value_preprocessors:
                for url_func in self.url_value_preprocessors[name]:
                    url_func(request.endpoint, request.view_args)

        for name in names:
            if name in self.before_request_funcs:
                for before_func in self.before_request_funcs[name]:
                    rv = self.ensure_sync(before_func)()

                    if rv is not None:
                        return rv  # type: ignore[no-any-return]

        return None

    def process_response(self, response: Response) -> Response:
        """Can be overridden in order to modify the response object
        before it's sent to the WSGI server.  By default this will
        call all the :meth:`after_request` decorated functions.

        .. versionchanged:: 0.5
           As of Flask 0.5 the functions registered for after request
           execution are called in reverse order of registration.

        :param response: a :attr:`response_class` object.
        :return: a new response object or the same, has to be an
                 instance of :attr:`response_class`.
        """
        ctx = request_ctx._get_current_object()  # type: ignore[attr-defined]

        for func in ctx._after_request_functions:
            response = self.ensure_sync(func)(response)

        for name in chain(request.blueprints, (None,)):
            if name in self.after_request_funcs:
                for func in reversed(self.after_request_funcs[name]):
                    response = self.ensure_sync(func)(response)

        if not self.session_interface.is_null_session(ctx.session):
            self.session_interface.save_session(self, ctx.session, response)

        return response

    def do_teardown_request(
        self,
        exc: BaseException | None = _sentinel,  # type: ignore[assignment]
    ) -> None:
        """Called after the request is dispatched and the response is
        returned, right before the request context is popped.

        This calls all functions decorated with
        :meth:`teardown_request`, and :meth:`Blueprint.teardown_request`
        if a blueprint handled the request. Finally, the
        :data:`request_tearing_down` signal is sent.

        This is called by
        :meth:`RequestContext.pop() <flask.ctx.RequestContext.pop>`,
        which may be delayed during testing to maintain access to
        resources.

        :param exc: An unhandled exception raised while dispatching the
            request. Detected from the current exception information if
            not passed. Passed to each teardown function.

        .. versionchanged:: 0.9
            Added the ``exc`` argument.
        """
        if exc is _sentinel:
            exc = sys.exc_info()[1]

        for name in chain(request.blueprints, (None,)):
            if name in self.teardown_request_funcs:
                for func in reversed(self.teardown_request_funcs[name]):
                    self.ensure_sync(func)(exc)

        request_tearing_down.send(self, _async_wrapper=self.ensure_sync, exc=exc)

    def do_teardown_appcontext(
        self,
        exc: BaseException | None = _sentinel,  # type: ignore[assignment]
    ) -> None:
        """Called right before the application context is popped.

        When handling a request, the application context is popped
        after the request context. See :meth:`do_teardown_request`.

        This calls all functions decorated with
        :meth:`teardown_appcontext`. Then the
        :data:`appcontext_tearing_down` signal is sent.

        This is called by
        :meth:`AppContext.pop() <flask.ctx.AppContext.pop>`.

        .. versionadded:: 0.9
        """
        if exc is _sentinel:
            exc = sys.exc_info()[1]

        for func in reversed(self.teardown_appcontext_funcs):
            self.ensure_sync(func)(exc)

        appcontext_tearing_down.send(self, _async_wrapper=self.ensure_sync, exc=exc)

    def app_context(self) -> AppContext:
        """Create an :class:`~flask.ctx.AppContext`. Use as a ``with``
        block to push the context, which will make :data:`current_app`
        point at this application.

        An application context is automatically pushed by
        :meth:`RequestContext.push() <flask.ctx.RequestContext.push>`
        when handling a request, and when running a CLI command. Use
        this to manually create a context outside of these situations.

        ::

            with app.app_context():
                init_db()

        See :doc:`/appcontext`.

        .. versionadded:: 0.9
        """
        return AppContext(self)

    def request_context(self, environ: WSGIEnvironment) -> RequestContext:
        """Create a :class:`~flask.ctx.RequestContext` representing a
        WSGI environment. Use a ``with`` block to push the context,
        which will make :data:`request` point at this request.

        See :doc:`/reqcontext`.

        Typically you should not call this from your own code. A request
        context is automatically pushed by the :meth:`wsgi_app` when
        handling a request. Use :meth:`test_request_context` to create
        an environment and context instead of this method.

        :param environ: a WSGI environment
        """
        return RequestContext(self, environ)

    def test_request_context(self, *args: t.Any, **kwargs: t.Any) -> RequestContext:
        """Create a :class:`~flask.ctx.RequestContext` for a WSGI
        environment created from the given values. This is mostly useful
        during testing, where you may want to run a function that uses
        request data without dispatching a full request.

        See :doc:`/reqcontext`.

        Use a ``with`` block to push the context, which will make
        :data:`request` point at the request for the created
        environment. ::

            with app.test_request_context(...):
                generate_report()

        When using the shell, it may be easier to push and pop the
        context manually to avoid indentation. ::

            ctx = app.test_request_context(...)
            ctx.push()
            ...
            ctx.pop()

        Takes the same arguments as Werkzeug's
        :class:`~werkzeug.test.EnvironBuilder`, with some defaults from
        the application. See the linked Werkzeug docs for most of the
        available arguments. Flask-specific behavior is listed here.

        :param path: URL path being requested.
        :param base_url: Base URL where the app is being served, which
            ``path`` is relative to. If not given, built from
            :data:`PREFERRED_URL_SCHEME`, ``subdomain``,
            :data:`SERVER_NAME`, and :data:`APPLICATION_ROOT`.
        :param subdomain: Subdomain name to append to
            :data:`SERVER_NAME`.
        :param url_scheme: Scheme to use instead of
            :data:`PREFERRED_URL_SCHEME`.
        :param data: The request body, either as a string or a dict of
            form keys and values.
        :param json: If given, this is serialized as JSON and passed as
            ``data``. Also defaults ``content_type`` to
            ``application/json``.
        :param args: other positional arguments passed to
            :class:`~werkzeug.test.EnvironBuilder`.
        :param kwargs: other keyword arguments passed to
            :class:`~werkzeug.test.EnvironBuilder`.
        """
        from .testing import EnvironBuilder

        builder = EnvironBuilder(self, *args, **kwargs)

        try:
            return self.request_context(builder.get_environ())
        finally:
            builder.close()

    def wsgi_app(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> cabc.Iterable[bytes]:
        """The actual WSGI application. This is not implemented in
        :meth:`__call__` so that middlewares can be applied without
        losing a reference to the app object. Instead of doing this::

            app = MyMiddleware(app)

        It's a better idea to do this instead::

            app.wsgi_app = MyMiddleware(app.wsgi_app)

        Then you still have the original application object around and
        can continue to call methods on it.

        .. versionchanged:: 0.7
            Teardown events for the request and app contexts are called
            even if an unhandled error occurs. Other events may not be
            called depending on when an error occurs during dispatch.
            See :ref:`callbacks-and-errors`.

        :param environ: A WSGI environment.
        :param start_response: A callable accepting a status code,
            a list of headers, and an optional exception context to
            start the response.
        """
        ctx = self.request_context(environ)
        error: BaseException | None = None
        try:
            try:
                ctx.push()
                response = self.full_dispatch_request()
            except Exception as e:
                error = e
                response = self.handle_exception(e)
            except:  # noqa: B001
                error = sys.exc_info()[1]
                raise
            return response(environ, start_response)
        finally:
            if "werkzeug.debug.preserve_context" in environ:
                environ["werkzeug.debug.preserve_context"](_cv_app.get())
                environ["werkzeug.debug.preserve_context"](_cv_request.get())

            if error is not None and self.should_ignore_error(error):
                error = None

            ctx.pop(error)

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> cabc.Iterable[bytes]:
        """The WSGI server calls the Flask application object as the
        WSGI application. This calls :meth:`wsgi_app`, which can be
        wrapped to apply middleware.
        """
        return self.wsgi_app(environ, start_response)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flatbuffers\flexbuffers.py ===
# Lint as: python3
# Copyright 2020 Google Inc. All rights reserved.
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
"""Implementation of FlexBuffers binary format.

For more info check https://google.github.io/flatbuffers/flexbuffers.html and
corresponding C++ implementation at
https://github.com/google/flatbuffers/blob/master/include/flatbuffers/flexbuffers.h
"""

# pylint: disable=invalid-name
# TODO(dkovalev): Add type hints everywhere, so tools like pytypes could work.

import array
import contextlib
import enum
import struct

__all__ = ('Type', 'Builder', 'GetRoot', 'Dumps', 'Loads')


class BitWidth(enum.IntEnum):
  """Supported bit widths of value types.

  These are used in the lower 2 bits of a type field to determine the size of
  the elements (and or size field) of the item pointed to (e.g. vector).
  """
  W8 = 0  # 2^0 = 1 byte
  W16 = 1  # 2^1 = 2 bytes
  W32 = 2  # 2^2 = 4 bytes
  W64 = 3  # 2^3 = 8 bytes

  @staticmethod
  def U(value):
    """Returns the minimum `BitWidth` to encode unsigned integer value."""
    assert value >= 0

    if value < (1 << 8):
      return BitWidth.W8
    elif value < (1 << 16):
      return BitWidth.W16
    elif value < (1 << 32):
      return BitWidth.W32
    elif value < (1 << 64):
      return BitWidth.W64
    else:
      raise ValueError('value is too big to encode: %s' % value)

  @staticmethod
  def I(value):
    """Returns the minimum `BitWidth` to encode signed integer value."""
    # -2^(n-1) <=     value < 2^(n-1)
    # -2^n     <= 2 * value < 2^n
    # 2 * value < 2^n, when value >= 0 or 2 * (-value) <= 2^n, when value < 0
    # 2 * value < 2^n, when value >= 0 or 2 * (-value) - 1 < 2^n, when value < 0
    #
    # if value >= 0:
    #   return BitWidth.U(2 * value)
    # else:
    #   return BitWidth.U(2 * (-value) - 1)  # ~x = -x - 1
    value *= 2
    return BitWidth.U(value if value >= 0 else ~value)

  @staticmethod
  def F(value):
    """Returns the `BitWidth` to encode floating point value."""
    if struct.unpack('<f', struct.pack('<f', value))[0] == value:
      return BitWidth.W32
    return BitWidth.W64

  @staticmethod
  def B(byte_width):
    return {
        1: BitWidth.W8,
        2: BitWidth.W16,
        4: BitWidth.W32,
        8: BitWidth.W64
    }[byte_width]


I = {1: 'b', 2: 'h', 4: 'i', 8: 'q'}  # Integer formats
U = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}  # Unsigned integer formats
F = {4: 'f', 8: 'd'}  # Floating point formats


def _Unpack(fmt, buf):
  return struct.unpack('<%s' % fmt[len(buf)], buf)[0]


def _UnpackVector(fmt, buf, length):
  byte_width = len(buf) // length
  return struct.unpack('<%d%s' % (length, fmt[byte_width]), buf)


def _Pack(fmt, value, byte_width):
  return struct.pack('<%s' % fmt[byte_width], value)


def _PackVector(fmt, values, byte_width):
  return struct.pack('<%d%s' % (len(values), fmt[byte_width]), *values)


def _Mutate(fmt, buf, value, byte_width, value_bit_width):
  if (1 << value_bit_width) <= byte_width:
    buf[:byte_width] = _Pack(fmt, value, byte_width)
    return True
  return False


# Computes how many bytes you'd have to pad to be able to write an
# "scalar_size" scalar if the buffer had grown to "buf_size",
# "scalar_size" is a power of two.
def _PaddingBytes(buf_size, scalar_size):
  # ((buf_size + (scalar_size - 1)) // scalar_size) * scalar_size - buf_size
  return -buf_size & (scalar_size - 1)


def _ShiftSlice(s, offset, length):
  start = offset + (0 if s.start is None else s.start)
  stop = offset + (length if s.stop is None else s.stop)
  return slice(start, stop, s.step)


# https://en.cppreference.com/w/cpp/algorithm/lower_bound
def _LowerBound(values, value, pred):
  """Implementation of C++ std::lower_bound() algorithm."""
  first, last = 0, len(values)
  count = last - first
  while count > 0:
    i = first
    step = count // 2
    i += step
    if pred(values[i], value):
      i += 1
      first = i
      count -= step + 1
    else:
      count = step
  return first


# https://en.cppreference.com/w/cpp/algorithm/binary_search
def _BinarySearch(values, value, pred=lambda x, y: x < y):
  """Implementation of C++ std::binary_search() algorithm."""
  index = _LowerBound(values, value, pred)
  if index != len(values) and not pred(value, values[index]):
    return index
  return -1


class Type(enum.IntEnum):
  """Supported types of encoded data.

  These are used as the upper 6 bits of a type field to indicate the actual
  type.
  """
  NULL = 0
  INT = 1
  UINT = 2
  FLOAT = 3
  # Types above stored inline, types below store an offset.
  KEY = 4
  STRING = 5
  INDIRECT_INT = 6
  INDIRECT_UINT = 7
  INDIRECT_FLOAT = 8
  MAP = 9
  VECTOR = 10  # Untyped.

  VECTOR_INT = 11  # Typed any size (stores no type table).
  VECTOR_UINT = 12
  VECTOR_FLOAT = 13
  VECTOR_KEY = 14
  # DEPRECATED, use VECTOR or VECTOR_KEY instead.
  # Read test.cpp/FlexBuffersDeprecatedTest() for details on why.
  VECTOR_STRING_DEPRECATED = 15

  VECTOR_INT2 = 16  # Typed tuple (no type table, no size field).
  VECTOR_UINT2 = 17
  VECTOR_FLOAT2 = 18
  VECTOR_INT3 = 19  # Typed triple (no type table, no size field).
  VECTOR_UINT3 = 20
  VECTOR_FLOAT3 = 21
  VECTOR_INT4 = 22  # Typed quad (no type table, no size field).
  VECTOR_UINT4 = 23
  VECTOR_FLOAT4 = 24

  BLOB = 25
  BOOL = 26
  VECTOR_BOOL = 36  # To do the same type of conversion of type to vector type

  @staticmethod
  def Pack(type_, bit_width):
    return (int(type_) << 2) | bit_width

  @staticmethod
  def Unpack(packed_type):
    return 1 << (packed_type & 0b11), Type(packed_type >> 2)

  @staticmethod
  def IsInline(type_):
    return type_ <= Type.FLOAT or type_ == Type.BOOL

  @staticmethod
  def IsTypedVector(type_):
    return Type.VECTOR_INT <= type_ <= Type.VECTOR_STRING_DEPRECATED or \
           type_ == Type.VECTOR_BOOL

  @staticmethod
  def IsTypedVectorElementType(type_):
    return Type.INT <= type_ <= Type.STRING or type_ == Type.BOOL

  @staticmethod
  def ToTypedVectorElementType(type_):
    if not Type.IsTypedVector(type_):
      raise ValueError('must be typed vector type')

    return Type(type_ - Type.VECTOR_INT + Type.INT)

  @staticmethod
  def IsFixedTypedVector(type_):
    return Type.VECTOR_INT2 <= type_ <= Type.VECTOR_FLOAT4

  @staticmethod
  def IsFixedTypedVectorElementType(type_):
    return Type.INT <= type_ <= Type.FLOAT

  @staticmethod
  def ToFixedTypedVectorElementType(type_):
    if not Type.IsFixedTypedVector(type_):
      raise ValueError('must be fixed typed vector type')

    # 3 types each, starting from length 2.
    fixed_type = type_ - Type.VECTOR_INT2
    return Type(fixed_type % 3 + Type.INT), fixed_type // 3 + 2

  @staticmethod
  def ToTypedVector(element_type, fixed_len=0):
    """Converts element type to corresponding vector type.

    Args:
      element_type: vector element type
      fixed_len: number of elements: 0 for typed vector; 2, 3, or 4 for fixed
        typed vector.

    Returns:
      Typed vector type or fixed typed vector type.
    """
    if fixed_len == 0:
      if not Type.IsTypedVectorElementType(element_type):
        raise ValueError('must be typed vector element type')
    else:
      if not Type.IsFixedTypedVectorElementType(element_type):
        raise ValueError('must be fixed typed vector element type')

    offset = element_type - Type.INT
    if fixed_len == 0:
      return Type(offset + Type.VECTOR_INT)  # TypedVector
    elif fixed_len == 2:
      return Type(offset + Type.VECTOR_INT2)  # FixedTypedVector
    elif fixed_len == 3:
      return Type(offset + Type.VECTOR_INT3)  # FixedTypedVector
    elif fixed_len == 4:
      return Type(offset + Type.VECTOR_INT4)  # FixedTypedVector
    else:
      raise ValueError('unsupported fixed_len: %s' % fixed_len)


class Buf:
  """Class to access underlying buffer object starting from the given offset."""

  def __init__(self, buf, offset):
    self._buf = buf
    self._offset = offset if offset >= 0 else len(buf) + offset
    self._length = len(buf) - self._offset

  def __getitem__(self, key):
    if isinstance(key, slice):
      return self._buf[_ShiftSlice(key, self._offset, self._length)]
    elif isinstance(key, int):
      return self._buf[self._offset + key]
    else:
      raise TypeError('invalid key type')

  def __setitem__(self, key, value):
    if isinstance(key, slice):
      self._buf[_ShiftSlice(key, self._offset, self._length)] = value
    elif isinstance(key, int):
      self._buf[self._offset + key] = key
    else:
      raise TypeError('invalid key type')

  def __repr__(self):
    return 'buf[%d:]' % self._offset

  def Find(self, sub):
    """Returns the lowest index where the sub subsequence is found."""
    return self._buf[self._offset:].find(sub)

  def Slice(self, offset):
    """Returns new `Buf` which starts from the given offset."""
    return Buf(self._buf, self._offset + offset)

  def Indirect(self, offset, byte_width):
    """Return new `Buf` based on the encoded offset (indirect encoding)."""
    return self.Slice(offset - _Unpack(U, self[offset:offset + byte_width]))


class Object:
  """Base class for all non-trivial data accessors."""
  __slots__ = '_buf', '_byte_width'

  def __init__(self, buf, byte_width):
    self._buf = buf
    self._byte_width = byte_width

  @property
  def ByteWidth(self):
    return self._byte_width


class Sized(Object):
  """Base class for all data accessors which need to read encoded size."""
  __slots__ = '_size',

  def __init__(self, buf, byte_width, size=0):
    super().__init__(buf, byte_width)
    if size == 0:
      self._size = _Unpack(U, self.SizeBytes)
    else:
      self._size = size

  @property
  def SizeBytes(self):
    return self._buf[-self._byte_width:0]

  def __len__(self):
    return self._size


class Blob(Sized):
  """Data accessor for the encoded blob bytes."""
  __slots__ = ()

  @property
  def Bytes(self):
    return self._buf[0:len(self)]

  def __repr__(self):
    return 'Blob(%s, size=%d)' % (self._buf, len(self))


class String(Sized):
  """Data accessor for the encoded string bytes."""
  __slots__ = ()

  @property
  def Bytes(self):
    return self._buf[0:len(self)]

  def Mutate(self, value):
    """Mutates underlying string bytes in place.

    Args:
      value: New string to replace the existing one. New string must have less
        or equal UTF-8-encoded bytes than the existing one to successfully
        mutate underlying byte buffer.

    Returns:
      Whether the value was mutated or not.
    """
    encoded = value.encode('utf-8')
    n = len(encoded)
    if n <= len(self):
      self._buf[-self._byte_width:0] = _Pack(U, n, self._byte_width)
      self._buf[0:n] = encoded
      self._buf[n:len(self)] = bytearray(len(self) - n)
      return True
    return False

  def __str__(self):
    return self.Bytes.decode('utf-8')

  def __repr__(self):
    return 'String(%s, size=%d)' % (self._buf, len(self))


class Key(Object):
  """Data accessor for the encoded key bytes."""
  __slots__ = ()

  def __init__(self, buf, byte_width):
    assert byte_width == 1
    super().__init__(buf, byte_width)

  @property
  def Bytes(self):
    return self._buf[0:len(self)]

  def __len__(self):
    return self._buf.Find(0)

  def __str__(self):
    return self.Bytes.decode('ascii')

  def __repr__(self):
    return 'Key(%s, size=%d)' % (self._buf, len(self))


class Vector(Sized):
  """Data accessor for the encoded vector bytes."""
  __slots__ = ()

  def __getitem__(self, index):
    if index < 0 or index >= len(self):
      raise IndexError('vector index %s is out of [0, %d) range' % \
          (index, len(self)))

    packed_type = self._buf[len(self) * self._byte_width + index]
    buf = self._buf.Slice(index * self._byte_width)
    return Ref.PackedType(buf, self._byte_width, packed_type)

  @property
  def Value(self):
    """Returns the underlying encoded data as a list object."""
    return [e.Value for e in self]

  def __repr__(self):
    return 'Vector(%s, byte_width=%d, size=%d)' % \
        (self._buf, self._byte_width, self._size)


class TypedVector(Sized):
  """Data accessor for the encoded typed vector or fixed typed vector bytes."""
  __slots__ = '_element_type', '_size'

  def __init__(self, buf, byte_width, element_type, size=0):
    super().__init__(buf, byte_width, size)

    if element_type == Type.STRING:
      # These can't be accessed as strings, since we don't know the bit-width
      # of the size field, see the declaration of
      # FBT_VECTOR_STRING_DEPRECATED above for details.
      # We change the type here to be keys, which are a subtype of strings,
      # and will ignore the size field. This will truncate strings with
      # embedded nulls.
      element_type = Type.KEY

    self._element_type = element_type

  @property
  def Bytes(self):
    return self._buf[:self._byte_width * len(self)]

  @property
  def ElementType(self):
    return self._element_type

  def __getitem__(self, index):
    if index < 0 or index >= len(self):
      raise IndexError('vector index %s is out of [0, %d) range' % \
          (index, len(self)))

    buf = self._buf.Slice(index * self._byte_width)
    return Ref(buf, self._byte_width, 1, self._element_type)

  @property
  def Value(self):
    """Returns underlying data as list object."""
    if not self:
      return []

    if self._element_type is Type.BOOL:
      return [bool(e) for e in _UnpackVector(U, self.Bytes, len(self))]
    elif self._element_type is Type.INT:
      return list(_UnpackVector(I, self.Bytes, len(self)))
    elif self._element_type is Type.UINT:
      return list(_UnpackVector(U, self.Bytes, len(self)))
    elif self._element_type is Type.FLOAT:
      return list(_UnpackVector(F, self.Bytes, len(self)))
    elif self._element_type is Type.KEY:
      return [e.AsKey for e in self]
    elif self._element_type is Type.STRING:
      return [e.AsString for e in self]
    else:
      raise TypeError('unsupported element_type: %s' % self._element_type)

  def __repr__(self):
    return 'TypedVector(%s, byte_width=%d, element_type=%s, size=%d)' % \
        (self._buf, self._byte_width, self._element_type, self._size)


class Map(Vector):
  """Data accessor for the encoded map bytes."""

  @staticmethod
  def CompareKeys(a, b):
    if isinstance(a, Ref):
      a = a.AsKeyBytes
    if isinstance(b, Ref):
      b = b.AsKeyBytes
    return a < b

  def __getitem__(self, key):
    if isinstance(key, int):
      return super().__getitem__(key)

    index = _BinarySearch(self.Keys, key.encode('ascii'), self.CompareKeys)
    if index != -1:
      return super().__getitem__(index)

    raise KeyError(key)

  @property
  def Keys(self):
    byte_width = _Unpack(U, self._buf[-2 * self._byte_width:-self._byte_width])
    buf = self._buf.Indirect(-3 * self._byte_width, self._byte_width)
    return TypedVector(buf, byte_width, Type.KEY)

  @property
  def Values(self):
    return Vector(self._buf, self._byte_width)

  @property
  def Value(self):
    return {k.Value: v.Value for k, v in zip(self.Keys, self.Values)}

  def __repr__(self):
    return 'Map(%s, size=%d)' % (self._buf, len(self))


class Ref:
  """Data accessor for the encoded data bytes."""
  __slots__ = '_buf', '_parent_width', '_byte_width', '_type'

  @staticmethod
  def PackedType(buf, parent_width, packed_type):
    byte_width, type_ = Type.Unpack(packed_type)
    return Ref(buf, parent_width, byte_width, type_)

  def __init__(self, buf, parent_width, byte_width, type_):
    self._buf = buf
    self._parent_width = parent_width
    self._byte_width = byte_width
    self._type = type_

  def __repr__(self):
    return 'Ref(%s, parent_width=%d, byte_width=%d, type_=%s)' % \
            (self._buf, self._parent_width, self._byte_width, self._type)

  @property
  def _Bytes(self):
    return self._buf[:self._parent_width]

  def _ConvertError(self, target_type):
    raise TypeError('cannot convert %s to %s' % (self._type, target_type))

  def _Indirect(self):
    return self._buf.Indirect(0, self._parent_width)

  @property
  def IsNull(self):
    return self._type is Type.NULL

  @property
  def IsBool(self):
    return self._type is Type.BOOL

  @property
  def AsBool(self):
    if self._type is Type.BOOL:
      return bool(_Unpack(U, self._Bytes))
    else:
      return self.AsInt != 0

  def MutateBool(self, value):
    """Mutates underlying boolean value bytes in place.

    Args:
      value: New boolean value.

    Returns:
      Whether the value was mutated or not.
    """
    return self.IsBool and \
           _Mutate(U, self._buf, value, self._parent_width, BitWidth.W8)

  @property
  def IsNumeric(self):
    return self.IsInt or self.IsFloat

  @property
  def IsInt(self):
    return self._type in (Type.INT, Type.INDIRECT_INT, Type.UINT,
                          Type.INDIRECT_UINT)

  @property
  def AsInt(self):
    """Returns current reference as integer value."""
    if self.IsNull:
      return 0
    elif self.IsBool:
      return int(self.AsBool)
    elif self._type is Type.INT:
      return _Unpack(I, self._Bytes)
    elif self._type is Type.INDIRECT_INT:
      return _Unpack(I, self._Indirect()[:self._byte_width])
    if self._type is Type.UINT:
      return _Unpack(U, self._Bytes)
    elif self._type is Type.INDIRECT_UINT:
      return _Unpack(U, self._Indirect()[:self._byte_width])
    elif self.IsString:
      return len(self.AsString)
    elif self.IsKey:
      return len(self.AsKey)
    elif self.IsBlob:
      return len(self.AsBlob)
    elif self.IsVector:
      return len(self.AsVector)
    elif self.IsTypedVector:
      return len(self.AsTypedVector)
    elif self.IsFixedTypedVector:
      return len(self.AsFixedTypedVector)
    else:
      raise self._ConvertError(Type.INT)

  def MutateInt(self, value):
    """Mutates underlying integer value bytes in place.

    Args:
      value: New integer value. It must fit to the byte size of the existing
        encoded value.

    Returns:
      Whether the value was mutated or not.
    """
    if self._type is Type.INT:
      return _Mutate(I, self._buf, value, self._parent_width, BitWidth.I(value))
    elif self._type is Type.INDIRECT_INT:
      return _Mutate(I, self._Indirect(), value, self._byte_width,
                     BitWidth.I(value))
    elif self._type is Type.UINT:
      return _Mutate(U, self._buf, value, self._parent_width, BitWidth.U(value))
    elif self._type is Type.INDIRECT_UINT:
      return _Mutate(U, self._Indirect(), value, self._byte_width,
                     BitWidth.U(value))
    else:
      return False

  @property
  def IsFloat(self):
    return self._type in (Type.FLOAT, Type.INDIRECT_FLOAT)

  @property
  def AsFloat(self):
    """Returns current reference as floating point value."""
    if self.IsNull:
      return 0.0
    elif self.IsBool:
      return float(self.AsBool)
    elif self.IsInt:
      return float(self.AsInt)
    elif self._type is Type.FLOAT:
      return _Unpack(F, self._Bytes)
    elif self._type is Type.INDIRECT_FLOAT:
      return _Unpack(F, self._Indirect()[:self._byte_width])
    elif self.IsString:
      return float(self.AsString)
    elif self.IsVector:
      return float(len(self.AsVector))
    elif self.IsTypedVector():
      return float(len(self.AsTypedVector))
    elif self.IsFixedTypedVector():
      return float(len(self.FixedTypedVector))
    else:
      raise self._ConvertError(Type.FLOAT)

  def MutateFloat(self, value):
    """Mutates underlying floating point value bytes in place.

    Args:
      value: New float value. It must fit to the byte size of the existing
        encoded value.

    Returns:
      Whether the value was mutated or not.
    """
    if self._type is Type.FLOAT:
      return _Mutate(F, self._buf, value, self._parent_width,
                     BitWidth.B(self._parent_width))
    elif self._type is Type.INDIRECT_FLOAT:
      return _Mutate(F, self._Indirect(), value, self._byte_width,
                     BitWidth.B(self._byte_width))
    else:
      return False

  @property
  def IsKey(self):
    return self._type is Type.KEY

  @property
  def AsKeyBytes(self):
    if self.IsKey:
      return Key(self._Indirect(), self._byte_width).Bytes
    else:
      raise self._ConvertError(Type.KEY)

  @property
  def AsKey(self):
    if self.IsKey:
      return str(Key(self._Indirect(), self._byte_width))
    else:
      raise self._ConvertError(Type.KEY)

  @property
  def IsString(self):
    return self._type is Type.STRING

  @property
  def AsStringBytes(self):
    if self.IsString:
      return String(self._Indirect(), self._byte_width).Bytes
    elif self.IsKey:
      return self.AsKeyBytes
    else:
      raise self._ConvertError(Type.STRING)

  @property
  def AsString(self):
    if self.IsString:
      return str(String(self._Indirect(), self._byte_width))
    elif self.IsKey:
      return self.AsKey
    else:
      raise self._ConvertError(Type.STRING)

  def MutateString(self, value):
    return String(self._Indirect(), self._byte_width).Mutate(value)

  @property
  def IsBlob(self):
    return self._type is Type.BLOB

  @property
  def AsBlob(self):
    if self.IsBlob:
      return Blob(self._Indirect(), self._byte_width).Bytes
    else:
      raise self._ConvertError(Type.BLOB)

  @property
  def IsAnyVector(self):
    return self.IsVector or self.IsTypedVector or self.IsFixedTypedVector()

  @property
  def IsVector(self):
    return self._type in (Type.VECTOR, Type.MAP)

  @property
  def AsVector(self):
    if self.IsVector:
      return Vector(self._Indirect(), self._byte_width)
    else:
      raise self._ConvertError(Type.VECTOR)

  @property
  def IsTypedVector(self):
    return Type.IsTypedVector(self._type)

  @property
  def AsTypedVector(self):
    if self.IsTypedVector:
      return TypedVector(self._Indirect(), self._byte_width,
                         Type.ToTypedVectorElementType(self._type))
    else:
      raise self._ConvertError('TYPED_VECTOR')

  @property
  def IsFixedTypedVector(self):
    return Type.IsFixedTypedVector(self._type)

  @property
  def AsFixedTypedVector(self):
    if self.IsFixedTypedVector:
      element_type, size = Type.ToFixedTypedVectorElementType(self._type)
      return TypedVector(self._Indirect(), self._byte_width, element_type, size)
    else:
      raise self._ConvertError('FIXED_TYPED_VECTOR')

  @property
  def IsMap(self):
    return self._type is Type.MAP

  @property
  def AsMap(self):
    if self.IsMap:
      return Map(self._Indirect(), self._byte_width)
    else:
      raise self._ConvertError(Type.MAP)

  @property
  def Value(self):
    """Converts current reference to value of corresponding type.

    This is equivalent to calling `AsInt` for integer values, `AsFloat` for
    floating point values, etc.

    Returns:
      Value of corresponding type.
    """
    if self.IsNull:
      return None
    elif self.IsBool:
      return self.AsBool
    elif self.IsInt:
      return self.AsInt
    elif self.IsFloat:
      return self.AsFloat
    elif self.IsString:
      return self.AsString
    elif self.IsKey:
      return self.AsKey
    elif self.IsBlob:
      return self.AsBlob
    elif self.IsMap:
      return self.AsMap.Value
    elif self.IsVector:
      return self.AsVector.Value
    elif self.IsTypedVector:
      return self.AsTypedVector.Value
    elif self.IsFixedTypedVector:
      return self.AsFixedTypedVector.Value
    else:
      raise TypeError('cannot convert %r to value' % self)


def _IsIterable(obj):
  try:
    iter(obj)
    return True
  except TypeError:
    return False


class Value:
  """Class to represent given value during the encoding process."""

  @staticmethod
  def Null():
    return Value(0, Type.NULL, BitWidth.W8)

  @staticmethod
  def Bool(value):
    return Value(value, Type.BOOL, BitWidth.W8)

  @staticmethod
  def Int(value, bit_width):
    return Value(value, Type.INT, bit_width)

  @staticmethod
  def UInt(value, bit_width):
    return Value(value, Type.UINT, bit_width)

  @staticmethod
  def Float(value, bit_width):
    return Value(value, Type.FLOAT, bit_width)

  @staticmethod
  def Key(offset):
    return Value(offset, Type.KEY, BitWidth.W8)

  def __init__(self, value, type_, min_bit_width):
    self._value = value
    self._type = type_

    # For scalars: of itself, for vector: of its elements, for string: length.
    self._min_bit_width = min_bit_width

  @property
  def Value(self):
    return self._value

  @property
  def Type(self):
    return self._type

  @property
  def MinBitWidth(self):
    return self._min_bit_width

  def StoredPackedType(self, parent_bit_width=BitWidth.W8):
    return Type.Pack(self._type, self.StoredWidth(parent_bit_width))

  # We have an absolute offset, but want to store a relative offset
  # elem_index elements beyond the current buffer end. Since whether
  # the relative offset fits in a certain byte_width depends on
  # the size of the elements before it (and their alignment), we have
  # to test for each size in turn.
  def ElemWidth(self, buf_size, elem_index=0):
    if Type.IsInline(self._type):
      return self._min_bit_width
    for byte_width in 1, 2, 4, 8:
      offset_loc = buf_size + _PaddingBytes(buf_size, byte_width) + \
                   elem_index * byte_width
      bit_width = BitWidth.U(offset_loc - self._value)
      if byte_width == (1 << bit_width):
        return bit_width
    raise ValueError('relative offset is too big')

  def StoredWidth(self, parent_bit_width=BitWidth.W8):
    if Type.IsInline(self._type):
      return max(self._min_bit_width, parent_bit_width)
    return self._min_bit_width

  def __repr__(self):
    return 'Value(%s, %s, %s)' % (self._value, self._type, self._min_bit_width)

  def __str__(self):
    return str(self._value)


def InMap(func):
  def wrapper(self, *args, **kwargs):
    if isinstance(args[0], str):
      self.Key(args[0])
      func(self, *args[1:], **kwargs)
    else:
      func(self, *args, **kwargs)
  return wrapper


def InMapForString(func):
  def wrapper(self, *args):
    if len(args) == 1:
      func(self, args[0])
    elif len(args) == 2:
      self.Key(args[0])
      func(self, args[1])
    else:
      raise ValueError('invalid number of arguments')
  return wrapper


class Pool:
  """Collection of (data, offset) pairs sorted by data for quick access."""

  def __init__(self):
    self._pool = []  # sorted list of (data, offset) tuples

  def FindOrInsert(self, data, offset):
    do = data, offset
    index = _BinarySearch(self._pool, do, lambda a, b: a[0] < b[0])
    if index != -1:
      _, offset = self._pool[index]
      return offset
    self._pool.insert(index, do)
    return None

  def Clear(self):
    self._pool = []

  @property
  def Elements(self):
    return [data for data, _ in self._pool]


class Builder:
  """Helper class to encode structural data into flexbuffers format."""

  def __init__(self,
               share_strings=False,
               share_keys=True,
               force_min_bit_width=BitWidth.W8):
    self._share_strings = share_strings
    self._share_keys = share_keys
    self._force_min_bit_width = force_min_bit_width

    self._string_pool = Pool()
    self._key_pool = Pool()

    self._finished = False
    self._buf = bytearray()
    self._stack = []

  def __len__(self):
    return len(self._buf)

  @property
  def StringPool(self):
    return self._string_pool

  @property
  def KeyPool(self):
    return self._key_pool

  def Clear(self):
    self._string_pool.Clear()
    self._key_pool.Clear()
    self._finished = False
    self._buf = bytearray()
    self._stack = []

  def Finish(self):
    """Finishes encoding process and returns underlying buffer."""
    if self._finished:
      raise RuntimeError('builder has been already finished')

    # If you hit this exception, you likely have objects that were never
    # included in a parent. You need to have exactly one root to finish a
    # buffer. Check your Start/End calls are matched, and all objects are inside
    # some other object.
    if len(self._stack) != 1:
      raise RuntimeError('internal stack size must be one')

    value = self._stack[0]
    byte_width = self._Align(value.ElemWidth(len(self._buf)))
    self._WriteAny(value, byte_width=byte_width)  # Root value
    self._Write(U, value.StoredPackedType(), byte_width=1)  # Root type
    self._Write(U, byte_width, byte_width=1)  # Root size

    self.finished = True
    return self._buf

  def _ReadKey(self, offset):
    key = self._buf[offset:]
    return key[:key.find(0)]

  def _Align(self, alignment):
    byte_width = 1 << alignment
    self._buf.extend(b'\x00' * _PaddingBytes(len(self._buf), byte_width))
    return byte_width

  def _Write(self, fmt, value, byte_width):
    self._buf.extend(_Pack(fmt, value, byte_width))

  def _WriteVector(self, fmt, values, byte_width):
    self._buf.extend(_PackVector(fmt, values, byte_width))

  def _WriteOffset(self, offset, byte_width):
    relative_offset = len(self._buf) - offset
    assert byte_width == 8 or relative_offset < (1 << (8 * byte_width))
    self._Write(U, relative_offset, byte_width)

  def _WriteAny(self, value, byte_width):
    fmt = {
        Type.NULL: U, Type.BOOL: U, Type.INT: I, Type.UINT: U, Type.FLOAT: F
    }.get(value.Type)
    if fmt:
      self._Write(fmt, value.Value, byte_width)
    else:
      self._WriteOffset(value.Value, byte_width)

  def _WriteBlob(self, data, append_zero, type_):
    bit_width = BitWidth.U(len(data))
    byte_width = self._Align(bit_width)
    self._Write(U, len(data), byte_width)
    loc = len(self._buf)
    self._buf.extend(data)
    if append_zero:
      self._buf.append(0)
    self._stack.append(Value(loc, type_, bit_width))
    return loc

  def _WriteScalarVector(self, element_type, byte_width, elements, fixed):
    """Writes scalar vector elements to the underlying buffer."""
    bit_width = BitWidth.B(byte_width)
    # If you get this exception, you're trying to write a vector with a size
    # field that is bigger than the scalars you're trying to write (e.g. a
    # byte vector > 255 elements). For such types, write a "blob" instead.
    if BitWidth.U(len(elements)) > bit_width:
      raise ValueError('too many elements for the given byte_width')

    self._Align(bit_width)
    if not fixed:
      self._Write(U, len(elements), byte_width)

    loc = len(self._buf)

    fmt = {Type.INT: I, Type.UINT: U, Type.FLOAT: F}.get(element_type)
    if not fmt:
      raise TypeError('unsupported element_type')
    self._WriteVector(fmt, elements, byte_width)

    type_ = Type.ToTypedVector(element_type, len(elements) if fixed else 0)
    self._stack.append(Value(loc, type_, bit_width))
    return loc

  def _CreateVector(self, elements, typed, fixed, keys=None):
    """Writes vector elements to the underlying buffer."""
    length = len(elements)

    if fixed and not typed:
      raise ValueError('fixed vector must be typed')

    # Figure out smallest bit width we can store this vector with.
    bit_width = max(self._force_min_bit_width, BitWidth.U(length))
    prefix_elems = 1  # Vector size
    if keys:
      bit_width = max(bit_width, keys.ElemWidth(len(self._buf)))
      prefix_elems += 2  # Offset to the keys vector and its byte width.

    vector_type = Type.KEY
    # Check bit widths and types for all elements.
    for i, e in enumerate(elements):
      bit_width = max(bit_width, e.ElemWidth(len(self._buf), prefix_elems + i))

      if typed:
        if i == 0:
          vector_type = e.Type
        else:
          if vector_type != e.Type:
            raise RuntimeError('typed vector elements must be of the same type')

    if fixed and not Type.IsFixedTypedVectorElementType(vector_type):
      raise RuntimeError('must be fixed typed vector element type')

    byte_width = self._Align(bit_width)
    # Write vector. First the keys width/offset if available, and size.
    if keys:
      self._WriteOffset(keys.Value, byte_width)
      self._Write(U, 1 << keys.MinBitWidth, byte_width)

    if not fixed:
      self._Write(U, length, byte_width)

    # Then the actual data.
    loc = len(self._buf)
    for e in elements:
      self._WriteAny(e, byte_width)

    # Then the types.
    if not typed:
      for e in elements:
        self._buf.append(e.StoredPackedType(bit_width))

    if keys:
      type_ = Type.MAP
    else:
      if typed:
        type_ = Type.ToTypedVector(vector_type, length if fixed else 0)
      else:
        type_ = Type.VECTOR

    return Value(loc, type_, bit_width)

  def _PushIndirect(self, value, type_, bit_width):
    byte_width = self._Align(bit_width)
    loc = len(self._buf)
    fmt = {
        Type.INDIRECT_INT: I,
        Type.INDIRECT_UINT: U,
        Type.INDIRECT_FLOAT: F
    }[type_]
    self._Write(fmt, value, byte_width)
    self._stack.append(Value(loc, type_, bit_width))

  @InMapForString
  def String(self, value):
    """Encodes string value."""
    reset_to = len(self._buf)
    encoded = value.encode('utf-8')
    loc = self._WriteBlob(encoded, append_zero=True, type_=Type.STRING)
    if self._share_strings:
      prev_loc = self._string_pool.FindOrInsert(encoded, loc)
      if prev_loc is not None:
        del self._buf[reset_to:]
        self._stack[-1]._value = loc = prev_loc  # pylint: disable=protected-access

    return loc

  @InMap
  def Blob(self, value):
    """Encodes binary blob value.

    Args:
      value: A byte/bytearray value to encode

    Returns:
      Offset of the encoded value in underlying the byte buffer.
    """
    return self._WriteBlob(value, append_zero=False, type_=Type.BLOB)

  def Key(self, value):
    """Encodes key value.

    Args:
      value: A byte/bytearray/str value to encode. Byte object must not contain
        zero bytes. String object must be convertible to ASCII.

    Returns:
      Offset of the encoded value in the underlying byte buffer.
    """
    if isinstance(value, (bytes, bytearray)):
      encoded = value
    else:
      encoded = value.encode('ascii')

    if 0 in encoded:
      raise ValueError('key contains zero byte')

    loc = len(self._buf)
    self._buf.extend(encoded)
    self._buf.append(0)
    if self._share_keys:
      prev_loc = self._key_pool.FindOrInsert(encoded, loc)
      if prev_loc is not None:
        del self._buf[loc:]
        loc = prev_loc

    self._stack.append(Value.Key(loc))
    return loc

  def Null(self, key=None):
    """Encodes None value."""
    if key:
      self.Key(key)
    self._stack.append(Value.Null())

  @InMap
  def Bool(self, value):
    """Encodes boolean value.

    Args:
      value: A boolean value.
    """
    self._stack.append(Value.Bool(value))

  @InMap
  def Int(self, value, byte_width=0):
    """Encodes signed integer value.

    Args:
      value: A signed integer value.
      byte_width: Number of bytes to use: 1, 2, 4, or 8.
    """
    bit_width = BitWidth.I(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._stack.append(Value.Int(value, bit_width))

  @InMap
  def IndirectInt(self, value, byte_width=0):
    """Encodes signed integer value indirectly.

    Args:
      value: A signed integer value.
      byte_width: Number of bytes to use: 1, 2, 4, or 8.
    """
    bit_width = BitWidth.I(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._PushIndirect(value, Type.INDIRECT_INT, bit_width)

  @InMap
  def UInt(self, value, byte_width=0):
    """Encodes unsigned integer value.

    Args:
      value: An unsigned integer value.
      byte_width: Number of bytes to use: 1, 2, 4, or 8.
    """
    bit_width = BitWidth.U(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._stack.append(Value.UInt(value, bit_width))

  @InMap
  def IndirectUInt(self, value, byte_width=0):
    """Encodes unsigned integer value indirectly.

    Args:
      value: An unsigned integer value.
      byte_width: Number of bytes to use: 1, 2, 4, or 8.
    """
    bit_width = BitWidth.U(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._PushIndirect(value, Type.INDIRECT_UINT, bit_width)

  @InMap
  def Float(self, value, byte_width=0):
    """Encodes floating point value.

    Args:
      value: A floating point value.
      byte_width: Number of bytes to use: 4 or 8.
    """
    bit_width = BitWidth.F(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._stack.append(Value.Float(value, bit_width))

  @InMap
  def IndirectFloat(self, value, byte_width=0):
    """Encodes floating point value indirectly.

    Args:
      value: A floating point value.
      byte_width: Number of bytes to use: 4 or 8.
    """
    bit_width = BitWidth.F(value) if byte_width == 0 else BitWidth.B(byte_width)
    self._PushIndirect(value, Type.INDIRECT_FLOAT, bit_width)

  def _StartVector(self):
    """Starts vector construction."""
    return len(self._stack)

  def _EndVector(self, start, typed, fixed):
    """Finishes vector construction by encodung its elements."""
    vec = self._CreateVector(self._stack[start:], typed, fixed)
    del self._stack[start:]
    self._stack.append(vec)
    return vec.Value

  @contextlib.contextmanager
  def Vector(self, key=None):
    if key:
      self.Key(key)

    try:
      start = self._StartVector()
      yield self
    finally:
      self._EndVector(start, typed=False, fixed=False)

  @InMap
  def VectorFromElements(self, elements):
    """Encodes sequence of any elements as a vector.

    Args:
      elements: sequence of elements, they may have different types.
    """
    with self.Vector():
      for e in elements:
        self.Add(e)

  @contextlib.contextmanager
  def TypedVector(self, key=None):
    if key:
      self.Key(key)

    try:
      start = self._StartVector()
      yield self
    finally:
      self._EndVector(start, typed=True, fixed=False)

  @InMap
  def TypedVectorFromElements(self, elements, element_type=None):
    """Encodes sequence of elements of the same type as typed vector.

    Args:
      elements: Sequence of elements, they must be of the same type.
      element_type: Suggested element type. Setting it to None means determining
        correct value automatically based on the given elements.
    """
    if isinstance(elements, array.array):
      if elements.typecode == 'f':
        self._WriteScalarVector(Type.FLOAT, 4, elements, fixed=False)
      elif elements.typecode == 'd':
        self._WriteScalarVector(Type.FLOAT, 8, elements, fixed=False)
      elif elements.typecode in ('b', 'h', 'i', 'l', 'q'):
        self._WriteScalarVector(
            Type.INT, elements.itemsize, elements, fixed=False)
      elif elements.typecode in ('B', 'H', 'I', 'L', 'Q'):
        self._WriteScalarVector(
            Type.UINT, elements.itemsize, elements, fixed=False)
      else:
        raise ValueError('unsupported array typecode: %s' % elements.typecode)
    else:
      add = self.Add if element_type is None else self.Adder(element_type)
      with self.TypedVector():
        for e in elements:
          add(e)

  @InMap
  def FixedTypedVectorFromElements(self,
                                   elements,
                                   element_type=None,
                                   byte_width=0):
    """Encodes sequence of elements of the same type as fixed typed vector.

    Args:
      elements: Sequence of elements, they must be of the same type. Allowed
        types are `Type.INT`, `Type.UINT`, `Type.FLOAT`. Allowed number of
        elements are 2, 3, or 4.
      element_type: Suggested element type. Setting it to None means determining
        correct value automatically based on the given elements.
      byte_width: Number of bytes to use per element. For `Type.INT` and
        `Type.UINT`: 1, 2, 4, or 8. For `Type.FLOAT`: 4 or 8. Setting it to 0
        means determining correct value automatically based on the given
        elements.
    """
    if not 2 <= len(elements) <= 4:
      raise ValueError('only 2, 3, or 4 elements are supported')

    types = {type(e) for e in elements}
    if len(types) != 1:
      raise TypeError('all elements must be of the same type')

    type_, = types

    if element_type is None:
      element_type = {int: Type.INT, float: Type.FLOAT}.get(type_)
      if not element_type:
        raise TypeError('unsupported element_type: %s' % type_)

    if byte_width == 0:
      width = {
          Type.UINT: BitWidth.U,
          Type.INT: BitWidth.I,
          Type.FLOAT: BitWidth.F
      }[element_type]
      byte_width = 1 << max(width(e) for e in elements)

    self._WriteScalarVector(element_type, byte_width, elements, fixed=True)

  def _StartMap(self):
    """Starts map construction."""
    return len(self._stack)

  def _EndMap(self, start):
    """Finishes map construction by encodung its elements."""
    # Interleaved keys and values on the stack.
    stack = self._stack[start:]

    if len(stack) % 2 != 0:
      raise RuntimeError('must be even number of keys and values')

    for key in stack[::2]:
      if key.Type is not Type.KEY:
        raise RuntimeError('all map keys must be of %s type' % Type.KEY)

    pairs = zip(stack[::2], stack[1::2])  # [(key, value), ...]
    pairs = sorted(pairs, key=lambda pair: self._ReadKey(pair[0].Value))

    del self._stack[start:]
    for pair in pairs:
      self._stack.extend(pair)

    keys = self._CreateVector(self._stack[start::2], typed=True, fixed=False)
    values = self._CreateVector(
        self._stack[start + 1::2], typed=False, fixed=False, keys=keys)

    del self._stack[start:]
    self._stack.append(values)
    return values.Value

  @contextlib.contextmanager
  def Map(self, key=None):
    if key:
      self.Key(key)

    try:
      start = self._StartMap()
      yield self
    finally:
      self._EndMap(start)

  def MapFromElements(self, elements):
    start = self._StartMap()
    for k, v in elements.items():
      self.Key(k)
      self.Add(v)
    self._EndMap(start)

  def Adder(self, type_):
    return {
        Type.BOOL: self.Bool,
        Type.INT: self.Int,
        Type.INDIRECT_INT: self.IndirectInt,
        Type.UINT: self.UInt,
        Type.INDIRECT_UINT: self.IndirectUInt,
        Type.FLOAT: self.Float,
        Type.INDIRECT_FLOAT: self.IndirectFloat,
        Type.KEY: self.Key,
        Type.BLOB: self.Blob,
        Type.STRING: self.String,
    }[type_]

  @InMapForString
  def Add(self, value):
    """Encodes value of any supported type."""
    if value is None:
      self.Null()
    elif isinstance(value, bool):
      self.Bool(value)
    elif isinstance(value, int):
      self.Int(value)
    elif isinstance(value, float):
      self.Float(value)
    elif isinstance(value, str):
      self.String(value)
    elif isinstance(value, (bytes, bytearray)):
      self.Blob(value)
    elif isinstance(value, dict):
      with self.Map():
        for k, v in value.items():
          self.Key(k)
          self.Add(v)
    elif isinstance(value, array.array):
      self.TypedVectorFromElements(value)
    elif _IsIterable(value):
      self.VectorFromElements(value)
    else:
      raise TypeError('unsupported python type: %s' % type(value))

  @property
  def LastValue(self):
    return self._stack[-1]

  @InMap
  def ReuseValue(self, value):
    self._stack.append(value)


def GetRoot(buf):
  """Returns root `Ref` object for the given buffer."""
  if len(buf) < 3:
    raise ValueError('buffer is too small')
  byte_width = buf[-1]
  return Ref.PackedType(
      Buf(buf, -(2 + byte_width)), byte_width, packed_type=buf[-2])


def Dumps(obj):
  """Returns bytearray with the encoded python object."""
  fbb = Builder()
  fbb.Add(obj)
  return fbb.Finish()


def Loads(buf):
  """Returns python object decoded from the buffer."""
  return GetRoot(buf).Value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\matmul_nbits_quantizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import importlib
import logging
import os

import numpy as np
import numpy.typing as npt
import onnx
from onnx.onnx_pb import GraphProto, ModelProto, NodeProto, TensorProto
from packaging import version

from onnxruntime.capi._pybind_state import quantize_matmul_4bits, quantize_matmul_8bits, quantize_qdq_matmul_4bits

from .calibrate import CalibrationDataReader
from .onnx_model import ONNXModel
from .quant_utils import QuantFormat, attribute_to_kwarg

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightOnlyQuantConfig:
    def __init__(
        self,
        algorithm: str,
        quant_format: QuantFormat,
        op_types_to_quantize: tuple[str, ...] | None = None,
        quant_axes: tuple[tuple[str, int], ...] | None = None,
        customized_weight_config: dict | None = None,
    ):
        """This is the Base class for Weight Only blockwise quantization Configuration.

        Args:
            algorithm:
                weight only quantize algorithm name.
            quant_format: QuantFormat{QOperator, QDQ}.
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
            op_types_to_quantize (optional):
                set of operator types to quantize. Default {MatMul}
            quant_axes (dict[str, int], optional):
                op:axis, which axis to quantize for an op. Default {MatMul: 0, Gather: 1}
            customized_weight_config:
                customized weight config for nodes if needed. It is dictionary with node name as key,
                and the value is a dict of customized config.
        """
        self.algorithm = algorithm
        self.quant_format = quant_format
        self.op_types_to_quantize = set(op_types_to_quantize) if op_types_to_quantize else {"MatMul"}
        self.quant_axes = dict(quant_axes) if quant_axes else {"MatMul": 0, "Gather": 1}
        self.customized_weight_config = customized_weight_config


class RTNWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        ratios=None,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize: tuple[str, ...] | None = None,
        customized_weight_config: dict | None = None,
    ):
        """
        This is a class for round-to-nearest (RTN) algorithm Weight Only Quant Configuration.
        RTN is the most straightforward way to quantize weight using scale maps.

        Args:
            ratios:
                percentile of clip. Defaults to {}.
            quant_format (QuantFormat{QOperator, QDQ}, optional):
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
                Defaults to QuantFormat.QOperator.
            op_types_to_quantize (optional):
                set of operator types to quantize.
            customized_weight_config:
                customized weight config for nodes if needed. It is dictionary with node name as key,
                and the value is a dict of customized config.
        """
        assert quant_format == QuantFormat.QOperator, "RTN only supports QOperator format"

        if ratios is None:
            ratios = {}
        super().__init__(
            algorithm="RTN",
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
            customized_weight_config=customized_weight_config,
        )
        self.ratios = ratios


class GPTQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        calibration_data_reader: CalibrationDataReader | None = None,
        percdamp=0.01,
        block_size=128,
        actorder=False,
        mse=False,
        perchannel=True,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize: tuple[str, ...] | None = None,
    ):
        """
        This is a class for GPTQ algorithm Weight Only Quant Configuration.
        GPTQ algorithm provides more accurate quantization but requires more computational resources.

        Args:
            calibration_data_reader:
                a calibration data reader. It enumerates calibration data and generates inputs for the original model.
            percdamp:
                percent of the average Hessian diagonal to use for dampening.
            block_size (int, optional):
                channel number in one block to execute a GPTQ quantization iteration.
            actorder (bool, optional):
                whether rearrange Hessian matrix considering the diag's value.
            mse (bool, optional):
                whether get scale and zero point with mse error.
            perchannel (bool, optional):
                whether quantize weight per-channel.
            quant_format (QuantFormat{QOperator, QDQ}, optional):
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
                Defaults to QuantFormat.QOperator.
            op_types_to_quantize (optional):
                set of operator types to quantize.
        """
        assert quant_format == QuantFormat.QOperator, "GPTQ only supports QOperator format"

        super().__init__(
            algorithm="GPTQ",
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
        )
        self.calibration_data_reader = calibration_data_reader
        self.percdamp = percdamp
        self.block_size = block_size
        self.actorder = actorder
        self.mse = mse
        self.perchannel = perchannel


class HQQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        block_size=128,
        bits=4,
        axis=1,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize: tuple[str, ...] | None = None,
        quant_axes: tuple[tuple[str, int], ...] | None = None,
    ):
        """
        This is a class for HQQ algorithm Weight Only Quant Configuration.
        HQQ algorithm quant weight without needing calibrate data.

        Args:
            block_size (int, optional):
                channel number in one block to execute a HQQ quantization iteration.
            bits (int, optional):
                how many bits to represent weight.
            axis (int, optional):
                0 or 1. which axis to quantize. https://arxiv.org/pdf/2309.15531.pdf
            quant_format (QuantFormat{QOperator, QDQ}, optional):
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
                Defaults to QuantFormat.QOperator.
            op_types_to_quantize (optional):
                set of operator types to quantize.
            quant_axes (dict[str, int], optional):
                op:axis, which axis to quantize for an op. Default {MatMul: 0, Gather: 1}
        """
        assert quant_format == QuantFormat.QOperator, "HQQ only supports QOperator format"

        super().__init__(
            algorithm="HQQ",
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
        )
        self.block_size = block_size
        self.bits = bits
        self.axis = axis


class DefaultWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        block_size: int = 128,
        is_symmetric: bool = False,
        accuracy_level: int | None = None,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize: tuple[str, ...] | None = None,
        quant_axes: tuple[tuple[str, int], ...] | None = None,
        bits: int = 4,
    ):
        """
        This is a class for weight only affine quantization configuration.

        Args:
            block_size (int, optional):
                channel number in one block to execute an affine quantization iteration.
            is_symmetric (bool, optional):
                whether quantize weight symmetrically.
            accuracy_level (int, optional):
                Accuracy level of the 4-bit quantized MatMul computation.
                Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details.
                (https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits)
            quant_format (QuantFormat{QOperator, QDQ}, optional):
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
                Defaults to QuantFormat.QOperator.
            op_types_to_quantize (optional):
                set of operator types to quantize.
            quant_axes (dict[str, int], optional):
                op:axis, which axis to quantize for an op. Default {MatMul: 0, Gather: 1}
            bits (int, optional):
                number of bits per element after quantization. Default 4.
        """
        super().__init__(
            algorithm="DEFAULT",
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
        )
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.bits = bits
        self.accuracy_level = accuracy_level


class NVAWQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        tokenizer_dir,
        dataset_name="cnn",
        cache_dir="./cache",
        calibration_method="awq_lite",
    ):
        """
        Configuration for the nvidia_awq quantization method.

        Args:
            tokenizer_dir (str): pathof the tokenizer dir.
            dataset_name (str): Name of the dataset.
            cache_dir (str): Directory for caching.
            calibration_method (str): calib method for nvidia_awq.
        """
        # Import torch and DataLoader
        try:
            import torch
            from torch.utils.data import DataLoader

            self.torch = torch
            self.DataLoader = DataLoader
        except ImportError:
            print(
                "Error: The 'torch' library is required but not installed. Please install it using 'pip install torch'."
            )
            raise ImportError("torch is not installed. Exiting.") from None

        # Import datasets
        try:
            from datasets import load_dataset

            self.load_dataset = load_dataset
        except ImportError:
            print(
                "Error: The 'datasets' library is required but not installed. Please install it using 'pip install datasets'."
            )
            raise ImportError("datasets is not installed. Exiting.") from None

        # Import transformers
        try:
            from transformers import AutoConfig, AutoTokenizer

            self.AutoConfig = AutoConfig
            self.AutoTokenizer = AutoTokenizer
        except ImportError:
            print(
                "Error: The 'transformers' library is required but not installed. Please install it using 'pip install transformers'."
            )
            raise ImportError("transformers is not installed. Exiting.") from None

        super().__init__(
            algorithm="nvidia_awq",
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=None,  # Assuming op_types_to_quantize is handled elsewhere
            quant_axes=None,  # Assuming quant_axes is handled elsewhere
        )

        # Determine the device
        device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

        calib_inputs = self.get_calib_inputs(
            dataset_name=dataset_name,
            model_name=tokenizer_dir,
            cache_dir=cache_dir,
            calib_size=32,
            batch_size=1,
            block_size=512,
            device=device,
            use_fp16=True,
            use_buffer_share=False,
            add_past_kv_inputs=True,
            max_calib_rows_to_load=128,
            add_position_ids=True,
        )

        self.calibration_data_reader = calib_inputs
        self.calibration_method = calibration_method

    def make_model_input(
        self,
        config,
        input_ids_arg,
        attention_mask_arg,
        add_past_kv_inputs,
        device,
        use_fp16,
        use_buffer_share,
        add_position_ids,
    ):
        # Access torch from the instance variable
        torch = self.torch

        input_ids = input_ids_arg
        attention_mask = attention_mask_arg

        if isinstance(input_ids_arg, list):
            input_ids = torch.tensor(input_ids_arg, device=device, dtype=torch.int64)
            attention_mask = torch.tensor(attention_mask_arg, device=device, dtype=torch.int64)

        inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
        }

        if add_position_ids:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            inputs["position_ids"] = position_ids.contiguous()

        if add_past_kv_inputs:
            torch_dtype = torch.float16 if use_fp16 else torch.float32
            batch_size, sequence_length = input_ids.shape
            max_sequence_length = config.max_position_embeddings
            num_heads, head_size = (
                config.num_key_value_heads,
                config.hidden_size // config.num_attention_heads,
            )
            for i in range(config.num_hidden_layers):
                past_key = torch.zeros(
                    batch_size,
                    num_heads,
                    max_sequence_length if use_buffer_share else 0,
                    head_size,
                    device=device,
                    dtype=torch_dtype,
                )
                past_value = torch.zeros(
                    batch_size,
                    num_heads,
                    max_sequence_length if use_buffer_share else 0,
                    head_size,
                    device=device,
                    dtype=torch_dtype,
                )
                inputs.update(
                    {
                        f"past_key_values.{i}.key": past_key.contiguous(),
                        f"past_key_values.{i}.value": past_value.contiguous(),
                    }
                )

        return inputs

    def get_calib_inputs(
        self,
        dataset_name,
        model_name,
        cache_dir,
        calib_size,
        batch_size,
        block_size,
        device,
        use_fp16,
        use_buffer_share,
        add_past_kv_inputs,
        max_calib_rows_to_load,
        add_position_ids,
    ):
        # Access transformers and datasets from the instance variables
        auto_config = self.AutoConfig
        auto_tokenizer = self.AutoTokenizer
        load_dataset = self.load_dataset

        config = auto_config.from_pretrained(
            model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer = auto_tokenizer.from_pretrained(
            model_name, use_auth_token=True, cache_dir=cache_dir, trust_remote_code=True
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token = tokenizer.eos_token

        assert calib_size <= max_calib_rows_to_load, "calib size should be no more than max_calib_rows_to_load"

        if "cnn" in dataset_name:
            dataset2 = load_dataset("cnn_dailymail", name="3.0.0", split="train").select(range(max_calib_rows_to_load))
            column = "article"
        elif "pile" in dataset_name:
            dataset2 = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            column = "text"
        else:
            raise ValueError(f'dataset "{dataset_name}" not supported')

        dataset2 = dataset2[column][:calib_size]
        batch_encoded = tokenizer.batch_encode_plus(
            dataset2, return_tensors="pt", padding=True, truncation=True, max_length=block_size
        )
        batch_encoded = batch_encoded.to(device)
        batch_encoded_input_ids = batch_encoded["input_ids"]
        batch_encoded_attention_mask = batch_encoded["attention_mask"]

        # Access DataLoader from the instance variable
        data_loader = self.DataLoader

        calib_dataloader_input_ids = data_loader(batch_encoded_input_ids, batch_size=batch_size, shuffle=False)
        calib_dataloader_attention_mask = data_loader(
            batch_encoded_attention_mask, batch_size=batch_size, shuffle=False
        )

        assert len(calib_dataloader_input_ids.dataset) == len(calib_dataloader_attention_mask.dataset)
        assert len(calib_dataloader_input_ids) == len(calib_dataloader_attention_mask)

        number_of_batched_samples = calib_size // batch_size

        batched_input_ids = []
        for idx, data in enumerate(calib_dataloader_input_ids):
            batched_input_ids.append(data)
            if idx == (number_of_batched_samples - 1):
                break

        batched_attention_mask = []
        for idx, data in enumerate(calib_dataloader_attention_mask):
            batched_attention_mask.append(data)
            if idx == (number_of_batched_samples - 1):
                break

        print(
            f"\n--Quantize-Script-- number_of_batched_samples={number_of_batched_samples}, "
            f"batch-input-ids-list-len={len(batched_input_ids)}, batched_attention_mask={len(batched_attention_mask)}\n"
        )

        batched_inputs_list = []
        for i in range(number_of_batched_samples):
            input_ids = batched_input_ids[i]
            attention_mask = batched_attention_mask[i]

            inputs = self.make_model_input(
                config,
                input_ids,
                attention_mask,
                add_past_kv_inputs,
                device,
                use_fp16,
                use_buffer_share,
                add_position_ids,
            )
            inputs = {input_name: torch_tensor.cpu().numpy() for input_name, torch_tensor in inputs.items()}
            batched_inputs_list.append(inputs)

        print(f"\n--Quantize-Script-- number of batched inputs = {len(batched_inputs_list)}\n")
        return batched_inputs_list


def is_divisible(val1, val2):
    return int(val2 * np.ceil(val1 / val2)) == val1


class HQQWeightOnlyQuantizer:
    def __init__(
        self,
        config: HQQWeightOnlyQuantConfig,
    ):
        self.config = config

    # Proximal solver || weight - dequantize(quantize(weight))||_p^p
    @staticmethod
    def optimize_weights(
        tensor,
        scale,
        zero,
        min_max: list[int],
        axis: int = 0,
        opt_params: dict | None = None,
        verbose=False,
    ):
        import torch

        opt_params = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20} if opt_params is None else opt_params
        lp_norm, beta, kappa, iters = (
            opt_params["lp_norm"],
            opt_params["beta"],
            opt_params["kappa"],
            opt_params["iters"],
        )

        dtype = torch.float16 if tensor.is_cuda else torch.float32
        w_f = tensor.to(dtype)
        scale = scale.to(dtype)
        zero = zero.to(dtype)

        def shrink_op(x, beta, p=lp_norm):
            if p == 1:
                return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
            else:
                return torch.sign(x) * torch.nn.functional.relu(
                    torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x) + 1e-8, p - 1)
                )

        best_error = 1e4
        for i in range(iters):
            w_q = torch.round(w_f * scale + zero).clamp(min_max[0], min_max[1])
            w_r = (w_q - zero) / scale
            w_e = shrink_op(w_f - w_r, beta)
            zero = torch.mean(w_q - (w_f - w_e) * scale, axis=axis, keepdim=True)
            beta *= kappa

            current_error = float(torch.abs(w_f - w_r).mean())
            if verbose:
                print(i, np.round(current_error, 6))
            if current_error < best_error:
                best_error = current_error
            else:
                break

        del w_f, w_q, w_r, w_e

        return scale, zero

    @staticmethod
    def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
        if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
            ori_int_tensor = ori_int_tensor.T
            pack_tensor = pack_tensor.T
        if bits in [2, 4, 8]:
            compress_ratio = pack_tensor.element_size() * 8 // bits
            for j in range(compress_ratio):
                pack_tensor[0:] |= ori_int_tensor[j::compress_ratio] << (bits * (j))
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

    # from Official implementation of Half-Quadratic Quantization (HQQ)
    def quantize_internal(
        self, tensor, bits=4, channel_wise=True, group_size=64, optimize=True, round_zero=True, axis=1
    ):
        import torch

        weight = tensor.float()
        ori_shape = weight.shape

        pad_len = (group_size - ori_shape[axis] % group_size) % group_size
        if axis == 1:
            weight = torch.nn.functional.pad(weight, (0, pad_len), "constant", 0)
        else:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_len), "constant", 0)
        shape = weight.shape

        # Reshape for grouping
        if (group_size is not None) and channel_wise:
            weight = weight.reshape([-1, group_size]) if (axis == 1) else weight.reshape([group_size, -1])

        # Get min/max values
        if channel_wise is False:
            _min, _max = weight.min(), weight.max()
            optimize = False
        else:
            _min = weight.min(axis=axis, keepdim=True)[0]
            _max = weight.max(axis=axis, keepdim=True)[0]

        max_v = 2**bits - 1
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via weight*scale + zero, the scale is inverted later on.
        # clamp to avoid half-precision problems
        scale = (max_v / (_max - _min)).clamp(max=2e4)
        #!!!!!!!!!!!!!!!
        min_max_axis = _max - _min
        if (min_max_axis == 0).sum().item() > 0:
            min_max_axis[min_max_axis == 0] = max_v
            scale = (max_v / min_max_axis).clamp(max=2e4)
        zero = -_min * scale

        if round_zero:
            zero = torch.round(zero)

        # Fine-tune weights
        if optimize:
            scale, zero = self.optimize_weights(tensor=weight, scale=scale, zero=zero, min_max=min_max, axis=axis)

        # Quantize
        # Necessary for fake quantization backprop
        w_q = torch.round(weight * scale + zero).clamp(min_max[0], min_max[1])
        w_q = w_q.reshape(shape).int()

        scale = 1.0 / scale
        if axis == 1:
            scale = scale.reshape(shape[0], -1)
            zero = zero.reshape(shape[0], -1)
        else:
            scale = scale.reshape(-1, shape[-1])
            zero = zero.reshape(-1, shape[-1])
        # cleanup
        del weight, _min, _max

        return w_q, scale.to(tensor.dtype), zero.to(tensor.dtype)

    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """
        Target node:        QOperator node:            QDQ nodes:
        MatMul              MatMulNBits                DeQuantizeLinear -> MatMul
        Gather              GatherBlockQuantized       Gather, Gather, Gather (optional) -> DequantizeLinear
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        If QOperator format, return the corresponding QOperator nodes.
        If QDQ format, return the corresdponging QDQ nodes.
        Gather (quantized data) + Gather (scales) + Gather (optional, zero points) -> DequantizeLinear is
        not supported yet because Gather does not support int4 data.
        """
        # With HQQ, zero points are in float. Current GatherBlockQuantized does not support float zero points.
        if node.op_type == "Gather":
            raise NotImplementedError("Gather quantization is not supported yet in HQQ")

        import torch

        logger.info(f"start to quantize {node.name} ...")
        input_b = node.input[1]
        b_pb, bs_graph = get_initializer(input_b, graph_stack)
        if b_pb is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return [node]  # only care about constant weight

        b_array = onnx.numpy_helper.to_array(b_pb)
        if len(b_array.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return [node]  # can only process 2-D matrix
        b_array_torch = torch.from_numpy(b_array)
        if torch.cuda.is_available():
            b_array_torch = b_array_torch.cuda()

        bits = self.config.bits
        quant_weight_torch, scales_torch, zero_points_torch = self.quantize_internal(
            b_array_torch.T, bits=bits, group_size=self.config.block_size
        )
        quant_weight_torch = quant_weight_torch.contiguous()
        scales_torch = scales_torch.contiguous()
        zero_points_torch = zero_points_torch.contiguous()

        packed_size = 8 // bits  # number of elements packed into one byte

        packed_torch = torch.zeros(
            (quant_weight_torch.shape[0], quant_weight_torch.shape[1] // packed_size),
            dtype=torch.uint8,
            device=quant_weight_torch.device,
        )
        self.pack_on_row_fast_248bit(packed_torch, quant_weight_torch, bits)
        scales = scales_torch.cpu().numpy()
        zero_points = zero_points_torch.cpu().numpy()
        # reshape to the predefined shape in MatmulNbits
        scales = scales.reshape(-1)
        zero_points = zero_points.reshape(-1)
        rows, cols = b_array_torch.shape
        block_size = self.config.block_size
        blob_size = block_size // packed_size
        k_blocks = (rows + block_size - 1) // block_size
        packed_torch = packed_torch.reshape(cols, k_blocks, blob_size)

        b_quant = onnx.numpy_helper.from_array(packed_torch.cpu().numpy())
        b_quant.name = b_pb.name + "_Q" + str(bits)
        for input in bs_graph.input:
            if input.name == input_b:
                bs_graph.input.remove(input)
                break

        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = b_pb.name + "_scales"
        bs_graph.initializer.extend([b_quant, scales_tensor])

        input_names = [node.input[0], b_quant.name, scales_tensor.name]
        zp_tensor = onnx.numpy_helper.from_array(zero_points)
        zp_tensor.name = b_pb.name + "_zero_points"
        bs_graph.initializer.extend([zp_tensor])
        input_names.append(zp_tensor.name)

        kwargs = {}
        rows, cols = b_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = bits
        kwargs["block_size"] = self.config.block_size

        matmul_q_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q" + str(bits) if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )

        logger.info(f"complete quantization of {node.name} ...")

        return [matmul_q_node]


def get_initializer(name, graph_path: list[GraphProto]) -> tuple[TensorProto, GraphProto]:
    for gid in range(len(graph_path) - 1, -1, -1):
        graph = graph_path[gid]
        for tensor in graph.initializer:
            if tensor.name == name:
                return tensor, graph
    return None, None


class DefaultWeightOnlyQuantizer:
    def __init__(self, config: DefaultWeightOnlyQuantConfig):
        self.config = config

    def qbits_block_quant(self, fp32weight: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4b/8b quantize fp32 weight to int4 using C++ kernels."""

        qbits = self.config.bits
        kpack = 8 // qbits
        if len(fp32weight.shape) != 2:
            raise ValueError("Current int4 block quantization only supports 2D tensors!")
        rows, cols = fp32weight.shape

        block_size = self.config.block_size
        k_blocks = (rows + block_size - 1) // block_size

        if self.config.quant_format == QuantFormat.QOperator:
            blob_size = (block_size + kpack - 1) // kpack
            padded_rows = k_blocks * block_size
            pad_len = padded_rows - rows
            if pad_len > 0:
                fp32weight = np.pad(fp32weight, ((0, pad_len), (0, 0)), "constant")

            # block wise quantization, each block comes from a single column
            packed = np.zeros((cols, k_blocks, blob_size), dtype="uint8")
            zero_point = np.zeros(cols * ((k_blocks + kpack - 1) // kpack), dtype="uint8")
            scales = np.zeros((cols * k_blocks), dtype=fp32weight.dtype)
            if qbits == 8:
                quantize_matmul_8bits(
                    packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric
                )
            else:
                quantize_matmul_4bits(
                    packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric
                )
        else:
            assert qbits == 4, "QDQ format only support 4 bits quantization"
            packed = np.zeros((rows * cols + 1) // 2, dtype="uint8")
            zero_point = np.zeros((cols * k_blocks + 1) // 2, dtype="uint8")
            scales = np.zeros((k_blocks, cols), dtype=fp32weight.dtype)
            quantize_qdq_matmul_4bits(
                packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric
            )

        return (packed, scales, zero_point)

    def quantize_matmul(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """
        Quantize weight B of MatMul node to int4 or int8.
        Currently only support 2D constant matrix and axis 0 blockwise quantization.
        """
        bits = self.config.bits
        if bits == 8:
            qtype = TensorProto.INT8 if self.config.is_symmetric else TensorProto.UINT8
        else:
            qtype = TensorProto.INT4 if self.config.is_symmetric else TensorProto.UINT4
        input_b = node.input[1]
        b_tensor, b_graph = get_initializer(input_b, graph_stack)
        if b_tensor is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return [node]  # only care about constant weight

        b_ndarray = onnx.numpy_helper.to_array(b_tensor)
        if len(b_ndarray.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return [node]  # can only process 2-D matrix

        packed, scales, zero_points = self.qbits_block_quant(b_ndarray)

        if self.config.quant_format == QuantFormat.QOperator:
            b_quant = onnx.numpy_helper.from_array(packed, b_tensor.name + f"_Q{bits}")
            scales_tensor = onnx.numpy_helper.from_array(scales, b_tensor.name + "_scales")
        else:
            b_quant = onnx.helper.make_tensor(
                b_tensor.name + f"_DQ_Q{bits}", qtype, b_ndarray.shape, packed.tobytes(), True
            )
            scales_tensor = onnx.numpy_helper.from_array(scales, b_tensor.name + "_DQ_scales")

        for input in b_graph.input:
            if input.name == input_b:
                b_graph.input.remove(input)
                break

        b_graph.initializer.extend([b_quant, scales_tensor])

        output_nodes = []

        if self.config.quant_format == QuantFormat.QOperator:
            input_names = [node.input[0], b_quant.name, scales_tensor.name]
            if not self.config.is_symmetric:
                zp_tensor = onnx.numpy_helper.from_array(zero_points, b_tensor.name + "_zero_points")
                input_names.append(zp_tensor.name)
                b_graph.initializer.extend([zp_tensor])
            kwargs = {}
            rows, cols = b_ndarray.shape
            kwargs["K"] = rows
            kwargs["N"] = cols
            kwargs["bits"] = bits
            kwargs["block_size"] = self.config.block_size
            if self.config.accuracy_level is not None:
                kwargs["accuracy_level"] = self.config.accuracy_level

            matmul_qbit_node = onnx.helper.make_node(
                "MatMulNBits",
                inputs=input_names,
                outputs=[node.output[0]],
                name=node.name + f"_Q{bits}" if node.name else "",
                domain="com.microsoft",
                **kwargs,
            )

            output_nodes.append(matmul_qbit_node)
        else:
            dq_input_names = [b_quant.name, scales_tensor.name]
            dq_output_names = [b_quant.name + "_output"]
            matmul_input_names = [node.input[0], dq_output_names[0]]
            matmul_output_names = [node.output[0]]
            if not self.config.is_symmetric:
                zp_tensor = onnx.helper.make_tensor(
                    b_tensor.name + "_DQ_zero_points", qtype, scales.shape, zero_points.tobytes(), True
                )
                dq_input_names.append(zp_tensor.name)
                b_graph.initializer.extend([zp_tensor])
            dq_kwargs = {"axis": 0, "block_size": self.config.block_size}
            dq_node = onnx.helper.make_node(
                "DequantizeLinear",
                inputs=dq_input_names,
                outputs=dq_output_names,
                name=node.name + f"_DQ_Q{bits}" if node.name else "",
                **dq_kwargs,
            )
            matmul_node = onnx.helper.make_node(
                "MatMul",
                inputs=matmul_input_names,
                outputs=matmul_output_names,
                name=node.name + f"_matmul_Q{bits}" if node.name else "",
            )
            output_nodes.extend([dq_node, matmul_node])

        return output_nodes

    @staticmethod
    def quant_slice_symmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        abs_max = np.where(np.abs(max_val) > np.abs(min_val), max_val, min_val)

        scale = abs_max / -8.0  # if max == min, max may be clipped
        quantized_slice = np.where(scale == 0, 0, data / scale).round().clip(-8, 7).astype(np.int8)

        return quantized_slice, scale

    @staticmethod
    def quant_slice_asymmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_val = np.minimum(data.min(axis=1, keepdims=True), 0)
        max_val = np.maximum(data.max(axis=1, keepdims=True), 0)

        scale = (max_val - min_val) / 15.0
        zero_point = np.where(scale == 0, 8, -min_val / scale).round().clip(0, 15).astype(np.uint8)
        quantized_slice = np.where(scale == 0, 8, data / scale + zero_point).round().clip(0, 15).astype(np.uint8)

        return quantized_slice, scale, zero_point

    @staticmethod
    def pack_int8_to_int4(data: np.ndarray) -> np.ndarray:
        """Pack int8 data to int4 and store in uint8 ndarray."""
        data_flat = data.reshape(-1)
        if len(data_flat) % 2 != 0:
            data_flat = np.append(data_flat, 0)
        quant_data_int4 = (data_flat[::2] & 0xF) | ((data_flat[1::2] & 0xF) << 4)

        return quant_data_int4.astype("uint8")

    @staticmethod
    def quantize_ndarray(
        data: np.ndarray,
        quantize_axis: int,
        block_size: int,
        is_symmetric: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Quantize ndarray data to int4 using numpy, return (quantized data, scales, zero points)."""
        # Get the shape of the matrix
        m = 1  # dimension of the matrix before the quantize axis
        k = data.shape[quantize_axis]  # dimension of the matrix along the quantize axis
        n = 1  # dimension of the matrix after the quantize axis
        for i, dim in enumerate(data.shape):
            if i < quantize_axis:
                m *= dim
            elif i > quantize_axis:
                n *= dim

        k_blocks = (k + block_size - 1) // block_size
        scales_shape = list(data.shape)
        scales_shape[quantize_axis] = k_blocks

        data_reshape = data.reshape((m, k, n))
        scales = np.zeros((m, k_blocks, n), dtype=data.dtype)
        if is_symmetric:
            quant_data_int8 = np.zeros((m, k, n), dtype="int8")
        else:
            quant_data_int8 = np.zeros((m, k, n), dtype="uint8")
            zero_point_int8 = np.zeros((m, k_blocks, n), dtype="uint8")

        # slice and quantize
        for i in range(0, k, block_size):
            end_idx = min(i + block_size, k)
            slice = data_reshape[:, i:end_idx, :]

            if is_symmetric:
                quantized_slice_int8, scale_slice = DefaultWeightOnlyQuantizer.quant_slice_symmetric(slice)
            else:
                quantized_slice_int8, scale_slice, zero_point_slice_int8 = (
                    DefaultWeightOnlyQuantizer.quant_slice_asymmetric(slice)
                )

            quant_data_int8[:, i:end_idx, :] = quantized_slice_int8
            j = i // block_size
            scales[:, j : (j + 1), :] = scale_slice
            if not is_symmetric:
                zero_point_int8[:, j : (j + 1), :] = zero_point_slice_int8

        # pack int8 to int4
        quant_data_int4 = DefaultWeightOnlyQuantizer.pack_int8_to_int4(quant_data_int8)
        zero_point_int4 = None
        if not is_symmetric:
            zero_point_int4 = DefaultWeightOnlyQuantizer.pack_int8_to_int4(zero_point_int8)
        scales = scales.reshape(scales_shape)
        return quant_data_int4, scales, zero_point_int4

    def quantize_gather(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """Quantize weight data of Gather node to int4."""
        assert self.config.quant_format == QuantFormat.QOperator, "Gather only supports QOperator format currently."

        qtype = TensorProto.INT4 if self.config.is_symmetric else TensorProto.UINT4
        data_arg = node.input[0]
        data_tensorproto, data_graphproto = get_initializer(data_arg, graph_stack)
        if data_tensorproto is None:
            logger.info("Gather doesn't have const weight. Skip quantization.")
            return [node]  # only care about constant weight

        data_ndarray = onnx.numpy_helper.to_array(data_tensorproto)
        data_rank = len(data_ndarray.shape)
        quantize_axis = self.config.quant_axes.get("Gather", 1)
        block_size = self.config.block_size

        assert quantize_axis < data_rank and quantize_axis >= -data_rank, "Invalid quantize axis for Gather node."
        assert block_size >= 16 and ((block_size - 1) & block_size == 0), "Invalid block size for Gather node."

        quantize_axis = (quantize_axis + data_rank) % data_rank
        quantized_data, scales, zero_points = self.quantize_ndarray(
            data_ndarray, quantize_axis, block_size, self.config.is_symmetric
        )

        for input in data_graphproto.input:
            if input.name == data_arg:
                data_graphproto.input.remove(input)
                break

        quantized_data_tensorproto = onnx.helper.make_tensor(
            data_tensorproto.name + "_Q4", qtype, data_ndarray.shape, quantized_data.tobytes(), True
        )
        scales_tensorproto = onnx.numpy_helper.from_array(scales, data_tensorproto.name + "_scales")
        input_names = [quantized_data_tensorproto.name, node.input[1], scales_tensorproto.name]
        data_graphproto.initializer.extend([quantized_data_tensorproto, scales_tensorproto])
        if not self.config.is_symmetric:
            zp_tensorproto = onnx.helper.make_tensor(
                data_tensorproto.name + "_zero_points", qtype, scales.shape, zero_points.tobytes(), True
            )
            input_names.append(zp_tensorproto.name)
            data_graphproto.initializer.extend([zp_tensorproto])

        try:
            gather_axis = onnx.helper.get_node_attr_value(node, "axis")
        except ValueError:
            gather_axis = 0

        kwargs = {
            "gather_axis": gather_axis,
            "quantize_axis": quantize_axis,
            "block_size": block_size,
        }

        gather_q4_node = onnx.helper.make_node(
            "GatherBlockQuantized",
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )

        return [gather_q4_node]

    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """
        Target node:        QOperator node:            QDQ nodes:
        MatMul              MatMulNBits                DeQuantizeLinear -> MatMul
        Gather              GatherBlockQuantized       Gather, Gather, Gather (optional) -> DequantizeLinear
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        If QOperator format, return the corresponding QOperator nodes.
        If QDQ format, return the corresdponging QDQ nodes.
        Gather (quantized data) + Gather (scales) + Gather (optional, zero points) -> DequantizeLinear is
        not supported yet because Gather does not support int4 data.
        """
        logger.info(f"start to quantize {node.name} ...")

        bits = self.config.bits
        if node.op_type == "MatMul":
            if bits == 8 and self.config.quant_format == QuantFormat.QDQ:
                logger.error("MatMul only supports QOperator format for 8 bits quantization.")
                return [node]
            results = self.quantize_matmul(node, graph_stack)
        elif node.op_type == "Gather":
            if self.config.bits != 4:
                logger.error("Gather only supports 4 bits quantization.")
                return [node]

            results = self.quantize_gather(node, graph_stack)
        else:
            logger.error(f"Unsupported operator {node.op_type} for weight only quantization. Skip quantization.")
            return [node]

        logger.info(f"complete quantization of {node.name} with {self.config.bits} bits ...")
        return results


class NVAWQWeightOnlyQuantizer:
    def __init__(
        self,
        config: NVAWQWeightOnlyQuantConfig,
    ):
        self.config = config

    def quantize_awq(self, model: ModelProto | str) -> ModelProto:
        """
        Perform nvidia_awq quantization using ModelOpt's int4 quantize function.

        Args:
            model (ModelProto): The ONNX model to quantize.

        Returns:
            ModelProto: The quantized ONNX model.
        """
        try:
            from modelopt.onnx.quantization.int4 import quantize as quantize_int4
        except ImportError:
            print(
                "Please ensure that the 'modelopt' package is installed. Please install it using pip install nvidia_modelopt."
            )
            raise ImportError(
                "modelopt is not installed. Please install it using pip install nvidia_modelopt. Exiting."
            ) from None

        logger.info("Starting nvidia_awq quantization...")

        # Prepare calibration inputs
        calib_inputs = self.config.calibration_data_reader

        # Perform quantization using ModelOpt's int4 quantize function
        quantized_model = quantize_int4(
            model,
            calibration_method=self.config.calibration_method,
            calibration_data_reader=calib_inputs,
        )

        logger.info("Completed nvidia_awq quantization.")
        return quantized_model


class MatMulNBitsQuantizer:
    """
    Target node:        QOperator node:            QDQ nodes:
    MatMul              MatMulNBits                DeQuantizeLinear -> MatMul
    Gather              GatherBlockQuantized       Gather, Gather, Gather (optional) -> DequantizeLinear

    Perform 4/8 bits quantization of constant weights for target nodes.
    If algo_config.quant_format is QOperator:
      - nodes are replaced by the corresponding QOperator nodes.
      - quantized weights are stored in the contrib ops.
    If algo_config.quant_format is QDQ:
      - the quantized weight is stored in a standard onnx node. For MatMul, it is DequantizeLinear. For Gather,
        it is the three Gathers, one for quantized data, one for scales and one for optional zero points.
      - The nodes are replaced by the corresponding QDQ nodes.
      - currently Gather is not supported in QDQ because Gather does not support int4 yet.
    Note:
      - for quantized gather, the memory usage of "DequantizeLinear + Gather" is the same as the original Gather
        during runtime. Therefor it is not recommended.
      - when a node is in nodes_to_exclude, and the node configuration in algo_config.customized_weight_config will be ignored.
    """

    def __init__(
        self,
        model: ModelProto | str,
        block_size: int = 128,
        is_symmetric: bool = False,
        accuracy_level: int | None = None,
        nodes_to_exclude=None,
        nodes_to_include: list[str] | None = None,
        quant_format=QuantFormat.QOperator,
        op_types_to_quantize: tuple[str, ...] | None = None,
        quant_axes: tuple[tuple[str, int], ...] | None = None,
        algo_config: WeightOnlyQuantConfig | None = None,
    ):
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        self.model = ONNXModel(onnx.load(model)) if isinstance(model, str) else ONNXModel(model)
        self.model_path = model if isinstance(model, str) else None
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.accuracy_level = accuracy_level
        self.nodes_to_exclude = set(nodes_to_exclude)
        self.nodes_to_include = set(nodes_to_include) if nodes_to_include else None
        self.node_quantizer = None

        if algo_config is None:
            algo_config = DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=is_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_format,
                op_types_to_quantize=op_types_to_quantize,
                quant_axes=quant_axes,
                bits=4,  # default to 4 bits
            )

        self.algo_config = algo_config
        if hasattr(self.algo_config, "bits"):
            assert self.algo_config.bits in [4, 8], "Only support 4 or 8 bits quantization"

        if algo_config.algorithm == "HQQ":
            self.node_quantizer = HQQWeightOnlyQuantizer(self.algo_config)
        elif algo_config.algorithm == "DEFAULT":
            self.node_quantizer = DefaultWeightOnlyQuantizer(self.algo_config)
        elif algo_config.algorithm == "nvidia_awq":
            self.node_quantizer = NVAWQWeightOnlyQuantizer(self.algo_config)

    def _process_subgraph(self, graph_stack: list[GraphProto]):
        new_nodes = []
        graph = graph_stack[-1]

        for node in graph.node:
            graph_attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # recursive call to take care of sub-graph
                        graph_stack.append(attr.g)
                        kv = {attr.name: self._process_subgraph(graph_stack)}
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([self._process_subgraph(graph_stack)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )
            out_nodes = []
            if node.name in self.nodes_to_exclude:
                logger.info(f"exclude to quantize {node.name} as specified by nodes_to_exclude...")
                out_nodes = [node]
            elif (self.nodes_to_include and node.name in self.nodes_to_include) or (
                node.op_type in self.algo_config.op_types_to_quantize
            ):
                out_nodes = self.node_quantizer.quantize(node, graph_stack)
            else:
                logger.info(f"skip to quantize {node.name} ...")
                out_nodes = [node]
            new_nodes.extend(out_nodes)

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    def _generate_q4_node_config(self):
        """Generate weight only quant configuration for nodes."""
        q4_node_config = {}
        for node in self.model.model.graph.node:
            if node.op_type in ["MatMul"]:
                if not all(self.model.get_initializer(i) is None for i in node.input):
                    template_config_q4 = {
                        "bits": 4,
                        "group_size": self.block_size,
                        "scheme": "sym" if self.is_symmetric else "asym",
                    }
                    if (
                        self.algo_config.customized_weight_config
                        and node.name in self.algo_config.customized_weight_config
                    ):
                        for key, value in self.algo_config.customized_weight_config[node.name].items():
                            if key in template_config_q4:
                                template_config_q4[key] = value
                    q4_node_config[node.name] = template_config_q4
        return q4_node_config

    def int4_quant_algo(self):
        """4b quantize a model with RTN or GPTQ algorithm. Please refer to
        https://github.com/intel/neural-compressor/blob/master/docs/source/quantization_weight_only.md
        for more details on weight only quantization using Intel® Neural Compressor.
        """

        def inc_dataloader():
            data_reader = copy.deepcopy(self.algo_config.calibration_data_reader)
            for data in data_reader:
                yield data, None

        kwargs = {}
        if self.accuracy_level is not None:
            kwargs["accuracy_level"] = self.accuracy_level
        weight_only_node_config = self._generate_q4_node_config()

        algorithm = self.algo_config.algorithm
        logger.info(f"start to quantize model with {algorithm} algorithm...")
        if algorithm == "RTN":
            from neural_compressor.adaptor.ox_utils.weight_only import rtn_quantize

            kwargs["ratios"] = self.algo_config.ratios

            """
            neural-compressor uses fp32 to represent the node that skip quantization, it does not mean this node is fp32 type though.
            https://github.com/intel/neural-compressor/blob/a617115b1490bbe6163c0024fb55bd260c8914df/neural_compressor/adaptor/ox_utils/weight_only.py#L343
            """
            for n in self.nodes_to_exclude:
                weight_only_node_config[n] = "fp32"

            self.model = rtn_quantize(
                model=self.model_path if self.model_path is not None else self.model.model,
                weight_config=weight_only_node_config,
                **kwargs,
            )
        elif algorithm == "GPTQ":
            from neural_compressor.adaptor.ox_utils.weight_only import gptq_quantize

            kwargs["percdamp"] = self.algo_config.percdamp
            kwargs["blocksize"] = self.algo_config.block_size
            kwargs["actorder"] = self.algo_config.actorder
            kwargs["mse"] = self.algo_config.mse
            kwargs["perchannel"] = self.algo_config.perchannel
            kwargs["n_samples"] = -1
            dataloader = inc_dataloader()

            self.model = gptq_quantize(
                model=self.model_path if self.model_path is not None else self.model.model,
                weight_config=weight_only_node_config,
                dataloader=dataloader,
                **kwargs,
            )
        logger.info(f"complete quantization of model with {algorithm} algorithm.")

    def process(self):
        if self.algo_config.algorithm in ["HQQ", "DEFAULT"]:
            # use a stack to keep track of sub-graphs
            graph_stack = [self.model.graph()]

            # Update domain opset
            if self.algo_config.quant_format == QuantFormat.QOperator:
                self.model.set_opset_import("com.microsoft", 1)

            if self.algo_config.quant_format == QuantFormat.QDQ or "Gather" in self.algo_config.op_types_to_quantize:
                opset_import = self.model.opset_import()
                for opset in opset_import:
                    if opset.domain in [None, "ai.onnx", ""] and opset.version < 21:
                        logger.warning(
                            "The opset of the input model is under 21 and doesn't support int4 data type. "
                            "Force to update it to opset 21, but the generated model may not be a valid model."
                        )
                        self.model.set_opset_import(opset.domain, 21)

            self._process_subgraph(graph_stack)
            self.model.clean_initializers()
        elif self.algo_config.algorithm == "nvidia_awq":
            # Handle nvidia_awq quantization
            logger.info("Processing nvidia_awq quantization...")
            self.model = self.node_quantizer.quantize_awq(
                self.model.model if self.model_path is None else self.model_path
            )
            logger.info("Completed nvidia_awq quantization.")
            self.model = ONNXModel(self.model)  # Ensure the model is wrapped back into ONNXModel
            self.model.clean_initializers()
        else:
            # use Intel® Neural Compressor for RTN or GPTQ weight-only quantize algorithm
            try:
                importlib.import_module("neural_compressor")
            except Exception as e:
                logging.error(f"{e}.")
                raise RuntimeError(
                    "neural-compressor is not correctly installed. Please check your environment."
                ) from e

            import neural_compressor

            assert version.parse(neural_compressor.__version__) >= version.parse("2.3.2"), (
                "Require neural-compressor >= 2.3.2 to support weight only quantization!"
            )

            self.int4_quant_algo()


def ort_convert_str_to_bool(value):
    return value.lower() in ("true", "1")


# Custom function to parse str:int pairs
def parse_key_value_pair(s):
    key, value = s.split(":")
    return key, int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Blockwise int4 quantization for MatMul 2D weight matrices.

A weight matrix is partitioned into into blocks, where each block is a
continguous subset inside each column. Each block is quantized into a
set of 4b integers with a scaling factor and an optional offset.
"""
    )

    parser.add_argument("--input_model", required=True, help="Path to the input model file")
    parser.add_argument("--output_model", required=True, help="Path to the output model file")
    parser.add_argument("--block_size", required=False, default=32, type=int, help="Block size for quantization")
    parser.add_argument(
        "--quant_method",
        default="default",
        type=str,
        choices=["default", "hqq", "rtn", "gptq", "nvidia_awq"],
        help="the algorithm used to quantize weight, \nrtn and gptq leverage Intel® Neural Compressor",
    )
    parser.add_argument("--bits", default=4, type=int, help="the target bits to represent weight")
    parser.add_argument(
        "--symmetric",
        required=False,
        default=True,
        const=True,
        nargs="?",
        type=ort_convert_str_to_bool,
        choices=[True, False],
        help="Indicate whether to quantize the model symmetrically, symmetric is not supported by hqq",
    )
    parser.add_argument(
        "--accuracy_level",
        required=False,
        type=int,
        help="Accuracy level of the 4-bit quantized MatMul computation. "
        "Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details "
        "(https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits).",
    )
    parser.add_argument("-v", "--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--nodes_to_exclude",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="Specify the nodes to be excluded from quantization with node names",
    )
    parser.add_argument(
        "--nodes_to_include",
        nargs="+",
        type=str,
        required=False,
        help="Specify the specific nodes to be included from quantization with node names",
    )
    parser.add_argument(
        "--quant_format",
        default="QOperator",
        type=str,
        choices=["QOperator", "QDQ"],
        help="QuantFormat {QOperator, QDQ}"
        "QOperator format quantizes the model with quantized operators directly."
        "QDQ format quantize the model by inserting DeQuantizeLinear before the MatMul.",
    )
    parser.add_argument(
        "--op_types_to_quantize",
        type=str,
        nargs="+",
        choices=["MatMul", "Gather"],
        help="op_types_to_quantize {MatMul, Gather}. Operators to quantize. Default is MatMul.",
    )
    parser.add_argument(
        "--quant_axes",
        type=parse_key_value_pair,
        nargs="+",
        required=False,
        help="Key-value pairs in op_type:axis_to_quantize separated by space."
        "Specify the axis to quantize for an op. Default {MatMul:0, Gather:1}"
        "Example: --quant_axes MatMul:0 Gather:1",
    )
    # Group arguments specific to nvidia_awq
    nv_awq_config = parser.add_argument_group("nvidia_awq", "Arguments specific to nvidia_awq quantization")
    nv_awq_config.add_argument(
        "--calib_dataset_name",
        type=str,
        default="cnn",
        help="Name of the calibration dataset for nvidia_awq.",
    )
    nv_awq_config.add_argument(
        "--tokenizer_dir",
        type=str,
        required=False,
        help="Path of the tokenizer dir.",
    )
    nv_awq_config.add_argument(
        "--calibration_method",
        type=str,
        required=False,
        choices=["awq", "awq_clip"],
        help="Support two options, awq implementation and weight clipping.",
    )
    nv_awq_config.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory for calibration data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_model_path = args.input_model
    output_model_path = args.output_model
    quant_format = QuantFormat[args.quant_format]
    op_types_to_quantize = tuple(args.op_types_to_quantize) if args.op_types_to_quantize else ("MatMul",)
    quant_axes = tuple(args.quant_axes) if args.quant_axes else None

    if os.path.exists(output_model_path):
        logger.error(f"file {output_model_path} already exists")
        raise Exception(f"file {output_model_path} already exists")

    if args.symmetric and args.quant_method == "hqq":
        logger.warning("symmetric is not supportted by hqq, will force to symmetric=False")
        args.symmetric = False

    model = onnx.load(input_model_path)
    if args.quant_method == "hqq":
        quant_config = HQQWeightOnlyQuantConfig(
            block_size=args.block_size, bits=args.bits, op_types_to_quantize=op_types_to_quantize, quant_axes=quant_axes
        )
    elif args.quant_method == "default":
        quant_config = DefaultWeightOnlyQuantConfig(
            block_size=args.block_size,
            is_symmetric=args.symmetric,
            accuracy_level=args.accuracy_level,
            quant_format=quant_format,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
            bits=args.bits,
        )
    elif args.quant_method == "rtn":
        quant_config = RTNWeightOnlyQuantConfig(op_types_to_quantize=op_types_to_quantize)
    elif args.quant_method == "gptq":
        quant_config = GPTQWeightOnlyQuantConfig(block_size=args.block_size, op_types_to_quantize=op_types_to_quantize)
    elif args.quant_method == "nvidia_awq":
        if quant_format == QuantFormat.QOperator:
            logger.warning("QOperator is not applicable to nvidia_awq. overriding the value to QDQ")
            quant_format = QuantFormat.QDQ

        model = input_model_path
        if args.calibration_method is not None:
            if args.calibration_method == "awq":
                calibration_method = "awq_lite"
            else:
                calibration_method = "awq_clip"
        else:
            calibration_method = "awq_lite"

        quant_config = NVAWQWeightOnlyQuantConfig(
            dataset_name=args.calib_dataset_name,
            tokenizer_dir=args.tokenizer_dir,
            cache_dir=args.cache_dir,
            calibration_method=calibration_method,
        )
    else:
        raise ValueError(f"Unsupported quantization method: {args.quant_method}")

    quant = MatMulNBitsQuantizer(
        model=model,
        accuracy_level=args.accuracy_level,
        nodes_to_exclude=args.nodes_to_exclude,
        nodes_to_include=args.nodes_to_include,
        algo_config=quant_config,
    )
    quant.process()
    quant.model.save_model_to_file(output_model_path, True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_sequence.py ===
# testing/suite/test_sequence.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from .. import config
from .. import fixtures
from ..assertions import eq_
from ..assertions import is_true
from ..config import requirements
from ..provision import normalize_sequence
from ..schema import Column
from ..schema import Table
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import Sequence
from ... import String
from ... import testing


class SequenceTest(fixtures.TablesTest):
    __requires__ = ("sequences",)
    __backend__ = True

    run_create_tables = "each"

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "seq_pk",
            metadata,
            Column(
                "id",
                Integer,
                normalize_sequence(config, Sequence("tab_id_seq")),
                primary_key=True,
            ),
            Column("data", String(50)),
        )

        Table(
            "seq_opt_pk",
            metadata,
            Column(
                "id",
                Integer,
                normalize_sequence(
                    config,
                    Sequence("tab_id_seq", data_type=Integer, optional=True),
                ),
                primary_key=True,
            ),
            Column("data", String(50)),
        )

        Table(
            "seq_no_returning",
            metadata,
            Column(
                "id",
                Integer,
                normalize_sequence(config, Sequence("noret_id_seq")),
                primary_key=True,
            ),
            Column("data", String(50)),
            implicit_returning=False,
        )

        if testing.requires.schemas.enabled:
            Table(
                "seq_no_returning_sch",
                metadata,
                Column(
                    "id",
                    Integer,
                    normalize_sequence(
                        config,
                        Sequence(
                            "noret_sch_id_seq", schema=config.test_schema
                        ),
                    ),
                    primary_key=True,
                ),
                Column("data", String(50)),
                implicit_returning=False,
                schema=config.test_schema,
            )

    def test_insert_roundtrip(self, connection):
        connection.execute(self.tables.seq_pk.insert(), dict(data="some data"))
        self._assert_round_trip(self.tables.seq_pk, connection)

    def test_insert_lastrowid(self, connection):
        r = connection.execute(
            self.tables.seq_pk.insert(), dict(data="some data")
        )
        eq_(
            r.inserted_primary_key, (testing.db.dialect.default_sequence_base,)
        )

    def test_nextval_direct(self, connection):
        r = connection.scalar(self.tables.seq_pk.c.id.default)
        eq_(r, testing.db.dialect.default_sequence_base)

    @requirements.sequences_optional
    def test_optional_seq(self, connection):
        r = connection.execute(
            self.tables.seq_opt_pk.insert(), dict(data="some data")
        )
        eq_(r.inserted_primary_key, (1,))

    def _assert_round_trip(self, table, conn):
        row = conn.execute(table.select()).first()
        eq_(row, (testing.db.dialect.default_sequence_base, "some data"))

    def test_insert_roundtrip_no_implicit_returning(self, connection):
        connection.execute(
            self.tables.seq_no_returning.insert(), dict(data="some data")
        )
        self._assert_round_trip(self.tables.seq_no_returning, connection)

    @testing.combinations((True,), (False,), argnames="implicit_returning")
    @testing.requires.schemas
    def test_insert_roundtrip_translate(self, connection, implicit_returning):
        seq_no_returning = Table(
            "seq_no_returning_sch",
            MetaData(),
            Column(
                "id",
                Integer,
                normalize_sequence(
                    config, Sequence("noret_sch_id_seq", schema="alt_schema")
                ),
                primary_key=True,
            ),
            Column("data", String(50)),
            implicit_returning=implicit_returning,
            schema="alt_schema",
        )

        connection = connection.execution_options(
            schema_translate_map={"alt_schema": config.test_schema}
        )
        connection.execute(seq_no_returning.insert(), dict(data="some data"))
        self._assert_round_trip(seq_no_returning, connection)

    @testing.requires.schemas
    def test_nextval_direct_schema_translate(self, connection):
        seq = normalize_sequence(
            config, Sequence("noret_sch_id_seq", schema="alt_schema")
        )
        connection = connection.execution_options(
            schema_translate_map={"alt_schema": config.test_schema}
        )

        r = connection.scalar(seq)
        eq_(r, testing.db.dialect.default_sequence_base)


class SequenceCompilerTest(testing.AssertsCompiledSQL, fixtures.TestBase):
    __requires__ = ("sequences",)
    __backend__ = True

    def test_literal_binds_inline_compile(self, connection):
        table = Table(
            "x",
            MetaData(),
            Column(
                "y", Integer, normalize_sequence(config, Sequence("y_seq"))
            ),
            Column("q", Integer),
        )

        stmt = table.insert().values(q=5)

        seq_nextval = connection.dialect.statement_compiler(
            statement=None, dialect=connection.dialect
        ).visit_sequence(normalize_sequence(config, Sequence("y_seq")))
        self.assert_compile(
            stmt,
            "INSERT INTO x (y, q) VALUES (%s, 5)" % (seq_nextval,),
            literal_binds=True,
            dialect=connection.dialect,
        )


class HasSequenceTest(fixtures.TablesTest):
    run_deletes = None

    __requires__ = ("sequences",)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        normalize_sequence(config, Sequence("user_id_seq", metadata=metadata))
        normalize_sequence(
            config,
            Sequence(
                "other_seq",
                metadata=metadata,
                nomaxvalue=True,
                nominvalue=True,
            ),
        )
        if testing.requires.schemas.enabled:
            normalize_sequence(
                config,
                Sequence(
                    "user_id_seq", schema=config.test_schema, metadata=metadata
                ),
            )
            normalize_sequence(
                config,
                Sequence(
                    "schema_seq", schema=config.test_schema, metadata=metadata
                ),
            )
        Table(
            "user_id_table",
            metadata,
            Column("id", Integer, primary_key=True),
        )

    def test_has_sequence(self, connection):
        eq_(inspect(connection).has_sequence("user_id_seq"), True)

    def test_has_sequence_cache(self, connection, metadata):
        insp = inspect(connection)
        eq_(insp.has_sequence("user_id_seq"), True)
        ss = normalize_sequence(config, Sequence("new_seq", metadata=metadata))
        eq_(insp.has_sequence("new_seq"), False)
        ss.create(connection)
        try:
            eq_(insp.has_sequence("new_seq"), False)
            insp.clear_cache()
            eq_(insp.has_sequence("new_seq"), True)
        finally:
            ss.drop(connection)

    def test_has_sequence_other_object(self, connection):
        eq_(inspect(connection).has_sequence("user_id_table"), False)

    @testing.requires.schemas
    def test_has_sequence_schema(self, connection):
        eq_(
            inspect(connection).has_sequence(
                "user_id_seq", schema=config.test_schema
            ),
            True,
        )

    def test_has_sequence_neg(self, connection):
        eq_(inspect(connection).has_sequence("some_sequence"), False)

    @testing.requires.schemas
    def test_has_sequence_schemas_neg(self, connection):
        eq_(
            inspect(connection).has_sequence(
                "some_sequence", schema=config.test_schema
            ),
            False,
        )

    @testing.requires.schemas
    def test_has_sequence_default_not_in_remote(self, connection):
        eq_(
            inspect(connection).has_sequence(
                "other_sequence", schema=config.test_schema
            ),
            False,
        )

    @testing.requires.schemas
    def test_has_sequence_remote_not_in_default(self, connection):
        eq_(inspect(connection).has_sequence("schema_seq"), False)

    def test_get_sequence_names(self, connection):
        exp = {"other_seq", "user_id_seq"}

        res = set(inspect(connection).get_sequence_names())
        is_true(res.intersection(exp) == exp)
        is_true("schema_seq" not in res)

    @testing.requires.schemas
    def test_get_sequence_names_no_sequence_schema(self, connection):
        eq_(
            inspect(connection).get_sequence_names(
                schema=config.test_schema_2
            ),
            [],
        )

    @testing.requires.schemas
    def test_get_sequence_names_sequences_schema(self, connection):
        eq_(
            sorted(
                inspect(connection).get_sequence_names(
                    schema=config.test_schema
                )
            ),
            ["schema_seq", "user_id_seq"],
        )


class HasSequenceTestEmpty(fixtures.TestBase):
    __requires__ = ("sequences",)
    __backend__ = True

    def test_get_sequence_names_no_sequence(self, connection):
        eq_(
            inspect(connection).get_sequence_names(),
            [],
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\hybrid.py ===
# ext/hybrid.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

r"""Define attributes on ORM-mapped classes that have "hybrid" behavior.

"hybrid" means the attribute has distinct behaviors defined at the
class level and at the instance level.

The :mod:`~sqlalchemy.ext.hybrid` extension provides a special form of
method decorator and has minimal dependencies on the rest of SQLAlchemy.
Its basic theory of operation can work with any descriptor-based expression
system.

Consider a mapping ``Interval``, representing integer ``start`` and ``end``
values. We can define higher level functions on mapped classes that produce SQL
expressions at the class level, and Python expression evaluation at the
instance level.  Below, each function decorated with :class:`.hybrid_method` or
:class:`.hybrid_property` may receive ``self`` as an instance of the class, or
may receive the class directly, depending on context::

    from __future__ import annotations

    from sqlalchemy.ext.hybrid import hybrid_method
    from sqlalchemy.ext.hybrid import hybrid_property
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column


    class Base(DeclarativeBase):
        pass


    class Interval(Base):
        __tablename__ = "interval"

        id: Mapped[int] = mapped_column(primary_key=True)
        start: Mapped[int]
        end: Mapped[int]

        def __init__(self, start: int, end: int):
            self.start = start
            self.end = end

        @hybrid_property
        def length(self) -> int:
            return self.end - self.start

        @hybrid_method
        def contains(self, point: int) -> bool:
            return (self.start <= point) & (point <= self.end)

        @hybrid_method
        def intersects(self, other: Interval) -> bool:
            return self.contains(other.start) | self.contains(other.end)

Above, the ``length`` property returns the difference between the
``end`` and ``start`` attributes.  With an instance of ``Interval``,
this subtraction occurs in Python, using normal Python descriptor
mechanics::

    >>> i1 = Interval(5, 10)
    >>> i1.length
    5

When dealing with the ``Interval`` class itself, the :class:`.hybrid_property`
descriptor evaluates the function body given the ``Interval`` class as
the argument, which when evaluated with SQLAlchemy expression mechanics
returns a new SQL expression:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import select
    >>> print(select(Interval.length))
    {printsql}SELECT interval."end" - interval.start AS length
    FROM interval{stop}


    >>> print(select(Interval).filter(Interval.length > 10))
    {printsql}SELECT interval.id, interval.start, interval."end"
    FROM interval
    WHERE interval."end" - interval.start > :param_1

Filtering methods such as :meth:`.Select.filter_by` are supported
with hybrid attributes as well:

.. sourcecode:: pycon+sql

    >>> print(select(Interval).filter_by(length=5))
    {printsql}SELECT interval.id, interval.start, interval."end"
    FROM interval
    WHERE interval."end" - interval.start = :param_1

The ``Interval`` class example also illustrates two methods,
``contains()`` and ``intersects()``, decorated with
:class:`.hybrid_method`. This decorator applies the same idea to
methods that :class:`.hybrid_property` applies to attributes.   The
methods return boolean values, and take advantage of the Python ``|``
and ``&`` bitwise operators to produce equivalent instance-level and
SQL expression-level boolean behavior:

.. sourcecode:: pycon+sql

    >>> i1.contains(6)
    True
    >>> i1.contains(15)
    False
    >>> i1.intersects(Interval(7, 18))
    True
    >>> i1.intersects(Interval(25, 29))
    False

    >>> print(select(Interval).filter(Interval.contains(15)))
    {printsql}SELECT interval.id, interval.start, interval."end"
    FROM interval
    WHERE interval.start <= :start_1 AND interval."end" > :end_1{stop}

    >>> ia = aliased(Interval)
    >>> print(select(Interval, ia).filter(Interval.intersects(ia)))
    {printsql}SELECT interval.id, interval.start,
    interval."end", interval_1.id AS interval_1_id,
    interval_1.start AS interval_1_start, interval_1."end" AS interval_1_end
    FROM interval, interval AS interval_1
    WHERE interval.start <= interval_1.start
        AND interval."end" > interval_1.start
        OR interval.start <= interval_1."end"
        AND interval."end" > interval_1."end"{stop}

.. _hybrid_distinct_expression:

Defining Expression Behavior Distinct from Attribute Behavior
--------------------------------------------------------------

In the previous section, our usage of the ``&`` and ``|`` bitwise operators
within the ``Interval.contains`` and ``Interval.intersects`` methods was
fortunate, considering our functions operated on two boolean values to return a
new one. In many cases, the construction of an in-Python function and a
SQLAlchemy SQL expression have enough differences that two separate Python
expressions should be defined. The :mod:`~sqlalchemy.ext.hybrid` decorator
defines a **modifier** :meth:`.hybrid_property.expression` for this purpose. As an
example we'll define the radius of the interval, which requires the usage of
the absolute value function::

    from sqlalchemy import ColumnElement
    from sqlalchemy import Float
    from sqlalchemy import func
    from sqlalchemy import type_coerce


    class Interval(Base):
        # ...

        @hybrid_property
        def radius(self) -> float:
            return abs(self.length) / 2

        @radius.inplace.expression
        @classmethod
        def _radius_expression(cls) -> ColumnElement[float]:
            return type_coerce(func.abs(cls.length) / 2, Float)

In the above example, the :class:`.hybrid_property` first assigned to the
name ``Interval.radius`` is amended by a subsequent method called
``Interval._radius_expression``, using the decorator
``@radius.inplace.expression``, which chains together two modifiers
:attr:`.hybrid_property.inplace` and :attr:`.hybrid_property.expression`.
The use of :attr:`.hybrid_property.inplace` indicates that the
:meth:`.hybrid_property.expression` modifier should mutate the
existing hybrid object at ``Interval.radius`` in place, without creating a
new object.   Notes on this modifier and its
rationale are discussed in the next section :ref:`hybrid_pep484_naming`.
The use of ``@classmethod`` is optional, and is strictly to give typing
tools a hint that ``cls`` in this case is expected to be the ``Interval``
class, and not an instance of ``Interval``.

.. note:: :attr:`.hybrid_property.inplace` as well as the use of ``@classmethod``
   for proper typing support are available as of SQLAlchemy 2.0.4, and will
   not work in earlier versions.

With ``Interval.radius`` now including an expression element, the SQL
function ``ABS()`` is returned when accessing ``Interval.radius``
at the class level:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import select
    >>> print(select(Interval).filter(Interval.radius > 5))
    {printsql}SELECT interval.id, interval.start, interval."end"
    FROM interval
    WHERE abs(interval."end" - interval.start) / :abs_1 > :param_1


.. _hybrid_pep484_naming:

Using ``inplace`` to create pep-484 compliant hybrid properties
---------------------------------------------------------------

In the previous section, a :class:`.hybrid_property` decorator is illustrated
which includes two separate method-level functions being decorated, both
to produce a single object attribute referenced as ``Interval.radius``.
There are actually several different modifiers we can use for
:class:`.hybrid_property` including :meth:`.hybrid_property.expression`,
:meth:`.hybrid_property.setter` and :meth:`.hybrid_property.update_expression`.

SQLAlchemy's :class:`.hybrid_property` decorator intends that adding on these
methods may be done in the identical manner as Python's built-in
``@property`` decorator, where idiomatic use is to continue to redefine the
attribute repeatedly, using the **same attribute name** each time, as in the
example below that illustrates the use of :meth:`.hybrid_property.setter` and
:meth:`.hybrid_property.expression` for the ``Interval.radius`` descriptor::

    # correct use, however is not accepted by pep-484 tooling


    class Interval(Base):
        # ...

        @hybrid_property
        def radius(self):
            return abs(self.length) / 2

        @radius.setter
        def radius(self, value):
            self.length = value * 2

        @radius.expression
        def radius(cls):
            return type_coerce(func.abs(cls.length) / 2, Float)

Above, there are three ``Interval.radius`` methods, but as each are decorated,
first by the :class:`.hybrid_property` decorator and then by the
``@radius`` name itself, the end effect is that ``Interval.radius`` is
a single attribute with three different functions contained within it.
This style of use is taken from `Python's documented use of @property
<https://docs.python.org/3/library/functions.html#property>`_.
It is important to note that the way both ``@property`` as well as
:class:`.hybrid_property` work, a **copy of the descriptor is made each time**.
That is, each call to ``@radius.expression``, ``@radius.setter`` etc.
make a new object entirely.  This allows the attribute to be re-defined in
subclasses without issue (see :ref:`hybrid_reuse_subclass` later in this
section for how this is used).

However, the above approach is not compatible with typing tools such as
mypy and pyright.  Python's own ``@property`` decorator does not have this
limitation only because
`these tools hardcode the behavior of @property
<https://github.com/python/typing/discussions/1102>`_, meaning this syntax
is not available to SQLAlchemy under :pep:`484` compliance.

In order to produce a reasonable syntax while remaining typing compliant,
the :attr:`.hybrid_property.inplace` decorator allows the same
decorator to be re-used with different method names, while still producing
a single decorator under one name::

    # correct use which is also accepted by pep-484 tooling


    class Interval(Base):
        # ...

        @hybrid_property
        def radius(self) -> float:
            return abs(self.length) / 2

        @radius.inplace.setter
        def _radius_setter(self, value: float) -> None:
            # for example only
            self.length = value * 2

        @radius.inplace.expression
        @classmethod
        def _radius_expression(cls) -> ColumnElement[float]:
            return type_coerce(func.abs(cls.length) / 2, Float)

Using :attr:`.hybrid_property.inplace` further qualifies the use of the
decorator that a new copy should not be made, thereby maintaining the
``Interval.radius`` name while allowing additional methods
``Interval._radius_setter`` and ``Interval._radius_expression`` to be
differently named.


.. versionadded:: 2.0.4 Added :attr:`.hybrid_property.inplace` to allow
   less verbose construction of composite :class:`.hybrid_property` objects
   while not having to use repeated method names.   Additionally allowed the
   use of ``@classmethod`` within :attr:`.hybrid_property.expression`,
   :attr:`.hybrid_property.update_expression`, and
   :attr:`.hybrid_property.comparator` to allow typing tools to identify
   ``cls`` as a class and not an instance in the method signature.


Defining Setters
----------------

The :meth:`.hybrid_property.setter` modifier allows the construction of a
custom setter method, that can modify values on the object::

    class Interval(Base):
        # ...

        @hybrid_property
        def length(self) -> int:
            return self.end - self.start

        @length.inplace.setter
        def _length_setter(self, value: int) -> None:
            self.end = self.start + value

The ``length(self, value)`` method is now called upon set::

    >>> i1 = Interval(5, 10)
    >>> i1.length
    5
    >>> i1.length = 12
    >>> i1.end
    17

.. _hybrid_bulk_update:

Allowing Bulk ORM Update
------------------------

A hybrid can define a custom "UPDATE" handler for when using
ORM-enabled updates, allowing the hybrid to be used in the
SET clause of the update.

Normally, when using a hybrid with :func:`_sql.update`, the SQL
expression is used as the column that's the target of the SET.  If our
``Interval`` class had a hybrid ``start_point`` that linked to
``Interval.start``, this could be substituted directly::

    from sqlalchemy import update

    stmt = update(Interval).values({Interval.start_point: 10})

However, when using a composite hybrid like ``Interval.length``, this
hybrid represents more than one column.   We can set up a handler that will
accommodate a value passed in the VALUES expression which can affect
this, using the :meth:`.hybrid_property.update_expression` decorator.
A handler that works similarly to our setter would be::

    from typing import List, Tuple, Any


    class Interval(Base):
        # ...

        @hybrid_property
        def length(self) -> int:
            return self.end - self.start

        @length.inplace.setter
        def _length_setter(self, value: int) -> None:
            self.end = self.start + value

        @length.inplace.update_expression
        def _length_update_expression(
            cls, value: Any
        ) -> List[Tuple[Any, Any]]:
            return [(cls.end, cls.start + value)]

Above, if we use ``Interval.length`` in an UPDATE expression, we get
a hybrid SET expression:

.. sourcecode:: pycon+sql


    >>> from sqlalchemy import update
    >>> print(update(Interval).values({Interval.length: 25}))
    {printsql}UPDATE interval SET "end"=(interval.start + :start_1)

This SET expression is accommodated by the ORM automatically.

.. seealso::

    :ref:`orm_expression_update_delete` - includes background on ORM-enabled
    UPDATE statements


Working with Relationships
--------------------------

There's no essential difference when creating hybrids that work with
related objects as opposed to column-based data. The need for distinct
expressions tends to be greater.  The two variants we'll illustrate
are the "join-dependent" hybrid, and the "correlated subquery" hybrid.

Join-Dependent Relationship Hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider the following declarative
mapping which relates a ``User`` to a ``SavingsAccount``::

    from __future__ import annotations

    from decimal import Decimal
    from typing import cast
    from typing import List
    from typing import Optional

    from sqlalchemy import ForeignKey
    from sqlalchemy import Numeric
    from sqlalchemy import String
    from sqlalchemy import SQLColumnExpression
    from sqlalchemy.ext.hybrid import hybrid_property
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column
    from sqlalchemy.orm import relationship


    class Base(DeclarativeBase):
        pass


    class SavingsAccount(Base):
        __tablename__ = "account"
        id: Mapped[int] = mapped_column(primary_key=True)
        user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
        balance: Mapped[Decimal] = mapped_column(Numeric(15, 5))

        owner: Mapped[User] = relationship(back_populates="accounts")


    class User(Base):
        __tablename__ = "user"
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))

        accounts: Mapped[List[SavingsAccount]] = relationship(
            back_populates="owner", lazy="selectin"
        )

        @hybrid_property
        def balance(self) -> Optional[Decimal]:
            if self.accounts:
                return self.accounts[0].balance
            else:
                return None

        @balance.inplace.setter
        def _balance_setter(self, value: Optional[Decimal]) -> None:
            assert value is not None

            if not self.accounts:
                account = SavingsAccount(owner=self)
            else:
                account = self.accounts[0]
            account.balance = value

        @balance.inplace.expression
        @classmethod
        def _balance_expression(cls) -> SQLColumnExpression[Optional[Decimal]]:
            return cast(
                "SQLColumnExpression[Optional[Decimal]]",
                SavingsAccount.balance,
            )

The above hybrid property ``balance`` works with the first
``SavingsAccount`` entry in the list of accounts for this user.   The
in-Python getter/setter methods can treat ``accounts`` as a Python
list available on ``self``.

.. tip:: The ``User.balance`` getter in the above example accesses the
   ``self.acccounts`` collection, which will normally be loaded via the
   :func:`.selectinload` loader strategy configured on the ``User.balance``
   :func:`_orm.relationship`. The default loader strategy when not otherwise
   stated on :func:`_orm.relationship` is :func:`.lazyload`, which emits SQL on
   demand. When using asyncio, on-demand loaders such as :func:`.lazyload` are
   not supported, so care should be taken to ensure the ``self.accounts``
   collection is accessible to this hybrid accessor when using asyncio.

At the expression level, it's expected that the ``User`` class will
be used in an appropriate context such that an appropriate join to
``SavingsAccount`` will be present:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import select
    >>> print(
    ...     select(User, User.balance)
    ...     .join(User.accounts)
    ...     .filter(User.balance > 5000)
    ... )
    {printsql}SELECT "user".id AS user_id, "user".name AS user_name,
    account.balance AS account_balance
    FROM "user" JOIN account ON "user".id = account.user_id
    WHERE account.balance > :balance_1

Note however, that while the instance level accessors need to worry
about whether ``self.accounts`` is even present, this issue expresses
itself differently at the SQL expression level, where we basically
would use an outer join:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import select
    >>> from sqlalchemy import or_
    >>> print(
    ...     select(User, User.balance)
    ...     .outerjoin(User.accounts)
    ...     .filter(or_(User.balance < 5000, User.balance == None))
    ... )
    {printsql}SELECT "user".id AS user_id, "user".name AS user_name,
    account.balance AS account_balance
    FROM "user" LEFT OUTER JOIN account ON "user".id = account.user_id
    WHERE account.balance <  :balance_1 OR account.balance IS NULL

Correlated Subquery Relationship Hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can, of course, forego being dependent on the enclosing query's usage
of joins in favor of the correlated subquery, which can portably be packed
into a single column expression. A correlated subquery is more portable, but
often performs more poorly at the SQL level. Using the same technique
illustrated at :ref:`mapper_column_property_sql_expressions`,
we can adjust our ``SavingsAccount`` example to aggregate the balances for
*all* accounts, and use a correlated subquery for the column expression::

    from __future__ import annotations

    from decimal import Decimal
    from typing import List

    from sqlalchemy import ForeignKey
    from sqlalchemy import func
    from sqlalchemy import Numeric
    from sqlalchemy import select
    from sqlalchemy import SQLColumnExpression
    from sqlalchemy import String
    from sqlalchemy.ext.hybrid import hybrid_property
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column
    from sqlalchemy.orm import relationship


    class Base(DeclarativeBase):
        pass


    class SavingsAccount(Base):
        __tablename__ = "account"
        id: Mapped[int] = mapped_column(primary_key=True)
        user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
        balance: Mapped[Decimal] = mapped_column(Numeric(15, 5))

        owner: Mapped[User] = relationship(back_populates="accounts")


    class User(Base):
        __tablename__ = "user"
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(100))

        accounts: Mapped[List[SavingsAccount]] = relationship(
            back_populates="owner", lazy="selectin"
        )

        @hybrid_property
        def balance(self) -> Decimal:
            return sum(
                (acc.balance for acc in self.accounts), start=Decimal("0")
            )

        @balance.inplace.expression
        @classmethod
        def _balance_expression(cls) -> SQLColumnExpression[Decimal]:
            return (
                select(func.sum(SavingsAccount.balance))
                .where(SavingsAccount.user_id == cls.id)
                .label("total_balance")
            )

The above recipe will give us the ``balance`` column which renders
a correlated SELECT:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import select
    >>> print(select(User).filter(User.balance > 400))
    {printsql}SELECT "user".id, "user".name
    FROM "user"
    WHERE (
        SELECT sum(account.balance) AS sum_1 FROM account
        WHERE account.user_id = "user".id
    ) > :param_1


.. _hybrid_custom_comparators:

Building Custom Comparators
---------------------------

The hybrid property also includes a helper that allows construction of
custom comparators. A comparator object allows one to customize the
behavior of each SQLAlchemy expression operator individually.  They
are useful when creating custom types that have some highly
idiosyncratic behavior on the SQL side.

.. note::  The :meth:`.hybrid_property.comparator` decorator introduced
   in this section **replaces** the use of the
   :meth:`.hybrid_property.expression` decorator.
   They cannot be used together.

The example class below allows case-insensitive comparisons on the attribute
named ``word_insensitive``::

    from __future__ import annotations

    from typing import Any

    from sqlalchemy import ColumnElement
    from sqlalchemy import func
    from sqlalchemy.ext.hybrid import Comparator
    from sqlalchemy.ext.hybrid import hybrid_property
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped
    from sqlalchemy.orm import mapped_column


    class Base(DeclarativeBase):
        pass


    class CaseInsensitiveComparator(Comparator[str]):
        def __eq__(self, other: Any) -> ColumnElement[bool]:  # type: ignore[override]  # noqa: E501
            return func.lower(self.__clause_element__()) == func.lower(other)


    class SearchWord(Base):
        __tablename__ = "searchword"

        id: Mapped[int] = mapped_column(primary_key=True)
        word: Mapped[str]

        @hybrid_property
        def word_insensitive(self) -> str:
            return self.word.lower()

        @word_insensitive.inplace.comparator
        @classmethod
        def _word_insensitive_comparator(cls) -> CaseInsensitiveComparator:
            return CaseInsensitiveComparator(cls.word)

Above, SQL expressions against ``word_insensitive`` will apply the ``LOWER()``
SQL function to both sides:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy import select
    >>> print(select(SearchWord).filter_by(word_insensitive="Trucks"))
    {printsql}SELECT searchword.id, searchword.word
    FROM searchword
    WHERE lower(searchword.word) = lower(:lower_1)


The ``CaseInsensitiveComparator`` above implements part of the
:class:`.ColumnOperators` interface.   A "coercion" operation like
lowercasing can be applied to all comparison operations (i.e. ``eq``,
``lt``, ``gt``, etc.) using :meth:`.Operators.operate`::

    class CaseInsensitiveComparator(Comparator):
        def operate(self, op, other, **kwargs):
            return op(
                func.lower(self.__clause_element__()),
                func.lower(other),
                **kwargs,
            )

.. _hybrid_reuse_subclass:

Reusing Hybrid Properties across Subclasses
-------------------------------------------

A hybrid can be referred to from a superclass, to allow modifying
methods like :meth:`.hybrid_property.getter`, :meth:`.hybrid_property.setter`
to be used to redefine those methods on a subclass.  This is similar to
how the standard Python ``@property`` object works::

    class FirstNameOnly(Base):
        # ...

        first_name: Mapped[str]

        @hybrid_property
        def name(self) -> str:
            return self.first_name

        @name.inplace.setter
        def _name_setter(self, value: str) -> None:
            self.first_name = value


    class FirstNameLastName(FirstNameOnly):
        # ...

        last_name: Mapped[str]

        # 'inplace' is not used here; calling getter creates a copy
        # of FirstNameOnly.name that is local to FirstNameLastName
        @FirstNameOnly.name.getter
        def name(self) -> str:
            return self.first_name + " " + self.last_name

        @name.inplace.setter
        def _name_setter(self, value: str) -> None:
            self.first_name, self.last_name = value.split(" ", 1)

Above, the ``FirstNameLastName`` class refers to the hybrid from
``FirstNameOnly.name`` to repurpose its getter and setter for the subclass.

When overriding :meth:`.hybrid_property.expression` and
:meth:`.hybrid_property.comparator` alone as the first reference to the
superclass, these names conflict with the same-named accessors on the class-
level :class:`.QueryableAttribute` object returned at the class level.  To
override these methods when referring directly to the parent class descriptor,
add the special qualifier :attr:`.hybrid_property.overrides`, which will de-
reference the instrumented attribute back to the hybrid object::

    class FirstNameLastName(FirstNameOnly):
        # ...

        last_name: Mapped[str]

        @FirstNameOnly.name.overrides.expression
        @classmethod
        def name(cls):
            return func.concat(cls.first_name, " ", cls.last_name)

Hybrid Value Objects
--------------------

Note in our previous example, if we were to compare the ``word_insensitive``
attribute of a ``SearchWord`` instance to a plain Python string, the plain
Python string would not be coerced to lower case - the
``CaseInsensitiveComparator`` we built, being returned by
``@word_insensitive.comparator``, only applies to the SQL side.

A more comprehensive form of the custom comparator is to construct a *Hybrid
Value Object*. This technique applies the target value or expression to a value
object which is then returned by the accessor in all cases.   The value object
allows control of all operations upon the value as well as how compared values
are treated, both on the SQL expression side as well as the Python value side.
Replacing the previous ``CaseInsensitiveComparator`` class with a new
``CaseInsensitiveWord`` class::

    class CaseInsensitiveWord(Comparator):
        "Hybrid value representing a lower case representation of a word."

        def __init__(self, word):
            if isinstance(word, basestring):
                self.word = word.lower()
            elif isinstance(word, CaseInsensitiveWord):
                self.word = word.word
            else:
                self.word = func.lower(word)

        def operate(self, op, other, **kwargs):
            if not isinstance(other, CaseInsensitiveWord):
                other = CaseInsensitiveWord(other)
            return op(self.word, other.word, **kwargs)

        def __clause_element__(self):
            return self.word

        def __str__(self):
            return self.word

        key = "word"
        "Label to apply to Query tuple results"

Above, the ``CaseInsensitiveWord`` object represents ``self.word``, which may
be a SQL function, or may be a Python native.   By overriding ``operate()`` and
``__clause_element__()`` to work in terms of ``self.word``, all comparison
operations will work against the "converted" form of ``word``, whether it be
SQL side or Python side. Our ``SearchWord`` class can now deliver the
``CaseInsensitiveWord`` object unconditionally from a single hybrid call::

    class SearchWord(Base):
        __tablename__ = "searchword"
        id: Mapped[int] = mapped_column(primary_key=True)
        word: Mapped[str]

        @hybrid_property
        def word_insensitive(self) -> CaseInsensitiveWord:
            return CaseInsensitiveWord(self.word)

The ``word_insensitive`` attribute now has case-insensitive comparison behavior
universally, including SQL expression vs. Python expression (note the Python
value is converted to lower case on the Python side here):

.. sourcecode:: pycon+sql

    >>> print(select(SearchWord).filter_by(word_insensitive="Trucks"))
    {printsql}SELECT searchword.id AS searchword_id, searchword.word AS searchword_word
    FROM searchword
    WHERE lower(searchword.word) = :lower_1

SQL expression versus SQL expression:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy.orm import aliased
    >>> sw1 = aliased(SearchWord)
    >>> sw2 = aliased(SearchWord)
    >>> print(
    ...     select(sw1.word_insensitive, sw2.word_insensitive).filter(
    ...         sw1.word_insensitive > sw2.word_insensitive
    ...     )
    ... )
    {printsql}SELECT lower(searchword_1.word) AS lower_1,
    lower(searchword_2.word) AS lower_2
    FROM searchword AS searchword_1, searchword AS searchword_2
    WHERE lower(searchword_1.word) > lower(searchword_2.word)

Python only expression::

    >>> ws1 = SearchWord(word="SomeWord")
    >>> ws1.word_insensitive == "sOmEwOrD"
    True
    >>> ws1.word_insensitive == "XOmEwOrX"
    False
    >>> print(ws1.word_insensitive)
    someword

The Hybrid Value pattern is very useful for any kind of value that may have
multiple representations, such as timestamps, time deltas, units of
measurement, currencies and encrypted passwords.

.. seealso::

    `Hybrids and Value Agnostic Types
    <https://techspot.zzzeek.org/2011/10/21/hybrids-and-value-agnostic-types/>`_
    - on the techspot.zzzeek.org blog

    `Value Agnostic Types, Part II
    <https://techspot.zzzeek.org/2011/10/29/value-agnostic-types-part-ii/>`_ -
    on the techspot.zzzeek.org blog


"""  # noqa

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .. import util
from ..orm import attributes
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.attributes import QueryableAttribute
from ..sql import roles
from ..sql._typing import is_has_clause_element
from ..sql.elements import ColumnElement
from ..sql.elements import SQLCoreOperations
from ..util.typing import Concatenate
from ..util.typing import Literal
from ..util.typing import ParamSpec
from ..util.typing import Protocol
from ..util.typing import Self

if TYPE_CHECKING:
    from ..orm.interfaces import MapperProperty
    from ..orm.util import AliasedInsp
    from ..sql import SQLColumnExpression
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _DMLColumnArgument
    from ..sql._typing import _HasClauseElement
    from ..sql._typing import _InfoType
    from ..sql.operators import OperatorType

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T", bound=Any)
_TE = TypeVar("_TE", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)
_T_con = TypeVar("_T_con", bound=Any, contravariant=True)


class HybridExtensionType(InspectionAttrExtensionType):
    HYBRID_METHOD = "HYBRID_METHOD"
    """Symbol indicating an :class:`InspectionAttr` that's
    of type :class:`.hybrid_method`.

    Is assigned to the :attr:`.InspectionAttr.extension_type`
    attribute.

    .. seealso::

        :attr:`_orm.Mapper.all_orm_attributes`

    """

    HYBRID_PROPERTY = "HYBRID_PROPERTY"
    """Symbol indicating an :class:`InspectionAttr` that's
        of type :class:`.hybrid_method`.

    Is assigned to the :attr:`.InspectionAttr.extension_type`
    attribute.

    .. seealso::

        :attr:`_orm.Mapper.all_orm_attributes`

    """


class _HybridGetterType(Protocol[_T_co]):
    def __call__(s, self: Any) -> _T_co: ...


class _HybridSetterType(Protocol[_T_con]):
    def __call__(s, self: Any, value: _T_con) -> None: ...


class _HybridUpdaterType(Protocol[_T_con]):
    def __call__(
        s,
        cls: Any,
        value: Union[_T_con, _ColumnExpressionArgument[_T_con]],
    ) -> List[Tuple[_DMLColumnArgument, Any]]: ...


class _HybridDeleterType(Protocol[_T_co]):
    def __call__(s, self: Any) -> None: ...


class _HybridExprCallableType(Protocol[_T_co]):
    def __call__(
        s, cls: Any
    ) -> Union[_HasClauseElement[_T_co], SQLColumnExpression[_T_co]]: ...


class _HybridComparatorCallableType(Protocol[_T]):
    def __call__(self, cls: Any) -> Comparator[_T]: ...


class _HybridClassLevelAccessor(QueryableAttribute[_T]):
    """Describe the object returned by a hybrid_property() when
    called as a class-level descriptor.

    """

    if TYPE_CHECKING:

        def getter(
            self, fget: _HybridGetterType[_T]
        ) -> hybrid_property[_T]: ...

        def setter(
            self, fset: _HybridSetterType[_T]
        ) -> hybrid_property[_T]: ...

        def deleter(
            self, fdel: _HybridDeleterType[_T]
        ) -> hybrid_property[_T]: ...

        @property
        def overrides(self) -> hybrid_property[_T]: ...

        def update_expression(
            self, meth: _HybridUpdaterType[_T]
        ) -> hybrid_property[_T]: ...


class hybrid_method(interfaces.InspectionAttrInfo, Generic[_P, _R]):
    """A decorator which allows definition of a Python object method with both
    instance-level and class-level behavior.

    """

    is_attribute = True
    extension_type = HybridExtensionType.HYBRID_METHOD

    def __init__(
        self,
        func: Callable[Concatenate[Any, _P], _R],
        expr: Optional[
            Callable[Concatenate[Any, _P], SQLCoreOperations[_R]]
        ] = None,
    ):
        """Create a new :class:`.hybrid_method`.

        Usage is typically via decorator::

            from sqlalchemy.ext.hybrid import hybrid_method


            class SomeClass:
                @hybrid_method
                def value(self, x, y):
                    return self._value + x + y

                @value.expression
                @classmethod
                def value(cls, x, y):
                    return func.some_function(cls._value, x, y)

        """
        self.func = func
        if expr is not None:
            self.expression(expr)
        else:
            self.expression(func)  # type: ignore

    @property
    def inplace(self) -> Self:
        """Return the inplace mutator for this :class:`.hybrid_method`.

        The :class:`.hybrid_method` class already performs "in place" mutation
        when the :meth:`.hybrid_method.expression` decorator is called,
        so this attribute returns Self.

        .. versionadded:: 2.0.4

        .. seealso::

            :ref:`hybrid_pep484_naming`

        """
        return self

    @overload
    def __get__(
        self, instance: Literal[None], owner: Type[object]
    ) -> Callable[_P, SQLCoreOperations[_R]]: ...

    @overload
    def __get__(
        self, instance: object, owner: Type[object]
    ) -> Callable[_P, _R]: ...

    def __get__(
        self, instance: Optional[object], owner: Type[object]
    ) -> Union[Callable[_P, _R], Callable[_P, SQLCoreOperations[_R]]]:
        if instance is None:
            return self.expr.__get__(owner, owner)  # type: ignore
        else:
            return self.func.__get__(instance, owner)  # type: ignore

    def expression(
        self, expr: Callable[Concatenate[Any, _P], SQLCoreOperations[_R]]
    ) -> hybrid_method[_P, _R]:
        """Provide a modifying decorator that defines a
        SQL-expression producing method."""

        self.expr = expr
        if not self.expr.__doc__:
            self.expr.__doc__ = self.func.__doc__
        return self


def _unwrap_classmethod(meth: _T) -> _T:
    if isinstance(meth, classmethod):
        return meth.__func__  # type: ignore
    else:
        return meth


class hybrid_property(interfaces.InspectionAttrInfo, ORMDescriptor[_T]):
    """A decorator which allows definition of a Python descriptor with both
    instance-level and class-level behavior.

    """

    is_attribute = True
    extension_type = HybridExtensionType.HYBRID_PROPERTY

    __name__: str

    def __init__(
        self,
        fget: _HybridGetterType[_T],
        fset: Optional[_HybridSetterType[_T]] = None,
        fdel: Optional[_HybridDeleterType[_T]] = None,
        expr: Optional[_HybridExprCallableType[_T]] = None,
        custom_comparator: Optional[Comparator[_T]] = None,
        update_expr: Optional[_HybridUpdaterType[_T]] = None,
    ):
        """Create a new :class:`.hybrid_property`.

        Usage is typically via decorator::

            from sqlalchemy.ext.hybrid import hybrid_property


            class SomeClass:
                @hybrid_property
                def value(self):
                    return self._value

                @value.setter
                def value(self, value):
                    self._value = value

        """
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.expr = _unwrap_classmethod(expr)
        self.custom_comparator = _unwrap_classmethod(custom_comparator)
        self.update_expr = _unwrap_classmethod(update_expr)
        util.update_wrapper(self, fget)  # type: ignore[arg-type]

    @overload
    def __get__(self, instance: Any, owner: Literal[None]) -> Self: ...

    @overload
    def __get__(
        self, instance: Literal[None], owner: Type[object]
    ) -> _HybridClassLevelAccessor[_T]: ...

    @overload
    def __get__(self, instance: object, owner: Type[object]) -> _T: ...

    def __get__(
        self, instance: Optional[object], owner: Optional[Type[object]]
    ) -> Union[hybrid_property[_T], _HybridClassLevelAccessor[_T], _T]:
        if owner is None:
            return self
        elif instance is None:
            return self._expr_comparator(owner)
        else:
            return self.fget(instance)

    def __set__(self, instance: object, value: Any) -> None:
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(instance, value)

    def __delete__(self, instance: object) -> None:
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(instance)

    def _copy(self, **kw: Any) -> hybrid_property[_T]:
        defaults = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
        defaults.update(**kw)
        return type(self)(**defaults)

    @property
    def overrides(self) -> Self:
        """Prefix for a method that is overriding an existing attribute.

        The :attr:`.hybrid_property.overrides` accessor just returns
        this hybrid object, which when called at the class level from
        a parent class, will de-reference the "instrumented attribute"
        normally returned at this level, and allow modifying decorators
        like :meth:`.hybrid_property.expression` and
        :meth:`.hybrid_property.comparator`
        to be used without conflicting with the same-named attributes
        normally present on the :class:`.QueryableAttribute`::

            class SuperClass:
                # ...

                @hybrid_property
                def foobar(self):
                    return self._foobar


            class SubClass(SuperClass):
                # ...

                @SuperClass.foobar.overrides.expression
                def foobar(cls):
                    return func.subfoobar(self._foobar)

        .. versionadded:: 1.2

        .. seealso::

            :ref:`hybrid_reuse_subclass`

        """
        return self

    class _InPlace(Generic[_TE]):
        """A builder helper for .hybrid_property.

        .. versionadded:: 2.0.4

        """

        __slots__ = ("attr",)

        def __init__(self, attr: hybrid_property[_TE]):
            self.attr = attr

        def _set(self, **kw: Any) -> hybrid_property[_TE]:
            for k, v in kw.items():
                setattr(self.attr, k, _unwrap_classmethod(v))
            return self.attr

        def getter(self, fget: _HybridGetterType[_TE]) -> hybrid_property[_TE]:
            return self._set(fget=fget)

        def setter(self, fset: _HybridSetterType[_TE]) -> hybrid_property[_TE]:
            return self._set(fset=fset)

        def deleter(
            self, fdel: _HybridDeleterType[_TE]
        ) -> hybrid_property[_TE]:
            return self._set(fdel=fdel)

        def expression(
            self, expr: _HybridExprCallableType[_TE]
        ) -> hybrid_property[_TE]:
            return self._set(expr=expr)

        def comparator(
            self, comparator: _HybridComparatorCallableType[_TE]
        ) -> hybrid_property[_TE]:
            return self._set(custom_comparator=comparator)

        def update_expression(
            self, meth: _HybridUpdaterType[_TE]
        ) -> hybrid_property[_TE]:
            return self._set(update_expr=meth)

    @property
    def inplace(self) -> _InPlace[_T]:
        """Return the inplace mutator for this :class:`.hybrid_property`.

        This is to allow in-place mutation of the hybrid, allowing the first
        hybrid method of a certain name to be re-used in order to add
        more methods without having to name those methods the same, e.g.::

            class Interval(Base):
                # ...

                @hybrid_property
                def radius(self) -> float:
                    return abs(self.length) / 2

                @radius.inplace.setter
                def _radius_setter(self, value: float) -> None:
                    self.length = value * 2

                @radius.inplace.expression
                def _radius_expression(cls) -> ColumnElement[float]:
                    return type_coerce(func.abs(cls.length) / 2, Float)

        .. versionadded:: 2.0.4

        .. seealso::

            :ref:`hybrid_pep484_naming`

        """
        return hybrid_property._InPlace(self)

    def getter(self, fget: _HybridGetterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a getter method.

        .. versionadded:: 1.2

        """

        return self._copy(fget=fget)

    def setter(self, fset: _HybridSetterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a setter method."""

        return self._copy(fset=fset)

    def deleter(self, fdel: _HybridDeleterType[_T]) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a deletion method."""

        return self._copy(fdel=fdel)

    def expression(
        self, expr: _HybridExprCallableType[_T]
    ) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a SQL-expression
        producing method.

        When a hybrid is invoked at the class level, the SQL expression given
        here is wrapped inside of a specialized :class:`.QueryableAttribute`,
        which is the same kind of object used by the ORM to represent other
        mapped attributes.   The reason for this is so that other class-level
        attributes such as docstrings and a reference to the hybrid itself may
        be maintained within the structure that's returned, without any
        modifications to the original SQL expression passed in.

        .. note::

           When referring to a hybrid property  from an owning class (e.g.
           ``SomeClass.some_hybrid``), an instance of
           :class:`.QueryableAttribute` is returned, representing the
           expression or comparator object as well as this  hybrid object.
           However, that object itself has accessors called ``expression`` and
           ``comparator``; so when attempting to override these decorators on a
           subclass, it may be necessary to qualify it using the
           :attr:`.hybrid_property.overrides` modifier first.  See that
           modifier for details.

        .. seealso::

            :ref:`hybrid_distinct_expression`

        """

        return self._copy(expr=expr)

    def comparator(
        self, comparator: _HybridComparatorCallableType[_T]
    ) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines a custom
        comparator producing method.

        The return value of the decorated method should be an instance of
        :class:`~.hybrid.Comparator`.

        .. note::  The :meth:`.hybrid_property.comparator` decorator
           **replaces** the use of the :meth:`.hybrid_property.expression`
           decorator.  They cannot be used together.

        When a hybrid is invoked at the class level, the
        :class:`~.hybrid.Comparator` object given here is wrapped inside of a
        specialized :class:`.QueryableAttribute`, which is the same kind of
        object used by the ORM to represent other mapped attributes.   The
        reason for this is so that other class-level attributes such as
        docstrings and a reference to the hybrid itself may be maintained
        within the structure that's returned, without any modifications to the
        original comparator object passed in.

        .. note::

           When referring to a hybrid property  from an owning class (e.g.
           ``SomeClass.some_hybrid``), an instance of
           :class:`.QueryableAttribute` is returned, representing the
           expression or comparator object as this  hybrid object.  However,
           that object itself has accessors called ``expression`` and
           ``comparator``; so when attempting to override these decorators on a
           subclass, it may be necessary to qualify it using the
           :attr:`.hybrid_property.overrides` modifier first.  See that
           modifier for details.

        """
        return self._copy(custom_comparator=comparator)

    def update_expression(
        self, meth: _HybridUpdaterType[_T]
    ) -> hybrid_property[_T]:
        """Provide a modifying decorator that defines an UPDATE tuple
        producing method.

        The method accepts a single value, which is the value to be
        rendered into the SET clause of an UPDATE statement.  The method
        should then process this value into individual column expressions
        that fit into the ultimate SET clause, and return them as a
        sequence of 2-tuples.  Each tuple
        contains a column expression as the key and a value to be rendered.

        E.g.::

            class Person(Base):
                # ...

                first_name = Column(String)
                last_name = Column(String)

                @hybrid_property
                def fullname(self):
                    return first_name + " " + last_name

                @fullname.update_expression
                def fullname(cls, value):
                    fname, lname = value.split(" ", 1)
                    return [(cls.first_name, fname), (cls.last_name, lname)]

        .. versionadded:: 1.2

        """
        return self._copy(update_expr=meth)

    @util.memoized_property
    def _expr_comparator(
        self,
    ) -> Callable[[Any], _HybridClassLevelAccessor[_T]]:
        if self.custom_comparator is not None:
            return self._get_comparator(self.custom_comparator)
        elif self.expr is not None:
            return self._get_expr(self.expr)
        else:
            return self._get_expr(cast(_HybridExprCallableType[_T], self.fget))

    def _get_expr(
        self, expr: _HybridExprCallableType[_T]
    ) -> Callable[[Any], _HybridClassLevelAccessor[_T]]:
        def _expr(cls: Any) -> ExprComparator[_T]:
            return ExprComparator(cls, expr(cls), self)

        util.update_wrapper(_expr, expr)

        return self._get_comparator(_expr)

    def _get_comparator(
        self, comparator: Any
    ) -> Callable[[Any], _HybridClassLevelAccessor[_T]]:
        proxy_attr = attributes.create_proxied_attribute(self)

        def expr_comparator(
            owner: Type[object],
        ) -> _HybridClassLevelAccessor[_T]:
            # because this is the descriptor protocol, we don't really know
            # what our attribute name is.  so search for it through the
            # MRO.
            for lookup in owner.__mro__:
                if self.__name__ in lookup.__dict__:
                    if lookup.__dict__[self.__name__] is self:
                        name = self.__name__
                        break
            else:
                name = attributes._UNKNOWN_ATTR_KEY  # type: ignore[assignment]

            return cast(
                "_HybridClassLevelAccessor[_T]",
                proxy_attr(
                    owner,
                    name,
                    self,
                    comparator(owner),
                    doc=comparator.__doc__ or self.__doc__,
                ),
            )

        return expr_comparator


class Comparator(interfaces.PropComparator[_T]):
    """A helper class that allows easy construction of custom
    :class:`~.orm.interfaces.PropComparator`
    classes for usage with hybrids."""

    def __init__(
        self, expression: Union[_HasClauseElement[_T], SQLColumnExpression[_T]]
    ):
        self.expression = expression

    def __clause_element__(self) -> roles.ColumnsClauseRole:
        expr = self.expression
        if is_has_clause_element(expr):
            ret_expr = expr.__clause_element__()
        else:
            if TYPE_CHECKING:
                assert isinstance(expr, ColumnElement)
            ret_expr = expr

        if TYPE_CHECKING:
            # see test_hybrid->test_expression_isnt_clause_element
            # that exercises the usual place this is caught if not
            # true
            assert isinstance(ret_expr, ColumnElement)
        return ret_expr

    @util.non_memoized_property
    def property(self) -> interfaces.MapperProperty[_T]:
        raise NotImplementedError()

    def adapt_to_entity(
        self, adapt_to_entity: AliasedInsp[Any]
    ) -> Comparator[_T]:
        # interesting....
        return self


class ExprComparator(Comparator[_T]):
    def __init__(
        self,
        cls: Type[Any],
        expression: Union[_HasClauseElement[_T], SQLColumnExpression[_T]],
        hybrid: hybrid_property[_T],
    ):
        self.cls = cls
        self.expression = expression
        self.hybrid = hybrid

    def __getattr__(self, key: str) -> Any:
        return getattr(self.expression, key)

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        return self.hybrid.info

    def _bulk_update_tuples(
        self, value: Any
    ) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
        if isinstance(self.expression, attributes.QueryableAttribute):
            return self.expression._bulk_update_tuples(value)
        elif self.hybrid.update_expr is not None:
            return self.hybrid.update_expr(self.cls, value)
        else:
            return [(self.expression, value)]

    @util.non_memoized_property
    def property(self) -> MapperProperty[_T]:
        # this accessor is not normally used, however is accessed by things
        # like ORM synonyms if the hybrid is used in this context; the
        # .property attribute is not necessarily accessible
        return self.expression.property  # type: ignore

    def operate(
        self, op: OperatorType, *other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return op(self.expression, *other, **kwargs)

    def reverse_operate(
        self, op: OperatorType, other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return op(other, self.expression, **kwargs)  # type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\__init__.py ===
# Colored terminal output for Python's logging module.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: June 11, 2021
# URL: https://coloredlogs.readthedocs.io

"""
Colored terminal output for Python's :mod:`logging` module.

.. contents::
   :local:

Getting started
===============

The easiest way to get started is by importing :mod:`coloredlogs` and calling
:mod:`coloredlogs.install()` (similar to :func:`logging.basicConfig()`):

 >>> import coloredlogs, logging
 >>> coloredlogs.install(level='DEBUG')
 >>> logger = logging.getLogger('some.module.name')
 >>> logger.info("this is an informational message")
 2015-10-22 19:13:52 peter-macbook some.module.name[28036] INFO this is an informational message

The :mod:`~coloredlogs.install()` function creates a :class:`ColoredFormatter`
that injects `ANSI escape sequences`_ into the log output.

.. _ANSI escape sequences: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors

Environment variables
=====================

The following environment variables can be used to configure the
:mod:`coloredlogs` module without writing any code:

=============================  ============================  ==================================
Environment variable           Default value                 Type of value
=============================  ============================  ==================================
``$COLOREDLOGS_AUTO_INSTALL``  'false'                       a boolean that controls whether
                                                             :func:`auto_install()` is called
``$COLOREDLOGS_LOG_LEVEL``     'INFO'                        a log level name
``$COLOREDLOGS_LOG_FORMAT``    :data:`DEFAULT_LOG_FORMAT`    a log format string
``$COLOREDLOGS_DATE_FORMAT``   :data:`DEFAULT_DATE_FORMAT`   a date/time format string
``$COLOREDLOGS_LEVEL_STYLES``  :data:`DEFAULT_LEVEL_STYLES`  see :func:`parse_encoded_styles()`
``$COLOREDLOGS_FIELD_STYLES``  :data:`DEFAULT_FIELD_STYLES`  see :func:`parse_encoded_styles()`
=============================  ============================  ==================================

If the environment variable `$NO_COLOR`_ is set (the value doesn't matter, even
an empty string will do) then :func:`coloredlogs.install()` will take this as a
hint that colors should not be used (unless the ``isatty=True`` override was
passed by the caller).

.. _$NO_COLOR: https://no-color.org/

Examples of customization
=========================

Here we'll take a look at some examples of how you can customize
:mod:`coloredlogs` using environment variables.

.. contents::
   :local:

About the defaults
------------------

Here's a screen shot of the default configuration for easy comparison with the
screen shots of the following customizations (this is the same screen shot that
is shown in the introduction):

.. image:: images/defaults.png
   :alt: Screen shot of colored logging with defaults.

The screen shot above was taken from ``urxvt`` which doesn't support faint text
colors, otherwise the color of green used for `debug` messages would have
differed slightly from the color of green used for `spam` messages.

Apart from the `faint` style of the `spam` level, the default configuration of
`coloredlogs` sticks to the eight color palette defined by the original ANSI
standard, in order to provide a somewhat consistent experience across terminals
and terminal emulators.

Available text styles and colors
--------------------------------

Of course you are free to customize the default configuration, in this case you
can use any text style or color that you know is supported by your terminal.
You can use the ``humanfriendly --demo`` command to try out the supported text
styles and colors:

.. image:: http://humanfriendly.readthedocs.io/en/latest/_images/ansi-demo.png
   :alt: Screen shot of the 'humanfriendly --demo' command.

Changing the log format
-----------------------

The simplest customization is to change the log format, for example:

.. literalinclude:: examples/custom-log-format.txt
   :language: console

Here's what that looks like in a terminal (I always work in terminals with a
black background and white text):

.. image:: images/custom-log-format.png
   :alt: Screen shot of colored logging with custom log format.

Changing the date/time format
-----------------------------

You can also change the date/time format, for example you can remove the date
part and leave only the time:

.. literalinclude:: examples/custom-datetime-format.txt
   :language: console

Here's what it looks like in a terminal:

.. image:: images/custom-datetime-format.png
   :alt: Screen shot of colored logging with custom date/time format.

Changing the colors/styles
--------------------------

Finally you can customize the colors and text styles that are used:

.. literalinclude:: examples/custom-colors.txt
   :language: console

Here's an explanation of the features used here:

- The numbers used in ``$COLOREDLOGS_LEVEL_STYLES`` demonstrate the use of 256
  color mode (the numbers refer to the 256 color mode palette which is fixed).

- The `success` level demonstrates the use of a text style (bold).

- The `critical` level demonstrates the use of a background color (red).

Of course none of this can be seen in the shell transcript quoted above, but
take a look at the following screen shot:

.. image:: images/custom-colors.png
   :alt: Screen shot of colored logging with custom colors.

.. _notes about log levels:

Some notes about log levels
===========================

With regards to the handling of log levels, the :mod:`coloredlogs` package
differs from Python's :mod:`logging` module in two aspects:

1. While the :mod:`logging` module uses the default logging level
   :data:`logging.WARNING`, the :mod:`coloredlogs` package has always used
   :data:`logging.INFO` as its default log level.

2. When logging to the terminal or system log is initialized by
   :func:`install()` or :func:`.enable_system_logging()` the effective
   level [#]_ of the selected logger [#]_ is compared against the requested
   level [#]_ and if the effective level is more restrictive than the requested
   level, the logger's level will be set to the requested level (this happens
   in :func:`adjust_level()`). The reason for this is to work around a
   combination of design choices in Python's :mod:`logging` module that can
   easily confuse people who aren't already intimately familiar with it:

   - All loggers are initialized with the level :data:`logging.NOTSET`.

   - When a logger's level is set to :data:`logging.NOTSET` the
     :func:`~logging.Logger.getEffectiveLevel()` method will
     fall back to the level of the parent logger.

   - The parent of all loggers is the root logger and the root logger has its
     level set to :data:`logging.WARNING` by default (after importing the
     :mod:`logging` module).

   Effectively all user defined loggers inherit the default log level
   :data:`logging.WARNING` from the root logger, which isn't very intuitive for
   those who aren't already familiar with the hierarchical nature of the
   :mod:`logging` module.

   By avoiding this potentially confusing behavior (see `#14`_, `#18`_, `#21`_,
   `#23`_ and `#24`_), while at the same time allowing the caller to specify a
   logger object, my goal and hope is to provide sane defaults that can easily
   be changed when the need arises.

   .. [#] Refer to :func:`logging.Logger.getEffectiveLevel()` for details.
   .. [#] The logger that is passed as an argument by the caller or the root
          logger which is selected as a default when no logger is provided.
   .. [#] The log level that is passed as an argument by the caller or the
          default log level :data:`logging.INFO` when no level is provided.

   .. _#14: https://github.com/xolox/python-coloredlogs/issues/14
   .. _#18: https://github.com/xolox/python-coloredlogs/issues/18
   .. _#21: https://github.com/xolox/python-coloredlogs/pull/21
   .. _#23: https://github.com/xolox/python-coloredlogs/pull/23
   .. _#24: https://github.com/xolox/python-coloredlogs/issues/24

Classes and functions
=====================
"""

# Standard library modules.
import collections
import logging
import os
import re
import socket
import sys

# External dependencies.
from humanfriendly import coerce_boolean
from humanfriendly.compat import coerce_string, is_string, on_windows
from humanfriendly.terminal import ANSI_COLOR_CODES, ansi_wrap, enable_ansi_support, terminal_supports_colors
from humanfriendly.text import format, split

# Semi-standard module versioning.
__version__ = '15.0.1'

DEFAULT_LOG_LEVEL = logging.INFO
"""The default log level for :mod:`coloredlogs` (:data:`logging.INFO`)."""

DEFAULT_LOG_FORMAT = '%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s'
"""The default log format for :class:`ColoredFormatter` objects (a string)."""

DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
"""The default date/time format for :class:`ColoredFormatter` objects (a string)."""

CHROOT_FILES = ['/etc/debian_chroot']
"""A list of filenames that indicate a chroot and contain the name of the chroot."""

DEFAULT_FIELD_STYLES = dict(
    asctime=dict(color='green'),
    hostname=dict(color='magenta'),
    levelname=dict(color='black', bold=True),
    name=dict(color='blue'),
    programname=dict(color='cyan'),
    username=dict(color='yellow'),
)
"""Mapping of log format names to default font styles."""

DEFAULT_LEVEL_STYLES = dict(
    spam=dict(color='green', faint=True),
    debug=dict(color='green'),
    verbose=dict(color='blue'),
    info=dict(),
    notice=dict(color='magenta'),
    warning=dict(color='yellow'),
    success=dict(color='green', bold=True),
    error=dict(color='red'),
    critical=dict(color='red', bold=True),
)
"""Mapping of log level names to default font styles."""

DEFAULT_FORMAT_STYLE = '%'
"""The default logging format style (a single character)."""

FORMAT_STYLE_PATTERNS = {
    '%': r'%\((\w+)\)[#0 +-]*\d*(?:\.\d+)?[hlL]?[diouxXeEfFgGcrs%]',
    '{': r'{(\w+)[^}]*}',
    '$': r'\$(\w+)|\${(\w+)}',
}
"""
A dictionary that maps the `style` characters ``%``, ``{`` and ``$`` (see the
documentation of the :class:`python3:logging.Formatter` class in Python 3.2+)
to strings containing regular expression patterns that can be used to parse
format strings in the corresponding style:

``%``
 A string containing a regular expression that matches a "percent conversion
 specifier" as defined in the `String Formatting Operations`_ section of the
 Python documentation. Here's an example of a logging format string in this
 format: ``%(levelname)s:%(name)s:%(message)s``.

``{``
 A string containing a regular expression that matches a "replacement field" as
 defined in the `Format String Syntax`_ section of the Python documentation.
 Here's an example of a logging format string in this format:
 ``{levelname}:{name}:{message}``.

``$``
 A string containing a regular expression that matches a "substitution
 placeholder" as defined in the `Template Strings`_ section of the Python
 documentation. Here's an example of a logging format string in this format:
 ``$levelname:$name:$message``.

These regular expressions are used by :class:`FormatStringParser` to introspect
and manipulate logging format strings.

.. _String Formatting Operations: https://docs.python.org/2/library/stdtypes.html#string-formatting
.. _Format String Syntax: https://docs.python.org/2/library/string.html#formatstrings
.. _Template Strings: https://docs.python.org/3/library/string.html#template-strings
"""


def auto_install():
    """
    Automatically call :func:`install()` when ``$COLOREDLOGS_AUTO_INSTALL`` is set.

    The `coloredlogs` package includes a `path configuration file`_ that
    automatically imports the :mod:`coloredlogs` module and calls
    :func:`auto_install()` when the environment variable
    ``$COLOREDLOGS_AUTO_INSTALL`` is set.

    This function uses :func:`~humanfriendly.coerce_boolean()` to check whether
    the value of ``$COLOREDLOGS_AUTO_INSTALL`` should be considered :data:`True`.

    .. _path configuration file: https://docs.python.org/2/library/site.html#module-site
    """
    if coerce_boolean(os.environ.get('COLOREDLOGS_AUTO_INSTALL', 'false')):
        install()


def install(level=None, **kw):
    """
    Enable colored terminal output for Python's :mod:`logging` module.

    :param level: The default logging level (an integer or a string with a
                  level name, defaults to :data:`DEFAULT_LOG_LEVEL`).
    :param logger: The logger to which the stream handler should be attached (a
                   :class:`~logging.Logger` object, defaults to the root logger).
    :param fmt: Set the logging format (a string like those accepted by
                :class:`~logging.Formatter`, defaults to
                :data:`DEFAULT_LOG_FORMAT`).
    :param datefmt: Set the date/time format (a string, defaults to
                    :data:`DEFAULT_DATE_FORMAT`).
    :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                  :data:`DEFAULT_FORMAT_STYLE`). See the documentation of the
                  :class:`python3:logging.Formatter` class in Python 3.2+. On
                  older Python versions only ``%`` is supported.
    :param milliseconds: :data:`True` to show milliseconds like :mod:`logging`
                         does by default, :data:`False` to hide milliseconds
                         (the default is :data:`False`, see `#16`_).
    :param level_styles: A dictionary with custom level styles (defaults to
                         :data:`DEFAULT_LEVEL_STYLES`).
    :param field_styles: A dictionary with custom field styles (defaults to
                         :data:`DEFAULT_FIELD_STYLES`).
    :param stream: The stream where log messages should be written to (a
                   file-like object). This defaults to :data:`None` which
                   means :class:`StandardErrorHandler` is used.
    :param isatty: :data:`True` to use a :class:`ColoredFormatter`,
                   :data:`False` to use a normal :class:`~logging.Formatter`
                   (defaults to auto-detection using
                   :func:`~humanfriendly.terminal.terminal_supports_colors()`).
    :param reconfigure: If :data:`True` (the default) multiple calls to
                        :func:`coloredlogs.install()` will each override
                        the previous configuration.
    :param use_chroot: Refer to :class:`HostNameFilter`.
    :param programname: Refer to :class:`ProgramNameFilter`.
    :param username: Refer to :class:`UserNameFilter`.
    :param syslog: If :data:`True` then :func:`.enable_system_logging()` will
                   be called without arguments (defaults to :data:`False`). The
                   `syslog` argument may also be a number or string, in this
                   case it is assumed to be a logging level which is passed on
                   to :func:`.enable_system_logging()`.

    The :func:`coloredlogs.install()` function is similar to
    :func:`logging.basicConfig()`, both functions take a lot of optional
    keyword arguments but try to do the right thing by default:

    1. If `reconfigure` is :data:`True` (it is by default) and an existing
       :class:`~logging.StreamHandler` is found that is connected to either
       :data:`~sys.stdout` or :data:`~sys.stderr` the handler will be removed.
       This means that first calling :func:`logging.basicConfig()` and then
       calling :func:`coloredlogs.install()` will replace the stream handler
       instead of adding a duplicate stream handler. If `reconfigure` is
       :data:`False` and an existing handler is found no further steps are
       taken (to avoid installing a duplicate stream handler).

    2. A :class:`~logging.StreamHandler` is created and connected to the stream
       given by the `stream` keyword argument (:data:`sys.stderr` by
       default). The stream handler's level is set to the value of the `level`
       keyword argument.

    3. A :class:`ColoredFormatter` is created if the `isatty` keyword argument
       allows it (or auto-detection allows it), otherwise a normal
       :class:`~logging.Formatter` is created. The formatter is initialized
       with the `fmt` and `datefmt` keyword arguments (or their computed
       defaults).

       The environment variable ``$NO_COLOR`` is taken as a hint by
       auto-detection that colors should not be used.

    4. :func:`HostNameFilter.install()`, :func:`ProgramNameFilter.install()`
       and :func:`UserNameFilter.install()` are called to enable the use of
       additional fields in the log format.

    5. If the logger's level is too restrictive it is relaxed (refer to `notes
       about log levels`_ for details).

    6. The formatter is added to the handler and the handler is added to the
       logger.

    .. _#16: https://github.com/xolox/python-coloredlogs/issues/16
    """
    logger = kw.get('logger') or logging.getLogger()
    reconfigure = kw.get('reconfigure', True)
    stream = kw.get('stream') or sys.stderr
    style = check_style(kw.get('style') or DEFAULT_FORMAT_STYLE)
    # Get the log level from an argument, environment variable or default and
    # convert the names of log levels to numbers to enable numeric comparison.
    if level is None:
        level = os.environ.get('COLOREDLOGS_LOG_LEVEL', DEFAULT_LOG_LEVEL)
    level = level_to_number(level)
    # Remove any existing stream handler that writes to stdout or stderr, even
    # if the stream handler wasn't created by coloredlogs because multiple
    # stream handlers (in the same hierarchy) writing to stdout or stderr would
    # create duplicate output.  `None' is a synonym for the possibly dynamic
    # value of the stderr attribute of the sys module.
    match_streams = ([sys.stdout, sys.stderr]
                     if stream in [sys.stdout, sys.stderr, None]
                     else [stream])
    match_handler = lambda handler: match_stream_handler(handler, match_streams)
    handler, logger = replace_handler(logger, match_handler, reconfigure)
    # Make sure reconfiguration is allowed or not relevant.
    if not (handler and not reconfigure):
        # Make it easy to enable system logging.
        syslog_enabled = kw.get('syslog')
        # We ignore the value `None' because it means the caller didn't opt in
        # to system logging and `False' because it means the caller explicitly
        # opted out of system logging.
        if syslog_enabled not in (None, False):
            from coloredlogs.syslog import enable_system_logging
            if syslog_enabled is True:
                # If the caller passed syslog=True then we leave the choice of
                # default log level up to the coloredlogs.syslog module.
                enable_system_logging()
            else:
                # Values other than (None, True, False) are assumed to
                # represent a logging level for system logging.
                enable_system_logging(level=syslog_enabled)
        # Figure out whether we can use ANSI escape sequences.
        use_colors = kw.get('isatty', None)
        # In the following indented block the expression (use_colors is None)
        # can be read as "auto detect is enabled and no reason has yet been
        # found to automatically disable color support".
        if use_colors or (use_colors is None):
            # Respect the user's choice not to have colors.
            if use_colors is None and 'NO_COLOR' in os.environ:
                # For details on this see https://no-color.org/.
                use_colors = False
            # Try to enable Windows native ANSI support or Colorama?
            if (use_colors or use_colors is None) and on_windows():
                # This can fail, in which case ANSI escape sequences would end
                # up being printed to the terminal in raw form. This is very
                # user hostile, so to avoid this happening we disable color
                # support on failure.
                use_colors = enable_ansi_support()
            # When auto detection is enabled, and so far we encountered no
            # reason to disable color support, then we will enable color
            # support if 'stream' is connected to a terminal.
            if use_colors is None:
                use_colors = terminal_supports_colors(stream)
        # Create a stream handler and make sure to preserve any filters
        # the current handler may have (if an existing handler is found).
        filters = handler.filters if handler else None
        if stream is sys.stderr:
            handler = StandardErrorHandler()
        else:
            handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        if filters:
            handler.filters = filters
        # Prepare the arguments to the formatter, allowing the caller to
        # customize the values of `fmt', `datefmt' and `style' as desired.
        formatter_options = dict(fmt=kw.get('fmt'), datefmt=kw.get('datefmt'))
        # Only pass the `style' argument to the formatter when the caller
        # provided an alternative logging format style. This prevents
        # TypeError exceptions on Python versions before 3.2.
        if style != DEFAULT_FORMAT_STYLE:
            formatter_options['style'] = style
        # Come up with a default log format?
        if not formatter_options['fmt']:
            # Use the log format defined by the environment variable
            # $COLOREDLOGS_LOG_FORMAT or fall back to the default.
            formatter_options['fmt'] = os.environ.get('COLOREDLOGS_LOG_FORMAT') or DEFAULT_LOG_FORMAT
        # If the caller didn't specify a date/time format we'll use the format
        # defined by the environment variable $COLOREDLOGS_DATE_FORMAT (or fall
        # back to the default).
        if not formatter_options['datefmt']:
            formatter_options['datefmt'] = os.environ.get('COLOREDLOGS_DATE_FORMAT') or DEFAULT_DATE_FORMAT
        # Python's logging module shows milliseconds by default through special
        # handling in the logging.Formatter.formatTime() method [1]. Because
        # coloredlogs always defines a `datefmt' it bypasses this special
        # handling, which is fine because ever since publishing coloredlogs
        # I've never needed millisecond precision ;-). However there are users
        # of coloredlogs that do want milliseconds to be shown [2] so we
        # provide a shortcut to make it easy.
        #
        # [1] https://stackoverflow.com/questions/6290739/python-logging-use-milliseconds-in-time-format
        # [2] https://github.com/xolox/python-coloredlogs/issues/16
        if kw.get('milliseconds'):
            parser = FormatStringParser(style=style)
            if not (parser.contains_field(formatter_options['fmt'], 'msecs')
                    or '%f' in formatter_options['datefmt']):
                pattern = parser.get_pattern('asctime')
                replacements = {'%': '%(msecs)03d', '{': '{msecs:03}', '$': '${msecs}'}
                formatter_options['fmt'] = pattern.sub(
                    r'\g<0>,' + replacements[style],
                    formatter_options['fmt'],
                )
        # Do we need to make %(hostname) available to the formatter?
        HostNameFilter.install(
            fmt=formatter_options['fmt'],
            handler=handler,
            style=style,
            use_chroot=kw.get('use_chroot', True),
        )
        # Do we need to make %(programname) available to the formatter?
        ProgramNameFilter.install(
            fmt=formatter_options['fmt'],
            handler=handler,
            programname=kw.get('programname'),
            style=style,
        )
        # Do we need to make %(username) available to the formatter?
        UserNameFilter.install(
            fmt=formatter_options['fmt'],
            handler=handler,
            username=kw.get('username'),
            style=style,
        )
        # Inject additional formatter arguments specific to ColoredFormatter?
        if use_colors:
            for name, environment_name in (('field_styles', 'COLOREDLOGS_FIELD_STYLES'),
                                           ('level_styles', 'COLOREDLOGS_LEVEL_STYLES')):
                value = kw.get(name)
                if value is None:
                    # If no styles have been specified we'll fall back
                    # to the styles defined by the environment variable.
                    environment_value = os.environ.get(environment_name)
                    if environment_value is not None:
                        value = parse_encoded_styles(environment_value)
                if value is not None:
                    formatter_options[name] = value
        # Create a (possibly colored) formatter.
        formatter_type = ColoredFormatter if use_colors else BasicFormatter
        handler.setFormatter(formatter_type(**formatter_options))
        # Adjust the level of the selected logger.
        adjust_level(logger, level)
        # Install the stream handler.
        logger.addHandler(handler)


def check_style(value):
    """
    Validate a logging format style.

    :param value: The logging format style to validate (any value).
    :returns: The logging format character (a string of one character).
    :raises: :exc:`~exceptions.ValueError` when the given style isn't supported.

    On Python 3.2+ this function accepts the logging format styles ``%``, ``{``
    and ``$`` while on older versions only ``%`` is accepted (because older
    Python versions don't support alternative logging format styles).
    """
    if sys.version_info[:2] >= (3, 2):
        if value not in FORMAT_STYLE_PATTERNS:
            msg = "Unsupported logging format style! (%r)"
            raise ValueError(format(msg, value))
    elif value != DEFAULT_FORMAT_STYLE:
        msg = "Format string styles other than %r require Python 3.2+!"
        raise ValueError(msg, DEFAULT_FORMAT_STYLE)
    return value


def increase_verbosity():
    """
    Increase the verbosity of the root handler by one defined level.

    Understands custom logging levels like defined by my ``verboselogs``
    module.
    """
    defined_levels = sorted(set(find_defined_levels().values()))
    current_index = defined_levels.index(get_level())
    selected_index = max(0, current_index - 1)
    set_level(defined_levels[selected_index])


def decrease_verbosity():
    """
    Decrease the verbosity of the root handler by one defined level.

    Understands custom logging levels like defined by my ``verboselogs``
    module.
    """
    defined_levels = sorted(set(find_defined_levels().values()))
    current_index = defined_levels.index(get_level())
    selected_index = min(current_index + 1, len(defined_levels) - 1)
    set_level(defined_levels[selected_index])


def is_verbose():
    """
    Check whether the log level of the root handler is set to a verbose level.

    :returns: ``True`` if the root handler is verbose, ``False`` if not.
    """
    return get_level() < DEFAULT_LOG_LEVEL


def get_level():
    """
    Get the logging level of the root handler.

    :returns: The logging level of the root handler (an integer) or
              :data:`DEFAULT_LOG_LEVEL` (if no root handler exists).
    """
    handler, logger = find_handler(logging.getLogger(), match_stream_handler)
    return handler.level if handler else DEFAULT_LOG_LEVEL


def set_level(level):
    """
    Set the logging level of the root handler.

    :param level: The logging level to filter on (an integer or string).

    If no root handler exists yet this automatically calls :func:`install()`.
    """
    handler, logger = find_handler(logging.getLogger(), match_stream_handler)
    if handler and logger:
        # Change the level of the existing handler.
        handler.setLevel(level_to_number(level))
        # Adjust the level of the selected logger.
        adjust_level(logger, level)
    else:
        # Create a new handler with the given level.
        install(level=level)


def adjust_level(logger, level):
    """
    Increase a logger's verbosity up to the requested level.

    :param logger: The logger to change (a :class:`~logging.Logger` object).
    :param level: The log level to enable (a string or number).

    This function is used by functions like :func:`install()`,
    :func:`increase_verbosity()` and :func:`.enable_system_logging()` to adjust
    a logger's level so that log messages up to the requested log level are
    propagated to the configured output handler(s).

    It uses :func:`logging.Logger.getEffectiveLevel()` to check whether
    `logger` propagates or swallows log messages of the requested `level` and
    sets the logger's level to the requested level if it would otherwise
    swallow log messages.

    Effectively this function will "widen the scope of logging" when asked to
    do so but it will never "narrow the scope of logging". This is because I am
    convinced that filtering of log messages should (primarily) be decided by
    handlers.
    """
    level = level_to_number(level)
    if logger.getEffectiveLevel() > level:
        logger.setLevel(level)


def find_defined_levels():
    """
    Find the defined logging levels.

    :returns: A dictionary with level names as keys and integers as values.

    Here's what the result looks like by default (when
    no custom levels or level names have been defined):

    >>> find_defined_levels()
    {'NOTSET': 0,
     'DEBUG': 10,
     'INFO': 20,
     'WARN': 30,
     'WARNING': 30,
     'ERROR': 40,
     'FATAL': 50,
     'CRITICAL': 50}
    """
    defined_levels = {}
    for name in dir(logging):
        if name.isupper():
            value = getattr(logging, name)
            if isinstance(value, int):
                defined_levels[name] = value
    return defined_levels


def level_to_number(value):
    """
    Coerce a logging level name to a number.

    :param value: A logging level (integer or string).
    :returns: The number of the log level (an integer).

    This function translates log level names into their numeric values..
    """
    if is_string(value):
        try:
            defined_levels = find_defined_levels()
            value = defined_levels[value.upper()]
        except KeyError:
            # Don't fail on unsupported log levels.
            value = DEFAULT_LOG_LEVEL
    return value


def find_level_aliases():
    """
    Find log level names which are aliases of each other.

    :returns: A dictionary that maps aliases to their canonical name.

    .. note:: Canonical names are chosen to be the alias with the longest
              string length so that e.g. ``WARN`` is an alias for ``WARNING``
              instead of the other way around.

    Here's what the result looks like by default (when
    no custom levels or level names have been defined):

    >>> from coloredlogs import find_level_aliases
    >>> find_level_aliases()
    {'WARN': 'WARNING', 'FATAL': 'CRITICAL'}
    """
    mapping = collections.defaultdict(list)
    for name, value in find_defined_levels().items():
        mapping[value].append(name)
    aliases = {}
    for value, names in mapping.items():
        if len(names) > 1:
            names = sorted(names, key=lambda n: len(n))
            canonical_name = names.pop()
            for alias in names:
                aliases[alias] = canonical_name
    return aliases


def parse_encoded_styles(text, normalize_key=None):
    """
    Parse text styles encoded in a string into a nested data structure.

    :param text: The encoded styles (a string).
    :returns: A dictionary in the structure of the :data:`DEFAULT_FIELD_STYLES`
              and :data:`DEFAULT_LEVEL_STYLES` dictionaries.

    Here's an example of how this function works:

    >>> from coloredlogs import parse_encoded_styles
    >>> from pprint import pprint
    >>> encoded_styles = 'debug=green;warning=yellow;error=red;critical=red,bold'
    >>> pprint(parse_encoded_styles(encoded_styles))
    {'debug': {'color': 'green'},
     'warning': {'color': 'yellow'},
     'error': {'color': 'red'},
     'critical': {'bold': True, 'color': 'red'}}
    """
    parsed_styles = {}
    for assignment in split(text, ';'):
        name, _, styles = assignment.partition('=')
        target = parsed_styles.setdefault(name, {})
        for token in split(styles, ','):
            # When this code was originally written, setting background colors
            # wasn't supported yet, so there was no need to disambiguate
            # between the text color and background color. This explains why
            # a color name or number implies setting the text color (for
            # backwards compatibility).
            if token.isdigit():
                target['color'] = int(token)
            elif token in ANSI_COLOR_CODES:
                target['color'] = token
            elif '=' in token:
                name, _, value = token.partition('=')
                if name in ('color', 'background'):
                    if value.isdigit():
                        target[name] = int(value)
                    elif value in ANSI_COLOR_CODES:
                        target[name] = value
            else:
                target[token] = True
    return parsed_styles


def find_hostname(use_chroot=True):
    """
    Find the host name to include in log messages.

    :param use_chroot: Use the name of the chroot when inside a chroot?
                       (boolean, defaults to :data:`True`)
    :returns: A suitable host name (a string).

    Looks for :data:`CHROOT_FILES` that have a nonempty first line (taken to be
    the chroot name). If none are found then :func:`socket.gethostname()` is
    used as a fall back.
    """
    for chroot_file in CHROOT_FILES:
        try:
            with open(chroot_file) as handle:
                first_line = next(handle)
                name = first_line.strip()
                if name:
                    return name
        except Exception:
            pass
    return socket.gethostname()


def find_program_name():
    """
    Select a suitable program name to embed in log messages.

    :returns: One of the following strings (in decreasing order of preference):

              1. The base name of the currently running Python program or
                 script (based on the value at index zero of :data:`sys.argv`).
              2. The base name of the Python executable (based on
                 :data:`sys.executable`).
              3. The string 'python'.
    """
    # Gotcha: sys.argv[0] is '-c' if Python is started with the -c option.
    return ((os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] != '-c' else '')
            or (os.path.basename(sys.executable) if sys.executable else '')
            or 'python')


def find_username():
    """
    Find the username to include in log messages.

    :returns: A suitable username (a string).

    On UNIX systems this uses the :mod:`pwd` module which means ``root`` will
    be reported when :man:`sudo` is used (as it should). If this fails (for
    example on Windows) then :func:`getpass.getuser()` is used as a fall back.
    """
    try:
        import pwd
        uid = os.getuid()
        entry = pwd.getpwuid(uid)
        return entry.pw_name
    except Exception:
        import getpass
        return getpass.getuser()


def replace_handler(logger, match_handler, reconfigure):
    """
    Prepare to replace a handler.

    :param logger: Refer to :func:`find_handler()`.
    :param match_handler: Refer to :func:`find_handler()`.
    :param reconfigure: :data:`True` if an existing handler should be replaced,
                        :data:`False` otherwise.
    :returns: A tuple of two values:

              1. The matched :class:`~logging.Handler` object or :data:`None`
                 if no handler was matched.
              2. The :class:`~logging.Logger` to which the matched handler was
                 attached or the logger given to :func:`replace_handler()`.
    """
    handler, other_logger = find_handler(logger, match_handler)
    if handler and other_logger and reconfigure:
        # Remove the existing handler from the logger that its attached to
        # so that we can install a new handler that behaves differently.
        other_logger.removeHandler(handler)
        # Switch to the logger that the existing handler was attached to so
        # that reconfiguration doesn't narrow the scope of logging.
        logger = other_logger
    return handler, logger


def find_handler(logger, match_handler):
    """
    Find a (specific type of) handler in the propagation tree of a logger.

    :param logger: The logger to check (a :class:`~logging.Logger` object).
    :param match_handler: A callable that receives a :class:`~logging.Handler`
                          object and returns :data:`True` to match a handler or
                          :data:`False` to skip that handler and continue
                          searching for a match.
    :returns: A tuple of two values:

              1. The matched :class:`~logging.Handler` object or :data:`None`
                 if no handler was matched.
              2. The :class:`~logging.Logger` object to which the handler is
                 attached or :data:`None` if no handler was matched.

    This function finds a logging handler (of the given type) attached to a
    logger or one of its parents (see :func:`walk_propagation_tree()`). It uses
    the undocumented :class:`~logging.Logger.handlers` attribute to find
    handlers attached to a logger, however it won't raise an exception if the
    attribute isn't available. The advantages of this approach are:

    - This works regardless of whether :mod:`coloredlogs` attached the handler
      or other Python code attached the handler.

    - This will correctly recognize the situation where the given logger has no
      handlers but :attr:`~logging.Logger.propagate` is enabled and the logger
      has a parent logger that does have a handler attached.
    """
    for logger in walk_propagation_tree(logger):
        for handler in getattr(logger, 'handlers', []):
            if match_handler(handler):
                return handler, logger
    return None, None


def match_stream_handler(handler, streams=[]):
    """
    Identify stream handlers writing to the given streams(s).

    :param handler: The :class:`~logging.Handler` class to check.
    :param streams: A sequence of streams to match (defaults to matching
                    :data:`~sys.stdout` and :data:`~sys.stderr`).
    :returns: :data:`True` if the handler is a :class:`~logging.StreamHandler`
              logging to the given stream(s), :data:`False` otherwise.

    This function can be used as a callback for :func:`find_handler()`.
    """
    return (isinstance(handler, logging.StreamHandler)
            and getattr(handler, 'stream') in (streams or (sys.stdout, sys.stderr)))


def walk_propagation_tree(logger):
    """
    Walk through the propagation hierarchy of the given logger.

    :param logger: The logger whose hierarchy to walk (a
                   :class:`~logging.Logger` object).
    :returns: A generator of :class:`~logging.Logger` objects.

    .. note:: This uses the undocumented :class:`logging.Logger.parent`
              attribute to find higher level loggers, however it won't
              raise an exception if the attribute isn't available.
    """
    while isinstance(logger, logging.Logger):
        # Yield the logger to our caller.
        yield logger
        # Check if the logger has propagation enabled.
        if logger.propagate:
            # Continue with the parent logger. We use getattr() because the
            # `parent' attribute isn't documented so properly speaking we
            # shouldn't break if it's not available.
            logger = getattr(logger, 'parent', None)
        else:
            # The propagation chain stops here.
            logger = None


class BasicFormatter(logging.Formatter):

    """
    Log :class:`~logging.Formatter` that supports ``%f`` for millisecond formatting.

    This class extends :class:`~logging.Formatter` to enable the use of ``%f``
    for millisecond formatting in date/time strings, to allow for the type of
    flexibility requested in issue `#45`_.

    .. _#45: https://github.com/xolox/python-coloredlogs/issues/45
    """

    def formatTime(self, record, datefmt=None):
        """
        Format the date/time of a log record.

        :param record: A :class:`~logging.LogRecord` object.
        :param datefmt: A date/time format string (defaults to :data:`DEFAULT_DATE_FORMAT`).
        :returns: The formatted date/time (a string).

        This method overrides :func:`~logging.Formatter.formatTime()` to set
        `datefmt` to :data:`DEFAULT_DATE_FORMAT` when the caller hasn't
        specified a date format.

        When `datefmt` contains the token ``%f`` it will be replaced by the
        value of ``%(msecs)03d`` (refer to issue `#45`_ for use cases).
        """
        # The default value of the following argument is defined here so
        # that Sphinx doesn't embed the default value in the generated
        # documentation (because the result is awkward to read).
        datefmt = datefmt or DEFAULT_DATE_FORMAT
        # Replace %f with the value of %(msecs)03d.
        if '%f' in datefmt:
            datefmt = datefmt.replace('%f', '%03d' % record.msecs)
        # Delegate the actual date/time formatting to the base formatter.
        return logging.Formatter.formatTime(self, record, datefmt)


class ColoredFormatter(BasicFormatter):

    """
    Log :class:`~logging.Formatter` that uses `ANSI escape sequences`_ to create colored logs.

    :class:`ColoredFormatter` inherits from :class:`BasicFormatter` to enable
    the use of ``%f`` for millisecond formatting in date/time strings.

    .. note:: If you want to use :class:`ColoredFormatter` on Windows then you
              need to call :func:`~humanfriendly.terminal.enable_ansi_support()`.
              This is done for you when you call :func:`coloredlogs.install()`.
    """

    def __init__(self, fmt=None, datefmt=None, style=DEFAULT_FORMAT_STYLE, level_styles=None, field_styles=None):
        """
        Initialize a :class:`ColoredFormatter` object.

        :param fmt: A log format string (defaults to :data:`DEFAULT_LOG_FORMAT`).
        :param datefmt: A date/time format string (defaults to :data:`None`,
                        but see the documentation of
                        :func:`BasicFormatter.formatTime()`).
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`)
        :param level_styles: A dictionary with custom level styles
                             (defaults to :data:`DEFAULT_LEVEL_STYLES`).
        :param field_styles: A dictionary with custom field styles
                             (defaults to :data:`DEFAULT_FIELD_STYLES`).
        :raises: Refer to :func:`check_style()`.

        This initializer uses :func:`colorize_format()` to inject ANSI escape
        sequences in the log format string before it is passed to the
        initializer of the base class.
        """
        self.nn = NameNormalizer()
        # The default values of the following arguments are defined here so
        # that Sphinx doesn't embed the default values in the generated
        # documentation (because the result is awkward to read).
        fmt = fmt or DEFAULT_LOG_FORMAT
        self.level_styles = self.nn.normalize_keys(DEFAULT_LEVEL_STYLES if level_styles is None else level_styles)
        self.field_styles = self.nn.normalize_keys(DEFAULT_FIELD_STYLES if field_styles is None else field_styles)
        # Rewrite the format string to inject ANSI escape sequences.
        kw = dict(fmt=self.colorize_format(fmt, style), datefmt=datefmt)
        # If we were given a non-default logging format style we pass it on
        # to our superclass. At this point check_style() will have already
        # complained that the use of alternative logging format styles
        # requires Python 3.2 or newer.
        if style != DEFAULT_FORMAT_STYLE:
            kw['style'] = style
        # Initialize the superclass with the rewritten format string.
        logging.Formatter.__init__(self, **kw)

    def colorize_format(self, fmt, style=DEFAULT_FORMAT_STYLE):
        """
        Rewrite a logging format string to inject ANSI escape sequences.

        :param fmt: The log format string.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :returns: The logging format string with ANSI escape sequences.

        This method takes a logging format string like the ones you give to
        :class:`logging.Formatter` and processes it as follows:

        1. First the logging format string is separated into formatting
           directives versus surrounding text (according to the given `style`).

        2. Then formatting directives and surrounding text are grouped
           based on whitespace delimiters (in the surrounding text).

        3. For each group styling is selected as follows:

           1. If the group contains a single formatting directive that has
              a style defined then the whole group is styled accordingly.

           2. If the group contains multiple formatting directives that
              have styles defined then each formatting directive is styled
              individually and surrounding text isn't styled.

        As an example consider the default log format (:data:`DEFAULT_LOG_FORMAT`)::

         %(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s

        The default field styles (:data:`DEFAULT_FIELD_STYLES`) define a style for the
        `name` field but not for the `process` field, however because both fields
        are part of the same whitespace delimited token they'll be highlighted
        together in the style defined for the `name` field.
        """
        result = []
        parser = FormatStringParser(style=style)
        for group in parser.get_grouped_pairs(fmt):
            applicable_styles = [self.nn.get(self.field_styles, token.name) for token in group if token.name]
            if sum(map(bool, applicable_styles)) == 1:
                # If exactly one (1) field style is available for the group of
                # tokens then all of the tokens will be styled the same way.
                # This provides a limited form of backwards compatibility with
                # the (intended) behavior of coloredlogs before the release of
                # version 10.
                result.append(ansi_wrap(
                    ''.join(token.text for token in group),
                    **next(s for s in applicable_styles if s)
                ))
            else:
                for token in group:
                    text = token.text
                    if token.name:
                        field_styles = self.nn.get(self.field_styles, token.name)
                        if field_styles:
                            text = ansi_wrap(text, **field_styles)
                    result.append(text)
        return ''.join(result)

    def format(self, record):
        """
        Apply level-specific styling to log records.

        :param record: A :class:`~logging.LogRecord` object.
        :returns: The result of :func:`logging.Formatter.format()`.

        This method injects ANSI escape sequences that are specific to the
        level of each log record (because such logic cannot be expressed in the
        syntax of a log format string). It works by making a copy of the log
        record, changing the `msg` field inside the copy and passing the copy
        into the :func:`~logging.Formatter.format()` method of the base
        class.
        """
        style = self.nn.get(self.level_styles, record.levelname)
        # After the introduction of the `Empty' class it was reported in issue
        # 33 that format() can be called when `Empty' has already been garbage
        # collected. This explains the (otherwise rather out of place) `Empty
        # is not None' check in the following `if' statement. The reasoning
        # here is that it's much better to log a message without formatting
        # then to raise an exception ;-).
        #
        # For more details refer to issue 33 on GitHub:
        # https://github.com/xolox/python-coloredlogs/issues/33
        if style and Empty is not None:
            # Due to the way that Python's logging module is structured and
            # documented the only (IMHO) clean way to customize its behavior is
            # to change incoming LogRecord objects before they get to the base
            # formatter. However we don't want to break other formatters and
            # handlers, so we copy the log record.
            #
            # In the past this used copy.copy() but as reported in issue 29
            # (which is reproducible) this can cause deadlocks. The following
            # Python voodoo is intended to accomplish the same thing as
            # copy.copy() without all of the generalization and overhead that
            # we don't need for our -very limited- use case.
            #
            # For more details refer to issue 29 on GitHub:
            # https://github.com/xolox/python-coloredlogs/issues/29
            copy = Empty()
            copy.__class__ = record.__class__
            copy.__dict__.update(record.__dict__)
            copy.msg = ansi_wrap(coerce_string(record.msg), **style)
            record = copy
        # Delegate the remaining formatting to the base formatter.
        return logging.Formatter.format(self, record)


class Empty(object):
    """An empty class used to copy :class:`~logging.LogRecord` objects without reinitializing them."""


class HostNameFilter(logging.Filter):

    """
    Log filter to enable the ``%(hostname)s`` format.

    Python's :mod:`logging` module doesn't expose the system's host name while
    I consider this to be a valuable addition. Fortunately it's very easy to
    expose additional fields in format strings: :func:`filter()` simply sets
    the ``hostname`` attribute of each :class:`~logging.LogRecord` object it
    receives and this is enough to enable the use of the ``%(hostname)s``
    expression in format strings.

    You can install this log filter as follows::

     >>> import coloredlogs, logging
     >>> handler = logging.StreamHandler()
     >>> handler.addFilter(coloredlogs.HostNameFilter())
     >>> handler.setFormatter(logging.Formatter('[%(hostname)s] %(message)s'))
     >>> logger = logging.getLogger()
     >>> logger.addHandler(handler)
     >>> logger.setLevel(logging.INFO)
     >>> logger.info("Does it work?")
     [peter-macbook] Does it work?

    Of course :func:`coloredlogs.install()` does all of this for you :-).
    """

    @classmethod
    def install(cls, handler, fmt=None, use_chroot=True, style=DEFAULT_FORMAT_STYLE):
        """
        Install the :class:`HostNameFilter` on a log handler (only if needed).

        :param fmt: The log format string to check for ``%(hostname)``.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :param handler: The logging handler on which to install the filter.
        :param use_chroot: Refer to :func:`find_hostname()`.

        If `fmt` is given the filter will only be installed if `fmt` uses the
        ``hostname`` field. If `fmt` is not given the filter is installed
        unconditionally.
        """
        if fmt:
            parser = FormatStringParser(style=style)
            if not parser.contains_field(fmt, 'hostname'):
                return
        handler.addFilter(cls(use_chroot))

    def __init__(self, use_chroot=True):
        """
        Initialize a :class:`HostNameFilter` object.

        :param use_chroot: Refer to :func:`find_hostname()`.
        """
        self.hostname = find_hostname(use_chroot)

    def filter(self, record):
        """Set each :class:`~logging.LogRecord`'s `hostname` field."""
        # Modify the record.
        record.hostname = self.hostname
        # Don't filter the record.
        return 1


class ProgramNameFilter(logging.Filter):

    """
    Log filter to enable the ``%(programname)s`` format.

    Python's :mod:`logging` module doesn't expose the name of the currently
    running program while I consider this to be a useful addition. Fortunately
    it's very easy to expose additional fields in format strings:
    :func:`filter()` simply sets the ``programname`` attribute of each
    :class:`~logging.LogRecord` object it receives and this is enough to enable
    the use of the ``%(programname)s`` expression in format strings.

    Refer to :class:`HostNameFilter` for an example of how to manually install
    these log filters.
    """

    @classmethod
    def install(cls, handler, fmt, programname=None, style=DEFAULT_FORMAT_STYLE):
        """
        Install the :class:`ProgramNameFilter` (only if needed).

        :param fmt: The log format string to check for ``%(programname)``.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :param handler: The logging handler on which to install the filter.
        :param programname: Refer to :func:`__init__()`.

        If `fmt` is given the filter will only be installed if `fmt` uses the
        ``programname`` field. If `fmt` is not given the filter is installed
        unconditionally.
        """
        if fmt:
            parser = FormatStringParser(style=style)
            if not parser.contains_field(fmt, 'programname'):
                return
        handler.addFilter(cls(programname))

    def __init__(self, programname=None):
        """
        Initialize a :class:`ProgramNameFilter` object.

        :param programname: The program name to use (defaults to the result of
                            :func:`find_program_name()`).
        """
        self.programname = programname or find_program_name()

    def filter(self, record):
        """Set each :class:`~logging.LogRecord`'s `programname` field."""
        # Modify the record.
        record.programname = self.programname
        # Don't filter the record.
        return 1


class UserNameFilter(logging.Filter):

    """
    Log filter to enable the ``%(username)s`` format.

    Python's :mod:`logging` module doesn't expose the username of the currently
    logged in user as requested in `#76`_. Given that :class:`HostNameFilter`
    and :class:`ProgramNameFilter` are already provided by `coloredlogs` it
    made sense to provide :class:`UserNameFilter` as well.

    Refer to :class:`HostNameFilter` for an example of how to manually install
    these log filters.

    .. _#76: https://github.com/xolox/python-coloredlogs/issues/76
    """

    @classmethod
    def install(cls, handler, fmt, username=None, style=DEFAULT_FORMAT_STYLE):
        """
        Install the :class:`UserNameFilter` (only if needed).

        :param fmt: The log format string to check for ``%(username)``.
        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :param handler: The logging handler on which to install the filter.
        :param username: Refer to :func:`__init__()`.

        If `fmt` is given the filter will only be installed if `fmt` uses the
        ``username`` field. If `fmt` is not given the filter is installed
        unconditionally.
        """
        if fmt:
            parser = FormatStringParser(style=style)
            if not parser.contains_field(fmt, 'username'):
                return
        handler.addFilter(cls(username))

    def __init__(self, username=None):
        """
        Initialize a :class:`UserNameFilter` object.

        :param username: The username to use (defaults to the
                         result of :func:`find_username()`).
        """
        self.username = username or find_username()

    def filter(self, record):
        """Set each :class:`~logging.LogRecord`'s `username` field."""
        # Modify the record.
        record.username = self.username
        # Don't filter the record.
        return 1


class StandardErrorHandler(logging.StreamHandler):

    """
    A :class:`~logging.StreamHandler` that gets the value of :data:`sys.stderr` for each log message.

    The :class:`StandardErrorHandler` class enables `monkey patching of
    sys.stderr <https://github.com/xolox/python-coloredlogs/pull/31>`_. It's
    basically the same as the ``logging._StderrHandler`` class present in
    Python 3 but it will be available regardless of Python version. This
    handler is used by :func:`coloredlogs.install()` to improve compatibility
    with the Python standard library.
    """

    def __init__(self, level=logging.NOTSET):
        """Initialize a :class:`StandardErrorHandler` object."""
        logging.Handler.__init__(self, level)

    @property
    def stream(self):
        """Get the value of :data:`sys.stderr` (a file-like object)."""
        return sys.stderr


class FormatStringParser(object):

    """
    Shallow logging format string parser.

    This class enables introspection and manipulation of logging format strings
    in the three styles supported by the :mod:`logging` module starting from
    Python 3.2 (``%``, ``{`` and ``$``).
    """

    def __init__(self, style=DEFAULT_FORMAT_STYLE):
        """
        Initialize a :class:`FormatStringParser` object.

        :param style: One of the characters ``%``, ``{`` or ``$`` (defaults to
                      :data:`DEFAULT_FORMAT_STYLE`).
        :raises: Refer to :func:`check_style()`.
        """
        self.style = check_style(style)
        self.capturing_pattern = FORMAT_STYLE_PATTERNS[style]
        # Remove the capture group around the mapping key / field name.
        self.raw_pattern = self.capturing_pattern.replace(r'(\w+)', r'\w+')
        # After removing the inner capture group we add an outer capture group
        # to make the pattern suitable for simple tokenization using re.split().
        self.tokenize_pattern = re.compile('(%s)' % self.raw_pattern, re.VERBOSE)
        # Compile a regular expression for finding field names.
        self.name_pattern = re.compile(self.capturing_pattern, re.VERBOSE)

    def contains_field(self, format_string, field_name):
        """
        Get the field names referenced by a format string.

        :param format_string: The logging format string.
        :returns: A list of strings with field names.
        """
        return field_name in self.get_field_names(format_string)

    def get_field_names(self, format_string):
        """
        Get the field names referenced by a format string.

        :param format_string: The logging format string.
        :returns: A list of strings with field names.
        """
        return self.name_pattern.findall(format_string)

    def get_grouped_pairs(self, format_string):
        """
        Group the results of :func:`get_pairs()` separated by whitespace.

        :param format_string: The logging format string.
        :returns: A list of lists of :class:`FormatStringToken` objects.
        """
        # Step 1: Split simple tokens (without a name) into
        # their whitespace parts and non-whitespace parts.
        separated = []
        pattern = re.compile(r'(\s+)')
        for token in self.get_pairs(format_string):
            if token.name:
                separated.append(token)
            else:
                separated.extend(
                    FormatStringToken(name=None, text=text)
                    for text in pattern.split(token.text) if text
                )
        # Step 2: Group tokens together based on whitespace.
        current_group = []
        grouped_pairs = []
        for token in separated:
            if token.text.isspace():
                if current_group:
                    grouped_pairs.append(current_group)
                grouped_pairs.append([token])
                current_group = []
            else:
                current_group.append(token)
        if current_group:
            grouped_pairs.append(current_group)
        return grouped_pairs

    def get_pairs(self, format_string):
        """
        Tokenize a logging format string and extract field names from tokens.

        :param format_string: The logging format string.
        :returns: A generator of :class:`FormatStringToken` objects.
        """
        for token in self.get_tokens(format_string):
            match = self.name_pattern.search(token)
            name = match.group(1) if match else None
            yield FormatStringToken(name=name, text=token)

    def get_pattern(self, field_name):
        """
        Get a regular expression to match a formatting directive that references the given field name.

        :param field_name: The name of the field to match (a string).
        :returns: A compiled regular expression object.
        """
        return re.compile(self.raw_pattern.replace(r'\w+', field_name), re.VERBOSE)

    def get_tokens(self, format_string):
        """
        Tokenize a logging format string.

        :param format_string: The logging format string.
        :returns: A list of strings with formatting directives separated from surrounding text.
        """
        return [t for t in self.tokenize_pattern.split(format_string) if t]


class FormatStringToken(collections.namedtuple('FormatStringToken', 'text, name')):

    """
    A named tuple for the results of :func:`FormatStringParser.get_pairs()`.

    .. attribute:: name

       The field name referenced in `text` (a string). If `text` doesn't
       contain a formatting directive this will be :data:`None`.

    .. attribute:: text

       The text extracted from the logging format string (a string).
    """


class NameNormalizer(object):

    """Responsible for normalizing field and level names."""

    def __init__(self):
        """Initialize a :class:`NameNormalizer` object."""
        self.aliases = {k.lower(): v.lower() for k, v in find_level_aliases().items()}

    def normalize_name(self, name):
        """
        Normalize a field or level name.

        :param name: The field or level name (a string).
        :returns: The normalized name (a string).

        Transforms all strings to lowercase and resolves level name aliases
        (refer to :func:`find_level_aliases()`) to their canonical name:

        >>> from coloredlogs import NameNormalizer
        >>> from humanfriendly import format_table
        >>> nn = NameNormalizer()
        >>> sample_names = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL']
        >>> print(format_table([(n, nn.normalize_name(n)) for n in sample_names]))
        -----------------------
        | DEBUG    | debug    |
        | INFO     | info     |
        | WARN     | warning  |
        | WARNING  | warning  |
        | ERROR    | error    |
        | FATAL    | critical |
        | CRITICAL | critical |
        -----------------------
        """
        name = name.lower()
        if name in self.aliases:
            name = self.aliases[name]
        return name

    def normalize_keys(self, value):
        """
        Normalize the keys of a dictionary using :func:`normalize_name()`.

        :param value: The dictionary to normalize.
        :returns: A dictionary with normalized keys.
        """
        return {self.normalize_name(k): v for k, v in value.items()}

    def get(self, normalized_dict, name):
        """
        Get a value from a dictionary after normalizing the key.

        :param normalized_dict: A dictionary produced by :func:`normalize_keys()`.
        :param name: A key to normalize and get from the dictionary.
        :returns: The value of the normalized key (if any).
        """
        return normalized_dict.get(self.normalize_name(name))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\descriptor.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Descriptors essentially contain exactly the information found in a .proto
file, in types that make this information accessible in Python.
"""

__author__ = 'robinson@google.com (Will Robinson)'

import abc
import binascii
import os
import threading
import warnings

from google.protobuf.internal import api_implementation

_USE_C_DESCRIPTORS = False
if api_implementation.Type() != 'python':
  # pylint: disable=protected-access
  _message = api_implementation._c_module
  # TODO: Remove this import after fix api_implementation
  if _message is None:
    from google.protobuf.pyext import _message
  _USE_C_DESCRIPTORS = True


class Error(Exception):
  """Base error for this module."""


class TypeTransformationError(Error):
  """Error transforming between python proto type and corresponding C++ type."""


if _USE_C_DESCRIPTORS:
  # This metaclass allows to override the behavior of code like
  #     isinstance(my_descriptor, FieldDescriptor)
  # and make it return True when the descriptor is an instance of the extension
  # type written in C++.
  class DescriptorMetaclass(type):

    def __instancecheck__(cls, obj):
      if super(DescriptorMetaclass, cls).__instancecheck__(obj):
        return True
      if isinstance(obj, cls._C_DESCRIPTOR_CLASS):
        return True
      return False
else:
  # The standard metaclass; nothing changes.
  DescriptorMetaclass = abc.ABCMeta


class _Lock(object):
  """Wrapper class of threading.Lock(), which is allowed by 'with'."""

  def __new__(cls):
    self = object.__new__(cls)
    self._lock = threading.Lock()  # pylint: disable=protected-access
    return self

  def __enter__(self):
    self._lock.acquire()

  def __exit__(self, exc_type, exc_value, exc_tb):
    self._lock.release()


_lock = threading.Lock()


def _Deprecated(name):
  if _Deprecated.count > 0:
    _Deprecated.count -= 1
    warnings.warn(
        'Call to deprecated create function %s(). Note: Create unlinked '
        'descriptors is going to go away. Please use get/find descriptors from '
        'generated code or query the descriptor_pool.'
        % name,
        category=DeprecationWarning, stacklevel=3)

# These must match the values in descriptor.proto, but we can't use them
# directly because we sometimes need to reference them in feature helpers
# below *during* the build of descriptor.proto.
_FEATURESET_MESSAGE_ENCODING_DELIMITED = 2
_FEATURESET_FIELD_PRESENCE_IMPLICIT = 2
_FEATURESET_FIELD_PRESENCE_LEGACY_REQUIRED = 3
_FEATURESET_REPEATED_FIELD_ENCODING_PACKED = 1
_FEATURESET_ENUM_TYPE_CLOSED = 2

# Deprecated warnings will print 100 times at most which should be enough for
# users to notice and do not cause timeout.
_Deprecated.count = 100


_internal_create_key = object()


class DescriptorBase(metaclass=DescriptorMetaclass):

  """Descriptors base class.

  This class is the base of all descriptor classes. It provides common options
  related functionality.

  Attributes:
    has_options:  True if the descriptor has non-default options.  Usually it is
      not necessary to read this -- just call GetOptions() which will happily
      return the default instance.  However, it's sometimes useful for
      efficiency, and also useful inside the protobuf implementation to avoid
      some bootstrapping issues.
    file (FileDescriptor): Reference to file info.
  """

  if _USE_C_DESCRIPTORS:
    # The class, or tuple of classes, that are considered as "virtual
    # subclasses" of this descriptor class.
    _C_DESCRIPTOR_CLASS = ()

  def __init__(self, file, options, serialized_options, options_class_name):
    """Initialize the descriptor given its options message and the name of the
    class of the options message. The name of the class is required in case
    the options message is None and has to be created.
    """
    self._features = None
    self.file = file
    self._options = options
    self._loaded_options = None
    self._options_class_name = options_class_name
    self._serialized_options = serialized_options

    # Does this descriptor have non-default options?
    self.has_options = (self._options is not None) or (
        self._serialized_options is not None
    )

  @property
  @abc.abstractmethod
  def _parent(self):
    pass

  def _InferLegacyFeatures(self, edition, options, features):
    """Infers features from proto2/proto3 syntax so that editions logic can be used everywhere.

    Args:
      edition: The edition to infer features for.
      options: The options for this descriptor that are being processed.
      features: The feature set object to modify with inferred features.
    """
    pass

  def _GetFeatures(self):
    if not self._features:
      self._LazyLoadOptions()
    return self._features

  def _ResolveFeatures(self, edition, raw_options):
    """Resolves features from the raw options of this descriptor.

    Args:
      edition: The edition to use for feature defaults.
      raw_options: The options for this descriptor that are being processed.

    Returns:
      A fully resolved feature set for making runtime decisions.
    """
    # pylint: disable=g-import-not-at-top
    from google.protobuf import descriptor_pb2

    if self._parent:
      features = descriptor_pb2.FeatureSet()
      features.CopyFrom(self._parent._GetFeatures())
    else:
      features = self.file.pool._CreateDefaultFeatures(edition)
    unresolved = descriptor_pb2.FeatureSet()
    unresolved.CopyFrom(raw_options.features)
    self._InferLegacyFeatures(edition, raw_options, unresolved)
    features.MergeFrom(unresolved)

    # Use the feature cache to reduce memory bloat.
    return self.file.pool._InternFeatures(features)

  def _LazyLoadOptions(self):
    """Lazily initializes descriptor options towards the end of the build."""
    if self._loaded_options:
      return

    # pylint: disable=g-import-not-at-top
    from google.protobuf import descriptor_pb2

    if not hasattr(descriptor_pb2, self._options_class_name):
      raise RuntimeError(
          'Unknown options class name %s!' % self._options_class_name
      )
    options_class = getattr(descriptor_pb2, self._options_class_name)
    features = None
    edition = self.file._edition

    if not self.has_options:
      if not self._features:
        features = self._ResolveFeatures(
            descriptor_pb2.Edition.Value(edition), options_class()
        )
      with _lock:
        self._loaded_options = options_class()
        if not self._features:
          self._features = features
    else:
      if not self._serialized_options:
        options = self._options
      else:
        options = _ParseOptions(options_class(), self._serialized_options)

      if not self._features:
        features = self._ResolveFeatures(
            descriptor_pb2.Edition.Value(edition), options
        )
      with _lock:
        self._loaded_options = options
        if not self._features:
          self._features = features
        if options.HasField('features'):
          options.ClearField('features')
          if not options.SerializeToString():
            self._loaded_options = options_class()
            self.has_options = False

  def GetOptions(self):
    """Retrieves descriptor options.

    Returns:
      The options set on this descriptor.
    """
    if not self._loaded_options:
      self._LazyLoadOptions()
    return self._loaded_options


class _NestedDescriptorBase(DescriptorBase):
  """Common class for descriptors that can be nested."""

  def __init__(self, options, options_class_name, name, full_name,
               file, containing_type, serialized_start=None,
               serialized_end=None, serialized_options=None):
    """Constructor.

    Args:
      options: Protocol message options or None to use default message options.
      options_class_name (str): The class name of the above options.
      name (str): Name of this protocol message type.
      full_name (str): Fully-qualified name of this protocol message type, which
        will include protocol "package" name and the name of any enclosing
        types.
      containing_type: if provided, this is a nested descriptor, with this
        descriptor as parent, otherwise None.
      serialized_start: The start index (inclusive) in block in the
        file.serialized_pb that describes this descriptor.
      serialized_end: The end index (exclusive) in block in the
        file.serialized_pb that describes this descriptor.
      serialized_options: Protocol message serialized options or None.
    """
    super(_NestedDescriptorBase, self).__init__(
        file, options, serialized_options, options_class_name
    )

    self.name = name
    # TODO: Add function to calculate full_name instead of having it in
    #             memory?
    self.full_name = full_name
    self.containing_type = containing_type

    self._serialized_start = serialized_start
    self._serialized_end = serialized_end

  def CopyToProto(self, proto):
    """Copies this to the matching proto in descriptor_pb2.

    Args:
      proto: An empty proto instance from descriptor_pb2.

    Raises:
      Error: If self couldn't be serialized, due to to few constructor
        arguments.
    """
    if (self.file is not None and
        self._serialized_start is not None and
        self._serialized_end is not None):
      proto.ParseFromString(self.file.serialized_pb[
          self._serialized_start:self._serialized_end])
    else:
      raise Error('Descriptor does not contain serialization.')


class Descriptor(_NestedDescriptorBase):

  """Descriptor for a protocol message type.

  Attributes:
      name (str): Name of this protocol message type.
      full_name (str): Fully-qualified name of this protocol message type,
          which will include protocol "package" name and the name of any
          enclosing types.
      containing_type (Descriptor): Reference to the descriptor of the type
          containing us, or None if this is top-level.
      fields (list[FieldDescriptor]): Field descriptors for all fields in
          this type.
      fields_by_number (dict(int, FieldDescriptor)): Same
          :class:`FieldDescriptor` objects as in :attr:`fields`, but indexed
          by "number" attribute in each FieldDescriptor.
      fields_by_name (dict(str, FieldDescriptor)): Same
          :class:`FieldDescriptor` objects as in :attr:`fields`, but indexed by
          "name" attribute in each :class:`FieldDescriptor`.
      nested_types (list[Descriptor]): Descriptor references
          for all protocol message types nested within this one.
      nested_types_by_name (dict(str, Descriptor)): Same Descriptor
          objects as in :attr:`nested_types`, but indexed by "name" attribute
          in each Descriptor.
      enum_types (list[EnumDescriptor]): :class:`EnumDescriptor` references
          for all enums contained within this type.
      enum_types_by_name (dict(str, EnumDescriptor)): Same
          :class:`EnumDescriptor` objects as in :attr:`enum_types`, but
          indexed by "name" attribute in each EnumDescriptor.
      enum_values_by_name (dict(str, EnumValueDescriptor)): Dict mapping
          from enum value name to :class:`EnumValueDescriptor` for that value.
      extensions (list[FieldDescriptor]): All extensions defined directly
          within this message type (NOT within a nested type).
      extensions_by_name (dict(str, FieldDescriptor)): Same FieldDescriptor
          objects as :attr:`extensions`, but indexed by "name" attribute of each
          FieldDescriptor.
      is_extendable (bool):  Does this type define any extension ranges?
      oneofs (list[OneofDescriptor]): The list of descriptors for oneof fields
          in this message.
      oneofs_by_name (dict(str, OneofDescriptor)): Same objects as in
          :attr:`oneofs`, but indexed by "name" attribute.
      file (FileDescriptor): Reference to file descriptor.
      is_map_entry: If the message type is a map entry.

  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.Descriptor

    def __new__(
        cls,
        name=None,
        full_name=None,
        filename=None,
        containing_type=None,
        fields=None,
        nested_types=None,
        enum_types=None,
        extensions=None,
        options=None,
        serialized_options=None,
        is_extendable=True,
        extension_ranges=None,
        oneofs=None,
        file=None,  # pylint: disable=redefined-builtin
        serialized_start=None,
        serialized_end=None,
        syntax=None,
        is_map_entry=False,
        create_key=None):
      _message.Message._CheckCalledFromGeneratedFile()
      return _message.default_pool.FindMessageTypeByName(full_name)

  # NOTE: The file argument redefining a builtin is nothing we can
  # fix right now since we don't know how many clients already rely on the
  # name of the argument.
  def __init__(self, name, full_name, filename, containing_type, fields,
               nested_types, enum_types, extensions, options=None,
               serialized_options=None,
               is_extendable=True, extension_ranges=None, oneofs=None,
               file=None, serialized_start=None, serialized_end=None,  # pylint: disable=redefined-builtin
               syntax=None, is_map_entry=False, create_key=None):
    """Arguments to __init__() are as described in the description
    of Descriptor fields above.

    Note that filename is an obsolete argument, that is not used anymore.
    Please use file.name to access this as an attribute.
    """
    if create_key is not _internal_create_key:
      _Deprecated('Descriptor')

    super(Descriptor, self).__init__(
        options, 'MessageOptions', name, full_name, file,
        containing_type, serialized_start=serialized_start,
        serialized_end=serialized_end, serialized_options=serialized_options)

    # We have fields in addition to fields_by_name and fields_by_number,
    # so that:
    #   1. Clients can index fields by "order in which they're listed."
    #   2. Clients can easily iterate over all fields with the terse
    #      syntax: for f in descriptor.fields: ...
    self.fields = fields
    for field in self.fields:
      field.containing_type = self
      field.file = file
    self.fields_by_number = dict((f.number, f) for f in fields)
    self.fields_by_name = dict((f.name, f) for f in fields)
    self._fields_by_camelcase_name = None

    self.nested_types = nested_types
    for nested_type in nested_types:
      nested_type.containing_type = self
    self.nested_types_by_name = dict((t.name, t) for t in nested_types)

    self.enum_types = enum_types
    for enum_type in self.enum_types:
      enum_type.containing_type = self
    self.enum_types_by_name = dict((t.name, t) for t in enum_types)
    self.enum_values_by_name = dict(
        (v.name, v) for t in enum_types for v in t.values)

    self.extensions = extensions
    for extension in self.extensions:
      extension.extension_scope = self
    self.extensions_by_name = dict((f.name, f) for f in extensions)
    self.is_extendable = is_extendable
    self.extension_ranges = extension_ranges
    self.oneofs = oneofs if oneofs is not None else []
    self.oneofs_by_name = dict((o.name, o) for o in self.oneofs)
    for oneof in self.oneofs:
      oneof.containing_type = self
      oneof.file = file
    self._is_map_entry = is_map_entry

  @property
  def _parent(self):
    return self.containing_type or self.file

  @property
  def fields_by_camelcase_name(self):
    """Same FieldDescriptor objects as in :attr:`fields`, but indexed by
    :attr:`FieldDescriptor.camelcase_name`.
    """
    if self._fields_by_camelcase_name is None:
      self._fields_by_camelcase_name = dict(
          (f.camelcase_name, f) for f in self.fields)
    return self._fields_by_camelcase_name

  def EnumValueName(self, enum, value):
    """Returns the string name of an enum value.

    This is just a small helper method to simplify a common operation.

    Args:
      enum: string name of the Enum.
      value: int, value of the enum.

    Returns:
      string name of the enum value.

    Raises:
      KeyError if either the Enum doesn't exist or the value is not a valid
        value for the enum.
    """
    return self.enum_types_by_name[enum].values_by_number[value].name

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.DescriptorProto.

    Args:
      proto: An empty descriptor_pb2.DescriptorProto.
    """
    # This function is overridden to give a better doc comment.
    super(Descriptor, self).CopyToProto(proto)


# TODO: We should have aggressive checking here,
# for example:
#   * If you specify a repeated field, you should not be allowed
#     to specify a default value.
#   * [Other examples here as needed].
#
# TODO: for this and other *Descriptor classes, we
# might also want to lock things down aggressively (e.g.,
# prevent clients from setting the attributes).  Having
# stronger invariants here in general will reduce the number
# of runtime checks we must do in reflection.py...
class FieldDescriptor(DescriptorBase):

  """Descriptor for a single field in a .proto file.

  Attributes:
    name (str): Name of this field, exactly as it appears in .proto.
    full_name (str): Name of this field, including containing scope.  This is
      particularly relevant for extensions.
    index (int): Dense, 0-indexed index giving the order that this
      field textually appears within its message in the .proto file.
    number (int): Tag number declared for this field in the .proto file.

    type (int): (One of the TYPE_* constants below) Declared type.
    cpp_type (int): (One of the CPPTYPE_* constants below) C++ type used to
      represent this field.

    label (int): (One of the LABEL_* constants below) Tells whether this
      field is optional, required, or repeated.
    has_default_value (bool): True if this field has a default value defined,
      otherwise false.
    default_value (Varies): Default value of this field.  Only
      meaningful for non-repeated scalar fields.  Repeated fields
      should always set this to [], and non-repeated composite
      fields should always set this to None.

    containing_type (Descriptor): Descriptor of the protocol message
      type that contains this field.  Set by the Descriptor constructor
      if we're passed into one.
      Somewhat confusingly, for extension fields, this is the
      descriptor of the EXTENDED message, not the descriptor
      of the message containing this field.  (See is_extension and
      extension_scope below).
    message_type (Descriptor): If a composite field, a descriptor
      of the message type contained in this field.  Otherwise, this is None.
    enum_type (EnumDescriptor): If this field contains an enum, a
      descriptor of that enum.  Otherwise, this is None.

    is_extension: True iff this describes an extension field.
    extension_scope (Descriptor): Only meaningful if is_extension is True.
      Gives the message that immediately contains this extension field.
      Will be None iff we're a top-level (file-level) extension field.

    options (descriptor_pb2.FieldOptions): Protocol message field options or
      None to use default field options.

    containing_oneof (OneofDescriptor): If the field is a member of a oneof
      union, contains its descriptor. Otherwise, None.

    file (FileDescriptor): Reference to file descriptor.
  """

  # Must be consistent with C++ FieldDescriptor::Type enum in
  # descriptor.h.
  #
  # TODO: Find a way to eliminate this repetition.
  TYPE_DOUBLE         = 1
  TYPE_FLOAT          = 2
  TYPE_INT64          = 3
  TYPE_UINT64         = 4
  TYPE_INT32          = 5
  TYPE_FIXED64        = 6
  TYPE_FIXED32        = 7
  TYPE_BOOL           = 8
  TYPE_STRING         = 9
  TYPE_GROUP          = 10
  TYPE_MESSAGE        = 11
  TYPE_BYTES          = 12
  TYPE_UINT32         = 13
  TYPE_ENUM           = 14
  TYPE_SFIXED32       = 15
  TYPE_SFIXED64       = 16
  TYPE_SINT32         = 17
  TYPE_SINT64         = 18
  MAX_TYPE            = 18

  # Must be consistent with C++ FieldDescriptor::CppType enum in
  # descriptor.h.
  #
  # TODO: Find a way to eliminate this repetition.
  CPPTYPE_INT32       = 1
  CPPTYPE_INT64       = 2
  CPPTYPE_UINT32      = 3
  CPPTYPE_UINT64      = 4
  CPPTYPE_DOUBLE      = 5
  CPPTYPE_FLOAT       = 6
  CPPTYPE_BOOL        = 7
  CPPTYPE_ENUM        = 8
  CPPTYPE_STRING      = 9
  CPPTYPE_MESSAGE     = 10
  MAX_CPPTYPE         = 10

  _PYTHON_TO_CPP_PROTO_TYPE_MAP = {
      TYPE_DOUBLE: CPPTYPE_DOUBLE,
      TYPE_FLOAT: CPPTYPE_FLOAT,
      TYPE_ENUM: CPPTYPE_ENUM,
      TYPE_INT64: CPPTYPE_INT64,
      TYPE_SINT64: CPPTYPE_INT64,
      TYPE_SFIXED64: CPPTYPE_INT64,
      TYPE_UINT64: CPPTYPE_UINT64,
      TYPE_FIXED64: CPPTYPE_UINT64,
      TYPE_INT32: CPPTYPE_INT32,
      TYPE_SFIXED32: CPPTYPE_INT32,
      TYPE_SINT32: CPPTYPE_INT32,
      TYPE_UINT32: CPPTYPE_UINT32,
      TYPE_FIXED32: CPPTYPE_UINT32,
      TYPE_BYTES: CPPTYPE_STRING,
      TYPE_STRING: CPPTYPE_STRING,
      TYPE_BOOL: CPPTYPE_BOOL,
      TYPE_MESSAGE: CPPTYPE_MESSAGE,
      TYPE_GROUP: CPPTYPE_MESSAGE
      }

  # Must be consistent with C++ FieldDescriptor::Label enum in
  # descriptor.h.
  #
  # TODO: Find a way to eliminate this repetition.
  LABEL_OPTIONAL      = 1
  LABEL_REQUIRED      = 2
  LABEL_REPEATED      = 3
  MAX_LABEL           = 3

  # Must be consistent with C++ constants kMaxNumber, kFirstReservedNumber,
  # and kLastReservedNumber in descriptor.h
  MAX_FIELD_NUMBER = (1 << 29) - 1
  FIRST_RESERVED_FIELD_NUMBER = 19000
  LAST_RESERVED_FIELD_NUMBER = 19999

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.FieldDescriptor

    def __new__(cls, name, full_name, index, number, type, cpp_type, label,
                default_value, message_type, enum_type, containing_type,
                is_extension, extension_scope, options=None,
                serialized_options=None,
                has_default_value=True, containing_oneof=None, json_name=None,
                file=None, create_key=None):  # pylint: disable=redefined-builtin
      _message.Message._CheckCalledFromGeneratedFile()
      if is_extension:
        return _message.default_pool.FindExtensionByName(full_name)
      else:
        return _message.default_pool.FindFieldByName(full_name)

  def __init__(self, name, full_name, index, number, type, cpp_type, label,
               default_value, message_type, enum_type, containing_type,
               is_extension, extension_scope, options=None,
               serialized_options=None,
               has_default_value=True, containing_oneof=None, json_name=None,
               file=None, create_key=None):  # pylint: disable=redefined-builtin
    """The arguments are as described in the description of FieldDescriptor
    attributes above.

    Note that containing_type may be None, and may be set later if necessary
    (to deal with circular references between message types, for example).
    Likewise for extension_scope.
    """
    if create_key is not _internal_create_key:
      _Deprecated('FieldDescriptor')

    super(FieldDescriptor, self).__init__(
        file, options, serialized_options, 'FieldOptions'
    )
    self.name = name
    self.full_name = full_name
    self._camelcase_name = None
    if json_name is None:
      self.json_name = _ToJsonName(name)
    else:
      self.json_name = json_name
    self.index = index
    self.number = number
    self._type = type
    self.cpp_type = cpp_type
    self._label = label
    self.has_default_value = has_default_value
    self.default_value = default_value
    self.containing_type = containing_type
    self.message_type = message_type
    self.enum_type = enum_type
    self.is_extension = is_extension
    self.extension_scope = extension_scope
    self.containing_oneof = containing_oneof
    if api_implementation.Type() == 'python':
      self._cdescriptor = None
    else:
      if is_extension:
        self._cdescriptor = _message.default_pool.FindExtensionByName(full_name)
      else:
        self._cdescriptor = _message.default_pool.FindFieldByName(full_name)

  @property
  def _parent(self):
    if self.containing_oneof:
      return self.containing_oneof
    if self.is_extension:
      return self.extension_scope or self.file
    return self.containing_type

  def _InferLegacyFeatures(self, edition, options, features):
    # pylint: disable=g-import-not-at-top
    from google.protobuf import descriptor_pb2

    if edition >= descriptor_pb2.Edition.EDITION_2023:
      return

    if self._label == FieldDescriptor.LABEL_REQUIRED:
      features.field_presence = (
          descriptor_pb2.FeatureSet.FieldPresence.LEGACY_REQUIRED
      )

    if self._type == FieldDescriptor.TYPE_GROUP:
      features.message_encoding = (
          descriptor_pb2.FeatureSet.MessageEncoding.DELIMITED
      )

    if options.HasField('packed'):
      features.repeated_field_encoding = (
          descriptor_pb2.FeatureSet.RepeatedFieldEncoding.PACKED
          if options.packed
          else descriptor_pb2.FeatureSet.RepeatedFieldEncoding.EXPANDED
      )

  @property
  def type(self):
    if (
        self._GetFeatures().message_encoding
        == _FEATURESET_MESSAGE_ENCODING_DELIMITED
        and self.message_type
        and not self.message_type.GetOptions().map_entry
        and not self.containing_type.GetOptions().map_entry
    ):
      return FieldDescriptor.TYPE_GROUP
    return self._type

  @type.setter
  def type(self, val):
    self._type = val

  @property
  def label(self):
    if (
        self._GetFeatures().field_presence
        == _FEATURESET_FIELD_PRESENCE_LEGACY_REQUIRED
    ):
      return FieldDescriptor.LABEL_REQUIRED
    return self._label

  @property
  def is_required(self):
    """Returns if the field is required."""
    return (
        self._GetFeatures().field_presence
        == _FEATURESET_FIELD_PRESENCE_LEGACY_REQUIRED
    )

  @property
  def is_repeated(self):
    """Returns if the field is repeated."""
    return self._label == FieldDescriptor.LABEL_REPEATED

  @property
  def camelcase_name(self):
    """Camelcase name of this field.

    Returns:
      str: the name in CamelCase.
    """
    if self._camelcase_name is None:
      self._camelcase_name = _ToCamelCase(self.name)
    return self._camelcase_name

  @property
  def has_presence(self):
    """Whether the field distinguishes between unpopulated and default values.

    Raises:
      RuntimeError: singular field that is not linked with message nor file.
    """
    if self.is_repeated:
      return False
    if (
        self.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE
        or self.is_extension
        or self.containing_oneof
    ):
      return True

    return (
        self._GetFeatures().field_presence
        != _FEATURESET_FIELD_PRESENCE_IMPLICIT
    )

  @property
  def is_packed(self):
    """Returns if the field is packed."""
    if not self.is_repeated:
      return False
    field_type = self.type
    if (field_type == FieldDescriptor.TYPE_STRING or
        field_type == FieldDescriptor.TYPE_GROUP or
        field_type == FieldDescriptor.TYPE_MESSAGE or
        field_type == FieldDescriptor.TYPE_BYTES):
      return False

    return (
        self._GetFeatures().repeated_field_encoding
        == _FEATURESET_REPEATED_FIELD_ENCODING_PACKED
    )

  @staticmethod
  def ProtoTypeToCppProtoType(proto_type):
    """Converts from a Python proto type to a C++ Proto Type.

    The Python ProtocolBuffer classes specify both the 'Python' datatype and the
    'C++' datatype - and they're not the same. This helper method should
    translate from one to another.

    Args:
      proto_type: the Python proto type (descriptor.FieldDescriptor.TYPE_*)
    Returns:
      int: descriptor.FieldDescriptor.CPPTYPE_*, the C++ type.
    Raises:
      TypeTransformationError: when the Python proto type isn't known.
    """
    try:
      return FieldDescriptor._PYTHON_TO_CPP_PROTO_TYPE_MAP[proto_type]
    except KeyError:
      raise TypeTransformationError('Unknown proto_type: %s' % proto_type)


class EnumDescriptor(_NestedDescriptorBase):

  """Descriptor for an enum defined in a .proto file.

  Attributes:
    name (str): Name of the enum type.
    full_name (str): Full name of the type, including package name
      and any enclosing type(s).

    values (list[EnumValueDescriptor]): List of the values
      in this enum.
    values_by_name (dict(str, EnumValueDescriptor)): Same as :attr:`values`,
      but indexed by the "name" field of each EnumValueDescriptor.
    values_by_number (dict(int, EnumValueDescriptor)): Same as :attr:`values`,
      but indexed by the "number" field of each EnumValueDescriptor.
    containing_type (Descriptor): Descriptor of the immediate containing
      type of this enum, or None if this is an enum defined at the
      top level in a .proto file.  Set by Descriptor's constructor
      if we're passed into one.
    file (FileDescriptor): Reference to file descriptor.
    options (descriptor_pb2.EnumOptions): Enum options message or
      None to use default enum options.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.EnumDescriptor

    def __new__(cls, name, full_name, filename, values,
                containing_type=None, options=None,
                serialized_options=None, file=None,  # pylint: disable=redefined-builtin
                serialized_start=None, serialized_end=None, create_key=None):
      _message.Message._CheckCalledFromGeneratedFile()
      return _message.default_pool.FindEnumTypeByName(full_name)

  def __init__(self, name, full_name, filename, values,
               containing_type=None, options=None,
               serialized_options=None, file=None,  # pylint: disable=redefined-builtin
               serialized_start=None, serialized_end=None, create_key=None):
    """Arguments are as described in the attribute description above.

    Note that filename is an obsolete argument, that is not used anymore.
    Please use file.name to access this as an attribute.
    """
    if create_key is not _internal_create_key:
      _Deprecated('EnumDescriptor')

    super(EnumDescriptor, self).__init__(
        options, 'EnumOptions', name, full_name, file,
        containing_type, serialized_start=serialized_start,
        serialized_end=serialized_end, serialized_options=serialized_options)

    self.values = values
    for value in self.values:
      value.file = file
      value.type = self
    self.values_by_name = dict((v.name, v) for v in values)
    # Values are reversed to ensure that the first alias is retained.
    self.values_by_number = dict((v.number, v) for v in reversed(values))

  @property
  def _parent(self):
    return self.containing_type or self.file

  @property
  def is_closed(self):
    """Returns true whether this is a "closed" enum.

    This means that it:
    - Has a fixed set of values, rather than being equivalent to an int32.
    - Encountering values not in this set causes them to be treated as unknown
      fields.
    - The first value (i.e., the default) may be nonzero.

    WARNING: Some runtimes currently have a quirk where non-closed enums are
    treated as closed when used as the type of fields defined in a
    `syntax = proto2;` file. This quirk is not present in all runtimes; as of
    writing, we know that:

    - C++, Java, and C++-based Python share this quirk.
    - UPB and UPB-based Python do not.
    - PHP and Ruby treat all enums as open regardless of declaration.

    Care should be taken when using this function to respect the target
    runtime's enum handling quirks.
    """
    return self._GetFeatures().enum_type == _FEATURESET_ENUM_TYPE_CLOSED

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.EnumDescriptorProto.

    Args:
      proto (descriptor_pb2.EnumDescriptorProto): An empty descriptor proto.
    """
    # This function is overridden to give a better doc comment.
    super(EnumDescriptor, self).CopyToProto(proto)


class EnumValueDescriptor(DescriptorBase):

  """Descriptor for a single value within an enum.

  Attributes:
    name (str): Name of this value.
    index (int): Dense, 0-indexed index giving the order that this
      value appears textually within its enum in the .proto file.
    number (int): Actual number assigned to this enum value.
    type (EnumDescriptor): :class:`EnumDescriptor` to which this value
      belongs.  Set by :class:`EnumDescriptor`'s constructor if we're
      passed into one.
    options (descriptor_pb2.EnumValueOptions): Enum value options message or
      None to use default enum value options options.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.EnumValueDescriptor

    def __new__(cls, name, index, number,
                type=None,  # pylint: disable=redefined-builtin
                options=None, serialized_options=None, create_key=None):
      _message.Message._CheckCalledFromGeneratedFile()
      # There is no way we can build a complete EnumValueDescriptor with the
      # given parameters (the name of the Enum is not known, for example).
      # Fortunately generated files just pass it to the EnumDescriptor()
      # constructor, which will ignore it, so returning None is good enough.
      return None

  def __init__(self, name, index, number,
               type=None,  # pylint: disable=redefined-builtin
               options=None, serialized_options=None, create_key=None):
    """Arguments are as described in the attribute description above."""
    if create_key is not _internal_create_key:
      _Deprecated('EnumValueDescriptor')

    super(EnumValueDescriptor, self).__init__(
        type.file if type else None,
        options,
        serialized_options,
        'EnumValueOptions',
    )
    self.name = name
    self.index = index
    self.number = number
    self.type = type

  @property
  def _parent(self):
    return self.type


class OneofDescriptor(DescriptorBase):
  """Descriptor for a oneof field.

  Attributes:
    name (str): Name of the oneof field.
    full_name (str): Full name of the oneof field, including package name.
    index (int): 0-based index giving the order of the oneof field inside
      its containing type.
    containing_type (Descriptor): :class:`Descriptor` of the protocol message
      type that contains this field.  Set by the :class:`Descriptor` constructor
      if we're passed into one.
    fields (list[FieldDescriptor]): The list of field descriptors this
      oneof can contain.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.OneofDescriptor

    def __new__(
        cls, name, full_name, index, containing_type, fields, options=None,
        serialized_options=None, create_key=None):
      _message.Message._CheckCalledFromGeneratedFile()
      return _message.default_pool.FindOneofByName(full_name)

  def __init__(
      self, name, full_name, index, containing_type, fields, options=None,
      serialized_options=None, create_key=None):
    """Arguments are as described in the attribute description above."""
    if create_key is not _internal_create_key:
      _Deprecated('OneofDescriptor')

    super(OneofDescriptor, self).__init__(
        containing_type.file if containing_type else None,
        options,
        serialized_options,
        'OneofOptions',
    )
    self.name = name
    self.full_name = full_name
    self.index = index
    self.containing_type = containing_type
    self.fields = fields

  @property
  def _parent(self):
    return self.containing_type


class ServiceDescriptor(_NestedDescriptorBase):

  """Descriptor for a service.

  Attributes:
    name (str): Name of the service.
    full_name (str): Full name of the service, including package name.
    index (int): 0-indexed index giving the order that this services
      definition appears within the .proto file.
    methods (list[MethodDescriptor]): List of methods provided by this
      service.
    methods_by_name (dict(str, MethodDescriptor)): Same
      :class:`MethodDescriptor` objects as in :attr:`methods_by_name`, but
      indexed by "name" attribute in each :class:`MethodDescriptor`.
    options (descriptor_pb2.ServiceOptions): Service options message or
      None to use default service options.
    file (FileDescriptor): Reference to file info.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.ServiceDescriptor

    def __new__(
        cls,
        name=None,
        full_name=None,
        index=None,
        methods=None,
        options=None,
        serialized_options=None,
        file=None,  # pylint: disable=redefined-builtin
        serialized_start=None,
        serialized_end=None,
        create_key=None):
      _message.Message._CheckCalledFromGeneratedFile()  # pylint: disable=protected-access
      return _message.default_pool.FindServiceByName(full_name)

  def __init__(self, name, full_name, index, methods, options=None,
               serialized_options=None, file=None,  # pylint: disable=redefined-builtin
               serialized_start=None, serialized_end=None, create_key=None):
    if create_key is not _internal_create_key:
      _Deprecated('ServiceDescriptor')

    super(ServiceDescriptor, self).__init__(
        options, 'ServiceOptions', name, full_name, file,
        None, serialized_start=serialized_start,
        serialized_end=serialized_end, serialized_options=serialized_options)
    self.index = index
    self.methods = methods
    self.methods_by_name = dict((m.name, m) for m in methods)
    # Set the containing service for each method in this service.
    for method in self.methods:
      method.file = self.file
      method.containing_service = self

  @property
  def _parent(self):
    return self.file

  def FindMethodByName(self, name):
    """Searches for the specified method, and returns its descriptor.

    Args:
      name (str): Name of the method.

    Returns:
      MethodDescriptor: The descriptor for the requested method.

    Raises:
      KeyError: if the method cannot be found in the service.
    """
    return self.methods_by_name[name]

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.ServiceDescriptorProto.

    Args:
      proto (descriptor_pb2.ServiceDescriptorProto): An empty descriptor proto.
    """
    # This function is overridden to give a better doc comment.
    super(ServiceDescriptor, self).CopyToProto(proto)


class MethodDescriptor(DescriptorBase):

  """Descriptor for a method in a service.

  Attributes:
    name (str): Name of the method within the service.
    full_name (str): Full name of method.
    index (int): 0-indexed index of the method inside the service.
    containing_service (ServiceDescriptor): The service that contains this
      method.
    input_type (Descriptor): The descriptor of the message that this method
      accepts.
    output_type (Descriptor): The descriptor of the message that this method
      returns.
    client_streaming (bool): Whether this method uses client streaming.
    server_streaming (bool): Whether this method uses server streaming.
    options (descriptor_pb2.MethodOptions or None): Method options message, or
      None to use default method options.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.MethodDescriptor

    def __new__(cls,
                name,
                full_name,
                index,
                containing_service,
                input_type,
                output_type,
                client_streaming=False,
                server_streaming=False,
                options=None,
                serialized_options=None,
                create_key=None):
      _message.Message._CheckCalledFromGeneratedFile()  # pylint: disable=protected-access
      return _message.default_pool.FindMethodByName(full_name)

  def __init__(self,
               name,
               full_name,
               index,
               containing_service,
               input_type,
               output_type,
               client_streaming=False,
               server_streaming=False,
               options=None,
               serialized_options=None,
               create_key=None):
    """The arguments are as described in the description of MethodDescriptor
    attributes above.

    Note that containing_service may be None, and may be set later if necessary.
    """
    if create_key is not _internal_create_key:
      _Deprecated('MethodDescriptor')

    super(MethodDescriptor, self).__init__(
        containing_service.file if containing_service else None,
        options,
        serialized_options,
        'MethodOptions',
    )
    self.name = name
    self.full_name = full_name
    self.index = index
    self.containing_service = containing_service
    self.input_type = input_type
    self.output_type = output_type
    self.client_streaming = client_streaming
    self.server_streaming = server_streaming

  @property
  def _parent(self):
    return self.containing_service

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.MethodDescriptorProto.

    Args:
      proto (descriptor_pb2.MethodDescriptorProto): An empty descriptor proto.

    Raises:
      Error: If self couldn't be serialized, due to too few constructor
        arguments.
    """
    if self.containing_service is not None:
      from google.protobuf import descriptor_pb2
      service_proto = descriptor_pb2.ServiceDescriptorProto()
      self.containing_service.CopyToProto(service_proto)
      proto.CopyFrom(service_proto.method[self.index])
    else:
      raise Error('Descriptor does not contain a service.')


class FileDescriptor(DescriptorBase):
  """Descriptor for a file. Mimics the descriptor_pb2.FileDescriptorProto.

  Note that :attr:`enum_types_by_name`, :attr:`extensions_by_name`, and
  :attr:`dependencies` fields are only set by the
  :py:mod:`google.protobuf.message_factory` module, and not by the generated
  proto code.

  Attributes:
    name (str): Name of file, relative to root of source tree.
    package (str): Name of the package
    edition (Edition): Enum value indicating edition of the file
    serialized_pb (bytes): Byte string of serialized
      :class:`descriptor_pb2.FileDescriptorProto`.
    dependencies (list[FileDescriptor]): List of other :class:`FileDescriptor`
      objects this :class:`FileDescriptor` depends on.
    public_dependencies (list[FileDescriptor]): A subset of
      :attr:`dependencies`, which were declared as "public".
    message_types_by_name (dict(str, Descriptor)): Mapping from message names to
      their :class:`Descriptor`.
    enum_types_by_name (dict(str, EnumDescriptor)): Mapping from enum names to
      their :class:`EnumDescriptor`.
    extensions_by_name (dict(str, FieldDescriptor)): Mapping from extension
      names declared at file scope to their :class:`FieldDescriptor`.
    services_by_name (dict(str, ServiceDescriptor)): Mapping from services'
      names to their :class:`ServiceDescriptor`.
    pool (DescriptorPool): The pool this descriptor belongs to.  When not passed
      to the constructor, the global default pool is used.
  """

  if _USE_C_DESCRIPTORS:
    _C_DESCRIPTOR_CLASS = _message.FileDescriptor

    def __new__(
        cls,
        name,
        package,
        options=None,
        serialized_options=None,
        serialized_pb=None,
        dependencies=None,
        public_dependencies=None,
        syntax=None,
        edition=None,
        pool=None,
        create_key=None,
    ):
      # FileDescriptor() is called from various places, not only from generated
      # files, to register dynamic proto files and messages.
      # pylint: disable=g-explicit-bool-comparison
      if serialized_pb:
        return _message.default_pool.AddSerializedFile(serialized_pb)
      else:
        return super(FileDescriptor, cls).__new__(cls)

  def __init__(
      self,
      name,
      package,
      options=None,
      serialized_options=None,
      serialized_pb=None,
      dependencies=None,
      public_dependencies=None,
      syntax=None,
      edition=None,
      pool=None,
      create_key=None,
  ):
    """Constructor."""
    if create_key is not _internal_create_key:
      _Deprecated('FileDescriptor')

    super(FileDescriptor, self).__init__(
        self, options, serialized_options, 'FileOptions'
    )

    if edition and edition != 'EDITION_UNKNOWN':
      self._edition = edition
    elif syntax == 'proto3':
      self._edition = 'EDITION_PROTO3'
    else:
      self._edition = 'EDITION_PROTO2'

    if pool is None:
      from google.protobuf import descriptor_pool
      pool = descriptor_pool.Default()
    self.pool = pool
    self.message_types_by_name = {}
    self.name = name
    self.package = package
    self.serialized_pb = serialized_pb

    self.enum_types_by_name = {}
    self.extensions_by_name = {}
    self.services_by_name = {}
    self.dependencies = (dependencies or [])
    self.public_dependencies = (public_dependencies or [])

  def CopyToProto(self, proto):
    """Copies this to a descriptor_pb2.FileDescriptorProto.

    Args:
      proto: An empty descriptor_pb2.FileDescriptorProto.
    """
    proto.ParseFromString(self.serialized_pb)

  @property
  def _parent(self):
    return None


def _ParseOptions(message, string):
  """Parses serialized options.

  This helper function is used to parse serialized options in generated
  proto2 files. It must not be used outside proto2.
  """
  message.ParseFromString(string)
  return message


def _ToCamelCase(name):
  """Converts name to camel-case and returns it."""
  capitalize_next = False
  result = []

  for c in name:
    if c == '_':
      if result:
        capitalize_next = True
    elif capitalize_next:
      result.append(c.upper())
      capitalize_next = False
    else:
      result += c

  # Lower-case the first letter.
  if result and result[0].isupper():
    result[0] = result[0].lower()
  return ''.join(result)


def _OptionsOrNone(descriptor_proto):
  """Returns the value of the field `options`, or None if it is not set."""
  if descriptor_proto.HasField('options'):
    return descriptor_proto.options
  else:
    return None


def _ToJsonName(name):
  """Converts name to Json name and returns it."""
  capitalize_next = False
  result = []

  for c in name:
    if c == '_':
      capitalize_next = True
    elif capitalize_next:
      result.append(c.upper())
      capitalize_next = False
    else:
      result += c

  return ''.join(result)


def MakeDescriptor(
    desc_proto,
    package='',
    build_file_if_cpp=True,
    syntax=None,
    edition=None,
    file_desc=None,
):
  """Make a protobuf Descriptor given a DescriptorProto protobuf.

  Handles nested descriptors. Note that this is limited to the scope of defining
  a message inside of another message. Composite fields can currently only be
  resolved if the message is defined in the same scope as the field.

  Args:
    desc_proto: The descriptor_pb2.DescriptorProto protobuf message.
    package: Optional package name for the new message Descriptor (string).
    build_file_if_cpp: Update the C++ descriptor pool if api matches. Set to
      False on recursion, so no duplicates are created.
    syntax: The syntax/semantics that should be used.  Set to "proto3" to get
      proto3 field presence semantics.
    edition: The edition that should be used if syntax is "edition".
    file_desc: A FileDescriptor to place this descriptor into.

  Returns:
    A Descriptor for protobuf messages.
  """
  # pylint: disable=g-import-not-at-top
  from google.protobuf import descriptor_pb2

  # Generate a random name for this proto file to prevent conflicts with any
  # imported ones. We need to specify a file name so the descriptor pool
  # accepts our FileDescriptorProto, but it is not important what that file
  # name is actually set to.
  proto_name = binascii.hexlify(os.urandom(16)).decode('ascii')

  if package:
    file_name = os.path.join(package.replace('.', '/'), proto_name + '.proto')
  else:
    file_name = proto_name + '.proto'

  if api_implementation.Type() != 'python' and build_file_if_cpp:
    # The C++ implementation requires all descriptors to be backed by the same
    # definition in the C++ descriptor pool. To do this, we build a
    # FileDescriptorProto with the same definition as this descriptor and build
    # it into the pool.
    file_descriptor_proto = descriptor_pb2.FileDescriptorProto()
    file_descriptor_proto.message_type.add().MergeFrom(desc_proto)

    if package:
      file_descriptor_proto.package = package
    file_descriptor_proto.name = file_name

    _message.default_pool.Add(file_descriptor_proto)
    result = _message.default_pool.FindFileByName(file_descriptor_proto.name)

    if _USE_C_DESCRIPTORS:
      return result.message_types_by_name[desc_proto.name]

  if file_desc is None:
    file_desc = FileDescriptor(
        pool=None,
        name=file_name,
        package=package,
        syntax=syntax,
        edition=edition,
        options=None,
        serialized_pb='',
        dependencies=[],
        public_dependencies=[],
        create_key=_internal_create_key,
    )
  full_message_name = [desc_proto.name]
  if package: full_message_name.insert(0, package)

  # Create Descriptors for enum types
  enum_types = {}
  for enum_proto in desc_proto.enum_type:
    full_name = '.'.join(full_message_name + [enum_proto.name])
    enum_desc = EnumDescriptor(
        enum_proto.name,
        full_name,
        None,
        [
            EnumValueDescriptor(
                enum_val.name,
                ii,
                enum_val.number,
                create_key=_internal_create_key,
            )
            for ii, enum_val in enumerate(enum_proto.value)
        ],
        file=file_desc,
        create_key=_internal_create_key,
    )
    enum_types[full_name] = enum_desc

  # Create Descriptors for nested types
  nested_types = {}
  for nested_proto in desc_proto.nested_type:
    full_name = '.'.join(full_message_name + [nested_proto.name])
    # Nested types are just those defined inside of the message, not all types
    # used by fields in the message, so no loops are possible here.
    nested_desc = MakeDescriptor(
        nested_proto,
        package='.'.join(full_message_name),
        build_file_if_cpp=False,
        syntax=syntax,
        edition=edition,
        file_desc=file_desc,
    )
    nested_types[full_name] = nested_desc

  fields = []
  for field_proto in desc_proto.field:
    full_name = '.'.join(full_message_name + [field_proto.name])
    enum_desc = None
    nested_desc = None
    if field_proto.json_name:
      json_name = field_proto.json_name
    else:
      json_name = None
    if field_proto.HasField('type_name'):
      type_name = field_proto.type_name
      full_type_name = '.'.join(full_message_name +
                                [type_name[type_name.rfind('.')+1:]])
      if full_type_name in nested_types:
        nested_desc = nested_types[full_type_name]
      elif full_type_name in enum_types:
        enum_desc = enum_types[full_type_name]
      # Else type_name references a non-local type, which isn't implemented
    field = FieldDescriptor(
        field_proto.name,
        full_name,
        field_proto.number - 1,
        field_proto.number,
        field_proto.type,
        FieldDescriptor.ProtoTypeToCppProtoType(field_proto.type),
        field_proto.label,
        None,
        nested_desc,
        enum_desc,
        None,
        False,
        None,
        options=_OptionsOrNone(field_proto),
        has_default_value=False,
        json_name=json_name,
        file=file_desc,
        create_key=_internal_create_key,
    )
    fields.append(field)

  desc_name = '.'.join(full_message_name)
  return Descriptor(
      desc_proto.name,
      desc_name,
      None,
      None,
      fields,
      list(nested_types.values()),
      list(enum_types.values()),
      [],
      options=_OptionsOrNone(desc_proto),
      file=file_desc,
      create_key=_internal_create_key,
  )