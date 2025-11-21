
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\types.py ===
# dialects/mysql/types.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


import datetime

from ... import exc
from ... import util
from ...sql import sqltypes


class _NumericType:
    """Base for MySQL numeric types.

    This is the base both for NUMERIC as well as INTEGER, hence
    it's a mixin.

    """

    def __init__(self, unsigned=False, zerofill=False, **kw):
        self.unsigned = unsigned
        self.zerofill = zerofill
        super().__init__(**kw)

    def __repr__(self):
        return util.generic_repr(
            self, to_inspect=[_NumericType, sqltypes.Numeric]
        )


class _FloatType(_NumericType, sqltypes.Float):
    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        if isinstance(self, (REAL, DOUBLE)) and (
            (precision is None and scale is not None)
            or (precision is not None and scale is None)
        ):
            raise exc.ArgumentError(
                "You must specify both precision and scale or omit "
                "both altogether."
            )
        super().__init__(precision=precision, asdecimal=asdecimal, **kw)
        self.scale = scale

    def __repr__(self):
        return util.generic_repr(
            self, to_inspect=[_FloatType, _NumericType, sqltypes.Float]
        )


class _IntegerType(_NumericType, sqltypes.Integer):
    def __init__(self, display_width=None, **kw):
        self.display_width = display_width
        super().__init__(**kw)

    def __repr__(self):
        return util.generic_repr(
            self, to_inspect=[_IntegerType, _NumericType, sqltypes.Integer]
        )


class _StringType(sqltypes.String):
    """Base for MySQL string types."""

    def __init__(
        self,
        charset=None,
        collation=None,
        ascii=False,  # noqa
        binary=False,
        unicode=False,
        national=False,
        **kw,
    ):
        self.charset = charset

        # allow collate= or collation=
        kw.setdefault("collation", kw.pop("collate", collation))

        self.ascii = ascii
        self.unicode = unicode
        self.binary = binary
        self.national = national
        super().__init__(**kw)

    def __repr__(self):
        return util.generic_repr(
            self, to_inspect=[_StringType, sqltypes.String]
        )


class _MatchType(sqltypes.Float, sqltypes.MatchType):
    def __init__(self, **kw):
        # TODO: float arguments?
        sqltypes.Float.__init__(self)
        sqltypes.MatchType.__init__(self)


class NUMERIC(_NumericType, sqltypes.NUMERIC):
    """MySQL NUMERIC type."""

    __visit_name__ = "NUMERIC"

    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        """Construct a NUMERIC.

        :param precision: Total digits in this number.  If scale and precision
          are both None, values are stored to limits allowed by the server.

        :param scale: The number of digits after the decimal point.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(
            precision=precision, scale=scale, asdecimal=asdecimal, **kw
        )


class DECIMAL(_NumericType, sqltypes.DECIMAL):
    """MySQL DECIMAL type."""

    __visit_name__ = "DECIMAL"

    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        """Construct a DECIMAL.

        :param precision: Total digits in this number.  If scale and precision
          are both None, values are stored to limits allowed by the server.

        :param scale: The number of digits after the decimal point.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(
            precision=precision, scale=scale, asdecimal=asdecimal, **kw
        )


class DOUBLE(_FloatType, sqltypes.DOUBLE):
    """MySQL DOUBLE type."""

    __visit_name__ = "DOUBLE"

    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        """Construct a DOUBLE.

        .. note::

            The :class:`.DOUBLE` type by default converts from float
            to Decimal, using a truncation that defaults to 10 digits.
            Specify either ``scale=n`` or ``decimal_return_scale=n`` in order
            to change this scale, or ``asdecimal=False`` to return values
            directly as Python floating points.

        :param precision: Total digits in this number.  If scale and precision
          are both None, values are stored to limits allowed by the server.

        :param scale: The number of digits after the decimal point.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(
            precision=precision, scale=scale, asdecimal=asdecimal, **kw
        )


class REAL(_FloatType, sqltypes.REAL):
    """MySQL REAL type."""

    __visit_name__ = "REAL"

    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        """Construct a REAL.

        .. note::

            The :class:`.REAL` type by default converts from float
            to Decimal, using a truncation that defaults to 10 digits.
            Specify either ``scale=n`` or ``decimal_return_scale=n`` in order
            to change this scale, or ``asdecimal=False`` to return values
            directly as Python floating points.

        :param precision: Total digits in this number.  If scale and precision
          are both None, values are stored to limits allowed by the server.

        :param scale: The number of digits after the decimal point.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(
            precision=precision, scale=scale, asdecimal=asdecimal, **kw
        )


class FLOAT(_FloatType, sqltypes.FLOAT):
    """MySQL FLOAT type."""

    __visit_name__ = "FLOAT"

    def __init__(self, precision=None, scale=None, asdecimal=False, **kw):
        """Construct a FLOAT.

        :param precision: Total digits in this number.  If scale and precision
          are both None, values are stored to limits allowed by the server.

        :param scale: The number of digits after the decimal point.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(
            precision=precision, scale=scale, asdecimal=asdecimal, **kw
        )

    def bind_processor(self, dialect):
        return None


class INTEGER(_IntegerType, sqltypes.INTEGER):
    """MySQL INTEGER type."""

    __visit_name__ = "INTEGER"

    def __init__(self, display_width=None, **kw):
        """Construct an INTEGER.

        :param display_width: Optional, maximum display width for this number.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(display_width=display_width, **kw)


class BIGINT(_IntegerType, sqltypes.BIGINT):
    """MySQL BIGINTEGER type."""

    __visit_name__ = "BIGINT"

    def __init__(self, display_width=None, **kw):
        """Construct a BIGINTEGER.

        :param display_width: Optional, maximum display width for this number.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(display_width=display_width, **kw)


class MEDIUMINT(_IntegerType):
    """MySQL MEDIUMINTEGER type."""

    __visit_name__ = "MEDIUMINT"

    def __init__(self, display_width=None, **kw):
        """Construct a MEDIUMINTEGER

        :param display_width: Optional, maximum display width for this number.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(display_width=display_width, **kw)


class TINYINT(_IntegerType):
    """MySQL TINYINT type."""

    __visit_name__ = "TINYINT"

    def __init__(self, display_width=None, **kw):
        """Construct a TINYINT.

        :param display_width: Optional, maximum display width for this number.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(display_width=display_width, **kw)


class SMALLINT(_IntegerType, sqltypes.SMALLINT):
    """MySQL SMALLINTEGER type."""

    __visit_name__ = "SMALLINT"

    def __init__(self, display_width=None, **kw):
        """Construct a SMALLINTEGER.

        :param display_width: Optional, maximum display width for this number.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(display_width=display_width, **kw)


class BIT(sqltypes.TypeEngine):
    """MySQL BIT type.

    This type is for MySQL 5.0.3 or greater for MyISAM, and 5.0.5 or greater
    for MyISAM, MEMORY, InnoDB and BDB.  For older versions, use a
    MSTinyInteger() type.

    """

    __visit_name__ = "BIT"

    def __init__(self, length=None):
        """Construct a BIT.

        :param length: Optional, number of bits.

        """
        self.length = length

    def result_processor(self, dialect, coltype):
        """Convert a MySQL's 64 bit, variable length binary string to a
        long."""

        if dialect.supports_native_bit:
            return None

        def process(value):
            if value is not None:
                v = 0
                for i in value:
                    if not isinstance(i, int):
                        i = ord(i)  # convert byte to int on Python 2
                    v = v << 8 | i
                return v
            return value

        return process


class TIME(sqltypes.TIME):
    """MySQL TIME type."""

    __visit_name__ = "TIME"

    def __init__(self, timezone=False, fsp=None):
        """Construct a MySQL TIME type.

        :param timezone: not used by the MySQL dialect.
        :param fsp: fractional seconds precision value.
         MySQL 5.6 supports storage of fractional seconds;
         this parameter will be used when emitting DDL
         for the TIME type.

         .. note::

            DBAPI driver support for fractional seconds may
            be limited; current support includes
            MySQL Connector/Python.

        """
        super().__init__(timezone=timezone)
        self.fsp = fsp

    def result_processor(self, dialect, coltype):
        time = datetime.time

        def process(value):
            # convert from a timedelta value
            if value is not None:
                microseconds = value.microseconds
                seconds = value.seconds
                minutes = seconds // 60
                return time(
                    minutes // 60,
                    minutes % 60,
                    seconds - minutes * 60,
                    microsecond=microseconds,
                )
            else:
                return None

        return process


class TIMESTAMP(sqltypes.TIMESTAMP):
    """MySQL TIMESTAMP type."""

    __visit_name__ = "TIMESTAMP"

    def __init__(self, timezone=False, fsp=None):
        """Construct a MySQL TIMESTAMP type.

        :param timezone: not used by the MySQL dialect.
        :param fsp: fractional seconds precision value.
         MySQL 5.6.4 supports storage of fractional seconds;
         this parameter will be used when emitting DDL
         for the TIMESTAMP type.

         .. note::

            DBAPI driver support for fractional seconds may
            be limited; current support includes
            MySQL Connector/Python.

        """
        super().__init__(timezone=timezone)
        self.fsp = fsp


class DATETIME(sqltypes.DATETIME):
    """MySQL DATETIME type."""

    __visit_name__ = "DATETIME"

    def __init__(self, timezone=False, fsp=None):
        """Construct a MySQL DATETIME type.

        :param timezone: not used by the MySQL dialect.
        :param fsp: fractional seconds precision value.
         MySQL 5.6.4 supports storage of fractional seconds;
         this parameter will be used when emitting DDL
         for the DATETIME type.

         .. note::

            DBAPI driver support for fractional seconds may
            be limited; current support includes
            MySQL Connector/Python.

        """
        super().__init__(timezone=timezone)
        self.fsp = fsp


class YEAR(sqltypes.TypeEngine):
    """MySQL YEAR type, for single byte storage of years 1901-2155."""

    __visit_name__ = "YEAR"

    def __init__(self, display_width=None):
        self.display_width = display_width


class TEXT(_StringType, sqltypes.TEXT):
    """MySQL TEXT type, for character storage encoded up to 2^16 bytes."""

    __visit_name__ = "TEXT"

    def __init__(self, length=None, **kw):
        """Construct a TEXT.

        :param length: Optional, if provided the server may optimize storage
          by substituting the smallest TEXT type sufficient to store
          ``length`` bytes of characters.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param national: Optional. If true, use the server's configured
          national character set.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        super().__init__(length=length, **kw)


class TINYTEXT(_StringType):
    """MySQL TINYTEXT type, for character storage encoded up to 2^8 bytes."""

    __visit_name__ = "TINYTEXT"

    def __init__(self, **kwargs):
        """Construct a TINYTEXT.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param national: Optional. If true, use the server's configured
          national character set.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        super().__init__(**kwargs)


class MEDIUMTEXT(_StringType):
    """MySQL MEDIUMTEXT type, for character storage encoded up
    to 2^24 bytes."""

    __visit_name__ = "MEDIUMTEXT"

    def __init__(self, **kwargs):
        """Construct a MEDIUMTEXT.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param national: Optional. If true, use the server's configured
          national character set.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        super().__init__(**kwargs)


class LONGTEXT(_StringType):
    """MySQL LONGTEXT type, for character storage encoded up to 2^32 bytes."""

    __visit_name__ = "LONGTEXT"

    def __init__(self, **kwargs):
        """Construct a LONGTEXT.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param national: Optional. If true, use the server's configured
          national character set.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        super().__init__(**kwargs)


class VARCHAR(_StringType, sqltypes.VARCHAR):
    """MySQL VARCHAR type, for variable-length character data."""

    __visit_name__ = "VARCHAR"

    def __init__(self, length=None, **kwargs):
        """Construct a VARCHAR.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param national: Optional. If true, use the server's configured
          national character set.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        super().__init__(length=length, **kwargs)


class CHAR(_StringType, sqltypes.CHAR):
    """MySQL CHAR type, for fixed-length character data."""

    __visit_name__ = "CHAR"

    def __init__(self, length=None, **kwargs):
        """Construct a CHAR.

        :param length: Maximum data length, in characters.

        :param binary: Optional, use the default binary collation for the
          national character set.  This does not affect the type of data
          stored, use a BINARY type for binary data.

        :param collation: Optional, request a particular collation.  Must be
          compatible with the national character set.

        """
        super().__init__(length=length, **kwargs)

    @classmethod
    def _adapt_string_for_cast(cls, type_):
        # copy the given string type into a CHAR
        # for the purposes of rendering a CAST expression
        type_ = sqltypes.to_instance(type_)
        if isinstance(type_, sqltypes.CHAR):
            return type_
        elif isinstance(type_, _StringType):
            return CHAR(
                length=type_.length,
                charset=type_.charset,
                collation=type_.collation,
                ascii=type_.ascii,
                binary=type_.binary,
                unicode=type_.unicode,
                national=False,  # not supported in CAST
            )
        else:
            return CHAR(length=type_.length)


class NVARCHAR(_StringType, sqltypes.NVARCHAR):
    """MySQL NVARCHAR type.

    For variable-length character data in the server's configured national
    character set.
    """

    __visit_name__ = "NVARCHAR"

    def __init__(self, length=None, **kwargs):
        """Construct an NVARCHAR.

        :param length: Maximum data length, in characters.

        :param binary: Optional, use the default binary collation for the
          national character set.  This does not affect the type of data
          stored, use a BINARY type for binary data.

        :param collation: Optional, request a particular collation.  Must be
          compatible with the national character set.

        """
        kwargs["national"] = True
        super().__init__(length=length, **kwargs)


class NCHAR(_StringType, sqltypes.NCHAR):
    """MySQL NCHAR type.

    For fixed-length character data in the server's configured national
    character set.
    """

    __visit_name__ = "NCHAR"

    def __init__(self, length=None, **kwargs):
        """Construct an NCHAR.

        :param length: Maximum data length, in characters.

        :param binary: Optional, use the default binary collation for the
          national character set.  This does not affect the type of data
          stored, use a BINARY type for binary data.

        :param collation: Optional, request a particular collation.  Must be
          compatible with the national character set.

        """
        kwargs["national"] = True
        super().__init__(length=length, **kwargs)


class TINYBLOB(sqltypes._Binary):
    """MySQL TINYBLOB type, for binary data up to 2^8 bytes."""

    __visit_name__ = "TINYBLOB"


class MEDIUMBLOB(sqltypes._Binary):
    """MySQL MEDIUMBLOB type, for binary data up to 2^24 bytes."""

    __visit_name__ = "MEDIUMBLOB"


class LONGBLOB(sqltypes._Binary):
    """MySQL LONGBLOB type, for binary data up to 2^32 bytes."""

    __visit_name__ = "LONGBLOB"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\ops\docstrings.py ===
"""
Templating for ops docstrings
"""
from __future__ import annotations


def make_flex_doc(op_name: str, typ: str) -> str:
    """
    Make the appropriate substitutions for the given operation and class-typ
    into either _flex_doc_SERIES or _flex_doc_FRAME to return the docstring
    to attach to a generated method.

    Parameters
    ----------
    op_name : str {'__add__', '__sub__', ... '__eq__', '__ne__', ...}
    typ : str {series, 'dataframe']}

    Returns
    -------
    doc : str
    """
    op_name = op_name.replace("__", "")
    op_desc = _op_descriptions[op_name]

    op_desc_op = op_desc["op"]
    assert op_desc_op is not None  # for mypy
    if op_name.startswith("r"):
        equiv = f"other {op_desc_op} {typ}"
    elif op_name == "divmod":
        equiv = f"{op_name}({typ}, other)"
    else:
        equiv = f"{typ} {op_desc_op} other"

    if typ == "series":
        base_doc = _flex_doc_SERIES
        if op_desc["reverse"]:
            base_doc += _see_also_reverse_SERIES.format(
                reverse=op_desc["reverse"], see_also_desc=op_desc["see_also_desc"]
            )
        doc_no_examples = base_doc.format(
            desc=op_desc["desc"],
            op_name=op_name,
            equiv=equiv,
            series_returns=op_desc["series_returns"],
        )
        ser_example = op_desc["series_examples"]
        if ser_example:
            doc = doc_no_examples + ser_example
        else:
            doc = doc_no_examples
    elif typ == "dataframe":
        if op_name in ["eq", "ne", "le", "lt", "ge", "gt"]:
            base_doc = _flex_comp_doc_FRAME
            doc = _flex_comp_doc_FRAME.format(
                op_name=op_name,
                desc=op_desc["desc"],
            )
        else:
            base_doc = _flex_doc_FRAME
            doc = base_doc.format(
                desc=op_desc["desc"],
                op_name=op_name,
                equiv=equiv,
                reverse=op_desc["reverse"],
            )
    else:
        raise AssertionError("Invalid typ argument.")
    return doc


_common_examples_algebra_SERIES = """
Examples
--------
>>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
>>> a
a    1.0
b    1.0
c    1.0
d    NaN
dtype: float64
>>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
>>> b
a    1.0
b    NaN
d    1.0
e    NaN
dtype: float64"""

_common_examples_comparison_SERIES = """
Examples
--------
>>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
>>> a
a    1.0
b    1.0
c    1.0
d    NaN
e    1.0
dtype: float64
>>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
>>> b
a    0.0
b    1.0
c    2.0
d    NaN
f    1.0
dtype: float64"""

_add_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.add(b, fill_value=0)
a    2.0
b    1.0
c    1.0
d    1.0
e    NaN
dtype: float64
"""
)

_sub_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.subtract(b, fill_value=0)
a    0.0
b    1.0
c    1.0
d   -1.0
e    NaN
dtype: float64
"""
)

_mul_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.multiply(b, fill_value=0)
a    1.0
b    0.0
c    0.0
d    0.0
e    NaN
dtype: float64
"""
)

_div_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.divide(b, fill_value=0)
a    1.0
b    inf
c    inf
d    0.0
e    NaN
dtype: float64
"""
)

_floordiv_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.floordiv(b, fill_value=0)
a    1.0
b    inf
c    inf
d    0.0
e    NaN
dtype: float64
"""
)

_divmod_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.divmod(b, fill_value=0)
(a    1.0
 b    inf
 c    inf
 d    0.0
 e    NaN
 dtype: float64,
 a    0.0
 b    NaN
 c    NaN
 d    0.0
 e    NaN
 dtype: float64)
"""
)

_mod_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.mod(b, fill_value=0)
a    0.0
b    NaN
c    NaN
d    0.0
e    NaN
dtype: float64
"""
)
_pow_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.pow(b, fill_value=0)
a    1.0
b    1.0
c    1.0
d    0.0
e    NaN
dtype: float64
"""
)

_ne_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.ne(b, fill_value=0)
a    False
b     True
c     True
d     True
e     True
dtype: bool
"""
)

_eq_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.eq(b, fill_value=0)
a     True
b    False
c    False
d    False
e    False
dtype: bool
"""
)

_lt_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.lt(b, fill_value=0)
a    False
b    False
c     True
d    False
e    False
f     True
dtype: bool
"""
)

_le_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.le(b, fill_value=0)
a    False
b     True
c     True
d    False
e    False
f     True
dtype: bool
"""
)

_gt_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.gt(b, fill_value=0)
a     True
b    False
c    False
d    False
e     True
f    False
dtype: bool
"""
)

_ge_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.ge(b, fill_value=0)
a     True
b     True
c    False
d    False
e     True
f    False
dtype: bool
"""
)

_returns_series = """Series\n    The result of the operation."""

_returns_tuple = """2-Tuple of Series\n    The result of the operation."""

_op_descriptions: dict[str, dict[str, str | None]] = {
    # Arithmetic Operators
    "add": {
        "op": "+",
        "desc": "Addition",
        "reverse": "radd",
        "series_examples": _add_example_SERIES,
        "series_returns": _returns_series,
    },
    "sub": {
        "op": "-",
        "desc": "Subtraction",
        "reverse": "rsub",
        "series_examples": _sub_example_SERIES,
        "series_returns": _returns_series,
    },
    "mul": {
        "op": "*",
        "desc": "Multiplication",
        "reverse": "rmul",
        "series_examples": _mul_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "mod": {
        "op": "%",
        "desc": "Modulo",
        "reverse": "rmod",
        "series_examples": _mod_example_SERIES,
        "series_returns": _returns_series,
    },
    "pow": {
        "op": "**",
        "desc": "Exponential power",
        "reverse": "rpow",
        "series_examples": _pow_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "truediv": {
        "op": "/",
        "desc": "Floating division",
        "reverse": "rtruediv",
        "series_examples": _div_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "floordiv": {
        "op": "//",
        "desc": "Integer division",
        "reverse": "rfloordiv",
        "series_examples": _floordiv_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "divmod": {
        "op": "divmod",
        "desc": "Integer division and modulo",
        "reverse": "rdivmod",
        "series_examples": _divmod_example_SERIES,
        "series_returns": _returns_tuple,
        "df_examples": None,
    },
    # Comparison Operators
    "eq": {
        "op": "==",
        "desc": "Equal to",
        "reverse": None,
        "series_examples": _eq_example_SERIES,
        "series_returns": _returns_series,
    },
    "ne": {
        "op": "!=",
        "desc": "Not equal to",
        "reverse": None,
        "series_examples": _ne_example_SERIES,
        "series_returns": _returns_series,
    },
    "lt": {
        "op": "<",
        "desc": "Less than",
        "reverse": None,
        "series_examples": _lt_example_SERIES,
        "series_returns": _returns_series,
    },
    "le": {
        "op": "<=",
        "desc": "Less than or equal to",
        "reverse": None,
        "series_examples": _le_example_SERIES,
        "series_returns": _returns_series,
    },
    "gt": {
        "op": ">",
        "desc": "Greater than",
        "reverse": None,
        "series_examples": _gt_example_SERIES,
        "series_returns": _returns_series,
    },
    "ge": {
        "op": ">=",
        "desc": "Greater than or equal to",
        "reverse": None,
        "series_examples": _ge_example_SERIES,
        "series_returns": _returns_series,
    },
}

_py_num_ref = """see
    `Python documentation
    <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
    for more details"""
_op_names = list(_op_descriptions.keys())
for key in _op_names:
    reverse_op = _op_descriptions[key]["reverse"]
    if reverse_op is not None:
        _op_descriptions[reverse_op] = _op_descriptions[key].copy()
        _op_descriptions[reverse_op]["reverse"] = key
        _op_descriptions[key][
            "see_also_desc"
        ] = f"Reverse of the {_op_descriptions[key]['desc']} operator, {_py_num_ref}"
        _op_descriptions[reverse_op][
            "see_also_desc"
        ] = f"Element-wise {_op_descriptions[key]['desc']}, {_py_num_ref}"

_flex_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``, but with support to substitute a fill_value for
missing data in either one of the inputs.

Parameters
----------
other : Series or scalar value
level : int or name
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : None or float value, default None (NaN)
    Fill existing missing (NaN) values, and any new element needed for
    successful Series alignment, with this value before computation.
    If data in both corresponding Series locations is missing
    the result of filling (at that location) will be missing.
axis : {{0 or 'index'}}
    Unused. Parameter needed for compatibility with DataFrame.

Returns
-------
{series_returns}
"""

_see_also_reverse_SERIES = """
See Also
--------
Series.{reverse} : {see_also_desc}.
"""

_flex_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``, but with support to substitute a fill_value
for missing data in one of the inputs. With reverse version, `{reverse}`.

Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

Parameters
----------
other : scalar, sequence, Series, dict or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}
    Whether to compare by the index (0 or 'index') or columns.
    (1 or 'columns'). For Series input, axis to match Series index on.
level : int or label
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : float or None, default None
    Fill existing missing (NaN) values, and any new element needed for
    successful DataFrame alignment, with this value before computation.
    If data in both corresponding DataFrame locations is missing
    the result will be missing.

Returns
-------
DataFrame
    Result of the arithmetic operation.

See Also
--------
DataFrame.add : Add DataFrames.
DataFrame.sub : Subtract DataFrames.
DataFrame.mul : Multiply DataFrames.
DataFrame.div : Divide DataFrames (float division).
DataFrame.truediv : Divide DataFrames (float division).
DataFrame.floordiv : Divide DataFrames (integer division).
DataFrame.mod : Calculate modulo (remainder after division).
DataFrame.pow : Calculate exponential power.

Notes
-----
Mismatched indices will be unioned together.

Examples
--------
>>> df = pd.DataFrame({{'angles': [0, 3, 4],
...                    'degrees': [360, 180, 360]}},
...                   index=['circle', 'triangle', 'rectangle'])
>>> df
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Add a scalar with operator version which return the same
results.

>>> df + 1
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

>>> df.add(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

Divide by constant with reverse version.

>>> df.div(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rdiv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

Subtract a list and Series by axis with operator version.

>>> df - [1, 2]
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

>>> df.sub([1, 2], axis='columns')
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

>>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
...        axis='index')
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

Multiply a dictionary by axis.

>>> df.mul({{'angles': 0, 'degrees': 2}})
            angles  degrees
circle           0      720
triangle         0      360
rectangle        0      720

>>> df.mul({{'circle': 0, 'triangle': 2, 'rectangle': 3}}, axis='index')
            angles  degrees
circle           0        0
triangle         6      360
rectangle       12     1080

Multiply a DataFrame of different shape with operator version.

>>> other = pd.DataFrame({{'angles': [0, 3, 4]}},
...                      index=['circle', 'triangle', 'rectangle'])
>>> other
           angles
circle          0
triangle        3
rectangle       4

>>> df * other
           angles  degrees
circle          0      NaN
triangle        9      NaN
rectangle      16      NaN

>>> df.mul(other, fill_value=0)
           angles  degrees
circle          0      0.0
triangle        9      0.0
rectangle      16      0.0

Divide by a MultiIndex by level.

>>> df_multindex = pd.DataFrame({{'angles': [0, 3, 4, 4, 5, 6],
...                              'degrees': [360, 180, 360, 360, 540, 720]}},
...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
...                                    ['circle', 'triangle', 'rectangle',
...                                     'square', 'pentagon', 'hexagon']])
>>> df_multindex
             angles  degrees
A circle          0      360
  triangle        3      180
  rectangle       4      360
B square          4      360
  pentagon        5      540
  hexagon         6      720

>>> df.div(df_multindex, level=1, fill_value=0)
             angles  degrees
A circle        NaN      1.0
  triangle      1.0      1.0
  rectangle     1.0      1.0
B square        0.0      0.0
  pentagon      0.0      0.0
  hexagon       0.0      0.0
"""

_flex_comp_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
operators.

Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
(rows or columns) and level for comparison.

Parameters
----------
other : scalar, sequence, Series, or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}, default 'columns'
    Whether to compare by the index (0 or 'index') or columns
    (1 or 'columns').
level : int or label
    Broadcast across a level, matching Index values on the passed
    MultiIndex level.

Returns
-------
DataFrame of bool
    Result of the comparison.

See Also
--------
DataFrame.eq : Compare DataFrames for equality elementwise.
DataFrame.ne : Compare DataFrames for inequality elementwise.
DataFrame.le : Compare DataFrames for less than inequality
    or equality elementwise.
DataFrame.lt : Compare DataFrames for strictly less than
    inequality elementwise.
DataFrame.ge : Compare DataFrames for greater than inequality
    or equality elementwise.
DataFrame.gt : Compare DataFrames for strictly greater than
    inequality elementwise.

Notes
-----
Mismatched indices will be unioned together.
`NaN` values are considered different (i.e. `NaN` != `NaN`).

Examples
--------
>>> df = pd.DataFrame({{'cost': [250, 150, 100],
...                    'revenue': [100, 250, 300]}},
...                   index=['A', 'B', 'C'])
>>> df
   cost  revenue
A   250      100
B   150      250
C   100      300

Comparison with a scalar, using either the operator or method:

>>> df == 100
    cost  revenue
A  False     True
B  False    False
C   True    False

>>> df.eq(100)
    cost  revenue
A  False     True
B  False    False
C   True    False

When `other` is a :class:`Series`, the columns of a DataFrame are aligned
with the index of `other` and broadcast:

>>> df != pd.Series([100, 250], index=["cost", "revenue"])
    cost  revenue
A   True     True
B   True    False
C  False     True

Use the method to control the broadcast axis:

>>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis='index')
   cost  revenue
A  True    False
B  True     True
C  True     True
D  True     True

When comparing to an arbitrary sequence, the number of columns must
match the number elements in `other`:

>>> df == [250, 100]
    cost  revenue
A   True     True
B  False    False
C  False    False

Use the method to control the axis:

>>> df.eq([250, 250, 100], axis='index')
    cost  revenue
A   True    False
B  False     True
C   True    False

Compare to a DataFrame of different shape.

>>> other = pd.DataFrame({{'revenue': [300, 250, 100, 150]}},
...                      index=['A', 'B', 'C', 'D'])
>>> other
   revenue
A      300
B      250
C      100
D      150

>>> df.gt(other)
    cost  revenue
A  False    False
B  False    False
C  False     True
D  False    False

Compare to a MultiIndex by level.

>>> df_multindex = pd.DataFrame({{'cost': [250, 150, 100, 150, 300, 220],
...                              'revenue': [100, 250, 300, 200, 175, 225]}},
...                             index=[['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'],
...                                    ['A', 'B', 'C', 'A', 'B', 'C']])
>>> df_multindex
      cost  revenue
Q1 A   250      100
   B   150      250
   C   100      300
Q2 A   150      200
   B   300      175
   C   220      225

>>> df.le(df_multindex, level=1)
       cost  revenue
Q1 A   True     True
   B   True     True
   C   True     True
Q2 A  False     True
   B   True    False
   C   True    False
"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\tomli\_parser.py ===
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Taneli Hukkinen
# Licensed to PSF under a Contributor Agreement.

from __future__ import annotations

from collections.abc import Iterable
import string
import sys
from types import MappingProxyType
from typing import IO, Any, Final, NamedTuple
import warnings

from ._re import (
    RE_DATETIME,
    RE_LOCALTIME,
    RE_NUMBER,
    match_to_datetime,
    match_to_localtime,
    match_to_number,
)
from ._types import Key, ParseFloat, Pos

# Inline tables/arrays are implemented using recursion. Pathologically
# nested documents cause pure Python to raise RecursionError (which is OK),
# but mypyc binary wheels will crash unrecoverably (not OK). According to
# mypyc docs this will be fixed in the future:
# https://mypyc.readthedocs.io/en/latest/differences_from_python.html#stack-overflows
# Before mypyc's fix is in, recursion needs to be limited by this library.
# Choosing `sys.getrecursionlimit()` as maximum inline table/array nesting
# level, as it allows more nesting than pure Python, but still seems a far
# lower number than where mypyc binaries crash.
MAX_INLINE_NESTING: Final = sys.getrecursionlimit()

ASCII_CTRL: Final = frozenset(chr(i) for i in range(32)) | frozenset(chr(127))

# Neither of these sets include quotation mark or backslash. They are
# currently handled as separate cases in the parser functions.
ILLEGAL_BASIC_STR_CHARS: Final = ASCII_CTRL - frozenset("\t")
ILLEGAL_MULTILINE_BASIC_STR_CHARS: Final = ASCII_CTRL - frozenset("\t\n")

ILLEGAL_LITERAL_STR_CHARS: Final = ILLEGAL_BASIC_STR_CHARS
ILLEGAL_MULTILINE_LITERAL_STR_CHARS: Final = ILLEGAL_MULTILINE_BASIC_STR_CHARS

ILLEGAL_COMMENT_CHARS: Final = ILLEGAL_BASIC_STR_CHARS

TOML_WS: Final = frozenset(" \t")
TOML_WS_AND_NEWLINE: Final = TOML_WS | frozenset("\n")
BARE_KEY_CHARS: Final = frozenset(string.ascii_letters + string.digits + "-_")
KEY_INITIAL_CHARS: Final = BARE_KEY_CHARS | frozenset("\"'")
HEXDIGIT_CHARS: Final = frozenset(string.hexdigits)

BASIC_STR_ESCAPE_REPLACEMENTS: Final = MappingProxyType(
    {
        "\\b": "\u0008",  # backspace
        "\\t": "\u0009",  # tab
        "\\n": "\u000A",  # linefeed
        "\\f": "\u000C",  # form feed
        "\\r": "\u000D",  # carriage return
        '\\"': "\u0022",  # quote
        "\\\\": "\u005C",  # backslash
    }
)


class DEPRECATED_DEFAULT:
    """Sentinel to be used as default arg during deprecation
    period of TOMLDecodeError's free-form arguments."""


class TOMLDecodeError(ValueError):
    """An error raised if a document is not valid TOML.

    Adds the following attributes to ValueError:
    msg: The unformatted error message
    doc: The TOML document being parsed
    pos: The index of doc where parsing failed
    lineno: The line corresponding to pos
    colno: The column corresponding to pos
    """

    def __init__(
        self,
        msg: str | type[DEPRECATED_DEFAULT] = DEPRECATED_DEFAULT,
        doc: str | type[DEPRECATED_DEFAULT] = DEPRECATED_DEFAULT,
        pos: Pos | type[DEPRECATED_DEFAULT] = DEPRECATED_DEFAULT,
        *args: Any,
    ):
        if (
            args
            or not isinstance(msg, str)
            or not isinstance(doc, str)
            or not isinstance(pos, int)
        ):
            warnings.warn(
                "Free-form arguments for TOMLDecodeError are deprecated. "
                "Please set 'msg' (str), 'doc' (str) and 'pos' (int) arguments only.",
                DeprecationWarning,
                stacklevel=2,
            )
            if pos is not DEPRECATED_DEFAULT:
                args = pos, *args
            if doc is not DEPRECATED_DEFAULT:
                args = doc, *args
            if msg is not DEPRECATED_DEFAULT:
                args = msg, *args
            ValueError.__init__(self, *args)
            return

        lineno = doc.count("\n", 0, pos) + 1
        if lineno == 1:
            colno = pos + 1
        else:
            colno = pos - doc.rindex("\n", 0, pos)

        if pos >= len(doc):
            coord_repr = "end of document"
        else:
            coord_repr = f"line {lineno}, column {colno}"
        errmsg = f"{msg} (at {coord_repr})"
        ValueError.__init__(self, errmsg)

        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = lineno
        self.colno = colno


def load(__fp: IO[bytes], *, parse_float: ParseFloat = float) -> dict[str, Any]:
    """Parse TOML from a binary file object."""
    b = __fp.read()
    try:
        s = b.decode()
    except AttributeError:
        raise TypeError(
            "File must be opened in binary mode, e.g. use `open('foo.toml', 'rb')`"
        ) from None
    return loads(s, parse_float=parse_float)


def loads(__s: str, *, parse_float: ParseFloat = float) -> dict[str, Any]:  # noqa: C901
    """Parse TOML from a string."""

    # The spec allows converting "\r\n" to "\n", even in string
    # literals. Let's do so to simplify parsing.
    try:
        src = __s.replace("\r\n", "\n")
    except (AttributeError, TypeError):
        raise TypeError(
            f"Expected str object, not '{type(__s).__qualname__}'"
        ) from None
    pos = 0
    out = Output(NestedDict(), Flags())
    header: Key = ()
    parse_float = make_safe_parse_float(parse_float)

    # Parse one statement at a time
    # (typically means one line in TOML source)
    while True:
        # 1. Skip line leading whitespace
        pos = skip_chars(src, pos, TOML_WS)

        # 2. Parse rules. Expect one of the following:
        #    - end of file
        #    - end of line
        #    - comment
        #    - key/value pair
        #    - append dict to list (and move to its namespace)
        #    - create dict (and move to its namespace)
        # Skip trailing whitespace when applicable.
        try:
            char = src[pos]
        except IndexError:
            break
        if char == "\n":
            pos += 1
            continue
        if char in KEY_INITIAL_CHARS:
            pos = key_value_rule(src, pos, out, header, parse_float)
            pos = skip_chars(src, pos, TOML_WS)
        elif char == "[":
            try:
                second_char: str | None = src[pos + 1]
            except IndexError:
                second_char = None
            out.flags.finalize_pending()
            if second_char == "[":
                pos, header = create_list_rule(src, pos, out)
            else:
                pos, header = create_dict_rule(src, pos, out)
            pos = skip_chars(src, pos, TOML_WS)
        elif char != "#":
            raise TOMLDecodeError("Invalid statement", src, pos)

        # 3. Skip comment
        pos = skip_comment(src, pos)

        # 4. Expect end of line or end of file
        try:
            char = src[pos]
        except IndexError:
            break
        if char != "\n":
            raise TOMLDecodeError(
                "Expected newline or end of document after a statement", src, pos
            )
        pos += 1

    return out.data.dict


class Flags:
    """Flags that map to parsed keys/namespaces."""

    # Marks an immutable namespace (inline array or inline table).
    FROZEN: Final = 0
    # Marks a nest that has been explicitly created and can no longer
    # be opened using the "[table]" syntax.
    EXPLICIT_NEST: Final = 1

    def __init__(self) -> None:
        self._flags: dict[str, dict] = {}
        self._pending_flags: set[tuple[Key, int]] = set()

    def add_pending(self, key: Key, flag: int) -> None:
        self._pending_flags.add((key, flag))

    def finalize_pending(self) -> None:
        for key, flag in self._pending_flags:
            self.set(key, flag, recursive=False)
        self._pending_flags.clear()

    def unset_all(self, key: Key) -> None:
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return
            cont = cont[k]["nested"]
        cont.pop(key[-1], None)

    def set(self, key: Key, flag: int, *, recursive: bool) -> None:  # noqa: A003
        cont = self._flags
        key_parent, key_stem = key[:-1], key[-1]
        for k in key_parent:
            if k not in cont:
                cont[k] = {"flags": set(), "recursive_flags": set(), "nested": {}}
            cont = cont[k]["nested"]
        if key_stem not in cont:
            cont[key_stem] = {"flags": set(), "recursive_flags": set(), "nested": {}}
        cont[key_stem]["recursive_flags" if recursive else "flags"].add(flag)

    def is_(self, key: Key, flag: int) -> bool:
        if not key:
            return False  # document root has no flags
        cont = self._flags
        for k in key[:-1]:
            if k not in cont:
                return False
            inner_cont = cont[k]
            if flag in inner_cont["recursive_flags"]:
                return True
            cont = inner_cont["nested"]
        key_stem = key[-1]
        if key_stem in cont:
            inner_cont = cont[key_stem]
            return flag in inner_cont["flags"] or flag in inner_cont["recursive_flags"]
        return False


class NestedDict:
    def __init__(self) -> None:
        # The parsed content of the TOML document
        self.dict: dict[str, Any] = {}

    def get_or_create_nest(
        self,
        key: Key,
        *,
        access_lists: bool = True,
    ) -> dict:
        cont: Any = self.dict
        for k in key:
            if k not in cont:
                cont[k] = {}
            cont = cont[k]
            if access_lists and isinstance(cont, list):
                cont = cont[-1]
            if not isinstance(cont, dict):
                raise KeyError("There is no nest behind this key")
        return cont

    def append_nest_to_list(self, key: Key) -> None:
        cont = self.get_or_create_nest(key[:-1])
        last_key = key[-1]
        if last_key in cont:
            list_ = cont[last_key]
            if not isinstance(list_, list):
                raise KeyError("An object other than list found behind this key")
            list_.append({})
        else:
            cont[last_key] = [{}]


class Output(NamedTuple):
    data: NestedDict
    flags: Flags


def skip_chars(src: str, pos: Pos, chars: Iterable[str]) -> Pos:
    try:
        while src[pos] in chars:
            pos += 1
    except IndexError:
        pass
    return pos


def skip_until(
    src: str,
    pos: Pos,
    expect: str,
    *,
    error_on: frozenset[str],
    error_on_eof: bool,
) -> Pos:
    try:
        new_pos = src.index(expect, pos)
    except ValueError:
        new_pos = len(src)
        if error_on_eof:
            raise TOMLDecodeError(f"Expected {expect!r}", src, new_pos) from None

    if not error_on.isdisjoint(src[pos:new_pos]):
        while src[pos] not in error_on:
            pos += 1
        raise TOMLDecodeError(f"Found invalid character {src[pos]!r}", src, pos)
    return new_pos


def skip_comment(src: str, pos: Pos) -> Pos:
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char == "#":
        return skip_until(
            src, pos + 1, "\n", error_on=ILLEGAL_COMMENT_CHARS, error_on_eof=False
        )
    return pos


def skip_comments_and_array_ws(src: str, pos: Pos) -> Pos:
    while True:
        pos_before_skip = pos
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        pos = skip_comment(src, pos)
        if pos == pos_before_skip:
            return pos


def create_dict_rule(src: str, pos: Pos, out: Output) -> tuple[Pos, Key]:
    pos += 1  # Skip "["
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)

    if out.flags.is_(key, Flags.EXPLICIT_NEST) or out.flags.is_(key, Flags.FROZEN):
        raise TOMLDecodeError(f"Cannot declare {key} twice", src, pos)
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.get_or_create_nest(key)
    except KeyError:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None

    if not src.startswith("]", pos):
        raise TOMLDecodeError(
            "Expected ']' at the end of a table declaration", src, pos
        )
    return pos + 1, key


def create_list_rule(src: str, pos: Pos, out: Output) -> tuple[Pos, Key]:
    pos += 2  # Skip "[["
    pos = skip_chars(src, pos, TOML_WS)
    pos, key = parse_key(src, pos)

    if out.flags.is_(key, Flags.FROZEN):
        raise TOMLDecodeError(f"Cannot mutate immutable namespace {key}", src, pos)
    # Free the namespace now that it points to another empty list item...
    out.flags.unset_all(key)
    # ...but this key precisely is still prohibited from table declaration
    out.flags.set(key, Flags.EXPLICIT_NEST, recursive=False)
    try:
        out.data.append_nest_to_list(key)
    except KeyError:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None

    if not src.startswith("]]", pos):
        raise TOMLDecodeError(
            "Expected ']]' at the end of an array declaration", src, pos
        )
    return pos + 2, key


def key_value_rule(
    src: str, pos: Pos, out: Output, header: Key, parse_float: ParseFloat
) -> Pos:
    pos, key, value = parse_key_value_pair(src, pos, parse_float, nest_lvl=0)
    key_parent, key_stem = key[:-1], key[-1]
    abs_key_parent = header + key_parent

    relative_path_cont_keys = (header + key[:i] for i in range(1, len(key)))
    for cont_key in relative_path_cont_keys:
        # Check that dotted key syntax does not redefine an existing table
        if out.flags.is_(cont_key, Flags.EXPLICIT_NEST):
            raise TOMLDecodeError(f"Cannot redefine namespace {cont_key}", src, pos)
        # Containers in the relative path can't be opened with the table syntax or
        # dotted key/value syntax in following table sections.
        out.flags.add_pending(cont_key, Flags.EXPLICIT_NEST)

    if out.flags.is_(abs_key_parent, Flags.FROZEN):
        raise TOMLDecodeError(
            f"Cannot mutate immutable namespace {abs_key_parent}", src, pos
        )

    try:
        nest = out.data.get_or_create_nest(abs_key_parent)
    except KeyError:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None
    if key_stem in nest:
        raise TOMLDecodeError("Cannot overwrite a value", src, pos)
    # Mark inline table and array namespaces recursively immutable
    if isinstance(value, (dict, list)):
        out.flags.set(header + key, Flags.FROZEN, recursive=True)
    nest[key_stem] = value
    return pos


def parse_key_value_pair(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, Key, Any]:
    pos, key = parse_key(src, pos)
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char != "=":
        raise TOMLDecodeError("Expected '=' after a key in a key/value pair", src, pos)
    pos += 1
    pos = skip_chars(src, pos, TOML_WS)
    pos, value = parse_value(src, pos, parse_float, nest_lvl)
    return pos, key, value


def parse_key(src: str, pos: Pos) -> tuple[Pos, Key]:
    pos, key_part = parse_key_part(src, pos)
    key: Key = (key_part,)
    pos = skip_chars(src, pos, TOML_WS)
    while True:
        try:
            char: str | None = src[pos]
        except IndexError:
            char = None
        if char != ".":
            return pos, key
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)
        pos, key_part = parse_key_part(src, pos)
        key += (key_part,)
        pos = skip_chars(src, pos, TOML_WS)


def parse_key_part(src: str, pos: Pos) -> tuple[Pos, str]:
    try:
        char: str | None = src[pos]
    except IndexError:
        char = None
    if char in BARE_KEY_CHARS:
        start_pos = pos
        pos = skip_chars(src, pos, BARE_KEY_CHARS)
        return pos, src[start_pos:pos]
    if char == "'":
        return parse_literal_str(src, pos)
    if char == '"':
        return parse_one_line_basic_str(src, pos)
    raise TOMLDecodeError("Invalid initial character for a key part", src, pos)


def parse_one_line_basic_str(src: str, pos: Pos) -> tuple[Pos, str]:
    pos += 1
    return parse_basic_str(src, pos, multiline=False)


def parse_array(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, list]:
    pos += 1
    array: list = []

    pos = skip_comments_and_array_ws(src, pos)
    if src.startswith("]", pos):
        return pos + 1, array
    while True:
        pos, val = parse_value(src, pos, parse_float, nest_lvl)
        array.append(val)
        pos = skip_comments_and_array_ws(src, pos)

        c = src[pos : pos + 1]
        if c == "]":
            return pos + 1, array
        if c != ",":
            raise TOMLDecodeError("Unclosed array", src, pos)
        pos += 1

        pos = skip_comments_and_array_ws(src, pos)
        if src.startswith("]", pos):
            return pos + 1, array


def parse_inline_table(
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, dict]:
    pos += 1
    nested_dict = NestedDict()
    flags = Flags()

    pos = skip_chars(src, pos, TOML_WS)
    if src.startswith("}", pos):
        return pos + 1, nested_dict.dict
    while True:
        pos, key, value = parse_key_value_pair(src, pos, parse_float, nest_lvl)
        key_parent, key_stem = key[:-1], key[-1]
        if flags.is_(key, Flags.FROZEN):
            raise TOMLDecodeError(f"Cannot mutate immutable namespace {key}", src, pos)
        try:
            nest = nested_dict.get_or_create_nest(key_parent, access_lists=False)
        except KeyError:
            raise TOMLDecodeError("Cannot overwrite a value", src, pos) from None
        if key_stem in nest:
            raise TOMLDecodeError(f"Duplicate inline table key {key_stem!r}", src, pos)
        nest[key_stem] = value
        pos = skip_chars(src, pos, TOML_WS)
        c = src[pos : pos + 1]
        if c == "}":
            return pos + 1, nested_dict.dict
        if c != ",":
            raise TOMLDecodeError("Unclosed inline table", src, pos)
        if isinstance(value, (dict, list)):
            flags.set(key, Flags.FROZEN, recursive=True)
        pos += 1
        pos = skip_chars(src, pos, TOML_WS)


def parse_basic_str_escape(
    src: str, pos: Pos, *, multiline: bool = False
) -> tuple[Pos, str]:
    escape_id = src[pos : pos + 2]
    pos += 2
    if multiline and escape_id in {"\\ ", "\\\t", "\\\n"}:
        # Skip whitespace until next non-whitespace character or end of
        # the doc. Error if non-whitespace is found before newline.
        if escape_id != "\\\n":
            pos = skip_chars(src, pos, TOML_WS)
            try:
                char = src[pos]
            except IndexError:
                return pos, ""
            if char != "\n":
                raise TOMLDecodeError("Unescaped '\\' in a string", src, pos)
            pos += 1
        pos = skip_chars(src, pos, TOML_WS_AND_NEWLINE)
        return pos, ""
    if escape_id == "\\u":
        return parse_hex_char(src, pos, 4)
    if escape_id == "\\U":
        return parse_hex_char(src, pos, 8)
    try:
        return pos, BASIC_STR_ESCAPE_REPLACEMENTS[escape_id]
    except KeyError:
        raise TOMLDecodeError("Unescaped '\\' in a string", src, pos) from None


def parse_basic_str_escape_multiline(src: str, pos: Pos) -> tuple[Pos, str]:
    return parse_basic_str_escape(src, pos, multiline=True)


def parse_hex_char(src: str, pos: Pos, hex_len: int) -> tuple[Pos, str]:
    hex_str = src[pos : pos + hex_len]
    if len(hex_str) != hex_len or not HEXDIGIT_CHARS.issuperset(hex_str):
        raise TOMLDecodeError("Invalid hex value", src, pos)
    pos += hex_len
    hex_int = int(hex_str, 16)
    if not is_unicode_scalar_value(hex_int):
        raise TOMLDecodeError(
            "Escaped character is not a Unicode scalar value", src, pos
        )
    return pos, chr(hex_int)


def parse_literal_str(src: str, pos: Pos) -> tuple[Pos, str]:
    pos += 1  # Skip starting apostrophe
    start_pos = pos
    pos = skip_until(
        src, pos, "'", error_on=ILLEGAL_LITERAL_STR_CHARS, error_on_eof=True
    )
    return pos + 1, src[start_pos:pos]  # Skip ending apostrophe


def parse_multiline_str(src: str, pos: Pos, *, literal: bool) -> tuple[Pos, str]:
    pos += 3
    if src.startswith("\n", pos):
        pos += 1

    if literal:
        delim = "'"
        end_pos = skip_until(
            src,
            pos,
            "'''",
            error_on=ILLEGAL_MULTILINE_LITERAL_STR_CHARS,
            error_on_eof=True,
        )
        result = src[pos:end_pos]
        pos = end_pos + 3
    else:
        delim = '"'
        pos, result = parse_basic_str(src, pos, multiline=True)

    # Add at maximum two extra apostrophes/quotes if the end sequence
    # is 4 or 5 chars long instead of just 3.
    if not src.startswith(delim, pos):
        return pos, result
    pos += 1
    if not src.startswith(delim, pos):
        return pos, result + delim
    pos += 1
    return pos, result + (delim * 2)


def parse_basic_str(src: str, pos: Pos, *, multiline: bool) -> tuple[Pos, str]:
    if multiline:
        error_on = ILLEGAL_MULTILINE_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape_multiline
    else:
        error_on = ILLEGAL_BASIC_STR_CHARS
        parse_escapes = parse_basic_str_escape
    result = ""
    start_pos = pos
    while True:
        try:
            char = src[pos]
        except IndexError:
            raise TOMLDecodeError("Unterminated string", src, pos) from None
        if char == '"':
            if not multiline:
                return pos + 1, result + src[start_pos:pos]
            if src.startswith('"""', pos):
                return pos + 3, result + src[start_pos:pos]
            pos += 1
            continue
        if char == "\\":
            result += src[start_pos:pos]
            pos, parsed_escape = parse_escapes(src, pos)
            result += parsed_escape
            start_pos = pos
            continue
        if char in error_on:
            raise TOMLDecodeError(f"Illegal character {char!r}", src, pos)
        pos += 1


def parse_value(  # noqa: C901
    src: str, pos: Pos, parse_float: ParseFloat, nest_lvl: int
) -> tuple[Pos, Any]:
    if nest_lvl > MAX_INLINE_NESTING:
        # Pure Python should have raised RecursionError already.
        # This ensures mypyc binaries eventually do the same.
        raise RecursionError(  # pragma: no cover
            "TOML inline arrays/tables are nested more than the allowed"
            f" {MAX_INLINE_NESTING} levels"
        )

    try:
        char: str | None = src[pos]
    except IndexError:
        char = None

    # IMPORTANT: order conditions based on speed of checking and likelihood

    # Basic strings
    if char == '"':
        if src.startswith('"""', pos):
            return parse_multiline_str(src, pos, literal=False)
        return parse_one_line_basic_str(src, pos)

    # Literal strings
    if char == "'":
        if src.startswith("'''", pos):
            return parse_multiline_str(src, pos, literal=True)
        return parse_literal_str(src, pos)

    # Booleans
    if char == "t":
        if src.startswith("true", pos):
            return pos + 4, True
    if char == "f":
        if src.startswith("false", pos):
            return pos + 5, False

    # Arrays
    if char == "[":
        return parse_array(src, pos, parse_float, nest_lvl + 1)

    # Inline tables
    if char == "{":
        return parse_inline_table(src, pos, parse_float, nest_lvl + 1)

    # Dates and times
    datetime_match = RE_DATETIME.match(src, pos)
    if datetime_match:
        try:
            datetime_obj = match_to_datetime(datetime_match)
        except ValueError as e:
            raise TOMLDecodeError("Invalid date or datetime", src, pos) from e
        return datetime_match.end(), datetime_obj
    localtime_match = RE_LOCALTIME.match(src, pos)
    if localtime_match:
        return localtime_match.end(), match_to_localtime(localtime_match)

    # Integers and "normal" floats.
    # The regex will greedily match any type starting with a decimal
    # char, so needs to be located after handling of dates and times.
    number_match = RE_NUMBER.match(src, pos)
    if number_match:
        return number_match.end(), match_to_number(number_match, parse_float)

    # Special floats
    first_three = src[pos : pos + 3]
    if first_three in {"inf", "nan"}:
        return pos + 3, parse_float(first_three)
    first_four = src[pos : pos + 4]
    if first_four in {"-inf", "+inf", "-nan", "+nan"}:
        return pos + 4, parse_float(first_four)

    raise TOMLDecodeError("Invalid value", src, pos)


def is_unicode_scalar_value(codepoint: int) -> bool:
    return (0 <= codepoint <= 55295) or (57344 <= codepoint <= 1114111)


def make_safe_parse_float(parse_float: ParseFloat) -> ParseFloat:
    """A decorator to make `parse_float` safe.

    `parse_float` must not return dicts or lists, because these types
    would be mixed with parsed TOML tables and arrays, thus confusing
    the parser. The returned decorated callable raises `ValueError`
    instead of returning illegal types.
    """
    # The default `float` callable never returns illegal types. Optimize it.
    if parse_float is float:
        return float

    def safe_parse_float(float_str: str) -> Any:
        float_value = parse_float(float_str)
        if isinstance(float_value, (dict, list)):
            raise ValueError("parse_float must not return dicts or lists")
        return float_value

    return safe_parse_float

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_subprocess.py ===
from __future__ import annotations

import gc
import os
import random
import signal
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from pathlib import Path as SyncPath
from signal import Signals
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
)
from unittest import mock

import pytest

import trio
from trio.testing import Matcher, RaisesGroup

from .. import (
    Event,
    Process,
    _core,
    fail_after,
    move_on_after,
    run_process,
    sleep,
    sleep_forever,
)
from .._core._tests.tutil import skip_if_fbsd_pipes_broken, slow
from ..lowlevel import open_process
from ..testing import MockClock, assert_no_checkpoints, wait_all_tasks_blocked

if TYPE_CHECKING:
    from types import FrameType

    from typing_extensions import TypeAlias

    from .._abc import ReceiveStream

if sys.platform == "win32":
    SignalType: TypeAlias = None
else:
    SignalType: TypeAlias = Signals

SIGKILL: SignalType
SIGTERM: SignalType
SIGUSR1: SignalType

posix = os.name == "posix"
if (not TYPE_CHECKING and posix) or sys.platform != "win32":
    from signal import SIGKILL, SIGTERM, SIGUSR1
else:
    SIGKILL, SIGTERM, SIGUSR1 = None, None, None


# Since Windows has very few command-line utilities generally available,
# all of our subprocesses are Python processes running short bits of
# (mostly) cross-platform code.
def python(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", "import sys; " + code]


EXIT_TRUE = python("sys.exit(0)")
EXIT_FALSE = python("sys.exit(1)")
CAT = python("sys.stdout.buffer.write(sys.stdin.buffer.read())")

if posix:

    def SLEEP(seconds: int) -> list[str]:
        return ["sleep", str(seconds)]

else:

    def SLEEP(seconds: int) -> list[str]:
        return python(f"import time; time.sleep({seconds})")


@asynccontextmanager
async def open_process_then_kill(  # type: ignore[misc, explicit-any]
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[Process]:
    proc = await open_process(*args, **kwargs)
    try:
        yield proc
    finally:
        proc.kill()
        await proc.wait()


@asynccontextmanager
async def run_process_in_nursery(  # type: ignore[misc, explicit-any]
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[Process]:
    async with _core.open_nursery() as nursery:
        kwargs.setdefault("check", False)
        value = await nursery.start(partial(run_process, *args, **kwargs))
        assert isinstance(value, Process)
        proc: Process = value
        yield proc
        nursery.cancel_scope.cancel()


background_process_param = pytest.mark.parametrize(
    "background_process",
    [open_process_then_kill, run_process_in_nursery],
    ids=["open_process", "run_process in nursery"],
)

BackgroundProcessType: TypeAlias = Callable[  # type: ignore[explicit-any]
    ...,
    AbstractAsyncContextManager[Process],
]


@background_process_param
async def test_basic(background_process: BackgroundProcessType) -> None:
    async with background_process(EXIT_TRUE) as proc:
        await proc.wait()
    assert isinstance(proc, Process)
    assert proc._pidfd is None
    assert proc.returncode == 0
    assert repr(proc) == f"<trio.Process {EXIT_TRUE}: exited with status 0>"

    async with background_process(EXIT_FALSE) as proc:
        await proc.wait()
    assert proc.returncode == 1
    assert repr(proc) == "<trio.Process {!r}: {}>".format(
        EXIT_FALSE,
        "exited with status 1",
    )


@background_process_param
async def test_basic_no_pidfd(background_process: BackgroundProcessType) -> None:
    with mock.patch("trio._subprocess.can_try_pidfd_open", new=False):
        async with background_process(EXIT_TRUE) as proc:
            assert proc._pidfd is None
            await proc.wait()
        assert isinstance(proc, Process)
        assert proc._pidfd is None
        assert proc.returncode == 0
        assert repr(proc) == f"<trio.Process {EXIT_TRUE}: exited with status 0>"

        async with background_process(EXIT_FALSE) as proc:
            await proc.wait()
        assert proc.returncode == 1
        assert repr(proc) == "<trio.Process {!r}: {}>".format(
            EXIT_FALSE,
            "exited with status 1",
        )


@background_process_param
async def test_auto_update_returncode(
    background_process: BackgroundProcessType,
) -> None:
    async with background_process(SLEEP(9999)) as p:
        assert p.returncode is None
        assert "running" in repr(p)
        p.kill()
        p._proc.wait()
        assert p.returncode is not None
        assert "exited" in repr(p)
        assert p._pidfd is None
        assert p.returncode is not None


@background_process_param
async def test_multi_wait(background_process: BackgroundProcessType) -> None:
    async with background_process(SLEEP(10)) as proc:
        # Check that wait (including multi-wait) tolerates being cancelled
        async with _core.open_nursery() as nursery:
            nursery.start_soon(proc.wait)
            nursery.start_soon(proc.wait)
            nursery.start_soon(proc.wait)
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()

        # Now try waiting for real
        async with _core.open_nursery() as nursery:
            nursery.start_soon(proc.wait)
            nursery.start_soon(proc.wait)
            nursery.start_soon(proc.wait)
            await wait_all_tasks_blocked()
            proc.kill()


@background_process_param
async def test_multi_wait_no_pidfd(background_process: BackgroundProcessType) -> None:
    with mock.patch("trio._subprocess.can_try_pidfd_open", new=False):
        async with background_process(SLEEP(10)) as proc:
            # Check that wait (including multi-wait) tolerates being cancelled
            async with _core.open_nursery() as nursery:
                nursery.start_soon(proc.wait)
                nursery.start_soon(proc.wait)
                nursery.start_soon(proc.wait)
                await wait_all_tasks_blocked()
                nursery.cancel_scope.cancel()

            # Now try waiting for real
            async with _core.open_nursery() as nursery:
                nursery.start_soon(proc.wait)
                nursery.start_soon(proc.wait)
                nursery.start_soon(proc.wait)
                await wait_all_tasks_blocked()
                proc.kill()


COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR = python(
    "data = sys.stdin.buffer.read(); "
    "sys.stdout.buffer.write(data); "
    "sys.stderr.buffer.write(data[::-1])",
)


@background_process_param
async def test_pipes(background_process: BackgroundProcessType) -> None:
    async with background_process(
        COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        msg = b"the quick brown fox jumps over the lazy dog"

        async def feed_input() -> None:
            assert proc.stdin is not None
            await proc.stdin.send_all(msg)
            await proc.stdin.aclose()

        async def check_output(stream: ReceiveStream, expected: bytes) -> None:
            seen = bytearray()
            async for chunk in stream:
                seen += chunk
            assert seen == expected

        assert proc.stdout is not None
        assert proc.stderr is not None

        async with _core.open_nursery() as nursery:
            # fail eventually if something is broken
            nursery.cancel_scope.deadline = _core.current_time() + 30.0
            nursery.start_soon(feed_input)
            nursery.start_soon(check_output, proc.stdout, msg)
            nursery.start_soon(check_output, proc.stderr, msg[::-1])

        assert not nursery.cancel_scope.cancelled_caught
        assert await proc.wait() == 0


@background_process_param
async def test_interactive(background_process: BackgroundProcessType) -> None:
    # Test some back-and-forth with a subprocess. This one works like so:
    # in: 32\n
    # out: 0000...0000\n (32 zeroes)
    # err: 1111...1111\n (64 ones)
    # in: 10\n
    # out: 2222222222\n (10 twos)
    # err: 3333....3333\n (20 threes)
    # in: EOF
    # out: EOF
    # err: EOF

    async with background_process(
        python(
            "idx = 0\n"
            "while True:\n"
            "    line = sys.stdin.readline()\n"
            "    if line == '': break\n"
            "    request = int(line.strip())\n"
            "    print(str(idx * 2) * request)\n"
            "    print(str(idx * 2 + 1) * request * 2, file=sys.stderr)\n"
            "    idx += 1\n",
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        newline = b"\n" if posix else b"\r\n"

        async def expect(idx: int, request: int) -> None:
            async with _core.open_nursery() as nursery:

                async def drain_one(
                    stream: ReceiveStream,
                    count: int,
                    digit: int,
                ) -> None:
                    while count > 0:
                        result = await stream.receive_some(count)
                        assert result == (f"{digit}".encode() * len(result))
                        count -= len(result)
                    assert count == 0
                    assert await stream.receive_some(len(newline)) == newline

                assert proc.stdout is not None
                assert proc.stderr is not None
                nursery.start_soon(drain_one, proc.stdout, request, idx * 2)
                nursery.start_soon(drain_one, proc.stderr, request * 2, idx * 2 + 1)

        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        with fail_after(5):
            await proc.stdin.send_all(b"12")
            await sleep(0.1)
            await proc.stdin.send_all(b"345" + newline)
            await expect(0, 12345)
            await proc.stdin.send_all(b"100" + newline + b"200" + newline)
            await expect(1, 100)
            await expect(2, 200)
            await proc.stdin.send_all(b"0" + newline)
            await expect(3, 0)
            await proc.stdin.send_all(b"999999")
            with move_on_after(0.1) as scope:
                await expect(4, 0)
            assert scope.cancelled_caught
            await proc.stdin.send_all(newline)
            await expect(4, 999999)
            await proc.stdin.aclose()
            assert await proc.stdout.receive_some(1) == b""
            assert await proc.stderr.receive_some(1) == b""
            await proc.wait()

    assert proc.returncode == 0


async def test_run() -> None:
    data = bytes(random.randint(0, 255) for _ in range(2**18))

    result = await run_process(
        CAT,
        stdin=data,
        capture_stdout=True,
        capture_stderr=True,
    )
    assert result.args == CAT
    assert result.returncode == 0
    assert result.stdout == data
    assert result.stderr == b""

    result = await run_process(CAT, capture_stdout=True)
    assert result.args == CAT
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr is None

    result = await run_process(
        COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR,
        stdin=data,
        capture_stdout=True,
        capture_stderr=True,
    )
    assert result.args == COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR
    assert result.returncode == 0
    assert result.stdout == data
    assert result.stderr == data[::-1]

    # invalid combinations
    with pytest.raises(UnicodeError):
        await run_process(CAT, stdin="oh no, it's text")

    pipe_stdout_error = r"^stdout=subprocess\.PIPE is only valid with nursery\.start, since that's the only way to access the pipe(; use nursery\.start or pass the data you want to write directly)*$"
    with pytest.raises(ValueError, match=pipe_stdout_error):
        await run_process(CAT, stdin=subprocess.PIPE)
    with pytest.raises(ValueError, match=pipe_stdout_error):
        await run_process(CAT, stdout=subprocess.PIPE)
    with pytest.raises(
        ValueError,
        match=pipe_stdout_error.replace("stdout", "stderr", 1),
    ):
        await run_process(CAT, stderr=subprocess.PIPE)
    with pytest.raises(
        ValueError,
        match=r"^can't specify both stdout and capture_stdout$",
    ):
        await run_process(CAT, capture_stdout=True, stdout=subprocess.DEVNULL)
    with pytest.raises(
        ValueError,
        match=r"^can't specify both stderr and capture_stderr$",
    ):
        await run_process(CAT, capture_stderr=True, stderr=None)


async def test_run_check() -> None:
    cmd = python("sys.stderr.buffer.write(b'test\\n'); sys.exit(1)")
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        await run_process(cmd, stdin=subprocess.DEVNULL, capture_stderr=True)
    assert excinfo.value.cmd == cmd
    assert excinfo.value.returncode == 1
    assert excinfo.value.stderr == b"test\n"
    assert excinfo.value.stdout is None

    result = await run_process(
        cmd,
        capture_stdout=True,
        capture_stderr=True,
        check=False,
    )
    assert result.args == cmd
    assert result.stdout == b""
    assert result.stderr == b"test\n"
    assert result.returncode == 1


@skip_if_fbsd_pipes_broken
async def test_run_with_broken_pipe() -> None:
    result = await run_process(
        [sys.executable, "-c", "import sys; sys.stdin.close()"],
        stdin=b"x" * 131072,
    )
    assert result.returncode == 0
    assert result.stdout is result.stderr is None


@background_process_param
async def test_stderr_stdout(background_process: BackgroundProcessType) -> None:
    async with background_process(
        COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        assert proc.stdio is not None
        assert proc.stdout is not None
        assert proc.stderr is None
        await proc.stdio.send_all(b"1234")
        await proc.stdio.send_eof()

        output = []
        while True:
            chunk = await proc.stdio.receive_some(16)
            if chunk == b"":
                break
            output.append(chunk)
        assert b"".join(output) == b"12344321"
    assert proc.returncode == 0

    # equivalent test with run_process()
    result = await run_process(
        COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR,
        stdin=b"1234",
        capture_stdout=True,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0
    assert result.stdout == b"12344321"
    assert result.stderr is None

    # this one hits the branch where stderr=STDOUT but stdout
    # is not redirected
    async with background_process(
        CAT,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        assert proc.stdout is None
        assert proc.stderr is None
        await proc.stdin.aclose()
        await proc.wait()
    assert proc.returncode == 0

    if posix:
        try:
            r, w = os.pipe()

            async with background_process(
                COPY_STDIN_TO_STDOUT_AND_BACKWARD_TO_STDERR,
                stdin=subprocess.PIPE,
                stdout=w,
                stderr=subprocess.STDOUT,
            ) as proc:
                os.close(w)
                assert proc.stdio is None
                assert proc.stdout is None
                assert proc.stderr is None
                await proc.stdin.send_all(b"1234")
                await proc.stdin.aclose()
                assert await proc.wait() == 0
                assert os.read(r, 4096) == b"12344321"
                assert os.read(r, 4096) == b""
        finally:
            os.close(r)


async def test_errors() -> None:
    with pytest.raises(TypeError) as excinfo:
        # call-overload on unix, call-arg on windows
        await open_process(["ls"], encoding="utf-8")  # type: ignore
    assert "unbuffered byte streams" in str(excinfo.value)
    assert "the 'encoding' option is not supported" in str(excinfo.value)

    if posix:
        with pytest.raises(TypeError) as excinfo:
            await open_process(["ls"], shell=True)
        with pytest.raises(TypeError) as excinfo:
            await open_process("ls", shell=False)


@background_process_param
async def test_signals(background_process: BackgroundProcessType) -> None:
    async def test_one_signal(
        send_it: Callable[[Process], None],
        signum: signal.Signals | None,
    ) -> None:
        with move_on_after(1.0) as scope:
            async with background_process(SLEEP(3600)) as proc:
                send_it(proc)
                await proc.wait()
        assert not scope.cancelled_caught
        if posix:
            assert signum is not None
            assert proc.returncode == -signum
        else:
            assert proc.returncode != 0

    await test_one_signal(Process.kill, SIGKILL)
    await test_one_signal(Process.terminate, SIGTERM)
    # Test that we can send arbitrary signals.
    #
    # We used to use SIGINT here, but it turns out that the Python interpreter
    # has race conditions that can cause it to explode in weird ways if it
    # tries to handle SIGINT during startup. SIGUSR1's default disposition is
    # to terminate the target process, and Python doesn't try to do anything
    # clever to handle it.
    if (not TYPE_CHECKING and posix) or sys.platform != "win32":
        await test_one_signal(lambda proc: proc.send_signal(SIGUSR1), SIGUSR1)


@pytest.mark.skipif(not posix, reason="POSIX specific")
@background_process_param
async def test_wait_reapable_fails(background_process: BackgroundProcessType) -> None:
    if TYPE_CHECKING and sys.platform == "win32":
        return
    old_sigchld = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    try:
        # With SIGCHLD disabled, the wait() syscall will wait for the
        # process to exit but then fail with ECHILD. Make sure we
        # support this case as the stdlib subprocess module does.
        async with background_process(SLEEP(3600)) as proc:
            async with _core.open_nursery() as nursery:
                nursery.start_soon(proc.wait)
                await wait_all_tasks_blocked()
                proc.kill()
                nursery.cancel_scope.deadline = _core.current_time() + 1.0
            assert not nursery.cancel_scope.cancelled_caught
            assert proc.returncode == 0  # exit status unknowable, so...
    finally:
        signal.signal(signal.SIGCHLD, old_sigchld)


@pytest.mark.skipif(not posix, reason="POSIX specific")
@background_process_param
async def test_wait_reapable_fails_no_pidfd(
    background_process: BackgroundProcessType,
) -> None:
    if TYPE_CHECKING and sys.platform == "win32":
        return
    with mock.patch("trio._subprocess.can_try_pidfd_open", new=False):
        old_sigchld = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        try:
            # With SIGCHLD disabled, the wait() syscall will wait for the
            # process to exit but then fail with ECHILD. Make sure we
            # support this case as the stdlib subprocess module does.
            async with background_process(SLEEP(3600)) as proc:
                async with _core.open_nursery() as nursery:
                    nursery.start_soon(proc.wait)
                    await wait_all_tasks_blocked()
                    proc.kill()
                    nursery.cancel_scope.deadline = _core.current_time() + 1.0
                assert not nursery.cancel_scope.cancelled_caught
                assert proc.returncode == 0  # exit status unknowable, so...
        finally:
            signal.signal(signal.SIGCHLD, old_sigchld)


@slow
def test_waitid_eintr() -> None:
    # This only matters on PyPy (where we're coding EINTR handling
    # ourselves) but the test works on all waitid platforms.
    from .._subprocess_platform import wait_child_exiting

    if TYPE_CHECKING and (sys.platform == "win32" or sys.platform == "darwin"):
        return

    if not wait_child_exiting.__module__.endswith("waitid"):
        pytest.skip("waitid only")

    # despite the TYPE_CHECKING early return silencing warnings about signal.SIGALRM etc
    # this import is still checked on win32&darwin and raises [attr-defined].
    # Linux doesn't raise [attr-defined] though, so we need [unused-ignore]
    from .._subprocess_platform.waitid import (  # type: ignore[attr-defined, unused-ignore]
        sync_wait_reapable,
    )

    got_alarm = False
    sleeper = subprocess.Popen(["sleep", "3600"])

    def on_alarm(sig: int, frame: FrameType | None) -> None:
        nonlocal got_alarm
        got_alarm = True
        sleeper.kill()

    old_sigalrm = signal.signal(signal.SIGALRM, on_alarm)
    try:
        signal.alarm(1)
        sync_wait_reapable(sleeper.pid)
        assert sleeper.wait(timeout=1) == -9
    finally:
        if sleeper.returncode is None:  # pragma: no cover
            # We only get here if something fails in the above;
            # if the test passes, wait() will reap the process
            sleeper.kill()
            sleeper.wait()
        signal.signal(signal.SIGALRM, old_sigalrm)


async def test_custom_deliver_cancel() -> None:
    custom_deliver_cancel_called = False

    async def custom_deliver_cancel(proc: Process) -> None:
        nonlocal custom_deliver_cancel_called
        custom_deliver_cancel_called = True
        proc.terminate()
        # Make sure this does get cancelled when the process exits, and that
        # the process really exited.
        try:
            await sleep_forever()
        finally:
            assert proc.returncode is not None

    async with _core.open_nursery() as nursery:
        nursery.start_soon(
            partial(run_process, SLEEP(9999), deliver_cancel=custom_deliver_cancel),
        )
        await wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()

    assert custom_deliver_cancel_called


def test_bad_deliver_cancel() -> None:
    async def custom_deliver_cancel(proc: Process) -> None:
        proc.terminate()
        raise ValueError("foo")

    async def do_stuff() -> None:
        async with _core.open_nursery() as nursery:
            nursery.start_soon(
                partial(run_process, SLEEP(9999), deliver_cancel=custom_deliver_cancel),
            )
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()

    # double wrap from our nursery + the internal nursery
    with RaisesGroup(RaisesGroup(Matcher(ValueError, "^foo$"))):
        _core.run(do_stuff, strict_exception_groups=True)


async def test_warn_on_failed_cancel_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    original_terminate = Process.terminate

    def broken_terminate(self: Process) -> NoReturn:
        original_terminate(self)
        raise OSError("whoops")

    monkeypatch.setattr(Process, "terminate", broken_terminate)

    with pytest.warns(RuntimeWarning, match=".*whoops.*"):  # noqa: PT031
        async with _core.open_nursery() as nursery:
            nursery.start_soon(run_process, SLEEP(9999))
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()


@pytest.mark.skipif(not posix, reason="posix only")
async def test_warn_on_cancel_SIGKILL_escalation(
    autojump_clock: MockClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Process, "terminate", lambda *args: None)

    with pytest.warns(RuntimeWarning, match=".*ignored SIGTERM.*"):  # noqa: PT031
        async with _core.open_nursery() as nursery:
            nursery.start_soon(run_process, SLEEP(9999))
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()


# the background_process_param exercises a lot of run_process cases, but it uses
# check=False, so lets have a test that uses check=True as well
async def test_run_process_background_fail() -> None:
    with RaisesGroup(subprocess.CalledProcessError):
        async with _core.open_nursery() as nursery:
            value = await nursery.start(run_process, EXIT_FALSE)
            assert isinstance(value, Process)
            proc: Process = value
    assert proc.returncode == 1


@pytest.mark.skipif(
    not SyncPath("/dev/fd").exists(),
    reason="requires a way to iterate through open files",
)
async def test_for_leaking_fds() -> None:
    gc.collect()  # address possible flakiness on PyPy

    starting_fds = set(SyncPath("/dev/fd").iterdir())
    await run_process(EXIT_TRUE)
    assert set(SyncPath("/dev/fd").iterdir()) == starting_fds

    with pytest.raises(subprocess.CalledProcessError):
        await run_process(EXIT_FALSE)
    assert set(SyncPath("/dev/fd").iterdir()) == starting_fds

    with pytest.raises(PermissionError):
        await run_process(["/dev/fd/0"])
    assert set(SyncPath("/dev/fd").iterdir()) == starting_fds


async def test_run_process_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # There's probably less extreme ways of triggering errors inside the nursery
    # in run_process.
    async def very_broken_open(*args: object, **kwargs: object) -> str:
        return "oops"

    monkeypatch.setattr(trio._subprocess, "open_process", very_broken_open)
    with RaisesGroup(AttributeError, AttributeError):
        await run_process(EXIT_TRUE, capture_stdout=True)


# regression test for #2209
async def test_subprocess_pidfd_unnotified() -> None:
    noticed_exit = None

    async def wait_and_tell(proc: Process) -> None:
        nonlocal noticed_exit
        noticed_exit = Event()
        await proc.wait()
        noticed_exit.set()

    proc = await open_process(SLEEP(9999))
    async with _core.open_nursery() as nursery:
        nursery.start_soon(wait_and_tell, proc)
        await wait_all_tasks_blocked()
        assert isinstance(noticed_exit, Event)
        proc.terminate()
        # without giving trio a chance to do so,
        with assert_no_checkpoints():
            # wait until the process has actually exited;
            proc._proc.wait()
            # force a call to poll (that closes the pidfd on linux)
            proc.poll()
        with move_on_after(5):
            # Some platforms use threads to wait for exit, so it might take a bit
            # for everything to notice
            await noticed_exit.wait()
        assert noticed_exit.is_set(), "child task wasn't woken after poll, DEADLOCK"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\utils.py ===
import enum
import json
import os
import re
import typing as t
from collections import abc
from collections import deque
from random import choice
from random import randrange
from threading import Lock
from types import CodeType
from urllib.parse import quote_from_bytes

import markupsafe

if t.TYPE_CHECKING:
    import typing_extensions as te

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


class _MissingType:
    def __repr__(self) -> str:
        return "missing"

    def __reduce__(self) -> str:
        return "missing"


missing: t.Any = _MissingType()
"""Special singleton representing missing values for the runtime."""

internal_code: t.MutableSet[CodeType] = set()

concat = "".join


def pass_context(f: F) -> F:
    """Pass the :class:`~jinja2.runtime.Context` as the first argument
    to the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    If only ``Context.eval_context`` is needed, use
    :func:`pass_eval_context`. If only ``Context.environment`` is
    needed, use :func:`pass_environment`.

    .. versionadded:: 3.0.0
        Replaces ``contextfunction`` and ``contextfilter``.
    """
    f.jinja_pass_arg = _PassArg.context  # type: ignore
    return f


def pass_eval_context(f: F) -> F:
    """Pass the :class:`~jinja2.nodes.EvalContext` as the first argument
    to the decorated function when called while rendering a template.
    See :ref:`eval-context`.

    Can be used on functions, filters, and tests.

    If only ``EvalContext.environment`` is needed, use
    :func:`pass_environment`.

    .. versionadded:: 3.0.0
        Replaces ``evalcontextfunction`` and ``evalcontextfilter``.
    """
    f.jinja_pass_arg = _PassArg.eval_context  # type: ignore
    return f


def pass_environment(f: F) -> F:
    """Pass the :class:`~jinja2.Environment` as the first argument to
    the decorated function when called while rendering a template.

    Can be used on functions, filters, and tests.

    .. versionadded:: 3.0.0
        Replaces ``environmentfunction`` and ``environmentfilter``.
    """
    f.jinja_pass_arg = _PassArg.environment  # type: ignore
    return f


class _PassArg(enum.Enum):
    context = enum.auto()
    eval_context = enum.auto()
    environment = enum.auto()

    @classmethod
    def from_obj(cls, obj: F) -> t.Optional["_PassArg"]:
        if hasattr(obj, "jinja_pass_arg"):
            return obj.jinja_pass_arg  # type: ignore

        return None


def internalcode(f: F) -> F:
    """Marks the function as internally used"""
    internal_code.add(f.__code__)
    return f


def is_undefined(obj: t.Any) -> bool:
    """Check if the object passed is undefined.  This does nothing more than
    performing an instance check against :class:`Undefined` but looks nicer.
    This can be used for custom filters or tests that want to react to
    undefined variables.  For example a custom default filter can look like
    this::

        def default(var, default=''):
            if is_undefined(var):
                return default
            return var
    """
    from .runtime import Undefined

    return isinstance(obj, Undefined)


def consume(iterable: t.Iterable[t.Any]) -> None:
    """Consumes an iterable without doing anything with it."""
    for _ in iterable:
        pass


def clear_caches() -> None:
    """Jinja keeps internal caches for environments and lexers.  These are
    used so that Jinja doesn't have to recreate environments and lexers all
    the time.  Normally you don't have to care about that but if you are
    measuring memory consumption you may want to clean the caches.
    """
    from .environment import get_spontaneous_environment
    from .lexer import _lexer_cache

    get_spontaneous_environment.cache_clear()
    _lexer_cache.clear()


def import_string(import_name: str, silent: bool = False) -> t.Any:
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If the `silent` is True the return value will be `None` if the import
    fails.

    :return: imported object
    """
    try:
        if ":" in import_name:
            module, obj = import_name.split(":", 1)
        elif "." in import_name:
            module, _, obj = import_name.rpartition(".")
        else:
            return __import__(import_name)
        return getattr(__import__(module, None, None, [obj]), obj)
    except (ImportError, AttributeError):
        if not silent:
            raise


def open_if_exists(filename: str, mode: str = "rb") -> t.Optional[t.IO[t.Any]]:
    """Returns a file descriptor for the filename if that file exists,
    otherwise ``None``.
    """
    if not os.path.isfile(filename):
        return None

    return open(filename, mode)


def object_type_repr(obj: t.Any) -> str:
    """Returns the name of the object's type.  For some recognized
    singletons the name of the object is returned instead. (For
    example for `None` and `Ellipsis`).
    """
    if obj is None:
        return "None"
    elif obj is Ellipsis:
        return "Ellipsis"

    cls = type(obj)

    if cls.__module__ == "builtins":
        return f"{cls.__name__} object"

    return f"{cls.__module__}.{cls.__name__} object"


def pformat(obj: t.Any) -> str:
    """Format an object using :func:`pprint.pformat`."""
    from pprint import pformat

    return pformat(obj)


_http_re = re.compile(
    r"""
    ^
    (
        (https?://|www\.)  # scheme or www
        (([\w%-]+\.)+)?  # subdomain
        (
            [a-z]{2,63}  # basic tld
        |
            xn--[\w%]{2,59}  # idna tld
        )
    |
        ([\w%-]{2,63}\.)+  # basic domain
        (com|net|int|edu|gov|org|info|mil)  # basic tld
    |
        (https?://)  # scheme
        (
            (([\d]{1,3})(\.[\d]{1,3}){3})  # IPv4
        |
            (\[([\da-f]{0,4}:){2}([\da-f]{0,4}:?){1,6}])  # IPv6
        )
    )
    (?::[\d]{1,5})?  # port
    (?:[/?#]\S*)?  # path, query, and fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
_email_re = re.compile(r"^\S+@\w[\w.-]*\.\w+$")


def urlize(
    text: str,
    trim_url_limit: t.Optional[int] = None,
    rel: t.Optional[str] = None,
    target: t.Optional[str] = None,
    extra_schemes: t.Optional[t.Iterable[str]] = None,
) -> str:
    """Convert URLs in text into clickable links.

    This may not recognize links in some situations. Usually, a more
    comprehensive formatter, such as a Markdown library, is a better
    choice.

    Works on ``http://``, ``https://``, ``www.``, ``mailto:``, and email
    addresses. Links with trailing punctuation (periods, commas, closing
    parentheses) and leading punctuation (opening parentheses) are
    recognized excluding the punctuation. Email addresses that include
    header fields are not recognized (for example,
    ``mailto:address@example.com?cc=copy@example.com``).

    :param text: Original text containing URLs to link.
    :param trim_url_limit: Shorten displayed URL values to this length.
    :param target: Add the ``target`` attribute to links.
    :param rel: Add the ``rel`` attribute to links.
    :param extra_schemes: Recognize URLs that start with these schemes
        in addition to the default behavior.

    .. versionchanged:: 3.0
        The ``extra_schemes`` parameter was added.

    .. versionchanged:: 3.0
        Generate ``https://`` links for URLs without a scheme.

    .. versionchanged:: 3.0
        The parsing rules were updated. Recognize email addresses with
        or without the ``mailto:`` scheme. Validate IP addresses. Ignore
        parentheses and brackets in more cases.
    """
    if trim_url_limit is not None:

        def trim_url(x: str) -> str:
            if len(x) > trim_url_limit:
                return f"{x[:trim_url_limit]}..."

            return x

    else:

        def trim_url(x: str) -> str:
            return x

    words = re.split(r"(\s+)", str(markupsafe.escape(text)))
    rel_attr = f' rel="{markupsafe.escape(rel)}"' if rel else ""
    target_attr = f' target="{markupsafe.escape(target)}"' if target else ""

    for i, word in enumerate(words):
        head, middle, tail = "", word, ""
        match = re.match(r"^([(<]|&lt;)+", middle)

        if match:
            head = match.group()
            middle = middle[match.end() :]

        # Unlike lead, which is anchored to the start of the string,
        # need to check that the string ends with any of the characters
        # before trying to match all of them, to avoid backtracking.
        if middle.endswith((")", ">", ".", ",", "\n", "&gt;")):
            match = re.search(r"([)>.,\n]|&gt;)+$", middle)

            if match:
                tail = match.group()
                middle = middle[: match.start()]

        # Prefer balancing parentheses in URLs instead of ignoring a
        # trailing character.
        for start_char, end_char in ("(", ")"), ("<", ">"), ("&lt;", "&gt;"):
            start_count = middle.count(start_char)

            if start_count <= middle.count(end_char):
                # Balanced, or lighter on the left
                continue

            # Move as many as possible from the tail to balance
            for _ in range(min(start_count, tail.count(end_char))):
                end_index = tail.index(end_char) + len(end_char)
                # Move anything in the tail before the end char too
                middle += tail[:end_index]
                tail = tail[end_index:]

        if _http_re.match(middle):
            if middle.startswith("https://") or middle.startswith("http://"):
                middle = (
                    f'<a href="{middle}"{rel_attr}{target_attr}>{trim_url(middle)}</a>'
                )
            else:
                middle = (
                    f'<a href="https://{middle}"{rel_attr}{target_attr}>'
                    f"{trim_url(middle)}</a>"
                )

        elif middle.startswith("mailto:") and _email_re.match(middle[7:]):
            middle = f'<a href="{middle}">{middle[7:]}</a>'

        elif (
            "@" in middle
            and not middle.startswith("www.")
            # ignore values like `@a@b`
            and not middle.startswith("@")
            and ":" not in middle
            and _email_re.match(middle)
        ):
            middle = f'<a href="mailto:{middle}">{middle}</a>'

        elif extra_schemes is not None:
            for scheme in extra_schemes:
                if middle != scheme and middle.startswith(scheme):
                    middle = f'<a href="{middle}"{rel_attr}{target_attr}>{middle}</a>'

        words[i] = f"{head}{middle}{tail}"

    return "".join(words)


def generate_lorem_ipsum(
    n: int = 5, html: bool = True, min: int = 20, max: int = 100
) -> str:
    """Generate some lorem ipsum for the template."""
    from .constants import LOREM_IPSUM_WORDS

    words = LOREM_IPSUM_WORDS.split()
    result = []

    for _ in range(n):
        next_capitalized = True
        last_comma = last_fullstop = 0
        word = None
        last = None
        p = []

        # each paragraph contains out of 20 to 100 words.
        for idx, _ in enumerate(range(randrange(min, max))):
            while True:
                word = choice(words)
                if word != last:
                    last = word
                    break
            if next_capitalized:
                word = word.capitalize()
                next_capitalized = False
            # add commas
            if idx - randrange(3, 8) > last_comma:
                last_comma = idx
                last_fullstop += 2
                word += ","
            # add end of sentences
            if idx - randrange(10, 20) > last_fullstop:
                last_comma = last_fullstop = idx
                word += "."
                next_capitalized = True
            p.append(word)

        # ensure that the paragraph ends with a dot.
        p_str = " ".join(p)

        if p_str.endswith(","):
            p_str = p_str[:-1] + "."
        elif not p_str.endswith("."):
            p_str += "."

        result.append(p_str)

    if not html:
        return "\n\n".join(result)
    return markupsafe.Markup(
        "\n".join(f"<p>{markupsafe.escape(x)}</p>" for x in result)
    )


def url_quote(obj: t.Any, charset: str = "utf-8", for_qs: bool = False) -> str:
    """Quote a string for use in a URL using the given charset.

    :param obj: String or bytes to quote. Other types are converted to
        string then encoded to bytes using the given charset.
    :param charset: Encode text to bytes using this charset.
    :param for_qs: Quote "/" and use "+" for spaces.
    """
    if not isinstance(obj, bytes):
        if not isinstance(obj, str):
            obj = str(obj)

        obj = obj.encode(charset)

    safe = b"" if for_qs else b"/"
    rv = quote_from_bytes(obj, safe)

    if for_qs:
        rv = rv.replace("%20", "+")

    return rv


@abc.MutableMapping.register
class LRUCache:
    """A simple LRU Cache implementation."""

    # this is fast for small capacities (something below 1000) but doesn't
    # scale.  But as long as it's only used as storage for templates this
    # won't do any harm.

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._mapping: t.Dict[t.Any, t.Any] = {}
        self._queue: te.Deque[t.Any] = deque()
        self._postinit()

    def _postinit(self) -> None:
        # alias all queue methods for faster lookup
        self._popleft = self._queue.popleft
        self._pop = self._queue.pop
        self._remove = self._queue.remove
        self._wlock = Lock()
        self._append = self._queue.append

    def __getstate__(self) -> t.Mapping[str, t.Any]:
        return {
            "capacity": self.capacity,
            "_mapping": self._mapping,
            "_queue": self._queue,
        }

    def __setstate__(self, d: t.Mapping[str, t.Any]) -> None:
        self.__dict__.update(d)
        self._postinit()

    def __getnewargs__(self) -> t.Tuple[t.Any, ...]:
        return (self.capacity,)

    def copy(self) -> "te.Self":
        """Return a shallow copy of the instance."""
        rv = self.__class__(self.capacity)
        rv._mapping.update(self._mapping)
        rv._queue.extend(self._queue)
        return rv

    def get(self, key: t.Any, default: t.Any = None) -> t.Any:
        """Return an item from the cache dict or `default`"""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: t.Any, default: t.Any = None) -> t.Any:
        """Set `default` if the key is not in the cache otherwise
        leave unchanged. Return the value of this key.
        """
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def clear(self) -> None:
        """Clear the cache."""
        with self._wlock:
            self._mapping.clear()
            self._queue.clear()

    def __contains__(self, key: t.Any) -> bool:
        """Check if a key exists in this cache."""
        return key in self._mapping

    def __len__(self) -> int:
        """Return the current size of the cache."""
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self._mapping!r}>"

    def __getitem__(self, key: t.Any) -> t.Any:
        """Get an item from the cache. Moves the item up so that it has the
        highest priority then.

        Raise a `KeyError` if it does not exist.
        """
        with self._wlock:
            rv = self._mapping[key]

            if self._queue[-1] != key:
                try:
                    self._remove(key)
                except ValueError:
                    # if something removed the key from the container
                    # when we read, ignore the ValueError that we would
                    # get otherwise.
                    pass

                self._append(key)

            return rv

    def __setitem__(self, key: t.Any, value: t.Any) -> None:
        """Sets the value for an item. Moves the item up so that it
        has the highest priority then.
        """
        with self._wlock:
            if key in self._mapping:
                self._remove(key)
            elif len(self._mapping) == self.capacity:
                del self._mapping[self._popleft()]

            self._append(key)
            self._mapping[key] = value

    def __delitem__(self, key: t.Any) -> None:
        """Remove an item from the cache dict.
        Raise a `KeyError` if it does not exist.
        """
        with self._wlock:
            del self._mapping[key]

            try:
                self._remove(key)
            except ValueError:
                pass

    def items(self) -> t.Iterable[t.Tuple[t.Any, t.Any]]:
        """Return a list of items."""
        result = [(key, self._mapping[key]) for key in list(self._queue)]
        result.reverse()
        return result

    def values(self) -> t.Iterable[t.Any]:
        """Return a list of all values."""
        return [x[1] for x in self.items()]

    def keys(self) -> t.Iterable[t.Any]:
        """Return a list of all keys ordered by most recent usage."""
        return list(self)

    def __iter__(self) -> t.Iterator[t.Any]:
        return reversed(tuple(self._queue))

    def __reversed__(self) -> t.Iterator[t.Any]:
        """Iterate over the keys in the cache dict, oldest items
        coming first.
        """
        return iter(tuple(self._queue))

    __copy__ = copy


def select_autoescape(
    enabled_extensions: t.Collection[str] = ("html", "htm", "xml"),
    disabled_extensions: t.Collection[str] = (),
    default_for_string: bool = True,
    default: bool = False,
) -> t.Callable[[t.Optional[str]], bool]:
    """Intelligently sets the initial value of autoescaping based on the
    filename of the template.  This is the recommended way to configure
    autoescaping if you do not want to write a custom function yourself.

    If you want to enable it for all templates created from strings or
    for all templates with `.html` and `.xml` extensions::

        from jinja2 import Environment, select_autoescape
        env = Environment(autoescape=select_autoescape(
            enabled_extensions=('html', 'xml'),
            default_for_string=True,
        ))

    Example configuration to turn it on at all times except if the template
    ends with `.txt`::

        from jinja2 import Environment, select_autoescape
        env = Environment(autoescape=select_autoescape(
            disabled_extensions=('txt',),
            default_for_string=True,
            default=True,
        ))

    The `enabled_extensions` is an iterable of all the extensions that
    autoescaping should be enabled for.  Likewise `disabled_extensions` is
    a list of all templates it should be disabled for.  If a template is
    loaded from a string then the default from `default_for_string` is used.
    If nothing matches then the initial value of autoescaping is set to the
    value of `default`.

    For security reasons this function operates case insensitive.

    .. versionadded:: 2.9
    """
    enabled_patterns = tuple(f".{x.lstrip('.').lower()}" for x in enabled_extensions)
    disabled_patterns = tuple(f".{x.lstrip('.').lower()}" for x in disabled_extensions)

    def autoescape(template_name: t.Optional[str]) -> bool:
        if template_name is None:
            return default_for_string
        template_name = template_name.lower()
        if template_name.endswith(enabled_patterns):
            return True
        if template_name.endswith(disabled_patterns):
            return False
        return default

    return autoescape


def htmlsafe_json_dumps(
    obj: t.Any, dumps: t.Optional[t.Callable[..., str]] = None, **kwargs: t.Any
) -> markupsafe.Markup:
    """Serialize an object to a string of JSON with :func:`json.dumps`,
    then replace HTML-unsafe characters with Unicode escapes and mark
    the result safe with :class:`~markupsafe.Markup`.

    This is available in templates as the ``|tojson`` filter.

    The following characters are escaped: ``<``, ``>``, ``&``, ``'``.

    The returned string is safe to render in HTML documents and
    ``<script>`` tags. The exception is in HTML attributes that are
    double quoted; either use single quotes or the ``|forceescape``
    filter.

    :param obj: The object to serialize to JSON.
    :param dumps: The ``dumps`` function to use. Defaults to
        ``env.policies["json.dumps_function"]``, which defaults to
        :func:`json.dumps`.
    :param kwargs: Extra arguments to pass to ``dumps``. Merged onto
        ``env.policies["json.dumps_kwargs"]``.

    .. versionchanged:: 3.0
        The ``dumper`` parameter is renamed to ``dumps``.

    .. versionadded:: 2.9
    """
    if dumps is None:
        dumps = json.dumps

    return markupsafe.Markup(
        dumps(obj, **kwargs)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


class Cycler:
    """Cycle through values by yield them one at a time, then restarting
    once the end is reached. Available as ``cycler`` in templates.

    Similar to ``loop.cycle``, but can be used outside loops or across
    multiple loops. For example, render a list of folders and files in a
    list, alternating giving them "odd" and "even" classes.

    .. code-block:: html+jinja

        {% set row_class = cycler("odd", "even") %}
        <ul class="browser">
        {% for folder in folders %}
          <li class="folder {{ row_class.next() }}">{{ folder }}
        {% endfor %}
        {% for file in files %}
          <li class="file {{ row_class.next() }}">{{ file }}
        {% endfor %}
        </ul>

    :param items: Each positional argument will be yielded in the order
        given for each cycle.

    .. versionadded:: 2.1
    """

    def __init__(self, *items: t.Any) -> None:
        if not items:
            raise RuntimeError("at least one item has to be provided")
        self.items = items
        self.pos = 0

    def reset(self) -> None:
        """Resets the current item to the first item."""
        self.pos = 0

    @property
    def current(self) -> t.Any:
        """Return the current item. Equivalent to the item that will be
        returned next time :meth:`next` is called.
        """
        return self.items[self.pos]

    def next(self) -> t.Any:
        """Return the current item, then advance :attr:`current` to the
        next item.
        """
        rv = self.current
        self.pos = (self.pos + 1) % len(self.items)
        return rv

    __next__ = next


class Joiner:
    """A joining helper for templates."""

    def __init__(self, sep: str = ", ") -> None:
        self.sep = sep
        self.used = False

    def __call__(self) -> str:
        if not self.used:
            self.used = True
            return ""
        return self.sep


class Namespace:
    """A namespace object that can hold arbitrary attributes.  It may be
    initialized from a dictionary or with keyword arguments."""

    def __init__(*args: t.Any, **kwargs: t.Any) -> None:  # noqa: B902
        self, args = args[0], args[1:]
        self.__attrs = dict(*args, **kwargs)

    def __getattribute__(self, name: str) -> t.Any:
        # __class__ is needed for the awaitable check in async mode
        if name in {"_Namespace__attrs", "__class__"}:
            return object.__getattribute__(self, name)
        try:
            return self.__attrs[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setitem__(self, name: str, value: t.Any) -> None:
        self.__attrs[name] = value

    def __repr__(self) -> str:
        return f"<Namespace {self.__attrs!r}>"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\sansio\response.py ===
from __future__ import annotations

import typing as t
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from http import HTTPStatus

from ..datastructures import CallbackDict
from ..datastructures import ContentRange
from ..datastructures import ContentSecurityPolicy
from ..datastructures import Headers
from ..datastructures import HeaderSet
from ..datastructures import ResponseCacheControl
from ..datastructures import WWWAuthenticate
from ..http import COEP
from ..http import COOP
from ..http import dump_age
from ..http import dump_cookie
from ..http import dump_header
from ..http import dump_options_header
from ..http import http_date
from ..http import HTTP_STATUS_CODES
from ..http import parse_age
from ..http import parse_cache_control_header
from ..http import parse_content_range_header
from ..http import parse_csp_header
from ..http import parse_date
from ..http import parse_options_header
from ..http import parse_set_header
from ..http import quote_etag
from ..http import unquote_etag
from ..utils import get_content_type
from ..utils import header_property

if t.TYPE_CHECKING:
    from ..datastructures.cache_control import _CacheControl


def _set_property(name: str, doc: str | None = None) -> property:
    def fget(self: Response) -> HeaderSet:
        def on_update(header_set: HeaderSet) -> None:
            if not header_set and name in self.headers:
                del self.headers[name]
            elif header_set:
                self.headers[name] = header_set.to_header()

        return parse_set_header(self.headers.get(name), on_update)

    def fset(
        self: Response,
        value: None | (str | dict[str, str | int] | t.Iterable[str]),
    ) -> None:
        if not value:
            del self.headers[name]
        elif isinstance(value, str):
            self.headers[name] = value
        else:
            self.headers[name] = dump_header(value)

    return property(fget, fset, doc=doc)


class Response:
    """Represents the non-IO parts of an HTTP response, specifically the
    status and headers but not the body.

    This class is not meant for general use. It should only be used when
    implementing WSGI, ASGI, or another HTTP application spec. Werkzeug
    provides a WSGI implementation at :cls:`werkzeug.wrappers.Response`.

    :param status: The status code for the response. Either an int, in
        which case the default status message is added, or a string in
        the form ``{code} {message}``, like ``404 Not Found``. Defaults
        to 200.
    :param headers: A :class:`~werkzeug.datastructures.Headers` object,
        or a list of ``(key, value)`` tuples that will be converted to a
        ``Headers`` object.
    :param mimetype: The mime type (content type without charset or
        other parameters) of the response. If the value starts with
        ``text/`` (or matches some other special cases), the charset
        will be added to create the ``content_type``.
    :param content_type: The full content type of the response.
        Overrides building the value from ``mimetype``.

    .. versionchanged:: 3.0
        The ``charset`` attribute was removed.

    .. versionadded:: 2.0
    """

    #: the default status if none is provided.
    default_status = 200

    #: the default mimetype if none is provided.
    default_mimetype: str | None = "text/plain"

    #: Warn if a cookie header exceeds this size. The default, 4093, should be
    #: safely `supported by most browsers <cookie_>`_. A cookie larger than
    #: this size will still be sent, but it may be ignored or handled
    #: incorrectly by some browsers. Set to 0 to disable this check.
    #:
    #: .. versionadded:: 0.13
    #:
    #: .. _`cookie`: http://browsercookielimits.squawky.net/
    max_cookie_size = 4093

    # A :class:`Headers` object representing the response headers.
    headers: Headers

    def __init__(
        self,
        status: int | str | HTTPStatus | None = None,
        headers: t.Mapping[str, str | t.Iterable[str]]
        | t.Iterable[tuple[str, str]]
        | None = None,
        mimetype: str | None = None,
        content_type: str | None = None,
    ) -> None:
        if isinstance(headers, Headers):
            self.headers = headers
        elif not headers:
            self.headers = Headers()
        else:
            self.headers = Headers(headers)

        if content_type is None:
            if mimetype is None and "content-type" not in self.headers:
                mimetype = self.default_mimetype
            if mimetype is not None:
                mimetype = get_content_type(mimetype, "utf-8")
            content_type = mimetype
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if status is None:
            status = self.default_status
        self.status = status  # type: ignore

    def __repr__(self) -> str:
        return f"<{type(self).__name__} [{self.status}]>"

    @property
    def status_code(self) -> int:
        """The HTTP status code as a number."""
        return self._status_code

    @status_code.setter
    def status_code(self, code: int) -> None:
        self.status = code  # type: ignore

    @property
    def status(self) -> str:
        """The HTTP status code as a string."""
        return self._status

    @status.setter
    def status(self, value: str | int | HTTPStatus) -> None:
        self._status, self._status_code = self._clean_status(value)

    def _clean_status(self, value: str | int | HTTPStatus) -> tuple[str, int]:
        if isinstance(value, (int, HTTPStatus)):
            status_code = int(value)
        else:
            value = value.strip()

            if not value:
                raise ValueError("Empty status argument")

            code_str, sep, _ = value.partition(" ")

            try:
                status_code = int(code_str)
            except ValueError:
                # only message
                return f"0 {value}", 0

            if sep:
                # code and message
                return value, status_code

        # only code, look up message
        try:
            status = f"{status_code} {HTTP_STATUS_CODES[status_code].upper()}"
        except KeyError:
            status = f"{status_code} UNKNOWN"

        return status, status_code

    def set_cookie(
        self,
        key: str,
        value: str = "",
        max_age: timedelta | int | None = None,
        expires: str | datetime | int | float | None = None,
        path: str | None = "/",
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = None,
        partitioned: bool = False,
    ) -> None:
        """Sets a cookie.

        A warning is raised if the size of the cookie header exceeds
        :attr:`max_cookie_size`, but the header will still be set.

        :param key: the key (name) of the cookie to be set.
        :param value: the value of the cookie.
        :param max_age: should be a number of seconds, or `None` (default) if
                        the cookie should last only as long as the client's
                        browser session.
        :param expires: should be a `datetime` object or UNIX timestamp.
        :param path: limits the cookie to a given path, per default it will
                     span the whole domain.
        :param domain: if you want to set a cross-domain cookie.  For example,
                       ``domain="example.com"`` will set a cookie that is
                       readable by the domain ``www.example.com``,
                       ``foo.example.com`` etc.  Otherwise, a cookie will only
                       be readable by the domain that set it.
        :param secure: If ``True``, the cookie will only be available
            via HTTPS.
        :param httponly: Disallow JavaScript access to the cookie.
        :param samesite: Limit the scope of the cookie to only be
            attached to requests that are "same-site".
        :param partitioned: If ``True``, the cookie will be partitioned.

        .. versionchanged:: 3.1
            The ``partitioned`` parameter was added.
        """
        self.headers.add(
            "Set-Cookie",
            dump_cookie(
                key,
                value=value,
                max_age=max_age,
                expires=expires,
                path=path,
                domain=domain,
                secure=secure,
                httponly=httponly,
                max_size=self.max_cookie_size,
                samesite=samesite,
                partitioned=partitioned,
            ),
        )

    def delete_cookie(
        self,
        key: str,
        path: str | None = "/",
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = None,
        partitioned: bool = False,
    ) -> None:
        """Delete a cookie.  Fails silently if key doesn't exist.

        :param key: the key (name) of the cookie to be deleted.
        :param path: if the cookie that should be deleted was limited to a
                     path, the path has to be defined here.
        :param domain: if the cookie that should be deleted was limited to a
                       domain, that domain has to be defined here.
        :param secure: If ``True``, the cookie will only be available
            via HTTPS.
        :param httponly: Disallow JavaScript access to the cookie.
        :param samesite: Limit the scope of the cookie to only be
            attached to requests that are "same-site".
        :param partitioned: If ``True``, the cookie will be partitioned.
        """
        self.set_cookie(
            key,
            expires=0,
            max_age=0,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
            partitioned=partitioned,
        )

    @property
    def is_json(self) -> bool:
        """Check if the mimetype indicates JSON data, either
        :mimetype:`application/json` or :mimetype:`application/*+json`.
        """
        mt = self.mimetype
        return mt is not None and (
            mt == "application/json"
            or mt.startswith("application/")
            and mt.endswith("+json")
        )

    # Common Descriptors

    @property
    def mimetype(self) -> str | None:
        """The mimetype (content type without charset etc.)"""
        ct = self.headers.get("content-type")

        if ct:
            return ct.split(";")[0].strip()
        else:
            return None

    @mimetype.setter
    def mimetype(self, value: str) -> None:
        self.headers["Content-Type"] = get_content_type(value, "utf-8")

    @property
    def mimetype_params(self) -> dict[str, str]:
        """The mimetype parameters as dict. For example if the
        content type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.5
        """

        def on_update(d: CallbackDict[str, str]) -> None:
            self.headers["Content-Type"] = dump_options_header(self.mimetype, d)

        d = parse_options_header(self.headers.get("content-type", ""))[1]
        return CallbackDict(d, on_update)

    location = header_property[str](
        "Location",
        doc="""The Location response-header field is used to redirect
        the recipient to a location other than the Request-URI for
        completion of the request or identification of a new
        resource.""",
    )
    age = header_property(
        "Age",
        None,
        parse_age,
        dump_age,  # type: ignore
        doc="""The Age response-header field conveys the sender's
        estimate of the amount of time since the response (or its
        revalidation) was generated at the origin server.

        Age values are non-negative decimal integers, representing time
        in seconds.""",
    )
    content_type = header_property[str](
        "Content-Type",
        doc="""The Content-Type entity-header field indicates the media
        type of the entity-body sent to the recipient or, in the case of
        the HEAD method, the media type that would have been sent had
        the request been a GET.""",
    )
    content_length = header_property(
        "Content-Length",
        None,
        int,
        str,
        doc="""The Content-Length entity-header field indicates the size
        of the entity-body, in decimal number of OCTETs, sent to the
        recipient or, in the case of the HEAD method, the size of the
        entity-body that would have been sent had the request been a
        GET.""",
    )
    content_location = header_property[str](
        "Content-Location",
        doc="""The Content-Location entity-header field MAY be used to
        supply the resource location for the entity enclosed in the
        message when that entity is accessible from a location separate
        from the requested resource's URI.""",
    )
    content_encoding = header_property[str](
        "Content-Encoding",
        doc="""The Content-Encoding entity-header field is used as a
        modifier to the media-type. When present, its value indicates
        what additional content codings have been applied to the
        entity-body, and thus what decoding mechanisms must be applied
        in order to obtain the media-type referenced by the Content-Type
        header field.""",
    )
    content_md5 = header_property[str](
        "Content-MD5",
        doc="""The Content-MD5 entity-header field, as defined in
        RFC 1864, is an MD5 digest of the entity-body for the purpose of
        providing an end-to-end message integrity check (MIC) of the
        entity-body. (Note: a MIC is good for detecting accidental
        modification of the entity-body in transit, but is not proof
        against malicious attacks.)""",
    )
    date = header_property(
        "Date",
        None,
        parse_date,
        http_date,
        doc="""The Date general-header field represents the date and
        time at which the message was originated, having the same
        semantics as orig-date in RFC 822.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
    )
    expires = header_property(
        "Expires",
        None,
        parse_date,
        http_date,
        doc="""The Expires entity-header field gives the date/time after
        which the response is considered stale. A stale cache entry may
        not normally be returned by a cache.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
    )
    last_modified = header_property(
        "Last-Modified",
        None,
        parse_date,
        http_date,
        doc="""The Last-Modified entity-header field indicates the date
        and time at which the origin server believes the variant was
        last modified.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """,
    )

    @property
    def retry_after(self) -> datetime | None:
        """The Retry-After response-header field can be used with a
        503 (Service Unavailable) response to indicate how long the
        service is expected to be unavailable to the requesting client.

        Time in seconds until expiration or date.

        .. versionchanged:: 2.0
            The datetime object is timezone-aware.
        """
        value = self.headers.get("retry-after")
        if value is None:
            return None

        try:
            seconds = int(value)
        except ValueError:
            return parse_date(value)

        return datetime.now(timezone.utc) + timedelta(seconds=seconds)

    @retry_after.setter
    def retry_after(self, value: datetime | int | str | None) -> None:
        if value is None:
            if "retry-after" in self.headers:
                del self.headers["retry-after"]
            return
        elif isinstance(value, datetime):
            value = http_date(value)
        else:
            value = str(value)
        self.headers["Retry-After"] = value

    vary = _set_property(
        "Vary",
        doc="""The Vary field value indicates the set of request-header
        fields that fully determines, while the response is fresh,
        whether a cache is permitted to use the response to reply to a
        subsequent request without revalidation.""",
    )
    content_language = _set_property(
        "Content-Language",
        doc="""The Content-Language entity-header field describes the
        natural language(s) of the intended audience for the enclosed
        entity. Note that this might not be equivalent to all the
        languages used within the entity-body.""",
    )
    allow = _set_property(
        "Allow",
        doc="""The Allow entity-header field lists the set of methods
        supported by the resource identified by the Request-URI. The
        purpose of this field is strictly to inform the recipient of
        valid methods associated with the resource. An Allow header
        field MUST be present in a 405 (Method Not Allowed)
        response.""",
    )

    # ETag

    @property
    def cache_control(self) -> ResponseCacheControl:
        """The Cache-Control general-header field is used to specify
        directives that MUST be obeyed by all caching mechanisms along the
        request/response chain.
        """

        def on_update(cache_control: _CacheControl) -> None:
            if not cache_control and "cache-control" in self.headers:
                del self.headers["cache-control"]
            elif cache_control:
                self.headers["Cache-Control"] = cache_control.to_header()

        return parse_cache_control_header(
            self.headers.get("cache-control"), on_update, ResponseCacheControl
        )

    def set_etag(self, etag: str, weak: bool = False) -> None:
        """Set the etag, and override the old one if there was one."""
        self.headers["ETag"] = quote_etag(etag, weak)

    def get_etag(self) -> tuple[str, bool] | tuple[None, None]:
        """Return a tuple in the form ``(etag, is_weak)``.  If there is no
        ETag the return value is ``(None, None)``.
        """
        return unquote_etag(self.headers.get("ETag"))

    accept_ranges = header_property[str](
        "Accept-Ranges",
        doc="""The `Accept-Ranges` header. Even though the name would
        indicate that multiple values are supported, it must be one
        string token only.

        The values ``'bytes'`` and ``'none'`` are common.

        .. versionadded:: 0.7""",
    )

    @property
    def content_range(self) -> ContentRange:
        """The ``Content-Range`` header as a
        :class:`~werkzeug.datastructures.ContentRange` object. Available
        even if the header is not set.

        .. versionadded:: 0.7
        """

        def on_update(rng: ContentRange) -> None:
            if not rng:
                del self.headers["content-range"]
            else:
                self.headers["Content-Range"] = rng.to_header()

        rv = parse_content_range_header(self.headers.get("content-range"), on_update)
        # always provide a content range object to make the descriptor
        # more user friendly.  It provides an unset() method that can be
        # used to remove the header quickly.
        if rv is None:
            rv = ContentRange(None, None, None, on_update=on_update)
        return rv

    @content_range.setter
    def content_range(self, value: ContentRange | str | None) -> None:
        if not value:
            del self.headers["content-range"]
        elif isinstance(value, str):
            self.headers["Content-Range"] = value
        else:
            self.headers["Content-Range"] = value.to_header()

    # Authorization

    @property
    def www_authenticate(self) -> WWWAuthenticate:
        """The ``WWW-Authenticate`` header parsed into a :class:`.WWWAuthenticate`
        object. Modifying the object will modify the header value.

        This header is not set by default. To set this header, assign an instance of
        :class:`.WWWAuthenticate` to this attribute.

        .. code-block:: python

            response.www_authenticate = WWWAuthenticate(
                "basic", {"realm": "Authentication Required"}
            )

        Multiple values for this header can be sent to give the client multiple options.
        Assign a list to set multiple headers. However, modifying the items in the list
        will not automatically update the header values, and accessing this attribute
        will only ever return the first value.

        To unset this header, assign ``None`` or use ``del``.

        .. versionchanged:: 2.3
            This attribute can be assigned to to set the header. A list can be assigned
            to set multiple header values. Use ``del`` to unset the header.

        .. versionchanged:: 2.3
            :class:`WWWAuthenticate` is no longer a ``dict``. The ``token`` attribute
            was added for auth challenges that use a token instead of parameters.
        """
        value = WWWAuthenticate.from_header(self.headers.get("WWW-Authenticate"))

        if value is None:
            value = WWWAuthenticate("basic")

        def on_update(value: WWWAuthenticate) -> None:
            self.www_authenticate = value

        value._on_update = on_update
        return value

    @www_authenticate.setter
    def www_authenticate(
        self, value: WWWAuthenticate | list[WWWAuthenticate] | None
    ) -> None:
        if not value:  # None or empty list
            del self.www_authenticate
        elif isinstance(value, list):
            # Clear any existing header by setting the first item.
            self.headers.set("WWW-Authenticate", value[0].to_header())

            for item in value[1:]:
                # Add additional header lines for additional items.
                self.headers.add("WWW-Authenticate", item.to_header())
        else:
            self.headers.set("WWW-Authenticate", value.to_header())

            def on_update(value: WWWAuthenticate) -> None:
                self.www_authenticate = value

            # When setting a single value, allow updating it directly.
            value._on_update = on_update

    @www_authenticate.deleter
    def www_authenticate(self) -> None:
        if "WWW-Authenticate" in self.headers:
            del self.headers["WWW-Authenticate"]

    # CSP

    @property
    def content_security_policy(self) -> ContentSecurityPolicy:
        """The ``Content-Security-Policy`` header as a
        :class:`~werkzeug.datastructures.ContentSecurityPolicy` object. Available
        even if the header is not set.

        The Content-Security-Policy header adds an additional layer of
        security to help detect and mitigate certain types of attacks.
        """

        def on_update(csp: ContentSecurityPolicy) -> None:
            if not csp:
                del self.headers["content-security-policy"]
            else:
                self.headers["Content-Security-Policy"] = csp.to_header()

        rv = parse_csp_header(self.headers.get("content-security-policy"), on_update)
        if rv is None:
            rv = ContentSecurityPolicy(None, on_update=on_update)
        return rv

    @content_security_policy.setter
    def content_security_policy(
        self, value: ContentSecurityPolicy | str | None
    ) -> None:
        if not value:
            del self.headers["content-security-policy"]
        elif isinstance(value, str):
            self.headers["Content-Security-Policy"] = value
        else:
            self.headers["Content-Security-Policy"] = value.to_header()

    @property
    def content_security_policy_report_only(self) -> ContentSecurityPolicy:
        """The ``Content-Security-policy-report-only`` header as a
        :class:`~werkzeug.datastructures.ContentSecurityPolicy` object. Available
        even if the header is not set.

        The Content-Security-Policy-Report-Only header adds a csp policy
        that is not enforced but is reported thereby helping detect
        certain types of attacks.
        """

        def on_update(csp: ContentSecurityPolicy) -> None:
            if not csp:
                del self.headers["content-security-policy-report-only"]
            else:
                self.headers["Content-Security-policy-report-only"] = csp.to_header()

        rv = parse_csp_header(
            self.headers.get("content-security-policy-report-only"), on_update
        )
        if rv is None:
            rv = ContentSecurityPolicy(None, on_update=on_update)
        return rv

    @content_security_policy_report_only.setter
    def content_security_policy_report_only(
        self, value: ContentSecurityPolicy | str | None
    ) -> None:
        if not value:
            del self.headers["content-security-policy-report-only"]
        elif isinstance(value, str):
            self.headers["Content-Security-policy-report-only"] = value
        else:
            self.headers["Content-Security-policy-report-only"] = value.to_header()

    # CORS

    @property
    def access_control_allow_credentials(self) -> bool:
        """Whether credentials can be shared by the browser to
        JavaScript code. As part of the preflight request it indicates
        whether credentials can be used on the cross origin request.
        """
        return "Access-Control-Allow-Credentials" in self.headers

    @access_control_allow_credentials.setter
    def access_control_allow_credentials(self, value: bool | None) -> None:
        if value is True:
            self.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            self.headers.pop("Access-Control-Allow-Credentials", None)

    access_control_allow_headers = header_property(
        "Access-Control-Allow-Headers",
        load_func=parse_set_header,
        dump_func=dump_header,
        doc="Which headers can be sent with the cross origin request.",
    )

    access_control_allow_methods = header_property(
        "Access-Control-Allow-Methods",
        load_func=parse_set_header,
        dump_func=dump_header,
        doc="Which methods can be used for the cross origin request.",
    )

    access_control_allow_origin = header_property[str](
        "Access-Control-Allow-Origin",
        doc="The origin or '*' for any origin that may make cross origin requests.",
    )

    access_control_expose_headers = header_property(
        "Access-Control-Expose-Headers",
        load_func=parse_set_header,
        dump_func=dump_header,
        doc="Which headers can be shared by the browser to JavaScript code.",
    )

    access_control_max_age = header_property(
        "Access-Control-Max-Age",
        load_func=int,
        dump_func=str,
        doc="The maximum age in seconds the access control settings can be cached for.",
    )

    cross_origin_opener_policy = header_property[COOP](
        "Cross-Origin-Opener-Policy",
        load_func=lambda value: COOP(value),
        dump_func=lambda value: value.value,
        default=COOP.UNSAFE_NONE,
        doc="""Allows control over sharing of browsing context group with cross-origin
        documents. Values must be a member of the :class:`werkzeug.http.COOP` enum.""",
    )

    cross_origin_embedder_policy = header_property[COEP](
        "Cross-Origin-Embedder-Policy",
        load_func=lambda value: COEP(value),
        dump_func=lambda value: value.value,
        default=COEP.UNSAFE_NONE,
        doc="""Prevents a document from loading any cross-origin resources that do not
        explicitly grant the document permission. Values must be a member of the
        :class:`werkzeug.http.COEP` enum.""",
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\sas\sas7bdat.py ===
"""
Read SAS7BDAT files

Based on code written by Jared Hobbs:
  https://bitbucket.org/jaredhobbs/sas7bdat

See also:
  https://github.com/BioStatMatt/sas7bdat

Partial documentation of the file format:
  https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf

Reference for binary data compression:
  http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
"""
from __future__ import annotations

from collections import abc
from datetime import (
    datetime,
    timedelta,
)
import sys
from typing import TYPE_CHECKING

import numpy as np

from pandas._config import get_option

from pandas._libs.byteswap import (
    read_double_with_byteswap,
    read_float_with_byteswap,
    read_uint16_with_byteswap,
    read_uint32_with_byteswap,
    read_uint64_with_byteswap,
)
from pandas._libs.sas import (
    Parser,
    get_subheader_index,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas.errors import EmptyDataError

import pandas as pd
from pandas import (
    DataFrame,
    Timestamp,
    isna,
)

from pandas.io.common import get_handle
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import ReaderBase

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadBuffer,
    )


_unix_origin = Timestamp("1970-01-01")
_sas_origin = Timestamp("1960-01-01")


def _parse_datetime(sas_datetime: float, unit: str):
    if isna(sas_datetime):
        return pd.NaT

    if unit == "s":
        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)

    elif unit == "d":
        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)

    else:
        raise ValueError("unit must be 'd' or 's'")


def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    """
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {'d', 's'}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    """
    td = (_sas_origin - _unix_origin).as_unit("s")
    if unit == "s":
        millis = cast_from_unit_vectorized(
            sas_datetimes._values, unit="s", out_unit="ms"
        )
        dt64ms = millis.view("M8[ms]") + td
        return pd.Series(dt64ms, index=sas_datetimes.index, copy=False)
    else:
        vals = np.array(sas_datetimes, dtype="M8[D]") + td
        return pd.Series(vals, dtype="M8[s]", index=sas_datetimes.index, copy=False)


class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes
    ctype: bytes
    length: int

    def __init__(
        self,
        col_id: int,
        # These can be bytes when convert_header_text is False
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ) -> None:
        self.col_id = col_id
        self.name = name
        self.label = label
        self.format = format
        self.ctype = ctype
        self.length = length


# SAS7BDAT represents a SAS data file in SAS7BDAT format.
class SAS7BDATReader(ReaderBase, abc.Iterator):
    """
    Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : str, 'infer', defaults to None
        String encoding acc. to Python standard encodings,
        encoding='infer' tries to detect the encoding from the file header,
        encoding=None will leave the data in binary format.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """

    _int_length: int
    _cached_page: bytes | None

    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        index=None,
        convert_dates: bool = True,
        blank_missing: bool = True,
        chunksize: int | None = None,
        encoding: str | None = None,
        convert_text: bool = True,
        convert_header_text: bool = True,
        compression: CompressionOptions = "infer",
    ) -> None:
        self.index = index
        self.convert_dates = convert_dates
        self.blank_missing = blank_missing
        self.chunksize = chunksize
        self.encoding = encoding
        self.convert_text = convert_text
        self.convert_header_text = convert_header_text

        self.default_encoding = "latin-1"
        self.compression = b""
        self.column_names_raw: list[bytes] = []
        self.column_names: list[str | bytes] = []
        self.column_formats: list[str | bytes] = []
        self.columns: list[_Column] = []

        self._current_page_data_subheader_pointers: list[tuple[int, int]] = []
        self._cached_page = None
        self._column_data_lengths: list[int] = []
        self._column_data_offsets: list[int] = []
        self._column_types: list[bytes] = []

        self._current_row_in_file_index = 0
        self._current_row_on_page_index = 0
        self._current_row_in_file_index = 0

        self.handles = get_handle(
            path_or_buf, "rb", is_text=False, compression=compression
        )

        self._path_or_buf = self.handles.handle

        # Same order as const.SASIndex
        self._subheader_processors = [
            self._process_rowsize_subheader,
            self._process_columnsize_subheader,
            self._process_subheader_counts,
            self._process_columntext_subheader,
            self._process_columnname_subheader,
            self._process_columnattributes_subheader,
            self._process_format_subheader,
            self._process_columnlist_subheader,
            None,  # Data
        ]

        try:
            self._get_properties()
            self._parse_metadata()
        except Exception:
            self.close()
            raise

    def column_data_lengths(self) -> np.ndarray:
        """Return a numpy int64 array of the column data lengths"""
        return np.asarray(self._column_data_lengths, dtype=np.int64)

    def column_data_offsets(self) -> np.ndarray:
        """Return a numpy int64 array of the column offsets"""
        return np.asarray(self._column_data_offsets, dtype=np.int64)

    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        return np.asarray(self._column_types, dtype=np.dtype("S1"))

    def close(self) -> None:
        self.handles.close()

    def _get_properties(self) -> None:
        # Check magic number
        self._path_or_buf.seek(0)
        self._cached_page = self._path_or_buf.read(288)
        if self._cached_page[0 : len(const.magic)] != const.magic:
            raise ValueError("magic number mismatch (not a SAS file?)")

        # Get alignment information
        buf = self._read_bytes(const.align_1_offset, const.align_1_length)
        if buf == const.u64_byte_checker_value:
            self.U64 = True
            self._int_length = 8
            self._page_bit_offset = const.page_bit_offset_x64
            self._subheader_pointer_length = const.subheader_pointer_length_x64
        else:
            self.U64 = False
            self._page_bit_offset = const.page_bit_offset_x86
            self._subheader_pointer_length = const.subheader_pointer_length_x86
            self._int_length = 4
        buf = self._read_bytes(const.align_2_offset, const.align_2_length)
        if buf == const.align_1_checker_value:
            align1 = const.align_2_value
        else:
            align1 = 0

        # Get endianness information
        buf = self._read_bytes(const.endianness_offset, const.endianness_length)
        if buf == b"\x01":
            self.byte_order = "<"
            self.need_byteswap = sys.byteorder == "big"
        else:
            self.byte_order = ">"
            self.need_byteswap = sys.byteorder == "little"

        # Get encoding information
        buf = self._read_bytes(const.encoding_offset, const.encoding_length)[0]
        if buf in const.encoding_names:
            self.inferred_encoding = const.encoding_names[buf]
            if self.encoding == "infer":
                self.encoding = self.inferred_encoding
        else:
            self.inferred_encoding = f"unknown (code={buf})"

        # Timestamp is epoch 01/01/1960
        epoch = datetime(1960, 1, 1)
        x = self._read_float(
            const.date_created_offset + align1, const.date_created_length
        )
        self.date_created = epoch + pd.to_timedelta(x, unit="s")
        x = self._read_float(
            const.date_modified_offset + align1, const.date_modified_length
        )
        self.date_modified = epoch + pd.to_timedelta(x, unit="s")

        self.header_length = self._read_uint(
            const.header_size_offset + align1, const.header_size_length
        )

        # Read the rest of the header into cached_page.
        buf = self._path_or_buf.read(self.header_length - 288)
        self._cached_page += buf
        # error: Argument 1 to "len" has incompatible type "Optional[bytes]";
        #  expected "Sized"
        if len(self._cached_page) != self.header_length:  # type: ignore[arg-type]
            raise ValueError("The SAS7BDAT file appears to be truncated.")

        self._page_length = self._read_uint(
            const.page_size_offset + align1, const.page_size_length
        )

    def __next__(self) -> DataFrame:
        da = self.read(nrows=self.chunksize or 1)
        if da.empty:
            self.close()
            raise StopIteration
        return da

    # Read a single float of the given width (4 or 8).
    def _read_float(self, offset: int, width: int):
        assert self._cached_page is not None
        if width == 4:
            return read_float_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 8:
            return read_double_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        else:
            self.close()
            raise ValueError("invalid float width")

    # Read a single unsigned integer of the given width (1, 2, 4 or 8).
    def _read_uint(self, offset: int, width: int) -> int:
        assert self._cached_page is not None
        if width == 1:
            return self._read_bytes(offset, 1)[0]
        elif width == 2:
            return read_uint16_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 4:
            return read_uint32_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        elif width == 8:
            return read_uint64_with_byteswap(
                self._cached_page, offset, self.need_byteswap
            )
        else:
            self.close()
            raise ValueError("invalid int width")

    def _read_bytes(self, offset: int, length: int):
        assert self._cached_page is not None
        if offset + length > len(self._cached_page):
            self.close()
            raise ValueError("The cached page is too small.")
        return self._cached_page[offset : offset + length]

    def _read_and_convert_header_text(self, offset: int, length: int) -> str | bytes:
        return self._convert_header_text(
            self._read_bytes(offset, length).rstrip(b"\x00 ")
        )

    def _parse_metadata(self) -> None:
        done = False
        while not done:
            self._cached_page = self._path_or_buf.read(self._page_length)
            if len(self._cached_page) <= 0:
                break
            if len(self._cached_page) != self._page_length:
                raise ValueError("Failed to read a meta data page from the SAS file.")
            done = self._process_page_meta()

    def _process_page_meta(self) -> bool:
        self._read_page_header()
        pt = const.page_meta_types + [const.page_amd_type, const.page_mix_type]
        if self._current_page_type in pt:
            self._process_page_metadata()
        is_data_page = self._current_page_type == const.page_data_type
        is_mix_page = self._current_page_type == const.page_mix_type
        return bool(
            is_data_page
            or is_mix_page
            or self._current_page_data_subheader_pointers != []
        )

    def _read_page_header(self) -> None:
        bit_offset = self._page_bit_offset
        tx = const.page_type_offset + bit_offset
        self._current_page_type = (
            self._read_uint(tx, const.page_type_length) & const.page_type_mask2
        )
        tx = const.block_count_offset + bit_offset
        self._current_page_block_count = self._read_uint(tx, const.block_count_length)
        tx = const.subheader_count_offset + bit_offset
        self._current_page_subheaders_count = self._read_uint(
            tx, const.subheader_count_length
        )

    def _process_page_metadata(self) -> None:
        bit_offset = self._page_bit_offset

        for i in range(self._current_page_subheaders_count):
            offset = const.subheader_pointers_offset + bit_offset
            total_offset = offset + self._subheader_pointer_length * i

            subheader_offset = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            subheader_length = self._read_uint(total_offset, self._int_length)
            total_offset += self._int_length

            subheader_compression = self._read_uint(total_offset, 1)
            total_offset += 1

            subheader_type = self._read_uint(total_offset, 1)

            if (
                subheader_length == 0
                or subheader_compression == const.truncated_subheader_id
            ):
                continue

            subheader_signature = self._read_bytes(subheader_offset, self._int_length)
            subheader_index = get_subheader_index(subheader_signature)
            subheader_processor = self._subheader_processors[subheader_index]

            if subheader_processor is None:
                f1 = subheader_compression in (const.compressed_subheader_id, 0)
                f2 = subheader_type == const.compressed_subheader_type
                if self.compression and f1 and f2:
                    self._current_page_data_subheader_pointers.append(
                        (subheader_offset, subheader_length)
                    )
                else:
                    self.close()
                    raise ValueError(
                        f"Unknown subheader signature {subheader_signature}"
                    )
            else:
                subheader_processor(subheader_offset, subheader_length)

    def _process_rowsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        lcs_offset = offset
        lcp_offset = offset
        if self.U64:
            lcs_offset += 682
            lcp_offset += 706
        else:
            lcs_offset += 354
            lcp_offset += 378

        self.row_length = self._read_uint(
            offset + const.row_length_offset_multiplier * int_len,
            int_len,
        )
        self.row_count = self._read_uint(
            offset + const.row_count_offset_multiplier * int_len,
            int_len,
        )
        self.col_count_p1 = self._read_uint(
            offset + const.col_count_p1_multiplier * int_len, int_len
        )
        self.col_count_p2 = self._read_uint(
            offset + const.col_count_p2_multiplier * int_len, int_len
        )
        mx = const.row_count_on_mix_page_offset_multiplier * int_len
        self._mix_page_row_count = self._read_uint(offset + mx, int_len)
        self._lcs = self._read_uint(lcs_offset, 2)
        self._lcp = self._read_uint(lcp_offset, 2)

    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        self.column_count = self._read_uint(offset, int_len)
        if self.col_count_p1 + self.col_count_p2 != self.column_count:
            print(
                f"Warning: column count mismatch ({self.col_count_p1} + "
                f"{self.col_count_p2} != {self.column_count})\n"
            )

    # Unknown purpose
    def _process_subheader_counts(self, offset: int, length: int) -> None:
        pass

    def _process_columntext_subheader(self, offset: int, length: int) -> None:
        offset += self._int_length
        text_block_size = self._read_uint(offset, const.text_block_size_length)

        buf = self._read_bytes(offset, text_block_size)
        cname_raw = buf[0:text_block_size].rstrip(b"\x00 ")
        self.column_names_raw.append(cname_raw)

        if len(self.column_names_raw) == 1:
            compression_literal = b""
            for cl in const.compression_literals:
                if cl in cname_raw:
                    compression_literal = cl
            self.compression = compression_literal
            offset -= self._int_length

            offset1 = offset + 16
            if self.U64:
                offset1 += 4

            buf = self._read_bytes(offset1, self._lcp)
            compression_literal = buf.rstrip(b"\x00")
            if compression_literal == b"":
                self._lcs = 0
                offset1 = offset + 32
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcp)
                self.creator_proc = buf[0 : self._lcp]
            elif compression_literal == const.rle_compression:
                offset1 = offset + 40
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcp)
                self.creator_proc = buf[0 : self._lcp]
            elif self._lcs > 0:
                self._lcp = 0
                offset1 = offset + 16
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcs)
                self.creator_proc = buf[0 : self._lcp]
            if hasattr(self, "creator_proc"):
                self.creator_proc = self._convert_header_text(self.creator_proc)

    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        column_name_pointers_count = (length - 2 * int_len - 12) // 8
        for i in range(column_name_pointers_count):
            text_subheader = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_text_subheader_offset
            )
            col_name_offset = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_offset_offset
            )
            col_name_length = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_length_offset
            )

            idx = self._read_uint(
                text_subheader, const.column_name_text_subheader_length
            )
            col_offset = self._read_uint(
                col_name_offset, const.column_name_offset_length
            )
            col_len = self._read_uint(col_name_length, const.column_name_length_length)

            name_raw = self.column_names_raw[idx]
            cname = name_raw[col_offset : col_offset + col_len]
            self.column_names.append(self._convert_header_text(cname))

    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        column_attributes_vectors_count = (length - 2 * int_len - 12) // (int_len + 8)
        for i in range(column_attributes_vectors_count):
            col_data_offset = (
                offset + int_len + const.column_data_offset_offset + i * (int_len + 8)
            )
            col_data_len = (
                offset
                + 2 * int_len
                + const.column_data_length_offset
                + i * (int_len + 8)
            )
            col_types = (
                offset + 2 * int_len + const.column_type_offset + i * (int_len + 8)
            )

            x = self._read_uint(col_data_offset, int_len)
            self._column_data_offsets.append(x)

            x = self._read_uint(col_data_len, const.column_data_length_length)
            self._column_data_lengths.append(x)

            x = self._read_uint(col_types, const.column_type_length)
            self._column_types.append(b"d" if x == 1 else b"s")

    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        # unknown purpose
        pass

    def _process_format_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        text_subheader_format = (
            offset + const.column_format_text_subheader_index_offset + 3 * int_len
        )
        col_format_offset = offset + const.column_format_offset_offset + 3 * int_len
        col_format_len = offset + const.column_format_length_offset + 3 * int_len
        text_subheader_label = (
            offset + const.column_label_text_subheader_index_offset + 3 * int_len
        )
        col_label_offset = offset + const.column_label_offset_offset + 3 * int_len
        col_label_len = offset + const.column_label_length_offset + 3 * int_len

        x = self._read_uint(
            text_subheader_format, const.column_format_text_subheader_index_length
        )
        format_idx = min(x, len(self.column_names_raw) - 1)

        format_start = self._read_uint(
            col_format_offset, const.column_format_offset_length
        )
        format_len = self._read_uint(col_format_len, const.column_format_length_length)

        label_idx = self._read_uint(
            text_subheader_label, const.column_label_text_subheader_index_length
        )
        label_idx = min(label_idx, len(self.column_names_raw) - 1)

        label_start = self._read_uint(
            col_label_offset, const.column_label_offset_length
        )
        label_len = self._read_uint(col_label_len, const.column_label_length_length)

        label_names = self.column_names_raw[label_idx]
        column_label = self._convert_header_text(
            label_names[label_start : label_start + label_len]
        )
        format_names = self.column_names_raw[format_idx]
        column_format = self._convert_header_text(
            format_names[format_start : format_start + format_len]
        )
        current_column_number = len(self.columns)

        col = _Column(
            current_column_number,
            self.column_names[current_column_number],
            column_label,
            column_format,
            self._column_types[current_column_number],
            self._column_data_lengths[current_column_number],
        )

        self.column_formats.append(column_format)
        self.columns.append(col)

    def read(self, nrows: int | None = None) -> DataFrame:
        if (nrows is None) and (self.chunksize is not None):
            nrows = self.chunksize
        elif nrows is None:
            nrows = self.row_count

        if len(self._column_types) == 0:
            self.close()
            raise EmptyDataError("No columns to parse from file")

        if nrows > 0 and self._current_row_in_file_index >= self.row_count:
            return DataFrame()

        nrows = min(nrows, self.row_count - self._current_row_in_file_index)

        nd = self._column_types.count(b"d")
        ns = self._column_types.count(b"s")

        self._string_chunk = np.empty((ns, nrows), dtype=object)
        self._byte_chunk = np.zeros((nd, 8 * nrows), dtype=np.uint8)

        self._current_row_in_chunk_index = 0
        p = Parser(self)
        p.read(nrows)

        rslt = self._chunk_to_dataframe()
        if self.index is not None:
            rslt = rslt.set_index(self.index)

        return rslt

    def _read_next_page(self):
        self._current_page_data_subheader_pointers = []
        self._cached_page = self._path_or_buf.read(self._page_length)
        if len(self._cached_page) <= 0:
            return True
        elif len(self._cached_page) != self._page_length:
            self.close()
            msg = (
                "failed to read complete page from file (read "
                f"{len(self._cached_page):d} of {self._page_length:d} bytes)"
            )
            raise ValueError(msg)

        self._read_page_header()
        if self._current_page_type in const.page_meta_types:
            self._process_page_metadata()

        if self._current_page_type not in const.page_meta_types + [
            const.page_data_type,
            const.page_mix_type,
        ]:
            return self._read_next_page()

        return False

    def _chunk_to_dataframe(self) -> DataFrame:
        n = self._current_row_in_chunk_index
        m = self._current_row_in_file_index
        ix = range(m - n, m)
        rslt = {}

        js, jb = 0, 0
        infer_string = get_option("future.infer_string")
        for j in range(self.column_count):
            name = self.column_names[j]

            if self._column_types[j] == b"d":
                col_arr = self._byte_chunk[jb, :].view(dtype=self.byte_order + "d")
                rslt[name] = pd.Series(col_arr, dtype=np.float64, index=ix, copy=False)
                if self.convert_dates:
                    if self.column_formats[j] in const.sas_date_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "d")
                    elif self.column_formats[j] in const.sas_datetime_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "s")
                jb += 1
            elif self._column_types[j] == b"s":
                rslt[name] = pd.Series(self._string_chunk[js, :], index=ix, copy=False)
                if self.convert_text and (self.encoding is not None):
                    rslt[name] = self._decode_string(rslt[name].str)
                    if infer_string:
                        rslt[name] = rslt[name].astype("str")

                js += 1
            else:
                self.close()
                raise ValueError(f"unknown column type {repr(self._column_types[j])}")

        df = DataFrame(rslt, columns=self.column_names, index=ix, copy=False)
        return df

    def _decode_string(self, b):
        return b.decode(self.encoding or self.default_encoding)

    def _convert_header_text(self, b: bytes) -> str | bytes:
        if self.convert_header_text:
            return self._decode_string(b)
        else:
            return b

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mssql\pyodbc.py ===
# dialects/mssql/pyodbc.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: mssql+pyodbc
    :name: PyODBC
    :dbapi: pyodbc
    :connectstring: mssql+pyodbc://<username>:<password>@<dsnname>
    :url: https://pypi.org/project/pyodbc/

Connecting to PyODBC
--------------------

The URL here is to be translated to PyODBC connection strings, as
detailed in `ConnectionStrings <https://code.google.com/p/pyodbc/wiki/ConnectionStrings>`_.

DSN Connections
^^^^^^^^^^^^^^^

A DSN connection in ODBC means that a pre-existing ODBC datasource is
configured on the client machine.   The application then specifies the name
of this datasource, which encompasses details such as the specific ODBC driver
in use as well as the network address of the database.   Assuming a datasource
is configured on the client, a basic DSN-based connection looks like::

    engine = create_engine("mssql+pyodbc://scott:tiger@some_dsn")

Which above, will pass the following connection string to PyODBC:

.. sourcecode:: text

    DSN=some_dsn;UID=scott;PWD=tiger

If the username and password are omitted, the DSN form will also add
the ``Trusted_Connection=yes`` directive to the ODBC string.

Hostname Connections
^^^^^^^^^^^^^^^^^^^^

Hostname-based connections are also supported by pyodbc.  These are often
easier to use than a DSN and have the additional advantage that the specific
database name to connect towards may be specified locally in the URL, rather
than it being fixed as part of a datasource configuration.

When using a hostname connection, the driver name must also be specified in the
query parameters of the URL.  As these names usually have spaces in them, the
name must be URL encoded which means using plus signs for spaces::

    engine = create_engine(
        "mssql+pyodbc://scott:tiger@myhost:port/databasename?driver=ODBC+Driver+17+for+SQL+Server"
    )

The ``driver`` keyword is significant to the pyodbc dialect and must be
specified in lowercase.

Any other names passed in the query string are passed through in the pyodbc
connect string, such as ``authentication``, ``TrustServerCertificate``, etc.
Multiple keyword arguments must be separated by an ampersand (``&``); these
will be translated to semicolons when the pyodbc connect string is generated
internally::

    e = create_engine(
        "mssql+pyodbc://scott:tiger@mssql2017:1433/test?"
        "driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
        "&authentication=ActiveDirectoryIntegrated"
    )

The equivalent URL can be constructed using :class:`_sa.engine.URL`::

    from sqlalchemy.engine import URL

    connection_url = URL.create(
        "mssql+pyodbc",
        username="scott",
        password="tiger",
        host="mssql2017",
        port=1433,
        database="test",
        query={
            "driver": "ODBC Driver 18 for SQL Server",
            "TrustServerCertificate": "yes",
            "authentication": "ActiveDirectoryIntegrated",
        },
    )

Pass through exact Pyodbc string
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A PyODBC connection string can also be sent in pyodbc's format directly, as
specified in `the PyODBC documentation
<https://github.com/mkleehammer/pyodbc/wiki/Connecting-to-databases>`_,
using the parameter ``odbc_connect``.  A :class:`_sa.engine.URL` object
can help make this easier::

    from sqlalchemy.engine import URL

    connection_string = "DRIVER={SQL Server Native Client 10.0};SERVER=dagger;DATABASE=test;UID=user;PWD=password"
    connection_url = URL.create(
        "mssql+pyodbc", query={"odbc_connect": connection_string}
    )

    engine = create_engine(connection_url)

.. _mssql_pyodbc_access_tokens:

Connecting to databases with access tokens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some database servers are set up to only accept access tokens for login. For
example, SQL Server allows the use of Azure Active Directory tokens to connect
to databases. This requires creating a credential object using the
``azure-identity`` library. More information about the authentication step can be
found in `Microsoft's documentation
<https://docs.microsoft.com/en-us/azure/developer/python/azure-sdk-authenticate?tabs=bash>`_.

After getting an engine, the credentials need to be sent to ``pyodbc.connect``
each time a connection is requested. One way to do this is to set up an event
listener on the engine that adds the credential token to the dialect's connect
call. This is discussed more generally in :ref:`engines_dynamic_tokens`. For
SQL Server in particular, this is passed as an ODBC connection attribute with
a data structure `described by Microsoft
<https://docs.microsoft.com/en-us/sql/connect/odbc/using-azure-active-directory#authenticating-with-an-access-token>`_.

The following code snippet will create an engine that connects to an Azure SQL
database using Azure credentials::

    import struct
    from sqlalchemy import create_engine, event
    from sqlalchemy.engine.url import URL
    from azure import identity

    # Connection option for access tokens, as defined in msodbcsql.h
    SQL_COPT_SS_ACCESS_TOKEN = 1256
    TOKEN_URL = "https://database.windows.net/"  # The token URL for any Azure SQL database

    connection_string = "mssql+pyodbc://@my-server.database.windows.net/myDb?driver=ODBC+Driver+17+for+SQL+Server"

    engine = create_engine(connection_string)

    azure_credentials = identity.DefaultAzureCredential()


    @event.listens_for(engine, "do_connect")
    def provide_token(dialect, conn_rec, cargs, cparams):
        # remove the "Trusted_Connection" parameter that SQLAlchemy adds
        cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")

        # create token credential
        raw_token = azure_credentials.get_token(TOKEN_URL).token.encode(
            "utf-16-le"
        )
        token_struct = struct.pack(
            f"<I{len(raw_token)}s", len(raw_token), raw_token
        )

        # apply it to keyword arguments
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}

.. tip::

    The ``Trusted_Connection`` token is currently added by the SQLAlchemy
    pyodbc dialect when no username or password is present.  This needs
    to be removed per Microsoft's
    `documentation for Azure access tokens
    <https://docs.microsoft.com/en-us/sql/connect/odbc/using-azure-active-directory#authenticating-with-an-access-token>`_,
    stating that a connection string when using an access token must not contain
    ``UID``, ``PWD``, ``Authentication`` or ``Trusted_Connection`` parameters.

.. _azure_synapse_ignore_no_transaction_on_rollback:

Avoiding transaction-related exceptions on Azure Synapse Analytics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Azure Synapse Analytics has a significant difference in its transaction
handling compared to plain SQL Server; in some cases an error within a Synapse
transaction can cause it to be arbitrarily terminated on the server side, which
then causes the DBAPI ``.rollback()`` method (as well as ``.commit()``) to
fail. The issue prevents the usual DBAPI contract of allowing ``.rollback()``
to pass silently if no transaction is present as the driver does not expect
this condition. The symptom of this failure is an exception with a message
resembling 'No corresponding transaction found. (111214)' when attempting to
emit a ``.rollback()`` after an operation had a failure of some kind.

This specific case can be handled by passing ``ignore_no_transaction_on_rollback=True`` to
the SQL Server dialect via the :func:`_sa.create_engine` function as follows::

    engine = create_engine(
        connection_url, ignore_no_transaction_on_rollback=True
    )

Using the above parameter, the dialect will catch ``ProgrammingError``
exceptions raised during ``connection.rollback()`` and emit a warning
if the error message contains code ``111214``, however will not raise
an exception.

.. versionadded:: 1.4.40  Added the
   ``ignore_no_transaction_on_rollback=True`` parameter.

Enable autocommit for Azure SQL Data Warehouse (DW) connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Azure SQL Data Warehouse does not support transactions,
and that can cause problems with SQLAlchemy's "autobegin" (and implicit
commit/rollback) behavior. We can avoid these problems by enabling autocommit
at both the pyodbc and engine levels::

    connection_url = sa.engine.URL.create(
        "mssql+pyodbc",
        username="scott",
        password="tiger",
        host="dw.azure.example.com",
        database="mydb",
        query={
            "driver": "ODBC Driver 17 for SQL Server",
            "autocommit": "True",
        },
    )

    engine = create_engine(connection_url).execution_options(
        isolation_level="AUTOCOMMIT"
    )

Avoiding sending large string parameters as TEXT/NTEXT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, for historical reasons, Microsoft's ODBC drivers for SQL Server
send long string parameters (greater than 4000 SBCS characters or 2000 Unicode
characters) as TEXT/NTEXT values. TEXT and NTEXT have been deprecated for many
years and are starting to cause compatibility issues with newer versions of
SQL_Server/Azure. For example, see `this
issue <https://github.com/mkleehammer/pyodbc/issues/835>`_.

Starting with ODBC Driver 18 for SQL Server we can override the legacy
behavior and pass long strings as varchar(max)/nvarchar(max) using the
``LongAsMax=Yes`` connection string parameter::

    connection_url = sa.engine.URL.create(
        "mssql+pyodbc",
        username="scott",
        password="tiger",
        host="mssqlserver.example.com",
        database="mydb",
        query={
            "driver": "ODBC Driver 18 for SQL Server",
            "LongAsMax": "Yes",
        },
    )

Pyodbc Pooling / connection close behavior
------------------------------------------

PyODBC uses internal `pooling
<https://github.com/mkleehammer/pyodbc/wiki/The-pyodbc-Module#pooling>`_ by
default, which means connections will be longer lived than they are within
SQLAlchemy itself.  As SQLAlchemy has its own pooling behavior, it is often
preferable to disable this behavior.  This behavior can only be disabled
globally at the PyODBC module level, **before** any connections are made::

    import pyodbc

    pyodbc.pooling = False

    # don't use the engine before pooling is set to False
    engine = create_engine("mssql+pyodbc://user:pass@dsn")

If this variable is left at its default value of ``True``, **the application
will continue to maintain active database connections**, even when the
SQLAlchemy engine itself fully discards a connection or if the engine is
disposed.

.. seealso::

    `pooling <https://github.com/mkleehammer/pyodbc/wiki/The-pyodbc-Module#pooling>`_ -
    in the PyODBC documentation.

Driver / Unicode Support
-------------------------

PyODBC works best with Microsoft ODBC drivers, particularly in the area
of Unicode support on both Python 2 and Python 3.

Using the FreeTDS ODBC drivers on Linux or OSX with PyODBC is **not**
recommended; there have been historically many Unicode-related issues
in this area, including before Microsoft offered ODBC drivers for Linux
and OSX.   Now that Microsoft offers drivers for all platforms, for
PyODBC support these are recommended.  FreeTDS remains relevant for
non-ODBC drivers such as pymssql where it works very well.


Rowcount Support
----------------

Previous limitations with the SQLAlchemy ORM's "versioned rows" feature with
Pyodbc have been resolved as of SQLAlchemy 2.0.5. See the notes at
:ref:`mssql_rowcount_versioning`.

.. _mssql_pyodbc_fastexecutemany:

Fast Executemany Mode
---------------------

The PyODBC driver includes support for a "fast executemany" mode of execution
which greatly reduces round trips for a DBAPI ``executemany()`` call when using
Microsoft ODBC drivers, for **limited size batches that fit in memory**.  The
feature is enabled by setting the attribute ``.fast_executemany`` on the DBAPI
cursor when an executemany call is to be used.   The SQLAlchemy PyODBC SQL
Server dialect supports this parameter by passing the
``fast_executemany`` parameter to
:func:`_sa.create_engine` , when using the **Microsoft ODBC driver only**::

    engine = create_engine(
        "mssql+pyodbc://scott:tiger@mssql2017:1433/test?driver=ODBC+Driver+17+for+SQL+Server",
        fast_executemany=True,
    )

.. versionchanged:: 2.0.9 - the ``fast_executemany`` parameter now has its
   intended effect of this PyODBC feature taking effect for all INSERT
   statements that are executed with multiple parameter sets, which don't
   include RETURNING.  Previously, SQLAlchemy 2.0's :term:`insertmanyvalues`
   feature would cause ``fast_executemany`` to not be used in most cases
   even if specified.

.. versionadded:: 1.3

.. seealso::

    `fast executemany <https://github.com/mkleehammer/pyodbc/wiki/Features-beyond-the-DB-API#fast_executemany>`_
    - on github

.. _mssql_pyodbc_setinputsizes:

Setinputsizes Support
-----------------------

As of version 2.0, the pyodbc ``cursor.setinputsizes()`` method is used for
all statement executions, except for ``cursor.executemany()`` calls when
fast_executemany=True where it is not supported (assuming
:ref:`insertmanyvalues <engine_insertmanyvalues>` is kept enabled,
"fastexecutemany" will not take place for INSERT statements in any case).

The use of ``cursor.setinputsizes()`` can be disabled by passing
``use_setinputsizes=False`` to :func:`_sa.create_engine`.

When ``use_setinputsizes`` is left at its default of ``True``, the
specific per-type symbols passed to ``cursor.setinputsizes()`` can be
programmatically customized using the :meth:`.DialectEvents.do_setinputsizes`
hook. See that method for usage examples.

.. versionchanged:: 2.0  The mssql+pyodbc dialect now defaults to using
   ``use_setinputsizes=True`` for all statement executions with the exception of
   cursor.executemany() calls when fast_executemany=True.  The behavior can
   be turned off by passing ``use_setinputsizes=False`` to
   :func:`_sa.create_engine`.

"""  # noqa


import datetime
import decimal
import re
import struct

from .base import _MSDateTime
from .base import _MSUnicode
from .base import _MSUnicodeText
from .base import BINARY
from .base import DATETIMEOFFSET
from .base import MSDialect
from .base import MSExecutionContext
from .base import VARBINARY
from .json import JSON as _MSJson
from .json import JSONIndexType as _MSJsonIndexType
from .json import JSONPathType as _MSJsonPathType
from ... import exc
from ... import types as sqltypes
from ... import util
from ...connectors.pyodbc import PyODBCConnector
from ...engine import cursor as _cursor


class _ms_numeric_pyodbc:
    """Turns Decimals with adjusted() < 0 or > 7 into strings.

    The routines here are needed for older pyodbc versions
    as well as current mxODBC versions.

    """

    def bind_processor(self, dialect):
        super_process = super().bind_processor(dialect)

        if not dialect._need_decimal_fix:
            return super_process

        def process(value):
            if self.asdecimal and isinstance(value, decimal.Decimal):
                adjusted = value.adjusted()
                if adjusted < 0:
                    return self._small_dec_to_string(value)
                elif adjusted > 7:
                    return self._large_dec_to_string(value)

            if super_process:
                return super_process(value)
            else:
                return value

        return process

    # these routines needed for older versions of pyodbc.
    # as of 2.1.8 this logic is integrated.

    def _small_dec_to_string(self, value):
        return "%s0.%s%s" % (
            (value < 0 and "-" or ""),
            "0" * (abs(value.adjusted()) - 1),
            "".join([str(nint) for nint in value.as_tuple()[1]]),
        )

    def _large_dec_to_string(self, value):
        _int = value.as_tuple()[1]
        if "E" in str(value):
            result = "%s%s%s" % (
                (value < 0 and "-" or ""),
                "".join([str(s) for s in _int]),
                "0" * (value.adjusted() - (len(_int) - 1)),
            )
        else:
            if (len(_int) - 1) > value.adjusted():
                result = "%s%s.%s" % (
                    (value < 0 and "-" or ""),
                    "".join([str(s) for s in _int][0 : value.adjusted() + 1]),
                    "".join([str(s) for s in _int][value.adjusted() + 1 :]),
                )
            else:
                result = "%s%s" % (
                    (value < 0 and "-" or ""),
                    "".join([str(s) for s in _int][0 : value.adjusted() + 1]),
                )
        return result


class _MSNumeric_pyodbc(_ms_numeric_pyodbc, sqltypes.Numeric):
    pass


class _MSFloat_pyodbc(_ms_numeric_pyodbc, sqltypes.Float):
    pass


class _ms_binary_pyodbc:
    """Wraps binary values in dialect-specific Binary wrapper.
    If the value is null, return a pyodbc-specific BinaryNull
    object to prevent pyODBC [and FreeTDS] from defaulting binary
    NULL types to SQLWCHAR and causing implicit conversion errors.
    """

    def bind_processor(self, dialect):
        if dialect.dbapi is None:
            return None

        DBAPIBinary = dialect.dbapi.Binary

        def process(value):
            if value is not None:
                return DBAPIBinary(value)
            else:
                # pyodbc-specific
                return dialect.dbapi.BinaryNull

        return process


class _ODBCDateTimeBindProcessor:
    """Add bind processors to handle datetimeoffset behaviors"""

    has_tz = False

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            elif isinstance(value, str):
                # if a string was passed directly, allow it through
                return value
            elif not value.tzinfo or (not self.timezone and not self.has_tz):
                # for DateTime(timezone=False)
                return value
            else:
                # for DATETIMEOFFSET or DateTime(timezone=True)
                #
                # Convert to string format required by T-SQL
                dto_string = value.strftime("%Y-%m-%d %H:%M:%S.%f %z")
                # offset needs a colon, e.g., -0700 -> -07:00
                # "UTC offset in the form (+-)HHMM[SS[.ffffff]]"
                # backend currently rejects seconds / fractional seconds
                dto_string = re.sub(
                    r"([\+\-]\d{2})([\d\.]+)$", r"\1:\2", dto_string
                )
                return dto_string

        return process


class _ODBCDateTime(_ODBCDateTimeBindProcessor, _MSDateTime):
    pass


class _ODBCDATETIMEOFFSET(_ODBCDateTimeBindProcessor, DATETIMEOFFSET):
    has_tz = True


class _VARBINARY_pyodbc(_ms_binary_pyodbc, VARBINARY):
    pass


class _BINARY_pyodbc(_ms_binary_pyodbc, BINARY):
    pass


class _String_pyodbc(sqltypes.String):
    def get_dbapi_type(self, dbapi):
        if self.length in (None, "max") or self.length >= 2000:
            return (dbapi.SQL_VARCHAR, 0, 0)
        else:
            return dbapi.SQL_VARCHAR


class _Unicode_pyodbc(_MSUnicode):
    def get_dbapi_type(self, dbapi):
        if self.length in (None, "max") or self.length >= 2000:
            return (dbapi.SQL_WVARCHAR, 0, 0)
        else:
            return dbapi.SQL_WVARCHAR


class _UnicodeText_pyodbc(_MSUnicodeText):
    def get_dbapi_type(self, dbapi):
        if self.length in (None, "max") or self.length >= 2000:
            return (dbapi.SQL_WVARCHAR, 0, 0)
        else:
            return dbapi.SQL_WVARCHAR


class _JSON_pyodbc(_MSJson):
    def get_dbapi_type(self, dbapi):
        return (dbapi.SQL_WVARCHAR, 0, 0)


class _JSONIndexType_pyodbc(_MSJsonIndexType):
    def get_dbapi_type(self, dbapi):
        return dbapi.SQL_WVARCHAR


class _JSONPathType_pyodbc(_MSJsonPathType):
    def get_dbapi_type(self, dbapi):
        return dbapi.SQL_WVARCHAR


class MSExecutionContext_pyodbc(MSExecutionContext):
    _embedded_scope_identity = False

    def pre_exec(self):
        """where appropriate, issue "select scope_identity()" in the same
        statement.

        Background on why "scope_identity()" is preferable to "@@identity":
        https://msdn.microsoft.com/en-us/library/ms190315.aspx

        Background on why we attempt to embed "scope_identity()" into the same
        statement as the INSERT:
        https://code.google.com/p/pyodbc/wiki/FAQs#How_do_I_retrieve_autogenerated/identity_values?

        """

        super().pre_exec()

        # don't embed the scope_identity select into an
        # "INSERT .. DEFAULT VALUES"
        if (
            self._select_lastrowid
            and self.dialect.use_scope_identity
            and len(self.parameters[0])
        ):
            self._embedded_scope_identity = True

            self.statement += "; select scope_identity()"

    def post_exec(self):
        if self._embedded_scope_identity:
            # Fetch the last inserted id from the manipulated statement
            # We may have to skip over a number of result sets with
            # no data (due to triggers, etc.)
            while True:
                try:
                    # fetchall() ensures the cursor is consumed
                    # without closing it (FreeTDS particularly)
                    rows = self.cursor.fetchall()
                except self.dialect.dbapi.Error:
                    # no way around this - nextset() consumes the previous set
                    # so we need to just keep flipping
                    self.cursor.nextset()
                else:
                    if not rows:
                        # async adapter drivers just return None here
                        self.cursor.nextset()
                        continue
                    row = rows[0]
                    break

            self._lastrowid = int(row[0])

            self.cursor_fetch_strategy = _cursor._NO_CURSOR_DML
        else:
            super().post_exec()


class MSDialect_pyodbc(PyODBCConnector, MSDialect):
    supports_statement_cache = True

    # note this parameter is no longer used by the ORM or default dialect
    # see #9414
    supports_sane_rowcount_returning = False

    execution_ctx_cls = MSExecutionContext_pyodbc

    colspecs = util.update_copy(
        MSDialect.colspecs,
        {
            sqltypes.Numeric: _MSNumeric_pyodbc,
            sqltypes.Float: _MSFloat_pyodbc,
            BINARY: _BINARY_pyodbc,
            # support DateTime(timezone=True)
            sqltypes.DateTime: _ODBCDateTime,
            DATETIMEOFFSET: _ODBCDATETIMEOFFSET,
            # SQL Server dialect has a VARBINARY that is just to support
            # "deprecate_large_types" w/ VARBINARY(max), but also we must
            # handle the usual SQL standard VARBINARY
            VARBINARY: _VARBINARY_pyodbc,
            sqltypes.VARBINARY: _VARBINARY_pyodbc,
            sqltypes.LargeBinary: _VARBINARY_pyodbc,
            sqltypes.String: _String_pyodbc,
            sqltypes.Unicode: _Unicode_pyodbc,
            sqltypes.UnicodeText: _UnicodeText_pyodbc,
            sqltypes.JSON: _JSON_pyodbc,
            sqltypes.JSON.JSONIndexType: _JSONIndexType_pyodbc,
            sqltypes.JSON.JSONPathType: _JSONPathType_pyodbc,
            # this excludes Enum from the string/VARCHAR thing for now
            # it looks like Enum's adaptation doesn't really support the
            # String type itself having a dialect-level impl
            sqltypes.Enum: sqltypes.Enum,
        },
    )

    def __init__(
        self,
        fast_executemany=False,
        use_setinputsizes=True,
        **params,
    ):
        super().__init__(use_setinputsizes=use_setinputsizes, **params)
        self.use_scope_identity = (
            self.use_scope_identity
            and self.dbapi
            and hasattr(self.dbapi.Cursor, "nextset")
        )
        self._need_decimal_fix = self.dbapi and self._dbapi_version() < (
            2,
            1,
            8,
        )
        self.fast_executemany = fast_executemany
        if fast_executemany:
            self.use_insertmanyvalues_wo_returning = False

    def _get_server_version_info(self, connection):
        try:
            # "Version of the instance of SQL Server, in the form
            # of 'major.minor.build.revision'"
            raw = connection.exec_driver_sql(
                "SELECT CAST(SERVERPROPERTY('ProductVersion') AS VARCHAR)"
            ).scalar()
        except exc.DBAPIError:
            # SQL Server docs indicate this function isn't present prior to
            # 2008.  Before we had the VARCHAR cast above, pyodbc would also
            # fail on this query.
            return super()._get_server_version_info(connection)
        else:
            version = []
            r = re.compile(r"[.\-]")
            for n in r.split(raw):
                try:
                    version.append(int(n))
                except ValueError:
                    pass
            return tuple(version)

    def on_connect(self):
        super_ = super().on_connect()

        def on_connect(conn):
            if super_ is not None:
                super_(conn)

            self._setup_timestampoffset_type(conn)

        return on_connect

    def _setup_timestampoffset_type(self, connection):
        # output converter function for datetimeoffset
        def _handle_datetimeoffset(dto_value):
            tup = struct.unpack("<6hI2h", dto_value)
            return datetime.datetime(
                tup[0],
                tup[1],
                tup[2],
                tup[3],
                tup[4],
                tup[5],
                tup[6] // 1000,
                datetime.timezone(
                    datetime.timedelta(hours=tup[7], minutes=tup[8])
                ),
            )

        odbc_SQL_SS_TIMESTAMPOFFSET = -155  # as defined in SQLNCLI.h
        connection.add_output_converter(
            odbc_SQL_SS_TIMESTAMPOFFSET, _handle_datetimeoffset
        )

    def do_executemany(self, cursor, statement, parameters, context=None):
        if self.fast_executemany:
            cursor.fast_executemany = True
        super().do_executemany(cursor, statement, parameters, context=context)

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.Error):
            code = e.args[0]
            if code in {
                "08S01",
                "01000",
                "01002",
                "08003",
                "08007",
                "08S02",
                "08001",
                "HYT00",
                "HY010",
                "10054",
            }:
                return True
        return super().is_disconnect(e, connection, cursor)


dialect = MSDialect_pyodbc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\filter.py ===
from __future__ import annotations
from collections import defaultdict
import re
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)
import warnings

from bs4._deprecation import _deprecated
from bs4.element import (
    AttributeDict,
    NavigableString,
    PageElement,
    ResultSet,
    Tag,
)
from bs4._typing import (
    _AtMostOneElement,
    _AttributeValue,
    _OneElement,
    _PageElementMatchFunction,
    _QueryResults,
    _RawAttributeValues,
    _RegularExpressionProtocol,
    _StrainableAttribute,
    _StrainableElement,
    _StrainableString,
    _StringMatchFunction,
    _TagMatchFunction,
)


class ElementFilter(object):
    """`ElementFilter` encapsulates the logic necessary to decide:

    1. whether a `PageElement` (a `Tag` or a `NavigableString`) matches a
    user-specified query.

    2. whether a given sequence of markup found during initial parsing
    should be turned into a `PageElement` at all, or simply discarded.

    The base class is the simplest `ElementFilter`. By default, it
    matches everything and allows all markup to become `PageElement`
    objects. You can make it more selective by passing in a
    user-defined match function, or defining a subclass.

    Most users of Beautiful Soup will never need to use
    `ElementFilter`, or its more capable subclass
    `SoupStrainer`. Instead, they will use methods like
    :py:meth:`Tag.find`, which will convert their arguments into
    `SoupStrainer` objects and run them against the tree.

    However, if you find yourself wanting to treat the arguments to
    Beautiful Soup's find_*() methods as first-class objects, those
    objects will be `SoupStrainer` objects. You can create them
    yourself and then make use of functions like
    `ElementFilter.filter()`.
    """

    match_function: Optional[_PageElementMatchFunction]

    def __init__(self, match_function: Optional[_PageElementMatchFunction] = None):
        """Pass in a match function to easily customize the behavior of
        `ElementFilter.match` without needing to subclass.

        :param match_function: A function that takes a `PageElement`
          and returns `True` if that `PageElement` matches some criteria.
        """
        self.match_function = match_function

    @property
    def includes_everything(self) -> bool:
        """Does this `ElementFilter` obviously include everything? If so,
        the filter process can be made much faster.

        The `ElementFilter` might turn out to include everything even
        if this returns `False`, but it won't include everything in an
        obvious way.

        The base `ElementFilter` implementation includes things based on
        the match function, so includes_everything is only true if
        there is no match function.
        """
        return not self.match_function

    @property
    def excludes_everything(self) -> bool:
        """Does this `ElementFilter` obviously exclude everything? If
        so, Beautiful Soup will issue a warning if you try to use it
        when parsing a document.

        The `ElementFilter` might turn out to exclude everything even
        if this returns `False`, but it won't exclude everything in an
        obvious way.

        The base `ElementFilter` implementation excludes things based
        on a match function we can't inspect, so excludes_everything
        is always false.
        """
        return False

    def match(self, element: PageElement, _known_rules:bool=False) -> bool:
        """Does the given PageElement match the rules set down by this
        ElementFilter?

        The base implementation delegates to the function passed in to
        the constructor.

        :param _known_rules: Defined for compatibility with
        SoupStrainer._match(). Used more for consistency than because
        we need the performance optimization.
        """
        if not _known_rules and self.includes_everything:
            return True
        if not self.match_function:
            return True
        return self.match_function(element)

    def filter(self, generator: Iterator[PageElement]) -> Iterator[_OneElement]:
        """The most generic search method offered by Beautiful Soup.

        Acts like Python's built-in `filter`, using
        `ElementFilter.match` as the filtering function.
        """
        # If there are no rules at all, don't bother filtering. Let
        # anything through.
        if self.includes_everything:
            for i in generator:
                yield i
        while True:
            try:
                i = next(generator)
            except StopIteration:
                break
            if i:
                if self.match(i, _known_rules=True):
                    yield cast("_OneElement", i)

    def find(self, generator: Iterator[PageElement]) -> _AtMostOneElement:
        """A lower-level equivalent of :py:meth:`Tag.find`.

        You can pass in your own generator for iterating over
        `PageElement` objects. The first one that matches this
        `ElementFilter` will be returned.

        :param generator: A way of iterating over `PageElement`
            objects.
        """
        for match in self.filter(generator):
            return match
        return None

    def find_all(
        self, generator: Iterator[PageElement], limit: Optional[int] = None
    ) -> _QueryResults:
        """A lower-level equivalent of :py:meth:`Tag.find_all`.

        You can pass in your own generator for iterating over
        `PageElement` objects. Only elements that match this
        `ElementFilter` will be returned in the :py:class:`ResultSet`.

        :param generator: A way of iterating over `PageElement`
            objects.

        :param limit: Stop looking after finding this many results.
        """
        results: _QueryResults = ResultSet(self)
        for match in self.filter(generator):
            results.append(match)
            if limit is not None and len(results) >= limit:
                break
        return results

    def allow_tag_creation(
        self, nsprefix: Optional[str], name: str, attrs: Optional[_RawAttributeValues]
    ) -> bool:
        """Based on the name and attributes of a tag, see whether this
        `ElementFilter` will allow a `Tag` object to even be created.

        By default, all tags are parsed. To change this, subclass
        `ElementFilter`.

        :param name: The name of the prospective tag.
        :param attrs: The attributes of the prospective tag.
        """
        return True

    def allow_string_creation(self, string: str) -> bool:
        """Based on the content of a string, see whether this
        `ElementFilter` will allow a `NavigableString` object based on
        this string to be added to the parse tree.

        By default, all strings are processed into `NavigableString`
        objects. To change this, subclass `ElementFilter`.

        :param str: The string under consideration.
        """
        return True


class MatchRule(object):
    """Each MatchRule encapsulates the logic behind a single argument
    passed in to one of the Beautiful Soup find* methods.
    """

    string: Optional[str]
    pattern: Optional[_RegularExpressionProtocol]
    present: Optional[bool]
    exclude_everything: Optional[bool]
    # TODO-TYPING: All MatchRule objects also have an attribute
    # ``function``, but the type of the function depends on the
    # subclass.

    def __init__(
        self,
        string: Optional[Union[str, bytes]] = None,
        pattern: Optional[_RegularExpressionProtocol] = None,
        function: Optional[Callable] = None,
        present: Optional[bool] = None,
        exclude_everything: Optional[bool] = None
    ):
        if isinstance(string, bytes):
            string = string.decode("utf8")
        self.string = string
        if isinstance(pattern, bytes):
            self.pattern = re.compile(pattern.decode("utf8"))
        elif isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern
        self.function = function
        self.present = present
        self.exclude_everything = exclude_everything

        values = [
            x
            for x in (self.string, self.pattern, self.function, self.present, self.exclude_everything)
            if x is not None
        ]
        if len(values) == 0:
            raise ValueError(
                "Either string, pattern, function, present, or exclude_everything must be provided."
            )
        if len(values) > 1:
            raise ValueError(
                "At most one of string, pattern, function, present, and exclude_everything must be provided."
            )

    def _base_match(self, string: Optional[str]) -> Optional[bool]:
        """Run the 'cheap' portion of a match, trying to get an answer without
        calling a potentially expensive custom function.

        :return: True or False if we have a (positive or negative)
        match; None if we need to keep trying.
        """
        # self.exclude_everything matches nothing.
        if self.exclude_everything:
            return False

        # self.present==True matches everything except None.
        if self.present is True:
            return string is not None

        # self.present==False matches _only_ None.
        if self.present is False:
            return string is None

        # self.string does an exact string match.
        if self.string is not None:
            # print(f"{self.string} ?= {string}")
            return self.string == string

        # self.pattern does a regular expression search.
        if self.pattern is not None:
            # print(f"{self.pattern} ?~ {string}")
            if string is None:
                return False
            return self.pattern.search(string) is not None

        return None

    def matches_string(self, string: Optional[str]) -> bool:
        _base_result = self._base_match(string)
        if _base_result is not None:
            # No need to invoke the test function.
            return _base_result
        if self.function is not None and not self.function(string):
            # print(f"{self.function}({string}) == False")
            return False
        return True

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls} string={self.string} pattern={self.pattern} function={self.function} present={self.present}>"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, MatchRule)
            and self.string == other.string
            and self.pattern == other.pattern
            and self.function == other.function
            and self.present == other.present
        )


class TagNameMatchRule(MatchRule):
    """A MatchRule implementing the rules for matches against tag name."""

    function: Optional[_TagMatchFunction]

    def matches_tag(self, tag: Tag) -> bool:
        base_value = self._base_match(tag.name)
        if base_value is not None:
            return base_value

        # The only remaining possibility is that the match is determined
        # by a function call. Call the function.
        function = cast(_TagMatchFunction, self.function)
        if function(tag):
            return True
        return False


class AttributeValueMatchRule(MatchRule):
    """A MatchRule implementing the rules for matches against attribute value."""

    function: Optional[_StringMatchFunction]


class StringMatchRule(MatchRule):
    """A MatchRule implementing the rules for matches against a NavigableString."""

    function: Optional[_StringMatchFunction]


class SoupStrainer(ElementFilter):
    """The `ElementFilter` subclass used internally by Beautiful Soup.

    A `SoupStrainer` encapsulates the logic necessary to perform the
    kind of matches supported by methods such as
    :py:meth:`Tag.find`. `SoupStrainer` objects are primarily created
    internally, but you can create one yourself and pass it in as
    ``parse_only`` to the `BeautifulSoup` constructor, to parse a
    subset of a large document.

    Internally, `SoupStrainer` objects work by converting the
    constructor arguments into `MatchRule` objects. Incoming
    tags/markup are matched against those rules.

    :param name: One or more restrictions on the tags found in a document.

    :param attrs: A dictionary that maps attribute names to
      restrictions on tags that use those attributes.

    :param string: One or more restrictions on the strings found in a
      document.

    :param kwargs: A dictionary that maps attribute names to restrictions
      on tags that use those attributes. These restrictions are additive to
      any specified in ``attrs``.

    """

    name_rules: List[TagNameMatchRule]
    attribute_rules: Dict[str, List[AttributeValueMatchRule]]
    string_rules: List[StringMatchRule]

    def __init__(
        self,
        name: Optional[_StrainableElement] = None,
        attrs: Dict[str, _StrainableAttribute] = {},
        string: Optional[_StrainableString] = None,
        **kwargs: _StrainableAttribute,
    ):
        if string is None and "text" in kwargs:
            string = cast(Optional[_StrainableString], kwargs.pop("text"))
            warnings.warn(
                "As of version 4.11.0, the 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if name is None and not attrs and not string and not kwargs:
            # Special case for backwards compatibility. Instantiating
            # a SoupStrainer with no arguments whatsoever gets you one
            # that matches all Tags, and only Tags.
            self.name_rules = [TagNameMatchRule(present=True)]
        else:
                self.name_rules = cast(
                    List[TagNameMatchRule], list(self._make_match_rules(name, TagNameMatchRule))
                )
        self.attribute_rules = defaultdict(list)

        if not isinstance(attrs, dict):
            # Passing something other than a dictionary as attrs is
            # sugar for matching that thing against the 'class'
            # attribute.
            attrs = {"class": attrs}

        for attrdict in attrs, kwargs:
            for attr, value in attrdict.items():
                if attr == "class_" and attrdict is kwargs:
                    # If you pass in 'class_' as part of kwargs, it's
                    # because class is a Python reserved word. If you
                    # pass it in as part of the attrs dict, it's
                    # because you really are looking for an attribute
                    # called 'class_'.
                    attr = "class"

                if value is None:
                    value = False
                for rule_obj in self._make_match_rules(value, AttributeValueMatchRule):
                    self.attribute_rules[attr].append(
                        cast(AttributeValueMatchRule, rule_obj)
                    )

        self.string_rules = cast(
            List[StringMatchRule], list(self._make_match_rules(string, StringMatchRule))
        )

        #: DEPRECATED 4.13.0: You shouldn't need to check this under
        #: any name (.string or .text), and if you do, you're probably
        #: not taking into account all of the types of values this
        #: variable might have. Look at the .string_rules list instead.
        self.__string = string

    @property
    def includes_everything(self) -> bool:
        """Check whether the provided rules will obviously include
        everything. (They might include everything even if this returns `False`,
        but not in an obvious way.)
        """
        return not self.name_rules and not self.string_rules and not self.attribute_rules

    @property
    def excludes_everything(self) -> bool:
        """Check whether the provided rules will obviously exclude
        everything. (They might exclude everything even if this returns `False`,
        but not in an obvious way.)
        """
        if (self.string_rules and (self.name_rules or self.attribute_rules)):
            # This is self-contradictory, so the rules exclude everything.
            return True

        # If there's a rule that ended up treated as an "exclude everything"
        # rule due to creating a logical inconsistency, then the rules
        # exclude everything.
        if any(x.exclude_everything for x in self.string_rules):
            return True
        if any(x.exclude_everything for x in self.name_rules):
            return True
        for ruleset in self.attribute_rules.values():
            if any(x.exclude_everything for x in ruleset):
                return True
        return False

    @property
    def string(self) -> Optional[_StrainableString]:
        ":meta private:"
        warnings.warn(
            "Access to deprecated property string. (Look at .string_rules instead) -- Deprecated since version 4.13.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.__string

    @property
    def text(self) -> Optional[_StrainableString]:
        ":meta private:"
        warnings.warn(
            "Access to deprecated property text. (Look at .string_rules instead) -- Deprecated since version 4.13.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.__string

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name_rules} attrs={self.attribute_rules} string={self.string_rules}>"

    @classmethod
    def _make_match_rules(
        cls,
        obj: Optional[Union[_StrainableElement, _StrainableAttribute]],
        rule_class: Type[MatchRule],
    ) -> Iterator[MatchRule]:
        """Convert a vaguely-specific 'object' into one or more well-defined
        `MatchRule` objects.

        :param obj: Some kind of object that corresponds to one or more
           matching rules.
        :param rule_class: Create instances of this `MatchRule` subclass.
        """
        if obj is None:
            return
        if isinstance(obj, (str, bytes)):
            yield rule_class(string=obj)
        elif isinstance(obj, bool):
            yield rule_class(present=obj)
        elif callable(obj):
            yield rule_class(function=obj)
        elif isinstance(obj, _RegularExpressionProtocol):
            yield rule_class(pattern=obj)
        elif hasattr(obj, "__iter__"):
            if not obj:
                # The attribute is being matched against the null set,
                # which means it should exclude everything.
                yield rule_class(exclude_everything=True)
            for o in obj:
                if not isinstance(o, (bytes, str)) and hasattr(o, "__iter__"):
                    # This is almost certainly the user's
                    # mistake. This list contains another list, which
                    # opens up the possibility of infinite
                    # self-reference. In the interests of avoiding
                    # infinite recursion, we'll treat this as an
                    # impossible match and issue a rule that excludes
                    # everything, rather than looking inside.
                    warnings.warn(
                        f"Ignoring nested list {o} to avoid the possibility of infinite recursion.",
                        stacklevel=5,
                    )
                    yield rule_class(exclude_everything=True)
                    continue
                for x in cls._make_match_rules(o, rule_class):
                    yield x
        else:
            yield rule_class(string=str(obj))

    def matches_tag(self, tag: Tag) -> bool:
        """Do the rules of this `SoupStrainer` trigger a match against the
        given `Tag`?

        If the `SoupStrainer` has any `TagNameMatchRule`, at least one
        must match the `Tag` or its `Tag.name`.

        If there are any `AttributeValueMatchRule` for a given
        attribute, at least one of them must match the attribute
        value.

        If there are any `StringMatchRule`, at least one must match,
        but a `SoupStrainer` that *only* contains `StringMatchRule`
        cannot match a `Tag`, only a `NavigableString`.
        """
        # If there are no rules at all, let anything through.
        #if self.includes_everything:
        #    return True

        # String rules cannot not match a Tag on their own.
        if not self.name_rules and not self.attribute_rules:
            return False

        # Optimization for a very common case where the user is
        # searching for a tag with one specific name, and we're
        # looking at a tag with a different name.
        if (
            not tag.prefix
            and len(self.name_rules) == 1
            and self.name_rules[0].string is not None
            and tag.name != self.name_rules[0].string
        ):
            return False

        # If there are name rules, at least one must match. It can
        # match either the Tag object itself or the prefixed name of
        # the tag.
        prefixed_name = None
        if tag.prefix:
            prefixed_name = f"{tag.prefix}:{tag.name}"
        if self.name_rules:
            name_matches = False
            for rule in self.name_rules:
                # attrs = " ".join(
                #     [f"{k}={v}" for k, v in sorted(tag.attrs.items())]
                # )
                # print(f"Testing <{tag.name} {attrs}>{tag.string}</{tag.name}> against {rule}")

                # If the rule contains a function, the function will be called
                # with `tag`. It will not be called a second time with
                # `prefixed_name`.
                if rule.matches_tag(tag) or (
                        not rule.function and prefixed_name is not None and rule.matches_string(prefixed_name)
                ):
                    name_matches = True
                    break

            if not name_matches:
                return False

        # If there are attribute rules for a given attribute, at least
        # one of them must match. If there are rules for multiple
        # attributes, each attribute must have at least one match.
        for attr, rules in self.attribute_rules.items():
            attr_value = tag.get(attr, None)
            this_attr_match = self._attribute_match(attr_value, rules)
            if not this_attr_match:
                return False

        # If there are string rules, at least one must match.
        if self.string_rules:
            _str = tag.string
            if _str is None:
                return False
            if not self.matches_any_string_rule(_str):
                return False
        return True

    def _attribute_match(
        self,
        attr_value: Optional[_AttributeValue],
        rules: Iterable[AttributeValueMatchRule],
    ) -> bool:
        attr_values: Sequence[Optional[str]]
        if isinstance(attr_value, list):
            attr_values = attr_value
        else:
            attr_values = [cast(str, attr_value)]

        def _match_attribute_value_helper(attr_values: Sequence[Optional[str]]) -> bool:
            for rule in rules:
                for attr_value in attr_values:
                    if rule.matches_string(attr_value):
                        return True
            return False

        this_attr_match = _match_attribute_value_helper(attr_values)
        if not this_attr_match and len(attr_values) > 1:
            # This cast converts Optional[str] to plain str.
            #
            # We know if there's more than one value, there can't be
            # any None in the list, because Beautiful Soup never uses
            # None as a value of a multi-valued attribute, and if None
            # is passed in as attr_value, it's turned into a list with
            # a single element (thus len(attr_values) > 1 fails).
            attr_values = cast(Sequence[str], attr_values)

            # Try again but treat the attribute value
            # as a single string.
            joined_attr_value = " ".join(attr_values)
            this_attr_match = _match_attribute_value_helper([joined_attr_value])
        return this_attr_match

    def allow_tag_creation(
        self, nsprefix: Optional[str], name: str, attrs: Optional[_RawAttributeValues]
    ) -> bool:
        """Based on the name and attributes of a tag, see whether this
        `SoupStrainer` will allow a `Tag` object to even be created.

        :param name: The name of the prospective tag.
        :param attrs: The attributes of the prospective tag.
        """
        if self.string_rules:
            # A SoupStrainer that has string rules can't be used to
            # manage tag creation, because the string rule can't be
            # evaluated until after the tag and all of its contents
            # have been parsed.
            return False
        prefixed_name = None
        if nsprefix:
            prefixed_name = f"{nsprefix}:{name}"
        if self.name_rules:
            # At least one name rule must match.
            name_match = False
            for rule in self.name_rules:
                for x in name, prefixed_name:
                    if x is not None:
                        if rule.matches_string(x):
                            name_match = True
                            break
            if not name_match:
                return False

        # For each attribute that has rules, at least one rule must
        # match.
        if attrs is None:
            attrs = AttributeDict()
        for attr, rules in self.attribute_rules.items():
            attr_value = attrs.get(attr)
            if not self._attribute_match(attr_value, rules):
                return False

        return True

    def allow_string_creation(self, string: str) -> bool:
        """Based on the content of a markup string, see whether this
        `SoupStrainer` will allow it to be instantiated as a
        `NavigableString` object, or whether it should be ignored.
        """
        if self.name_rules or self.attribute_rules:
            # A SoupStrainer that has name or attribute rules won't
            # match any strings; it's designed to match tags with
            # certain properties.
            return False
        if not self.string_rules:
            # A SoupStrainer with no string rules will match
            # all strings.
            return True
        if not self.matches_any_string_rule(string):
            return False
        return True

    def matches_any_string_rule(self, string: str) -> bool:
        """See whether the content of a string matches any of
        this `SoupStrainer`'s string rules.
        """
        if not self.string_rules:
            return True
        for string_rule in self.string_rules:
            if string_rule.matches_string(string):
                return True
        return False

    def match(self, element: PageElement, _known_rules: bool=False) -> bool:
        """Does the given `PageElement` match the rules set down by this
        `SoupStrainer`?

        The find_* methods rely heavily on this method to find matches.

        :param element: A `PageElement`.
        :param _known_rules: Set to true in the common case where
           we already checked and found at least one rule in this SoupStrainer
           that might exclude a PageElement. Without this, we need
           to check .includes_everything every time, just to be safe.
        :return: `True` if the element matches this `SoupStrainer`'s rules; `False` otherwise.
        """
        # If there are no rules at all, let anything through.
        if not _known_rules and self.includes_everything:
            return True
        if isinstance(element, Tag):
            return self.matches_tag(element)
        assert isinstance(element, NavigableString)
        if not (self.name_rules or self.attribute_rules):
            # A NavigableString can only match a SoupStrainer that
            # does not define any name or attribute rules.
            # Then it comes down to the string rules.
            return self.matches_any_string_rule(element)
        return False

    @_deprecated("allow_tag_creation", "4.13.0")
    def search_tag(self, name: str, attrs: Optional[_RawAttributeValues]) -> bool:
        """A less elegant version of `allow_tag_creation`. Deprecated as of 4.13.0"""
        ":meta private:"
        return self.allow_tag_creation(None, name, attrs)

    @_deprecated("match", "4.13.0")
    def search(self, element: PageElement) -> Optional[PageElement]:
        """A less elegant version of match(). Deprecated as of 4.13.0.

        :meta private:
        """
        return element if self.match(element) else None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\polyutils.py ===
"""
Utility classes and functions for the polynomial modules.

This module provides: error and warning objects; a polynomial base class;
and some routines used in both the `polynomial` and `chebyshev` modules.

Functions
---------

.. autosummary::
   :toctree: generated/

   as_series    convert list of array_likes into 1-D arrays of common type.
   trimseq      remove trailing zeros.
   trimcoef     remove small trailing coefficients.
   getdomain    return the domain appropriate for a given set of abscissae.
   mapdomain    maps points between domains.
   mapparms     parameters of the linear map between domains.

"""
import functools
import operator
import warnings

import numpy as np
from numpy._core.multiarray import dragon4_positional, dragon4_scientific
from numpy.exceptions import RankWarning

__all__ = [
    'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain', 'mapparms',
    'format_float']

#
# Helper functions to convert inputs to 1-D arrays
#
def trimseq(seq):
    """Remove small Poly series coefficients.

    Parameters
    ----------
    seq : sequence
        Sequence of Poly series coefficients.

    Returns
    -------
    series : sequence
        Subsequence with trailing zeros removed. If the resulting sequence
        would be empty, return the first element. The returned sequence may
        or may not be a view.

    Notes
    -----
    Do not lose the type info if the sequence contains unknown objects.

    """
    if len(seq) == 0 or seq[-1] != 0:
        return seq
    else:
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != 0:
                break
        return seq[:i + 1]


def as_series(alist, trim=True):
    """
    Return argument as a list of 1-d arrays.

    The returned list contains array(s) of dtype double, complex double, or
    object.  A 1-d argument of shape ``(N,)`` is parsed into ``N`` arrays of
    size one; a 2-d argument of shape ``(M,N)`` is parsed into ``M`` arrays
    of size ``N`` (i.e., is "parsed by row"); and a higher dimensional array
    raises a Value Error if it is not first reshaped into either a 1-d or 2-d
    array.

    Parameters
    ----------
    alist : array_like
        A 1- or 2-d array_like
    trim : boolean, optional
        When True, trailing zeros are removed from the inputs.
        When False, the inputs are passed through intact.

    Returns
    -------
    [a1, a2,...] : list of 1-D arrays
        A copy of the input data as a list of 1-d arrays.

    Raises
    ------
    ValueError
        Raised when `as_series` cannot convert its input to 1-d arrays, or at
        least one of the resulting arrays is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> a = np.arange(4)
    >>> pu.as_series(a)
    [array([0.]), array([1.]), array([2.]), array([3.])]
    >>> b = np.arange(6).reshape((2,3))
    >>> pu.as_series(b)
    [array([0., 1., 2.]), array([3., 4., 5.])]

    >>> pu.as_series((1, np.arange(3), np.arange(2, dtype=np.float16)))
    [array([1.]), array([0., 1., 2.]), array([0., 1.])]

    >>> pu.as_series([2, [1.1, 0.]])
    [array([2.]), array([1.1])]

    >>> pu.as_series([2, [1.1, 0.]], trim=False)
    [array([2.]), array([1.1, 0. ])]

    """
    arrays = [np.array(a, ndmin=1, copy=None) for a in alist]
    for a in arrays:
        if a.size == 0:
            raise ValueError("Coefficient array is empty")
        if a.ndim != 1:
            raise ValueError("Coefficient array is not 1-d")
    if trim:
        arrays = [trimseq(a) for a in arrays]

    try:
        dtype = np.common_type(*arrays)
    except Exception as e:
        object_dtype = np.dtypes.ObjectDType()
        has_one_object_type = False
        ret = []
        for a in arrays:
            if a.dtype != object_dtype:
                tmp = np.empty(len(a), dtype=object_dtype)
                tmp[:] = a[:]
                ret.append(tmp)
            else:
                has_one_object_type = True
                ret.append(a.copy())
        if not has_one_object_type:
            raise ValueError("Coefficient arrays have no common type") from e
    else:
        ret = [np.array(a, copy=True, dtype=dtype) for a in arrays]
    return ret


def trimcoef(c, tol=0):
    """
    Remove "small" "trailing" coefficients from a polynomial.

    "Small" means "small in absolute value" and is controlled by the
    parameter `tol`; "trailing" means highest order coefficient(s), e.g., in
    ``[0, 1, 1, 0, 0]`` (which represents ``0 + x + x**2 + 0*x**3 + 0*x**4``)
    both the 3-rd and 4-th order coefficients would be "trimmed."

    Parameters
    ----------
    c : array_like
        1-d array of coefficients, ordered from lowest order to highest.
    tol : number, optional
        Trailing (i.e., highest order) elements with absolute value less
        than or equal to `tol` (default value is zero) are removed.

    Returns
    -------
    trimmed : ndarray
        1-d array with trailing zeros removed.  If the resulting series
        would be empty, a series containing a single zero is returned.

    Raises
    ------
    ValueError
        If `tol` < 0

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> pu.trimcoef((0,0,3,0,5,0,0))
    array([0.,  0.,  3.,  0.,  5.])
    >>> pu.trimcoef((0,0,1e-3,0,1e-5,0,0),1e-3) # item == tol is trimmed
    array([0.])
    >>> i = complex(0,1) # works for complex
    >>> pu.trimcoef((3e-4,1e-3*(1-i),5e-4,2e-5*(1+i)), 1e-3)
    array([0.0003+0.j   , 0.001 -0.001j])

    """
    if tol < 0:
        raise ValueError("tol must be non-negative")

    [c] = as_series([c])
    [ind] = np.nonzero(np.abs(c) > tol)
    if len(ind) == 0:
        return c[:1] * 0
    else:
        return c[:ind[-1] + 1].copy()

def getdomain(x):
    """
    Return a domain suitable for given abscissae.

    Find a domain suitable for a polynomial or Chebyshev series
    defined at the values supplied.

    Parameters
    ----------
    x : array_like
        1-d array of abscissae whose domain will be determined.

    Returns
    -------
    domain : ndarray
        1-d array containing two values.  If the inputs are complex, then
        the two returned points are the lower left and upper right corners
        of the smallest rectangle (aligned with the axes) in the complex
        plane containing the points `x`. If the inputs are real, then the
        two points are the ends of the smallest interval containing the
        points `x`.

    See Also
    --------
    mapparms, mapdomain

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> points = np.arange(4)**2 - 5; points
    array([-5, -4, -1,  4])
    >>> pu.getdomain(points)
    array([-5.,  4.])
    >>> c = np.exp(complex(0,1)*np.pi*np.arange(12)/6) # unit circle
    >>> pu.getdomain(c)
    array([-1.-1.j,  1.+1.j])

    """
    [x] = as_series([x], trim=False)
    if x.dtype.char in np.typecodes['Complex']:
        rmin, rmax = x.real.min(), x.real.max()
        imin, imax = x.imag.min(), x.imag.max()
        return np.array((complex(rmin, imin), complex(rmax, imax)))
    else:
        return np.array((x.min(), x.max()))

def mapparms(old, new):
    """
    Linear map parameters between domains.

    Return the parameters of the linear map ``offset + scale*x`` that maps
    `old` to `new` such that ``old[i] -> new[i]``, ``i = 0, 1``.

    Parameters
    ----------
    old, new : array_like
        Domains. Each domain must (successfully) convert to a 1-d array
        containing precisely two values.

    Returns
    -------
    offset, scale : scalars
        The map ``L(x) = offset + scale*x`` maps the first domain to the
        second.

    See Also
    --------
    getdomain, mapdomain

    Notes
    -----
    Also works for complex numbers, and thus can be used to calculate the
    parameters required to map any line in the complex plane to any other
    line therein.

    Examples
    --------
    >>> from numpy.polynomial import polyutils as pu
    >>> pu.mapparms((-1,1),(-1,1))
    (0.0, 1.0)
    >>> pu.mapparms((1,-1),(-1,1))
    (-0.0, -1.0)
    >>> i = complex(0,1)
    >>> pu.mapparms((-i,-1),(1,i))
    ((1+1j), (1-0j))

    """
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl

def mapdomain(x, old, new):
    """
    Apply linear map to input points.

    The linear map ``offset + scale*x`` that maps the domain `old` to
    the domain `new` is applied to the points `x`.

    Parameters
    ----------
    x : array_like
        Points to be mapped. If `x` is a subtype of ndarray the subtype
        will be preserved.
    old, new : array_like
        The two domains that determine the map.  Each must (successfully)
        convert to 1-d arrays containing precisely two values.

    Returns
    -------
    x_out : ndarray
        Array of points of the same shape as `x`, after application of the
        linear map between the two domains.

    See Also
    --------
    getdomain, mapparms

    Notes
    -----
    Effectively, this implements:

    .. math::
        x\\_out = new[0] + m(x - old[0])

    where

    .. math::
        m = \\frac{new[1]-new[0]}{old[1]-old[0]}

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polyutils as pu
    >>> old_domain = (-1,1)
    >>> new_domain = (0,2*np.pi)
    >>> x = np.linspace(-1,1,6); x
    array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ])
    >>> x_out = pu.mapdomain(x, old_domain, new_domain); x_out
    array([ 0.        ,  1.25663706,  2.51327412,  3.76991118,  5.02654825, # may vary
            6.28318531])
    >>> x - pu.mapdomain(x_out, new_domain, old_domain)
    array([0., 0., 0., 0., 0., 0.])

    Also works for complex numbers (and thus can be used to map any line in
    the complex plane to any other line therein).

    >>> i = complex(0,1)
    >>> old = (-1 - i, 1 + i)
    >>> new = (-1 + i, 1 - i)
    >>> z = np.linspace(old[0], old[1], 6); z
    array([-1. -1.j , -0.6-0.6j, -0.2-0.2j,  0.2+0.2j,  0.6+0.6j,  1. +1.j ])
    >>> new_z = pu.mapdomain(z, old, new); new_z
    array([-1.0+1.j , -0.6+0.6j, -0.2+0.2j,  0.2-0.2j,  0.6-0.6j,  1.0-1.j ]) # may vary

    """
    if type(x) not in (int, float, complex) and not isinstance(x, np.generic):
        x = np.asanyarray(x)
    off, scl = mapparms(old, new)
    return off + scl * x


def _nth_slice(i, ndim):
    sl = [np.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)


def _vander_nd(vander_fs, points, degrees):
    r"""
    A generalization of the Vandermonde matrix for N dimensions

    The result is built by combining the results of 1d Vandermonde matrices,

    .. math::
        W[i_0, \ldots, i_M, j_0, \ldots, j_N] = \prod_{k=0}^N{V_k(x_k)[i_0, \ldots, i_M, j_k]}

    where

    .. math::
        N &= \texttt{len(points)} = \texttt{len(degrees)} = \texttt{len(vander\_fs)} \\
        M &= \texttt{points[k].ndim} \\
        V_k &= \texttt{vander\_fs[k]} \\
        x_k &= \texttt{points[k]} \\
        0 \le j_k &\le \texttt{degrees[k]}

    Expanding the one-dimensional :math:`V_k` functions gives:

    .. math::
        W[i_0, \ldots, i_M, j_0, \ldots, j_N] = \prod_{k=0}^N{B_{k, j_k}(x_k[i_0, \ldots, i_M])}

    where :math:`B_{k,m}` is the m'th basis of the polynomial construction used along
    dimension :math:`k`. For a regular polynomial, :math:`B_{k, m}(x) = P_m(x) = x^m`.

    Parameters
    ----------
    vander_fs : Sequence[function(array_like, int) -> ndarray]
        The 1d vander function to use for each axis, such as ``polyvander``
    points : Sequence[array_like]
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
        This must be the same length as `vander_fs`.
    degrees : Sequence[int]
        The maximum degree (inclusive) to use for each axis.
        This must be the same length as `vander_fs`.

    Returns
    -------
    vander_nd : ndarray
        An array of shape ``points[0].shape + tuple(d + 1 for d in degrees)``.
    """  # noqa: E501
    n_dims = len(vander_fs)
    if n_dims != len(points):
        raise ValueError(
            f"Expected {n_dims} dimensions of sample points, got {len(points)}")
    if n_dims != len(degrees):
        raise ValueError(
            f"Expected {n_dims} dimensions of degrees, got {len(degrees)}")
    if n_dims == 0:
        raise ValueError("Unable to guess a dtype or shape when no points are given")

    # convert to the same shape and type
    points = tuple(np.asarray(tuple(points)) + 0.0)

    # produce the vandermonde matrix for each dimension, placing the last
    # axis of each in an independent trailing axis of the output
    vander_arrays = (
        vander_fs[i](points[i], degrees[i])[(...,) + _nth_slice(i, n_dims)]
        for i in range(n_dims)
    )

    # we checked this wasn't empty already, so no `initial` needed
    return functools.reduce(operator.mul, vander_arrays)


def _vander_nd_flat(vander_fs, points, degrees):
    """
    Like `_vander_nd`, but flattens the last ``len(degrees)`` axes into a single axis

    Used to implement the public ``<type>vander<n>d`` functions.
    """
    v = _vander_nd(vander_fs, points, degrees)
    return v.reshape(v.shape[:-len(degrees)] + (-1,))


def _fromroots(line_f, mul_f, roots):
    """
    Helper function used to implement the ``<type>fromroots`` functions.

    Parameters
    ----------
    line_f : function(float, float) -> ndarray
        The ``<type>line`` function, such as ``polyline``
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    roots
        See the ``<type>fromroots`` functions for more detail
    """
    if len(roots) == 0:
        return np.ones(1)
    else:
        [roots] = as_series([roots], trim=False)
        roots.sort()
        p = [line_f(-r, 1) for r in roots]
        n = len(p)
        while n > 1:
            m, r = divmod(n, 2)
            tmp = [mul_f(p[i], p[i + m]) for i in range(m)]
            if r:
                tmp[0] = mul_f(tmp[0], p[-1])
            p = tmp
            n = m
        return p[0]


def _valnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>val<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>val<n>d`` functions for more detail
    """
    args = [np.asanyarray(a) for a in args]
    shape0 = args[0].shape
    if not all(a.shape == shape0 for a in args[1:]):
        if len(args) == 3:
            raise ValueError('x, y, z are incompatible')
        elif len(args) == 2:
            raise ValueError('x, y are incompatible')
        else:
            raise ValueError('ordinates are incompatible')
    it = iter(args)
    x0 = next(it)

    # use tensor on only the first
    c = val_f(x0, c)
    for xi in it:
        c = val_f(xi, c, tensor=False)
    return c


def _gridnd(val_f, c, *args):
    """
    Helper function used to implement the ``<type>grid<n>d`` functions.

    Parameters
    ----------
    val_f : function(array_like, array_like, tensor: bool) -> array_like
        The ``<type>val`` function, such as ``polyval``
    c, args
        See the ``<type>grid<n>d`` functions for more detail
    """
    for xi in args:
        c = val_f(xi, c)
    return c


def _div(mul_f, c1, c2):
    """
    Helper function used to implement the ``<type>div`` functions.

    Implementation uses repeated subtraction of c2 multiplied by the nth basis.
    For some polynomial types, a more efficient approach may be possible.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> array_like
        The ``<type>mul`` function, such as ``polymul``
    c1, c2
        See the ``<type>div`` functions for more detail
    """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError  # FIXME: add message with details to exception

    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
        rem = c1
        for i in range(lc1 - lc2, - 1, -1):
            p = mul_f([0] * i + [1], c2)
            q = rem[-1] / p[-1]
            rem = rem[:-1] - q * p[:-1]
            quo[i] = q
        return quo, trimseq(rem)


def _add(c1, c2):
    """ Helper function used to implement the ``<type>add`` functions. """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[:c2.size] += c2
        ret = c1
    else:
        c2[:c1.size] += c1
        ret = c2
    return trimseq(ret)


def _sub(c1, c2):
    """ Helper function used to implement the ``<type>sub`` functions. """
    # c1, c2 are trimmed copies
    [c1, c2] = as_series([c1, c2])
    if len(c1) > len(c2):
        c1[:c2.size] -= c2
        ret = c1
    else:
        c2 = -c2
        c2[:c1.size] += c1
        ret = c2
    return trimseq(ret)


def _fit(vander_f, x, y, deg, rcond=None, full=False, w=None):
    """
    Helper function used to implement the ``<type>fit`` functions.

    Parameters
    ----------
    vander_f : function(array_like, int) -> ndarray
        The 1d vander function, such as ``polyvander``
    c1, c2
        See the ``<type>fit`` functions for more detail
    """
    x = np.asarray(x) + 0.0
    y = np.asarray(y) + 0.0
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int or non-empty 1-D array of int")
    if deg.min() < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if len(x) != len(y):
        raise TypeError("expected x and y to have same length")

    if deg.ndim == 0:
        lmax = deg
        order = lmax + 1
        van = vander_f(x, lmax)
    else:
        deg = np.sort(deg)
        lmax = deg[-1]
        order = len(deg)
        van = vander_f(x, lmax)[:, deg]

    # set up the least squares matrices in transposed form
    lhs = van.T
    rhs = y.T
    if w is not None:
        w = np.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(x) != len(w):
            raise TypeError("expected x and w to have same length")
        # apply weights. Don't use inplace operations as they
        # can cause problems with NA.
        lhs = lhs * w
        rhs = rhs * w

    # set rcond
    if rcond is None:
        rcond = len(x) * np.finfo(x.dtype).eps

    # Determine the norms of the design matrix columns.
    if issubclass(lhs.dtype.type, np.complexfloating):
        scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
    else:
        scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
    c = (c.T / scl).T

    # Expand c to include non-fitted coefficients which are set to zero
    if deg.ndim > 0:
        if c.ndim == 2:
            cc = np.zeros((lmax + 1, c.shape[1]), dtype=c.dtype)
        else:
            cc = np.zeros(lmax + 1, dtype=c.dtype)
        cc[deg] = c
        c = cc

    # warn on rank reduction
    if rank != order and not full:
        msg = "The fit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, [resids, rank, s, rcond]
    else:
        return c


def _pow(mul_f, c, pow, maxpower):
    """
    Helper function used to implement the ``<type>pow`` functions.

    Parameters
    ----------
    mul_f : function(array_like, array_like) -> ndarray
        The ``<type>mul`` function, such as ``polymul``
    c : array_like
        1-D array of array of series coefficients
    pow, maxpower
        See the ``<type>pow`` functions for more detail
    """
    # c is a trimmed copy
    [c] = as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return np.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        # This can be made more efficient by using powers of two
        # in the usual way.
        prd = c
        for i in range(2, power + 1):
            prd = mul_f(prd, c)
        return prd


def _as_int(x, desc):
    """
    Like `operator.index`, but emits a custom exception when passed an
    incorrect type

    Parameters
    ----------
    x : int-like
        Value to interpret as an integer
    desc : str
        description to include in any error message

    Raises
    ------
    TypeError : if x is a float or non-numeric
    """
    try:
        return operator.index(x)
    except TypeError as e:
        raise TypeError(f"{desc} must be an integer, received {x}") from e


def format_float(x, parens=False):
    if not np.issubdtype(type(x), np.floating):
        return str(x)

    opts = np.get_printoptions()

    if np.isnan(x):
        return opts['nanstr']
    elif np.isinf(x):
        return opts['infstr']

    exp_format = False
    if x != 0:
        a = np.abs(x)
        if a >= 1.e8 or a < 10**min(0, -(opts['precision'] - 1) // 2):
            exp_format = True

    trim, unique = '0', True
    if opts['floatmode'] == 'fixed':
        trim, unique = 'k', False

    if exp_format:
        s = dragon4_scientific(x, precision=opts['precision'],
                               unique=unique, trim=trim,
                               sign=opts['sign'] == '+')
        if parens:
            s = '(' + s + ')'
    else:
        s = dragon4_positional(x, precision=opts['precision'],
                               fractional=True,
                               unique=unique, trim=trim,
                               sign=opts['sign'] == '+')
    return s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\licenses\_spdx.py ===

from __future__ import annotations

from typing import TypedDict

class SPDXLicense(TypedDict):
    id: str
    deprecated: bool

class SPDXException(TypedDict):
    id: str
    deprecated: bool


VERSION = '3.25.0'

LICENSES: dict[str, SPDXLicense] = {
    '0bsd': {'id': '0BSD', 'deprecated': False},
    '3d-slicer-1.0': {'id': '3D-Slicer-1.0', 'deprecated': False},
    'aal': {'id': 'AAL', 'deprecated': False},
    'abstyles': {'id': 'Abstyles', 'deprecated': False},
    'adacore-doc': {'id': 'AdaCore-doc', 'deprecated': False},
    'adobe-2006': {'id': 'Adobe-2006', 'deprecated': False},
    'adobe-display-postscript': {'id': 'Adobe-Display-PostScript', 'deprecated': False},
    'adobe-glyph': {'id': 'Adobe-Glyph', 'deprecated': False},
    'adobe-utopia': {'id': 'Adobe-Utopia', 'deprecated': False},
    'adsl': {'id': 'ADSL', 'deprecated': False},
    'afl-1.1': {'id': 'AFL-1.1', 'deprecated': False},
    'afl-1.2': {'id': 'AFL-1.2', 'deprecated': False},
    'afl-2.0': {'id': 'AFL-2.0', 'deprecated': False},
    'afl-2.1': {'id': 'AFL-2.1', 'deprecated': False},
    'afl-3.0': {'id': 'AFL-3.0', 'deprecated': False},
    'afmparse': {'id': 'Afmparse', 'deprecated': False},
    'agpl-1.0': {'id': 'AGPL-1.0', 'deprecated': True},
    'agpl-1.0-only': {'id': 'AGPL-1.0-only', 'deprecated': False},
    'agpl-1.0-or-later': {'id': 'AGPL-1.0-or-later', 'deprecated': False},
    'agpl-3.0': {'id': 'AGPL-3.0', 'deprecated': True},
    'agpl-3.0-only': {'id': 'AGPL-3.0-only', 'deprecated': False},
    'agpl-3.0-or-later': {'id': 'AGPL-3.0-or-later', 'deprecated': False},
    'aladdin': {'id': 'Aladdin', 'deprecated': False},
    'amd-newlib': {'id': 'AMD-newlib', 'deprecated': False},
    'amdplpa': {'id': 'AMDPLPA', 'deprecated': False},
    'aml': {'id': 'AML', 'deprecated': False},
    'aml-glslang': {'id': 'AML-glslang', 'deprecated': False},
    'ampas': {'id': 'AMPAS', 'deprecated': False},
    'antlr-pd': {'id': 'ANTLR-PD', 'deprecated': False},
    'antlr-pd-fallback': {'id': 'ANTLR-PD-fallback', 'deprecated': False},
    'any-osi': {'id': 'any-OSI', 'deprecated': False},
    'apache-1.0': {'id': 'Apache-1.0', 'deprecated': False},
    'apache-1.1': {'id': 'Apache-1.1', 'deprecated': False},
    'apache-2.0': {'id': 'Apache-2.0', 'deprecated': False},
    'apafml': {'id': 'APAFML', 'deprecated': False},
    'apl-1.0': {'id': 'APL-1.0', 'deprecated': False},
    'app-s2p': {'id': 'App-s2p', 'deprecated': False},
    'apsl-1.0': {'id': 'APSL-1.0', 'deprecated': False},
    'apsl-1.1': {'id': 'APSL-1.1', 'deprecated': False},
    'apsl-1.2': {'id': 'APSL-1.2', 'deprecated': False},
    'apsl-2.0': {'id': 'APSL-2.0', 'deprecated': False},
    'arphic-1999': {'id': 'Arphic-1999', 'deprecated': False},
    'artistic-1.0': {'id': 'Artistic-1.0', 'deprecated': False},
    'artistic-1.0-cl8': {'id': 'Artistic-1.0-cl8', 'deprecated': False},
    'artistic-1.0-perl': {'id': 'Artistic-1.0-Perl', 'deprecated': False},
    'artistic-2.0': {'id': 'Artistic-2.0', 'deprecated': False},
    'aswf-digital-assets-1.0': {'id': 'ASWF-Digital-Assets-1.0', 'deprecated': False},
    'aswf-digital-assets-1.1': {'id': 'ASWF-Digital-Assets-1.1', 'deprecated': False},
    'baekmuk': {'id': 'Baekmuk', 'deprecated': False},
    'bahyph': {'id': 'Bahyph', 'deprecated': False},
    'barr': {'id': 'Barr', 'deprecated': False},
    'bcrypt-solar-designer': {'id': 'bcrypt-Solar-Designer', 'deprecated': False},
    'beerware': {'id': 'Beerware', 'deprecated': False},
    'bitstream-charter': {'id': 'Bitstream-Charter', 'deprecated': False},
    'bitstream-vera': {'id': 'Bitstream-Vera', 'deprecated': False},
    'bittorrent-1.0': {'id': 'BitTorrent-1.0', 'deprecated': False},
    'bittorrent-1.1': {'id': 'BitTorrent-1.1', 'deprecated': False},
    'blessing': {'id': 'blessing', 'deprecated': False},
    'blueoak-1.0.0': {'id': 'BlueOak-1.0.0', 'deprecated': False},
    'boehm-gc': {'id': 'Boehm-GC', 'deprecated': False},
    'borceux': {'id': 'Borceux', 'deprecated': False},
    'brian-gladman-2-clause': {'id': 'Brian-Gladman-2-Clause', 'deprecated': False},
    'brian-gladman-3-clause': {'id': 'Brian-Gladman-3-Clause', 'deprecated': False},
    'bsd-1-clause': {'id': 'BSD-1-Clause', 'deprecated': False},
    'bsd-2-clause': {'id': 'BSD-2-Clause', 'deprecated': False},
    'bsd-2-clause-darwin': {'id': 'BSD-2-Clause-Darwin', 'deprecated': False},
    'bsd-2-clause-first-lines': {'id': 'BSD-2-Clause-first-lines', 'deprecated': False},
    'bsd-2-clause-freebsd': {'id': 'BSD-2-Clause-FreeBSD', 'deprecated': True},
    'bsd-2-clause-netbsd': {'id': 'BSD-2-Clause-NetBSD', 'deprecated': True},
    'bsd-2-clause-patent': {'id': 'BSD-2-Clause-Patent', 'deprecated': False},
    'bsd-2-clause-views': {'id': 'BSD-2-Clause-Views', 'deprecated': False},
    'bsd-3-clause': {'id': 'BSD-3-Clause', 'deprecated': False},
    'bsd-3-clause-acpica': {'id': 'BSD-3-Clause-acpica', 'deprecated': False},
    'bsd-3-clause-attribution': {'id': 'BSD-3-Clause-Attribution', 'deprecated': False},
    'bsd-3-clause-clear': {'id': 'BSD-3-Clause-Clear', 'deprecated': False},
    'bsd-3-clause-flex': {'id': 'BSD-3-Clause-flex', 'deprecated': False},
    'bsd-3-clause-hp': {'id': 'BSD-3-Clause-HP', 'deprecated': False},
    'bsd-3-clause-lbnl': {'id': 'BSD-3-Clause-LBNL', 'deprecated': False},
    'bsd-3-clause-modification': {'id': 'BSD-3-Clause-Modification', 'deprecated': False},
    'bsd-3-clause-no-military-license': {'id': 'BSD-3-Clause-No-Military-License', 'deprecated': False},
    'bsd-3-clause-no-nuclear-license': {'id': 'BSD-3-Clause-No-Nuclear-License', 'deprecated': False},
    'bsd-3-clause-no-nuclear-license-2014': {'id': 'BSD-3-Clause-No-Nuclear-License-2014', 'deprecated': False},
    'bsd-3-clause-no-nuclear-warranty': {'id': 'BSD-3-Clause-No-Nuclear-Warranty', 'deprecated': False},
    'bsd-3-clause-open-mpi': {'id': 'BSD-3-Clause-Open-MPI', 'deprecated': False},
    'bsd-3-clause-sun': {'id': 'BSD-3-Clause-Sun', 'deprecated': False},
    'bsd-4-clause': {'id': 'BSD-4-Clause', 'deprecated': False},
    'bsd-4-clause-shortened': {'id': 'BSD-4-Clause-Shortened', 'deprecated': False},
    'bsd-4-clause-uc': {'id': 'BSD-4-Clause-UC', 'deprecated': False},
    'bsd-4.3reno': {'id': 'BSD-4.3RENO', 'deprecated': False},
    'bsd-4.3tahoe': {'id': 'BSD-4.3TAHOE', 'deprecated': False},
    'bsd-advertising-acknowledgement': {'id': 'BSD-Advertising-Acknowledgement', 'deprecated': False},
    'bsd-attribution-hpnd-disclaimer': {'id': 'BSD-Attribution-HPND-disclaimer', 'deprecated': False},
    'bsd-inferno-nettverk': {'id': 'BSD-Inferno-Nettverk', 'deprecated': False},
    'bsd-protection': {'id': 'BSD-Protection', 'deprecated': False},
    'bsd-source-beginning-file': {'id': 'BSD-Source-beginning-file', 'deprecated': False},
    'bsd-source-code': {'id': 'BSD-Source-Code', 'deprecated': False},
    'bsd-systemics': {'id': 'BSD-Systemics', 'deprecated': False},
    'bsd-systemics-w3works': {'id': 'BSD-Systemics-W3Works', 'deprecated': False},
    'bsl-1.0': {'id': 'BSL-1.0', 'deprecated': False},
    'busl-1.1': {'id': 'BUSL-1.1', 'deprecated': False},
    'bzip2-1.0.5': {'id': 'bzip2-1.0.5', 'deprecated': True},
    'bzip2-1.0.6': {'id': 'bzip2-1.0.6', 'deprecated': False},
    'c-uda-1.0': {'id': 'C-UDA-1.0', 'deprecated': False},
    'cal-1.0': {'id': 'CAL-1.0', 'deprecated': False},
    'cal-1.0-combined-work-exception': {'id': 'CAL-1.0-Combined-Work-Exception', 'deprecated': False},
    'caldera': {'id': 'Caldera', 'deprecated': False},
    'caldera-no-preamble': {'id': 'Caldera-no-preamble', 'deprecated': False},
    'catharon': {'id': 'Catharon', 'deprecated': False},
    'catosl-1.1': {'id': 'CATOSL-1.1', 'deprecated': False},
    'cc-by-1.0': {'id': 'CC-BY-1.0', 'deprecated': False},
    'cc-by-2.0': {'id': 'CC-BY-2.0', 'deprecated': False},
    'cc-by-2.5': {'id': 'CC-BY-2.5', 'deprecated': False},
    'cc-by-2.5-au': {'id': 'CC-BY-2.5-AU', 'deprecated': False},
    'cc-by-3.0': {'id': 'CC-BY-3.0', 'deprecated': False},
    'cc-by-3.0-at': {'id': 'CC-BY-3.0-AT', 'deprecated': False},
    'cc-by-3.0-au': {'id': 'CC-BY-3.0-AU', 'deprecated': False},
    'cc-by-3.0-de': {'id': 'CC-BY-3.0-DE', 'deprecated': False},
    'cc-by-3.0-igo': {'id': 'CC-BY-3.0-IGO', 'deprecated': False},
    'cc-by-3.0-nl': {'id': 'CC-BY-3.0-NL', 'deprecated': False},
    'cc-by-3.0-us': {'id': 'CC-BY-3.0-US', 'deprecated': False},
    'cc-by-4.0': {'id': 'CC-BY-4.0', 'deprecated': False},
    'cc-by-nc-1.0': {'id': 'CC-BY-NC-1.0', 'deprecated': False},
    'cc-by-nc-2.0': {'id': 'CC-BY-NC-2.0', 'deprecated': False},
    'cc-by-nc-2.5': {'id': 'CC-BY-NC-2.5', 'deprecated': False},
    'cc-by-nc-3.0': {'id': 'CC-BY-NC-3.0', 'deprecated': False},
    'cc-by-nc-3.0-de': {'id': 'CC-BY-NC-3.0-DE', 'deprecated': False},
    'cc-by-nc-4.0': {'id': 'CC-BY-NC-4.0', 'deprecated': False},
    'cc-by-nc-nd-1.0': {'id': 'CC-BY-NC-ND-1.0', 'deprecated': False},
    'cc-by-nc-nd-2.0': {'id': 'CC-BY-NC-ND-2.0', 'deprecated': False},
    'cc-by-nc-nd-2.5': {'id': 'CC-BY-NC-ND-2.5', 'deprecated': False},
    'cc-by-nc-nd-3.0': {'id': 'CC-BY-NC-ND-3.0', 'deprecated': False},
    'cc-by-nc-nd-3.0-de': {'id': 'CC-BY-NC-ND-3.0-DE', 'deprecated': False},
    'cc-by-nc-nd-3.0-igo': {'id': 'CC-BY-NC-ND-3.0-IGO', 'deprecated': False},
    'cc-by-nc-nd-4.0': {'id': 'CC-BY-NC-ND-4.0', 'deprecated': False},
    'cc-by-nc-sa-1.0': {'id': 'CC-BY-NC-SA-1.0', 'deprecated': False},
    'cc-by-nc-sa-2.0': {'id': 'CC-BY-NC-SA-2.0', 'deprecated': False},
    'cc-by-nc-sa-2.0-de': {'id': 'CC-BY-NC-SA-2.0-DE', 'deprecated': False},
    'cc-by-nc-sa-2.0-fr': {'id': 'CC-BY-NC-SA-2.0-FR', 'deprecated': False},
    'cc-by-nc-sa-2.0-uk': {'id': 'CC-BY-NC-SA-2.0-UK', 'deprecated': False},
    'cc-by-nc-sa-2.5': {'id': 'CC-BY-NC-SA-2.5', 'deprecated': False},
    'cc-by-nc-sa-3.0': {'id': 'CC-BY-NC-SA-3.0', 'deprecated': False},
    'cc-by-nc-sa-3.0-de': {'id': 'CC-BY-NC-SA-3.0-DE', 'deprecated': False},
    'cc-by-nc-sa-3.0-igo': {'id': 'CC-BY-NC-SA-3.0-IGO', 'deprecated': False},
    'cc-by-nc-sa-4.0': {'id': 'CC-BY-NC-SA-4.0', 'deprecated': False},
    'cc-by-nd-1.0': {'id': 'CC-BY-ND-1.0', 'deprecated': False},
    'cc-by-nd-2.0': {'id': 'CC-BY-ND-2.0', 'deprecated': False},
    'cc-by-nd-2.5': {'id': 'CC-BY-ND-2.5', 'deprecated': False},
    'cc-by-nd-3.0': {'id': 'CC-BY-ND-3.0', 'deprecated': False},
    'cc-by-nd-3.0-de': {'id': 'CC-BY-ND-3.0-DE', 'deprecated': False},
    'cc-by-nd-4.0': {'id': 'CC-BY-ND-4.0', 'deprecated': False},
    'cc-by-sa-1.0': {'id': 'CC-BY-SA-1.0', 'deprecated': False},
    'cc-by-sa-2.0': {'id': 'CC-BY-SA-2.0', 'deprecated': False},
    'cc-by-sa-2.0-uk': {'id': 'CC-BY-SA-2.0-UK', 'deprecated': False},
    'cc-by-sa-2.1-jp': {'id': 'CC-BY-SA-2.1-JP', 'deprecated': False},
    'cc-by-sa-2.5': {'id': 'CC-BY-SA-2.5', 'deprecated': False},
    'cc-by-sa-3.0': {'id': 'CC-BY-SA-3.0', 'deprecated': False},
    'cc-by-sa-3.0-at': {'id': 'CC-BY-SA-3.0-AT', 'deprecated': False},
    'cc-by-sa-3.0-de': {'id': 'CC-BY-SA-3.0-DE', 'deprecated': False},
    'cc-by-sa-3.0-igo': {'id': 'CC-BY-SA-3.0-IGO', 'deprecated': False},
    'cc-by-sa-4.0': {'id': 'CC-BY-SA-4.0', 'deprecated': False},
    'cc-pddc': {'id': 'CC-PDDC', 'deprecated': False},
    'cc0-1.0': {'id': 'CC0-1.0', 'deprecated': False},
    'cddl-1.0': {'id': 'CDDL-1.0', 'deprecated': False},
    'cddl-1.1': {'id': 'CDDL-1.1', 'deprecated': False},
    'cdl-1.0': {'id': 'CDL-1.0', 'deprecated': False},
    'cdla-permissive-1.0': {'id': 'CDLA-Permissive-1.0', 'deprecated': False},
    'cdla-permissive-2.0': {'id': 'CDLA-Permissive-2.0', 'deprecated': False},
    'cdla-sharing-1.0': {'id': 'CDLA-Sharing-1.0', 'deprecated': False},
    'cecill-1.0': {'id': 'CECILL-1.0', 'deprecated': False},
    'cecill-1.1': {'id': 'CECILL-1.1', 'deprecated': False},
    'cecill-2.0': {'id': 'CECILL-2.0', 'deprecated': False},
    'cecill-2.1': {'id': 'CECILL-2.1', 'deprecated': False},
    'cecill-b': {'id': 'CECILL-B', 'deprecated': False},
    'cecill-c': {'id': 'CECILL-C', 'deprecated': False},
    'cern-ohl-1.1': {'id': 'CERN-OHL-1.1', 'deprecated': False},
    'cern-ohl-1.2': {'id': 'CERN-OHL-1.2', 'deprecated': False},
    'cern-ohl-p-2.0': {'id': 'CERN-OHL-P-2.0', 'deprecated': False},
    'cern-ohl-s-2.0': {'id': 'CERN-OHL-S-2.0', 'deprecated': False},
    'cern-ohl-w-2.0': {'id': 'CERN-OHL-W-2.0', 'deprecated': False},
    'cfitsio': {'id': 'CFITSIO', 'deprecated': False},
    'check-cvs': {'id': 'check-cvs', 'deprecated': False},
    'checkmk': {'id': 'checkmk', 'deprecated': False},
    'clartistic': {'id': 'ClArtistic', 'deprecated': False},
    'clips': {'id': 'Clips', 'deprecated': False},
    'cmu-mach': {'id': 'CMU-Mach', 'deprecated': False},
    'cmu-mach-nodoc': {'id': 'CMU-Mach-nodoc', 'deprecated': False},
    'cnri-jython': {'id': 'CNRI-Jython', 'deprecated': False},
    'cnri-python': {'id': 'CNRI-Python', 'deprecated': False},
    'cnri-python-gpl-compatible': {'id': 'CNRI-Python-GPL-Compatible', 'deprecated': False},
    'coil-1.0': {'id': 'COIL-1.0', 'deprecated': False},
    'community-spec-1.0': {'id': 'Community-Spec-1.0', 'deprecated': False},
    'condor-1.1': {'id': 'Condor-1.1', 'deprecated': False},
    'copyleft-next-0.3.0': {'id': 'copyleft-next-0.3.0', 'deprecated': False},
    'copyleft-next-0.3.1': {'id': 'copyleft-next-0.3.1', 'deprecated': False},
    'cornell-lossless-jpeg': {'id': 'Cornell-Lossless-JPEG', 'deprecated': False},
    'cpal-1.0': {'id': 'CPAL-1.0', 'deprecated': False},
    'cpl-1.0': {'id': 'CPL-1.0', 'deprecated': False},
    'cpol-1.02': {'id': 'CPOL-1.02', 'deprecated': False},
    'cronyx': {'id': 'Cronyx', 'deprecated': False},
    'crossword': {'id': 'Crossword', 'deprecated': False},
    'crystalstacker': {'id': 'CrystalStacker', 'deprecated': False},
    'cua-opl-1.0': {'id': 'CUA-OPL-1.0', 'deprecated': False},
    'cube': {'id': 'Cube', 'deprecated': False},
    'curl': {'id': 'curl', 'deprecated': False},
    'cve-tou': {'id': 'cve-tou', 'deprecated': False},
    'd-fsl-1.0': {'id': 'D-FSL-1.0', 'deprecated': False},
    'dec-3-clause': {'id': 'DEC-3-Clause', 'deprecated': False},
    'diffmark': {'id': 'diffmark', 'deprecated': False},
    'dl-de-by-2.0': {'id': 'DL-DE-BY-2.0', 'deprecated': False},
    'dl-de-zero-2.0': {'id': 'DL-DE-ZERO-2.0', 'deprecated': False},
    'doc': {'id': 'DOC', 'deprecated': False},
    'docbook-schema': {'id': 'DocBook-Schema', 'deprecated': False},
    'docbook-xml': {'id': 'DocBook-XML', 'deprecated': False},
    'dotseqn': {'id': 'Dotseqn', 'deprecated': False},
    'drl-1.0': {'id': 'DRL-1.0', 'deprecated': False},
    'drl-1.1': {'id': 'DRL-1.1', 'deprecated': False},
    'dsdp': {'id': 'DSDP', 'deprecated': False},
    'dtoa': {'id': 'dtoa', 'deprecated': False},
    'dvipdfm': {'id': 'dvipdfm', 'deprecated': False},
    'ecl-1.0': {'id': 'ECL-1.0', 'deprecated': False},
    'ecl-2.0': {'id': 'ECL-2.0', 'deprecated': False},
    'ecos-2.0': {'id': 'eCos-2.0', 'deprecated': True},
    'efl-1.0': {'id': 'EFL-1.0', 'deprecated': False},
    'efl-2.0': {'id': 'EFL-2.0', 'deprecated': False},
    'egenix': {'id': 'eGenix', 'deprecated': False},
    'elastic-2.0': {'id': 'Elastic-2.0', 'deprecated': False},
    'entessa': {'id': 'Entessa', 'deprecated': False},
    'epics': {'id': 'EPICS', 'deprecated': False},
    'epl-1.0': {'id': 'EPL-1.0', 'deprecated': False},
    'epl-2.0': {'id': 'EPL-2.0', 'deprecated': False},
    'erlpl-1.1': {'id': 'ErlPL-1.1', 'deprecated': False},
    'etalab-2.0': {'id': 'etalab-2.0', 'deprecated': False},
    'eudatagrid': {'id': 'EUDatagrid', 'deprecated': False},
    'eupl-1.0': {'id': 'EUPL-1.0', 'deprecated': False},
    'eupl-1.1': {'id': 'EUPL-1.1', 'deprecated': False},
    'eupl-1.2': {'id': 'EUPL-1.2', 'deprecated': False},
    'eurosym': {'id': 'Eurosym', 'deprecated': False},
    'fair': {'id': 'Fair', 'deprecated': False},
    'fbm': {'id': 'FBM', 'deprecated': False},
    'fdk-aac': {'id': 'FDK-AAC', 'deprecated': False},
    'ferguson-twofish': {'id': 'Ferguson-Twofish', 'deprecated': False},
    'frameworx-1.0': {'id': 'Frameworx-1.0', 'deprecated': False},
    'freebsd-doc': {'id': 'FreeBSD-DOC', 'deprecated': False},
    'freeimage': {'id': 'FreeImage', 'deprecated': False},
    'fsfap': {'id': 'FSFAP', 'deprecated': False},
    'fsfap-no-warranty-disclaimer': {'id': 'FSFAP-no-warranty-disclaimer', 'deprecated': False},
    'fsful': {'id': 'FSFUL', 'deprecated': False},
    'fsfullr': {'id': 'FSFULLR', 'deprecated': False},
    'fsfullrwd': {'id': 'FSFULLRWD', 'deprecated': False},
    'ftl': {'id': 'FTL', 'deprecated': False},
    'furuseth': {'id': 'Furuseth', 'deprecated': False},
    'fwlw': {'id': 'fwlw', 'deprecated': False},
    'gcr-docs': {'id': 'GCR-docs', 'deprecated': False},
    'gd': {'id': 'GD', 'deprecated': False},
    'gfdl-1.1': {'id': 'GFDL-1.1', 'deprecated': True},
    'gfdl-1.1-invariants-only': {'id': 'GFDL-1.1-invariants-only', 'deprecated': False},
    'gfdl-1.1-invariants-or-later': {'id': 'GFDL-1.1-invariants-or-later', 'deprecated': False},
    'gfdl-1.1-no-invariants-only': {'id': 'GFDL-1.1-no-invariants-only', 'deprecated': False},
    'gfdl-1.1-no-invariants-or-later': {'id': 'GFDL-1.1-no-invariants-or-later', 'deprecated': False},
    'gfdl-1.1-only': {'id': 'GFDL-1.1-only', 'deprecated': False},
    'gfdl-1.1-or-later': {'id': 'GFDL-1.1-or-later', 'deprecated': False},
    'gfdl-1.2': {'id': 'GFDL-1.2', 'deprecated': True},
    'gfdl-1.2-invariants-only': {'id': 'GFDL-1.2-invariants-only', 'deprecated': False},
    'gfdl-1.2-invariants-or-later': {'id': 'GFDL-1.2-invariants-or-later', 'deprecated': False},
    'gfdl-1.2-no-invariants-only': {'id': 'GFDL-1.2-no-invariants-only', 'deprecated': False},
    'gfdl-1.2-no-invariants-or-later': {'id': 'GFDL-1.2-no-invariants-or-later', 'deprecated': False},
    'gfdl-1.2-only': {'id': 'GFDL-1.2-only', 'deprecated': False},
    'gfdl-1.2-or-later': {'id': 'GFDL-1.2-or-later', 'deprecated': False},
    'gfdl-1.3': {'id': 'GFDL-1.3', 'deprecated': True},
    'gfdl-1.3-invariants-only': {'id': 'GFDL-1.3-invariants-only', 'deprecated': False},
    'gfdl-1.3-invariants-or-later': {'id': 'GFDL-1.3-invariants-or-later', 'deprecated': False},
    'gfdl-1.3-no-invariants-only': {'id': 'GFDL-1.3-no-invariants-only', 'deprecated': False},
    'gfdl-1.3-no-invariants-or-later': {'id': 'GFDL-1.3-no-invariants-or-later', 'deprecated': False},
    'gfdl-1.3-only': {'id': 'GFDL-1.3-only', 'deprecated': False},
    'gfdl-1.3-or-later': {'id': 'GFDL-1.3-or-later', 'deprecated': False},
    'giftware': {'id': 'Giftware', 'deprecated': False},
    'gl2ps': {'id': 'GL2PS', 'deprecated': False},
    'glide': {'id': 'Glide', 'deprecated': False},
    'glulxe': {'id': 'Glulxe', 'deprecated': False},
    'glwtpl': {'id': 'GLWTPL', 'deprecated': False},
    'gnuplot': {'id': 'gnuplot', 'deprecated': False},
    'gpl-1.0': {'id': 'GPL-1.0', 'deprecated': True},
    'gpl-1.0+': {'id': 'GPL-1.0+', 'deprecated': True},
    'gpl-1.0-only': {'id': 'GPL-1.0-only', 'deprecated': False},
    'gpl-1.0-or-later': {'id': 'GPL-1.0-or-later', 'deprecated': False},
    'gpl-2.0': {'id': 'GPL-2.0', 'deprecated': True},
    'gpl-2.0+': {'id': 'GPL-2.0+', 'deprecated': True},
    'gpl-2.0-only': {'id': 'GPL-2.0-only', 'deprecated': False},
    'gpl-2.0-or-later': {'id': 'GPL-2.0-or-later', 'deprecated': False},
    'gpl-2.0-with-autoconf-exception': {'id': 'GPL-2.0-with-autoconf-exception', 'deprecated': True},
    'gpl-2.0-with-bison-exception': {'id': 'GPL-2.0-with-bison-exception', 'deprecated': True},
    'gpl-2.0-with-classpath-exception': {'id': 'GPL-2.0-with-classpath-exception', 'deprecated': True},
    'gpl-2.0-with-font-exception': {'id': 'GPL-2.0-with-font-exception', 'deprecated': True},
    'gpl-2.0-with-gcc-exception': {'id': 'GPL-2.0-with-GCC-exception', 'deprecated': True},
    'gpl-3.0': {'id': 'GPL-3.0', 'deprecated': True},
    'gpl-3.0+': {'id': 'GPL-3.0+', 'deprecated': True},
    'gpl-3.0-only': {'id': 'GPL-3.0-only', 'deprecated': False},
    'gpl-3.0-or-later': {'id': 'GPL-3.0-or-later', 'deprecated': False},
    'gpl-3.0-with-autoconf-exception': {'id': 'GPL-3.0-with-autoconf-exception', 'deprecated': True},
    'gpl-3.0-with-gcc-exception': {'id': 'GPL-3.0-with-GCC-exception', 'deprecated': True},
    'graphics-gems': {'id': 'Graphics-Gems', 'deprecated': False},
    'gsoap-1.3b': {'id': 'gSOAP-1.3b', 'deprecated': False},
    'gtkbook': {'id': 'gtkbook', 'deprecated': False},
    'gutmann': {'id': 'Gutmann', 'deprecated': False},
    'haskellreport': {'id': 'HaskellReport', 'deprecated': False},
    'hdparm': {'id': 'hdparm', 'deprecated': False},
    'hidapi': {'id': 'HIDAPI', 'deprecated': False},
    'hippocratic-2.1': {'id': 'Hippocratic-2.1', 'deprecated': False},
    'hp-1986': {'id': 'HP-1986', 'deprecated': False},
    'hp-1989': {'id': 'HP-1989', 'deprecated': False},
    'hpnd': {'id': 'HPND', 'deprecated': False},
    'hpnd-dec': {'id': 'HPND-DEC', 'deprecated': False},
    'hpnd-doc': {'id': 'HPND-doc', 'deprecated': False},
    'hpnd-doc-sell': {'id': 'HPND-doc-sell', 'deprecated': False},
    'hpnd-export-us': {'id': 'HPND-export-US', 'deprecated': False},
    'hpnd-export-us-acknowledgement': {'id': 'HPND-export-US-acknowledgement', 'deprecated': False},
    'hpnd-export-us-modify': {'id': 'HPND-export-US-modify', 'deprecated': False},
    'hpnd-export2-us': {'id': 'HPND-export2-US', 'deprecated': False},
    'hpnd-fenneberg-livingston': {'id': 'HPND-Fenneberg-Livingston', 'deprecated': False},
    'hpnd-inria-imag': {'id': 'HPND-INRIA-IMAG', 'deprecated': False},
    'hpnd-intel': {'id': 'HPND-Intel', 'deprecated': False},
    'hpnd-kevlin-henney': {'id': 'HPND-Kevlin-Henney', 'deprecated': False},
    'hpnd-markus-kuhn': {'id': 'HPND-Markus-Kuhn', 'deprecated': False},
    'hpnd-merchantability-variant': {'id': 'HPND-merchantability-variant', 'deprecated': False},
    'hpnd-mit-disclaimer': {'id': 'HPND-MIT-disclaimer', 'deprecated': False},
    'hpnd-netrek': {'id': 'HPND-Netrek', 'deprecated': False},
    'hpnd-pbmplus': {'id': 'HPND-Pbmplus', 'deprecated': False},
    'hpnd-sell-mit-disclaimer-xserver': {'id': 'HPND-sell-MIT-disclaimer-xserver', 'deprecated': False},
    'hpnd-sell-regexpr': {'id': 'HPND-sell-regexpr', 'deprecated': False},
    'hpnd-sell-variant': {'id': 'HPND-sell-variant', 'deprecated': False},
    'hpnd-sell-variant-mit-disclaimer': {'id': 'HPND-sell-variant-MIT-disclaimer', 'deprecated': False},
    'hpnd-sell-variant-mit-disclaimer-rev': {'id': 'HPND-sell-variant-MIT-disclaimer-rev', 'deprecated': False},
    'hpnd-uc': {'id': 'HPND-UC', 'deprecated': False},
    'hpnd-uc-export-us': {'id': 'HPND-UC-export-US', 'deprecated': False},
    'htmltidy': {'id': 'HTMLTIDY', 'deprecated': False},
    'ibm-pibs': {'id': 'IBM-pibs', 'deprecated': False},
    'icu': {'id': 'ICU', 'deprecated': False},
    'iec-code-components-eula': {'id': 'IEC-Code-Components-EULA', 'deprecated': False},
    'ijg': {'id': 'IJG', 'deprecated': False},
    'ijg-short': {'id': 'IJG-short', 'deprecated': False},
    'imagemagick': {'id': 'ImageMagick', 'deprecated': False},
    'imatix': {'id': 'iMatix', 'deprecated': False},
    'imlib2': {'id': 'Imlib2', 'deprecated': False},
    'info-zip': {'id': 'Info-ZIP', 'deprecated': False},
    'inner-net-2.0': {'id': 'Inner-Net-2.0', 'deprecated': False},
    'intel': {'id': 'Intel', 'deprecated': False},
    'intel-acpi': {'id': 'Intel-ACPI', 'deprecated': False},
    'interbase-1.0': {'id': 'Interbase-1.0', 'deprecated': False},
    'ipa': {'id': 'IPA', 'deprecated': False},
    'ipl-1.0': {'id': 'IPL-1.0', 'deprecated': False},
    'isc': {'id': 'ISC', 'deprecated': False},
    'isc-veillard': {'id': 'ISC-Veillard', 'deprecated': False},
    'jam': {'id': 'Jam', 'deprecated': False},
    'jasper-2.0': {'id': 'JasPer-2.0', 'deprecated': False},
    'jpl-image': {'id': 'JPL-image', 'deprecated': False},
    'jpnic': {'id': 'JPNIC', 'deprecated': False},
    'json': {'id': 'JSON', 'deprecated': False},
    'kastrup': {'id': 'Kastrup', 'deprecated': False},
    'kazlib': {'id': 'Kazlib', 'deprecated': False},
    'knuth-ctan': {'id': 'Knuth-CTAN', 'deprecated': False},
    'lal-1.2': {'id': 'LAL-1.2', 'deprecated': False},
    'lal-1.3': {'id': 'LAL-1.3', 'deprecated': False},
    'latex2e': {'id': 'Latex2e', 'deprecated': False},
    'latex2e-translated-notice': {'id': 'Latex2e-translated-notice', 'deprecated': False},
    'leptonica': {'id': 'Leptonica', 'deprecated': False},
    'lgpl-2.0': {'id': 'LGPL-2.0', 'deprecated': True},
    'lgpl-2.0+': {'id': 'LGPL-2.0+', 'deprecated': True},
    'lgpl-2.0-only': {'id': 'LGPL-2.0-only', 'deprecated': False},
    'lgpl-2.0-or-later': {'id': 'LGPL-2.0-or-later', 'deprecated': False},
    'lgpl-2.1': {'id': 'LGPL-2.1', 'deprecated': True},
    'lgpl-2.1+': {'id': 'LGPL-2.1+', 'deprecated': True},
    'lgpl-2.1-only': {'id': 'LGPL-2.1-only', 'deprecated': False},
    'lgpl-2.1-or-later': {'id': 'LGPL-2.1-or-later', 'deprecated': False},
    'lgpl-3.0': {'id': 'LGPL-3.0', 'deprecated': True},
    'lgpl-3.0+': {'id': 'LGPL-3.0+', 'deprecated': True},
    'lgpl-3.0-only': {'id': 'LGPL-3.0-only', 'deprecated': False},
    'lgpl-3.0-or-later': {'id': 'LGPL-3.0-or-later', 'deprecated': False},
    'lgpllr': {'id': 'LGPLLR', 'deprecated': False},
    'libpng': {'id': 'Libpng', 'deprecated': False},
    'libpng-2.0': {'id': 'libpng-2.0', 'deprecated': False},
    'libselinux-1.0': {'id': 'libselinux-1.0', 'deprecated': False},
    'libtiff': {'id': 'libtiff', 'deprecated': False},
    'libutil-david-nugent': {'id': 'libutil-David-Nugent', 'deprecated': False},
    'liliq-p-1.1': {'id': 'LiLiQ-P-1.1', 'deprecated': False},
    'liliq-r-1.1': {'id': 'LiLiQ-R-1.1', 'deprecated': False},
    'liliq-rplus-1.1': {'id': 'LiLiQ-Rplus-1.1', 'deprecated': False},
    'linux-man-pages-1-para': {'id': 'Linux-man-pages-1-para', 'deprecated': False},
    'linux-man-pages-copyleft': {'id': 'Linux-man-pages-copyleft', 'deprecated': False},
    'linux-man-pages-copyleft-2-para': {'id': 'Linux-man-pages-copyleft-2-para', 'deprecated': False},
    'linux-man-pages-copyleft-var': {'id': 'Linux-man-pages-copyleft-var', 'deprecated': False},
    'linux-openib': {'id': 'Linux-OpenIB', 'deprecated': False},
    'loop': {'id': 'LOOP', 'deprecated': False},
    'lpd-document': {'id': 'LPD-document', 'deprecated': False},
    'lpl-1.0': {'id': 'LPL-1.0', 'deprecated': False},
    'lpl-1.02': {'id': 'LPL-1.02', 'deprecated': False},
    'lppl-1.0': {'id': 'LPPL-1.0', 'deprecated': False},
    'lppl-1.1': {'id': 'LPPL-1.1', 'deprecated': False},
    'lppl-1.2': {'id': 'LPPL-1.2', 'deprecated': False},
    'lppl-1.3a': {'id': 'LPPL-1.3a', 'deprecated': False},
    'lppl-1.3c': {'id': 'LPPL-1.3c', 'deprecated': False},
    'lsof': {'id': 'lsof', 'deprecated': False},
    'lucida-bitmap-fonts': {'id': 'Lucida-Bitmap-Fonts', 'deprecated': False},
    'lzma-sdk-9.11-to-9.20': {'id': 'LZMA-SDK-9.11-to-9.20', 'deprecated': False},
    'lzma-sdk-9.22': {'id': 'LZMA-SDK-9.22', 'deprecated': False},
    'mackerras-3-clause': {'id': 'Mackerras-3-Clause', 'deprecated': False},
    'mackerras-3-clause-acknowledgment': {'id': 'Mackerras-3-Clause-acknowledgment', 'deprecated': False},
    'magaz': {'id': 'magaz', 'deprecated': False},
    'mailprio': {'id': 'mailprio', 'deprecated': False},
    'makeindex': {'id': 'MakeIndex', 'deprecated': False},
    'martin-birgmeier': {'id': 'Martin-Birgmeier', 'deprecated': False},
    'mcphee-slideshow': {'id': 'McPhee-slideshow', 'deprecated': False},
    'metamail': {'id': 'metamail', 'deprecated': False},
    'minpack': {'id': 'Minpack', 'deprecated': False},
    'miros': {'id': 'MirOS', 'deprecated': False},
    'mit': {'id': 'MIT', 'deprecated': False},
    'mit-0': {'id': 'MIT-0', 'deprecated': False},
    'mit-advertising': {'id': 'MIT-advertising', 'deprecated': False},
    'mit-cmu': {'id': 'MIT-CMU', 'deprecated': False},
    'mit-enna': {'id': 'MIT-enna', 'deprecated': False},
    'mit-feh': {'id': 'MIT-feh', 'deprecated': False},
    'mit-festival': {'id': 'MIT-Festival', 'deprecated': False},
    'mit-khronos-old': {'id': 'MIT-Khronos-old', 'deprecated': False},
    'mit-modern-variant': {'id': 'MIT-Modern-Variant', 'deprecated': False},
    'mit-open-group': {'id': 'MIT-open-group', 'deprecated': False},
    'mit-testregex': {'id': 'MIT-testregex', 'deprecated': False},
    'mit-wu': {'id': 'MIT-Wu', 'deprecated': False},
    'mitnfa': {'id': 'MITNFA', 'deprecated': False},
    'mmixware': {'id': 'MMIXware', 'deprecated': False},
    'motosoto': {'id': 'Motosoto', 'deprecated': False},
    'mpeg-ssg': {'id': 'MPEG-SSG', 'deprecated': False},
    'mpi-permissive': {'id': 'mpi-permissive', 'deprecated': False},
    'mpich2': {'id': 'mpich2', 'deprecated': False},
    'mpl-1.0': {'id': 'MPL-1.0', 'deprecated': False},
    'mpl-1.1': {'id': 'MPL-1.1', 'deprecated': False},
    'mpl-2.0': {'id': 'MPL-2.0', 'deprecated': False},
    'mpl-2.0-no-copyleft-exception': {'id': 'MPL-2.0-no-copyleft-exception', 'deprecated': False},
    'mplus': {'id': 'mplus', 'deprecated': False},
    'ms-lpl': {'id': 'MS-LPL', 'deprecated': False},
    'ms-pl': {'id': 'MS-PL', 'deprecated': False},
    'ms-rl': {'id': 'MS-RL', 'deprecated': False},
    'mtll': {'id': 'MTLL', 'deprecated': False},
    'mulanpsl-1.0': {'id': 'MulanPSL-1.0', 'deprecated': False},
    'mulanpsl-2.0': {'id': 'MulanPSL-2.0', 'deprecated': False},
    'multics': {'id': 'Multics', 'deprecated': False},
    'mup': {'id': 'Mup', 'deprecated': False},
    'naist-2003': {'id': 'NAIST-2003', 'deprecated': False},
    'nasa-1.3': {'id': 'NASA-1.3', 'deprecated': False},
    'naumen': {'id': 'Naumen', 'deprecated': False},
    'nbpl-1.0': {'id': 'NBPL-1.0', 'deprecated': False},
    'ncbi-pd': {'id': 'NCBI-PD', 'deprecated': False},
    'ncgl-uk-2.0': {'id': 'NCGL-UK-2.0', 'deprecated': False},
    'ncl': {'id': 'NCL', 'deprecated': False},
    'ncsa': {'id': 'NCSA', 'deprecated': False},
    'net-snmp': {'id': 'Net-SNMP', 'deprecated': True},
    'netcdf': {'id': 'NetCDF', 'deprecated': False},
    'newsletr': {'id': 'Newsletr', 'deprecated': False},
    'ngpl': {'id': 'NGPL', 'deprecated': False},
    'nicta-1.0': {'id': 'NICTA-1.0', 'deprecated': False},
    'nist-pd': {'id': 'NIST-PD', 'deprecated': False},
    'nist-pd-fallback': {'id': 'NIST-PD-fallback', 'deprecated': False},
    'nist-software': {'id': 'NIST-Software', 'deprecated': False},
    'nlod-1.0': {'id': 'NLOD-1.0', 'deprecated': False},
    'nlod-2.0': {'id': 'NLOD-2.0', 'deprecated': False},
    'nlpl': {'id': 'NLPL', 'deprecated': False},
    'nokia': {'id': 'Nokia', 'deprecated': False},
    'nosl': {'id': 'NOSL', 'deprecated': False},
    'noweb': {'id': 'Noweb', 'deprecated': False},
    'npl-1.0': {'id': 'NPL-1.0', 'deprecated': False},
    'npl-1.1': {'id': 'NPL-1.1', 'deprecated': False},
    'nposl-3.0': {'id': 'NPOSL-3.0', 'deprecated': False},
    'nrl': {'id': 'NRL', 'deprecated': False},
    'ntp': {'id': 'NTP', 'deprecated': False},
    'ntp-0': {'id': 'NTP-0', 'deprecated': False},
    'nunit': {'id': 'Nunit', 'deprecated': True},
    'o-uda-1.0': {'id': 'O-UDA-1.0', 'deprecated': False},
    'oar': {'id': 'OAR', 'deprecated': False},
    'occt-pl': {'id': 'OCCT-PL', 'deprecated': False},
    'oclc-2.0': {'id': 'OCLC-2.0', 'deprecated': False},
    'odbl-1.0': {'id': 'ODbL-1.0', 'deprecated': False},
    'odc-by-1.0': {'id': 'ODC-By-1.0', 'deprecated': False},
    'offis': {'id': 'OFFIS', 'deprecated': False},
    'ofl-1.0': {'id': 'OFL-1.0', 'deprecated': False},
    'ofl-1.0-no-rfn': {'id': 'OFL-1.0-no-RFN', 'deprecated': False},
    'ofl-1.0-rfn': {'id': 'OFL-1.0-RFN', 'deprecated': False},
    'ofl-1.1': {'id': 'OFL-1.1', 'deprecated': False},
    'ofl-1.1-no-rfn': {'id': 'OFL-1.1-no-RFN', 'deprecated': False},
    'ofl-1.1-rfn': {'id': 'OFL-1.1-RFN', 'deprecated': False},
    'ogc-1.0': {'id': 'OGC-1.0', 'deprecated': False},
    'ogdl-taiwan-1.0': {'id': 'OGDL-Taiwan-1.0', 'deprecated': False},
    'ogl-canada-2.0': {'id': 'OGL-Canada-2.0', 'deprecated': False},
    'ogl-uk-1.0': {'id': 'OGL-UK-1.0', 'deprecated': False},
    'ogl-uk-2.0': {'id': 'OGL-UK-2.0', 'deprecated': False},
    'ogl-uk-3.0': {'id': 'OGL-UK-3.0', 'deprecated': False},
    'ogtsl': {'id': 'OGTSL', 'deprecated': False},
    'oldap-1.1': {'id': 'OLDAP-1.1', 'deprecated': False},
    'oldap-1.2': {'id': 'OLDAP-1.2', 'deprecated': False},
    'oldap-1.3': {'id': 'OLDAP-1.3', 'deprecated': False},
    'oldap-1.4': {'id': 'OLDAP-1.4', 'deprecated': False},
    'oldap-2.0': {'id': 'OLDAP-2.0', 'deprecated': False},
    'oldap-2.0.1': {'id': 'OLDAP-2.0.1', 'deprecated': False},
    'oldap-2.1': {'id': 'OLDAP-2.1', 'deprecated': False},
    'oldap-2.2': {'id': 'OLDAP-2.2', 'deprecated': False},
    'oldap-2.2.1': {'id': 'OLDAP-2.2.1', 'deprecated': False},
    'oldap-2.2.2': {'id': 'OLDAP-2.2.2', 'deprecated': False},
    'oldap-2.3': {'id': 'OLDAP-2.3', 'deprecated': False},
    'oldap-2.4': {'id': 'OLDAP-2.4', 'deprecated': False},
    'oldap-2.5': {'id': 'OLDAP-2.5', 'deprecated': False},
    'oldap-2.6': {'id': 'OLDAP-2.6', 'deprecated': False},
    'oldap-2.7': {'id': 'OLDAP-2.7', 'deprecated': False},
    'oldap-2.8': {'id': 'OLDAP-2.8', 'deprecated': False},
    'olfl-1.3': {'id': 'OLFL-1.3', 'deprecated': False},
    'oml': {'id': 'OML', 'deprecated': False},
    'openpbs-2.3': {'id': 'OpenPBS-2.3', 'deprecated': False},
    'openssl': {'id': 'OpenSSL', 'deprecated': False},
    'openssl-standalone': {'id': 'OpenSSL-standalone', 'deprecated': False},
    'openvision': {'id': 'OpenVision', 'deprecated': False},
    'opl-1.0': {'id': 'OPL-1.0', 'deprecated': False},
    'opl-uk-3.0': {'id': 'OPL-UK-3.0', 'deprecated': False},
    'opubl-1.0': {'id': 'OPUBL-1.0', 'deprecated': False},
    'oset-pl-2.1': {'id': 'OSET-PL-2.1', 'deprecated': False},
    'osl-1.0': {'id': 'OSL-1.0', 'deprecated': False},
    'osl-1.1': {'id': 'OSL-1.1', 'deprecated': False},
    'osl-2.0': {'id': 'OSL-2.0', 'deprecated': False},
    'osl-2.1': {'id': 'OSL-2.1', 'deprecated': False},
    'osl-3.0': {'id': 'OSL-3.0', 'deprecated': False},
    'padl': {'id': 'PADL', 'deprecated': False},
    'parity-6.0.0': {'id': 'Parity-6.0.0', 'deprecated': False},
    'parity-7.0.0': {'id': 'Parity-7.0.0', 'deprecated': False},
    'pddl-1.0': {'id': 'PDDL-1.0', 'deprecated': False},
    'php-3.0': {'id': 'PHP-3.0', 'deprecated': False},
    'php-3.01': {'id': 'PHP-3.01', 'deprecated': False},
    'pixar': {'id': 'Pixar', 'deprecated': False},
    'pkgconf': {'id': 'pkgconf', 'deprecated': False},
    'plexus': {'id': 'Plexus', 'deprecated': False},
    'pnmstitch': {'id': 'pnmstitch', 'deprecated': False},
    'polyform-noncommercial-1.0.0': {'id': 'PolyForm-Noncommercial-1.0.0', 'deprecated': False},
    'polyform-small-business-1.0.0': {'id': 'PolyForm-Small-Business-1.0.0', 'deprecated': False},
    'postgresql': {'id': 'PostgreSQL', 'deprecated': False},
    'ppl': {'id': 'PPL', 'deprecated': False},
    'psf-2.0': {'id': 'PSF-2.0', 'deprecated': False},
    'psfrag': {'id': 'psfrag', 'deprecated': False},
    'psutils': {'id': 'psutils', 'deprecated': False},
    'python-2.0': {'id': 'Python-2.0', 'deprecated': False},
    'python-2.0.1': {'id': 'Python-2.0.1', 'deprecated': False},
    'python-ldap': {'id': 'python-ldap', 'deprecated': False},
    'qhull': {'id': 'Qhull', 'deprecated': False},
    'qpl-1.0': {'id': 'QPL-1.0', 'deprecated': False},
    'qpl-1.0-inria-2004': {'id': 'QPL-1.0-INRIA-2004', 'deprecated': False},
    'radvd': {'id': 'radvd', 'deprecated': False},
    'rdisc': {'id': 'Rdisc', 'deprecated': False},
    'rhecos-1.1': {'id': 'RHeCos-1.1', 'deprecated': False},
    'rpl-1.1': {'id': 'RPL-1.1', 'deprecated': False},
    'rpl-1.5': {'id': 'RPL-1.5', 'deprecated': False},
    'rpsl-1.0': {'id': 'RPSL-1.0', 'deprecated': False},
    'rsa-md': {'id': 'RSA-MD', 'deprecated': False},
    'rscpl': {'id': 'RSCPL', 'deprecated': False},
    'ruby': {'id': 'Ruby', 'deprecated': False},
    'ruby-pty': {'id': 'Ruby-pty', 'deprecated': False},
    'sax-pd': {'id': 'SAX-PD', 'deprecated': False},
    'sax-pd-2.0': {'id': 'SAX-PD-2.0', 'deprecated': False},
    'saxpath': {'id': 'Saxpath', 'deprecated': False},
    'scea': {'id': 'SCEA', 'deprecated': False},
    'schemereport': {'id': 'SchemeReport', 'deprecated': False},
    'sendmail': {'id': 'Sendmail', 'deprecated': False},
    'sendmail-8.23': {'id': 'Sendmail-8.23', 'deprecated': False},
    'sgi-b-1.0': {'id': 'SGI-B-1.0', 'deprecated': False},
    'sgi-b-1.1': {'id': 'SGI-B-1.1', 'deprecated': False},
    'sgi-b-2.0': {'id': 'SGI-B-2.0', 'deprecated': False},
    'sgi-opengl': {'id': 'SGI-OpenGL', 'deprecated': False},
    'sgp4': {'id': 'SGP4', 'deprecated': False},
    'shl-0.5': {'id': 'SHL-0.5', 'deprecated': False},
    'shl-0.51': {'id': 'SHL-0.51', 'deprecated': False},
    'simpl-2.0': {'id': 'SimPL-2.0', 'deprecated': False},
    'sissl': {'id': 'SISSL', 'deprecated': False},
    'sissl-1.2': {'id': 'SISSL-1.2', 'deprecated': False},
    'sl': {'id': 'SL', 'deprecated': False},
    'sleepycat': {'id': 'Sleepycat', 'deprecated': False},
    'smlnj': {'id': 'SMLNJ', 'deprecated': False},
    'smppl': {'id': 'SMPPL', 'deprecated': False},
    'snia': {'id': 'SNIA', 'deprecated': False},
    'snprintf': {'id': 'snprintf', 'deprecated': False},
    'softsurfer': {'id': 'softSurfer', 'deprecated': False},
    'soundex': {'id': 'Soundex', 'deprecated': False},
    'spencer-86': {'id': 'Spencer-86', 'deprecated': False},
    'spencer-94': {'id': 'Spencer-94', 'deprecated': False},
    'spencer-99': {'id': 'Spencer-99', 'deprecated': False},
    'spl-1.0': {'id': 'SPL-1.0', 'deprecated': False},
    'ssh-keyscan': {'id': 'ssh-keyscan', 'deprecated': False},
    'ssh-openssh': {'id': 'SSH-OpenSSH', 'deprecated': False},
    'ssh-short': {'id': 'SSH-short', 'deprecated': False},
    'ssleay-standalone': {'id': 'SSLeay-standalone', 'deprecated': False},
    'sspl-1.0': {'id': 'SSPL-1.0', 'deprecated': False},
    'standardml-nj': {'id': 'StandardML-NJ', 'deprecated': True},
    'sugarcrm-1.1.3': {'id': 'SugarCRM-1.1.3', 'deprecated': False},
    'sun-ppp': {'id': 'Sun-PPP', 'deprecated': False},
    'sun-ppp-2000': {'id': 'Sun-PPP-2000', 'deprecated': False},
    'sunpro': {'id': 'SunPro', 'deprecated': False},
    'swl': {'id': 'SWL', 'deprecated': False},
    'swrule': {'id': 'swrule', 'deprecated': False},
    'symlinks': {'id': 'Symlinks', 'deprecated': False},
    'tapr-ohl-1.0': {'id': 'TAPR-OHL-1.0', 'deprecated': False},
    'tcl': {'id': 'TCL', 'deprecated': False},
    'tcp-wrappers': {'id': 'TCP-wrappers', 'deprecated': False},
    'termreadkey': {'id': 'TermReadKey', 'deprecated': False},
    'tgppl-1.0': {'id': 'TGPPL-1.0', 'deprecated': False},
    'threeparttable': {'id': 'threeparttable', 'deprecated': False},
    'tmate': {'id': 'TMate', 'deprecated': False},
    'torque-1.1': {'id': 'TORQUE-1.1', 'deprecated': False},
    'tosl': {'id': 'TOSL', 'deprecated': False},
    'tpdl': {'id': 'TPDL', 'deprecated': False},
    'tpl-1.0': {'id': 'TPL-1.0', 'deprecated': False},
    'ttwl': {'id': 'TTWL', 'deprecated': False},
    'ttyp0': {'id': 'TTYP0', 'deprecated': False},
    'tu-berlin-1.0': {'id': 'TU-Berlin-1.0', 'deprecated': False},
    'tu-berlin-2.0': {'id': 'TU-Berlin-2.0', 'deprecated': False},
    'ubuntu-font-1.0': {'id': 'Ubuntu-font-1.0', 'deprecated': False},
    'ucar': {'id': 'UCAR', 'deprecated': False},
    'ucl-1.0': {'id': 'UCL-1.0', 'deprecated': False},
    'ulem': {'id': 'ulem', 'deprecated': False},
    'umich-merit': {'id': 'UMich-Merit', 'deprecated': False},
    'unicode-3.0': {'id': 'Unicode-3.0', 'deprecated': False},
    'unicode-dfs-2015': {'id': 'Unicode-DFS-2015', 'deprecated': False},
    'unicode-dfs-2016': {'id': 'Unicode-DFS-2016', 'deprecated': False},
    'unicode-tou': {'id': 'Unicode-TOU', 'deprecated': False},
    'unixcrypt': {'id': 'UnixCrypt', 'deprecated': False},
    'unlicense': {'id': 'Unlicense', 'deprecated': False},
    'upl-1.0': {'id': 'UPL-1.0', 'deprecated': False},
    'urt-rle': {'id': 'URT-RLE', 'deprecated': False},
    'vim': {'id': 'Vim', 'deprecated': False},
    'vostrom': {'id': 'VOSTROM', 'deprecated': False},
    'vsl-1.0': {'id': 'VSL-1.0', 'deprecated': False},
    'w3c': {'id': 'W3C', 'deprecated': False},
    'w3c-19980720': {'id': 'W3C-19980720', 'deprecated': False},
    'w3c-20150513': {'id': 'W3C-20150513', 'deprecated': False},
    'w3m': {'id': 'w3m', 'deprecated': False},
    'watcom-1.0': {'id': 'Watcom-1.0', 'deprecated': False},
    'widget-workshop': {'id': 'Widget-Workshop', 'deprecated': False},
    'wsuipa': {'id': 'Wsuipa', 'deprecated': False},
    'wtfpl': {'id': 'WTFPL', 'deprecated': False},
    'wxwindows': {'id': 'wxWindows', 'deprecated': True},
    'x11': {'id': 'X11', 'deprecated': False},
    'x11-distribute-modifications-variant': {'id': 'X11-distribute-modifications-variant', 'deprecated': False},
    'x11-swapped': {'id': 'X11-swapped', 'deprecated': False},
    'xdebug-1.03': {'id': 'Xdebug-1.03', 'deprecated': False},
    'xerox': {'id': 'Xerox', 'deprecated': False},
    'xfig': {'id': 'Xfig', 'deprecated': False},
    'xfree86-1.1': {'id': 'XFree86-1.1', 'deprecated': False},
    'xinetd': {'id': 'xinetd', 'deprecated': False},
    'xkeyboard-config-zinoviev': {'id': 'xkeyboard-config-Zinoviev', 'deprecated': False},
    'xlock': {'id': 'xlock', 'deprecated': False},
    'xnet': {'id': 'Xnet', 'deprecated': False},
    'xpp': {'id': 'xpp', 'deprecated': False},
    'xskat': {'id': 'XSkat', 'deprecated': False},
    'xzoom': {'id': 'xzoom', 'deprecated': False},
    'ypl-1.0': {'id': 'YPL-1.0', 'deprecated': False},
    'ypl-1.1': {'id': 'YPL-1.1', 'deprecated': False},
    'zed': {'id': 'Zed', 'deprecated': False},
    'zeeff': {'id': 'Zeeff', 'deprecated': False},
    'zend-2.0': {'id': 'Zend-2.0', 'deprecated': False},
    'zimbra-1.3': {'id': 'Zimbra-1.3', 'deprecated': False},
    'zimbra-1.4': {'id': 'Zimbra-1.4', 'deprecated': False},
    'zlib': {'id': 'Zlib', 'deprecated': False},
    'zlib-acknowledgement': {'id': 'zlib-acknowledgement', 'deprecated': False},
    'zpl-1.1': {'id': 'ZPL-1.1', 'deprecated': False},
    'zpl-2.0': {'id': 'ZPL-2.0', 'deprecated': False},
    'zpl-2.1': {'id': 'ZPL-2.1', 'deprecated': False},
}

EXCEPTIONS: dict[str, SPDXException] = {
    '389-exception': {'id': '389-exception', 'deprecated': False},
    'asterisk-exception': {'id': 'Asterisk-exception', 'deprecated': False},
    'asterisk-linking-protocols-exception': {'id': 'Asterisk-linking-protocols-exception', 'deprecated': False},
    'autoconf-exception-2.0': {'id': 'Autoconf-exception-2.0', 'deprecated': False},
    'autoconf-exception-3.0': {'id': 'Autoconf-exception-3.0', 'deprecated': False},
    'autoconf-exception-generic': {'id': 'Autoconf-exception-generic', 'deprecated': False},
    'autoconf-exception-generic-3.0': {'id': 'Autoconf-exception-generic-3.0', 'deprecated': False},
    'autoconf-exception-macro': {'id': 'Autoconf-exception-macro', 'deprecated': False},
    'bison-exception-1.24': {'id': 'Bison-exception-1.24', 'deprecated': False},
    'bison-exception-2.2': {'id': 'Bison-exception-2.2', 'deprecated': False},
    'bootloader-exception': {'id': 'Bootloader-exception', 'deprecated': False},
    'classpath-exception-2.0': {'id': 'Classpath-exception-2.0', 'deprecated': False},
    'clisp-exception-2.0': {'id': 'CLISP-exception-2.0', 'deprecated': False},
    'cryptsetup-openssl-exception': {'id': 'cryptsetup-OpenSSL-exception', 'deprecated': False},
    'digirule-foss-exception': {'id': 'DigiRule-FOSS-exception', 'deprecated': False},
    'ecos-exception-2.0': {'id': 'eCos-exception-2.0', 'deprecated': False},
    'erlang-otp-linking-exception': {'id': 'erlang-otp-linking-exception', 'deprecated': False},
    'fawkes-runtime-exception': {'id': 'Fawkes-Runtime-exception', 'deprecated': False},
    'fltk-exception': {'id': 'FLTK-exception', 'deprecated': False},
    'fmt-exception': {'id': 'fmt-exception', 'deprecated': False},
    'font-exception-2.0': {'id': 'Font-exception-2.0', 'deprecated': False},
    'freertos-exception-2.0': {'id': 'freertos-exception-2.0', 'deprecated': False},
    'gcc-exception-2.0': {'id': 'GCC-exception-2.0', 'deprecated': False},
    'gcc-exception-2.0-note': {'id': 'GCC-exception-2.0-note', 'deprecated': False},
    'gcc-exception-3.1': {'id': 'GCC-exception-3.1', 'deprecated': False},
    'gmsh-exception': {'id': 'Gmsh-exception', 'deprecated': False},
    'gnat-exception': {'id': 'GNAT-exception', 'deprecated': False},
    'gnome-examples-exception': {'id': 'GNOME-examples-exception', 'deprecated': False},
    'gnu-compiler-exception': {'id': 'GNU-compiler-exception', 'deprecated': False},
    'gnu-javamail-exception': {'id': 'gnu-javamail-exception', 'deprecated': False},
    'gpl-3.0-interface-exception': {'id': 'GPL-3.0-interface-exception', 'deprecated': False},
    'gpl-3.0-linking-exception': {'id': 'GPL-3.0-linking-exception', 'deprecated': False},
    'gpl-3.0-linking-source-exception': {'id': 'GPL-3.0-linking-source-exception', 'deprecated': False},
    'gpl-cc-1.0': {'id': 'GPL-CC-1.0', 'deprecated': False},
    'gstreamer-exception-2005': {'id': 'GStreamer-exception-2005', 'deprecated': False},
    'gstreamer-exception-2008': {'id': 'GStreamer-exception-2008', 'deprecated': False},
    'i2p-gpl-java-exception': {'id': 'i2p-gpl-java-exception', 'deprecated': False},
    'kicad-libraries-exception': {'id': 'KiCad-libraries-exception', 'deprecated': False},
    'lgpl-3.0-linking-exception': {'id': 'LGPL-3.0-linking-exception', 'deprecated': False},
    'libpri-openh323-exception': {'id': 'libpri-OpenH323-exception', 'deprecated': False},
    'libtool-exception': {'id': 'Libtool-exception', 'deprecated': False},
    'linux-syscall-note': {'id': 'Linux-syscall-note', 'deprecated': False},
    'llgpl': {'id': 'LLGPL', 'deprecated': False},
    'llvm-exception': {'id': 'LLVM-exception', 'deprecated': False},
    'lzma-exception': {'id': 'LZMA-exception', 'deprecated': False},
    'mif-exception': {'id': 'mif-exception', 'deprecated': False},
    'nokia-qt-exception-1.1': {'id': 'Nokia-Qt-exception-1.1', 'deprecated': True},
    'ocaml-lgpl-linking-exception': {'id': 'OCaml-LGPL-linking-exception', 'deprecated': False},
    'occt-exception-1.0': {'id': 'OCCT-exception-1.0', 'deprecated': False},
    'openjdk-assembly-exception-1.0': {'id': 'OpenJDK-assembly-exception-1.0', 'deprecated': False},
    'openvpn-openssl-exception': {'id': 'openvpn-openssl-exception', 'deprecated': False},
    'pcre2-exception': {'id': 'PCRE2-exception', 'deprecated': False},
    'ps-or-pdf-font-exception-20170817': {'id': 'PS-or-PDF-font-exception-20170817', 'deprecated': False},
    'qpl-1.0-inria-2004-exception': {'id': 'QPL-1.0-INRIA-2004-exception', 'deprecated': False},
    'qt-gpl-exception-1.0': {'id': 'Qt-GPL-exception-1.0', 'deprecated': False},
    'qt-lgpl-exception-1.1': {'id': 'Qt-LGPL-exception-1.1', 'deprecated': False},
    'qwt-exception-1.0': {'id': 'Qwt-exception-1.0', 'deprecated': False},
    'romic-exception': {'id': 'romic-exception', 'deprecated': False},
    'rrdtool-floss-exception-2.0': {'id': 'RRDtool-FLOSS-exception-2.0', 'deprecated': False},
    'sane-exception': {'id': 'SANE-exception', 'deprecated': False},
    'shl-2.0': {'id': 'SHL-2.0', 'deprecated': False},
    'shl-2.1': {'id': 'SHL-2.1', 'deprecated': False},
    'stunnel-exception': {'id': 'stunnel-exception', 'deprecated': False},
    'swi-exception': {'id': 'SWI-exception', 'deprecated': False},
    'swift-exception': {'id': 'Swift-exception', 'deprecated': False},
    'texinfo-exception': {'id': 'Texinfo-exception', 'deprecated': False},
    'u-boot-exception-2.0': {'id': 'u-boot-exception-2.0', 'deprecated': False},
    'ubdl-exception': {'id': 'UBDL-exception', 'deprecated': False},
    'universal-foss-exception-1.0': {'id': 'Universal-FOSS-exception-1.0', 'deprecated': False},
    'vsftpd-openssl-exception': {'id': 'vsftpd-openssl-exception', 'deprecated': False},
    'wxwindows-exception-3.1': {'id': 'WxWindows-exception-3.1', 'deprecated': False},
    'x11vnc-openssl-exception': {'id': 'x11vnc-openssl-exception', 'deprecated': False},
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\licenses\_spdx.py ===

from __future__ import annotations

from typing import TypedDict

class SPDXLicense(TypedDict):
    id: str
    deprecated: bool

class SPDXException(TypedDict):
    id: str
    deprecated: bool


VERSION = '3.25.0'

LICENSES: dict[str, SPDXLicense] = {
    '0bsd': {'id': '0BSD', 'deprecated': False},
    '3d-slicer-1.0': {'id': '3D-Slicer-1.0', 'deprecated': False},
    'aal': {'id': 'AAL', 'deprecated': False},
    'abstyles': {'id': 'Abstyles', 'deprecated': False},
    'adacore-doc': {'id': 'AdaCore-doc', 'deprecated': False},
    'adobe-2006': {'id': 'Adobe-2006', 'deprecated': False},
    'adobe-display-postscript': {'id': 'Adobe-Display-PostScript', 'deprecated': False},
    'adobe-glyph': {'id': 'Adobe-Glyph', 'deprecated': False},
    'adobe-utopia': {'id': 'Adobe-Utopia', 'deprecated': False},
    'adsl': {'id': 'ADSL', 'deprecated': False},
    'afl-1.1': {'id': 'AFL-1.1', 'deprecated': False},
    'afl-1.2': {'id': 'AFL-1.2', 'deprecated': False},
    'afl-2.0': {'id': 'AFL-2.0', 'deprecated': False},
    'afl-2.1': {'id': 'AFL-2.1', 'deprecated': False},
    'afl-3.0': {'id': 'AFL-3.0', 'deprecated': False},
    'afmparse': {'id': 'Afmparse', 'deprecated': False},
    'agpl-1.0': {'id': 'AGPL-1.0', 'deprecated': True},
    'agpl-1.0-only': {'id': 'AGPL-1.0-only', 'deprecated': False},
    'agpl-1.0-or-later': {'id': 'AGPL-1.0-or-later', 'deprecated': False},
    'agpl-3.0': {'id': 'AGPL-3.0', 'deprecated': True},
    'agpl-3.0-only': {'id': 'AGPL-3.0-only', 'deprecated': False},
    'agpl-3.0-or-later': {'id': 'AGPL-3.0-or-later', 'deprecated': False},
    'aladdin': {'id': 'Aladdin', 'deprecated': False},
    'amd-newlib': {'id': 'AMD-newlib', 'deprecated': False},
    'amdplpa': {'id': 'AMDPLPA', 'deprecated': False},
    'aml': {'id': 'AML', 'deprecated': False},
    'aml-glslang': {'id': 'AML-glslang', 'deprecated': False},
    'ampas': {'id': 'AMPAS', 'deprecated': False},
    'antlr-pd': {'id': 'ANTLR-PD', 'deprecated': False},
    'antlr-pd-fallback': {'id': 'ANTLR-PD-fallback', 'deprecated': False},
    'any-osi': {'id': 'any-OSI', 'deprecated': False},
    'apache-1.0': {'id': 'Apache-1.0', 'deprecated': False},
    'apache-1.1': {'id': 'Apache-1.1', 'deprecated': False},
    'apache-2.0': {'id': 'Apache-2.0', 'deprecated': False},
    'apafml': {'id': 'APAFML', 'deprecated': False},
    'apl-1.0': {'id': 'APL-1.0', 'deprecated': False},
    'app-s2p': {'id': 'App-s2p', 'deprecated': False},
    'apsl-1.0': {'id': 'APSL-1.0', 'deprecated': False},
    'apsl-1.1': {'id': 'APSL-1.1', 'deprecated': False},
    'apsl-1.2': {'id': 'APSL-1.2', 'deprecated': False},
    'apsl-2.0': {'id': 'APSL-2.0', 'deprecated': False},
    'arphic-1999': {'id': 'Arphic-1999', 'deprecated': False},
    'artistic-1.0': {'id': 'Artistic-1.0', 'deprecated': False},
    'artistic-1.0-cl8': {'id': 'Artistic-1.0-cl8', 'deprecated': False},
    'artistic-1.0-perl': {'id': 'Artistic-1.0-Perl', 'deprecated': False},
    'artistic-2.0': {'id': 'Artistic-2.0', 'deprecated': False},
    'aswf-digital-assets-1.0': {'id': 'ASWF-Digital-Assets-1.0', 'deprecated': False},
    'aswf-digital-assets-1.1': {'id': 'ASWF-Digital-Assets-1.1', 'deprecated': False},
    'baekmuk': {'id': 'Baekmuk', 'deprecated': False},
    'bahyph': {'id': 'Bahyph', 'deprecated': False},
    'barr': {'id': 'Barr', 'deprecated': False},
    'bcrypt-solar-designer': {'id': 'bcrypt-Solar-Designer', 'deprecated': False},
    'beerware': {'id': 'Beerware', 'deprecated': False},
    'bitstream-charter': {'id': 'Bitstream-Charter', 'deprecated': False},
    'bitstream-vera': {'id': 'Bitstream-Vera', 'deprecated': False},
    'bittorrent-1.0': {'id': 'BitTorrent-1.0', 'deprecated': False},
    'bittorrent-1.1': {'id': 'BitTorrent-1.1', 'deprecated': False},
    'blessing': {'id': 'blessing', 'deprecated': False},
    'blueoak-1.0.0': {'id': 'BlueOak-1.0.0', 'deprecated': False},
    'boehm-gc': {'id': 'Boehm-GC', 'deprecated': False},
    'borceux': {'id': 'Borceux', 'deprecated': False},
    'brian-gladman-2-clause': {'id': 'Brian-Gladman-2-Clause', 'deprecated': False},
    'brian-gladman-3-clause': {'id': 'Brian-Gladman-3-Clause', 'deprecated': False},
    'bsd-1-clause': {'id': 'BSD-1-Clause', 'deprecated': False},
    'bsd-2-clause': {'id': 'BSD-2-Clause', 'deprecated': False},
    'bsd-2-clause-darwin': {'id': 'BSD-2-Clause-Darwin', 'deprecated': False},
    'bsd-2-clause-first-lines': {'id': 'BSD-2-Clause-first-lines', 'deprecated': False},
    'bsd-2-clause-freebsd': {'id': 'BSD-2-Clause-FreeBSD', 'deprecated': True},
    'bsd-2-clause-netbsd': {'id': 'BSD-2-Clause-NetBSD', 'deprecated': True},
    'bsd-2-clause-patent': {'id': 'BSD-2-Clause-Patent', 'deprecated': False},
    'bsd-2-clause-views': {'id': 'BSD-2-Clause-Views', 'deprecated': False},
    'bsd-3-clause': {'id': 'BSD-3-Clause', 'deprecated': False},
    'bsd-3-clause-acpica': {'id': 'BSD-3-Clause-acpica', 'deprecated': False},
    'bsd-3-clause-attribution': {'id': 'BSD-3-Clause-Attribution', 'deprecated': False},
    'bsd-3-clause-clear': {'id': 'BSD-3-Clause-Clear', 'deprecated': False},
    'bsd-3-clause-flex': {'id': 'BSD-3-Clause-flex', 'deprecated': False},
    'bsd-3-clause-hp': {'id': 'BSD-3-Clause-HP', 'deprecated': False},
    'bsd-3-clause-lbnl': {'id': 'BSD-3-Clause-LBNL', 'deprecated': False},
    'bsd-3-clause-modification': {'id': 'BSD-3-Clause-Modification', 'deprecated': False},
    'bsd-3-clause-no-military-license': {'id': 'BSD-3-Clause-No-Military-License', 'deprecated': False},
    'bsd-3-clause-no-nuclear-license': {'id': 'BSD-3-Clause-No-Nuclear-License', 'deprecated': False},
    'bsd-3-clause-no-nuclear-license-2014': {'id': 'BSD-3-Clause-No-Nuclear-License-2014', 'deprecated': False},
    'bsd-3-clause-no-nuclear-warranty': {'id': 'BSD-3-Clause-No-Nuclear-Warranty', 'deprecated': False},
    'bsd-3-clause-open-mpi': {'id': 'BSD-3-Clause-Open-MPI', 'deprecated': False},
    'bsd-3-clause-sun': {'id': 'BSD-3-Clause-Sun', 'deprecated': False},
    'bsd-4-clause': {'id': 'BSD-4-Clause', 'deprecated': False},
    'bsd-4-clause-shortened': {'id': 'BSD-4-Clause-Shortened', 'deprecated': False},
    'bsd-4-clause-uc': {'id': 'BSD-4-Clause-UC', 'deprecated': False},
    'bsd-4.3reno': {'id': 'BSD-4.3RENO', 'deprecated': False},
    'bsd-4.3tahoe': {'id': 'BSD-4.3TAHOE', 'deprecated': False},
    'bsd-advertising-acknowledgement': {'id': 'BSD-Advertising-Acknowledgement', 'deprecated': False},
    'bsd-attribution-hpnd-disclaimer': {'id': 'BSD-Attribution-HPND-disclaimer', 'deprecated': False},
    'bsd-inferno-nettverk': {'id': 'BSD-Inferno-Nettverk', 'deprecated': False},
    'bsd-protection': {'id': 'BSD-Protection', 'deprecated': False},
    'bsd-source-beginning-file': {'id': 'BSD-Source-beginning-file', 'deprecated': False},
    'bsd-source-code': {'id': 'BSD-Source-Code', 'deprecated': False},
    'bsd-systemics': {'id': 'BSD-Systemics', 'deprecated': False},
    'bsd-systemics-w3works': {'id': 'BSD-Systemics-W3Works', 'deprecated': False},
    'bsl-1.0': {'id': 'BSL-1.0', 'deprecated': False},
    'busl-1.1': {'id': 'BUSL-1.1', 'deprecated': False},
    'bzip2-1.0.5': {'id': 'bzip2-1.0.5', 'deprecated': True},
    'bzip2-1.0.6': {'id': 'bzip2-1.0.6', 'deprecated': False},
    'c-uda-1.0': {'id': 'C-UDA-1.0', 'deprecated': False},
    'cal-1.0': {'id': 'CAL-1.0', 'deprecated': False},
    'cal-1.0-combined-work-exception': {'id': 'CAL-1.0-Combined-Work-Exception', 'deprecated': False},
    'caldera': {'id': 'Caldera', 'deprecated': False},
    'caldera-no-preamble': {'id': 'Caldera-no-preamble', 'deprecated': False},
    'catharon': {'id': 'Catharon', 'deprecated': False},
    'catosl-1.1': {'id': 'CATOSL-1.1', 'deprecated': False},
    'cc-by-1.0': {'id': 'CC-BY-1.0', 'deprecated': False},
    'cc-by-2.0': {'id': 'CC-BY-2.0', 'deprecated': False},
    'cc-by-2.5': {'id': 'CC-BY-2.5', 'deprecated': False},
    'cc-by-2.5-au': {'id': 'CC-BY-2.5-AU', 'deprecated': False},
    'cc-by-3.0': {'id': 'CC-BY-3.0', 'deprecated': False},
    'cc-by-3.0-at': {'id': 'CC-BY-3.0-AT', 'deprecated': False},
    'cc-by-3.0-au': {'id': 'CC-BY-3.0-AU', 'deprecated': False},
    'cc-by-3.0-de': {'id': 'CC-BY-3.0-DE', 'deprecated': False},
    'cc-by-3.0-igo': {'id': 'CC-BY-3.0-IGO', 'deprecated': False},
    'cc-by-3.0-nl': {'id': 'CC-BY-3.0-NL', 'deprecated': False},
    'cc-by-3.0-us': {'id': 'CC-BY-3.0-US', 'deprecated': False},
    'cc-by-4.0': {'id': 'CC-BY-4.0', 'deprecated': False},
    'cc-by-nc-1.0': {'id': 'CC-BY-NC-1.0', 'deprecated': False},
    'cc-by-nc-2.0': {'id': 'CC-BY-NC-2.0', 'deprecated': False},
    'cc-by-nc-2.5': {'id': 'CC-BY-NC-2.5', 'deprecated': False},
    'cc-by-nc-3.0': {'id': 'CC-BY-NC-3.0', 'deprecated': False},
    'cc-by-nc-3.0-de': {'id': 'CC-BY-NC-3.0-DE', 'deprecated': False},
    'cc-by-nc-4.0': {'id': 'CC-BY-NC-4.0', 'deprecated': False},
    'cc-by-nc-nd-1.0': {'id': 'CC-BY-NC-ND-1.0', 'deprecated': False},
    'cc-by-nc-nd-2.0': {'id': 'CC-BY-NC-ND-2.0', 'deprecated': False},
    'cc-by-nc-nd-2.5': {'id': 'CC-BY-NC-ND-2.5', 'deprecated': False},
    'cc-by-nc-nd-3.0': {'id': 'CC-BY-NC-ND-3.0', 'deprecated': False},
    'cc-by-nc-nd-3.0-de': {'id': 'CC-BY-NC-ND-3.0-DE', 'deprecated': False},
    'cc-by-nc-nd-3.0-igo': {'id': 'CC-BY-NC-ND-3.0-IGO', 'deprecated': False},
    'cc-by-nc-nd-4.0': {'id': 'CC-BY-NC-ND-4.0', 'deprecated': False},
    'cc-by-nc-sa-1.0': {'id': 'CC-BY-NC-SA-1.0', 'deprecated': False},
    'cc-by-nc-sa-2.0': {'id': 'CC-BY-NC-SA-2.0', 'deprecated': False},
    'cc-by-nc-sa-2.0-de': {'id': 'CC-BY-NC-SA-2.0-DE', 'deprecated': False},
    'cc-by-nc-sa-2.0-fr': {'id': 'CC-BY-NC-SA-2.0-FR', 'deprecated': False},
    'cc-by-nc-sa-2.0-uk': {'id': 'CC-BY-NC-SA-2.0-UK', 'deprecated': False},
    'cc-by-nc-sa-2.5': {'id': 'CC-BY-NC-SA-2.5', 'deprecated': False},
    'cc-by-nc-sa-3.0': {'id': 'CC-BY-NC-SA-3.0', 'deprecated': False},
    'cc-by-nc-sa-3.0-de': {'id': 'CC-BY-NC-SA-3.0-DE', 'deprecated': False},
    'cc-by-nc-sa-3.0-igo': {'id': 'CC-BY-NC-SA-3.0-IGO', 'deprecated': False},
    'cc-by-nc-sa-4.0': {'id': 'CC-BY-NC-SA-4.0', 'deprecated': False},
    'cc-by-nd-1.0': {'id': 'CC-BY-ND-1.0', 'deprecated': False},
    'cc-by-nd-2.0': {'id': 'CC-BY-ND-2.0', 'deprecated': False},
    'cc-by-nd-2.5': {'id': 'CC-BY-ND-2.5', 'deprecated': False},
    'cc-by-nd-3.0': {'id': 'CC-BY-ND-3.0', 'deprecated': False},
    'cc-by-nd-3.0-de': {'id': 'CC-BY-ND-3.0-DE', 'deprecated': False},
    'cc-by-nd-4.0': {'id': 'CC-BY-ND-4.0', 'deprecated': False},
    'cc-by-sa-1.0': {'id': 'CC-BY-SA-1.0', 'deprecated': False},
    'cc-by-sa-2.0': {'id': 'CC-BY-SA-2.0', 'deprecated': False},
    'cc-by-sa-2.0-uk': {'id': 'CC-BY-SA-2.0-UK', 'deprecated': False},
    'cc-by-sa-2.1-jp': {'id': 'CC-BY-SA-2.1-JP', 'deprecated': False},
    'cc-by-sa-2.5': {'id': 'CC-BY-SA-2.5', 'deprecated': False},
    'cc-by-sa-3.0': {'id': 'CC-BY-SA-3.0', 'deprecated': False},
    'cc-by-sa-3.0-at': {'id': 'CC-BY-SA-3.0-AT', 'deprecated': False},
    'cc-by-sa-3.0-de': {'id': 'CC-BY-SA-3.0-DE', 'deprecated': False},
    'cc-by-sa-3.0-igo': {'id': 'CC-BY-SA-3.0-IGO', 'deprecated': False},
    'cc-by-sa-4.0': {'id': 'CC-BY-SA-4.0', 'deprecated': False},
    'cc-pddc': {'id': 'CC-PDDC', 'deprecated': False},
    'cc0-1.0': {'id': 'CC0-1.0', 'deprecated': False},
    'cddl-1.0': {'id': 'CDDL-1.0', 'deprecated': False},
    'cddl-1.1': {'id': 'CDDL-1.1', 'deprecated': False},
    'cdl-1.0': {'id': 'CDL-1.0', 'deprecated': False},
    'cdla-permissive-1.0': {'id': 'CDLA-Permissive-1.0', 'deprecated': False},
    'cdla-permissive-2.0': {'id': 'CDLA-Permissive-2.0', 'deprecated': False},
    'cdla-sharing-1.0': {'id': 'CDLA-Sharing-1.0', 'deprecated': False},
    'cecill-1.0': {'id': 'CECILL-1.0', 'deprecated': False},
    'cecill-1.1': {'id': 'CECILL-1.1', 'deprecated': False},
    'cecill-2.0': {'id': 'CECILL-2.0', 'deprecated': False},
    'cecill-2.1': {'id': 'CECILL-2.1', 'deprecated': False},
    'cecill-b': {'id': 'CECILL-B', 'deprecated': False},
    'cecill-c': {'id': 'CECILL-C', 'deprecated': False},
    'cern-ohl-1.1': {'id': 'CERN-OHL-1.1', 'deprecated': False},
    'cern-ohl-1.2': {'id': 'CERN-OHL-1.2', 'deprecated': False},
    'cern-ohl-p-2.0': {'id': 'CERN-OHL-P-2.0', 'deprecated': False},
    'cern-ohl-s-2.0': {'id': 'CERN-OHL-S-2.0', 'deprecated': False},
    'cern-ohl-w-2.0': {'id': 'CERN-OHL-W-2.0', 'deprecated': False},
    'cfitsio': {'id': 'CFITSIO', 'deprecated': False},
    'check-cvs': {'id': 'check-cvs', 'deprecated': False},
    'checkmk': {'id': 'checkmk', 'deprecated': False},
    'clartistic': {'id': 'ClArtistic', 'deprecated': False},
    'clips': {'id': 'Clips', 'deprecated': False},
    'cmu-mach': {'id': 'CMU-Mach', 'deprecated': False},
    'cmu-mach-nodoc': {'id': 'CMU-Mach-nodoc', 'deprecated': False},
    'cnri-jython': {'id': 'CNRI-Jython', 'deprecated': False},
    'cnri-python': {'id': 'CNRI-Python', 'deprecated': False},
    'cnri-python-gpl-compatible': {'id': 'CNRI-Python-GPL-Compatible', 'deprecated': False},
    'coil-1.0': {'id': 'COIL-1.0', 'deprecated': False},
    'community-spec-1.0': {'id': 'Community-Spec-1.0', 'deprecated': False},
    'condor-1.1': {'id': 'Condor-1.1', 'deprecated': False},
    'copyleft-next-0.3.0': {'id': 'copyleft-next-0.3.0', 'deprecated': False},
    'copyleft-next-0.3.1': {'id': 'copyleft-next-0.3.1', 'deprecated': False},
    'cornell-lossless-jpeg': {'id': 'Cornell-Lossless-JPEG', 'deprecated': False},
    'cpal-1.0': {'id': 'CPAL-1.0', 'deprecated': False},
    'cpl-1.0': {'id': 'CPL-1.0', 'deprecated': False},
    'cpol-1.02': {'id': 'CPOL-1.02', 'deprecated': False},
    'cronyx': {'id': 'Cronyx', 'deprecated': False},
    'crossword': {'id': 'Crossword', 'deprecated': False},
    'crystalstacker': {'id': 'CrystalStacker', 'deprecated': False},
    'cua-opl-1.0': {'id': 'CUA-OPL-1.0', 'deprecated': False},
    'cube': {'id': 'Cube', 'deprecated': False},
    'curl': {'id': 'curl', 'deprecated': False},
    'cve-tou': {'id': 'cve-tou', 'deprecated': False},
    'd-fsl-1.0': {'id': 'D-FSL-1.0', 'deprecated': False},
    'dec-3-clause': {'id': 'DEC-3-Clause', 'deprecated': False},
    'diffmark': {'id': 'diffmark', 'deprecated': False},
    'dl-de-by-2.0': {'id': 'DL-DE-BY-2.0', 'deprecated': False},
    'dl-de-zero-2.0': {'id': 'DL-DE-ZERO-2.0', 'deprecated': False},
    'doc': {'id': 'DOC', 'deprecated': False},
    'docbook-schema': {'id': 'DocBook-Schema', 'deprecated': False},
    'docbook-xml': {'id': 'DocBook-XML', 'deprecated': False},
    'dotseqn': {'id': 'Dotseqn', 'deprecated': False},
    'drl-1.0': {'id': 'DRL-1.0', 'deprecated': False},
    'drl-1.1': {'id': 'DRL-1.1', 'deprecated': False},
    'dsdp': {'id': 'DSDP', 'deprecated': False},
    'dtoa': {'id': 'dtoa', 'deprecated': False},
    'dvipdfm': {'id': 'dvipdfm', 'deprecated': False},
    'ecl-1.0': {'id': 'ECL-1.0', 'deprecated': False},
    'ecl-2.0': {'id': 'ECL-2.0', 'deprecated': False},
    'ecos-2.0': {'id': 'eCos-2.0', 'deprecated': True},
    'efl-1.0': {'id': 'EFL-1.0', 'deprecated': False},
    'efl-2.0': {'id': 'EFL-2.0', 'deprecated': False},
    'egenix': {'id': 'eGenix', 'deprecated': False},
    'elastic-2.0': {'id': 'Elastic-2.0', 'deprecated': False},
    'entessa': {'id': 'Entessa', 'deprecated': False},
    'epics': {'id': 'EPICS', 'deprecated': False},
    'epl-1.0': {'id': 'EPL-1.0', 'deprecated': False},
    'epl-2.0': {'id': 'EPL-2.0', 'deprecated': False},
    'erlpl-1.1': {'id': 'ErlPL-1.1', 'deprecated': False},
    'etalab-2.0': {'id': 'etalab-2.0', 'deprecated': False},
    'eudatagrid': {'id': 'EUDatagrid', 'deprecated': False},
    'eupl-1.0': {'id': 'EUPL-1.0', 'deprecated': False},
    'eupl-1.1': {'id': 'EUPL-1.1', 'deprecated': False},
    'eupl-1.2': {'id': 'EUPL-1.2', 'deprecated': False},
    'eurosym': {'id': 'Eurosym', 'deprecated': False},
    'fair': {'id': 'Fair', 'deprecated': False},
    'fbm': {'id': 'FBM', 'deprecated': False},
    'fdk-aac': {'id': 'FDK-AAC', 'deprecated': False},
    'ferguson-twofish': {'id': 'Ferguson-Twofish', 'deprecated': False},
    'frameworx-1.0': {'id': 'Frameworx-1.0', 'deprecated': False},
    'freebsd-doc': {'id': 'FreeBSD-DOC', 'deprecated': False},
    'freeimage': {'id': 'FreeImage', 'deprecated': False},
    'fsfap': {'id': 'FSFAP', 'deprecated': False},
    'fsfap-no-warranty-disclaimer': {'id': 'FSFAP-no-warranty-disclaimer', 'deprecated': False},
    'fsful': {'id': 'FSFUL', 'deprecated': False},
    'fsfullr': {'id': 'FSFULLR', 'deprecated': False},
    'fsfullrwd': {'id': 'FSFULLRWD', 'deprecated': False},
    'ftl': {'id': 'FTL', 'deprecated': False},
    'furuseth': {'id': 'Furuseth', 'deprecated': False},
    'fwlw': {'id': 'fwlw', 'deprecated': False},
    'gcr-docs': {'id': 'GCR-docs', 'deprecated': False},
    'gd': {'id': 'GD', 'deprecated': False},
    'gfdl-1.1': {'id': 'GFDL-1.1', 'deprecated': True},
    'gfdl-1.1-invariants-only': {'id': 'GFDL-1.1-invariants-only', 'deprecated': False},
    'gfdl-1.1-invariants-or-later': {'id': 'GFDL-1.1-invariants-or-later', 'deprecated': False},
    'gfdl-1.1-no-invariants-only': {'id': 'GFDL-1.1-no-invariants-only', 'deprecated': False},
    'gfdl-1.1-no-invariants-or-later': {'id': 'GFDL-1.1-no-invariants-or-later', 'deprecated': False},
    'gfdl-1.1-only': {'id': 'GFDL-1.1-only', 'deprecated': False},
    'gfdl-1.1-or-later': {'id': 'GFDL-1.1-or-later', 'deprecated': False},
    'gfdl-1.2': {'id': 'GFDL-1.2', 'deprecated': True},
    'gfdl-1.2-invariants-only': {'id': 'GFDL-1.2-invariants-only', 'deprecated': False},
    'gfdl-1.2-invariants-or-later': {'id': 'GFDL-1.2-invariants-or-later', 'deprecated': False},
    'gfdl-1.2-no-invariants-only': {'id': 'GFDL-1.2-no-invariants-only', 'deprecated': False},
    'gfdl-1.2-no-invariants-or-later': {'id': 'GFDL-1.2-no-invariants-or-later', 'deprecated': False},
    'gfdl-1.2-only': {'id': 'GFDL-1.2-only', 'deprecated': False},
    'gfdl-1.2-or-later': {'id': 'GFDL-1.2-or-later', 'deprecated': False},
    'gfdl-1.3': {'id': 'GFDL-1.3', 'deprecated': True},
    'gfdl-1.3-invariants-only': {'id': 'GFDL-1.3-invariants-only', 'deprecated': False},
    'gfdl-1.3-invariants-or-later': {'id': 'GFDL-1.3-invariants-or-later', 'deprecated': False},
    'gfdl-1.3-no-invariants-only': {'id': 'GFDL-1.3-no-invariants-only', 'deprecated': False},
    'gfdl-1.3-no-invariants-or-later': {'id': 'GFDL-1.3-no-invariants-or-later', 'deprecated': False},
    'gfdl-1.3-only': {'id': 'GFDL-1.3-only', 'deprecated': False},
    'gfdl-1.3-or-later': {'id': 'GFDL-1.3-or-later', 'deprecated': False},
    'giftware': {'id': 'Giftware', 'deprecated': False},
    'gl2ps': {'id': 'GL2PS', 'deprecated': False},
    'glide': {'id': 'Glide', 'deprecated': False},
    'glulxe': {'id': 'Glulxe', 'deprecated': False},
    'glwtpl': {'id': 'GLWTPL', 'deprecated': False},
    'gnuplot': {'id': 'gnuplot', 'deprecated': False},
    'gpl-1.0': {'id': 'GPL-1.0', 'deprecated': True},
    'gpl-1.0+': {'id': 'GPL-1.0+', 'deprecated': True},
    'gpl-1.0-only': {'id': 'GPL-1.0-only', 'deprecated': False},
    'gpl-1.0-or-later': {'id': 'GPL-1.0-or-later', 'deprecated': False},
    'gpl-2.0': {'id': 'GPL-2.0', 'deprecated': True},
    'gpl-2.0+': {'id': 'GPL-2.0+', 'deprecated': True},
    'gpl-2.0-only': {'id': 'GPL-2.0-only', 'deprecated': False},
    'gpl-2.0-or-later': {'id': 'GPL-2.0-or-later', 'deprecated': False},
    'gpl-2.0-with-autoconf-exception': {'id': 'GPL-2.0-with-autoconf-exception', 'deprecated': True},
    'gpl-2.0-with-bison-exception': {'id': 'GPL-2.0-with-bison-exception', 'deprecated': True},
    'gpl-2.0-with-classpath-exception': {'id': 'GPL-2.0-with-classpath-exception', 'deprecated': True},
    'gpl-2.0-with-font-exception': {'id': 'GPL-2.0-with-font-exception', 'deprecated': True},
    'gpl-2.0-with-gcc-exception': {'id': 'GPL-2.0-with-GCC-exception', 'deprecated': True},
    'gpl-3.0': {'id': 'GPL-3.0', 'deprecated': True},
    'gpl-3.0+': {'id': 'GPL-3.0+', 'deprecated': True},
    'gpl-3.0-only': {'id': 'GPL-3.0-only', 'deprecated': False},
    'gpl-3.0-or-later': {'id': 'GPL-3.0-or-later', 'deprecated': False},
    'gpl-3.0-with-autoconf-exception': {'id': 'GPL-3.0-with-autoconf-exception', 'deprecated': True},
    'gpl-3.0-with-gcc-exception': {'id': 'GPL-3.0-with-GCC-exception', 'deprecated': True},
    'graphics-gems': {'id': 'Graphics-Gems', 'deprecated': False},
    'gsoap-1.3b': {'id': 'gSOAP-1.3b', 'deprecated': False},
    'gtkbook': {'id': 'gtkbook', 'deprecated': False},
    'gutmann': {'id': 'Gutmann', 'deprecated': False},
    'haskellreport': {'id': 'HaskellReport', 'deprecated': False},
    'hdparm': {'id': 'hdparm', 'deprecated': False},
    'hidapi': {'id': 'HIDAPI', 'deprecated': False},
    'hippocratic-2.1': {'id': 'Hippocratic-2.1', 'deprecated': False},
    'hp-1986': {'id': 'HP-1986', 'deprecated': False},
    'hp-1989': {'id': 'HP-1989', 'deprecated': False},
    'hpnd': {'id': 'HPND', 'deprecated': False},
    'hpnd-dec': {'id': 'HPND-DEC', 'deprecated': False},
    'hpnd-doc': {'id': 'HPND-doc', 'deprecated': False},
    'hpnd-doc-sell': {'id': 'HPND-doc-sell', 'deprecated': False},
    'hpnd-export-us': {'id': 'HPND-export-US', 'deprecated': False},
    'hpnd-export-us-acknowledgement': {'id': 'HPND-export-US-acknowledgement', 'deprecated': False},
    'hpnd-export-us-modify': {'id': 'HPND-export-US-modify', 'deprecated': False},
    'hpnd-export2-us': {'id': 'HPND-export2-US', 'deprecated': False},
    'hpnd-fenneberg-livingston': {'id': 'HPND-Fenneberg-Livingston', 'deprecated': False},
    'hpnd-inria-imag': {'id': 'HPND-INRIA-IMAG', 'deprecated': False},
    'hpnd-intel': {'id': 'HPND-Intel', 'deprecated': False},
    'hpnd-kevlin-henney': {'id': 'HPND-Kevlin-Henney', 'deprecated': False},
    'hpnd-markus-kuhn': {'id': 'HPND-Markus-Kuhn', 'deprecated': False},
    'hpnd-merchantability-variant': {'id': 'HPND-merchantability-variant', 'deprecated': False},
    'hpnd-mit-disclaimer': {'id': 'HPND-MIT-disclaimer', 'deprecated': False},
    'hpnd-netrek': {'id': 'HPND-Netrek', 'deprecated': False},
    'hpnd-pbmplus': {'id': 'HPND-Pbmplus', 'deprecated': False},
    'hpnd-sell-mit-disclaimer-xserver': {'id': 'HPND-sell-MIT-disclaimer-xserver', 'deprecated': False},
    'hpnd-sell-regexpr': {'id': 'HPND-sell-regexpr', 'deprecated': False},
    'hpnd-sell-variant': {'id': 'HPND-sell-variant', 'deprecated': False},
    'hpnd-sell-variant-mit-disclaimer': {'id': 'HPND-sell-variant-MIT-disclaimer', 'deprecated': False},
    'hpnd-sell-variant-mit-disclaimer-rev': {'id': 'HPND-sell-variant-MIT-disclaimer-rev', 'deprecated': False},
    'hpnd-uc': {'id': 'HPND-UC', 'deprecated': False},
    'hpnd-uc-export-us': {'id': 'HPND-UC-export-US', 'deprecated': False},
    'htmltidy': {'id': 'HTMLTIDY', 'deprecated': False},
    'ibm-pibs': {'id': 'IBM-pibs', 'deprecated': False},
    'icu': {'id': 'ICU', 'deprecated': False},
    'iec-code-components-eula': {'id': 'IEC-Code-Components-EULA', 'deprecated': False},
    'ijg': {'id': 'IJG', 'deprecated': False},
    'ijg-short': {'id': 'IJG-short', 'deprecated': False},
    'imagemagick': {'id': 'ImageMagick', 'deprecated': False},
    'imatix': {'id': 'iMatix', 'deprecated': False},
    'imlib2': {'id': 'Imlib2', 'deprecated': False},
    'info-zip': {'id': 'Info-ZIP', 'deprecated': False},
    'inner-net-2.0': {'id': 'Inner-Net-2.0', 'deprecated': False},
    'intel': {'id': 'Intel', 'deprecated': False},
    'intel-acpi': {'id': 'Intel-ACPI', 'deprecated': False},
    'interbase-1.0': {'id': 'Interbase-1.0', 'deprecated': False},
    'ipa': {'id': 'IPA', 'deprecated': False},
    'ipl-1.0': {'id': 'IPL-1.0', 'deprecated': False},
    'isc': {'id': 'ISC', 'deprecated': False},
    'isc-veillard': {'id': 'ISC-Veillard', 'deprecated': False},
    'jam': {'id': 'Jam', 'deprecated': False},
    'jasper-2.0': {'id': 'JasPer-2.0', 'deprecated': False},
    'jpl-image': {'id': 'JPL-image', 'deprecated': False},
    'jpnic': {'id': 'JPNIC', 'deprecated': False},
    'json': {'id': 'JSON', 'deprecated': False},
    'kastrup': {'id': 'Kastrup', 'deprecated': False},
    'kazlib': {'id': 'Kazlib', 'deprecated': False},
    'knuth-ctan': {'id': 'Knuth-CTAN', 'deprecated': False},
    'lal-1.2': {'id': 'LAL-1.2', 'deprecated': False},
    'lal-1.3': {'id': 'LAL-1.3', 'deprecated': False},
    'latex2e': {'id': 'Latex2e', 'deprecated': False},
    'latex2e-translated-notice': {'id': 'Latex2e-translated-notice', 'deprecated': False},
    'leptonica': {'id': 'Leptonica', 'deprecated': False},
    'lgpl-2.0': {'id': 'LGPL-2.0', 'deprecated': True},
    'lgpl-2.0+': {'id': 'LGPL-2.0+', 'deprecated': True},
    'lgpl-2.0-only': {'id': 'LGPL-2.0-only', 'deprecated': False},
    'lgpl-2.0-or-later': {'id': 'LGPL-2.0-or-later', 'deprecated': False},
    'lgpl-2.1': {'id': 'LGPL-2.1', 'deprecated': True},
    'lgpl-2.1+': {'id': 'LGPL-2.1+', 'deprecated': True},
    'lgpl-2.1-only': {'id': 'LGPL-2.1-only', 'deprecated': False},
    'lgpl-2.1-or-later': {'id': 'LGPL-2.1-or-later', 'deprecated': False},
    'lgpl-3.0': {'id': 'LGPL-3.0', 'deprecated': True},
    'lgpl-3.0+': {'id': 'LGPL-3.0+', 'deprecated': True},
    'lgpl-3.0-only': {'id': 'LGPL-3.0-only', 'deprecated': False},
    'lgpl-3.0-or-later': {'id': 'LGPL-3.0-or-later', 'deprecated': False},
    'lgpllr': {'id': 'LGPLLR', 'deprecated': False},
    'libpng': {'id': 'Libpng', 'deprecated': False},
    'libpng-2.0': {'id': 'libpng-2.0', 'deprecated': False},
    'libselinux-1.0': {'id': 'libselinux-1.0', 'deprecated': False},
    'libtiff': {'id': 'libtiff', 'deprecated': False},
    'libutil-david-nugent': {'id': 'libutil-David-Nugent', 'deprecated': False},
    'liliq-p-1.1': {'id': 'LiLiQ-P-1.1', 'deprecated': False},
    'liliq-r-1.1': {'id': 'LiLiQ-R-1.1', 'deprecated': False},
    'liliq-rplus-1.1': {'id': 'LiLiQ-Rplus-1.1', 'deprecated': False},
    'linux-man-pages-1-para': {'id': 'Linux-man-pages-1-para', 'deprecated': False},
    'linux-man-pages-copyleft': {'id': 'Linux-man-pages-copyleft', 'deprecated': False},
    'linux-man-pages-copyleft-2-para': {'id': 'Linux-man-pages-copyleft-2-para', 'deprecated': False},
    'linux-man-pages-copyleft-var': {'id': 'Linux-man-pages-copyleft-var', 'deprecated': False},
    'linux-openib': {'id': 'Linux-OpenIB', 'deprecated': False},
    'loop': {'id': 'LOOP', 'deprecated': False},
    'lpd-document': {'id': 'LPD-document', 'deprecated': False},
    'lpl-1.0': {'id': 'LPL-1.0', 'deprecated': False},
    'lpl-1.02': {'id': 'LPL-1.02', 'deprecated': False},
    'lppl-1.0': {'id': 'LPPL-1.0', 'deprecated': False},
    'lppl-1.1': {'id': 'LPPL-1.1', 'deprecated': False},
    'lppl-1.2': {'id': 'LPPL-1.2', 'deprecated': False},
    'lppl-1.3a': {'id': 'LPPL-1.3a', 'deprecated': False},
    'lppl-1.3c': {'id': 'LPPL-1.3c', 'deprecated': False},
    'lsof': {'id': 'lsof', 'deprecated': False},
    'lucida-bitmap-fonts': {'id': 'Lucida-Bitmap-Fonts', 'deprecated': False},
    'lzma-sdk-9.11-to-9.20': {'id': 'LZMA-SDK-9.11-to-9.20', 'deprecated': False},
    'lzma-sdk-9.22': {'id': 'LZMA-SDK-9.22', 'deprecated': False},
    'mackerras-3-clause': {'id': 'Mackerras-3-Clause', 'deprecated': False},
    'mackerras-3-clause-acknowledgment': {'id': 'Mackerras-3-Clause-acknowledgment', 'deprecated': False},
    'magaz': {'id': 'magaz', 'deprecated': False},
    'mailprio': {'id': 'mailprio', 'deprecated': False},
    'makeindex': {'id': 'MakeIndex', 'deprecated': False},
    'martin-birgmeier': {'id': 'Martin-Birgmeier', 'deprecated': False},
    'mcphee-slideshow': {'id': 'McPhee-slideshow', 'deprecated': False},
    'metamail': {'id': 'metamail', 'deprecated': False},
    'minpack': {'id': 'Minpack', 'deprecated': False},
    'miros': {'id': 'MirOS', 'deprecated': False},
    'mit': {'id': 'MIT', 'deprecated': False},
    'mit-0': {'id': 'MIT-0', 'deprecated': False},
    'mit-advertising': {'id': 'MIT-advertising', 'deprecated': False},
    'mit-cmu': {'id': 'MIT-CMU', 'deprecated': False},
    'mit-enna': {'id': 'MIT-enna', 'deprecated': False},
    'mit-feh': {'id': 'MIT-feh', 'deprecated': False},
    'mit-festival': {'id': 'MIT-Festival', 'deprecated': False},
    'mit-khronos-old': {'id': 'MIT-Khronos-old', 'deprecated': False},
    'mit-modern-variant': {'id': 'MIT-Modern-Variant', 'deprecated': False},
    'mit-open-group': {'id': 'MIT-open-group', 'deprecated': False},
    'mit-testregex': {'id': 'MIT-testregex', 'deprecated': False},
    'mit-wu': {'id': 'MIT-Wu', 'deprecated': False},
    'mitnfa': {'id': 'MITNFA', 'deprecated': False},
    'mmixware': {'id': 'MMIXware', 'deprecated': False},
    'motosoto': {'id': 'Motosoto', 'deprecated': False},
    'mpeg-ssg': {'id': 'MPEG-SSG', 'deprecated': False},
    'mpi-permissive': {'id': 'mpi-permissive', 'deprecated': False},
    'mpich2': {'id': 'mpich2', 'deprecated': False},
    'mpl-1.0': {'id': 'MPL-1.0', 'deprecated': False},
    'mpl-1.1': {'id': 'MPL-1.1', 'deprecated': False},
    'mpl-2.0': {'id': 'MPL-2.0', 'deprecated': False},
    'mpl-2.0-no-copyleft-exception': {'id': 'MPL-2.0-no-copyleft-exception', 'deprecated': False},
    'mplus': {'id': 'mplus', 'deprecated': False},
    'ms-lpl': {'id': 'MS-LPL', 'deprecated': False},
    'ms-pl': {'id': 'MS-PL', 'deprecated': False},
    'ms-rl': {'id': 'MS-RL', 'deprecated': False},
    'mtll': {'id': 'MTLL', 'deprecated': False},
    'mulanpsl-1.0': {'id': 'MulanPSL-1.0', 'deprecated': False},
    'mulanpsl-2.0': {'id': 'MulanPSL-2.0', 'deprecated': False},
    'multics': {'id': 'Multics', 'deprecated': False},
    'mup': {'id': 'Mup', 'deprecated': False},
    'naist-2003': {'id': 'NAIST-2003', 'deprecated': False},
    'nasa-1.3': {'id': 'NASA-1.3', 'deprecated': False},
    'naumen': {'id': 'Naumen', 'deprecated': False},
    'nbpl-1.0': {'id': 'NBPL-1.0', 'deprecated': False},
    'ncbi-pd': {'id': 'NCBI-PD', 'deprecated': False},
    'ncgl-uk-2.0': {'id': 'NCGL-UK-2.0', 'deprecated': False},
    'ncl': {'id': 'NCL', 'deprecated': False},
    'ncsa': {'id': 'NCSA', 'deprecated': False},
    'net-snmp': {'id': 'Net-SNMP', 'deprecated': True},
    'netcdf': {'id': 'NetCDF', 'deprecated': False},
    'newsletr': {'id': 'Newsletr', 'deprecated': False},
    'ngpl': {'id': 'NGPL', 'deprecated': False},
    'nicta-1.0': {'id': 'NICTA-1.0', 'deprecated': False},
    'nist-pd': {'id': 'NIST-PD', 'deprecated': False},
    'nist-pd-fallback': {'id': 'NIST-PD-fallback', 'deprecated': False},
    'nist-software': {'id': 'NIST-Software', 'deprecated': False},
    'nlod-1.0': {'id': 'NLOD-1.0', 'deprecated': False},
    'nlod-2.0': {'id': 'NLOD-2.0', 'deprecated': False},
    'nlpl': {'id': 'NLPL', 'deprecated': False},
    'nokia': {'id': 'Nokia', 'deprecated': False},
    'nosl': {'id': 'NOSL', 'deprecated': False},
    'noweb': {'id': 'Noweb', 'deprecated': False},
    'npl-1.0': {'id': 'NPL-1.0', 'deprecated': False},
    'npl-1.1': {'id': 'NPL-1.1', 'deprecated': False},
    'nposl-3.0': {'id': 'NPOSL-3.0', 'deprecated': False},
    'nrl': {'id': 'NRL', 'deprecated': False},
    'ntp': {'id': 'NTP', 'deprecated': False},
    'ntp-0': {'id': 'NTP-0', 'deprecated': False},
    'nunit': {'id': 'Nunit', 'deprecated': True},
    'o-uda-1.0': {'id': 'O-UDA-1.0', 'deprecated': False},
    'oar': {'id': 'OAR', 'deprecated': False},
    'occt-pl': {'id': 'OCCT-PL', 'deprecated': False},
    'oclc-2.0': {'id': 'OCLC-2.0', 'deprecated': False},
    'odbl-1.0': {'id': 'ODbL-1.0', 'deprecated': False},
    'odc-by-1.0': {'id': 'ODC-By-1.0', 'deprecated': False},
    'offis': {'id': 'OFFIS', 'deprecated': False},
    'ofl-1.0': {'id': 'OFL-1.0', 'deprecated': False},
    'ofl-1.0-no-rfn': {'id': 'OFL-1.0-no-RFN', 'deprecated': False},
    'ofl-1.0-rfn': {'id': 'OFL-1.0-RFN', 'deprecated': False},
    'ofl-1.1': {'id': 'OFL-1.1', 'deprecated': False},
    'ofl-1.1-no-rfn': {'id': 'OFL-1.1-no-RFN', 'deprecated': False},
    'ofl-1.1-rfn': {'id': 'OFL-1.1-RFN', 'deprecated': False},
    'ogc-1.0': {'id': 'OGC-1.0', 'deprecated': False},
    'ogdl-taiwan-1.0': {'id': 'OGDL-Taiwan-1.0', 'deprecated': False},
    'ogl-canada-2.0': {'id': 'OGL-Canada-2.0', 'deprecated': False},
    'ogl-uk-1.0': {'id': 'OGL-UK-1.0', 'deprecated': False},
    'ogl-uk-2.0': {'id': 'OGL-UK-2.0', 'deprecated': False},
    'ogl-uk-3.0': {'id': 'OGL-UK-3.0', 'deprecated': False},
    'ogtsl': {'id': 'OGTSL', 'deprecated': False},
    'oldap-1.1': {'id': 'OLDAP-1.1', 'deprecated': False},
    'oldap-1.2': {'id': 'OLDAP-1.2', 'deprecated': False},
    'oldap-1.3': {'id': 'OLDAP-1.3', 'deprecated': False},
    'oldap-1.4': {'id': 'OLDAP-1.4', 'deprecated': False},
    'oldap-2.0': {'id': 'OLDAP-2.0', 'deprecated': False},
    'oldap-2.0.1': {'id': 'OLDAP-2.0.1', 'deprecated': False},
    'oldap-2.1': {'id': 'OLDAP-2.1', 'deprecated': False},
    'oldap-2.2': {'id': 'OLDAP-2.2', 'deprecated': False},
    'oldap-2.2.1': {'id': 'OLDAP-2.2.1', 'deprecated': False},
    'oldap-2.2.2': {'id': 'OLDAP-2.2.2', 'deprecated': False},
    'oldap-2.3': {'id': 'OLDAP-2.3', 'deprecated': False},
    'oldap-2.4': {'id': 'OLDAP-2.4', 'deprecated': False},
    'oldap-2.5': {'id': 'OLDAP-2.5', 'deprecated': False},
    'oldap-2.6': {'id': 'OLDAP-2.6', 'deprecated': False},
    'oldap-2.7': {'id': 'OLDAP-2.7', 'deprecated': False},
    'oldap-2.8': {'id': 'OLDAP-2.8', 'deprecated': False},
    'olfl-1.3': {'id': 'OLFL-1.3', 'deprecated': False},
    'oml': {'id': 'OML', 'deprecated': False},
    'openpbs-2.3': {'id': 'OpenPBS-2.3', 'deprecated': False},
    'openssl': {'id': 'OpenSSL', 'deprecated': False},
    'openssl-standalone': {'id': 'OpenSSL-standalone', 'deprecated': False},
    'openvision': {'id': 'OpenVision', 'deprecated': False},
    'opl-1.0': {'id': 'OPL-1.0', 'deprecated': False},
    'opl-uk-3.0': {'id': 'OPL-UK-3.0', 'deprecated': False},
    'opubl-1.0': {'id': 'OPUBL-1.0', 'deprecated': False},
    'oset-pl-2.1': {'id': 'OSET-PL-2.1', 'deprecated': False},
    'osl-1.0': {'id': 'OSL-1.0', 'deprecated': False},
    'osl-1.1': {'id': 'OSL-1.1', 'deprecated': False},
    'osl-2.0': {'id': 'OSL-2.0', 'deprecated': False},
    'osl-2.1': {'id': 'OSL-2.1', 'deprecated': False},
    'osl-3.0': {'id': 'OSL-3.0', 'deprecated': False},
    'padl': {'id': 'PADL', 'deprecated': False},
    'parity-6.0.0': {'id': 'Parity-6.0.0', 'deprecated': False},
    'parity-7.0.0': {'id': 'Parity-7.0.0', 'deprecated': False},
    'pddl-1.0': {'id': 'PDDL-1.0', 'deprecated': False},
    'php-3.0': {'id': 'PHP-3.0', 'deprecated': False},
    'php-3.01': {'id': 'PHP-3.01', 'deprecated': False},
    'pixar': {'id': 'Pixar', 'deprecated': False},
    'pkgconf': {'id': 'pkgconf', 'deprecated': False},
    'plexus': {'id': 'Plexus', 'deprecated': False},
    'pnmstitch': {'id': 'pnmstitch', 'deprecated': False},
    'polyform-noncommercial-1.0.0': {'id': 'PolyForm-Noncommercial-1.0.0', 'deprecated': False},
    'polyform-small-business-1.0.0': {'id': 'PolyForm-Small-Business-1.0.0', 'deprecated': False},
    'postgresql': {'id': 'PostgreSQL', 'deprecated': False},
    'ppl': {'id': 'PPL', 'deprecated': False},
    'psf-2.0': {'id': 'PSF-2.0', 'deprecated': False},
    'psfrag': {'id': 'psfrag', 'deprecated': False},
    'psutils': {'id': 'psutils', 'deprecated': False},
    'python-2.0': {'id': 'Python-2.0', 'deprecated': False},
    'python-2.0.1': {'id': 'Python-2.0.1', 'deprecated': False},
    'python-ldap': {'id': 'python-ldap', 'deprecated': False},
    'qhull': {'id': 'Qhull', 'deprecated': False},
    'qpl-1.0': {'id': 'QPL-1.0', 'deprecated': False},
    'qpl-1.0-inria-2004': {'id': 'QPL-1.0-INRIA-2004', 'deprecated': False},
    'radvd': {'id': 'radvd', 'deprecated': False},
    'rdisc': {'id': 'Rdisc', 'deprecated': False},
    'rhecos-1.1': {'id': 'RHeCos-1.1', 'deprecated': False},
    'rpl-1.1': {'id': 'RPL-1.1', 'deprecated': False},
    'rpl-1.5': {'id': 'RPL-1.5', 'deprecated': False},
    'rpsl-1.0': {'id': 'RPSL-1.0', 'deprecated': False},
    'rsa-md': {'id': 'RSA-MD', 'deprecated': False},
    'rscpl': {'id': 'RSCPL', 'deprecated': False},
    'ruby': {'id': 'Ruby', 'deprecated': False},
    'ruby-pty': {'id': 'Ruby-pty', 'deprecated': False},
    'sax-pd': {'id': 'SAX-PD', 'deprecated': False},
    'sax-pd-2.0': {'id': 'SAX-PD-2.0', 'deprecated': False},
    'saxpath': {'id': 'Saxpath', 'deprecated': False},
    'scea': {'id': 'SCEA', 'deprecated': False},
    'schemereport': {'id': 'SchemeReport', 'deprecated': False},
    'sendmail': {'id': 'Sendmail', 'deprecated': False},
    'sendmail-8.23': {'id': 'Sendmail-8.23', 'deprecated': False},
    'sgi-b-1.0': {'id': 'SGI-B-1.0', 'deprecated': False},
    'sgi-b-1.1': {'id': 'SGI-B-1.1', 'deprecated': False},
    'sgi-b-2.0': {'id': 'SGI-B-2.0', 'deprecated': False},
    'sgi-opengl': {'id': 'SGI-OpenGL', 'deprecated': False},
    'sgp4': {'id': 'SGP4', 'deprecated': False},
    'shl-0.5': {'id': 'SHL-0.5', 'deprecated': False},
    'shl-0.51': {'id': 'SHL-0.51', 'deprecated': False},
    'simpl-2.0': {'id': 'SimPL-2.0', 'deprecated': False},
    'sissl': {'id': 'SISSL', 'deprecated': False},
    'sissl-1.2': {'id': 'SISSL-1.2', 'deprecated': False},
    'sl': {'id': 'SL', 'deprecated': False},
    'sleepycat': {'id': 'Sleepycat', 'deprecated': False},
    'smlnj': {'id': 'SMLNJ', 'deprecated': False},
    'smppl': {'id': 'SMPPL', 'deprecated': False},
    'snia': {'id': 'SNIA', 'deprecated': False},
    'snprintf': {'id': 'snprintf', 'deprecated': False},
    'softsurfer': {'id': 'softSurfer', 'deprecated': False},
    'soundex': {'id': 'Soundex', 'deprecated': False},
    'spencer-86': {'id': 'Spencer-86', 'deprecated': False},
    'spencer-94': {'id': 'Spencer-94', 'deprecated': False},
    'spencer-99': {'id': 'Spencer-99', 'deprecated': False},
    'spl-1.0': {'id': 'SPL-1.0', 'deprecated': False},
    'ssh-keyscan': {'id': 'ssh-keyscan', 'deprecated': False},
    'ssh-openssh': {'id': 'SSH-OpenSSH', 'deprecated': False},
    'ssh-short': {'id': 'SSH-short', 'deprecated': False},
    'ssleay-standalone': {'id': 'SSLeay-standalone', 'deprecated': False},
    'sspl-1.0': {'id': 'SSPL-1.0', 'deprecated': False},
    'standardml-nj': {'id': 'StandardML-NJ', 'deprecated': True},
    'sugarcrm-1.1.3': {'id': 'SugarCRM-1.1.3', 'deprecated': False},
    'sun-ppp': {'id': 'Sun-PPP', 'deprecated': False},
    'sun-ppp-2000': {'id': 'Sun-PPP-2000', 'deprecated': False},
    'sunpro': {'id': 'SunPro', 'deprecated': False},
    'swl': {'id': 'SWL', 'deprecated': False},
    'swrule': {'id': 'swrule', 'deprecated': False},
    'symlinks': {'id': 'Symlinks', 'deprecated': False},
    'tapr-ohl-1.0': {'id': 'TAPR-OHL-1.0', 'deprecated': False},
    'tcl': {'id': 'TCL', 'deprecated': False},
    'tcp-wrappers': {'id': 'TCP-wrappers', 'deprecated': False},
    'termreadkey': {'id': 'TermReadKey', 'deprecated': False},
    'tgppl-1.0': {'id': 'TGPPL-1.0', 'deprecated': False},
    'threeparttable': {'id': 'threeparttable', 'deprecated': False},
    'tmate': {'id': 'TMate', 'deprecated': False},
    'torque-1.1': {'id': 'TORQUE-1.1', 'deprecated': False},
    'tosl': {'id': 'TOSL', 'deprecated': False},
    'tpdl': {'id': 'TPDL', 'deprecated': False},
    'tpl-1.0': {'id': 'TPL-1.0', 'deprecated': False},
    'ttwl': {'id': 'TTWL', 'deprecated': False},
    'ttyp0': {'id': 'TTYP0', 'deprecated': False},
    'tu-berlin-1.0': {'id': 'TU-Berlin-1.0', 'deprecated': False},
    'tu-berlin-2.0': {'id': 'TU-Berlin-2.0', 'deprecated': False},
    'ubuntu-font-1.0': {'id': 'Ubuntu-font-1.0', 'deprecated': False},
    'ucar': {'id': 'UCAR', 'deprecated': False},
    'ucl-1.0': {'id': 'UCL-1.0', 'deprecated': False},
    'ulem': {'id': 'ulem', 'deprecated': False},
    'umich-merit': {'id': 'UMich-Merit', 'deprecated': False},
    'unicode-3.0': {'id': 'Unicode-3.0', 'deprecated': False},
    'unicode-dfs-2015': {'id': 'Unicode-DFS-2015', 'deprecated': False},
    'unicode-dfs-2016': {'id': 'Unicode-DFS-2016', 'deprecated': False},
    'unicode-tou': {'id': 'Unicode-TOU', 'deprecated': False},
    'unixcrypt': {'id': 'UnixCrypt', 'deprecated': False},
    'unlicense': {'id': 'Unlicense', 'deprecated': False},
    'upl-1.0': {'id': 'UPL-1.0', 'deprecated': False},
    'urt-rle': {'id': 'URT-RLE', 'deprecated': False},
    'vim': {'id': 'Vim', 'deprecated': False},
    'vostrom': {'id': 'VOSTROM', 'deprecated': False},
    'vsl-1.0': {'id': 'VSL-1.0', 'deprecated': False},
    'w3c': {'id': 'W3C', 'deprecated': False},
    'w3c-19980720': {'id': 'W3C-19980720', 'deprecated': False},
    'w3c-20150513': {'id': 'W3C-20150513', 'deprecated': False},
    'w3m': {'id': 'w3m', 'deprecated': False},
    'watcom-1.0': {'id': 'Watcom-1.0', 'deprecated': False},
    'widget-workshop': {'id': 'Widget-Workshop', 'deprecated': False},
    'wsuipa': {'id': 'Wsuipa', 'deprecated': False},
    'wtfpl': {'id': 'WTFPL', 'deprecated': False},
    'wxwindows': {'id': 'wxWindows', 'deprecated': True},
    'x11': {'id': 'X11', 'deprecated': False},
    'x11-distribute-modifications-variant': {'id': 'X11-distribute-modifications-variant', 'deprecated': False},
    'x11-swapped': {'id': 'X11-swapped', 'deprecated': False},
    'xdebug-1.03': {'id': 'Xdebug-1.03', 'deprecated': False},
    'xerox': {'id': 'Xerox', 'deprecated': False},
    'xfig': {'id': 'Xfig', 'deprecated': False},
    'xfree86-1.1': {'id': 'XFree86-1.1', 'deprecated': False},
    'xinetd': {'id': 'xinetd', 'deprecated': False},
    'xkeyboard-config-zinoviev': {'id': 'xkeyboard-config-Zinoviev', 'deprecated': False},
    'xlock': {'id': 'xlock', 'deprecated': False},
    'xnet': {'id': 'Xnet', 'deprecated': False},
    'xpp': {'id': 'xpp', 'deprecated': False},
    'xskat': {'id': 'XSkat', 'deprecated': False},
    'xzoom': {'id': 'xzoom', 'deprecated': False},
    'ypl-1.0': {'id': 'YPL-1.0', 'deprecated': False},
    'ypl-1.1': {'id': 'YPL-1.1', 'deprecated': False},
    'zed': {'id': 'Zed', 'deprecated': False},
    'zeeff': {'id': 'Zeeff', 'deprecated': False},
    'zend-2.0': {'id': 'Zend-2.0', 'deprecated': False},
    'zimbra-1.3': {'id': 'Zimbra-1.3', 'deprecated': False},
    'zimbra-1.4': {'id': 'Zimbra-1.4', 'deprecated': False},
    'zlib': {'id': 'Zlib', 'deprecated': False},
    'zlib-acknowledgement': {'id': 'zlib-acknowledgement', 'deprecated': False},
    'zpl-1.1': {'id': 'ZPL-1.1', 'deprecated': False},
    'zpl-2.0': {'id': 'ZPL-2.0', 'deprecated': False},
    'zpl-2.1': {'id': 'ZPL-2.1', 'deprecated': False},
}

EXCEPTIONS: dict[str, SPDXException] = {
    '389-exception': {'id': '389-exception', 'deprecated': False},
    'asterisk-exception': {'id': 'Asterisk-exception', 'deprecated': False},
    'asterisk-linking-protocols-exception': {'id': 'Asterisk-linking-protocols-exception', 'deprecated': False},
    'autoconf-exception-2.0': {'id': 'Autoconf-exception-2.0', 'deprecated': False},
    'autoconf-exception-3.0': {'id': 'Autoconf-exception-3.0', 'deprecated': False},
    'autoconf-exception-generic': {'id': 'Autoconf-exception-generic', 'deprecated': False},
    'autoconf-exception-generic-3.0': {'id': 'Autoconf-exception-generic-3.0', 'deprecated': False},
    'autoconf-exception-macro': {'id': 'Autoconf-exception-macro', 'deprecated': False},
    'bison-exception-1.24': {'id': 'Bison-exception-1.24', 'deprecated': False},
    'bison-exception-2.2': {'id': 'Bison-exception-2.2', 'deprecated': False},
    'bootloader-exception': {'id': 'Bootloader-exception', 'deprecated': False},
    'classpath-exception-2.0': {'id': 'Classpath-exception-2.0', 'deprecated': False},
    'clisp-exception-2.0': {'id': 'CLISP-exception-2.0', 'deprecated': False},
    'cryptsetup-openssl-exception': {'id': 'cryptsetup-OpenSSL-exception', 'deprecated': False},
    'digirule-foss-exception': {'id': 'DigiRule-FOSS-exception', 'deprecated': False},
    'ecos-exception-2.0': {'id': 'eCos-exception-2.0', 'deprecated': False},
    'erlang-otp-linking-exception': {'id': 'erlang-otp-linking-exception', 'deprecated': False},
    'fawkes-runtime-exception': {'id': 'Fawkes-Runtime-exception', 'deprecated': False},
    'fltk-exception': {'id': 'FLTK-exception', 'deprecated': False},
    'fmt-exception': {'id': 'fmt-exception', 'deprecated': False},
    'font-exception-2.0': {'id': 'Font-exception-2.0', 'deprecated': False},
    'freertos-exception-2.0': {'id': 'freertos-exception-2.0', 'deprecated': False},
    'gcc-exception-2.0': {'id': 'GCC-exception-2.0', 'deprecated': False},
    'gcc-exception-2.0-note': {'id': 'GCC-exception-2.0-note', 'deprecated': False},
    'gcc-exception-3.1': {'id': 'GCC-exception-3.1', 'deprecated': False},
    'gmsh-exception': {'id': 'Gmsh-exception', 'deprecated': False},
    'gnat-exception': {'id': 'GNAT-exception', 'deprecated': False},
    'gnome-examples-exception': {'id': 'GNOME-examples-exception', 'deprecated': False},
    'gnu-compiler-exception': {'id': 'GNU-compiler-exception', 'deprecated': False},
    'gnu-javamail-exception': {'id': 'gnu-javamail-exception', 'deprecated': False},
    'gpl-3.0-interface-exception': {'id': 'GPL-3.0-interface-exception', 'deprecated': False},
    'gpl-3.0-linking-exception': {'id': 'GPL-3.0-linking-exception', 'deprecated': False},
    'gpl-3.0-linking-source-exception': {'id': 'GPL-3.0-linking-source-exception', 'deprecated': False},
    'gpl-cc-1.0': {'id': 'GPL-CC-1.0', 'deprecated': False},
    'gstreamer-exception-2005': {'id': 'GStreamer-exception-2005', 'deprecated': False},
    'gstreamer-exception-2008': {'id': 'GStreamer-exception-2008', 'deprecated': False},
    'i2p-gpl-java-exception': {'id': 'i2p-gpl-java-exception', 'deprecated': False},
    'kicad-libraries-exception': {'id': 'KiCad-libraries-exception', 'deprecated': False},
    'lgpl-3.0-linking-exception': {'id': 'LGPL-3.0-linking-exception', 'deprecated': False},
    'libpri-openh323-exception': {'id': 'libpri-OpenH323-exception', 'deprecated': False},
    'libtool-exception': {'id': 'Libtool-exception', 'deprecated': False},
    'linux-syscall-note': {'id': 'Linux-syscall-note', 'deprecated': False},
    'llgpl': {'id': 'LLGPL', 'deprecated': False},
    'llvm-exception': {'id': 'LLVM-exception', 'deprecated': False},
    'lzma-exception': {'id': 'LZMA-exception', 'deprecated': False},
    'mif-exception': {'id': 'mif-exception', 'deprecated': False},
    'nokia-qt-exception-1.1': {'id': 'Nokia-Qt-exception-1.1', 'deprecated': True},
    'ocaml-lgpl-linking-exception': {'id': 'OCaml-LGPL-linking-exception', 'deprecated': False},
    'occt-exception-1.0': {'id': 'OCCT-exception-1.0', 'deprecated': False},
    'openjdk-assembly-exception-1.0': {'id': 'OpenJDK-assembly-exception-1.0', 'deprecated': False},
    'openvpn-openssl-exception': {'id': 'openvpn-openssl-exception', 'deprecated': False},
    'pcre2-exception': {'id': 'PCRE2-exception', 'deprecated': False},
    'ps-or-pdf-font-exception-20170817': {'id': 'PS-or-PDF-font-exception-20170817', 'deprecated': False},
    'qpl-1.0-inria-2004-exception': {'id': 'QPL-1.0-INRIA-2004-exception', 'deprecated': False},
    'qt-gpl-exception-1.0': {'id': 'Qt-GPL-exception-1.0', 'deprecated': False},
    'qt-lgpl-exception-1.1': {'id': 'Qt-LGPL-exception-1.1', 'deprecated': False},
    'qwt-exception-1.0': {'id': 'Qwt-exception-1.0', 'deprecated': False},
    'romic-exception': {'id': 'romic-exception', 'deprecated': False},
    'rrdtool-floss-exception-2.0': {'id': 'RRDtool-FLOSS-exception-2.0', 'deprecated': False},
    'sane-exception': {'id': 'SANE-exception', 'deprecated': False},
    'shl-2.0': {'id': 'SHL-2.0', 'deprecated': False},
    'shl-2.1': {'id': 'SHL-2.1', 'deprecated': False},
    'stunnel-exception': {'id': 'stunnel-exception', 'deprecated': False},
    'swi-exception': {'id': 'SWI-exception', 'deprecated': False},
    'swift-exception': {'id': 'Swift-exception', 'deprecated': False},
    'texinfo-exception': {'id': 'Texinfo-exception', 'deprecated': False},
    'u-boot-exception-2.0': {'id': 'u-boot-exception-2.0', 'deprecated': False},
    'ubdl-exception': {'id': 'UBDL-exception', 'deprecated': False},
    'universal-foss-exception-1.0': {'id': 'Universal-FOSS-exception-1.0', 'deprecated': False},
    'vsftpd-openssl-exception': {'id': 'vsftpd-openssl-exception', 'deprecated': False},
    'wxwindows-exception-3.1': {'id': 'WxWindows-exception-3.1', 'deprecated': False},
    'x11vnc-openssl-exception': {'id': 'x11vnc-openssl-exception', 'deprecated': False},
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_guest_mode.py ===
from __future__ import annotations

import asyncio
import contextlib
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from functools import partial
from math import inf
from typing import (
    TYPE_CHECKING,
    NoReturn,
    TypeVar,
    cast,
)

import pytest
import sniffio
from outcome import Outcome

import trio
import trio.testing
from trio.abc import Clock, Instrument

from .tutil import gc_collect_harder, restore_unraisablehook

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from trio._channel import MemorySendChannel

T = TypeVar("T")
InHost: TypeAlias = Callable[[Callable[[], object]], None]


# The simplest possible "host" loop.
# Nice features:
# - we can run code "outside" of trio using the schedule function passed to
#   our main
# - final result is returned
# - any unhandled exceptions cause an immediate crash
def trivial_guest_run(
    trio_fn: Callable[[InHost], Awaitable[T]],
    *,
    in_host_after_start: Callable[[], None] | None = None,
    host_uses_signal_set_wakeup_fd: bool = False,
    clock: Clock | None = None,
    instruments: Sequence[Instrument] = (),
    restrict_keyboard_interrupt_to_checkpoints: bool = False,
    strict_exception_groups: bool = True,
) -> T:
    todo: queue.Queue[tuple[str, Outcome[T] | Callable[[], object]]] = queue.Queue()

    host_thread = threading.current_thread()

    def run_sync_soon_threadsafe(fn: Callable[[], object]) -> None:
        nonlocal todo
        if host_thread is threading.current_thread():  # pragma: no cover
            crash = partial(
                pytest.fail,
                "run_sync_soon_threadsafe called from host thread",
            )
            todo.put(("run", crash))
        todo.put(("run", fn))

    def run_sync_soon_not_threadsafe(fn: Callable[[], object]) -> None:
        nonlocal todo
        if host_thread is not threading.current_thread():  # pragma: no cover
            crash = partial(
                pytest.fail,
                "run_sync_soon_not_threadsafe called from worker thread",
            )
            todo.put(("run", crash))
        todo.put(("run", fn))

    def done_callback(outcome: Outcome[T]) -> None:
        nonlocal todo
        todo.put(("unwrap", outcome))

    trio.lowlevel.start_guest_run(
        trio_fn,
        run_sync_soon_not_threadsafe,
        run_sync_soon_threadsafe=run_sync_soon_threadsafe,
        run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe,
        done_callback=done_callback,
        host_uses_signal_set_wakeup_fd=host_uses_signal_set_wakeup_fd,
        clock=clock,
        instruments=instruments,
        restrict_keyboard_interrupt_to_checkpoints=restrict_keyboard_interrupt_to_checkpoints,
        strict_exception_groups=strict_exception_groups,
    )
    if in_host_after_start is not None:
        in_host_after_start()

    try:
        while True:
            op, obj = todo.get()
            if op == "run":
                assert not isinstance(obj, Outcome)
                obj()
            elif op == "unwrap":
                assert isinstance(obj, Outcome)
                return obj.unwrap()
            else:  # pragma: no cover
                raise NotImplementedError(f"{op!r} not handled")
    finally:
        # Make sure that exceptions raised here don't capture these, so that
        # if an exception does cause us to abandon a run then the Trio state
        # has a chance to be GC'ed and warn about it.
        del todo, run_sync_soon_threadsafe, done_callback


def test_guest_trivial() -> None:
    async def trio_return(in_host: InHost) -> str:
        await trio.lowlevel.checkpoint()
        return "ok"

    assert trivial_guest_run(trio_return) == "ok"

    async def trio_fail(in_host: InHost) -> NoReturn:
        raise KeyError("whoopsiedaisy")

    with pytest.raises(KeyError, match="whoopsiedaisy"):
        trivial_guest_run(trio_fail)


def test_guest_can_do_io() -> None:
    async def trio_main(in_host: InHost) -> None:
        record = []
        a, b = trio.socket.socketpair()
        with a, b:
            async with trio.open_nursery() as nursery:

                async def do_receive() -> None:
                    record.append(await a.recv(1))

                nursery.start_soon(do_receive)
                await trio.testing.wait_all_tasks_blocked()

                await b.send(b"x")

        assert record == [b"x"]

    trivial_guest_run(trio_main)


def test_guest_is_initialized_when_start_returns() -> None:
    trio_token = None
    record = []

    async def trio_main(in_host: InHost) -> str:
        record.append("main task ran")
        await trio.lowlevel.checkpoint()
        assert trio.lowlevel.current_trio_token() is trio_token
        return "ok"

    def after_start() -> None:
        # We should get control back before the main task executes any code
        assert record == []

        nonlocal trio_token
        trio_token = trio.lowlevel.current_trio_token()
        trio_token.run_sync_soon(record.append, "run_sync_soon cb ran")

        @trio.lowlevel.spawn_system_task
        async def early_task() -> None:
            record.append("system task ran")
            await trio.lowlevel.checkpoint()

    res = trivial_guest_run(trio_main, in_host_after_start=after_start)
    assert res == "ok"
    assert set(record) == {"system task ran", "main task ran", "run_sync_soon cb ran"}

    class BadClock(Clock):
        def start_clock(self) -> NoReturn:
            raise ValueError("whoops")

        def current_time(self) -> float:
            raise NotImplementedError()

        def deadline_to_sleep_time(self, deadline: float) -> float:
            raise NotImplementedError()

    def after_start_never_runs() -> None:  # pragma: no cover
        pytest.fail("shouldn't get here")

    # Errors during initialization (which can only be TrioInternalErrors)
    # are raised out of start_guest_run, not out of the done_callback
    with pytest.raises(trio.TrioInternalError):
        trivial_guest_run(
            trio_main,
            clock=BadClock(),
            in_host_after_start=after_start_never_runs,
        )


def test_host_can_directly_wake_trio_task() -> None:
    async def trio_main(in_host: InHost) -> str:
        ev = trio.Event()
        in_host(ev.set)
        await ev.wait()
        return "ok"

    assert trivial_guest_run(trio_main) == "ok"


def test_host_altering_deadlines_wakes_trio_up() -> None:
    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:
        with trio.CancelScope() as cscope:
            in_host(lambda: set_deadline(cscope, -inf))
            await trio.sleep_forever()
        assert cscope.cancelled_caught

        with trio.CancelScope() as cscope:
            # also do a change that doesn't affect the next deadline, just to
            # exercise that path
            in_host(lambda: set_deadline(cscope, 1e6))
            in_host(lambda: set_deadline(cscope, -inf))
            await trio.sleep(999)
        assert cscope.cancelled_caught

        return "ok"

    assert trivial_guest_run(trio_main) == "ok"


def test_guest_mode_sniffio_integration() -> None:
    current_async_library = sniffio.current_async_library
    sniffio_library = sniffio.thread_local

    async def trio_main(in_host: InHost) -> str:
        async def synchronize() -> None:
            """Wait for all in_host() calls issued so far to complete."""
            evt = trio.Event()
            in_host(evt.set)
            await evt.wait()

        # Host and guest have separate sniffio_library contexts
        in_host(partial(setattr, sniffio_library, "name", "nullio"))
        await synchronize()
        assert current_async_library() == "trio"

        record = []
        in_host(lambda: record.append(current_async_library()))
        await synchronize()
        assert record == ["nullio"]
        assert current_async_library() == "trio"

        return "ok"

    try:
        assert trivial_guest_run(trio_main) == "ok"
    finally:
        sniffio_library.name = None


def test_guest_mode_trio_context_detection() -> None:
    def check(thing: bool) -> None:
        assert thing

    assert not trio.lowlevel.in_trio_run()
    assert not trio.lowlevel.in_trio_task()

    async def trio_main(in_host: InHost) -> None:
        for _ in range(2):
            assert trio.lowlevel.in_trio_run()
            assert trio.lowlevel.in_trio_task()

            in_host(lambda: check(trio.lowlevel.in_trio_run()))
            in_host(lambda: check(not trio.lowlevel.in_trio_task()))

    trivial_guest_run(trio_main)
    assert not trio.lowlevel.in_trio_run()
    assert not trio.lowlevel.in_trio_task()


def test_warn_set_wakeup_fd_overwrite() -> None:
    assert signal.set_wakeup_fd(-1) == -1

    async def trio_main(in_host: InHost) -> str:
        return "ok"

    a, b = socket.socketpair()
    with a, b:
        a.setblocking(False)

        # Warn if there's already a wakeup fd
        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match="signal handling code.*collided"):
                assert trivial_guest_run(trio_main) == "ok"
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()

        signal.set_wakeup_fd(a.fileno())
        try:
            with pytest.warns(RuntimeWarning, match="signal handling code.*collided"):
                assert (
                    trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=False)
                    == "ok"
                )
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()

        # Don't warn if there isn't already a wakeup fd
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert trivial_guest_run(trio_main) == "ok"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                trivial_guest_run(trio_main, host_uses_signal_set_wakeup_fd=True)
                == "ok"
            )

        # If there's already a wakeup fd, but we've been told to trust it,
        # then it's left alone and there's no warning
        signal.set_wakeup_fd(a.fileno())
        try:

            async def trio_check_wakeup_fd_unaltered(in_host: InHost) -> str:
                fd = signal.set_wakeup_fd(-1)
                assert fd == a.fileno()
                signal.set_wakeup_fd(fd)
                return "ok"

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert (
                    trivial_guest_run(
                        trio_check_wakeup_fd_unaltered,
                        host_uses_signal_set_wakeup_fd=True,
                    )
                    == "ok"
                )
        finally:
            assert signal.set_wakeup_fd(-1) == a.fileno()


def test_host_wakeup_doesnt_trigger_wait_all_tasks_blocked() -> None:
    # This is designed to hit the branch in unrolled_run where:
    #   idle_primed=True
    #   runner.runq is empty
    #   events is Truth-y
    # ...and confirm that in this case, wait_all_tasks_blocked does not get
    # triggered.
    def set_deadline(cscope: trio.CancelScope, new_deadline: float) -> None:
        print(f"setting deadline {new_deadline}")
        cscope.deadline = new_deadline

    async def trio_main(in_host: InHost) -> str:
        async def sit_in_wait_all_tasks_blocked(watb_cscope: trio.CancelScope) -> None:
            with watb_cscope:
                # Overall point of this test is that this
                # wait_all_tasks_blocked should *not* return normally, but
                # only by cancellation.
                await trio.testing.wait_all_tasks_blocked(cushion=9999)
                raise AssertionError(  # pragma: no cover
                    "wait_all_tasks_blocked should *not* return normally, "
                    "only by cancellation.",
                )
            assert watb_cscope.cancelled_caught

        async def get_woken_by_host_deadline(watb_cscope: trio.CancelScope) -> None:
            with trio.CancelScope() as cscope:
                print("scheduling stuff to happen")

                # Altering the deadline from the host, to something in the
                # future, will cause the run loop to wake up, but then
                # discover that there is nothing to do and go back to sleep.
                # This should *not* trigger wait_all_tasks_blocked.
                #
                # So the 'before_io_wait' here will wait until we're blocking
                # with the wait_all_tasks_blocked primed, and then schedule a
                # deadline change. The critical test is that this should *not*
                # wake up 'sit_in_wait_all_tasks_blocked'.
                #
                # The after we've had a chance to wake up
                # 'sit_in_wait_all_tasks_blocked', we want the test to
                # actually end. So in after_io_wait we schedule a second host
                # call to tear things down.
                class InstrumentHelper(Instrument):
                    def __init__(self) -> None:
                        self.primed = False

                    def before_io_wait(self, timeout: float) -> None:
                        print(f"before_io_wait({timeout})")
                        if timeout == 9999:  # pragma: no branch
                            assert not self.primed
                            in_host(lambda: set_deadline(cscope, 1e9))
                            self.primed = True

                    def after_io_wait(self, timeout: float) -> None:
                        if self.primed:  # pragma: no branch
                            print("instrument triggered")
                            in_host(lambda: cscope.cancel())
                            trio.lowlevel.remove_instrument(self)

                trio.lowlevel.add_instrument(InstrumentHelper())
                await trio.sleep_forever()
            assert cscope.cancelled_caught
            watb_cscope.cancel()

        async with trio.open_nursery() as nursery:
            watb_cscope = trio.CancelScope()
            nursery.start_soon(sit_in_wait_all_tasks_blocked, watb_cscope)
            await trio.testing.wait_all_tasks_blocked()
            nursery.start_soon(get_woken_by_host_deadline, watb_cscope)

        return "ok"

    assert trivial_guest_run(trio_main) == "ok"


@restore_unraisablehook()
def test_guest_warns_if_abandoned() -> None:
    # This warning is emitted from the garbage collector. So we have to make
    # sure that our abandoned run is garbage. The easiest way to do this is to
    # put it into a function, so that we're sure all the local state,
    # traceback frames, etc. are garbage once it returns.
    def do_abandoned_guest_run() -> None:
        async def abandoned_main(in_host: InHost) -> None:
            in_host(lambda: 1 / 0)
            while True:
                await trio.lowlevel.checkpoint()

        with pytest.raises(ZeroDivisionError):
            trivial_guest_run(abandoned_main)

    with pytest.warns(  # noqa: PT031
        RuntimeWarning,
        match="Trio guest run got abandoned",
    ):
        do_abandoned_guest_run()
        gc_collect_harder()

    # If you have problems some day figuring out what's holding onto a
    # reference to the unrolled_run generator and making this test fail,
    # then this might be useful to help track it down. (It assumes you
    # also hack start_guest_run so that it does 'global W; W =
    # weakref(unrolled_run_gen)'.)
    #
    # import gc
    # print(trio._core._run.W)
    # targets = [trio._core._run.W()]
    # for i in range(15):
    #     new_targets = []
    #     for target in targets:
    #         new_targets += gc.get_referrers(target)
    #         new_targets.remove(targets)
    #     print("#####################")
    #     print(f"depth {i}: {len(new_targets)}")
    #     print(new_targets)
    #     targets = new_targets

    with pytest.raises(RuntimeError):
        trio.current_time()


def aiotrio_run(
    trio_fn: Callable[[], Awaitable[T]],
    *,
    pass_not_threadsafe: bool = True,
    run_sync_soon_not_threadsafe: InHost | None = None,
    host_uses_signal_set_wakeup_fd: bool = False,
    clock: Clock | None = None,
    instruments: Sequence[Instrument] = (),
    restrict_keyboard_interrupt_to_checkpoints: bool = False,
    strict_exception_groups: bool = True,
) -> T:
    loop = asyncio.new_event_loop()

    async def aio_main() -> T:
        nonlocal run_sync_soon_not_threadsafe
        trio_done_fut: asyncio.Future[Outcome[T]] = loop.create_future()

        def trio_done_callback(main_outcome: Outcome[T]) -> None:
            print(f"trio_fn finished: {main_outcome!r}")
            trio_done_fut.set_result(main_outcome)

        if pass_not_threadsafe:
            run_sync_soon_not_threadsafe = cast("InHost", loop.call_soon)

        trio.lowlevel.start_guest_run(
            trio_fn,
            run_sync_soon_threadsafe=loop.call_soon_threadsafe,
            done_callback=trio_done_callback,
            run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe,
            host_uses_signal_set_wakeup_fd=host_uses_signal_set_wakeup_fd,
            clock=clock,
            instruments=instruments,
            restrict_keyboard_interrupt_to_checkpoints=restrict_keyboard_interrupt_to_checkpoints,
            strict_exception_groups=strict_exception_groups,
        )

        return (await trio_done_fut).unwrap()

    try:
        # can't use asyncio.run because that fails on Windows (3.8, x64, with
        # Komodia LSP) and segfaults on Windows (3.9, x64, with Komodia LSP)
        return loop.run_until_complete(aio_main())
    finally:
        loop.close()


def test_guest_mode_on_asyncio() -> None:
    async def trio_main() -> str:
        print("trio_main!")

        to_trio, from_aio = trio.open_memory_channel[int](float("inf"))
        from_trio: asyncio.Queue[int] = asyncio.Queue()

        aio_task = asyncio.ensure_future(aio_pingpong(from_trio, to_trio))

        # Make sure we have at least one tick where we don't need to go into
        # the thread
        await trio.lowlevel.checkpoint()

        from_trio.put_nowait(0)

        async for n in from_aio:
            print(f"trio got: {n}")
            from_trio.put_nowait(n + 1)
            if n >= 10:
                aio_task.cancel()
                return "trio-main-done"

        raise AssertionError("should never be reached")  # pragma: no cover

    async def aio_pingpong(
        from_trio: asyncio.Queue[int],
        to_trio: MemorySendChannel[int],
    ) -> None:
        print("aio_pingpong!")

        try:
            while True:
                n = await from_trio.get()
                print(f"aio got: {n}")
                to_trio.send_nowait(n + 1)
        except asyncio.CancelledError:
            raise
        except:  # pragma: no cover
            traceback.print_exc()
            raise

    assert (
        aiotrio_run(
            trio_main,
            # Not all versions of asyncio we test on can actually be trusted,
            # but this test doesn't care about signal handling, and it's
            # easier to just avoid the warnings.
            host_uses_signal_set_wakeup_fd=True,
        )
        == "trio-main-done"
    )

    assert (
        aiotrio_run(
            trio_main,
            # Also check that passing only call_soon_threadsafe works, via the
            # fallback path where we use it for everything.
            pass_not_threadsafe=False,
            host_uses_signal_set_wakeup_fd=True,
        )
        == "trio-main-done"
    )


def test_guest_mode_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    with monkeypatch.context() as m:

        async def crash_in_run_loop(in_host: InHost) -> None:
            m.setattr("trio._core._run.GLOBAL_RUN_CONTEXT.runner.runq", "HI")
            await trio.sleep(1)

        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_run_loop)

    with monkeypatch.context() as m:

        async def crash_in_io(in_host: InHost) -> None:
            m.setattr("trio._core._run.TheIOManager.get_events", None)
            await trio.lowlevel.checkpoint()

        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_io)

    with monkeypatch.context() as m:

        async def crash_in_worker_thread_io(in_host: InHost) -> None:
            t = threading.current_thread()
            old_get_events = trio._core._run.TheIOManager.get_events

            def bad_get_events(
                self: trio._core._run.TheIOManager,
                timeout: float,
            ) -> trio._core._run.EventResult:
                if threading.current_thread() is not t:
                    raise ValueError("oh no!")
                else:
                    return old_get_events(self, timeout)

            m.setattr("trio._core._run.TheIOManager.get_events", bad_get_events)

            await trio.sleep(1)

        with pytest.raises(trio.TrioInternalError):
            trivial_guest_run(crash_in_worker_thread_io)

    gc_collect_harder()


def test_guest_mode_ki() -> None:
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    # Check SIGINT in Trio func and in host func
    async def trio_main(in_host: InHost) -> None:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)

        # Host SIGINT should get injected into Trio
        in_host(partial(signal.raise_signal, signal.SIGINT))
        await trio.sleep(10)

    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main)
    assert excinfo.value.__context__ is None
    # Signal handler should be restored properly on exit
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    # Also check chaining in the case where KI is injected after main exits
    final_exc = KeyError("whoa")

    async def trio_main_raising(in_host: InHost) -> NoReturn:
        in_host(partial(signal.raise_signal, signal.SIGINT))
        raise final_exc

    with pytest.raises(KeyboardInterrupt) as excinfo:
        trivial_guest_run(trio_main_raising)
    assert excinfo.value.__context__ is final_exc

    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_guest_mode_autojump_clock_threshold_changing() -> None:
    # This is super obscure and probably no-one will ever notice, but
    # technically mutating the MockClock.autojump_threshold from the host
    # should wake up the guest, so let's test it.

    clock = trio.testing.MockClock()

    DURATION = 120

    async def trio_main(in_host: InHost) -> None:
        assert trio.current_time() == 0
        in_host(lambda: setattr(clock, "autojump_threshold", 0))
        await trio.sleep(DURATION)
        assert trio.current_time() == DURATION

    start = time.monotonic()
    trivial_guest_run(trio_main, clock=clock)
    end = time.monotonic()
    # Should be basically instantaneous, but we'll leave a generous buffer to
    # account for any CI weirdness
    assert end - start < DURATION / 2


@restore_unraisablehook()
def test_guest_mode_asyncgens() -> None:
    record = set()

    async def agen(label: str) -> AsyncGenerator[int, None]:
        assert sniffio.current_async_library() == label
        try:
            yield 1
        finally:
            library = sniffio.current_async_library()
            with contextlib.suppress(trio.Cancelled):
                await sys.modules[library].sleep(0)
            record.add((label, library))

    async def iterate_in_aio() -> None:
        await agen("asyncio").asend(None)

    async def trio_main() -> None:
        task = asyncio.ensure_future(iterate_in_aio())
        done_evt = trio.Event()
        task.add_done_callback(lambda _: done_evt.set())
        with trio.fail_after(1):
            await done_evt.wait()

        await agen("trio").asend(None)

        gc_collect_harder()

    aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True)

    assert record == {("asyncio", "asyncio"), ("trio", "trio")}


@restore_unraisablehook()
def test_guest_mode_asyncgens_garbage_collection() -> None:
    record: set[tuple[str, str, bool]] = set()

    async def agen(label: str) -> AsyncGenerator[int, None]:
        class A:
            pass

        a = A()
        a_wr = weakref.ref(a)
        assert sniffio.current_async_library() == label
        try:
            yield 1
        finally:
            library = sniffio.current_async_library()
            with contextlib.suppress(trio.Cancelled):
                await sys.modules[library].sleep(0)

            del a
            if sys.implementation.name == "pypy":
                gc_collect_harder()

            record.add((label, library, a_wr() is None))

    async def iterate_in_aio() -> None:
        await agen("asyncio").asend(None)

    async def trio_main() -> None:
        task = asyncio.ensure_future(iterate_in_aio())
        done_evt = trio.Event()
        task.add_done_callback(lambda _: done_evt.set())
        with trio.fail_after(1):
            await done_evt.wait()

        await agen("trio").asend(None)

        gc_collect_harder()

    aiotrio_run(trio_main, host_uses_signal_set_wakeup_fd=True)

    assert record == {("asyncio", "asyncio", True), ("trio", "trio", True)}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\instrumentation.py ===
# orm/instrumentation.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Defines SQLAlchemy's system of class instrumentation.

This module is usually not directly visible to user applications, but
defines a large part of the ORM's interactivity.

instrumentation.py deals with registration of end-user classes
for state tracking.   It interacts closely with state.py
and attributes.py which establish per-instance and per-class-attribute
instrumentation, respectively.

The class instrumentation system can be customized on a per-class
or global basis using the :mod:`sqlalchemy.ext.instrumentation`
module, which provides the means to build and specify
alternate instrumentation forms.

.. versionchanged: 0.8
   The instrumentation extension system was moved out of the
   ORM and into the external :mod:`sqlalchemy.ext.instrumentation`
   package.  When that package is imported, it installs
   itself within sqlalchemy.orm so that its more comprehensive
   resolution mechanics take effect.

"""


from __future__ import annotations

from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol

if TYPE_CHECKING:
    from ._typing import _RegistryType
    from .attributes import AttributeImpl
    from .attributes import QueryableAttribute
    from .collections import _AdaptedCollectionProtocol
    from .collections import _CollectionFactoryType
    from .decl_base import _MapperConfig
    from .events import InstanceEvents
    from .mapper import Mapper
    from .state import InstanceState
    from ..event import dispatcher

_T = TypeVar("_T", bound=Any)
DEL_ATTR = util.symbol("DEL_ATTR")


class _ExpiredAttributeLoaderProto(Protocol):
    def __call__(
        self,
        state: state.InstanceState[Any],
        toload: Set[str],
        passive: base.PassiveFlag,
    ) -> None: ...


class _ManagerFactory(Protocol):
    def __call__(self, class_: Type[_O]) -> ClassManager[_O]: ...


class ClassManager(
    HasMemoized,
    Dict[str, "QueryableAttribute[Any]"],
    Generic[_O],
    EventTarget,
):
    """Tracks state information at the class level."""

    dispatch: dispatcher[ClassManager[_O]]

    MANAGER_ATTR = base.DEFAULT_MANAGER_ATTR
    STATE_ATTR = base.DEFAULT_STATE_ATTR

    _state_setter = staticmethod(util.attrsetter(STATE_ATTR))

    expired_attribute_loader: _ExpiredAttributeLoaderProto
    "previously known as deferred_scalar_loader"

    init_method: Optional[Callable[..., None]]
    original_init: Optional[Callable[..., None]] = None

    factory: Optional[_ManagerFactory]

    declarative_scan: Optional[weakref.ref[_MapperConfig]] = None

    registry: _RegistryType

    if not TYPE_CHECKING:
        # starts as None during setup
        registry = None

    class_: Type[_O]

    _bases: List[ClassManager[Any]]

    @property
    @util.deprecated(
        "1.4",
        message="The ClassManager.deferred_scalar_loader attribute is now "
        "named expired_attribute_loader",
    )
    def deferred_scalar_loader(self):
        return self.expired_attribute_loader

    @deferred_scalar_loader.setter
    @util.deprecated(
        "1.4",
        message="The ClassManager.deferred_scalar_loader attribute is now "
        "named expired_attribute_loader",
    )
    def deferred_scalar_loader(self, obj):
        self.expired_attribute_loader = obj

    def __init__(self, class_):
        self.class_ = class_
        self.info = {}
        self.new_init = None
        self.local_attrs = {}
        self.originals = {}
        self._finalized = False
        self.factory = None
        self.init_method = None

        self._bases = [
            mgr
            for mgr in cast(
                "List[Optional[ClassManager[Any]]]",
                [
                    opt_manager_of_class(base)
                    for base in self.class_.__bases__
                    if isinstance(base, type)
                ],
            )
            if mgr is not None
        ]

        for base_ in self._bases:
            self.update(base_)

        cast(
            "InstanceEvents", self.dispatch._events
        )._new_classmanager_instance(class_, self)

        for basecls in class_.__mro__:
            mgr = opt_manager_of_class(basecls)
            if mgr is not None:
                self.dispatch._update(mgr.dispatch)

        self.manage()

        if "__del__" in class_.__dict__:
            util.warn(
                "__del__() method on class %s will "
                "cause unreachable cycles and memory leaks, "
                "as SQLAlchemy instrumentation often creates "
                "reference cycles.  Please remove this method." % class_
            )

    def _update_state(
        self,
        finalize: bool = False,
        mapper: Optional[Mapper[_O]] = None,
        registry: Optional[_RegistryType] = None,
        declarative_scan: Optional[_MapperConfig] = None,
        expired_attribute_loader: Optional[
            _ExpiredAttributeLoaderProto
        ] = None,
        init_method: Optional[Callable[..., None]] = None,
    ) -> None:
        if mapper:
            self.mapper = mapper  #
        if registry:
            registry._add_manager(self)
        if declarative_scan:
            self.declarative_scan = weakref.ref(declarative_scan)
        if expired_attribute_loader:
            self.expired_attribute_loader = expired_attribute_loader

        if init_method:
            assert not self._finalized, (
                "class is already instrumented, "
                "init_method %s can't be applied" % init_method
            )
            self.init_method = init_method

        if not self._finalized:
            self.original_init = (
                self.init_method
                if self.init_method is not None
                and self.class_.__init__ is object.__init__
                else self.class_.__init__
            )

        if finalize and not self._finalized:
            self._finalize()

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        self._instrument_init()

        _instrumentation_factory.dispatch.class_instrument(self.class_)

    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return other is self

    @property
    def is_mapped(self) -> bool:
        return "mapper" in self.__dict__

    @HasMemoized.memoized_attribute
    def _all_key_set(self):
        return frozenset(self)

    @HasMemoized.memoized_attribute
    def _collection_impl_keys(self):
        return frozenset(
            [attr.key for attr in self.values() if attr.impl.collection]
        )

    @HasMemoized.memoized_attribute
    def _scalar_loader_impls(self):
        return frozenset(
            [
                attr.impl
                for attr in self.values()
                if attr.impl.accepts_scalar_loader
            ]
        )

    @HasMemoized.memoized_attribute
    def _loader_impls(self):
        return frozenset([attr.impl for attr in self.values()])

    @util.memoized_property
    def mapper(self) -> Mapper[_O]:
        # raises unless self.mapper has been assigned
        raise exc.UnmappedClassError(self.class_)

    def _all_sqla_attributes(self, exclude=None):
        """return an iterator of all classbound attributes that are
        implement :class:`.InspectionAttr`.

        This includes :class:`.QueryableAttribute` as well as extension
        types such as :class:`.hybrid_property` and
        :class:`.AssociationProxy`.

        """

        found: Dict[str, Any] = {}

        # constraints:
        # 1. yield keys in cls.__dict__ order
        # 2. if a subclass has the same key as a superclass, include that
        #    key as part of the ordering of the superclass, because an
        #    overridden key is usually installed by the mapper which is going
        #    on a different ordering
        # 3. don't use getattr() as this fires off descriptors

        for supercls in self.class_.__mro__[0:-1]:
            inherits = supercls.__mro__[1]
            for key in supercls.__dict__:
                found.setdefault(key, supercls)
                if key in inherits.__dict__:
                    continue
                val = found[key].__dict__[key]
                if (
                    isinstance(val, interfaces.InspectionAttr)
                    and val.is_attribute
                ):
                    yield key, val

    def _get_class_attr_mro(self, key, default=None):
        """return an attribute on the class without tripping it."""

        for supercls in self.class_.__mro__:
            if key in supercls.__dict__:
                return supercls.__dict__[key]
        else:
            return default

    def _attr_has_impl(self, key: str) -> bool:
        """Return True if the given attribute is fully initialized.

        i.e. has an impl.
        """

        return key in self and self[key].impl is not None

    def _subclass_manager(self, cls: Type[_T]) -> ClassManager[_T]:
        """Create a new ClassManager for a subclass of this ClassManager's
        class.

        This is called automatically when attributes are instrumented so that
        the attributes can be propagated to subclasses against their own
        class-local manager, without the need for mappers etc. to have already
        pre-configured managers for the full class hierarchy.   Mappers
        can post-configure the auto-generated ClassManager when needed.

        """
        return register_class(cls, finalize=False)

    def _instrument_init(self):
        self.new_init = _generate_init(self.class_, self, self.original_init)
        self.install_member("__init__", self.new_init)

    @util.memoized_property
    def _state_constructor(self) -> Type[state.InstanceState[_O]]:
        self.dispatch.first_init(self, self.class_)
        return state.InstanceState

    def manage(self):
        """Mark this instance as the manager for its class."""

        setattr(self.class_, self.MANAGER_ATTR, self)

    @util.hybridmethod
    def manager_getter(self):
        return _default_manager_getter

    @util.hybridmethod
    def state_getter(self):
        """Return a (instance) -> InstanceState callable.

        "state getter" callables should raise either KeyError or
        AttributeError if no InstanceState could be found for the
        instance.
        """

        return _default_state_getter

    @util.hybridmethod
    def dict_getter(self):
        return _default_dict_getter

    def instrument_attribute(
        self,
        key: str,
        inst: QueryableAttribute[Any],
        propagated: bool = False,
    ) -> None:
        if propagated:
            if key in self.local_attrs:
                return  # don't override local attr with inherited attr
        else:
            self.local_attrs[key] = inst
            self.install_descriptor(key, inst)
        self._reset_memoizations()
        self[key] = inst

        for cls in self.class_.__subclasses__():
            manager = self._subclass_manager(cls)
            manager.instrument_attribute(key, inst, True)

    def subclass_managers(self, recursive):
        for cls in self.class_.__subclasses__():
            mgr = opt_manager_of_class(cls)
            if mgr is not None and mgr is not self:
                yield mgr
                if recursive:
                    yield from mgr.subclass_managers(True)

    def post_configure_attribute(self, key):
        _instrumentation_factory.dispatch.attribute_instrument(
            self.class_, key, self[key]
        )

    def uninstrument_attribute(self, key, propagated=False):
        if key not in self:
            return
        if propagated:
            if key in self.local_attrs:
                return  # don't get rid of local attr
        else:
            del self.local_attrs[key]
            self.uninstall_descriptor(key)
        self._reset_memoizations()
        del self[key]
        for cls in self.class_.__subclasses__():
            manager = opt_manager_of_class(cls)
            if manager:
                manager.uninstrument_attribute(key, True)

    def unregister(self) -> None:
        """remove all instrumentation established by this ClassManager."""

        for key in list(self.originals):
            self.uninstall_member(key)

        self.mapper = None
        self.dispatch = None  # type: ignore
        self.new_init = None
        self.info.clear()

        for key in list(self):
            if key in self.local_attrs:
                self.uninstrument_attribute(key)

        if self.MANAGER_ATTR in self.class_.__dict__:
            delattr(self.class_, self.MANAGER_ATTR)

    def install_descriptor(
        self, key: str, inst: QueryableAttribute[Any]
    ) -> None:
        if key in (self.STATE_ATTR, self.MANAGER_ATTR):
            raise KeyError(
                "%r: requested attribute name conflicts with "
                "instrumentation attribute of the same name." % key
            )
        setattr(self.class_, key, inst)

    def uninstall_descriptor(self, key: str) -> None:
        delattr(self.class_, key)

    def install_member(self, key: str, implementation: Any) -> None:
        if key in (self.STATE_ATTR, self.MANAGER_ATTR):
            raise KeyError(
                "%r: requested attribute name conflicts with "
                "instrumentation attribute of the same name." % key
            )
        self.originals.setdefault(key, self.class_.__dict__.get(key, DEL_ATTR))
        setattr(self.class_, key, implementation)

    def uninstall_member(self, key: str) -> None:
        original = self.originals.pop(key, None)
        if original is not DEL_ATTR:
            setattr(self.class_, key, original)
        else:
            delattr(self.class_, key)

    def instrument_collection_class(
        self, key: str, collection_class: Type[Collection[Any]]
    ) -> _CollectionFactoryType:
        return collections.prepare_instrumentation(collection_class)

    def initialize_collection(
        self,
        key: str,
        state: InstanceState[_O],
        factory: _CollectionFactoryType,
    ) -> Tuple[collections.CollectionAdapter, _AdaptedCollectionProtocol]:
        user_data = factory()
        impl = self.get_impl(key)
        assert _is_collection_attribute_impl(impl)
        adapter = collections.CollectionAdapter(impl, state, user_data)
        return adapter, user_data

    def is_instrumented(self, key: str, search: bool = False) -> bool:
        if search:
            return key in self
        else:
            return key in self.local_attrs

    def get_impl(self, key: str) -> AttributeImpl:
        return self[key].impl

    @property
    def attributes(self) -> Iterable[Any]:
        return iter(self.values())

    # InstanceState management

    def new_instance(self, state: Optional[InstanceState[_O]] = None) -> _O:
        # here, we would prefer _O to be bound to "object"
        # so that mypy sees that __new__ is present.   currently
        # it's bound to Any as there were other problems not having
        # it that way but these can be revisited
        instance = self.class_.__new__(self.class_)
        if state is None:
            state = self._state_constructor(instance, self)
        self._state_setter(instance, state)
        return instance

    def setup_instance(
        self, instance: _O, state: Optional[InstanceState[_O]] = None
    ) -> None:
        if state is None:
            state = self._state_constructor(instance, self)
        self._state_setter(instance, state)

    def teardown_instance(self, instance: _O) -> None:
        delattr(instance, self.STATE_ATTR)

    def _serialize(
        self, state: InstanceState[_O], state_dict: Dict[str, Any]
    ) -> _SerializeManager:
        return _SerializeManager(state, state_dict)

    def _new_state_if_none(
        self, instance: _O
    ) -> Union[Literal[False], InstanceState[_O]]:
        """Install a default InstanceState if none is present.

        A private convenience method used by the __init__ decorator.

        """
        if hasattr(instance, self.STATE_ATTR):
            return False
        elif self.class_ is not instance.__class__ and self.is_mapped:
            # this will create a new ClassManager for the
            # subclass, without a mapper.  This is likely a
            # user error situation but allow the object
            # to be constructed, so that it is usable
            # in a non-ORM context at least.
            return self._subclass_manager(
                instance.__class__
            )._new_state_if_none(instance)
        else:
            state = self._state_constructor(instance, self)
            self._state_setter(instance, state)
            return state

    def has_state(self, instance: _O) -> bool:
        return hasattr(instance, self.STATE_ATTR)

    def has_parent(
        self, state: InstanceState[_O], key: str, optimistic: bool = False
    ) -> bool:
        """TODO"""
        return self.get_impl(key).hasparent(state, optimistic=optimistic)

    def __bool__(self) -> bool:
        """All ClassManagers are non-zero regardless of attribute state."""
        return True

    def __repr__(self) -> str:
        return "<%s of %r at %x>" % (
            self.__class__.__name__,
            self.class_,
            id(self),
        )


class _SerializeManager:
    """Provide serialization of a :class:`.ClassManager`.

    The :class:`.InstanceState` uses ``__init__()`` on serialize
    and ``__call__()`` on deserialize.

    """

    def __init__(self, state: state.InstanceState[Any], d: Dict[str, Any]):
        self.class_ = state.class_
        manager = state.manager
        manager.dispatch.pickle(state, d)

    def __call__(self, state, inst, state_dict):
        state.manager = manager = opt_manager_of_class(self.class_)
        if manager is None:
            raise exc.UnmappedInstanceError(
                inst,
                "Cannot deserialize object of type %r - "
                "no mapper() has "
                "been configured for this class within the current "
                "Python process!" % self.class_,
            )
        elif manager.is_mapped and not manager.mapper.configured:
            manager.mapper._check_configure()

        # setup _sa_instance_state ahead of time so that
        # unpickle events can access the object normally.
        # see [ticket:2362]
        if inst is not None:
            manager.setup_instance(inst, state)
        manager.dispatch.unpickle(state, state_dict)


class InstrumentationFactory(EventTarget):
    """Factory for new ClassManager instances."""

    dispatch: dispatcher[InstrumentationFactory]

    def create_manager_for_cls(self, class_: Type[_O]) -> ClassManager[_O]:
        assert class_ is not None
        assert opt_manager_of_class(class_) is None

        # give a more complicated subclass
        # a chance to do what it wants here
        manager, factory = self._locate_extended_factory(class_)

        if factory is None:
            factory = ClassManager
            manager = ClassManager(class_)
        else:
            assert manager is not None

        self._check_conflicts(class_, factory)

        manager.factory = factory

        return manager

    def _locate_extended_factory(
        self, class_: Type[_O]
    ) -> Tuple[Optional[ClassManager[_O]], Optional[_ManagerFactory]]:
        """Overridden by a subclass to do an extended lookup."""
        return None, None

    def _check_conflicts(
        self, class_: Type[_O], factory: Callable[[Type[_O]], ClassManager[_O]]
    ) -> None:
        """Overridden by a subclass to test for conflicting factories."""

    def unregister(self, class_: Type[_O]) -> None:
        manager = manager_of_class(class_)
        manager.unregister()
        self.dispatch.class_uninstrument(class_)


# this attribute is replaced by sqlalchemy.ext.instrumentation
# when imported.
_instrumentation_factory = InstrumentationFactory()

# these attributes are replaced by sqlalchemy.ext.instrumentation
# when a non-standard InstrumentationManager class is first
# used to instrument a class.
instance_state = _default_state_getter = base.instance_state

instance_dict = _default_dict_getter = base.instance_dict

manager_of_class = _default_manager_getter = base.manager_of_class
opt_manager_of_class = _default_opt_manager_getter = base.opt_manager_of_class


def register_class(
    class_: Type[_O],
    finalize: bool = True,
    mapper: Optional[Mapper[_O]] = None,
    registry: Optional[_RegistryType] = None,
    declarative_scan: Optional[_MapperConfig] = None,
    expired_attribute_loader: Optional[_ExpiredAttributeLoaderProto] = None,
    init_method: Optional[Callable[..., None]] = None,
) -> ClassManager[_O]:
    """Register class instrumentation.

    Returns the existing or newly created class manager.

    """

    manager = opt_manager_of_class(class_)
    if manager is None:
        manager = _instrumentation_factory.create_manager_for_cls(class_)
    manager._update_state(
        mapper=mapper,
        registry=registry,
        declarative_scan=declarative_scan,
        expired_attribute_loader=expired_attribute_loader,
        init_method=init_method,
        finalize=finalize,
    )

    return manager


def unregister_class(class_):
    """Unregister class instrumentation."""

    _instrumentation_factory.unregister(class_)


def is_instrumented(instance, key):
    """Return True if the given attribute on the given instance is
    instrumented by the attributes package.

    This function may be used regardless of instrumentation
    applied directly to the class, i.e. no descriptors are required.

    """
    return manager_of_class(instance.__class__).is_instrumented(
        key, search=True
    )


def _generate_init(class_, class_manager, original_init):
    """Build an __init__ decorator that triggers ClassManager events."""

    # TODO: we should use the ClassManager's notion of the
    # original '__init__' method, once ClassManager is fixed
    # to always reference that.

    if original_init is None:
        original_init = class_.__init__

    # Go through some effort here and don't change the user's __init__
    # calling signature, including the unlikely case that it has
    # a return value.
    # FIXME: need to juggle local names to avoid constructor argument
    # clashes.
    func_body = """\
def __init__(%(apply_pos)s):
    new_state = class_manager._new_state_if_none(%(self_arg)s)
    if new_state:
        return new_state._initialize_instance(%(apply_kw)s)
    else:
        return original_init(%(apply_kw)s)
"""
    func_vars = util.format_argspec_init(original_init, grouped=False)
    func_text = func_body % func_vars

    func_defaults = getattr(original_init, "__defaults__", None)
    func_kw_defaults = getattr(original_init, "__kwdefaults__", None)

    env = locals().copy()
    env["__name__"] = __name__
    exec(func_text, env)
    __init__ = env["__init__"]
    __init__.__doc__ = original_init.__doc__
    __init__._sa_original_init = original_init

    if func_defaults:
        __init__.__defaults__ = func_defaults
    if func_kw_defaults:
        __init__.__kwdefaults__ = func_kw_defaults

    return __init__