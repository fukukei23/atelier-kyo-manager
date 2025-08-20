
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\parser\isoparser.py ===
# -*- coding: utf-8 -*-
"""
This module offers a parser for ISO-8601 strings

It is intended to support all valid date, time and datetime formats per the
ISO-8601 specification.

..versionadded:: 2.7.0
"""
from datetime import datetime, timedelta, time, date
import calendar
from dateutil import tz

from functools import wraps

import re
import six

__all__ = ["isoparse", "isoparser"]


def _takes_ascii(f):
    @wraps(f)
    def func(self, str_in, *args, **kwargs):
        # If it's a stream, read the whole thing
        str_in = getattr(str_in, 'read', lambda: str_in)()

        # If it's unicode, turn it into bytes, since ISO-8601 only covers ASCII
        if isinstance(str_in, six.text_type):
            # ASCII is the same in UTF-8
            try:
                str_in = str_in.encode('ascii')
            except UnicodeEncodeError as e:
                msg = 'ISO-8601 strings should contain only ASCII characters'
                six.raise_from(ValueError(msg), e)

        return f(self, str_in, *args, **kwargs)

    return func


class isoparser(object):
    def __init__(self, sep=None):
        """
        :param sep:
            A single character that separates date and time portions. If
            ``None``, the parser will accept any single character.
            For strict ISO-8601 adherence, pass ``'T'``.
        """
        if sep is not None:
            if (len(sep) != 1 or ord(sep) >= 128 or sep in '0123456789'):
                raise ValueError('Separator must be a single, non-numeric ' +
                                 'ASCII character')

            sep = sep.encode('ascii')

        self._sep = sep

    @_takes_ascii
    def isoparse(self, dt_str):
        """
        Parse an ISO-8601 datetime string into a :class:`datetime.datetime`.

        An ISO-8601 datetime string consists of a date portion, followed
        optionally by a time portion - the date and time portions are separated
        by a single character separator, which is ``T`` in the official
        standard. Incomplete date formats (such as ``YYYY-MM``) may *not* be
        combined with a time portion.

        Supported date formats are:

        Common:

        - ``YYYY``
        - ``YYYY-MM``
        - ``YYYY-MM-DD`` or ``YYYYMMDD``

        Uncommon:

        - ``YYYY-Www`` or ``YYYYWww`` - ISO week (day defaults to 0)
        - ``YYYY-Www-D`` or ``YYYYWwwD`` - ISO week and day

        The ISO week and day numbering follows the same logic as
        :func:`datetime.date.isocalendar`.

        Supported time formats are:

        - ``hh``
        - ``hh:mm`` or ``hhmm``
        - ``hh:mm:ss`` or ``hhmmss``
        - ``hh:mm:ss.ssssss`` (Up to 6 sub-second digits)

        Midnight is a special case for `hh`, as the standard supports both
        00:00 and 24:00 as a representation. The decimal separator can be
        either a dot or a comma.


        .. caution::

            Support for fractional components other than seconds is part of the
            ISO-8601 standard, but is not currently implemented in this parser.

        Supported time zone offset formats are:

        - `Z` (UTC)
        - `±HH:MM`
        - `±HHMM`
        - `±HH`

        Offsets will be represented as :class:`dateutil.tz.tzoffset` objects,
        with the exception of UTC, which will be represented as
        :class:`dateutil.tz.tzutc`. Time zone offsets equivalent to UTC (such
        as `+00:00`) will also be represented as :class:`dateutil.tz.tzutc`.

        :param dt_str:
            A string or stream containing only an ISO-8601 datetime string

        :return:
            Returns a :class:`datetime.datetime` representing the string.
            Unspecified components default to their lowest value.

        .. warning::

            As of version 2.7.0, the strictness of the parser should not be
            considered a stable part of the contract. Any valid ISO-8601 string
            that parses correctly with the default settings will continue to
            parse correctly in future versions, but invalid strings that
            currently fail (e.g. ``2017-01-01T00:00+00:00:00``) are not
            guaranteed to continue failing in future versions if they encode
            a valid date.

        .. versionadded:: 2.7.0
        """
        components, pos = self._parse_isodate(dt_str)

        if len(dt_str) > pos:
            if self._sep is None or dt_str[pos:pos + 1] == self._sep:
                components += self._parse_isotime(dt_str[pos + 1:])
            else:
                raise ValueError('String contains unknown ISO components')

        if len(components) > 3 and components[3] == 24:
            components[3] = 0
            return datetime(*components) + timedelta(days=1)

        return datetime(*components)

    @_takes_ascii
    def parse_isodate(self, datestr):
        """
        Parse the date portion of an ISO string.

        :param datestr:
            The string portion of an ISO string, without a separator

        :return:
            Returns a :class:`datetime.date` object
        """
        components, pos = self._parse_isodate(datestr)
        if pos < len(datestr):
            raise ValueError('String contains unknown ISO ' +
                             'components: {!r}'.format(datestr.decode('ascii')))
        return date(*components)

    @_takes_ascii
    def parse_isotime(self, timestr):
        """
        Parse the time portion of an ISO string.

        :param timestr:
            The time portion of an ISO string, without a separator

        :return:
            Returns a :class:`datetime.time` object
        """
        components = self._parse_isotime(timestr)
        if components[0] == 24:
            components[0] = 0
        return time(*components)

    @_takes_ascii
    def parse_tzstr(self, tzstr, zero_as_utc=True):
        """
        Parse a valid ISO time zone string.

        See :func:`isoparser.isoparse` for details on supported formats.

        :param tzstr:
            A string representing an ISO time zone offset

        :param zero_as_utc:
            Whether to return :class:`dateutil.tz.tzutc` for zero-offset zones

        :return:
            Returns :class:`dateutil.tz.tzoffset` for offsets and
            :class:`dateutil.tz.tzutc` for ``Z`` and (if ``zero_as_utc`` is
            specified) offsets equivalent to UTC.
        """
        return self._parse_tzstr(tzstr, zero_as_utc=zero_as_utc)

    # Constants
    _DATE_SEP = b'-'
    _TIME_SEP = b':'
    _FRACTION_REGEX = re.compile(b'[\\.,]([0-9]+)')

    def _parse_isodate(self, dt_str):
        try:
            return self._parse_isodate_common(dt_str)
        except ValueError:
            return self._parse_isodate_uncommon(dt_str)

    def _parse_isodate_common(self, dt_str):
        len_str = len(dt_str)
        components = [1, 1, 1]

        if len_str < 4:
            raise ValueError('ISO string too short')

        # Year
        components[0] = int(dt_str[0:4])
        pos = 4
        if pos >= len_str:
            return components, pos

        has_sep = dt_str[pos:pos + 1] == self._DATE_SEP
        if has_sep:
            pos += 1

        # Month
        if len_str - pos < 2:
            raise ValueError('Invalid common month')

        components[1] = int(dt_str[pos:pos + 2])
        pos += 2

        if pos >= len_str:
            if has_sep:
                return components, pos
            else:
                raise ValueError('Invalid ISO format')

        if has_sep:
            if dt_str[pos:pos + 1] != self._DATE_SEP:
                raise ValueError('Invalid separator in ISO string')
            pos += 1

        # Day
        if len_str - pos < 2:
            raise ValueError('Invalid common day')
        components[2] = int(dt_str[pos:pos + 2])
        return components, pos + 2

    def _parse_isodate_uncommon(self, dt_str):
        if len(dt_str) < 4:
            raise ValueError('ISO string too short')

        # All ISO formats start with the year
        year = int(dt_str[0:4])

        has_sep = dt_str[4:5] == self._DATE_SEP

        pos = 4 + has_sep       # Skip '-' if it's there
        if dt_str[pos:pos + 1] == b'W':
            # YYYY-?Www-?D?
            pos += 1
            weekno = int(dt_str[pos:pos + 2])
            pos += 2

            dayno = 1
            if len(dt_str) > pos:
                if (dt_str[pos:pos + 1] == self._DATE_SEP) != has_sep:
                    raise ValueError('Inconsistent use of dash separator')

                pos += has_sep

                dayno = int(dt_str[pos:pos + 1])
                pos += 1

            base_date = self._calculate_weekdate(year, weekno, dayno)
        else:
            # YYYYDDD or YYYY-DDD
            if len(dt_str) - pos < 3:
                raise ValueError('Invalid ordinal day')

            ordinal_day = int(dt_str[pos:pos + 3])
            pos += 3

            if ordinal_day < 1 or ordinal_day > (365 + calendar.isleap(year)):
                raise ValueError('Invalid ordinal day' +
                                 ' {} for year {}'.format(ordinal_day, year))

            base_date = date(year, 1, 1) + timedelta(days=ordinal_day - 1)

        components = [base_date.year, base_date.month, base_date.day]
        return components, pos

    def _calculate_weekdate(self, year, week, day):
        """
        Calculate the day of corresponding to the ISO year-week-day calendar.

        This function is effectively the inverse of
        :func:`datetime.date.isocalendar`.

        :param year:
            The year in the ISO calendar

        :param week:
            The week in the ISO calendar - range is [1, 53]

        :param day:
            The day in the ISO calendar - range is [1 (MON), 7 (SUN)]

        :return:
            Returns a :class:`datetime.date`
        """
        if not 0 < week < 54:
            raise ValueError('Invalid week: {}'.format(week))

        if not 0 < day < 8:     # Range is 1-7
            raise ValueError('Invalid weekday: {}'.format(day))

        # Get week 1 for the specific year:
        jan_4 = date(year, 1, 4)   # Week 1 always has January 4th in it
        week_1 = jan_4 - timedelta(days=jan_4.isocalendar()[2] - 1)

        # Now add the specific number of weeks and days to get what we want
        week_offset = (week - 1) * 7 + (day - 1)
        return week_1 + timedelta(days=week_offset)

    def _parse_isotime(self, timestr):
        len_str = len(timestr)
        components = [0, 0, 0, 0, None]
        pos = 0
        comp = -1

        if len_str < 2:
            raise ValueError('ISO time too short')

        has_sep = False

        while pos < len_str and comp < 5:
            comp += 1

            if timestr[pos:pos + 1] in b'-+Zz':
                # Detect time zone boundary
                components[-1] = self._parse_tzstr(timestr[pos:])
                pos = len_str
                break

            if comp == 1 and timestr[pos:pos+1] == self._TIME_SEP:
                has_sep = True
                pos += 1
            elif comp == 2 and has_sep:
                if timestr[pos:pos+1] != self._TIME_SEP:
                    raise ValueError('Inconsistent use of colon separator')
                pos += 1

            if comp < 3:
                # Hour, minute, second
                components[comp] = int(timestr[pos:pos + 2])
                pos += 2

            if comp == 3:
                # Fraction of a second
                frac = self._FRACTION_REGEX.match(timestr[pos:])
                if not frac:
                    continue

                us_str = frac.group(1)[:6]  # Truncate to microseconds
                components[comp] = int(us_str) * 10**(6 - len(us_str))
                pos += len(frac.group())

        if pos < len_str:
            raise ValueError('Unused components in ISO string')

        if components[0] == 24:
            # Standard supports 00:00 and 24:00 as representations of midnight
            if any(component != 0 for component in components[1:4]):
                raise ValueError('Hour may only be 24 at 24:00:00.000')

        return components

    def _parse_tzstr(self, tzstr, zero_as_utc=True):
        if tzstr == b'Z' or tzstr == b'z':
            return tz.UTC

        if len(tzstr) not in {3, 5, 6}:
            raise ValueError('Time zone offset must be 1, 3, 5 or 6 characters')

        if tzstr[0:1] == b'-':
            mult = -1
        elif tzstr[0:1] == b'+':
            mult = 1
        else:
            raise ValueError('Time zone offset requires sign')

        hours = int(tzstr[1:3])
        if len(tzstr) == 3:
            minutes = 0
        else:
            minutes = int(tzstr[(4 if tzstr[3:4] == self._TIME_SEP else 3):])

        if zero_as_utc and hours == 0 and minutes == 0:
            return tz.UTC
        else:
            if minutes > 59:
                raise ValueError('Invalid minutes in time zone offset')

            if hours > 23:
                raise ValueError('Invalid hours in time zone offset')

            return tz.tzoffset(None, mult * (hours * 60 + minutes) * 60)


DEFAULT_ISOPARSER = isoparser()
isoparse = DEFAULT_ISOPARSER.isoparse

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\methods\describe.py ===
"""
Module responsible for execution of NDFrame.describe() method.

Method NDFrame.describe() delegates actual execution to function describe_ndframe().
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    cast,
)

import numpy as np

from pandas._libs.tslibs import Timestamp
from pandas._typing import (
    DtypeObj,
    NDFrameT,
    npt,
)
from pandas.util._validators import validate_percentile

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)

from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat

from pandas.io.formats.format import format_percentiles

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas import (
        DataFrame,
        Series,
    )


def describe_ndframe(
    *,
    obj: NDFrameT,
    include: str | Sequence[str] | None,
    exclude: str | Sequence[str] | None,
    percentiles: Sequence[float] | np.ndarray | None,
) -> NDFrameT:
    """Describe series or dataframe.

    Called from pandas.core.generic.NDFrame.describe()

    Parameters
    ----------
    obj: DataFrame or Series
        Either dataframe or series to be described.
    include : 'all', list-like of dtypes or None (default), optional
        A white list of data types to include in the result. Ignored for ``Series``.
    exclude : list-like of dtypes or None (default), optional,
        A black list of data types to omit from the result. Ignored for ``Series``.
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. All should fall between 0 and 1.
        The default is ``[.25, .5, .75]``, which returns the 25th, 50th, and
        75th percentiles.

    Returns
    -------
    Dataframe or series description.
    """
    percentiles = _refine_percentiles(percentiles)

    describer: NDFrameDescriberAbstract
    if obj.ndim == 1:
        describer = SeriesDescriber(
            obj=cast("Series", obj),
        )
    else:
        describer = DataFrameDescriber(
            obj=cast("DataFrame", obj),
            include=include,
            exclude=exclude,
        )

    result = describer.describe(percentiles=percentiles)
    return cast(NDFrameT, result)


class NDFrameDescriberAbstract(ABC):
    """Abstract class for describing dataframe or series.

    Parameters
    ----------
    obj : Series or DataFrame
        Object to be described.
    """

    def __init__(self, obj: DataFrame | Series) -> None:
        self.obj = obj

    @abstractmethod
    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame | Series:
        """Do describe either series or dataframe.

        Parameters
        ----------
        percentiles : list-like of numbers
            The percentiles to include in the output.
        """


class SeriesDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating series description."""

    obj: Series

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> Series:
        describe_func = select_describe_func(
            self.obj,
        )
        return describe_func(self.obj, percentiles)


class DataFrameDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating dataobj description.

    Parameters
    ----------
    obj : DataFrame
        DataFrame to be described.
    include : 'all', list-like of dtypes or None
        A white list of data types to include in the result.
    exclude : list-like of dtypes or None
        A black list of data types to omit from the result.
    """

    obj: DataFrame

    def __init__(
        self,
        obj: DataFrame,
        *,
        include: str | Sequence[str] | None,
        exclude: str | Sequence[str] | None,
    ) -> None:
        self.include = include
        self.exclude = exclude

        if obj.ndim == 2 and obj.columns.size == 0:
            raise ValueError("Cannot describe a DataFrame without columns")

        super().__init__(obj)

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> DataFrame:
        data = self._select_data()

        ldesc: list[Series] = []
        for _, series in data.items():
            describe_func = select_describe_func(series)
            ldesc.append(describe_func(series, percentiles))

        col_names = reorder_columns(ldesc)
        d = concat(
            [x.reindex(col_names, copy=False) for x in ldesc],
            axis=1,
            sort=False,
        )
        d.columns = data.columns.copy()
        return d

    def _select_data(self) -> DataFrame:
        """Select columns to be described."""
        if (self.include is None) and (self.exclude is None):
            # when some numerics are found, keep only numerics
            default_include: list[npt.DTypeLike] = [np.number, "datetime"]
            data = self.obj.select_dtypes(include=default_include)
            if len(data.columns) == 0:
                data = self.obj
        elif self.include == "all":
            if self.exclude is not None:
                msg = "exclude must be None when include is 'all'"
                raise ValueError(msg)
            data = self.obj
        else:
            data = self.obj.select_dtypes(
                include=self.include,
                exclude=self.exclude,
            )
        return data


def reorder_columns(ldesc: Sequence[Series]) -> list[Hashable]:
    """Set a convenient order for rows for display."""
    names: list[Hashable] = []
    seen_names: set[Hashable] = set()
    ldesc_indexes = sorted((x.index for x in ldesc), key=len)
    for idxnames in ldesc_indexes:
        for name in idxnames:
            if name not in seen_names:
                seen_names.add(name)
                names.append(name)
    return names


def describe_numeric_1d(series: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing numerical data.

    Parameters
    ----------
    series : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    stat_index = ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]
    d = (
        [series.count(), series.mean(), series.std(), series.min()]
        + series.quantile(percentiles).tolist()
        + [series.max()]
    )
    # GH#48340 - always return float on non-complex numeric data
    dtype: DtypeObj | None
    if isinstance(series.dtype, ExtensionDtype):
        if isinstance(series.dtype, ArrowDtype):
            if series.dtype.kind == "m":
                # GH53001: describe timedeltas with object dtype
                dtype = None
            else:
                import pyarrow as pa

                dtype = ArrowDtype(pa.float64())
        else:
            dtype = Float64Dtype()
    elif series.dtype.kind in "iufb":
        # i.e. numeric but exclude complex dtype
        dtype = np.dtype("float")
    else:
        dtype = None
    return Series(d, index=stat_index, name=series.name, dtype=dtype)


def describe_categorical_1d(
    data: Series,
    percentiles_ignored: Sequence[float],
) -> Series:
    """Describe series containing categorical data.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ["count", "unique", "top", "freq"]
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    if count_unique > 0:
        top, freq = objcounts.index[0], objcounts.iloc[0]
        dtype = None
    else:
        # If the DataFrame is empty, set 'top' and 'freq' to None
        # to maintain output shape consistency
        top, freq = np.nan, np.nan
        dtype = "object"

    result = [data.count(), count_unique, top, freq]

    from pandas import Series

    return Series(result, index=names, name=data.name, dtype=dtype)


def describe_timestamp_as_categorical_1d(
    data: Series,
    percentiles_ignored: Sequence[float],
) -> Series:
    """Describe series containing timestamp data treated as categorical.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles_ignored : list-like of numbers
        Ignored, but in place to unify interface.
    """
    names = ["count", "unique"]
    objcounts = data.value_counts()
    count_unique = len(objcounts[objcounts != 0])
    result: list[float | Timestamp] = [data.count(), count_unique]
    dtype = None
    if count_unique > 0:
        top, freq = objcounts.index[0], objcounts.iloc[0]
        tz = data.dt.tz
        asint = data.dropna().values.view("i8")
        top = Timestamp(top)
        if top.tzinfo is not None and tz is not None:
            # Don't tz_localize(None) if key is already tz-aware
            top = top.tz_convert(tz)
        else:
            top = top.tz_localize(tz)
        names += ["top", "freq", "first", "last"]
        result += [
            top,
            freq,
            Timestamp(asint.min(), tz=tz),
            Timestamp(asint.max(), tz=tz),
        ]

    # If the DataFrame is empty, set 'top' and 'freq' to None
    # to maintain output shape consistency
    else:
        names += ["top", "freq"]
        result += [np.nan, np.nan]
        dtype = "object"

    from pandas import Series

    return Series(result, index=names, name=data.name, dtype=dtype)


def describe_timestamp_1d(data: Series, percentiles: Sequence[float]) -> Series:
    """Describe series containing datetime64 dtype.

    Parameters
    ----------
    data : Series
        Series to be described.
    percentiles : list-like of numbers
        The percentiles to include in the output.
    """
    # GH-30164
    from pandas import Series

    formatted_percentiles = format_percentiles(percentiles)

    stat_index = ["count", "mean", "min"] + formatted_percentiles + ["max"]
    d = (
        [data.count(), data.mean(), data.min()]
        + data.quantile(percentiles).tolist()
        + [data.max()]
    )
    return Series(d, index=stat_index, name=data.name)


def select_describe_func(
    data: Series,
) -> Callable:
    """Select proper function for describing series based on data type.

    Parameters
    ----------
    data : Series
        Series to be described.
    """
    if is_bool_dtype(data.dtype):
        return describe_categorical_1d
    elif is_numeric_dtype(data):
        return describe_numeric_1d
    elif data.dtype.kind == "M" or isinstance(data.dtype, DatetimeTZDtype):
        return describe_timestamp_1d
    elif data.dtype.kind == "m":
        return describe_numeric_1d
    else:
        return describe_categorical_1d


def _refine_percentiles(
    percentiles: Sequence[float] | np.ndarray | None,
) -> npt.NDArray[np.float64]:
    """
    Ensure that percentiles are unique and sorted.

    Parameters
    ----------
    percentiles : list-like of numbers, optional
        The percentiles to include in the output.
    """
    if percentiles is None:
        return np.array([0.25, 0.5, 0.75])

    # explicit conversion of `percentiles` to list
    percentiles = list(percentiles)

    # get them all to be in [0, 1]
    validate_percentile(percentiles)

    # median should always be included
    if 0.5 not in percentiles:
        percentiles.append(0.5)

    percentiles = np.asarray(percentiles)

    # sort and check for duplicates
    unique_pcts = np.unique(percentiles)
    assert percentiles is not None
    if len(unique_pcts) < len(percentiles):
        raise ValueError("percentiles cannot contain duplicates")

    return unique_pcts

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\storage.py ===
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

from typing import Dict, List, Optional, Union

from selenium.webdriver.common.bidi.common import command_builder


class SameSite:
    """Represents the possible same site values for cookies."""

    STRICT = "strict"
    LAX = "lax"
    NONE = "none"


class BytesValue:
    """Represents a bytes value."""

    TYPE_BASE64 = "base64"
    TYPE_STRING = "string"

    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

    def to_dict(self) -> Dict:
        """Converts the BytesValue to a dictionary.

        Returns:
        -------
            Dict: A dictionary representation of the BytesValue.
        """
        return {"type": self.type, "value": self.value}


class Cookie:
    """Represents a cookie."""

    def __init__(
        self,
        name: str,
        value: BytesValue,
        domain: str,
        path: Optional[str] = None,
        size: Optional[int] = None,
        http_only: Optional[bool] = None,
        secure: Optional[bool] = None,
        same_site: Optional[str] = None,
        expiry: Optional[int] = None,
    ):
        self.name = name
        self.value = value
        self.domain = domain
        self.path = path
        self.size = size
        self.http_only = http_only
        self.secure = secure
        self.same_site = same_site
        self.expiry = expiry

    @classmethod
    def from_dict(cls, data: Dict) -> "Cookie":
        """Creates a Cookie instance from a dictionary.

        Parameters:
        -----------
            data: A dictionary containing the cookie information.

        Returns:
        -------
            Cookie: A new instance of Cookie.
        """
        value = BytesValue(data.get("value", {}).get("type"), data.get("value", {}).get("value"))

        return cls(
            name=data.get("name"),
            value=value,
            domain=data.get("domain"),
            path=data.get("path"),
            size=data.get("size"),
            http_only=data.get("httpOnly"),
            secure=data.get("secure"),
            same_site=data.get("sameSite"),
            expiry=data.get("expiry"),
        )


class CookieFilter:
    """Represents a filter for cookies."""

    def __init__(
        self,
        name: Optional[str] = None,
        value: Optional[BytesValue] = None,
        domain: Optional[str] = None,
        path: Optional[str] = None,
        size: Optional[int] = None,
        http_only: Optional[bool] = None,
        secure: Optional[bool] = None,
        same_site: Optional[str] = None,
        expiry: Optional[int] = None,
    ):
        self.name = name
        self.value = value
        self.domain = domain
        self.path = path
        self.size = size
        self.http_only = http_only
        self.secure = secure
        self.same_site = same_site
        self.expiry = expiry

    def to_dict(self) -> Dict:
        """Converts the CookieFilter to a dictionary.

        Returns:
        -------
            Dict: A dictionary representation of the CookieFilter.
        """
        result = {}
        if self.name is not None:
            result["name"] = self.name
        if self.value is not None:
            result["value"] = self.value.to_dict()
        if self.domain is not None:
            result["domain"] = self.domain
        if self.path is not None:
            result["path"] = self.path
        if self.size is not None:
            result["size"] = self.size
        if self.http_only is not None:
            result["httpOnly"] = self.http_only
        if self.secure is not None:
            result["secure"] = self.secure
        if self.same_site is not None:
            result["sameSite"] = self.same_site
        if self.expiry is not None:
            result["expiry"] = self.expiry
        return result


class PartitionKey:
    """Represents a storage partition key."""

    def __init__(self, user_context: Optional[str] = None, source_origin: Optional[str] = None):
        self.user_context = user_context
        self.source_origin = source_origin

    @classmethod
    def from_dict(cls, data: Dict) -> "PartitionKey":
        """Creates a PartitionKey instance from a dictionary.

        Parameters:
        -----------
            data: A dictionary containing the partition key information.

        Returns:
        -------
            PartitionKey: A new instance of PartitionKey.
        """
        return cls(
            user_context=data.get("userContext"),
            source_origin=data.get("sourceOrigin"),
        )


class BrowsingContextPartitionDescriptor:
    """Represents a browsing context partition descriptor."""

    def __init__(self, context: str):
        self.type = "context"
        self.context = context

    def to_dict(self) -> Dict:
        """Converts the BrowsingContextPartitionDescriptor to a dictionary.

        Returns:
        -------
            Dict: A dictionary representation of the BrowsingContextPartitionDescriptor.
        """
        return {"type": self.type, "context": self.context}


class StorageKeyPartitionDescriptor:
    """Represents a storage key partition descriptor."""

    def __init__(self, user_context: Optional[str] = None, source_origin: Optional[str] = None):
        self.type = "storageKey"
        self.user_context = user_context
        self.source_origin = source_origin

    def to_dict(self) -> Dict:
        """Converts the StorageKeyPartitionDescriptor to a dictionary.

        Returns:
        -------
            Dict: A dictionary representation of the StorageKeyPartitionDescriptor.
        """
        result = {"type": self.type}
        if self.user_context is not None:
            result["userContext"] = self.user_context
        if self.source_origin is not None:
            result["sourceOrigin"] = self.source_origin
        return result


class PartialCookie:
    """Represents a partial cookie for setting."""

    def __init__(
        self,
        name: str,
        value: BytesValue,
        domain: str,
        path: Optional[str] = None,
        http_only: Optional[bool] = None,
        secure: Optional[bool] = None,
        same_site: Optional[str] = None,
        expiry: Optional[int] = None,
    ):
        self.name = name
        self.value = value
        self.domain = domain
        self.path = path
        self.http_only = http_only
        self.secure = secure
        self.same_site = same_site
        self.expiry = expiry

    def to_dict(self) -> Dict:
        """Converts the PartialCookie to a dictionary.

        Returns:
        -------
            Dict: A dictionary representation of the PartialCookie.
        """
        result = {
            "name": self.name,
            "value": self.value.to_dict(),
            "domain": self.domain,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.http_only is not None:
            result["httpOnly"] = self.http_only
        if self.secure is not None:
            result["secure"] = self.secure
        if self.same_site is not None:
            result["sameSite"] = self.same_site
        if self.expiry is not None:
            result["expiry"] = self.expiry
        return result


class GetCookiesResult:
    """Represents the result of a getCookies command."""

    def __init__(self, cookies: List[Cookie], partition_key: PartitionKey):
        self.cookies = cookies
        self.partition_key = partition_key

    @classmethod
    def from_dict(cls, data: Dict) -> "GetCookiesResult":
        """Creates a GetCookiesResult instance from a dictionary.

        Parameters:
        -----------
            data: A dictionary containing the get cookies result information.

        Returns:
        -------
            GetCookiesResult: A new instance of GetCookiesResult.
        """
        cookies = [Cookie.from_dict(cookie) for cookie in data.get("cookies", [])]
        partition_key = PartitionKey.from_dict(data.get("partitionKey", {}))
        return cls(cookies=cookies, partition_key=partition_key)


class SetCookieResult:
    """Represents the result of a setCookie command."""

    def __init__(self, partition_key: PartitionKey):
        self.partition_key = partition_key

    @classmethod
    def from_dict(cls, data: Dict) -> "SetCookieResult":
        """Creates a SetCookieResult instance from a dictionary.

        Parameters:
        -----------
            data: A dictionary containing the set cookie result information.

        Returns:
        -------
            SetCookieResult: A new instance of SetCookieResult.
        """
        partition_key = PartitionKey.from_dict(data.get("partitionKey", {}))
        return cls(partition_key=partition_key)


class DeleteCookiesResult:
    """Represents the result of a deleteCookies command."""

    def __init__(self, partition_key: PartitionKey):
        self.partition_key = partition_key

    @classmethod
    def from_dict(cls, data: Dict) -> "DeleteCookiesResult":
        """Creates a DeleteCookiesResult instance from a dictionary.

        Parameters:
        -----------
            data: A dictionary containing the delete cookies result information.

        Returns:
        -------
            DeleteCookiesResult: A new instance of DeleteCookiesResult.
        """
        partition_key = PartitionKey.from_dict(data.get("partitionKey", {}))
        return cls(partition_key=partition_key)


class Storage:
    """BiDi implementation of the storage module."""

    def __init__(self, conn):
        self.conn = conn

    def get_cookies(
        self,
        filter: Optional[CookieFilter] = None,
        partition: Optional[Union[BrowsingContextPartitionDescriptor, StorageKeyPartitionDescriptor]] = None,
    ) -> GetCookiesResult:
        """Retrieves cookies that match the given parameters.

        Parameters:
        -----------
            filter: Optional filter to match cookies.
            partition: Optional partition descriptor.

        Returns:
        -------
            GetCookiesResult: The result of the get cookies command.
        """
        params = {}
        if filter is not None:
            params["filter"] = filter.to_dict()
        if partition is not None:
            params["partition"] = partition.to_dict()

        result = self.conn.execute(command_builder("storage.getCookies", params))
        return GetCookiesResult.from_dict(result)

    def set_cookie(
        self,
        cookie: PartialCookie,
        partition: Optional[Union[BrowsingContextPartitionDescriptor, StorageKeyPartitionDescriptor]] = None,
    ) -> SetCookieResult:
        """Sets a cookie in the browser.

        Parameters:
        -----------
            cookie: The cookie to set.
            partition: Optional partition descriptor.

        Returns:
        -------
            SetCookieResult: The result of the set cookie command.
        """
        params = {"cookie": cookie.to_dict()}
        if partition is not None:
            params["partition"] = partition.to_dict()

        result = self.conn.execute(command_builder("storage.setCookie", params))
        return SetCookieResult.from_dict(result)

    def delete_cookies(
        self,
        filter: Optional[CookieFilter] = None,
        partition: Optional[Union[BrowsingContextPartitionDescriptor, StorageKeyPartitionDescriptor]] = None,
    ) -> DeleteCookiesResult:
        """Deletes cookies that match the given parameters.

        Parameters:
        -----------
            filter: Optional filter to match cookies to delete.
            partition: Optional partition descriptor.

        Returns:
        -------
            DeleteCookiesResult: The result of the delete cookies command.
        """
        params = {}
        if filter is not None:
            params["filter"] = filter.to_dict()
        if partition is not None:
            params["partition"] = partition.to_dict()

        result = self.conn.execute(command_builder("storage.deleteCookies", params))
        return DeleteCookiesResult.from_dict(result)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\containers.py ===
"""Module for SymPy containers

    (SymPy objects that store other SymPy objects)

    The containers implemented in this module are subclassed to Basic.
    They are supposed to work seamlessly within the SymPy framework.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableSet
from typing import Any, Callable

from .basic import Basic
from .sorting import default_sort_key, ordered
from .sympify import _sympify, sympify, _sympy_converter, SympifyError
from sympy.core.kind import Kind
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


class Tuple(Basic):
    """
    Wrapper around the builtin tuple object.

    Explanation
    ===========

    The Tuple is a subclass of Basic, so that it works well in the
    SymPy framework.  The wrapped tuple is available as self.args, but
    you can also access elements or slices with [:] syntax.

    Parameters
    ==========

    sympify : bool
        If ``False``, ``sympify`` is not called on ``args``. This
        can be used for speedups for very large tuples where the
        elements are known to already be SymPy objects.

    Examples
    ========

    >>> from sympy import Tuple, symbols
    >>> a, b, c, d = symbols('a b c d')
    >>> Tuple(a, b, c)[1:]
    (b, c)
    >>> Tuple(a, b, c).subs(a, d)
    (d, b, c)

    """

    def __new__(cls, *args, **kwargs):
        if kwargs.get('sympify', True):
            args = (sympify(arg) for arg in args)
        obj = Basic.__new__(cls, *args)
        return obj

    def __getitem__(self, i):
        if isinstance(i, slice):
            indices = i.indices(len(self))
            return Tuple(*(self.args[j] for j in range(*indices)))
        return self.args[i]

    def __len__(self):
        return len(self.args)

    def __contains__(self, item):
        return item in self.args

    def __iter__(self):
        return iter(self.args)

    def __add__(self, other):
        if isinstance(other, Tuple):
            return Tuple(*(self.args + other.args))
        elif isinstance(other, tuple):
            return Tuple(*(self.args + other))
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Tuple):
            return Tuple(*(other.args + self.args))
        elif isinstance(other, tuple):
            return Tuple(*(other + self.args))
        else:
            return NotImplemented

    def __mul__(self, other):
        try:
            n = as_int(other)
        except ValueError:
            raise TypeError("Can't multiply sequence by non-integer of type '%s'" % type(other))
        return self.func(*(self.args*n))

    __rmul__ = __mul__

    def __eq__(self, other):
        if isinstance(other, Basic):
            return super().__eq__(other)
        return self.args == other

    def __ne__(self, other):
        if isinstance(other, Basic):
            return super().__ne__(other)
        return self.args != other

    def __hash__(self):
        return hash(self.args)

    def _to_mpmath(self, prec):
        return tuple(a._to_mpmath(prec) for a in self.args)

    def __lt__(self, other):
        return _sympify(self.args < other.args)

    def __le__(self, other):
        return _sympify(self.args <= other.args)

    # XXX: Basic defines count() as something different, so we can't
    # redefine it here. Originally this lead to cse() test failure.
    def tuple_count(self, value) -> int:
        """Return number of occurrences of value."""
        return self.args.count(value)

    def index(self, value, start=None, stop=None):
        """Searches and returns the first index of the value."""
        # XXX: One would expect:
        #
        # return self.args.index(value, start, stop)
        #
        # here. Any trouble with that? Yes:
        #
        # >>> (1,).index(1, None, None)
        # Traceback (most recent call last):
        #   File "<stdin>", line 1, in <module>
        # TypeError: slice indices must be integers or None or have an __index__ method
        #
        # See: http://bugs.python.org/issue13340

        if start is None and stop is None:
            return self.args.index(value)
        elif stop is None:
            return self.args.index(value, start)
        else:
            return self.args.index(value, start, stop)

    @property
    def kind(self):
        """
        The kind of a Tuple instance.

        The kind of a Tuple is always of :class:`TupleKind` but
        parametrised by the number of elements and the kind of each element.

        Examples
        ========

        >>> from sympy import Tuple, Matrix
        >>> Tuple(1, 2).kind
        TupleKind(NumberKind, NumberKind)
        >>> Tuple(Matrix([1, 2]), 1).kind
        TupleKind(MatrixKind(NumberKind), NumberKind)
        >>> Tuple(1, 2).kind.element_kind
        (NumberKind, NumberKind)

        See Also
        ========

        sympy.matrices.kind.MatrixKind
        sympy.core.kind.NumberKind
        """
        return TupleKind(*(i.kind for i in self.args))

_sympy_converter[tuple] = lambda tup: Tuple(*tup)





def tuple_wrapper(method):
    """
    Decorator that converts any tuple in the function arguments into a Tuple.

    Explanation
    ===========

    The motivation for this is to provide simple user interfaces.  The user can
    call a function with regular tuples in the argument, and the wrapper will
    convert them to Tuples before handing them to the function.

    Explanation
    ===========

    >>> from sympy.core.containers import tuple_wrapper
    >>> def f(*args):
    ...    return args
    >>> g = tuple_wrapper(f)

    The decorated function g sees only the Tuple argument:

    >>> g(0, (1, 2), 3)
    (0, (1, 2), 3)

    """
    def wrap_tuples(*args, **kw_args):
        newargs = []
        for arg in args:
            if isinstance(arg, tuple):
                newargs.append(Tuple(*arg))
            else:
                newargs.append(arg)
        return method(*newargs, **kw_args)
    return wrap_tuples


class Dict(Basic):
    """
    Wrapper around the builtin dict object.

    Explanation
    ===========

    The Dict is a subclass of Basic, so that it works well in the
    SymPy framework.  Because it is immutable, it may be included
    in sets, but its values must all be given at instantiation and
    cannot be changed afterwards.  Otherwise it behaves identically
    to the Python dict.

    Examples
    ========

    >>> from sympy import Dict, Symbol

    >>> D = Dict({1: 'one', 2: 'two'})
    >>> for key in D:
    ...    if key == 1:
    ...        print('%s %s' % (key, D[key]))
    1 one

    The args are sympified so the 1 and 2 are Integers and the values
    are Symbols. Queries automatically sympify args so the following work:

    >>> 1 in D
    True
    >>> D.has(Symbol('one')) # searches keys and values
    True
    >>> 'one' in D # not in the keys
    False
    >>> D[1]
    one

    """

    elements: frozenset[Tuple]
    _dict: dict[Basic, Basic]

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], (dict, Dict)):
            items = [Tuple(k, v) for k, v in args[0].items()]
        elif iterable(args) and all(len(arg) == 2 for arg in args):
            items = [Tuple(k, v) for k, v in args]
        else:
            raise TypeError('Pass Dict args as Dict((k1, v1), ...) or Dict({k1: v1, ...})')
        elements = frozenset(items)
        obj = Basic.__new__(cls, *ordered(items))
        obj.elements = elements
        obj._dict = dict(items)  # In case Tuple decides it wants to sympify
        return obj

    def __getitem__(self, key):
        """x.__getitem__(y) <==> x[y]"""
        try:
            key = _sympify(key)
        except SympifyError:
            raise KeyError(key)

        return self._dict[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("SymPy Dicts are Immutable")

    def items(self):
        '''Returns a set-like object providing a view on dict's items.
        '''
        return self._dict.items()

    def keys(self):
        '''Returns the list of the dict's keys.'''
        return self._dict.keys()

    def values(self):
        '''Returns the list of the dict's values.'''
        return self._dict.values()

    def __iter__(self):
        '''x.__iter__() <==> iter(x)'''
        return iter(self._dict)

    def __len__(self):
        '''x.__len__() <==> len(x)'''
        return self._dict.__len__()

    def get(self, key, default=None):
        '''Returns the value for key if the key is in the dictionary.'''
        try:
            key = _sympify(key)
        except SympifyError:
            return default
        return self._dict.get(key, default)

    def __contains__(self, key):
        '''D.__contains__(k) -> True if D has a key k, else False'''
        try:
            key = _sympify(key)
        except SympifyError:
            return False
        return key in self._dict

    def __lt__(self, other):
        return _sympify(self.args < other.args)

    @property
    def _sorted_args(self):
        return tuple(sorted(self.args, key=default_sort_key))

    def __eq__(self, other):
        if isinstance(other, dict):
            return self == Dict(other)
        return super().__eq__(other)

    __hash__ : Callable[[Basic], Any] = Basic.__hash__

# this handles dict, defaultdict, OrderedDict
_sympy_converter[dict] = lambda d: Dict(*d.items())

class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        if iterable:
            self.map = OrderedDict((item, None) for item in iterable)
        else:
            self.map = OrderedDict()

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        self.map[key] = None

    def discard(self, key):
        self.map.pop(key)

    def pop(self, last=True):
        return self.map.popitem(last=last)[0]

    def __iter__(self):
        yield from self.map.keys()

    def __repr__(self):
        if not self.map:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.map.keys()))

    def intersection(self, other):
        return self.__class__([val for val in self if val in other])

    def difference(self, other):
        return self.__class__([val for val in self if val not in other])

    def update(self, iterable):
        for val in iterable:
            self.add(val)

class TupleKind(Kind):
    """
    TupleKind is a subclass of Kind, which is used to define Kind of ``Tuple``.

    Parameters of TupleKind will be kinds of all the arguments in Tuples, for
    example

    Parameters
    ==========

    args : tuple(element_kind)
       element_kind is kind of element.
       args is tuple of kinds of element

    Examples
    ========

    >>> from sympy import Tuple
    >>> Tuple(1, 2).kind
    TupleKind(NumberKind, NumberKind)
    >>> Tuple(1, 2).kind.element_kind
    (NumberKind, NumberKind)

    See Also
    ========

    sympy.core.kind.NumberKind
    MatrixKind
    sympy.sets.sets.SetKind
    """
    def __new__(cls, *args):
        obj = super().__new__(cls, *args)
        obj.element_kind = args
        return obj

    def __repr__(self):
        return "TupleKind{}".format(self.element_kind)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\utils.py ===
from __future__ import annotations

import importlib
import logging
import unicodedata
from codecs import IncrementalDecoder
from encodings.aliases import aliases
from functools import lru_cache
from re import findall
from typing import Generator

from _multibytecodec import (  # type: ignore[import-not-found,import]
    MultibyteIncrementalDecoder,
)

from .constant import (
    ENCODING_MARKS,
    IANA_SUPPORTED_SIMILAR,
    RE_POSSIBLE_ENCODING_INDICATION,
    UNICODE_RANGES_COMBINED,
    UNICODE_SECONDARY_RANGE_KEYWORD,
    UTF8_MAXIMAL_ALLOCATION,
    COMMON_CJK_CHARACTERS,
)


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_accentuated(character: str) -> bool:
    try:
        description: str = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False
    return (
        "WITH GRAVE" in description
        or "WITH ACUTE" in description
        or "WITH CEDILLA" in description
        or "WITH DIAERESIS" in description
        or "WITH CIRCUMFLEX" in description
        or "WITH TILDE" in description
        or "WITH MACRON" in description
        or "WITH RING ABOVE" in description
    )


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def remove_accent(character: str) -> str:
    decomposed: str = unicodedata.decomposition(character)
    if not decomposed:
        return character

    codes: list[str] = decomposed.split(" ")

    return chr(int(codes[0], 16))


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def unicode_range(character: str) -> str | None:
    """
    Retrieve the Unicode range official name from a single character.
    """
    character_ord: int = ord(character)

    for range_name, ord_range in UNICODE_RANGES_COMBINED.items():
        if character_ord in ord_range:
            return range_name

    return None


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_latin(character: str) -> bool:
    try:
        description: str = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False
    return "LATIN" in description


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_punctuation(character: str) -> bool:
    character_category: str = unicodedata.category(character)

    if "P" in character_category:
        return True

    character_range: str | None = unicode_range(character)

    if character_range is None:
        return False

    return "Punctuation" in character_range


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_symbol(character: str) -> bool:
    character_category: str = unicodedata.category(character)

    if "S" in character_category or "N" in character_category:
        return True

    character_range: str | None = unicode_range(character)

    if character_range is None:
        return False

    return "Forms" in character_range and character_category != "Lo"


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_emoticon(character: str) -> bool:
    character_range: str | None = unicode_range(character)

    if character_range is None:
        return False

    return "Emoticons" in character_range or "Pictographs" in character_range


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_separator(character: str) -> bool:
    if character.isspace() or character in {"｜", "+", "<", ">"}:
        return True

    character_category: str = unicodedata.category(character)

    return "Z" in character_category or character_category in {"Po", "Pd", "Pc"}


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_case_variable(character: str) -> bool:
    return character.islower() != character.isupper()


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_cjk(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "CJK" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_hiragana(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "HIRAGANA" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_katakana(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "KATAKANA" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_hangul(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "HANGUL" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_thai(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "THAI" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_arabic(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "ARABIC" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_arabic_isolated_form(character: str) -> bool:
    try:
        character_name = unicodedata.name(character)
    except ValueError:  # Defensive: unicode database outdated?
        return False

    return "ARABIC" in character_name and "ISOLATED FORM" in character_name


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_cjk_uncommon(character: str) -> bool:
    return character not in COMMON_CJK_CHARACTERS


@lru_cache(maxsize=len(UNICODE_RANGES_COMBINED))
def is_unicode_range_secondary(range_name: str) -> bool:
    return any(keyword in range_name for keyword in UNICODE_SECONDARY_RANGE_KEYWORD)


@lru_cache(maxsize=UTF8_MAXIMAL_ALLOCATION)
def is_unprintable(character: str) -> bool:
    return (
        character.isspace() is False  # includes \n \t \r \v
        and character.isprintable() is False
        and character != "\x1a"  # Why? Its the ASCII substitute character.
        and character != "\ufeff"  # bug discovered in Python,
        # Zero Width No-Break Space located in 	Arabic Presentation Forms-B, Unicode 1.1 not acknowledged as space.
    )


def any_specified_encoding(sequence: bytes, search_zone: int = 8192) -> str | None:
    """
    Extract using ASCII-only decoder any specified encoding in the first n-bytes.
    """
    if not isinstance(sequence, bytes):
        raise TypeError

    seq_len: int = len(sequence)

    results: list[str] = findall(
        RE_POSSIBLE_ENCODING_INDICATION,
        sequence[: min(seq_len, search_zone)].decode("ascii", errors="ignore"),
    )

    if len(results) == 0:
        return None

    for specified_encoding in results:
        specified_encoding = specified_encoding.lower().replace("-", "_")

        encoding_alias: str
        encoding_iana: str

        for encoding_alias, encoding_iana in aliases.items():
            if encoding_alias == specified_encoding:
                return encoding_iana
            if encoding_iana == specified_encoding:
                return encoding_iana

    return None


@lru_cache(maxsize=128)
def is_multi_byte_encoding(name: str) -> bool:
    """
    Verify is a specific encoding is a multi byte one based on it IANA name
    """
    return name in {
        "utf_8",
        "utf_8_sig",
        "utf_16",
        "utf_16_be",
        "utf_16_le",
        "utf_32",
        "utf_32_le",
        "utf_32_be",
        "utf_7",
    } or issubclass(
        importlib.import_module(f"encodings.{name}").IncrementalDecoder,
        MultibyteIncrementalDecoder,
    )


def identify_sig_or_bom(sequence: bytes) -> tuple[str | None, bytes]:
    """
    Identify and extract SIG/BOM in given sequence.
    """

    for iana_encoding in ENCODING_MARKS:
        marks: bytes | list[bytes] = ENCODING_MARKS[iana_encoding]

        if isinstance(marks, bytes):
            marks = [marks]

        for mark in marks:
            if sequence.startswith(mark):
                return iana_encoding, mark

    return None, b""


def should_strip_sig_or_bom(iana_encoding: str) -> bool:
    return iana_encoding not in {"utf_16", "utf_32"}


def iana_name(cp_name: str, strict: bool = True) -> str:
    """Returns the Python normalized encoding name (Not the IANA official name)."""
    cp_name = cp_name.lower().replace("-", "_")

    encoding_alias: str
    encoding_iana: str

    for encoding_alias, encoding_iana in aliases.items():
        if cp_name in [encoding_alias, encoding_iana]:
            return encoding_iana

    if strict:
        raise ValueError(f"Unable to retrieve IANA for '{cp_name}'")

    return cp_name


def cp_similarity(iana_name_a: str, iana_name_b: str) -> float:
    if is_multi_byte_encoding(iana_name_a) or is_multi_byte_encoding(iana_name_b):
        return 0.0

    decoder_a = importlib.import_module(f"encodings.{iana_name_a}").IncrementalDecoder
    decoder_b = importlib.import_module(f"encodings.{iana_name_b}").IncrementalDecoder

    id_a: IncrementalDecoder = decoder_a(errors="ignore")
    id_b: IncrementalDecoder = decoder_b(errors="ignore")

    character_match_count: int = 0

    for i in range(255):
        to_be_decoded: bytes = bytes([i])
        if id_a.decode(to_be_decoded) == id_b.decode(to_be_decoded):
            character_match_count += 1

    return character_match_count / 254


def is_cp_similar(iana_name_a: str, iana_name_b: str) -> bool:
    """
    Determine if two code page are at least 80% similar. IANA_SUPPORTED_SIMILAR dict was generated using
    the function cp_similarity.
    """
    return (
        iana_name_a in IANA_SUPPORTED_SIMILAR
        and iana_name_b in IANA_SUPPORTED_SIMILAR[iana_name_a]
    )


def set_logging_handler(
    name: str = "charset_normalizer",
    level: int = logging.INFO,
    format_string: str = "%(asctime)s | %(levelname)s | %(message)s",
) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(handler)


def cut_sequence_chunks(
    sequences: bytes,
    encoding_iana: str,
    offsets: range,
    chunk_size: int,
    bom_or_sig_available: bool,
    strip_sig_or_bom: bool,
    sig_payload: bytes,
    is_multi_byte_decoder: bool,
    decoded_payload: str | None = None,
) -> Generator[str, None, None]:
    if decoded_payload and is_multi_byte_decoder is False:
        for i in offsets:
            chunk = decoded_payload[i : i + chunk_size]
            if not chunk:
                break
            yield chunk
    else:
        for i in offsets:
            chunk_end = i + chunk_size
            if chunk_end > len(sequences) + 8:
                continue

            cut_sequence = sequences[i : i + chunk_size]

            if bom_or_sig_available and strip_sig_or_bom is False:
                cut_sequence = sig_payload + cut_sequence

            chunk = cut_sequence.decode(
                encoding_iana,
                errors="ignore" if is_multi_byte_decoder else "strict",
            )

            # multi-byte bad cutting detector and adjustment
            # not the cleanest way to perform that fix but clever enough for now.
            if is_multi_byte_decoder and i > 0:
                chunk_partial_size_chk: int = min(chunk_size, 16)

                if (
                    decoded_payload
                    and chunk[:chunk_partial_size_chk] not in decoded_payload
                ):
                    for j in range(i, i - 4, -1):
                        cut_sequence = sequences[j:chunk_end]

                        if bom_or_sig_available and strip_sig_or_bom is False:
                            cut_sequence = sig_payload + cut_sequence

                        chunk = cut_sequence.decode(encoding_iana, errors="ignore")

                        if chunk[:chunk_partial_size_chk] in decoded_payload:
                            break

            yield chunk

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\phi2\inference_example.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import numpy as np
import torch
from transformers import AutoTokenizer

import onnxruntime as ort

pt_to_np = {
    "torch.int32": np.int32,
    "torch.int64": np.int64,
    "torch.float32": np.float32,
    "torch.float16": np.float16,
}


def cuda_memcpy(dst, src):
    from cuda import cudart

    cudart.cudaMemcpy(
        dst.data_ptr(),
        src.data_ptr(),
        src.element_size() * src.nelement(),
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    )


class ORTGenerator:
    def __init__(self, decoder_path):
        self.onnx_decoder_path = decoder_path
        self.num_heads = 32
        self.head_size = 80
        self.num_layers = 32
        self.max_sequence_length = 2048
        self.device_id = 0
        self.use_cuda_graph = False
        self.use_traced_inputs = False
        self.static_inputs_map = {}

    def append_static_inputs(self, batch_size):
        # Only use this function with GQA and with use_cuda_graph=True
        if batch_size in self.static_inputs_map:
            return

        cpu_device = torch.device("cpu")
        cuda_device = torch.device("cuda", self.device_id)

        static_io = {}
        static_io["input_ids"] = torch.zeros((batch_size, 1), dtype=torch.int32, device=cuda_device)
        static_io["step"] = torch.tensor([0], dtype=torch.int64, device=cuda_device)
        static_io["seqlens_k"] = torch.tensor(batch_size * [0], dtype=torch.int32, device=cuda_device)
        static_io["total_sequence_length"] = torch.tensor([0], dtype=torch.int32, device=cpu_device)

        cache_shape = (batch_size, self.num_heads, self.max_sequence_length, self.head_size)
        for i in range(self.num_layers):
            cache = torch.zeros(cache_shape, device=cuda_device, dtype=torch.float16)
            static_io.update({f"past_key_{i}": cache.contiguous(), f"past_value_{i}": cache.clone().contiguous()})

        static_io["logits"] = torch.zeros((batch_size, 1, 51200), dtype=torch.float16, device=cuda_device)

        self.static_inputs_map[batch_size] = static_io

    def get_initial_inputs_and_outputs(self, encodings_dict):
        self.torch_dtype = torch.float16 if self.use_fp16 else torch.float32

        input_ids = torch.tensor(encodings_dict["input_ids"], device=self.device, dtype=torch.int32)
        attention_mask = torch.tensor(encodings_dict["attention_mask"], device=self.device, dtype=torch.int32)

        batch_size, sequence_length = input_ids.shape

        self.use_traced_inputs = (
            self.use_cuda_graph
            and (batch_size in self.static_inputs_map)
            and self.use_buffer_share
            and not self.packed_kv
        )

        step = (
            torch.tensor([0], device=self.device, dtype=torch.int64)
            if not self.use_traced_inputs
            else self.static_inputs_map[batch_size]["step"]
        )

        seqlens_k = (
            torch.tensor(batch_size * [0], device=self.device, dtype=torch.int32)
            if not self.use_traced_inputs
            else self.static_inputs_map[batch_size]["seqlens_k"]
        )
        cuda_memcpy(seqlens_k, attention_mask.sum(1).sub(1).to(torch.int32))

        total_seq_length = (
            torch.tensor([0], device=torch.device("cpu"), dtype=torch.int32)
            if not self.use_traced_inputs
            else self.static_inputs_map[batch_size]["total_sequence_length"]
        )
        total_seq_length[0] = sequence_length

        inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
        }

        if self.use_step:
            inputs["step"] = step.contiguous()

        if self.use_cuda_graph:
            inputs["seqlens_k"] = seqlens_k.contiguous()
            inputs["total_sequence_length"] = total_seq_length.contiguous()
            del inputs["attention_mask"]

        past_seq_length = self.max_sequence_length if self.use_buffer_share else 0
        past_shape = (
            (2, batch_size, self.num_heads, past_seq_length, self.head_size)
            if self.packed_kv
            else (batch_size, self.num_heads, past_seq_length, self.head_size)
        )

        if not self.use_traced_inputs:
            for i in range(self.num_layers):
                past = torch.zeros(past_shape, device=self.device, dtype=self.torch_dtype)
                (
                    inputs.update({f"past_key_{i}": past.contiguous(), f"past_value_{i}": past.clone().contiguous()})
                    if not self.packed_kv
                    else inputs.update({f"past_{i}": past.contiguous()})
                )
        else:
            for i in range(self.num_layers):
                inputs.update(
                    {
                        f"past_key_{i}": self.static_inputs_map[batch_size][f"past_key_{i}"].contiguous(),
                        f"past_value_{i}": self.static_inputs_map[batch_size][f"past_value_{i}"].contiguous(),
                    }
                )

        logits = torch.zeros(batch_size, sequence_length, 51200, device=self.device, dtype=self.torch_dtype)
        outputs = {"logits": logits.contiguous()}

        if not self.use_buffer_share:
            present_shape = (
                (2, batch_size, self.num_heads, sequence_length, self.head_size)
                if self.packed_kv
                else (batch_size, self.num_heads, sequence_length, self.head_size)
            )
            for i in range(self.num_layers):
                present = torch.zeros(present_shape, device=self.device, dtype=self.torch_dtype)
                (
                    outputs.update(
                        {f"present_key_{i}": present.contiguous(), f"present_value_{i}": present.contiguous()}
                    )
                    if not self.packed_kv
                    else outputs.update({f"present_{i}": present.contiguous()})
                )

        return inputs, outputs

    def apply_io_binding(self, model: ort.InferenceSession, inputs: dict, outputs: dict):
        io_binding = model.io_binding()
        device = None

        for k, v in inputs.items():
            io_binding.bind_input(
                name=k,
                device_type=v.device.type,
                device_id=0 if v.device.type == "cpu" else v.device.index,
                element_type=pt_to_np[repr(v.dtype)],
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr(),
            )
            device = v.device

        for output in model.get_outputs():
            name = output.name
            if self.use_buffer_share and "present" in name:
                v = inputs[name.replace("present", "past")]
                io_binding.bind_output(
                    name=name,
                    device_type=v.device.type,
                    device_id=v.device.index,
                    element_type=(np.float16 if self.use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )
            else:
                v = outputs[name]
                io_binding.bind_output(
                    name=name,
                    device_type=device.type,
                    device_id=0 if device.type == "cpu" else device.index,
                    element_type=(np.float16 if self.use_fp16 else np.float32),
                    shape=tuple(v.shape),
                    buffer_ptr=v.data_ptr(),
                )

        return io_binding

    def create_session(
        self, device_id, use_fp16=True, use_buffer_share=True, packed_kv=False, use_step=False, use_cuda_graph=False
    ):
        self.device_id = device_id
        sess_options = ort.SessionOptions()
        sess_options.log_verbosity_level = 4
        sess_options.log_severity_level = 4
        self.use_cuda_graph = use_cuda_graph
        ep = (
            ("CUDAExecutionProvider", {"device_id": self.device_id, "enable_cuda_graph": self.use_cuda_graph})
            if self.device_id >= 0
            else "CPUExecutionProvider"
        )
        self.sess = ort.InferenceSession(self.onnx_decoder_path, sess_options=sess_options, providers=[ep])
        self.ro = ort.RunOptions()

        self.device = torch.device("cuda", self.device_id) if torch.cuda.is_available() else torch.device("cpu")
        self.use_fp16 = use_fp16
        self.use_buffer_share = use_buffer_share
        self.packed_kv = packed_kv
        self.use_step = use_step

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        self.tokenizer.pad_token = "[PAD]"

    def generate_impl(self, encodings_dict, max_length, cuda_graph_annotation, benchmark=False):
        inputs, outputs = self.get_initial_inputs_and_outputs(encodings_dict)

        all_token_ids = inputs["input_ids"].clone()
        batch_size, sequence_length = all_token_ids.shape

        current_length = sequence_length
        has_eos = torch.zeros(batch_size, device=self.device, dtype=torch.bool)

        if benchmark:
            import time

            latency = []

        prompt_run = True
        while current_length < max_length:
            io_binding = self.apply_io_binding(self.sess, inputs, outputs)

            if benchmark:
                start = time.time()

            io_binding.synchronize_inputs()
            if prompt_run:
                if self.use_cuda_graph:
                    # Disable CUDA graph for the prompt run
                    self.ro.add_run_config_entry("gpu_graph_id", "-1")
                self.sess.run_with_iobinding(io_binding, self.ro)
                if self.use_cuda_graph:
                    # Enable CUDA graph for the decoding run
                    self.ro.add_run_config_entry(
                        "gpu_graph_id", str(cuda_graph_annotation) if self.use_traced_inputs else "-1"
                    )
                prompt_run = False
            else:
                self.sess.run_with_iobinding(io_binding, self.ro)
            io_binding.synchronize_outputs()

            if benchmark:
                end = time.time()
                latency.append(end - start)

            # Sample with argmax (greedy search)
            next_token_logits = outputs["logits"][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Check if we previously reached EOS token id or if generated token id is EOS token id
            has_eos = has_eos | next_tokens == self.tokenizer.eos_token_id

            # Determine which new tokens to add to list of all token ids
            # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
            tokens_to_add = next_tokens.masked_fill(has_eos, self.tokenizer.eos_token_id).reshape([batch_size, 1])
            all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

            # Return early if all batch entries have reached EOS token id
            if torch.all(has_eos):
                break

            # Update inputs for next inference run
            current_length += 1

            inputs["input_ids"] = tokens_to_add.to(torch.int32)
            if self.use_traced_inputs:
                cuda_memcpy(self.static_inputs_map[batch_size]["input_ids"], inputs["input_ids"])
                inputs["input_ids"] = self.static_inputs_map[batch_size]["input_ids"]

            if self.use_step:
                inputs["step"] = torch.tensor([current_length - 1], device=self.device, dtype=torch.int64)
                if self.use_traced_inputs:
                    cuda_memcpy(self.static_inputs_map[batch_size]["step"], inputs["step"])
                    inputs["step"] = self.static_inputs_map[batch_size]["step"]

            if self.use_cuda_graph:
                previous_seqlens_k = inputs["seqlens_k"]
                inputs["seqlens_k"] = (previous_seqlens_k + (~has_eos).reshape(batch_size, 1)).to(torch.int32)
                inputs["total_sequence_length"][0] = current_length
                if self.use_traced_inputs:
                    cuda_memcpy(self.static_inputs_map[batch_size]["seqlens_k"], inputs["seqlens_k"])
                    inputs["seqlens_k"] = self.static_inputs_map[batch_size]["seqlens_k"]
                    self.static_inputs_map[batch_size]["total_sequence_length"][0] = inputs["total_sequence_length"][0]
                    inputs["total_sequence_length"] = self.static_inputs_map[batch_size]["total_sequence_length"]
            else:
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], (~has_eos).reshape(batch_size, 1)], 1
                ).to(torch.int32)

            # Set logits to zeros for next inference run and re-use memory buffer
            if outputs["logits"].shape[1] != 1:
                outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
                if self.use_traced_inputs:
                    outputs["logits"] = self.static_inputs_map[batch_size]["logits"]
            outputs["logits"].zero_()

            if not self.use_buffer_share:
                for i in range(self.num_layers):
                    if not self.packed_kv:
                        inputs[f"past_key_{i}"] = outputs[f"present_key_{i}"]
                        inputs[f"past_value_{i}"] = outputs[f"present_value_{i}"]
                    else:
                        inputs[f"past_{i}"] = outputs[f"present_{i}"]

                new_sequence_length = inputs["attention_mask"].shape[1]
                present_shape = (
                    (2, batch_size, self.num_heads, new_sequence_length, self.head_size)
                    if self.packed_kv
                    else (batch_size, self.num_heads, new_sequence_length, self.head_size)
                )
                for i in range(self.num_layers):
                    present = torch.zeros(present_shape, device=self.device, dtype=self.torch_dtype)
                    (
                        outputs.update(
                            {
                                f"present_key_{i}": present.contiguous(),
                                f"present_value_{i}": present.clone().contiguous(),
                            }
                        )
                        if not self.packed_kv
                        else outputs.update({f"present_{i}": present.contiguous()})
                    )

        if benchmark:
            print(
                f"Batch size: {batch_size}, Sequence length: {sequence_length}, Token num: {max_length - sequence_length}"
            )
            print(f"Prompt letency: {1000 * latency[0]}ms, Token latency: {1000 * np.mean(latency[1:])}ms")
            return

        texts = self.tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)
        return texts

    def generate(self, prompt, max_length, cuda_graph_annotation):
        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding=True)

        return self.generate_impl(encodings_dict, max_length, cuda_graph_annotation)

    def generate_benchmark(self, prompt_shape, token_num, cuda_graph_annotation):
        batch_size, sequence_length = prompt_shape
        max_length = sequence_length + token_num

        encodings_dict = {}
        encodings_dict["input_ids"] = torch.randint(0, 50264, (batch_size, sequence_length), dtype=torch.int32).tolist()
        encodings_dict["attention_mask"] = torch.ones((batch_size, sequence_length), dtype=torch.int32).tolist()

        # Warm up run
        self.generate_impl(encodings_dict, max_length, cuda_graph_annotation, benchmark=False)

        # Benchmark run
        self.generate_impl(encodings_dict, max_length, cuda_graph_annotation, benchmark=True)


def run_phi2(
    onnx_model_path,
    use_buffer_share,
    device_id,
    packed_kv=False,
    use_fp16=True,
    use_step=False,
    use_cuda_graph=False,
    run_benchmark=False,
):
    generator = ORTGenerator(onnx_model_path)
    generator.create_session(device_id, use_fp16, use_buffer_share, packed_kv, use_step, use_cuda_graph)

    def simple_run(prompt):
        example_batch_size = len(prompt)
        if use_cuda_graph:
            generator.append_static_inputs(batch_size=example_batch_size)
        texts = generator.generate(prompt, max_length=210, cuda_graph_annotation=example_batch_size)

        for i in range(len(texts)):
            print("Prompt: ", prompt[i])
            print("Texts: ", texts[i])

    prompt = [
        '''```python
    def print_prime(n):
    """
    Print all primes between 1 and n
    """'''
    ]

    if not run_benchmark:
        simple_run(prompt)

    # Run simple benchmark. Time the decoder only.
    if run_benchmark:
        token_num = 32
        for batch_size in [1, 2, 4, 8]:
            generator.append_static_inputs(batch_size)
            for sequence_length in [16, 512]:
                prompt_shape = (batch_size, sequence_length)
                generator.generate_benchmark(prompt_shape, token_num, cuda_graph_annotation=batch_size)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\sparse\accessor.py ===
"""Sparse accessor"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.dtypes import SparseDtype

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays.sparse.array import SparseArray

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )


class BaseAccessor:
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None) -> None:
        self._parent = data
        self._validate(data)

    def _validate(self, data):
        raise NotImplementedError


@delegate_names(
    SparseArray, ["npoints", "density", "fill_value", "sp_values"], typ="property"
)
class SparseAccessor(BaseAccessor, PandasDelegate):
    """
    Accessor for SparseSparse from other sparse matrix data types.

    Examples
    --------
    >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
    >>> ser.sparse.density
    0.6
    >>> ser.sparse.sp_values
    array([2, 2, 2])
    """

    def _validate(self, data):
        if not isinstance(data.dtype, SparseDtype):
            raise AttributeError(self._validation_msg)

    def _delegate_property_get(self, name: str, *args, **kwargs):
        return getattr(self._parent.array, name)

    def _delegate_method(self, name: str, *args, **kwargs):
        if name == "from_coo":
            return self.from_coo(*args, **kwargs)
        elif name == "to_coo":
            return self.to_coo(*args, **kwargs)
        else:
            raise ValueError

    @classmethod
    def from_coo(cls, A, dense_index: bool = False) -> Series:
        """
        Create a Series with sparse values from a scipy.sparse.coo_matrix.

        Parameters
        ----------
        A : scipy.sparse.coo_matrix
        dense_index : bool, default False
            If False (default), the index consists of only the
            coords of the non-null entries of the original coo_matrix.
            If True, the index consists of the full sorted
            (row, col) coordinates of the coo_matrix.

        Returns
        -------
        s : Series
            A Series with sparse values.

        Examples
        --------
        >>> from scipy import sparse

        >>> A = sparse.coo_matrix(
        ...     ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(3, 4)
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>

        >>> A.todense()
        matrix([[0., 0., 1., 2.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> ss = pd.Series.sparse.from_coo(A)
        >>> ss
        0  2    1.0
           3    2.0
        1  0    3.0
        dtype: Sparse[float64, nan]
        """
        from pandas import Series
        from pandas.core.arrays.sparse.scipy_sparse import coo_to_sparse_series

        result = coo_to_sparse_series(A, dense_index=dense_index)
        result = Series(result.array, index=result.index, copy=False)

        return result

    def to_coo(self, row_levels=(0,), column_levels=(1,), sort_labels: bool = False):
        """
        Create a scipy.sparse.coo_matrix from a Series with MultiIndex.

        Use row_levels and column_levels to determine the row and column
        coordinates respectively. row_levels and column_levels are the names
        (labels) or numbers of the levels. {row_levels, column_levels} must be
        a partition of the MultiIndex level names (or numbers).

        Parameters
        ----------
        row_levels : tuple/list
        column_levels : tuple/list
        sort_labels : bool, default False
            Sort the row and column labels before forming the sparse matrix.
            When `row_levels` and/or `column_levels` refer to a single level,
            set to `True` for a faster execution.

        Returns
        -------
        y : scipy.sparse.coo_matrix
        rows : list (row labels)
        columns : list (column labels)

        Examples
        --------
        >>> s = pd.Series([3.0, np.nan, 1.0, 3.0, np.nan, np.nan])
        >>> s.index = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, 2, "a", 0),
        ...         (1, 2, "a", 1),
        ...         (1, 1, "b", 0),
        ...         (1, 1, "b", 1),
        ...         (2, 1, "b", 0),
        ...         (2, 1, "b", 1)
        ...     ],
        ...     names=["A", "B", "C", "D"],
        ... )
        >>> s
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: float64

        >>> ss = s.astype("Sparse")
        >>> ss
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: Sparse[float64, nan]

        >>> A, rows, columns = ss.sparse.to_coo(
        ...     row_levels=["A", "B"], column_levels=["C", "D"], sort_labels=True
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>
        >>> A.todense()
        matrix([[0., 0., 1., 3.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> rows
        [(1, 1), (1, 2), (2, 1)]
        >>> columns
        [('a', 0), ('a', 1), ('b', 0), ('b', 1)]
        """
        from pandas.core.arrays.sparse.scipy_sparse import sparse_series_to_coo

        A, rows, columns = sparse_series_to_coo(
            self._parent, row_levels, column_levels, sort_labels=sort_labels
        )
        return A, rows, columns

    def to_dense(self) -> Series:
        """
        Convert a Series from sparse values to dense.

        Returns
        -------
        Series:
            A Series with the same values, stored as a dense array.

        Examples
        --------
        >>> series = pd.Series(pd.arrays.SparseArray([0, 1, 0]))
        >>> series
        0    0
        1    1
        2    0
        dtype: Sparse[int64, 0]

        >>> series.sparse.to_dense()
        0    0
        1    1
        2    0
        dtype: int64
        """
        from pandas import Series

        return Series(
            self._parent.array.to_dense(),
            index=self._parent.index,
            name=self._parent.name,
            copy=False,
        )


class SparseFrameAccessor(BaseAccessor, PandasDelegate):
    """
    DataFrame accessor for sparse data.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 0, 0],
    ...                   "b": [3, 0, 0, 4]}, dtype="Sparse[int]")
    >>> df.sparse.density
    0.5
    """

    def _validate(self, data):
        dtypes = data.dtypes
        if not all(isinstance(t, SparseDtype) for t in dtypes):
            raise AttributeError(self._validation_msg)

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None) -> DataFrame:
        """
        Create a new DataFrame from a scipy sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            Must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex.

        Returns
        -------
        DataFrame
            Each column of the DataFrame is stored as a
            :class:`arrays.SparseArray`.

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3, dtype=float)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0  1.0    0    0
        1    0  1.0    0
        2    0    0  1.0
        """
        from pandas._libs.sparse import IntIndex

        from pandas import DataFrame

        data = data.tocsc()
        index, columns = cls._prep_index(data, index, columns)
        n_rows, n_columns = data.shape
        # We need to make sure indices are sorted, as we create
        # IntIndex with no input validation (i.e. check_integrity=False ).
        # Indices may already be sorted in scipy in which case this adds
        # a small overhead.
        data.sort_indices()
        indices = data.indices
        indptr = data.indptr
        array_data = data.data
        dtype = SparseDtype(array_data.dtype, 0)
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[i + 1])
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)
            arr = SparseArray._simple_new(array_data[sl], idx, dtype)
            arrays.append(arr)
        return DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )

    def to_dense(self) -> DataFrame:
        """
        Convert a DataFrame with sparse values to dense.

        Returns
        -------
        DataFrame
            A DataFrame with the same values stored as dense arrays.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0])})
        >>> df.sparse.to_dense()
           A
        0  0
        1  1
        2  0
        """
        from pandas import DataFrame

        data = {k: v.array.to_dense() for k, v in self._parent.items()}
        return DataFrame(data, index=self._parent.index, columns=self._parent.columns)

    def to_coo(self):
        """
        Return the contents of the frame as a sparse SciPy COO matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            If the caller is heterogeneous and contains booleans or objects,
            the result will be of dtype=object. See Notes.

        Notes
        -----
        The dtype will be the lowest-common-denominator type (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32. By numpy.find_common_type convention, mixing int64 and
        and uint64 will result in a float64 dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.to_coo()
        <COOrdinate sparse matrix of dtype 'int64'
            with 2 stored elements and shape (4, 1)>
        """
        import_optional_dependency("scipy")
        from scipy.sparse import coo_matrix

        dtype = find_common_type(self._parent.dtypes.to_list())
        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype

        cols, rows, data = [], [], []
        for col, (_, ser) in enumerate(self._parent.items()):
            sp_arr = ser.array
            if sp_arr.fill_value != 0:
                raise ValueError("fill value must be 0 when converting to COO matrix")

            row = sp_arr.sp_index.indices
            cols.append(np.repeat(col, len(row)))
            rows.append(row)
            data.append(sp_arr.sp_values.astype(dtype, copy=False))

        cols = np.concatenate(cols)
        rows = np.concatenate(rows)
        data = np.concatenate(data)
        return coo_matrix((data, (rows, cols)), shape=self._parent.shape)

    @property
    def density(self) -> float:
        """
        Ratio of non-sparse points to total (dense) data points.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.density
        0.5
        """
        tmp = np.mean([column.array.density for _, column in self._parent.items()])
        return tmp

    @staticmethod
    def _prep_index(data, index, columns):
        from pandas.core.indexes.api import (
            default_index,
            ensure_index,
        )

        N, K = data.shape
        if index is None:
            index = default_index(N)
        else:
            index = ensure_index(index)
        if columns is None:
            columns = default_index(K)
        else:
            columns = ensure_index(columns)

        if len(columns) != K:
            raise ValueError(f"Column length mismatch: {len(columns)} vs. {K}")
        if len(index) != N:
            raise ValueError(f"Index length mismatch: {len(index)} vs. {N}")
        return index, columns

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\service_worker.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: ServiceWorker (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target


class RegistrationID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RegistrationID:
        return cls(json)

    def __repr__(self):
        return 'RegistrationID({})'.format(super().__repr__())


@dataclass
class ServiceWorkerRegistration:
    '''
    ServiceWorker registration.
    '''
    registration_id: RegistrationID

    scope_url: str

    is_deleted: bool

    def to_json(self):
        json = dict()
        json['registrationId'] = self.registration_id.to_json()
        json['scopeURL'] = self.scope_url
        json['isDeleted'] = self.is_deleted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            registration_id=RegistrationID.from_json(json['registrationId']),
            scope_url=str(json['scopeURL']),
            is_deleted=bool(json['isDeleted']),
        )


class ServiceWorkerVersionRunningStatus(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ServiceWorkerVersionStatus(enum.Enum):
    NEW = "new"
    INSTALLING = "installing"
    INSTALLED = "installed"
    ACTIVATING = "activating"
    ACTIVATED = "activated"
    REDUNDANT = "redundant"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ServiceWorkerVersion:
    '''
    ServiceWorker version.
    '''
    version_id: str

    registration_id: RegistrationID

    script_url: str

    running_status: ServiceWorkerVersionRunningStatus

    status: ServiceWorkerVersionStatus

    #: The Last-Modified header value of the main script.
    script_last_modified: typing.Optional[float] = None

    #: The time at which the response headers of the main script were received from the server.
    #: For cached script it is the last time the cache entry was validated.
    script_response_time: typing.Optional[float] = None

    controlled_clients: typing.Optional[typing.List[target.TargetID]] = None

    target_id: typing.Optional[target.TargetID] = None

    router_rules: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['versionId'] = self.version_id
        json['registrationId'] = self.registration_id.to_json()
        json['scriptURL'] = self.script_url
        json['runningStatus'] = self.running_status.to_json()
        json['status'] = self.status.to_json()
        if self.script_last_modified is not None:
            json['scriptLastModified'] = self.script_last_modified
        if self.script_response_time is not None:
            json['scriptResponseTime'] = self.script_response_time
        if self.controlled_clients is not None:
            json['controlledClients'] = [i.to_json() for i in self.controlled_clients]
        if self.target_id is not None:
            json['targetId'] = self.target_id.to_json()
        if self.router_rules is not None:
            json['routerRules'] = self.router_rules
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            version_id=str(json['versionId']),
            registration_id=RegistrationID.from_json(json['registrationId']),
            script_url=str(json['scriptURL']),
            running_status=ServiceWorkerVersionRunningStatus.from_json(json['runningStatus']),
            status=ServiceWorkerVersionStatus.from_json(json['status']),
            script_last_modified=float(json['scriptLastModified']) if 'scriptLastModified' in json else None,
            script_response_time=float(json['scriptResponseTime']) if 'scriptResponseTime' in json else None,
            controlled_clients=[target.TargetID.from_json(i) for i in json['controlledClients']] if 'controlledClients' in json else None,
            target_id=target.TargetID.from_json(json['targetId']) if 'targetId' in json else None,
            router_rules=str(json['routerRules']) if 'routerRules' in json else None,
        )


@dataclass
class ServiceWorkerErrorMessage:
    '''
    ServiceWorker error message.
    '''
    error_message: str

    registration_id: RegistrationID

    version_id: str

    source_url: str

    line_number: int

    column_number: int

    def to_json(self):
        json = dict()
        json['errorMessage'] = self.error_message
        json['registrationId'] = self.registration_id.to_json()
        json['versionId'] = self.version_id
        json['sourceURL'] = self.source_url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_message=str(json['errorMessage']),
            registration_id=RegistrationID.from_json(json['registrationId']),
            version_id=str(json['versionId']),
            source_url=str(json['sourceURL']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


def deliver_push_message(
        origin: str,
        registration_id: RegistrationID,
        data: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param data:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['data'] = data
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.deliverPushMessage',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.disable',
    }
    json = yield cmd_dict


def dispatch_sync_event(
        origin: str,
        registration_id: RegistrationID,
        tag: str,
        last_chance: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param tag:
    :param last_chance:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    params['lastChance'] = last_chance
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.dispatchSyncEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_periodic_sync_event(
        origin: str,
        registration_id: RegistrationID,
        tag: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param tag:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.dispatchPeriodicSyncEvent',
        'params': params,
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.enable',
    }
    json = yield cmd_dict


def inspect_worker(
        version_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param version_id:
    '''
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.inspectWorker',
        'params': params,
    }
    json = yield cmd_dict


def set_force_update_on_page_load(
        force_update_on_page_load: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param force_update_on_page_load:
    '''
    params: T_JSON_DICT = dict()
    params['forceUpdateOnPageLoad'] = force_update_on_page_load
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.setForceUpdateOnPageLoad',
        'params': params,
    }
    json = yield cmd_dict


def skip_waiting(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.skipWaiting',
        'params': params,
    }
    json = yield cmd_dict


def start_worker(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.startWorker',
        'params': params,
    }
    json = yield cmd_dict


def stop_all_workers() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.stopAllWorkers',
    }
    json = yield cmd_dict


def stop_worker(
        version_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param version_id:
    '''
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.stopWorker',
        'params': params,
    }
    json = yield cmd_dict


def unregister(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.unregister',
        'params': params,
    }
    json = yield cmd_dict


def update_registration(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.updateRegistration',
        'params': params,
    }
    json = yield cmd_dict


@event_class('ServiceWorker.workerErrorReported')
@dataclass
class WorkerErrorReported:
    error_message: ServiceWorkerErrorMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerErrorReported:
        return cls(
            error_message=ServiceWorkerErrorMessage.from_json(json['errorMessage'])
        )


@event_class('ServiceWorker.workerRegistrationUpdated')
@dataclass
class WorkerRegistrationUpdated:
    registrations: typing.List[ServiceWorkerRegistration]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerRegistrationUpdated:
        return cls(
            registrations=[ServiceWorkerRegistration.from_json(i) for i in json['registrations']]
        )


@event_class('ServiceWorker.workerVersionUpdated')
@dataclass
class WorkerVersionUpdated:
    versions: typing.List[ServiceWorkerVersion]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerVersionUpdated:
        return cls(
            versions=[ServiceWorkerVersion.from_json(i) for i in json['versions']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\service_worker.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: ServiceWorker (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target


class RegistrationID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RegistrationID:
        return cls(json)

    def __repr__(self):
        return 'RegistrationID({})'.format(super().__repr__())


@dataclass
class ServiceWorkerRegistration:
    '''
    ServiceWorker registration.
    '''
    registration_id: RegistrationID

    scope_url: str

    is_deleted: bool

    def to_json(self):
        json = dict()
        json['registrationId'] = self.registration_id.to_json()
        json['scopeURL'] = self.scope_url
        json['isDeleted'] = self.is_deleted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            registration_id=RegistrationID.from_json(json['registrationId']),
            scope_url=str(json['scopeURL']),
            is_deleted=bool(json['isDeleted']),
        )


class ServiceWorkerVersionRunningStatus(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ServiceWorkerVersionStatus(enum.Enum):
    NEW = "new"
    INSTALLING = "installing"
    INSTALLED = "installed"
    ACTIVATING = "activating"
    ACTIVATED = "activated"
    REDUNDANT = "redundant"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ServiceWorkerVersion:
    '''
    ServiceWorker version.
    '''
    version_id: str

    registration_id: RegistrationID

    script_url: str

    running_status: ServiceWorkerVersionRunningStatus

    status: ServiceWorkerVersionStatus

    #: The Last-Modified header value of the main script.
    script_last_modified: typing.Optional[float] = None

    #: The time at which the response headers of the main script were received from the server.
    #: For cached script it is the last time the cache entry was validated.
    script_response_time: typing.Optional[float] = None

    controlled_clients: typing.Optional[typing.List[target.TargetID]] = None

    target_id: typing.Optional[target.TargetID] = None

    router_rules: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['versionId'] = self.version_id
        json['registrationId'] = self.registration_id.to_json()
        json['scriptURL'] = self.script_url
        json['runningStatus'] = self.running_status.to_json()
        json['status'] = self.status.to_json()
        if self.script_last_modified is not None:
            json['scriptLastModified'] = self.script_last_modified
        if self.script_response_time is not None:
            json['scriptResponseTime'] = self.script_response_time
        if self.controlled_clients is not None:
            json['controlledClients'] = [i.to_json() for i in self.controlled_clients]
        if self.target_id is not None:
            json['targetId'] = self.target_id.to_json()
        if self.router_rules is not None:
            json['routerRules'] = self.router_rules
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            version_id=str(json['versionId']),
            registration_id=RegistrationID.from_json(json['registrationId']),
            script_url=str(json['scriptURL']),
            running_status=ServiceWorkerVersionRunningStatus.from_json(json['runningStatus']),
            status=ServiceWorkerVersionStatus.from_json(json['status']),
            script_last_modified=float(json['scriptLastModified']) if 'scriptLastModified' in json else None,
            script_response_time=float(json['scriptResponseTime']) if 'scriptResponseTime' in json else None,
            controlled_clients=[target.TargetID.from_json(i) for i in json['controlledClients']] if 'controlledClients' in json else None,
            target_id=target.TargetID.from_json(json['targetId']) if 'targetId' in json else None,
            router_rules=str(json['routerRules']) if 'routerRules' in json else None,
        )


@dataclass
class ServiceWorkerErrorMessage:
    '''
    ServiceWorker error message.
    '''
    error_message: str

    registration_id: RegistrationID

    version_id: str

    source_url: str

    line_number: int

    column_number: int

    def to_json(self):
        json = dict()
        json['errorMessage'] = self.error_message
        json['registrationId'] = self.registration_id.to_json()
        json['versionId'] = self.version_id
        json['sourceURL'] = self.source_url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_message=str(json['errorMessage']),
            registration_id=RegistrationID.from_json(json['registrationId']),
            version_id=str(json['versionId']),
            source_url=str(json['sourceURL']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


def deliver_push_message(
        origin: str,
        registration_id: RegistrationID,
        data: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param data:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['data'] = data
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.deliverPushMessage',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.disable',
    }
    json = yield cmd_dict


def dispatch_sync_event(
        origin: str,
        registration_id: RegistrationID,
        tag: str,
        last_chance: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param tag:
    :param last_chance:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    params['lastChance'] = last_chance
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.dispatchSyncEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_periodic_sync_event(
        origin: str,
        registration_id: RegistrationID,
        tag: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param tag:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.dispatchPeriodicSyncEvent',
        'params': params,
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.enable',
    }
    json = yield cmd_dict


def inspect_worker(
        version_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param version_id:
    '''
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.inspectWorker',
        'params': params,
    }
    json = yield cmd_dict


def set_force_update_on_page_load(
        force_update_on_page_load: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param force_update_on_page_load:
    '''
    params: T_JSON_DICT = dict()
    params['forceUpdateOnPageLoad'] = force_update_on_page_load
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.setForceUpdateOnPageLoad',
        'params': params,
    }
    json = yield cmd_dict


def skip_waiting(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.skipWaiting',
        'params': params,
    }
    json = yield cmd_dict


def start_worker(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.startWorker',
        'params': params,
    }
    json = yield cmd_dict


def stop_all_workers() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.stopAllWorkers',
    }
    json = yield cmd_dict


def stop_worker(
        version_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param version_id:
    '''
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.stopWorker',
        'params': params,
    }
    json = yield cmd_dict


def unregister(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.unregister',
        'params': params,
    }
    json = yield cmd_dict


def update_registration(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.updateRegistration',
        'params': params,
    }
    json = yield cmd_dict


@event_class('ServiceWorker.workerErrorReported')
@dataclass
class WorkerErrorReported:
    error_message: ServiceWorkerErrorMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerErrorReported:
        return cls(
            error_message=ServiceWorkerErrorMessage.from_json(json['errorMessage'])
        )


@event_class('ServiceWorker.workerRegistrationUpdated')
@dataclass
class WorkerRegistrationUpdated:
    registrations: typing.List[ServiceWorkerRegistration]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerRegistrationUpdated:
        return cls(
            registrations=[ServiceWorkerRegistration.from_json(i) for i in json['registrations']]
        )


@event_class('ServiceWorker.workerVersionUpdated')
@dataclass
class WorkerVersionUpdated:
    versions: typing.List[ServiceWorkerVersion]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerVersionUpdated:
        return cls(
            versions=[ServiceWorkerVersion.from_json(i) for i in json['versions']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\service_worker.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: ServiceWorker (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target


class RegistrationID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RegistrationID:
        return cls(json)

    def __repr__(self):
        return 'RegistrationID({})'.format(super().__repr__())


@dataclass
class ServiceWorkerRegistration:
    '''
    ServiceWorker registration.
    '''
    registration_id: RegistrationID

    scope_url: str

    is_deleted: bool

    def to_json(self):
        json = dict()
        json['registrationId'] = self.registration_id.to_json()
        json['scopeURL'] = self.scope_url
        json['isDeleted'] = self.is_deleted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            registration_id=RegistrationID.from_json(json['registrationId']),
            scope_url=str(json['scopeURL']),
            is_deleted=bool(json['isDeleted']),
        )


class ServiceWorkerVersionRunningStatus(enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ServiceWorkerVersionStatus(enum.Enum):
    NEW = "new"
    INSTALLING = "installing"
    INSTALLED = "installed"
    ACTIVATING = "activating"
    ACTIVATED = "activated"
    REDUNDANT = "redundant"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ServiceWorkerVersion:
    '''
    ServiceWorker version.
    '''
    version_id: str

    registration_id: RegistrationID

    script_url: str

    running_status: ServiceWorkerVersionRunningStatus

    status: ServiceWorkerVersionStatus

    #: The Last-Modified header value of the main script.
    script_last_modified: typing.Optional[float] = None

    #: The time at which the response headers of the main script were received from the server.
    #: For cached script it is the last time the cache entry was validated.
    script_response_time: typing.Optional[float] = None

    controlled_clients: typing.Optional[typing.List[target.TargetID]] = None

    target_id: typing.Optional[target.TargetID] = None

    router_rules: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['versionId'] = self.version_id
        json['registrationId'] = self.registration_id.to_json()
        json['scriptURL'] = self.script_url
        json['runningStatus'] = self.running_status.to_json()
        json['status'] = self.status.to_json()
        if self.script_last_modified is not None:
            json['scriptLastModified'] = self.script_last_modified
        if self.script_response_time is not None:
            json['scriptResponseTime'] = self.script_response_time
        if self.controlled_clients is not None:
            json['controlledClients'] = [i.to_json() for i in self.controlled_clients]
        if self.target_id is not None:
            json['targetId'] = self.target_id.to_json()
        if self.router_rules is not None:
            json['routerRules'] = self.router_rules
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            version_id=str(json['versionId']),
            registration_id=RegistrationID.from_json(json['registrationId']),
            script_url=str(json['scriptURL']),
            running_status=ServiceWorkerVersionRunningStatus.from_json(json['runningStatus']),
            status=ServiceWorkerVersionStatus.from_json(json['status']),
            script_last_modified=float(json['scriptLastModified']) if 'scriptLastModified' in json else None,
            script_response_time=float(json['scriptResponseTime']) if 'scriptResponseTime' in json else None,
            controlled_clients=[target.TargetID.from_json(i) for i in json['controlledClients']] if 'controlledClients' in json else None,
            target_id=target.TargetID.from_json(json['targetId']) if 'targetId' in json else None,
            router_rules=str(json['routerRules']) if 'routerRules' in json else None,
        )


@dataclass
class ServiceWorkerErrorMessage:
    '''
    ServiceWorker error message.
    '''
    error_message: str

    registration_id: RegistrationID

    version_id: str

    source_url: str

    line_number: int

    column_number: int

    def to_json(self):
        json = dict()
        json['errorMessage'] = self.error_message
        json['registrationId'] = self.registration_id.to_json()
        json['versionId'] = self.version_id
        json['sourceURL'] = self.source_url
        json['lineNumber'] = self.line_number
        json['columnNumber'] = self.column_number
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            error_message=str(json['errorMessage']),
            registration_id=RegistrationID.from_json(json['registrationId']),
            version_id=str(json['versionId']),
            source_url=str(json['sourceURL']),
            line_number=int(json['lineNumber']),
            column_number=int(json['columnNumber']),
        )


def deliver_push_message(
        origin: str,
        registration_id: RegistrationID,
        data: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param data:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['data'] = data
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.deliverPushMessage',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.disable',
    }
    json = yield cmd_dict


def dispatch_sync_event(
        origin: str,
        registration_id: RegistrationID,
        tag: str,
        last_chance: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param tag:
    :param last_chance:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    params['lastChance'] = last_chance
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.dispatchSyncEvent',
        'params': params,
    }
    json = yield cmd_dict


def dispatch_periodic_sync_event(
        origin: str,
        registration_id: RegistrationID,
        tag: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param origin:
    :param registration_id:
    :param tag:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['registrationId'] = registration_id.to_json()
    params['tag'] = tag
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.dispatchPeriodicSyncEvent',
        'params': params,
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.enable',
    }
    json = yield cmd_dict


def inspect_worker(
        version_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param version_id:
    '''
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.inspectWorker',
        'params': params,
    }
    json = yield cmd_dict


def set_force_update_on_page_load(
        force_update_on_page_load: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param force_update_on_page_load:
    '''
    params: T_JSON_DICT = dict()
    params['forceUpdateOnPageLoad'] = force_update_on_page_load
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.setForceUpdateOnPageLoad',
        'params': params,
    }
    json = yield cmd_dict


def skip_waiting(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.skipWaiting',
        'params': params,
    }
    json = yield cmd_dict


def start_worker(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.startWorker',
        'params': params,
    }
    json = yield cmd_dict


def stop_all_workers() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.stopAllWorkers',
    }
    json = yield cmd_dict


def stop_worker(
        version_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param version_id:
    '''
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.stopWorker',
        'params': params,
    }
    json = yield cmd_dict


def unregister(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.unregister',
        'params': params,
    }
    json = yield cmd_dict


def update_registration(
        scope_url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param scope_url:
    '''
    params: T_JSON_DICT = dict()
    params['scopeURL'] = scope_url
    cmd_dict: T_JSON_DICT = {
        'method': 'ServiceWorker.updateRegistration',
        'params': params,
    }
    json = yield cmd_dict


@event_class('ServiceWorker.workerErrorReported')
@dataclass
class WorkerErrorReported:
    error_message: ServiceWorkerErrorMessage

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerErrorReported:
        return cls(
            error_message=ServiceWorkerErrorMessage.from_json(json['errorMessage'])
        )


@event_class('ServiceWorker.workerRegistrationUpdated')
@dataclass
class WorkerRegistrationUpdated:
    registrations: typing.List[ServiceWorkerRegistration]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerRegistrationUpdated:
        return cls(
            registrations=[ServiceWorkerRegistration.from_json(i) for i in json['registrations']]
        )


@event_class('ServiceWorker.workerVersionUpdated')
@dataclass
class WorkerVersionUpdated:
    versions: typing.List[ServiceWorkerVersion]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WorkerVersionUpdated:
        return cls(
            versions=[ServiceWorkerVersion.from_json(i) for i in json['versions']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_highlevel_socket.py ===
# "High-level" networking interface
from __future__ import annotations

import errno
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, overload

import trio

from . import socket as tsocket
from ._util import ConflictDetector, final
from .abc import HalfCloseableStream, Listener

if TYPE_CHECKING:
    from collections.abc import Generator

    from typing_extensions import Buffer

    from ._socket import SocketType

# XX TODO: this number was picked arbitrarily. We should do experiments to
# tune it. (Or make it dynamic -- one idea is to start small and increase it
# if we observe single reads filling up the whole buffer, at least within some
# limits.)
DEFAULT_RECEIVE_SIZE = 65536

_closed_stream_errnos = {
    # Unix
    errno.EBADF,
    # Windows
    errno.ENOTSOCK,
}


@contextmanager
def _translate_socket_errors_to_stream_errors() -> Generator[None, None, None]:
    try:
        yield
    except OSError as exc:
        if exc.errno in _closed_stream_errnos:
            raise trio.ClosedResourceError("this socket was already closed") from None
        else:
            raise trio.BrokenResourceError(f"socket connection broken: {exc}") from exc


@final
class SocketStream(HalfCloseableStream):
    """An implementation of the :class:`trio.abc.HalfCloseableStream`
    interface based on a raw network socket.

    Args:
      socket: The Trio socket object to wrap. Must have type ``SOCK_STREAM``,
          and be connected.

    By default for TCP sockets, :class:`SocketStream` enables ``TCP_NODELAY``,
    and (on platforms where it's supported) enables ``TCP_NOTSENT_LOWAT`` with
    a reasonable buffer size (currently 16 KiB) – see `issue #72
    <https://github.com/python-trio/trio/issues/72>`__ for discussion. You can
    of course override these defaults by calling :meth:`setsockopt`.

    Once a :class:`SocketStream` object is constructed, it implements the full
    :class:`trio.abc.HalfCloseableStream` interface. In addition, it provides
    a few extra features:

    .. attribute:: socket

       The Trio socket object that this stream wraps.

    """

    def __init__(self, socket: SocketType) -> None:
        if not isinstance(socket, tsocket.SocketType):
            raise TypeError("SocketStream requires a Trio socket object")
        if socket.type != tsocket.SOCK_STREAM:
            raise ValueError("SocketStream requires a SOCK_STREAM socket")

        self.socket = socket
        self._send_conflict_detector = ConflictDetector(
            "another task is currently sending data on this SocketStream",
        )

        # Socket defaults:

        # Not supported on e.g. unix domain sockets
        with suppress(OSError):
            self.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NODELAY, True)

        if hasattr(tsocket, "TCP_NOTSENT_LOWAT"):
            # 16 KiB is pretty arbitrary and could probably do with some
            # tuning. (Apple is also setting this by default in CFNetwork
            # apparently -- I'm curious what value they're using, though I
            # couldn't find it online trivially. CFNetwork-129.20 source
            # has no mentions of TCP_NOTSENT_LOWAT. This presentation says
            # "typically 8 kilobytes":
            # http://devstreaming.apple.com/videos/wwdc/2015/719ui2k57m/719/719_your_app_and_next_generation_networks.pdf?dl=1
            # ). The theory is that you want it to be bandwidth *
            # rescheduling interval.
            with suppress(OSError):
                self.setsockopt(tsocket.IPPROTO_TCP, tsocket.TCP_NOTSENT_LOWAT, 2**14)

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        if self.socket.did_shutdown_SHUT_WR:
            raise trio.ClosedResourceError("can't send data after sending EOF")
        with self._send_conflict_detector:
            with _translate_socket_errors_to_stream_errors():
                with memoryview(data) as data:
                    if not data:
                        if self.socket.fileno() == -1:
                            raise trio.ClosedResourceError("socket was already closed")
                        await trio.lowlevel.checkpoint()
                        return
                    total_sent = 0
                    while total_sent < len(data):
                        with data[total_sent:] as remaining:
                            sent = await self.socket.send(remaining)
                        total_sent += sent

    async def wait_send_all_might_not_block(self) -> None:
        with self._send_conflict_detector:
            if self.socket.fileno() == -1:
                raise trio.ClosedResourceError
            with _translate_socket_errors_to_stream_errors():
                await self.socket.wait_writable()

    async def send_eof(self) -> None:
        with self._send_conflict_detector:
            await trio.lowlevel.checkpoint()
            # On macOS, calling shutdown a second time raises ENOTCONN, but
            # send_eof needs to be idempotent.
            if self.socket.did_shutdown_SHUT_WR:
                return
            with _translate_socket_errors_to_stream_errors():
                self.socket.shutdown(tsocket.SHUT_WR)

    async def receive_some(self, max_bytes: int | None = None) -> bytes:
        if max_bytes is None:
            max_bytes = DEFAULT_RECEIVE_SIZE
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        with _translate_socket_errors_to_stream_errors():
            return await self.socket.recv(max_bytes)

    async def aclose(self) -> None:
        self.socket.close()
        await trio.lowlevel.checkpoint()

    # __aenter__, __aexit__ inherited from HalfCloseableStream are OK

    @overload
    def setsockopt(self, level: int, option: int, value: int | Buffer) -> None: ...

    @overload
    def setsockopt(self, level: int, option: int, value: None, length: int) -> None: ...

    def setsockopt(
        self,
        level: int,
        option: int,
        value: int | Buffer | None,
        length: int | None = None,
    ) -> None:
        """Set an option on the underlying socket.

        See :meth:`socket.socket.setsockopt` for details.

        """
        if length is None:
            if value is None:
                raise TypeError(
                    "invalid value for argument 'value', must not be None when specifying length",
                )
            return self.socket.setsockopt(level, option, value)
        if value is not None:
            raise TypeError(
                f"invalid value for argument 'value': {value!r}, must be None when specifying optlen",
            )
        return self.socket.setsockopt(level, option, value, length)

    @overload
    def getsockopt(self, level: int, option: int) -> int: ...

    @overload
    def getsockopt(self, level: int, option: int, buffersize: int) -> bytes: ...

    def getsockopt(self, level: int, option: int, buffersize: int = 0) -> int | bytes:
        """Check the current value of an option on the underlying socket.

        See :meth:`socket.socket.getsockopt` for details.

        """
        # This is to work around
        #   https://bitbucket.org/pypy/pypy/issues/2561
        # We should be able to drop it when the next PyPy3 beta is released.
        if buffersize == 0:
            return self.socket.getsockopt(level, option)
        else:
            return self.socket.getsockopt(level, option, buffersize)


################################################################
# SocketListener
################################################################

# Accept error handling
# =====================
#
# Literature review
# -----------------
#
# Here's a list of all the possible errors that accept() can return, according
# to the POSIX spec or the Linux, FreeBSD, macOS, and Windows docs:
#
# Can't happen with a Trio socket:
# - EAGAIN/(WSA)EWOULDBLOCK
# - EINTR
# - WSANOTINITIALISED
# - WSAEINPROGRESS: a blocking call is already in progress
# - WSAEINTR: someone called WSACancelBlockingCall, but we don't make blocking
#   calls in the first place
#
# Something is wrong with our call:
# - EBADF: not a file descriptor
# - (WSA)EINVAL: socket isn't listening, or (Linux, BSD) bad flags
# - (WSA)ENOTSOCK: not a socket
# - (WSA)EOPNOTSUPP: this kind of socket doesn't support accept
# - (Linux, FreeBSD, Windows) EFAULT: the sockaddr pointer points to readonly
#   memory
#
# Something is wrong with the environment:
# - (WSA)EMFILE: this process hit its fd limit
# - ENFILE: the system hit its fd limit
# - (WSA)ENOBUFS, ENOMEM: unspecified memory problems
#
# Something is wrong with the connection we were going to accept. There's a
# ton of variability between systems here:
# - ECONNABORTED: documented everywhere, but apparently only the BSDs do this
#   (signals a connection was closed/reset before being accepted)
# - EPROTO: unspecified protocol error
# - (Linux) EPERM: firewall rule prevented connection
# - (Linux) ENETDOWN, EPROTO, ENOPROTOOPT, EHOSTDOWN, ENONET, EHOSTUNREACH,
#   EOPNOTSUPP, ENETUNREACH, ENOSR, ESOCKTNOSUPPORT, EPROTONOSUPPORT,
#   ETIMEDOUT, ... or any other error that the socket could give, because
#   apparently if an error happens on a connection before it's accept()ed,
#   Linux will report that error from accept().
# - (Windows) WSAECONNRESET, WSAENETDOWN
#
#
# Code review
# -----------
#
# What do other libraries do?
#
# Twisted on Unix or when using nonblocking I/O on Windows:
# - ignores EPERM, with comment about Linux firewalls
# - logs and ignores EMFILE, ENOBUFS, ENFILE, ENOMEM, ECONNABORTED
#   Comment notes that ECONNABORTED is a BSDism and that Linux returns the
#   socket before having it fail, and macOS just silently discards it.
# - other errors are raised, which is logged + kills the socket
# ref: src/twisted/internet/tcp.py, Port.doRead
#
# Twisted using IOCP on Windows:
# - logs and ignores all errors
# ref: src/twisted/internet/iocpreactor/tcp.py, Port.handleAccept
#
# Tornado:
# - ignore ECONNABORTED (comments notes that it was observed on FreeBSD)
# - everything else raised, but all this does (by default) is cause it to be
#   logged and then ignored
# (ref: tornado/netutil.py, tornado/ioloop.py)
#
# libuv on Unix:
# - ignores ECONNABORTED
# - does a "trick" for EMFILE or ENFILE
# - all other errors passed to the connection_cb to be handled
# (ref: src/unix/stream.c:uv__server_io, uv__emfile_trick)
#
# libuv on Windows:
# src/win/tcp.c:uv_tcp_queue_accept
#   this calls AcceptEx, and then arranges to call:
# src/win/tcp.c:uv_process_tcp_accept_req
#   this gets the result from AcceptEx. If the original AcceptEx call failed,
#   then "we stop accepting connections and report this error to the
#   connection callback". I think this is for things like ENOTSOCK. If
#   AcceptEx successfully queues an overlapped operation, and then that
#   reports an error, it's just discarded.
#
# asyncio, selector mode:
# - ignores EWOULDBLOCK, EINTR, ECONNABORTED
# - on EMFILE, ENFILE, ENOBUFS, ENOMEM, logs an error and then disables the
#   listening loop for 1 second
# - everything else raises, but then the event loop just logs and ignores it
# (selector_events.py: BaseSelectorEventLoop._accept_connection)
#
#
# What should we do?
# ------------------
#
# When accept() returns an error, we can either ignore it or raise it.
#
# We have a long list of errors that should be ignored, and a long list of
# errors that should be raised. The big question is what to do with an error
# that isn't on either list. On Linux apparently you can get nearly arbitrary
# errors from accept() and they should be ignored, because it just indicates a
# socket that crashed before it began, and there isn't really anything to be
# done about this, plus on other platforms you may not get any indication at
# all, so programs have to tolerate not getting any indication too. OTOH if we
# get an unexpected error then it could indicate something arbitrarily bad --
# after all, it's unexpected.
#
# Given that we know that other libraries seem to be getting along fine with a
# fairly minimal list of errors to ignore, I think we'll be OK if we write
# down that list and then raise on everything else.
#
# The other question is what to do about the capacity problem errors: EMFILE,
# ENFILE, ENOBUFS, ENOMEM. Just flat out ignoring these is clearly not optimal
# -- at the very least you want to log them, and probably you want to take
# some remedial action. And if we ignore them then it prevents higher levels
# from doing anything clever with them. So we raise them.

_ignorable_accept_errno_names = [
    # Linux can do this when the a connection is denied by the firewall
    "EPERM",
    # BSDs with an early close/reset
    "ECONNABORTED",
    # All the other miscellany noted above -- may not happen in practice, but
    # whatever.
    "EPROTO",
    "ENETDOWN",
    "ENOPROTOOPT",
    "EHOSTDOWN",
    "ENONET",
    "EHOSTUNREACH",
    "EOPNOTSUPP",
    "ENETUNREACH",
    "ENOSR",
    "ESOCKTNOSUPPORT",
    "EPROTONOSUPPORT",
    "ETIMEDOUT",
    "ECONNRESET",
]

# Not all errnos are defined on all platforms
_ignorable_accept_errnos: set[int] = set()
for name in _ignorable_accept_errno_names:
    with suppress(AttributeError):
        _ignorable_accept_errnos.add(getattr(errno, name))


@final
class SocketListener(Listener[SocketStream]):
    """A :class:`~trio.abc.Listener` that uses a listening socket to accept
    incoming connections as :class:`SocketStream` objects.

    Args:
      socket: The Trio socket object to wrap. Must have type ``SOCK_STREAM``,
          and be listening.

    Note that the :class:`SocketListener` "takes ownership" of the given
    socket; closing the :class:`SocketListener` will also close the socket.

    .. attribute:: socket

       The Trio socket object that this stream wraps.

    """

    def __init__(self, socket: SocketType) -> None:
        if not isinstance(socket, tsocket.SocketType):
            raise TypeError("SocketListener requires a Trio socket object")
        if socket.type != tsocket.SOCK_STREAM:
            raise ValueError("SocketListener requires a SOCK_STREAM socket")
        try:
            listening = socket.getsockopt(tsocket.SOL_SOCKET, tsocket.SO_ACCEPTCONN)
        except OSError:
            # SO_ACCEPTCONN fails on macOS; we just have to trust the user.
            pass
        else:
            if not listening:
                raise ValueError("SocketListener requires a listening socket")

        self.socket = socket

    async def accept(self) -> SocketStream:
        """Accept an incoming connection.

        Returns:
          :class:`SocketStream`

        Raises:
          OSError: if the underlying call to ``accept`` raises an unexpected
              error.
          ClosedResourceError: if you already closed the socket.

        This method handles routine errors like ``ECONNABORTED``, but passes
        other errors on to its caller. In particular, it does *not* make any
        special effort to handle resource exhaustion errors like ``EMFILE``,
        ``ENFILE``, ``ENOBUFS``, ``ENOMEM``.

        """
        while True:
            try:
                sock, _ = await self.socket.accept()
            except OSError as exc:
                if exc.errno in _closed_stream_errnos:
                    raise trio.ClosedResourceError from None
                if exc.errno not in _ignorable_accept_errnos:
                    raise
            else:
                return SocketStream(sock)

    async def aclose(self) -> None:
        """Close this listener and its underlying socket."""
        self.socket.close()
        await trio.lowlevel.checkpoint()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\type_checkers.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Provides type checking routines.

This module defines type checking utilities in the forms of dictionaries:

VALUE_CHECKERS: A dictionary of field types and a value validation object.
TYPE_TO_BYTE_SIZE_FN: A dictionary with field types and a size computing
  function.
TYPE_TO_SERIALIZE_METHOD: A dictionary with field types and serialization
  function.
FIELD_TYPE_TO_WIRE_TYPE: A dictionary with field typed and their
  corresponding wire types.
TYPE_TO_DESERIALIZE_METHOD: A dictionary with field types and deserialization
  function.
"""

__author__ = 'robinson@google.com (Will Robinson)'

import struct
import numbers

from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import wire_format
from google.protobuf import descriptor

_FieldDescriptor = descriptor.FieldDescriptor


def TruncateToFourByteFloat(original):
  return struct.unpack('<f', struct.pack('<f', original))[0]


def ToShortestFloat(original):
  """Returns the shortest float that has same value in wire."""
  # All 4 byte floats have between 6 and 9 significant digits, so we
  # start with 6 as the lower bound.
  # It has to be iterative because use '.9g' directly can not get rid
  # of the noises for most values. For example if set a float_field=0.9
  # use '.9g' will print 0.899999976.
  precision = 6
  rounded = float('{0:.{1}g}'.format(original, precision))
  while TruncateToFourByteFloat(rounded) != original:
    precision += 1
    rounded = float('{0:.{1}g}'.format(original, precision))
  return rounded


def GetTypeChecker(field):
  """Returns a type checker for a message field of the specified types.

  Args:
    field: FieldDescriptor object for this field.

  Returns:
    An instance of TypeChecker which can be used to verify the types
    of values assigned to a field of the specified type.
  """
  if (field.cpp_type == _FieldDescriptor.CPPTYPE_STRING and
      field.type == _FieldDescriptor.TYPE_STRING):
    return UnicodeValueChecker()
  if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
    if field.enum_type.is_closed:
      return EnumValueChecker(field.enum_type)
    else:
      # When open enums are supported, any int32 can be assigned.
      return _VALUE_CHECKERS[_FieldDescriptor.CPPTYPE_INT32]
  return _VALUE_CHECKERS[field.cpp_type]


# None of the typecheckers below make any attempt to guard against people
# subclassing builtin types and doing weird things.  We're not trying to
# protect against malicious clients here, just people accidentally shooting
# themselves in the foot in obvious ways.
class TypeChecker(object):

  """Type checker used to catch type errors as early as possible
  when the client is setting scalar fields in protocol messages.
  """

  def __init__(self, *acceptable_types):
    self._acceptable_types = acceptable_types

  def CheckValue(self, proposed_value):
    """Type check the provided value and return it.

    The returned value might have been normalized to another type.
    """
    if not isinstance(proposed_value, self._acceptable_types):
      message = ('%.1024r has type %s, but expected one of: %s' %
                 (proposed_value, type(proposed_value), self._acceptable_types))
      raise TypeError(message)
    return proposed_value


class TypeCheckerWithDefault(TypeChecker):

  def __init__(self, default_value, *acceptable_types):
    TypeChecker.__init__(self, *acceptable_types)
    self._default_value = default_value

  def DefaultValue(self):
    return self._default_value


class BoolValueChecker(object):
  """Type checker used for bool fields."""

  def CheckValue(self, proposed_value):
    if not hasattr(proposed_value, '__index__') or (
        type(proposed_value).__module__ == 'numpy' and
        type(proposed_value).__name__ == 'ndarray'):
      message = ('%.1024r has type %s, but expected one of: %s' %
                 (proposed_value, type(proposed_value), (bool, int)))
      raise TypeError(message)
    return bool(proposed_value)

  def DefaultValue(self):
    return False


# IntValueChecker and its subclasses perform integer type-checks
# and bounds-checks.
class IntValueChecker(object):

  """Checker used for integer fields.  Performs type-check and range check."""

  def CheckValue(self, proposed_value):
    if not hasattr(proposed_value, '__index__') or (
        type(proposed_value).__module__ == 'numpy' and
        type(proposed_value).__name__ == 'ndarray'):
      message = ('%.1024r has type %s, but expected one of: %s' %
                 (proposed_value, type(proposed_value), (int,)))
      raise TypeError(message)

    if not self._MIN <= int(proposed_value) <= self._MAX:
      raise ValueError('Value out of range: %d' % proposed_value)
    # We force all values to int to make alternate implementations where the
    # distinction is more significant (e.g. the C++ implementation) simpler.
    proposed_value = int(proposed_value)
    return proposed_value

  def DefaultValue(self):
    return 0


class EnumValueChecker(object):

  """Checker used for enum fields.  Performs type-check and range check."""

  def __init__(self, enum_type):
    self._enum_type = enum_type

  def CheckValue(self, proposed_value):
    if not isinstance(proposed_value, numbers.Integral):
      message = ('%.1024r has type %s, but expected one of: %s' %
                 (proposed_value, type(proposed_value), (int,)))
      raise TypeError(message)
    if int(proposed_value) not in self._enum_type.values_by_number:
      raise ValueError('Unknown enum value: %d' % proposed_value)
    return proposed_value

  def DefaultValue(self):
    return self._enum_type.values[0].number


class UnicodeValueChecker(object):

  """Checker used for string fields.

  Always returns a unicode value, even if the input is of type str.
  """

  def CheckValue(self, proposed_value):
    if not isinstance(proposed_value, (bytes, str)):
      message = ('%.1024r has type %s, but expected one of: %s' %
                 (proposed_value, type(proposed_value), (bytes, str)))
      raise TypeError(message)

    # If the value is of type 'bytes' make sure that it is valid UTF-8 data.
    if isinstance(proposed_value, bytes):
      try:
        proposed_value = proposed_value.decode('utf-8')
      except UnicodeDecodeError:
        raise ValueError('%.1024r has type bytes, but isn\'t valid UTF-8 '
                         'encoding. Non-UTF-8 strings must be converted to '
                         'unicode objects before being added.' %
                         (proposed_value))
    else:
      try:
        proposed_value.encode('utf8')
      except UnicodeEncodeError:
        raise ValueError('%.1024r isn\'t a valid unicode string and '
                         'can\'t be encoded in UTF-8.'%
                         (proposed_value))

    return proposed_value

  def DefaultValue(self):
    return u""


class Int32ValueChecker(IntValueChecker):
  # We're sure to use ints instead of longs here since comparison may be more
  # efficient.
  _MIN = -2147483648
  _MAX = 2147483647


class Uint32ValueChecker(IntValueChecker):
  _MIN = 0
  _MAX = (1 << 32) - 1


class Int64ValueChecker(IntValueChecker):
  _MIN = -(1 << 63)
  _MAX = (1 << 63) - 1


class Uint64ValueChecker(IntValueChecker):
  _MIN = 0
  _MAX = (1 << 64) - 1


# The max 4 bytes float is about 3.4028234663852886e+38
_FLOAT_MAX = float.fromhex('0x1.fffffep+127')
_FLOAT_MIN = -_FLOAT_MAX
_MAX_FLOAT_AS_DOUBLE_ROUNDED = 3.4028235677973366e38
_INF = float('inf')
_NEG_INF = float('-inf')


class DoubleValueChecker(object):
  """Checker used for double fields.

  Performs type-check and range check.
  """

  def CheckValue(self, proposed_value):
    """Check and convert proposed_value to float."""
    if (not hasattr(proposed_value, '__float__') and
        not hasattr(proposed_value, '__index__')) or (
            type(proposed_value).__module__ == 'numpy' and
            type(proposed_value).__name__ == 'ndarray'):
      message = ('%.1024r has type %s, but expected one of: int, float' %
                 (proposed_value, type(proposed_value)))
      raise TypeError(message)
    return float(proposed_value)

  def DefaultValue(self):
    return 0.0


class FloatValueChecker(DoubleValueChecker):
  """Checker used for float fields.

  Performs type-check and range check.

  Values exceeding a 32-bit float will be converted to inf/-inf.
  """

  def CheckValue(self, proposed_value):
    """Check and convert proposed_value to float."""
    converted_value = super().CheckValue(proposed_value)
    # This inf rounding matches the C++ proto SafeDoubleToFloat logic.
    if converted_value > _FLOAT_MAX:
      if converted_value <= _MAX_FLOAT_AS_DOUBLE_ROUNDED:
        return _FLOAT_MAX
      return _INF
    if converted_value < _FLOAT_MIN:
      if converted_value >= -_MAX_FLOAT_AS_DOUBLE_ROUNDED:
        return _FLOAT_MIN
      return _NEG_INF

    return TruncateToFourByteFloat(converted_value)

# Type-checkers for all scalar CPPTYPEs.
_VALUE_CHECKERS = {
    _FieldDescriptor.CPPTYPE_INT32: Int32ValueChecker(),
    _FieldDescriptor.CPPTYPE_INT64: Int64ValueChecker(),
    _FieldDescriptor.CPPTYPE_UINT32: Uint32ValueChecker(),
    _FieldDescriptor.CPPTYPE_UINT64: Uint64ValueChecker(),
    _FieldDescriptor.CPPTYPE_DOUBLE: DoubleValueChecker(),
    _FieldDescriptor.CPPTYPE_FLOAT: FloatValueChecker(),
    _FieldDescriptor.CPPTYPE_BOOL: BoolValueChecker(),
    _FieldDescriptor.CPPTYPE_STRING: TypeCheckerWithDefault(b'', bytes),
}


# Map from field type to a function F, such that F(field_num, value)
# gives the total byte size for a value of the given type.  This
# byte size includes tag information and any other additional space
# associated with serializing "value".
TYPE_TO_BYTE_SIZE_FN = {
    _FieldDescriptor.TYPE_DOUBLE: wire_format.DoubleByteSize,
    _FieldDescriptor.TYPE_FLOAT: wire_format.FloatByteSize,
    _FieldDescriptor.TYPE_INT64: wire_format.Int64ByteSize,
    _FieldDescriptor.TYPE_UINT64: wire_format.UInt64ByteSize,
    _FieldDescriptor.TYPE_INT32: wire_format.Int32ByteSize,
    _FieldDescriptor.TYPE_FIXED64: wire_format.Fixed64ByteSize,
    _FieldDescriptor.TYPE_FIXED32: wire_format.Fixed32ByteSize,
    _FieldDescriptor.TYPE_BOOL: wire_format.BoolByteSize,
    _FieldDescriptor.TYPE_STRING: wire_format.StringByteSize,
    _FieldDescriptor.TYPE_GROUP: wire_format.GroupByteSize,
    _FieldDescriptor.TYPE_MESSAGE: wire_format.MessageByteSize,
    _FieldDescriptor.TYPE_BYTES: wire_format.BytesByteSize,
    _FieldDescriptor.TYPE_UINT32: wire_format.UInt32ByteSize,
    _FieldDescriptor.TYPE_ENUM: wire_format.EnumByteSize,
    _FieldDescriptor.TYPE_SFIXED32: wire_format.SFixed32ByteSize,
    _FieldDescriptor.TYPE_SFIXED64: wire_format.SFixed64ByteSize,
    _FieldDescriptor.TYPE_SINT32: wire_format.SInt32ByteSize,
    _FieldDescriptor.TYPE_SINT64: wire_format.SInt64ByteSize
    }


# Maps from field types to encoder constructors.
TYPE_TO_ENCODER = {
    _FieldDescriptor.TYPE_DOUBLE: encoder.DoubleEncoder,
    _FieldDescriptor.TYPE_FLOAT: encoder.FloatEncoder,
    _FieldDescriptor.TYPE_INT64: encoder.Int64Encoder,
    _FieldDescriptor.TYPE_UINT64: encoder.UInt64Encoder,
    _FieldDescriptor.TYPE_INT32: encoder.Int32Encoder,
    _FieldDescriptor.TYPE_FIXED64: encoder.Fixed64Encoder,
    _FieldDescriptor.TYPE_FIXED32: encoder.Fixed32Encoder,
    _FieldDescriptor.TYPE_BOOL: encoder.BoolEncoder,
    _FieldDescriptor.TYPE_STRING: encoder.StringEncoder,
    _FieldDescriptor.TYPE_GROUP: encoder.GroupEncoder,
    _FieldDescriptor.TYPE_MESSAGE: encoder.MessageEncoder,
    _FieldDescriptor.TYPE_BYTES: encoder.BytesEncoder,
    _FieldDescriptor.TYPE_UINT32: encoder.UInt32Encoder,
    _FieldDescriptor.TYPE_ENUM: encoder.EnumEncoder,
    _FieldDescriptor.TYPE_SFIXED32: encoder.SFixed32Encoder,
    _FieldDescriptor.TYPE_SFIXED64: encoder.SFixed64Encoder,
    _FieldDescriptor.TYPE_SINT32: encoder.SInt32Encoder,
    _FieldDescriptor.TYPE_SINT64: encoder.SInt64Encoder,
    }


# Maps from field types to sizer constructors.
TYPE_TO_SIZER = {
    _FieldDescriptor.TYPE_DOUBLE: encoder.DoubleSizer,
    _FieldDescriptor.TYPE_FLOAT: encoder.FloatSizer,
    _FieldDescriptor.TYPE_INT64: encoder.Int64Sizer,
    _FieldDescriptor.TYPE_UINT64: encoder.UInt64Sizer,
    _FieldDescriptor.TYPE_INT32: encoder.Int32Sizer,
    _FieldDescriptor.TYPE_FIXED64: encoder.Fixed64Sizer,
    _FieldDescriptor.TYPE_FIXED32: encoder.Fixed32Sizer,
    _FieldDescriptor.TYPE_BOOL: encoder.BoolSizer,
    _FieldDescriptor.TYPE_STRING: encoder.StringSizer,
    _FieldDescriptor.TYPE_GROUP: encoder.GroupSizer,
    _FieldDescriptor.TYPE_MESSAGE: encoder.MessageSizer,
    _FieldDescriptor.TYPE_BYTES: encoder.BytesSizer,
    _FieldDescriptor.TYPE_UINT32: encoder.UInt32Sizer,
    _FieldDescriptor.TYPE_ENUM: encoder.EnumSizer,
    _FieldDescriptor.TYPE_SFIXED32: encoder.SFixed32Sizer,
    _FieldDescriptor.TYPE_SFIXED64: encoder.SFixed64Sizer,
    _FieldDescriptor.TYPE_SINT32: encoder.SInt32Sizer,
    _FieldDescriptor.TYPE_SINT64: encoder.SInt64Sizer,
    }


# Maps from field type to a decoder constructor.
TYPE_TO_DECODER = {
    _FieldDescriptor.TYPE_DOUBLE: decoder.DoubleDecoder,
    _FieldDescriptor.TYPE_FLOAT: decoder.FloatDecoder,
    _FieldDescriptor.TYPE_INT64: decoder.Int64Decoder,
    _FieldDescriptor.TYPE_UINT64: decoder.UInt64Decoder,
    _FieldDescriptor.TYPE_INT32: decoder.Int32Decoder,
    _FieldDescriptor.TYPE_FIXED64: decoder.Fixed64Decoder,
    _FieldDescriptor.TYPE_FIXED32: decoder.Fixed32Decoder,
    _FieldDescriptor.TYPE_BOOL: decoder.BoolDecoder,
    _FieldDescriptor.TYPE_STRING: decoder.StringDecoder,
    _FieldDescriptor.TYPE_GROUP: decoder.GroupDecoder,
    _FieldDescriptor.TYPE_MESSAGE: decoder.MessageDecoder,
    _FieldDescriptor.TYPE_BYTES: decoder.BytesDecoder,
    _FieldDescriptor.TYPE_UINT32: decoder.UInt32Decoder,
    _FieldDescriptor.TYPE_ENUM: decoder.EnumDecoder,
    _FieldDescriptor.TYPE_SFIXED32: decoder.SFixed32Decoder,
    _FieldDescriptor.TYPE_SFIXED64: decoder.SFixed64Decoder,
    _FieldDescriptor.TYPE_SINT32: decoder.SInt32Decoder,
    _FieldDescriptor.TYPE_SINT64: decoder.SInt64Decoder,
    }

# Maps from field type to expected wiretype.
FIELD_TYPE_TO_WIRE_TYPE = {
    _FieldDescriptor.TYPE_DOUBLE: wire_format.WIRETYPE_FIXED64,
    _FieldDescriptor.TYPE_FLOAT: wire_format.WIRETYPE_FIXED32,
    _FieldDescriptor.TYPE_INT64: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_UINT64: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_INT32: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_FIXED64: wire_format.WIRETYPE_FIXED64,
    _FieldDescriptor.TYPE_FIXED32: wire_format.WIRETYPE_FIXED32,
    _FieldDescriptor.TYPE_BOOL: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_STRING:
      wire_format.WIRETYPE_LENGTH_DELIMITED,
    _FieldDescriptor.TYPE_GROUP: wire_format.WIRETYPE_START_GROUP,
    _FieldDescriptor.TYPE_MESSAGE:
      wire_format.WIRETYPE_LENGTH_DELIMITED,
    _FieldDescriptor.TYPE_BYTES:
      wire_format.WIRETYPE_LENGTH_DELIMITED,
    _FieldDescriptor.TYPE_UINT32: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_ENUM: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_SFIXED32: wire_format.WIRETYPE_FIXED32,
    _FieldDescriptor.TYPE_SFIXED64: wire_format.WIRETYPE_FIXED64,
    _FieldDescriptor.TYPE_SINT32: wire_format.WIRETYPE_VARINT,
    _FieldDescriptor.TYPE_SINT64: wire_format.WIRETYPE_VARINT,
    }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\execution_providers\qnn\mixed_precision_overrides_utils.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass

import onnx

from ...quant_utils import QuantType
from ...tensor_quant_overrides import QuantTypeInfo, TensorQuantOverridesHelper


@dataclass
class TensorTypeRequest:
    """
    Bundles desired quantization type requests for a tensor. A distinction is made between the
    produced type and the consumed type.
    """

    # The tensor's quant type at the producer end. If None, assumed to be the default activation quant type.
    producer: QuantTypeInfo | None

    # The tensor's quant type received by a set of consumer nodes.
    # If None, assumed to be the default activation quant type for all consumers.
    # consumers[1] is a set of consumer node names.
    consumers: tuple[QuantTypeInfo, set[str]] | None


class MixedPrecisionTensorQuantOverridesFixer:
    """
    Helper that generates tensor quantization overrides for mixed-precision QDQ models.

    Specifically, this helper fixes an initial set of quantization overrides that assign a non-default
    activation quantization type to one or more tensors by doing the following:
     - Inferring which other tensors need to be overridden to the non-default activation quantization type.
     - Inserting quantization data type conversions.

    Example:
    --------

    Float model:

    input_0 --> Op1 --> Op3 --> Op5 --> Op6 --> output_0
                                 ^
                                 |
    input_1 --> Op2 -+-> Op4 ----+
                     |
                     +-> Op7 --> output_1
                     |
                     +-> Op8 --> output_2

    If we'd like to quantize this model to uint8 precision, but would like to make sure tensor "Op4_out"
    is quantized to 16-bit, then we would specify the following initial tensor quantization overrides:

    ```
    init_overrides = {"Op4_out": [{"quant_type": QuantType.QUInt16}]}
    ```

    These initial overrides may not create a valid model because Op4 and Op5 may require both the input and output
    to be the same type (e.g., uint16). This helper fixes the overrides so that input/output data types
    are valid:

    ```
    overrides = TensorQuantOverridesHelper(init_overrides)

    fixer = MixedPrecisionTensorQuantOverridesFixer.create_from_model(overrides, model, QuantType.QUInt8)
    fixer.apply(
        default_activation_qtype=QuantType.QUInt8,
        default_activation_symmetric=False,
    )
    ```

    The above snippet generates the following "fixed" overrides (get via overrides.get_dict()):

    {
      "Op2_out": [{"quant_type": QUInt8, "convert": {"quant_type": QUInt16, "recv_nodes": {"Op4"}}}],
      "Op3_out": [{"quant_type": QUInt8, "convert": {"quant_type": QUInt16, "recv_nodes": {"Op5"}}}],
      "Op4_out": [{"quant_type": QUInt16}],
      "Op5_out": [{"quant_type": QUInt16, "convert": {"quant_type": QUInt8, "recv_nodes": {"Op6"}}}]
    }

    How to interpret the fixed overrides:
    - Op2's output is consumed by Op4, Op7, and Op8. Op4 consumes the converted u16 type,
      but Op7 and Op8 consume the original u8 type.
    - Op3's output is converted from u8 to u16. Op5 consumes the converted u16 type.
    - Op4's output is just u16 (not converted). All consumers of Op4_out get the u16 type.
    - Op5's output is converted from u16 to u8. Op6 consumes the u8 type.
    """

    def __init__(
        self,
        overrides: TensorQuantOverridesHelper,
        producers: dict[str, onnx.NodeProto],
        consumers: dict[str, list[onnx.NodeProto]],
        value_infos: dict[str, onnx.ValueInfoProto],
        initializers: dict[str, onnx.TensorProto],
    ):
        """
        Params:
            overrides: The initial tensor quantization overrides to fix.
            producers: Dictionary that maps a tensor name to the producer node that generates the tensor.
            consumers: Dictionary that maps a tensor name to the consumer nodes that take the tensor as input.
            value_infos: Dictionary that maps a tensor name to its onnx.ValueInfoProto.
            initializers: Dictionary that maps an initializer name to its onnx.TensorProto.
        """
        self.overrides = overrides
        self.consumers = consumers
        self.producers = producers
        self.value_infos = value_infos
        self.initializers = initializers

    @staticmethod
    def create_from_model(
        overrides: TensorQuantOverridesHelper, model: onnx.ModelProto, default_activation_qtype: QuantType
    ) -> MixedPrecisionTensorQuantOverridesFixer:
        """
        Helper function that creates an instance of this class from a loaded ONNX model.

        Params:
            overrides: The initial tensor quantization overrides to fix.
            model: Loaded ONNX model
            default_activation_qtype: The intended default activation quantization type.
                                      Used to validate the initial overrides.

        Returns:
            Initialized MixedPrecisionTensorQuantOverridesFixer object
        """
        model = onnx.shape_inference.infer_shapes(model)  # Need to infer shapes to get value_infos

        # Build dictionaries that enable convenient lookups of initializers and value_infos by name.
        initializers = {initializer.name: initializer for initializer in model.graph.initializer}
        value_infos = {vi.name: vi for vi in model.graph.value_info}
        value_infos.update({ot.name: ot for ot in model.graph.output})
        value_infos.update({it.name: it for it in model.graph.input})

        # Ensure that the user-provided initial overrides are actually valid.
        valid, err = overrides.is_valid(initializers, set(value_infos), default_activation_qtype)
        if not valid:
            pprint_overrides = overrides.pprint_str(indent=4)
            logging.error(f"Provided invalid tensor quantization overrides:\n{pprint_overrides}")
            raise ValueError(err)

        consumers = {}
        producers = {}

        # Build dictionaries that map a tensor name to the consumer or producer nodes.
        for node in model.graph.node:
            for input_name in node.input:
                if input_name:
                    if input_name not in consumers:
                        consumers[input_name] = []

                    consumers[input_name].append(node)

            for output_name in node.output:
                producers[output_name] = node

        return MixedPrecisionTensorQuantOverridesFixer(overrides, producers, consumers, value_infos, initializers)

    def apply(
        self,
        default_activation_qtype: QuantType,
        default_activation_symmetric: bool,
    ):
        """
        Fixes the initial tensor quantization overrides (in-place) for use in mixed-precision QDQ models.

        Params:
            default_activation_qtype: The intended default activation quantization type.
            default_activation_symmetric: The intended default symmetry used to quantize activations.
        """
        type_requests = self.get_desired_tensor_types(default_activation_qtype, default_activation_symmetric)

        # Use type requests to "fix" tensor quantization overrides by adding
        # quantization type conversions where necessary.
        for tensor_name, type_req in type_requests.items():
            all_consumers = {node.name for node in self.consumers.get(tensor_name, [])}
            has_producer_req = type_req.producer is not None
            has_consumer_req = bool(type_req.consumers)

            # Only producer type: Add conversion back to default activation type
            if has_producer_req and not has_consumer_req:
                self._update_converted_tensor(
                    tensor_name, type_req.producer, QuantTypeInfo(default_activation_qtype), all_consumers
                )
            # Only consumers
            elif not has_producer_req and has_consumer_req:
                prod_type_info = self.overrides.get_node_output_qtype_info(tensor_name, default_activation_qtype)
                consumer_type_info = type_req.consumers[0]

                if prod_type_info != consumer_type_info:
                    self._update_converted_tensor(
                        tensor_name, prod_type_info, consumer_type_info, type_req.consumers[1]
                    )
                else:
                    if not self._check_nodes_are_not_convert_consumers(tensor_name, type_req.consumers[1]):
                        raise ValueError(
                            f"Tensor override for '{tensor_name}' converts the type for consumers that need the original type."
                        )
            # Both producer and consumers
            elif has_producer_req and has_consumer_req:
                prod_type_info = type_req.producer
                consumer_type_info = type_req.consumers[0]

                if prod_type_info != consumer_type_info:
                    self._update_converted_tensor(
                        tensor_name, prod_type_info, consumer_type_info, type_req.consumers[1]
                    )
                else:
                    consumers_for_original_type = all_consumers.difference(type_req.consumers[1])

                    if len(consumers_for_original_type) == 0:
                        # All consumers want the overridden type, so no need for convert nodes!
                        # Just add the override to the new new if not already present.
                        if tensor_name not in self.overrides:
                            self.overrides[tensor_name] = [{}]
                            prod_type_info.save_to_dict(self.overrides[tensor_name][0])

                        assert "convert" not in self.overrides[tensor_name][0]
                    else:
                        # Some consumers don't want the overridden type.
                        self._update_converted_tensor(
                            tensor_name,
                            prod_type_info,
                            QuantTypeInfo(default_activation_qtype),
                            consumers_for_original_type,
                        )
            else:
                raise ValueError(f"TypeRequest for tensor {tensor_name} has no producer or consumers.")

        # Done. Check if the overrides are valid.
        valid, err = self.overrides.is_valid(self.initializers, set(self.value_infos), default_activation_qtype)
        if not valid:
            pprint_overrides = self.overrides.pprint_str(indent=4)
            logging.error(
                f"Generated invalid tensor quantization overrides for mixed-precision QDQ model:\n{pprint_overrides}"
            )
            raise ValueError(err)

    def get_desired_tensor_types(
        self,
        default_activation_qtype: QuantType,
        default_activation_symmetric: bool,
    ) -> dict[str, TensorTypeRequest]:
        """
        Iterates through the initial tensor quantization overrides and builds a set of TensorTypeRequests objects
        that describe the quantization types required at each tensor. These TensorTypeRequests objects are ultimately
        used to generated the "fixed" overrides.

        Params:
            default_activation_qtype: The intended default activation quantization type.
            default_activation_symmetric: The intended default symmetry used to quantize activations.

        Returns:
            TensorTypeRequest objects as a dict that maps a tensor name to its requested types.
        """
        type_requests = {}
        default_activation_type_info = QuantTypeInfo(default_activation_qtype, default_activation_symmetric)

        # Scan tensor overrides for type conversion requests.
        for tensor_name, override_list in self.overrides.items():
            if not self.__is_tensor_quantizable(tensor_name):
                continue  # Skip non-quantizable tensors (e.g., not a float)

            if tensor_name in self.initializers:
                continue  # Skip initializers

            if not override_list or len(override_list) > 1:
                continue  # Skip per-channel stuff

            override_dict = override_list[0]
            quant_type_info = QuantTypeInfo.load_from_dict(override_dict, default_activation_type_info.quant_type)
            producer_node = self.producers.get(tensor_name)  # None if this is a model input

            if quant_type_info != default_activation_type_info and "convert" not in override_dict:
                if producer_node is not None:
                    self._add_type_requests_for_node(type_requests, quant_type_info, producer_node)

                # Find all consumer nodes of `tensor_name` and update their inputs/outputs to the new type.
                for consumer_node in self.consumers.get(tensor_name, []):
                    self._add_type_requests_for_node(type_requests, quant_type_info, consumer_node)

        return type_requests

    def _add_type_requests_for_node(
        self,
        type_requests: dict[str, TensorTypeRequest],
        quant_type_info: QuantTypeInfo,
        node: onnx.NodeProto,
    ):
        """
        Adds TensorTypeRequest objects for a given node, assuming that we want all its inputs and outputs
        to have the same quantization type (as specified by the `quant_type_info` parameter).

        Params:
            type_requests: Dictionary of type requests to append to for this node.
            quant_type_info: The quantization type to use for inputs and outputs.
            node: The node for which the TensorTypeRequest objects are created and added to type_requests.
        """
        # Add output side
        for output_name in node.output:
            if not self.__is_tensor_quantizable(output_name):
                continue

            if output_name not in type_requests:
                type_requests[output_name] = TensorTypeRequest(quant_type_info, None)
            else:
                if (
                    type_requests[output_name].producer is not None
                    and type_requests[output_name].producer != quant_type_info
                ):
                    raise ValueError(f"Tensor {output_name} has multiple types.")

                type_requests[output_name].producer = quant_type_info

        # Add the consumer side
        for input_name in node.input:
            if input_name and input_name not in self.initializers and self.__is_tensor_quantizable(input_name):
                if input_name not in type_requests:
                    type_requests[input_name] = TensorTypeRequest(None, None)

                if type_requests[input_name].consumers is None:
                    type_requests[input_name].consumers = (quant_type_info, set())

                if type_requests[input_name].consumers[0] != quant_type_info:
                    raise ValueError(f"Tensor {input_name} has consumers requesting different types.")

                if not node.name:
                    raise ValueError(
                        f"Node of type {node.op_type} with output 0 {node.output[0]} does not have a name!"
                    )

                type_requests[input_name].consumers[1].add(node.name)

    def _update_converted_tensor(
        self,
        tensor_name: str,
        producer_type_info: QuantTypeInfo,
        consumer_type_info: QuantTypeInfo,
        consumer_names: set[str],
    ):
        """
        Updates the tensor quantization overrides for a tensor that is converted from one type to another.

        Params:
            tensor_name: The name of the tensor for which to update overrides.
            producer_type_info: Info for the tensor's produced type.
            consumer_type_info: Info for the tensor's consumed (i.e., converted) type.
            consumer_names: Nodes names of consumers that consume the converted type.
        """
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            self.overrides[tensor_name] = [{}]
            producer_type_info.save_to_dict(self.overrides[tensor_name][0])

        overrides = self.overrides[tensor_name][0]
        if producer_type_info != QuantTypeInfo.load_from_dict(overrides):
            raise ValueError(f"Desired producer quant_type for {tensor_name} doesn't match existing type.")

        if consumer_names:
            if "convert" not in overrides:
                overrides["convert"] = {}
                consumer_type_info.save_to_dict(overrides["convert"])

            convert_dict = overrides["convert"]
            if consumer_type_info != QuantTypeInfo.load_from_dict(convert_dict):
                raise ValueError(f"Desired consumer quant_type for {tensor_name} doesn't match existing type.")

            if "recv_nodes" not in convert_dict:
                convert_dict["recv_nodes"] = set()

            convert_dict["recv_nodes"].update(consumer_names)

    def _check_nodes_are_not_convert_consumers(self, tensor_name: str, node_names: set[str]):
        """
        Returns true if the given nodes do not consume/receive a converted quantization type.

        Params:
            tensor_name: The name of the tensor to check.
            node_names: Set of node names that should not be consumers of the converted type.
        """
        if tensor_name not in self.overrides or not self.overrides[tensor_name]:
            return True

        overrides = self.overrides[tensor_name][0]

        if "convert" not in overrides:
            return True

        convert_dict = overrides["convert"]

        if "recv_nodes" not in convert_dict:
            return False

        return not convert_dict["recv_nodes"].intersection(node_names)

    def __is_tensor_quantizable(self, tensor_name):
        weight = self.initializers.get(tensor_name)
        if weight is not None:
            if weight.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
                return True
        elif tensor_name in self.value_infos:
            vi = self.value_infos[tensor_name]
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
            ):
                return True

        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\onnx_model_utils.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import logging
import pathlib

import onnx
from onnx import version_converter

import onnxruntime as ort


def iterate_graph_per_node_func(graph, per_node_func, **func_args):
    """
    Iterate the graph including subgraphs calling the per_node_func for each node.
    :param graph: Graph to iterate
    :param per_node_func: Function to call for each node. Signature is fn(node: onnx:NodeProto, **kwargs)
    :param func_args: The keyword args to pass through.
    """

    for node in graph.node:
        per_node_func(node, **func_args)
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField("g"):
                iterate_graph_per_node_func(attr.g, per_node_func, **func_args)


def iterate_graph_per_graph_func(graph, per_graph_func, **func_args):
    """
    Iterate the graph including subgraphs calling the per_graph_func for each Graph.
    :param graph: Graph to iterate
    :param per_graph_func: Function to call for each graph. Signature is fn(graph: onnx:GraphProto, **kwargs)
    :param func_args: The keyword args to pass through.
    """

    per_graph_func(graph, **func_args)

    for node in graph.node:
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField("g"):
                iterate_graph_per_graph_func(attr.g, per_graph_func, **func_args)


def get_opsets_imported(model: onnx.ModelProto):
    """
    Get the opsets imported by the model
    :param model: Model to check.
    :return: Map of domain to opset.
    """
    opsets = {}
    for entry in model.opset_import:
        # if empty it's ai.onnx
        domain = entry.domain or "ai.onnx"
        opsets[domain] = entry.version

    return opsets


def update_onnx_opset(
    model_path: pathlib.Path,
    opset: int,
    out_path: pathlib.Path | None = None,
    logger: logging.Logger | None = None,
):
    """
    Helper to update the opset of a model using onnx version_converter. Target opset must be greater than current opset.
    :param model_path: Path to model to update
    :param opset: Opset to update model to
    :param out_path: Optional output path for updated model to be saved to.
    :param logger: Optional logger for diagnostic output
    :returns: Updated onnx.ModelProto
    """

    model_path_str = str(model_path.resolve(strict=True))
    if logger:
        logger.info("Updating %s to opset %d", model_path_str, opset)

    model = onnx.load(model_path_str)

    new_model = version_converter.convert_version(model, opset)

    if out_path:
        onnx.save(new_model, str(out_path))
        if logger:
            logger.info("Saved updated model to %s", out_path)

    return new_model


def optimize_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    log_level: int = 3,
    use_external_initializers: bool = False,
):
    """
    Optimize an ONNX model using ONNX Runtime to the specified level
    :param model_path: Path to ONNX model
    :param output_path: Path to save optimized model to.
    :param level: onnxruntime.GraphOptimizationLevel to use. Default is ORT_ENABLE_BASIC.
    :param log_level: Log level. Defaults to Error (3) so we don't get output about unused initializers being removed.
                      Warning (2) or Info (1) may be desirable in some scenarios.
    :param use_external_initializers: Set flag to write initializers to an external file. Required if model > 2GB.
                                      Requires onnxruntime 1.17+
    """
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_path.resolve())
    so.graph_optimization_level = level
    so.log_severity_level = log_level

    # save using external initializers so models > 2 GB are handled
    if use_external_initializers:
        major, minor, rest = ort.__version__.split(".", 3)
        if (int(major), int(minor)) >= (1, 17):
            so.add_session_config_entry("session.optimized_model_external_initializers_file_name", "external_data.pb")
        else:
            raise ValueError(
                "ONNX Runtime 1.17 or higher required to save initializers as external data when optimizing model. "
                f"Current ONNX Runtime version is {ort.__version__}"
            )

    # create session to optimize. this will write the updated model to output_path
    _ = ort.InferenceSession(str(model_path.resolve(strict=True)), so, providers=["CPUExecutionProvider"])


def _replace_symbolic_dim_value(graph: onnx.GraphProto, **kwargs):
    param_to_replace = kwargs["dim_param"]
    value = kwargs["value"]

    def update_dim_values(value_infos):
        for vi in value_infos:
            if vi.type.HasField("tensor_type"):
                shape = vi.type.tensor_type.shape
                if shape:
                    for dim in shape.dim:
                        if dim.HasField("dim_param") and dim.dim_param == param_to_replace:
                            dim.Clear()
                            dim.dim_value = value

    update_dim_values(graph.input)
    update_dim_values(graph.output)
    update_dim_values(graph.value_info)


def _remove_invalid_dim_values_impl(graph: onnx.GraphProto):
    def clear_invalid_values(value):
        if value.type.HasField("tensor_type"):
            shape = value.type.tensor_type.shape
            if shape:
                for dim in shape.dim:
                    if dim.HasField("dim_value") and dim.dim_value < 1:
                        dim.Clear()

    for i in graph.input:
        clear_invalid_values(i)

    for o in graph.output:
        clear_invalid_values(o)

    for vi in graph.value_info:
        clear_invalid_values(vi)


def remove_invalid_dim_values(graph: onnx.GraphProto):
    """
    Iterate the graph and subgraphs, unsetting any dim_value entries that have a value of less than 1.
    These are typically erroneously inserted by a converter to represent a dynamic dimension.
    :param graph: GraphProto to update
    """
    iterate_graph_per_graph_func(graph, _remove_invalid_dim_values_impl)


def make_dim_param_fixed(graph: onnx.GraphProto, param_name: str, value: int):
    """
    Iterate all values in the graph, replacing dim_param in a tensor shape with the provided value.
    :param graph: GraphProto to update
    :param param_name: dim_param to set
    :param value: value to use
    """
    iterate_graph_per_graph_func(graph, _replace_symbolic_dim_value, dim_param=param_name, value=value)


def make_input_shape_fixed(graph: onnx.GraphProto, input_name: str, fixed_shape: [int]):
    """
    Update the named graph input to set shape to the provided value. This can be used to set unknown dims as well
    as to replace dim values.
    If setting the input shape replaces a dim_param, update any other values in the graph that use the dim_param.
    :param graph: Graph to update
    :param input_name: Name of graph input to update.
    :param fixed_shape: Shape to use.
    """

    # remove any invalid dim values first. typically this is a dim_value of -1.
    remove_invalid_dim_values(graph)

    for i in graph.input:
        if i.name == input_name:
            if not i.type.HasField("tensor_type"):
                raise ValueError(f"Input {input_name} is not a tensor")

            # graph inputs are required to have a shape to provide the rank
            shape = i.type.tensor_type.shape
            if len(shape.dim) != len(fixed_shape):
                raise ValueError(f"Rank mismatch. Existing:{len(shape.dim)} Replacement:{len(fixed_shape)}")

            for idx, dim in enumerate(shape.dim):
                # check any existing fixed dims match
                if dim.HasField("dim_value"):
                    if dim.dim_value != fixed_shape[idx]:
                        raise ValueError(
                            f"Can't replace existing fixed size of {dim.dim_value} with {fixed_shape[idx]} "
                            f"for dimension {idx + 1}"
                        )
                elif dim.HasField("dim_param"):
                    # replacing a dim_param so have to do that through the entire graph
                    make_dim_param_fixed(graph, dim.dim_param, fixed_shape[idx])
                else:
                    # replacing an unknown dim
                    dim.Clear()
                    dim.dim_value = fixed_shape[idx]

            return

    raise ValueError(
        f"Input {input_name} was not found in graph inputs. "
        f"Valid input names are: {','.join([i.name for i in graph.input])}"
    )


def fix_output_shapes(model: onnx.ModelProto):
    """
    Update the output shapesof a model where the input shape/s were made fixed, if possible.
    This is mainly to make the model usage clearer if the output shapes can be inferred from the new input shapes.
    :param model: Model that had input shapes fixed.
    """

    # get a version of the model with shape inferencing info in it. this will provide fixed output shapes if possible.
    m2 = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(m2)

    for idx, o in enumerate(model.graph.output):
        if not is_fixed_size_tensor(o):
            new_o = m2.graph.output[idx]
            if is_fixed_size_tensor(new_o):
                o.type.tensor_type.shape.CopyFrom(new_o.type.tensor_type.shape)


def _create_producer_consumer_link(
    node_to_producers: dict, node_to_consumers: dict, producer: onnx.NodeProto, consumer: onnx.NodeProto
):
    """
    Create links between two nodes for a value produced by one and consumed by the other.
    :param node_to_producers: Map of NodeProto to set of nodes that produce values the node consumes as inputs.
    :param node_to_consumers: Map of NodeProto to set of nodes that consume values the node produces as outputs.
    :param producer: Producer node
    :param consumer: Consumer node
    """

    if consumer not in node_to_producers:
        node_to_producers[consumer] = set()

    if producer not in node_to_consumers:
        node_to_consumers[producer] = set()

    # add entry mapping this node to the producer of this input
    node_to_producers[consumer].add(producer)
    node_to_consumers[producer].add(consumer)


def _map_node_dependencies(graph: onnx.GraphProto, node_to_producers: dict, node_to_consumers: dict):
    graph_inputs = {i.name for i in graph.input}
    initializers = {i.name for i in graph.initializer}

    # map of value name to node that creates it. copy parent values but override if values get shadowed
    producers = {}

    implicit_inputs = set()

    def is_local_value(value):
        return value in producers or value in initializers or value in graph_inputs

    for node in graph.node:
        inputs = list(node.input)

        for attr in node.attribute:
            if attr.HasField("g"):
                subgraph_implicit_inputs = _map_node_dependencies(attr.g, node_to_producers, node_to_consumers)
                inputs += subgraph_implicit_inputs

        for i in inputs:
            if not i:
                # missing optional input
                continue

            if is_local_value(i):
                if i in producers:
                    producer = producers[i]
                    _create_producer_consumer_link(node_to_producers, node_to_consumers, producer, node)
            else:
                implicit_inputs.add(i)

        for o in node.output:
            producers[o] = node

    return implicit_inputs


def get_producer_consumer_maps(graph: onnx.GraphProto):
    """
    Get maps for connections between the node that produces each value and the nodes that consume the value.
    Processing includes subgraphs. As the map key is a Node instance from the Graph there should be no ambiguity.
    :param graph: Graph to process.
    :return: Tuple with two maps.
             First is node_to_producers map of a node to set of all nodes producing input it consumes.
             Second is node_to_consumers map of a node to set of all nodes consuming output it creates.
             e.g. NodeA and NodeB provide inputs to NodeC. NodeC provides input to NodeD
             node_to_consumers[NodeA] = set([NodeC])
             node_to_consumers[NodeB] = set([NodeC])
             node_to_producers[NodeC] = set([NodeA, NodeB])
             node_to_consumers[NodeC] = set([NodeD])
             node_to_producers[NodeD] = set([NodeC])
    """

    # use a hash of the object id for NodeProto.
    # we need this for the partitioning checker where we keep maps with nodes as the key.
    onnx.NodeProto.__hash__ = lambda self: id(self)

    node_to_producers = {}  # map of node instance to nodes producing input values it consumes
    node_to_consumers = {}  # map of node instance to nodes consuming output values it produces

    implicit_inputs = _map_node_dependencies(graph, node_to_producers, node_to_consumers)

    # top level graph should have no implicit inputs
    if implicit_inputs:
        raise ValueError(
            f"This appears to be an invalid model with missing inputs of {','.join(sorted(implicit_inputs))}"
        )

    return node_to_producers, node_to_consumers


def is_fixed_size_tensor(value: onnx.ValueInfoProto):
    """
    Check if value is a tensor with a fixed shape.
    :param value: onnx.ValueInfoProto to check
    :return: True if value is a tensor, with a shape, where all dimensions have fixed values.
    """

    is_fixed = False
    if value.type.HasField("tensor_type"):
        shape = value.type.tensor_type.shape
        if shape:
            is_fixed = True  # scalar has no dims so set to True and unset if we hit a dim without a valid value
            for dim in shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    continue

                # anything else means it's a dynamic value
                is_fixed = False
                break

    return is_fixed


def get_optimization_level(level):
    """Convert string to GraphOptimizationLevel."""
    if level == "disable":
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if level == "basic":
        # Constant folding and other optimizations that only use ONNX operators
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if level == "extended":
        # Optimizations using custom operators, excluding NCHWc and NHWC layout optimizers
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if level == "all":
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    raise ValueError("Invalid optimization level of " + level)


class ModelProtoWithShapeInfo:
    """
    Class to load an ONNX model and run shape inferencing on it to populate the ValueInfo.
    The model_with_shape_info property will contain the updated model.
    If the model is > 2GB and uses external data a temporary file is required to run shape inferencing successfully.
    This helper class handles automatic removal of the temporary file.
    """

    def __init__(self, model_path: pathlib.Path):
        """
        :param model_path: Path to ONNX model to load and run shape inferencing on.
        """

        self.model_path = model_path

        model = onnx.load(str(model_path))
        self.model_with_shape_info = onnx.shape_inference.infer_shapes(model, strict_mode=True)

        # ONNX has a silent failure from the call to infer_shapes when the model is > 2GB.
        # We detect that by checking the nodes in the returned model.
        self._tmp_model_path = None
        if len(model.graph.node) > 0 and len(self.model_with_shape_info.graph.node) == 0:
            self._tmp_model_path = pathlib.Path(model_path).with_suffix(".temp_with_shapeinf.onnx")
            onnx.shape_inference.infer_shapes_path(str(model_path), str(self._tmp_model_path), strict_mode=True)
            self.model_with_shape_info = onnx.load(str(self._tmp_model_path))

    def __del__(self):
        if self._tmp_model_path:
            self._tmp_model_path.unlink(missing_ok=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\gpt2\benchmark_gpt2.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script benchmarks gpt2 model with past state.
# For gpt2 model without past state, use benchmark.py to measure performance.

import argparse
import csv
import logging
import os
from datetime import datetime

import psutil
import torch
from benchmark_helper import (
    Precision,
    create_onnxruntime_session,
    get_ort_environment_variables,
    prepare_environment,
    setup_logger,
)
from gpt2_helper import DEFAULT_TOLERANCE, MODEL_CLASSES, PRETRAINED_GPT2_MODELS, Gpt2Helper
from packaging import version
from quantize_helper import QuantizeHelper
from transformers import AutoConfig
from transformers import __version__ as transformers_version

logger = logging.getLogger("")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=True,
        type=str,
        help="Model path, or pretrained model name selected in the list: " + ", ".join(PRETRAINED_GPT2_MODELS),
    )

    parser.add_argument(
        "--model_class",
        required=False,
        type=str,
        default="GPT2LMHeadModel",
        choices=list(MODEL_CLASSES.keys()),
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--onnx_dir",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Directory to store onnx models",
    )

    parser.add_argument(
        "--test_times",
        required=False,
        default=100,
        type=int,
        help="Number of repeat times to get average inference latency.",
    )

    parser.add_argument(
        "-v",
        "--validate_onnx",
        required=False,
        action="store_true",
        help="Validate ONNX model",
    )

    parser.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model",
    )
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        required=False,
        choices=[0, 1, 2],
        help="Stage in generation: 1 (initial decoder), 2 (decoder), 0 (both). "
        "1 - decode the first token when past_sequence_length is zero; "
        "2 - decode the remaining tokens when past_sequence_length is not zero; "
        "0 - one onnx model for both stages 1 and 2. "
        "Note that we will optimize 1 and 2 differently for best performance.",
    )

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "-p",
        "--precision",
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. fp32 for full precision, fp16 for half precision, and int8 for quantization",
    )

    parser.add_argument("--torchscript", required=False, action="store_true", help="use Torchscript")
    parser.set_defaults(torchscript=False)

    parser.add_argument("-b", "--batch_sizes", nargs="+", type=int, default=[1], help="batch size")

    parser.add_argument(
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[1],
        help="sequence lengths (excluding past)",
    )

    parser.add_argument(
        "-s",
        "--past_sequence_lengths",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128, 256],
        help="past sequence lengths",
    )

    parser.add_argument(
        "-r",
        "--result_csv",
        required=False,
        default=None,
        help="CSV file for saving summary results.",
    )

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="Threads to use")

    parser.add_argument("--include_copy_output_latency", required=False, action="store_true")
    parser.set_defaults(include_copy_output_latency=False)

    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("--output_torch_latency", required=False, action="store_true")
    parser.set_defaults(output_torch_latency=False)

    parser.add_argument("--disable_io_binding", required=False, action="store_true")
    parser.set_defaults(disable_io_binding=False)

    args = parser.parse_args(argv)

    return args


def main(args):
    if version.parse(transformers_version) < version.parse(
        "3.1.0"
    ):  # past_key_values name does not exist in 3.0.2 or older
        raise RuntimeError("This tool requires transformers 3.1.0 or later.")

    logger.info(f"Arguments:{args}")
    if args.precision == Precision.FLOAT16:
        assert args.optimize_onnx and args.use_gpu, "fp16 requires --optimize_onnx --use_gpu"

    if args.precision == Precision.INT8:
        assert not args.use_gpu, "quantization only supports CPU"

    if args.stage == 1:
        assert args.past_sequence_lengths == [0], "past_sequence_lengths shall be 0 for stage==1 (init decoder)"

    torch.set_num_threads(psutil.cpu_count(logical=True) if args.thread_num <= 0 else args.thread_num)
    print(torch.__config__.parallel_info())

    cache_dir = args.cache_dir
    output_dir = args.onnx_dir
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    model_class = MODEL_CLASSES[args.model_class][0]
    gpt2helper = Gpt2Helper
    config = AutoConfig.from_pretrained(args.model_name_or_path, torchscript=args.torchscript, cache_dir=cache_dir)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=cache_dir)

    # This script does not support float16 for PyTorch.
    # if args.float16:
    #    model.half()

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)
    use_external_data_format = config.n_layer > 24  # TODO: find a way to check model size > 2GB
    onnx_model_paths = gpt2helper.get_onnx_paths(
        output_dir,
        args.model_name_or_path,
        args.model_class,
        has_past=True,
        new_folder=use_external_data_format,
    )

    onnx_model_path = onnx_model_paths["raw"]
    use_padding = MODEL_CLASSES[args.model_class][2]
    gpt2helper.export_onnx(
        model,
        device,
        onnx_model_path,
        args.verbose,
        use_external_data_format,
        has_position_ids=use_padding,
        has_attention_mask=use_padding,
    )

    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        onnx_model_path = onnx_model_paths[str(args.precision) if args.precision != Precision.INT8 else "fp32"]
        gpt2helper.optimize_onnx(
            onnx_model_paths["raw"],
            onnx_model_path,
            args.precision == Precision.FLOAT16,
            model.config.num_attention_heads,
            model.config.hidden_size,
            use_external_data_format,
            auto_mixed_precision=True,
            stage=args.stage,
        )

        if args.precision == Precision.INT8:
            logger.info("quantizing model...")
            QuantizeHelper.quantize_onnx_model(onnx_model_path, onnx_model_paths["int8"], use_external_data_format)
            model = QuantizeHelper.quantize_torch_model(model)
            logger.info("finished quantizing model")
            onnx_model_path = onnx_model_paths["int8"]

    if args.torchscript:
        model = gpt2helper.torchscript(
            model,
            config,
            device,
            has_position_ids=use_padding,
            has_attention_mask=use_padding,
        )

    session = create_onnxruntime_session(
        onnx_model_path,
        args.use_gpu,
        enable_all_optimization=False,
        num_threads=args.thread_num,
        verbose=args.verbose,
    )
    if session is None:
        return

    # Allocate output buffers for IO Binding
    max_output_shapes = gpt2helper.get_output_shapes(
        max(args.batch_sizes),
        max(args.past_sequence_lengths),
        max(args.sequence_lengths),
        config,
        args.model_class,
    )
    output_buffers = gpt2helper.get_output_buffers(max_output_shapes, device, args.precision == Precision.FLOAT16)

    csv_filename = args.result_csv or "benchmark_result_{}.csv".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_filename, mode="a", newline="") as csv_file:
        column_names = [
            "model_name",
            "model_class",
            "stage",
            "environment_variables",
            "gpu",
            "precision",
            "optimizer",
            "torchscript",
            "batch_size",
            "sequence_length",
            "past_sequence_length",
            "disable_io_binding",
            "torch_latency",
            "onnxruntime_latency",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        for batch_size in args.batch_sizes:
            for sequence_length in args.sequence_lengths:
                for past_sequence_length in args.past_sequence_lengths:
                    assert batch_size > 0 and sequence_length > 0 and past_sequence_length >= 0
                    logger.debug(
                        "Running test for batch_size=%d sequence_length=%d past_sequence_length=%d ...",
                        batch_size,
                        sequence_length,
                        past_sequence_length,
                    )

                    dummy_inputs = gpt2helper.get_dummy_inputs(
                        batch_size,
                        past_sequence_length,
                        sequence_length,
                        config.num_attention_heads,
                        config.hidden_size,
                        config.n_layer,
                        config.vocab_size,
                        device,
                        float16=(args.precision == Precision.FLOAT16),
                        has_position_ids=use_padding,
                        has_attention_mask=use_padding,
                    )
                    output_shapes = gpt2helper.get_output_shapes(
                        batch_size,
                        past_sequence_length,
                        sequence_length,
                        config,
                        args.model_class,
                    )

                    try:
                        if args.validate_onnx or args.output_torch_latency:
                            outputs, torch_latency = gpt2helper.pytorch_inference(model, dummy_inputs, args.test_times)

                            # Dump Torch output shape
                            for i, value in enumerate(outputs):
                                if isinstance(value, tuple):
                                    logger.debug(
                                        f"torch output {i} is tuple of size {len(value)}, shape {value[0].shape}"
                                    )
                                else:
                                    logger.debug(f"torch output {i} shape {value.shape}")
                        else:
                            outputs = None
                            torch_latency = None

                        if args.disable_io_binding:
                            ort_outputs, ort_latency = gpt2helper.onnxruntime_inference(
                                session, dummy_inputs, args.test_times
                            )
                        else:
                            ort_outputs, ort_latency = gpt2helper.onnxruntime_inference_with_binded_io(
                                session,
                                dummy_inputs,
                                output_buffers,
                                output_shapes,
                                args.test_times,
                                return_numpy=False,
                                include_copy_output_latency=args.include_copy_output_latency,
                            )

                        if args.validate_onnx:
                            copy_outputs = ort_outputs
                            if not args.disable_io_binding:
                                # Results of IO binding might be in GPU. Copy outputs to CPU for comparison.
                                copy_outputs = []
                                for output in ort_outputs:
                                    copy_outputs.append(output.cpu().numpy())

                            if gpt2helper.compare_outputs(
                                outputs,
                                copy_outputs,
                                model_class=args.model_class,
                                rtol=DEFAULT_TOLERANCE[args.precision],
                                atol=DEFAULT_TOLERANCE[args.precision],
                            ):
                                logger.info(
                                    f"Pytorch and ONNX Runtime outputs are all close (tolerance={DEFAULT_TOLERANCE[args.precision]})."
                                )

                        logger.info(
                            "batch_size=%d, sequence_length=%d, past_sequence_length=%d, onnxruntime_latency=%.2f %s %s",
                            batch_size,
                            sequence_length,
                            past_sequence_length,
                            ort_latency,
                            "(disable_io_binding)" if args.disable_io_binding else "",
                            ", torch_latency={torch_latency}" if torch_latency else "",
                        )

                        row = {
                            "model_name": args.model_name_or_path,
                            "model_class": args.model_class,
                            "stage": args.stage,
                            "environment_variables": get_ort_environment_variables(),
                            "gpu": args.use_gpu,
                            "precision": args.precision,
                            "optimizer": args.optimize_onnx,
                            "torchscript": args.torchscript,
                            "batch_size": batch_size,
                            "sequence_length": sequence_length,
                            "past_sequence_length": past_sequence_length,
                            "disable_io_binding": args.disable_io_binding,
                            "torch_latency": f"{torch_latency:.2f}" if torch_latency else "None",
                            "onnxruntime_latency": f"{ort_latency:.2f}",
                        }
                        csv_writer.writerow(row)
                    except Exception:
                        logger.error("Exception", exc_info=True)  # noqa: G201
                        return None

    logger.info(f"Results are saved to file {csv_filename}")
    return csv_filename


if __name__ == "__main__":
    args = parse_arguments()
    setup_logger(args.verbose)
    main(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\longformer\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This script converts Longformer model from huggingface transformers 4.0 or later to ONNX.
# It translates LongformerSelfAttention to the LongformerAttention operator in ONNX Runtime.
#
# Before running this script, prepare a python environment in Linux with PyTorch 1.9.0 and other packages installed.
# Then run "python setup.py install" in ./torch_extensions directory. If your python version is not 3.8, you will need
# update this script with correct name of longformer_attention.cpython-*.so (search TODO below).
#
# It is tested in Ubuntu 18.04 with python 3.8, onnxruntime-gpu 1.11.0, PyTorch 1.9.0, transformers 4.18.0.
# Warning: Using  PyTorch 1.10 or newer version might encounter issue in exporting, but they are fine for benchmarking.
#
# Example commands to export longformer base model in Linux:
#   conda create -n longformer python=3.8
#   conda activate longformer
#   python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#   python3 -m pip install coloredlogs flatbuffers numpy packaging sympy protobuf==3.20.1 onnx==1.12.0 transformers==4.18.0
#   python3 -m pip install -i https://test.pypi.org/simple/ ort-nightly-gpu
#   cd ./torch_extensions
#   rm -rf build
#   python setup.py install
#   cd ..
#   python convert_to_onnx.py --model longformer-base-4096 --precision fp16 --optimize_onnx
#   python convert_to_onnx.py --model longformer-base-4096 --precision fp16 --optimize_onnx --no_merge_qkv
#
# GPU is not needed for this script. You can run it in CPU. For --optimize_onnx, you can use either onnxruntime or onnxruntime-gpu package.
#
# For inference of the onnx model, you will need onnxruntime-gpu 1.7.0 or newer version.

import argparse
import inspect
from pathlib import Path

import torch
import transformers
from longformer_helper import PRETRAINED_LONGFORMER_MODELS
from onnx import load_model
from onnx_model_bert import BertOnnxModel
from packaging import version
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch_onnx_export_helper import torch_onnx_export
from transformers import LongformerModel, LongformerSelfAttention

# Supports format 0 or 1
weight_bias_format = 0


@parse_args("v", "v", "v", "v", "v", "v", "v", "i", "i")
def my_longformer_attention(
    g,
    input,
    weight,
    bias,
    mask,
    global_weight,
    global_bias,
    global_mask,
    num_heads,
    window,
):
    return g.op(
        "com.microsoft::LongformerAttention",
        input,
        weight,
        bias,
        mask,
        global_weight,
        global_bias,
        global_mask,
        num_heads_i=num_heads,
        window_i=window,
    )


# namespace is onnxruntime which is registered in longformer_attention.cpp
register_custom_op_symbolic("onnxruntime::LongformerAttention", my_longformer_attention, 9)

# TODO: search the directory to find correct output filename of "python setup.py install" when python version is not 3.8
torch.ops.load_library(
    r"./torch_extensions/build/lib.linux-x86_64-3.8/longformer_attention.cpython-38-x86_64-linux-gnu.so"
)


def parse_arguments():
    """Parse arguments

    Returns:
        args: Namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        type=str,
        default="longformer-base-4096",
        help="Checkpoint directory or pre-trained model names in the list: "
        + ", ".join(PRETRAINED_LONGFORMER_MODELS.keys()),
    )

    parser.add_argument(
        "--export_padding",
        required=False,
        action="store_true",
        help="Export padding logic to ONNX graph. If not enabled, user need pad input so that sequence length is multiple of window size.",
    )
    parser.set_defaults(export_padding=False)

    parser.add_argument(
        "--no_merge_qkv",
        required=False,
        action="store_true",
        help="Stack the weights of q, k and v on dimension 0 instead of dimension 1.",
    )
    parser.set_defaults(no_merge_qkv=False)

    parser.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model.",
    )
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision of model to run: fp32 for full precision, fp16 for mixed precision",
    )

    args = parser.parse_args()
    return args


# Create a dummy input for ONNX export.
def get_dummy_inputs(config, export_padding, device):
    # When sequence length is multiple of windows size, there is no padding logic in ONNX graph
    sequence_length = config.attention_window[0] + 1 if export_padding else config.attention_window[0]

    # Create dummy inputs
    input_ids = torch.arange(sequence_length).unsqueeze(0).to(device)

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    attention_mask[:, sequence_length - 1] = 0  # last token is masked

    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=device)
    global_attention_mask[:, 0] = 1  # first token is global token

    return input_ids, attention_mask, global_attention_mask


# A new function to replace LongformerSelfAttention.forward
# For transformers 4.0.0
def my_longformer_self_attention_forward_4(
    self,
    hidden_states,
    attention_mask=None,
    is_index_masked=None,
    is_index_global_attn=None,
    is_global_attn=None,
):
    global_mask = is_index_global_attn.int()
    # The following check is based on the dummy inputs (only the first token is global).
    assert (
        len(global_mask.shape) == 2
        and global_mask.shape[0] == 1
        and global_mask.count_nonzero().item() == 1
        and global_mask.tolist()[0][0] == 1
    )

    input_mask = is_index_masked.float()
    # TODO: The filtering value may be -10000.0 or -inf. Check the huggingface implementation.
    input_mask = input_mask.masked_fill(is_index_masked, -10000.0)
    # Yet another way to generate input_mask = torch.masked_fill(attention_mask, is_index_global_attn, 0.0)

    # TODO: add postprocessing of ONNX model to calculate based on graph input: input_mask = (attention_mask - 1) * 10000.0
    # TODO: add postprocessing of ONNX model to use graph input directly: global_mask = global_attention_mask

    # The following check is based on the dummy inputs (only the last token is masked).
    assert (
        len(input_mask.shape) == 2
        and input_mask.shape[0] == 1
        and input_mask.count_nonzero().item() == 1
        and input_mask.tolist()[0][-1] == -10000.0
    )

    weight = torch.stack(
        (
            self.query.weight.transpose(0, 1),
            self.key.weight.transpose(0, 1),
            self.value.weight.transpose(0, 1),
        ),
        dim=weight_bias_format,
    )

    if weight_bias_format == 1:
        # shape is (hidden_size, 3*hidden_size) for format 1, otherwise (3, hidden_size, hidden_size) by default
        weight = weight.reshape(self.embed_dim, 3 * self.embed_dim)

    global_weight = torch.stack(
        (
            self.query_global.weight.transpose(0, 1),
            self.key_global.weight.transpose(0, 1),
            self.value_global.weight.transpose(0, 1),
        ),
        dim=weight_bias_format,
    )

    if weight_bias_format == 1:
        global_weight = global_weight.reshape(self.embed_dim, 3 * self.embed_dim)

    if weight_bias_format == 1:
        bias = torch.stack((self.query.bias, self.key.bias, self.value.bias), dim=0)
        bias = bias.reshape(3 * self.embed_dim)
        global_bias = torch.stack((self.query_global.bias, self.key_global.bias, self.value_global.bias), dim=0)
        global_bias = global_bias.reshape(3 * self.embed_dim)
    else:
        bias = torch.stack(
            (self.query.bias, self.key.bias, self.value.bias, self.key_global.bias, self.value_global.bias), dim=0
        )
        bias = bias.reshape(5 * self.embed_dim)
        global_bias = self.query_global.bias
        global_bias = global_bias.reshape(1 * self.embed_dim)

    attn_output = torch.ops.onnxruntime.LongformerAttention(
        hidden_states,
        weight,
        bias,
        input_mask,
        global_weight,
        global_bias,
        global_mask,
        self.num_heads,
        self.one_sided_attn_window_size,
    )

    assert attn_output.size() == hidden_states.size(), "Unexpected size"

    outputs = (attn_output,)
    return outputs


# For transformers 4.3.0
def my_longformer_self_attention_forward_4_3(
    self,
    hidden_states,
    attention_mask=None,
    is_index_masked=None,
    is_index_global_attn=None,
    is_global_attn=None,
    output_attentions=False,
):
    assert output_attentions is False
    return my_longformer_self_attention_forward_4(
        self,
        hidden_states,
        attention_mask,
        is_index_masked,
        is_index_global_attn,
        is_global_attn,
    )


# For transformers 4.3.2 or later versions
def my_longformer_self_attention_forward_4_3_2(
    self,
    hidden_states,
    attention_mask=None,
    layer_head_mask=None,
    is_index_masked=None,
    is_index_global_attn=None,
    is_global_attn=None,
    output_attentions=False,
):
    assert output_attentions is False
    assert layer_head_mask is None
    return my_longformer_self_attention_forward_4(
        self,
        hidden_states,
        attention_mask,
        is_index_masked,
        is_index_global_attn,
        is_global_attn,
    )


def export_longformer(model: LongformerModel, onnx_model_path: str, export_padding: bool):
    """Export longformer model to ONNX

    Args:
        model (LongformerModel): longformer model
        onnx_model_path (str): output onnx path
        export_padding (bool): whether export padding logic to ONNX so that input string can be any length.

    Raises:
        RuntimeError: This tool requires transformers 4.0.0 or later.
        RuntimeError: LongformerSelfAttention.forward arguments are different.
    """
    input_ids, attention_mask, global_attention_mask = get_dummy_inputs(
        model.config, export_padding, device=torch.device("cpu")
    )

    _ = model(
        input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
    )

    if version.parse(transformers.__version__) < version.parse("4.0.0"):
        raise RuntimeError("This tool requires transformers 4.0.0 or later.")

    # Here we replace LongformerSelfAttention.forward using our implementation for exporting ONNX model
    key = " ".join(inspect.getfullargspec(LongformerSelfAttention.forward).args)
    args_to_func = {
        "self hidden_states attention_mask layer_head_mask is_index_masked is_index_global_attn is_global_attn output_attentions": my_longformer_self_attention_forward_4_3_2,
        "self hidden_states attention_mask is_index_masked is_index_global_attn is_global_attn output_attentions": my_longformer_self_attention_forward_4_3,
        "self hidden_states attention_mask is_index_masked is_index_global_attn is_global_attn": my_longformer_self_attention_forward_4,
    }

    if key not in args_to_func:
        print(
            "Current arguments",
            inspect.getfullargspec(LongformerSelfAttention.forward).args,
        )
        raise RuntimeError(
            "LongformerSelfAttention.forward arguments are different. Please install supported version (like transformers 4.3.0)."
        )

    # Store for restoring later
    original_forward = LongformerSelfAttention.forward

    LongformerSelfAttention.forward = args_to_func[key]

    example_inputs = (input_ids, attention_mask, global_attention_mask)

    Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

    torch_onnx_export(
        model,
        example_inputs,
        onnx_model_path,
        opset_version=12,
        input_names=["input_ids", "attention_mask", "global_attention_mask"],
        output_names=["last_state", "pooler"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "global_attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_state": {0: "batch_size", 1: "sequence_length"},
            "pooler": {0: "batch_size", 1: "sequence_length"},
        },
        custom_opsets={"com.microsoft": 1},
    )
    print(f"ONNX model exported to {onnx_model_path}")

    # Restore original implementation:
    LongformerSelfAttention.forward = original_forward


def optimize_longformer(onnx_model_path: str, fp32_model_path: str, fp16_model_path=None):
    """Optimize longformer onnx model

    Args:
        onnx_model_path (str): path of original ONNX model.
        fp32_model_path (str): path of optimized fp32 model.
        fp16_model_path (str, optional): path of optimized fp16 model. Defaults to None.
    """
    model = load_model(onnx_model_path, format=None, load_external_data=True)
    optimizer = BertOnnxModel(model)
    optimizer.optimize()

    use_external_data_format = False
    if fp32_model_path:
        optimizer.save_model_to_file(fp32_model_path, use_external_data_format)
        print(f"optimized fp32 model saved to {fp32_model_path}")

    if fp16_model_path:
        optimizer.convert_float_to_float16(keep_io_types=True)
        optimizer.save_model_to_file(fp16_model_path, use_external_data_format)
        print(f"optimized fp16 model saved to {fp16_model_path}")


def main(args):
    model_name = args.model
    onnx_model_path = model_name + ".onnx"

    global weight_bias_format  # noqa: PLW0603
    weight_bias_format = 0 if args.no_merge_qkv else 1

    model = LongformerModel.from_pretrained(PRETRAINED_LONGFORMER_MODELS[model_name])

    export_longformer(model, onnx_model_path, args.export_padding)

    if args.optimize_onnx or args.precision != "fp32":
        fp32_model_path = model_name + f"_f{weight_bias_format}" + "_fp32.onnx"
        fp16_model_path = model_name + f"_f{weight_bias_format}" + "_fp16.onnx" if args.precision == "fp16" else None
        optimize_longformer(onnx_model_path, fp32_model_path, fp16_model_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\multipledispatch\dispatcher.py ===
from __future__ import annotations

from warnings import warn
import inspect
from .conflict import ordering, ambiguities, super_signature, AmbiguityWarning
from .utils import expand_tuples
import itertools as itl


class MDNotImplementedError(NotImplementedError):
    """ A NotImplementedError for multiple dispatch """


### Functions for on_ambiguity

def ambiguity_warn(dispatcher, ambiguities):
    """ Raise warning when ambiguity is detected

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also:
        Dispatcher.add
        warning_text
    """
    warn(warning_text(dispatcher.name, ambiguities), AmbiguityWarning)


class RaiseNotImplementedError:
    """Raise ``NotImplementedError`` when called."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        types = tuple(type(a) for a in args)
        raise NotImplementedError(
            "Ambiguous signature for %s: <%s>" % (
            self.dispatcher.name, str_signature(types)
        ))

def ambiguity_register_error_ignore_dup(dispatcher, ambiguities):
    """
    If super signature for ambiguous types is duplicate types, ignore it.
    Else, register instance of ``RaiseNotImplementedError`` for ambiguous types.

    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher

    See Also:
        Dispatcher.add
        ambiguity_warn
    """
    for amb in ambiguities:
        signature = tuple(super_signature(amb))
        if len(set(signature)) == 1:
            continue
        dispatcher.add(
            signature, RaiseNotImplementedError(dispatcher),
            on_ambiguity=ambiguity_register_error_ignore_dup
        )

###


_unresolved_dispatchers: set[Dispatcher] = set()
_resolve = [True]


def halt_ordering():
    _resolve[0] = False


def restart_ordering(on_ambiguity=ambiguity_warn):
    _resolve[0] = True
    while _unresolved_dispatchers:
        dispatcher = _unresolved_dispatchers.pop()
        dispatcher.reorder(on_ambiguity=on_ambiguity)


class Dispatcher:
    """ Dispatch methods based on type signature

    Use ``dispatch`` to add implementations

    Examples
    --------

    >>> from sympy.multipledispatch import dispatch
    >>> @dispatch(int)
    ... def f(x):
    ...     return x + 1

    >>> @dispatch(float)
    ... def f(x): # noqa: F811
    ...     return x - 1

    >>> f(3)
    4
    >>> f(3.0)
    2.0
    """
    __slots__ = '__name__', 'name', 'funcs', 'ordering', '_cache', 'doc'

    def __init__(self, name, doc=None):
        self.name = self.__name__ = name
        self.funcs = {}
        self._cache = {}
        self.ordering = []
        self.doc = doc

    def register(self, *types, **kwargs):
        """ Register dispatcher with new implementation

        >>> from sympy.multipledispatch.dispatcher import Dispatcher
        >>> f = Dispatcher('f')
        >>> @f.register(int)
        ... def inc(x):
        ...     return x + 1

        >>> @f.register(float)
        ... def dec(x):
        ...     return x - 1

        >>> @f.register(list)
        ... @f.register(tuple)
        ... def reverse(x):
        ...     return x[::-1]

        >>> f(1)
        2

        >>> f(1.0)
        0.0

        >>> f([1, 2, 3])
        [3, 2, 1]
        """
        def _(func):
            self.add(types, func, **kwargs)
            return func
        return _

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return sig.parameters.values()

    @classmethod
    def get_func_annotations(cls, func):
        """ Get annotations of function positional parameters
        """
        params = cls.get_func_params(func)
        if params:
            Parameter = inspect.Parameter

            params = (param for param in params
                      if param.kind in
                      (Parameter.POSITIONAL_ONLY,
                       Parameter.POSITIONAL_OR_KEYWORD))

            annotations = tuple(
                param.annotation
                for param in params)

            if not any(ann is Parameter.empty for ann in annotations):
                return annotations

    def add(self, signature, func, on_ambiguity=ambiguity_warn):
        """ Add new types/method pair to dispatcher

        >>> from sympy.multipledispatch import Dispatcher
        >>> D = Dispatcher('add')
        >>> D.add((int, int), lambda x, y: x + y)
        >>> D.add((float, float), lambda x, y: x + y)

        >>> D(1, 2)
        3
        >>> D(1, 2.0)
        Traceback (most recent call last):
        ...
        NotImplementedError: Could not find signature for add: <int, float>

        When ``add`` detects a warning it calls the ``on_ambiguity`` callback
        with a dispatcher/itself, and a set of ambiguous type signature pairs
        as inputs.  See ``ambiguity_warn`` for an example.
        """
        # Handle annotations
        if not signature:
            annotations = self.get_func_annotations(func)
            if annotations:
                signature = annotations

        # Handle union types
        if any(isinstance(typ, tuple) for typ in signature):
            for typs in expand_tuples(signature):
                self.add(typs, func, on_ambiguity)
            return

        for typ in signature:
            if not isinstance(typ, type):
                str_sig = ', '.join(c.__name__ if isinstance(c, type)
                                    else str(c) for c in signature)
                raise TypeError("Tried to dispatch on non-type: %s\n"
                                "In signature: <%s>\n"
                                "In function: %s" %
                                (typ, str_sig, self.name))

        self.funcs[signature] = func
        self.reorder(on_ambiguity=on_ambiguity)
        self._cache.clear()

    def reorder(self, on_ambiguity=ambiguity_warn):
        if _resolve[0]:
            self.ordering = ordering(self.funcs)
            amb = ambiguities(self.funcs)
            if amb:
                on_ambiguity(self, amb)
        else:
            _unresolved_dispatchers.add(self)

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if not func:
                raise NotImplementedError(
                    'Could not find signature for %s: <%s>' %
                    (self.name, str_signature(types)))
            self._cache[types] = func
        try:
            return func(*args, **kwargs)

        except MDNotImplementedError:
            funcs = self.dispatch_iter(*types)
            next(funcs)  # burn first
            for func in funcs:
                try:
                    return func(*args, **kwargs)
                except MDNotImplementedError:
                    pass
            raise NotImplementedError("Matching functions for "
                                      "%s: <%s> found, but none completed successfully"
                                      % (self.name, str_signature(types)))

    def __str__(self):
        return "<dispatched %s>" % self.name
    __repr__ = __str__

    def dispatch(self, *types):
        """ Deterimine appropriate implementation for this type signature

        This method is internal.  Users should call this object as a function.
        Implementation resolution occurs within the ``__call__`` method.

        >>> from sympy.multipledispatch import dispatch
        >>> @dispatch(int)
        ... def inc(x):
        ...     return x + 1

        >>> implementation = inc.dispatch(int)
        >>> implementation(3)
        4

        >>> print(inc.dispatch(float))
        None

        See Also:
            ``sympy.multipledispatch.conflict`` - module to determine resolution order
        """

        if types in self.funcs:
            return self.funcs[types]

        try:
            return next(self.dispatch_iter(*types))
        except StopIteration:
            return None

    def dispatch_iter(self, *types):
        n = len(types)
        for signature in self.ordering:
            if len(signature) == n and all(map(issubclass, types, signature)):
                result = self.funcs[signature]
                yield result

    def resolve(self, types):
        """ Deterimine appropriate implementation for this type signature

        .. deprecated:: 0.4.4
            Use ``dispatch(*types)`` instead
        """
        warn("resolve() is deprecated, use dispatch(*types)",
             DeprecationWarning)

        return self.dispatch(*types)

    def __getstate__(self):
        return {'name': self.name,
                'funcs': self.funcs}

    def __setstate__(self, d):
        self.name = d['name']
        self.funcs = d['funcs']
        self.ordering = ordering(self.funcs)
        self._cache = {}

    @property
    def __doc__(self):
        docs = ["Multiply dispatched method: %s" % self.name]

        if self.doc:
            docs.append(self.doc)

        other = []
        for sig in self.ordering[::-1]:
            func = self.funcs[sig]
            if func.__doc__:
                s = 'Inputs: <%s>\n' % str_signature(sig)
                s += '-' * len(s) + '\n'
                s += func.__doc__.strip()
                docs.append(s)
            else:
                other.append(str_signature(sig))

        if other:
            docs.append('Other signatures:\n    ' + '\n    '.join(other))

        return '\n\n'.join(docs)

    def _help(self, *args):
        return self.dispatch(*map(type, args)).__doc__

    def help(self, *args, **kwargs):
        """ Print docstring for the function corresponding to inputs """
        print(self._help(*args))

    def _source(self, *args):
        func = self.dispatch(*map(type, args))
        if not func:
            raise TypeError("No function found")
        return source(func)

    def source(self, *args, **kwargs):
        """ Print source code for the function corresponding to inputs """
        print(self._source(*args))


def source(func):
    s = 'File: %s\n\n' % inspect.getsourcefile(func)
    s = s + inspect.getsource(func)
    return s


class MethodDispatcher(Dispatcher):
    """ Dispatch methods based on type signature

    See Also:
        Dispatcher
    """

    @classmethod
    def get_func_params(cls, func):
        if hasattr(inspect, "signature"):
            sig = inspect.signature(func)
            return itl.islice(sig.parameters.values(), 1, None)

    def __get__(self, instance, owner):
        self.obj = instance
        self.cls = owner
        return self

    def __call__(self, *args, **kwargs):
        types = tuple([type(arg) for arg in args])
        func = self.dispatch(*types)
        if not func:
            raise NotImplementedError('Could not find signature for %s: <%s>' %
                                      (self.name, str_signature(types)))
        return func(self.obj, *args, **kwargs)


def str_signature(sig):
    """ String representation of type signature

    >>> from sympy.multipledispatch.dispatcher import str_signature
    >>> str_signature((int, float))
    'int, float'
    """
    return ', '.join(cls.__name__ for cls in sig)


def warning_text(name, amb):
    """ The text for ambiguity warnings """
    text = "\nAmbiguities exist in dispatched function %s\n\n" % (name)
    text += "The following signatures may result in ambiguous behavior:\n"
    for pair in amb:
        text += "\t" + \
            ', '.join('[' + str_signature(s) + ']' for s in pair) + "\n"
    text += "\n\nConsider making the following additions:\n\n"
    text += '\n\n'.join(['@dispatch(' + str_signature(super_signature(s))
                         + ')\ndef %s(...)' % name for s in amb])
    return text

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\interval_arithmetic.py ===
"""
Interval Arithmetic for plotting.
This module does not implement interval arithmetic accurately and
hence cannot be used for purposes other than plotting. If you want
to use interval arithmetic, use mpmath's interval arithmetic.

The module implements interval arithmetic using numpy and
python floating points. The rounding up and down is not handled
and hence this is not an accurate implementation of interval
arithmetic.

The module uses numpy for speed which cannot be achieved with mpmath.
"""

# Q: Why use numpy? Why not simply use mpmath's interval arithmetic?
# A: mpmath's interval arithmetic simulates a floating point unit
# and hence is slow, while numpy evaluations are orders of magnitude
# faster.

# Q: Why create a separate class for intervals? Why not use SymPy's
# Interval Sets?
# A: The functionalities that will be required for plotting is quite
# different from what Interval Sets implement.

# Q: Why is rounding up and down according to IEEE754 not handled?
# A: It is not possible to do it in both numpy and python. An external
# library has to used, which defeats the whole purpose i.e., speed. Also
# rounding is handled for very few functions in those libraries.

# Q Will my plots be affected?
# A It will not affect most of the plots. The interval arithmetic
# module based suffers the same problems as that of floating point
# arithmetic.

from sympy.core.numbers import int_valued
from sympy.core.logic import fuzzy_and
from sympy.simplify.simplify import nsimplify

from .interval_membership import intervalMembership


class interval:
    """ Represents an interval containing floating points as start and
    end of the interval
    The is_valid variable tracks whether the interval obtained as the
    result of the function is in the domain and is continuous.
    - True: Represents the interval result of a function is continuous and
            in the domain of the function.
    - False: The interval argument of the function was not in the domain of
             the function, hence the is_valid of the result interval is False
    - None: The function was not continuous over the interval or
            the function's argument interval is partly in the domain of the
            function

    A comparison between an interval and a real number, or a
    comparison between two intervals may return ``intervalMembership``
    of two 3-valued logic values.
    """

    def __init__(self, *args, is_valid=True, **kwargs):
        self.is_valid = is_valid
        if len(args) == 1:
            if isinstance(args[0], interval):
                self.start, self.end = args[0].start, args[0].end
            else:
                self.start = float(args[0])
                self.end = float(args[0])
        elif len(args) == 2:
            if args[0] < args[1]:
                self.start = float(args[0])
                self.end = float(args[1])
            else:
                self.start = float(args[1])
                self.end = float(args[0])

        else:
            raise ValueError("interval takes a maximum of two float values "
                            "as arguments")

    @property
    def mid(self):
        return (self.start + self.end) / 2.0

    @property
    def width(self):
        return self.end - self.start

    def __repr__(self):
        return "interval(%f, %f)" % (self.start, self.end)

    def __str__(self):
        return "[%f, %f]" % (self.start, self.end)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            if self.end < other:
                return intervalMembership(True, self.is_valid)
            elif self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        elif isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end < other. start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            if self.start > other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__lt__(self)
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(True, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(False, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(True, valid)
            elif self.__lt__(other)[0] is not None:
                return intervalMembership(False, valid)
            else:
                return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            if self.start == other and self.end == other:
                return intervalMembership(False, self.is_valid)
            if other in self:
                return intervalMembership(None, self.is_valid)
            else:
                return intervalMembership(True, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.start == other.start and self.end == other.end:
                return intervalMembership(False, valid)
            if not self.__lt__(other)[0] is None:
                return intervalMembership(True, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)):
            if self.end <= other:
                return intervalMembership(True, self.is_valid)
            if self.start > other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)

        if isinstance(other, interval):
            valid = fuzzy_and([self.is_valid, other.is_valid])
            if self.end <= other.start:
                return intervalMembership(True, valid)
            if self.start > other.end:
                return intervalMembership(False, valid)
            return intervalMembership(None, valid)
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            if self.start >= other:
                return intervalMembership(True, self.is_valid)
            elif self.end < other:
                return intervalMembership(False, self.is_valid)
            else:
                return intervalMembership(None, self.is_valid)
        elif isinstance(other, interval):
            return other.__le__(self)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if self.is_valid:
                return interval(self.start + other, self.end + other)
            else:
                start = self.start + other
                end = self.end + other
                return interval(start, end, is_valid=self.is_valid)

        elif isinstance(other, interval):
            start = self.start + other.start
            end = self.end + other.end
            valid = fuzzy_and([self.is_valid, other.is_valid])
            return interval(start, end, is_valid=valid)
        else:
            return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            start = self.start - other
            end = self.end - other
            return interval(start, end, is_valid=self.is_valid)

        elif isinstance(other, interval):
            start = self.start - other.end
            end = self.end - other.start
            valid = fuzzy_and([self.is_valid, other.is_valid])
            return interval(start, end, is_valid=valid)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            start = other - self.end
            end = other - self.start
            return interval(start, end, is_valid=self.is_valid)
        elif isinstance(other, interval):
            return other.__sub__(self)
        else:
            return NotImplemented

    def __neg__(self):
        if self.is_valid:
            return interval(-self.end, -self.start)
        else:
            return interval(-self.end, -self.start, is_valid=self.is_valid)

    def __mul__(self, other):
        if isinstance(other, interval):
            if self.is_valid is False or other.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif self.is_valid is None or other.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
                inters = []
                inters.append(self.start * other.start)
                inters.append(self.end * other.start)
                inters.append(self.start * other.end)
                inters.append(self.end * other.end)
                start = min(inters)
                end = max(inters)
                return interval(start, end)
        elif isinstance(other, (int, float)):
            return interval(self.start*other, self.end*other, is_valid=self.is_valid)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __contains__(self, other):
        if isinstance(other, (int, float)):
            return self.start <= other and self.end >= other
        else:
            return self.start <= other.start and other.end <= self.end

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = interval(other)
            return other.__truediv__(self)
        elif isinstance(other, interval):
            return other.__truediv__(self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        # Both None and False are handled
        if not self.is_valid:
            # Don't divide as the value is not valid
            return interval(-float('inf'), float('inf'), is_valid=self.is_valid)
        if isinstance(other, (int, float)):
            if other == 0:
                # Divide by zero encountered. valid nowhere
                return interval(-float('inf'), float('inf'), is_valid=False)
            else:
                return interval(self.start / other, self.end / other)

        elif isinstance(other, interval):
            if other.is_valid is False or self.is_valid is False:
                return interval(-float('inf'), float('inf'), is_valid=False)
            elif other.is_valid is None or self.is_valid is None:
                return interval(-float('inf'), float('inf'), is_valid=None)
            else:
               # denominator contains both signs, i.e. being divided by zero
               # return the whole real line with is_valid = None
                if 0 in other:
                    return interval(-float('inf'), float('inf'), is_valid=None)

                # denominator negative
                this = self
                if other.end < 0:
                    this = -this
                    other = -other

                # denominator positive
                inters = []
                inters.append(this.start / other.start)
                inters.append(this.end / other.start)
                inters.append(this.start / other.end)
                inters.append(this.end / other.end)
                start = max(inters)
                end = min(inters)
                return interval(start, end)
        else:
            return NotImplemented

    def __pow__(self, other):
        # Implements only power to an integer.
        from .lib_interval import exp, log
        if not self.is_valid:
            return self
        if isinstance(other, interval):
            return exp(other * log(self))
        elif isinstance(other, (float, int)):
            if other < 0:
                return 1 / self.__pow__(abs(other))
            else:
                if int_valued(other):
                    return _pow_int(self, other)
                else:
                    return _pow_float(self, other)
        else:
            return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (float, int)):
            if not self.is_valid:
                #Don't do anything
                return self
            elif other < 0:
                if self.width > 0:
                    return interval(-float('inf'), float('inf'), is_valid=False)
                else:
                    power_rational = nsimplify(self.start)
                    num, denom = power_rational.as_numer_denom()
                    if denom % 2 == 0:
                        return interval(-float('inf'), float('inf'),
                                        is_valid=False)
                    else:
                        start = -abs(other)**self.start
                        end = start
                        return interval(start, end)
            else:
                return interval(other**self.start, other**self.end)
        elif isinstance(other, interval):
            return other.__pow__(self)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.is_valid, self.start, self.end))


def _pow_float(inter, power):
    """Evaluates an interval raised to a floating point."""
    power_rational = nsimplify(power)
    num, denom = power_rational.as_numer_denom()
    if num % 2 == 0:
        start = abs(inter.start)**power
        end = abs(inter.end)**power
        if start < 0:
            ret = interval(0, max(start, end))
        else:
            ret = interval(start, end)
        return ret
    elif denom % 2 == 0:
        if inter.end < 0:
            return interval(-float('inf'), float('inf'), is_valid=False)
        elif inter.start < 0:
            return interval(0, inter.end**power, is_valid=None)
        else:
            return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0:
            start = -abs(inter.start)**power
        else:
            start = inter.start**power

        if inter.end < 0:
            end = -abs(inter.end)**power
        else:
            end = inter.end**power

        return interval(start, end, is_valid=inter.is_valid)


def _pow_int(inter, power):
    """Evaluates an interval raised to an integer power"""
    power = int(power)
    if power & 1:
        return interval(inter.start**power, inter.end**power)
    else:
        if inter.start < 0 and inter.end > 0:
            start = 0
            end = max(inter.start**power, inter.end**power)
            return interval(start, end)
        else:
            return interval(inter.start**power, inter.end**power)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\parsers\c_parser_wrapper.py ===
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import (
    lib,
    parsers,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DtypeWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.concat import (
    concat_compat,
    union_categoricals,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core.indexes.api import ensure_index_from_sequences

from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    ParserError,
    is_index_col,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeArg,
        DtypeObj,
        ReadCsvBuffer,
    )

    from pandas import (
        Index,
        MultiIndex,
    )


class CParserWrapper(ParserBase):
    low_memory: bool
    _reader: parsers.TextReader

    def __init__(self, src: ReadCsvBuffer[str], **kwds) -> None:
        super().__init__(kwds)
        self.kwds = kwds
        kwds = kwds.copy()

        self.low_memory = kwds.pop("low_memory", False)

        # #2442
        # error: Cannot determine type of 'index_col'
        kwds["allow_leading_cols"] = (
            self.index_col is not False  # type: ignore[has-type]
        )

        # GH20529, validate usecol arg before TextReader
        kwds["usecols"] = self.usecols

        # Have to pass int, would break tests using TextReader directly otherwise :(
        kwds["on_bad_lines"] = self.on_bad_lines.value

        for key in (
            "storage_options",
            "encoding",
            "memory_map",
            "compression",
        ):
            kwds.pop(key, None)

        kwds["dtype"] = ensure_dtype_objs(kwds.get("dtype", None))
        if "dtype_backend" not in kwds or kwds["dtype_backend"] is lib.no_default:
            kwds["dtype_backend"] = "numpy"
        if kwds["dtype_backend"] == "pyarrow":
            # Fail here loudly instead of in cython after reading
            import_optional_dependency("pyarrow")
        self._reader = parsers.TextReader(src, **kwds)

        self.unnamed_cols = self._reader.unnamed_cols

        # error: Cannot determine type of 'names'
        passed_names = self.names is None  # type: ignore[has-type]

        if self._reader.header is None:
            self.names = None
        else:
            # error: Cannot determine type of 'names'
            # error: Cannot determine type of 'index_names'
            (
                self.names,  # type: ignore[has-type]
                self.index_names,
                self.col_names,
                passed_names,
            ) = self._extract_multi_indexer_columns(
                self._reader.header,
                self.index_names,  # type: ignore[has-type]
                passed_names,
            )

        # error: Cannot determine type of 'names'
        if self.names is None:  # type: ignore[has-type]
            self.names = list(range(self._reader.table_width))

        # gh-9755
        #
        # need to set orig_names here first
        # so that proper indexing can be done
        # with _set_noconvert_columns
        #
        # once names has been filtered, we will
        # then set orig_names again to names
        # error: Cannot determine type of 'names'
        self.orig_names = self.names[:]  # type: ignore[has-type]

        if self.usecols:
            usecols = self._evaluate_usecols(self.usecols, self.orig_names)

            # GH 14671
            # assert for mypy, orig_names is List or None, None would error in issubset
            assert self.orig_names is not None
            if self.usecols_dtype == "string" and not set(usecols).issubset(
                self.orig_names
            ):
                self._validate_usecols_names(usecols, self.orig_names)

            # error: Cannot determine type of 'names'
            if len(self.names) > len(usecols):  # type: ignore[has-type]
                # error: Cannot determine type of 'names'
                self.names = [  # type: ignore[has-type]
                    n
                    # error: Cannot determine type of 'names'
                    for i, n in enumerate(self.names)  # type: ignore[has-type]
                    if (i in usecols or n in usecols)
                ]

            # error: Cannot determine type of 'names'
            if len(self.names) < len(usecols):  # type: ignore[has-type]
                # error: Cannot determine type of 'names'
                self._validate_usecols_names(
                    usecols,
                    self.names,  # type: ignore[has-type]
                )

        # error: Cannot determine type of 'names'
        self._validate_parse_dates_presence(self.names)  # type: ignore[has-type]
        self._set_noconvert_columns()

        # error: Cannot determine type of 'names'
        self.orig_names = self.names  # type: ignore[has-type]

        if not self._has_complex_date_col:
            # error: Cannot determine type of 'index_col'
            if self._reader.leading_cols == 0 and is_index_col(
                self.index_col  # type: ignore[has-type]
            ):
                self._name_processed = True
                (
                    index_names,
                    # error: Cannot determine type of 'names'
                    self.names,  # type: ignore[has-type]
                    self.index_col,
                ) = self._clean_index_names(
                    # error: Cannot determine type of 'names'
                    self.names,  # type: ignore[has-type]
                    # error: Cannot determine type of 'index_col'
                    self.index_col,  # type: ignore[has-type]
                )

                if self.index_names is None:
                    self.index_names = index_names

            if self._reader.header is None and not passed_names:
                assert self.index_names is not None
                self.index_names = [None] * len(self.index_names)

        self._implicit_index = self._reader.leading_cols > 0

    def close(self) -> None:
        # close handles opened by C parser
        try:
            self._reader.close()
        except ValueError:
            pass

    def _set_noconvert_columns(self) -> None:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
        assert self.orig_names is not None
        # error: Cannot determine type of 'names'

        # much faster than using orig_names.index(x) xref GH#44106
        names_dict = {x: i for i, x in enumerate(self.orig_names)}
        col_indices = [names_dict[x] for x in self.names]  # type: ignore[has-type]
        # error: Cannot determine type of 'names'
        noconvert_columns = self._set_noconvert_dtype_columns(
            col_indices,
            self.names,  # type: ignore[has-type]
        )
        for col in noconvert_columns:
            self._reader.set_noconvert(col)

    def read(
        self,
        nrows: int | None = None,
    ) -> tuple[
        Index | MultiIndex | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, ArrayLike],
    ]:
        index: Index | MultiIndex | None
        column_names: Sequence[Hashable] | MultiIndex
        try:
            if self.low_memory:
                chunks = self._reader.read_low_memory(nrows)
                # destructive to chunks
                data = _concatenate_chunks(chunks)

            else:
                data = self._reader.read(nrows)
        except StopIteration:
            if self._first_chunk:
                self._first_chunk = False
                names = dedup_names(
                    self.orig_names,
                    is_potential_multi_index(self.orig_names, self.index_col),
                )
                index, columns, col_dict = self._get_empty_meta(
                    names,
                    dtype=self.dtype,
                )
                columns = self._maybe_make_multi_index_columns(columns, self.col_names)

                if self.usecols is not None:
                    columns = self._filter_usecols(columns)

                col_dict = {k: v for k, v in col_dict.items() if k in columns}

                return index, columns, col_dict

            else:
                self.close()
                raise

        # Done with first read, next time raise StopIteration
        self._first_chunk = False

        # error: Cannot determine type of 'names'
        names = self.names  # type: ignore[has-type]

        if self._reader.leading_cols:
            if self._has_complex_date_col:
                raise NotImplementedError("file structure not yet supported")

            # implicit index, no index names
            arrays = []

            if self.index_col and self._reader.leading_cols != len(self.index_col):
                raise ParserError(
                    "Could not construct index. Requested to use "
                    f"{len(self.index_col)} number of columns, but "
                    f"{self._reader.leading_cols} left to parse."
                )

            for i in range(self._reader.leading_cols):
                if self.index_col is None:
                    values = data.pop(i)
                else:
                    values = data.pop(self.index_col[i])

                values = self._maybe_parse_dates(values, i, try_parse_dates=True)
                arrays.append(values)

            index = ensure_index_from_sequences(arrays)

            if self.usecols is not None:
                names = self._filter_usecols(names)

            names = dedup_names(names, is_potential_multi_index(names, self.index_col))

            # rename dict keys
            data_tups = sorted(data.items())
            data = {k: v for k, (i, v) in zip(names, data_tups)}

            column_names, date_data = self._do_date_conversions(names, data)

            # maybe create a mi on the columns
            column_names = self._maybe_make_multi_index_columns(
                column_names, self.col_names
            )

        else:
            # rename dict keys
            data_tups = sorted(data.items())

            # ugh, mutation

            # assert for mypy, orig_names is List or None, None would error in list(...)
            assert self.orig_names is not None
            names = list(self.orig_names)
            names = dedup_names(names, is_potential_multi_index(names, self.index_col))

            if self.usecols is not None:
                names = self._filter_usecols(names)

            # columns as list
            alldata = [x[1] for x in data_tups]
            if self.usecols is None:
                self._check_data_length(names, alldata)

            data = {k: v for k, (i, v) in zip(names, data_tups)}

            names, date_data = self._do_date_conversions(names, data)
            index, column_names = self._make_index(date_data, alldata, names)

        return index, column_names, date_data

    def _filter_usecols(self, names: Sequence[Hashable]) -> Sequence[Hashable]:
        # hackish
        usecols = self._evaluate_usecols(self.usecols, names)
        if usecols is not None and len(names) != len(usecols):
            names = [
                name for i, name in enumerate(names) if i in usecols or name in usecols
            ]
        return names

    def _maybe_parse_dates(self, values, index: int, try_parse_dates: bool = True):
        if try_parse_dates and self._should_parse_dates(index):
            values = self._date_conv(
                values,
                col=self.index_names[index] if self.index_names is not None else None,
            )
        return values


def _concatenate_chunks(chunks: list[dict[int, ArrayLike]]) -> dict:
    """
    Concatenate chunks of data read with low_memory=True.

    The tricky part is handling Categoricals, where different chunks
    may have different inferred categories.
    """
    names = list(chunks[0].keys())
    warning_columns = []

    result: dict = {}
    for name in names:
        arrs = [chunk.pop(name) for chunk in chunks]
        # Check each arr for consistent types.
        dtypes = {a.dtype for a in arrs}
        non_cat_dtypes = {x for x in dtypes if not isinstance(x, CategoricalDtype)}

        dtype = dtypes.pop()
        if isinstance(dtype, CategoricalDtype):
            result[name] = union_categoricals(arrs, sort_categories=False)
        else:
            result[name] = concat_compat(arrs)
            if len(non_cat_dtypes) > 1 and result[name].dtype == np.dtype(object):
                warning_columns.append(str(name))

    if warning_columns:
        warning_names = ",".join(warning_columns)
        warning_message = " ".join(
            [
                f"Columns ({warning_names}) have mixed types. "
                f"Specify dtype option on import or set low_memory=False."
            ]
        )
        warnings.warn(warning_message, DtypeWarning, stacklevel=find_stack_level())
    return result


def ensure_dtype_objs(
    dtype: DtypeArg | dict[Hashable, DtypeArg] | None
) -> DtypeObj | dict[Hashable, DtypeObj] | None:
    """
    Ensure we have either None, a dtype object, or a dictionary mapping to
    dtype objects.
    """
    if isinstance(dtype, defaultdict):
        # "None" not callable  [misc]
        default_dtype = pandas_dtype(dtype.default_factory())  # type: ignore[misc]
        dtype_converted: defaultdict = defaultdict(lambda: default_dtype)
        for key in dtype.keys():
            dtype_converted[key] = pandas_dtype(dtype[key])
        return dtype_converted
    elif isinstance(dtype, dict):
        return {k: pandas_dtype(dtype[k]) for k in dtype}
    elif dtype is not None:
        return pandas_dtype(dtype)
    return dtype

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pyproject_hooks\_impl.py ===
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from os.path import abspath
from os.path import join as pjoin
from subprocess import STDOUT, check_call, check_output
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional, Sequence

from ._in_process import _in_proc_script_path

if TYPE_CHECKING:
    from typing import Protocol

    class SubprocessRunner(Protocol):
        """A protocol for the subprocess runner."""

        def __call__(
            self,
            cmd: Sequence[str],
            cwd: Optional[str] = None,
            extra_environ: Optional[Mapping[str, str]] = None,
        ) -> None:
            ...


def write_json(obj: Mapping[str, Any], path: str, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kwargs)


def read_json(path: str) -> Mapping[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class BackendUnavailable(Exception):
    """Will be raised if the backend cannot be imported in the hook process."""

    def __init__(
        self,
        traceback: str,
        message: Optional[str] = None,
        backend_name: Optional[str] = None,
        backend_path: Optional[Sequence[str]] = None,
    ) -> None:
        # Preserving arg order for the sake of API backward compatibility.
        self.backend_name = backend_name
        self.backend_path = backend_path
        self.traceback = traceback
        super().__init__(message or "Error while importing backend")


class HookMissing(Exception):
    """Will be raised on missing hooks (if a fallback can't be used)."""

    def __init__(self, hook_name: str) -> None:
        super().__init__(hook_name)
        self.hook_name = hook_name


class UnsupportedOperation(Exception):
    """May be raised by build_sdist if the backend indicates that it can't."""

    def __init__(self, traceback: str) -> None:
        self.traceback = traceback


def default_subprocess_runner(
    cmd: Sequence[str],
    cwd: Optional[str] = None,
    extra_environ: Optional[Mapping[str, str]] = None,
) -> None:
    """The default method of calling the wrapper subprocess.

    This uses :func:`subprocess.check_call` under the hood.
    """
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)

    check_call(cmd, cwd=cwd, env=env)


def quiet_subprocess_runner(
    cmd: Sequence[str],
    cwd: Optional[str] = None,
    extra_environ: Optional[Mapping[str, str]] = None,
) -> None:
    """Call the subprocess while suppressing output.

    This uses :func:`subprocess.check_output` under the hood.
    """
    env = os.environ.copy()
    if extra_environ:
        env.update(extra_environ)

    check_output(cmd, cwd=cwd, env=env, stderr=STDOUT)


def norm_and_check(source_tree: str, requested: str) -> str:
    """Normalise and check a backend path.

    Ensure that the requested backend path is specified as a relative path,
    and resolves to a location under the given source tree.

    Return an absolute version of the requested path.
    """
    if os.path.isabs(requested):
        raise ValueError("paths must be relative")

    abs_source = os.path.abspath(source_tree)
    abs_requested = os.path.normpath(os.path.join(abs_source, requested))
    # We have to use commonprefix for Python 2.7 compatibility. So we
    # normalise case to avoid problems because commonprefix is a character
    # based comparison :-(
    norm_source = os.path.normcase(abs_source)
    norm_requested = os.path.normcase(abs_requested)
    if os.path.commonprefix([norm_source, norm_requested]) != norm_source:
        raise ValueError("paths must be inside source tree")

    return abs_requested


class BuildBackendHookCaller:
    """A wrapper to call the build backend hooks for a source directory."""

    def __init__(
        self,
        source_dir: str,
        build_backend: str,
        backend_path: Optional[Sequence[str]] = None,
        runner: Optional["SubprocessRunner"] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        """
        :param source_dir: The source directory to invoke the build backend for
        :param build_backend: The build backend spec
        :param backend_path: Additional path entries for the build backend spec
        :param runner: The :ref:`subprocess runner <Subprocess Runners>` to use
        :param python_executable:
            The Python executable used to invoke the build backend
        """
        if runner is None:
            runner = default_subprocess_runner

        self.source_dir = abspath(source_dir)
        self.build_backend = build_backend
        if backend_path:
            backend_path = [norm_and_check(self.source_dir, p) for p in backend_path]
        self.backend_path = backend_path
        self._subprocess_runner = runner
        if not python_executable:
            python_executable = sys.executable
        self.python_executable = python_executable

    @contextmanager
    def subprocess_runner(self, runner: "SubprocessRunner") -> Iterator[None]:
        """A context manager for temporarily overriding the default
        :ref:`subprocess runner <Subprocess Runners>`.

        :param runner: The new subprocess runner to use within the context.

        .. code-block:: python

            hook_caller = BuildBackendHookCaller(...)
            with hook_caller.subprocess_runner(quiet_subprocess_runner):
                ...
        """
        prev = self._subprocess_runner
        self._subprocess_runner = runner
        try:
            yield
        finally:
            self._subprocess_runner = prev

    def _supported_features(self) -> Sequence[str]:
        """Return the list of optional features supported by the backend."""
        return self._call_hook("_supported_features", {})

    def get_requires_for_build_wheel(
        self,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[str]:
        """Get additional dependencies required for building a wheel.

        :param config_settings: The configuration settings for the build backend
        :returns: A list of :pep:`dependency specifiers <508>`.

        .. admonition:: Fallback

            If the build backend does not defined a hook with this name, an
            empty list will be returned.
        """
        return self._call_hook(
            "get_requires_for_build_wheel", {"config_settings": config_settings}
        )

    def prepare_metadata_for_build_wheel(
        self,
        metadata_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        _allow_fallback: bool = True,
    ) -> str:
        """Prepare a ``*.dist-info`` folder with metadata for this project.

        :param metadata_directory: The directory to write the metadata to
        :param config_settings: The configuration settings for the build backend
        :param _allow_fallback:
            Whether to allow the fallback to building a wheel and extracting
            the metadata from it. Should be passed as a keyword argument only.

        :returns: Name of the newly created subfolder within
                  ``metadata_directory``, containing the metadata.

        .. admonition:: Fallback

            If the build backend does not define a hook with this name and
            ``_allow_fallback`` is truthy, the backend will be asked to build a
            wheel via the ``build_wheel`` hook and the dist-info extracted from
            that will be returned.
        """
        return self._call_hook(
            "prepare_metadata_for_build_wheel",
            {
                "metadata_directory": abspath(metadata_directory),
                "config_settings": config_settings,
                "_allow_fallback": _allow_fallback,
            },
        )

    def build_wheel(
        self,
        wheel_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        """Build a wheel from this project.

        :param wheel_directory: The directory to write the wheel to
        :param config_settings: The configuration settings for the build backend
        :param metadata_directory: The directory to reuse existing metadata from
        :returns:
            The name of the newly created wheel within ``wheel_directory``.

        .. admonition:: Interaction with fallback

            If the ``build_wheel`` hook was called in the fallback for
            :meth:`prepare_metadata_for_build_wheel`, the build backend would
            not be invoked. Instead, the previously built wheel will be copied
            to ``wheel_directory`` and the name of that file will be returned.
        """
        if metadata_directory is not None:
            metadata_directory = abspath(metadata_directory)
        return self._call_hook(
            "build_wheel",
            {
                "wheel_directory": abspath(wheel_directory),
                "config_settings": config_settings,
                "metadata_directory": metadata_directory,
            },
        )

    def get_requires_for_build_editable(
        self,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[str]:
        """Get additional dependencies required for building an editable wheel.

        :param config_settings: The configuration settings for the build backend
        :returns: A list of :pep:`dependency specifiers <508>`.

        .. admonition:: Fallback

            If the build backend does not defined a hook with this name, an
            empty list will be returned.
        """
        return self._call_hook(
            "get_requires_for_build_editable", {"config_settings": config_settings}
        )

    def prepare_metadata_for_build_editable(
        self,
        metadata_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        _allow_fallback: bool = True,
    ) -> Optional[str]:
        """Prepare a ``*.dist-info`` folder with metadata for this project.

        :param metadata_directory: The directory to write the metadata to
        :param config_settings: The configuration settings for the build backend
        :param _allow_fallback:
            Whether to allow the fallback to building a wheel and extracting
            the metadata from it. Should be passed as a keyword argument only.
        :returns: Name of the newly created subfolder within
                  ``metadata_directory``, containing the metadata.

        .. admonition:: Fallback

            If the build backend does not define a hook with this name and
            ``_allow_fallback`` is truthy, the backend will be asked to build a
            wheel via the ``build_editable`` hook and the dist-info
            extracted from that will be returned.
        """
        return self._call_hook(
            "prepare_metadata_for_build_editable",
            {
                "metadata_directory": abspath(metadata_directory),
                "config_settings": config_settings,
                "_allow_fallback": _allow_fallback,
            },
        )

    def build_editable(
        self,
        wheel_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
        metadata_directory: Optional[str] = None,
    ) -> str:
        """Build an editable wheel from this project.

        :param wheel_directory: The directory to write the wheel to
        :param config_settings: The configuration settings for the build backend
        :param metadata_directory: The directory to reuse existing metadata from
        :returns:
            The name of the newly created wheel within ``wheel_directory``.

        .. admonition:: Interaction with fallback

            If the ``build_editable`` hook was called in the fallback for
            :meth:`prepare_metadata_for_build_editable`, the build backend
            would not be invoked. Instead, the previously built wheel will be
            copied to ``wheel_directory`` and the name of that file will be
            returned.
        """
        if metadata_directory is not None:
            metadata_directory = abspath(metadata_directory)
        return self._call_hook(
            "build_editable",
            {
                "wheel_directory": abspath(wheel_directory),
                "config_settings": config_settings,
                "metadata_directory": metadata_directory,
            },
        )

    def get_requires_for_build_sdist(
        self,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[str]:
        """Get additional dependencies required for building an sdist.

        :returns: A list of :pep:`dependency specifiers <508>`.
        """
        return self._call_hook(
            "get_requires_for_build_sdist", {"config_settings": config_settings}
        )

    def build_sdist(
        self,
        sdist_directory: str,
        config_settings: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Build an sdist from this project.

        :returns:
            The name of the newly created sdist within ``wheel_directory``.
        """
        return self._call_hook(
            "build_sdist",
            {
                "sdist_directory": abspath(sdist_directory),
                "config_settings": config_settings,
            },
        )

    def _call_hook(self, hook_name: str, kwargs: Mapping[str, Any]) -> Any:
        extra_environ = {"_PYPROJECT_HOOKS_BUILD_BACKEND": self.build_backend}

        if self.backend_path:
            backend_path = os.pathsep.join(self.backend_path)
            extra_environ["_PYPROJECT_HOOKS_BACKEND_PATH"] = backend_path

        with tempfile.TemporaryDirectory() as td:
            hook_input = {"kwargs": kwargs}
            write_json(hook_input, pjoin(td, "input.json"), indent=2)

            # Run the hook in a subprocess
            with _in_proc_script_path() as script:
                python = self.python_executable
                self._subprocess_runner(
                    [python, abspath(str(script)), hook_name, td],
                    cwd=self.source_dir,
                    extra_environ=extra_environ,
                )

            data = read_json(pjoin(td, "output.json"))
            if data.get("unsupported"):
                raise UnsupportedOperation(data.get("traceback", ""))
            if data.get("no_backend"):
                raise BackendUnavailable(
                    data.get("traceback", ""),
                    message=data.get("backend_error", ""),
                    backend_name=self.build_backend,
                    backend_path=self.backend_path,
                )
            if data.get("hook_missing"):
                raise HookMissing(data.get("missing_hook_name") or hook_name)
            return data["return_val"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\qexpr.py ===
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray, scipy_sparse_matrix,
    to_sympy, to_numpy, to_scipy_sparse
)

__all__ = [
    'QuantumError',
    'QExpr'
]


#-----------------------------------------------------------------------------
# Error handling
#-----------------------------------------------------------------------------

class QuantumError(Exception):
    pass


def _qsympify_sequence(seq):
    """Convert elements of a sequence to standard form.

    This is like sympify, but it performs special logic for arguments passed
    to QExpr. The following conversions are done:

    * (list, tuple, Tuple) => _qsympify_sequence each element and convert
      sequence to a Tuple.
    * basestring => Symbol
    * Matrix => Matrix
    * other => sympify

    Strings are passed to Symbol, not sympify to make sure that variables like
    'pi' are kept as Symbols, not the SymPy built-in number subclasses.

    Examples
    ========

    >>> from sympy.physics.quantum.qexpr import _qsympify_sequence
    >>> _qsympify_sequence((1,2,[3,4,[1,]]))
    (1, 2, (3, 4, (1,)))

    """

    return tuple(__qsympify_sequence_helper(seq))


def __qsympify_sequence_helper(seq):
    """
       Helper function for _qsympify_sequence
       This function does the actual work.
    """
    #base case. If not a list, do Sympification
    if not is_sequence(seq):
        if isinstance(seq, Matrix):
            return seq
        elif isinstance(seq, str):
            return Symbol(seq)
        else:
            return sympify(seq)

    # base condition, when seq is QExpr and also
    # is iterable.
    if isinstance(seq, QExpr):
        return seq

    #if list, recurse on each item in the list
    result = [__qsympify_sequence_helper(item) for item in seq]

    return Tuple(*result)


#-----------------------------------------------------------------------------
# Basic Quantum Expression from which all objects descend
#-----------------------------------------------------------------------------

class QExpr(Expr):
    """A base class for all quantum object like operators and states."""

    # In sympy, slots are for instance attributes that are computed
    # dynamically by the __new__ method. They are not part of args, but they
    # derive from args.

    # The Hilbert space a quantum Object belongs to.
    __slots__ = ('hilbert_space', )

    is_commutative = False

    # The separator used in printing the label.
    _label_separator = ''

    def __new__(cls, *args, **kwargs):
        """Construct a new quantum object.

        Parameters
        ==========

        args : tuple
            The list of numbers or parameters that uniquely specify the
            quantum object. For a state, this will be its symbol or its
            set of quantum numbers.

        Examples
        ========

        >>> from sympy.physics.quantum.qexpr import QExpr
        >>> q = QExpr(0)
        >>> q
        0
        >>> q.label
        (0,)
        >>> q.hilbert_space
        H
        >>> q.args
        (0,)
        >>> q.is_commutative
        False
        """

        # First compute args and call Expr.__new__ to create the instance
        args = cls._eval_args(args, **kwargs)
        if len(args) == 0:
            args = cls._eval_args(tuple(cls.default_args()), **kwargs)
        inst = Expr.__new__(cls, *args)
        # Now set the slots on the instance
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _new_rawargs(cls, hilbert_space, *args, **old_assumptions):
        """Create new instance of this class with hilbert_space and args.

        This is used to bypass the more complex logic in the ``__new__``
        method in cases where you already have the exact ``hilbert_space``
        and ``args``. This should be used when you are positive these
        arguments are valid, in their final, proper form and want to optimize
        the creation of the object.
        """

        obj = Expr.__new__(cls, *args, **old_assumptions)
        obj.hilbert_space = hilbert_space
        return obj

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def label(self):
        """The label is the unique set of identifiers for the object.

        Usually, this will include all of the information about the state
        *except* the time (in the case of time-dependent objects).

        This must be a tuple, rather than a Tuple.
        """
        if len(self.args) == 0:  # If there is no label specified, return the default
            return self._eval_args(list(self.default_args()))
        else:
            return self.args

    @property
    def is_symbolic(self):
        return True

    @classmethod
    def default_args(self):
        """If no arguments are specified, then this will return a default set
        of arguments to be run through the constructor.

        NOTE: Any classes that override this MUST return a tuple of arguments.
        Should be overridden by subclasses to specify the default arguments for kets and operators
        """
        raise NotImplementedError("No default arguments for this class!")

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_adjoint(self):
        obj = Expr._eval_adjoint(self)
        if obj is None:
            obj = Expr.__new__(Dagger, self)
        if isinstance(obj, QExpr):
            obj.hilbert_space = self.hilbert_space
        return obj

    @classmethod
    def _eval_args(cls, args):
        """Process the args passed to the __new__ method.

        This simply runs args through _qsympify_sequence.
        """
        return _qsympify_sequence(args)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """Compute the Hilbert space instance from the args.
        """
        from sympy.physics.quantum.hilbert import HilbertSpace
        return HilbertSpace()

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    # Utilities for printing: these operate on raw SymPy objects

    def _print_sequence(self, seq, sep, printer, *args):
        result = []
        for item in seq:
            result.append(printer._print(item, *args))
        return sep.join(result)

    def _print_sequence_pretty(self, seq, sep, printer, *args):
        pform = printer._print(seq[0], *args)
        for item in seq[1:]:
            pform = prettyForm(*pform.right(sep))
            pform = prettyForm(*pform.right(printer._print(item, *args)))
        return pform

    # Utilities for printing: these operate prettyForm objects

    def _print_subscript_pretty(self, a, b):
        top = prettyForm(*b.left(' '*a.width()))
        bot = prettyForm(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))

    def _print_superscript_pretty(self, a, b):
        return a**b

    def _print_parens_pretty(self, pform, left='(', right=')'):
        return prettyForm(*pform.parens(left=left, right=right))

    # Printing of labels (i.e. args)

    def _print_label(self, printer, *args):
        """Prints the label of the QExpr

        This method prints self.label, using self._label_separator to separate
        the elements. This method should not be overridden, instead, override
        _print_contents to change printing behavior.
        """
        return self._print_sequence(
            self.label, self._label_separator, printer, *args
        )

    def _print_label_repr(self, printer, *args):
        return self._print_sequence(
            self.label, ',', printer, *args
        )

    def _print_label_pretty(self, printer, *args):
        return self._print_sequence_pretty(
            self.label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        return self._print_sequence(
            self.label, self._label_separator, printer, *args
        )

    # Printing of contents (default to label)

    def _print_contents(self, printer, *args):
        """Printer for contents of QExpr

        Handles the printing of any unique identifying contents of a QExpr to
        print as its contents, such as any variables or quantum numbers. The
        default is to print the label, which is almost always the args. This
        should not include printing of any brackets or parentheses.
        """
        return self._print_label(printer, *args)

    def _print_contents_pretty(self, printer, *args):
        return self._print_label_pretty(printer, *args)

    def _print_contents_latex(self, printer, *args):
        return self._print_label_latex(printer, *args)

    # Main printing methods

    def _sympystr(self, printer, *args):
        """Default printing behavior of QExpr objects

        Handles the default printing of a QExpr. To add other things to the
        printing of the object, such as an operator name to operators or
        brackets to states, the class should override the _print/_pretty/_latex
        functions directly and make calls to _print_contents where appropriate.
        This allows things like InnerProduct to easily control its printing the
        printing of contents.
        """
        return self._print_contents(printer, *args)

    def _sympyrepr(self, printer, *args):
        classname = self.__class__.__name__
        label = self._print_label_repr(printer, *args)
        return '%s(%s)' % (classname, label)

    def _pretty(self, printer, *args):
        pform = self._print_contents_pretty(printer, *args)
        return pform

    def _latex(self, printer, *args):
        return self._print_contents_latex(printer, *args)

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_default_basis(self, **options):
        raise NotImplementedError('This object does not have a default basis')

    def _represent(self, *, basis=None, **options):
        """Represent this object in a given basis.

        This method dispatches to the actual methods that perform the
        representation. Subclases of QExpr should define various methods to
        determine how the object will be represented in various bases. The
        format of these methods is::

            def _represent_BasisName(self, basis, **options):

        Thus to define how a quantum object is represented in the basis of
        the operator Position, you would define::

            def _represent_Position(self, basis, **options):

        Usually, basis object will be instances of Operator subclasses, but
        there is a chance we will relax this in the future to accommodate other
        types of basis sets that are not associated with an operator.

        If the ``format`` option is given it can be ("sympy", "numpy",
        "scipy.sparse"). This will ensure that any matrices that result from
        representing the object are returned in the appropriate matrix format.

        Parameters
        ==========

        basis : Operator
            The Operator whose basis functions will be used as the basis for
            representation.
        options : dict
            A dictionary of key/value pairs that give options and hints for
            the representation, such as the number of basis functions to
            be used.
        """
        if basis is None:
            result = self._represent_default_basis(**options)
        else:
            result = dispatch_method(self, '_represent', basis, **options)

        # If we get a matrix representation, convert it to the right format.
        format = options.get('format', 'sympy')
        result = self._format_represent(result, format)
        return result

    def _format_represent(self, result, format):
        if format == 'sympy' and not isinstance(result, Matrix):
            return to_sympy(result)
        elif format == 'numpy' and not isinstance(result, numpy_ndarray):
            return to_numpy(result)
        elif format == 'scipy.sparse' and \
                not isinstance(result, scipy_sparse_matrix):
            return to_scipy_sparse(result)

        return result


def split_commutative_parts(e):
    """Split into commutative and non-commutative parts."""
    c_part, nc_part = e.args_cnc()
    c_part = list(c_part)
    return c_part, nc_part


def split_qexpr_parts(e):
    """Split an expression into Expr and noncommutative QExpr parts."""
    expr_part = []
    qexpr_part = []
    for arg in e.args:
        if not isinstance(arg, QExpr):
            expr_part.append(arg)
        else:
            qexpr_part.append(arg)
    return expr_part, qexpr_part


def dispatch_method(self, basename, arg, **options):
    """Dispatch a method to the proper handlers."""
    method_name = '%s_%s' % (basename, arg.__class__.__name__)
    if hasattr(self, method_name):
        f = getattr(self, method_name)
        # This can raise and we will allow it to propagate.
        result = f(arg, **options)
        if result is not None:
            return result
    raise NotImplementedError(
        "%s.%s cannot handle: %r" %
        (self.__class__.__name__, basename, arg)
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\bccache.py ===
"""The optional bytecode cache system. This is useful if you have very
complex template situations and the compilation of all those templates
slows down your application too much.

Situations where this is useful are often forking web applications that
are initialized on the first request.
"""

import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .environment import Environment

    class _MemcachedClient(te.Protocol):
        def get(self, key: str) -> bytes: ...

        def set(
            self, key: str, value: bytes, timeout: t.Optional[int] = None
        ) -> None: ...


bc_version = 5
# Magic bytes to identify Jinja bytecode cache files. Contains the
# Python major and minor version to avoid loading incompatible bytecode
# if a project upgrades its Python version.
bc_magic = (
    b"j2"
    + pickle.dumps(bc_version, 2)
    + pickle.dumps((sys.version_info[0] << 24) | sys.version_info[1], 2)
)


class Bucket:
    """Buckets are used to store the bytecode for one template.  It's created
    and initialized by the bytecode cache and passed to the loading functions.

    The buckets get an internal checksum from the cache assigned and use this
    to automatically reject outdated cache material.  Individual bytecode
    cache subclasses don't have to care about cache invalidation.
    """

    def __init__(self, environment: "Environment", key: str, checksum: str) -> None:
        self.environment = environment
        self.key = key
        self.checksum = checksum
        self.reset()

    def reset(self) -> None:
        """Resets the bucket (unloads the bytecode)."""
        self.code: t.Optional[CodeType] = None

    def load_bytecode(self, f: t.BinaryIO) -> None:
        """Loads bytecode from a file or file like object."""
        # make sure the magic header is correct
        magic = f.read(len(bc_magic))
        if magic != bc_magic:
            self.reset()
            return
        # the source code of the file changed, we need to reload
        checksum = pickle.load(f)
        if self.checksum != checksum:
            self.reset()
            return
        # if marshal_load fails then we need to reload
        try:
            self.code = marshal.load(f)
        except (EOFError, ValueError, TypeError):
            self.reset()
            return

    def write_bytecode(self, f: t.IO[bytes]) -> None:
        """Dump the bytecode into the file or file like object passed."""
        if self.code is None:
            raise TypeError("can't write empty bucket")
        f.write(bc_magic)
        pickle.dump(self.checksum, f, 2)
        marshal.dump(self.code, f)

    def bytecode_from_string(self, string: bytes) -> None:
        """Load bytecode from bytes."""
        self.load_bytecode(BytesIO(string))

    def bytecode_to_string(self) -> bytes:
        """Return the bytecode as bytes."""
        out = BytesIO()
        self.write_bytecode(out)
        return out.getvalue()


class BytecodeCache:
    """To implement your own bytecode cache you have to subclass this class
    and override :meth:`load_bytecode` and :meth:`dump_bytecode`.  Both of
    these methods are passed a :class:`~jinja2.bccache.Bucket`.

    A very basic bytecode cache that saves the bytecode on the file system::

        from os import path

        class MyCache(BytecodeCache):

            def __init__(self, directory):
                self.directory = directory

            def load_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                if path.exists(filename):
                    with open(filename, 'rb') as f:
                        bucket.load_bytecode(f)

            def dump_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                with open(filename, 'wb') as f:
                    bucket.write_bytecode(f)

    A more advanced version of a filesystem based bytecode cache is part of
    Jinja.
    """

    def load_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to load bytecode into a
        bucket.  If they are not able to find code in the cache for the
        bucket, it must not do anything.
        """
        raise NotImplementedError()

    def dump_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to write the bytecode
        from a bucket back to the cache.  If it unable to do so it must not
        fail silently but raise an exception.
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """Clears the cache.  This method is not used by Jinja but should be
        implemented to allow applications to clear the bytecode cache used
        by a particular environment.
        """

    def get_cache_key(
        self, name: str, filename: t.Optional[t.Union[str]] = None
    ) -> str:
        """Returns the unique hash key for this template name."""
        hash = sha1(name.encode("utf-8"))

        if filename is not None:
            hash.update(f"|{filename}".encode())

        return hash.hexdigest()

    def get_source_checksum(self, source: str) -> str:
        """Returns a checksum for the source."""
        return sha1(source.encode("utf-8")).hexdigest()

    def get_bucket(
        self,
        environment: "Environment",
        name: str,
        filename: t.Optional[str],
        source: str,
    ) -> Bucket:
        """Return a cache bucket for the given template.  All arguments are
        mandatory but filename may be `None`.
        """
        key = self.get_cache_key(name, filename)
        checksum = self.get_source_checksum(source)
        bucket = Bucket(environment, key, checksum)
        self.load_bytecode(bucket)
        return bucket

    def set_bucket(self, bucket: Bucket) -> None:
        """Put the bucket into the cache."""
        self.dump_bytecode(bucket)


class FileSystemBytecodeCache(BytecodeCache):
    """A bytecode cache that stores bytecode on the filesystem.  It accepts
    two arguments: The directory where the cache items are stored and a
    pattern string that is used to build the filename.

    If no directory is specified a default cache directory is selected.  On
    Windows the user's temp directory is used, on UNIX systems a directory
    is created for the user in the system temp directory.

    The pattern can be used to have multiple separate caches operate on the
    same directory.  The default pattern is ``'__jinja2_%s.cache'``.  ``%s``
    is replaced with the cache key.

    >>> bcc = FileSystemBytecodeCache('/tmp/jinja_cache', '%s.cache')

    This bytecode cache supports clearing of the cache using the clear method.
    """

    def __init__(
        self, directory: t.Optional[str] = None, pattern: str = "__jinja2_%s.cache"
    ) -> None:
        if directory is None:
            directory = self._get_default_cache_dir()
        self.directory = directory
        self.pattern = pattern

    def _get_default_cache_dir(self) -> str:
        def _unsafe_dir() -> "te.NoReturn":
            raise RuntimeError(
                "Cannot determine safe temp directory.  You "
                "need to explicitly provide one."
            )

        tmpdir = tempfile.gettempdir()

        # On windows the temporary directory is used specific unless
        # explicitly forced otherwise.  We can just use that.
        if os.name == "nt":
            return tmpdir
        if not hasattr(os, "getuid"):
            _unsafe_dir()

        dirname = f"_jinja2-cache-{os.getuid()}"
        actual_dir = os.path.join(tmpdir, dirname)

        try:
            os.mkdir(actual_dir, stat.S_IRWXU)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        try:
            os.chmod(actual_dir, stat.S_IRWXU)
            actual_dir_stat = os.lstat(actual_dir)
            if (
                actual_dir_stat.st_uid != os.getuid()
                or not stat.S_ISDIR(actual_dir_stat.st_mode)
                or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU
            ):
                _unsafe_dir()
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        actual_dir_stat = os.lstat(actual_dir)
        if (
            actual_dir_stat.st_uid != os.getuid()
            or not stat.S_ISDIR(actual_dir_stat.st_mode)
            or stat.S_IMODE(actual_dir_stat.st_mode) != stat.S_IRWXU
        ):
            _unsafe_dir()

        return actual_dir

    def _get_cache_filename(self, bucket: Bucket) -> str:
        return os.path.join(self.directory, self.pattern % (bucket.key,))

    def load_bytecode(self, bucket: Bucket) -> None:
        filename = self._get_cache_filename(bucket)

        # Don't test for existence before opening the file, since the
        # file could disappear after the test before the open.
        try:
            f = open(filename, "rb")
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            # PermissionError can occur on Windows when an operation is
            # in progress, such as calling clear().
            return

        with f:
            bucket.load_bytecode(f)

    def dump_bytecode(self, bucket: Bucket) -> None:
        # Write to a temporary file, then rename to the real name after
        # writing. This avoids another process reading the file before
        # it is fully written.
        name = self._get_cache_filename(bucket)
        f = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=os.path.dirname(name),
            prefix=os.path.basename(name),
            suffix=".tmp",
            delete=False,
        )

        def remove_silent() -> None:
            try:
                os.remove(f.name)
            except OSError:
                # Another process may have called clear(). On Windows,
                # another program may be holding the file open.
                pass

        try:
            with f:
                bucket.write_bytecode(f)
        except BaseException:
            remove_silent()
            raise

        try:
            os.replace(f.name, name)
        except OSError:
            # Another process may have called clear(). On Windows,
            # another program may be holding the file open.
            remove_silent()
        except BaseException:
            remove_silent()
            raise

    def clear(self) -> None:
        # imported lazily here because google app-engine doesn't support
        # write access on the file system and the function does not exist
        # normally.
        from os import remove

        files = fnmatch.filter(os.listdir(self.directory), self.pattern % ("*",))
        for filename in files:
            try:
                remove(os.path.join(self.directory, filename))
            except OSError:
                pass


class MemcachedBytecodeCache(BytecodeCache):
    """This class implements a bytecode cache that uses a memcache cache for
    storing the information.  It does not enforce a specific memcache library
    (tummy's memcache or cmemcache) but will accept any class that provides
    the minimal interface required.

    Libraries compatible with this class:

    -   `cachelib <https://github.com/pallets/cachelib>`_
    -   `python-memcached <https://pypi.org/project/python-memcached/>`_

    (Unfortunately the django cache interface is not compatible because it
    does not support storing binary data, only text. You can however pass
    the underlying cache client to the bytecode cache which is available
    as `django.core.cache.cache._client`.)

    The minimal interface for the client passed to the constructor is this:

    .. class:: MinimalClientInterface

        .. method:: set(key, value[, timeout])

            Stores the bytecode in the cache.  `value` is a string and
            `timeout` the timeout of the key.  If timeout is not provided
            a default timeout or no timeout should be assumed, if it's
            provided it's an integer with the number of seconds the cache
            item should exist.

        .. method:: get(key)

            Returns the value for the cache key.  If the item does not
            exist in the cache the return value must be `None`.

    The other arguments to the constructor are the prefix for all keys that
    is added before the actual cache key and the timeout for the bytecode in
    the cache system.  We recommend a high (or no) timeout.

    This bytecode cache does not support clearing of used items in the cache.
    The clear method is a no-operation function.

    .. versionadded:: 2.7
       Added support for ignoring memcache errors through the
       `ignore_memcache_errors` parameter.
    """

    def __init__(
        self,
        client: "_MemcachedClient",
        prefix: str = "jinja2/bytecode/",
        timeout: t.Optional[int] = None,
        ignore_memcache_errors: bool = True,
    ):
        self.client = client
        self.prefix = prefix
        self.timeout = timeout
        self.ignore_memcache_errors = ignore_memcache_errors

    def load_bytecode(self, bucket: Bucket) -> None:
        try:
            code = self.client.get(self.prefix + bucket.key)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
        else:
            bucket.bytecode_from_string(code)

    def dump_bytecode(self, bucket: Bucket) -> None:
        key = self.prefix + bucket.key
        value = bucket.bytecode_to_string()

        try:
            if self.timeout is not None:
                self.client.set(key, value, self.timeout)
            else:
                self.client.set(key, value)
        except Exception:
            if not self.ignore_memcache_errors:
                raise

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\effect.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    String,
    Set,
    Bool,
    Integer,
    Float,
)

from .colors import ColorChoice


class TintEffect(Serialisable):

    tagname = "tint"

    hue = Integer()
    amt = Integer()

    def __init__(self,
                 hue=0,
                 amt=0,
                ):
        self.hue = hue
        self.amt = amt


class LuminanceEffect(Serialisable):

    tagname = "lum"

    bright = Integer() #Pct ?
    contrast = Integer() #Pct#

    def __init__(self,
                 bright=0,
                 contrast=0,
                ):
        self.bright = bright
        self.contrast = contrast


class HSLEffect(Serialisable):

    hue = Integer()
    sat = Integer()
    lum = Integer()

    def __init__(self,
                 hue=None,
                 sat=None,
                 lum=None,
                ):
        self.hue = hue
        self.sat = sat
        self.lum = lum


class GrayscaleEffect(Serialisable):

    tagname = "grayscl"


class FillOverlayEffect(Serialisable):

    blend = Set(values=(['over', 'mult', 'screen', 'darken', 'lighten']))

    def __init__(self,
                 blend=None,
                ):
        self.blend = blend


class DuotoneEffect(Serialisable):

    pass

class ColorReplaceEffect(Serialisable):

    pass

class Color(Serialisable):

    pass

class ColorChangeEffect(Serialisable):

    useA = Bool(allow_none=True)
    clrFrom = Typed(expected_type=Color, )
    clrTo = Typed(expected_type=Color, )

    def __init__(self,
                 useA=None,
                 clrFrom=None,
                 clrTo=None,
                ):
        self.useA = useA
        self.clrFrom = clrFrom
        self.clrTo = clrTo


class BlurEffect(Serialisable):

    rad = Float()
    grow = Bool(allow_none=True)

    def __init__(self,
                 rad=None,
                 grow=None,
                ):
        self.rad = rad
        self.grow = grow


class BiLevelEffect(Serialisable):

    thresh = Integer()

    def __init__(self,
                 thresh=None,
                ):
        self.thresh = thresh


class AlphaReplaceEffect(Serialisable):

    a = Integer()

    def __init__(self,
                 a=None,
                ):
        self.a = a


class AlphaModulateFixedEffect(Serialisable):

    amt = Integer()

    def __init__(self,
                 amt=None,
                ):
        self.amt = amt


class EffectContainer(Serialisable):

    type = Set(values=(['sib', 'tree']))
    name = String(allow_none=True)

    def __init__(self,
                 type=None,
                 name=None,
                ):
        self.type = type
        self.name = name


class AlphaModulateEffect(Serialisable):

    cont = Typed(expected_type=EffectContainer, )

    def __init__(self,
                 cont=None,
                ):
        self.cont = cont


class AlphaInverseEffect(Serialisable):

    pass

class AlphaFloorEffect(Serialisable):

    pass

class AlphaCeilingEffect(Serialisable):

    pass

class AlphaBiLevelEffect(Serialisable):

    thresh = Integer()

    def __init__(self,
                 thresh=None,
                ):
        self.thresh = thresh


class GlowEffect(ColorChoice):

    rad = Float()
    # uses element group EG_ColorChoice
    scrgbClr = ColorChoice.scrgbClr
    srgbClr = ColorChoice.srgbClr
    hslClr = ColorChoice.hslClr
    sysClr = ColorChoice.sysClr
    schemeClr = ColorChoice.schemeClr
    prstClr = ColorChoice.prstClr

    __elements__ = ('scrgbClr', 'srgbClr', 'hslClr', 'sysClr', 'schemeClr', 'prstClr')

    def __init__(self,
                 rad=None,
                 **kw
                ):
        self.rad = rad
        super().__init__(**kw)


class InnerShadowEffect(ColorChoice):

    blurRad = Float()
    dist = Float()
    dir = Integer()
    # uses element group EG_ColorChoice
    scrgbClr = ColorChoice.scrgbClr
    srgbClr = ColorChoice.srgbClr
    hslClr = ColorChoice.hslClr
    sysClr = ColorChoice.sysClr
    schemeClr = ColorChoice.schemeClr
    prstClr = ColorChoice.prstClr

    __elements__ = ('scrgbClr', 'srgbClr', 'hslClr', 'sysClr', 'schemeClr', 'prstClr')

    def __init__(self,
                 blurRad=None,
                 dist=None,
                 dir=None,
                 **kw
                 ):
        self.blurRad = blurRad
        self.dist = dist
        self.dir = dir
        super().__init__(**kw)


class OuterShadow(ColorChoice):

    tagname = "outerShdw"

    blurRad = Float(allow_none=True)
    dist = Float(allow_none=True)
    dir = Integer(allow_none=True)
    sx = Integer(allow_none=True)
    sy = Integer(allow_none=True)
    kx = Integer(allow_none=True)
    ky = Integer(allow_none=True)
    algn = Set(values=['tl', 't', 'tr', 'l', 'ctr', 'r', 'bl', 'b', 'br'])
    rotWithShape = Bool(allow_none=True)
    # uses element group EG_ColorChoice
    scrgbClr = ColorChoice.scrgbClr
    srgbClr = ColorChoice.srgbClr
    hslClr = ColorChoice.hslClr
    sysClr = ColorChoice.sysClr
    schemeClr = ColorChoice.schemeClr
    prstClr = ColorChoice.prstClr

    __elements__ = ('scrgbClr', 'srgbClr', 'hslClr', 'sysClr', 'schemeClr', 'prstClr')

    def __init__(self,
                 blurRad=None,
                 dist=None,
                 dir=None,
                 sx=None,
                 sy=None,
                 kx=None,
                 ky=None,
                 algn=None,
                 rotWithShape=None,
                 **kw
                ):
        self.blurRad = blurRad
        self.dist = dist
        self.dir = dir
        self.sx = sx
        self.sy = sy
        self.kx = kx
        self.ky = ky
        self.algn = algn
        self.rotWithShape = rotWithShape
        super().__init__(**kw)


class PresetShadowEffect(ColorChoice):

    prst = Set(values=(['shdw1', 'shdw2', 'shdw3', 'shdw4', 'shdw5', 'shdw6',
                        'shdw7', 'shdw8', 'shdw9', 'shdw10', 'shdw11', 'shdw12', 'shdw13',
                        'shdw14', 'shdw15', 'shdw16', 'shdw17', 'shdw18', 'shdw19', 'shdw20']))
    dist = Float()
    dir = Integer()
    # uses element group EG_ColorChoice
    scrgbClr = ColorChoice.scrgbClr
    srgbClr = ColorChoice.srgbClr
    hslClr = ColorChoice.hslClr
    sysClr = ColorChoice.sysClr
    schemeClr = ColorChoice.schemeClr
    prstClr = ColorChoice.prstClr

    __elements__ = ('scrgbClr', 'srgbClr', 'hslClr', 'sysClr', 'schemeClr', 'prstClr')

    def __init__(self,
                 prst=None,
                 dist=None,
                 dir=None,
                 **kw
                ):
        self.prst = prst
        self.dist = dist
        self.dir = dir
        super().__init__(**kw)


class ReflectionEffect(Serialisable):

    blurRad = Float()
    stA = Integer()
    stPos = Integer()
    endA = Integer()
    endPos = Integer()
    dist = Float()
    dir = Integer()
    fadeDir = Integer()
    sx = Integer()
    sy = Integer()
    kx = Integer()
    ky = Integer()
    algn = Set(values=(['tl', 't', 'tr', 'l', 'ctr', 'r', 'bl', 'b', 'br']))
    rotWithShape = Bool(allow_none=True)

    def __init__(self,
                 blurRad=None,
                 stA=None,
                 stPos=None,
                 endA=None,
                 endPos=None,
                 dist=None,
                 dir=None,
                 fadeDir=None,
                 sx=None,
                 sy=None,
                 kx=None,
                 ky=None,
                 algn=None,
                 rotWithShape=None,
                ):
        self.blurRad = blurRad
        self.stA = stA
        self.stPos = stPos
        self.endA = endA
        self.endPos = endPos
        self.dist = dist
        self.dir = dir
        self.fadeDir = fadeDir
        self.sx = sx
        self.sy = sy
        self.kx = kx
        self.ky = ky
        self.algn = algn
        self.rotWithShape = rotWithShape


class SoftEdgesEffect(Serialisable):

    rad = Float()

    def __init__(self,
                 rad=None,
                ):
        self.rad = rad


class EffectList(Serialisable):

    blur = Typed(expected_type=BlurEffect, allow_none=True)
    fillOverlay = Typed(expected_type=FillOverlayEffect, allow_none=True)
    glow = Typed(expected_type=GlowEffect, allow_none=True)
    innerShdw = Typed(expected_type=InnerShadowEffect, allow_none=True)
    outerShdw = Typed(expected_type=OuterShadow, allow_none=True)
    prstShdw = Typed(expected_type=PresetShadowEffect, allow_none=True)
    reflection = Typed(expected_type=ReflectionEffect, allow_none=True)
    softEdge = Typed(expected_type=SoftEdgesEffect, allow_none=True)

    __elements__ = ('blur', 'fillOverlay', 'glow', 'innerShdw', 'outerShdw',
                    'prstShdw', 'reflection', 'softEdge')

    def __init__(self,
                 blur=None,
                 fillOverlay=None,
                 glow=None,
                 innerShdw=None,
                 outerShdw=None,
                 prstShdw=None,
                 reflection=None,
                 softEdge=None,
                ):
        self.blur = blur
        self.fillOverlay = fillOverlay
        self.glow = glow
        self.innerShdw = innerShdw
        self.outerShdw = outerShdw
        self.prstShdw = prstShdw
        self.reflection = reflection
        self.softEdge = softEdge

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\boolean.py ===
from __future__ import annotations

import numbers
from typing import (
    TYPE_CHECKING,
    ClassVar,
    cast,
)

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna

from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    import pyarrow

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
        type_t,
    )


@register_extension_dtype
class BooleanDtype(BaseMaskedDtype):
    """
    Extension dtype for boolean data.

    .. warning::

       BooleanDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.BooleanDtype()
    BooleanDtype
    """

    name: ClassVar[str] = "boolean"

    # https://github.com/python/mypy/issues/4125
    # error: Signature of "type" incompatible with supertype "BaseMaskedDtype"
    @property
    def type(self) -> type:  # type: ignore[override]
        return np.bool_

    @property
    def kind(self) -> str:
        return "b"

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype("bool")

    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return BooleanArray

    def __repr__(self) -> str:
        return "BooleanDtype"

    @property
    def _is_boolean(self) -> bool:
        return True

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BooleanArray:
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        if array.type != pyarrow.bool_() and not pyarrow.types.is_null(array.type):
            raise TypeError(f"Expected array of boolean type, got {array.type} instead")

        if isinstance(array, pyarrow.Array):
            chunks = [array]
            length = len(array)
        else:
            # pyarrow.ChunkedArray
            chunks = array.chunks
            length = array.length()

        if pyarrow.types.is_null(array.type):
            mask = np.ones(length, dtype=bool)
            # No need to init data, since all null
            data = np.empty(length, dtype=bool)
            return BooleanArray(data, mask)

        results = []
        for arr in chunks:
            buflist = arr.buffers()
            data = pyarrow.BooleanArray.from_buffers(
                arr.type, len(arr), [None, buflist[1]], offset=arr.offset
            ).to_numpy(zero_copy_only=False)
            if arr.null_count != 0:
                mask = pyarrow.BooleanArray.from_buffers(
                    arr.type, len(arr), [None, buflist[0]], offset=arr.offset
                ).to_numpy(zero_copy_only=False)
                mask = ~mask
            else:
                mask = np.zeros(len(arr), dtype=bool)

            bool_arr = BooleanArray(data, mask)
            results.append(bool_arr)

        if not results:
            return BooleanArray(
                np.array([], dtype=np.bool_), np.array([], dtype=np.bool_)
            )
        else:
            return BooleanArray._concat_same_type(results)


def coerce_to_array(
    values, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    if isinstance(values, BooleanArray):
        if mask is not None:
            raise ValueError("cannot pass mask for BooleanArray input")
        values, mask = values._data, values._mask
        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    mask_values = None
    if isinstance(values, np.ndarray) and values.dtype == np.bool_:
        if copy:
            values = values.copy()
    elif isinstance(values, np.ndarray) and values.dtype.kind in "iufcb":
        mask_values = isna(values)

        values_bool = np.zeros(len(values), dtype=bool)
        values_bool[~mask_values] = values[~mask_values].astype(bool)

        if not np.all(
            values_bool[~mask_values].astype(values.dtype) == values[~mask_values]
        ):
            raise TypeError("Need to pass bool-like values")

        values = values_bool
    else:
        values_object = np.asarray(values, dtype=object)

        inferred_dtype = lib.infer_dtype(values_object, skipna=True)
        integer_like = ("floating", "integer", "mixed-integer-float")
        if inferred_dtype not in ("boolean", "empty") + integer_like:
            raise TypeError("Need to pass bool-like values")

        # mypy does not narrow the type of mask_values to npt.NDArray[np.bool_]
        # within this branch, it assumes it can also be None
        mask_values = cast("npt.NDArray[np.bool_]", isna(values_object))
        values = np.zeros(len(values), dtype=bool)
        values[~mask_values] = values_object[~mask_values].astype(bool)

        # if the values were integer-like, validate it were actually 0/1's
        if (inferred_dtype in integer_like) and not (
            np.all(
                values[~mask_values].astype(float)
                == values_object[~mask_values].astype(float)
            )
        ):
            raise TypeError("Need to pass bool-like values")

    if mask is None and mask_values is None:
        mask = np.zeros(values.shape, dtype=bool)
    elif mask is None:
        mask = mask_values
    else:
        if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
            if mask_values is not None:
                mask = mask | mask_values
            else:
                if copy:
                    mask = mask.copy()
        else:
            mask = np.array(mask, dtype=bool)
            if mask_values is not None:
                mask = mask | mask_values

    if values.shape != mask.shape:
        raise ValueError("values.shape and mask.shape must match")

    return values, mask


class BooleanArray(BaseMaskedArray):
    """
    Array of boolean (True/False) data with missing values.

    This is a pandas Extension array for boolean data, under the hood
    represented by 2 numpy arrays: a boolean array with the data and
    a boolean array with the mask (True indicating missing).

    BooleanArray implements Kleene logic (sometimes called three-value
    logic) for logical operations. See :ref:`boolean.kleene` for more.

    To construct an BooleanArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples
    below).

    .. warning::

       BooleanArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d boolean-dtype array with the data.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values (True
        indicates missing).
    copy : bool, default False
        Whether to copy the `values` and `mask` arrays.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    BooleanArray

    Examples
    --------
    Create an BooleanArray with :func:`pandas.array`:

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = False
    # Fill values used for any/all
    # Incompatible types in assignment (expression has type "bool", base class
    # "BaseMaskedArray" defined the type as "<typing special form>")
    _truthy_value = True  # type: ignore[assignment]
    _falsey_value = False  # type: ignore[assignment]
    _TRUE_VALUES = {"True", "TRUE", "true", "1", "1.0"}
    _FALSE_VALUES = {"False", "FALSE", "false", "0", "0.0"}

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        result = super()._simple_new(values, mask)
        result._dtype = BooleanDtype()
        return result

    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = False
    ) -> None:
        if not (isinstance(values, np.ndarray) and values.dtype == np.bool_):
            raise TypeError(
                "values should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        self._dtype = BooleanDtype()
        super().__init__(values, mask, copy=copy)

    @property
    def dtype(self) -> BooleanDtype:
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: list[str],
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
        true_values: list[str] | None = None,
        false_values: list[str] | None = None,
    ) -> BooleanArray:
        true_values_union = cls._TRUE_VALUES.union(true_values or [])
        false_values_union = cls._FALSE_VALUES.union(false_values or [])

        def map_string(s) -> bool:
            if s in true_values_union:
                return True
            elif s in false_values_union:
                return False
            else:
                raise ValueError(f"{s} cannot be cast to bool")

        scalars = np.array(strings, dtype=object)
        mask = isna(scalars)
        scalars[~mask] = list(map(map_string, scalars[~mask]))
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number, bool, np.bool_)

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if dtype:
            assert dtype == "boolean"
        return coerce_to_array(value, copy=copy)

    def _logical_method(self, other, op):
        assert op.__name__ in {"or_", "ror_", "and_", "rand_", "xor", "rxor"}
        other_is_scalar = lib.is_scalar(other)
        mask = None

        if isinstance(other, BooleanArray):
            other, mask = other._data, other._mask
        elif is_list_like(other):
            other = np.asarray(other, dtype="bool")
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            other, mask = coerce_to_array(other, copy=False)
        elif isinstance(other, np.bool_):
            other = other.item()

        if other_is_scalar and other is not libmissing.NA and not lib.is_bool(other):
            raise TypeError(
                "'other' should be pandas.NA or a bool. "
                f"Got {type(other).__name__} instead."
            )

        if not other_is_scalar and len(self) != len(other):
            raise ValueError("Lengths must match")

        if op.__name__ in {"or_", "ror_"}:
            result, mask = ops.kleene_or(self._data, other, self._mask, mask)
        elif op.__name__ in {"and_", "rand_"}:
            result, mask = ops.kleene_and(self._data, other, self._mask, mask)
        else:
            # i.e. xor, rxor
            result, mask = ops.kleene_xor(self._data, other, self._mask, mask)

        # i.e. BooleanArray
        return self._maybe_mask_result(result, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> BaseMaskedArray:
        data = self._data
        mask = self._mask
        if name in ("cummin", "cummax"):
            op = getattr(masked_accumulations, name)
            data, mask = op(data, mask, skipna=skipna, **kwargs)
            return self._simple_new(data, mask)
        else:
            from pandas.core.arrays import IntegerArray

            return IntegerArray(data.astype(int), mask)._accumulate(
                name, skipna=skipna, **kwargs
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\base.py ===
"""
Base class for the internal managers. Both BlockManager and ArrayManager
inherit from this class.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
)

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    find_common_type,
    np_can_hold_element,
)
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    SparseDtype,
)

from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
    Index,
    default_index,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        Self,
        Shape,
    )


class _AlreadyWarned:
    def __init__(self):
        # This class is used on the manager level to the block level to
        # ensure that we warn only once. The block method can update the
        # warned_already option without returning a value to keep the
        # interface consistent. This is only a temporary solution for
        # CoW warnings.
        self.warned_already = False


class DataManager(PandasObject):
    # TODO share more methods/attributes

    axes: list[Index]

    @property
    def items(self) -> Index:
        raise AbstractMethodError(self)

    @final
    def __len__(self) -> int:
        return len(self.items)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> Shape:
        return tuple(len(ax) for ax in self.axes)

    @final
    def _validate_set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        old_len = len(self.axes[axis])
        new_len = len(new_labels)

        if axis == 1 and len(self.items) == 0:
            # If we are setting the index on a DataFrame with no columns,
            #  it is OK to change the length.
            pass

        elif new_len != old_len:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, new "
                f"values have {new_len} elements"
            )

    def reindex_indexer(
        self,
        new_axis,
        indexer,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool = True,
        only_slice: bool = False,
    ) -> Self:
        raise AbstractMethodError(self)

    @final
    def reindex_axis(
        self,
        new_index: Index,
        axis: AxisInt,
        fill_value=None,
        only_slice: bool = False,
    ) -> Self:
        """
        Conform data manager to new index.
        """
        new_index, indexer = self.axes[axis].reindex(new_index)

        return self.reindex_indexer(
            new_index,
            indexer,
            axis=axis,
            fill_value=fill_value,
            copy=False,
            only_slice=only_slice,
        )

    def _equal_values(self, other: Self) -> bool:
        """
        To be implemented by the subclasses. Only check the column values
        assuming shape and indexes have already been checked.
        """
        raise AbstractMethodError(self)

    @final
    def equals(self, other: object) -> bool:
        """
        Implementation for DataFrame.equals
        """
        if not isinstance(other, type(self)):
            return False

        self_axes, other_axes = self.axes, other.axes
        if len(self_axes) != len(other_axes):
            return False
        if not all(ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes)):
            return False

        return self._equal_values(other)

    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        raise AbstractMethodError(self)

    def apply_with_block(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        raise AbstractMethodError(self)

    @final
    def isna(self, func) -> Self:
        return self.apply("apply", func=func)

    @final
    def fillna(self, value, limit: int | None, inplace: bool, downcast) -> Self:
        if limit is not None:
            # Do this validation even if we go through one of the no-op paths
            limit = libalgos.validate_limit(None, limit=limit)

        return self.apply_with_block(
            "fillna",
            value=value,
            limit=limit,
            inplace=inplace,
            downcast=downcast,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    @final
    def where(self, other, cond, align: bool) -> Self:
        if align:
            align_keys = ["other", "cond"]
        else:
            align_keys = ["cond"]
            other = extract_array(other, extract_numpy=True)

        return self.apply_with_block(
            "where",
            align_keys=align_keys,
            other=other,
            cond=cond,
            using_cow=using_copy_on_write(),
        )

    @final
    def putmask(self, mask, new, align: bool = True, warn: bool = True) -> Self:
        if align:
            align_keys = ["new", "mask"]
        else:
            align_keys = ["mask"]
            new = extract_array(new, extract_numpy=True)

        already_warned = None
        if warn_copy_on_write():
            already_warned = _AlreadyWarned()
            if not warn:
                already_warned.warned_already = True

        return self.apply_with_block(
            "putmask",
            align_keys=align_keys,
            mask=mask,
            new=new,
            using_cow=using_copy_on_write(),
            already_warned=already_warned,
        )

    @final
    def round(self, decimals: int, using_cow: bool = False) -> Self:
        return self.apply_with_block(
            "round",
            decimals=decimals,
            using_cow=using_cow,
        )

    @final
    def replace(self, to_replace, value, inplace: bool) -> Self:
        inplace = validate_bool_kwarg(inplace, "inplace")
        # NDFrame.replace ensures the not-is_list_likes here
        assert not lib.is_list_like(to_replace)
        assert not lib.is_list_like(value)
        return self.apply_with_block(
            "replace",
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    @final
    def replace_regex(self, **kwargs) -> Self:
        return self.apply_with_block(
            "_replace_regex",
            **kwargs,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    @final
    def replace_list(
        self,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> Self:
        """do a list replace"""
        inplace = validate_bool_kwarg(inplace, "inplace")

        bm = self.apply_with_block(
            "replace_list",
            src_list=src_list,
            dest_list=dest_list,
            inplace=inplace,
            regex=regex,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )
        bm._consolidate_inplace()
        return bm

    def interpolate(self, inplace: bool, **kwargs) -> Self:
        return self.apply_with_block(
            "interpolate",
            inplace=inplace,
            **kwargs,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    def pad_or_backfill(self, inplace: bool, **kwargs) -> Self:
        return self.apply_with_block(
            "pad_or_backfill",
            inplace=inplace,
            **kwargs,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    def shift(self, periods: int, fill_value) -> Self:
        if fill_value is lib.no_default:
            fill_value = None

        return self.apply_with_block("shift", periods=periods, fill_value=fill_value)

    # --------------------------------------------------------------------
    # Consolidation: No-ops for all but BlockManager

    def is_consolidated(self) -> bool:
        return True

    def consolidate(self) -> Self:
        return self

    def _consolidate_inplace(self) -> None:
        return


class SingleDataManager(DataManager):
    @property
    def ndim(self) -> Literal[1]:
        return 1

    @final
    @property
    def array(self) -> ArrayLike:
        """
        Quick access to the backing array of the Block or SingleArrayManager.
        """
        # error: "SingleDataManager" has no attribute "arrays"; maybe "array"
        return self.arrays[0]  # type: ignore[attr-defined]

    def setitem_inplace(self, indexer, value, warn: bool = True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        arr = self.array

        # EAs will do this validation in their own __setitem__ methods.
        if isinstance(arr, np.ndarray):
            # Note: checking for ndarray instead of np.dtype means we exclude
            #  dt64/td64, which do their own validation.
            value = np_can_hold_element(arr.dtype, value)

        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == 1:
            # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
            value = value[0, ...]

        arr[indexer] = value

    def grouped_reduce(self, func):
        arr = self.array
        res = func(arr)
        index = default_index(len(res))

        mgr = type(self).from_array(res, index)
        return mgr

    @classmethod
    def from_array(cls, arr: ArrayLike, index: Index):
        raise AbstractMethodError(cls)


def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    if not len(dtypes):
        return None

    return find_common_type(dtypes)


def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    # TODO: https://github.com/pandas-dev/pandas/issues/22791
    # Give EAs some input on what happens here. Sparse needs this.
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype("object")
    elif dtype == np.dtype(str):
        dtype = np.dtype("object")
    return dtype