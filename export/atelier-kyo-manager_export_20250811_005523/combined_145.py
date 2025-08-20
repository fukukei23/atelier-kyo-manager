
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\index\sources.py ===
import logging
import mimetypes
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from pip._vendor.packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)

from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url, url_to_path
from pip._internal.vcs import is_url

logger = logging.getLogger(__name__)

FoundCandidates = Iterable[InstallationCandidate]
FoundLinks = Iterable[Link]
CandidatesFromPage = Callable[[Link], Iterable[InstallationCandidate]]
PageValidator = Callable[[Link], bool]


class LinkSource:
    @property
    def link(self) -> Optional[Link]:
        """Returns the underlying link, if there's one."""
        raise NotImplementedError()

    def page_candidates(self) -> FoundCandidates:
        """Candidates found by parsing an archive listing HTML file."""
        raise NotImplementedError()

    def file_links(self) -> FoundLinks:
        """Links found by specifying archives directly."""
        raise NotImplementedError()


def _is_html_file(file_url: str) -> bool:
    return mimetypes.guess_type(file_url, strict=False)[0] == "text/html"


class _FlatDirectoryToUrls:
    """Scans directory and caches results"""

    def __init__(self, path: str) -> None:
        self._path = path
        self._page_candidates: List[str] = []
        self._project_name_to_urls: Dict[str, List[str]] = defaultdict(list)
        self._scanned_directory = False

    def _scan_directory(self) -> None:
        """Scans directory once and populates both page_candidates
        and project_name_to_urls at the same time
        """
        for entry in os.scandir(self._path):
            url = path_to_url(entry.path)
            if _is_html_file(url):
                self._page_candidates.append(url)
                continue

            # File must have a valid wheel or sdist name,
            # otherwise not worth considering as a package
            try:
                project_filename = parse_wheel_filename(entry.name)[0]
            except InvalidWheelFilename:
                try:
                    project_filename = parse_sdist_filename(entry.name)[0]
                except InvalidSdistFilename:
                    continue

            self._project_name_to_urls[project_filename].append(url)
        self._scanned_directory = True

    @property
    def page_candidates(self) -> List[str]:
        if not self._scanned_directory:
            self._scan_directory()

        return self._page_candidates

    @property
    def project_name_to_urls(self) -> Dict[str, List[str]]:
        if not self._scanned_directory:
            self._scan_directory()

        return self._project_name_to_urls


class _FlatDirectorySource(LinkSource):
    """Link source specified by ``--find-links=<path-to-dir>``.

    This looks the content of the directory, and returns:

    * ``page_candidates``: Links listed on each HTML file in the directory.
    * ``file_candidates``: Archives in the directory.
    """

    _paths_to_urls: Dict[str, _FlatDirectoryToUrls] = {}

    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        path: str,
        project_name: str,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._project_name = canonicalize_name(project_name)

        # Get existing instance of _FlatDirectoryToUrls if it exists
        if path in self._paths_to_urls:
            self._path_to_urls = self._paths_to_urls[path]
        else:
            self._path_to_urls = _FlatDirectoryToUrls(path=path)
            self._paths_to_urls[path] = self._path_to_urls

    @property
    def link(self) -> Optional[Link]:
        return None

    def page_candidates(self) -> FoundCandidates:
        for url in self._path_to_urls.page_candidates:
            yield from self._candidates_from_page(Link(url))

    def file_links(self) -> FoundLinks:
        for url in self._path_to_urls.project_name_to_urls[self._project_name]:
            yield Link(url)


class _LocalFileSource(LinkSource):
    """``--find-links=<path-or-url>`` or ``--[extra-]index-url=<path-or-url>``.

    If a URL is supplied, it must be a ``file:`` URL. If a path is supplied to
    the option, it is converted to a URL first. This returns:

    * ``page_candidates``: Links listed on an HTML file.
    * ``file_candidates``: The non-HTML file.
    """

    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        link: Link,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._link = link

    @property
    def link(self) -> Optional[Link]:
        return self._link

    def page_candidates(self) -> FoundCandidates:
        if not _is_html_file(self._link.url):
            return
        yield from self._candidates_from_page(self._link)

    def file_links(self) -> FoundLinks:
        if _is_html_file(self._link.url):
            return
        yield self._link


class _RemoteFileSource(LinkSource):
    """``--find-links=<url>`` or ``--[extra-]index-url=<url>``.

    This returns:

    * ``page_candidates``: Links listed on an HTML file.
    * ``file_candidates``: The non-HTML file.
    """

    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        page_validator: PageValidator,
        link: Link,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._page_validator = page_validator
        self._link = link

    @property
    def link(self) -> Optional[Link]:
        return self._link

    def page_candidates(self) -> FoundCandidates:
        if not self._page_validator(self._link):
            return
        yield from self._candidates_from_page(self._link)

    def file_links(self) -> FoundLinks:
        yield self._link


class _IndexDirectorySource(LinkSource):
    """``--[extra-]index-url=<path-to-directory>``.

    This is treated like a remote URL; ``candidates_from_page`` contains logic
    for this by appending ``index.html`` to the link.
    """

    def __init__(
        self,
        candidates_from_page: CandidatesFromPage,
        link: Link,
    ) -> None:
        self._candidates_from_page = candidates_from_page
        self._link = link

    @property
    def link(self) -> Optional[Link]:
        return self._link

    def page_candidates(self) -> FoundCandidates:
        yield from self._candidates_from_page(self._link)

    def file_links(self) -> FoundLinks:
        return ()


def build_source(
    location: str,
    *,
    candidates_from_page: CandidatesFromPage,
    page_validator: PageValidator,
    expand_dir: bool,
    cache_link_parsing: bool,
    project_name: str,
) -> Tuple[Optional[str], Optional[LinkSource]]:
    path: Optional[str] = None
    url: Optional[str] = None
    if os.path.exists(location):  # Is a local path.
        url = path_to_url(location)
        path = location
    elif location.startswith("file:"):  # A file: URL.
        url = location
        path = url_to_path(location)
    elif is_url(location):
        url = location

    if url is None:
        msg = (
            "Location '%s' is ignored: "
            "it is either a non-existing path or lacks a specific scheme."
        )
        logger.warning(msg, location)
        return (None, None)

    if path is None:
        source: LinkSource = _RemoteFileSource(
            candidates_from_page=candidates_from_page,
            page_validator=page_validator,
            link=Link(url, cache_link_parsing=cache_link_parsing),
        )
        return (url, source)

    if os.path.isdir(path):
        if expand_dir:
            source = _FlatDirectorySource(
                candidates_from_page=candidates_from_page,
                path=path,
                project_name=project_name,
            )
        else:
            source = _IndexDirectorySource(
                candidates_from_page=candidates_from_page,
                link=Link(url, cache_link_parsing=cache_link_parsing),
            )
        return (url, source)
    elif os.path.isfile(path):
        source = _LocalFileSource(
            candidates_from_page=candidates_from_page,
            link=Link(url, cache_link_parsing=cache_link_parsing),
        )
        return (url, source)
    logger.warning(
        "Location '%s' is ignored: it is neither a file nor a directory.",
        location,
    )
    return (url, None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\lineeditor\history.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************

import re, operator, string, sys, os, io

from pyreadline3.logger import log
from pyreadline3.unicode_helper import ensure_str, ensure_unicode

from . import lineobj


class EscapeHistory(Exception):
    pass


class LineHistory(object):
    def __init__(self):
        self.history = []
        self._history_length = 100
        self._history_cursor = 0
        # Cannot expand unicode strings correctly on python2.4
        self.history_filename = os.path.expanduser(ensure_str("~/.history"))
        self.lastcommand = None
        self.query = ""
        self.last_search_for = ""

    def get_current_history_length(self):
        """Return the number of lines currently in the history.
        (This is different from get_history_length(), which returns
        the maximum number of lines that will be written to a history file.)"""
        value = len(self.history)
        log("get_current_history_length:%d" % value)
        return value

    def get_history_length(self):
        """Return the desired length of the history file. Negative values imply
        unlimited history file size."""
        value = self._history_length
        log("get_history_length:%d" % value)
        return value

    def get_history_item(self, index):
        """Return the current contents of history item at index (starts with index 1)."""
        item = self.history[index - 1]
        log("get_history_item: index:%d item:%r" % (index, item))
        return item.get_line_text()

    def set_history_length(self, value):
        log("set_history_length: old:%d new:%d" % (self._history_length, value))
        self._history_length = value

    def get_history_cursor(self):
        value = self._history_cursor
        log("get_history_cursor:%d" % value)
        return value

    def set_history_cursor(self, value):
        log("set_history_cursor: old:%d new:%d" % (self._history_cursor, value))
        self._history_cursor = value

    history_length = property(get_history_length, set_history_length)
    history_cursor = property(get_history_cursor, set_history_cursor)

    def clear_history(self):
        """Clear readline history."""
        self.history[:] = []
        self.history_cursor = 0

    def read_history_file(self, filename=None):
        """Load a readline history file."""
        if filename is None:
            filename = self.history_filename
        try:
            with io.open(filename, "rt", errors="replace") as fd:
                for line in fd:
                    self.add_history(lineobj.ReadLineTextBuffer(line.rstrip()))
        except IOError:
            self.history = []
            self.history_cursor = 0

    def write_history_file(self, filename=None):
        """Save a readline history file."""
        if filename is None:
            filename = self.history_filename
        with io.open(filename, "wt", errors="replace") as fp:
            fp.writelines(
                tuple(
                    line.get_line_text()+"\n"
                    for line in self.history[-self.history_length :]
                )
            )

    def add_history(self, line):
        """Append a line to the history buffer, as if it was the last line typed."""
        line = ensure_unicode(line)
        if not hasattr(line, "get_line_text"):
            line = lineobj.ReadLineTextBuffer(line)
        if not line.get_line_text():
            pass
        elif (
            len(self.history) > 0
            and self.history[-1].get_line_text() == line.get_line_text()
        ):
            pass
        else:
            self.history.append(line)
        self.history_cursor = len(self.history)

    def previous_history(self, current):  # (C-p)
        """Move back through the history list, fetching the previous command."""
        if self.history_cursor == len(self.history):
            # do not use add_history since we do not want to increment cursor
            self.history.append(current.copy())

        if self.history_cursor > 0:
            self.history_cursor -= 1
            current.set_line(self.history[self.history_cursor].get_line_text())
            current.point = lineobj.EndOfLine

    def next_history(self, current):  # (C-n)
        """Move forward through the history list, fetching the next command."""
        if self.history_cursor < len(self.history) - 1:
            self.history_cursor += 1
            current.set_line(self.history[self.history_cursor].get_line_text())

    def beginning_of_history(self):  # (M-<)
        """Move to the first line in the history."""
        self.history_cursor = 0
        if len(self.history) > 0:
            self.l_buffer = self.history[0]

    def end_of_history(self, current):  # (M->)
        """Move to the end of the input history, i.e., the line currently
        being entered."""
        self.history_cursor = len(self.history)
        current.set_line(self.history[-1].get_line_text())

    def reverse_search_history(self, searchfor, startpos=None):
        if startpos is None:
            startpos = self.history_cursor
        origpos = startpos

        result = lineobj.ReadLineTextBuffer("")

        for idx, line in list(enumerate(self.history))[startpos:0:-1]:
            if searchfor in line:
                startpos = idx
                break

        # If we get a new search without change in search term it means
        # someone pushed ctrl-r and we should find the next match
        if self.last_search_for == searchfor and startpos > 0:
            startpos -= 1
            for idx, line in list(enumerate(self.history))[startpos:0:-1]:
                if searchfor in line:
                    startpos = idx
                    break

        if self.history:
            result = self.history[startpos].get_line_text()
        else:
            result = ""
        self.history_cursor = startpos
        self.last_search_for = searchfor
        log(
            "reverse_search_history: old:%d new:%d result:%r"
            % (origpos, self.history_cursor, result)
        )
        return result

    def forward_search_history(self, searchfor, startpos=None):
        if startpos is None:
            startpos = min(
                self.history_cursor, max(0, self.get_current_history_length() - 1)
            )
        # origpos = startpos

        result = lineobj.ReadLineTextBuffer("")

        for idx, line in list(enumerate(self.history))[startpos:]:
            if searchfor in line:
                startpos = idx
                break

        # If we get a new search without change in search term it means
        # someone pushed ctrl-r and we should find the next match
        if (
            self.last_search_for == searchfor
            and startpos < self.get_current_history_length() - 1
        ):
            startpos += 1
            for idx, line in list(enumerate(self.history))[startpos:]:
                if searchfor in line:
                    startpos = idx
                    break

        if self.history:
            result = self.history[startpos].get_line_text()
        else:
            result = ""
        self.history_cursor = startpos
        self.last_search_for = searchfor
        return result

    def _search(self, direction, partial):
        try:
            if (
                self.lastcommand != self.history_search_forward
                and self.lastcommand != self.history_search_backward
            ):
                self.query = "".join(partial[0 : partial.point].get_line_text())
            hcstart = max(self.history_cursor, 0)
            hc = self.history_cursor + direction
            while (direction < 0 and hc >= 0) or (
                direction > 0 and hc < len(self.history)
            ):
                h = self.history[hc]
                if not self.query:
                    self.history_cursor = hc
                    result = lineobj.ReadLineTextBuffer(h, point=len(h.get_line_text()))
                    return result
                elif h.get_line_text().startswith(self.query) and (
                    h != partial.get_line_text()
                ):
                    self.history_cursor = hc
                    result = lineobj.ReadLineTextBuffer(h, point=partial.point)
                    return result
                hc += direction
            else:
                if len(self.history) == 0:
                    pass
                elif hc >= len(self.history) and not self.query:
                    self.history_cursor = len(self.history)
                    return lineobj.ReadLineTextBuffer("", point=0)
                elif (
                    self.history[max(min(hcstart, len(self.history) - 1), 0)]
                    .get_line_text()
                    .startswith(self.query)
                    and self.query
                ):
                    return lineobj.ReadLineTextBuffer(
                        self.history[max(min(hcstart, len(self.history) - 1), 0)],
                        point=partial.point,
                    )
                else:
                    return lineobj.ReadLineTextBuffer(partial, point=partial.point)
                return lineobj.ReadLineTextBuffer(
                    self.query, point=min(len(self.query), partial.point)
                )
        except IndexError:
            raise

    def history_search_forward(self, partial):  # ()
        """Search forward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        return self._search(1, partial)

    def history_search_backward(self, partial):  # ()
        """Search backward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""

        return self._search(-1, partial)


if __name__ == "__main__":
    q = LineHistory()
    r = LineHistory()
    s = LineHistory()
    RL = lineobj.ReadLineTextBuffer
    q.add_history(RL("aaaa"))
    q.add_history(RL("aaba"))
    q.add_history(RL("aaca"))
    q.add_history(RL("akca"))
    q.add_history(RL("bbb"))
    q.add_history(RL("ako"))
    r.add_history(RL("ako"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\tools\timedeltas.py ===
"""
timedelta support tools
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    NaTType,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    disallow_ambiguous_unit,
    parse_timedelta_unit,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.arrays.timedeltas import sequence_to_td64ns

if TYPE_CHECKING:
    from collections.abc import Hashable
    from datetime import timedelta

    from pandas._libs.tslibs.timedeltas import UnitChoices
    from pandas._typing import (
        ArrayLike,
        DateTimeErrorChoices,
    )

    from pandas import (
        Index,
        Series,
        TimedeltaIndex,
    )


@overload
def to_timedelta(
    arg: str | float | timedelta,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Timedelta:
    ...


@overload
def to_timedelta(
    arg: Series,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Series:
    ...


@overload
def to_timedelta(
    arg: list | tuple | range | ArrayLike | Index,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaIndex:
    ...


def to_timedelta(
    arg: str
    | int
    | float
    | timedelta
    | list
    | tuple
    | range
    | ArrayLike
    | Index
    | Series,
    unit: UnitChoices | None = None,
    errors: DateTimeErrorChoices = "raise",
) -> Timedelta | TimedeltaIndex | Series:
    """
    Convert argument to timedelta.

    Timedeltas are absolute differences in times, expressed in difference
    units (e.g. days, hours, minutes, seconds). This method converts
    an argument from a recognized timedelta format / value into
    a Timedelta type.

    Parameters
    ----------
    arg : str, timedelta, list-like or Series
        The data to be converted to timedelta.

        .. versionchanged:: 2.0
            Strings with units 'M', 'Y' and 'y' do not represent
            unambiguous timedelta values and will raise an exception.

    unit : str, optional
        Denotes the unit of the arg for numeric `arg`. Defaults to ``"ns"``.

        Possible values:

        * 'W'
        * 'D' / 'days' / 'day'
        * 'hours' / 'hour' / 'hr' / 'h' / 'H'
        * 'm' / 'minute' / 'min' / 'minutes' / 'T'
        * 's' / 'seconds' / 'sec' / 'second' / 'S'
        * 'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis' / 'L'
        * 'us' / 'microseconds' / 'microsecond' / 'micro' / 'micros' / 'U'
        * 'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'

        Must not be specified when `arg` contains strings and ``errors="raise"``.

        .. deprecated:: 2.2.0
            Units 'H', 'T', 'S', 'L', 'U' and 'N' are deprecated and will be removed
            in a future version. Please use 'h', 'min', 's', 'ms', 'us', and 'ns'
            instead of 'H', 'T', 'S', 'L', 'U' and 'N'.

    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'ignore', then invalid parsing will return the input.

    Returns
    -------
    timedelta
        If parsing succeeded.
        Return type depends on input:

        - list-like: TimedeltaIndex of timedelta64 dtype
        - Series: Series of timedelta64 dtype
        - scalar: Timedelta

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_datetime : Convert argument to datetime.
    convert_dtypes : Convert dtypes.

    Notes
    -----
    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds for string inputs.

    Examples
    --------
    Parsing a single string to a Timedelta:

    >>> pd.to_timedelta('1 days 06:05:01.00003')
    Timedelta('1 days 06:05:01.000030')
    >>> pd.to_timedelta('15.5us')
    Timedelta('0 days 00:00:00.000015500')

    Parsing a list or array of strings:

    >>> pd.to_timedelta(['1 days 06:05:01.00003', '15.5us', 'nan'])
    TimedeltaIndex(['1 days 06:05:01.000030', '0 days 00:00:00.000015500', NaT],
                   dtype='timedelta64[ns]', freq=None)

    Converting numbers by specifying the `unit` keyword argument:

    >>> pd.to_timedelta(np.arange(5), unit='s')
    TimedeltaIndex(['0 days 00:00:00', '0 days 00:00:01', '0 days 00:00:02',
                    '0 days 00:00:03', '0 days 00:00:04'],
                   dtype='timedelta64[ns]', freq=None)
    >>> pd.to_timedelta(np.arange(5), unit='d')
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq=None)
    """
    if unit is not None:
        unit = parse_timedelta_unit(unit)
        disallow_ambiguous_unit(unit)

    if errors not in ("ignore", "raise", "coerce"):
        raise ValueError("errors must be one of 'ignore', 'raise', or 'coerce'.")
    if errors == "ignore":
        # GH#54467
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a future version. "
            "Use to_timedelta without passing `errors` and catch exceptions "
            "explicitly instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if arg is None:
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, unit=unit, errors=errors)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, unit=unit, errors=errors, name=arg.name)
    elif isinstance(arg, np.ndarray) and arg.ndim == 0:
        # extract array scalar and process below
        # error: Incompatible types in assignment (expression has type "object",
        # variable has type "Union[str, int, float, timedelta, List[Any],
        # Tuple[Any, ...], Union[Union[ExtensionArray, ndarray[Any, Any]], Index,
        # Series]]")  [assignment]
        arg = lib.item_from_zerodim(arg)  # type: ignore[assignment]
    elif is_list_like(arg) and getattr(arg, "ndim", 1) == 1:
        return _convert_listlike(arg, unit=unit, errors=errors)
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, timedelta, list, tuple, 1-d array, or Series"
        )

    if isinstance(arg, str) and unit is not None:
        raise ValueError("unit must not be specified if the input is/contains a str")

    # ...so it must be a scalar value. Return scalar.
    return _coerce_scalar_to_timedelta_type(arg, unit=unit, errors=errors)


def _coerce_scalar_to_timedelta_type(
    r, unit: UnitChoices | None = "ns", errors: DateTimeErrorChoices = "raise"
):
    """Convert string 'r' to a timedelta object."""
    result: Timedelta | NaTType

    try:
        result = Timedelta(r, unit)
    except ValueError:
        if errors == "raise":
            raise
        if errors == "ignore":
            return r

        # coerce
        result = NaT

    return result


def _convert_listlike(
    arg,
    unit: UnitChoices | None = None,
    errors: DateTimeErrorChoices = "raise",
    name: Hashable | None = None,
):
    """Convert a list of objects to a timedelta index object."""
    arg_dtype = getattr(arg, "dtype", None)
    if isinstance(arg, (list, tuple)) or arg_dtype is None:
        # This is needed only to ensure that in the case where we end up
        #  returning arg (errors == "ignore"), and where the input is a
        #  generator, we return a useful list-like instead of a
        #  used-up generator
        if not hasattr(arg, "__array__"):
            arg = list(arg)
        arg = np.array(arg, dtype=object)
    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.kind == "m":
        return arg

    try:
        td64arr = sequence_to_td64ns(arg, unit=unit, errors=errors, copy=False)[0]
    except ValueError:
        if errors == "ignore":
            return arg
        else:
            # This else-block accounts for the cases when errors='raise'
            # and errors='coerce'. If errors == 'raise', these errors
            # should be raised. If errors == 'coerce', we shouldn't
            # expect any errors to be raised, since all parsing errors
            # cause coercion to pd.NaT. However, if an error / bug is
            # introduced that causes an Exception to be raised, we would
            # like to surface it.
            raise

    from pandas import TimedeltaIndex

    value = TimedeltaIndex(td64arr, name=name)
    return value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\autofill.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Autofill (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page


@dataclass
class CreditCard:
    #: 16-digit credit card number.
    number: str

    #: Name of the credit card owner.
    name: str

    #: 2-digit expiry month.
    expiry_month: str

    #: 4-digit expiry year.
    expiry_year: str

    #: 3-digit card verification code.
    cvc: str

    def to_json(self):
        json = dict()
        json['number'] = self.number
        json['name'] = self.name
        json['expiryMonth'] = self.expiry_month
        json['expiryYear'] = self.expiry_year
        json['cvc'] = self.cvc
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            number=str(json['number']),
            name=str(json['name']),
            expiry_month=str(json['expiryMonth']),
            expiry_year=str(json['expiryYear']),
            cvc=str(json['cvc']),
        )


@dataclass
class AddressField:
    #: address field name, for example GIVEN_NAME.
    name: str

    #: address field value, for example Jon Doe.
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
class AddressFields:
    '''
    A list of address fields.
    '''
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fields=[AddressField.from_json(i) for i in json['fields']],
        )


@dataclass
class Address:
    #: fields and values defining an address.
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fields=[AddressField.from_json(i) for i in json['fields']],
        )


@dataclass
class AddressUI:
    '''
    Defines how an address can be displayed like in chrome://settings/addresses.
    Address UI is a two dimensional array, each inner array is an "address information line", and when rendered in a UI surface should be displayed as such.
    The following address UI for instance:
    [[{name: "GIVE_NAME", value: "Jon"}, {name: "FAMILY_NAME", value: "Doe"}], [{name: "CITY", value: "Munich"}, {name: "ZIP", value: "81456"}]]
    should allow the receiver to render:
    Jon Doe
    Munich 81456
    '''
    #: A two dimension array containing the representation of values from an address profile.
    address_fields: typing.List[AddressFields]

    def to_json(self):
        json = dict()
        json['addressFields'] = [i.to_json() for i in self.address_fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            address_fields=[AddressFields.from_json(i) for i in json['addressFields']],
        )


class FillingStrategy(enum.Enum):
    '''
    Specified whether a filled field was done so by using the html autocomplete attribute or autofill heuristics.
    '''
    AUTOCOMPLETE_ATTRIBUTE = "autocompleteAttribute"
    AUTOFILL_INFERRED = "autofillInferred"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FilledField:
    #: The type of the field, e.g text, password etc.
    html_type: str

    #: the html id
    id_: str

    #: the html name
    name: str

    #: the field value
    value: str

    #: The actual field type, e.g FAMILY_NAME
    autofill_type: str

    #: The filling strategy
    filling_strategy: FillingStrategy

    #: The frame the field belongs to
    frame_id: page.FrameId

    #: The form field's DOM node
    field_id: dom.BackendNodeId

    def to_json(self):
        json = dict()
        json['htmlType'] = self.html_type
        json['id'] = self.id_
        json['name'] = self.name
        json['value'] = self.value
        json['autofillType'] = self.autofill_type
        json['fillingStrategy'] = self.filling_strategy.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['fieldId'] = self.field_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            html_type=str(json['htmlType']),
            id_=str(json['id']),
            name=str(json['name']),
            value=str(json['value']),
            autofill_type=str(json['autofillType']),
            filling_strategy=FillingStrategy.from_json(json['fillingStrategy']),
            frame_id=page.FrameId.from_json(json['frameId']),
            field_id=dom.BackendNodeId.from_json(json['fieldId']),
        )


def trigger(
        field_id: dom.BackendNodeId,
        frame_id: typing.Optional[page.FrameId] = None,
        card: CreditCard = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Trigger autofill on a form identified by the fieldId.
    If the field and related form cannot be autofilled, returns an error.

    :param field_id: Identifies a field that serves as an anchor for autofill.
    :param frame_id: *(Optional)* Identifies the frame that field belongs to.
    :param card: Credit card information to fill out the form. Credit card data is not saved.
    '''
    params: T_JSON_DICT = dict()
    params['fieldId'] = field_id.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    params['card'] = card.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.trigger',
        'params': params,
    }
    json = yield cmd_dict


def set_addresses(
        addresses: typing.List[Address]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set addresses so that developers can verify their forms implementation.

    :param addresses:
    '''
    params: T_JSON_DICT = dict()
    params['addresses'] = [i.to_json() for i in addresses]
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.setAddresses',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables autofill domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables autofill domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.enable',
    }
    json = yield cmd_dict


@event_class('Autofill.addressFormFilled')
@dataclass
class AddressFormFilled:
    '''
    Emitted when an address form is filled.
    '''
    #: Information about the fields that were filled
    filled_fields: typing.List[FilledField]
    #: An UI representation of the address used to fill the form.
    #: Consists of a 2D array where each child represents an address/profile line.
    address_ui: AddressUI

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddressFormFilled:
        return cls(
            filled_fields=[FilledField.from_json(i) for i in json['filledFields']],
            address_ui=AddressUI.from_json(json['addressUi'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\autofill.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Autofill (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page


@dataclass
class CreditCard:
    #: 16-digit credit card number.
    number: str

    #: Name of the credit card owner.
    name: str

    #: 2-digit expiry month.
    expiry_month: str

    #: 4-digit expiry year.
    expiry_year: str

    #: 3-digit card verification code.
    cvc: str

    def to_json(self):
        json = dict()
        json['number'] = self.number
        json['name'] = self.name
        json['expiryMonth'] = self.expiry_month
        json['expiryYear'] = self.expiry_year
        json['cvc'] = self.cvc
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            number=str(json['number']),
            name=str(json['name']),
            expiry_month=str(json['expiryMonth']),
            expiry_year=str(json['expiryYear']),
            cvc=str(json['cvc']),
        )


@dataclass
class AddressField:
    #: address field name, for example GIVEN_NAME.
    name: str

    #: address field value, for example Jon Doe.
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
class AddressFields:
    '''
    A list of address fields.
    '''
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fields=[AddressField.from_json(i) for i in json['fields']],
        )


@dataclass
class Address:
    #: fields and values defining an address.
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fields=[AddressField.from_json(i) for i in json['fields']],
        )


@dataclass
class AddressUI:
    '''
    Defines how an address can be displayed like in chrome://settings/addresses.
    Address UI is a two dimensional array, each inner array is an "address information line", and when rendered in a UI surface should be displayed as such.
    The following address UI for instance:
    [[{name: "GIVE_NAME", value: "Jon"}, {name: "FAMILY_NAME", value: "Doe"}], [{name: "CITY", value: "Munich"}, {name: "ZIP", value: "81456"}]]
    should allow the receiver to render:
    Jon Doe
    Munich 81456
    '''
    #: A two dimension array containing the representation of values from an address profile.
    address_fields: typing.List[AddressFields]

    def to_json(self):
        json = dict()
        json['addressFields'] = [i.to_json() for i in self.address_fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            address_fields=[AddressFields.from_json(i) for i in json['addressFields']],
        )


class FillingStrategy(enum.Enum):
    '''
    Specified whether a filled field was done so by using the html autocomplete attribute or autofill heuristics.
    '''
    AUTOCOMPLETE_ATTRIBUTE = "autocompleteAttribute"
    AUTOFILL_INFERRED = "autofillInferred"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FilledField:
    #: The type of the field, e.g text, password etc.
    html_type: str

    #: the html id
    id_: str

    #: the html name
    name: str

    #: the field value
    value: str

    #: The actual field type, e.g FAMILY_NAME
    autofill_type: str

    #: The filling strategy
    filling_strategy: FillingStrategy

    #: The frame the field belongs to
    frame_id: page.FrameId

    #: The form field's DOM node
    field_id: dom.BackendNodeId

    def to_json(self):
        json = dict()
        json['htmlType'] = self.html_type
        json['id'] = self.id_
        json['name'] = self.name
        json['value'] = self.value
        json['autofillType'] = self.autofill_type
        json['fillingStrategy'] = self.filling_strategy.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['fieldId'] = self.field_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            html_type=str(json['htmlType']),
            id_=str(json['id']),
            name=str(json['name']),
            value=str(json['value']),
            autofill_type=str(json['autofillType']),
            filling_strategy=FillingStrategy.from_json(json['fillingStrategy']),
            frame_id=page.FrameId.from_json(json['frameId']),
            field_id=dom.BackendNodeId.from_json(json['fieldId']),
        )


def trigger(
        field_id: dom.BackendNodeId,
        frame_id: typing.Optional[page.FrameId] = None,
        card: CreditCard = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Trigger autofill on a form identified by the fieldId.
    If the field and related form cannot be autofilled, returns an error.

    :param field_id: Identifies a field that serves as an anchor for autofill.
    :param frame_id: *(Optional)* Identifies the frame that field belongs to.
    :param card: Credit card information to fill out the form. Credit card data is not saved.
    '''
    params: T_JSON_DICT = dict()
    params['fieldId'] = field_id.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    params['card'] = card.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.trigger',
        'params': params,
    }
    json = yield cmd_dict


def set_addresses(
        addresses: typing.List[Address]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set addresses so that developers can verify their forms implementation.

    :param addresses:
    '''
    params: T_JSON_DICT = dict()
    params['addresses'] = [i.to_json() for i in addresses]
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.setAddresses',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables autofill domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables autofill domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.enable',
    }
    json = yield cmd_dict


@event_class('Autofill.addressFormFilled')
@dataclass
class AddressFormFilled:
    '''
    Emitted when an address form is filled.
    '''
    #: Information about the fields that were filled
    filled_fields: typing.List[FilledField]
    #: An UI representation of the address used to fill the form.
    #: Consists of a 2D array where each child represents an address/profile line.
    address_ui: AddressUI

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddressFormFilled:
        return cls(
            filled_fields=[FilledField.from_json(i) for i in json['filledFields']],
            address_ui=AddressUI.from_json(json['addressUi'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\autofill.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Autofill (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page


@dataclass
class CreditCard:
    #: 16-digit credit card number.
    number: str

    #: Name of the credit card owner.
    name: str

    #: 2-digit expiry month.
    expiry_month: str

    #: 4-digit expiry year.
    expiry_year: str

    #: 3-digit card verification code.
    cvc: str

    def to_json(self):
        json = dict()
        json['number'] = self.number
        json['name'] = self.name
        json['expiryMonth'] = self.expiry_month
        json['expiryYear'] = self.expiry_year
        json['cvc'] = self.cvc
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            number=str(json['number']),
            name=str(json['name']),
            expiry_month=str(json['expiryMonth']),
            expiry_year=str(json['expiryYear']),
            cvc=str(json['cvc']),
        )


@dataclass
class AddressField:
    #: address field name, for example GIVEN_NAME.
    name: str

    #: address field value, for example Jon Doe.
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
class AddressFields:
    '''
    A list of address fields.
    '''
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fields=[AddressField.from_json(i) for i in json['fields']],
        )


@dataclass
class Address:
    #: fields and values defining an address.
    fields: typing.List[AddressField]

    def to_json(self):
        json = dict()
        json['fields'] = [i.to_json() for i in self.fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            fields=[AddressField.from_json(i) for i in json['fields']],
        )


@dataclass
class AddressUI:
    '''
    Defines how an address can be displayed like in chrome://settings/addresses.
    Address UI is a two dimensional array, each inner array is an "address information line", and when rendered in a UI surface should be displayed as such.
    The following address UI for instance:
    [[{name: "GIVE_NAME", value: "Jon"}, {name: "FAMILY_NAME", value: "Doe"}], [{name: "CITY", value: "Munich"}, {name: "ZIP", value: "81456"}]]
    should allow the receiver to render:
    Jon Doe
    Munich 81456
    '''
    #: A two dimension array containing the representation of values from an address profile.
    address_fields: typing.List[AddressFields]

    def to_json(self):
        json = dict()
        json['addressFields'] = [i.to_json() for i in self.address_fields]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            address_fields=[AddressFields.from_json(i) for i in json['addressFields']],
        )


class FillingStrategy(enum.Enum):
    '''
    Specified whether a filled field was done so by using the html autocomplete attribute or autofill heuristics.
    '''
    AUTOCOMPLETE_ATTRIBUTE = "autocompleteAttribute"
    AUTOFILL_INFERRED = "autofillInferred"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class FilledField:
    #: The type of the field, e.g text, password etc.
    html_type: str

    #: the html id
    id_: str

    #: the html name
    name: str

    #: the field value
    value: str

    #: The actual field type, e.g FAMILY_NAME
    autofill_type: str

    #: The filling strategy
    filling_strategy: FillingStrategy

    #: The frame the field belongs to
    frame_id: page.FrameId

    #: The form field's DOM node
    field_id: dom.BackendNodeId

    def to_json(self):
        json = dict()
        json['htmlType'] = self.html_type
        json['id'] = self.id_
        json['name'] = self.name
        json['value'] = self.value
        json['autofillType'] = self.autofill_type
        json['fillingStrategy'] = self.filling_strategy.to_json()
        json['frameId'] = self.frame_id.to_json()
        json['fieldId'] = self.field_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            html_type=str(json['htmlType']),
            id_=str(json['id']),
            name=str(json['name']),
            value=str(json['value']),
            autofill_type=str(json['autofillType']),
            filling_strategy=FillingStrategy.from_json(json['fillingStrategy']),
            frame_id=page.FrameId.from_json(json['frameId']),
            field_id=dom.BackendNodeId.from_json(json['fieldId']),
        )


def trigger(
        field_id: dom.BackendNodeId,
        frame_id: typing.Optional[page.FrameId] = None,
        card: CreditCard = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Trigger autofill on a form identified by the fieldId.
    If the field and related form cannot be autofilled, returns an error.

    :param field_id: Identifies a field that serves as an anchor for autofill.
    :param frame_id: *(Optional)* Identifies the frame that field belongs to.
    :param card: Credit card information to fill out the form. Credit card data is not saved.
    '''
    params: T_JSON_DICT = dict()
    params['fieldId'] = field_id.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    params['card'] = card.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.trigger',
        'params': params,
    }
    json = yield cmd_dict


def set_addresses(
        addresses: typing.List[Address]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set addresses so that developers can verify their forms implementation.

    :param addresses:
    '''
    params: T_JSON_DICT = dict()
    params['addresses'] = [i.to_json() for i in addresses]
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.setAddresses',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables autofill domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables autofill domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Autofill.enable',
    }
    json = yield cmd_dict


@event_class('Autofill.addressFormFilled')
@dataclass
class AddressFormFilled:
    '''
    Emitted when an address form is filled.
    '''
    #: Information about the fields that were filled
    filled_fields: typing.List[FilledField]
    #: An UI representation of the address used to fill the form.
    #: Consists of a 2D array where each child represents an address/profile line.
    address_ui: AddressUI

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AddressFormFilled:
        return cls(
            filled_fields=[FilledField.from_json(i) for i in json['filledFields']],
            address_ui=AddressUI.from_json(json['addressUi'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\__init__.py ===
# __init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any

from . import util as _util
from .engine import AdaptedConnection as AdaptedConnection
from .engine import BaseRow as BaseRow
from .engine import BindTyping as BindTyping
from .engine import ChunkedIteratorResult as ChunkedIteratorResult
from .engine import Compiled as Compiled
from .engine import Connection as Connection
from .engine import create_engine as create_engine
from .engine import create_mock_engine as create_mock_engine
from .engine import create_pool_from_url as create_pool_from_url
from .engine import CreateEnginePlugin as CreateEnginePlugin
from .engine import CursorResult as CursorResult
from .engine import Dialect as Dialect
from .engine import Engine as Engine
from .engine import engine_from_config as engine_from_config
from .engine import ExceptionContext as ExceptionContext
from .engine import ExecutionContext as ExecutionContext
from .engine import FrozenResult as FrozenResult
from .engine import Inspector as Inspector
from .engine import IteratorResult as IteratorResult
from .engine import make_url as make_url
from .engine import MappingResult as MappingResult
from .engine import MergedResult as MergedResult
from .engine import NestedTransaction as NestedTransaction
from .engine import Result as Result
from .engine import result_tuple as result_tuple
from .engine import ResultProxy as ResultProxy
from .engine import RootTransaction as RootTransaction
from .engine import Row as Row
from .engine import RowMapping as RowMapping
from .engine import ScalarResult as ScalarResult
from .engine import Transaction as Transaction
from .engine import TwoPhaseTransaction as TwoPhaseTransaction
from .engine import TypeCompiler as TypeCompiler
from .engine import URL as URL
from .inspection import inspect as inspect
from .pool import AssertionPool as AssertionPool
from .pool import AsyncAdaptedQueuePool as AsyncAdaptedQueuePool
from .pool import (
    FallbackAsyncAdaptedQueuePool as FallbackAsyncAdaptedQueuePool,
)
from .pool import NullPool as NullPool
from .pool import Pool as Pool
from .pool import PoolProxiedConnection as PoolProxiedConnection
from .pool import PoolResetState as PoolResetState
from .pool import QueuePool as QueuePool
from .pool import SingletonThreadPool as SingletonThreadPool
from .pool import StaticPool as StaticPool
from .schema import BaseDDLElement as BaseDDLElement
from .schema import BLANK_SCHEMA as BLANK_SCHEMA
from .schema import CheckConstraint as CheckConstraint
from .schema import Column as Column
from .schema import ColumnDefault as ColumnDefault
from .schema import Computed as Computed
from .schema import Constraint as Constraint
from .schema import DDL as DDL
from .schema import DDLElement as DDLElement
from .schema import DefaultClause as DefaultClause
from .schema import ExecutableDDLElement as ExecutableDDLElement
from .schema import FetchedValue as FetchedValue
from .schema import ForeignKey as ForeignKey
from .schema import ForeignKeyConstraint as ForeignKeyConstraint
from .schema import Identity as Identity
from .schema import Index as Index
from .schema import insert_sentinel as insert_sentinel
from .schema import MetaData as MetaData
from .schema import PrimaryKeyConstraint as PrimaryKeyConstraint
from .schema import Sequence as Sequence
from .schema import Table as Table
from .schema import UniqueConstraint as UniqueConstraint
from .sql import ColumnExpressionArgument as ColumnExpressionArgument
from .sql import NotNullable as NotNullable
from .sql import Nullable as Nullable
from .sql import SelectLabelStyle as SelectLabelStyle
from .sql.expression import Alias as Alias
from .sql.expression import alias as alias
from .sql.expression import AliasedReturnsRows as AliasedReturnsRows
from .sql.expression import all_ as all_
from .sql.expression import and_ as and_
from .sql.expression import any_ as any_
from .sql.expression import asc as asc
from .sql.expression import between as between
from .sql.expression import BinaryExpression as BinaryExpression
from .sql.expression import bindparam as bindparam
from .sql.expression import BindParameter as BindParameter
from .sql.expression import bitwise_not as bitwise_not
from .sql.expression import BooleanClauseList as BooleanClauseList
from .sql.expression import CacheKey as CacheKey
from .sql.expression import Case as Case
from .sql.expression import case as case
from .sql.expression import Cast as Cast
from .sql.expression import cast as cast
from .sql.expression import ClauseElement as ClauseElement
from .sql.expression import ClauseList as ClauseList
from .sql.expression import collate as collate
from .sql.expression import CollectionAggregate as CollectionAggregate
from .sql.expression import column as column
from .sql.expression import ColumnClause as ColumnClause
from .sql.expression import ColumnCollection as ColumnCollection
from .sql.expression import ColumnElement as ColumnElement
from .sql.expression import ColumnOperators as ColumnOperators
from .sql.expression import CompoundSelect as CompoundSelect
from .sql.expression import CTE as CTE
from .sql.expression import cte as cte
from .sql.expression import custom_op as custom_op
from .sql.expression import Delete as Delete
from .sql.expression import delete as delete
from .sql.expression import desc as desc
from .sql.expression import distinct as distinct
from .sql.expression import except_ as except_
from .sql.expression import except_all as except_all
from .sql.expression import Executable as Executable
from .sql.expression import Exists as Exists
from .sql.expression import exists as exists
from .sql.expression import Extract as Extract
from .sql.expression import extract as extract
from .sql.expression import false as false
from .sql.expression import False_ as False_
from .sql.expression import FromClause as FromClause
from .sql.expression import FromGrouping as FromGrouping
from .sql.expression import func as func
from .sql.expression import funcfilter as funcfilter
from .sql.expression import Function as Function
from .sql.expression import FunctionElement as FunctionElement
from .sql.expression import FunctionFilter as FunctionFilter
from .sql.expression import GenerativeSelect as GenerativeSelect
from .sql.expression import Grouping as Grouping
from .sql.expression import HasCTE as HasCTE
from .sql.expression import HasPrefixes as HasPrefixes
from .sql.expression import HasSuffixes as HasSuffixes
from .sql.expression import Insert as Insert
from .sql.expression import insert as insert
from .sql.expression import intersect as intersect
from .sql.expression import intersect_all as intersect_all
from .sql.expression import Join as Join
from .sql.expression import join as join
from .sql.expression import Label as Label
from .sql.expression import label as label
from .sql.expression import LABEL_STYLE_DEFAULT as LABEL_STYLE_DEFAULT
from .sql.expression import (
    LABEL_STYLE_DISAMBIGUATE_ONLY as LABEL_STYLE_DISAMBIGUATE_ONLY,
)
from .sql.expression import LABEL_STYLE_NONE as LABEL_STYLE_NONE
from .sql.expression import (
    LABEL_STYLE_TABLENAME_PLUS_COL as LABEL_STYLE_TABLENAME_PLUS_COL,
)
from .sql.expression import lambda_stmt as lambda_stmt
from .sql.expression import LambdaElement as LambdaElement
from .sql.expression import Lateral as Lateral
from .sql.expression import lateral as lateral
from .sql.expression import literal as literal
from .sql.expression import literal_column as literal_column
from .sql.expression import modifier as modifier
from .sql.expression import not_ as not_
from .sql.expression import Null as Null
from .sql.expression import null as null
from .sql.expression import nulls_first as nulls_first
from .sql.expression import nulls_last as nulls_last
from .sql.expression import nullsfirst as nullsfirst
from .sql.expression import nullslast as nullslast
from .sql.expression import Operators as Operators
from .sql.expression import or_ as or_
from .sql.expression import outerjoin as outerjoin
from .sql.expression import outparam as outparam
from .sql.expression import Over as Over
from .sql.expression import over as over
from .sql.expression import quoted_name as quoted_name
from .sql.expression import ReleaseSavepointClause as ReleaseSavepointClause
from .sql.expression import ReturnsRows as ReturnsRows
from .sql.expression import (
    RollbackToSavepointClause as RollbackToSavepointClause,
)
from .sql.expression import SavepointClause as SavepointClause
from .sql.expression import ScalarSelect as ScalarSelect
from .sql.expression import Select as Select
from .sql.expression import select as select
from .sql.expression import Selectable as Selectable
from .sql.expression import SelectBase as SelectBase
from .sql.expression import SQLColumnExpression as SQLColumnExpression
from .sql.expression import StatementLambdaElement as StatementLambdaElement
from .sql.expression import Subquery as Subquery
from .sql.expression import table as table
from .sql.expression import TableClause as TableClause
from .sql.expression import TableSample as TableSample
from .sql.expression import tablesample as tablesample
from .sql.expression import TableValuedAlias as TableValuedAlias
from .sql.expression import text as text
from .sql.expression import TextAsFrom as TextAsFrom
from .sql.expression import TextClause as TextClause
from .sql.expression import TextualSelect as TextualSelect
from .sql.expression import true as true
from .sql.expression import True_ as True_
from .sql.expression import try_cast as try_cast
from .sql.expression import TryCast as TryCast
from .sql.expression import Tuple as Tuple
from .sql.expression import tuple_ as tuple_
from .sql.expression import type_coerce as type_coerce
from .sql.expression import TypeClause as TypeClause
from .sql.expression import TypeCoerce as TypeCoerce
from .sql.expression import UnaryExpression as UnaryExpression
from .sql.expression import union as union
from .sql.expression import union_all as union_all
from .sql.expression import Update as Update
from .sql.expression import update as update
from .sql.expression import UpdateBase as UpdateBase
from .sql.expression import Values as Values
from .sql.expression import values as values
from .sql.expression import ValuesBase as ValuesBase
from .sql.expression import Visitable as Visitable
from .sql.expression import within_group as within_group
from .sql.expression import WithinGroup as WithinGroup
from .types import ARRAY as ARRAY
from .types import BIGINT as BIGINT
from .types import BigInteger as BigInteger
from .types import BINARY as BINARY
from .types import BLOB as BLOB
from .types import BOOLEAN as BOOLEAN
from .types import Boolean as Boolean
from .types import CHAR as CHAR
from .types import CLOB as CLOB
from .types import DATE as DATE
from .types import Date as Date
from .types import DATETIME as DATETIME
from .types import DateTime as DateTime
from .types import DECIMAL as DECIMAL
from .types import DOUBLE as DOUBLE
from .types import Double as Double
from .types import DOUBLE_PRECISION as DOUBLE_PRECISION
from .types import Enum as Enum
from .types import FLOAT as FLOAT
from .types import Float as Float
from .types import INT as INT
from .types import INTEGER as INTEGER
from .types import Integer as Integer
from .types import Interval as Interval
from .types import JSON as JSON
from .types import LargeBinary as LargeBinary
from .types import NCHAR as NCHAR
from .types import NUMERIC as NUMERIC
from .types import Numeric as Numeric
from .types import NVARCHAR as NVARCHAR
from .types import PickleType as PickleType
from .types import REAL as REAL
from .types import SMALLINT as SMALLINT
from .types import SmallInteger as SmallInteger
from .types import String as String
from .types import TEXT as TEXT
from .types import Text as Text
from .types import TIME as TIME
from .types import Time as Time
from .types import TIMESTAMP as TIMESTAMP
from .types import TupleType as TupleType
from .types import TypeDecorator as TypeDecorator
from .types import Unicode as Unicode
from .types import UnicodeText as UnicodeText
from .types import UUID as UUID
from .types import Uuid as Uuid
from .types import VARBINARY as VARBINARY
from .types import VARCHAR as VARCHAR

__version__ = "2.0.41"


def __go(lcls: Any) -> None:
    _util.preloaded.import_prefix("sqlalchemy")

    from . import exc

    exc._version_token = "".join(__version__.split(".")[0:2])


__go(locals())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\middleware\shared_data.py ===
"""
Serve Shared Static Files
=========================

.. autoclass:: SharedDataMiddleware
    :members: is_allowed

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import collections.abc as cabc
import importlib.util
import mimetypes
import os
import posixpath
import typing as t
from datetime import datetime
from datetime import timezone
from io import BytesIO
from time import time
from zlib import adler32

from ..http import http_date
from ..http import is_resource_modified
from ..security import safe_join
from ..utils import get_content_type
from ..wsgi import get_path_info
from ..wsgi import wrap_file

_TOpener = t.Callable[[], tuple[t.IO[bytes], datetime, int]]
_TLoader = t.Callable[[t.Optional[str]], tuple[t.Optional[str], t.Optional[_TOpener]]]

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class SharedDataMiddleware:
    """A WSGI middleware which provides static content for development
    environments or simple server setups. Its usage is quite simple::

        import os
        from werkzeug.middleware.shared_data import SharedDataMiddleware

        app = SharedDataMiddleware(app, {
            '/shared': os.path.join(os.path.dirname(__file__), 'shared')
        })

    The contents of the folder ``./shared`` will now be available on
    ``http://example.com/shared/``.  This is pretty useful during development
    because a standalone media server is not required. Files can also be
    mounted on the root folder and still continue to use the application because
    the shared data middleware forwards all unhandled requests to the
    application, even if the requests are below one of the shared folders.

    If `pkg_resources` is available you can also tell the middleware to serve
    files from package data::

        app = SharedDataMiddleware(app, {
            '/static': ('myapplication', 'static')
        })

    This will then serve the ``static`` folder in the `myapplication`
    Python package.

    The optional `disallow` parameter can be a list of :func:`~fnmatch.fnmatch`
    rules for files that are not accessible from the web.  If `cache` is set to
    `False` no caching headers are sent.

    Currently the middleware does not support non-ASCII filenames. If the
    encoding on the file system happens to match the encoding of the URI it may
    work but this could also be by accident. We strongly suggest using ASCII
    only file names for static files.

    The middleware will guess the mimetype using the Python `mimetype`
    module.  If it's unable to figure out the charset it will fall back
    to `fallback_mimetype`.

    :param app: the application to wrap.  If you don't want to wrap an
                application you can pass it :exc:`NotFound`.
    :param exports: a list or dict of exported files and folders.
    :param disallow: a list of :func:`~fnmatch.fnmatch` rules.
    :param cache: enable or disable caching headers.
    :param cache_timeout: the cache timeout in seconds for the headers.
    :param fallback_mimetype: The fallback mimetype for unknown files.

    .. versionchanged:: 1.0
        The default ``fallback_mimetype`` is
        ``application/octet-stream``. If a filename looks like a text
        mimetype, the ``utf-8`` charset is added to it.

    .. versionadded:: 0.6
        Added ``fallback_mimetype``.

    .. versionchanged:: 0.5
        Added ``cache_timeout``.
    """

    def __init__(
        self,
        app: WSGIApplication,
        exports: (
            cabc.Mapping[str, str | tuple[str, str]]
            | t.Iterable[tuple[str, str | tuple[str, str]]]
        ),
        disallow: None = None,
        cache: bool = True,
        cache_timeout: int = 60 * 60 * 12,
        fallback_mimetype: str = "application/octet-stream",
    ) -> None:
        self.app = app
        self.exports: list[tuple[str, _TLoader]] = []
        self.cache = cache
        self.cache_timeout = cache_timeout

        if isinstance(exports, cabc.Mapping):
            exports = exports.items()

        for key, value in exports:
            if isinstance(value, tuple):
                loader = self.get_package_loader(*value)
            elif isinstance(value, str):
                if os.path.isfile(value):
                    loader = self.get_file_loader(value)
                else:
                    loader = self.get_directory_loader(value)
            else:
                raise TypeError(f"unknown def {value!r}")

            self.exports.append((key, loader))

        if disallow is not None:
            from fnmatch import fnmatch

            self.is_allowed = lambda x: not fnmatch(x, disallow)

        self.fallback_mimetype = fallback_mimetype

    def is_allowed(self, filename: str) -> bool:
        """Subclasses can override this method to disallow the access to
        certain files.  However by providing `disallow` in the constructor
        this method is overwritten.
        """
        return True

    def _opener(self, filename: str) -> _TOpener:
        return lambda: (
            open(filename, "rb"),
            datetime.fromtimestamp(os.path.getmtime(filename), tz=timezone.utc),
            int(os.path.getsize(filename)),
        )

    def get_file_loader(self, filename: str) -> _TLoader:
        return lambda x: (os.path.basename(filename), self._opener(filename))

    def get_package_loader(self, package: str, package_path: str) -> _TLoader:
        load_time = datetime.now(timezone.utc)
        spec = importlib.util.find_spec(package)
        reader = spec.loader.get_resource_reader(package)  # type: ignore[union-attr]

        def loader(
            path: str | None,
        ) -> tuple[str | None, _TOpener | None]:
            if path is None:
                return None, None

            path = safe_join(package_path, path)

            if path is None:
                return None, None

            basename = posixpath.basename(path)

            try:
                resource = reader.open_resource(path)
            except OSError:
                return None, None

            if isinstance(resource, BytesIO):
                return (
                    basename,
                    lambda: (resource, load_time, len(resource.getvalue())),
                )

            return (
                basename,
                lambda: (
                    resource,
                    datetime.fromtimestamp(
                        os.path.getmtime(resource.name), tz=timezone.utc
                    ),
                    os.path.getsize(resource.name),
                ),
            )

        return loader

    def get_directory_loader(self, directory: str) -> _TLoader:
        def loader(
            path: str | None,
        ) -> tuple[str | None, _TOpener | None]:
            if path is not None:
                path = safe_join(directory, path)

                if path is None:
                    return None, None
            else:
                path = directory

            if os.path.isfile(path):
                return os.path.basename(path), self._opener(path)

            return None, None

        return loader

    def generate_etag(self, mtime: datetime, file_size: int, real_filename: str) -> str:
        fn_str = os.fsencode(real_filename)
        timestamp = mtime.timestamp()
        checksum = adler32(fn_str) & 0xFFFFFFFF
        return f"wzsdm-{timestamp}-{file_size}-{checksum}"

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        path = get_path_info(environ)
        file_loader = None

        for search_path, loader in self.exports:
            if search_path == path:
                real_filename, file_loader = loader(None)

                if file_loader is not None:
                    break

            if not search_path.endswith("/"):
                search_path += "/"

            if path.startswith(search_path):
                real_filename, file_loader = loader(path[len(search_path) :])

                if file_loader is not None:
                    break

        if file_loader is None or not self.is_allowed(real_filename):  # type: ignore
            return self.app(environ, start_response)

        guessed_type = mimetypes.guess_type(real_filename)  # type: ignore
        mime_type = get_content_type(guessed_type[0] or self.fallback_mimetype, "utf-8")
        f, mtime, file_size = file_loader()

        headers = [("Date", http_date())]

        if self.cache:
            timeout = self.cache_timeout
            etag = self.generate_etag(mtime, file_size, real_filename)  # type: ignore
            headers += [
                ("Etag", f'"{etag}"'),
                ("Cache-Control", f"max-age={timeout}, public"),
            ]

            if not is_resource_modified(environ, etag, last_modified=mtime):
                f.close()
                start_response("304 Not Modified", headers)
                return []

            headers.append(("Expires", http_date(time() + timeout)))
        else:
            headers.append(("Cache-Control", "public"))

        headers.extend(
            (
                ("Content-Type", mime_type),
                ("Content-Length", str(file_size)),
                ("Last-Modified", http_date(mtime)),
            )
        )
        start_response("200 OK", headers)
        return wrap_file(environ, f)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_headers.py ===
import re
from typing import AnyStr, cast, List, overload, Sequence, Tuple, TYPE_CHECKING, Union

from ._abnf import field_name, field_value
from ._util import bytesify, LocalProtocolError, validate

if TYPE_CHECKING:
    from ._events import Request

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

CONTENT_LENGTH_MAX_DIGITS = 20  # allow up to 1 billion TB - 1


# Facts
# -----
#
# Headers are:
#   keys: case-insensitive ascii
#   values: mixture of ascii and raw bytes
#
# "Historically, HTTP has allowed field content with text in the ISO-8859-1
# charset [ISO-8859-1], supporting other charsets only through use of
# [RFC2047] encoding.  In practice, most HTTP header field values use only a
# subset of the US-ASCII charset [USASCII]. Newly defined header fields SHOULD
# limit their field values to US-ASCII octets.  A recipient SHOULD treat other
# octets in field content (obs-text) as opaque data."
# And it deprecates all non-ascii values
#
# Leading/trailing whitespace in header names is forbidden
#
# Values get leading/trailing whitespace stripped
#
# Content-Disposition actually needs to contain unicode semantically; to
# accomplish this it has a terrifically weird way of encoding the filename
# itself as ascii (and even this still has lots of cross-browser
# incompatibilities)
#
# Order is important:
# "a proxy MUST NOT change the order of these field values when forwarding a
# message"
# (and there are several headers where the order indicates a preference)
#
# Multiple occurences of the same header:
# "A sender MUST NOT generate multiple header fields with the same field name
# in a message unless either the entire field value for that header field is
# defined as a comma-separated list [or the header is Set-Cookie which gets a
# special exception]" - RFC 7230. (cookies are in RFC 6265)
#
# So every header aside from Set-Cookie can be merged by b", ".join if it
# occurs repeatedly. But, of course, they can't necessarily be split by
# .split(b","), because quoting.
#
# Given all this mess (case insensitive, duplicates allowed, order is
# important, ...), there doesn't appear to be any standard way to handle
# headers in Python -- they're almost like dicts, but... actually just
# aren't. For now we punt and just use a super simple representation: headers
# are a list of pairs
#
#   [(name1, value1), (name2, value2), ...]
#
# where all entries are bytestrings, names are lowercase and have no
# leading/trailing whitespace, and values are bytestrings with no
# leading/trailing whitespace. Searching and updating are done via naive O(n)
# methods.
#
# Maybe a dict-of-lists would be better?

_content_length_re = re.compile(rb"[0-9]+")
_field_name_re = re.compile(field_name.encode("ascii"))
_field_value_re = re.compile(field_value.encode("ascii"))


class Headers(Sequence[Tuple[bytes, bytes]]):
    """
    A list-like interface that allows iterating over headers as byte-pairs
    of (lowercased-name, value).

    Internally we actually store the representation as three-tuples,
    including both the raw original casing, in order to preserve casing
    over-the-wire, and the lowercased name, for case-insensitive comparisions.

    r = Request(
        method="GET",
        target="/",
        headers=[("Host", "example.org"), ("Connection", "keep-alive")],
        http_version="1.1",
    )
    assert r.headers == [
        (b"host", b"example.org"),
        (b"connection", b"keep-alive")
    ]
    assert r.headers.raw_items() == [
        (b"Host", b"example.org"),
        (b"Connection", b"keep-alive")
    ]
    """

    __slots__ = "_full_items"

    def __init__(self, full_items: List[Tuple[bytes, bytes, bytes]]) -> None:
        self._full_items = full_items

    def __bool__(self) -> bool:
        return bool(self._full_items)

    def __eq__(self, other: object) -> bool:
        return list(self) == list(other)  # type: ignore

    def __len__(self) -> int:
        return len(self._full_items)

    def __repr__(self) -> str:
        return "<Headers(%s)>" % repr(list(self))

    def __getitem__(self, idx: int) -> Tuple[bytes, bytes]:  # type: ignore[override]
        _, name, value = self._full_items[idx]
        return (name, value)

    def raw_items(self) -> List[Tuple[bytes, bytes]]:
        return [(raw_name, value) for raw_name, _, value in self._full_items]


HeaderTypes = Union[
    List[Tuple[bytes, bytes]],
    List[Tuple[bytes, str]],
    List[Tuple[str, bytes]],
    List[Tuple[str, str]],
]


@overload
def normalize_and_validate(headers: Headers, _parsed: Literal[True]) -> Headers:
    ...


@overload
def normalize_and_validate(headers: HeaderTypes, _parsed: Literal[False]) -> Headers:
    ...


@overload
def normalize_and_validate(
    headers: Union[Headers, HeaderTypes], _parsed: bool = False
) -> Headers:
    ...


def normalize_and_validate(
    headers: Union[Headers, HeaderTypes], _parsed: bool = False
) -> Headers:
    new_headers = []
    seen_content_length = None
    saw_transfer_encoding = False
    for name, value in headers:
        # For headers coming out of the parser, we can safely skip some steps,
        # because it always returns bytes and has already run these regexes
        # over the data:
        if not _parsed:
            name = bytesify(name)
            value = bytesify(value)
            validate(_field_name_re, name, "Illegal header name {!r}", name)
            validate(_field_value_re, value, "Illegal header value {!r}", value)
        assert isinstance(name, bytes)
        assert isinstance(value, bytes)

        raw_name = name
        name = name.lower()
        if name == b"content-length":
            lengths = {length.strip() for length in value.split(b",")}
            if len(lengths) != 1:
                raise LocalProtocolError("conflicting Content-Length headers")
            value = lengths.pop()
            validate(_content_length_re, value, "bad Content-Length")
            if len(value) > CONTENT_LENGTH_MAX_DIGITS:
                raise LocalProtocolError("bad Content-Length")
            if seen_content_length is None:
                seen_content_length = value
                new_headers.append((raw_name, name, value))
            elif seen_content_length != value:
                raise LocalProtocolError("conflicting Content-Length headers")
        elif name == b"transfer-encoding":
            # "A server that receives a request message with a transfer coding
            # it does not understand SHOULD respond with 501 (Not
            # Implemented)."
            # https://tools.ietf.org/html/rfc7230#section-3.3.1
            if saw_transfer_encoding:
                raise LocalProtocolError(
                    "multiple Transfer-Encoding headers", error_status_hint=501
                )
            # "All transfer-coding names are case-insensitive"
            # -- https://tools.ietf.org/html/rfc7230#section-4
            value = value.lower()
            if value != b"chunked":
                raise LocalProtocolError(
                    "Only Transfer-Encoding: chunked is supported",
                    error_status_hint=501,
                )
            saw_transfer_encoding = True
            new_headers.append((raw_name, name, value))
        else:
            new_headers.append((raw_name, name, value))
    return Headers(new_headers)


def get_comma_header(headers: Headers, name: bytes) -> List[bytes]:
    # Should only be used for headers whose value is a list of
    # comma-separated, case-insensitive values.
    #
    # The header name `name` is expected to be lower-case bytes.
    #
    # Connection: meets these criteria (including cast insensitivity).
    #
    # Content-Length: technically is just a single value (1*DIGIT), but the
    # standard makes reference to implementations that do multiple values, and
    # using this doesn't hurt. Ditto, case insensitivity doesn't things either
    # way.
    #
    # Transfer-Encoding: is more complex (allows for quoted strings), so
    # splitting on , is actually wrong. For example, this is legal:
    #
    #    Transfer-Encoding: foo; options="1,2", chunked
    #
    # and should be parsed as
    #
    #    foo; options="1,2"
    #    chunked
    #
    # but this naive function will parse it as
    #
    #    foo; options="1
    #    2"
    #    chunked
    #
    # However, this is okay because the only thing we are going to do with
    # any Transfer-Encoding is reject ones that aren't just "chunked", so
    # both of these will be treated the same anyway.
    #
    # Expect: the only legal value is the literal string
    # "100-continue". Splitting on commas is harmless. Case insensitive.
    #
    out: List[bytes] = []
    for _, found_name, found_raw_value in headers._full_items:
        if found_name == name:
            found_raw_value = found_raw_value.lower()
            for found_split_value in found_raw_value.split(b","):
                found_split_value = found_split_value.strip()
                if found_split_value:
                    out.append(found_split_value)
    return out


def set_comma_header(headers: Headers, name: bytes, new_values: List[bytes]) -> Headers:
    # The header name `name` is expected to be lower-case bytes.
    #
    # Note that when we store the header we use title casing for the header
    # names, in order to match the conventional HTTP header style.
    #
    # Simply calling `.title()` is a blunt approach, but it's correct
    # here given the cases where we're using `set_comma_header`...
    #
    # Connection, Content-Length, Transfer-Encoding.
    new_headers: List[Tuple[bytes, bytes]] = []
    for found_raw_name, found_name, found_raw_value in headers._full_items:
        if found_name != name:
            new_headers.append((found_raw_name, found_raw_value))
    for new_value in new_values:
        new_headers.append((name.title(), new_value))
    return normalize_and_validate(new_headers)


def has_expect_100_continue(request: "Request") -> bool:
    # https://tools.ietf.org/html/rfc7231#section-5.1.1
    # "A server that receives a 100-continue expectation in an HTTP/1.0 request
    # MUST ignore that expectation."
    if request.http_version < b"1.1":
        return False
    expect = get_comma_header(request.headers, b"expect")
    return b"100-continue" in expect

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\named_styles.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.compat import safe_string

from openpyxl.descriptors import (
    Typed,
    Integer,
    Bool,
    String,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.serialisable import Serialisable

from .fills import PatternFill, Fill
from .fonts import Font
from .borders import Border
from .alignment import Alignment
from .protection import Protection
from .numbers import (
    NumberFormatDescriptor,
    BUILTIN_FORMATS_MAX_SIZE,
    BUILTIN_FORMATS_REVERSE,
)
from .cell_style import (
    StyleArray,
    CellStyle,
)


class NamedStyle(Serialisable):

    """
    Named and editable styles
    """

    font = Typed(expected_type=Font)
    fill = Typed(expected_type=Fill)
    border = Typed(expected_type=Border)
    alignment = Typed(expected_type=Alignment)
    number_format = NumberFormatDescriptor()
    protection = Typed(expected_type=Protection)
    builtinId = Integer(allow_none=True)
    hidden = Bool(allow_none=True)
    name = String()
    _wb = None
    _style = StyleArray()


    def __init__(self,
                 name="Normal",
                 font=None,
                 fill=None,
                 border=None,
                 alignment=None,
                 number_format=None,
                 protection=None,
                 builtinId=None,
                 hidden=False,
                 ):
        self.name = name
        self.font = font or Font()
        self.fill = fill or PatternFill()
        self.border = border or Border()
        self.alignment = alignment or Alignment()
        self.number_format = number_format
        self.protection = protection or Protection()
        self.builtinId = builtinId
        self.hidden = hidden
        self._wb = None
        self._style = StyleArray()


    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if getattr(self, '_wb', None) and attr in (
           'font', 'fill', 'border', 'alignment', 'number_format', 'protection',
            ):
            self._recalculate()


    def __iter__(self):
        for key in ('name', 'builtinId', 'hidden', 'xfId'):
            value = getattr(self, key, None)
            if value is not None:
                yield key, safe_string(value)


    def bind(self, wb):
        """
        Bind a named style to a workbook
        """
        self._wb = wb
        self._recalculate()


    def _recalculate(self):
        self._style.fontId =  self._wb._fonts.add(self.font)
        self._style.borderId = self._wb._borders.add(self.border)
        self._style.fillId =  self._wb._fills.add(self.fill)
        self._style.protectionId = self._wb._protections.add(self.protection)
        self._style.alignmentId = self._wb._alignments.add(self.alignment)
        fmt = self.number_format
        if fmt in BUILTIN_FORMATS_REVERSE:
            fmt = BUILTIN_FORMATS_REVERSE[fmt]
        else:
            fmt = self._wb._number_formats.add(self.number_format) + (
                  BUILTIN_FORMATS_MAX_SIZE)
        self._style.numFmtId = fmt


    def as_tuple(self):
        """Return a style array representing the current style"""
        return self._style


    def as_xf(self):
        """
        Return equivalent XfStyle
        """
        xf = CellStyle.from_array(self._style)
        xf.xfId = None
        xf.pivotButton = None
        xf.quotePrefix = None
        if self.alignment != Alignment():
            xf.alignment = self.alignment
        if self.protection != Protection():
            xf.protection = self.protection
        return xf


    def as_name(self):
        """
        Return relevant named style

        """
        named = _NamedCellStyle(
            name=self.name,
            builtinId=self.builtinId,
            hidden=self.hidden,
            xfId=self._style.xfId
        )
        return named


class NamedStyleList(list):
    """
    Named styles are editable and can be applied to multiple objects

    As only the index is stored in referencing objects the order mus
    be preserved.

    Returns a list of NamedStyles
    """

    def __init__(self, iterable=()):
        """
        Allow a list of named styles to be passed in and index them.
        """

        for idx, s in enumerate(iterable, len(self)):
            s._style.xfId = idx
        super().__init__(iterable)


    @property
    def names(self):
        return [s.name for s in self]


    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)


        for idx, name in enumerate(self.names):
            if name == key:
                return self[idx]

        raise KeyError("No named style with the name{0} exists".format(key))

    def append(self, style):
        if not isinstance(style, NamedStyle):
            raise TypeError("""Only NamedStyle instances can be added""")
        elif style.name in self.names: # hotspot
            raise ValueError("""Style {0} exists already""".format(style.name))
        style._style.xfId = (len(self))
        super().append(style)


class _NamedCellStyle(Serialisable):

    """
    Pointer-based representation of named styles in XML
    xfId refers to the corresponding CellStyleXfs

    Not used in client code.
    """

    tagname = "cellStyle"

    name = String()
    xfId = Integer()
    builtinId = Integer(allow_none=True)
    iLevel = Integer(allow_none=True)
    hidden = Bool(allow_none=True)
    customBuiltin = Bool(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()


    def __init__(self,
                 name=None,
                 xfId=None,
                 builtinId=None,
                 iLevel=None,
                 hidden=None,
                 customBuiltin=None,
                 extLst=None,
                ):
        self.name = name
        self.xfId = xfId
        self.builtinId = builtinId
        self.iLevel = iLevel
        self.hidden = hidden
        self.customBuiltin = customBuiltin


class _NamedCellStyleList(Serialisable):
    """
    Container for named cell style objects

    Not used in client code
    """

    tagname = "cellStyles"

    count = Integer(allow_none=True)
    cellStyle = Sequence(expected_type=_NamedCellStyle)

    __attrs__ = ("count",)

    def __init__(self,
                 count=None,
                 cellStyle=(),
                ):
        self.cellStyle = cellStyle


    @property
    def count(self):
        return len(self.cellStyle)


    def remove_duplicates(self):
        """
        Some applications contain duplicate definitions either by name or
        referenced style.

        As the references are 0-based indices, styles are sorted by
        index.

        Returns a list of style references with duplicates removed
        """

        def sort_fn(v):
            return v.xfId

        styles = []
        names = set()
        ids = set()

        for ns in sorted(self.cellStyle, key=sort_fn):
            if ns.xfId in ids or ns.name in names: # skip duplicates
                continue
            ids.add(ns.xfId)
            names.add(ns.name)

            styles.append(ns)

        return styles

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\ordinals.py ===
from sympy.core import Basic, Integer
import operator


class OmegaPower(Basic):
    """
    Represents ordinal exponential and multiplication terms one of the
    building blocks of the :class:`Ordinal` class.
    In ``OmegaPower(a, b)``, ``a`` represents exponent and ``b`` represents multiplicity.
    """
    def __new__(cls, a, b):
        if isinstance(b, int):
            b = Integer(b)
        if not isinstance(b, Integer) or b <= 0:
            raise TypeError("multiplicity must be a positive integer")

        if not isinstance(a, Ordinal):
            a = Ordinal.convert(a)

        return Basic.__new__(cls, a, b)

    @property
    def exp(self):
        return self.args[0]

    @property
    def mult(self):
        return self.args[1]

    def _compare_term(self, other, op):
        if self.exp == other.exp:
            return op(self.mult, other.mult)
        else:
            return op(self.exp, other.exp)

    def __eq__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self.args == other.args

    def __hash__(self):
        return Basic.__hash__(self)

    def __lt__(self, other):
        if not isinstance(other, OmegaPower):
            try:
                other = OmegaPower(0, other)
            except TypeError:
                return NotImplemented
        return self._compare_term(other, operator.lt)


class Ordinal(Basic):
    """
    Represents ordinals in Cantor normal form.

    Internally, this class is just a list of instances of OmegaPower.

    Examples
    ========
    >>> from sympy import Ordinal, OmegaPower
    >>> from sympy.sets.ordinals import omega
    >>> w = omega
    >>> w.is_limit_ordinal
    True
    >>> Ordinal(OmegaPower(w + 1, 1), OmegaPower(3, 2))
    w**(w + 1) + w**3*2
    >>> 3 + w
    w
    >>> (w + 1) * w
    w**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ordinal_arithmetic
    """
    def __new__(cls, *terms):
        obj = super().__new__(cls, *terms)
        powers = [i.exp for i in obj.args]
        if not all(powers[i] >= powers[i+1] for i in range(len(powers) - 1)):
            raise ValueError("powers must be in decreasing order")
        return obj

    @property
    def terms(self):
        return self.args

    @property
    def leading_term(self):
        if self == ord0:
            raise ValueError("ordinal zero has no leading term")
        return self.terms[0]

    @property
    def trailing_term(self):
        if self == ord0:
            raise ValueError("ordinal zero has no trailing term")
        return self.terms[-1]

    @property
    def is_successor_ordinal(self):
        try:
            return self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    def is_limit_ordinal(self):
        try:
            return not self.trailing_term.exp == ord0
        except ValueError:
            return False

    @property
    def degree(self):
        return self.leading_term.exp

    @classmethod
    def convert(cls, integer_value):
        if integer_value == 0:
            return ord0
        return Ordinal(OmegaPower(0, integer_value))

    def __eq__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return self.terms == other.terms

    def __hash__(self):
        return hash(self.args)

    def __lt__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        for term_self, term_other in zip(self.terms, other.terms):
            if term_self != term_other:
                return term_self < term_other
        return len(self.terms) < len(other.terms)

    def __le__(self, other):
        return (self == other or self < other)

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def __str__(self):
        net_str = ""
        plus_count = 0
        if self == ord0:
            return 'ord0'
        for i in self.terms:
            if plus_count:
                net_str += " + "

            if i.exp == ord0:
                net_str += str(i.mult)
            elif i.exp == 1:
                net_str += 'w'
            elif len(i.exp.terms) > 1 or i.exp.is_limit_ordinal:
                net_str += 'w**(%s)'%i.exp
            else:
                net_str += 'w**%s'%i.exp

            if not i.mult == 1 and not i.exp == ord0:
                net_str += '*%s'%i.mult

            plus_count += 1
        return(net_str)

    __repr__ = __str__

    def __add__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        if other == ord0:
            return self
        a_terms = list(self.terms)
        b_terms = list(other.terms)
        r = len(a_terms) - 1
        b_exp = other.degree
        while r >= 0 and a_terms[r].exp < b_exp:
            r -= 1
        if r < 0:
            terms = b_terms
        elif a_terms[r].exp == b_exp:
            sum_term = OmegaPower(b_exp, a_terms[r].mult + other.leading_term.mult)
            terms = a_terms[:r] + [sum_term] + b_terms[1:]
        else:
            terms = a_terms[:r+1] + b_terms
        return Ordinal(*terms)

    def __radd__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return other + self

    def __mul__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        if ord0 in (self, other):
            return ord0
        a_exp = self.degree
        a_mult = self.leading_term.mult
        summation = []
        if other.is_limit_ordinal:
            for arg in other.terms:
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))

        else:
            for arg in other.terms[:-1]:
                summation.append(OmegaPower(a_exp + arg.exp, arg.mult))
            b_mult = other.trailing_term.mult
            summation.append(OmegaPower(a_exp, a_mult*b_mult))
            summation += list(self.terms[1:])
        return Ordinal(*summation)

    def __rmul__(self, other):
        if not isinstance(other, Ordinal):
            try:
                other = Ordinal.convert(other)
            except TypeError:
                return NotImplemented
        return other * self

    def __pow__(self, other):
        if not self == omega:
            return NotImplemented
        return Ordinal(OmegaPower(other, 1))


class OrdinalZero(Ordinal):
    """The ordinal zero.

    OrdinalZero can be imported as ``ord0``.
    """
    pass


class OrdinalOmega(Ordinal):
    """The ordinal omega which forms the base of all ordinals in cantor normal form.

    OrdinalOmega can be imported as ``omega``.

    Examples
    ========

    >>> from sympy.sets.ordinals import omega
    >>> omega + omega
    w*2
    """
    def __new__(cls):
        return Ordinal.__new__(cls)

    @property
    def terms(self):
        return (OmegaPower(1, 1),)


ord0 = OrdinalZero()
omega = OrdinalOmega()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\debug\repr.py ===
"""Object representations for debugging purposes. Unlike the default
repr, these expose more information and produce HTML instead of ASCII.

Together with the CSS and JavaScript of the debugger this gives a
colorful and more compact output.
"""

from __future__ import annotations

import codecs
import re
import sys
import typing as t
from collections import deque
from traceback import format_exception_only

from markupsafe import escape

missing = object()
_paragraph_re = re.compile(r"(?:\r\n|\r|\n){2,}")
RegexType = type(_paragraph_re)

HELP_HTML = """\
<div class=box>
  <h3>%(title)s</h3>
  <pre class=help>%(text)s</pre>
</div>\
"""
OBJECT_DUMP_HTML = """\
<div class=box>
  <h3>%(title)s</h3>
  %(repr)s
  <table>%(items)s</table>
</div>\
"""


def debug_repr(obj: object) -> str:
    """Creates a debug repr of an object as HTML string."""
    return DebugReprGenerator().repr(obj)


def dump(obj: object = missing) -> None:
    """Print the object details to stdout._write (for the interactive
    console of the web debugger.
    """
    gen = DebugReprGenerator()
    if obj is missing:
        rv = gen.dump_locals(sys._getframe(1).f_locals)
    else:
        rv = gen.dump_object(obj)
    sys.stdout._write(rv)  # type: ignore


class _Helper:
    """Displays an HTML version of the normal help, for the interactive
    debugger only because it requires a patched sys.stdout.
    """

    def __repr__(self) -> str:
        return "Type help(object) for help about object."

    def __call__(self, topic: t.Any | None = None) -> None:
        if topic is None:
            sys.stdout._write(f"<span class=help>{self!r}</span>")  # type: ignore
            return
        import pydoc

        pydoc.help(topic)
        rv = sys.stdout.reset()  # type: ignore
        paragraphs = _paragraph_re.split(rv)
        if len(paragraphs) > 1:
            title = paragraphs[0]
            text = "\n\n".join(paragraphs[1:])
        else:
            title = "Help"
            text = paragraphs[0]
        sys.stdout._write(HELP_HTML % {"title": title, "text": text})  # type: ignore


helper = _Helper()


def _add_subclass_info(inner: str, obj: object, base: type | tuple[type, ...]) -> str:
    if isinstance(base, tuple):
        for cls in base:
            if type(obj) is cls:
                return inner
    elif type(obj) is base:
        return inner
    module = ""
    if obj.__class__.__module__ not in ("__builtin__", "exceptions"):
        module = f'<span class="module">{obj.__class__.__module__}.</span>'
    return f"{module}{type(obj).__name__}({inner})"


def _sequence_repr_maker(
    left: str, right: str, base: type, limit: int = 8
) -> t.Callable[[DebugReprGenerator, t.Iterable[t.Any], bool], str]:
    def proxy(self: DebugReprGenerator, obj: t.Iterable[t.Any], recursive: bool) -> str:
        if recursive:
            return _add_subclass_info(f"{left}...{right}", obj, base)
        buf = [left]
        have_extended_section = False
        for idx, item in enumerate(obj):
            if idx:
                buf.append(", ")
            if idx == limit:
                buf.append('<span class="extended">')
                have_extended_section = True
            buf.append(self.repr(item))
        if have_extended_section:
            buf.append("</span>")
        buf.append(right)
        return _add_subclass_info("".join(buf), obj, base)

    return proxy


class DebugReprGenerator:
    def __init__(self) -> None:
        self._stack: list[t.Any] = []

    list_repr = _sequence_repr_maker("[", "]", list)
    tuple_repr = _sequence_repr_maker("(", ")", tuple)
    set_repr = _sequence_repr_maker("set([", "])", set)
    frozenset_repr = _sequence_repr_maker("frozenset([", "])", frozenset)
    deque_repr = _sequence_repr_maker(
        '<span class="module">collections.</span>deque([', "])", deque
    )

    def regex_repr(self, obj: t.Pattern[t.AnyStr]) -> str:
        pattern = repr(obj.pattern)
        pattern = codecs.decode(pattern, "unicode-escape", "ignore")
        pattern = f"r{pattern}"
        return f're.compile(<span class="string regex">{pattern}</span>)'

    def string_repr(self, obj: str | bytes, limit: int = 70) -> str:
        buf = ['<span class="string">']
        r = repr(obj)

        # shorten the repr when the hidden part would be at least 3 chars
        if len(r) - limit > 2:
            buf.extend(
                (
                    escape(r[:limit]),
                    '<span class="extended">',
                    escape(r[limit:]),
                    "</span>",
                )
            )
        else:
            buf.append(escape(r))

        buf.append("</span>")
        out = "".join(buf)

        # if the repr looks like a standard string, add subclass info if needed
        if r[0] in "'\"" or (r[0] == "b" and r[1] in "'\""):
            return _add_subclass_info(out, obj, (bytes, str))

        # otherwise, assume the repr distinguishes the subclass already
        return out

    def dict_repr(
        self,
        d: dict[int, None] | dict[str, int] | dict[str | int, int],
        recursive: bool,
        limit: int = 5,
    ) -> str:
        if recursive:
            return _add_subclass_info("{...}", d, dict)
        buf = ["{"]
        have_extended_section = False
        for idx, (key, value) in enumerate(d.items()):
            if idx:
                buf.append(", ")
            if idx == limit - 1:
                buf.append('<span class="extended">')
                have_extended_section = True
            buf.append(
                f'<span class="pair"><span class="key">{self.repr(key)}</span>:'
                f' <span class="value">{self.repr(value)}</span></span>'
            )
        if have_extended_section:
            buf.append("</span>")
        buf.append("}")
        return _add_subclass_info("".join(buf), d, dict)

    def object_repr(self, obj: t.Any) -> str:
        r = repr(obj)
        return f'<span class="object">{escape(r)}</span>'

    def dispatch_repr(self, obj: t.Any, recursive: bool) -> str:
        if obj is helper:
            return f'<span class="help">{helper!r}</span>'
        if isinstance(obj, (int, float, complex)):
            return f'<span class="number">{obj!r}</span>'
        if isinstance(obj, str) or isinstance(obj, bytes):
            return self.string_repr(obj)
        if isinstance(obj, RegexType):
            return self.regex_repr(obj)
        if isinstance(obj, list):
            return self.list_repr(obj, recursive)
        if isinstance(obj, tuple):
            return self.tuple_repr(obj, recursive)
        if isinstance(obj, set):
            return self.set_repr(obj, recursive)
        if isinstance(obj, frozenset):
            return self.frozenset_repr(obj, recursive)
        if isinstance(obj, dict):
            return self.dict_repr(obj, recursive)
        if isinstance(obj, deque):
            return self.deque_repr(obj, recursive)
        return self.object_repr(obj)

    def fallback_repr(self) -> str:
        try:
            info = "".join(format_exception_only(*sys.exc_info()[:2]))
        except Exception:
            info = "?"
        return (
            '<span class="brokenrepr">'
            f"&lt;broken repr ({escape(info.strip())})&gt;</span>"
        )

    def repr(self, obj: object) -> str:
        recursive = False
        for item in self._stack:
            if item is obj:
                recursive = True
                break
        self._stack.append(obj)
        try:
            try:
                return self.dispatch_repr(obj, recursive)
            except Exception:
                return self.fallback_repr()
        finally:
            self._stack.pop()

    def dump_object(self, obj: object) -> str:
        repr = None
        items: list[tuple[str, str]] | None = None

        if isinstance(obj, dict):
            title = "Contents of"
            items = []
            for key, value in obj.items():
                if not isinstance(key, str):
                    items = None
                    break
                items.append((key, self.repr(value)))
        if items is None:
            items = []
            repr = self.repr(obj)
            for key in dir(obj):
                try:
                    items.append((key, self.repr(getattr(obj, key))))
                except Exception:
                    pass
            title = "Details for"
        title += f" {object.__repr__(obj)[1:-1]}"
        return self.render_object_dump(items, title, repr)

    def dump_locals(self, d: dict[str, t.Any]) -> str:
        items = [(key, self.repr(value)) for key, value in d.items()]
        return self.render_object_dump(items, "Local variables in frame")

    def render_object_dump(
        self, items: list[tuple[str, str]], title: str, repr: str | None = None
    ) -> str:
        html_items = []
        for key, value in items:
            html_items.append(f"<tr><th>{escape(key)}<td><pre class=repr>{value}</pre>")
        if not html_items:
            html_items.append("<tr><td><em>Nothing</em>")
        return OBJECT_DUMP_HTML % {
            "title": escape(title),
            "repr": f"<pre class=repr>{repr if repr else ''}</pre>",
            "items": "\n".join(html_items),
        }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\resolution\resolvelib\provider.py ===
import math
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from pip._vendor.resolvelib.providers import AbstractProvider

from pip._internal.req.req_install import InstallRequirement

from .base import Candidate, Constraint, Requirement
from .candidates import REQUIRES_PYTHON_IDENTIFIER
from .factory import Factory
from .requirements import ExplicitRequirement

if TYPE_CHECKING:
    from pip._vendor.resolvelib.providers import Preference
    from pip._vendor.resolvelib.resolvers import RequirementInformation

    PreferenceInformation = RequirementInformation[Requirement, Candidate]

    _ProviderBase = AbstractProvider[Requirement, Candidate, str]
else:
    _ProviderBase = AbstractProvider

# Notes on the relationship between the provider, the factory, and the
# candidate and requirement classes.
#
# The provider is a direct implementation of the resolvelib class. Its role
# is to deliver the API that resolvelib expects.
#
# Rather than work with completely abstract "requirement" and "candidate"
# concepts as resolvelib does, pip has concrete classes implementing these two
# ideas. The API of Requirement and Candidate objects are defined in the base
# classes, but essentially map fairly directly to the equivalent provider
# methods. In particular, `find_matches` and `is_satisfied_by` are
# requirement methods, and `get_dependencies` is a candidate method.
#
# The factory is the interface to pip's internal mechanisms. It is stateless,
# and is created by the resolver and held as a property of the provider. It is
# responsible for creating Requirement and Candidate objects, and provides
# services to those objects (access to pip's finder and preparer).


D = TypeVar("D")
V = TypeVar("V")


def _get_with_identifier(
    mapping: Mapping[str, V],
    identifier: str,
    default: D,
) -> Union[D, V]:
    """Get item from a package name lookup mapping with a resolver identifier.

    This extra logic is needed when the target mapping is keyed by package
    name, which cannot be directly looked up with an identifier (which may
    contain requested extras). Additional logic is added to also look up a value
    by "cleaning up" the extras from the identifier.
    """
    if identifier in mapping:
        return mapping[identifier]
    # HACK: Theoretically we should check whether this identifier is a valid
    # "NAME[EXTRAS]" format, and parse out the name part with packaging or
    # some regular expression. But since pip's resolver only spits out three
    # kinds of identifiers: normalized PEP 503 names, normalized names plus
    # extras, and Requires-Python, we can cheat a bit here.
    name, open_bracket, _ = identifier.partition("[")
    if open_bracket and name in mapping:
        return mapping[name]
    return default


class PipProvider(_ProviderBase):
    """Pip's provider implementation for resolvelib.

    :params constraints: A mapping of constraints specified by the user. Keys
        are canonicalized project names.
    :params ignore_dependencies: Whether the user specified ``--no-deps``.
    :params upgrade_strategy: The user-specified upgrade strategy.
    :params user_requested: A set of canonicalized package names that the user
        supplied for pip to install/upgrade.
    """

    def __init__(
        self,
        factory: Factory,
        constraints: Dict[str, Constraint],
        ignore_dependencies: bool,
        upgrade_strategy: str,
        user_requested: Dict[str, int],
    ) -> None:
        self._factory = factory
        self._constraints = constraints
        self._ignore_dependencies = ignore_dependencies
        self._upgrade_strategy = upgrade_strategy
        self._user_requested = user_requested

    def identify(self, requirement_or_candidate: Union[Requirement, Candidate]) -> str:
        return requirement_or_candidate.name

    def narrow_requirement_selection(
        self,
        identifiers: Iterable[str],
        resolutions: Mapping[str, Candidate],
        candidates: Mapping[str, Iterator[Candidate]],
        information: Mapping[str, Iterator["PreferenceInformation"]],
        backtrack_causes: Sequence["PreferenceInformation"],
    ) -> Iterable[str]:
        """Produce a subset of identifiers that should be considered before others.

        Currently pip narrows the following selection:
            * Requires-Python, if present is always returned by itself
            * Backtrack causes are considered next because they can be identified
              in linear time here, whereas because get_preference() is called
              for each identifier, it would be quadratic to check for them there.
              Further, the current backtrack causes likely need to be resolved
              before other requirements as a resolution can't be found while
              there is a conflict.
        """
        backtrack_identifiers = set()
        for info in backtrack_causes:
            backtrack_identifiers.add(info.requirement.name)
            if info.parent is not None:
                backtrack_identifiers.add(info.parent.name)

        current_backtrack_causes = []
        for identifier in identifiers:
            # Requires-Python has only one candidate and the check is basically
            # free, so we always do it first to avoid needless work if it fails.
            # This skips calling get_preference() for all other identifiers.
            if identifier == REQUIRES_PYTHON_IDENTIFIER:
                return [identifier]

            # Check if this identifier is a backtrack cause
            if identifier in backtrack_identifiers:
                current_backtrack_causes.append(identifier)
                continue

        if current_backtrack_causes:
            return current_backtrack_causes

        return identifiers

    def get_preference(
        self,
        identifier: str,
        resolutions: Mapping[str, Candidate],
        candidates: Mapping[str, Iterator[Candidate]],
        information: Mapping[str, Iterable["PreferenceInformation"]],
        backtrack_causes: Sequence["PreferenceInformation"],
    ) -> "Preference":
        """Produce a sort key for given requirement based on preference.

        The lower the return value is, the more preferred this group of
        arguments is.

        Currently pip considers the following in order:

        * Any requirement that is "direct", e.g., points to an explicit URL.
        * Any requirement that is "pinned", i.e., contains the operator ``===``
          or ``==`` without a wildcard.
        * Any requirement that imposes an upper version limit, i.e., contains the
          operator ``<``, ``<=``, ``~=``, or ``==`` with a wildcard. Because
          pip prioritizes the latest version, preferring explicit upper bounds
          can rule out infeasible candidates sooner. This does not imply that
          upper bounds are good practice; they can make dependency management
          and resolution harder.
        * Order user-specified requirements as they are specified, placing
          other requirements afterward.
        * Any "non-free" requirement, i.e., one that contains at least one
          operator, such as ``>=`` or ``!=``.
        * Alphabetical order for consistency (aids debuggability).
        """
        try:
            next(iter(information[identifier]))
        except StopIteration:
            # There is no information for this identifier, so there's no known
            # candidates.
            has_information = False
        else:
            has_information = True

        if not has_information:
            direct = False
            ireqs: Tuple[Optional[InstallRequirement], ...] = ()
        else:
            # Go through the information and for each requirement,
            # check if it's explicit (e.g., a direct link) and get the
            # InstallRequirement (the second element) from get_candidate_lookup()
            directs, ireqs = zip(
                *(
                    (isinstance(r, ExplicitRequirement), r.get_candidate_lookup()[1])
                    for r, _ in information[identifier]
                )
            )
            direct = any(directs)

        operators: list[tuple[str, str]] = [
            (specifier.operator, specifier.version)
            for specifier_set in (ireq.specifier for ireq in ireqs if ireq)
            for specifier in specifier_set
        ]

        pinned = any(((op[:2] == "==") and ("*" not in ver)) for op, ver in operators)
        upper_bounded = any(
            ((op in ("<", "<=", "~=")) or (op == "==" and "*" in ver))
            for op, ver in operators
        )
        unfree = bool(operators)
        requested_order = self._user_requested.get(identifier, math.inf)

        return (
            not direct,
            not pinned,
            not upper_bounded,
            requested_order,
            not unfree,
            identifier,
        )

    def find_matches(
        self,
        identifier: str,
        requirements: Mapping[str, Iterator[Requirement]],
        incompatibilities: Mapping[str, Iterator[Candidate]],
    ) -> Iterable[Candidate]:
        def _eligible_for_upgrade(identifier: str) -> bool:
            """Are upgrades allowed for this project?

            This checks the upgrade strategy, and whether the project was one
            that the user specified in the command line, in order to decide
            whether we should upgrade if there's a newer version available.

            (Note that we don't need access to the `--upgrade` flag, because
            an upgrade strategy of "to-satisfy-only" means that `--upgrade`
            was not specified).
            """
            if self._upgrade_strategy == "eager":
                return True
            elif self._upgrade_strategy == "only-if-needed":
                user_order = _get_with_identifier(
                    self._user_requested,
                    identifier,
                    default=None,
                )
                return user_order is not None
            return False

        constraint = _get_with_identifier(
            self._constraints,
            identifier,
            default=Constraint.empty(),
        )
        return self._factory.find_candidates(
            identifier=identifier,
            requirements=requirements,
            constraint=constraint,
            prefers_installed=(not _eligible_for_upgrade(identifier)),
            incompatibilities=incompatibilities,
            is_satisfied_by=self.is_satisfied_by,
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def is_satisfied_by(requirement: Requirement, candidate: Candidate) -> bool:
        return requirement.is_satisfied_by(candidate)

    def get_dependencies(self, candidate: Candidate) -> Iterable[Requirement]:
        with_requires = not self._ignore_dependencies
        # iter_dependencies() can perform nontrivial work so delay until needed.
        return (r for r in candidate.iter_dependencies(with_requires) if r is not None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\fed_cm.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: FedCm (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class LoginState(enum.Enum):
    '''
    Whether this is a sign-up or sign-in action for this account, i.e.
    whether this account has ever been used to sign in to this RP before.
    '''
    SIGN_IN = "SignIn"
    SIGN_UP = "SignUp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DialogType(enum.Enum):
    '''
    The types of FedCM dialogs.
    '''
    ACCOUNT_CHOOSER = "AccountChooser"
    AUTO_REAUTHN = "AutoReauthn"
    CONFIRM_IDP_LOGIN = "ConfirmIdpLogin"
    ERROR = "Error"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DialogButton(enum.Enum):
    '''
    The buttons on the FedCM dialog.
    '''
    CONFIRM_IDP_LOGIN_CONTINUE = "ConfirmIdpLoginContinue"
    ERROR_GOT_IT = "ErrorGotIt"
    ERROR_MORE_DETAILS = "ErrorMoreDetails"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AccountUrlType(enum.Enum):
    '''
    The URLs that each account has
    '''
    TERMS_OF_SERVICE = "TermsOfService"
    PRIVACY_POLICY = "PrivacyPolicy"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class Account:
    '''
    Corresponds to IdentityRequestAccount
    '''
    account_id: str

    email: str

    name: str

    given_name: str

    picture_url: str

    idp_config_url: str

    idp_login_url: str

    login_state: LoginState

    #: These two are only set if the loginState is signUp
    terms_of_service_url: typing.Optional[str] = None

    privacy_policy_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['accountId'] = self.account_id
        json['email'] = self.email
        json['name'] = self.name
        json['givenName'] = self.given_name
        json['pictureUrl'] = self.picture_url
        json['idpConfigUrl'] = self.idp_config_url
        json['idpLoginUrl'] = self.idp_login_url
        json['loginState'] = self.login_state.to_json()
        if self.terms_of_service_url is not None:
            json['termsOfServiceUrl'] = self.terms_of_service_url
        if self.privacy_policy_url is not None:
            json['privacyPolicyUrl'] = self.privacy_policy_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            account_id=str(json['accountId']),
            email=str(json['email']),
            name=str(json['name']),
            given_name=str(json['givenName']),
            picture_url=str(json['pictureUrl']),
            idp_config_url=str(json['idpConfigUrl']),
            idp_login_url=str(json['idpLoginUrl']),
            login_state=LoginState.from_json(json['loginState']),
            terms_of_service_url=str(json['termsOfServiceUrl']) if 'termsOfServiceUrl' in json else None,
            privacy_policy_url=str(json['privacyPolicyUrl']) if 'privacyPolicyUrl' in json else None,
        )


def enable(
        disable_rejection_delay: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param disable_rejection_delay: *(Optional)* Allows callers to disable the promise rejection delay that would normally happen, if this is unimportant to what's being tested. (step 4 of https://fedidcg.github.io/FedCM/#browser-api-rp-sign-in)
    '''
    params: T_JSON_DICT = dict()
    if disable_rejection_delay is not None:
        params['disableRejectionDelay'] = disable_rejection_delay
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.enable',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.disable',
    }
    json = yield cmd_dict


def select_account(
        dialog_id: str,
        account_index: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param account_index:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.selectAccount',
        'params': params,
    }
    json = yield cmd_dict


def click_dialog_button(
        dialog_id: str,
        dialog_button: DialogButton
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param dialog_button:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['dialogButton'] = dialog_button.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.clickDialogButton',
        'params': params,
    }
    json = yield cmd_dict


def open_url(
        dialog_id: str,
        account_index: int,
        account_url_type: AccountUrlType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param account_index:
    :param account_url_type:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    params['accountUrlType'] = account_url_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.openUrl',
        'params': params,
    }
    json = yield cmd_dict


def dismiss_dialog(
        dialog_id: str,
        trigger_cooldown: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param trigger_cooldown: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    if trigger_cooldown is not None:
        params['triggerCooldown'] = trigger_cooldown
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.dismissDialog',
        'params': params,
    }
    json = yield cmd_dict


def reset_cooldown() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets the cooldown time, if any, to allow the next FedCM call to show
    a dialog even if one was recently dismissed by the user.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.resetCooldown',
    }
    json = yield cmd_dict


@event_class('FedCm.dialogShown')
@dataclass
class DialogShown:
    dialog_id: str
    dialog_type: DialogType
    accounts: typing.List[Account]
    #: These exist primarily so that the caller can verify the
    #: RP context was used appropriately.
    title: str
    subtitle: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogShown:
        return cls(
            dialog_id=str(json['dialogId']),
            dialog_type=DialogType.from_json(json['dialogType']),
            accounts=[Account.from_json(i) for i in json['accounts']],
            title=str(json['title']),
            subtitle=str(json['subtitle']) if 'subtitle' in json else None
        )


@event_class('FedCm.dialogClosed')
@dataclass
class DialogClosed:
    '''
    Triggered when a dialog is closed, either by user action, JS abort,
    or a command below.
    '''
    dialog_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogClosed:
        return cls(
            dialog_id=str(json['dialogId'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\fed_cm.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: FedCm (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class LoginState(enum.Enum):
    '''
    Whether this is a sign-up or sign-in action for this account, i.e.
    whether this account has ever been used to sign in to this RP before.
    '''
    SIGN_IN = "SignIn"
    SIGN_UP = "SignUp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DialogType(enum.Enum):
    '''
    The types of FedCM dialogs.
    '''
    ACCOUNT_CHOOSER = "AccountChooser"
    AUTO_REAUTHN = "AutoReauthn"
    CONFIRM_IDP_LOGIN = "ConfirmIdpLogin"
    ERROR = "Error"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DialogButton(enum.Enum):
    '''
    The buttons on the FedCM dialog.
    '''
    CONFIRM_IDP_LOGIN_CONTINUE = "ConfirmIdpLoginContinue"
    ERROR_GOT_IT = "ErrorGotIt"
    ERROR_MORE_DETAILS = "ErrorMoreDetails"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AccountUrlType(enum.Enum):
    '''
    The URLs that each account has
    '''
    TERMS_OF_SERVICE = "TermsOfService"
    PRIVACY_POLICY = "PrivacyPolicy"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class Account:
    '''
    Corresponds to IdentityRequestAccount
    '''
    account_id: str

    email: str

    name: str

    given_name: str

    picture_url: str

    idp_config_url: str

    idp_login_url: str

    login_state: LoginState

    #: These two are only set if the loginState is signUp
    terms_of_service_url: typing.Optional[str] = None

    privacy_policy_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['accountId'] = self.account_id
        json['email'] = self.email
        json['name'] = self.name
        json['givenName'] = self.given_name
        json['pictureUrl'] = self.picture_url
        json['idpConfigUrl'] = self.idp_config_url
        json['idpLoginUrl'] = self.idp_login_url
        json['loginState'] = self.login_state.to_json()
        if self.terms_of_service_url is not None:
            json['termsOfServiceUrl'] = self.terms_of_service_url
        if self.privacy_policy_url is not None:
            json['privacyPolicyUrl'] = self.privacy_policy_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            account_id=str(json['accountId']),
            email=str(json['email']),
            name=str(json['name']),
            given_name=str(json['givenName']),
            picture_url=str(json['pictureUrl']),
            idp_config_url=str(json['idpConfigUrl']),
            idp_login_url=str(json['idpLoginUrl']),
            login_state=LoginState.from_json(json['loginState']),
            terms_of_service_url=str(json['termsOfServiceUrl']) if 'termsOfServiceUrl' in json else None,
            privacy_policy_url=str(json['privacyPolicyUrl']) if 'privacyPolicyUrl' in json else None,
        )


def enable(
        disable_rejection_delay: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param disable_rejection_delay: *(Optional)* Allows callers to disable the promise rejection delay that would normally happen, if this is unimportant to what's being tested. (step 4 of https://fedidcg.github.io/FedCM/#browser-api-rp-sign-in)
    '''
    params: T_JSON_DICT = dict()
    if disable_rejection_delay is not None:
        params['disableRejectionDelay'] = disable_rejection_delay
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.enable',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.disable',
    }
    json = yield cmd_dict


def select_account(
        dialog_id: str,
        account_index: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param account_index:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.selectAccount',
        'params': params,
    }
    json = yield cmd_dict


def click_dialog_button(
        dialog_id: str,
        dialog_button: DialogButton
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param dialog_button:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['dialogButton'] = dialog_button.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.clickDialogButton',
        'params': params,
    }
    json = yield cmd_dict


def open_url(
        dialog_id: str,
        account_index: int,
        account_url_type: AccountUrlType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param account_index:
    :param account_url_type:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    params['accountUrlType'] = account_url_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.openUrl',
        'params': params,
    }
    json = yield cmd_dict


def dismiss_dialog(
        dialog_id: str,
        trigger_cooldown: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param trigger_cooldown: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    if trigger_cooldown is not None:
        params['triggerCooldown'] = trigger_cooldown
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.dismissDialog',
        'params': params,
    }
    json = yield cmd_dict


def reset_cooldown() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets the cooldown time, if any, to allow the next FedCM call to show
    a dialog even if one was recently dismissed by the user.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.resetCooldown',
    }
    json = yield cmd_dict


@event_class('FedCm.dialogShown')
@dataclass
class DialogShown:
    dialog_id: str
    dialog_type: DialogType
    accounts: typing.List[Account]
    #: These exist primarily so that the caller can verify the
    #: RP context was used appropriately.
    title: str
    subtitle: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogShown:
        return cls(
            dialog_id=str(json['dialogId']),
            dialog_type=DialogType.from_json(json['dialogType']),
            accounts=[Account.from_json(i) for i in json['accounts']],
            title=str(json['title']),
            subtitle=str(json['subtitle']) if 'subtitle' in json else None
        )


@event_class('FedCm.dialogClosed')
@dataclass
class DialogClosed:
    '''
    Triggered when a dialog is closed, either by user action, JS abort,
    or a command below.
    '''
    dialog_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogClosed:
        return cls(
            dialog_id=str(json['dialogId'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\fed_cm.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: FedCm (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class LoginState(enum.Enum):
    '''
    Whether this is a sign-up or sign-in action for this account, i.e.
    whether this account has ever been used to sign in to this RP before.
    '''
    SIGN_IN = "SignIn"
    SIGN_UP = "SignUp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DialogType(enum.Enum):
    '''
    The types of FedCM dialogs.
    '''
    ACCOUNT_CHOOSER = "AccountChooser"
    AUTO_REAUTHN = "AutoReauthn"
    CONFIRM_IDP_LOGIN = "ConfirmIdpLogin"
    ERROR = "Error"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DialogButton(enum.Enum):
    '''
    The buttons on the FedCM dialog.
    '''
    CONFIRM_IDP_LOGIN_CONTINUE = "ConfirmIdpLoginContinue"
    ERROR_GOT_IT = "ErrorGotIt"
    ERROR_MORE_DETAILS = "ErrorMoreDetails"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AccountUrlType(enum.Enum):
    '''
    The URLs that each account has
    '''
    TERMS_OF_SERVICE = "TermsOfService"
    PRIVACY_POLICY = "PrivacyPolicy"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class Account:
    '''
    Corresponds to IdentityRequestAccount
    '''
    account_id: str

    email: str

    name: str

    given_name: str

    picture_url: str

    idp_config_url: str

    idp_login_url: str

    login_state: LoginState

    #: These two are only set if the loginState is signUp
    terms_of_service_url: typing.Optional[str] = None

    privacy_policy_url: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['accountId'] = self.account_id
        json['email'] = self.email
        json['name'] = self.name
        json['givenName'] = self.given_name
        json['pictureUrl'] = self.picture_url
        json['idpConfigUrl'] = self.idp_config_url
        json['idpLoginUrl'] = self.idp_login_url
        json['loginState'] = self.login_state.to_json()
        if self.terms_of_service_url is not None:
            json['termsOfServiceUrl'] = self.terms_of_service_url
        if self.privacy_policy_url is not None:
            json['privacyPolicyUrl'] = self.privacy_policy_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            account_id=str(json['accountId']),
            email=str(json['email']),
            name=str(json['name']),
            given_name=str(json['givenName']),
            picture_url=str(json['pictureUrl']),
            idp_config_url=str(json['idpConfigUrl']),
            idp_login_url=str(json['idpLoginUrl']),
            login_state=LoginState.from_json(json['loginState']),
            terms_of_service_url=str(json['termsOfServiceUrl']) if 'termsOfServiceUrl' in json else None,
            privacy_policy_url=str(json['privacyPolicyUrl']) if 'privacyPolicyUrl' in json else None,
        )


def enable(
        disable_rejection_delay: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param disable_rejection_delay: *(Optional)* Allows callers to disable the promise rejection delay that would normally happen, if this is unimportant to what's being tested. (step 4 of https://fedidcg.github.io/FedCM/#browser-api-rp-sign-in)
    '''
    params: T_JSON_DICT = dict()
    if disable_rejection_delay is not None:
        params['disableRejectionDelay'] = disable_rejection_delay
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.enable',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.disable',
    }
    json = yield cmd_dict


def select_account(
        dialog_id: str,
        account_index: int
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param account_index:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.selectAccount',
        'params': params,
    }
    json = yield cmd_dict


def click_dialog_button(
        dialog_id: str,
        dialog_button: DialogButton
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param dialog_button:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['dialogButton'] = dialog_button.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.clickDialogButton',
        'params': params,
    }
    json = yield cmd_dict


def open_url(
        dialog_id: str,
        account_index: int,
        account_url_type: AccountUrlType
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param account_index:
    :param account_url_type:
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    params['accountUrlType'] = account_url_type.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.openUrl',
        'params': params,
    }
    json = yield cmd_dict


def dismiss_dialog(
        dialog_id: str,
        trigger_cooldown: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param dialog_id:
    :param trigger_cooldown: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    if trigger_cooldown is not None:
        params['triggerCooldown'] = trigger_cooldown
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.dismissDialog',
        'params': params,
    }
    json = yield cmd_dict


def reset_cooldown() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets the cooldown time, if any, to allow the next FedCM call to show
    a dialog even if one was recently dismissed by the user.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'FedCm.resetCooldown',
    }
    json = yield cmd_dict


@event_class('FedCm.dialogShown')
@dataclass
class DialogShown:
    dialog_id: str
    dialog_type: DialogType
    accounts: typing.List[Account]
    #: These exist primarily so that the caller can verify the
    #: RP context was used appropriately.
    title: str
    subtitle: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogShown:
        return cls(
            dialog_id=str(json['dialogId']),
            dialog_type=DialogType.from_json(json['dialogType']),
            accounts=[Account.from_json(i) for i in json['accounts']],
            title=str(json['title']),
            subtitle=str(json['subtitle']) if 'subtitle' in json else None
        )


@event_class('FedCm.dialogClosed')
@dataclass
class DialogClosed:
    '''
    Triggered when a dialog is closed, either by user action, JS abort,
    or a command below.
    '''
    dialog_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DialogClosed:
        return cls(
            dialog_id=str(json['dialogId'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\asyncio\base.py ===
# ext/asyncio/base.py
# Copyright (C) 2020-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import abc
import functools
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Generator
from typing import Generic
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
import weakref

from . import exc as async_exc
from ... import util
from ...util.typing import Literal
from ...util.typing import Self

_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)


_PT = TypeVar("_PT", bound=Any)


class ReversibleProxy(Generic[_PT]):
    _proxy_objects: ClassVar[
        Dict[weakref.ref[Any], weakref.ref[ReversibleProxy[Any]]]
    ] = {}
    __slots__ = ("__weakref__",)

    @overload
    def _assign_proxied(self, target: _PT) -> _PT: ...

    @overload
    def _assign_proxied(self, target: None) -> None: ...

    def _assign_proxied(self, target: Optional[_PT]) -> Optional[_PT]:
        if target is not None:
            target_ref: weakref.ref[_PT] = weakref.ref(
                target, ReversibleProxy._target_gced
            )
            proxy_ref = weakref.ref(
                self,
                functools.partial(ReversibleProxy._target_gced, target_ref),
            )
            ReversibleProxy._proxy_objects[target_ref] = proxy_ref

        return target

    @classmethod
    def _target_gced(
        cls,
        ref: weakref.ref[_PT],
        proxy_ref: Optional[weakref.ref[Self]] = None,  # noqa: U100
    ) -> None:
        cls._proxy_objects.pop(ref, None)

    @classmethod
    def _regenerate_proxy_for_target(
        cls, target: _PT, **additional_kw: Any
    ) -> Self:
        raise NotImplementedError()

    @overload
    @classmethod
    def _retrieve_proxy_for_target(
        cls, target: _PT, regenerate: Literal[True] = ..., **additional_kw: Any
    ) -> Self: ...

    @overload
    @classmethod
    def _retrieve_proxy_for_target(
        cls, target: _PT, regenerate: bool = True, **additional_kw: Any
    ) -> Optional[Self]: ...

    @classmethod
    def _retrieve_proxy_for_target(
        cls, target: _PT, regenerate: bool = True, **additional_kw: Any
    ) -> Optional[Self]:
        try:
            proxy_ref = cls._proxy_objects[weakref.ref(target)]
        except KeyError:
            pass
        else:
            proxy = proxy_ref()
            if proxy is not None:
                return proxy  # type: ignore

        if regenerate:
            return cls._regenerate_proxy_for_target(target, **additional_kw)
        else:
            return None


class StartableContext(Awaitable[_T_co], abc.ABC):
    __slots__ = ()

    @abc.abstractmethod
    async def start(self, is_ctxmanager: bool = False) -> _T_co:
        raise NotImplementedError()

    def __await__(self) -> Generator[Any, Any, _T_co]:
        return self.start().__await__()

    async def __aenter__(self) -> _T_co:
        return await self.start(is_ctxmanager=True)

    @abc.abstractmethod
    async def __aexit__(
        self, type_: Any, value: Any, traceback: Any
    ) -> Optional[bool]:
        pass

    def _raise_for_not_started(self) -> NoReturn:
        raise async_exc.AsyncContextNotStarted(
            "%s context has not been started and object has not been awaited."
            % (self.__class__.__name__)
        )


class GeneratorStartableContext(StartableContext[_T_co]):
    __slots__ = ("gen",)

    gen: AsyncGenerator[_T_co, Any]

    def __init__(
        self,
        func: Callable[..., AsyncIterator[_T_co]],
        args: Tuple[Any, ...],
        kwds: Dict[str, Any],
    ):
        self.gen = func(*args, **kwds)  # type: ignore

    async def start(self, is_ctxmanager: bool = False) -> _T_co:
        try:
            start_value = await util.anext_(self.gen)
        except StopAsyncIteration:
            raise RuntimeError("generator didn't yield") from None

        # if not a context manager, then interrupt the generator, don't
        # let it complete.   this step is technically not needed, as the
        # generator will close in any case at gc time.  not clear if having
        # this here is a good idea or not (though it helps for clarity IMO)
        if not is_ctxmanager:
            await self.gen.aclose()

        return start_value

    async def __aexit__(
        self, typ: Any, value: Any, traceback: Any
    ) -> Optional[bool]:
        # vendored from contextlib.py
        if typ is None:
            try:
                await util.anext_(self.gen)
            except StopAsyncIteration:
                return False
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                # Need to force instantiation so we can reliably
                # tell if we get the same exception back
                value = typ()
            try:
                await self.gen.athrow(value)
            except StopAsyncIteration as exc:
                # Suppress StopIteration *unless* it's the same exception that
                # was passed to throw().  This prevents a StopIteration
                # raised inside the "with" statement from being suppressed.
                return exc is not value
            except RuntimeError as exc:
                # Don't re-raise the passed in exception. (issue27122)
                if exc is value:
                    return False
                # Avoid suppressing if a Stop(Async)Iteration exception
                # was passed to athrow() and later wrapped into a RuntimeError
                # (see PEP 479 for sync generators; async generators also
                # have this behavior). But do this only if the exception
                # wrapped
                # by the RuntimeError is actully Stop(Async)Iteration (see
                # issue29692).
                if (
                    isinstance(value, (StopIteration, StopAsyncIteration))
                    and exc.__cause__ is value
                ):
                    return False
                raise
            except BaseException as exc:
                # only re-raise if it's *not* the exception that was
                # passed to throw(), because __exit__() must not raise
                # an exception unless __exit__() itself failed.  But throw()
                # has to raise the exception to signal propagation, so this
                # fixes the impedance mismatch between the throw() protocol
                # and the __exit__() protocol.
                if exc is not value:
                    raise
                return False
            raise RuntimeError("generator didn't stop after athrow()")


def asyncstartablecontext(
    func: Callable[..., AsyncIterator[_T_co]]
) -> Callable[..., GeneratorStartableContext[_T_co]]:
    """@asyncstartablecontext decorator.

    the decorated function can be called either as ``async with fn()``, **or**
    ``await fn()``.   This is decidedly different from what
    ``@contextlib.asynccontextmanager`` supports, and the usage pattern
    is different as well.

    Typical usage:

    .. sourcecode:: text

        @asyncstartablecontext
        async def some_async_generator(<arguments>):
            <setup>
            try:
                yield <value>
            except GeneratorExit:
                # return value was awaited, no context manager is present
                # and caller will .close() the resource explicitly
                pass
            else:
                <context manager cleanup>


    Above, ``GeneratorExit`` is caught if the function were used as an
    ``await``.  In this case, it's essential that the cleanup does **not**
    occur, so there should not be a ``finally`` block.

    If ``GeneratorExit`` is not invoked, this means we're in ``__aexit__``
    and we were invoked as a context manager, and cleanup should proceed.


    """

    @functools.wraps(func)
    def helper(*args: Any, **kwds: Any) -> GeneratorStartableContext[_T_co]:
        return GeneratorStartableContext(func, args, kwds)

    return helper


class ProxyComparable(ReversibleProxy[_PT]):
    __slots__ = ()

    @util.ro_non_memoized_property
    def _proxied(self) -> _PT:
        raise NotImplementedError()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self._proxied == other._proxied
        )

    def __ne__(self, other: Any) -> bool:
        return (
            not isinstance(other, self.__class__)
            or self._proxied != other._proxied
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_timeouts.py ===
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, TypeVar

import outcome
import pytest

import trio

from .. import _core
from .._core._tests.tutil import slow
from .._timeouts import (
    TooSlowError,
    fail_after,
    fail_at,
    move_on_after,
    move_on_at,
    sleep,
    sleep_forever,
    sleep_until,
)
from ..testing import assert_checkpoints

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

T = TypeVar("T")


async def check_takes_about(f: Callable[[], Awaitable[T]], expected_dur: float) -> T:
    start = time.perf_counter()
    result = await outcome.acapture(f)
    dur = time.perf_counter() - start
    print(dur / expected_dur)
    # 1.5 is an arbitrary fudge factor because there's always some delay
    # between when we become eligible to wake up and when we actually do. We
    # used to sleep for 0.05, and regularly observed overruns of 1.6x on
    # Appveyor, and then started seeing overruns of 2.3x on Travis's macOS, so
    # now we bumped up the sleep to 1 second, marked the tests as slow, and
    # hopefully now the proportional error will be less huge.
    #
    # We also also for durations that are a hair shorter than expected. For
    # example, here's a run on Windows where a 1.0 second sleep was measured
    # to take 0.9999999999999858 seconds:
    #   https://ci.appveyor.com/project/njsmith/trio/build/1.0.768/job/3lbdyxl63q3h9s21
    # I believe that what happened here is that Windows's low clock resolution
    # meant that our calls to time.monotonic() returned exactly the same
    # values as the calls inside the actual run loop, but the two subtractions
    # returned slightly different values because the run loop's clock adds a
    # random floating point offset to both times, which should cancel out, but
    # lol floating point we got slightly different rounding errors. (That
    # value above is exactly 128 ULPs below 1.0, which would make sense if it
    # started as a 1 ULP error at a different dynamic range.)
    assert (1 - 1e-8) <= (dur / expected_dur) < 1.5

    return result.unwrap()


# How long to (attempt to) sleep for when testing. Smaller numbers make the
# test suite go faster.
TARGET = 1.0


@slow
async def test_sleep() -> None:
    async def sleep_1() -> None:
        await sleep_until(_core.current_time() + TARGET)

    await check_takes_about(sleep_1, TARGET)

    async def sleep_2() -> None:
        await sleep(TARGET)

    await check_takes_about(sleep_2, TARGET)

    with assert_checkpoints():
        await sleep(0)
    # This also serves as a test of the trivial move_on_at
    with move_on_at(_core.current_time()):
        with pytest.raises(_core.Cancelled):
            await sleep(0)


@slow
async def test_move_on_after() -> None:
    async def sleep_3() -> None:
        with move_on_after(TARGET):
            await sleep(100)

    await check_takes_about(sleep_3, TARGET)


async def test_cannot_wake_sleep_forever() -> None:
    # Test an error occurs if you manually wake sleep_forever().
    task = trio.lowlevel.current_task()

    async def wake_task() -> None:
        await trio.lowlevel.checkpoint()
        trio.lowlevel.reschedule(task, outcome.Value(None))

    async with trio.open_nursery() as nursery:
        nursery.start_soon(wake_task)
        with pytest.raises(RuntimeError):
            await trio.sleep_forever()


class TimeoutScope(Protocol):
    def __call__(self, seconds: float, *, shield: bool) -> trio.CancelScope: ...


@pytest.mark.parametrize("scope", [move_on_after, fail_after])
async def test_context_shields_from_outer(scope: TimeoutScope) -> None:
    with _core.CancelScope() as outer, scope(TARGET, shield=True) as inner:
        outer.cancel()
        try:
            await trio.lowlevel.checkpoint()
        except trio.Cancelled:  # pragma: no cover
            pytest.fail("shield didn't work")
        inner.shield = False
        with pytest.raises(trio.Cancelled):
            await trio.lowlevel.checkpoint()


@slow
async def test_move_on_after_moves_on_even_if_shielded() -> None:
    async def task() -> None:
        with _core.CancelScope() as outer, move_on_after(TARGET, shield=True):
            outer.cancel()
            # The outer scope is cancelled, but this task is protected by the
            # shield, so it manages to get to sleep until deadline is met
            await sleep_forever()

    await check_takes_about(task, TARGET)


@slow
async def test_fail_after_fails_even_if_shielded() -> None:
    async def task() -> None:
        # fmt: off
        # Remove after 3.9 unsupported, black formats in a way that breaks if
        # you do `-X oldparser`
        with pytest.raises(TooSlowError), _core.CancelScope() as outer, fail_after(
            TARGET,
            shield=True,
        ):
            # fmt: on
            outer.cancel()
            # The outer scope is cancelled, but this task is protected by the
            # shield, so it manages to get to sleep until deadline is met
            await sleep_forever()

    await check_takes_about(task, TARGET)


@slow
async def test_fail() -> None:
    async def sleep_4() -> None:
        with fail_at(_core.current_time() + TARGET):
            await sleep(100)

    with pytest.raises(TooSlowError):
        await check_takes_about(sleep_4, TARGET)

    with fail_at(_core.current_time() + 100):
        await sleep(0)

    async def sleep_5() -> None:
        with fail_after(TARGET):
            await sleep(100)

    with pytest.raises(TooSlowError):
        await check_takes_about(sleep_5, TARGET)

    with fail_after(100):
        await sleep(0)


async def test_timeouts_raise_value_error() -> None:
    # deadlines are allowed to be negative, but not delays.
    # neither delays nor deadlines are allowed to be NaN

    nan = float("nan")

    for fun, val in (
        (sleep, -1),
        (sleep, nan),
        (sleep_until, nan),
    ):
        with pytest.raises(
            ValueError,
            match=r"^(deadline|`seconds`) must (not )*be (non-negative|NaN)$",
        ):
            await fun(val)

    for cm, val in (
        (fail_after, -1),
        (fail_after, nan),
        (fail_at, nan),
        (move_on_after, -1),
        (move_on_after, nan),
        (move_on_at, nan),
    ):
        with pytest.raises(
            ValueError,
            match=r"^(deadline|`seconds`) must (not )*be (non-negative|NaN)$",
        ):
            with cm(val):
                pass  # pragma: no cover


async def test_timeout_deadline_on_entry(mock_clock: _core.MockClock) -> None:
    rcs = move_on_after(5)
    assert rcs.relative_deadline == 5

    mock_clock.jump(3)
    start = _core.current_time()
    with rcs as cs:
        assert cs.is_relative is None

        # This would previously be start+2
        assert cs.deadline == start + 5
        assert cs.relative_deadline == 5

        cs.deadline = start + 3
        assert cs.deadline == start + 3
        assert cs.relative_deadline == 3

        cs.relative_deadline = 4
        assert cs.deadline == start + 4
        assert cs.relative_deadline == 4

    rcs = move_on_after(5)
    assert rcs.shield is False
    rcs.shield = True
    assert rcs.shield is True

    mock_clock.jump(3)
    start = _core.current_time()
    with rcs as cs:
        assert cs.deadline == start + 5

        assert rcs is cs


async def test_invalid_access_unentered(mock_clock: _core.MockClock) -> None:
    cs = move_on_after(5)
    mock_clock.jump(3)
    start = _core.current_time()

    match_str = "^unentered relative cancel scope does not have an absolute deadline"
    with pytest.warns(DeprecationWarning, match=match_str):
        assert cs.deadline == start + 5
    mock_clock.jump(1)
    # this is hella sketchy, but they *have* been warned
    with pytest.warns(DeprecationWarning, match=match_str):
        assert cs.deadline == start + 6

    with pytest.warns(DeprecationWarning, match=match_str):
        cs.deadline = 7
    # now transformed into absolute
    assert cs.deadline == 7
    assert not cs.is_relative

    cs = move_on_at(5)

    match_str = (
        "^unentered non-relative cancel scope does not have a relative deadline$"
    )
    with pytest.raises(RuntimeError, match=match_str):
        assert cs.relative_deadline
    with pytest.raises(RuntimeError, match=match_str):
        cs.relative_deadline = 7


@pytest.mark.xfail(reason="not implemented")
async def test_fail_access_before_entering() -> None:  # pragma: no cover
    my_fail_at = fail_at(5)
    assert my_fail_at.deadline  # type: ignore[attr-defined]
    my_fail_after = fail_after(5)
    assert my_fail_after.relative_deadline  # type: ignore[attr-defined]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\qfunctions.py ===
from .functions import defun, defun_wrapped

@defun
def qp(ctx, a, q=None, n=None, **kwargs):
    r"""
    Evaluates the q-Pochhammer symbol (or q-rising factorial)

    .. math ::

        (a; q)_n = \prod_{k=0}^{n-1} (1-a q^k)

    where `n = \infty` is permitted if `|q| < 1`. Called with two arguments,
    ``qp(a,q)`` computes `(a;q)_{\infty}`; with a single argument, ``qp(q)``
    computes `(q;q)_{\infty}`. The special case

    .. math ::

        \phi(q) = (q; q)_{\infty} = \prod_{k=1}^{\infty} (1-q^k) =
            \sum_{k=-\infty}^{\infty} (-1)^k q^{(3k^2-k)/2}

    is also known as the Euler function, or (up to a factor `q^{-1/24}`)
    the Dedekind eta function.

    **Examples**

    If `n` is a positive integer, the function amounts to a finite product::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qp(2,3,5)
        -725305.0
        >>> fprod(1-2*3**k for k in range(5))
        -725305.0
        >>> qp(2,3,0)
        1.0

    Complex arguments are allowed::

        >>> qp(2-1j, 0.75j)
        (0.4628842231660149089976379 + 4.481821753552703090628793j)

    The regular Pochhammer symbol `(a)_n` is obtained in the
    following limit as `q \to 1`::

        >>> a, n = 4, 7
        >>> limit(lambda q: qp(q**a,q,n) / (1-q)**n, 1)
        604800.0
        >>> rf(a,n)
        604800.0

    The Taylor series of the reciprocal Euler function gives
    the partition function `P(n)`, i.e. the number of ways of writing
    `n` as a sum of positive integers::

        >>> taylor(lambda q: 1/qp(q), 0, 10)
        [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 15.0, 22.0, 30.0, 42.0]

    Special values include::

        >>> qp(0)
        1.0
        >>> findroot(diffun(qp), -0.4)   # location of maximum
        -0.4112484791779547734440257
        >>> qp(_)
        1.228348867038575112586878

    The q-Pochhammer symbol is related to the Jacobi theta functions.
    For example, the following identity holds::

        >>> q = mpf(0.5)    # arbitrary
        >>> qp(q)
        0.2887880950866024212788997
        >>> root(3,-2)*root(q,-24)*jtheta(2,pi/6,root(q,6))
        0.2887880950866024212788997

    """
    a = ctx.convert(a)
    if n is None:
        n = ctx.inf
    else:
        n = ctx.convert(n)
    if n < 0:
        raise ValueError("n cannot be negative")
    if q is None:
        q = a
    else:
        q = ctx.convert(q)
    if n == 0:
        return ctx.one + 0*(a+q)
    infinite = (n == ctx.inf)
    same = (a == q)
    if infinite:
        if abs(q) >= 1:
            if same and (q == -1 or q == 1):
                return ctx.zero * q
            raise ValueError("q-function only defined for |q| < 1")
        elif q == 0:
            return ctx.one - a
    maxterms = kwargs.get('maxterms', 50*ctx.prec)
    if infinite and same:
        # Euler's pentagonal theorem
        def terms():
            t = 1
            yield t
            k = 1
            x1 = q
            x2 = q**2
            while 1:
                yield (-1)**k * x1
                yield (-1)**k * x2
                x1 *= q**(3*k+1)
                x2 *= q**(3*k+2)
                k += 1
                if k > maxterms:
                    raise ctx.NoConvergence
        return ctx.sum_accurately(terms)
    # return ctx.nprod(lambda k: 1-a*q**k, [0,n-1])
    def factors():
        k = 0
        r = ctx.one
        while 1:
            yield 1 - a*r
            r *= q
            k += 1
            if k >= n:
                return
            if k > maxterms:
                raise ctx.NoConvergence
    return ctx.mul_accurately(factors)

@defun_wrapped
def qgamma(ctx, z, q, **kwargs):
    r"""
    Evaluates the q-gamma function

    .. math ::

        \Gamma_q(z) = \frac{(q; q)_{\infty}}{(q^z; q)_{\infty}} (1-q)^{1-z}.


    **Examples**

    Evaluation for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qgamma(4,0.75)
        4.046875
        >>> qgamma(6,6)
        121226245.0
        >>> qgamma(3+4j, 0.5j)
        (0.1663082382255199834630088 + 0.01952474576025952984418217j)

    The q-gamma function satisfies a functional equation similar
    to that of the ordinary gamma function::

        >>> q = mpf(0.25)
        >>> z = mpf(2.5)
        >>> qgamma(z+1,q)
        1.428277424823760954685912
        >>> (1-q**z)/(1-q)*qgamma(z,q)
        1.428277424823760954685912

    """
    if abs(q) > 1:
        return ctx.qgamma(z,1/q)*q**((z-2)*(z-1)*0.5)
    return ctx.qp(q, q, None, **kwargs) / \
        ctx.qp(q**z, q, None, **kwargs) * (1-q)**(1-z)

@defun_wrapped
def qfac(ctx, z, q, **kwargs):
    r"""
    Evaluates the q-factorial,

    .. math ::

        [n]_q! = (1+q)(1+q+q^2)\cdots(1+q+\cdots+q^{n-1})

    or more generally

    .. math ::

        [z]_q! = \frac{(q;q)_z}{(1-q)^z}.

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qfac(0,0)
        1.0
        >>> qfac(4,3)
        2080.0
        >>> qfac(5,6)
        121226245.0
        >>> qfac(1+1j, 2+1j)
        (0.4370556551322672478613695 + 0.2609739839216039203708921j)

    """
    if ctx.isint(z) and ctx._re(z) > 0:
        n = int(ctx._re(z))
        return ctx.qp(q, q, n, **kwargs) / (1-q)**n
    return ctx.qgamma(z+1, q, **kwargs)

@defun
def qhyper(ctx, a_s, b_s, q, z, **kwargs):
    r"""
    Evaluates the basic hypergeometric series or hypergeometric q-series

    .. math ::

        \,_r\phi_s \left[\begin{matrix}
            a_1 & a_2 & \ldots & a_r \\
            b_1 & b_2 & \ldots & b_s
        \end{matrix} ; q,z \right] =
        \sum_{n=0}^\infty
        \frac{(a_1;q)_n, \ldots, (a_r;q)_n}
             {(b_1;q)_n, \ldots, (b_s;q)_n}
        \left((-1)^n q^{n\choose 2}\right)^{1+s-r}
        \frac{z^n}{(q;q)_n}

    where `(a;q)_n` denotes the q-Pochhammer symbol (see :func:`~mpmath.qp`).

    **Examples**

    Evaluation works for real and complex arguments::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qhyper([0.5], [2.25], 0.25, 4)
        -0.1975849091263356009534385
        >>> qhyper([0.5], [2.25], 0.25-0.25j, 4)
        (2.806330244925716649839237 + 3.568997623337943121769938j)
        >>> qhyper([1+j], [2,3+0.5j], 0.25, 3+4j)
        (9.112885171773400017270226 - 1.272756997166375050700388j)

    Comparing with a summation of the defining series, using
    :func:`~mpmath.nsum`::

        >>> b, q, z = 3, 0.25, 0.5
        >>> qhyper([], [b], q, z)
        0.6221136748254495583228324
        >>> nsum(lambda n: z**n / qp(q,q,n)/qp(b,q,n) * q**(n*(n-1)), [0,inf])
        0.6221136748254495583228324

    """
    #a_s = [ctx._convert_param(a)[0] for a in a_s]
    #b_s = [ctx._convert_param(b)[0] for b in b_s]
    #q = ctx._convert_param(q)[0]
    a_s = [ctx.convert(a) for a in a_s]
    b_s = [ctx.convert(b) for b in b_s]
    q = ctx.convert(q)
    z = ctx.convert(z)
    r = len(a_s)
    s = len(b_s)
    d = 1+s-r
    maxterms = kwargs.get('maxterms', 50*ctx.prec)
    def terms():
        t = ctx.one
        yield t
        qk = 1
        k = 0
        x = 1
        while 1:
            for a in a_s:
                p = 1 - a*qk
                t *= p
            for b in b_s:
                p = 1 - b*qk
                if not p:
                    raise ValueError
                t /= p
            t *= z
            x *= (-1)**d * qk ** d
            qk *= q
            t /= (1 - qk)
            k += 1
            yield t * x
            if k > maxterms:
                raise ctx.NoConvergence
    return ctx.sum_accurately(terms)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\configuration.py ===
import logging
import os
import subprocess
from optparse import Values
from typing import Any, List, Optional

from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.configuration import (
    Configuration,
    Kind,
    get_configuration_files,
    kinds,
)
from pip._internal.exceptions import PipError
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_prog, write_output

logger = logging.getLogger(__name__)


class ConfigurationCommand(Command):
    """
    Manage local and global configuration.

    Subcommands:

    - list: List the active configuration (or from the file specified)
    - edit: Edit the configuration file in an editor
    - get: Get the value associated with command.option
    - set: Set the command.option=value
    - unset: Unset the value associated with command.option
    - debug: List the configuration files and values defined under them

    Configuration keys should be dot separated command and option name,
    with the special prefix "global" affecting any command. For example,
    "pip config set global.index-url https://example.org/" would configure
    the index url for all commands, but "pip config set download.timeout 10"
    would configure a 10 second timeout only for "pip download" commands.

    If none of --user, --global and --site are passed, a virtual
    environment configuration file is used if one is active and the file
    exists. Otherwise, all modifications happen to the user file by
    default.
    """

    ignore_require_venv = True
    usage = """
        %prog [<file-option>] list
        %prog [<file-option>] [--editor <editor-path>] edit

        %prog [<file-option>] get command.option
        %prog [<file-option>] set command.option value
        %prog [<file-option>] unset command.option
        %prog [<file-option>] debug
    """

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "--editor",
            dest="editor",
            action="store",
            default=None,
            help=(
                "Editor to use to edit the file. Uses VISUAL or EDITOR "
                "environment variables if not provided."
            ),
        )

        self.cmd_opts.add_option(
            "--global",
            dest="global_file",
            action="store_true",
            default=False,
            help="Use the system-wide configuration file only",
        )

        self.cmd_opts.add_option(
            "--user",
            dest="user_file",
            action="store_true",
            default=False,
            help="Use the user configuration file only",
        )

        self.cmd_opts.add_option(
            "--site",
            dest="site_file",
            action="store_true",
            default=False,
            help="Use the current environment configuration file only",
        )

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        handlers = {
            "list": self.list_values,
            "edit": self.open_in_editor,
            "get": self.get_name,
            "set": self.set_name_value,
            "unset": self.unset_name,
            "debug": self.list_config_values,
        }

        # Determine action
        if not args or args[0] not in handlers:
            logger.error(
                "Need an action (%s) to perform.",
                ", ".join(sorted(handlers)),
            )
            return ERROR

        action = args[0]

        # Determine which configuration files are to be loaded
        #    Depends on whether the command is modifying.
        try:
            load_only = self._determine_file(
                options, need_value=(action in ["get", "set", "unset", "edit"])
            )
        except PipError as e:
            logger.error(e.args[0])
            return ERROR

        # Load a new configuration
        self.configuration = Configuration(
            isolated=options.isolated_mode, load_only=load_only
        )
        self.configuration.load()

        # Error handling happens here, not in the action-handlers.
        try:
            handlers[action](options, args[1:])
        except PipError as e:
            logger.error(e.args[0])
            return ERROR

        return SUCCESS

    def _determine_file(self, options: Values, need_value: bool) -> Optional[Kind]:
        file_options = [
            key
            for key, value in (
                (kinds.USER, options.user_file),
                (kinds.GLOBAL, options.global_file),
                (kinds.SITE, options.site_file),
            )
            if value
        ]

        if not file_options:
            if not need_value:
                return None
            # Default to user, unless there's a site file.
            elif any(
                os.path.exists(site_config_file)
                for site_config_file in get_configuration_files()[kinds.SITE]
            ):
                return kinds.SITE
            else:
                return kinds.USER
        elif len(file_options) == 1:
            return file_options[0]

        raise PipError(
            "Need exactly one file to operate upon "
            "(--user, --site, --global) to perform."
        )

    def list_values(self, options: Values, args: List[str]) -> None:
        self._get_n_args(args, "list", n=0)

        for key, value in sorted(self.configuration.items()):
            write_output("%s=%r", key, value)

    def get_name(self, options: Values, args: List[str]) -> None:
        key = self._get_n_args(args, "get [name]", n=1)
        value = self.configuration.get_value(key)

        write_output("%s", value)

    def set_name_value(self, options: Values, args: List[str]) -> None:
        key, value = self._get_n_args(args, "set [name] [value]", n=2)
        self.configuration.set_value(key, value)

        self._save_configuration()

    def unset_name(self, options: Values, args: List[str]) -> None:
        key = self._get_n_args(args, "unset [name]", n=1)
        self.configuration.unset_value(key)

        self._save_configuration()

    def list_config_values(self, options: Values, args: List[str]) -> None:
        """List config key-value pairs across different config files"""
        self._get_n_args(args, "debug", n=0)

        self.print_env_var_values()
        # Iterate over config files and print if they exist, and the
        # key-value pairs present in them if they do
        for variant, files in sorted(self.configuration.iter_config_files()):
            write_output("%s:", variant)
            for fname in files:
                with indent_log():
                    file_exists = os.path.exists(fname)
                    write_output("%s, exists: %r", fname, file_exists)
                    if file_exists:
                        self.print_config_file_values(variant)

    def print_config_file_values(self, variant: Kind) -> None:
        """Get key-value pairs from the file of a variant"""
        for name, value in self.configuration.get_values_in_config(variant).items():
            with indent_log():
                write_output("%s: %s", name, value)

    def print_env_var_values(self) -> None:
        """Get key-values pairs present as environment variables"""
        write_output("%s:", "env_var")
        with indent_log():
            for key, value in sorted(self.configuration.get_environ_vars()):
                env_var = f"PIP_{key.upper()}"
                write_output("%s=%r", env_var, value)

    def open_in_editor(self, options: Values, args: List[str]) -> None:
        editor = self._determine_editor(options)

        fname = self.configuration.get_file_to_edit()
        if fname is None:
            raise PipError("Could not determine appropriate file.")
        elif '"' in fname:
            # This shouldn't happen, unless we see a username like that.
            # If that happens, we'd appreciate a pull request fixing this.
            raise PipError(
                f'Can not open an editor for a file name containing "\n{fname}'
            )

        try:
            subprocess.check_call(f'{editor} "{fname}"', shell=True)
        except FileNotFoundError as e:
            if not e.filename:
                e.filename = editor
            raise
        except subprocess.CalledProcessError as e:
            raise PipError(f"Editor Subprocess exited with exit code {e.returncode}")

    def _get_n_args(self, args: List[str], example: str, n: int) -> Any:
        """Helper to make sure the command got the right number of arguments"""
        if len(args) != n:
            msg = (
                f"Got unexpected number of arguments, expected {n}. "
                f'(example: "{get_prog()} config {example}")'
            )
            raise PipError(msg)

        if n == 1:
            return args[0]
        else:
            return args

    def _save_configuration(self) -> None:
        # We successfully ran a modifying command. Need to save the
        # configuration.
        try:
            self.configuration.save()
        except Exception:
            logger.exception(
                "Unable to save configuration. Please report this as a bug."
            )
            raise PipError("Internal Error.")

    def _determine_editor(self, options: Values) -> str:
        if options.editor is not None:
            return options.editor
        elif "VISUAL" in os.environ:
            return os.environ["VISUAL"]
        elif "EDITOR" in os.environ:
            return os.environ["EDITOR"]
        else:
            raise PipError("Could not determine editor to use.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\sam2_image_onnx_predictor.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

import numpy as np
import torch
from PIL.Image import Image
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2_utils import decoder_shape_dict, encoder_shape_dict

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession

logger = logging.getLogger(__name__)


def create_ort_session(
    onnx_path: str,
    session_options=None,
    provider="CUDAExecutionProvider",
    enable_cuda_graph=False,
    use_tf32=True,
) -> InferenceSession:
    if provider == "CUDAExecutionProvider":
        device_id = torch.cuda.current_device()
        provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
        provider_options["use_tf32"] = int(use_tf32)
        providers = [(provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    logger.info("Using providers: %s", providers)
    return InferenceSession(onnx_path, session_options, providers=providers)


def create_session(
    onnx_path: str,
    session_options=None,
    provider="CUDAExecutionProvider",
    device: str | torch.device = "cuda",
    enable_cuda_graph=False,
) -> CudaSession:
    ort_session = create_ort_session(
        onnx_path, session_options, provider, enable_cuda_graph=enable_cuda_graph, use_tf32=True
    )
    cuda_session = CudaSession(ort_session, device=torch.device(device), enable_cuda_graph=enable_cuda_graph)
    return cuda_session


class SAM2ImageOnnxPredictor(SAM2ImagePredictor):
    def __init__(
        self,
        sam_model: SAM2Base,
        image_encoder_onnx_path: str = "",
        image_decoder_onnx_path: str = "",
        image_decoder_multi_onnx_path: str = "",
        provider: str = "CUDAExecutionProvider",
        device: str | torch.device = "cuda",
        onnx_dtype: torch.dtype = torch.float32,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to compute the image embedding for an image, and then allow mask prediction given prompts.

        Arguments:
          sam_model (SAM2Base): The model to use for mask prediction.
          onnx_directory (str): The path of the directory that contains encoder and decoder onnx models.
          onnx_dtype (torch.dtype): The data type to use for ONNX inputs.
          mask_threshold (float): The threshold to convert mask logits to binary masks. Default is 0.0.
          max_hole_area (float): If max_hole_area > 0, we fill small holes in up to
                                 the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (float): If max_sprinkle_area > 0, we remove small sprinkles up to
                                     the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__(
            sam_model, mask_threshold=mask_threshold, max_hole_area=max_hole_area, max_sprinkle_area=max_sprinkle_area
        )

        logger.debug("self.device=%s, device=%s", self.device, device)

        # This model is exported by image_encoder.py.
        self.encoder_session = create_session(
            image_encoder_onnx_path,
            session_options=None,
            provider=provider,
            device=device,
            enable_cuda_graph=False,
        )
        self.onnx_dtype = onnx_dtype

        # This model is exported by image_decoder.py. It outputs only one mask.
        self.decoder_session = create_session(
            image_decoder_onnx_path,
            session_options=None,
            provider=provider,
            device=device,
            enable_cuda_graph=False,
        )

        # This model is exported by image_decoder.py. It outputs multiple (3) masks.
        self.decoder_session_multi_out = create_session(
            image_decoder_multi_onnx_path,
            session_options=None,
            provider=provider,
            device=device,
            enable_cuda_graph=False,
        )

    @torch.no_grad()
    def set_image(self, image: np.ndarray | Image):
        """
        Calculates the image embeddings for the provided image.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format.
              The image should be in HWC format if np.ndarray, or WHC format if PIL Image with pixel values in [0, 255].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            # For numpy array image, we assume (HxWxC) format.
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert len(input_image.shape) == 4 and input_image.shape[1] == 3, (
            f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        )

        # Computing image embeddings for the provided image
        io_shapes = encoder_shape_dict(batch_size=1, height=input_image.shape[2], width=input_image.shape[3])
        self.encoder_session.allocate_buffers(io_shapes)

        feed_dict = {"image": input_image.to(self.onnx_dtype).to(self.device)}

        for key, value in feed_dict.items():
            logger.debug(f"{key}: {value.shape}, {value.dtype}")
        logger.debug(f"encoder onnx: {self.encoder_session.ort_session._model_path}")

        ort_outputs = self.encoder_session.infer(feed_dict)

        self._features = {
            "image_embed": ort_outputs["image_embeddings"],
            "high_res_feats": [ort_outputs[f"image_features_{i}"] for i in range(2)],
        }
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def _predict(
        self,
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        assert not return_logits  # onnx model is exported for returning bool masks.

        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        assert concat_points is not None
        num_labels = concat_points[0].shape[0]
        shape_dict = decoder_shape_dict(
            original_image_height=self._orig_hw[img_idx][0],
            original_image_width=self._orig_hw[img_idx][1],
            num_labels=num_labels,
            max_points=concat_points[0].shape[1],
            num_masks=3 if multimask_output else 1,
        )
        if multimask_output:
            decoder_session = self.decoder_session_multi_out
        else:
            decoder_session = self.decoder_session

        decoder_session.allocate_buffers(shape_dict)

        image_features_0 = self._features["high_res_feats"][0][img_idx].unsqueeze(0)
        image_features_1 = self._features["high_res_feats"][1][img_idx].unsqueeze(0)
        image_embeddings = self._features["image_embed"][img_idx].unsqueeze(0)

        if mask_input is None:
            input_masks = torch.zeros(num_labels, 1, 256, 256, dtype=self.onnx_dtype, device=self.device)
            has_input_masks = torch.zeros(num_labels, dtype=self.onnx_dtype, device=self.device)
        else:
            input_masks = mask_input[img_idx].unsqueeze(0).repeat(num_labels, 1, 1, 1)
            has_input_masks = torch.ones(num_labels, dtype=self.onnx_dtype, device=self.device)

        feed_dict = {
            "image_embeddings": image_embeddings.contiguous().to(dtype=self.onnx_dtype).to(self.device),
            "image_features_0": image_features_0.contiguous().to(dtype=self.onnx_dtype).to(self.device),
            "image_features_1": image_features_1.contiguous().to(dtype=self.onnx_dtype).to(self.device),
            "point_coords": concat_points[0].to(dtype=self.onnx_dtype).to(self.device),
            "point_labels": concat_points[1].to(dtype=torch.int32).to(self.device),
            "input_masks": input_masks.to(dtype=self.onnx_dtype).to(self.device),
            "has_input_masks": has_input_masks.to(dtype=self.onnx_dtype).to(self.device),
            "original_image_size": torch.tensor(self._orig_hw[img_idx], dtype=torch.int32, device=self.device),
        }

        for key, value in feed_dict.items():
            logger.debug(f"{key}: {value.shape}, {value.dtype}")
        logger.debug(f"decoder onnx: {self.decoder_session.ort_session._model_path}")

        ort_outputs = decoder_session.infer(feed_dict)

        masks = ort_outputs["masks"]
        iou_predictions = ort_outputs["iou_predictions"]
        low_res_masks = ort_outputs["low_res_masks"]

        return torch.Tensor(masks), torch.Tensor(iou_predictions), torch.Tensor(low_res_masks)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\handlers\ntheory.py ===
"""
Handlers for keys related to number theory: prime, even, odd, etc.
"""

from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, NumberSymbol, Rational, int_valued)
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime

from sympy.multipledispatch import MDNotImplementedError

from ..predicates.ntheory import (PrimePredicate, CompositePredicate,
    EvenPredicate, OddPredicate)


# PrimePredicate

def _PrimePredicate_number(expr, assumptions):
    # helper method
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    if exact:
        return isprime(i)
    # when not exact, we won't give a True or False
    # since the number represents an approximate value

@PrimePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_prime
    if ret is None:
        raise MDNotImplementedError
    return ret

@PrimePredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(Mul)
def _(expr, assumptions):
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    for arg in expr.args:
        if not ask(Q.integer(arg), assumptions):
            return None
    for arg in expr.args:
        if arg.is_number and arg.is_composite:
            return False

@PrimePredicate.register(Pow)
def _(expr, assumptions):
    """
    Integer**Integer     -> !Prime
    """
    if expr.is_number:
        return _PrimePredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions) and \
            ask(Q.integer(expr.base), assumptions):
        prime_base = ask(Q.prime(expr.base), assumptions)
        if prime_base is False:
            return False
        is_exp_one = ask(Q.eq(expr.exp, 1), assumptions)
        if is_exp_one is False:
            return False
        if prime_base is True and is_exp_one is True:
            return True

@PrimePredicate.register(Integer)
def _(expr, assumptions):
    return isprime(expr)

@PrimePredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

@PrimePredicate.register(Float)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _PrimePredicate_number(expr, assumptions)

@PrimePredicate.register(NaN)
def _(expr, assumptions):
    return None


# CompositePredicate

@CompositePredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_composite
    if ret is None:
        raise MDNotImplementedError
    return ret

@CompositePredicate.register(Basic)
def _(expr, assumptions):
    _positive = ask(Q.positive(expr), assumptions)
    if _positive:
        _integer = ask(Q.integer(expr), assumptions)
        if _integer:
            _prime = ask(Q.prime(expr), assumptions)
            if _prime is None:
                return
            # Positive integer which is not prime is not
            # necessarily composite
            _is_one = ask(Q.eq(expr, 1), assumptions)
            if _is_one:
                return False
            if _is_one is None:
                return None
            return not _prime
        else:
            return _integer
    else:
        return _positive


# EvenPredicate

def _EvenPredicate_number(expr, assumptions):
    # helper method
    if isinstance(expr, (float, Float)):
        if int_valued(expr):
            return None
        return False
    try:
        i = int(expr.round())
    except TypeError:
        return False
    if not (expr - i).equals(0):
        return False
    return i % 2 == 0

@EvenPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_even
    if ret is None:
        raise MDNotImplementedError
    return ret

@EvenPredicate.register(Basic)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Mul)
def _(expr, assumptions):
    """
    Even * Integer    -> Even
    Even * Odd        -> Even
    Integer * Odd     -> ?
    Odd * Odd         -> Odd
    Even * Even       -> Even
    Integer * Integer -> Even if Integer + Integer = Odd
    otherwise         -> ?
    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    even, odd, irrational, acc = False, 0, False, 1
    for arg in expr.args:
        # check for all integers and at least one even
        if ask(Q.integer(arg), assumptions):
            if ask(Q.even(arg), assumptions):
                even = True
            elif ask(Q.odd(arg), assumptions):
                odd += 1
            elif not even and acc != 1:
                if ask(Q.odd(acc + arg), assumptions):
                    even = True
        elif ask(Q.irrational(arg), assumptions):
            # one irrational makes the result False
            # two makes it undefined
            if irrational:
                break
            irrational = True
        else:
            break
        acc = arg
    else:
        if irrational:
            return False
        if even:
            return True
        if odd == len(expr.args):
            return False

@EvenPredicate.register(Add)
def _(expr, assumptions):
    """
    Even + Odd  -> Odd
    Even + Even -> Even
    Odd  + Odd  -> Even

    """
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    _result = True
    for arg in expr.args:
        if ask(Q.even(arg), assumptions):
            pass
        elif ask(Q.odd(arg), assumptions):
            _result = not _result
        else:
            break
    else:
        return _result

@EvenPredicate.register(Pow)
def _(expr, assumptions):
    if expr.is_number:
        return _EvenPredicate_number(expr, assumptions)
    if ask(Q.integer(expr.exp), assumptions):
        if ask(Q.positive(expr.exp), assumptions):
            return ask(Q.even(expr.base), assumptions)
        elif ask(~Q.negative(expr.exp) & Q.odd(expr.base), assumptions):
            return False
        elif expr.base is S.NegativeOne:
            return False

@EvenPredicate.register(Integer)
def _(expr, assumptions):
    return not bool(expr.p & 1)

@EvenPredicate.register_many(Rational, Infinity, NegativeInfinity, ImaginaryUnit)
def _(expr, assumptions):
    return False

@EvenPredicate.register(NumberSymbol)
def _(expr, assumptions):
    return _EvenPredicate_number(expr, assumptions)

@EvenPredicate.register(Abs)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(re)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return ask(Q.even(expr.args[0]), assumptions)

@EvenPredicate.register(im)
def _(expr, assumptions):
    if ask(Q.real(expr.args[0]), assumptions):
        return True

@EvenPredicate.register(NaN)
def _(expr, assumptions):
    return None


# OddPredicate

@OddPredicate.register(Expr)
def _(expr, assumptions):
    ret = expr.is_odd
    if ret is None:
        raise MDNotImplementedError
    return ret

@OddPredicate.register(Basic)
def _(expr, assumptions):
    _integer = ask(Q.integer(expr), assumptions)
    if _integer:
        _even = ask(Q.even(expr), assumptions)
        if _even is None:
            return None
        return not _even
    return _integer

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\graph.py ===
from sympy.utilities.iterables import \
    flatten, connected_components, strongly_connected_components
from .exceptions import NonSquareMatrixError


def _connected_components(M):
    """Returns the list of connected vertices of the graph when
    a square matrix is viewed as a weighted graph.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])
    >>> A.connected_components()
    [[0, 3, 8], [1, 6, 7], [2, 4, 5]]

    Notes
    =====

    Even if any symbolic elements of the matrix can be indeterminate
    to be zero mathematically, this only takes the account of the
    structural aspect of the matrix, so they will considered to be
    nonzero.
    """
    if not M.is_square:
        raise NonSquareMatrixError

    V = range(M.rows)
    E = sorted(M.todok().keys())
    return connected_components((V, E))


def _strongly_connected_components(M):
    """Returns the list of strongly connected vertices of the graph when
    a square matrix is viewed as a weighted graph.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])
    >>> A.strongly_connected_components()
    [[0, 4, 6], [2, 3, 7], [1, 5, 8]]
    """
    if not M.is_square:
        raise NonSquareMatrixError

    # RepMatrix uses the more efficient DomainMatrix.scc() method
    rep = getattr(M, '_rep', None)
    if rep is not None:
        return rep.scc()

    V = range(M.rows)
    E = sorted(M.todok().keys())
    return strongly_connected_components((V, E))


def _connected_components_decomposition(M):
    """Decomposes a square matrix into block diagonal form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Returns
    =======

    P, B : PermutationMatrix, BlockDiagMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block diagonal matrix of
        the result of the permutation.

        If you would like to get the diagonal blocks from the
        BlockDiagMatrix, see
        :meth:`~sympy.matrices.expressions.blockmatrix.BlockDiagMatrix.get_diag_blocks`.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [66, 0, 0, 68, 0, 0, 0, 0, 67],
    ...     [0, 55, 0, 0, 0, 0, 54, 53, 0],
    ...     [0, 0, 0, 0, 1, 2, 0, 0, 0],
    ...     [86, 0, 0, 88, 0, 0, 0, 0, 87],
    ...     [0, 0, 10, 0, 11, 12, 0, 0, 0],
    ...     [0, 0, 20, 0, 21, 22, 0, 0, 0],
    ...     [0, 45, 0, 0, 0, 0, 44, 43, 0],
    ...     [0, 35, 0, 0, 0, 0, 34, 33, 0],
    ...     [76, 0, 0, 78, 0, 0, 0, 0, 77]])

    >>> P, B = A.connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((1 3)(2 8 5 7 4 6))
    >>> pprint(B)
    [[66  68  67]                            ]
    [[          ]                            ]
    [[86  88  87]       0             0      ]
    [[          ]                            ]
    [[76  78  77]                            ]
    [                                        ]
    [              [55  54  53]              ]
    [              [          ]              ]
    [     0        [45  44  43]       0      ]
    [              [          ]              ]
    [              [35  34  33]              ]
    [                                        ]
    [                            [0   1   2 ]]
    [                            [          ]]
    [     0             0        [10  11  12]]
    [                            [          ]]
    [                            [20  21  22]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T*B*P == A
    True

    Notes
    =====

    This problem corresponds to the finding of the connected components
    of a graph, when a matrix is viewed as a weighted graph.
    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix
    from sympy.matrices.expressions.permutation import PermutationMatrix

    iblocks = M.connected_components()

    p = Permutation(flatten(iblocks))
    P = PermutationMatrix(p)

    blocks = []
    for b in iblocks:
        blocks.append(M[b, b])
    B = BlockDiagMatrix(*blocks)
    return P, B


def _strongly_connected_components_decomposition(M, lower=True):
    """Decomposes a square matrix into block triangular form only
    using the permutations.

    Explanation
    ===========

    The decomposition is in a form of $A = P^{-1} B P$ where $P$ is a
    permutation matrix and $B$ is a block diagonal matrix.

    Parameters
    ==========

    lower : bool
        Makes $B$ lower block triangular when ``True``.
        Otherwise, makes $B$ upper block triangular.

    Returns
    =======

    P, B : PermutationMatrix, BlockMatrix
        *P* is a permutation matrix for the similarity transform
        as in the explanation. And *B* is the block triangular matrix of
        the result of the permutation.

    Examples
    ========

    >>> from sympy import Matrix, pprint
    >>> A = Matrix([
    ...     [44, 0, 0, 0, 43, 0, 45, 0, 0],
    ...     [0, 66, 62, 61, 0, 68, 0, 60, 67],
    ...     [0, 0, 22, 21, 0, 0, 0, 20, 0],
    ...     [0, 0, 12, 11, 0, 0, 0, 10, 0],
    ...     [34, 0, 0, 0, 33, 0, 35, 0, 0],
    ...     [0, 86, 82, 81, 0, 88, 0, 80, 87],
    ...     [54, 0, 0, 0, 53, 0, 55, 0, 0],
    ...     [0, 0, 2, 1, 0, 0, 0, 0, 0],
    ...     [0, 76, 72, 71, 0, 78, 0, 70, 77]])

    A lower block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition()
    >>> pprint(P)
    PermutationMatrix((8)(1 4 3 2 6)(5 7))
    >>> pprint(B)
    [[44  43  45]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[34  33  35]   [0  0  0]     [0  0  0]  ]
    [[          ]   [       ]     [       ]  ]
    [[54  53  55]   [0  0  0]     [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [62  61  60]  [66  68  67]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [82  81  80]  [86  88  87]]
    [ [       ]    [          ]  [          ]]
    [ [0  0  0]    [72  71  70]  [76  78  77]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True

    An upper block triangular decomposition:

    >>> P, B = A.strongly_connected_components_decomposition(lower=False)
    >>> pprint(P)
    PermutationMatrix((0 1 5 7 4 3 2 8 6))
    >>> pprint(B)
    [[66  68  67]  [62  61  60]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[86  88  87]  [82  81  80]   [0  0  0]  ]
    [[          ]  [          ]   [       ]  ]
    [[76  78  77]  [72  71  70]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]    [22  21  20]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [12  11  10]   [0  0  0]  ]
    [ [       ]    [          ]   [       ]  ]
    [ [0  0  0]    [2   1   0 ]   [0  0  0]  ]
    [                                        ]
    [ [0  0  0]     [0  0  0]    [44  43  45]]
    [ [       ]     [       ]    [          ]]
    [ [0  0  0]     [0  0  0]    [34  33  35]]
    [ [       ]     [       ]    [          ]]
    [ [0  0  0]     [0  0  0]    [54  53  55]]

    >>> P = P.as_explicit()
    >>> B = B.as_explicit()
    >>> P.T * B * P == A
    True
    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    from sympy.matrices.expressions.permutation import PermutationMatrix

    iblocks = M.strongly_connected_components()
    if not lower:
        iblocks = list(reversed(iblocks))

    p = Permutation(flatten(iblocks))
    P = PermutationMatrix(p)

    rows = []
    for a in iblocks:
        cols = []
        for b in iblocks:
            cols.append(M[a, b])
        rows.append(cols)
    B = BlockMatrix(rows)
    return P, B

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\sym_expr.py ===
from sympy.printing import pycode, ccode, fcode
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

lfortran = import_module('lfortran')
cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

if lfortran:
    from sympy.parsing.fortran.fortran_parser import src_to_sympy
if cin:
    from sympy.parsing.c.c_parser import parse_c

@doctest_depends_on(modules=['lfortran', 'clang.cindex'])
class SymPyExpression:  # type: ignore
    """Class to store and handle SymPy expressions

    This class will hold SymPy Expressions and handle the API for the
    conversion to and from different languages.

    It works with the C and the Fortran Parser to generate SymPy expressions
    which are stored here and which can be converted to multiple language's
    source code.

    Notes
    =====

    The module and its API are currently under development and experimental
    and can be changed during development.

    The Fortran parser does not support numeric assignments, so all the
    variables have been Initialized to zero.

    The module also depends on external dependencies:

    - LFortran which is required to use the Fortran parser
    - Clang which is required for the C parser

    Examples
    ========

    Example of parsing C code:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... int a,b;
    ... float c = 2, d =4;
    ... '''
    >>> a = SymPyExpression(src, 'c')
    >>> a.return_expr()
    [Declaration(Variable(a, type=intc)),
    Declaration(Variable(b, type=intc)),
    Declaration(Variable(c, type=float32, value=2.0)),
    Declaration(Variable(d, type=float32, value=4.0))]

    An example of variable definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src2 = '''
    ... integer :: a, b, c, d
    ... real :: p, q, r, s
    ... '''
    >>> p = SymPyExpression()
    >>> p.convert_to_expr(src2, 'f')
    >>> p.convert_to_c()
    ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0']

    An example of Assignment:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src3 = '''
    ... integer :: a, b, c, d, e
    ... d = a + b - c
    ... e = b * d + c * e / a
    ... '''
    >>> p = SymPyExpression(src3, 'f')
    >>> p.convert_to_python()
    ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'e = 0', 'd = a + b - c', 'e = b*d + c*e/a']

    An example of function definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... integer function f(a,b)
    ... integer, intent(in) :: a, b
    ... integer :: r
    ... end function
    ... '''
    >>> a = SymPyExpression(src, 'f')
    >>> a.convert_to_python()
    ['def f(a, b):\\n   f = 0\\n    r = 0\\n    return f']

    """

    def __init__(self, source_code = None, mode = None):
        """Constructor for SymPyExpression class"""
        super().__init__()
        if not(mode or source_code):
            self._expr = []
        elif mode:
            if source_code:
                if mode.lower() == 'f':
                    if lfortran:
                        self._expr = src_to_sympy(source_code)
                    else:
                        raise ImportError("LFortran is not installed, cannot parse Fortran code")
                elif mode.lower() == 'c':
                    if cin:
                        self._expr = parse_c(source_code)
                    else:
                        raise ImportError("Clang is not installed, cannot parse C code")
                else:
                    raise NotImplementedError(
                        'Parser for specified language is not implemented'
                    )
            else:
                raise ValueError('Source code not present')
        else:
            raise ValueError('Please specify a mode for conversion')

    def convert_to_expr(self, src_code, mode):
        """Converts the given source code to SymPy Expressions

        Attributes
        ==========

        src_code : String
            the source code or filename of the source code that is to be
            converted

        mode: String
            the mode to determine which parser is to be used according to
            the language of the source code
            f or F for Fortran
            c or C for C/C++

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b) result(r)
        ... integer, intent(in) :: a, b
        ... integer :: x
        ... r = a + b -x
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(r, type=integer, value=0)),
        Declaration(Variable(x, type=integer, value=0)),
        Assignment(Variable(r), a + b - x),
        Return(Variable(r))
        ))]




        """
        if mode.lower() == 'f':
            if lfortran:
                self._expr = src_to_sympy(src_code)
            else:
                raise ImportError("LFortran is not installed, cannot parse Fortran code")
        elif mode.lower() == 'c':
            if cin:
                self._expr = parse_c(src_code)
            else:
                raise ImportError("Clang is not installed, cannot parse C code")
        else:
            raise NotImplementedError(
                "Parser for specified language has not been implemented"
            )

    def convert_to_python(self):
        """Returns a list with Python code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_python()
        ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'p = 0.0', 'q = 0.0', 'r = 0.0', 's = 0.0', 'c = a/b', 'd = c/a', 's = p/q', 'r = q/p']

        """
        self._pycode = []
        for iter in self._expr:
            self._pycode.append(pycode(iter))
        return self._pycode

    def convert_to_c(self):
        """Returns a list with the c source code for the SymPy expressions


        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src2, 'f')
        >>> p.convert_to_c()
        ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0', 'c = a/b;', 'd = c/a;', 's = p/q;', 'r = q/p;']

        """
        self._ccode = []
        for iter in self._expr:
            self._ccode.append(ccode(iter))
        return self._ccode

    def convert_to_fortran(self):
        """Returns a list with the fortran source code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_fortran()
        ['      integer*4 a', '      integer*4 b', '      integer*4 c', '      integer*4 d', '      real*8 p', '      real*8 q', '      real*8 r', '      real*8 s', '      c = a/b', '      d = c/a', '      s = p/q', '      r = q/p']

        """
        self._fcode = []
        for iter in self._expr:
            self._fcode.append(fcode(iter))
        return self._fcode

    def return_expr(self):
        """Returns the expression list

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b)
        ... integer, intent(in) :: a, b
        ... integer :: r
        ... r = a+b
        ... f = r
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(f, type=integer, value=0)),
        Declaration(Variable(r, type=integer, value=0)),
        Assignment(Variable(f), Variable(r)),
        Return(Variable(f))
        ))]

        """
        return self._expr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\operatorset.py ===
""" A module for mapping operators to their corresponding eigenstates
and vice versa

It contains a global dictionary with eigenstate-operator pairings.
If a new state-operator pair is created, this dictionary should be
updated as well.

It also contains functions operators_to_state and state_to_operators
for mapping between the two. These can handle both classes and
instances of operators and states. See the individual function
descriptions for details.

TODO List:
- Update the dictionary with a complete list of state-operator pairs
"""

from sympy.physics.quantum.cartesian import (XOp, YOp, ZOp, XKet, PxOp, PxKet,
                                             PositionKet3D)
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import StateBase, BraBase, Ket
from sympy.physics.quantum.spin import (JxOp, JyOp, JzOp, J2Op, JxKet, JyKet,
                                        JzKet)

__all__ = [
    'operators_to_state',
    'state_to_operators'
]

#state_mapping stores the mappings between states and their associated
#operators or tuples of operators. This should be updated when new
#classes are written! Entries are of the form PxKet : PxOp or
#something like 3DKet : (ROp, ThetaOp, PhiOp)

#frozenset is used so that the reverse mapping can be made
#(regular sets are not hashable because they are mutable
state_mapping = { JxKet: frozenset((J2Op, JxOp)),
                  JyKet: frozenset((J2Op, JyOp)),
                  JzKet: frozenset((J2Op, JzOp)),
                  Ket: Operator,
                  PositionKet3D: frozenset((XOp, YOp, ZOp)),
                  PxKet: PxOp,
                  XKet: XOp }

op_mapping = {v: k for k, v in state_mapping.items()}


def operators_to_state(operators, **options):
    """ Returns the eigenstate of the given operator or set of operators

    A global function for mapping operator classes to their associated
    states. It takes either an Operator or a set of operators and
    returns the state associated with these.

    This function can handle both instances of a given operator or
    just the class itself (i.e. both XOp() and XOp)

    There are multiple use cases to consider:

    1) A class or set of classes is passed: First, we try to
    instantiate default instances for these operators. If this fails,
    then the class is simply returned. If we succeed in instantiating
    default instances, then we try to call state._operators_to_state
    on the operator instances. If this fails, the class is returned.
    Otherwise, the instance returned by _operators_to_state is returned.

    2) An instance or set of instances is passed: In this case,
    state._operators_to_state is called on the instances passed. If
    this fails, a state class is returned. If the method returns an
    instance, that instance is returned.

    In both cases, if the operator class or set does not exist in the
    state_mapping dictionary, None is returned.

    Parameters
    ==========

    arg: Operator or set
         The class or instance of the operator or set of operators
         to be mapped to a state

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp
    >>> from sympy.physics.quantum.operatorset import operators_to_state
    >>> from sympy.physics.quantum.operator import Operator
    >>> operators_to_state(XOp)
    |x>
    >>> operators_to_state(XOp())
    |x>
    >>> operators_to_state(PxOp)
    |px>
    >>> operators_to_state(PxOp())
    |px>
    >>> operators_to_state(Operator)
    |psi>
    >>> operators_to_state(Operator())
    |psi>
    """

    if not (isinstance(operators, (Operator, set)) or issubclass(operators, Operator)):
        raise NotImplementedError("Argument is not an Operator or a set!")

    if isinstance(operators, set):
        for s in operators:
            if not (isinstance(s, Operator)
                   or issubclass(s, Operator)):
                raise NotImplementedError("Set is not all Operators!")

        ops = frozenset(operators)

        if ops in op_mapping:  # ops is a list of classes in this case
            #Try to get an object from default instances of the
            #operators...if this fails, return the class
            try:
                op_instances = [op() for op in ops]
                ret = _get_state(op_mapping[ops], set(op_instances), **options)
            except NotImplementedError:
                ret = op_mapping[ops]

            return ret
        else:
            tmp = [type(o) for o in ops]
            classes = frozenset(tmp)

            if classes in op_mapping:
                ret = _get_state(op_mapping[classes], ops, **options)
            else:
                ret = None

            return ret
    else:
        if operators in op_mapping:
            try:
                op_instance = operators()
                ret = _get_state(op_mapping[operators], op_instance, **options)
            except NotImplementedError:
                ret = op_mapping[operators]

            return ret
        elif type(operators) in op_mapping:
            return _get_state(op_mapping[type(operators)], operators, **options)
        else:
            return None


def state_to_operators(state, **options):
    """ Returns the operator or set of operators corresponding to the
    given eigenstate

    A global function for mapping state classes to their associated
    operators or sets of operators. It takes either a state class
    or instance.

    This function can handle both instances of a given state or just
    the class itself (i.e. both XKet() and XKet)

    There are multiple use cases to consider:

    1) A state class is passed: In this case, we first try
    instantiating a default instance of the class. If this succeeds,
    then we try to call state._state_to_operators on that instance.
    If the creation of the default instance or if the calling of
    _state_to_operators fails, then either an operator class or set of
    operator classes is returned. Otherwise, the appropriate
    operator instances are returned.

    2) A state instance is returned: Here, state._state_to_operators
    is called for the instance. If this fails, then a class or set of
    operator classes is returned. Otherwise, the instances are returned.

    In either case, if the state's class does not exist in
    state_mapping, None is returned.

    Parameters
    ==========

    arg: StateBase class or instance (or subclasses)
         The class or instance of the state to be mapped to an
         operator or set of operators

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XKet, PxKet, XBra, PxBra
    >>> from sympy.physics.quantum.operatorset import state_to_operators
    >>> from sympy.physics.quantum.state import Ket, Bra
    >>> state_to_operators(XKet)
    X
    >>> state_to_operators(XKet())
    X
    >>> state_to_operators(PxKet)
    Px
    >>> state_to_operators(PxKet())
    Px
    >>> state_to_operators(PxBra)
    Px
    >>> state_to_operators(XBra)
    X
    >>> state_to_operators(Ket)
    O
    >>> state_to_operators(Bra)
    O
    """

    if not (isinstance(state, StateBase) or issubclass(state, StateBase)):
        raise NotImplementedError("Argument is not a state!")

    if state in state_mapping:  # state is a class
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state]), **options)
        except (NotImplementedError, TypeError):
            ret = state_mapping[state]
    elif type(state) in state_mapping:
        ret = _get_ops(state,
                       _make_set(state_mapping[type(state)]), **options)
    elif isinstance(state, BraBase) and state.dual_class() in state_mapping:
        ret = _get_ops(state,
                       _make_set(state_mapping[state.dual_class()]))
    elif issubclass(state, BraBase) and state.dual_class() in state_mapping:
        state_inst = _make_default(state)
        try:
            ret = _get_ops(state_inst,
                           _make_set(state_mapping[state.dual_class()]))
        except (NotImplementedError, TypeError):
            ret = state_mapping[state.dual_class()]
    else:
        ret = None

    return _make_set(ret)


def _make_default(expr):
    # XXX: Catching TypeError like this is a bad way of distinguishing between
    # classes and instances. The logic using this function should be rewritten
    # somehow.
    try:
        ret = expr()
    except TypeError:
        ret = expr

    return ret


def _get_state(state_class, ops, **options):
    # Try to get a state instance from the operator INSTANCES.
    # If this fails, get the class
    try:
        ret = state_class._operators_to_state(ops, **options)
    except NotImplementedError:
        ret = _make_default(state_class)

    return ret


def _get_ops(state_inst, op_classes, **options):
    # Try to get operator instances from the state INSTANCE.
    # If this fails, just return the classes
    try:
        ret = state_inst._state_to_operators(op_classes, **options)
    except NotImplementedError:
        if isinstance(op_classes, (set, tuple, frozenset)):
            ret = tuple(_make_default(x) for x in op_classes)
        else:
            ret = _make_default(op_classes)

    if isinstance(ret, set) and len(ret) == 1:
        return ret[0]

    return ret


def _make_set(ops):
    if isinstance(ops, (tuple, list, frozenset)):
        return set(ops)
    else:
        return ops

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_path.py ===
from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING, Union

import pytest

import trio
from trio._file_io import AsyncIOWrapper

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@pytest.fixture
def path(tmp_path: pathlib.Path) -> trio.Path:
    return trio.Path(tmp_path / "test")


def method_pair(
    path: str,
    method_name: str,
) -> tuple[Callable[[], object], Callable[[], Awaitable[object]]]:
    sync_path = pathlib.Path(path)
    async_path = trio.Path(path)
    return getattr(sync_path, method_name), getattr(async_path, method_name)


@pytest.mark.skipif(os.name == "nt", reason="OS is not posix")
def test_instantiate_posix() -> None:
    assert isinstance(trio.Path(), trio.PosixPath)


@pytest.mark.skipif(os.name != "nt", reason="OS is not Windows")
def test_instantiate_windows() -> None:
    assert isinstance(trio.Path(), trio.WindowsPath)


async def test_open_is_async_context_manager(path: trio.Path) -> None:
    async with await path.open("w") as f:
        assert isinstance(f, AsyncIOWrapper)

    assert f.closed


def test_magic() -> None:
    path = trio.Path("test")

    assert str(path) == "test"
    assert bytes(path) == b"test"


EitherPathType = Union[type[trio.Path], type[pathlib.Path]]
PathOrStrType = Union[EitherPathType, type[str]]
cls_pairs: list[tuple[EitherPathType, EitherPathType]] = [
    (trio.Path, pathlib.Path),
    (pathlib.Path, trio.Path),
    (trio.Path, trio.Path),
]


@pytest.mark.parametrize(("cls_a", "cls_b"), cls_pairs)
def test_cmp_magic(cls_a: EitherPathType, cls_b: EitherPathType) -> None:
    a, b = cls_a(""), cls_b("")
    assert a == b
    assert not a != b  # noqa: SIM202  # negate-not-equal-op

    a, b = cls_a("a"), cls_b("b")
    assert a < b
    assert b > a

    # this is intentionally testing equivalence with none, due to the
    # other=sentinel logic in _forward_magic
    assert not a == None  # noqa
    assert not b == None  # noqa


# upstream python3.8 bug: we should also test (pathlib.Path, trio.Path), but
# __*div__ does not properly raise NotImplementedError like the other comparison
# magic, so trio.Path's implementation does not get dispatched
cls_pairs_str: list[tuple[PathOrStrType, PathOrStrType]] = [
    (trio.Path, pathlib.Path),
    (trio.Path, trio.Path),
    (trio.Path, str),
    (str, trio.Path),
]


@pytest.mark.parametrize(("cls_a", "cls_b"), cls_pairs_str)
def test_div_magic(cls_a: PathOrStrType, cls_b: PathOrStrType) -> None:
    a, b = cls_a("a"), cls_b("b")

    result = a / b  # type: ignore[operator]
    # Type checkers think str / str could happen. Check each combo manually in type_tests/.
    assert isinstance(result, trio.Path)
    assert str(result) == os.path.join("a", "b")


@pytest.mark.parametrize(
    ("cls_a", "cls_b"),
    [(trio.Path, pathlib.Path), (trio.Path, trio.Path)],
)
@pytest.mark.parametrize("path", ["foo", "foo/bar/baz", "./foo"])
def test_hash_magic(
    cls_a: EitherPathType,
    cls_b: EitherPathType,
    path: str,
) -> None:
    a, b = cls_a(path), cls_b(path)
    assert hash(a) == hash(b)


def test_forwarded_properties(path: trio.Path) -> None:
    # use `name` as a representative of forwarded properties

    assert "name" in dir(path)
    assert path.name == "test"


def test_async_method_signature(path: trio.Path) -> None:
    # use `resolve` as a representative of wrapped methods

    assert path.resolve.__name__ == "resolve"
    assert path.resolve.__qualname__ == "Path.resolve"

    assert path.resolve.__doc__ is not None
    assert path.resolve.__qualname__ in path.resolve.__doc__


@pytest.mark.parametrize("method_name", ["is_dir", "is_file"])
async def test_compare_async_stat_methods(method_name: str) -> None:
    method, async_method = method_pair(".", method_name)

    result = method()
    async_result = await async_method()

    assert result == async_result


def test_invalid_name_not_wrapped(path: trio.Path) -> None:
    with pytest.raises(AttributeError):
        getattr(path, "invalid_fake_attr")  # noqa: B009  # "get-attr-with-constant"


@pytest.mark.parametrize("method_name", ["absolute", "resolve"])
async def test_async_methods_rewrap(method_name: str) -> None:
    method, async_method = method_pair(".", method_name)

    result = method()
    async_result = await async_method()

    assert isinstance(async_result, trio.Path)
    assert str(result) == str(async_result)


def test_forward_methods_rewrap(path: trio.Path, tmp_path: pathlib.Path) -> None:
    with_name = path.with_name("foo")
    with_suffix = path.with_suffix(".py")

    assert isinstance(with_name, trio.Path)
    assert with_name == tmp_path / "foo"
    assert isinstance(with_suffix, trio.Path)
    assert with_suffix == tmp_path / "test.py"


def test_forward_properties_rewrap(path: trio.Path) -> None:
    assert isinstance(path.parent, trio.Path)


def test_forward_methods_without_rewrap(path: trio.Path) -> None:
    assert "totally-unique-path" in str(path.joinpath("totally-unique-path"))


def test_repr() -> None:
    path = trio.Path(".")

    assert repr(path) == "trio.Path('.')"


@pytest.mark.parametrize("meth", [trio.Path.__init__, trio.Path.joinpath])
async def test_path_wraps_path(
    path: trio.Path,
    meth: Callable[[trio.Path, trio.Path], object],
) -> None:
    wrapped = await path.absolute()
    result = meth(path, wrapped)
    if result is None:
        result = path

    assert wrapped == result


def test_path_nonpath() -> None:
    with pytest.raises(TypeError):
        trio.Path(1)  # type: ignore


async def test_open_file_can_open_path(path: trio.Path) -> None:
    async with await trio.open_file(path, "w") as f:
        assert f.name == os.fspath(path)


async def test_globmethods(path: trio.Path) -> None:
    # Populate a directory tree
    await path.mkdir()
    await (path / "foo").mkdir()
    await (path / "foo" / "_bar.txt").write_bytes(b"")
    await (path / "bar.txt").write_bytes(b"")
    await (path / "bar.dat").write_bytes(b"")

    # Path.glob
    for _pattern, _results in {
        "*.txt": {"bar.txt"},
        "**/*.txt": {"_bar.txt", "bar.txt"},
    }.items():
        entries = set()
        for entry in await path.glob(_pattern):
            assert isinstance(entry, trio.Path)
            entries.add(entry.name)

        assert entries == _results

    # Path.rglob
    entries = set()
    for entry in await path.rglob("*.txt"):
        assert isinstance(entry, trio.Path)
        entries.add(entry.name)

    assert entries == {"_bar.txt", "bar.txt"}


async def test_as_uri(path: trio.Path) -> None:
    path = await path.parent.resolve()

    assert path.as_uri().startswith("file:///")


async def test_iterdir(path: trio.Path) -> None:
    # Populate a directory
    await path.mkdir()
    await (path / "foo").mkdir()
    await (path / "bar.txt").write_bytes(b"")

    entries = set()
    for entry in await path.iterdir():
        assert isinstance(entry, trio.Path)
        entries.add(entry.name)

    assert entries == {"bar.txt", "foo"}


async def test_classmethods() -> None:
    assert isinstance(await trio.Path.home(), trio.Path)

    # pathlib.Path has only two classmethods
    assert str(await trio.Path.home()) == os.path.expanduser("~")
    assert str(await trio.Path.cwd()) == os.getcwd()

    # Wrapped method has docstring
    assert trio.Path.home.__doc__


@pytest.mark.parametrize(
    "wrapper",
    [
        trio._path._wraps_async,
        trio._path._wrap_method,
        trio._path._wrap_method_path,
        trio._path._wrap_method_path_iterable,
    ],
)
def test_wrapping_without_docstrings(
    wrapper: Callable[[Callable[[], None]], Callable[[], None]],
) -> None:
    @wrapper
    def func_without_docstring() -> None: ...  # pragma: no cover

    assert func_without_docstring.__doc__ is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\style.py ===
from __future__ import annotations

from collections.abc import (
    Collection,
    Iterator,
)
import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import matplotlib as mpl
import matplotlib.colors
import numpy as np

from pandas._typing import MatplotlibColor as Color
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


def get_standard_colors(
    num_colors: int,
    colormap: Colormap | None = None,
    color_type: str = "default",
    color: dict[str, Color] | Color | Collection[Color] | None = None,
):
    """
    Get standard colors based on `colormap`, `color_type` or `color` inputs.

    Parameters
    ----------
    num_colors : int
        Minimum number of colors to be returned.
        Ignored if `color` is a dictionary.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap` are not None.
    color : dict or str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a dictionary, or a single color (single color string,
        or sequence of floats representing a single color),
        or a sequence of colors.

    Returns
    -------
    dict or list
        Standard colors. Can either be a mapping if `color` was a dictionary,
        or a list of colors with a length of `num_colors` or more.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if isinstance(color, dict):
        return color

    colors = _derive_colors(
        color=color,
        colormap=colormap,
        color_type=color_type,
        num_colors=num_colors,
    )

    return list(_cycle_colors(colors, num_colors=num_colors))


def _derive_colors(
    *,
    color: Color | Collection[Color] | None,
    colormap: str | Colormap | None,
    color_type: str,
    num_colors: int,
) -> list[Color]:
    """
    Derive colors from either `colormap`, `color_type` or `color` inputs.

    Get a list of colors either from `colormap`, or from `color`,
    or from `color_type` (if both `colormap` and `color` are None).

    Parameters
    ----------
    color : str or sequence, optional
        Color(s) to be used for deriving sequence of colors.
        Can be either be a single color (single color string, or sequence of floats
        representing a single color), or a sequence of colors.
    colormap : :py:class:`matplotlib.colors.Colormap`, optional
        Matplotlib colormap.
        When provided, the resulting colors will be derived from the colormap.
    color_type : {"default", "random"}, optional
        Type of colors to derive. Used if provided `color` and `colormap` are None.
        Ignored if either `color` or `colormap`` are not None.
    num_colors : int
        Number of colors to be extracted.

    Returns
    -------
    list
        List of colors extracted.

    Warns
    -----
    UserWarning
        If both `colormap` and `color` are provided.
        Parameter `color` will override.
    """
    if color is None and colormap is not None:
        return _get_colors_from_colormap(colormap, num_colors=num_colors)
    elif color is not None:
        if colormap is not None:
            warnings.warn(
                "'color' and 'colormap' cannot be used simultaneously. Using 'color'",
                stacklevel=find_stack_level(),
            )
        return _get_colors_from_color(color)
    else:
        return _get_colors_from_color_type(color_type, num_colors=num_colors)


def _cycle_colors(colors: list[Color], num_colors: int) -> Iterator[Color]:
    """Cycle colors until achieving max of `num_colors` or length of `colors`.

    Extra colors will be ignored by matplotlib if there are more colors
    than needed and nothing needs to be done here.
    """
    max_colors = max(num_colors, len(colors))
    yield from itertools.islice(itertools.cycle(colors), max_colors)


def _get_colors_from_colormap(
    colormap: str | Colormap,
    num_colors: int,
) -> list[Color]:
    """Get colors from colormap."""
    cmap = _get_cmap_instance(colormap)
    return [cmap(num) for num in np.linspace(0, 1, num=num_colors)]


def _get_cmap_instance(colormap: str | Colormap) -> Colormap:
    """Get instance of matplotlib colormap."""
    if isinstance(colormap, str):
        cmap = colormap
        colormap = mpl.colormaps[colormap]
        if colormap is None:
            raise ValueError(f"Colormap {cmap} is not recognized")
    return colormap


def _get_colors_from_color(
    color: Color | Collection[Color],
) -> list[Color]:
    """Get colors from user input color."""
    if len(color) == 0:
        raise ValueError(f"Invalid color argument: {color}")

    if _is_single_color(color):
        color = cast(Color, color)
        return [color]

    color = cast(Collection[Color], color)
    return list(_gen_list_of_colors_from_iterable(color))


def _is_single_color(color: Color | Collection[Color]) -> bool:
    """Check if `color` is a single color, not a sequence of colors.

    Single color is of these kinds:
        - Named color "red", "C0", "firebrick"
        - Alias "g"
        - Sequence of floats, such as (0.1, 0.2, 0.3) or (0.1, 0.2, 0.3, 0.4).

    See Also
    --------
    _is_single_string_color
    """
    if isinstance(color, str) and _is_single_string_color(color):
        # GH #36972
        return True

    if _is_floats_color(color):
        return True

    return False


def _gen_list_of_colors_from_iterable(color: Collection[Color]) -> Iterator[Color]:
    """
    Yield colors from string of several letters or from collection of colors.
    """
    for x in color:
        if _is_single_color(x):
            yield x
        else:
            raise ValueError(f"Invalid color {x}")


def _is_floats_color(color: Color | Collection[Color]) -> bool:
    """Check if color comprises a sequence of floats representing color."""
    return bool(
        is_list_like(color)
        and (len(color) == 3 or len(color) == 4)
        and all(isinstance(x, (int, float)) for x in color)
    )


def _get_colors_from_color_type(color_type: str, num_colors: int) -> list[Color]:
    """Get colors from user input color type."""
    if color_type == "default":
        return _get_default_colors(num_colors)
    elif color_type == "random":
        return _get_random_colors(num_colors)
    else:
        raise ValueError("color_type must be either 'default' or 'random'")


def _get_default_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of default colors from matplotlib rc params."""
    import matplotlib.pyplot as plt

    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
    return colors[0:num_colors]


def _get_random_colors(num_colors: int) -> list[Color]:
    """Get `num_colors` of random colors."""
    return [_random_color(num) for num in range(num_colors)]


def _random_color(column: int) -> list[float]:
    """Get a random color represented as a list of length 3"""
    # GH17525 use common._random_state to avoid resetting the seed
    rs = com.random_state(column)
    return rs.rand(3).tolist()


def _is_single_string_color(color: Color) -> bool:
    """Check if `color` is a single string color.

    Examples of single string colors:
        - 'r'
        - 'g'
        - 'red'
        - 'green'
        - 'C3'
        - 'firebrick'

    Parameters
    ----------
    color : Color
        Color string or sequence of floats.

    Returns
    -------
    bool
        True if `color` looks like a valid color.
        False otherwise.
    """
    conv = matplotlib.colors.ColorConverter()
    try:
        # error: Argument 1 to "to_rgba" of "ColorConverter" has incompatible type
        # "str | Sequence[float]"; expected "tuple[float, float, float] | ..."
        conv.to_rgba(color)  # type: ignore[arg-type]
    except ValueError:
        return False
    else:
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\expressiondomain.py ===
"""Implementation of :class:`ExpressionDomain` class. """


from sympy.core import sympify, SympifyError
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyutils import PicklableWithSlots
from sympy.utilities import public

eflags = {"deep": False, "mul": True, "power_exp": False, "power_base": False,
              "basic": False, "multinomial": False, "log": False}

@public
class ExpressionDomain(Field, CharacteristicZero, SimpleDomain):
    """A class for arbitrary expressions. """

    is_SymbolicDomain = is_EX = True

    class Expression(DomainElement, PicklableWithSlots):
        """An arbitrary expression. """

        __slots__ = ('ex',)

        def __init__(self, ex):
            if not isinstance(ex, self.__class__):
                self.ex = sympify(ex)
            else:
                self.ex = ex.ex

        def __repr__(f):
            return 'EX(%s)' % repr(f.ex)

        def __str__(f):
            return 'EX(%s)' % str(f.ex)

        def __hash__(self):
            return hash((self.__class__.__name__, self.ex))

        def parent(self):
            return EX

        def as_expr(f):
            return f.ex

        def numer(f):
            return f.__class__(f.ex.as_numer_denom()[0])

        def denom(f):
            return f.__class__(f.ex.as_numer_denom()[1])

        def simplify(f, ex):
            return f.__class__(ex.cancel().expand(**eflags))

        def __abs__(f):
            return f.__class__(abs(f.ex))

        def __neg__(f):
            return f.__class__(-f.ex)

        def _to_ex(f, g):
            try:
                return f.__class__(g)
            except SympifyError:
                return None

        def __lt__(f, g):
            return f.ex.sort_key() < g.ex.sort_key()

        def __add__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return g
            else:
                return f.simplify(f.ex + g.ex)

        def __radd__(f, g):
            return f.simplify(f.__class__(g).ex + f.ex)

        def __sub__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented
            elif g == EX.zero:
                return f
            elif f == EX.zero:
                return -g
            else:
                return f.simplify(f.ex - g.ex)

        def __rsub__(f, g):
            return f.simplify(f.__class__(g).ex - f.ex)

        def __mul__(f, g):
            g = f._to_ex(g)

            if g is None:
                return NotImplemented

            if EX.zero in (f, g):
                return EX.zero
            elif f.ex.is_Number and g.ex.is_Number:
                return f.__class__(f.ex*g.ex)

            return f.simplify(f.ex*g.ex)

        def __rmul__(f, g):
            return f.simplify(f.__class__(g).ex*f.ex)

        def __pow__(f, n):
            n = f._to_ex(n)

            if n is not None:
                return f.simplify(f.ex**n.ex)
            else:
                return NotImplemented

        def __truediv__(f, g):
            g = f._to_ex(g)

            if g is not None:
                return f.simplify(f.ex/g.ex)
            else:
                return NotImplemented

        def __rtruediv__(f, g):
            return f.simplify(f.__class__(g).ex/f.ex)

        def __eq__(f, g):
            return f.ex == f.__class__(g).ex

        def __ne__(f, g):
            return not f == g

        def __bool__(f):
            return not f.ex.is_zero

        def gcd(f, g):
            from sympy.polys import gcd
            return f.__class__(gcd(f.ex, f.__class__(g).ex))

        def lcm(f, g):
            from sympy.polys import lcm
            return f.__class__(lcm(f.ex, f.__class__(g).ex))

    dtype = Expression

    zero = Expression(0)
    one = Expression(1)

    rep = 'EX'

    has_assoc_Ring = False
    has_assoc_Field = True

    def __init__(self):
        pass

    def __eq__(self, other):
        if isinstance(other, ExpressionDomain):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        return hash("EX")

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return a.as_expr()

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        return self.dtype(a)

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_GaussianIntegerRing(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_GaussianRationalField(K1, a, K0):
        """Convert a ``GaussianRational`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_AlgebraicField(K1, a, K0):
        """Convert an ``ANP`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ComplexField(K1, a, K0):
        """Convert a mpmath ``mpc`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_FractionField(K1, a, K0):
        """Convert a ``DMF`` object to ``dtype``. """
        return K1(K0.to_sympy(a))

    def from_ExpressionDomain(K1, a, K0):
        """Convert a ``EX`` object to ``dtype``. """
        return a

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self  # XXX: EX is not a ring but we don't have much choice here.

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return a.ex.as_coeff_mul()[0].is_positive

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return a.ex.could_extract_minus_sign()

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return a.ex.as_coeff_mul()[0].is_nonpositive

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return a.ex.as_coeff_mul()[0].is_nonnegative

    def numer(self, a):
        """Returns numerator of ``a``. """
        return a.numer()

    def denom(self, a):
        """Returns denominator of ``a``. """
        return a.denom()

    def gcd(self, a, b):
        return self(1)

    def lcm(self, a, b):
        return a.lcm(b)


EX = ExpressionDomain()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\_request_methods.py ===
from __future__ import annotations

import json as _json
import typing
from urllib.parse import urlencode

from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .filepost import _TYPE_FIELDS, encode_multipart_formdata
from .response import BaseHTTPResponse

__all__ = ["RequestMethods"]

_TYPE_ENCODE_URL_FIELDS = typing.Union[
    typing.Sequence[tuple[str, typing.Union[str, bytes]]],
    typing.Mapping[str, typing.Union[str, bytes]],
]


class RequestMethods:
    """
    Convenience mixin for classes who implement a :meth:`urlopen` method, such
    as :class:`urllib3.HTTPConnectionPool` and
    :class:`urllib3.PoolManager`.

    Provides behavior for making common types of HTTP request methods and
    decides which type of request field encoding to use.

    Specifically,

    :meth:`.request_encode_url` is for sending requests whose fields are
    encoded in the URL (such as GET, HEAD, DELETE).

    :meth:`.request_encode_body` is for sending requests whose fields are
    encoded in the *body* of the request using multipart or www-form-urlencoded
    (such as for POST, PUT, PATCH).

    :meth:`.request` is for making any kind of request, it will look up the
    appropriate encoding format and use one of the above two methods to make
    the request.

    Initializer parameters:

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.
    """

    _encode_url_methods = {"DELETE", "GET", "HEAD", "OPTIONS"}

    def __init__(self, headers: typing.Mapping[str, str] | None = None) -> None:
        self.headers = headers or {}

    def urlopen(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        encode_multipart: bool = True,
        multipart_boundary: str | None = None,
        **kw: typing.Any,
    ) -> BaseHTTPResponse:  # Abstract
        raise NotImplementedError(
            "Classes extending RequestMethods must implement "
            "their own ``urlopen`` method."
        )

    def request(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        fields: _TYPE_FIELDS | None = None,
        headers: typing.Mapping[str, str] | None = None,
        json: typing.Any | None = None,
        **urlopen_kw: typing.Any,
    ) -> BaseHTTPResponse:
        """
        Make a request using :meth:`urlopen` with the appropriate encoding of
        ``fields`` based on the ``method`` used.

        This is a convenience method that requires the least amount of manual
        effort. It can be used in most situations, while still having the
        option to drop down to more specific methods when necessary, such as
        :meth:`request_encode_url`, :meth:`request_encode_body`,
        or even the lowest level :meth:`urlopen`.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param body:
            Data to send in the request body, either :class:`str`, :class:`bytes`,
            an iterable of :class:`str`/:class:`bytes`, or a file-like object.

        :param fields:
            Data to encode and send in the URL or request body, depending on ``method``.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param json:
            Data to encode and send as JSON with UTF-encoded in the request body.
            The ``"Content-Type"`` header will be set to ``"application/json"``
            unless specified otherwise.
        """
        method = method.upper()

        if json is not None and body is not None:
            raise TypeError(
                "request got values for both 'body' and 'json' parameters which are mutually exclusive"
            )

        if json is not None:
            if headers is None:
                headers = self.headers

            if not ("content-type" in map(str.lower, headers.keys())):
                headers = HTTPHeaderDict(headers)
                headers["Content-Type"] = "application/json"

            body = _json.dumps(json, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )

        if body is not None:
            urlopen_kw["body"] = body

        if method in self._encode_url_methods:
            return self.request_encode_url(
                method,
                url,
                fields=fields,  # type: ignore[arg-type]
                headers=headers,
                **urlopen_kw,
            )
        else:
            return self.request_encode_body(
                method, url, fields=fields, headers=headers, **urlopen_kw
            )

    def request_encode_url(
        self,
        method: str,
        url: str,
        fields: _TYPE_ENCODE_URL_FIELDS | None = None,
        headers: typing.Mapping[str, str] | None = None,
        **urlopen_kw: str,
    ) -> BaseHTTPResponse:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the url. This is useful for request methods like GET, HEAD, DELETE, etc.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param fields:
            Data to encode and send in the URL.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.
        """
        if headers is None:
            headers = self.headers

        extra_kw: dict[str, typing.Any] = {"headers": headers}
        extra_kw.update(urlopen_kw)

        if fields:
            url += "?" + urlencode(fields)

        return self.urlopen(method, url, **extra_kw)

    def request_encode_body(
        self,
        method: str,
        url: str,
        fields: _TYPE_FIELDS | None = None,
        headers: typing.Mapping[str, str] | None = None,
        encode_multipart: bool = True,
        multipart_boundary: str | None = None,
        **urlopen_kw: str,
    ) -> BaseHTTPResponse:
        """
        Make a request using :meth:`urlopen` with the ``fields`` encoded in
        the body. This is useful for request methods like POST, PUT, PATCH, etc.

        When ``encode_multipart=True`` (default), then
        :func:`urllib3.encode_multipart_formdata` is used to encode
        the payload with the appropriate content type. Otherwise
        :func:`urllib.parse.urlencode` is used with the
        'application/x-www-form-urlencoded' content type.

        Multipart encoding must be used when posting files, and it's reasonably
        safe to use it in other times too. However, it may break request
        signing, such as with OAuth.

        Supports an optional ``fields`` parameter of key/value strings AND
        key/filetuple. A filetuple is a (filename, data, MIME type) tuple where
        the MIME type is optional. For example::

            fields = {
                'foo': 'bar',
                'fakefile': ('foofile.txt', 'contents of foofile'),
                'realfile': ('barfile.txt', open('realfile').read()),
                'typedfile': ('bazfile.bin', open('bazfile').read(),
                              'image/jpeg'),
                'nonamefile': 'contents of nonamefile field',
            }

        When uploading a file, providing a filename (the first parameter of the
        tuple) is optional but recommended to best mimic behavior of browsers.

        Note that if ``headers`` are supplied, the 'Content-Type' header will
        be overwritten because it depends on the dynamic random boundary string
        which is used to compose the body of the request. The random boundary
        string can be explicitly set with the ``multipart_boundary`` parameter.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param fields:
            Data to encode and send in the request body.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param encode_multipart:
            If True, encode the ``fields`` using the multipart/form-data MIME
            format.

        :param multipart_boundary:
            If not specified, then a random boundary will be generated using
            :func:`urllib3.filepost.choose_boundary`.
        """
        if headers is None:
            headers = self.headers

        extra_kw: dict[str, typing.Any] = {"headers": HTTPHeaderDict(headers)}
        body: bytes | str

        if fields:
            if "body" in urlopen_kw:
                raise TypeError(
                    "request got values for both 'fields' and 'body', can only specify one."
                )

            if encode_multipart:
                body, content_type = encode_multipart_formdata(
                    fields, boundary=multipart_boundary
                )
            else:
                body, content_type = (
                    urlencode(fields),  # type: ignore[arg-type]
                    "application/x-www-form-urlencoded",
                )

            extra_kw["body"] = body
            extra_kw["headers"].setdefault("Content-Type", content_type)

        extra_kw.update(urlopen_kw)

        return self.urlopen(method, url, **extra_kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\colorama\ansitowin32.py ===
# Copyright Jonathan Hartley 2013. BSD 3-Clause license, see LICENSE file.
import re
import sys
import os

from .ansi import AnsiFore, AnsiBack, AnsiStyle, Style, BEL
from .winterm import enable_vt_processing, WinTerm, WinColor, WinStyle
from .win32 import windll, winapi_test


winterm = None
if windll is not None:
    winterm = WinTerm()


class StreamWrapper(object):
    '''
    Wraps a stream (such as stdout), acting as a transparent proxy for all
    attribute access apart from method 'write()', which is delegated to our
    Converter instance.
    '''
    def __init__(self, wrapped, converter):
        # double-underscore everything to prevent clashes with names of
        # attributes on the wrapped stream object.
        self.__wrapped = wrapped
        self.__convertor = converter

    def __getattr__(self, name):
        return getattr(self.__wrapped, name)

    def __enter__(self, *args, **kwargs):
        # special method lookup bypasses __getattr__/__getattribute__, see
        # https://stackoverflow.com/questions/12632894/why-doesnt-getattr-work-with-exit
        # thus, contextlib magic methods are not proxied via __getattr__
        return self.__wrapped.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        return self.__wrapped.__exit__(*args, **kwargs)

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def write(self, text):
        self.__convertor.write(text)

    def isatty(self):
        stream = self.__wrapped
        if 'PYCHARM_HOSTED' in os.environ:
            if stream is not None and (stream is sys.__stdout__ or stream is sys.__stderr__):
                return True
        try:
            stream_isatty = stream.isatty
        except AttributeError:
            return False
        else:
            return stream_isatty()

    @property
    def closed(self):
        stream = self.__wrapped
        try:
            return stream.closed
        # AttributeError in the case that the stream doesn't support being closed
        # ValueError for the case that the stream has already been detached when atexit runs
        except (AttributeError, ValueError):
            return True


class AnsiToWin32(object):
    '''
    Implements a 'write()' method which, on Windows, will strip ANSI character
    sequences from the text, and if outputting to a tty, will convert them into
    win32 function calls.
    '''
    ANSI_CSI_RE = re.compile('\001?\033\\[((?:\\d|;)*)([a-zA-Z])\002?')   # Control Sequence Introducer
    ANSI_OSC_RE = re.compile('\001?\033\\]([^\a]*)(\a)\002?')             # Operating System Command

    def __init__(self, wrapped, convert=None, strip=None, autoreset=False):
        # The wrapped stream (normally sys.stdout or sys.stderr)
        self.wrapped = wrapped

        # should we reset colors to defaults after every .write()
        self.autoreset = autoreset

        # create the proxy wrapping our output stream
        self.stream = StreamWrapper(wrapped, self)

        on_windows = os.name == 'nt'
        # We test if the WinAPI works, because even if we are on Windows
        # we may be using a terminal that doesn't support the WinAPI
        # (e.g. Cygwin Terminal). In this case it's up to the terminal
        # to support the ANSI codes.
        conversion_supported = on_windows and winapi_test()
        try:
            fd = wrapped.fileno()
        except Exception:
            fd = -1
        system_has_native_ansi = not on_windows or enable_vt_processing(fd)
        have_tty = not self.stream.closed and self.stream.isatty()
        need_conversion = conversion_supported and not system_has_native_ansi

        # should we strip ANSI sequences from our output?
        if strip is None:
            strip = need_conversion or not have_tty
        self.strip = strip

        # should we should convert ANSI sequences into win32 calls?
        if convert is None:
            convert = need_conversion and have_tty
        self.convert = convert

        # dict of ansi codes to win32 functions and parameters
        self.win32_calls = self.get_win32_calls()

        # are we wrapping stderr?
        self.on_stderr = self.wrapped is sys.stderr

    def should_wrap(self):
        '''
        True if this class is actually needed. If false, then the output
        stream will not be affected, nor will win32 calls be issued, so
        wrapping stdout is not actually required. This will generally be
        False on non-Windows platforms, unless optional functionality like
        autoreset has been requested using kwargs to init()
        '''
        return self.convert or self.strip or self.autoreset

    def get_win32_calls(self):
        if self.convert and winterm:
            return {
                AnsiStyle.RESET_ALL: (winterm.reset_all, ),
                AnsiStyle.BRIGHT: (winterm.style, WinStyle.BRIGHT),
                AnsiStyle.DIM: (winterm.style, WinStyle.NORMAL),
                AnsiStyle.NORMAL: (winterm.style, WinStyle.NORMAL),
                AnsiFore.BLACK: (winterm.fore, WinColor.BLACK),
                AnsiFore.RED: (winterm.fore, WinColor.RED),
                AnsiFore.GREEN: (winterm.fore, WinColor.GREEN),
                AnsiFore.YELLOW: (winterm.fore, WinColor.YELLOW),
                AnsiFore.BLUE: (winterm.fore, WinColor.BLUE),
                AnsiFore.MAGENTA: (winterm.fore, WinColor.MAGENTA),
                AnsiFore.CYAN: (winterm.fore, WinColor.CYAN),
                AnsiFore.WHITE: (winterm.fore, WinColor.GREY),
                AnsiFore.RESET: (winterm.fore, ),
                AnsiFore.LIGHTBLACK_EX: (winterm.fore, WinColor.BLACK, True),
                AnsiFore.LIGHTRED_EX: (winterm.fore, WinColor.RED, True),
                AnsiFore.LIGHTGREEN_EX: (winterm.fore, WinColor.GREEN, True),
                AnsiFore.LIGHTYELLOW_EX: (winterm.fore, WinColor.YELLOW, True),
                AnsiFore.LIGHTBLUE_EX: (winterm.fore, WinColor.BLUE, True),
                AnsiFore.LIGHTMAGENTA_EX: (winterm.fore, WinColor.MAGENTA, True),
                AnsiFore.LIGHTCYAN_EX: (winterm.fore, WinColor.CYAN, True),
                AnsiFore.LIGHTWHITE_EX: (winterm.fore, WinColor.GREY, True),
                AnsiBack.BLACK: (winterm.back, WinColor.BLACK),
                AnsiBack.RED: (winterm.back, WinColor.RED),
                AnsiBack.GREEN: (winterm.back, WinColor.GREEN),
                AnsiBack.YELLOW: (winterm.back, WinColor.YELLOW),
                AnsiBack.BLUE: (winterm.back, WinColor.BLUE),
                AnsiBack.MAGENTA: (winterm.back, WinColor.MAGENTA),
                AnsiBack.CYAN: (winterm.back, WinColor.CYAN),
                AnsiBack.WHITE: (winterm.back, WinColor.GREY),
                AnsiBack.RESET: (winterm.back, ),
                AnsiBack.LIGHTBLACK_EX: (winterm.back, WinColor.BLACK, True),
                AnsiBack.LIGHTRED_EX: (winterm.back, WinColor.RED, True),
                AnsiBack.LIGHTGREEN_EX: (winterm.back, WinColor.GREEN, True),
                AnsiBack.LIGHTYELLOW_EX: (winterm.back, WinColor.YELLOW, True),
                AnsiBack.LIGHTBLUE_EX: (winterm.back, WinColor.BLUE, True),
                AnsiBack.LIGHTMAGENTA_EX: (winterm.back, WinColor.MAGENTA, True),
                AnsiBack.LIGHTCYAN_EX: (winterm.back, WinColor.CYAN, True),
                AnsiBack.LIGHTWHITE_EX: (winterm.back, WinColor.GREY, True),
            }
        return dict()

    def write(self, text):
        if self.strip or self.convert:
            self.write_and_convert(text)
        else:
            self.wrapped.write(text)
            self.wrapped.flush()
        if self.autoreset:
            self.reset_all()


    def reset_all(self):
        if self.convert:
            self.call_win32('m', (0,))
        elif not self.strip and not self.stream.closed:
            self.wrapped.write(Style.RESET_ALL)


    def write_and_convert(self, text):
        '''
        Write the given text to our wrapped stream, stripping any ANSI
        sequences from the text, and optionally converting them into win32
        calls.
        '''
        cursor = 0
        text = self.convert_osc(text)
        for match in self.ANSI_CSI_RE.finditer(text):
            start, end = match.span()
            self.write_plain_text(text, cursor, start)
            self.convert_ansi(*match.groups())
            cursor = end
        self.write_plain_text(text, cursor, len(text))


    def write_plain_text(self, text, start, end):
        if start < end:
            self.wrapped.write(text[start:end])
            self.wrapped.flush()


    def convert_ansi(self, paramstring, command):
        if self.convert:
            params = self.extract_params(command, paramstring)
            self.call_win32(command, params)


    def extract_params(self, command, paramstring):
        if command in 'Hf':
            params = tuple(int(p) if len(p) != 0 else 1 for p in paramstring.split(';'))
            while len(params) < 2:
                # defaults:
                params = params + (1,)
        else:
            params = tuple(int(p) for p in paramstring.split(';') if len(p) != 0)
            if len(params) == 0:
                # defaults:
                if command in 'JKm':
                    params = (0,)
                elif command in 'ABCD':
                    params = (1,)

        return params


    def call_win32(self, command, params):
        if command == 'm':
            for param in params:
                if param in self.win32_calls:
                    func_args = self.win32_calls[param]
                    func = func_args[0]
                    args = func_args[1:]
                    kwargs = dict(on_stderr=self.on_stderr)
                    func(*args, **kwargs)
        elif command in 'J':
            winterm.erase_screen(params[0], on_stderr=self.on_stderr)
        elif command in 'K':
            winterm.erase_line(params[0], on_stderr=self.on_stderr)
        elif command in 'Hf':     # cursor position - absolute
            winterm.set_cursor_position(params, on_stderr=self.on_stderr)
        elif command in 'ABCD':   # cursor position - relative
            n = params[0]
            # A - up, B - down, C - forward, D - back
            x, y = {'A': (0, -n), 'B': (0, n), 'C': (n, 0), 'D': (-n, 0)}[command]
            winterm.cursor_adjust(x, y, on_stderr=self.on_stderr)


    def convert_osc(self, text):
        for match in self.ANSI_OSC_RE.finditer(text):
            start, end = match.span()
            text = text[:start] + text[end:]
            paramstring, command = match.groups()
            if command == BEL:
                if paramstring.count(";") == 1:
                    params = paramstring.split(";")
                    # 0 - change title and icon (we will only change title)
                    # 1 - change icon (we don't support this)
                    # 2 - change title
                    if params[0] in '02':
                        winterm.set_title(params[1])
        return text


    def flush(self):
        self.wrapped.flush()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\mariadbconnector.py ===
# dialects/mysql/mariadbconnector.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""

.. dialect:: mysql+mariadbconnector
    :name: MariaDB Connector/Python
    :dbapi: mariadb
    :connectstring: mariadb+mariadbconnector://<user>:<password>@<host>[:<port>]/<dbname>
    :url: https://pypi.org/project/mariadb/

Driver Status
-------------

MariaDB Connector/Python enables Python programs to access MariaDB and MySQL
databases using an API which is compliant with the Python DB API 2.0 (PEP-249).
It is written in C and uses MariaDB Connector/C client library for client server
communication.

Note that the default driver for a ``mariadb://`` connection URI continues to
be ``mysqldb``. ``mariadb+mariadbconnector://`` is required to use this driver.

.. mariadb: https://github.com/mariadb-corporation/mariadb-connector-python

"""  # noqa
import re
from uuid import UUID as _python_UUID

from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes


mariadb_cpy_minimum_version = (1, 0, 1)


class _MariaDBUUID(sqltypes.UUID[sqltypes._UUID_RETURN]):
    # work around JIRA issue
    # https://jira.mariadb.org/browse/CONPY-270.  When that issue is fixed,
    # this type can be removed.
    def result_processor(self, dialect, coltype):
        if self.as_uuid:

            def process(value):
                if value is not None:
                    if hasattr(value, "decode"):
                        value = value.decode("ascii")
                    value = _python_UUID(value)
                return value

            return process
        else:

            def process(value):
                if value is not None:
                    if hasattr(value, "decode"):
                        value = value.decode("ascii")
                    value = str(_python_UUID(value))
                return value

            return process


class MySQLExecutionContext_mariadbconnector(MySQLExecutionContext):
    _lastrowid = None

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(buffered=False)

    def create_default_cursor(self):
        return self._dbapi_connection.cursor(buffered=True)

    def post_exec(self):
        super().post_exec()

        self._rowcount = self.cursor.rowcount

        if self.isinsert and self.compiled.postfetch_lastrowid:
            self._lastrowid = self.cursor.lastrowid

    def get_lastrowid(self):
        return self._lastrowid


class MySQLCompiler_mariadbconnector(MySQLCompiler):
    pass


class MySQLDialect_mariadbconnector(MySQLDialect):
    driver = "mariadbconnector"
    supports_statement_cache = True

    # set this to True at the module level to prevent the driver from running
    # against a backend that server detects as MySQL. currently this appears to
    # be unnecessary as MariaDB client libraries have always worked against
    # MySQL databases.   However, if this changes at some point, this can be
    # adjusted, but PLEASE ADD A TEST in test/dialect/mysql/test_dialect.py if
    # this change is made at some point to ensure the correct exception
    # is raised at the correct point when running the driver against
    # a MySQL backend.
    # is_mariadb = True

    supports_unicode_statements = True
    encoding = "utf8mb4"
    convert_unicode = True
    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    supports_native_decimal = True
    default_paramstyle = "qmark"
    execution_ctx_cls = MySQLExecutionContext_mariadbconnector
    statement_compiler = MySQLCompiler_mariadbconnector

    supports_server_side_cursors = True

    colspecs = util.update_copy(
        MySQLDialect.colspecs, {sqltypes.Uuid: _MariaDBUUID}
    )

    @util.memoized_property
    def _dbapi_version(self):
        if self.dbapi and hasattr(self.dbapi, "__version__"):
            return tuple(
                [
                    int(x)
                    for x in re.findall(
                        r"(\d+)(?:[-\.]?|$)", self.dbapi.__version__
                    )
                ]
            )
        else:
            return (99, 99, 99)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.paramstyle = "qmark"
        if self.dbapi is not None:
            if self._dbapi_version < mariadb_cpy_minimum_version:
                raise NotImplementedError(
                    "The minimum required version for MariaDB "
                    "Connector/Python is %s"
                    % ".".join(str(x) for x in mariadb_cpy_minimum_version)
                )

    @classmethod
    def import_dbapi(cls):
        return __import__("mariadb")

    def is_disconnect(self, e, connection, cursor):
        if super().is_disconnect(e, connection, cursor):
            return True
        elif isinstance(e, self.dbapi.Error):
            str_e = str(e).lower()
            return "not connected" in str_e or "isn't valid" in str_e
        else:
            return False

    def create_connect_args(self, url):
        opts = url.translate_connect_args()
        opts.update(url.query)

        int_params = [
            "connect_timeout",
            "read_timeout",
            "write_timeout",
            "client_flag",
            "port",
            "pool_size",
        ]
        bool_params = [
            "local_infile",
            "ssl_verify_cert",
            "ssl",
            "pool_reset_connection",
            "compress",
        ]

        for key in int_params:
            util.coerce_kw_type(opts, key, int)
        for key in bool_params:
            util.coerce_kw_type(opts, key, bool)

        # FOUND_ROWS must be set in CLIENT_FLAGS to enable
        # supports_sane_rowcount.
        client_flag = opts.get("client_flag", 0)
        if self.dbapi is not None:
            try:
                CLIENT_FLAGS = __import__(
                    self.dbapi.__name__ + ".constants.CLIENT"
                ).constants.CLIENT
                client_flag |= CLIENT_FLAGS.FOUND_ROWS
            except (AttributeError, ImportError):
                self.supports_sane_rowcount = False
            opts["client_flag"] = client_flag
        return [[], opts]

    def _extract_error_code(self, exception):
        try:
            rc = exception.errno
        except:
            rc = -1
        return rc

    def _detect_charset(self, connection):
        return "utf8mb4"

    def get_isolation_level_values(self, dbapi_connection):
        return (
            "SERIALIZABLE",
            "READ UNCOMMITTED",
            "READ COMMITTED",
            "REPEATABLE READ",
            "AUTOCOMMIT",
        )

    def set_isolation_level(self, connection, level):
        if level == "AUTOCOMMIT":
            connection.autocommit = True
        else:
            connection.autocommit = False
            super().set_isolation_level(connection, level)

    def do_begin_twophase(self, connection, xid):
        connection.execute(
            sql.text("XA BEGIN :xid").bindparams(
                sql.bindparam("xid", xid, literal_execute=True)
            )
        )

    def do_prepare_twophase(self, connection, xid):
        connection.execute(
            sql.text("XA END :xid").bindparams(
                sql.bindparam("xid", xid, literal_execute=True)
            )
        )
        connection.execute(
            sql.text("XA PREPARE :xid").bindparams(
                sql.bindparam("xid", xid, literal_execute=True)
            )
        )

    def do_rollback_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        if not is_prepared:
            connection.execute(
                sql.text("XA END :xid").bindparams(
                    sql.bindparam("xid", xid, literal_execute=True)
                )
            )
        connection.execute(
            sql.text("XA ROLLBACK :xid").bindparams(
                sql.bindparam("xid", xid, literal_execute=True)
            )
        )

    def do_commit_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        if not is_prepared:
            self.do_prepare_twophase(connection, xid)
        connection.execute(
            sql.text("XA COMMIT :xid").bindparams(
                sql.bindparam("xid", xid, literal_execute=True)
            )
        )


dialect = MySQLDialect_mariadbconnector

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\partitions_.py ===
from mpmath.libmp import (fzero, from_int, from_rational,
    fone, fhalf, bitcount, to_int, mpf_mul, mpf_div, mpf_sub,
    mpf_add, mpf_sqrt, mpf_pi, mpf_cosh_sinh, mpf_cos, mpf_sin)
from .residue_ntheory import _sqrt_mod_prime_power, is_quad_residue
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo

import math
from itertools import count

def _pre():
    maxn = 10**5
    global _factor, _totient
    _factor = [0]*maxn
    _totient = [1]*maxn
    lim = int(maxn**0.5) + 5
    for i in range(2, lim):
        if _factor[i] == 0:
            for j in range(i*i, maxn, i):
                if _factor[j] == 0:
                    _factor[j] = i
    for i in range(2, maxn):
        if _factor[i] == 0:
            _factor[i] = i
            _totient[i] = i-1
            continue
        x = _factor[i]
        y = i//x
        if y % x == 0:
            _totient[i] = _totient[y]*x
        else:
            _totient[i] = _totient[y]*(x - 1)

def _a(n, k, prec):
    """ Compute the inner sum in HRR formula [1]_

    References
    ==========

    .. [1] https://msp.org/pjm/1956/6-1/pjm-v6-n1-p18-p.pdf

    """
    if k == 1:
        return fone

    k1 = k
    e = 0
    p = _factor[k]
    while k1 % p == 0:
        k1 //= p
        e += 1
    k2 = k//k1 # k2 = p^e
    v = 1 - 24*n
    pi = mpf_pi(prec)

    if k1 == 1:
        # k  = p^e
        if p == 2:
            mod = 8*k
            v = mod + v % mod
            v = (v*pow(9, k - 1, mod)) % mod
            m = _sqrt_mod_prime_power(v, 2, e + 3)[0]
            arg = mpf_div(mpf_mul(
                from_int(4*m), pi, prec), from_int(mod), prec)
            return mpf_mul(mpf_mul(
                from_int((-1)**e*(2 - (m % 4))),
                mpf_sqrt(from_int(k), prec), prec),
                mpf_sin(arg, prec), prec)
        if p == 3:
            mod = 3*k
            v = mod + v % mod
            if e > 1:
                v = (v*pow(64, k//3 - 1, mod)) % mod
            m = _sqrt_mod_prime_power(v, 3, e + 1)[0]
            arg = mpf_div(mpf_mul(from_int(4*m), pi, prec),
                from_int(mod), prec)
            return mpf_mul(mpf_mul(
                from_int(2*(-1)**(e + 1)*(3 - 2*(m % 3))),
                mpf_sqrt(from_int(k//3), prec), prec),
                mpf_sin(arg, prec), prec)
        v = k + v % k
        jacobi3 = -1 if k % 12 in [5, 7] else 1
        if v % p == 0:
            if e == 1:
                return mpf_mul(
                    from_int(jacobi3),
                    mpf_sqrt(from_int(k), prec), prec)
            return fzero
        if not is_quad_residue(v, p):
            return fzero
        _phi = p**(e - 1)*(p - 1)
        v = (v*pow(576, _phi - 1, k))
        m = _sqrt_mod_prime_power(v, p, e)[0]
        arg = mpf_div(
            mpf_mul(from_int(4*m), pi, prec),
            from_int(k), prec)
        return mpf_mul(mpf_mul(
            from_int(2*jacobi3),
            mpf_sqrt(from_int(k), prec), prec),
            mpf_cos(arg, prec), prec)

    if p != 2 or e >= 3:
        d1, d2 = math.gcd(k1, 24), math.gcd(k2, 24)
        e = 24//(d1*d2)
        n1 = ((d2*e*n + (k2**2 - 1)//d1)*
            pow(e*k2*k2*d2, _totient[k1] - 1, k1)) % k1
        n2 = ((d1*e*n + (k1**2 - 1)//d2)*
            pow(e*k1*k1*d1, _totient[k2] - 1, k2)) % k2
        return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
    if e == 2:
        n1 = ((8*n + 5)*pow(128, _totient[k1] - 1, k1)) % k1
        n2 = (4 + ((n - 2 - (k1**2 - 1)//8)*(k1**2)) % 4) % 4
        return mpf_mul(mpf_mul(
            from_int(-1),
            _a(n1, k1, prec), prec),
            _a(n2, k2, prec))
    n1 = ((8*n + 1)*pow(32, _totient[k1] - 1, k1)) % k1
    n2 = (2 + (n - (k1**2 - 1)//8) % 2) % 2
    return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)

def _d(n, j, prec, sq23pi, sqrt8):
    """
    Compute the sinh term in the outer sum of the HRR formula.
    The constants sqrt(2/3*pi) and sqrt(8) must be precomputed.
    """
    j = from_int(j)
    pi = mpf_pi(prec)
    a = mpf_div(sq23pi, j, prec)
    b = mpf_sub(from_int(n), from_rational(1, 24, prec), prec)
    c = mpf_sqrt(b, prec)
    ch, sh = mpf_cosh_sinh(mpf_mul(a, c), prec)
    D = mpf_div(
        mpf_sqrt(j, prec),
        mpf_mul(mpf_mul(sqrt8, b), pi), prec)
    E = mpf_sub(mpf_mul(a, ch), mpf_div(sh, c, prec), prec)
    return mpf_mul(D, E)


@recurrence_memo([1, 1])
def _partition_rec(n: int, prev) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int
        nonnegative integer

    """
    v = 0
    penta = 0 # pentagonal number: 1, 5, 12, ...
    for i in count():
        penta += 3*i + 1
        np = n - penta
        if np < 0:
            break
        s = prev[np]
        np -= i + 1
        # np = n - gp where gp = generalized pentagonal: 2, 7, 15, ...
        if 0 <= np:
            s += prev[np]
        v += -s if i % 2 else s
    return v


def _partition(n: int) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int

    """
    if n < 0:
        return 0
    if (n <= 200_000 and n - _partition_rec.cache_length() < 70 or
            _partition_rec.cache_length() == 2 and n < 14_400):
        # There will be 2*10**5 elements created here
        # and n elements created by partition, so in case we
        # are going to be working with small n, we just
        # use partition to calculate (and cache) the values
        # since lookup is used there while summation, using
        # _factor and _totient, will be used below. But we
        # only do so if n is relatively close to the length
        # of the cache since doing 1 calculation here is about
        # the same as adding 70 elements to the cache. In addition,
        # the startup here costs about the same as calculating the first
        # 14,400 values via partition, so we delay startup here unless n
        # is smaller than that.
        return _partition_rec(n)
    if '_factor' not in globals():
        _pre()
    # Estimate number of bits in p(n). This formula could be tidied
    pbits = int((
        math.pi*(2*n/3.)**0.5 -
        math.log(4*n))/math.log(10) + 1) * \
        math.log2(10)
    prec = p = int(pbits*1.1 + 100)

    # find the number of terms needed so rounded sum will be accurate
    # using Rademacher's bound M(n, N) for the remainder after a partial
    # sum of N terms (https://arxiv.org/pdf/1205.5991.pdf, (1.8))
    c1 = 44*math.pi**2/(225*math.sqrt(3))
    c2 = math.pi*math.sqrt(2)/75
    c3 = math.pi*math.sqrt(2/3)
    def _M(n, N):
        sqrt = math.sqrt
        return c1/sqrt(N) + c2*sqrt(N/(n - 1))*math.sinh(c3*sqrt(n)/N)
    big = max(9, math.ceil(n**0.5))  # should be too large (for n > 65, ceil should work)
    assert _M(n, big) < 0.5  # else double big until too large
    while big > 40 and _M(n, big) < 0.5:
        big //= 2
    small = big
    big = small*2
    while big - small > 1:
        N = (big + small)//2
        if (er := _M(n, N)) < 0.5:
            big = N
        elif er >= 0.5:
            small = N
    M = big  # done with function M; now have value

    # sanity check for expected size of answer
    if M > 10**5:  # i.e. M > maxn
        raise ValueError("Input too big")  # i.e. n > 149832547102

    # calculate it
    s = fzero
    sq23pi = mpf_mul(mpf_sqrt(from_rational(2, 3, p), p), mpf_pi(p), p)
    sqrt8 = mpf_sqrt(from_int(8), p)
    for q in range(1, M):
        a = _a(n, q, p)
        d = _d(n, q, p, sq23pi, sqrt8)
        s = mpf_add(s, mpf_mul(a, d), prec)
        # On average, the terms decrease rapidly in magnitude.
        # Dynamically reducing the precision greatly improves
        # performance.
        p = bitcount(abs(to_int(d))) + 50
    return int(to_int(mpf_add(s, fhalf, prec)))


@deprecated("""\
The `sympy.ntheory.partitions_.npartitions` has been moved to `sympy.functions.combinatorial.numbers.partition`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def npartitions(n, verbose=False):
    """
    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    .. deprecated:: 1.13

        The ``npartitions`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.partition`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import partition
    >>> partition(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    """
    from sympy.functions.combinatorial.numbers import partition as func_partition
    return func_partition(n)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\emscripten\response.py ===
from __future__ import annotations

import json as _json
import logging
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException as HTTPException
from io import BytesIO, IOBase

from ...exceptions import InvalidHeader, TimeoutError
from ...response import BaseHTTPResponse
from ...util.retry import Retry
from .request import EmscriptenRequest

if typing.TYPE_CHECKING:
    from ..._base_connection import BaseHTTPConnection, BaseHTTPSConnection

log = logging.getLogger(__name__)


@dataclass
class EmscriptenResponse:
    status_code: int
    headers: dict[str, str]
    body: IOBase | bytes
    request: EmscriptenRequest


class EmscriptenHttpResponseWrapper(BaseHTTPResponse):
    def __init__(
        self,
        internal_response: EmscriptenResponse,
        url: str | None = None,
        connection: BaseHTTPConnection | BaseHTTPSConnection | None = None,
    ):
        self._pool = None  # set by pool class
        self._body = None
        self._response = internal_response
        self._url = url
        self._connection = connection
        self._closed = False
        super().__init__(
            headers=internal_response.headers,
            status=internal_response.status_code,
            request_url=url,
            version=0,
            version_string="HTTP/?",
            reason="",
            decode_content=True,
        )
        self.length_remaining = self._init_length(self._response.request.method)
        self.length_is_certain = False

    @property
    def url(self) -> str | None:
        return self._url

    @url.setter
    def url(self, url: str | None) -> None:
        self._url = url

    @property
    def connection(self) -> BaseHTTPConnection | BaseHTTPSConnection | None:
        return self._connection

    @property
    def retries(self) -> Retry | None:
        return self._retries

    @retries.setter
    def retries(self, retries: Retry | None) -> None:
        # Override the request_url if retries has a redirect location.
        self._retries = retries

    def stream(
        self, amt: int | None = 2**16, decode_content: bool | None = None
    ) -> typing.Generator[bytes]:
        """
        A generator wrapper for the read() method. A call will block until
        ``amt`` bytes have been read from the connection or until the
        connection is closed.

        :param amt:
            How much of the content to read. The generator will return up to
            much data per iteration, but may return less. This is particularly
            likely when using compressed data. However, the empty string will
            never be returned.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        while True:
            data = self.read(amt=amt, decode_content=decode_content)

            if data:
                yield data
            else:
                break

    def _init_length(self, request_method: str | None) -> int | None:
        length: int | None
        content_length: str | None = self.headers.get("content-length")

        if content_length is not None:
            try:
                # RFC 7230 section 3.3.2 specifies multiple content lengths can
                # be sent in a single Content-Length header
                # (e.g. Content-Length: 42, 42). This line ensures the values
                # are all valid ints and that as long as the `set` length is 1,
                # all values are the same. Otherwise, the header is invalid.
                lengths = {int(val) for val in content_length.split(",")}
                if len(lengths) > 1:
                    raise InvalidHeader(
                        "Content-Length contained multiple "
                        "unmatching values (%s)" % content_length
                    )
                length = lengths.pop()
            except ValueError:
                length = None
            else:
                if length < 0:
                    length = None

        else:  # if content_length is None
            length = None

        # Check for responses that shouldn't include a body
        if (
            self.status in (204, 304)
            or 100 <= self.status < 200
            or request_method == "HEAD"
        ):
            length = 0

        return length

    def read(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,  # ignored because browser decodes always
        cache_content: bool = False,
    ) -> bytes:
        if (
            self._closed
            or self._response is None
            or (isinstance(self._response.body, IOBase) and self._response.body.closed)
        ):
            return b""

        with self._error_catcher():
            # body has been preloaded as a string by XmlHttpRequest
            if not isinstance(self._response.body, IOBase):
                self.length_remaining = len(self._response.body)
                self.length_is_certain = True
                # wrap body in IOStream
                self._response.body = BytesIO(self._response.body)
            if amt is not None and amt >= 0:
                # don't cache partial content
                cache_content = False
                data = self._response.body.read(amt)
            else:  # read all we can (and cache it)
                data = self._response.body.read()
                if cache_content:
                    self._body = data
            if self.length_remaining is not None:
                self.length_remaining = max(self.length_remaining - len(data), 0)
            if len(data) == 0 or (
                self.length_is_certain and self.length_remaining == 0
            ):
                # definitely finished reading, close response stream
                self._response.body.close()
            return typing.cast(bytes, data)

    def read_chunked(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
    ) -> typing.Generator[bytes]:
        # chunked is handled by browser
        while True:
            bytes = self.read(amt, decode_content)
            if not bytes:
                break
            yield bytes

    def release_conn(self) -> None:
        if not self._pool or not self._connection:
            return None

        self._pool._put_conn(self._connection)
        self._connection = None

    def drain_conn(self) -> None:
        self.close()

    @property
    def data(self) -> bytes:
        if self._body:
            return self._body
        else:
            return self.read(cache_content=True)

    def json(self) -> typing.Any:
        """
        Deserializes the body of the HTTP response as a Python object.

        The body of the HTTP response must be encoded using UTF-8, as per
        `RFC 8529 Section 8.1 <https://www.rfc-editor.org/rfc/rfc8259#section-8.1>`_.

        To use a custom JSON decoder pass the result of :attr:`HTTPResponse.data` to
        your custom decoder instead.

        If the body of the HTTP response is not decodable to UTF-8, a
        `UnicodeDecodeError` will be raised. If the body of the HTTP response is not a
        valid JSON document, a `json.JSONDecodeError` will be raised.

        Read more :ref:`here <json_content>`.

        :returns: The body of the HTTP response as a Python object.
        """
        data = self.data.decode("utf-8")
        return _json.loads(data)

    def close(self) -> None:
        if not self._closed:
            if isinstance(self._response.body, IOBase):
                self._response.body.close()
            if self._connection:
                self._connection.close()
                self._connection = None
            self._closed = True

    @contextmanager
    def _error_catcher(self) -> typing.Generator[None]:
        """
        Catch Emscripten specific exceptions thrown by fetch.py,
        instead re-raising urllib3 variants, so that low-level exceptions
        are not leaked in the high-level api.

        On exit, release the connection back to the pool.
        """
        from .fetch import _RequestError, _TimeoutError  # avoid circular import

        clean_exit = False

        try:
            yield
            # If no exception is thrown, we should avoid cleaning up
            # unnecessarily.
            clean_exit = True
        except _TimeoutError as e:
            raise TimeoutError(str(e))
        except _RequestError as e:
            raise HTTPException(str(e))
        finally:
            # If we didn't terminate cleanly, we need to throw away our
            # connection.
            if not clean_exit:
                # The response may not be closed but we're not going to use it
                # anymore so close it now
                if (
                    isinstance(self._response.body, IOBase)
                    and not self._response.body.closed
                ):
                    self._response.body.close()
                # release the connection back to the pool
                self.release_conn()
            else:
                # If we have read everything from the response stream,
                # return the connection back to the pool.
                if (
                    isinstance(self._response.body, IOBase)
                    and self._response.body.closed
                ):
                    self.release_conn()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\formatter.py ===
from __future__ import annotations
from typing import Callable, Dict, Iterable, Optional, Set, Tuple, TYPE_CHECKING, Union
from typing_extensions import TypeAlias
from bs4.dammit import EntitySubstitution

if TYPE_CHECKING:
    from bs4._typing import _AttributeValue


class Formatter(EntitySubstitution):
    """Describes a strategy to use when outputting a parse tree to a string.

    Some parts of this strategy come from the distinction between
    HTML4, HTML5, and XML. Others are configurable by the user.

    Formatters are passed in as the `formatter` argument to methods
    like `bs4.element.Tag.encode`. Most people won't need to
    think about formatters, and most people who need to think about
    them can pass in one of these predefined strings as `formatter`
    rather than making a new Formatter object:

    For HTML documents:
     * 'html' - HTML entity substitution for generic HTML documents. (default)
     * 'html5' - HTML entity substitution for HTML5 documents, as
                 well as some optimizations in the way tags are rendered.
     * 'html5-4.12.0' - The version of the 'html5' formatter used prior to
                        Beautiful Soup 4.13.0.
     * 'minimal' - Only make the substitutions necessary to guarantee
                   valid HTML.
     * None - Do not perform any substitution. This will be faster
              but may result in invalid markup.

    For XML documents:
     * 'html' - Entity substitution for XHTML documents.
     * 'minimal' - Only make the substitutions necessary to guarantee
                   valid XML. (default)
     * None - Do not perform any substitution. This will be faster
              but may result in invalid markup.

    """

    #: Constant name denoting HTML markup
    HTML: str = "html"

    #: Constant name denoting XML markup
    XML: str = "xml"

    #: Default values for the various constructor options when the
    #: markup language is HTML.
    HTML_DEFAULTS: Dict[str, Set[str]] = dict(
        cdata_containing_tags=set(["script", "style"]),
    )

    language: Optional[str]  #: :meta private:
    entity_substitution: Optional[_EntitySubstitutionFunction]  #: :meta private:
    void_element_close_prefix: str  #: :meta private:
    cdata_containing_tags: Set[str]  #: :meta private:
    indent: str  #: :meta private:

    #: If this is set to true by the constructor, then attributes whose
    #: values are sent to the empty string will be treated as HTML
    #: boolean attributes. (Attributes whose value is None are always
    #: rendered this way.)
    empty_attributes_are_booleans: bool

    def _default(
        self, language: str, value: Optional[Set[str]], kwarg: str
    ) -> Set[str]:
        if value is not None:
            return value
        if language == self.XML:
            # When XML is the markup language in use, all of the
            # defaults are the empty list.
            return set()

        # Otherwise, it depends on what's in HTML_DEFAULTS.
        return self.HTML_DEFAULTS[kwarg]

    def __init__(
        self,
        language: Optional[str] = None,
        entity_substitution: Optional[_EntitySubstitutionFunction] = None,
        void_element_close_prefix: str = "/",
        cdata_containing_tags: Optional[Set[str]] = None,
        empty_attributes_are_booleans: bool = False,
        indent: Union[int,str] = 1,
    ):
        r"""Constructor.

        :param language: This should be `Formatter.XML` if you are formatting
           XML markup and `Formatter.HTML` if you are formatting HTML markup.

        :param entity_substitution: A function to call to replace special
           characters with XML/HTML entities. For examples, see
           bs4.dammit.EntitySubstitution.substitute_html and substitute_xml.
        :param void_element_close_prefix: By default, void elements
           are represented as <tag/> (XML rules) rather than <tag>
           (HTML rules). To get <tag>, pass in the empty string.
        :param cdata_containing_tags: The set of tags that are defined
           as containing CDATA in this dialect. For example, in HTML,
           <script> and <style> tags are defined as containing CDATA,
           and their contents should not be formatted.
        :param empty_attributes_are_booleans: If this is set to true,
          then attributes whose values are sent to the empty string
          will be treated as `HTML boolean
          attributes<https://dev.w3.org/html5/spec-LC/common-microsyntaxes.html#boolean-attributes>`_. (Attributes
          whose value is None are always rendered this way.)
        :param indent: If indent is a non-negative integer or string,
            then the contents of elements will be indented
            appropriately when pretty-printing. An indent level of 0,
            negative, or "" will only insert newlines. Using a
            positive integer indent indents that many spaces per
            level. If indent is a string (such as "\t"), that string
            is used to indent each level. The default behavior is to
            indent one space per level.

        """
        self.language = language or self.HTML
        self.entity_substitution = entity_substitution
        self.void_element_close_prefix = void_element_close_prefix
        self.cdata_containing_tags = self._default(
            self.language, cdata_containing_tags, "cdata_containing_tags"
        )
        self.empty_attributes_are_booleans = empty_attributes_are_booleans
        if indent is None:
            indent = 0
        indent_str: str
        if isinstance(indent, int):
            if indent < 0:
                indent = 0
            indent_str = " " * indent
        elif isinstance(indent, str):
            indent_str = indent
        else:
            indent_str = " "
        self.indent = indent_str

    def substitute(self, ns: str) -> str:
        """Process a string that needs to undergo entity substitution.
        This may be a string encountered in an attribute value or as
        text.

        :param ns: A string.
        :return: The same string but with certain characters replaced by named
           or numeric entities.
        """
        if not self.entity_substitution:
            return ns
        from .element import NavigableString

        if (
            isinstance(ns, NavigableString)
            and ns.parent is not None
            and ns.parent.name in self.cdata_containing_tags
        ):
            # Do nothing.
            return ns
        # Substitute.
        return self.entity_substitution(ns)

    def attribute_value(self, value: str) -> str:
        """Process the value of an attribute.

        :param ns: A string.
        :return: A string with certain characters replaced by named
           or numeric entities.
        """
        return self.substitute(value)

    def attributes(
        self, tag: bs4.element.Tag
    ) -> Iterable[Tuple[str, Optional[_AttributeValue]]]:
        """Reorder a tag's attributes however you want.

        By default, attributes are sorted alphabetically. This makes
        behavior consistent between Python 2 and Python 3, and preserves
        backwards compatibility with older versions of Beautiful Soup.

        If `empty_attributes_are_booleans` is True, then
        attributes whose values are set to the empty string will be
        treated as boolean attributes.
        """
        if tag.attrs is None:
            return []

        items: Iterable[Tuple[str, _AttributeValue]] = list(tag.attrs.items())
        return sorted(
            (k, (None if self.empty_attributes_are_booleans and v == "" else v))
            for k, v in items
        )


class HTMLFormatter(Formatter):
    """A generic Formatter for HTML."""

    REGISTRY: Dict[Optional[str], HTMLFormatter] = {}

    def __init__(
        self,
        entity_substitution: Optional[_EntitySubstitutionFunction] = None,
        void_element_close_prefix: str = "/",
        cdata_containing_tags: Optional[Set[str]] = None,
        empty_attributes_are_booleans: bool = False,
        indent: Union[int,str] = 1,
    ):
        super(HTMLFormatter, self).__init__(
            self.HTML,
            entity_substitution,
            void_element_close_prefix,
            cdata_containing_tags,
            empty_attributes_are_booleans,
            indent=indent
        )


class XMLFormatter(Formatter):
    """A generic Formatter for XML."""

    REGISTRY: Dict[Optional[str], XMLFormatter] = {}

    def __init__(
        self,
        entity_substitution: Optional[_EntitySubstitutionFunction] = None,
        void_element_close_prefix: str = "/",
        cdata_containing_tags: Optional[Set[str]] = None,
        empty_attributes_are_booleans: bool = False,
        indent: Union[int,str] = 1,
    ):
        super(XMLFormatter, self).__init__(
            self.XML,
            entity_substitution,
            void_element_close_prefix,
            cdata_containing_tags,
            empty_attributes_are_booleans,
            indent=indent,
        )


# Set up aliases for the default formatters.
HTMLFormatter.REGISTRY["html"] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html
)

HTMLFormatter.REGISTRY["html5"] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html5,
    void_element_close_prefix="",
    empty_attributes_are_booleans=True,
)
HTMLFormatter.REGISTRY["html5-4.12"] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html,
    void_element_close_prefix="",
    empty_attributes_are_booleans=True,
)
HTMLFormatter.REGISTRY["minimal"] = HTMLFormatter(
    entity_substitution=EntitySubstitution.substitute_xml
)
HTMLFormatter.REGISTRY[None] = HTMLFormatter(entity_substitution=None)
XMLFormatter.REGISTRY["html"] = XMLFormatter(
    entity_substitution=EntitySubstitution.substitute_html
)
XMLFormatter.REGISTRY["minimal"] = XMLFormatter(
    entity_substitution=EntitySubstitution.substitute_xml
)

XMLFormatter.REGISTRY[None] = XMLFormatter(entity_substitution=None)

# Define type aliases to improve readability.
#

#: A function to call to replace special characters with XML or HTML
#: entities.
_EntitySubstitutionFunction: TypeAlias = Callable[[str], str]

# Many of the output-centered methods take an argument that can either
# be a Formatter object or the name of a Formatter to be looked up.
_FormatterOrName = Union[Formatter, str]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\integerring.py ===
"""Implementation of :class:`IntegerRing` class. """

from sympy.external.gmpy import MPZ, GROUND_TYPES

from sympy.core.numbers import int_valued
from sympy.polys.domains.groundtypes import (
    SymPyInteger,
    factorial,
    gcdex, gcd, lcm, sqrt, is_square, sqrtrem,
)

from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.ring import Ring
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

import math

@public
class IntegerRing(Ring, CharacteristicZero, SimpleDomain):
    r"""The domain ``ZZ`` representing the integers `\mathbb{Z}`.

    The :py:class:`IntegerRing` class represents the ring of integers as a
    :py:class:`~.Domain` in the domain system. :py:class:`IntegerRing` is a
    super class of :py:class:`PythonIntegerRing` and
    :py:class:`GMPYIntegerRing` one of which will be the implementation for
    :ref:`ZZ` depending on whether or not ``gmpy`` or ``gmpy2`` is installed.

    See also
    ========

    Domain
    """

    rep = 'ZZ'
    alias = 'ZZ'
    dtype = MPZ
    zero = dtype(0)
    one = dtype(1)
    tp = type(one)


    is_IntegerRing = is_ZZ = True
    is_Numerical = True
    is_PID = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self):
        """Allow instantiation of this domain. """

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        if isinstance(other, IntegerRing):
            return True
        else:
            return NotImplemented

    def __hash__(self):
        """Compute a hash value for this domain. """
        return hash('ZZ')

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to ``dtype``. """
        if a.is_Integer:
            return MPZ(a.p)
        elif int_valued(a):
            return MPZ(int(a))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def get_field(self):
        r"""Return the associated field of fractions :ref:`QQ`

        Returns
        =======

        :ref:`QQ`:
            The associated field of fractions :ref:`QQ`, a
            :py:class:`~.Domain` representing the rational numbers
            `\mathbb{Q}`.

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.get_field()
        QQ
        """
        from sympy.polys.domains import QQ
        return QQ

    def algebraic_field(self, *extension, alias=None):
        r"""Returns an algebraic field, i.e. `\mathbb{Q}(\alpha, \ldots)`.

        Parameters
        ==========

        *extension : One or more :py:class:`~.Expr`.
            Generators of the extension. These should be expressions that are
            algebraic over `\mathbb{Q}`.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            If provided, this will be used as the alias symbol for the
            primitive element of the returned :py:class:`~.AlgebraicField`.

        Returns
        =======

        :py:class:`~.AlgebraicField`
            A :py:class:`~.Domain` representing the algebraic field extension.

        Examples
        ========

        >>> from sympy import ZZ, sqrt
        >>> ZZ.algebraic_field(sqrt(2))
        QQ<sqrt(2)>
        """
        return self.get_field().algebraic_field(*extension, alias=alias)

    def from_AlgebraicField(K1, a, K0):
        """Convert a :py:class:`~.ANP` object to :ref:`ZZ`.

        See :py:meth:`~.Domain.convert`.
        """
        if a.is_ground:
            return K1.convert(a.LC(), K0.dom)

    def log(self, a, b):
        r"""Logarithm of *a* to the base *b*.

        Parameters
        ==========

        a: number
        b: number

        Returns
        =======

        $\\lfloor\log(a, b)\\rfloor$:
            Floor of the logarithm of *a* to the base *b*

        Examples
        ========

        >>> from sympy import ZZ
        >>> ZZ.log(ZZ(8), ZZ(2))
        3
        >>> ZZ.log(ZZ(9), ZZ(2))
        3

        Notes
        =====

        This function uses ``math.log`` which is based on ``float`` so it will
        fail for large integer arguments.
        """
        return self.dtype(int(math.log(int(a), b)))

    def from_FF(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_FF_python(K1, a, K0):
        """Convert ``ModularInteger(int)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

    def from_ZZ(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return MPZ(a)

    def from_ZZ_python(K1, a, K0):
        """Convert Python's ``int`` to GMPY's ``mpz``. """
        return MPZ(a)

    def from_QQ(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_QQ_python(K1, a, K0):
        """Convert Python's ``Fraction`` to GMPY's ``mpz``. """
        if a.denominator == 1:
            return MPZ(a.numerator)

    def from_FF_gmpy(K1, a, K0):
        """Convert ``ModularInteger(mpz)`` to GMPY's ``mpz``. """
        return MPZ(K0.to_int(a))

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
            # XXX: If MPZ is flint.fmpz and p is a gmpy2.mpz, then we need
            # to convert via int because fmpz and mpz do not know about each
            # other.
            return MPZ(int(p))

    def from_GaussianIntegerRing(K1, a, K0):
        if a.y == 0:
            return a.x

    def from_EX(K1, a, K0):
        """Convert ``Expression`` to GMPY's ``mpz``. """
        if a.is_Integer:
            return K1.from_sympy(a)

    def gcdex(self, a, b):
        """Compute extended GCD of ``a`` and ``b``. """
        h, s, t = gcdex(a, b)
        # XXX: This conditional logic should be handled somewhere else.
        if GROUND_TYPES == 'gmpy':
            return s, t, h
        else:
            return h, s, t

    def gcd(self, a, b):
        """Compute GCD of ``a`` and ``b``. """
        return gcd(a, b)

    def lcm(self, a, b):
        """Compute LCM of ``a`` and ``b``. """
        return lcm(a, b)

    def sqrt(self, a):
        """Compute square root of ``a``. """
        return sqrt(a)

    def is_square(self, a):
        """Return ``True`` if ``a`` is a square.

        Explanation
        ===========
        An integer is a square if and only if there exists an integer
        ``b`` such that ``b * b == a``.
        """
        return is_square(a)

    def exsqrt(self, a):
        """Non-negative square root of ``a`` if ``a`` is a square.

        See also
        ========
        is_square
        """
        if a < 0:
            return None
        root, rem = sqrtrem(a)
        if rem != 0:
            return None
        return root

    def factorial(self, a):
        """Compute factorial of ``a``. """
        return factorial(a)


ZZ = IntegerRing()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\type_e.py ===
import itertools

from .cartan_type import Standard_Cartan
from sympy.core.backend import eye, Rational
from sympy.core.singleton import S

class TypeE(Standard_Cartan):

    def __new__(cls, n):
        if n < 6 or n > 8:
            raise ValueError("Invalid value of n")
        return Standard_Cartan.__new__(cls, "E", n)

    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("E6")
        >>> c.dimension()
        8
        """

        return 8

    def basic_root(self, i, j):
        """
        This is a method just to generate roots
        with a -1 in the ith position and a 1
        in the jth position.

        """

        root = [0]*8
        root[i] = -1
        root[j] = 1
        return root

    def simple_root(self, i):
        """
        Every Lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        This method returns the ith simple root for E_n.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("E6")
        >>> c.simple_root(2)
        [1, 1, 0, 0, 0, 0, 0, 0]
        """
        n = self.n
        if i == 1:
            root = [-0.5]*8
            root[0] = 0.5
            root[7] = 0.5
            return root
        elif i == 2:
            root = [0]*8
            root[1] = 1
            root[0] = 1
            return root
        else:
            if i in (7, 8) and n == 6:
                raise ValueError("E6 only has six simple roots!")
            if i == 8 and n == 7:
                raise ValueError("E7 only has seven simple roots!")

            return self.basic_root(i - 3, i - 2)

    def positive_roots(self):
        """
        This method generates all the positive roots of
        A_n.  This is half of all of the roots of E_n;
        by multiplying all the positive roots by -1 we
        get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}
        """
        n = self.n
        neghalf = Rational(-1, 2)
        poshalf = S.Half
        if n == 6:
            posroots = {}
            k = 0
            for i in range(n-1):
                for j in range(i+1, n-1):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root

            root = [poshalf, poshalf, poshalf, poshalf, poshalf,
                    neghalf, neghalf, poshalf]
            for a, b, c, d, e in itertools.product(
                    range(2), range(2), range(2), range(2), range(2)):
                if (a + b + c + d + e)%2 == 0:
                    k += 1
                    if a == 1:
                        root[0] = neghalf
                    if b == 1:
                        root[1] = neghalf
                    if c == 1:
                        root[2] = neghalf
                    if d == 1:
                        root[3] = neghalf
                    if e == 1:
                        root[4] = neghalf
                    posroots[k] = root[:]
            return posroots
        if n == 7:
            posroots = {}
            k = 0
            for i in range(n-1):
                for j in range(i+1, n-1):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root

            k += 1
            posroots[k] = [0, 0, 0, 0, 0, 1, 1, 0]
            root = [poshalf, poshalf, poshalf, poshalf, poshalf,
                    neghalf, neghalf, poshalf]
            for a, b, c, d, e, f in itertools.product(
                    range(2), range(2), range(2), range(2), range(2), range(2)):
                if (a + b + c + d + e + f)%2 == 0:
                    k += 1
                    if a == 1:
                        root[0] = neghalf
                    if b == 1:
                        root[1] = neghalf
                    if c == 1:
                        root[2] = neghalf
                    if d == 1:
                        root[3] = neghalf
                    if e == 1:
                        root[4] = neghalf
                    if f == 1:
                        root[5] = poshalf
                    posroots[k] = root[:]
            return posroots
        if n == 8:
            posroots = {}
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    k += 1
                    root = self.basic_root(i, j)
                    posroots[k] = root
                    k += 1
                    root = self.basic_root(i, j)
                    root[i] = 1
                    posroots[k] = root

            root = [poshalf, poshalf, poshalf, poshalf, poshalf,
                    neghalf, neghalf, poshalf]
            for a, b, c, d, e, f, g in itertools.product(
                    range(2), range(2), range(2), range(2), range(2),
                    range(2), range(2)):
                if (a + b + c + d + e + f + g)%2 == 0:
                    k += 1
                    if a == 1:
                        root[0] = neghalf
                    if b == 1:
                        root[1] = neghalf
                    if c == 1:
                        root[2] = neghalf
                    if d == 1:
                        root[3] = neghalf
                    if e == 1:
                        root[4] = neghalf
                    if f == 1:
                        root[5] = poshalf
                    if g == 1:
                        root[6] = poshalf
                    posroots[k] = root[:]
            return posroots



    def roots(self):
        """
        Returns the total number of roots of E_n
        """

        n = self.n
        if n == 6:
            return 72
        if n == 7:
            return 126
        if n == 8:
            return 240


    def cartan_matrix(self):
        """
        Returns the Cartan matrix for G_2
        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('A4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]])


        """

        n = self.n
        m = 2*eye(n)
        for i in range(3, n - 1):
            m[i, i+1] = -1
            m[i, i-1] = -1
        m[0, 2] = m[2, 0] = -1
        m[1, 3] = m[3, 1] = -1
        m[2, 3] = -1
        m[n-1, n-2] = -1
        return m


    def basis(self):
        """
        Returns the number of independent generators of E_n
        """

        n = self.n
        if n == 6:
            return 78
        if n == 7:
            return 133
        if n == 8:
            return 248

    def dynkin_diagram(self):
        n = self.n
        diag = " "*8 + str(2) + "\n"
        diag += " "*8 + "0\n"
        diag += " "*8 + "|\n"
        diag += " "*8 + "|\n"
        diag += "---".join("0" for i in range(1, n)) + "\n"
        diag += "1   " + "   ".join(str(i) for i in range(3, n+1))
        return diag