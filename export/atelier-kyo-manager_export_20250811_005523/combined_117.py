
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\retry.py ===
from __future__ import absolute_import

import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile

from ..exceptions import (
    ConnectTimeoutError,
    InvalidHeader,
    MaxRetryError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    ResponseError,
)
from ..packages import six

log = logging.getLogger(__name__)


# Data structure for representing the metadata of requests that result in a retry.
RequestHistory = namedtuple(
    "RequestHistory", ["method", "url", "error", "status", "redirect_location"]
)


# TODO: In v2 we can remove this sentinel and metaclass with deprecated options.
_Default = object()


class _RetryMeta(type):
    @property
    def DEFAULT_METHOD_WHITELIST(cls):
        warnings.warn(
            "Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_ALLOWED_METHODS' instead",
            DeprecationWarning,
        )
        return cls.DEFAULT_ALLOWED_METHODS

    @DEFAULT_METHOD_WHITELIST.setter
    def DEFAULT_METHOD_WHITELIST(cls, value):
        warnings.warn(
            "Using 'Retry.DEFAULT_METHOD_WHITELIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_ALLOWED_METHODS' instead",
            DeprecationWarning,
        )
        cls.DEFAULT_ALLOWED_METHODS = value

    @property
    def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls):
        warnings.warn(
            "Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead",
            DeprecationWarning,
        )
        return cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT

    @DEFAULT_REDIRECT_HEADERS_BLACKLIST.setter
    def DEFAULT_REDIRECT_HEADERS_BLACKLIST(cls, value):
        warnings.warn(
            "Using 'Retry.DEFAULT_REDIRECT_HEADERS_BLACKLIST' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_REMOVE_HEADERS_ON_REDIRECT' instead",
            DeprecationWarning,
        )
        cls.DEFAULT_REMOVE_HEADERS_ON_REDIRECT = value

    @property
    def BACKOFF_MAX(cls):
        warnings.warn(
            "Using 'Retry.BACKOFF_MAX' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_BACKOFF_MAX' instead",
            DeprecationWarning,
        )
        return cls.DEFAULT_BACKOFF_MAX

    @BACKOFF_MAX.setter
    def BACKOFF_MAX(cls, value):
        warnings.warn(
            "Using 'Retry.BACKOFF_MAX' is deprecated and "
            "will be removed in v2.0. Use 'Retry.DEFAULT_BACKOFF_MAX' instead",
            DeprecationWarning,
        )
        cls.DEFAULT_BACKOFF_MAX = value


@six.add_metaclass(_RetryMeta)
class Retry(object):
    """Retry configuration.

    Each retry attempt will create a new Retry object with updated values, so
    they can be safely reused.

    Retries can be defined as a default for a pool::

        retries = Retry(connect=5, read=2, redirect=5)
        http = PoolManager(retries=retries)
        response = http.request('GET', 'http://example.com/')

    Or per-request (which overrides the default for the pool)::

        response = http.request('GET', 'http://example.com/', retries=Retry(10))

    Retries can be disabled by passing ``False``::

        response = http.request('GET', 'http://example.com/', retries=False)

    Errors will be wrapped in :class:`~urllib3.exceptions.MaxRetryError` unless
    retries are disabled, in which case the causing exception will be raised.

    :param int total:
        Total number of retries to allow. Takes precedence over other counts.

        Set to ``None`` to remove this constraint and fall back on other
        counts.

        Set to ``0`` to fail on the first retry.

        Set to ``False`` to disable and imply ``raise_on_redirect=False``.

    :param int connect:
        How many connection-related errors to retry on.

        These are errors raised before the request is sent to the remote server,
        which we assume has not triggered the server to process the request.

        Set to ``0`` to fail on the first retry of this type.

    :param int read:
        How many times to retry on read errors.

        These errors are raised after the request was sent to the server, so the
        request may have side-effects.

        Set to ``0`` to fail on the first retry of this type.

    :param int redirect:
        How many redirects to perform. Limit this to avoid infinite redirect
        loops.

        A redirect is a HTTP response with a status code 301, 302, 303, 307 or
        308.

        Set to ``0`` to fail on the first retry of this type.

        Set to ``False`` to disable and imply ``raise_on_redirect=False``.

    :param int status:
        How many times to retry on bad status codes.

        These are retries made on responses, where status code matches
        ``status_forcelist``.

        Set to ``0`` to fail on the first retry of this type.

    :param int other:
        How many times to retry on other errors.

        Other errors are errors that are not connect, read, redirect or status errors.
        These errors might be raised after the request was sent to the server, so the
        request might have side-effects.

        Set to ``0`` to fail on the first retry of this type.

        If ``total`` is not set, it's a good idea to set this to 0 to account
        for unexpected edge cases and avoid infinite retry loops.

    :param iterable allowed_methods:
        Set of uppercased HTTP method verbs that we should retry on.

        By default, we only retry on methods which are considered to be
        idempotent (multiple requests with the same parameters end with the
        same state). See :attr:`Retry.DEFAULT_ALLOWED_METHODS`.

        Set to a ``False`` value to retry on any verb.

        .. warning::

            Previously this parameter was named ``method_whitelist``, that
            usage is deprecated in v1.26.0 and will be removed in v2.0.

    :param iterable status_forcelist:
        A set of integer HTTP status codes that we should force a retry on.
        A retry is initiated if the request method is in ``allowed_methods``
        and the response status code is in ``status_forcelist``.

        By default, this is disabled with ``None``.

    :param float backoff_factor:
        A backoff factor to apply between attempts after the second try
        (most errors are resolved immediately by a second try without a
        delay). urllib3 will sleep for::

            {backoff factor} * (2 ** ({number of total retries} - 1))

        seconds. If the backoff_factor is 0.1, then :func:`.sleep` will sleep
        for [0.0s, 0.2s, 0.4s, ...] between retries. It will never be longer
        than :attr:`Retry.DEFAULT_BACKOFF_MAX`.

        By default, backoff is disabled (set to 0).

    :param bool raise_on_redirect: Whether, if the number of redirects is
        exhausted, to raise a MaxRetryError, or to return a response with a
        response code in the 3xx range.

    :param bool raise_on_status: Similar meaning to ``raise_on_redirect``:
        whether we should raise an exception, or return a response,
        if status falls in ``status_forcelist`` range and retries have
        been exhausted.

    :param tuple history: The history of the request encountered during
        each call to :meth:`~Retry.increment`. The list is in the order
        the requests occurred. Each list item is of class :class:`RequestHistory`.

    :param bool respect_retry_after_header:
        Whether to respect Retry-After header on status codes defined as
        :attr:`Retry.RETRY_AFTER_STATUS_CODES` or not.

    :param iterable remove_headers_on_redirect:
        Sequence of headers to remove from the request when a response
        indicating a redirect is returned before firing off the redirected
        request.
    """

    #: Default methods to be used for ``allowed_methods``
    DEFAULT_ALLOWED_METHODS = frozenset(
        ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )

    #: Default status codes to be used for ``status_forcelist``
    RETRY_AFTER_STATUS_CODES = frozenset([413, 429, 503])

    #: Default headers to be used for ``remove_headers_on_redirect``
    DEFAULT_REMOVE_HEADERS_ON_REDIRECT = frozenset(
        ["Cookie", "Authorization", "Proxy-Authorization"]
    )

    #: Maximum backoff time.
    DEFAULT_BACKOFF_MAX = 120

    def __init__(
        self,
        total=10,
        connect=None,
        read=None,
        redirect=None,
        status=None,
        other=None,
        allowed_methods=_Default,
        status_forcelist=None,
        backoff_factor=0,
        raise_on_redirect=True,
        raise_on_status=True,
        history=None,
        respect_retry_after_header=True,
        remove_headers_on_redirect=_Default,
        # TODO: Deprecated, remove in v2.0
        method_whitelist=_Default,
    ):

        if method_whitelist is not _Default:
            if allowed_methods is not _Default:
                raise ValueError(
                    "Using both 'allowed_methods' and "
                    "'method_whitelist' together is not allowed. "
                    "Instead only use 'allowed_methods'"
                )
            warnings.warn(
                "Using 'method_whitelist' with Retry is deprecated and "
                "will be removed in v2.0. Use 'allowed_methods' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            allowed_methods = method_whitelist
        if allowed_methods is _Default:
            allowed_methods = self.DEFAULT_ALLOWED_METHODS
        if remove_headers_on_redirect is _Default:
            remove_headers_on_redirect = self.DEFAULT_REMOVE_HEADERS_ON_REDIRECT

        self.total = total
        self.connect = connect
        self.read = read
        self.status = status
        self.other = other

        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False

        self.redirect = redirect
        self.status_forcelist = status_forcelist or set()
        self.allowed_methods = allowed_methods
        self.backoff_factor = backoff_factor
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = history or tuple()
        self.respect_retry_after_header = respect_retry_after_header
        self.remove_headers_on_redirect = frozenset(
            [h.lower() for h in remove_headers_on_redirect]
        )

    def new(self, **kw):
        params = dict(
            total=self.total,
            connect=self.connect,
            read=self.read,
            redirect=self.redirect,
            status=self.status,
            other=self.other,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            history=self.history,
            remove_headers_on_redirect=self.remove_headers_on_redirect,
            respect_retry_after_header=self.respect_retry_after_header,
        )

        # TODO: If already given in **kw we use what's given to us
        # If not given we need to figure out what to pass. We decide
        # based on whether our class has the 'method_whitelist' property
        # and if so we pass the deprecated 'method_whitelist' otherwise
        # we use 'allowed_methods'. Remove in v2.0
        if "method_whitelist" not in kw and "allowed_methods" not in kw:
            if "method_whitelist" in self.__dict__:
                warnings.warn(
                    "Using 'method_whitelist' with Retry is deprecated and "
                    "will be removed in v2.0. Use 'allowed_methods' instead",
                    DeprecationWarning,
                )
                params["method_whitelist"] = self.allowed_methods
            else:
                params["allowed_methods"] = self.allowed_methods

        params.update(kw)
        return type(self)(**params)

    @classmethod
    def from_int(cls, retries, redirect=True, default=None):
        """Backwards-compatibility for the old retries format."""
        if retries is None:
            retries = default if default is not None else cls.DEFAULT

        if isinstance(retries, Retry):
            return retries

        redirect = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect)
        log.debug("Converted retries value: %r -> %r", retries, new_retries)
        return new_retries

    def get_backoff_time(self):
        """Formula for computing the current backoff

        :rtype: float
        """
        # We want to consider only the last consecutive errors sequence (Ignore redirects).
        consecutive_errors_len = len(
            list(
                takewhile(lambda x: x.redirect_location is None, reversed(self.history))
            )
        )
        if consecutive_errors_len <= 1:
            return 0

        backoff_value = self.backoff_factor * (2 ** (consecutive_errors_len - 1))
        return min(self.DEFAULT_BACKOFF_MAX, backoff_value)

    def parse_retry_after(self, retry_after):
        # Whitespace: https://tools.ietf.org/html/rfc7230#section-3.2.4
        if re.match(r"^\s*[0-9]+\s*$", retry_after):
            seconds = int(retry_after)
        else:
            retry_date_tuple = email.utils.parsedate_tz(retry_after)
            if retry_date_tuple is None:
                raise InvalidHeader("Invalid Retry-After header: %s" % retry_after)
            if retry_date_tuple[9] is None:  # Python 2
                # Assume UTC if no timezone was specified
                # On Python2.7, parsedate_tz returns None for a timezone offset
                # instead of 0 if no timezone is given, where mktime_tz treats
                # a None timezone offset as local time.
                retry_date_tuple = retry_date_tuple[:9] + (0,) + retry_date_tuple[10:]

            retry_date = email.utils.mktime_tz(retry_date_tuple)
            seconds = retry_date - time.time()

        if seconds < 0:
            seconds = 0

        return seconds

    def get_retry_after(self, response):
        """Get the value of Retry-After in seconds."""

        retry_after = response.headers.get("Retry-After")

        if retry_after is None:
            return None

        return self.parse_retry_after(retry_after)

    def sleep_for_retry(self, response=None):
        retry_after = self.get_retry_after(response)
        if retry_after:
            time.sleep(retry_after)
            return True

        return False

    def _sleep_backoff(self):
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)

    def sleep(self, response=None):
        """Sleep between retry attempts.

        This method will respect a server's ``Retry-After`` response header
        and sleep the duration of the time requested. If that is not present, it
        will use an exponential backoff. By default, the backoff factor is 0 and
        this method will return immediately.
        """

        if self.respect_retry_after_header and response:
            slept = self.sleep_for_retry(response)
            if slept:
                return

        self._sleep_backoff()

    def _is_connection_error(self, err):
        """Errors when we're fairly sure that the server did not receive the
        request, so it should be safe to retry.
        """
        if isinstance(err, ProxyError):
            err = err.original_error
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err):
        """Errors that occur after the request has been started, so we should
        assume that the server began processing it.
        """
        return isinstance(err, (ReadTimeoutError, ProtocolError))

    def _is_method_retryable(self, method):
        """Checks if a given HTTP method should be retried upon, depending if
        it is included in the allowed_methods
        """
        # TODO: For now favor if the Retry implementation sets its own method_whitelist
        # property outside of our constructor to avoid breaking custom implementations.
        if "method_whitelist" in self.__dict__:
            warnings.warn(
                "Using 'method_whitelist' with Retry is deprecated and "
                "will be removed in v2.0. Use 'allowed_methods' instead",
                DeprecationWarning,
            )
            allowed_methods = self.method_whitelist
        else:
            allowed_methods = self.allowed_methods

        if allowed_methods and method.upper() not in allowed_methods:
            return False
        return True

    def is_retry(self, method, status_code, has_retry_after=False):
        """Is this method/status code retryable? (Based on allowlists and control
        variables such as the number of total retries to allow, whether to
        respect the Retry-After header, whether this header is present, and
        whether the returned status code is on the list of status codes to
        be retried upon on the presence of the aforementioned header)
        """
        if not self._is_method_retryable(method):
            return False

        if self.status_forcelist and status_code in self.status_forcelist:
            return True

        return (
            self.total
            and self.respect_retry_after_header
            and has_retry_after
            and (status_code in self.RETRY_AFTER_STATUS_CODES)
        )

    def is_exhausted(self):
        """Are we out of retries?"""
        retry_counts = (
            self.total,
            self.connect,
            self.read,
            self.redirect,
            self.status,
            self.other,
        )
        retry_counts = list(filter(None, retry_counts))
        if not retry_counts:
            return False

        return min(retry_counts) < 0

    def increment(
        self,
        method=None,
        url=None,
        response=None,
        error=None,
        _pool=None,
        _stacktrace=None,
    ):
        """Return a new Retry object with incremented retry counters.

        :param response: A response object, or None, if the server did not
            return a response.
        :type response: :class:`~urllib3.response.HTTPResponse`
        :param Exception error: An error encountered during the request, or
            None if the response was received successfully.

        :return: A new ``Retry`` object.
        """
        if self.total is False and error:
            # Disabled, indicate to re-raise the error.
            raise six.reraise(type(error), error, _stacktrace)

        total = self.total
        if total is not None:
            total -= 1

        connect = self.connect
        read = self.read
        redirect = self.redirect
        status_count = self.status
        other = self.other
        cause = "unknown"
        status = None
        redirect_location = None

        if error and self._is_connection_error(error):
            # Connect retry?
            if connect is False:
                raise six.reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1

        elif error and self._is_read_error(error):
            # Read retry?
            if read is False or not self._is_method_retryable(method):
                raise six.reraise(type(error), error, _stacktrace)
            elif read is not None:
                read -= 1

        elif error:
            # Other retry?
            if other is not None:
                other -= 1

        elif response and response.get_redirect_location():
            # Redirect retry?
            if redirect is not None:
                redirect -= 1
            cause = "too many redirects"
            redirect_location = response.get_redirect_location()
            status = response.status

        else:
            # Incrementing because of a server error like a 500 in
            # status_forcelist and the given method is in the allowed_methods
            cause = ResponseError.GENERIC_ERROR
            if response and response.status:
                if status_count is not None:
                    status_count -= 1
                cause = ResponseError.SPECIFIC_ERROR.format(status_code=response.status)
                status = response.status

        history = self.history + (
            RequestHistory(method, url, error, status, redirect_location),
        )

        new_retry = self.new(
            total=total,
            connect=connect,
            read=read,
            redirect=redirect,
            status=status_count,
            other=other,
            history=history,
        )

        if new_retry.is_exhausted():
            raise MaxRetryError(_pool, url, error or ResponseError(cause))

        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)

        return new_retry

    def __repr__(self):
        return (
            "{cls.__name__}(total={self.total}, connect={self.connect}, "
            "read={self.read}, redirect={self.redirect}, status={self.status})"
        ).format(cls=type(self), self=self)

    def __getattr__(self, item):
        if item == "method_whitelist":
            # TODO: Remove this deprecated alias in v2.0
            warnings.warn(
                "Using 'method_whitelist' with Retry is deprecated and "
                "will be removed in v2.0. Use 'allowed_methods' instead",
                DeprecationWarning,
            )
            return self.allowed_methods
        try:
            return getattr(super(Retry, self), item)
        except AttributeError:
            return getattr(Retry, item)


# For backwards compatibility (equivalent to pre-v1.9):
Retry.DEFAULT = Retry(3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\color.py ===
import re
import sys
from colorsys import rgb_to_hls
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

from ._palettes import EIGHT_BIT_PALETTE, STANDARD_PALETTE, WINDOWS_PALETTE
from .color_triplet import ColorTriplet
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME

if TYPE_CHECKING:  # pragma: no cover
    from .terminal_theme import TerminalTheme
    from .text import Text


WINDOWS = sys.platform == "win32"


class ColorSystem(IntEnum):
    """One of the 3 color system supported by terminals."""

    STANDARD = 1
    EIGHT_BIT = 2
    TRUECOLOR = 3
    WINDOWS = 4

    def __repr__(self) -> str:
        return f"ColorSystem.{self.name}"

    def __str__(self) -> str:
        return repr(self)


class ColorType(IntEnum):
    """Type of color stored in Color class."""

    DEFAULT = 0
    STANDARD = 1
    EIGHT_BIT = 2
    TRUECOLOR = 3
    WINDOWS = 4

    def __repr__(self) -> str:
        return f"ColorType.{self.name}"


ANSI_COLOR_NAMES = {
    "black": 0,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "magenta": 5,
    "cyan": 6,
    "white": 7,
    "bright_black": 8,
    "bright_red": 9,
    "bright_green": 10,
    "bright_yellow": 11,
    "bright_blue": 12,
    "bright_magenta": 13,
    "bright_cyan": 14,
    "bright_white": 15,
    "grey0": 16,
    "gray0": 16,
    "navy_blue": 17,
    "dark_blue": 18,
    "blue3": 20,
    "blue1": 21,
    "dark_green": 22,
    "deep_sky_blue4": 25,
    "dodger_blue3": 26,
    "dodger_blue2": 27,
    "green4": 28,
    "spring_green4": 29,
    "turquoise4": 30,
    "deep_sky_blue3": 32,
    "dodger_blue1": 33,
    "green3": 40,
    "spring_green3": 41,
    "dark_cyan": 36,
    "light_sea_green": 37,
    "deep_sky_blue2": 38,
    "deep_sky_blue1": 39,
    "spring_green2": 47,
    "cyan3": 43,
    "dark_turquoise": 44,
    "turquoise2": 45,
    "green1": 46,
    "spring_green1": 48,
    "medium_spring_green": 49,
    "cyan2": 50,
    "cyan1": 51,
    "dark_red": 88,
    "deep_pink4": 125,
    "purple4": 55,
    "purple3": 56,
    "blue_violet": 57,
    "orange4": 94,
    "grey37": 59,
    "gray37": 59,
    "medium_purple4": 60,
    "slate_blue3": 62,
    "royal_blue1": 63,
    "chartreuse4": 64,
    "dark_sea_green4": 71,
    "pale_turquoise4": 66,
    "steel_blue": 67,
    "steel_blue3": 68,
    "cornflower_blue": 69,
    "chartreuse3": 76,
    "cadet_blue": 73,
    "sky_blue3": 74,
    "steel_blue1": 81,
    "pale_green3": 114,
    "sea_green3": 78,
    "aquamarine3": 79,
    "medium_turquoise": 80,
    "chartreuse2": 112,
    "sea_green2": 83,
    "sea_green1": 85,
    "aquamarine1": 122,
    "dark_slate_gray2": 87,
    "dark_magenta": 91,
    "dark_violet": 128,
    "purple": 129,
    "light_pink4": 95,
    "plum4": 96,
    "medium_purple3": 98,
    "slate_blue1": 99,
    "yellow4": 106,
    "wheat4": 101,
    "grey53": 102,
    "gray53": 102,
    "light_slate_grey": 103,
    "light_slate_gray": 103,
    "medium_purple": 104,
    "light_slate_blue": 105,
    "dark_olive_green3": 149,
    "dark_sea_green": 108,
    "light_sky_blue3": 110,
    "sky_blue2": 111,
    "dark_sea_green3": 150,
    "dark_slate_gray3": 116,
    "sky_blue1": 117,
    "chartreuse1": 118,
    "light_green": 120,
    "pale_green1": 156,
    "dark_slate_gray1": 123,
    "red3": 160,
    "medium_violet_red": 126,
    "magenta3": 164,
    "dark_orange3": 166,
    "indian_red": 167,
    "hot_pink3": 168,
    "medium_orchid3": 133,
    "medium_orchid": 134,
    "medium_purple2": 140,
    "dark_goldenrod": 136,
    "light_salmon3": 173,
    "rosy_brown": 138,
    "grey63": 139,
    "gray63": 139,
    "medium_purple1": 141,
    "gold3": 178,
    "dark_khaki": 143,
    "navajo_white3": 144,
    "grey69": 145,
    "gray69": 145,
    "light_steel_blue3": 146,
    "light_steel_blue": 147,
    "yellow3": 184,
    "dark_sea_green2": 157,
    "light_cyan3": 152,
    "light_sky_blue1": 153,
    "green_yellow": 154,
    "dark_olive_green2": 155,
    "dark_sea_green1": 193,
    "pale_turquoise1": 159,
    "deep_pink3": 162,
    "magenta2": 200,
    "hot_pink2": 169,
    "orchid": 170,
    "medium_orchid1": 207,
    "orange3": 172,
    "light_pink3": 174,
    "pink3": 175,
    "plum3": 176,
    "violet": 177,
    "light_goldenrod3": 179,
    "tan": 180,
    "misty_rose3": 181,
    "thistle3": 182,
    "plum2": 183,
    "khaki3": 185,
    "light_goldenrod2": 222,
    "light_yellow3": 187,
    "grey84": 188,
    "gray84": 188,
    "light_steel_blue1": 189,
    "yellow2": 190,
    "dark_olive_green1": 192,
    "honeydew2": 194,
    "light_cyan1": 195,
    "red1": 196,
    "deep_pink2": 197,
    "deep_pink1": 199,
    "magenta1": 201,
    "orange_red1": 202,
    "indian_red1": 204,
    "hot_pink": 206,
    "dark_orange": 208,
    "salmon1": 209,
    "light_coral": 210,
    "pale_violet_red1": 211,
    "orchid2": 212,
    "orchid1": 213,
    "orange1": 214,
    "sandy_brown": 215,
    "light_salmon1": 216,
    "light_pink1": 217,
    "pink1": 218,
    "plum1": 219,
    "gold1": 220,
    "navajo_white1": 223,
    "misty_rose1": 224,
    "thistle1": 225,
    "yellow1": 226,
    "light_goldenrod1": 227,
    "khaki1": 228,
    "wheat1": 229,
    "cornsilk1": 230,
    "grey100": 231,
    "gray100": 231,
    "grey3": 232,
    "gray3": 232,
    "grey7": 233,
    "gray7": 233,
    "grey11": 234,
    "gray11": 234,
    "grey15": 235,
    "gray15": 235,
    "grey19": 236,
    "gray19": 236,
    "grey23": 237,
    "gray23": 237,
    "grey27": 238,
    "gray27": 238,
    "grey30": 239,
    "gray30": 239,
    "grey35": 240,
    "gray35": 240,
    "grey39": 241,
    "gray39": 241,
    "grey42": 242,
    "gray42": 242,
    "grey46": 243,
    "gray46": 243,
    "grey50": 244,
    "gray50": 244,
    "grey54": 245,
    "gray54": 245,
    "grey58": 246,
    "gray58": 246,
    "grey62": 247,
    "gray62": 247,
    "grey66": 248,
    "gray66": 248,
    "grey70": 249,
    "gray70": 249,
    "grey74": 250,
    "gray74": 250,
    "grey78": 251,
    "gray78": 251,
    "grey82": 252,
    "gray82": 252,
    "grey85": 253,
    "gray85": 253,
    "grey89": 254,
    "gray89": 254,
    "grey93": 255,
    "gray93": 255,
}


class ColorParseError(Exception):
    """The color could not be parsed."""


RE_COLOR = re.compile(
    r"""^
\#([0-9a-f]{6})$|
color\(([0-9]{1,3})\)$|
rgb\(([\d\s,]+)\)$
""",
    re.VERBOSE,
)


@rich_repr
class Color(NamedTuple):
    """Terminal color definition."""

    name: str
    """The name of the color (typically the input to Color.parse)."""
    type: ColorType
    """The type of the color."""
    number: Optional[int] = None
    """The color number, if a standard color, or None."""
    triplet: Optional[ColorTriplet] = None
    """A triplet of color components, if an RGB color."""

    def __rich__(self) -> "Text":
        """Displays the actual color if Rich printed."""
        from .style import Style
        from .text import Text

        return Text.assemble(
            f"<color {self.name!r} ({self.type.name.lower()})",
            ("⬤", Style(color=self)),
            " >",
        )

    def __rich_repr__(self) -> Result:
        yield self.name
        yield self.type
        yield "number", self.number, None
        yield "triplet", self.triplet, None

    @property
    def system(self) -> ColorSystem:
        """Get the native color system for this color."""
        if self.type == ColorType.DEFAULT:
            return ColorSystem.STANDARD
        return ColorSystem(int(self.type))

    @property
    def is_system_defined(self) -> bool:
        """Check if the color is ultimately defined by the system."""
        return self.system not in (ColorSystem.EIGHT_BIT, ColorSystem.TRUECOLOR)

    @property
    def is_default(self) -> bool:
        """Check if the color is a default color."""
        return self.type == ColorType.DEFAULT

    def get_truecolor(
        self, theme: Optional["TerminalTheme"] = None, foreground: bool = True
    ) -> ColorTriplet:
        """Get an equivalent color triplet for this color.

        Args:
            theme (TerminalTheme, optional): Optional terminal theme, or None to use default. Defaults to None.
            foreground (bool, optional): True for a foreground color, or False for background. Defaults to True.

        Returns:
            ColorTriplet: A color triplet containing RGB components.
        """

        if theme is None:
            theme = DEFAULT_TERMINAL_THEME
        if self.type == ColorType.TRUECOLOR:
            assert self.triplet is not None
            return self.triplet
        elif self.type == ColorType.EIGHT_BIT:
            assert self.number is not None
            return EIGHT_BIT_PALETTE[self.number]
        elif self.type == ColorType.STANDARD:
            assert self.number is not None
            return theme.ansi_colors[self.number]
        elif self.type == ColorType.WINDOWS:
            assert self.number is not None
            return WINDOWS_PALETTE[self.number]
        else:  # self.type == ColorType.DEFAULT:
            assert self.number is None
            return theme.foreground_color if foreground else theme.background_color

    @classmethod
    def from_ansi(cls, number: int) -> "Color":
        """Create a Color number from it's 8-bit ansi number.

        Args:
            number (int): A number between 0-255 inclusive.

        Returns:
            Color: A new Color instance.
        """
        return cls(
            name=f"color({number})",
            type=(ColorType.STANDARD if number < 16 else ColorType.EIGHT_BIT),
            number=number,
        )

    @classmethod
    def from_triplet(cls, triplet: "ColorTriplet") -> "Color":
        """Create a truecolor RGB color from a triplet of values.

        Args:
            triplet (ColorTriplet): A color triplet containing red, green and blue components.

        Returns:
            Color: A new color object.
        """
        return cls(name=triplet.hex, type=ColorType.TRUECOLOR, triplet=triplet)

    @classmethod
    def from_rgb(cls, red: float, green: float, blue: float) -> "Color":
        """Create a truecolor from three color components in the range(0->255).

        Args:
            red (float): Red component in range 0-255.
            green (float): Green component in range 0-255.
            blue (float): Blue component in range 0-255.

        Returns:
            Color: A new color object.
        """
        return cls.from_triplet(ColorTriplet(int(red), int(green), int(blue)))

    @classmethod
    def default(cls) -> "Color":
        """Get a Color instance representing the default color.

        Returns:
            Color: Default color.
        """
        return cls(name="default", type=ColorType.DEFAULT)

    @classmethod
    @lru_cache(maxsize=1024)
    def parse(cls, color: str) -> "Color":
        """Parse a color definition."""
        original_color = color
        color = color.lower().strip()

        if color == "default":
            return cls(color, type=ColorType.DEFAULT)

        color_number = ANSI_COLOR_NAMES.get(color)
        if color_number is not None:
            return cls(
                color,
                type=(ColorType.STANDARD if color_number < 16 else ColorType.EIGHT_BIT),
                number=color_number,
            )

        color_match = RE_COLOR.match(color)
        if color_match is None:
            raise ColorParseError(f"{original_color!r} is not a valid color")

        color_24, color_8, color_rgb = color_match.groups()
        if color_24:
            triplet = ColorTriplet(
                int(color_24[0:2], 16), int(color_24[2:4], 16), int(color_24[4:6], 16)
            )
            return cls(color, ColorType.TRUECOLOR, triplet=triplet)

        elif color_8:
            number = int(color_8)
            if number > 255:
                raise ColorParseError(f"color number must be <= 255 in {color!r}")
            return cls(
                color,
                type=(ColorType.STANDARD if number < 16 else ColorType.EIGHT_BIT),
                number=number,
            )

        else:  #  color_rgb:
            components = color_rgb.split(",")
            if len(components) != 3:
                raise ColorParseError(
                    f"expected three components in {original_color!r}"
                )
            red, green, blue = components
            triplet = ColorTriplet(int(red), int(green), int(blue))
            if not all(component <= 255 for component in triplet):
                raise ColorParseError(
                    f"color components must be <= 255 in {original_color!r}"
                )
            return cls(color, ColorType.TRUECOLOR, triplet=triplet)

    @lru_cache(maxsize=1024)
    def get_ansi_codes(self, foreground: bool = True) -> Tuple[str, ...]:
        """Get the ANSI escape codes for this color."""
        _type = self.type
        if _type == ColorType.DEFAULT:
            return ("39" if foreground else "49",)

        elif _type == ColorType.WINDOWS:
            number = self.number
            assert number is not None
            fore, back = (30, 40) if number < 8 else (82, 92)
            return (str(fore + number if foreground else back + number),)

        elif _type == ColorType.STANDARD:
            number = self.number
            assert number is not None
            fore, back = (30, 40) if number < 8 else (82, 92)
            return (str(fore + number if foreground else back + number),)

        elif _type == ColorType.EIGHT_BIT:
            assert self.number is not None
            return ("38" if foreground else "48", "5", str(self.number))

        else:  # self.standard == ColorStandard.TRUECOLOR:
            assert self.triplet is not None
            red, green, blue = self.triplet
            return ("38" if foreground else "48", "2", str(red), str(green), str(blue))

    @lru_cache(maxsize=1024)
    def downgrade(self, system: ColorSystem) -> "Color":
        """Downgrade a color system to a system with fewer colors."""

        if self.type in (ColorType.DEFAULT, system):
            return self
        # Convert to 8-bit color from truecolor color
        if system == ColorSystem.EIGHT_BIT and self.system == ColorSystem.TRUECOLOR:
            assert self.triplet is not None
            _h, l, s = rgb_to_hls(*self.triplet.normalized)
            # If saturation is under 15% assume it is grayscale
            if s < 0.15:
                gray = round(l * 25.0)
                if gray == 0:
                    color_number = 16
                elif gray == 25:
                    color_number = 231
                else:
                    color_number = 231 + gray
                return Color(self.name, ColorType.EIGHT_BIT, number=color_number)

            red, green, blue = self.triplet
            six_red = red / 95 if red < 95 else 1 + (red - 95) / 40
            six_green = green / 95 if green < 95 else 1 + (green - 95) / 40
            six_blue = blue / 95 if blue < 95 else 1 + (blue - 95) / 40

            color_number = (
                16 + 36 * round(six_red) + 6 * round(six_green) + round(six_blue)
            )
            return Color(self.name, ColorType.EIGHT_BIT, number=color_number)

        # Convert to standard from truecolor or 8-bit
        elif system == ColorSystem.STANDARD:
            if self.system == ColorSystem.TRUECOLOR:
                assert self.triplet is not None
                triplet = self.triplet
            else:  # self.system == ColorSystem.EIGHT_BIT
                assert self.number is not None
                triplet = ColorTriplet(*EIGHT_BIT_PALETTE[self.number])

            color_number = STANDARD_PALETTE.match(triplet)
            return Color(self.name, ColorType.STANDARD, number=color_number)

        elif system == ColorSystem.WINDOWS:
            if self.system == ColorSystem.TRUECOLOR:
                assert self.triplet is not None
                triplet = self.triplet
            else:  # self.system == ColorSystem.EIGHT_BIT
                assert self.number is not None
                if self.number < 16:
                    return Color(self.name, ColorType.WINDOWS, number=self.number)
                triplet = ColorTriplet(*EIGHT_BIT_PALETTE[self.number])

            color_number = WINDOWS_PALETTE.match(triplet)
            return Color(self.name, ColorType.WINDOWS, number=color_number)

        return self


def parse_rgb_hex(hex_color: str) -> ColorTriplet:
    """Parse six hex characters in to RGB triplet."""
    assert len(hex_color) == 6, "must be 6 characters"
    color = ColorTriplet(
        int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    )
    return color


def blend_rgb(
    color1: ColorTriplet, color2: ColorTriplet, cross_fade: float = 0.5
) -> ColorTriplet:
    """Blend one RGB color in to another."""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    new_color = ColorTriplet(
        int(r1 + (r2 - r1) * cross_fade),
        int(g1 + (g2 - g1) * cross_fade),
        int(b1 + (b2 - b1) * cross_fade),
    )
    return new_color


if __name__ == "__main__":  # pragma: no cover
    from .console import Console
    from .table import Table
    from .text import Text

    console = Console()

    table = Table(show_footer=False, show_edge=True)
    table.add_column("Color", width=10, overflow="ellipsis")
    table.add_column("Number", justify="right", style="yellow")
    table.add_column("Name", style="green")
    table.add_column("Hex", style="blue")
    table.add_column("RGB", style="magenta")

    colors = sorted((v, k) for k, v in ANSI_COLOR_NAMES.items())
    for color_number, name in colors:
        if "grey" in name:
            continue
        color_cell = Text(" " * 10, style=f"on {name}")
        if color_number < 16:
            table.add_row(color_cell, f"{color_number}", Text(f'"{name}"'))
        else:
            color = EIGHT_BIT_PALETTE[color_number]  # type: ignore[has-type]
            table.add_row(
                color_cell, str(color_number), Text(f'"{name}"'), color.hex, color.rgb
            )

    console.print(table)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\subsets.py ===
from itertools import combinations

from sympy.combinatorics.graycode import GrayCode


class Subset():
    """
    Represents a basic subset object.

    Explanation
    ===========

    We generate subsets using essentially two techniques,
    binary enumeration and lexicographic enumeration.
    The Subset class takes two arguments, the first one
    describes the initial subset to consider and the second
    describes the superset.

    Examples
    ========

    >>> from sympy.combinatorics import Subset
    >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
    >>> a.next_binary().subset
    ['b']
    >>> a.prev_binary().subset
    ['c']
    """

    _rank_binary = None
    _rank_lex = None
    _rank_graycode = None
    _subset = None
    _superset = None

    def __new__(cls, subset, superset):
        """
        Default constructor.

        It takes the ``subset`` and its ``superset`` as its parameters.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']
        >>> a.superset
        ['a', 'b', 'c', 'd']
        >>> a.size
        2
        """
        if len(subset) > len(superset):
            raise ValueError('Invalid arguments have been provided. The '
                             'superset must be larger than the subset.')
        for elem in subset:
            if elem not in superset:
                raise ValueError('The superset provided is invalid as it does '
                                 'not contain the element {}'.format(elem))
        obj = object.__new__(cls)
        obj._subset = subset
        obj._superset = superset
        return obj

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        whether both objects are of the class Subset and if the values
        of the subset and superset attributes are the same.
        """
        if not isinstance(other, Subset):
            return NotImplemented
        return self.subset == other.subset and self.superset == other.superset

    def iterate_binary(self, k):
        """
        This is a helper function. It iterates over the
        binary subsets by ``k`` steps. This variable can be
        both positive or negative.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(-2).subset
        ['d']
        >>> a = Subset(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(2).subset
        []

        See Also
        ========

        next_binary, prev_binary
        """
        bin_list = Subset.bitlist_from_subset(self.subset, self.superset)
        n = (int(''.join(bin_list), 2) + k) % 2**self.superset_size
        bits = bin(n)[2:].rjust(self.superset_size, '0')
        return Subset.subset_from_bitlist(self.superset, bits)

    def next_binary(self):
        """
        Generates the next binary ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        ['b']
        >>> a = Subset(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_binary().subset
        []

        See Also
        ========

        prev_binary, iterate_binary
        """
        return self.iterate_binary(1)

    def prev_binary(self):
        """
        Generates the previous binary ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['a', 'b', 'c', 'd']
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.prev_binary().subset
        ['c']

        See Also
        ========

        next_binary, iterate_binary
        """
        return self.iterate_binary(-1)

    def next_lexicographic(self):
        """
        Generates the next lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        ['d']
        >>> a = Subset(['d'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        []

        See Also
        ========

        prev_lexicographic
        """
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)

        if i in indices:
            if i - 1 in indices:
                indices.remove(i - 1)
            else:
                indices.remove(i)
                i = i - 1
                while i >= 0 and i not in indices:
                    i = i - 1
                if i >= 0:
                    indices.remove(i)
                    indices.append(i+1)
        else:
            while i not in indices and i >= 0:
                i = i - 1
            indices.append(i + 1)

        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def prev_lexicographic(self):
        """
        Generates the previous lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['d']
        >>> a = Subset(['c','d'], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['c']

        See Also
        ========

        next_lexicographic
        """
        i = self.superset_size - 1
        indices = Subset.subset_indices(self.subset, self.superset)

        while i >= 0 and i not in indices:
            i = i - 1

        if i == 0 or i - 1 in indices:
            indices.remove(i)
        else:
            if i >= 0:
                indices.remove(i)
                indices.append(i - 1)
            indices.append(self.superset_size - 1)

        ret_set = []
        super_set = self.superset
        for i in indices:
            ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    def iterate_graycode(self, k):
        """
        Helper function used for prev_gray and next_gray.
        It performs ``k`` step overs to get the respective Gray codes.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.iterate_graycode(3).subset
        [1, 4]
        >>> a.iterate_graycode(-2).subset
        [1, 2, 4]

        See Also
        ========

        next_gray, prev_gray
        """
        unranked_code = GrayCode.unrank(self.superset_size,
                                       (self.rank_gray + k) % self.cardinality)
        return Subset.subset_from_bitlist(self.superset,
                                          unranked_code)

    def next_gray(self):
        """
        Generates the next Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.next_gray().subset
        [1, 3]

        See Also
        ========

        iterate_graycode, prev_gray
        """
        return self.iterate_graycode(1)

    def prev_gray(self):
        """
        Generates the previous Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([2, 3, 4], [1, 2, 3, 4, 5])
        >>> a.prev_gray().subset
        [2, 3, 4, 5]

        See Also
        ========

        iterate_graycode, next_gray
        """
        return self.iterate_graycode(-1)

    @property
    def rank_binary(self):
        """
        Computes the binary ordered rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a','b','c','d'])
        >>> a.rank_binary
        0
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_binary
        3

        See Also
        ========

        iterate_binary, unrank_binary
        """
        if self._rank_binary is None:
            self._rank_binary = int("".join(
                Subset.bitlist_from_subset(self.subset,
                                           self.superset)), 2)
        return self._rank_binary

    @property
    def rank_lexicographic(self):
        """
        Computes the lexicographic ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_lexicographic
        14
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_lexicographic
        43
        """
        if self._rank_lex is None:
            def _ranklex(self, subset_index, i, n):
                if subset_index == [] or i > n:
                    return 0
                if i in subset_index:
                    subset_index.remove(i)
                    return 1 + _ranklex(self, subset_index, i + 1, n)
                return 2**(n - i - 1) + _ranklex(self, subset_index, i + 1, n)
            indices = Subset.subset_indices(self.subset, self.superset)
            self._rank_lex = _ranklex(self, indices, 0, self.superset_size)
        return self._rank_lex

    @property
    def rank_gray(self):
        """
        Computes the Gray code ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c','d'], ['a','b','c','d'])
        >>> a.rank_gray
        2
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_gray
        27

        See Also
        ========

        iterate_graycode, unrank_gray
        """
        if self._rank_graycode is None:
            bits = Subset.bitlist_from_subset(self.subset, self.superset)
            self._rank_graycode = GrayCode(len(bits), start=bits).rank
        return self._rank_graycode

    @property
    def subset(self):
        """
        Gets the subset represented by the current instance.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.subset
        ['c', 'd']

        See Also
        ========

        superset, size, superset_size, cardinality
        """
        return self._subset

    @property
    def size(self):
        """
        Gets the size of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.size
        2

        See Also
        ========

        subset, superset, superset_size, cardinality
        """
        return len(self.subset)

    @property
    def superset(self):
        """
        Gets the superset of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset
        ['a', 'b', 'c', 'd']

        See Also
        ========

        subset, size, superset_size, cardinality
        """
        return self._superset

    @property
    def superset_size(self):
        """
        Returns the size of the superset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset_size
        4

        See Also
        ========

        subset, superset, size, cardinality
        """
        return len(self.superset)

    @property
    def cardinality(self):
        """
        Returns the number of all possible subsets.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.cardinality
        16

        See Also
        ========

        subset, superset, size, superset_size
        """
        return 2**(self.superset_size)

    @classmethod
    def subset_from_bitlist(self, super_set, bitlist):
        """
        Gets the subset defined by the bitlist.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.subset_from_bitlist(['a', 'b', 'c', 'd'], '0011').subset
        ['c', 'd']

        See Also
        ========

        bitlist_from_subset
        """
        if len(super_set) != len(bitlist):
            raise ValueError("The sizes of the lists are not equal")
        ret_set = []
        for i in range(len(bitlist)):
            if bitlist[i] == '1':
                ret_set.append(super_set[i])
        return Subset(ret_set, super_set)

    @classmethod
    def bitlist_from_subset(self, subset, superset):
        """
        Gets the bitlist corresponding to a subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.bitlist_from_subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        '0011'

        See Also
        ========

        subset_from_bitlist
        """
        bitlist = ['0'] * len(superset)
        if isinstance(subset, Subset):
            subset = subset.subset
        for i in Subset.subset_indices(subset, superset):
            bitlist[i] = '1'
        return ''.join(bitlist)

    @classmethod
    def unrank_binary(self, rank, superset):
        """
        Gets the binary ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_binary(4, ['a', 'b', 'c', 'd']).subset
        ['b']

        See Also
        ========

        iterate_binary, rank_binary
        """
        bits = bin(rank)[2:].rjust(len(superset), '0')
        return Subset.subset_from_bitlist(superset, bits)

    @classmethod
    def unrank_gray(self, rank, superset):
        """
        Gets the Gray code ordered subset of the specified rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> Subset.unrank_gray(4, ['a', 'b', 'c']).subset
        ['a', 'b']
        >>> Subset.unrank_gray(0, ['a', 'b', 'c']).subset
        []

        See Also
        ========

        iterate_graycode, rank_gray
        """
        graycode_bitlist = GrayCode.unrank(len(superset), rank)
        return Subset.subset_from_bitlist(superset, graycode_bitlist)

    @classmethod
    def subset_indices(self, subset, superset):
        """Return indices of subset in superset in a list; the list is empty
        if all elements of ``subset`` are not in ``superset``.

        Examples
        ========

            >>> from sympy.combinatorics import Subset
            >>> superset = [1, 3, 2, 5, 4]
            >>> Subset.subset_indices([3, 2, 1], superset)
            [1, 2, 0]
            >>> Subset.subset_indices([1, 6], superset)
            []
            >>> Subset.subset_indices([], superset)
            []

        """
        a, b = superset, subset
        sb = set(b)
        d = {}
        for i, ai in enumerate(a):
            if ai in sb:
                d[ai] = i
                sb.remove(ai)
                if not sb:
                    break
        else:
            return []
        return [d[bi] for bi in b]


def ksubsets(superset, k):
    """
    Finds the subsets of size ``k`` in lexicographic order.

    This uses the itertools generator.

    Examples
    ========

    >>> from sympy.combinatorics.subsets import ksubsets
    >>> list(ksubsets([1, 2, 3], 2))
    [(1, 2), (1, 3), (2, 3)]
    >>> list(ksubsets([1, 2, 3, 4, 5], 2))
    [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), \
    (2, 5), (3, 4), (3, 5), (4, 5)]

    See Also
    ========

    Subset
    """
    return combinations(superset, k)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\model.py ===
import types
import weakref

from .lock import allocate_lock
from .error import CDefError, VerificationError, VerificationMissing

# type qualifiers
Q_CONST    = 0x01
Q_RESTRICT = 0x02
Q_VOLATILE = 0x04

def qualify(quals, replace_with):
    if quals & Q_CONST:
        replace_with = ' const ' + replace_with.lstrip()
    if quals & Q_VOLATILE:
        replace_with = ' volatile ' + replace_with.lstrip()
    if quals & Q_RESTRICT:
        # It seems that __restrict is supported by gcc and msvc.
        # If you hit some different compiler, add a #define in
        # _cffi_include.h for it (and in its copies, documented there)
        replace_with = ' __restrict ' + replace_with.lstrip()
    return replace_with


class BaseTypeByIdentity(object):
    is_array_type = False
    is_raw_function = False

    def get_c_name(self, replace_with='', context='a C file', quals=0):
        result = self.c_name_with_marker
        assert result.count('&') == 1
        # some logic duplication with ffi.getctype()... :-(
        replace_with = replace_with.strip()
        if replace_with:
            if replace_with.startswith('*') and '&[' in result:
                replace_with = '(%s)' % replace_with
            elif not replace_with[0] in '[(':
                replace_with = ' ' + replace_with
        replace_with = qualify(quals, replace_with)
        result = result.replace('&', replace_with)
        if '$' in result:
            raise VerificationError(
                "cannot generate '%s' in %s: unknown type name"
                % (self._get_c_name(), context))
        return result

    def _get_c_name(self):
        return self.c_name_with_marker.replace('&', '')

    def has_c_name(self):
        return '$' not in self._get_c_name()

    def is_integer_type(self):
        return False

    def get_cached_btype(self, ffi, finishlist, can_delay=False):
        try:
            BType = ffi._cached_btypes[self]
        except KeyError:
            BType = self.build_backend_type(ffi, finishlist)
            BType2 = ffi._cached_btypes.setdefault(self, BType)
            assert BType2 is BType
        return BType

    def __repr__(self):
        return '<%s>' % (self._get_c_name(),)

    def _get_items(self):
        return [(name, getattr(self, name)) for name in self._attrs_]


class BaseType(BaseTypeByIdentity):

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self._get_items() == other._get_items())

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.__class__, tuple(self._get_items())))


class VoidType(BaseType):
    _attrs_ = ()

    def __init__(self):
        self.c_name_with_marker = 'void&'

    def build_backend_type(self, ffi, finishlist):
        return global_cache(self, ffi, 'new_void_type')

void_type = VoidType()


class BasePrimitiveType(BaseType):
    def is_complex_type(self):
        return False


class PrimitiveType(BasePrimitiveType):
    _attrs_ = ('name',)

    ALL_PRIMITIVE_TYPES = {
        'char':               'c',
        'short':              'i',
        'int':                'i',
        'long':               'i',
        'long long':          'i',
        'signed char':        'i',
        'unsigned char':      'i',
        'unsigned short':     'i',
        'unsigned int':       'i',
        'unsigned long':      'i',
        'unsigned long long': 'i',
        'float':              'f',
        'double':             'f',
        'long double':        'f',
        '_cffi_float_complex_t': 'j',
        '_cffi_double_complex_t': 'j',
        '_Bool':              'i',
        # the following types are not primitive in the C sense
        'wchar_t':            'c',
        'char16_t':           'c',
        'char32_t':           'c',
        'int8_t':             'i',
        'uint8_t':            'i',
        'int16_t':            'i',
        'uint16_t':           'i',
        'int32_t':            'i',
        'uint32_t':           'i',
        'int64_t':            'i',
        'uint64_t':           'i',
        'int_least8_t':       'i',
        'uint_least8_t':      'i',
        'int_least16_t':      'i',
        'uint_least16_t':     'i',
        'int_least32_t':      'i',
        'uint_least32_t':     'i',
        'int_least64_t':      'i',
        'uint_least64_t':     'i',
        'int_fast8_t':        'i',
        'uint_fast8_t':       'i',
        'int_fast16_t':       'i',
        'uint_fast16_t':      'i',
        'int_fast32_t':       'i',
        'uint_fast32_t':      'i',
        'int_fast64_t':       'i',
        'uint_fast64_t':      'i',
        'intptr_t':           'i',
        'uintptr_t':          'i',
        'intmax_t':           'i',
        'uintmax_t':          'i',
        'ptrdiff_t':          'i',
        'size_t':             'i',
        'ssize_t':            'i',
        }

    def __init__(self, name):
        assert name in self.ALL_PRIMITIVE_TYPES
        self.name = name
        self.c_name_with_marker = name + '&'

    def is_char_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'c'
    def is_integer_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'i'
    def is_float_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'f'
    def is_complex_type(self):
        return self.ALL_PRIMITIVE_TYPES[self.name] == 'j'

    def build_backend_type(self, ffi, finishlist):
        return global_cache(self, ffi, 'new_primitive_type', self.name)


class UnknownIntegerType(BasePrimitiveType):
    _attrs_ = ('name',)

    def __init__(self, name):
        self.name = name
        self.c_name_with_marker = name + '&'

    def is_integer_type(self):
        return True

    def build_backend_type(self, ffi, finishlist):
        raise NotImplementedError("integer type '%s' can only be used after "
                                  "compilation" % self.name)

class UnknownFloatType(BasePrimitiveType):
    _attrs_ = ('name', )

    def __init__(self, name):
        self.name = name
        self.c_name_with_marker = name + '&'

    def build_backend_type(self, ffi, finishlist):
        raise NotImplementedError("float type '%s' can only be used after "
                                  "compilation" % self.name)


class BaseFunctionType(BaseType):
    _attrs_ = ('args', 'result', 'ellipsis', 'abi')

    def __init__(self, args, result, ellipsis, abi=None):
        self.args = args
        self.result = result
        self.ellipsis = ellipsis
        self.abi = abi
        #
        reprargs = [arg._get_c_name() for arg in self.args]
        if self.ellipsis:
            reprargs.append('...')
        reprargs = reprargs or ['void']
        replace_with = self._base_pattern % (', '.join(reprargs),)
        if abi is not None:
            replace_with = replace_with[:1] + abi + ' ' + replace_with[1:]
        self.c_name_with_marker = (
            self.result.c_name_with_marker.replace('&', replace_with))


class RawFunctionType(BaseFunctionType):
    # Corresponds to a C type like 'int(int)', which is the C type of
    # a function, but not a pointer-to-function.  The backend has no
    # notion of such a type; it's used temporarily by parsing.
    _base_pattern = '(&)(%s)'
    is_raw_function = True

    def build_backend_type(self, ffi, finishlist):
        raise CDefError("cannot render the type %r: it is a function "
                        "type, not a pointer-to-function type" % (self,))

    def as_function_pointer(self):
        return FunctionPtrType(self.args, self.result, self.ellipsis, self.abi)


class FunctionPtrType(BaseFunctionType):
    _base_pattern = '(*&)(%s)'

    def build_backend_type(self, ffi, finishlist):
        result = self.result.get_cached_btype(ffi, finishlist)
        args = []
        for tp in self.args:
            args.append(tp.get_cached_btype(ffi, finishlist))
        abi_args = ()
        if self.abi == "__stdcall":
            if not self.ellipsis:    # __stdcall ignored for variadic funcs
                try:
                    abi_args = (ffi._backend.FFI_STDCALL,)
                except AttributeError:
                    pass
        return global_cache(self, ffi, 'new_function_type',
                            tuple(args), result, self.ellipsis, *abi_args)

    def as_raw_function(self):
        return RawFunctionType(self.args, self.result, self.ellipsis, self.abi)


class PointerType(BaseType):
    _attrs_ = ('totype', 'quals')

    def __init__(self, totype, quals=0):
        self.totype = totype
        self.quals = quals
        extra = " *&"
        if totype.is_array_type:
            extra = "(%s)" % (extra.lstrip(),)
        extra = qualify(quals, extra)
        self.c_name_with_marker = totype.c_name_with_marker.replace('&', extra)

    def build_backend_type(self, ffi, finishlist):
        BItem = self.totype.get_cached_btype(ffi, finishlist, can_delay=True)
        return global_cache(self, ffi, 'new_pointer_type', BItem)

voidp_type = PointerType(void_type)

def ConstPointerType(totype):
    return PointerType(totype, Q_CONST)

const_voidp_type = ConstPointerType(void_type)


class NamedPointerType(PointerType):
    _attrs_ = ('totype', 'name')

    def __init__(self, totype, name, quals=0):
        PointerType.__init__(self, totype, quals)
        self.name = name
        self.c_name_with_marker = name + '&'


class ArrayType(BaseType):
    _attrs_ = ('item', 'length')
    is_array_type = True

    def __init__(self, item, length):
        self.item = item
        self.length = length
        #
        if length is None:
            brackets = '&[]'
        elif length == '...':
            brackets = '&[/*...*/]'
        else:
            brackets = '&[%s]' % length
        self.c_name_with_marker = (
            self.item.c_name_with_marker.replace('&', brackets))

    def length_is_unknown(self):
        return isinstance(self.length, str)

    def resolve_length(self, newlength):
        return ArrayType(self.item, newlength)

    def build_backend_type(self, ffi, finishlist):
        if self.length_is_unknown():
            raise CDefError("cannot render the type %r: unknown length" %
                            (self,))
        self.item.get_cached_btype(ffi, finishlist)   # force the item BType
        BPtrItem = PointerType(self.item).get_cached_btype(ffi, finishlist)
        return global_cache(self, ffi, 'new_array_type', BPtrItem, self.length)

char_array_type = ArrayType(PrimitiveType('char'), None)


class StructOrUnionOrEnum(BaseTypeByIdentity):
    _attrs_ = ('name',)
    forcename = None

    def build_c_name_with_marker(self):
        name = self.forcename or '%s %s' % (self.kind, self.name)
        self.c_name_with_marker = name + '&'

    def force_the_name(self, forcename):
        self.forcename = forcename
        self.build_c_name_with_marker()

    def get_official_name(self):
        assert self.c_name_with_marker.endswith('&')
        return self.c_name_with_marker[:-1]


class StructOrUnion(StructOrUnionOrEnum):
    fixedlayout = None
    completed = 0
    partial = False
    packed = 0

    def __init__(self, name, fldnames, fldtypes, fldbitsize, fldquals=None):
        self.name = name
        self.fldnames = fldnames
        self.fldtypes = fldtypes
        self.fldbitsize = fldbitsize
        self.fldquals = fldquals
        self.build_c_name_with_marker()

    def anonymous_struct_fields(self):
        if self.fldtypes is not None:
            for name, type in zip(self.fldnames, self.fldtypes):
                if name == '' and isinstance(type, StructOrUnion):
                    yield type

    def enumfields(self, expand_anonymous_struct_union=True):
        fldquals = self.fldquals
        if fldquals is None:
            fldquals = (0,) * len(self.fldnames)
        for name, type, bitsize, quals in zip(self.fldnames, self.fldtypes,
                                              self.fldbitsize, fldquals):
            if (name == '' and isinstance(type, StructOrUnion)
                    and expand_anonymous_struct_union):
                # nested anonymous struct/union
                for result in type.enumfields():
                    yield result
            else:
                yield (name, type, bitsize, quals)

    def force_flatten(self):
        # force the struct or union to have a declaration that lists
        # directly all fields returned by enumfields(), flattening
        # nested anonymous structs/unions.
        names = []
        types = []
        bitsizes = []
        fldquals = []
        for name, type, bitsize, quals in self.enumfields():
            names.append(name)
            types.append(type)
            bitsizes.append(bitsize)
            fldquals.append(quals)
        self.fldnames = tuple(names)
        self.fldtypes = tuple(types)
        self.fldbitsize = tuple(bitsizes)
        self.fldquals = tuple(fldquals)

    def get_cached_btype(self, ffi, finishlist, can_delay=False):
        BType = StructOrUnionOrEnum.get_cached_btype(self, ffi, finishlist,
                                                     can_delay)
        if not can_delay:
            self.finish_backend_type(ffi, finishlist)
        return BType

    def finish_backend_type(self, ffi, finishlist):
        if self.completed:
            if self.completed != 2:
                raise NotImplementedError("recursive structure declaration "
                                          "for '%s'" % (self.name,))
            return
        BType = ffi._cached_btypes[self]
        #
        self.completed = 1
        #
        if self.fldtypes is None:
            pass    # not completing it: it's an opaque struct
            #
        elif self.fixedlayout is None:
            fldtypes = [tp.get_cached_btype(ffi, finishlist)
                        for tp in self.fldtypes]
            lst = list(zip(self.fldnames, fldtypes, self.fldbitsize))
            extra_flags = ()
            if self.packed:
                if self.packed == 1:
                    extra_flags = (8,)    # SF_PACKED
                else:
                    extra_flags = (0, self.packed)
            ffi._backend.complete_struct_or_union(BType, lst, self,
                                                  -1, -1, *extra_flags)
            #
        else:
            fldtypes = []
            fieldofs, fieldsize, totalsize, totalalignment = self.fixedlayout
            for i in range(len(self.fldnames)):
                fsize = fieldsize[i]
                ftype = self.fldtypes[i]
                #
                if isinstance(ftype, ArrayType) and ftype.length_is_unknown():
                    # fix the length to match the total size
                    BItemType = ftype.item.get_cached_btype(ffi, finishlist)
                    nlen, nrest = divmod(fsize, ffi.sizeof(BItemType))
                    if nrest != 0:
                        self._verification_error(
                            "field '%s.%s' has a bogus size?" % (
                            self.name, self.fldnames[i] or '{}'))
                    ftype = ftype.resolve_length(nlen)
                    self.fldtypes = (self.fldtypes[:i] + (ftype,) +
                                     self.fldtypes[i+1:])
                #
                BFieldType = ftype.get_cached_btype(ffi, finishlist)
                if isinstance(ftype, ArrayType) and ftype.length is None:
                    assert fsize == 0
                else:
                    bitemsize = ffi.sizeof(BFieldType)
                    if bitemsize != fsize:
                        self._verification_error(
                            "field '%s.%s' is declared as %d bytes, but is "
                            "really %d bytes" % (self.name,
                                                 self.fldnames[i] or '{}',
                                                 bitemsize, fsize))
                fldtypes.append(BFieldType)
            #
            lst = list(zip(self.fldnames, fldtypes, self.fldbitsize, fieldofs))
            ffi._backend.complete_struct_or_union(BType, lst, self,
                                                  totalsize, totalalignment)
        self.completed = 2

    def _verification_error(self, msg):
        raise VerificationError(msg)

    def check_not_partial(self):
        if self.partial and self.fixedlayout is None:
            raise VerificationMissing(self._get_c_name())

    def build_backend_type(self, ffi, finishlist):
        self.check_not_partial()
        finishlist.append(self)
        #
        return global_cache(self, ffi, 'new_%s_type' % self.kind,
                            self.get_official_name(), key=self)


class StructType(StructOrUnion):
    kind = 'struct'


class UnionType(StructOrUnion):
    kind = 'union'


class EnumType(StructOrUnionOrEnum):
    kind = 'enum'
    partial = False
    partial_resolved = False

    def __init__(self, name, enumerators, enumvalues, baseinttype=None):
        self.name = name
        self.enumerators = enumerators
        self.enumvalues = enumvalues
        self.baseinttype = baseinttype
        self.build_c_name_with_marker()

    def force_the_name(self, forcename):
        StructOrUnionOrEnum.force_the_name(self, forcename)
        if self.forcename is None:
            name = self.get_official_name()
            self.forcename = '$' + name.replace(' ', '_')

    def check_not_partial(self):
        if self.partial and not self.partial_resolved:
            raise VerificationMissing(self._get_c_name())

    def build_backend_type(self, ffi, finishlist):
        self.check_not_partial()
        base_btype = self.build_baseinttype(ffi, finishlist)
        return global_cache(self, ffi, 'new_enum_type',
                            self.get_official_name(),
                            self.enumerators, self.enumvalues,
                            base_btype, key=self)

    def build_baseinttype(self, ffi, finishlist):
        if self.baseinttype is not None:
            return self.baseinttype.get_cached_btype(ffi, finishlist)
        #
        if self.enumvalues:
            smallest_value = min(self.enumvalues)
            largest_value = max(self.enumvalues)
        else:
            import warnings
            try:
                # XXX!  The goal is to ensure that the warnings.warn()
                # will not suppress the warning.  We want to get it
                # several times if we reach this point several times.
                __warningregistry__.clear()
            except NameError:
                pass
            warnings.warn("%r has no values explicitly defined; "
                          "guessing that it is equivalent to 'unsigned int'"
                          % self._get_c_name())
            smallest_value = largest_value = 0
        if smallest_value < 0:   # needs a signed type
            sign = 1
            candidate1 = PrimitiveType("int")
            candidate2 = PrimitiveType("long")
        else:
            sign = 0
            candidate1 = PrimitiveType("unsigned int")
            candidate2 = PrimitiveType("unsigned long")
        btype1 = candidate1.get_cached_btype(ffi, finishlist)
        btype2 = candidate2.get_cached_btype(ffi, finishlist)
        size1 = ffi.sizeof(btype1)
        size2 = ffi.sizeof(btype2)
        if (smallest_value >= ((-1) << (8*size1-1)) and
            largest_value < (1 << (8*size1-sign))):
            return btype1
        if (smallest_value >= ((-1) << (8*size2-1)) and
            largest_value < (1 << (8*size2-sign))):
            return btype2
        raise CDefError("%s values don't all fit into either 'long' "
                        "or 'unsigned long'" % self._get_c_name())

def unknown_type(name, structname=None):
    if structname is None:
        structname = '$%s' % name
    tp = StructType(structname, None, None, None)
    tp.force_the_name(name)
    tp.origin = "unknown_type"
    return tp

def unknown_ptr_type(name, structname=None):
    if structname is None:
        structname = '$$%s' % name
    tp = StructType(structname, None, None, None)
    return NamedPointerType(tp, name)


global_lock = allocate_lock()
_typecache_cffi_backend = weakref.WeakValueDictionary()

def get_typecache(backend):
    # returns _typecache_cffi_backend if backend is the _cffi_backend
    # module, or type(backend).__typecache if backend is an instance of
    # CTypesBackend (or some FakeBackend class during tests)
    if isinstance(backend, types.ModuleType):
        return _typecache_cffi_backend
    with global_lock:
        if not hasattr(type(backend), '__typecache'):
            type(backend).__typecache = weakref.WeakValueDictionary()
        return type(backend).__typecache

def global_cache(srctype, ffi, funcname, *args, **kwds):
    key = kwds.pop('key', (funcname, args))
    assert not kwds
    try:
        return ffi._typecache[key]
    except KeyError:
        pass
    try:
        res = getattr(ffi._backend, funcname)(*args)
    except NotImplementedError as e:
        raise NotImplementedError("%s: %r: %s" % (funcname, srctype, e))
    # note that setdefault() on WeakValueDictionary is not atomic
    # and contains a rare bug (http://bugs.python.org/issue19542);
    # we have to use a lock and do it ourselves
    cache = ffi._typecache
    with global_lock:
        res1 = cache.get(key)
        if res1 is None:
            cache[key] = res
            return res
        else:
            return res1

def pointer_cache(ffi, BType):
    return global_cache('?', ffi, 'new_pointer_type', BType)

def attach_exception_info(e, name):
    if e.args and type(e.args[0]) is str:
        e.args = ('%s: %s' % (name, e.args[0]),) + e.args[1:]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\ntheory.py ===
# sympy.external.ntheory
#
# This module provides pure Python implementations of some number theory
# functions that are alternately used from gmpy2 if it is installed.

import math

import mpmath.libmp as mlib


_small_trailing = [0] * 256
for j in range(1, 8):
    _small_trailing[1 << j :: 1 << (j + 1)] = [j] * (1 << (7 - j))


def bit_scan1(x, n=0):
    if not x:
        return
    x = abs(x >> n)
    low_byte = x & 0xFF
    if low_byte:
        return _small_trailing[low_byte] + n

    t = 8 + n
    x >>= 8
    # 2**m is quick for z up through 2**30
    z = x.bit_length() - 1
    if x == 1 << z:
        return z + t

    if z < 300:
        # fixed 8-byte reduction
        while not x & 0xFF:
            x >>= 8
            t += 8
    else:
        # binary reduction important when there might be a large
        # number of trailing 0s
        p = z >> 1
        while not x & 0xFF:
            while x & ((1 << p) - 1):
                p >>= 1
            x >>= p
            t += p
    return t + _small_trailing[x & 0xFF]


def bit_scan0(x, n=0):
    return bit_scan1(x + (1 << n), n)


def remove(x, f):
    if f < 2:
        raise ValueError("factor must be > 1")
    if x == 0:
        return 0, 0
    if f == 2:
        b = bit_scan1(x)
        return x >> b, b
    m = 0
    y, rem = divmod(x, f)
    while not rem:
        x = y
        m += 1
        if m > 5:
            pow_list = [f**2]
            while pow_list:
                _f = pow_list[-1]
                y, rem = divmod(x, _f)
                if not rem:
                    m += 1 << len(pow_list)
                    x = y
                    pow_list.append(_f**2)
                else:
                    pow_list.pop()
        y, rem = divmod(x, f)
    return x, m


def factorial(x):
    """Return x!."""
    return int(mlib.ifac(int(x)))


def sqrt(x):
    """Integer square root of x."""
    return int(mlib.isqrt(int(x)))


def sqrtrem(x):
    """Integer square root of x and remainder."""
    s, r = mlib.sqrtrem(int(x))
    return (int(s), int(r))


gcd = math.gcd
lcm = math.lcm


def _sign(n):
    if n < 0:
        return -1, -n
    return 1, n


def gcdext(a, b):
    if not a or not b:
        g = abs(a) or abs(b)
        if not g:
            return (0, 0, 0)
        return (g, a // g, b // g)

    x_sign, a = _sign(a)
    y_sign, b = _sign(b)
    x, r = 1, 0
    y, s = 0, 1

    while b:
        q, c = divmod(a, b)
        a, b = b, c
        x, r = r, x - q*r
        y, s = s, y - q*s

    return (a, x * x_sign, y * y_sign)


def is_square(x):
    """Return True if x is a square number."""
    if x < 0:
        return False

    # Note that the possible values of y**2 % n for a given n are limited.
    # For example, when n=4, y**2 % n can only take 0 or 1.
    # In other words, if x % 4 is 2 or 3, then x is not a square number.
    # Mathematically, it determines if it belongs to the set {y**2 % n},
    # but implementationally, it can be realized as a logical conjunction
    # with an n-bit integer.
    # see https://mersenneforum.org/showpost.php?p=110896
    # def magic(n):
    #     s = {y**2 % n for y in range(n)}
    #     s = set(range(n)) - s
    #     return sum(1 << bit for bit in s)
    # >>> print(hex(magic(128)))
    # 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec
    # >>> print(hex(magic(99)))
    # 0x5f6f9ffb6fb7ddfcb75befdec
    # >>> print(hex(magic(91)))
    # 0x6fd1bfcfed5f3679d3ebdec
    # >>> print(hex(magic(85)))
    # 0xdef9ae771ffe3b9d67dec
    if 0xfdfdfdedfdfdfdecfdfdfdedfdfcfdec & (1 << (x & 127)):
        return False  # e.g. 2, 3
    m = x % 765765 # 765765 = 99 * 91 * 85
    if 0x5f6f9ffb6fb7ddfcb75befdec & (1 << (m % 99)):
        return False  # e.g. 17, 68
    if 0x6fd1bfcfed5f3679d3ebdec & (1 << (m % 91)):
        return False  # e.g. 97, 388
    if 0xdef9ae771ffe3b9d67dec & (1 << (m % 85)):
        return False  # e.g. 793, 1408
    return mlib.sqrtrem(int(x))[1] == 0


def invert(x, m):
    """Modular inverse of x modulo m.

    Returns y such that x*y == 1 mod m.

    Uses ``math.pow`` but reproduces the behaviour of ``gmpy2.invert``
    which raises ZeroDivisionError if no inverse exists.
    """
    try:
        return pow(x, -1, m)
    except ValueError:
        raise ZeroDivisionError("invert() no inverse exists")


def legendre(x, y):
    """Legendre symbol (x / y).

    Following the implementation of gmpy2,
    the error is raised only when y is an even number.
    """
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd prime")
    x %= y
    if not x:
        return 0
    if pow(x, (y - 1) // 2, y) == 1:
        return 1
    return -1


def jacobi(x, y):
    """Jacobi symbol (x / y)."""
    if y <= 0 or not y % 2:
        raise ValueError("y should be an odd positive integer")
    x %= y
    if not x:
        return int(y == 1)
    if y == 1 or x == 1:
        return 1
    if gcd(x, y) != 1:
        return 0
    j = 1
    while x != 0:
        while x % 2 == 0 and x > 0:
            x >>= 1
            if y % 8 in [3, 5]:
                j = -j
        x, y = y, x
        if x % 4 == y % 4 == 3:
            j = -j
        x %= y
    return j


def kronecker(x, y):
    """Kronecker symbol (x / y)."""
    if gcd(x, y) != 1:
        return 0
    if y == 0:
        return 1
    sign = -1 if y < 0 and x < 0 else 1
    y = abs(y)
    s = bit_scan1(y)
    y >>= s
    if s % 2 and x % 8 in [3, 5]:
        sign = -sign
    return sign * jacobi(x, y)


def iroot(y, n):
    if y < 0:
        raise ValueError("y must be nonnegative")
    if n < 1:
        raise ValueError("n must be positive")
    if y in (0, 1):
        return y, True
    if n == 1:
        return y, True
    if n == 2:
        x, rem = mlib.sqrtrem(y)
        return int(x), not rem
    if n >= y.bit_length():
        return 1, False
    # Get initial estimate for Newton's method. Care must be taken to
    # avoid overflow
    try:
        guess = int(y**(1./n) + 0.5)
    except OverflowError:
        exp = math.log2(y)/n
        if exp > 53:
            shift = int(exp - 53)
            guess = int(2.0**(exp - shift) + 1) << shift
        else:
            guess = int(2.0**exp)
    if guess > 2**50:
        # Newton iteration
        xprev, x = -1, guess
        while 1:
            t = x**(n - 1)
            xprev, x = x, ((n - 1)*x + y//t)//n
            if abs(x - xprev) < 2:
                break
    else:
        x = guess
    # Compensate
    t = x**n
    while t < y:
        x += 1
        t = x**n
    while t > y:
        x -= 1
        t = x**n
    return x, t == y


def is_fermat_prp(n, a):
    if a < 2:
        raise ValueError("is_fermat_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_fermat_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_fermat_prp() requires gcd(n,a) == 1")
    return pow(a, n - 1, n) == 1


def is_euler_prp(n, a):
    if a < 2:
        raise ValueError("is_euler_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_euler_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_euler_prp() requires gcd(n,a) == 1")
    return pow(a, n >> 1, n) == jacobi(a, n) % n


def _is_strong_prp(n, a):
    s = bit_scan1(n - 1)
    a = pow(a, n >> s, n)
    if a == 1 or a == n - 1:
        return True
    for _ in range(s - 1):
        a = pow(a, 2, n)
        if a == n - 1:
            return True
        if a == 1:
            return False
    return False


def is_strong_prp(n, a):
    if a < 2:
        raise ValueError("is_strong_prp() requires 'a' greater than or equal to 2")
    if n < 1:
        raise ValueError("is_strong_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    a %= n
    if gcd(n, a) != 1:
        raise ValueError("is_strong_prp() requires gcd(n,a) == 1")
    return _is_strong_prp(n, a)


def _lucas_sequence(n, P, Q, k):
    r"""Return the modular Lucas sequence (U_k, V_k, Q_k).

    Explanation
    ===========

    Given a Lucas sequence defined by P, Q, returns the kth values for
    U and V, along with Q^k, all modulo n. This is intended for use with
    possibly very large values of n and k, where the combinatorial functions
    would be completely unusable.

    .. math ::
        U_k = \begin{cases}
             0 & \text{if } k = 0\\
             1 & \text{if } k = 1\\
             PU_{k-1} - QU_{k-2} & \text{if } k > 1
        \end{cases}\\
        V_k = \begin{cases}
             2 & \text{if } k = 0\\
             P & \text{if } k = 1\\
             PV_{k-1} - QV_{k-2} & \text{if } k > 1
        \end{cases}

    The modular Lucas sequences are used in numerous places in number theory,
    especially in the Lucas compositeness tests and the various n + 1 proofs.

    Parameters
    ==========

    n : int
        n is an odd number greater than or equal to 3
    P : int
    Q : int
        D determined by D = P**2 - 4*Q is non-zero
    k : int
        k is a nonnegative integer

    Returns
    =======

    U, V, Qk : (int, int, int)
        `(U_k \bmod{n}, V_k \bmod{n}, Q^k \bmod{n})`

    Examples
    ========

    >>> from sympy.external.ntheory import _lucas_sequence
    >>> N = 10**2000 + 4561
    >>> sol = U, V, Qk = _lucas_sequence(N, 3, 1, N//2); sol
    (0, 2, 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_sequence

    """
    if k == 0:
        return (0, 2, 1)
    D = P**2 - 4*Q
    U = 1
    V = P
    Qk = Q % n
    if Q == 1:
        # Optimization for extra strong tests.
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2) % n
            if b == "1":
                U, V = U*P + V, V*P + U*D
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = U >> 1, V >> 1
    elif P == 1 and Q == -1:
        # Small optimization for 50% of Selfridge parameters.
        for b in bin(k)[3:]:
            U = (U*V) % n
            if Qk == 1:
                V = (V*V - 2) % n
            else:
                V = (V*V + 2) % n
                Qk = 1
            if b == "1":
                # new_U = (U + V) // 2
                # new_V = (5*U + V) // 2 = 2*U + new_U
                U, V  = U + V, U << 1
                if U & 1:
                    U += n
                U >>= 1
                V += U
                Qk = -1
        Qk %= n
    elif P == 1:
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2*Qk) % n
            Qk *= Qk
            if b == "1":
                # new_U = (U + V) // 2
                # new_V = new_U - 2*Q*U
                U, V  = U + V, (Q*U) << 1
                if U & 1:
                    U += n
                U >>= 1
                V = U - V
                Qk *= Q
            Qk %= n
    else:
        # The general case with any P and Q.
        for b in bin(k)[3:]:
            U = (U*V) % n
            V = (V*V - 2*Qk) % n
            Qk *= Qk
            if b == "1":
                U, V = U*P + V, V*P + U*D
                if U & 1:
                    U += n
                if V & 1:
                    V += n
                U, V = U >> 1, V >> 1
                Qk *= Q
            Qk %= n
    return (U % n, V % n, Qk)


def is_fibonacci_prp(n, p, q):
    d = p**2 - 4*q
    if d == 0 or p <= 0 or q not in [1, -1]:
        raise ValueError("invalid values for p,q in is_fibonacci_prp()")
    if n < 1:
        raise ValueError("is_fibonacci_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _lucas_sequence(n, p, q, n)[1] == p % n


def is_lucas_prp(n, p, q):
    d = p**2 - 4*q
    if d == 0:
        raise ValueError("invalid values for p,q in is_lucas_prp()")
    if n < 1:
        raise ValueError("is_lucas_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    if gcd(n, q*d) not in [1, n]:
        raise ValueError("is_lucas_prp() requires gcd(n,2*q*D) == 1")
    return _lucas_sequence(n, p, q, n - jacobi(d, n))[0] == 0


def _is_selfridge_prp(n):
    """Lucas compositeness test with the Selfridge parameters for n.

    Explanation
    ===========

    The Lucas compositeness test checks whether n is a prime number.
    The test can be run with arbitrary parameters ``P`` and ``Q``, which also change the performance of the test.
    So, which parameters are most effective for running the Lucas compositeness test?
    As an algorithm for determining ``P`` and ``Q``, Selfridge proposed method A [1]_ page 1401
    (Since two methods were proposed, referred to simply as A and B in the paper,
    we will refer to one of them as "method A").

    method A fixes ``P = 1``. Then, ``D`` defined by ``D = P**2 - 4Q`` is varied from 5, -7, 9, -11, 13, and so on,
    with the first ``D`` being ``jacobi(D, n) == -1``. Once ``D`` is determined,
    ``Q`` is determined to be ``(P**2 - D)//4``.

    References
    ==========

    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf

    """
    for D in range(5, 1_000_000, 2):
        if D & 2: # if D % 4 == 3
            D = -D
        j = jacobi(D, n)
        if j == -1:
            return _lucas_sequence(n, 1, (1-D) // 4, n + 1)[0] == 0
        if j == 0 and D % n:
            return False
        # When j == -1 is hard to find, suspect a square number
        if D == 13 and is_square(n):
            return False
    raise ValueError("appropriate value for D cannot be found in is_selfridge_prp()")


def is_selfridge_prp(n):
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_selfridge_prp(n)


def is_strong_lucas_prp(n, p, q):
    D = p**2 - 4*q
    if D == 0:
        raise ValueError("invalid values for p,q in is_strong_lucas_prp()")
    if n < 1:
        raise ValueError("is_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    if gcd(n, q*D) not in [1, n]:
        raise ValueError("is_strong_lucas_prp() requires gcd(n,2*q*D) == 1")
    j = jacobi(D, n)
    s = bit_scan1(n - j)
    U, V, Qk = _lucas_sequence(n, p, q, (n - j) >> s)
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V = (V*V - 2*Qk) % n
        if V == 0:
            return True
        Qk = pow(Qk, 2, n)
    return False


def _is_strong_selfridge_prp(n):
    for D in range(5, 1_000_000, 2):
        if D & 2: # if D % 4 == 3
            D = -D
        j = jacobi(D, n)
        if j == -1:
            s = bit_scan1(n + 1)
            U, V, Qk = _lucas_sequence(n, 1, (1-D) // 4, (n + 1) >> s)
            if U == 0 or V == 0:
                return True
            for _ in range(s - 1):
                V = (V*V - 2*Qk) % n
                if V == 0:
                    return True
                Qk = pow(Qk, 2, n)
            return False
        if j == 0 and D % n:
            return False
        # When j == -1 is hard to find, suspect a square number
        if D == 13 and is_square(n):
            return False
    raise ValueError("appropriate value for D cannot be found in is_strong_selfridge_prp()")


def is_strong_selfridge_prp(n):
    if n < 1:
        raise ValueError("is_strong_selfridge_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_selfridge_prp(n)


def is_bpsw_prp(n):
    if n < 1:
        raise ValueError("is_bpsw_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_prp(n, 2) and _is_selfridge_prp(n)


def is_strong_bpsw_prp(n):
    if n < 1:
        raise ValueError("is_strong_bpsw_prp() requires 'n' be greater than 0")
    if n == 1:
        return False
    if n % 2 == 0:
        return n == 2
    return _is_strong_prp(n, 2) and _is_strong_selfridge_prp(n)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\quadrature.py ===
from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
                                    hermite_poly, jacobi_poly)
from sympy.polys.rootoftools import RootOf


def gauss_legendre(n, n_digits):
    r"""
    Computes the Gauss-Legendre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Legendre quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2}{\left(1-x_i^2\right) \left(P'_n(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_legendre
    >>> x, w = gauss_legendre(3, 5)
    >>> x
    [-0.7746, 0, 0.7746]
    >>> w
    [0.55556, 0.88889, 0.55556]
    >>> x, w = gauss_legendre(4, 5)
    >>> x
    [-0.86114, -0.33998, 0.33998, 0.86114]
    >>> w
    [0.34785, 0.65215, 0.65215, 0.34785]

    See Also
    ========

    gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/legendre_rule/legendre_rule.html
    """
    x = Dummy("x")
    p = legendre_poly(n, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/((1-r**2) * pd.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_laguerre(n, n_digits):
    r"""
    Computes the Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_0^{\infty} e^{-x} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)


    The nodes `x_i` of an order `n` quadrature rule are the roots of `L_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{x_i}{(n+1)^2 \left(L_{n+1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_laguerre
    >>> x, w = gauss_laguerre(3, 5)
    >>> x
    [0.41577, 2.2943, 6.2899]
    >>> w
    [0.71109, 0.27852, 0.010389]
    >>> x, w = gauss_laguerre(6, 5)
    >>> x
    [0.22285, 1.1889, 2.9927, 5.7751, 9.8375, 15.983]
    >>> w
    [0.45896, 0.417, 0.11337, 0.010399, 0.00026102, 8.9855e-7]

    See Also
    ========

    gauss_legendre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/laguerre_rule/laguerre_rule.html
    """
    x = Dummy("x")
    p = laguerre_poly(n, x, polys=True)
    p1 = laguerre_poly(n+1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((r/((n+1)**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_hermite(n, n_digits):
    r"""
    Computes the Gauss-Hermite quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Hermite quadrature approximates the integral:

    .. math::
        \int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `H_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{2^{n-1} n! \sqrt{\pi}}{n^2 \left(H_{n-1}(x_i)\right)^2}

    Parameters
    ==========

    n :
        The order of quadrature.
    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : The ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_hermite
    >>> x, w = gauss_hermite(3, 5)
    >>> x
    [-1.2247, 0, 1.2247]
    >>> w
    [0.29541, 1.1816, 0.29541]

    >>> x, w = gauss_hermite(6, 5)
    >>> x
    [-2.3506, -1.3358, -0.43608, 0.43608, 1.3358, 2.3506]
    >>> w
    [0.00453, 0.15707, 0.72463, 0.72463, 0.15707, 0.00453]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_gen_laguerre, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss-Hermite_Quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/hermite_rule/hermite_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_hermite_rule/gen_hermite_rule.html
    """
    x = Dummy("x")
    p = hermite_poly(n, x, polys=True)
    p1 = hermite_poly(n-1, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append(((2**(n-1) * factorial(n) * sqrt(pi)) /
                 (n**2 * p1.subs(x, r)**2)).n(n_digits))
    return xi, w


def gauss_gen_laguerre(n, alpha, n_digits):
    r"""
    Computes the generalized Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The generalized Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \int_{0}^\infty x^{\alpha} e^{-x} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `L^{\alpha}_n` and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\Gamma(\alpha+n)}
                {n \Gamma(n) L^{\alpha}_{n-1}(x_i) L^{\alpha+1}_{n-1}(x_i)}

    Parameters
    ==========

    n :
        The order of quadrature.

    alpha :
        The exponent of the singularity, `\alpha > -1`.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_gen_laguerre
    >>> x, w = gauss_gen_laguerre(3, -S.Half, 5)
    >>> x
    [0.19016, 1.7845, 5.5253]
    >>> w
    [1.4493, 0.31413, 0.00906]

    >>> x, w = gauss_gen_laguerre(4, 3*S.Half, 5)
    >>> x
    [0.97851, 2.9904, 6.3193, 11.712]
    >>> w
    [0.53087, 0.67721, 0.11895, 0.0023152]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_laguerre_rule/gen_laguerre_rule.html
    """
    x = Dummy("x")
    p = laguerre_poly(n, x, alpha=alpha, polys=True)
    p1 = laguerre_poly(n-1, x, alpha=alpha, polys=True)
    p2 = laguerre_poly(n-1, x, alpha=alpha+1, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((gamma(alpha+n) /
                 (n*gamma(n)*p1.subs(x, r)*p2.subs(x, r))).n(n_digits))
    return xi, w


def gauss_chebyshev_t(n, n_digits):
    r"""
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the first kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^{1} \frac{1}{\sqrt{1-x^2}} f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `T_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\pi}{n}

    Parameters
    ==========

    n :
        The order of quadrature.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_t
    >>> x, w = gauss_chebyshev_t(3, 5)
    >>> x
    [0.86602, 0, -0.86602]
    >>> w
    [1.0472, 1.0472, 1.0472]

    >>> x, w = gauss_chebyshev_t(6, 5)
    >>> x
    [0.96593, 0.70711, 0.25882, -0.25882, -0.70711, -0.96593]
    >>> w
    [0.5236, 0.5236, 0.5236, 0.5236, 0.5236, 0.5236]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev1_rule/chebyshev1_rule.html
    """
    xi = []
    w = []
    for i in range(1, n+1):
        xi.append((cos((2*i-S.One)/(2*n)*S.Pi)).n(n_digits))
        w.append((S.Pi/n).n(n_digits))
    return xi, w


def gauss_chebyshev_u(n, n_digits):
    r"""
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the second kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the second kind approximates the
    integral:

    .. math::
        \int_{-1}^{1} \sqrt{1-x^2} f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `U_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \frac{\pi}{n+1} \sin^2 \left(\frac{i}{n+1}\pi\right)

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_u
    >>> x, w = gauss_chebyshev_u(3, 5)
    >>> x
    [0.70711, 0, -0.70711]
    >>> w
    [0.3927, 0.7854, 0.3927]

    >>> x, w = gauss_chebyshev_u(6, 5)
    >>> x
    [0.90097, 0.62349, 0.22252, -0.22252, -0.62349, -0.90097]
    >>> w
    [0.084489, 0.27433, 0.42658, 0.42658, 0.27433, 0.084489]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_t, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev2_rule/chebyshev2_rule.html
    """
    xi = []
    w = []
    for i in range(1, n+1):
        xi.append((cos(i/(n+S.One)*S.Pi)).n(n_digits))
        w.append((S.Pi/(n+S.One)*sin(i*S.Pi/(n+S.One))**2).n(n_digits))
    return xi, w


def gauss_jacobi(n, alpha, beta, n_digits):
    r"""
    Computes the Gauss-Jacobi quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Jacobi quadrature of the first kind approximates the integral:

    .. math::
        \int_{-1}^1 (1-x)^\alpha (1+x)^\beta f(x)\,dx \approx
            \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `P^{(\alpha,\beta)}_n` and the weights `w_i` are given by:

    .. math::
        w_i = -\frac{2n+\alpha+\beta+2}{n+\alpha+\beta+1}
              \frac{\Gamma(n+\alpha+1)\Gamma(n+\beta+1)}
              {\Gamma(n+\alpha+\beta+1)(n+1)!}
              \frac{2^{\alpha+\beta}}{P'_n(x_i)
              P^{(\alpha,\beta)}_{n+1}(x_i)}

    Parameters
    ==========

    n : the order of quadrature

    alpha : the first parameter of the Jacobi Polynomial, `\alpha > -1`

    beta : the second parameter of the Jacobi Polynomial, `\beta > -1`

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_jacobi
    >>> x, w = gauss_jacobi(3, S.Half, -S.Half, 5)
    >>> x
    [-0.90097, -0.22252, 0.62349]
    >>> w
    [1.7063, 1.0973, 0.33795]

    >>> x, w = gauss_jacobi(6, 1, 1, 5)
    >>> x
    [-0.87174, -0.5917, -0.2093, 0.2093, 0.5917, 0.87174]
    >>> w
    [0.050584, 0.22169, 0.39439, 0.39439, 0.22169, 0.050584]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre,
    gauss_chebyshev_t, gauss_chebyshev_u, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_rule/jacobi_rule.html
    .. [3] https://people.sc.fsu.edu/~jburkardt/cpp_src/gegenbauer_rule/gegenbauer_rule.html
    """
    x = Dummy("x")
    p = jacobi_poly(n, alpha, beta, x, polys=True)
    pd = p.diff(x)
    pn = jacobi_poly(n+1, alpha, beta, x, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((
            - (2*n+alpha+beta+2) / (n+alpha+beta+S.One) *
            (gamma(n+alpha+1)*gamma(n+beta+1)) /
            (gamma(n+alpha+beta+S.One)*gamma(n+2)) *
            2**(alpha+beta) / (pd.subs(x, r) * pn.subs(x, r))).n(n_digits))
    return xi, w


def gauss_lobatto(n, n_digits):
    r"""
    Computes the Gauss-Lobatto quadrature [1]_ points and weights.

    Explanation
    ===========

    The Gauss-Lobatto quadrature approximates the integral:

    .. math::
        \int_{-1}^1 f(x)\,dx \approx \sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `P'_(n-1)`
    and the weights `w_i` are given by:

    .. math::
        &w_i = \frac{2}{n(n-1) \left[P_{n-1}(x_i)\right]^2},\quad x\neq\pm 1\\
        &w_i = \frac{2}{n(n-1)},\quad x=\pm 1

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_lobatto
    >>> x, w = gauss_lobatto(3, 5)
    >>> x
    [-1, 0, 1]
    >>> w
    [0.33333, 1.3333, 0.33333]
    >>> x, w = gauss_lobatto(4, 5)
    >>> x
    [-1, -0.44721, 0.44721, 1]
    >>> w
    [0.16667, 0.83333, 0.83333, 0.16667]

    See Also
    ========

    gauss_legendre,gauss_laguerre, gauss_gen_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    .. [2] https://web.archive.org/web/20200118141346/http://people.math.sfu.ca/~cbm/aands/page_888.htm
    """
    x = Dummy("x")
    p = legendre_poly(n-1, x, polys=True)
    pd = p.diff(x)
    xi = []
    w = []
    for r in pd.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One/10**(n_digits+2))
        xi.append(r.n(n_digits))
        w.append((2/(n*(n-1) * p.subs(x, r)**2)).n(n_digits))

    xi.insert(0, -1)
    xi.append(1)
    w.insert(0, (S(2)/(n*(n-1))).n(n_digits))
    w.append((S(2)/(n*(n-1))).n(n_digits))
    return xi, w

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\period.py ===
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import index as libindex
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    Period,
    Resolution,
    Tick,
)
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_integer
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
    raise_on_incompatible,
    validate_dtype_freq,
)
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import maybe_extract_name
from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from pandas.core.indexes.datetimes import (
    DatetimeIndex,
    Index,
)
from pandas.core.indexes.extension import inherit_names

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )


_index_doc_kwargs = dict(ibase._index_doc_kwargs)
_index_doc_kwargs.update({"target_klass": "PeriodIndex or list of Periods"})
_shared_doc_kwargs = {
    "klass": "PeriodArray",
}

# --- Period index sketch


def _new_PeriodIndex(cls, **d):
    # GH13277 for unpickling
    values = d.pop("data")
    if values.dtype == "int64":
        freq = d.pop("freq", None)
        dtype = PeriodDtype(freq)
        values = PeriodArray(values, dtype=dtype)
        return cls._simple_new(values, **d)
    else:
        return cls(values, **d)


@inherit_names(
    ["strftime", "start_time", "end_time"] + PeriodArray._field_ops,
    PeriodArray,
    wrap=True,
)
@inherit_names(["is_leap_year"], PeriodArray)
class PeriodIndex(DatetimeIndexOpsMixin):
    """
    Immutable ndarray holding ordinal values indicating regular periods in time.

    Index keys are boxed to Period objects which carries the metadata (eg,
    frequency information).

    Parameters
    ----------
    data : array-like (1d int np.ndarray or PeriodArray), optional
        Optional period-like data to construct index with.
    copy : bool
        Make a copy of input ndarray.
    freq : str or period object, optional
        One of pandas period strings or corresponding objects.
    year : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    month : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    quarter : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    day : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    hour : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    minute : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    second : int, array, or Series, default None

        .. deprecated:: 2.2.0
           Use PeriodIndex.from_fields instead.
    dtype : str or PeriodDtype, default None

    Attributes
    ----------
    day
    dayofweek
    day_of_week
    dayofyear
    day_of_year
    days_in_month
    daysinmonth
    end_time
    freq
    freqstr
    hour
    is_leap_year
    minute
    month
    quarter
    qyear
    second
    start_time
    week
    weekday
    weekofyear
    year

    Methods
    -------
    asfreq
    strftime
    to_timestamp
    from_fields
    from_ordinals

    See Also
    --------
    Index : The base pandas Index type.
    Period : Represents a period of time.
    DatetimeIndex : Index with datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    period_range : Create a fixed-frequency PeriodIndex.

    Examples
    --------
    >>> idx = pd.PeriodIndex.from_fields(year=[2000, 2002], quarter=[1, 3])
    >>> idx
    PeriodIndex(['2000Q1', '2002Q3'], dtype='period[Q-DEC]')
    """

    _typ = "periodindex"

    _data: PeriodArray
    freq: BaseOffset
    dtype: PeriodDtype

    _data_cls = PeriodArray
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.PeriodEngine]:
        return libindex.PeriodEngine

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        # for compat with DatetimeIndex
        return self.dtype._resolution_obj

    # --------------------------------------------------------------------
    # methods that dispatch to array and wrap result in Index
    # These are defined here instead of via inherit_names for mypy

    @doc(
        PeriodArray.asfreq,
        other="pandas.arrays.PeriodArray",
        other_name="PeriodArray",
        **_shared_doc_kwargs,
    )
    def asfreq(self, freq=None, how: str = "E") -> Self:
        arr = self._data.asfreq(freq, how)
        return type(self)._simple_new(arr, name=self.name)

    @doc(PeriodArray.to_timestamp)
    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeIndex:
        arr = self._data.to_timestamp(freq, how)
        return DatetimeIndex._simple_new(arr, name=self.name)

    @property
    @doc(PeriodArray.hour.fget)
    def hour(self) -> Index:
        return Index(self._data.hour, name=self.name)

    @property
    @doc(PeriodArray.minute.fget)
    def minute(self) -> Index:
        return Index(self._data.minute, name=self.name)

    @property
    @doc(PeriodArray.second.fget)
    def second(self) -> Index:
        return Index(self._data.second, name=self.name)

    # ------------------------------------------------------------------------
    # Index Constructors

    def __new__(
        cls,
        data=None,
        ordinal=None,
        freq=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
        **fields,
    ) -> Self:
        valid_field_set = {
            "year",
            "month",
            "day",
            "quarter",
            "hour",
            "minute",
            "second",
        }

        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        if not set(fields).issubset(valid_field_set):
            argument = next(iter(set(fields) - valid_field_set))
            raise TypeError(f"__new__() got an unexpected keyword argument {argument}")
        elif len(fields):
            # GH#55960
            warnings.warn(
                "Constructing PeriodIndex from fields is deprecated. Use "
                "PeriodIndex.from_fields instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if ordinal is not None:
            # GH#55960
            warnings.warn(
                "The 'ordinal' keyword in PeriodIndex is deprecated and will "
                "be removed in a future version. Use PeriodIndex.from_ordinals "
                "instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        name = maybe_extract_name(name, data, cls)

        if data is None and ordinal is None:
            # range-based.
            if not fields:
                # test_pickle_compat_construction
                cls._raise_scalar_data_error(None)
            data = cls.from_fields(**fields, freq=freq)._data
            copy = False

        elif fields:
            if data is not None:
                raise ValueError("Cannot pass both data and fields")
            raise ValueError("Cannot pass both ordinal and fields")

        else:
            freq = validate_dtype_freq(dtype, freq)

            # PeriodIndex allow PeriodIndex(period_index, freq=different)
            # Let's not encourage that kind of behavior in PeriodArray.

            if freq and isinstance(data, cls) and data.freq != freq:
                # TODO: We can do some of these with no-copy / coercion?
                # e.g. D -> 2D seems to be OK
                data = data.asfreq(freq)

            if data is None and ordinal is not None:
                ordinal = np.asarray(ordinal, dtype=np.int64)
                dtype = PeriodDtype(freq)
                data = PeriodArray(ordinal, dtype=dtype)
            elif data is not None and ordinal is not None:
                raise ValueError("Cannot pass both data and ordinal")
            else:
                # don't pass copy here, since we copy later.
                data = period_array(data=data, freq=freq)

        if copy:
            data = data.copy()

        return cls._simple_new(data, name=name, refs=refs)

    @classmethod
    def from_fields(
        cls,
        *,
        year=None,
        quarter=None,
        month=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
        freq=None,
    ) -> Self:
        fields = {
            "year": year,
            "quarter": quarter,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        arr = PeriodArray._from_fields(fields=fields, freq=freq)
        return cls._simple_new(arr)

    @classmethod
    def from_ordinals(cls, ordinals, *, freq, name=None) -> Self:
        ordinals = np.asarray(ordinals, dtype=np.int64)
        dtype = PeriodDtype(freq)
        data = PeriodArray._simple_new(ordinals, dtype=dtype)
        return cls._simple_new(data, name=name)

    # ------------------------------------------------------------------------
    # Data

    @property
    def values(self) -> npt.NDArray[np.object_]:
        return np.asarray(self, dtype=object)

    def _maybe_convert_timedelta(self, other) -> int | npt.NDArray[np.int64]:
        """
        Convert timedelta-like input to an integer multiple of self.freq

        Parameters
        ----------
        other : timedelta, np.timedelta64, DateOffset, int, np.ndarray

        Returns
        -------
        converted : int, np.ndarray[int64]

        Raises
        ------
        IncompatibleFrequency : if the input cannot be written as a multiple
            of self.freq.  Note IncompatibleFrequency subclasses ValueError.
        """
        if isinstance(other, (timedelta, np.timedelta64, Tick, np.ndarray)):
            if isinstance(self.freq, Tick):
                # _check_timedeltalike_freq_compat will raise if incompatible
                delta = self._data._check_timedeltalike_freq_compat(other)
                return delta
        elif isinstance(other, BaseOffset):
            if other.base == self.freq.base:
                return other.n

            raise raise_on_incompatible(self, other)
        elif is_integer(other):
            assert isinstance(other, int)
            return other

        # raise when input doesn't have freq
        raise raise_on_incompatible(self, None)

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return self.dtype == dtype

    # ------------------------------------------------------------------------
    # Index Methods

    def asof_locs(self, where: Index, mask: npt.NDArray[np.bool_]) -> np.ndarray:
        """
        where : array of timestamps
        mask : np.ndarray[bool]
            Array of booleans where data is not NA.
        """
        if isinstance(where, DatetimeIndex):
            where = PeriodIndex(where._values, freq=self.freq)
        elif not isinstance(where, PeriodIndex):
            raise TypeError("asof_locs `where` must be DatetimeIndex or PeriodIndex")

        return super().asof_locs(where, mask)

    @property
    def is_full(self) -> bool:
        """
        Returns True if this PeriodIndex is range-like in that all Periods
        between start and end are present, in order.
        """
        if len(self) == 0:
            return True
        if not self.is_monotonic_increasing:
            raise ValueError("Index is not monotonic")
        values = self.asi8
        return bool(((values[1:] - values[:-1]) < 2).all())

    @property
    def inferred_type(self) -> str:
        # b/c data is represented as ints make sure we can't have ambiguous
        # indexing
        return "period"

    # ------------------------------------------------------------------------
    # Indexing Methods

    def _convert_tolerance(self, tolerance, target):
        # Returned tolerance must be in dtype/units so that
        #  `|self._get_engine_target() - target._engine_target()| <= tolerance`
        #  is meaningful.  Since PeriodIndex returns int64 for engine_target,
        #  we may need to convert timedelta64 tolerance to int64.
        tolerance = super()._convert_tolerance(tolerance, target)

        if self.dtype == target.dtype:
            # convert tolerance to i8
            tolerance = self._maybe_convert_timedelta(tolerance)

        return tolerance

    def get_loc(self, key):
        """
        Get integer location for requested label.

        Parameters
        ----------
        key : Period, NaT, str, or datetime
            String or datetime key must be parsable as Period.

        Returns
        -------
        loc : int or ndarray[int64]

        Raises
        ------
        KeyError
            Key is not present in the index.
        TypeError
            If key is listlike or otherwise not hashable.
        """
        orig_key = key

        self._check_indexing_error(key)

        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except ValueError as err:
                # A string with invalid format
                raise KeyError(f"Cannot interpret '{key}' as period") from err

            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err

            if reso == self._resolution_obj:
                # the reso < self._resolution_obj case goes
                #  through _get_string_slice
                key = self._cast_partial_indexing_scalar(parsed)
            else:
                raise KeyError(key)

        elif isinstance(key, Period):
            self._disallow_mismatched_indexing(key)

        elif isinstance(key, datetime):
            key = self._cast_partial_indexing_scalar(key)

        else:
            # in particular integer, which Period constructor would cast to string
            raise KeyError(key)

        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    def _disallow_mismatched_indexing(self, key: Period) -> None:
        if key._dtype != self.dtype:
            raise KeyError(key)

    def _cast_partial_indexing_scalar(self, label: datetime) -> Period:
        try:
            period = Period(label, freq=self.freq)
        except ValueError as err:
            # we cannot construct the Period
            raise KeyError(label) from err
        return period

    @doc(DatetimeIndexOpsMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        if isinstance(label, datetime):
            label = self._cast_partial_indexing_scalar(label)

        return super()._maybe_cast_slice_bound(label, side)

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: datetime):
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        iv = Period(parsed, freq=freq)
        return (iv.asfreq(self.freq, how="start"), iv.asfreq(self.freq, how="end"))

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq=None) -> Self:
        if freq is not None:
            raise TypeError(
                f"`freq` argument is not supported for {type(self).__name__}.shift"
            )
        return self + periods


def period_range(
    start=None,
    end=None,
    periods: int | None = None,
    freq=None,
    name: Hashable | None = None,
) -> PeriodIndex:
    """
    Return a fixed frequency PeriodIndex.

    The day (calendar) is the default frequency.

    Parameters
    ----------
    start : str, datetime, date, pandas.Timestamp, or period-like, default None
        Left bound for generating periods.
    end : str, datetime, date, pandas.Timestamp, or period-like, default None
        Right bound for generating periods.
    periods : int, default None
        Number of periods to generate.
    freq : str or DateOffset, optional
        Frequency alias. By default the freq is taken from `start` or `end`
        if those are Period objects. Otherwise, the default is ``"D"`` for
        daily frequency.
    name : str, default None
        Name of the resulting PeriodIndex.

    Returns
    -------
    PeriodIndex

    Notes
    -----
    Of the three parameters: ``start``, ``end``, and ``periods``, exactly two
    must be specified.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.period_range(start='2017-01-01', end='2018-01-01', freq='M')
    PeriodIndex(['2017-01', '2017-02', '2017-03', '2017-04', '2017-05', '2017-06',
             '2017-07', '2017-08', '2017-09', '2017-10', '2017-11', '2017-12',
             '2018-01'],
            dtype='period[M]')

    If ``start`` or ``end`` are ``Period`` objects, they will be used as anchor
    endpoints for a ``PeriodIndex`` with frequency matching that of the
    ``period_range`` constructor.

    >>> pd.period_range(start=pd.Period('2017Q1', freq='Q'),
    ...                 end=pd.Period('2017Q2', freq='Q'), freq='M')
    PeriodIndex(['2017-03', '2017-04', '2017-05', '2017-06'],
                dtype='period[M]')
    """
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )
    if freq is None and (not isinstance(start, Period) and not isinstance(end, Period)):
        freq = "D"

    data, freq = PeriodArray._generate_range(start, end, periods, freq)
    dtype = PeriodDtype(freq)
    data = PeriodArray(data, dtype=dtype)
    return PeriodIndex(data, name=name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\modes\notemacs.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import os

import pyreadline3.lineeditor.history as history
import pyreadline3.lineeditor.lineobj as lineobj
import pyreadline3.logger as logger
from pyreadline3.logger import log

from . import basemode


class NotEmacsMode(basemode.BaseMode):
    mode = "notemacs"

    def __init__(self, rlobj):
        super().__init__(rlobj)

    def __repr__(self):
        return "<NotEmacsMode>"

    def _readline_from_keyboard(self):
        c = self.console
        while True:
            self._update_line()
            event = c.getkeypress()
            if self.next_meta:
                self.next_meta = False
                control, meta, shift, code = event.keyinfo
                event.keyinfo = (control, True, shift, code)

            # Process exit keys. Only exit on empty line
            if event.keyinfo in self.exit_dispatch:
                if lineobj.EndOfLine(self.l_buffer) == 0:
                    raise EOFError

            dispatch_func = self.key_dispatch.get(event.keyinfo, self.self_insert)
            log("readline from keyboard:%s" % (event.keyinfo,))
            r = None
            if dispatch_func:
                r = dispatch_func(event)
                self.l_buffer.push_undo()

            self.previous_func = dispatch_func
            if r:
                self._update_line()
                break

    def readline(self, prompt=""):
        """Try to act like GNU readline."""
        # handle startup_hook
        if self.first_prompt:
            self.first_prompt = False
            if self.startup_hook:
                try:
                    self.startup_hook()
                except BaseException:
                    print("startup hook failed")
                    traceback.print_exc()

        c = self.console
        self.l_buffer.reset_line()
        self.prompt = prompt
        self._print_prompt()

        if self.pre_input_hook:
            try:
                self.pre_input_hook()
            except BaseException:
                print("pre_input_hook failed")
                traceback.print_exc()
                self.pre_input_hook = None

        log("in readline: %s" % self.paste_line_buffer)
        if len(self.paste_line_buffer) > 0:
            self.l_buffer = lineobj.ReadlineTextBuffer(self.paste_line_buffer[0])
            self._update_line()
            self.paste_line_buffer = self.paste_line_buffer[1:]
            c.write("\r\n")
        else:
            self._readline_from_keyboard()
            c.write("\r\n")

        self.add_history(self.l_buffer.copy())

        log("returning(%s)" % self.l_buffer.get_line_text())
        return self.l_buffer.get_line_text() + "\n"

    # Methods below here are bindable emacs functions

    def beginning_of_line(self, e):  # (C-a)
        """Move to the start of the current line."""
        self.l_buffer.beginning_of_line()

    def end_of_line(self, e):  # (C-e)
        """Move to the end of the line."""
        self.l_buffer.end_of_line()

    def forward_char(self, e):  # (C-f)
        """Move forward a character."""
        self.l_buffer.forward_char()

    def backward_char(self, e):  # (C-b)
        """Move back a character."""
        self.l_buffer.backward_char()

    def forward_word(self, e):  # (M-f)
        """Move forward to the end of the next word. Words are composed of
        letters and digits."""
        self.l_buffer.forward_word()

    def backward_word(self, e):  # (M-b)
        """Move back to the start of the current or previous word. Words are
        composed of letters and digits."""
        self.l_buffer.backward_word()

    def clear_screen(self, e):  # (C-l)
        """Clear the screen and redraw the current line, leaving the current
        line at the top of the screen."""
        self.console.page()

    def redraw_current_line(self, e):  # ()
        """Refresh the current line. By default, this is unbound."""
        pass

    def accept_line(self, e):  # (Newline or Return)
        """Accept the line regardless of where the cursor is. If this line
        is non-empty, it may be added to the history list for future recall
        with add_history(). If this line is a modified history line, the
        history line is restored to its original state."""
        return True

    # History commands
    def previous_history(self, e):  # (C-p)
        """Move back through the history list, fetching the previous command."""
        self._history.previous_history(self.l_buffer)

    def next_history(self, e):  # (C-n)
        """Move forward through the history list, fetching the next command."""
        self._history.next_history(self.l_buffer)

    def beginning_of_history(self, e):  # (M-<)
        """Move to the first line in the history."""
        self._history.beginning_of_history()

    def end_of_history(self, e):  # (M->)
        """Move to the end of the input history, i.e., the line currently
        being entered."""
        self._history.end_of_history(self.l_buffer)

    def _i_search(self, searchfun, direction, init_event):
        c = self.console
        line = self.get_line_buffer()
        query = ""
        hc_start = self._history.history_cursor  # + direction
        while True:
            x, y = self.prompt_end_pos
            c.pos(0, y)
            if direction < 0:
                prompt = "reverse-i-search"
            else:
                prompt = "forward-i-search"

            scroll = c.write_scrolling("%s`%s': %s" % (prompt, query, line))
            self._update_prompt_pos(scroll)
            self._clear_after()

            event = c.getkeypress()
            if event.keysym == "BackSpace":
                if len(query) > 0:
                    query = query[:-1]
                    self._history.history_cursor = hc_start
                else:
                    self._bell()
            elif (
                event.char in string.letters + string.digits + string.punctuation + " "
            ):
                self._history.history_cursor = hc_start
                query += event.char
            elif event.keyinfo == init_event.keyinfo:
                self._history.history_cursor += direction
                line = searchfun(query)
                pass
            else:
                if event.keysym != "Return":
                    self._bell()
                break
            line = searchfun(query)

        px, py = self.prompt_begin_pos
        c.pos(0, py)
        self.l_buffer.set_line(line)
        self._print_prompt()
        self._history.history_cursor = len(self._history.history)

    def reverse_search_history(self, e):  # (C-r)
        """Search backward starting at the current line and moving up
        through the history as necessary. This is an incremental search."""
        #        print("HEJ")
        #        self.console.bell()
        self._i_search(self._history.reverse_search_history, -1, e)

    def forward_search_history(self, e):  # (C-s)
        """Search forward starting at the current line and moving down
        through the the history as necessary. This is an incremental search."""
        #        print("HEJ")
        #        self.console.bell()
        self._i_search(self._history.forward_search_history, 1, e)

    def non_incremental_reverse_search_history(self, e):  # (M-p)
        """Search backward starting at the current line and moving up
        through the history as necessary using a non-incremental search for
        a string supplied by the user."""
        self._history.non_incremental_reverse_search_history(self.l_buffer)

    def non_incremental_forward_search_history(self, e):  # (M-n)
        """Search forward starting at the current line and moving down
        through the the history as necessary using a non-incremental search
        for a string supplied by the user."""
        self._history.non_incremental_reverse_search_history(self.l_buffer)

    def history_search_forward(self, e):  # ()
        """Search forward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        self.l_buffer = self._history.history_search_forward(self.l_buffer)

    def history_search_backward(self, e):  # ()
        """Search backward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        self.l_buffer = self._history.history_search_backward(self.l_buffer)

    def yank_nth_arg(self, e):  # (M-C-y)
        """Insert the first argument to the previous command (usually the
        second word on the previous line) at point. With an argument n,
        insert the nth word from the previous command (the words in the
        previous command begin with word 0). A negative argument inserts the
        nth word from the end of the previous command."""
        pass

    def yank_last_arg(self, e):  # (M-. or M-_)
        """Insert last argument to the previous command (the last word of
        the previous history entry). With an argument, behave exactly like
        yank-nth-arg. Successive calls to yank-last-arg move back through
        the history list, inserting the last argument of each line in turn."""
        pass

    def delete_char(self, e):  # (C-d)
        """Delete the character at point. If point is at the beginning of
        the line, there are no characters in the line, and the last
        character typed was not bound to delete-char, then return EOF."""
        self.l_buffer.delete_char()

    def backward_delete_char(self, e):  # (Rubout)
        """Delete the character behind the cursor. A numeric argument means
        to kill the characters instead of deleting them."""
        self.l_buffer.backward_delete_char()

    def forward_backward_delete_char(self, e):  # ()
        """Delete the character under the cursor, unless the cursor is at
        the end of the line, in which case the character behind the cursor
        is deleted. By default, this is not bound to a key."""
        pass

    def quoted_insert(self, e):  # (C-q or C-v)
        """Add the next character typed to the line verbatim. This is how to
        insert key sequences like C-q, for example."""
        e = self.console.getkeypress()
        self.insert_text(e.char)

    def tab_insert(self, e):  # (M-TAB)
        """Insert a tab character."""
        cursor = min(self.l_buffer.point, len(self.l_buffer.line_buffer))
        ws = " " * (self.tabstop - (cursor % self.tabstop))
        self.insert_text(ws)

    def self_insert(self, e):  # (a, b, A, 1, !, ...)
        """Insert yourself."""
        if (
            ord(e.char) != 0
        ):  # don't insert null character in buffer, can happen with dead keys.
            self.insert_text(e.char)

    def transpose_chars(self, e):  # (C-t)
        """Drag the character before the cursor forward over the character
        at the cursor, moving the cursor forward as well. If the insertion
        point is at the end of the line, then this transposes the last two
        characters of the line. Negative arguments have no effect."""
        self.l_buffer.transpose_chars()

    def transpose_words(self, e):  # (M-t)
        """Drag the word before point past the word after point, moving
        point past that word as well. If the insertion point is at the end
        of the line, this transposes the last two words on the line."""
        self.l_buffer.transpose_words()

    def upcase_word(self, e):  # (M-u)
        """Uppercase the current (or following) word. With a negative
        argument, uppercase the previous word, but do not move the cursor."""
        self.l_buffer.upcase_word()

    def downcase_word(self, e):  # (M-l)
        """Lowercase the current (or following) word. With a negative
        argument, lowercase the previous word, but do not move the cursor."""
        self.l_buffer.downcase_word()

    def capitalize_word(self, e):  # (M-c)
        """Capitalize the current (or following) word. With a negative
        argument, capitalize the previous word, but do not move the cursor."""
        self.l_buffer.capitalize_word()

    def overwrite_mode(self, e):  # ()
        """Toggle overwrite mode. With an explicit positive numeric
        argument, switches to overwrite mode. With an explicit non-positive
        numeric argument, switches to insert mode. This command affects only
        emacs mode; vi mode does overwrite differently. Each call to
        readline() starts in insert mode. In overwrite mode, characters
        bound to self-insert replace the text at point rather than pushing
        the text to the right. Characters bound to backward-delete-char
        replace the character before point with a space."""
        pass

    def kill_line(self, e):  # (C-k)
        """Kill the text from point to the end of the line."""
        self.l_buffer.kill_line()

    def backward_kill_line(self, e):  # (C-x Rubout)
        """Kill backward to the beginning of the line."""
        self.l_buffer.backward_kill_line()

    def unix_line_discard(self, e):  # (C-u)
        """Kill backward from the cursor to the beginning of the current line."""
        # how is this different from backward_kill_line?
        self.l_buffer.unix_line_discard()

    def kill_whole_line(self, e):  # ()
        """Kill all characters on the current line, no matter where point
        is. By default, this is unbound."""
        self.l_buffer.kill_whole_line()

    def kill_word(self, e):  # (M-d)
        """Kill from point to the end of the current word, or if between
        words, to the end of the next word. Word boundaries are the same as
        forward-word."""
        self.l_buffer.kill_word()

    def backward_kill_word(self, e):  # (M-DEL)
        """Kill the word behind point. Word boundaries are the same as
        backward-word."""
        self.l_buffer.backward_kill_word()

    def unix_word_rubout(self, e):  # (C-w)
        """Kill the word behind point, using white space as a word
        boundary. The killed text is saved on the kill-ring."""
        self.l_buffer.unix_word_rubout()

    def delete_horizontal_space(self, e):  # ()
        """Delete all spaces and tabs around point. By default, this is unbound."""
        pass

    def kill_region(self, e):  # ()
        """Kill the text in the current region. By default, this command is unbound."""
        pass

    def copy_region_as_kill(self, e):  # ()
        """Copy the text in the region to the kill buffer, so it can be
        yanked right away. By default, this command is unbound."""
        pass

    def copy_region_to_clipboard(self, e):  # ()
        """Copy the text in the region to the windows clipboard."""
        if self.enable_win32_clipboard:
            mark = min(self.l_buffer.mark, len(self.l_buffer.line_buffer))
            cursor = min(self.l_buffer.point, len(self.l_buffer.line_buffer))
            if self.l_buffer.mark == -1:
                return
            begin = min(cursor, mark)
            end = max(cursor, mark)
            toclipboard = "".join(self.l_buffer.line_buffer[begin:end])
            clipboard.set_clipboard_text(str(toclipboard))

    def copy_backward_word(self, e):  # ()
        """Copy the word before point to the kill buffer. The word
        boundaries are the same as backward-word. By default, this command
        is unbound."""
        pass

    def copy_forward_word(self, e):  # ()
        """Copy the word following point to the kill buffer. The word
        boundaries are the same as forward-word. By default, this command is
        unbound."""
        pass

    def paste(self, e):
        """Paste windows clipboard"""
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            self.insert_text(txt)

    def paste_mulitline_code(self, e):
        """Paste windows clipboard"""
        reg = re.compile("\r?\n")
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(False)
            t = reg.split(txt)
            t = [row for row in t if row.strip() != ""]  # remove empty lines
            if t != [""]:
                self.insert_text(t[0])
                self.add_history(self.l_buffer.copy())
                self.paste_line_buffer = t[1:]
                log("multi: %s" % self.paste_line_buffer)
                return True
            else:
                return False

    def ipython_paste(self, e):
        """Paste windows clipboard. If enable_ipython_paste_list_of_lists is
        True then try to convert tabseparated data to repr of list of lists or
        repr of array"""
        if self.enable_win32_clipboard:
            txt = clipboard.get_clipboard_text_and_convert(
                self.enable_ipython_paste_list_of_lists
            )
            if self.enable_ipython_paste_for_paths:
                if len(txt) < 300 and ("\t" not in txt) and ("\n" not in txt):
                    txt = txt.replace("\\", "/").replace(" ", r"\ ")
            self.insert_text(txt)

    def yank(self, e):  # (C-y)
        """Yank the top of the kill ring into the buffer at point."""
        pass

    def yank_pop(self, e):  # (M-y)
        """Rotate the kill-ring, and yank the new top. You can only do this
        if the prior command is yank or yank-pop."""
        pass

    def digit_argument(self, e):  # (M-0, M-1, ... M--)
        """Add this digit to the argument already accumulating, or start a
        new argument. M-- starts a negative argument."""
        pass

    def universal_argument(self, e):  # ()
        """This is another way to specify an argument. If this command is
        followed by one or more digits, optionally with a leading minus
        sign, those digits define the argument. If the command is followed
        by digits, executing universal-argument again ends the numeric
        argument, but is otherwise ignored. As a special case, if this
        command is immediately followed by a character that is neither a
        digit or minus sign, the argument count for the next command is
        multiplied by four. The argument count is initially one, so
        executing this function the first time makes the argument count
        four, a second time makes the argument count sixteen, and so on. By
        default, this is not bound to a key."""
        pass

    def delete_char_or_list(self, e):  # ()
        """Deletes the character under the cursor if not at the beginning or
        end of the line (like delete-char). If at the end of the line,
        behaves identically to possible-completions. This command is unbound
        by default."""
        pass

    def start_kbd_macro(self, e):  # (C-x ()
        """Begin saving the characters typed into the current keyboard macro."""
        pass

    def end_kbd_macro(self, e):  # (C-x ))
        """Stop saving the characters typed into the current keyboard macro
        and save the definition."""
        pass

    def call_last_kbd_macro(self, e):  # (C-x e)
        """Re-execute the last keyboard macro defined, by making the
        characters in the macro appear as if typed at the keyboard."""
        pass

    def re_read_init_file(self, e):  # (C-x C-r)
        """Read in the contents of the inputrc file, and incorporate any
        bindings or variable assignments found there."""
        pass

    def abort(self, e):  # (C-g)
        """Abort the current editing command and ring the terminals bell
        (subject to the setting of bell-style)."""
        self._bell()

    def do_uppercase_version(self, e):  # (M-a, M-b, M-x, ...)
        """If the metafied character x is lowercase, run the command that is
        bound to the corresponding uppercase character."""
        pass

    def prefix_meta(self, e):  # (ESC)
        """Metafy the next character typed. This is for keyboards without a
        meta key. Typing ESC f is equivalent to typing M-f."""
        self.next_meta = True

    def undo(self, e):  # (C-_ or C-x C-u)
        """Incremental undo, separately remembered for each line."""
        self.l_buffer.pop_undo()

    def revert_line(self, e):  # (M-r)
        """Undo all changes made to this line. This is like executing the
        undo command enough times to get back to the beginning."""
        pass

    def tilde_expand(self, e):  # (M-~)
        """Perform tilde expansion on the current word."""
        pass

    def set_mark(self, e):  # (C-@)
        """Set the mark to the point. If a numeric argument is supplied, the
        mark is set to that position."""
        self.l_buffer.set_mark()

    def exchange_point_and_mark(self, e):  # (C-x C-x)
        """Swap the point with the mark. The current cursor position is set
        to the saved position, and the old cursor position is saved as the
        mark."""
        pass

    def character_search(self, e):  # (C-])
        """A character is read and point is moved to the next occurrence of
        that character. A negative count searches for previous occurrences."""
        pass

    def character_search_backward(self, e):  # (M-C-])
        """A character is read and point is moved to the previous occurrence
        of that character. A negative count searches for subsequent
        occurrences."""
        pass

    def insert_comment(self, e):  # (M-#)
        """Without a numeric argument, the value of the comment-begin
        variable is inserted at the beginning of the current line. If a
        numeric argument is supplied, this command acts as a toggle: if the
        characters at the beginning of the line do not match the value of
        comment-begin, the value is inserted, otherwise the characters in
        comment-begin are deleted from the beginning of the line. In either
        case, the line is accepted as if a newline had been typed."""
        pass

    def dump_functions(self, e):  # ()
        """Print all of the functions and their key bindings to the Readline
        output stream. If a numeric argument is supplied, the output is
        formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        pass

    def dump_variables(self, e):  # ()
        """Print all of the settable variables and their values to the
        Readline output stream. If a numeric argument is supplied, the
        output is formatted in such a way that it can be made part of an
        inputrc file. This command is unbound by default."""
        pass

    def dump_macros(self, e):  # ()
        """Print all of the Readline key sequences bound to macros and the
        strings they output. If a numeric argument is supplied, the output
        is formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        pass

    # Create key bindings:

    def init_editing_mode(self, e):  # (C-e)
        """When in vi command mode, this causes a switch to emacs editing
        mode."""

        self._bind_exit_key("Control-d")
        self._bind_exit_key("Control-z")

        # I often accidentally hold the shift or control while typing space
        self._bind_key("Shift-space", self.self_insert)
        self._bind_key("Control-space", self.self_insert)
        self._bind_key("Return", self.accept_line)
        self._bind_key("Left", self.backward_char)
        self._bind_key("Control-b", self.backward_char)
        self._bind_key("Right", self.forward_char)
        self._bind_key("Control-f", self.forward_char)
        self._bind_key("BackSpace", self.backward_delete_char)
        self._bind_key("Home", self.beginning_of_line)
        self._bind_key("End", self.end_of_line)
        self._bind_key("Delete", self.delete_char)
        self._bind_key("Control-d", self.delete_char)
        self._bind_key("Clear", self.clear_screen)


# make it case insensitive
def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m:
        return ""
    prefix = m[0]
    for item in m:
        for i in range(len(prefix)):
            if prefix[: i + 1].lower() != item[: i + 1].lower():
                prefix = prefix[:i]
                if i == 0:
                    return ""
                break
    return prefix

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_exports.py ===
from __future__ import annotations  # isort: split

import __future__  # Regular import, not special!

import enum
import functools
import importlib
import inspect
import json
import socket as stdlib_socket
import sys
import types
from pathlib import Path, PurePath
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

import attrs
import pytest

import trio
import trio.testing
from trio._tests.pytest_plugin import RUN_SLOW, skip_if_optional_else_raise

from .. import _core, _util
from .._core._tests.tutil import slow

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

mypy_cache_updated = False


try:  # If installed, check both versions of this class.
    from typing_extensions import Protocol as Protocol_ext
except ImportError:  # pragma: no cover
    Protocol_ext = Protocol  # type: ignore[assignment]


def _ensure_mypy_cache_updated() -> None:
    # This pollutes the `empty` dir. Should this be changed?
    try:
        from mypy.api import run
    except ImportError as error:
        skip_if_optional_else_raise(error)

    global mypy_cache_updated
    if not mypy_cache_updated:
        # mypy cache was *probably* already updated by the other tests,
        # but `pytest -k ...` might run just this test on its own
        result = run(
            [
                "--config-file=",
                "--cache-dir=./.mypy_cache",
                "--no-error-summary",
                "-c",
                "import trio",
            ],
        )
        assert not result[1]  # stderr
        assert not result[0]  # stdout
        mypy_cache_updated = True


def test_core_is_properly_reexported() -> None:
    # Each export from _core should be re-exported by exactly one of these
    # three modules:
    sources = [trio, trio.lowlevel, trio.testing]
    for symbol in dir(_core):
        if symbol.startswith("_"):
            continue
        found = 0
        for source in sources:
            if symbol in dir(source) and getattr(source, symbol) is getattr(
                _core,
                symbol,
            ):
                found += 1
        print(symbol, found)
        assert found == 1


def class_is_final(cls: type) -> bool:
    """Check if a class cannot be subclassed."""
    try:
        # new_class() handles metaclasses properly, type(...) does not.
        types.new_class("SubclassTester", (cls,))
    except TypeError:
        return True
    else:
        return False


def iter_modules(
    module: types.ModuleType,
    only_public: bool,
) -> Iterator[types.ModuleType]:
    yield module
    for name, class_ in module.__dict__.items():
        if name.startswith("_") and only_public:
            continue
        if not isinstance(class_, ModuleType):
            continue
        if not class_.__name__.startswith(module.__name__):  # pragma: no cover
            continue
        if class_ is module:  # pragma: no cover
            continue
        yield from iter_modules(class_, only_public)


PUBLIC_MODULES = list(iter_modules(trio, only_public=True))
ALL_MODULES = list(iter_modules(trio, only_public=False))
PUBLIC_MODULE_NAMES = [m.__name__ for m in PUBLIC_MODULES]


# It doesn't make sense for downstream redistributors to run this test, since
# they might be using a newer version of Python with additional symbols which
# won't be reflected in trio.socket, and this shouldn't cause downstream test
# runs to start failing.
@pytest.mark.redistributors_should_skip
# Static analysis tools often have trouble with alpha releases, where Python's
# internals are in flux, grammar may not have settled down, etc.
@pytest.mark.skipif(
    sys.version_info.releaselevel == "alpha",
    reason="skip static introspection tools on Python dev/alpha releases",
)
@pytest.mark.parametrize("modname", PUBLIC_MODULE_NAMES)
@pytest.mark.parametrize("tool", ["pylint", "jedi", "mypy", "pyright_verifytypes"])
@pytest.mark.filterwarnings(
    # https://github.com/pypa/setuptools/issues/3274
    "ignore:module 'sre_constants' is deprecated:DeprecationWarning",
)
def test_static_tool_sees_all_symbols(tool: str, modname: str, tmp_path: Path) -> None:
    module = importlib.import_module(modname)

    def no_underscores(symbols: Iterable[str]) -> set[str]:
        return {symbol for symbol in symbols if not symbol.startswith("_")}

    runtime_names = no_underscores(dir(module))

    # ignore deprecated module `tests` being invisible
    if modname == "trio":
        runtime_names.discard("tests")

    # Ignore any __future__ feature objects, if imported under that name.
    for name in __future__.all_feature_names:
        if getattr(module, name, None) is getattr(__future__, name):
            runtime_names.remove(name)

    if tool == "pylint":
        try:
            from pylint.lint import PyLinter
        except ImportError as error:
            skip_if_optional_else_raise(error)

        linter = PyLinter()
        assert module.__file__ is not None
        ast = linter.get_ast(module.__file__, modname)
        static_names = no_underscores(ast)  # type: ignore[arg-type]
    elif tool == "jedi":
        if sys.implementation.name != "cpython":
            pytest.skip("jedi does not support pypy")

        try:
            import jedi
        except ImportError as error:
            skip_if_optional_else_raise(error)

        # Simulate typing "import trio; trio.<TAB>"
        script = jedi.Script(f"import {modname}; {modname}.")
        completions = script.complete()
        static_names = no_underscores(c.name for c in completions)
    elif tool == "mypy":
        if not RUN_SLOW:  # pragma: no cover
            pytest.skip("use --run-slow to check against mypy")

        cache = Path.cwd() / ".mypy_cache"

        _ensure_mypy_cache_updated()

        trio_cache = next(cache.glob("*/trio"))
        _, modname = (modname + ".").split(".", 1)
        modname = modname[:-1]
        mod_cache = trio_cache / modname if modname else trio_cache
        if mod_cache.is_dir():  # pragma: no coverage
            mod_cache = mod_cache / "__init__.data.json"
        else:
            mod_cache = trio_cache / (modname + ".data.json")

        assert mod_cache.exists()
        assert mod_cache.is_file()
        with mod_cache.open() as cache_file:
            cache_json = json.loads(cache_file.read())
            static_names = no_underscores(
                key
                for key, value in cache_json["names"].items()
                if not key.startswith(".") and value["kind"] == "Gdef"
            )
    elif tool == "pyright_verifytypes":
        if not RUN_SLOW:  # pragma: no cover
            pytest.skip("use --run-slow to check against pyright")

        try:
            import pyright  # noqa: F401
        except ImportError as error:
            skip_if_optional_else_raise(error)
        import subprocess

        res = subprocess.run(
            ["pyright", f"--verifytypes={modname}", "--outputjson"],
            capture_output=True,
        )
        current_result = json.loads(res.stdout)

        static_names = {
            x["name"][len(modname) + 1 :]
            for x in current_result["typeCompleteness"]["symbols"]
            if x["name"].startswith(modname)
        }
    else:  # pragma: no cover
        raise AssertionError()

    # It's expected that the static set will contain more names than the
    # runtime set:
    # - static tools are sometimes sloppy and include deleted names
    # - some symbols are platform-specific at runtime, but always show up in
    #   static analysis (e.g. in trio.socket or trio.lowlevel)
    # So we check that the runtime names are a subset of the static names.
    missing_names = runtime_names - static_names

    # ignore warnings about deprecated module tests
    missing_names -= {"tests"}

    if missing_names:  # pragma: no cover
        print(f"{tool} can't see the following names in {modname}:")
        print()
        for name in sorted(missing_names):
            print(f"    {name}")
        raise AssertionError()


# this could be sped up by only invoking mypy once per module, or even once for all
# modules, instead of once per class.
@slow
# see comment on test_static_tool_sees_all_symbols
@pytest.mark.redistributors_should_skip
# Static analysis tools often have trouble with alpha releases, where Python's
# internals are in flux, grammar may not have settled down, etc.
@pytest.mark.skipif(
    sys.version_info.releaselevel == "alpha",
    reason="skip static introspection tools on Python dev/alpha releases",
)
@pytest.mark.parametrize("module_name", PUBLIC_MODULE_NAMES)
@pytest.mark.parametrize("tool", ["jedi", "mypy"])
def test_static_tool_sees_class_members(
    tool: str,
    module_name: str,
    tmp_path: Path,
) -> None:
    module = PUBLIC_MODULES[PUBLIC_MODULE_NAMES.index(module_name)]

    # ignore hidden, but not dunder, symbols
    def no_hidden(symbols: Iterable[str]) -> set[str]:
        return {
            symbol
            for symbol in symbols
            if (not symbol.startswith("_")) or symbol.startswith("__")
        }

    if tool == "jedi" and sys.implementation.name != "cpython":
        pytest.skip("jedi does not support pypy")

    if tool == "mypy":
        cache = Path.cwd() / ".mypy_cache"

        _ensure_mypy_cache_updated()

        trio_cache = next(cache.glob("*/trio"))
        modname = module_name
        _, modname = (modname + ".").split(".", 1)
        modname = modname[:-1]
        mod_cache = trio_cache / modname if modname else trio_cache
        if mod_cache.is_dir():
            mod_cache = mod_cache / "__init__.data.json"
        else:
            mod_cache = trio_cache / (modname + ".data.json")

        assert mod_cache.exists()
        assert mod_cache.is_file()
        with mod_cache.open() as cache_file:
            cache_json = json.loads(cache_file.read())

        # skip a bunch of file-system activity (probably can un-memoize?)
        @functools.lru_cache
        def lookup_symbol(symbol: str) -> dict[str, str]:
            topname, *modname, name = symbol.split(".")
            version = next(cache.glob("3.*/"))
            mod_cache = version / topname
            if not mod_cache.is_dir():
                mod_cache = version / (topname + ".data.json")

            if modname:
                for piece in modname[:-1]:
                    mod_cache /= piece
                next_cache = mod_cache / modname[-1]
                if next_cache.is_dir():  # pragma: no coverage
                    mod_cache = next_cache / "__init__.data.json"
                else:
                    mod_cache = mod_cache / (modname[-1] + ".data.json")
            elif mod_cache.is_dir():
                mod_cache /= "__init__.data.json"
            with mod_cache.open() as f:
                return json.loads(f.read())["names"][name]  # type: ignore[no-any-return]

    errors: dict[str, object] = {}
    for class_name, class_ in module.__dict__.items():
        if not isinstance(class_, type):
            continue
        if module_name == "trio.socket" and class_name in dir(stdlib_socket):
            continue

        # Ignore classes that don't use attrs, they only define their members once
        # __init__ is called (and reason they don't use attrs is because they're going
        # to be reimplemented in pytest).
        # Not 100% that's the case, and it works locally, so whatever /shrug
        if class_ is trio.testing.RaisesGroup or class_ is trio.testing.Matcher:
            continue

        # dir() and inspect.getmembers doesn't display properties from the metaclass
        # also ignore some dunder methods that tend to differ but are of no consequence
        ignore_names = set(dir(type(class_))) | {
            "__annotations__",
            "__attrs_attrs__",
            "__attrs_own_setattr__",
            "__callable_proto_members_only__",
            "__class_getitem__",
            "__final__",
            "__getstate__",
            "__match_args__",
            "__order__",
            "__orig_bases__",
            "__parameters__",
            "__protocol_attrs__",
            "__setstate__",
            "__slots__",
            "__weakref__",
            # ignore errors about dunders inherited from stdlib that tools might
            # not see
            "__copy__",
            "__deepcopy__",
        }

        if type(class_) is type:
            # C extension classes don't have these dunders, but Python classes do
            ignore_names.add("__firstlineno__")
            ignore_names.add("__static_attributes__")

        # pypy seems to have some additional dunders that differ
        if sys.implementation.name == "pypy":
            ignore_names |= {
                "__basicsize__",
                "__dictoffset__",
                "__itemsize__",
                "__sizeof__",
                "__weakrefoffset__",
                "__unicode__",
            }

        # inspect.getmembers sees `name` and `value` in Enums, otherwise
        # it behaves the same way as `dir`
        # runtime_names = no_underscores(dir(class_))
        runtime_names = (
            no_hidden(x[0] for x in inspect.getmembers(class_)) - ignore_names
        )

        if tool == "jedi":
            try:
                import jedi
            except ImportError as error:
                skip_if_optional_else_raise(error)

            script = jedi.Script(
                f"from {module_name} import {class_name}; {class_name}.",
            )
            completions = script.complete()
            static_names = no_hidden(c.name for c in completions) - ignore_names

        elif tool == "mypy":
            # load the cached type information
            cached_type_info = cache_json["names"][class_name]
            assert (
                "node" not in cached_type_info
            ), "previously this was an 'if' but it seems it's no longer possible for this cache to contain 'node', if this assert raises for you please let us know!"
            cached_type_info = lookup_symbol(cached_type_info["cross_ref"])

            assert "node" in cached_type_info
            node = cached_type_info["node"]
            static_names = no_hidden(
                k for k in node.get("names", ()) if not k.startswith(".")
            )
            for symbol in node["mro"][1:]:
                node = lookup_symbol(symbol)["node"]
                static_names |= no_hidden(
                    k for k in node.get("names", ()) if not k.startswith(".")
                )
            static_names -= ignore_names

        else:  # pragma: no cover
            raise AssertionError("unknown tool")

        missing = runtime_names - static_names
        extra = static_names - runtime_names

        # using .remove() instead of .delete() to get an error in case they start not
        # being missing

        if (
            tool == "jedi"
            and BaseException in class_.__mro__
            and sys.version_info >= (3, 11)
        ):
            missing.remove("add_note")

        if (
            tool == "mypy"
            and BaseException in class_.__mro__
            and sys.version_info >= (3, 11)
        ):
            extra.remove("__notes__")

        if tool == "mypy" and attrs.has(class_):
            # e.g. __trio__core__run_CancelScope_AttrsAttributes__
            before = len(extra)
            extra = {e for e in extra if not e.endswith("AttrsAttributes__")}
            assert len(extra) == before - 1

        # mypy does not see these attributes in Enum subclasses
        if (
            tool == "mypy"
            and enum.Enum in class_.__mro__
            and sys.version_info >= (3, 12)
        ):
            # Another attribute, in 3.12+ only.
            extra.remove("__signature__")

        # TODO: this *should* be visible via `dir`!!
        if tool == "mypy" and class_ == trio.Nursery:
            extra.remove("cancel_scope")

        # These are (mostly? solely?) *runtime* attributes, often set in
        # __init__, which doesn't show up with dir() or inspect.getmembers,
        # but we get them in the way we query mypy & jedi
        EXTRAS = {
            trio.DTLSChannel: {"peer_address", "endpoint"},
            trio.DTLSEndpoint: {"socket", "incoming_packets_buffer"},
            trio.Process: {"args", "pid", "stderr", "stdin", "stdio", "stdout"},
            trio.SSLListener: {"transport_listener"},
            trio.SSLStream: {"transport_stream"},
            trio.SocketListener: {"socket"},
            trio.SocketStream: {"socket"},
            trio.testing.MemoryReceiveStream: {"close_hook", "receive_some_hook"},
            trio.testing.MemorySendStream: {
                "close_hook",
                "send_all_hook",
                "wait_send_all_might_not_block_hook",
            },
        }
        if tool == "mypy" and class_ in EXTRAS:
            before = len(extra)
            extra -= EXTRAS[class_]
            assert len(extra) == before - len(EXTRAS[class_])

        # TODO: why is this? Is it a problem?
        # see https://github.com/python-trio/trio/pull/2631#discussion_r1185615916
        if class_ == trio.StapledStream:
            extra.remove("receive_stream")
            extra.remove("send_stream")

        # I have not researched why these are missing, should maybe create an issue
        # upstream with jedi
        if tool == "jedi" and sys.version_info >= (3, 11):
            if class_ in (
                trio.DTLSChannel,
                trio.MemoryReceiveChannel,
                trio.MemorySendChannel,
                trio.SSLListener,
                trio.SocketListener,
            ):
                missing.remove("__aenter__")
                missing.remove("__aexit__")
            if class_ in (trio.DTLSChannel, trio.MemoryReceiveChannel):
                missing.remove("__aiter__")
                missing.remove("__anext__")

        if class_ in (trio.Path, trio.WindowsPath, trio.PosixPath):
            # These are from inherited subclasses.
            missing -= PurePath.__dict__.keys()
            # These are unix-only.
            if tool == "mypy" and sys.platform == "win32":
                missing -= {"owner", "is_mount", "group"}
            if tool == "jedi" and sys.platform == "win32":
                extra -= {"owner", "is_mount", "group"}

        # not sure why jedi in particular ignores this (static?) method in 3.13
        # (especially given the method is from 3.12....)
        if (
            tool == "jedi"
            and sys.version_info >= (3, 13)
            and class_ in (trio.Path, trio.WindowsPath, trio.PosixPath)
        ):
            missing.remove("with_segments")

        if sys.version_info >= (3, 13) and attrs.has(class_):
            missing.remove("__replace__")

        if missing or extra:  # pragma: no cover
            errors[f"{module_name}.{class_name}"] = {
                "missing": missing,
                "extra": extra,
            }

    # `assert not errors` will not print the full content of errors, even with
    # `--verbose`, so we manually print it
    if errors:  # pragma: no cover
        from pprint import pprint

        print(f"\n{tool} can't see the following symbols in {module_name}:")
        pprint(errors)
    assert not errors


def test_nopublic_is_final() -> None:
    """Check all NoPublicConstructor classes are also @final."""
    assert class_is_final(_util.NoPublicConstructor)  # This is itself final.

    for module in ALL_MODULES:
        for class_ in module.__dict__.values():
            if isinstance(class_, _util.NoPublicConstructor):
                assert class_is_final(class_)


def test_classes_are_final() -> None:
    # Sanity checks.
    assert not class_is_final(object)
    assert class_is_final(bool)

    for module in PUBLIC_MODULES:
        for name, class_ in module.__dict__.items():
            if not isinstance(class_, type):
                continue
            # Deprecated classes are exported with a leading underscore
            if name.startswith("_"):  # pragma: no cover
                continue

            # Abstract classes can be subclassed, because that's the whole
            # point of ABCs
            if inspect.isabstract(class_):
                continue
            # Same with protocols, but only direct children.
            if Protocol in class_.__bases__ or Protocol_ext in class_.__bases__:
                continue
            # Exceptions are allowed to be subclassed, because exception
            # subclassing isn't used to inherit behavior.
            if issubclass(class_, BaseException):
                continue
            # These are classes that are conceptually abstract, but
            # inspect.isabstract returns False for boring reasons.
            if class_ is trio.abc.Instrument or class_ is trio.socket.SocketType:
                continue
            # ... insert other special cases here ...

            # The `Path` class needs to support inheritance to allow `WindowsPath` and `PosixPath`.
            if class_ is trio.Path:
                continue
            # don't care about the *Statistics classes
            if name.endswith("Statistics"):
                continue

            assert class_is_final(class_)


# Plugin might not be running, especially if running from an installed version.
@pytest.mark.skipif(
    not hasattr(attrs.field, "trio_modded"),
    reason="Pytest plugin not installed.",
)
def test_pyright_recognizes_init_attributes() -> None:
    """Check whether we provide `alias` for all underscore prefixed attributes.

    Attrs always sets the `alias` attribute on fields, so a pytest plugin is used
    to monkeypatch `field()` to record whether an alias was defined in the metadata.
    See `_trio_check_attrs_aliases`.
    """
    for module in PUBLIC_MODULES:
        for class_ in module.__dict__.values():
            if not attrs.has(class_):
                continue
            if isinstance(class_, _util.NoPublicConstructor):
                continue

            attributes = [
                attr
                for attr in attrs.fields(class_)
                if attr.init
                if attr.alias
                not in (
                    attr.name,
                    # trio_original_args may not be present in autoattribs
                    attr.metadata.get("trio_original_args", {}).get("alias"),
                )
            ]

            assert attributes == [], class_

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\galois.py ===
r"""
Construct transitive subgroups of symmetric groups, useful in Galois theory.

Besides constructing instances of the :py:class:`~.PermutationGroup` class to
represent the transitive subgroups of $S_n$ for small $n$, this module provides
*names* for these groups.

In some applications, it may be preferable to know the name of a group,
rather than receive an instance of the :py:class:`~.PermutationGroup`
class, and then have to do extra work to determine which group it is, by
checking various properties.

Names are instances of ``Enum`` classes defined in this module. With a name in
hand, the name's ``get_perm_group`` method can then be used to retrieve a
:py:class:`~.PermutationGroup`.

The names used for groups in this module are taken from [1].

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict
from enum import Enum
import itertools

from sympy.combinatorics.named_groups import (
    SymmetricGroup, AlternatingGroup, CyclicGroup, DihedralGroup,
    set_symmetric_group_properties, set_alternating_group_properties,
)
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation


class S1TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S1.
    """
    S1 = "S1"

    def get_perm_group(self):
        return SymmetricGroup(1)


class S2TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S2.
    """
    S2 = "S2"

    def get_perm_group(self):
        return SymmetricGroup(2)


class S3TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S3.
    """
    A3 = "A3"
    S3 = "S3"

    def get_perm_group(self):
        if self == S3TransitiveSubgroups.A3:
            return AlternatingGroup(3)
        elif self == S3TransitiveSubgroups.S3:
            return SymmetricGroup(3)


class S4TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S4.
    """
    C4 = "C4"
    V = "V"
    D4 = "D4"
    A4 = "A4"
    S4 = "S4"

    def get_perm_group(self):
        if self == S4TransitiveSubgroups.C4:
            return CyclicGroup(4)
        elif self == S4TransitiveSubgroups.V:
            return four_group()
        elif self == S4TransitiveSubgroups.D4:
            return DihedralGroup(4)
        elif self == S4TransitiveSubgroups.A4:
            return AlternatingGroup(4)
        elif self == S4TransitiveSubgroups.S4:
            return SymmetricGroup(4)


class S5TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S5.
    """
    C5 = "C5"
    D5 = "D5"
    M20 = "M20"
    A5 = "A5"
    S5 = "S5"

    def get_perm_group(self):
        if self == S5TransitiveSubgroups.C5:
            return CyclicGroup(5)
        elif self == S5TransitiveSubgroups.D5:
            return DihedralGroup(5)
        elif self == S5TransitiveSubgroups.M20:
            return M20()
        elif self == S5TransitiveSubgroups.A5:
            return AlternatingGroup(5)
        elif self == S5TransitiveSubgroups.S5:
            return SymmetricGroup(5)


class S6TransitiveSubgroups(Enum):
    """
    Names for the transitive subgroups of S6.
    """
    C6 = "C6"
    S3 = "S3"
    D6 = "D6"
    A4 = "A4"
    G18 = "G18"
    A4xC2 = "A4 x C2"
    S4m = "S4-"
    S4p = "S4+"
    G36m = "G36-"
    G36p = "G36+"
    S4xC2 = "S4 x C2"
    PSL2F5 = "PSL2(F5)"
    G72 = "G72"
    PGL2F5 = "PGL2(F5)"
    A6 = "A6"
    S6 = "S6"

    def get_perm_group(self):
        if self == S6TransitiveSubgroups.C6:
            return CyclicGroup(6)
        elif self == S6TransitiveSubgroups.S3:
            return S3_in_S6()
        elif self == S6TransitiveSubgroups.D6:
            return DihedralGroup(6)
        elif self == S6TransitiveSubgroups.A4:
            return A4_in_S6()
        elif self == S6TransitiveSubgroups.G18:
            return G18()
        elif self == S6TransitiveSubgroups.A4xC2:
            return A4xC2()
        elif self == S6TransitiveSubgroups.S4m:
            return S4m()
        elif self == S6TransitiveSubgroups.S4p:
            return S4p()
        elif self == S6TransitiveSubgroups.G36m:
            return G36m()
        elif self == S6TransitiveSubgroups.G36p:
            return G36p()
        elif self == S6TransitiveSubgroups.S4xC2:
            return S4xC2()
        elif self == S6TransitiveSubgroups.PSL2F5:
            return PSL2F5()
        elif self == S6TransitiveSubgroups.G72:
            return G72()
        elif self == S6TransitiveSubgroups.PGL2F5:
            return PGL2F5()
        elif self == S6TransitiveSubgroups.A6:
            return AlternatingGroup(6)
        elif self == S6TransitiveSubgroups.S6:
            return SymmetricGroup(6)


def four_group():
    """
    Return a representation of the Klein four-group as a transitive subgroup
    of S4.
    """
    return PermutationGroup(
        Permutation(0, 1)(2, 3),
        Permutation(0, 2)(1, 3)
    )


def M20():
    """
    Return a representation of the metacyclic group M20, a transitive subgroup
    of S5 that is one of the possible Galois groups for polys of degree 5.

    Notes
    =====

    See [1], Page 323.

    """
    G = PermutationGroup(Permutation(0, 1, 2, 3, 4), Permutation(1, 2, 4, 3))
    G._degree = 5
    G._order = 20
    G._is_transitive = True
    G._is_sym = False
    G._is_alt = False
    G._is_cyclic = False
    G._is_dihedral = False
    return G


def S3_in_S6():
    """
    Return a representation of S3 as a transitive subgroup of S6.

    Notes
    =====

    The representation is found by viewing the group as the symmetries of a
    triangular prism.

    """
    G = PermutationGroup(Permutation(0, 1, 2)(3, 4, 5), Permutation(0, 3)(2, 4)(1, 5))
    set_symmetric_group_properties(G, 3, 6)
    return G


def A4_in_S6():
    """
    Return a representation of A4 as a transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4))
    set_alternating_group_properties(G, 4, 6)
    return G


def S4m():
    """
    Return a representation of the S4- transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3))
    set_symmetric_group_properties(G, 4, 6)
    return G


def S4p():
    """
    Return a representation of the S4+ transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(Permutation(0, 2, 4, 1)(3, 5), Permutation(0, 3)(4, 5))
    set_symmetric_group_properties(G, 4, 6)
    return G


def A4xC2():
    """
    Return a representation of the (A4 x C2) transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 1, 2)(3, 5, 4),
        Permutation(5)(2, 4))


def S4xC2():
    """
    Return a representation of the (S4 x C2) transitive subgroup of S6.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(1, 4, 5, 3), Permutation(0, 4)(1, 5)(2, 3),
        Permutation(1, 4)(3, 5))


def G18():
    """
    Return a representation of the group G18, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C2.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(0, 4)(1, 5)(2, 3))


def G36m():
    """
    Return a representation of the group G36-, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C2^2.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(1, 2)(3, 5), Permutation(0, 4)(1, 5)(2, 3))


def G36p():
    """
    Return a representation of the group G36+, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with C4.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2), Permutation(3, 4, 5),
        Permutation(0, 5, 2, 3)(1, 4))


def G72():
    """
    Return a representation of the group G72, a transitive subgroup of S6
    isomorphic to the semidirect product of C3^2 with D4.

    Notes
    =====

    See [1], Page 325.

    """
    return PermutationGroup(
        Permutation(5)(0, 1, 2),
        Permutation(0, 4, 1, 3)(2, 5), Permutation(0, 3)(1, 4)(2, 5))


def PSL2F5():
    r"""
    Return a representation of the group $PSL_2(\mathbb{F}_5)$, as a transitive
    subgroup of S6, isomorphic to $A_5$.

    Notes
    =====

    This was computed using :py:func:`~.find_transitive_subgroups_of_S6`.

    """
    G = PermutationGroup(
        Permutation(0, 4, 5)(1, 3, 2), Permutation(0, 4, 3, 1, 5))
    set_alternating_group_properties(G, 5, 6)
    return G


def PGL2F5():
    r"""
    Return a representation of the group $PGL_2(\mathbb{F}_5)$, as a transitive
    subgroup of S6, isomorphic to $S_5$.

    Notes
    =====

    See [1], Page 325.

    """
    G = PermutationGroup(
        Permutation(0, 1, 2, 3, 4), Permutation(0, 5)(1, 2)(3, 4))
    set_symmetric_group_properties(G, 5, 6)
    return G


def find_transitive_subgroups_of_S6(*targets, print_report=False):
    r"""
    Search for certain transitive subgroups of $S_6$.

    The symmetric group $S_6$ has 16 different transitive subgroups, up to
    conjugacy. Some are more easily constructed than others. For example, the
    dihedral group $D_6$ is immediately found, but it is not at all obvious how
    to realize $S_4$ or $S_5$ *transitively* within $S_6$.

    In some cases there are well-known constructions that can be used. For
    example, $S_5$ is isomorphic to $PGL_2(\mathbb{F}_5)$, which acts in a
    natural way on the projective line $P^1(\mathbb{F}_5)$, a set of order 6.

    In absence of such special constructions however, we can simply search for
    generators. For example, transitive instances of $A_4$ and $S_4$ can be
    found within $S_6$ in this way.

    Once we are engaged in such searches, it may then be easier (if less
    elegant) to find even those groups like $S_5$ that do have special
    constructions, by mere search.

    This function locates generators for transitive instances in $S_6$ of the
    following subgroups:

    * $A_4$
    * $S_4^-$ ($S_4$ not contained within $A_6$)
    * $S_4^+$ ($S_4$ contained within $A_6$)
    * $A_4 \times C_2$
    * $S_4 \times C_2$
    * $G_{18}   = C_3^2 \rtimes C_2$
    * $G_{36}^- = C_3^2 \rtimes C_2^2$
    * $G_{36}^+ = C_3^2 \rtimes C_4$
    * $G_{72}   = C_3^2 \rtimes D_4$
    * $A_5$
    * $S_5$

    Note: Each of these groups also has a dedicated function in this module
    that returns the group immediately, using generators that were found by
    this search procedure.

    The search procedure serves as a record of how these generators were
    found. Also, due to randomness in the generation of the elements of
    permutation groups, it can be called again, in order to (probably) get
    different generators for the same groups.

    Parameters
    ==========

    targets : list of :py:class:`~.S6TransitiveSubgroups` values
        The groups you want to find.

    print_report : bool (default False)
        If True, print to stdout the generators found for each group.

    Returns
    =======

    dict
        mapping each name in *targets* to the :py:class:`~.PermutationGroup`
        that was found

    References
    ==========

    .. [2] https://en.wikipedia.org/wiki/Projective_linear_group#Exceptional_isomorphisms
    .. [3] https://en.wikipedia.org/wiki/Automorphisms_of_the_symmetric_and_alternating_groups#PGL%282,5%29

    """
    def elts_by_order(G):
        """Sort the elements of a group by their order. """
        elts = defaultdict(list)
        for g in G.elements:
            elts[g.order()].append(g)
        return elts

    def order_profile(G, name=None):
        """Determine how many elements a group has, of each order. """
        elts = elts_by_order(G)
        profile = {o:len(e) for o, e in elts.items()}
        if name:
            print(f'{name}: ' + ' '.join(f'{len(profile[r])}@{r}' for r in sorted(profile.keys())))
        return profile

    S6 = SymmetricGroup(6)
    A6 = AlternatingGroup(6)
    S6_by_order = elts_by_order(S6)

    def search(existing_gens, needed_gen_orders, order, alt=None, profile=None, anti_profile=None):
        """
        Find a transitive subgroup of S6.

        Parameters
        ==========

        existing_gens : list of Permutation
            Optionally empty list of generators that must be in the group.

        needed_gen_orders : list of positive int
            Nonempty list of the orders of the additional generators that are
            to be found.

        order: int
            The order of the group being sought.

        alt: bool, None
            If True, require the group to be contained in A6.
            If False, require the group not to be contained in A6.

        profile : dict
            If given, the group's order profile must equal this.

        anti_profile : dict
            If given, the group's order profile must *not* equal this.

        """
        for gens in itertools.product(*[S6_by_order[n] for n in needed_gen_orders]):
            if len(set(gens)) < len(gens):
                continue
            G = PermutationGroup(existing_gens + list(gens))
            if G.order() == order and G.is_transitive():
                if alt is not None and G.is_subgroup(A6) != alt:
                    continue
                if profile and order_profile(G) != profile:
                    continue
                if anti_profile and order_profile(G) == anti_profile:
                    continue
                return G

    def match_known_group(G, alt=None):
        needed = [g.order() for g in G.generators]
        return search([], needed, G.order(), alt=alt, profile=order_profile(G))

    found = {}

    def finish_up(name, G):
        found[name] = G
        if print_report:
            print("=" * 40)
            print(f"{name}:")
            print(G.generators)

    if S6TransitiveSubgroups.A4 in targets or S6TransitiveSubgroups.A4xC2 in targets:
        A4_in_S6 = match_known_group(AlternatingGroup(4))
        finish_up(S6TransitiveSubgroups.A4, A4_in_S6)

    if S6TransitiveSubgroups.S4m in targets or S6TransitiveSubgroups.S4xC2 in targets:
        S4m_in_S6 = match_known_group(SymmetricGroup(4), alt=False)
        finish_up(S6TransitiveSubgroups.S4m, S4m_in_S6)

    if S6TransitiveSubgroups.S4p in targets:
        S4p_in_S6 = match_known_group(SymmetricGroup(4), alt=True)
        finish_up(S6TransitiveSubgroups.S4p, S4p_in_S6)

    if S6TransitiveSubgroups.A4xC2 in targets:
        A4xC2_in_S6 = search(A4_in_S6.generators, [2], 24, anti_profile=order_profile(SymmetricGroup(4)))
        finish_up(S6TransitiveSubgroups.A4xC2, A4xC2_in_S6)

    if S6TransitiveSubgroups.S4xC2 in targets:
        S4xC2_in_S6 = search(S4m_in_S6.generators, [2], 48)
        finish_up(S6TransitiveSubgroups.S4xC2, S4xC2_in_S6)

    # For the normal factor N = C3^2 in any of the G_n subgroups, we take one
    # obvious instance of C3^2 in S6:
    N_gens = [Permutation(5)(0, 1, 2), Permutation(5)(3, 4, 5)]

    if S6TransitiveSubgroups.G18 in targets:
        G18_in_S6 = search(N_gens, [2], 18)
        finish_up(S6TransitiveSubgroups.G18, G18_in_S6)

    if S6TransitiveSubgroups.G36m in targets:
        G36m_in_S6 = search(N_gens, [2, 2], 36, alt=False)
        finish_up(S6TransitiveSubgroups.G36m, G36m_in_S6)

    if S6TransitiveSubgroups.G36p in targets:
        G36p_in_S6 = search(N_gens, [4], 36, alt=True)
        finish_up(S6TransitiveSubgroups.G36p, G36p_in_S6)

    if S6TransitiveSubgroups.G72 in targets:
        G72_in_S6 = search(N_gens, [4, 2], 72)
        finish_up(S6TransitiveSubgroups.G72, G72_in_S6)

    # The PSL2(F5) and PGL2(F5) subgroups are isomorphic to A5 and S5, resp.

    if S6TransitiveSubgroups.PSL2F5 in targets:
        PSL2F5_in_S6 = match_known_group(AlternatingGroup(5))
        finish_up(S6TransitiveSubgroups.PSL2F5, PSL2F5_in_S6)

    if S6TransitiveSubgroups.PGL2F5 in targets:
        PGL2F5_in_S6 = match_known_group(SymmetricGroup(5))
        finish_up(S6TransitiveSubgroups.PGL2F5, PGL2F5_in_S6)

    # There is little need to "search" for any of the groups C6, S3, D6, A6,
    # or S6, since they all have obvious realizations within S6. However, we
    # support them here just in case a random representation is desired.

    if S6TransitiveSubgroups.C6 in targets:
        C6 = match_known_group(CyclicGroup(6))
        finish_up(S6TransitiveSubgroups.C6, C6)

    if S6TransitiveSubgroups.S3 in targets:
        S3 = match_known_group(SymmetricGroup(3))
        finish_up(S6TransitiveSubgroups.S3, S3)

    if S6TransitiveSubgroups.D6 in targets:
        D6 = match_known_group(DihedralGroup(6))
        finish_up(S6TransitiveSubgroups.D6, D6)

    if S6TransitiveSubgroups.A6 in targets:
        A6 = match_known_group(A6)
        finish_up(S6TransitiveSubgroups.A6, A6)

    if S6TransitiveSubgroups.S6 in targets:
        S6 = match_known_group(S6)
        finish_up(S6TransitiveSubgroups.S6, S6)

    return found

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\benchmark.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import ast
import datetime
import gc
import logging
import os
import sys
import time

import numpy as np
import psutil
import torch
import whisper
from benchmark_helper import measure_memory, setup_logger
from onnxruntime_extensions import get_library_path
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import trange
from transformers import AutoModelForSpeechSeq2Seq, WhisperConfig, WhisperProcessor

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_inputs(args: argparse.Namespace):
    if args.benchmark_type not in {"hf-pt-eager", "hf-pt-compile", "hf-ort", "ort"}:
        raise Exception("Unable to auto-detect inputs for provided model")

    def load_via_ffmpeg():
        audio = whisper.load_audio(args.audio_path)
        audio = whisper.pad_or_trim(audio)
        return audio

    def load_via_numpy():
        with open(args.audio_path, "rb") as f:
            audio = np.asarray(list(f.read()), dtype=np.uint8)
            audio = np.array([audio])
        return audio

    inputs = {
        "max_length": args.max_length,
        "min_length": args.min_length,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "length_penalty": args.length_penalty,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.benchmark_type == "ort":
        # convert_to_onnx export or ONNX E2E solution created by Olive
        for k, v in inputs.items():
            inputs[k] = np.array([v], dtype=np.float32 if "penalty" in k else np.int32)
        if args.has_decoder_input_ids:
            inputs["decoder_input_ids"] = np.array([args.decoder_input_ids], dtype=np.int32)
        if args.has_logits_processor:
            inputs["logits_processor"] = np.array([args.logits_processor], dtype=np.int32)
        if args.has_temperature:
            inputs["temperature"] = np.array([args.temperature], dtype=np.float32)

    # Measure time taken to load audio file
    logger.info(f"Load audio: {args.audio_path}")
    load_audio_fn = lambda onnx_e2e: load_via_numpy() if onnx_e2e else load_via_ffmpeg()  # noqa: E731
    time_fn(args, load_audio_fn, args.has_audio_stream)
    audio_data = load_audio_fn(args.has_audio_stream)

    if args.has_audio_stream:
        # ONNX E2E solution created by Olive
        inputs["audio_stream"] = audio_data
        return inputs

    # Measure time taken to get input features
    logger.info("Feature extraction: ")
    return_type = "np" if args.benchmark_type == "ort" else "pt"
    processor_fn = lambda audio: args.processor.feature_extractor(  # noqa: E731
        [audio], return_tensors=return_type, sampling_rate=args.sampling_rate
    ).input_features
    time_fn(args, processor_fn, audio_data)
    input_features = processor_fn(audio_data)

    if args.benchmark_type == "ort":
        # convert_to_onnx export
        inputs["input_features"] = input_features
        return inputs

    inputs["inputs"] = input_features.to(
        dtype=torch.float16 if args.use_fp16 else torch.float32, device=args.target_device
    )
    inputs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    inputs["early_stopping"] = True
    inputs["use_cache"] = True

    if args.decoder_input_ids:
        inputs["forced_decoder_ids"] = args.decoder_input_ids

    return inputs


def get_model(args: argparse.Namespace):
    model, sess_options = None, None
    start_time, end_time = None, None

    # There are multiple sources that the model could come from:
    # 1) Benchmark Whisper from Hugging Face
    # 2) Benchmark Whisper ONNX model from Optimum export (without pre/post processing)
    # 3) Benchmark Whisper ONNX E2E model from Olive (with pre/post processing)

    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        source = args.hf_pt_model_path if args.hf_pt_model_path else args.model_name
        start_time = time.time()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            source,
            torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
            use_cache=True,
        ).to(args.target_device)
        end_time = time.time()

        if args.benchmark_type == "hf-pt-compile":
            model = torch.compile(model)

    elif args.benchmark_type in {"hf-ort", "ort"}:
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = args.profile
        sess_options.register_custom_ops_library(get_library_path())
        if args.verbose:
            sess_options.log_verbosity_level = 1
            sess_options.log_severity_level = 1
            if args.tune:
                ort.set_default_logger_severity(0)
                ort.set_default_logger_verbosity(0)

    else:
        raise Exception(f"Cannot recognize {args.benchmark_type}")

    if args.benchmark_type == "hf-ort":
        # Optimum export
        provider = args.execution_provider[0] if type(args.execution_provider) is tuple else args.execution_provider
        provider_options = args.execution_provider[1] if type(args.execution_provider) is tuple else None

        start_time = time.time()
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            args.hf_ort_dir_path,
            provider=provider,
            provider_options=provider_options,
            session_options=sess_options,
            use_io_binding=True,  # Avoid memory copy overhead
        )
        end_time = time.time()

    if args.benchmark_type == "ort":
        # convert_to_onnx.py export
        logger.info(f"Loading model from {args.ort_model_path}")
        start_time = time.time()
        model = ort.InferenceSession(
            args.ort_model_path,
            sess_options,
            providers=[args.execution_provider],
        )
        end_time = time.time()

    logger.info(f"Loaded model in {end_time - start_time} s")

    return model


def time_fn(args, fn, inputs):
    warmup_inputs = inputs[0] if type(inputs) is tuple else inputs
    benchmark_inputs = inputs[1] if type(inputs) is tuple else inputs
    torch_device = torch.device(args.target_device)

    # Warm up
    warmup_range = (
        range(args.warmup_runs)
        if args.benchmark_type == "ort"
        else trange(args.warmup_runs, file=sys.stdout, desc="Warm up")
    )

    if args.verbose:
        outputs = fn(warmup_inputs)
        logger.info(outputs)

    for _ in warmup_range:
        fn(warmup_inputs)

    # Benchmark
    if args.device != "cpu":
        torch.cuda.synchronize(torch_device)
    start_time = time.time()

    bench_range = (
        range(args.num_runs)
        if args.benchmark_type == "ort"
        else trange(args.num_runs, file=sys.stdout, desc="Benchmark")
    )
    for _ in bench_range:
        fn(benchmark_inputs)

    if args.device != "cpu":
        torch.cuda.synchronize(torch_device)
    end_time = time.time()

    # Newline print after trange in order to print metrics on new lines without progress bar on same line
    if args.benchmark_type != "ort":
        logger.info("")

    batch_size = 1
    latency = (end_time - start_time) / args.num_runs
    throughput = batch_size / latency

    logger.info(f"Latency: {latency} s")
    logger.info(f"Throughput: {throughput} qps")
    return


def profile_fn(args, fn, inputs, inputs_type):
    # Filename prefix format:
    # "<benchmark-type>-<precision>-<device>_<inference-step>_<inputs-type>_<current-time>"
    prefix = f"{args.benchmark_type.lower()}-{args.precision}-{args.device}_{fn.__name__.replace('_', '-')}_{inputs_type}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}"
    filename = None

    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile"}:
        # Profile PyTorch kernels
        with profile(  # noqa: SIM117
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True
        ) as prof:
            with record_function("model_inference"):
                fn(inputs)
        prof_data = prof.key_averages(group_by_stack_n=5).table(sort_by=args.pt_filter_by, row_limit=args.pt_num_rows)

        filename = os.path.join(args.log_folder, f"{prefix}.log")
        with open(filename, "w") as f:
            f.write(prof_data)

    else:
        # Profile ORT kernels
        fn(inputs)

        # Set new log name for ORT profile log generated
        filename = f"{prefix}.json"

    return filename


def measure_fn(args, fn, inputs):
    # Measure CPU usage
    pid = os.getpid()
    process = psutil.Process(pid)
    process.cpu_percent(interval=0.1)

    fn(inputs)
    logger.info(f"CPU usage: {process.cpu_percent(interval=None)}%")

    # Measure memory usage
    gc.collect()
    torch.cuda.empty_cache()
    measure_memory(is_gpu=(args.device != "cpu"), func=lambda: fn(inputs), monitor_type=args.monitor_type)

    # Flush output so memory usage is printed
    sys.stdout.flush()


def run_hf_inference(args, inputs, model):
    # Inference steps to measure
    def get_pred_ids(inputs):
        # Inference pass with predicted token ids generation
        predicted_ids = model.generate(**inputs)
        return predicted_ids

    def gen_and_dec(inputs):
        # Inference pass with generation and decoding
        predicted_ids = get_pred_ids(inputs)
        transcription = []
        for _ in range(args.num_return_sequences):
            transcription.append(args.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
        return predicted_ids, transcription

    # Examples of other inference steps that can be measured:
    # To use, uncomment the function and assign it to `generate_fn`

    # def get_logits(inputs):
    #     # Inference pass without decoding
    #     outputs = model(**inputs)
    #     return outputs

    generate_fn = gen_and_dec

    if args.benchmark_type == "hf-pt-compile":
        # Run forward pass once with each set of inputs to process through Dynamo
        generate_fn(inputs)

    if args.profile:
        new_logname = profile_fn(args, generate_fn, inputs, "gen-and-dec")
        if args.benchmark_type == "hf-ort":
            # Rename log files per model component and turn profiling off to stop appending to log
            new_prefix = new_logname[: -len(".json")]

            old_logname = model.encoder.session.end_profiling()
            new_logname = new_prefix + "-encoder.json"
            if os.path.isfile(old_logname):
                logger.warning(f"Renaming {old_logname} to {new_logname}")
                os.rename(old_logname, os.path.join(args.log_folder, new_logname))

            old_logname = model.decoder.session.end_profiling()
            new_logname = new_prefix + "-decoder.json"
            if os.path.isfile(old_logname):
                logger.warning(f"Renaming {old_logname} to {new_logname}")
                os.rename(old_logname, os.path.join(args.log_folder, new_logname))

            old_logname = model.decoder_with_past.session.end_profiling()
            new_logname = new_prefix + "-decoder-with-past.json"
            if os.path.isfile(old_logname):
                logger.warning(f"Renaming {old_logname} to {new_logname}")
                os.rename(old_logname, os.path.join(args.log_folder, new_logname))

        return

    # PyTorch evaluations
    logger.info("\nEvaluating PyTorch...")
    time_fn(args, generate_fn, inputs)
    predicted_ids, transcription = generate_fn(inputs)
    logger.info(f"Generated token length: {len(predicted_ids[0])} tokens")
    logger.info(f"Transcription: {transcription[0]}")
    measure_fn(args, generate_fn, inputs)


def run_ort_inference(args, inputs, model):
    def prepare_ort_inputs(inputs, warmup=False):
        # Check that all model inputs will be provided
        model_inputs = {model_input.name for model_input in model.get_inputs()}
        user_inputs = set(inputs.keys())
        missing_inputs = model_inputs - user_inputs
        if len(missing_inputs):
            logger.error(f"The following model inputs are missing: {missing_inputs}")
            raise Exception("There are missing inputs to the model. Please add them and try again.")

        if warmup and args.tune:
            inputs["min_length"] = inputs["max_length"]

        # Remove unnecessary inputs from model inputs
        unnecessary_inputs = user_inputs - model_inputs
        if len(unnecessary_inputs):
            for unnecessary_input in unnecessary_inputs:
                logger.info(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
                del inputs[unnecessary_input]

        # Add IO bindings for non-CPU execution providers
        if args.device != "cpu":
            io_binding = model.io_binding()
            for k, v in inputs.items():
                io_binding.bind_cpu_input(k, v)
            for output in model.get_outputs():
                io_binding.bind_output(output.name, device_type=args.device, device_id=args.device_id)
            return io_binding

        return inputs

    def with_io_binding(io_binding):
        # Inference pass with IO binding
        model.run_with_iobinding(io_binding)
        return io_binding

    def without_io_binding(inputs):
        # Inference pass without IO binding
        outputs = model.run(None, inputs)
        return outputs

    def handle_output(output):
        if args.eos_token_id in output:
            first_end = np.where(output == args.eos_token_id)[0][0]
            return output[: first_end + 1]

        return output

    generate_fn = with_io_binding if args.device != "cpu" else without_io_binding
    ort_inputs = prepare_ort_inputs(inputs)

    if args.profile:
        new_logname = profile_fn(args, generate_fn, ort_inputs, "e2e")

        # Turn profiling off to stop appending to log file
        old_logname = model.end_profiling()
        logger.warning(f"Renaming {old_logname} to {new_logname}")
        os.rename(old_logname, os.path.join(args.log_folder, new_logname))

        return

    # ORT evaluation
    logger.info("\nEvaluating ONNX Runtime...")
    ort_evaluate_inputs = ort_inputs
    if args.tune:
        ort_warmup_inputs = prepare_ort_inputs(inputs, warmup=True)
        ort_evaluate_inputs = (ort_warmup_inputs, ort_inputs)

    time_fn(args, generate_fn, ort_evaluate_inputs)
    ort_outputs = generate_fn(ort_inputs)
    if args.device != "cpu":
        ort_outputs = ort_outputs.copy_outputs_to_cpu()
    ort_outputs = ort_outputs[0]

    if args.has_audio_stream:
        # ONNX E2E model from Olive produces transcribed output
        logger.info(f"Transcription: {ort_outputs[0][0]}")
    else:
        # convert_to_onnx model produces generated ids
        actual_output = handle_output(ort_outputs[0][0])
        logger.info(f"Generated token length: {len(actual_output)} tokens")
        transcription = args.processor.batch_decode(ort_outputs[0], skip_special_tokens=True)[0]
        # print to stdout as the output for comparison
        print(f"{transcription}")

    measure_fn(args, generate_fn, ort_inputs)


def run_inference(args, inputs, model):
    if args.benchmark_type in {"hf-pt-eager", "hf-pt-compile", "hf-ort"}:
        run_hf_inference(args, inputs, model)
    elif args.benchmark_type == "ort":
        run_ort_inference(args, inputs, model)
    else:
        raise Exception(f"Cannot recognize {args.benchmark_type}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bt",
        "--benchmark-type",
        type=str,
        required=True,
        choices=["hf-pt-eager", "hf-pt-compile", "hf-ort", "ort"],
    )

    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face name of model (e.g. 'openai/whisper-large-v2')",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        required=True,
        default="fp32",
        choices=["int8", "fp16", "fp32"],
        help="Precision for model. For ONNX models, the model's precision should be set before running this script.",
    )

    parser.add_argument(
        "--hf-pt-model-path",
        type=str,
        default="",
        help="Path to directory containing all PyTorch files (e.g. tokenizer, PyTorch model)",
    )
    parser.add_argument(
        "--hf-ort-dir-path",
        type=str,
        default="",
        help="Path to directory containing all ONNX files (e.g. tokenizer, encoder, decoder, decoder_with_past)",
    )
    parser.add_argument(
        "--ort-model-path",
        type=str,
        default="",
        help="Path to ONNX model",
    )

    # Args for running and evaluating the model
    parser.add_argument("-a", "--audio-path", type=str, required=True, help="Path to audio file for E2E evaluation")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "rocm"],
    )
    parser.add_argument("-id", "--device-id", type=int, default=0)
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)

    # Optional args:
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Sampling rate for audio (in Hz)")

    # Args for decoding logic
    # Required args:
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)

    # Optional args for E2E solution:
    parser.add_argument(
        "--decoder-input-ids",
        type=str,
        default="[]",
        help="The forced decoder ids for generation. Format is [start token, timestamp token, language token, task token]. Default is [start token]. See `decoder_input_ids` in https://github.com/microsoft/Olive/tree/main/examples/whisper for details.",
    )
    parser.add_argument(
        "--logits-processor",
        type=int,
        default=1,
        help="Whether to use timestamps logits processor or not (0 for false, 1 for true).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature value for generation.",
    )

    # Args for accessing detailed info
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--pt-filter-by", type=str, default="self_cpu_time_total", help="What to filter PyTorch profiler by"
    )
    parser.add_argument("--pt-num-rows", type=int, default=1000, help="Number of rows for PyTorch profiler to display")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--log-folder", type=str, default=os.path.join("."), help="Folder to cache log files")
    parser.add_argument(
        "--tune",
        default=False,
        action="store_true",
        help="Only used by ROCm EP, enable TunableOp tuning to select fastest kernel",
    )

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.monitor_type = args.device
    # Set runtime properties
    if "ort" in args.benchmark_type:
        args.execution_provider = f"{args.device.upper()}ExecutionProvider"
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})
        elif args.execution_provider == "ROCMExecutionProvider":
            args.execution_provider = (
                args.execution_provider,
                {
                    "device_id": args.device_id,
                    "tunable_op_enable": 1,
                    "tunable_op_tuning_enable": 1 if args.tune else 0,
                },
            )
            args.device = "cuda"

    # Check that model paths have been specified for any benchmarking with ORT
    if args.benchmark_type == "hf-ort":
        assert args.hf_ort_dir_path, "Please specify a path to `--hf-ort-dir-path`"
    if args.benchmark_type == "ort":
        assert args.ort_model_path, "Please specify a path to `--ort-model-path`"

    # Convert decoder_input_ids string to list of ids
    # (e.g. "[1, 50257]" for Hugging Face or "[50257]" for ORT)
    args.decoder_input_ids = ast.literal_eval(args.decoder_input_ids)

    return args


def main():
    args = parse_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True

    config = WhisperConfig.from_pretrained(args.model_name)
    processor = WhisperProcessor.from_pretrained(args.model_name)
    target_device = f"cuda:{args.device_id}" if args.device != "cpu" else args.device
    use_fp16 = args.precision == "fp16"

    setattr(args, "processor", processor)  # noqa: B010
    setattr(args, "target_device", target_device)  # noqa: B010
    setattr(args, "use_fp16", use_fp16)  # noqa: B010
    setattr(args, "has_audio_stream", False)  # noqa: B010
    setattr(args, "eos_token_id", config.eos_token_id)  # noqa: B010

    logger.info(f"Forced decoder prompt ids: {args.decoder_input_ids}")

    # Measure cost to transcribe audio
    model = get_model(args)
    if args.benchmark_type == "ort":
        # Check for optional inputs that could have been added during export
        ort_model_inputs = {model_input.name for model_input in model.get_inputs()}
        args.has_audio_stream = "audio_stream" in ort_model_inputs
        setattr(args, "has_decoder_input_ids", "decoder_input_ids" in ort_model_inputs)  # noqa: B010
        setattr(args, "has_logits_processor", "logits_processor" in ort_model_inputs)  # noqa: B010
        setattr(args, "has_temperature", "temperature" in ort_model_inputs)  # noqa: B010

        if args.decoder_input_ids == []:
            args.decoder_input_ids = [config.decoder_start_token_id]

    inputs = get_inputs(args)
    run_inference(args, inputs, model)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\matrix_distributions.py ===
from math import prod

from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
                            MatrixSymbol, MatrixBase, Transpose, MatrixSet,
                            matrix2numpy)
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
                            _symbol_converter, MatrixDomain, Distribution)
from sympy.external import import_module


################################################################################
#------------------------Matrix Probability Space------------------------------#
################################################################################
class MatrixPSpace(PSpace):
    """
    Represents probability space for
    Matrix Distributions.
    """
    def __new__(cls, sym, distribution, dim_n, dim_m):
        sym = _symbol_converter(sym)
        dim_n, dim_m = _sympify(dim_n), _sympify(dim_m)
        if not (dim_n.is_integer and dim_m.is_integer):
            raise ValueError("Dimensions should be integers")
        return Basic.__new__(cls, sym, distribution, dim_n, dim_m)

    distribution = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])

    @property
    def domain(self):
        return MatrixDomain(self.symbol, self.distribution.set)

    @property
    def value(self):
        return RandomMatrixSymbol(self.symbol, self.args[2], self.args[3], self)

    @property
    def values(self):
        return {self.value}

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 1 or (not isinstance(expr, RandomMatrixSymbol)):
            raise NotImplementedError("Currently, no algorithm has been "
                    "implemented to handle general expressions containing "
                    "multiple matrix distributions.")
        return self.distribution.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomMatrixSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}


def rv(symbol, cls, args):
    args = list(map(sympify, args))
    dist = cls(*args)
    dist.check(*args)
    dim = dist.dimension
    pspace = MatrixPSpace(symbol, dist, dim[0], dim[1])
    return pspace.value


class SampleMatrixScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        from scipy import stats as scipy_stats
        import numpy
        scipy_rv_map = {
            'WishartDistribution': lambda dist, size, rand_state: scipy_stats.wishart.rvs(
                df=int(dist.n), scale=matrix2numpy(dist.scale_matrix, float), size=size),
            'MatrixNormalDistribution': lambda dist, size, rand_state: scipy_stats.matrix_normal.rvs(
                mean=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float), size=size, random_state=rand_state)
        }

        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = scipy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        samp = scipy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixNumpy:
    """Returns the sample from numpy of the given distribution"""

    ### TODO: Add tests after adding matrix distributions in numpy_rv_map
    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        numpy_rv_map = {
        }

        sample_shape = {
        }

        dist_list = numpy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        samp = numpy_rv_map[dist.__class__.__name__](dist, prod(size), rand_state)
        return samp.reshape(size + sample_shape[dist.__class__.__name__](dist))


class SampleMatrixPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {
            'MatrixNormalDistribution': lambda dist: pymc.MatrixNormal('X',
                mu=matrix2numpy(dist.location_matrix, float),
                rowcov=matrix2numpy(dist.scale_matrix_1, float),
                colcov=matrix2numpy(dist.scale_matrix_2, float),
                shape=dist.location_matrix.shape),
            'WishartDistribution': lambda dist: pymc.WishartBartlett('X',
                nu=int(dist.n), S=matrix2numpy(dist.scale_matrix, float))
        }

        sample_shape = {
            'WishartDistribution': lambda dist: dist.scale_matrix.shape,
            'MatrixNormalDistribution' : lambda dist: dist.location_matrix.shape
        }

        dist_list = pymc_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None
        import logging
        logging.getLogger("pymc").setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samps = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)['X']
        return samps.reshape(size + sample_shape[dist.__class__.__name__](dist))

_get_sample_class_matrixrv = {
    'scipy': SampleMatrixScipy,
    'pymc3': SampleMatrixPymc,
    'pymc': SampleMatrixPymc,
    'numpy': SampleMatrixNumpy
}

################################################################################
#-------------------------Matrix Distribution----------------------------------#
################################################################################

class MatrixDistribution(Distribution, NamedArgsMixin):
    """
    Abstract class for Matrix Distribution.
    """
    def __new__(cls, *args):
        args = [ImmutableMatrix(arg) if isinstance(arg, list)
                else _sympify(arg) for arg in args]
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    def __call__(self, expr):
        if isinstance(expr, list):
            expr = ImmutableMatrix(expr)
        return self.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """

        libraries = ['scipy', 'numpy', 'pymc3', 'pymc']
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        samps = _get_sample_class_matrixrv[library](self, size, seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

################################################################################
#------------------------Matrix Distribution Types-----------------------------#
################################################################################

#-------------------------------------------------------------------------------
# Matrix Gamma distribution ----------------------------------------------------

class MatrixGammaDistribution(MatrixDistribution):

    _argnames = ('alpha', 'beta', 'scale_matrix')

    @staticmethod
    def check(alpha, beta, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        _value_check(beta.is_positive, "Scale parameter should be positive.")

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        alpha, beta, scale_matrix = self.alpha, self.beta, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        sigma_inv_x = - Inverse(scale_matrix)*x / beta
        term1 = exp(Trace(sigma_inv_x))/((beta**(p*alpha)) * multigamma(alpha, p))
        term2 = (Determinant(scale_matrix))**(-alpha)
        term3 = (Determinant(x))**(alpha - S(p + 1)/2)
        return term1 * term2 * term3

def MatrixGamma(symbol, alpha, beta, scale_matrix):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    alpha: Positive Real number
        Shape Parameter
    beta: Positive Real number
        Scale Parameter
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, MatrixGamma
    >>> from sympy import MatrixSymbol, symbols
    >>> a, b = symbols('a b', positive=True)
    >>> M = MatrixGamma('M', a, b, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(M)(X).doit()
    exp(Trace(Matrix([
    [-2/3,  1/3],
    [ 1/3, -2/3]])*X)/b)*Determinant(X)**(a - 3/2)/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))
    >>> density(M)([[1, 0], [0, 1]]).doit()
    exp(-4/(3*b))/(3**a*sqrt(pi)*b**(2*a)*gamma(a)*gamma(a - 1/2))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_gamma_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, MatrixGammaDistribution, (alpha, beta, scale_matrix))

#-------------------------------------------------------------------------------
# Wishart Distribution ---------------------------------------------------------

class WishartDistribution(MatrixDistribution):

    _argnames = ('n', 'scale_matrix')

    @staticmethod
    def check(n, scale_matrix):
        if not isinstance(scale_matrix, MatrixSymbol):
            _value_check(scale_matrix.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix.is_square, "Should "
        "be square matrix")
        _value_check(n.is_positive, "Shape parameter should be positive.")

    @property
    def set(self):
        k = self.scale_matrix.shape[0]
        return MatrixSet(k, k, S.Reals)

    @property
    def dimension(self):
        return self.scale_matrix.shape

    def pdf(self, x):
        n, scale_matrix = self.n, self.scale_matrix
        p = scale_matrix.shape[0]
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        sigma_inv_x = - Inverse(scale_matrix)*x / S(2)
        term1 = exp(Trace(sigma_inv_x))/((2**(p*n/S(2))) * multigamma(n/S(2), p))
        term2 = (Determinant(scale_matrix))**(-n/S(2))
        term3 = (Determinant(x))**(S(n - p - 1)/2)
        return term1 * term2 * term3

def Wishart(symbol, n, scale_matrix):
    """
    Creates a random variable with Wishart Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    n: Positive Real number
        Represents degrees of freedom
    scale_matrix: Positive definite real square matrix
        Scale Matrix

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, Wishart
    >>> from sympy import MatrixSymbol, symbols
    >>> n = symbols('n', positive=True)
    >>> W = Wishart('W', n, [[2, 1], [1, 2]])
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(W)(X).doit()
    exp(Trace(Matrix([
    [-1/3,  1/6],
    [ 1/6, -1/3]])*X))*Determinant(X)**(n/2 - 3/2)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))
    >>> density(W)([[1, 0], [0, 1]]).doit()
    exp(-2/3)/(2**n*3**(n/2)*sqrt(pi)*gamma(n/2)*gamma(n/2 - 1/2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wishart_distribution

    """
    if isinstance(scale_matrix, list):
        scale_matrix = ImmutableMatrix(scale_matrix)
    return rv(symbol, WishartDistribution, (n, scale_matrix))

#-------------------------------------------------------------------------------
# Matrix Normal distribution ---------------------------------------------------

class MatrixNormalDistribution(MatrixDistribution):

    _argnames = ('location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite, "The shape "
                "matrix must be positive definite.")
        _value_check(scale_matrix_1.is_square, "Scale matrix 1 should be "
        "be square matrix")
        _value_check(scale_matrix_2.is_square, "Scale matrix 2 should be "
        "be square matrix")
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == n, "Scale matrix 1 should be"
        " of shape %s x %s"% (str(n), str(n)))
        _value_check(scale_matrix_2.shape[0] == p, "Scale matrix 2 should be"
        " of shape %s x %s"% (str(p), str(p)))

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        M, U, V = self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                    "or MatrixSymbol" % str(x))
        term1 = Inverse(V)*Transpose(x - M)*Inverse(U)*(x - M)
        num = exp(-Trace(term1)/S(2))
        den = (2*pi)**(S(n*p)/2) * Determinant(U)**(S(p)/2) * Determinant(V)**(S(n)/2)
        return num/den

def MatrixNormal(symbol, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Normal Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    location_matrix: Real ``n x p`` matrix
        Represents degrees of freedom
    scale_matrix_1: Positive definite matrix
        Scale Matrix of shape ``n x n``
    scale_matrix_2: Positive definite matrix
        Scale Matrix of shape ``p x p``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> from sympy.stats import density, MatrixNormal
    >>> M = MatrixNormal('M', [[1, 2]], [1], [[1, 0], [0, 1]])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X).doit()
    exp(-Trace((Matrix([
    [-1],
    [-2]]) + X.T)*(Matrix([[-1, -2]]) + X))/2)/(2*pi)
    >>> density(M)([[3, 4]]).doit()
    exp(-4)/(2*pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_normal_distribution

    """
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    args = (location_matrix, scale_matrix_1, scale_matrix_2)
    return rv(symbol, MatrixNormalDistribution, args)

#-------------------------------------------------------------------------------
# Matrix Student's T distribution ---------------------------------------------------

class MatrixStudentTDistribution(MatrixDistribution):

    _argnames = ('nu', 'location_matrix', 'scale_matrix_1', 'scale_matrix_2')

    @staticmethod
    def check(nu, location_matrix, scale_matrix_1, scale_matrix_2):
        if not isinstance(scale_matrix_1, MatrixSymbol):
            _value_check(scale_matrix_1.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        if not isinstance(scale_matrix_2, MatrixSymbol):
            _value_check(scale_matrix_2.is_positive_definite != False, "The shape "
                                                              "matrix must be positive definite.")
        _value_check(scale_matrix_1.is_square != False, "Scale matrix 1 should be "
                                               "be square matrix")
        _value_check(scale_matrix_2.is_square != False, "Scale matrix 2 should be "
                                               "be square matrix")
        n = location_matrix.shape[0]
        p = location_matrix.shape[1]
        _value_check(scale_matrix_1.shape[0] == p, "Scale matrix 1 should be"
                                                   " of shape %s x %s" % (str(p), str(p)))
        _value_check(scale_matrix_2.shape[0] == n, "Scale matrix 2 should be"
                                                   " of shape %s x %s" % (str(n), str(n)))
        _value_check(nu.is_positive != False, "Degrees of freedom must be positive")

    @property
    def set(self):
        n, p = self.location_matrix.shape
        return MatrixSet(n, p, S.Reals)

    @property
    def dimension(self):
        return self.location_matrix.shape

    def pdf(self, x):
        from sympy.matrices.dense import eye
        if isinstance(x, list):
            x = ImmutableMatrix(x)
        if not isinstance(x, (MatrixBase, MatrixSymbol)):
            raise ValueError("%s should be an isinstance of Matrix "
                             "or MatrixSymbol" % str(x))
        nu, M, Omega, Sigma = self.nu, self.location_matrix, self.scale_matrix_1, self.scale_matrix_2
        n, p = M.shape

        K = multigamma((nu + n + p - 1)/2, p) * Determinant(Omega)**(-n/2) * Determinant(Sigma)**(-p/2) \
            / ((pi)**(n*p/2) * multigamma((nu + p - 1)/2, p))
        return K * (Determinant(eye(n) + Inverse(Sigma)*(x - M)*Inverse(Omega)*Transpose(x - M))) \
               **(-(nu + n + p -1)/2)



def MatrixStudentT(symbol, nu, location_matrix, scale_matrix_1, scale_matrix_2):
    """
    Creates a random variable with Matrix Gamma Distribution.

    The density of the said distribution can be found at [1].

    Parameters
    ==========

    nu: Positive Real number
        degrees of freedom
    location_matrix: Positive definite real square matrix
        Location Matrix of shape ``n x p``
    scale_matrix_1: Positive definite real square matrix
        Scale Matrix of shape ``p x p``
    scale_matrix_2: Positive definite real square matrix
        Scale Matrix of shape ``n x n``

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy import MatrixSymbol,symbols
    >>> from sympy.stats import density, MatrixStudentT
    >>> v = symbols('v',positive=True)
    >>> M = MatrixStudentT('M', v, [[1, 2]], [[1, 0], [0, 1]], [1])
    >>> X = MatrixSymbol('X', 1, 2)
    >>> density(M)(X)
    gamma(v/2 + 1)*Determinant((Matrix([[-1, -2]]) + X)*(Matrix([
    [-1],
    [-2]]) + X.T) + Matrix([[1]]))**(-v/2 - 1)/(pi**1.0*gamma(v/2)*Determinant(Matrix([[1]]))**1.0*Determinant(Matrix([
    [1, 0],
    [0, 1]]))**0.5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Matrix_t-distribution

    """
    if isinstance(location_matrix, list):
        location_matrix = ImmutableMatrix(location_matrix)
    if isinstance(scale_matrix_1, list):
        scale_matrix_1 = ImmutableMatrix(scale_matrix_1)
    if isinstance(scale_matrix_2, list):
        scale_matrix_2 = ImmutableMatrix(scale_matrix_2)
    args = (nu, location_matrix, scale_matrix_1, scale_matrix_2)
    return rv(symbol, MatrixStudentTDistribution, args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\benchmark_e2e.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This is an end-to-end benchmarking script for the Hugging Face LLaMA-2 model.
#
# Prerequisites:
# 1) Install `huggingface-cli`:
#
# $ pip install huggingface_hub
#
# 2) Authenticate with Hugging Face's CLI:
#
# $ huggingface-cli login
#
# 3) Accept Meta's license in Hugging Face to access the models at https://huggingface.co/meta-llama/
#
# 4) Install the latest ONNX Runtime version
#
# $ pip install onnxruntime-gpu
#
# 5) Install flash attention v2
#
# $ pip install flash-attn --no-build-isolation
#
# 6) Install bitsandbytes
#
# $ pip install bitsandbytes

from __future__ import annotations

import argparse
import datetime
import gc
import itertools
import json
import logging
import os
import textwrap
import time

import numpy as np
import pandas as pd
import torch
from benchmark_helper import setup_logger
from llama_inputs import add_io_bindings_as_tensors, get_initial_inputs_and_outputs
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_model(args: argparse.Namespace):
    if args.benchmark_type in {"pt-eager", "pt-compile"}:
        model = None
        if args.onnx_precision == "int4" and args.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                args.hf_dir_path if args.hf_dir_path != "" else args.model_name,
                cache_dir=args.cache_dir,
                torch_dtype=args.torch_dtype,
                use_auth_token=args.auth,
                trust_remote_code=args.trust,
                use_cache=True,
                attn_implementation="flash_attention_2",
                quantization_config=bnb_config,
                max_memory={args.device_id: "80GB"},
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.hf_dir_path if args.hf_dir_path != "" else args.model_name,
                    cache_dir=args.cache_dir,
                    torch_dtype=args.torch_dtype,
                    use_auth_token=args.auth,
                    trust_remote_code=args.trust,
                    use_cache=True,
                    attn_implementation=("flash_attention_2" if args.device == "cuda" else "sdpa"),
                ).to(args.target_device)
            except Exception as e:
                # When flash_attention or sdpa doesn't support a model, it throws an exception.
                # Rather than stopping a process, run as eager mode.
                print("Try to load a model using eager mode: ", e)
                model = AutoModelForCausalLM.from_pretrained(
                    args.hf_dir_path if args.hf_dir_path != "" else args.model_name,
                    cache_dir=args.cache_dir,
                    torch_dtype=args.torch_dtype,
                    use_auth_token=args.auth,
                    trust_remote_code=args.trust,
                    use_cache=True,
                    attn_implementation="eager",
                ).to(args.target_device)

        model.eval()

        if args.benchmark_type == "pt-compile":
            model = torch.compile(model)

    else:
        sess_options = ort.SessionOptions()
        ep = (
            ("CUDAExecutionProvider", {"device_id": args.device_id})
            if args.device == "cuda"
            else "CPUExecutionProvider"
        )
        model = ort.InferenceSession(args.onnx_model_path, sess_options=sess_options, providers=[ep])

    return model


def run_inference(args, model, runs, inputs, outputs):
    if args.benchmark_type == "pt-compile":
        with torch.no_grad():
            outputs = model(**inputs)

    # Synchronize inputs
    io_binding = None
    if args.benchmark_type in {"pt-eager", "pt-compile"}:
        if args.device != "cpu":
            torch.cuda.synchronize(args.target_device)
    else:
        io_binding = add_io_bindings_as_tensors(model, inputs, outputs, args.use_fp16, args.use_buffer_share)
        io_binding.synchronize_inputs()

    # Run inference
    start = time.perf_counter()
    for _ in range(runs):
        if args.benchmark_type in {"pt-eager", "pt-compile"}:
            with torch.no_grad():
                outputs = model(**inputs)
                if args.device != "cpu":
                    torch.cuda.synchronize(args.target_device)
        else:
            model.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

    end = time.perf_counter()
    avg = (end - start) / runs
    return avg, outputs


def prepare_model_for_inference(args, model, config, tokenizer, prompt_length, prompt):
    clear_cache()
    inputs, outputs = get_initial_inputs_and_outputs(
        config, tokenizer, prompt_length, prompt, args.target_device, args.use_fp16, args.use_buffer_share, args.engine
    )
    _, outputs = run_inference(args, model, args.warmup_runs, inputs, outputs)
    return inputs, outputs


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def save_results(results, filename, gen_length):
    df = pd.DataFrame(
        results,
        columns=[
            "Batch Size",
            "Prompt Length",
            "Prompt Processing Latency (ms)",
            "Prompt Processing Throughput (tps)",
            "Sampling Latency (ms)",
            "Sampling Throughput (tps)",
            "First Token Generated Latency (ms)",
            "First Token Generated Throughput (tps)",
            f"Average Latency of First {gen_length // 2} Tokens Generated (ms)",
            f"Average Throughput of First {gen_length // 2} Tokens Generated (tps)",
            f"Average Latency of First {gen_length} Tokens Generated (ms)",
            f"Average Throughput of First {gen_length} Tokens Generated (tps)",
            "Wall-Clock Latency (s)",
            "Wall-Clock Throughput (tps)",
        ],
    )

    df.to_csv(filename, index=False)
    logger.info(f"Results saved in {filename}!")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bt",
        "--benchmark-type",
        type=str,
        required=True,
        choices=["pt-eager", "pt-compile", "ort"],
    )

    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=False,
        help="Hugging Face name of model (e.g. 'meta-llama/Llama-2-7b-hf')",
    )

    parser.add_argument(
        "-a",
        "--auth",
        default=False,
        action="store_true",
        help="Use Hugging Face authentication token to access model",
    )

    parser.add_argument(
        "-t",
        "--trust",
        default=False,
        action="store_true",
        help="Whether or not to allow for custom models defined on the Hugging Face Hub in their own modeling files",
    )

    parser.add_argument(
        "-c",
        "--cache-dir",
        type=str,
        default=os.path.join(".", "model_cache"),
        help="Path to directory containing all Hugging Face files (e.g. config, tokenizer, PyTorch model). Use when loading model as `AutoModel.from_pretrained(model_name, cache_dir=cache_dir)`.",
    )

    parser.add_argument(
        "--hf-dir-path",
        type=str,
        default="",
        help="Path to directory containing all Hugging Face files (e.g. config, tokenizer, PyTorch model). Use when loading model as `AutoModel.from_pretrained(folder_path)`.",
    )

    parser.add_argument(
        "-o",
        "--onnx-model-path",
        required=False,
        help="Path to ONNX model",
    )

    parser.add_argument(
        "-f",
        "--prompts-file",
        required=True,
        default=os.path.join(".", "models", "llama", "prompts.json"),
        help="JSON file containing entries in the format 'prompt length: prompt' where prompt length = tokenized length of prompt",
    )

    parser.add_argument(
        "--use_buffer_share",
        default=False,
        action="store_true",
        help="Use when GroupQueryAttention (GQA) is in ONNX model",
    )

    (
        parser.add_argument(
            "--anomaly-filtering",
            default=False,
            action="store_true",
            help="Use this flag to filter anomaly accelerator times for tokens generated. \
              This may give more accurate latency and throughput metrics for tokens generated. \
              Wall-clock metrics are still reported with anomaly times though.",
        ),
    )

    parser.add_argument(
        "-b",
        "--batch-sizes",
        default="1 2",
    )

    parser.add_argument(
        "-s",
        "--prompt-lengths",
        default="16 64 256 1024",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        type=str,
        default="fp32",
        choices=["int4", "int8", "fp16", "fp32"],
        help="Precision for model. For ONNX models, the model's precision should be set before running this script.",
    )

    parser.add_argument(
        "-g",
        "--generation-length",
        type=int,
        default=256,
        help="Number of new tokens to generate",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )

    parser.add_argument("-id", "--device-id", type=int, default=0)
    parser.add_argument("-w", "--warmup-runs", type=int, default=5)
    parser.add_argument("-n", "--num-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)

    args = parser.parse_args()

    # Set seed properties
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set runtime properties
    if "ort" in args.benchmark_type:
        setattr(args, "execution_provider", f"{args.device.upper()}ExecutionProvider")  # noqa: B010
        if args.execution_provider == "CUDAExecutionProvider":
            args.execution_provider = (args.execution_provider, {"device_id": args.device_id})

    # Check that paths have been specified for any benchmarking with ORT
    if args.benchmark_type == "ort":
        assert args.onnx_model_path, "Please specify a path to `--onnx-model-path`"

    args.batch_sizes = args.batch_sizes.split(" ")
    args.prompt_lengths = args.prompt_lengths.split(" ")

    # Use FP32 precision for FP32, INT8, INT4 CPU models, use FP16 precision for FP16 and INT4 GPU models
    setattr(args, "onnx_precision", args.precision)  # noqa: B010
    args.precision = (
        "fp32" if args.precision in {"int8", "fp32"} or (args.precision == "int4" and args.device == "cpu") else "fp16"
    )

    target_device = f"cuda:{args.device_id}" if args.device != "cpu" else args.device
    torch_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    engine = "ort" if args.benchmark_type == "ort" else "pt"
    setattr(args, "target_device", target_device)  # noqa: B010
    setattr(args, "torch_dtype", torch_dtype)  # noqa: B010
    setattr(args, "engine", engine)  # noqa: B010
    setattr(args, "use_fp16", args.precision == "fp16")  # noqa: B010

    args.use_buffer_share = args.use_buffer_share and engine == "ort"

    return args


def main():
    args = get_args()
    setup_logger(False)
    logger.info(args.__dict__)

    # Get prompts and prompt sizes
    size_to_prompt = None
    with open(args.prompts_file) as f:
        size_to_prompt = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})

    # Get config, tokenizer, and model
    config = AutoConfig.from_pretrained(
        args.hf_dir_path if args.hf_dir_path != "" else args.model_name,
        cache_dir=args.cache_dir,
        use_auth_token=args.auth,
        trust_remote_code=args.trust,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_dir_path if args.hf_dir_path != "" else args.model_name,
        cache_dir=args.cache_dir,
        use_auth_token=args.auth,
        trust_remote_code=args.trust,
    )
    model = get_model(args)

    all_csv_metrics = []
    for batch_size, prompt_length in itertools.product(args.batch_sizes, args.prompt_lengths):
        batch_size, prompt_length = int(batch_size), int(prompt_length)  # noqa: PLW2901
        logger.info(f"Running batch size = {batch_size}, prompt length = {prompt_length}")
        clear_cache()
        max_length = prompt_length + args.generation_length

        if prompt_length not in size_to_prompt:
            raise NotImplementedError(
                textwrap.dedent(
                    f"""
                                A prompt of size {prompt_length} was not found in '{args.prompts_file}'. There are a couple of solutions to fix this.
                                1) You can change one of the keys in '{args.prompts_file}' to be {prompt_length}.
                                    If {prompt_length} < actual prompt's length, the benchmark E2E tool will repeat the first word in the prompt until {prompt_length} = actual prompt's length.
                                    If {prompt_length} > actual prompt's length, the benchmark E2E tool will automatically trim the actual prompt's length so that {prompt_length} = actual prompt's length.
                                2) You can add a new key-value entry in '{args.prompts_file}' of the form '{prompt_length}': 'your prompt goes here'.
                """
                )
            )
        prompt = [size_to_prompt[prompt_length]] * batch_size
        csv_metrics = [batch_size, prompt_length]

        try:
            # Measure prompt processing
            logger.info("Measuring prompt processing...")
            inputs, outputs = prepare_model_for_inference(args, model, config, tokenizer, prompt_length, prompt)
            accelerator_prompt_latency_s, outputs = run_inference(args, model, args.num_runs, inputs, outputs)

            # Calculate prompt metrics
            accelerator_prompt_latency_ms = accelerator_prompt_latency_s * 1000
            accelerator_prompt_thrpt = batch_size * (prompt_length / accelerator_prompt_latency_s)
            logger.info(f"Average Latency of Prompt Processing: {accelerator_prompt_latency_ms} ms")
            logger.info(
                f"Average Throughput of Prompt Processing: {batch_size * (prompt_length / accelerator_prompt_latency_s)} tps"
            )
            csv_metrics.extend([accelerator_prompt_latency_ms, accelerator_prompt_thrpt])

            # Measure token generation
            logger.info("Measuring token generation...")
            clear_cache()
            inputs, outputs = prepare_model_for_inference(args, model, config, tokenizer, prompt_length, prompt)

            all_token_ids = inputs["input_ids"].clone()
            current_length = all_token_ids.shape[-1]
            num_heads = config.num_key_value_heads
            head_size = (
                config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
            )

            has_eos = torch.zeros(batch_size, device=args.target_device, dtype=torch.bool)

            # 0th entry will have prompt accelerator time, 1st entry onwards will have token generation accelerator time
            accelerator_times = []
            sampling_times = []  # cost to sample after each model run

            wall_clock_start_time = time.perf_counter()
            while current_length <= max_length:
                # Run inference
                accelerator_time_latency_s, outputs = run_inference(args, model, 1, inputs, outputs)
                accelerator_times.append(accelerator_time_latency_s)

                # Sample with argmax (greedy search)
                sampling_start_time = time.perf_counter()
                if outputs["logits"].shape[1] > 1:
                    prompt_end_indices = inputs["attention_mask"].sum(1) - 1
                    idxs = (
                        prompt_end_indices.unsqueeze(dim=1)
                        .repeat(1, config.vocab_size)
                        .view(batch_size, 1, config.vocab_size)
                    )
                    next_token_logits = torch.gather(outputs["logits"], 1, idxs).squeeze()
                else:
                    next_token_logits = outputs["logits"][:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Check if we previously reached EOS token id or if generated token id is EOS token id
                has_eos = has_eos | next_tokens == tokenizer.eos_token_id

                # Determine which new tokens to add to list of all token ids
                # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
                tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape([batch_size, 1])
                sampling_end_time = time.perf_counter()
                sampling_times.append(sampling_end_time - sampling_start_time)

                all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)
                current_length += 1

                # Update inputs for next inference run
                inputs["input_ids"] = tokens_to_add
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"], (~has_eos).to(torch.int64).reshape(batch_size, 1)], 1
                )
                if "position_ids" in inputs:
                    inputs["position_ids"] = torch.max(inputs["position_ids"], dim=1)[0].reshape(batch_size, 1) + 1

                # Set logits to zeros for next inference run and re-use memory buffer
                if outputs["logits"].shape[1] != 1:
                    outputs["logits"] = outputs["logits"][:, :1, :].contiguous()
                outputs["logits"].zero_()

                # Update KV caches for next inference run
                if args.engine == "pt":
                    # Update KV caches for PyTorch
                    inputs["past_key_values"] = outputs["past_key_values"]
                elif not args.use_buffer_share:
                    # Update KV caches for ONNX Runtime if buffer sharing is not used
                    for i in range(config.num_hidden_layers):
                        inputs[f"past_key_values.{i}.key"] = outputs[f"present.{i}.key"]
                        inputs[f"past_key_values.{i}.value"] = outputs[f"present.{i}.value"]

                    new_sequence_length = inputs["attention_mask"].shape[1]
                    for i in range(config.num_hidden_layers):
                        present_key = torch.zeros(
                            batch_size,
                            num_heads,
                            new_sequence_length,
                            head_size,
                            device=args.target_device,
                            dtype=args.torch_dtype,
                        )
                        present_value = torch.zeros(
                            batch_size,
                            num_heads,
                            new_sequence_length,
                            head_size,
                            device=args.target_device,
                            dtype=args.torch_dtype,
                        )
                        outputs.update(
                            {
                                f"present.{i}.key": present_key.contiguous(),
                                f"present.{i}.value": present_value.contiguous(),
                            }
                        )

            wall_clock_end_time = time.perf_counter()

            # Filter out any anomaly accelerator times (e.g. for `torch.compile`)
            accelerator_times.pop(0)  # Remove prompt processing time
            if args.anomaly_filtering:
                anomaly_threshold_factor = 10
                min_time_s = min(accelerator_times)
                orig_size = len(accelerator_times)
                accelerator_times = list(
                    filter(lambda acc_time: acc_time < anomaly_threshold_factor * min_time_s, accelerator_times)
                )
                new_size = len(accelerator_times)
                logger.info(
                    f"Filtered out {orig_size - new_size} anomaly accelerator times that are {anomaly_threshold_factor}x greater than {min_time_s * 1000} ms..."
                )

            #######################################################
            # Calculate sampling and first token generated metrics
            #######################################################

            # Calculate sampling metrics
            avg_sampling_latency_s = sum(sampling_times) / len(sampling_times)
            avg_sampling_latency_ms = avg_sampling_latency_s * 1000
            avg_sampling_thrpt = batch_size * (1 / avg_sampling_latency_s)
            logger.info(f"Average Latency of Sampling: {avg_sampling_latency_ms} ms")
            logger.info(f"Average Throughput of Sampling: {avg_sampling_thrpt} tps")

            # Calculate first token generated metrics
            first_token_latency_s = accelerator_times[0]
            first_token_latency_ms = first_token_latency_s * 1000
            first_token_thrpt = batch_size * (1 / first_token_latency_s)
            logger.info(f"Latency of First Token Generated: {first_token_latency_ms} ms")
            logger.info(f"Throughput of First Token Generated: {first_token_thrpt} tps")

            ####################################################
            # Calculate first `halfway` token generated metrics
            ####################################################

            halfway = args.generation_length // 2
            halfway_token_latency_s = sum(accelerator_times[:halfway]) / len(accelerator_times[:halfway])
            halfway_token_latency_ms = halfway_token_latency_s * 1000
            halfway_token_thrpt = batch_size * (1 / halfway_token_latency_s)
            logger.info(f"Average Latency of First {halfway} Tokens Generated: {halfway_token_latency_ms} ms")
            logger.info(f"Average Throughput of First {halfway} Tokens Generated: {halfway_token_thrpt} tps")

            #########################################
            # Calculate all tokens generated metrics
            #########################################

            all_token_latency_s = sum(accelerator_times) / len(accelerator_times)
            all_token_latency_ms = all_token_latency_s * 1000
            all_token_thrpt = batch_size * (1 / all_token_latency_s)
            logger.info(
                f"Average Latency of First {args.generation_length} Tokens Generated: {all_token_latency_ms} ms"
            )
            logger.info(f"Average Throughput of First {args.generation_length} Tokens Generated: {all_token_thrpt} tps")

            ###############################
            # Calculate wall clock metrics
            ###############################

            wall_clock_latency_s = wall_clock_end_time - wall_clock_start_time
            wall_clock_thrpt = batch_size * ((prompt_length + args.generation_length) / wall_clock_latency_s)
            logger.info(f"Wall-Clock Latency: {wall_clock_latency_s} s")
            logger.info(
                f"Wall-Clock Throughput: {batch_size * ((prompt_length + args.generation_length) / wall_clock_latency_s)} tps"
            )

            # Add metrics to CSV
            logger.info("Adding results to CSV")
            csv_metrics.extend(
                [
                    avg_sampling_latency_ms,
                    avg_sampling_thrpt,
                    first_token_latency_ms,
                    first_token_thrpt,
                    halfway_token_latency_ms,
                    halfway_token_thrpt,
                    all_token_latency_ms,
                    all_token_thrpt,
                    wall_clock_latency_s,
                    wall_clock_thrpt,
                ]
            )
            all_csv_metrics.append(csv_metrics)

        except Exception as e:
            logger.info(f"Could not benchmark at batch size = {batch_size}, prompt length = {prompt_length} - {e}")

    filename = f"benchmark_{args.engine}_e2e_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.csv"
    save_results(all_csv_metrics, filename, args.generation_length)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\models\link.py ===
import functools
import itertools
import logging
import os
import posixpath
import re
import urllib.parse
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
    pairwise,
    redact_auth_from_url,
    split_auth_from_netloc,
    splitext,
)
from pip._internal.utils.urls import path_to_url, url_to_path

if TYPE_CHECKING:
    from pip._internal.index.collector import IndexContent

logger = logging.getLogger(__name__)


# Order matters, earlier hashes have a precedence over later hashes for what
# we will pick to use.
_SUPPORTED_HASHES = ("sha512", "sha384", "sha256", "sha224", "sha1", "md5")


@dataclass(frozen=True)
class LinkHash:
    """Links to content may have embedded hash values. This class parses those.

    `name` must be any member of `_SUPPORTED_HASHES`.

    This class can be converted to and from `ArchiveInfo`. While ArchiveInfo intends to
    be JSON-serializable to conform to PEP 610, this class contains the logic for
    parsing a hash name and value for correctness, and then checking whether that hash
    conforms to a schema with `.is_hash_allowed()`."""

    name: str
    value: str

    _hash_url_fragment_re = re.compile(
        # NB: we do not validate that the second group (.*) is a valid hex
        # digest. Instead, we simply keep that string in this class, and then check it
        # against Hashes when hash-checking is needed. This is easier to debug than
        # proactively discarding an invalid hex digest, as we handle incorrect hashes
        # and malformed hashes in the same place.
        r"[#&]({choices})=([^&]*)".format(
            choices="|".join(re.escape(hash_name) for hash_name in _SUPPORTED_HASHES)
        ),
    )

    def __post_init__(self) -> None:
        assert self.name in _SUPPORTED_HASHES

    @classmethod
    @functools.lru_cache(maxsize=None)
    def find_hash_url_fragment(cls, url: str) -> Optional["LinkHash"]:
        """Search a string for a checksum algorithm name and encoded output value."""
        match = cls._hash_url_fragment_re.search(url)
        if match is None:
            return None
        name, value = match.groups()
        return cls(name=name, value=value)

    def as_dict(self) -> Dict[str, str]:
        return {self.name: self.value}

    def as_hashes(self) -> Hashes:
        """Return a Hashes instance which checks only for the current hash."""
        return Hashes({self.name: [self.value]})

    def is_hash_allowed(self, hashes: Optional[Hashes]) -> bool:
        """
        Return True if the current hash is allowed by `hashes`.
        """
        if hashes is None:
            return False
        return hashes.is_hash_allowed(self.name, hex_digest=self.value)


@dataclass(frozen=True)
class MetadataFile:
    """Information about a core metadata file associated with a distribution."""

    hashes: Optional[Dict[str, str]]

    def __post_init__(self) -> None:
        if self.hashes is not None:
            assert all(name in _SUPPORTED_HASHES for name in self.hashes)


def supported_hashes(hashes: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    # Remove any unsupported hash types from the mapping. If this leaves no
    # supported hashes, return None
    if hashes is None:
        return None
    hashes = {n: v for n, v in hashes.items() if n in _SUPPORTED_HASHES}
    if not hashes:
        return None
    return hashes


def _clean_url_path_part(part: str) -> str:
    """
    Clean a "part" of a URL path (i.e. after splitting on "@" characters).
    """
    # We unquote prior to quoting to make sure nothing is double quoted.
    return urllib.parse.quote(urllib.parse.unquote(part))


def _clean_file_url_path(part: str) -> str:
    """
    Clean the first part of a URL path that corresponds to a local
    filesystem path (i.e. the first part after splitting on "@" characters).
    """
    # We unquote prior to quoting to make sure nothing is double quoted.
    # Also, on Windows the path part might contain a drive letter which
    # should not be quoted. On Linux where drive letters do not
    # exist, the colon should be quoted. We rely on urllib.request
    # to do the right thing here.
    return urllib.request.pathname2url(urllib.request.url2pathname(part))


# percent-encoded:                   /
_reserved_chars_re = re.compile("(@|%2F)", re.IGNORECASE)


def _clean_url_path(path: str, is_local_path: bool) -> str:
    """
    Clean the path portion of a URL.
    """
    if is_local_path:
        clean_func = _clean_file_url_path
    else:
        clean_func = _clean_url_path_part

    # Split on the reserved characters prior to cleaning so that
    # revision strings in VCS URLs are properly preserved.
    parts = _reserved_chars_re.split(path)

    cleaned_parts = []
    for to_clean, reserved in pairwise(itertools.chain(parts, [""])):
        cleaned_parts.append(clean_func(to_clean))
        # Normalize %xx escapes (e.g. %2f -> %2F)
        cleaned_parts.append(reserved.upper())

    return "".join(cleaned_parts)


def _ensure_quoted_url(url: str) -> str:
    """
    Make sure a link is fully quoted.
    For example, if ' ' occurs in the URL, it will be replaced with "%20",
    and without double-quoting other characters.
    """
    # Split the URL into parts according to the general structure
    # `scheme://netloc/path?query#fragment`.
    result = urllib.parse.urlsplit(url)
    # If the netloc is empty, then the URL refers to a local filesystem path.
    is_local_path = not result.netloc
    path = _clean_url_path(result.path, is_local_path=is_local_path)
    return urllib.parse.urlunsplit(result._replace(path=path))


def _absolute_link_url(base_url: str, url: str) -> str:
    """
    A faster implementation of urllib.parse.urljoin with a shortcut
    for absolute http/https URLs.
    """
    if url.startswith(("https://", "http://")):
        return url
    else:
        return urllib.parse.urljoin(base_url, url)


@functools.total_ordering
class Link:
    """Represents a parsed link from a Package Index's simple URL"""

    __slots__ = [
        "_parsed_url",
        "_url",
        "_path",
        "_hashes",
        "comes_from",
        "requires_python",
        "yanked_reason",
        "metadata_file_data",
        "cache_link_parsing",
        "egg_fragment",
    ]

    def __init__(
        self,
        url: str,
        comes_from: Optional[Union[str, "IndexContent"]] = None,
        requires_python: Optional[str] = None,
        yanked_reason: Optional[str] = None,
        metadata_file_data: Optional[MetadataFile] = None,
        cache_link_parsing: bool = True,
        hashes: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        :param url: url of the resource pointed to (href of the link)
        :param comes_from: instance of IndexContent where the link was found,
            or string.
        :param requires_python: String containing the `Requires-Python`
            metadata field, specified in PEP 345. This may be specified by
            a data-requires-python attribute in the HTML link tag, as
            described in PEP 503.
        :param yanked_reason: the reason the file has been yanked, if the
            file has been yanked, or None if the file hasn't been yanked.
            This is the value of the "data-yanked" attribute, if present, in
            a simple repository HTML link. If the file has been yanked but
            no reason was provided, this should be the empty string. See
            PEP 592 for more information and the specification.
        :param metadata_file_data: the metadata attached to the file, or None if
            no such metadata is provided. This argument, if not None, indicates
            that a separate metadata file exists, and also optionally supplies
            hashes for that file.
        :param cache_link_parsing: A flag that is used elsewhere to determine
            whether resources retrieved from this link should be cached. PyPI
            URLs should generally have this set to False, for example.
        :param hashes: A mapping of hash names to digests to allow us to
            determine the validity of a download.
        """

        # The comes_from, requires_python, and metadata_file_data arguments are
        # only used by classmethods of this class, and are not used in client
        # code directly.

        # url can be a UNC windows share
        if url.startswith("\\\\"):
            url = path_to_url(url)

        self._parsed_url = urllib.parse.urlsplit(url)
        # Store the url as a private attribute to prevent accidentally
        # trying to set a new value.
        self._url = url
        # The .path property is hot, so calculate its value ahead of time.
        self._path = urllib.parse.unquote(self._parsed_url.path)

        link_hash = LinkHash.find_hash_url_fragment(url)
        hashes_from_link = {} if link_hash is None else link_hash.as_dict()
        if hashes is None:
            self._hashes = hashes_from_link
        else:
            self._hashes = {**hashes, **hashes_from_link}

        self.comes_from = comes_from
        self.requires_python = requires_python if requires_python else None
        self.yanked_reason = yanked_reason
        self.metadata_file_data = metadata_file_data

        self.cache_link_parsing = cache_link_parsing
        self.egg_fragment = self._egg_fragment()

    @classmethod
    def from_json(
        cls,
        file_data: Dict[str, Any],
        page_url: str,
    ) -> Optional["Link"]:
        """
        Convert an pypi json document from a simple repository page into a Link.
        """
        file_url = file_data.get("url")
        if file_url is None:
            return None

        url = _ensure_quoted_url(_absolute_link_url(page_url, file_url))
        pyrequire = file_data.get("requires-python")
        yanked_reason = file_data.get("yanked")
        hashes = file_data.get("hashes", {})

        # PEP 714: Indexes must use the name core-metadata, but
        # clients should support the old name as a fallback for compatibility.
        metadata_info = file_data.get("core-metadata")
        if metadata_info is None:
            metadata_info = file_data.get("dist-info-metadata")

        # The metadata info value may be a boolean, or a dict of hashes.
        if isinstance(metadata_info, dict):
            # The file exists, and hashes have been supplied
            metadata_file_data = MetadataFile(supported_hashes(metadata_info))
        elif metadata_info:
            # The file exists, but there are no hashes
            metadata_file_data = MetadataFile(None)
        else:
            # False or not present: the file does not exist
            metadata_file_data = None

        # The Link.yanked_reason expects an empty string instead of a boolean.
        if yanked_reason and not isinstance(yanked_reason, str):
            yanked_reason = ""
        # The Link.yanked_reason expects None instead of False.
        elif not yanked_reason:
            yanked_reason = None

        return cls(
            url,
            comes_from=page_url,
            requires_python=pyrequire,
            yanked_reason=yanked_reason,
            hashes=hashes,
            metadata_file_data=metadata_file_data,
        )

    @classmethod
    def from_element(
        cls,
        anchor_attribs: Dict[str, Optional[str]],
        page_url: str,
        base_url: str,
    ) -> Optional["Link"]:
        """
        Convert an anchor element's attributes in a simple repository page to a Link.
        """
        href = anchor_attribs.get("href")
        if not href:
            return None

        url = _ensure_quoted_url(_absolute_link_url(base_url, href))
        pyrequire = anchor_attribs.get("data-requires-python")
        yanked_reason = anchor_attribs.get("data-yanked")

        # PEP 714: Indexes must use the name data-core-metadata, but
        # clients should support the old name as a fallback for compatibility.
        metadata_info = anchor_attribs.get("data-core-metadata")
        if metadata_info is None:
            metadata_info = anchor_attribs.get("data-dist-info-metadata")
        # The metadata info value may be the string "true", or a string of
        # the form "hashname=hashval"
        if metadata_info == "true":
            # The file exists, but there are no hashes
            metadata_file_data = MetadataFile(None)
        elif metadata_info is None:
            # The file does not exist
            metadata_file_data = None
        else:
            # The file exists, and hashes have been supplied
            hashname, sep, hashval = metadata_info.partition("=")
            if sep == "=":
                metadata_file_data = MetadataFile(supported_hashes({hashname: hashval}))
            else:
                # Error - data is wrong. Treat as no hashes supplied.
                logger.debug(
                    "Index returned invalid data-dist-info-metadata value: %s",
                    metadata_info,
                )
                metadata_file_data = MetadataFile(None)

        return cls(
            url,
            comes_from=page_url,
            requires_python=pyrequire,
            yanked_reason=yanked_reason,
            metadata_file_data=metadata_file_data,
        )

    def __str__(self) -> str:
        if self.requires_python:
            rp = f" (requires-python:{self.requires_python})"
        else:
            rp = ""
        if self.comes_from:
            return f"{self.redacted_url} (from {self.comes_from}){rp}"
        else:
            return self.redacted_url

    def __repr__(self) -> str:
        return f"<Link {self}>"

    def __hash__(self) -> int:
        return hash(self.url)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Link):
            return NotImplemented
        return self.url == other.url

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Link):
            return NotImplemented
        return self.url < other.url

    @property
    def url(self) -> str:
        return self._url

    @property
    def redacted_url(self) -> str:
        return redact_auth_from_url(self.url)

    @property
    def filename(self) -> str:
        path = self.path.rstrip("/")
        name = posixpath.basename(path)
        if not name:
            # Make sure we don't leak auth information if the netloc
            # includes a username and password.
            netloc, user_pass = split_auth_from_netloc(self.netloc)
            return netloc

        name = urllib.parse.unquote(name)
        assert name, f"URL {self._url!r} produced no filename"
        return name

    @property
    def file_path(self) -> str:
        return url_to_path(self.url)

    @property
    def scheme(self) -> str:
        return self._parsed_url.scheme

    @property
    def netloc(self) -> str:
        """
        This can contain auth information.
        """
        return self._parsed_url.netloc

    @property
    def path(self) -> str:
        return self._path

    def splitext(self) -> Tuple[str, str]:
        return splitext(posixpath.basename(self.path.rstrip("/")))

    @property
    def ext(self) -> str:
        return self.splitext()[1]

    @property
    def url_without_fragment(self) -> str:
        scheme, netloc, path, query, fragment = self._parsed_url
        return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))

    _egg_fragment_re = re.compile(r"[#&]egg=([^&]*)")

    # Per PEP 508.
    _project_name_re = re.compile(
        r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", re.IGNORECASE
    )

    def _egg_fragment(self) -> Optional[str]:
        match = self._egg_fragment_re.search(self._url)
        if not match:
            return None

        # An egg fragment looks like a PEP 508 project name, along with
        # an optional extras specifier. Anything else is invalid.
        project_name = match.group(1)
        if not self._project_name_re.match(project_name):
            deprecated(
                reason=f"{self} contains an egg fragment with a non-PEP 508 name.",
                replacement="to use the req @ url syntax, and remove the egg fragment",
                gone_in="25.2",
                issue=13157,
            )

        return project_name

    _subdirectory_fragment_re = re.compile(r"[#&]subdirectory=([^&]*)")

    @property
    def subdirectory_fragment(self) -> Optional[str]:
        match = self._subdirectory_fragment_re.search(self._url)
        if not match:
            return None
        return match.group(1)

    def metadata_link(self) -> Optional["Link"]:
        """Return a link to the associated core metadata file (if any)."""
        if self.metadata_file_data is None:
            return None
        metadata_url = f"{self.url_without_fragment}.metadata"
        if self.metadata_file_data.hashes is None:
            return Link(metadata_url)
        return Link(metadata_url, hashes=self.metadata_file_data.hashes)

    def as_hashes(self) -> Hashes:
        return Hashes({k: [v] for k, v in self._hashes.items()})

    @property
    def hash(self) -> Optional[str]:
        return next(iter(self._hashes.values()), None)

    @property
    def hash_name(self) -> Optional[str]:
        return next(iter(self._hashes), None)

    @property
    def show_url(self) -> str:
        return posixpath.basename(self._url.split("#", 1)[0].split("?", 1)[0])

    @property
    def is_file(self) -> bool:
        return self.scheme == "file"

    def is_existing_dir(self) -> bool:
        return self.is_file and os.path.isdir(self.file_path)

    @property
    def is_wheel(self) -> bool:
        return self.ext == WHEEL_EXTENSION

    @property
    def is_vcs(self) -> bool:
        from pip._internal.vcs import vcs

        return self.scheme in vcs.all_schemes

    @property
    def is_yanked(self) -> bool:
        return self.yanked_reason is not None

    @property
    def has_hash(self) -> bool:
        return bool(self._hashes)

    def is_hash_allowed(self, hashes: Optional[Hashes]) -> bool:
        """
        Return True if the link has a hash and it is allowed by `hashes`.
        """
        if hashes is None:
            return False
        return any(hashes.is_hash_allowed(k, v) for k, v in self._hashes.items())


class _CleanResult(NamedTuple):
    """Convert link for equivalency check.

    This is used in the resolver to check whether two URL-specified requirements
    likely point to the same distribution and can be considered equivalent. This
    equivalency logic avoids comparing URLs literally, which can be too strict
    (e.g. "a=1&b=2" vs "b=2&a=1") and produce conflicts unexpecting to users.

    Currently this does three things:

    1. Drop the basic auth part. This is technically wrong since a server can
       serve different content based on auth, but if it does that, it is even
       impossible to guarantee two URLs without auth are equivalent, since
       the user can input different auth information when prompted. So the
       practical solution is to assume the auth doesn't affect the response.
    2. Parse the query to avoid the ordering issue. Note that ordering under the
       same key in the query are NOT cleaned; i.e. "a=1&a=2" and "a=2&a=1" are
       still considered different.
    3. Explicitly drop most of the fragment part, except ``subdirectory=`` and
       hash values, since it should have no impact the downloaded content. Note
       that this drops the "egg=" part historically used to denote the requested
       project (and extras), which is wrong in the strictest sense, but too many
       people are supplying it inconsistently to cause superfluous resolution
       conflicts, so we choose to also ignore them.
    """

    parsed: urllib.parse.SplitResult
    query: Dict[str, List[str]]
    subdirectory: str
    hashes: Dict[str, str]


def _clean_link(link: Link) -> _CleanResult:
    parsed = link._parsed_url
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    # According to RFC 8089, an empty host in file: means localhost.
    if parsed.scheme == "file" and not netloc:
        netloc = "localhost"
    fragment = urllib.parse.parse_qs(parsed.fragment)
    if "egg" in fragment:
        logger.debug("Ignoring egg= fragment in %s", link)
    try:
        # If there are multiple subdirectory values, use the first one.
        # This matches the behavior of Link.subdirectory_fragment.
        subdirectory = fragment["subdirectory"][0]
    except (IndexError, KeyError):
        subdirectory = ""
    # If there are multiple hash values under the same algorithm, use the
    # first one. This matches the behavior of Link.hash_value.
    hashes = {k: fragment[k][0] for k in _SUPPORTED_HASHES if k in fragment}
    return _CleanResult(
        parsed=parsed._replace(netloc=netloc, query="", fragment=""),
        query=urllib.parse.parse_qs(parsed.query),
        subdirectory=subdirectory,
        hashes=hashes,
    )


@functools.lru_cache(maxsize=None)
def links_equivalent(link1: Link, link2: Link) -> bool:
    return _clean_link(link1) == _clean_link(link2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_threads.py ===
from __future__ import annotations

import contextlib
import contextvars
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar

import attrs
import outcome
from attrs import define
from sniffio import current_async_library_cvar

import trio

from ._core import (
    RunVar,
    TrioToken,
    checkpoint,
    disable_ki_protection,
    enable_ki_protection,
    start_thread_soon,
)
from ._sync import CapacityLimiter, Event
from ._util import coroutine_or_error

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from typing_extensions import TypeVarTuple, Unpack

    from trio._core._traps import RaiseCancelT

    Ts = TypeVarTuple("Ts")

RetT = TypeVar("RetT")


class _ParentTaskData(threading.local):
    """Global due to Threading API, thread local storage for data related to the
    parent task of native Trio threads."""

    token: TrioToken
    abandon_on_cancel: bool
    cancel_register: list[RaiseCancelT | None]
    task_register: list[trio.lowlevel.Task | None]


PARENT_TASK_DATA = _ParentTaskData()

_limiter_local: RunVar[CapacityLimiter] = RunVar("limiter")
# I pulled this number out of the air; it isn't based on anything. Probably we
# should make some kind of measurements to pick a good value.
DEFAULT_LIMIT = 40
_thread_counter = count()


@define
class _ActiveThreadCount:
    count: int
    event: Event


_active_threads_local: RunVar[_ActiveThreadCount] = RunVar("active_threads")


@contextlib.contextmanager
def _track_active_thread() -> Generator[None, None, None]:
    try:
        active_threads_local = _active_threads_local.get()
    except LookupError:
        active_threads_local = _ActiveThreadCount(0, Event())
        _active_threads_local.set(active_threads_local)

    active_threads_local.count += 1
    try:
        yield
    finally:
        active_threads_local.count -= 1
        if active_threads_local.count == 0:
            active_threads_local.event.set()
            active_threads_local.event = Event()


async def wait_all_threads_completed() -> None:
    """Wait until no threads are still running tasks.

    This is intended to be used when testing code with trio.to_thread to
    make sure no tasks are still making progress in a thread. See the
    following code for a usage example::

        async def wait_all_settled():
            while True:
                await trio.testing.wait_all_threads_complete()
                await trio.testing.wait_all_tasks_blocked()
                if trio.testing.active_thread_count() == 0:
                    break
    """

    await checkpoint()

    try:
        active_threads_local = _active_threads_local.get()
    except LookupError:
        # If there would have been active threads, the
        # _active_threads_local would have been set
        return

    while active_threads_local.count != 0:
        await active_threads_local.event.wait()


def active_thread_count() -> int:
    """Returns the number of threads that are currently running a task

    See `trio.testing.wait_all_threads_completed`
    """
    try:
        return _active_threads_local.get().count
    except LookupError:
        return 0


def current_default_thread_limiter() -> CapacityLimiter:
    """Get the default `~trio.CapacityLimiter` used by
    `trio.to_thread.run_sync`.

    The most common reason to call this would be if you want to modify its
    :attr:`~trio.CapacityLimiter.total_tokens` attribute.

    """
    try:
        limiter = _limiter_local.get()
    except LookupError:
        limiter = CapacityLimiter(DEFAULT_LIMIT)
        _limiter_local.set(limiter)
    return limiter


# Eventually we might build this into a full-fledged deadlock-detection
# system; see https://github.com/python-trio/trio/issues/182
# But for now we just need an object to stand in for the thread, so we can
# keep track of who's holding the CapacityLimiter's token.
@attrs.frozen(eq=False, slots=False)
class ThreadPlaceholder:
    name: str


# Types for the to_thread_run_sync message loop
@attrs.frozen(eq=False, slots=False)
class Run(Generic[RetT]):  # type: ignore[explicit-any]
    afn: Callable[..., Awaitable[RetT]]  # type: ignore[explicit-any]
    args: tuple[object, ...]
    context: contextvars.Context = attrs.field(
        init=False,
        factory=contextvars.copy_context,
    )
    queue: stdlib_queue.SimpleQueue[outcome.Outcome[RetT]] = attrs.field(
        init=False,
        factory=stdlib_queue.SimpleQueue,
    )

    @disable_ki_protection
    async def unprotected_afn(self) -> RetT:
        coro = coroutine_or_error(self.afn, *self.args)
        return await coro

    async def run(self) -> None:
        # we use extra checkpoints to pick up and reset any context changes
        task = trio.lowlevel.current_task()
        old_context = task.context
        task.context = self.context.copy()
        await trio.lowlevel.cancel_shielded_checkpoint()
        result = await outcome.acapture(self.unprotected_afn)
        task.context = old_context
        await trio.lowlevel.cancel_shielded_checkpoint()
        self.queue.put_nowait(result)

    async def run_system(self) -> None:
        result = await outcome.acapture(self.unprotected_afn)
        self.queue.put_nowait(result)

    def run_in_host_task(self, token: TrioToken) -> None:
        task_register = PARENT_TASK_DATA.task_register

        def in_trio_thread() -> None:
            task = task_register[0]
            assert task is not None, "guaranteed by abandon_on_cancel semantics"
            trio.lowlevel.reschedule(task, outcome.Value(self))

        token.run_sync_soon(in_trio_thread)

    def run_in_system_nursery(self, token: TrioToken) -> None:
        def in_trio_thread() -> None:
            try:
                trio.lowlevel.spawn_system_task(
                    self.run_system,
                    name=self.afn,
                    context=self.context,
                )
            except RuntimeError:  # system nursery is closed
                self.queue.put_nowait(
                    outcome.Error(trio.RunFinishedError("system nursery is closed")),
                )

        token.run_sync_soon(in_trio_thread)


@attrs.frozen(eq=False, slots=False)
class RunSync(Generic[RetT]):  # type: ignore[explicit-any]
    fn: Callable[..., RetT]  # type: ignore[explicit-any]
    args: tuple[object, ...]
    context: contextvars.Context = attrs.field(
        init=False,
        factory=contextvars.copy_context,
    )
    queue: stdlib_queue.SimpleQueue[outcome.Outcome[RetT]] = attrs.field(
        init=False,
        factory=stdlib_queue.SimpleQueue,
    )

    @disable_ki_protection
    def unprotected_fn(self) -> RetT:
        ret = self.context.run(self.fn, *self.args)

        if inspect.iscoroutine(ret):
            # Manually close coroutine to avoid RuntimeWarnings
            ret.close()
            raise TypeError(
                "Trio expected a synchronous function, but {!r} appears to be "
                "asynchronous".format(getattr(self.fn, "__qualname__", self.fn)),
            )

        return ret

    def run_sync(self) -> None:
        result = outcome.capture(self.unprotected_fn)
        self.queue.put_nowait(result)

    def run_in_host_task(self, token: TrioToken) -> None:
        task_register = PARENT_TASK_DATA.task_register

        def in_trio_thread() -> None:
            task = task_register[0]
            assert task is not None, "guaranteed by abandon_on_cancel semantics"
            trio.lowlevel.reschedule(task, outcome.Value(self))

        token.run_sync_soon(in_trio_thread)

    def run_in_system_nursery(self, token: TrioToken) -> None:
        token.run_sync_soon(self.run_sync)


@enable_ki_protection
async def to_thread_run_sync(
    sync_fn: Callable[[Unpack[Ts]], RetT],
    *args: Unpack[Ts],
    thread_name: str | None = None,
    abandon_on_cancel: bool = False,
    limiter: CapacityLimiter | None = None,
) -> RetT:
    """Convert a blocking operation into an async operation using a thread.

    These two lines are equivalent::

        sync_fn(*args)
        await trio.to_thread.run_sync(sync_fn, *args)

    except that if ``sync_fn`` takes a long time, then the first line will
    block the Trio loop while it runs, while the second line allows other Trio
    tasks to continue working while ``sync_fn`` runs. This is accomplished by
    pushing the call to ``sync_fn(*args)`` off into a worker thread.

    From inside the worker thread, you can get back into Trio using the
    functions in `trio.from_thread`.

    Args:
      sync_fn: An arbitrary synchronous callable.
      *args: Positional arguments to pass to sync_fn. If you need keyword
          arguments, use :func:`functools.partial`.
      abandon_on_cancel (bool): Whether to abandon this thread upon
          cancellation of this operation. See discussion below.
      thread_name (str): Optional string to set the name of the thread.
          Will always set `threading.Thread.name`, but only set the os name
          if pthread.h is available (i.e. most POSIX installations).
          pthread names are limited to 15 characters, and can be read from
          ``/proc/<PID>/task/<SPID>/comm`` or with ``ps -eT``, among others.
          Defaults to ``{sync_fn.__name__|None} from {trio.lowlevel.current_task().name}``.
      limiter (None, or CapacityLimiter-like object):
          An object used to limit the number of simultaneous threads. Most
          commonly this will be a `~trio.CapacityLimiter`, but it could be
          anything providing compatible
          :meth:`~trio.CapacityLimiter.acquire_on_behalf_of` and
          :meth:`~trio.CapacityLimiter.release_on_behalf_of` methods. This
          function will call ``acquire_on_behalf_of`` before starting the
          thread, and ``release_on_behalf_of`` after the thread has finished.

          If None (the default), uses the default `~trio.CapacityLimiter`, as
          returned by :func:`current_default_thread_limiter`.

    **Cancellation handling**: Cancellation is a tricky issue here, because
    neither Python nor the operating systems it runs on provide any general
    mechanism for cancelling an arbitrary synchronous function running in a
    thread. This function will always check for cancellation on entry, before
    starting the thread. But once the thread is running, there are two ways it
    can handle being cancelled:

    * If ``abandon_on_cancel=False``, the function ignores the cancellation and
      keeps going, just like if we had called ``sync_fn`` synchronously. This
      is the default behavior.

    * If ``abandon_on_cancel=True``, then this function immediately raises
      `~trio.Cancelled`. In this case **the thread keeps running in
      background** – we just abandon it to do whatever it's going to do, and
      silently discard any return value or errors that it raises. Only use
      this if you know that the operation is safe and side-effect free. (For
      example: :func:`trio.socket.getaddrinfo` uses a thread with
      ``abandon_on_cancel=True``, because it doesn't really affect anything if a
      stray hostname lookup keeps running in the background.)

      The ``limiter`` is only released after the thread has *actually*
      finished – which in the case of cancellation may be some time after this
      function has returned. If :func:`trio.run` finishes before the thread
      does, then the limiter release method will never be called at all.

    .. warning::

       You should not use this function to call long-running CPU-bound
       functions! In addition to the usual GIL-related reasons why using
       threads for CPU-bound work is not very effective in Python, there is an
       additional problem: on CPython, `CPU-bound threads tend to "starve out"
       IO-bound threads <https://bugs.python.org/issue7946>`__, so using
       threads for CPU-bound work is likely to adversely affect the main
       thread running Trio. If you need to do this, you're better off using a
       worker process, or perhaps PyPy (which still has a GIL, but may do a
       better job of fairly allocating CPU time between threads).

    Returns:
      Whatever ``sync_fn(*args)`` returns.

    Raises:
      Exception: Whatever ``sync_fn(*args)`` raises.

    """
    await trio.lowlevel.checkpoint_if_cancelled()
    # raise early if abandon_on_cancel.__bool__ raises
    # and give a new name to ensure mypy knows it's never None
    abandon_bool = bool(abandon_on_cancel)
    if limiter is None:
        limiter = current_default_thread_limiter()

    # Holds a reference to the task that's blocked in this function waiting
    # for the result – or None if this function was cancelled and we should
    # discard the result.
    task_register: list[trio.lowlevel.Task | None] = [trio.lowlevel.current_task()]
    # Holds a reference to the raise_cancel function provided if a cancellation
    # is attempted against this task - or None if no such delivery has happened.
    cancel_register: list[RaiseCancelT | None] = [None]  # type: ignore[assignment]
    name = f"trio.to_thread.run_sync-{next(_thread_counter)}"
    placeholder = ThreadPlaceholder(name)

    # This function gets scheduled into the Trio run loop to deliver the
    # thread's result.
    def report_back_in_trio_thread_fn(result: outcome.Outcome[RetT]) -> None:
        def do_release_then_return_result() -> RetT:
            # release_on_behalf_of is an arbitrary user-defined method, so it
            # might raise an error. If it does, we want that error to
            # replace the regular return value, and if the regular return was
            # already an exception then we want them to chain.
            try:
                return result.unwrap()
            finally:
                limiter.release_on_behalf_of(placeholder)

        result = outcome.capture(do_release_then_return_result)
        if task_register[0] is not None:
            trio.lowlevel.reschedule(task_register[0], outcome.Value(result))

    current_trio_token = trio.lowlevel.current_trio_token()

    if thread_name is None:
        thread_name = f"{getattr(sync_fn, '__name__', None)} from {trio.lowlevel.current_task().name}"

    def worker_fn() -> RetT:
        PARENT_TASK_DATA.token = current_trio_token
        PARENT_TASK_DATA.abandon_on_cancel = abandon_bool
        PARENT_TASK_DATA.cancel_register = cancel_register
        PARENT_TASK_DATA.task_register = task_register
        try:
            ret = context.run(sync_fn, *args)

            if inspect.iscoroutine(ret):
                # Manually close coroutine to avoid RuntimeWarnings
                ret.close()
                raise TypeError(
                    "Trio expected a sync function, but {!r} appears to be "
                    "asynchronous".format(getattr(sync_fn, "__qualname__", sync_fn)),
                )

            return ret
        finally:
            del PARENT_TASK_DATA.token
            del PARENT_TASK_DATA.abandon_on_cancel
            del PARENT_TASK_DATA.cancel_register
            del PARENT_TASK_DATA.task_register

    context = contextvars.copy_context()
    # Trio doesn't use current_async_library_cvar, but if someone
    # else set it, it would now shine through since
    # sniffio.thread_local isn't set in the new thread. Make sure
    # the new thread sees that it's not running in async context.
    context.run(current_async_library_cvar.set, None)

    def deliver_worker_fn_result(result: outcome.Outcome[RetT]) -> None:
        # If the entire run finished, the task we're trying to contact is
        # certainly long gone -- it must have been cancelled and abandoned
        # us. Just ignore the error in this case.
        with contextlib.suppress(trio.RunFinishedError):
            current_trio_token.run_sync_soon(report_back_in_trio_thread_fn, result)

    await limiter.acquire_on_behalf_of(placeholder)
    with _track_active_thread():
        try:
            start_thread_soon(worker_fn, deliver_worker_fn_result, thread_name)
        except:
            limiter.release_on_behalf_of(placeholder)
            raise

        def abort(raise_cancel: RaiseCancelT) -> trio.lowlevel.Abort:
            # fill so from_thread_check_cancelled can raise
            cancel_register[0] = raise_cancel
            if abandon_bool:
                # empty so report_back_in_trio_thread_fn cannot reschedule
                task_register[0] = None
                return trio.lowlevel.Abort.SUCCEEDED
            else:
                return trio.lowlevel.Abort.FAILED

        while True:
            # wait_task_rescheduled return value cannot be typed
            msg_from_thread: outcome.Outcome[RetT] | Run[object] | RunSync[object] = (
                await trio.lowlevel.wait_task_rescheduled(abort)
            )
            if isinstance(msg_from_thread, outcome.Outcome):
                return msg_from_thread.unwrap()
            elif isinstance(msg_from_thread, Run):
                await msg_from_thread.run()
            elif isinstance(msg_from_thread, RunSync):
                msg_from_thread.run_sync()
            else:  # pragma: no cover, internal debugging guard TODO: use assert_never
                raise TypeError(
                    f"trio.to_thread.run_sync received unrecognized thread message {msg_from_thread!r}.",
                )
            del msg_from_thread


def from_thread_check_cancelled() -> None:
    """Raise `trio.Cancelled` if the associated Trio task entered a cancelled status.

     Only applicable to threads spawned by `trio.to_thread.run_sync`. Poll to allow
     ``abandon_on_cancel=False`` threads to raise :exc:`~trio.Cancelled` at a suitable
     place, or to end abandoned ``abandon_on_cancel=True`` threads sooner than they may
     otherwise.

    Raises:
        Cancelled: If the corresponding call to `trio.to_thread.run_sync` has had a
            delivery of cancellation attempted against it, regardless of the value of
            ``abandon_on_cancel`` supplied as an argument to it.
        RuntimeError: If this thread is not spawned from `trio.to_thread.run_sync`.

    .. note::

       To be precise, :func:`~trio.from_thread.check_cancelled` checks whether the task
       running :func:`trio.to_thread.run_sync` has ever been cancelled since the last
       time it was running a :func:`trio.from_thread.run` or :func:`trio.from_thread.run_sync`
       function. It may raise `trio.Cancelled` even if a cancellation occurred that was
       later hidden by a modification to `trio.CancelScope.shield` between the cancelled
       `~trio.CancelScope` and :func:`trio.to_thread.run_sync`. This differs from the
       behavior of normal Trio checkpoints, which raise `~trio.Cancelled` only if the
       cancellation is still active when the checkpoint executes. The distinction here is
       *exceedingly* unlikely to be relevant to your application, but we mention it
       for completeness.
    """
    try:
        raise_cancel = PARENT_TASK_DATA.cancel_register[0]
    except AttributeError:
        raise RuntimeError(
            "this thread wasn't created by Trio, can't check for cancellation",
        ) from None
    if raise_cancel is not None:
        raise_cancel()


def _send_message_to_trio(
    trio_token: TrioToken | None,
    message_to_trio: Run[RetT] | RunSync[RetT],
) -> RetT:
    """Shared logic of from_thread functions"""
    token_provided = trio_token is not None

    if not token_provided:
        try:
            trio_token = PARENT_TASK_DATA.token
        except AttributeError:
            raise RuntimeError(
                "this thread wasn't created by Trio, pass kwarg trio_token=...",
            ) from None
    elif not isinstance(trio_token, TrioToken):
        raise RuntimeError("Passed kwarg trio_token is not of type TrioToken")

    # Avoid deadlock by making sure we're not called from Trio thread
    try:
        trio.lowlevel.current_task()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("this is a blocking function; call it from a thread")

    if token_provided or PARENT_TASK_DATA.abandon_on_cancel:
        message_to_trio.run_in_system_nursery(trio_token)
    else:
        message_to_trio.run_in_host_task(trio_token)

    return message_to_trio.queue.get().unwrap()


def from_thread_run(
    afn: Callable[[Unpack[Ts]], Awaitable[RetT]],
    *args: Unpack[Ts],
    trio_token: TrioToken | None = None,
) -> RetT:
    """Run the given async function in the parent Trio thread, blocking until it
    is complete.

    Returns:
      Whatever ``afn(*args)`` returns.

    Returns or raises whatever the given function returns or raises. It
    can also raise exceptions of its own:

    Raises:
        RunFinishedError: if the corresponding call to :func:`trio.run` has
            already completed, or if the run has started its final cleanup phase
            and can no longer spawn new system tasks.
        Cancelled: If the original call to :func:`trio.to_thread.run_sync` is cancelled
            (if *trio_token* is None) or the call to :func:`trio.run` completes
            (if *trio_token* is not None) while ``afn(*args)`` is running,
            then *afn* is likely to raise :exc:`trio.Cancelled`.
        RuntimeError: if you try calling this from inside the Trio thread,
            which would otherwise cause a deadlock, or if no ``trio_token`` was
            provided, and we can't infer one from context.
        TypeError: if ``afn`` is not an asynchronous function.

    **Locating a TrioToken**: There are two ways to specify which
    `trio.run` loop to reenter:

        - Spawn this thread from `trio.to_thread.run_sync`. Trio will
          automatically capture the relevant Trio token and use it
          to re-enter the same Trio task.
        - Pass a keyword argument, ``trio_token`` specifying a specific
          `trio.run` loop to re-enter. This is useful in case you have a
          "foreign" thread, spawned using some other framework, and still want
          to enter Trio, or if you want to use a new system task to call ``afn``,
          maybe to avoid the cancellation context of a corresponding
          `trio.to_thread.run_sync` task. You can get this token from
          :func:`trio.lowlevel.current_trio_token`.
    """
    return _send_message_to_trio(trio_token, Run(afn, args))


def from_thread_run_sync(
    fn: Callable[[Unpack[Ts]], RetT],
    *args: Unpack[Ts],
    trio_token: TrioToken | None = None,
) -> RetT:
    """Run the given sync function in the parent Trio thread, blocking until it
    is complete.

    Returns:
      Whatever ``fn(*args)`` returns.

    Returns or raises whatever the given function returns or raises. It
    can also raise exceptions of its own:

    Raises:
        RunFinishedError: if the corresponding call to `trio.run` has
            already completed.
        RuntimeError: if you try calling this from inside the Trio thread,
            which would otherwise cause a deadlock or if no ``trio_token`` was
            provided, and we can't infer one from context.
        TypeError: if ``fn`` is an async function.

    **Locating a TrioToken**: There are two ways to specify which
    `trio.run` loop to reenter:

        - Spawn this thread from `trio.to_thread.run_sync`. Trio will
          automatically capture the relevant Trio token and use it when you
          want to re-enter Trio.
        - Pass a keyword argument, ``trio_token`` specifying a specific
          `trio.run` loop to re-enter. This is useful in case you have a
          "foreign" thread, spawned using some other framework, and still want
          to enter Trio, or if you want to use a new system task to call ``fn``,
          maybe to avoid the cancellation context of a corresponding
          `trio.to_thread.run_sync` task.
    """
    return _send_message_to_trio(trio_token, RunSync(fn, args))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\_parse_latex_antlr.py ===
# Ported from latex2sympy by @augustt198
# https://github.com/augustt198/latex2sympy
# See license in LICENSE.txt
from importlib.metadata import version
import sympy
from sympy.external import import_module
from sympy.printing.str import StrPrinter
from sympy.physics.quantum.state import Bra, Ket

from .errors import LaTeXParsingError


LaTeXParser = LaTeXLexer = MathErrorListener = None

try:
    LaTeXParser = import_module('sympy.parsing.latex._antlr.latexparser',
                                import_kwargs={'fromlist': ['LaTeXParser']}).LaTeXParser
    LaTeXLexer = import_module('sympy.parsing.latex._antlr.latexlexer',
                               import_kwargs={'fromlist': ['LaTeXLexer']}).LaTeXLexer
except Exception:
    pass

ErrorListener = import_module('antlr4.error.ErrorListener',
                              warn_not_installed=True,
                              import_kwargs={'fromlist': ['ErrorListener']}
                              )



if ErrorListener:
    class MathErrorListener(ErrorListener.ErrorListener):  # type:ignore # noqa:F811
        def __init__(self, src):
            super(ErrorListener.ErrorListener, self).__init__()
            self.src = src

        def syntaxError(self, recog, symbol, line, col, msg, e):
            fmt = "%s\n%s\n%s"
            marker = "~" * col + "^"

            if msg.startswith("missing"):
                err = fmt % (msg, self.src, marker)
            elif msg.startswith("no viable"):
                err = fmt % ("I expected something else here", self.src, marker)
            elif msg.startswith("mismatched"):
                names = LaTeXParser.literalNames
                expected = [
                    names[i] for i in e.getExpectedTokens() if i < len(names)
                ]
                if len(expected) < 10:
                    expected = " ".join(expected)
                    err = (fmt % ("I expected one of these: " + expected, self.src,
                                  marker))
                else:
                    err = (fmt % ("I expected something else here", self.src,
                                  marker))
            else:
                err = fmt % ("I don't understand this", self.src, marker)
            raise LaTeXParsingError(err)


def parse_latex(sympy, strict=False):
    antlr4 = import_module('antlr4')

    if None in [antlr4, MathErrorListener] or \
            not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError("LaTeX parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime) or"
                          " conda (antlr-python-runtime), version 4.11")

    sympy = sympy.strip()
    matherror = MathErrorListener(sympy)

    stream = antlr4.InputStream(sympy)
    lex = LaTeXLexer(stream)
    lex.removeErrorListeners()
    lex.addErrorListener(matherror)

    tokens = antlr4.CommonTokenStream(lex)
    parser = LaTeXParser(tokens)

    # remove default console error listener
    parser.removeErrorListeners()
    parser.addErrorListener(matherror)

    relation = parser.math().relation()
    if strict and (relation.start.start != 0 or relation.stop.stop != len(sympy) - 1):
        raise LaTeXParsingError("Invalid LaTeX")
    expr = convert_relation(relation)

    return expr


def convert_relation(rel):
    if rel.expr():
        return convert_expr(rel.expr())

    lh = convert_relation(rel.relation(0))
    rh = convert_relation(rel.relation(1))
    if rel.LT():
        return sympy.StrictLessThan(lh, rh)
    elif rel.LTE():
        return sympy.LessThan(lh, rh)
    elif rel.GT():
        return sympy.StrictGreaterThan(lh, rh)
    elif rel.GTE():
        return sympy.GreaterThan(lh, rh)
    elif rel.EQUAL():
        return sympy.Eq(lh, rh)
    elif rel.NEQ():
        return sympy.Ne(lh, rh)


def convert_expr(expr):
    return convert_add(expr.additive())


def convert_add(add):
    if add.ADD():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        return sympy.Add(lh, rh, evaluate=False)
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        if hasattr(rh, "is_Atom") and rh.is_Atom:
            return sympy.Add(lh, -1 * rh, evaluate=False)
        return sympy.Add(lh, sympy.Mul(-1, rh, evaluate=False), evaluate=False)
    else:
        return convert_mp(add.mp())


def convert_mp(mp):
    if hasattr(mp, 'mp'):
        mp_left = mp.mp(0)
        mp_right = mp.mp(1)
    else:
        mp_left = mp.mp_nofunc(0)
        mp_right = mp.mp_nofunc(1)

    if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, rh, evaluate=False)
    elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
    else:
        if hasattr(mp, 'unary'):
            return convert_unary(mp.unary())
        else:
            return convert_unary(mp.unary_nofunc())


def convert_unary(unary):
    if hasattr(unary, 'unary'):
        nested_unary = unary.unary()
    else:
        nested_unary = unary.unary_nofunc()
    if hasattr(unary, 'postfix_nofunc'):
        first = unary.postfix()
        tail = unary.postfix_nofunc()
        postfix = [first] + tail
    else:
        postfix = unary.postfix()

    if unary.ADD():
        return convert_unary(nested_unary)
    elif unary.SUB():
        numabs = convert_unary(nested_unary)
        # Use Integer(-n) instead of Mul(-1, n)
        return -numabs
    elif postfix:
        return convert_postfix_list(postfix)


def convert_postfix_list(arr, i=0):
    if i >= len(arr):
        raise LaTeXParsingError("Index out of bounds")

    res = convert_postfix(arr[i])
    if isinstance(res, sympy.Expr):
        if i == len(arr) - 1:
            return res  # nothing to multiply by
        else:
            if i > 0:
                left = convert_postfix(arr[i - 1])
                right = convert_postfix(arr[i + 1])
                if isinstance(left, sympy.Expr) and isinstance(
                        right, sympy.Expr):
                    left_syms = convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                    right_syms = convert_postfix(arr[i + 1]).atoms(
                        sympy.Symbol)
                    # if the left and right sides contain no variables and the
                    # symbol in between is 'x', treat as multiplication.
                    if not (left_syms or right_syms) and str(res) == 'x':
                        return convert_postfix_list(arr, i + 1)
            # multiply by next
            return sympy.Mul(
                res, convert_postfix_list(arr, i + 1), evaluate=False)
    else:  # must be derivative
        wrt = res[0]
        if i == len(arr) - 1:
            raise LaTeXParsingError("Expected expression for derivative")
        else:
            expr = convert_postfix_list(arr, i + 1)
            return sympy.Derivative(expr, wrt)


def do_subs(expr, at):
    if at.expr():
        at_expr = convert_expr(at.expr())
        syms = at_expr.atoms(sympy.Symbol)
        if len(syms) == 0:
            return expr
        elif len(syms) > 0:
            sym = next(iter(syms))
            return expr.subs(sym, at_expr)
    elif at.equality():
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        return expr.subs(lh, rh)


def convert_postfix(postfix):
    if hasattr(postfix, 'exp'):
        exp_nested = postfix.exp()
    else:
        exp_nested = postfix.exp_nofunc()

    exp = convert_exp(exp_nested)
    for op in postfix.postfix_op():
        if op.BANG():
            if isinstance(exp, list):
                raise LaTeXParsingError("Cannot apply postfix to derivative")
            exp = sympy.factorial(exp, evaluate=False)
        elif op.eval_at():
            ev = op.eval_at()
            at_b = None
            at_a = None
            if ev.eval_at_sup():
                at_b = do_subs(exp, ev.eval_at_sup())
            if ev.eval_at_sub():
                at_a = do_subs(exp, ev.eval_at_sub())
            if at_b is not None and at_a is not None:
                exp = sympy.Add(at_b, -1 * at_a, evaluate=False)
            elif at_b is not None:
                exp = at_b
            elif at_a is not None:
                exp = at_a

    return exp


def convert_exp(exp):
    if hasattr(exp, 'exp'):
        exp_nested = exp.exp()
    else:
        exp_nested = exp.exp_nofunc()

    if exp_nested:
        base = convert_exp(exp_nested)
        if isinstance(base, list):
            raise LaTeXParsingError("Cannot raise derivative to power")
        if exp.atom():
            exponent = convert_atom(exp.atom())
        elif exp.expr():
            exponent = convert_expr(exp.expr())
        return sympy.Pow(base, exponent, evaluate=False)
    else:
        if hasattr(exp, 'comp'):
            return convert_comp(exp.comp())
        else:
            return convert_comp(exp.comp_nofunc())


def convert_comp(comp):
    if comp.group():
        return convert_expr(comp.group().expr())
    elif comp.abs_group():
        return sympy.Abs(convert_expr(comp.abs_group().expr()), evaluate=False)
    elif comp.atom():
        return convert_atom(comp.atom())
    elif comp.floor():
        return convert_floor(comp.floor())
    elif comp.ceil():
        return convert_ceil(comp.ceil())
    elif comp.func():
        return convert_func(comp.func())


def convert_atom(atom):
    if atom.LETTER():
        sname = atom.LETTER().getText()
        if atom.subexpr():
            if atom.subexpr().expr():  # subscript is expr
                subscript = convert_expr(atom.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(atom.subexpr().atom())
            sname += '_{' + StrPrinter().doprint(subscript) + '}'
        if atom.SINGLE_QUOTES():
            sname += atom.SINGLE_QUOTES().getText()  # put after subscript for easy identify
        return sympy.Symbol(sname)
    elif atom.SYMBOL():
        s = atom.SYMBOL().getText()[1:]
        if s == "infty":
            return sympy.oo
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    subscript = convert_expr(atom.subexpr().expr())
                else:  # subscript is atom
                    subscript = convert_atom(atom.subexpr().atom())
                subscriptName = StrPrinter().doprint(subscript)
                s += '_{' + subscriptName + '}'
            return sympy.Symbol(s)
    elif atom.number():
        s = atom.number().getText().replace(",", "")
        return sympy.Number(s)
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return sympy.Symbol('d' + var.name)
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return sympy.Symbol(text)
    elif atom.frac():
        return convert_frac(atom.frac())
    elif atom.binom():
        return convert_binom(atom.binom())
    elif atom.bra():
        val = convert_expr(atom.bra().expr())
        return Bra(val)
    elif atom.ket():
        val = convert_expr(atom.ket().expr())
        return Ket(val)


def rule2text(ctx):
    stream = ctx.start.getInputStream()
    # starting index of starting token
    startIdx = ctx.start.start
    # stopping index of stopping token
    stopIdx = ctx.stop.stop

    return stream.getText(startIdx, stopIdx)


def convert_frac(frac):
    diff_op = False
    partial_op = False
    if frac.lower and frac.upper:
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1
        if (frac.lower.start == frac.lower.stop
                and frac.lower.start.type == LaTeXLexer.DIFFERENTIAL):
            wrt = get_differential_var_str(frac.lower.start.text)
            diff_op = True
        elif (lower_itv_len == 2 and frac.lower.start.type == LaTeXLexer.SYMBOL
              and frac.lower.start.text == '\\partial'
              and (frac.lower.stop.type == LaTeXLexer.LETTER
                   or frac.lower.stop.type == LaTeXLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == LaTeXLexer.SYMBOL:
                wrt = wrt[1:]

        if diff_op or partial_op:
            wrt = sympy.Symbol(wrt)
            if (diff_op and frac.upper.start == frac.upper.stop
                    and frac.upper.start.type == LaTeXLexer.LETTER
                    and frac.upper.start.text == 'd'):
                return [wrt]
            elif (partial_op and frac.upper.start == frac.upper.stop
                  and frac.upper.start.type == LaTeXLexer.SYMBOL
                  and frac.upper.start.text == '\\partial'):
                return [wrt]
            upper_text = rule2text(frac.upper)

            expr_top = None
            if diff_op and upper_text.startswith('d'):
                expr_top = parse_latex(upper_text[1:])
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = parse_latex(upper_text[len('\\partial'):])
            if expr_top:
                return sympy.Derivative(expr_top, wrt)
    if frac.upper:
        expr_top = convert_expr(frac.upper)
    else:
        expr_top = sympy.Number(frac.upperd.text)
    if frac.lower:
        expr_bot = convert_expr(frac.lower)
    else:
        expr_bot = sympy.Number(frac.lowerd.text)
    inverse_denom = sympy.Pow(expr_bot, -1, evaluate=False)
    if expr_top == 1:
        return inverse_denom
    else:
        return sympy.Mul(expr_top, inverse_denom, evaluate=False)

def convert_binom(binom):
    expr_n = convert_expr(binom.n)
    expr_k = convert_expr(binom.k)
    return sympy.binomial(expr_n, expr_k, evaluate=False)

def convert_floor(floor):
    val = convert_expr(floor.val)
    return sympy.floor(val, evaluate=False)

def convert_ceil(ceil):
    val = convert_expr(ceil.val)
    return sympy.ceiling(val, evaluate=False)

def convert_func(func):
    if func.func_normal():
        if func.L_PAREN():  # function called with parenthesis
            arg = convert_func_arg(func.func_arg())
        else:
            arg = convert_func_arg(func.func_arg_noparens())

        name = func.func_normal().start.text[1:]

        # change arc<trig> -> a<trig>
        if name in [
                "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot"
        ]:
            name = "a" + name[3:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        if name in ["arsinh", "arcosh", "artanh"]:
            name = "a" + name[2:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if name == "exp":
            expr = sympy.exp(arg, evaluate=False)

        if name in ("log", "lg", "ln"):
            if func.subexpr():
                if func.subexpr().expr():
                    base = convert_expr(func.subexpr().expr())
                else:
                    base = convert_atom(func.subexpr().atom())
            elif name == "lg":  # ISO 80000-2:2019
                base = 10
            elif name in ("ln", "log"):  # SymPy's latex printer prints ln as log by default
                base = sympy.E
            expr = sympy.log(arg, base, evaluate=False)

        func_pow = None
        should_pow = True
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())

        if name in [
                "sin", "cos", "tan", "csc", "sec", "cot", "sinh", "cosh",
                "tanh"
        ]:
            if func_pow == -1:
                name = "a" + name
                should_pow = False
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if func_pow and should_pow:
            expr = sympy.Pow(expr, func_pow, evaluate=False)

        return expr
    elif func.LETTER() or func.SYMBOL():
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = func.SYMBOL().getText()[1:]
        fname = str(fname)  # can't be unicode
        if func.subexpr():
            if func.subexpr().expr():  # subscript is expr
                subscript = convert_expr(func.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(func.subexpr().atom())
            subscriptName = StrPrinter().doprint(subscript)
            fname += '_{' + subscriptName + '}'
        if func.SINGLE_QUOTES():
            fname += func.SINGLE_QUOTES().getText()
        input_args = func.args()
        output_args = []
        while input_args.args():  # handle multiple arguments to function
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))
        return sympy.Function(fname)(*output_args)
    elif func.FUNC_INT():
        return handle_integral(func)
    elif func.FUNC_SQRT():
        expr = convert_expr(func.base)
        if func.root:
            r = convert_expr(func.root)
            return sympy.root(expr, r, evaluate=False)
        else:
            return sympy.sqrt(expr, evaluate=False)
    elif func.FUNC_OVERLINE():
        expr = convert_expr(func.base)
        return sympy.conjugate(expr, evaluate=False)
    elif func.FUNC_SUM():
        return handle_sum_or_prod(func, "summation")
    elif func.FUNC_PROD():
        return handle_sum_or_prod(func, "product")
    elif func.FUNC_LIM():
        return handle_limit(func)


def convert_func_arg(arg):
    if hasattr(arg, 'expr'):
        return convert_expr(arg.expr())
    else:
        return convert_mp(arg.mp_nofunc())


def handle_integral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = sympy.Symbol(s[2:])
                else:
                    int_var = sympy.Symbol(s[1:])
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            # Assume dx by default
            int_var = sympy.Symbol('x')

    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        if func.supexpr().atom():
            upper = convert_atom(func.supexpr().atom())
        else:
            upper = convert_expr(func.supexpr().expr())
        return sympy.Integral(integrand, (int_var, lower, upper))
    else:
        return sympy.Integral(integrand, int_var)


def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    iter_var = convert_expr(func.subeq().equality().expr(0))
    start = convert_expr(func.subeq().equality().expr(1))
    if func.supexpr().expr():  # ^{expr}
        end = convert_expr(func.supexpr().expr())
    else:  # ^atom
        end = convert_atom(func.supexpr().atom())

    if name == "summation":
        return sympy.Sum(val, (iter_var, start, end))
    elif name == "product":
        return sympy.Product(val, (iter_var, start, end))


def handle_limit(func):
    sub = func.limit_sub()
    if sub.LETTER():
        var = sympy.Symbol(sub.LETTER().getText())
    elif sub.SYMBOL():
        var = sympy.Symbol(sub.SYMBOL().getText()[1:])
    else:
        var = sympy.Symbol('x')
    if sub.SUB():
        direction = "-"
    elif sub.ADD():
        direction = "+"
    else:
        direction = "+-"
    approaching = convert_expr(sub.expr())
    content = convert_mp(func.mp())

    return sympy.Limit(content, var, approaching, direction)


def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return sympy.Symbol(text)


def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
            idx = i
            break
    text = text[idx:]
    if text[0] == "\\":
        text = text[1:]
    return text

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\concrete\products.py ===
from __future__ import annotations

from .expr_with_intlimits import ExprWithIntLimits
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import RisingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import quo, roots


class Product(ExprWithIntLimits):
    r"""
    Represents unevaluated products.

    Explanation
    ===========

    ``Product`` represents a finite or infinite product, with the first
    argument being the general form of terms in the series, and the second
    argument being ``(dummy_variable, start, end)``, with ``dummy_variable``
    taking all integer values from ``start`` through ``end``. In accordance
    with long-standing mathematical convention, the end term is included in
    the product.

    Finite products
    ===============

    For finite products (and products with symbolic limits assumed to be finite)
    we follow the analogue of the summation convention described by Karr [1],
    especially definition 3 of section 1.4. The product:

    .. math::

        \prod_{m \leq i < n} f(i)

    has *the obvious meaning* for `m < n`, namely:

    .. math::

        \prod_{m \leq i < n} f(i) = f(m) f(m+1) \cdot \ldots \cdot f(n-2) f(n-1)

    with the upper limit value `f(n)` excluded. The product over an empty set is
    one if and only if `m = n`:

    .. math::

        \prod_{m \leq i < n} f(i) = 1  \quad \mathrm{for} \quad  m = n

    Finally, for all other products over empty sets we assume the following
    definition:

    .. math::

        \prod_{m \leq i < n} f(i) = \frac{1}{\prod_{n \leq i < m} f(i)}  \quad \mathrm{for} \quad  m > n

    It is important to note that above we define all products with the upper
    limit being exclusive. This is in contrast to the usual mathematical notation,
    but does not affect the product convention. Indeed we have:

    .. math::

        \prod_{m \leq i < n} f(i) = \prod_{i = m}^{n - 1} f(i)

    where the difference in notation is intentional to emphasize the meaning,
    with limits typeset on the top being inclusive.

    Examples
    ========

    >>> from sympy.abc import a, b, i, k, m, n, x
    >>> from sympy import Product, oo
    >>> Product(k, (k, 1, m))
    Product(k, (k, 1, m))
    >>> Product(k, (k, 1, m)).doit()
    factorial(m)
    >>> Product(k**2,(k, 1, m))
    Product(k**2, (k, 1, m))
    >>> Product(k**2,(k, 1, m)).doit()
    factorial(m)**2

    Wallis' product for pi:

    >>> W = Product(2*i/(2*i-1) * 2*i/(2*i+1), (i, 1, oo))
    >>> W
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    Direct computation currently fails:

    >>> W.doit()
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, oo))

    But we can approach the infinite product by a limit of finite products:

    >>> from sympy import limit
    >>> W2 = Product(2*i/(2*i-1)*2*i/(2*i+1), (i, 1, n))
    >>> W2
    Product(4*i**2/((2*i - 1)*(2*i + 1)), (i, 1, n))
    >>> W2e = W2.doit()
    >>> W2e
    4**n*factorial(n)**2/(2**(2*n)*RisingFactorial(1/2, n)*RisingFactorial(3/2, n))
    >>> limit(W2e, n, oo)
    pi/2

    By the same formula we can compute sin(pi/2):

    >>> from sympy import combsimp, pi, gamma, simplify
    >>> P = pi * x * Product(1 - x**2/k**2, (k, 1, n))
    >>> P = P.subs(x, pi/2)
    >>> P
    pi**2*Product(1 - pi**2/(4*k**2), (k, 1, n))/2
    >>> Pe = P.doit()
    >>> Pe
    pi**2*RisingFactorial(1 - pi/2, n)*RisingFactorial(1 + pi/2, n)/(2*factorial(n)**2)
    >>> limit(Pe, n, oo).gammasimp()
    sin(pi**2/2)
    >>> Pe.rewrite(gamma)
    (-1)**n*pi**2*gamma(pi/2)*gamma(n + 1 + pi/2)/(2*gamma(1 + pi/2)*gamma(-n + pi/2)*gamma(n + 1)**2)

    Products with the lower limit being larger than the upper one:

    >>> Product(1/i, (i, 6, 1)).doit()
    120
    >>> Product(i, (i, 2, 5)).doit()
    120

    The empty product:

    >>> Product(i, (i, n, n-1)).doit()
    1

    An example showing that the symbolic result of a product is still
    valid for seemingly nonsensical values of the limits. Then the Karr
    convention allows us to give a perfectly valid interpretation to
    those products by interchanging the limits according to the above rules:

    >>> P = Product(2, (i, 10, n)).doit()
    >>> P
    2**(n - 9)
    >>> P.subs(n, 5)
    1/16
    >>> Product(2, (i, 10, 5)).doit()
    1/16
    >>> 1/Product(2, (i, 6, 9)).doit()
    1/16

    An explicit example of the Karr summation convention applied to products:

    >>> P1 = Product(x, (i, a, b)).doit()
    >>> P1
    x**(-a + b + 1)
    >>> P2 = Product(x, (i, b+1, a-1)).doit()
    >>> P2
    x**(a - b - 1)
    >>> simplify(P1 * P2)
    1

    And another one:

    >>> P1 = Product(i, (i, b, a)).doit()
    >>> P1
    RisingFactorial(b, a - b + 1)
    >>> P2 = Product(i, (i, a+1, b-1)).doit()
    >>> P2
    RisingFactorial(a + 1, -a + b - 1)
    >>> P1 * P2
    RisingFactorial(b, a - b + 1)*RisingFactorial(a + 1, -a + b - 1)
    >>> combsimp(P1 * P2)
    1

    See Also
    ========

    Sum, summation
    product

    References
    ==========

    .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
           Volume 28 Issue 2, April 1981, Pages 305-350
           https://dl.acm.org/doi/10.1145/322248.322255
    .. [2] https://en.wikipedia.org/wiki/Multiplication#Capital_Pi_notation
    .. [3] https://en.wikipedia.org/wiki/Empty_product
    """

    __slots__ = ()

    limits: tuple[tuple[Symbol, Expr, Expr]]

    def __new__(cls, function, *symbols, **assumptions):
        obj = ExprWithIntLimits.__new__(cls, function, *symbols, **assumptions)
        return obj

    def _eval_rewrite_as_Sum(self, *args, **kwargs):
        return exp(Sum(log(self.function), *self.limits))

    @property
    def term(self):
        return self._args[0]
    function = term

    def _eval_is_zero(self):
        if self.has_empty_sequence:
            return False

        z = self.term.is_zero
        if z is True:
            return True
        if self.has_finite_limits:
            # A Product is zero only if its term is zero assuming finite limits.
            return z

    def _eval_is_extended_real(self):
        if self.has_empty_sequence:
            return True

        return self.function.is_extended_real

    def _eval_is_positive(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_positive and self.has_finite_limits:
            return True

    def _eval_is_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_nonnegative and self.has_finite_limits:
            return True

    def _eval_is_extended_nonnegative(self):
        if self.has_empty_sequence:
            return True
        if self.function.is_extended_nonnegative:
            return True

    def _eval_is_extended_nonpositive(self):
        if self.has_empty_sequence:
            return True

    def _eval_is_finite(self):
        if self.has_finite_limits and self.function.is_finite:
            return True

    def doit(self, **hints):
        # first make sure any definite limits have product
        # variables with matching assumptions
        reps = {}
        for xab in self.limits:
            d = _dummy_with_inherited_properties_concrete(xab)
            if d:
                reps[xab[0]] = d
        if reps:
            undo = {v: k for k, v in reps.items()}
            did = self.xreplace(reps).doit(**hints)
            if isinstance(did, tuple):  # when separate=True
                did = tuple([i.xreplace(undo) for i in did])
            else:
                did = did.xreplace(undo)
            return did

        from sympy.simplify.powsimp import powsimp
        f = self.function
        for index, limit in enumerate(self.limits):
            i, a, b = limit
            dif = b - a
            if dif.is_integer and dif.is_negative:
                a, b = b + 1, a - 1
                f = 1 / f

            g = self._eval_product(f, (i, a, b))
            if g in (None, S.NaN):
                return self.func(powsimp(f), *self.limits[index:])
            else:
                f = g

        if hints.get('deep', True):
            return f.doit(**hints)
        else:
            return powsimp(f)

    def _eval_conjugate(self):
        return self.func(self.function.conjugate(), *self.limits)

    def _eval_product(self, term, limits):

        (k, a, n) = limits

        if k not in term.free_symbols:
            if (term - 1).is_zero:
                return S.One
            return term**(n - a + 1)

        if a == n:
            return term.subs(k, a)

        from .delta import deltaproduct, _has_simple_delta
        if term.has(KroneckerDelta) and _has_simple_delta(term, limits[0]):
            return deltaproduct(term, limits)

        dif = n - a
        definite = dif.is_Integer
        if definite and (dif < 100):
            return self._eval_product_direct(term, limits)

        elif term.is_polynomial(k):
            poly = term.as_poly(k)

            A = B = Q = S.One

            all_roots = roots(poly)

            M = 0
            for r, m in all_roots.items():
                M += m
                A *= RisingFactorial(a - r, n - a + 1)**m
                Q *= (n - r)**m

            if M < poly.degree():
                arg = quo(poly, Q.as_poly(k))
                B = self.func(arg, (k, a, n)).doit()

            return poly.LC()**(n - a + 1) * A * B

        elif term.is_Add:
            factored = factor_terms(term, fraction=True)
            if factored.is_Mul:
                return self._eval_product(factored, (k, a, n))

        elif term.is_Mul:
            # Factor in part without the summation variable and part with
            without_k, with_k = term.as_coeff_mul(k)

            if len(with_k) >= 2:
                # More than one term including k, so still a multiplication
                exclude, include = [], []
                for t in with_k:
                    p = self._eval_product(t, (k, a, n))

                    if p is not None:
                        exclude.append(p)
                    else:
                        include.append(t)

                if not exclude:
                    return None
                else:
                    arg = term._new_rawargs(*include)
                    A = Mul(*exclude)
                    B = self.func(arg, (k, a, n)).doit()
                    return without_k**(n - a + 1)*A * B
            else:
                # Just a single term
                p = self._eval_product(with_k[0], (k, a, n))
                if p is None:
                    p = self.func(with_k[0], (k, a, n)).doit()
                return without_k**(n - a + 1)*p


        elif term.is_Pow:
            if not term.base.has(k):
                s = summation(term.exp, (k, a, n))

                return term.base**s
            elif not term.exp.has(k):
                p = self._eval_product(term.base, (k, a, n))

                if p is not None:
                    return p**term.exp

        elif isinstance(term, Product):
            evaluated = term.doit()
            f = self._eval_product(evaluated, limits)
            if f is None:
                return self.func(evaluated, limits)
            else:
                return f

        if definite:
            return self._eval_product_direct(term, limits)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import product_simplify
        rv = product_simplify(self, **kwargs)
        return rv.doit() if kwargs['doit'] else rv

    def _eval_transpose(self):
        if self.is_commutative:
            return self.func(self.function.transpose(), *self.limits)
        return None

    def _eval_product_direct(self, term, limits):
        (k, a, n) = limits
        return Mul(*[term.subs(k, a + i) for i in range(n - a + 1)])

    def _eval_derivative(self, x):
        if isinstance(x, Symbol) and x not in self.free_symbols:
            return S.Zero
        f, limits = self.function, list(self.limits)
        limit = limits.pop(-1)
        if limits:
            f = self.func(f, *limits)
        i, a, b = limit
        if x in a.free_symbols or x in b.free_symbols:
            return None
        h = Dummy()
        rv = Sum( Product(f, (i, a, h - 1)) * Product(f, (i, h + 1, b)) * Derivative(f, x, evaluate=True).subs(i, h), (h, a, b))
        return rv

    def is_convergent(self):
        r"""
        See docs of :obj:`.Sum.is_convergent()` for explanation of convergence
        in SymPy.

        Explanation
        ===========

        The infinite product:

        .. math::

            \prod_{1 \leq i < \infty} f(i)

        is defined by the sequence of partial products:

        .. math::

            \prod_{i=1}^{n} f(i) = f(1) f(2) \cdots f(n)

        as n increases without bound. The product converges to a non-zero
        value if and only if the sum:

        .. math::

            \sum_{1 \leq i < \infty} \log{f(n)}

        converges.

        Examples
        ========

        >>> from sympy import Product, Symbol, cos, pi, exp, oo
        >>> n = Symbol('n', integer=True)
        >>> Product(n/(n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Product(1/n**2, (n, 1, oo)).is_convergent()
        False
        >>> Product(cos(pi/n), (n, 1, oo)).is_convergent()
        True
        >>> Product(exp(-n**2), (n, 1, oo)).is_convergent()
        False

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Infinite_product
        """
        sequence_term = self.function
        log_sum = log(sequence_term)
        lim = self.limits
        try:
            is_conv = Sum(log_sum, *lim).is_convergent()
        except NotImplementedError:
            if Sum(sequence_term - 1, *lim).is_absolutely_convergent() is S.true:
                return S.true
            raise NotImplementedError("The algorithm to find the product convergence of %s "
                                        "is not yet implemented" % (sequence_term))
        return is_conv

    def reverse_order(expr, *indices):
        """
        Reverse the order of a limit in a Product.

        Explanation
        ===========

        ``reverse_order(expr, *indices)`` reverses some limits in the expression
        ``expr`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import gamma, Product, simplify, Sum
        >>> from sympy.abc import x, y, a, b, c, d
        >>> P = Product(x, (x, a, b))
        >>> Pr = P.reverse_order(x)
        >>> Pr
        Product(1/x, (x, b + 1, a - 1))
        >>> Pr = Pr.doit()
        >>> Pr
        1/RisingFactorial(b + 1, a - b - 1)
        >>> simplify(Pr.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), b > -1), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))
        >>> P = P.doit()
        >>> P
        RisingFactorial(a, -a + b + 1)
        >>> simplify(P.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), a > 0), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x*y, (x, a, b), (y, c, d))
        >>> S
        Sum(x*y, (x, a, b), (y, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x*y, (x, b + 1, a - 1), (y, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x*y, (x, b + 1, a - 1), (y, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255

        """
        l_indices = list(indices)

        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = expr.index(indx)

        e = 1
        limits = []
        for i, limit in enumerate(expr.limits):
            l = limit
            if i in l_indices:
                e = -e
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)

        return Product(expr.function ** e, *limits)


def product(*args, **kwargs):
    r"""
    Compute the product.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Sum or
    Integral. product(f, (i, a, b)) computes the product of f with
    respect to i from a to b, i.e.,

    ::

                                     b
                                   _____
        product(f(n), (i, a, b)) = |   | f(n)
                                   |   |
                                   i = a

    If it cannot compute the product, it returns an unevaluated Product object.
    Repeated products can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import product, symbols
    >>> i, n, m, k = symbols('i n m k', integer=True)

    >>> product(i, (i, 1, k))
    factorial(k)
    >>> product(m, (i, 1, k))
    m**k
    >>> product(i, (i, 1, k), (k, 1, n))
    Product(factorial(k), (k, 1, n))

    """

    prod = Product(*args, **kwargs)

    if isinstance(prod, Product):
        return prod.doit(deep=False)
    else:
        return prod