
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\retry.py ===
from __future__ import annotations

import email
import logging
import random
import re
import time
import typing
from itertools import takewhile
from types import TracebackType

from ..exceptions import (
    ConnectTimeoutError,
    InvalidHeader,
    MaxRetryError,
    ProtocolError,
    ProxyError,
    ReadTimeoutError,
    ResponseError,
)
from .util import reraise

if typing.TYPE_CHECKING:
    from typing_extensions import Self

    from ..connectionpool import ConnectionPool
    from ..response import BaseHTTPResponse

log = logging.getLogger(__name__)


# Data structure for representing the metadata of requests that result in a retry.
class RequestHistory(typing.NamedTuple):
    method: str | None
    url: str | None
    error: Exception | None
    status: int | None
    redirect_location: str | None


class Retry:
    """Retry configuration.

    Each retry attempt will create a new Retry object with updated values, so
    they can be safely reused.

    Retries can be defined as a default for a pool:

    .. code-block:: python

        retries = Retry(connect=5, read=2, redirect=5)
        http = PoolManager(retries=retries)
        response = http.request("GET", "https://example.com/")

    Or per-request (which overrides the default for the pool):

    .. code-block:: python

        response = http.request("GET", "https://example.com/", retries=Retry(10))

    Retries can be disabled by passing ``False``:

    .. code-block:: python

        response = http.request("GET", "https://example.com/", retries=False)

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

    :param Collection allowed_methods:
        Set of uppercased HTTP method verbs that we should retry on.

        By default, we only retry on methods which are considered to be
        idempotent (multiple requests with the same parameters end with the
        same state). See :attr:`Retry.DEFAULT_ALLOWED_METHODS`.

        Set to a ``None`` value to retry on any verb.

    :param Collection status_forcelist:
        A set of integer HTTP status codes that we should force a retry on.
        A retry is initiated if the request method is in ``allowed_methods``
        and the response status code is in ``status_forcelist``.

        By default, this is disabled with ``None``.

    :param float backoff_factor:
        A backoff factor to apply between attempts after the second try
        (most errors are resolved immediately by a second try without a
        delay). urllib3 will sleep for::

            {backoff factor} * (2 ** ({number of previous retries}))

        seconds. If `backoff_jitter` is non-zero, this sleep is extended by::

            random.uniform(0, {backoff jitter})

        seconds. For example, if the backoff_factor is 0.1, then :func:`Retry.sleep` will
        sleep for [0.0s, 0.2s, 0.4s, 0.8s, ...] between retries. No backoff will ever
        be longer than `backoff_max`.

        By default, backoff is disabled (factor set to 0).

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

    :param Collection remove_headers_on_redirect:
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

    #: Default maximum backoff time.
    DEFAULT_BACKOFF_MAX = 120

    # Backward compatibility; assigned outside of the class.
    DEFAULT: typing.ClassVar[Retry]

    def __init__(
        self,
        total: bool | int | None = 10,
        connect: int | None = None,
        read: int | None = None,
        redirect: bool | int | None = None,
        status: int | None = None,
        other: int | None = None,
        allowed_methods: typing.Collection[str] | None = DEFAULT_ALLOWED_METHODS,
        status_forcelist: typing.Collection[int] | None = None,
        backoff_factor: float = 0,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        raise_on_redirect: bool = True,
        raise_on_status: bool = True,
        history: tuple[RequestHistory, ...] | None = None,
        respect_retry_after_header: bool = True,
        remove_headers_on_redirect: typing.Collection[
            str
        ] = DEFAULT_REMOVE_HEADERS_ON_REDIRECT,
        backoff_jitter: float = 0.0,
    ) -> None:
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
        self.backoff_max = backoff_max
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = history or ()
        self.respect_retry_after_header = respect_retry_after_header
        self.remove_headers_on_redirect = frozenset(
            h.lower() for h in remove_headers_on_redirect
        )
        self.backoff_jitter = backoff_jitter

    def new(self, **kw: typing.Any) -> Self:
        params = dict(
            total=self.total,
            connect=self.connect,
            read=self.read,
            redirect=self.redirect,
            status=self.status,
            other=self.other,
            allowed_methods=self.allowed_methods,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            backoff_max=self.backoff_max,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            history=self.history,
            remove_headers_on_redirect=self.remove_headers_on_redirect,
            respect_retry_after_header=self.respect_retry_after_header,
            backoff_jitter=self.backoff_jitter,
        )

        params.update(kw)
        return type(self)(**params)  # type: ignore[arg-type]

    @classmethod
    def from_int(
        cls,
        retries: Retry | bool | int | None,
        redirect: bool | int | None = True,
        default: Retry | bool | int | None = None,
    ) -> Retry:
        """Backwards-compatibility for the old retries format."""
        if retries is None:
            retries = default if default is not None else cls.DEFAULT

        if isinstance(retries, Retry):
            return retries

        redirect = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect)
        log.debug("Converted retries value: %r -> %r", retries, new_retries)
        return new_retries

    def get_backoff_time(self) -> float:
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
        if self.backoff_jitter != 0.0:
            backoff_value += random.random() * self.backoff_jitter
        return float(max(0, min(self.backoff_max, backoff_value)))

    def parse_retry_after(self, retry_after: str) -> float:
        seconds: float
        # Whitespace: https://tools.ietf.org/html/rfc7230#section-3.2.4
        if re.match(r"^\s*[0-9]+\s*$", retry_after):
            seconds = int(retry_after)
        else:
            retry_date_tuple = email.utils.parsedate_tz(retry_after)
            if retry_date_tuple is None:
                raise InvalidHeader(f"Invalid Retry-After header: {retry_after}")

            retry_date = email.utils.mktime_tz(retry_date_tuple)
            seconds = retry_date - time.time()

        seconds = max(seconds, 0)

        return seconds

    def get_retry_after(self, response: BaseHTTPResponse) -> float | None:
        """Get the value of Retry-After in seconds."""

        retry_after = response.headers.get("Retry-After")

        if retry_after is None:
            return None

        return self.parse_retry_after(retry_after)

    def sleep_for_retry(self, response: BaseHTTPResponse) -> bool:
        retry_after = self.get_retry_after(response)
        if retry_after:
            time.sleep(retry_after)
            return True

        return False

    def _sleep_backoff(self) -> None:
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)

    def sleep(self, response: BaseHTTPResponse | None = None) -> None:
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

    def _is_connection_error(self, err: Exception) -> bool:
        """Errors when we're fairly sure that the server did not receive the
        request, so it should be safe to retry.
        """
        if isinstance(err, ProxyError):
            err = err.original_error
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err: Exception) -> bool:
        """Errors that occur after the request has been started, so we should
        assume that the server began processing it.
        """
        return isinstance(err, (ReadTimeoutError, ProtocolError))

    def _is_method_retryable(self, method: str) -> bool:
        """Checks if a given HTTP method should be retried upon, depending if
        it is included in the allowed_methods
        """
        if self.allowed_methods and method.upper() not in self.allowed_methods:
            return False
        return True

    def is_retry(
        self, method: str, status_code: int, has_retry_after: bool = False
    ) -> bool:
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

        return bool(
            self.total
            and self.respect_retry_after_header
            and has_retry_after
            and (status_code in self.RETRY_AFTER_STATUS_CODES)
        )

    def is_exhausted(self) -> bool:
        """Are we out of retries?"""
        retry_counts = [
            x
            for x in (
                self.total,
                self.connect,
                self.read,
                self.redirect,
                self.status,
                self.other,
            )
            if x
        ]
        if not retry_counts:
            return False

        return min(retry_counts) < 0

    def increment(
        self,
        method: str | None = None,
        url: str | None = None,
        response: BaseHTTPResponse | None = None,
        error: Exception | None = None,
        _pool: ConnectionPool | None = None,
        _stacktrace: TracebackType | None = None,
    ) -> Self:
        """Return a new Retry object with incremented retry counters.

        :param response: A response object, or None, if the server did not
            return a response.
        :type response: :class:`~urllib3.response.BaseHTTPResponse`
        :param Exception error: An error encountered during the request, or
            None if the response was received successfully.

        :return: A new ``Retry`` object.
        """
        if self.total is False and error:
            # Disabled, indicate to re-raise the error.
            raise reraise(type(error), error, _stacktrace)

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
                raise reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1

        elif error and self._is_read_error(error):
            # Read retry?
            if read is False or method is None or not self._is_method_retryable(method):
                raise reraise(type(error), error, _stacktrace)
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
            response_redirect_location = response.get_redirect_location()
            if response_redirect_location:
                redirect_location = response_redirect_location
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
            reason = error or ResponseError(cause)
            raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]

        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)

        return new_retry

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(total={self.total}, connect={self.connect}, "
            f"read={self.read}, redirect={self.redirect}, status={self.status})"
        )


# For backwards compatibility (equivalent to pre-v1.9):
Retry.DEFAULT = Retry(3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\parser.py ===
"""
This module started out as largely a copy paste from the stdlib's
optparse module with the features removed that we do not need from
optparse because we implement them in Click on a higher level (for
instance type handling, help formatting and a lot more).

The plan is to remove more and more from here over time.

The reason this is a different module and not optparse from the stdlib
is that there are differences in 2.x and 3.x about the error messages
generated and optparse in the stdlib uses gettext for no good reason
and might cause us issues.

Click uses parts of optparse written by Gregory P. Ward and maintained
by the Python Software Foundation. This is limited to code in parser.py.

Copyright 2001-2006 Gregory P. Ward. All rights reserved.
Copyright 2002-2006 Python Software Foundation. All rights reserved.
"""

# This code uses parts of optparse written by Gregory P. Ward and
# maintained by the Python Software Foundation.
# Copyright 2001-2006 Gregory P. Ward
# Copyright 2002-2006 Python Software Foundation
from __future__ import annotations

import collections.abc as cabc
import typing as t
from collections import deque
from gettext import gettext as _
from gettext import ngettext

from .exceptions import BadArgumentUsage
from .exceptions import BadOptionUsage
from .exceptions import NoSuchOption
from .exceptions import UsageError

if t.TYPE_CHECKING:
    from .core import Argument as CoreArgument
    from .core import Context
    from .core import Option as CoreOption
    from .core import Parameter as CoreParameter

V = t.TypeVar("V")

# Sentinel value that indicates an option was passed as a flag without a
# value but is not a flag option. Option.consume_value uses this to
# prompt or use the flag_value.
_flag_needs_value = object()


def _unpack_args(
    args: cabc.Sequence[str], nargs_spec: cabc.Sequence[int]
) -> tuple[cabc.Sequence[str | cabc.Sequence[str | None] | None], list[str]]:
    """Given an iterable of arguments and an iterable of nargs specifications,
    it returns a tuple with all the unpacked arguments at the first index
    and all remaining arguments as the second.

    The nargs specification is the number of arguments that should be consumed
    or `-1` to indicate that this position should eat up all the remainders.

    Missing items are filled with `None`.
    """
    args = deque(args)
    nargs_spec = deque(nargs_spec)
    rv: list[str | tuple[str | None, ...] | None] = []
    spos: int | None = None

    def _fetch(c: deque[V]) -> V | None:
        try:
            if spos is None:
                return c.popleft()
            else:
                return c.pop()
        except IndexError:
            return None

    while nargs_spec:
        nargs = _fetch(nargs_spec)

        if nargs is None:
            continue

        if nargs == 1:
            rv.append(_fetch(args))
        elif nargs > 1:
            x = [_fetch(args) for _ in range(nargs)]

            # If we're reversed, we're pulling in the arguments in reverse,
            # so we need to turn them around.
            if spos is not None:
                x.reverse()

            rv.append(tuple(x))
        elif nargs < 0:
            if spos is not None:
                raise TypeError("Cannot have two nargs < 0")

            spos = len(rv)
            rv.append(None)

    # spos is the position of the wildcard (star).  If it's not `None`,
    # we fill it with the remainder.
    if spos is not None:
        rv[spos] = tuple(args)
        args = []
        rv[spos + 1 :] = reversed(rv[spos + 1 :])

    return tuple(rv), list(args)


def _split_opt(opt: str) -> tuple[str, str]:
    first = opt[:1]
    if first.isalnum():
        return "", opt
    if opt[1:2] == first:
        return opt[:2], opt[2:]
    return first, opt[1:]


def _normalize_opt(opt: str, ctx: Context | None) -> str:
    if ctx is None or ctx.token_normalize_func is None:
        return opt
    prefix, opt = _split_opt(opt)
    return f"{prefix}{ctx.token_normalize_func(opt)}"


class _Option:
    def __init__(
        self,
        obj: CoreOption,
        opts: cabc.Sequence[str],
        dest: str | None,
        action: str | None = None,
        nargs: int = 1,
        const: t.Any | None = None,
    ):
        self._short_opts = []
        self._long_opts = []
        self.prefixes: set[str] = set()

        for opt in opts:
            prefix, value = _split_opt(opt)
            if not prefix:
                raise ValueError(f"Invalid start character for option ({opt})")
            self.prefixes.add(prefix[0])
            if len(prefix) == 1 and len(value) == 1:
                self._short_opts.append(opt)
            else:
                self._long_opts.append(opt)
                self.prefixes.add(prefix)

        if action is None:
            action = "store"

        self.dest = dest
        self.action = action
        self.nargs = nargs
        self.const = const
        self.obj = obj

    @property
    def takes_value(self) -> bool:
        return self.action in ("store", "append")

    def process(self, value: t.Any, state: _ParsingState) -> None:
        if self.action == "store":
            state.opts[self.dest] = value  # type: ignore
        elif self.action == "store_const":
            state.opts[self.dest] = self.const  # type: ignore
        elif self.action == "append":
            state.opts.setdefault(self.dest, []).append(value)  # type: ignore
        elif self.action == "append_const":
            state.opts.setdefault(self.dest, []).append(self.const)  # type: ignore
        elif self.action == "count":
            state.opts[self.dest] = state.opts.get(self.dest, 0) + 1  # type: ignore
        else:
            raise ValueError(f"unknown action '{self.action}'")
        state.order.append(self.obj)


class _Argument:
    def __init__(self, obj: CoreArgument, dest: str | None, nargs: int = 1):
        self.dest = dest
        self.nargs = nargs
        self.obj = obj

    def process(
        self,
        value: str | cabc.Sequence[str | None] | None,
        state: _ParsingState,
    ) -> None:
        if self.nargs > 1:
            assert value is not None
            holes = sum(1 for x in value if x is None)
            if holes == len(value):
                value = None
            elif holes != 0:
                raise BadArgumentUsage(
                    _("Argument {name!r} takes {nargs} values.").format(
                        name=self.dest, nargs=self.nargs
                    )
                )

        if self.nargs == -1 and self.obj.envvar is not None and value == ():
            # Replace empty tuple with None so that a value from the
            # environment may be tried.
            value = None

        state.opts[self.dest] = value  # type: ignore
        state.order.append(self.obj)


class _ParsingState:
    def __init__(self, rargs: list[str]) -> None:
        self.opts: dict[str, t.Any] = {}
        self.largs: list[str] = []
        self.rargs = rargs
        self.order: list[CoreParameter] = []


class _OptionParser:
    """The option parser is an internal class that is ultimately used to
    parse options and arguments.  It's modelled after optparse and brings
    a similar but vastly simplified API.  It should generally not be used
    directly as the high level Click classes wrap it for you.

    It's not nearly as extensible as optparse or argparse as it does not
    implement features that are implemented on a higher level (such as
    types or defaults).

    :param ctx: optionally the :class:`~click.Context` where this parser
                should go with.

    .. deprecated:: 8.2
        Will be removed in Click 9.0.
    """

    def __init__(self, ctx: Context | None = None) -> None:
        #: The :class:`~click.Context` for this parser.  This might be
        #: `None` for some advanced use cases.
        self.ctx = ctx
        #: This controls how the parser deals with interspersed arguments.
        #: If this is set to `False`, the parser will stop on the first
        #: non-option.  Click uses this to implement nested subcommands
        #: safely.
        self.allow_interspersed_args: bool = True
        #: This tells the parser how to deal with unknown options.  By
        #: default it will error out (which is sensible), but there is a
        #: second mode where it will ignore it and continue processing
        #: after shifting all the unknown options into the resulting args.
        self.ignore_unknown_options: bool = False

        if ctx is not None:
            self.allow_interspersed_args = ctx.allow_interspersed_args
            self.ignore_unknown_options = ctx.ignore_unknown_options

        self._short_opt: dict[str, _Option] = {}
        self._long_opt: dict[str, _Option] = {}
        self._opt_prefixes = {"-", "--"}
        self._args: list[_Argument] = []

    def add_option(
        self,
        obj: CoreOption,
        opts: cabc.Sequence[str],
        dest: str | None,
        action: str | None = None,
        nargs: int = 1,
        const: t.Any | None = None,
    ) -> None:
        """Adds a new option named `dest` to the parser.  The destination
        is not inferred (unlike with optparse) and needs to be explicitly
        provided.  Action can be any of ``store``, ``store_const``,
        ``append``, ``append_const`` or ``count``.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        opts = [_normalize_opt(opt, self.ctx) for opt in opts]
        option = _Option(obj, opts, dest, action=action, nargs=nargs, const=const)
        self._opt_prefixes.update(option.prefixes)
        for opt in option._short_opts:
            self._short_opt[opt] = option
        for opt in option._long_opts:
            self._long_opt[opt] = option

    def add_argument(self, obj: CoreArgument, dest: str | None, nargs: int = 1) -> None:
        """Adds a positional argument named `dest` to the parser.

        The `obj` can be used to identify the option in the order list
        that is returned from the parser.
        """
        self._args.append(_Argument(obj, dest=dest, nargs=nargs))

    def parse_args(
        self, args: list[str]
    ) -> tuple[dict[str, t.Any], list[str], list[CoreParameter]]:
        """Parses positional arguments and returns ``(values, args, order)``
        for the parsed options and arguments as well as the leftover
        arguments if there are any.  The order is a list of objects as they
        appear on the command line.  If arguments appear multiple times they
        will be memorized multiple times as well.
        """
        state = _ParsingState(args)
        try:
            self._process_args_for_options(state)
            self._process_args_for_args(state)
        except UsageError:
            if self.ctx is None or not self.ctx.resilient_parsing:
                raise
        return state.opts, state.largs, state.order

    def _process_args_for_args(self, state: _ParsingState) -> None:
        pargs, args = _unpack_args(
            state.largs + state.rargs, [x.nargs for x in self._args]
        )

        for idx, arg in enumerate(self._args):
            arg.process(pargs[idx], state)

        state.largs = args
        state.rargs = []

    def _process_args_for_options(self, state: _ParsingState) -> None:
        while state.rargs:
            arg = state.rargs.pop(0)
            arglen = len(arg)
            # Double dashes always handled explicitly regardless of what
            # prefixes are valid.
            if arg == "--":
                return
            elif arg[:1] in self._opt_prefixes and arglen > 1:
                self._process_opts(arg, state)
            elif self.allow_interspersed_args:
                state.largs.append(arg)
            else:
                state.rargs.insert(0, arg)
                return

        # Say this is the original argument list:
        # [arg0, arg1, ..., arg(i-1), arg(i), arg(i+1), ..., arg(N-1)]
        #                            ^
        # (we are about to process arg(i)).
        #
        # Then rargs is [arg(i), ..., arg(N-1)] and largs is a *subset* of
        # [arg0, ..., arg(i-1)] (any options and their arguments will have
        # been removed from largs).
        #
        # The while loop will usually consume 1 or more arguments per pass.
        # If it consumes 1 (eg. arg is an option that takes no arguments),
        # then after _process_arg() is done the situation is:
        #
        #   largs = subset of [arg0, ..., arg(i)]
        #   rargs = [arg(i+1), ..., arg(N-1)]
        #
        # If allow_interspersed_args is false, largs will always be
        # *empty* -- still a subset of [arg0, ..., arg(i-1)], but
        # not a very interesting subset!

    def _match_long_opt(
        self, opt: str, explicit_value: str | None, state: _ParsingState
    ) -> None:
        if opt not in self._long_opt:
            from difflib import get_close_matches

            possibilities = get_close_matches(opt, self._long_opt)
            raise NoSuchOption(opt, possibilities=possibilities, ctx=self.ctx)

        option = self._long_opt[opt]
        if option.takes_value:
            # At this point it's safe to modify rargs by injecting the
            # explicit value, because no exception is raised in this
            # branch.  This means that the inserted value will be fully
            # consumed.
            if explicit_value is not None:
                state.rargs.insert(0, explicit_value)

            value = self._get_value_from_state(opt, option, state)

        elif explicit_value is not None:
            raise BadOptionUsage(
                opt, _("Option {name!r} does not take a value.").format(name=opt)
            )

        else:
            value = None

        option.process(value, state)

    def _match_short_opt(self, arg: str, state: _ParsingState) -> None:
        stop = False
        i = 1
        prefix = arg[0]
        unknown_options = []

        for ch in arg[1:]:
            opt = _normalize_opt(f"{prefix}{ch}", self.ctx)
            option = self._short_opt.get(opt)
            i += 1

            if not option:
                if self.ignore_unknown_options:
                    unknown_options.append(ch)
                    continue
                raise NoSuchOption(opt, ctx=self.ctx)
            if option.takes_value:
                # Any characters left in arg?  Pretend they're the
                # next arg, and stop consuming characters of arg.
                if i < len(arg):
                    state.rargs.insert(0, arg[i:])
                    stop = True

                value = self._get_value_from_state(opt, option, state)

            else:
                value = None

            option.process(value, state)

            if stop:
                break

        # If we got any unknown options we recombine the string of the
        # remaining options and re-attach the prefix, then report that
        # to the state as new larg.  This way there is basic combinatorics
        # that can be achieved while still ignoring unknown arguments.
        if self.ignore_unknown_options and unknown_options:
            state.largs.append(f"{prefix}{''.join(unknown_options)}")

    def _get_value_from_state(
        self, option_name: str, option: _Option, state: _ParsingState
    ) -> t.Any:
        nargs = option.nargs

        if len(state.rargs) < nargs:
            if option.obj._flag_needs_value:
                # Option allows omitting the value.
                value = _flag_needs_value
            else:
                raise BadOptionUsage(
                    option_name,
                    ngettext(
                        "Option {name!r} requires an argument.",
                        "Option {name!r} requires {nargs} arguments.",
                        nargs,
                    ).format(name=option_name, nargs=nargs),
                )
        elif nargs == 1:
            next_rarg = state.rargs[0]

            if (
                option.obj._flag_needs_value
                and isinstance(next_rarg, str)
                and next_rarg[:1] in self._opt_prefixes
                and len(next_rarg) > 1
            ):
                # The next arg looks like the start of an option, don't
                # use it as the value if omitting the value is allowed.
                value = _flag_needs_value
            else:
                value = state.rargs.pop(0)
        else:
            value = tuple(state.rargs[:nargs])
            del state.rargs[:nargs]

        return value

    def _process_opts(self, arg: str, state: _ParsingState) -> None:
        explicit_value = None
        # Long option handling happens in two parts.  The first part is
        # supporting explicitly attached values.  In any case, we will try
        # to long match the option first.
        if "=" in arg:
            long_opt, explicit_value = arg.split("=", 1)
        else:
            long_opt = arg
        norm_long_opt = _normalize_opt(long_opt, self.ctx)

        # At this point we will match the (assumed) long option through
        # the long option matching code.  Note that this allows options
        # like "-foo" to be matched as long options.
        try:
            self._match_long_opt(norm_long_opt, explicit_value, state)
        except NoSuchOption:
            # At this point the long option matching failed, and we need
            # to try with short options.  However there is a special rule
            # which says, that if we have a two character options prefix
            # (applies to "--foo" for instance), we do not dispatch to the
            # short option code and will instead raise the no option
            # error.
            if arg[:2] not in self._opt_prefixes:
                self._match_short_opt(arg, state)
                return

            if not self.ignore_unknown_options:
                raise

            state.largs.append(arg)


def __getattr__(name: str) -> object:
    import warnings

    if name in {
        "OptionParser",
        "Argument",
        "Option",
        "split_opt",
        "normalize_opt",
        "ParsingState",
    }:
        warnings.warn(
            f"'parser.{name}' is deprecated and will be removed in Click 9.0."
            " The old parser is available in 'optparse'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[f"_{name}"]

    if name == "split_arg_string":
        from .shell_completion import split_arg_string

        warnings.warn(
            "Importing 'parser.split_arg_string' is deprecated, it will only be"
            " available in 'shell_completion' in Click 9.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return split_arg_string

    raise AttributeError(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\util.py ===
from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime

rmul = Permutation.rmul
_af_new = Permutation._af_new

############################################
#
# Utilities for computational group theory
#
############################################


def _base_ordering(base, degree):
    r"""
    Order `\{0, 1, \dots, n-1\}` so that base points come first and in order.

    Parameters
    ==========

    base : the base
    degree : the degree of the associated permutation group

    Returns
    =======

    A list ``base_ordering`` such that ``base_ordering[point]`` is the
    number of ``point`` in the ordering.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _base_ordering
    >>> S = SymmetricGroup(4)
    >>> S.schreier_sims()
    >>> _base_ordering(S.base, S.degree)
    [0, 1, 2, 3]

    Notes
    =====

    This is used in backtrack searches, when we define a relation `\ll` on
    the underlying set for a permutation group of degree `n`,
    `\{0, 1, \dots, n-1\}`, so that if `(b_1, b_2, \dots, b_k)` is a base we
    have `b_i \ll b_j` whenever `i<j` and `b_i \ll a` for all
    `i\in\{1,2, \dots, k\}` and `a` is not in the base. The idea is developed
    and applied to backtracking algorithms in [1], pp.108-132. The points
    that are not in the base are taken in increasing order.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    base_len = len(base)
    ordering = [0]*degree
    for i in range(base_len):
        ordering[base[i]] = i
    current = base_len
    for i in range(degree):
        if i not in base:
            ordering[i] = current
            current += 1
    return ordering


def _check_cycles_alt_sym(perm):
    """
    Checks for cycles of prime length p with n/2 < p < n-2.

    Explanation
    ===========

    Here `n` is the degree of the permutation. This is a helper function for
    the function is_alt_sym from sympy.combinatorics.perm_groups.

    Examples
    ========

    >>> from sympy.combinatorics.util import _check_cycles_alt_sym
    >>> from sympy.combinatorics import Permutation
    >>> a = Permutation([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    >>> _check_cycles_alt_sym(a)
    False
    >>> b = Permutation([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
    >>> _check_cycles_alt_sym(b)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.is_alt_sym

    """
    n = perm.size
    af = perm.array_form
    current_len = 0
    total_len = 0
    used = set()
    for i in range(n//2):
        if i not in used and i < n//2 - total_len:
            current_len = 1
            used.add(i)
            j = i
            while af[j] != i:
                current_len += 1
                j = af[j]
                used.add(j)
            total_len += current_len
            if current_len > n//2 and current_len < n - 2 and isprime(current_len):
                return True
    return False


def _distribute_gens_by_base(base, gens):
    r"""
    Distribute the group elements ``gens`` by membership in basic stabilizers.

    Explanation
    ===========

    Notice that for a base `(b_1, b_2, \dots, b_k)`, the basic stabilizers
    are defined as `G^{(i)} = G_{b_1, \dots, b_{i-1}}` for
    `i \in\{1, 2, \dots, k\}`.

    Parameters
    ==========

    base : a sequence of points in `\{0, 1, \dots, n-1\}`
    gens : a list of elements of a permutation group of degree `n`.

    Returns
    =======
    list
        List of length `k`, where `k` is the length of *base*. The `i`-th entry
        contains those elements in *gens* which fix the first `i` elements of
        *base* (so that the `0`-th entry is equal to *gens* itself). If no
        element fixes the first `i` elements of *base*, the `i`-th element is
        set to a list containing the identity element.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> D.strong_gens
    [(0 1 2), (0 2), (1 2)]
    >>> D.base
    [0, 1]
    >>> _distribute_gens_by_base(D.base, D.strong_gens)
    [[(0 1 2), (0 2), (1 2)],
     [(1 2)]]

    See Also
    ========

    _strong_gens_from_distr, _orbits_transversals_from_bsgs,
    _handle_precomputed_bsgs

    """
    base_len = len(base)
    degree = gens[0].size
    stabs = [[] for _ in range(base_len)]
    max_stab_index = 0
    for gen in gens:
        j = 0
        while j < base_len - 1 and gen._array_form[base[j]] == base[j]:
            j += 1
        if j > max_stab_index:
            max_stab_index = j
        for k in range(j + 1):
            stabs[k].append(gen)
    for i in range(max_stab_index + 1, base_len):
        stabs[i].append(_af_new(list(range(degree))))
    return stabs


def _handle_precomputed_bsgs(base, strong_gens, transversals=None,
                             basic_orbits=None, strong_gens_distr=None):
    """
    Calculate BSGS-related structures from those present.

    Explanation
    ===========

    The base and strong generating set must be provided; if any of the
    transversals, basic orbits or distributed strong generators are not
    provided, they will be calculated from the base and strong generating set.

    Parameters
    ==========

    base : the base
    strong_gens : the strong generators
    transversals : basic transversals
    basic_orbits : basic orbits
    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Returns
    =======

    (transversals, basic_orbits, strong_gens_distr)
        where *transversals* are the basic transversals, *basic_orbits* are the
        basic orbits, and *strong_gens_distr* are the strong generators distributed
        by membership in basic stabilizers.

    Examples
    ========

    >>> from sympy.combinatorics.named_groups import DihedralGroup
    >>> from sympy.combinatorics.util import _handle_precomputed_bsgs
    >>> D = DihedralGroup(3)
    >>> D.schreier_sims()
    >>> _handle_precomputed_bsgs(D.base, D.strong_gens,
    ... basic_orbits=D.basic_orbits)
    ([{0: (2), 1: (0 1 2), 2: (0 2)}, {1: (2), 2: (1 2)}], [[0, 1, 2], [1, 2]], [[(0 1 2), (0 2), (1 2)], [(1 2)]])

    See Also
    ========

    _orbits_transversals_from_bsgs, _distribute_gens_by_base

    """
    if strong_gens_distr is None:
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    if transversals is None:
        if basic_orbits is None:
            basic_orbits, transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr)
        else:
            transversals = \
                _orbits_transversals_from_bsgs(base, strong_gens_distr,
                                           transversals_only=True)
    else:
        if basic_orbits is None:
            base_len = len(base)
            basic_orbits = [None]*base_len
            for i in range(base_len):
                basic_orbits[i] = list(transversals[i].keys())
    return transversals, basic_orbits, strong_gens_distr


def _orbits_transversals_from_bsgs(base, strong_gens_distr,
                                   transversals_only=False, slp=False):
    """
    Compute basic orbits and transversals from a base and strong generating set.

    Explanation
    ===========

    The generators are provided as distributed across the basic stabilizers.
    If the optional argument ``transversals_only`` is set to True, only the
    transversals are returned.

    Parameters
    ==========

    base : The base.
    strong_gens_distr : Strong generators distributed by membership in basic stabilizers.
    transversals_only : bool, default: False
        A flag switching between returning only the
        transversals and both orbits and transversals.
    slp : bool, default: False
        If ``True``, return a list of dictionaries containing the
        generator presentations of the elements of the transversals,
        i.e. the list of indices of generators from ``strong_gens_distr[i]``
        such that their product is the relevant transversal element.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _distribute_gens_by_base
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> (S.base, strong_gens_distr)
    ([0, 1], [[(0 1 2), (2)(0 1), (1 2)], [(1 2)]])

    See Also
    ========

    _distribute_gens_by_base, _handle_precomputed_bsgs

    """
    from sympy.combinatorics.perm_groups import _orbit_transversal
    base_len = len(base)
    degree = strong_gens_distr[0][0].size
    transversals = [None]*base_len
    slps = [None]*base_len
    if transversals_only is False:
        basic_orbits = [None]*base_len
    for i in range(base_len):
        transversals[i], slps[i] = _orbit_transversal(degree, strong_gens_distr[i],
                                 base[i], pairs=True, slp=True)
        transversals[i] = dict(transversals[i])
        if transversals_only is False:
            basic_orbits[i] = list(transversals[i].keys())
    if transversals_only:
        return transversals
    else:
        if not slp:
            return basic_orbits, transversals
        return basic_orbits, transversals, slps


def _remove_gens(base, strong_gens, basic_orbits=None, strong_gens_distr=None):
    """
    Remove redundant generators from a strong generating set.

    Parameters
    ==========

    base : a base
    strong_gens : a strong generating set relative to *base*
    basic_orbits : basic orbits
    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Returns
    =======

    A strong generating set with respect to ``base`` which is a subset of
    ``strong_gens``.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _remove_gens
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> S = SymmetricGroup(15)
    >>> base, strong_gens = S.schreier_sims_incremental()
    >>> new_gens = _remove_gens(base, strong_gens)
    >>> len(new_gens)
    14
    >>> _verify_bsgs(S, base, new_gens)
    True

    Notes
    =====

    This procedure is outlined in [1],p.95.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    from sympy.combinatorics.perm_groups import _orbit
    base_len = len(base)
    degree = strong_gens[0].size
    if strong_gens_distr is None:
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    if basic_orbits is None:
        basic_orbits = []
        for i in range(base_len):
            basic_orbit = _orbit(degree, strong_gens_distr[i], base[i])
            basic_orbits.append(basic_orbit)
    strong_gens_distr.append([])
    res = strong_gens[:]
    for i in range(base_len - 1, -1, -1):
        gens_copy = strong_gens_distr[i][:]
        for gen in strong_gens_distr[i]:
            if gen not in strong_gens_distr[i + 1]:
                temp_gens = gens_copy[:]
                temp_gens.remove(gen)
                if temp_gens == []:
                    continue
                temp_orbit = _orbit(degree, temp_gens, base[i])
                if temp_orbit == basic_orbits[i]:
                    gens_copy.remove(gen)
                    res.remove(gen)
    return res


def _strip(g, base, orbits, transversals):
    """
    Attempt to decompose a permutation using a (possibly partial) BSGS
    structure.

    Explanation
    ===========

    This is done by treating the sequence ``base`` as an actual base, and
    the orbits ``orbits`` and transversals ``transversals`` as basic orbits and
    transversals relative to it.

    This process is called "sifting". A sift is unsuccessful when a certain
    orbit element is not found or when after the sift the decomposition
    does not end with the identity element.

    The argument ``transversals`` is a list of dictionaries that provides
    transversal elements for the orbits ``orbits``.

    Parameters
    ==========

    g : permutation to be decomposed
    base : sequence of points
    orbits : list
        A list in which the ``i``-th entry is an orbit of ``base[i]``
        under some subgroup of the pointwise stabilizer of `
        `base[0], base[1], ..., base[i - 1]``. The groups themselves are implicit
        in this function since the only information we need is encoded in the orbits
        and transversals
    transversals : list
        A list of orbit transversals associated with the orbits *orbits*.

    Examples
    ========

    >>> from sympy.combinatorics import Permutation, SymmetricGroup
    >>> from sympy.combinatorics.util import _strip
    >>> S = SymmetricGroup(5)
    >>> S.schreier_sims()
    >>> g = Permutation([0, 2, 3, 1, 4])
    >>> _strip(g, S.base, S.basic_orbits, S.basic_transversals)
    ((4), 5)

    Notes
    =====

    The algorithm is described in [1],pp.89-90. The reason for returning
    both the current state of the element being decomposed and the level
    at which the sifting ends is that they provide important information for
    the randomized version of the Schreier-Sims algorithm.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E."Handbook of computational group theory"

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims
    sympy.combinatorics.perm_groups.PermutationGroup.schreier_sims_random

    """
    h = g._array_form
    base_len = len(base)
    for i in range(base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in orbits[i]:
            return _af_new(h), i + 1
        u = transversals[i][beta]._array_form
        h = _af_rmul(_af_invert(u), h)
    return _af_new(h), base_len + 1


def _strip_af(h, base, orbits, transversals, j, slp=[], slps={}):
    """
    optimized _strip, with h, transversals and result in array form
    if the stripped elements is the identity, it returns False, base_len + 1

    j    h[base[i]] == base[i] for i <= j

    """
    base_len = len(base)
    for i in range(j+1, base_len):
        beta = h[base[i]]
        if beta == base[i]:
            continue
        if beta not in orbits[i]:
            if not slp:
                return h, i + 1
            return h, i + 1, slp
        u = transversals[i][beta]
        if h == u:
            if not slp:
                return False, base_len + 1
            return False, base_len + 1, slp
        h = _af_rmul(_af_invert(u), h)
        if slp:
            u_slp = slps[i][beta][:]
            u_slp.reverse()
            u_slp = [(i, (g,)) for g in u_slp]
            slp = u_slp + slp
    if not slp:
        return h, base_len + 1
    return h, base_len + 1, slp


def _strong_gens_from_distr(strong_gens_distr):
    """
    Retrieve strong generating set from generators of basic stabilizers.

    This is just the union of the generators of the first and second basic
    stabilizers.

    Parameters
    ==========

    strong_gens_distr : strong generators distributed by membership in basic stabilizers

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import (_strong_gens_from_distr,
    ... _distribute_gens_by_base)
    >>> S = SymmetricGroup(3)
    >>> S.schreier_sims()
    >>> S.strong_gens
    [(0 1 2), (2)(0 1), (1 2)]
    >>> strong_gens_distr = _distribute_gens_by_base(S.base, S.strong_gens)
    >>> _strong_gens_from_distr(strong_gens_distr)
    [(0 1 2), (2)(0 1), (1 2)]

    See Also
    ========

    _distribute_gens_by_base

    """
    if len(strong_gens_distr) == 1:
        return strong_gens_distr[0][:]
    else:
        result = strong_gens_distr[0]
        for gen in strong_gens_distr[1]:
            if gen not in result:
                result.append(gen)
        return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\interactive\printing.py ===
"""Tools for setting up printing in interactive sessions. """

from io import BytesIO

from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
from sympy.external import import_module


def _init_python_printing(stringify_func, **settings):
    """Setup printing in Python interactive session. """
    import sys
    import builtins

    def _displayhook(arg):
        """Python's pretty-printer display hook.

           This function was adapted from:

            https://www.python.org/dev/peps/pep-0217/

        """
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg, **settings))
            builtins._ = arg

    sys.displayhook = _displayhook


def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor,
                           backcolor, fontsize, latex_mode, print_builtin,
                           latex_printer, scale, **settings):
    """Setup printing in IPython interactive session. """
    IPython = import_module("IPython", min_module_version="1.0")
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass

    # Guess best font color if none was given based on the ip.colors string.
    # From the IPython documentation:
    #   It has four case-insensitive values: 'nocolor', 'neutral', 'linux',
    #   'lightbg'. The default is neutral, which should be legible on either
    #   dark or light terminal backgrounds. linux is optimised for dark
    #   backgrounds and lightbg for light ones.
    if forecolor is None:
        color = ip.colors.lower()
        if color == 'lightbg':
            forecolor = 'Black'
        elif color == 'linux':
            forecolor = 'White'
        else:
            # No idea, go with gray.
            forecolor = 'Gray'
        debug("init_printing: Automatic foreground color:", forecolor)

    if use_latex == "svg":
        extra_preamble = "\n\\special{color %s}" % forecolor
    else:
        extra_preamble = ""

    imagesize = 'tight'
    offset = "0cm,0cm"
    resolution = round(150*scale)
    dvi = r"-T %s -D %d -bg %s -fg %s -O %s" % (
        imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()

    svg_scale = 150/72*scale
    dvioptions_svg = ["--no-fonts", "--scale={}".format(svg_scale)]

    debug("init_printing: DVIOPTIONS:", dvioptions)
    debug("init_printing: DVIOPTIONS_SVG:", dvioptions_svg)

    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        """caller for pretty, for use in IPython 0.11"""
        if _can_print(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))

    def _preview_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='png', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions, fontsize=fontsize)
        except Exception as e:
            # IPython swallows exceptions
            debug("png printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue()

    def _svg_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='svg', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions_svg, fontsize=fontsize)
        except Exception as e:
            # IPython swallows exceptions
            debug("svg printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue().decode('utf-8')

    def _matplotlib_wrapper(o):
        # mathtext can't render some LaTeX commands. For example, it can't
        # render any LaTeX environments such as array or matrix. So here we
        # ensure that if mathtext fails to render, we return None.
        try:
            try:
                return latex_to_png(o, color=forecolor, scale=scale)
            except TypeError: #  Old IPython version without color and scale
                return latex_to_png(o)
        except ValueError as e:
            debug('matplotlib exception caught:', repr(e))
            return None


    # Hook methods for builtin SymPy printers
    printing_hooks = ('_latex', '_sympystr', '_pretty', '_sympyrepr')


    def _can_print(o):
        """Return True if type o can be printed with one of the SymPy printers.

        If o is a container type, this is True if and only if every element of
        o can be printed in this way.
        """

        try:
            # If you're adding another type, make sure you add it to printable_types
            # later in this file as well

            builtin_types = (list, tuple, set, frozenset)
            if isinstance(o, builtin_types):
                # If the object is a custom subclass with a custom str or
                # repr, use that instead.
                if (type(o).__str__ not in (i.__str__ for i in builtin_types) or
                    type(o).__repr__ not in (i.__repr__ for i in builtin_types)):
                    return False
                return all(_can_print(i) for i in o)
            elif isinstance(o, dict):
                return all(_can_print(i) and _can_print(o[i]) for i in o)
            elif isinstance(o, bool):
                return False
            elif isinstance(o, Printable):
                # types known to SymPy
                return True
            elif any(hasattr(o, hook) for hook in printing_hooks):
                # types which add support themselves
                return True
            elif isinstance(o, (float, int)) and print_builtin:
                return True
            return False
        except RuntimeError:
            return False
            # This is in case maximum recursion depth is reached.
            # Since RecursionError is for versions of Python 3.5+
            # so this is to guard against RecursionError for older versions.

    def _print_latex_png(o):
        """
        A function that returns a png rendered by an external latex
        distribution, falling back to matplotlib rendering
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _preview_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' Falling back to matplotlib backend')
                if latex_mode != 'inline':
                    s = latex(o, mode='inline', **settings)
                return _matplotlib_wrapper(s)

    def _print_latex_svg(o):
        """
        A function that returns a svg rendered by an external latex
        distribution, no fallback available.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _svg_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' No fallback available.')

    def _print_latex_matplotlib(o):
        """
        A function that returns a png rendered by mathtext
        """
        if _can_print(o):
            s = latex(o, mode='inline', **settings)
            return _matplotlib_wrapper(s)

    def _print_latex_text(o):
        """
        A function to generate the latex representation of SymPy expressions.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                return '$\\displaystyle %s$' % s
            return s

    # Printable is our own type, so we handle it with methods instead of
    # the approach required by builtin types. This allows downstream
    # packages to override the methods in their own subclasses of Printable,
    # which avoids the effects of gh-16002.
    printable_types = [float, tuple, list, set, frozenset, dict, int]

    plaintext_formatter = ip.display_formatter.formatters['text/plain']

    # Exception to the rule above: IPython has better dispatching rules
    # for plaintext printing (xref ipython/ipython#8938), and we can't
    # use `_repr_pretty_` without hitting a recursion error in _print_plain.
    for cls in printable_types + [Printable]:
        plaintext_formatter.for_type(cls, _print_plain)

    svg_formatter = ip.display_formatter.formatters['image/svg+xml']
    if use_latex in ('svg', ):
        debug("init_printing: using svg formatter")
        for cls in printable_types:
            svg_formatter.for_type(cls, _print_latex_svg)
        Printable._repr_svg_ = _print_latex_svg
    else:
        debug("init_printing: not using any svg formatter")
        for cls in printable_types:
            # Better way to set this, but currently does not work in IPython
            #png_formatter.for_type(cls, None)
            if cls in svg_formatter.type_printers:
                svg_formatter.type_printers.pop(cls)
        Printable._repr_svg_ = Printable._repr_disabled

    png_formatter = ip.display_formatter.formatters['image/png']
    if use_latex in (True, 'png'):
        debug("init_printing: using png formatter")
        for cls in printable_types:
            png_formatter.for_type(cls, _print_latex_png)
        Printable._repr_png_ = _print_latex_png
    elif use_latex == 'matplotlib':
        debug("init_printing: using matplotlib formatter")
        for cls in printable_types:
            png_formatter.for_type(cls, _print_latex_matplotlib)
        Printable._repr_png_ = _print_latex_matplotlib
    else:
        debug("init_printing: not using any png formatter")
        for cls in printable_types:
            # Better way to set this, but currently does not work in IPython
            #png_formatter.for_type(cls, None)
            if cls in png_formatter.type_printers:
                png_formatter.type_printers.pop(cls)
        Printable._repr_png_ = Printable._repr_disabled

    latex_formatter = ip.display_formatter.formatters['text/latex']
    if use_latex in (True, 'mathjax'):
        debug("init_printing: using mathjax formatter")
        for cls in printable_types:
            latex_formatter.for_type(cls, _print_latex_text)
        Printable._repr_latex_ = _print_latex_text
    else:
        debug("init_printing: not using text/latex formatter")
        for cls in printable_types:
            # Better way to set this, but currently does not work in IPython
            #latex_formatter.for_type(cls, None)
            if cls in latex_formatter.type_printers:
                latex_formatter.type_printers.pop(cls)
        Printable._repr_latex_ = Printable._repr_disabled

def _is_ipython(shell):
    """Is a shell instance an IPython shell?"""
    # shortcut, so we don't import IPython if we don't have to
    from sys import modules
    if 'IPython' not in modules:
        return False
    try:
        from IPython.core.interactiveshell import InteractiveShell
    except ImportError:
        # IPython < 0.11
        try:
            from IPython.iplib import InteractiveShell
        except ImportError:
            # Reaching this points means IPython has changed in a backward-incompatible way
            # that we don't know about. Warn?
            return False
    return isinstance(shell, InteractiveShell)

# Used by the doctester to override the default for no_global
NO_GLOBAL = False

def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    r"""
    Initializes pretty-printer depending on the environment.

    Parameters
    ==========

    pretty_print : bool, default=True
        If ``True``, use :func:`~.pretty_print` to stringify or the provided pretty
        printer; if ``False``, use :func:`~.sstrrepr` to stringify or the provided string
        printer.
    order : string or None, default='lex'
        There are a few different settings for this parameter:
        ``'lex'`` (default), which is lexographic order;
        ``'grlex'``, which is graded lexographic order;
        ``'grevlex'``, which is reversed graded lexographic order;
        ``'old'``, which is used for compatibility reasons and for long expressions;
        ``None``, which sets it to lex.
    use_unicode : bool or None, default=None
        If ``True``, use unicode characters;
        if ``False``, do not use unicode characters;
        if ``None``, make a guess based on the environment.
    use_latex : string, bool, or None, default=None
        If ``True``, use default LaTeX rendering in GUI interfaces (png and
        mathjax);
        if ``False``, do not use LaTeX rendering;
        if ``None``, make a guess based on the environment;
        if ``'png'``, enable LaTeX rendering with an external LaTeX compiler,
        falling back to matplotlib if external compilation fails;
        if ``'matplotlib'``, enable LaTeX rendering with matplotlib;
        if ``'mathjax'``, enable LaTeX text generation, for example MathJax
        rendering in IPython notebook or text rendering in LaTeX documents;
        if ``'svg'``, enable LaTeX rendering with an external latex compiler,
        no fallback
    wrap_line : bool
        If True, lines will wrap at the end; if False, they will not wrap
        but continue as one line. This is only relevant if ``pretty_print`` is
        True.
    num_columns : int or None, default=None
        If ``int``, number of columns before wrapping is set to num_columns; if
        ``None``, number of columns before wrapping is set to terminal width.
        This is only relevant if ``pretty_print`` is ``True``.
    no_global : bool, default=False
        If ``True``, the settings become system wide;
        if ``False``, use just for this console/session.
    ip : An interactive console
        This can either be an instance of IPython,
        or a class that derives from code.InteractiveConsole.
    euler : bool, optional, default=False
        Loads the euler package in the LaTeX preamble for handwritten style
        fonts (https://www.ctan.org/pkg/euler).
    forecolor : string or None, optional, default=None
        DVI setting for foreground color. ``None`` means that either ``'Black'``,
        ``'White'``, or ``'Gray'`` will be selected based on a guess of the IPython
        terminal color setting. See notes.
    backcolor : string, optional, default='Transparent'
        DVI setting for background color. See notes.
    fontsize : string or int, optional, default='10pt'
        A font size to pass to the LaTeX documentclass function in the
        preamble. Note that the options are limited by the documentclass.
        Consider using scale instead.
    latex_mode : string, optional, default='plain'
        The mode used in the LaTeX printer. Can be one of:
        ``{'inline'|'plain'|'equation'|'equation*'}``.
    print_builtin : boolean, optional, default=True
        If ``True`` then floats and integers will be printed. If ``False`` the
        printer will only print SymPy types.
    str_printer : function, optional, default=None
        A custom string printer function. This should mimic
        :func:`~.sstrrepr`.
    pretty_printer : function, optional, default=None
        A custom pretty printer. This should mimic :func:`~.pretty`.
    latex_printer : function, optional, default=None
        A custom LaTeX printer. This should mimic :func:`~.latex`.
    scale : float, optional, default=1.0
        Scale the LaTeX output when using the ``'png'`` or ``'svg'`` backends.
        Useful for high dpi screens.
    settings :
        Any additional settings for the ``latex`` and ``pretty`` commands can
        be used to fine-tune the output.

    Examples
    ========

    >>> from sympy.interactive import init_printing
    >>> from sympy import Symbol, sqrt
    >>> from sympy.abc import x, y
    >>> sqrt(5)
    sqrt(5)
    >>> init_printing(pretty_print=True) # doctest: +SKIP
    >>> sqrt(5) # doctest: +SKIP
      ___
    \/ 5
    >>> theta = Symbol('theta') # doctest: +SKIP
    >>> init_printing(use_unicode=True) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    >>> init_printing(use_unicode=False) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    theta
    >>> init_printing(order='lex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grlex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grevlex') # doctest: +SKIP
    >>> str(y * x**2 + x * y**2) # doctest: +SKIP
    x**2*y + x*y**2
    >>> init_printing(order='old') # doctest: +SKIP
    >>> str(x**2 + y**2 + x + y) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(num_columns=10) # doctest: +SKIP
    >>> x**2 + x + y**2 + y # doctest: +SKIP
    x + y +
    x**2 + y**2

    Notes
    =====

    The foreground and background colors can be selected when using ``'png'`` or
    ``'svg'`` LaTeX rendering. Note that before the ``init_printing`` command is
    executed, the LaTeX rendering is handled by the IPython console and not SymPy.

    The colors can be selected among the 68 standard colors known to ``dvips``,
    for a list see [1]_. In addition, the background color can be
    set to  ``'Transparent'`` (which is the default value).

    When using the ``'Auto'`` foreground color, the guess is based on the
    ``colors`` variable in the IPython console, see [2]_. Hence, if
    that variable is set correctly in your IPython console, there is a high
    chance that the output will be readable, although manual settings may be
    needed.


    References
    ==========

    .. [1] https://en.wikibooks.org/wiki/LaTeX/Colors#The_68_standard_colors_known_to_dvips

    .. [2] https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors

    See Also
    ========

    sympy.printing.latex
    sympy.printing.pretty

    """
    import sys
    from sympy.printing.printer import Printer

    if pretty_print:
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            from sympy.printing import pretty as stringify_func
    else:
        if str_printer is not None:
            stringify_func = str_printer
        else:
            from sympy.printing import sstrrepr as stringify_func

    # Even if ip is not passed, double check that not in IPython shell
    in_ipython = False
    if ip is None:
        try:
            ip = get_ipython()
        except NameError:
            pass
        else:
            in_ipython = (ip is not None)

    if ip and not in_ipython:
        in_ipython = _is_ipython(ip)

    if in_ipython and pretty_print:
        try:
            from IPython.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            # This will be True if we are in the qtconsole or notebook
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) \
                    and 'ipython-console' not in ''.join(sys.argv):
                if use_unicode is None:
                    debug("init_printing: Setting use_unicode to True")
                    use_unicode = True
                if use_latex is None:
                    debug("init_printing: Setting use_latex to True")
                    use_latex = True

    if not NO_GLOBAL and not no_global:
        Printer.set_global_settings(order=order, use_unicode=use_unicode,
                                    wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func

        if pretty_print:
            stringify_func = lambda expr, **settings: \
                             _stringify_func(expr, order=order,
                                             use_unicode=use_unicode,
                                             wrap_line=wrap_line,
                                             num_columns=num_columns,
                                             **settings)
        else:
            stringify_func = \
                lambda expr, **settings: _stringify_func(
                    expr, order=order, **settings)

    if in_ipython:
        mode_in_settings = settings.pop("mode", None)
        if mode_in_settings:
            debug("init_printing: Mode is not able to be set due to internals"
                  "of IPython printing")
        _init_ipython_printing(ip, stringify_func, use_latex, euler,
                               forecolor, backcolor, fontsize, latex_mode,
                               print_builtin, latex_printer, scale,
                               **settings)
    else:
        _init_python_printing(stringify_func, **settings)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\matrices\calculus.py ===
from ..libmp.backend import xrange

# TODO: should use diagonalization-based algorithms

class MatrixCalculusMethods(object):

    def _exp_pade(ctx, a):
        """
        Exponential of a matrix using Pade approximants.

        See G. H. Golub, C. F. van Loan 'Matrix Computations',
        third Ed., page 572

        TODO:
         - find a good estimate for q
         - reduce the number of matrix multiplications to improve
           performance
        """
        def eps_pade(p):
            return ctx.mpf(2)**(3-2*p) * \
                ctx.factorial(p)**2/(ctx.factorial(2*p)**2 * (2*p + 1))
        q = 4
        extraq = 8
        while 1:
            if eps_pade(q) < ctx.eps:
                break
            q += 1
        q += extraq
        j = int(max(1, ctx.mag(ctx.mnorm(a,'inf'))))
        extra = q
        prec = ctx.prec
        ctx.dps += extra + 3
        try:
            a = a/2**j
            na = a.rows
            den = ctx.eye(na)
            num = ctx.eye(na)
            x = ctx.eye(na)
            c = ctx.mpf(1)
            for k in range(1, q+1):
                c *= ctx.mpf(q - k + 1)/((2*q - k + 1) * k)
                x = a*x
                cx = c*x
                num += cx
                den += (-1)**k * cx
            f = ctx.lu_solve_mat(den, num)
            for k in range(j):
                f = f*f
        finally:
            ctx.prec = prec
        return f*1

    def expm(ctx, A, method='taylor'):
        r"""
        Computes the matrix exponential of a square matrix `A`, which is defined
        by the power series

        .. math ::

            \exp(A) = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \ldots

        With method='taylor', the matrix exponential is computed
        using the Taylor series. With method='pade', Pade approximants
        are used instead.

        **Examples**

        Basic examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> expm(zeros(3))
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]
            >>> expm(eye(3))
            [2.71828182845905               0.0               0.0]
            [             0.0  2.71828182845905               0.0]
            [             0.0               0.0  2.71828182845905]
            >>> expm([[1,1,0],[1,0,1],[0,1,0]])
            [ 3.86814500615414  2.26812870852145  0.841130841230196]
            [ 2.26812870852145  2.44114713886289   1.42699786729125]
            [0.841130841230196  1.42699786729125    1.6000162976327]
            >>> expm([[1,1,0],[1,0,1],[0,1,0]], method='pade')
            [ 3.86814500615414  2.26812870852145  0.841130841230196]
            [ 2.26812870852145  2.44114713886289   1.42699786729125]
            [0.841130841230196  1.42699786729125    1.6000162976327]
            >>> expm([[1+j, 0], [1+j,1]])
            [(1.46869393991589 + 2.28735528717884j)                        0.0]
            [  (1.03776739863568 + 3.536943175722j)  (2.71828182845905 + 0.0j)]

        Matrices with large entries are allowed::

            >>> expm(matrix([[1,2],[2,3]])**25)
            [5.65024064048415e+2050488462815550  9.14228140091932e+2050488462815550]
            [9.14228140091932e+2050488462815550  1.47925220414035e+2050488462815551]

        The identity `\exp(A+B) = \exp(A) \exp(B)` does not hold for
        noncommuting matrices::

            >>> A = hilbert(3)
            >>> B = A + eye(3)
            >>> chop(mnorm(A*B - B*A))
            0.0
            >>> chop(mnorm(expm(A+B) - expm(A)*expm(B)))
            0.0
            >>> B = A + ones(3)
            >>> mnorm(A*B - B*A)
            1.8
            >>> mnorm(expm(A+B) - expm(A)*expm(B))
            42.0927851137247

        """
        if method == 'pade':
            prec = ctx.prec
            try:
                A = ctx.matrix(A)
                ctx.prec += 2*A.rows
                res = ctx._exp_pade(A)
            finally:
                ctx.prec = prec
            return res
        A = ctx.matrix(A)
        prec = ctx.prec
        j = int(max(1, ctx.mag(ctx.mnorm(A,'inf'))))
        j += int(0.5*prec**0.5)
        try:
            ctx.prec += 10 + 2*j
            tol = +ctx.eps
            A = A/2**j
            T = A
            Y = A**0 + A
            k = 2
            while 1:
                T *= A * (1/ctx.mpf(k))
                if ctx.mnorm(T, 'inf') < tol:
                    break
                Y += T
                k += 1
            for k in xrange(j):
                Y = Y*Y
        finally:
            ctx.prec = prec
        Y *= 1
        return Y

    def cosm(ctx, A):
        r"""
        Gives the cosine of a square matrix `A`, defined in analogy
        with the matrix exponential.

        Examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> cosm(X)
            [0.54030230586814               0.0               0.0]
            [             0.0  0.54030230586814               0.0]
            [             0.0               0.0  0.54030230586814]
            >>> X = hilbert(3)
            >>> cosm(X)
            [ 0.424403834569555  -0.316643413047167  -0.221474945949293]
            [-0.316643413047167   0.820646708837824  -0.127183694770039]
            [-0.221474945949293  -0.127183694770039   0.909236687217541]
            >>> X = matrix([[1+j,-2],[0,-j]])
            >>> cosm(X)
            [(0.833730025131149 - 0.988897705762865j)  (1.07485840848393 - 0.17192140544213j)]
            [                                     0.0               (1.54308063481524 + 0.0j)]
        """
        B = 0.5 * (ctx.expm(A*ctx.j) + ctx.expm(A*(-ctx.j)))
        if not sum(A.apply(ctx.im).apply(abs)):
            B = B.apply(ctx.re)
        return B

    def sinm(ctx, A):
        r"""
        Gives the sine of a square matrix `A`, defined in analogy
        with the matrix exponential.

        Examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> sinm(X)
            [0.841470984807897                0.0                0.0]
            [              0.0  0.841470984807897                0.0]
            [              0.0                0.0  0.841470984807897]
            >>> X = hilbert(3)
            >>> sinm(X)
            [0.711608512150994  0.339783913247439  0.220742837314741]
            [0.339783913247439  0.244113865695532  0.187231271174372]
            [0.220742837314741  0.187231271174372  0.155816730769635]
            >>> X = matrix([[1+j,-2],[0,-j]])
            >>> sinm(X)
            [(1.29845758141598 + 0.634963914784736j)  (-1.96751511930922 + 0.314700021761367j)]
            [                                    0.0                  (0.0 - 1.1752011936438j)]
        """
        B = (-0.5j) * (ctx.expm(A*ctx.j) - ctx.expm(A*(-ctx.j)))
        if not sum(A.apply(ctx.im).apply(abs)):
            B = B.apply(ctx.re)
        return B

    def _sqrtm_rot(ctx, A, _may_rotate):
        # If the iteration fails to converge, cheat by performing
        # a rotation by a complex number
        u = ctx.j**0.3
        return ctx.sqrtm(u*A, _may_rotate) / ctx.sqrt(u)

    def sqrtm(ctx, A, _may_rotate=2):
        r"""
        Computes a square root of the square matrix `A`, i.e. returns
        a matrix `B = A^{1/2}` such that `B^2 = A`. The square root
        of a matrix, if it exists, is not unique.

        **Examples**

        Square roots of some simple matrices::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> sqrtm([[1,0], [0,1]])
            [1.0  0.0]
            [0.0  1.0]
            >>> sqrtm([[0,0], [0,0]])
            [0.0  0.0]
            [0.0  0.0]
            >>> sqrtm([[2,0],[0,1]])
            [1.4142135623731  0.0]
            [            0.0  1.0]
            >>> sqrtm([[1,1],[1,0]])
            [ (0.920442065259926 - 0.21728689675164j)  (0.568864481005783 + 0.351577584254143j)]
            [(0.568864481005783 + 0.351577584254143j)  (0.351577584254143 - 0.568864481005783j)]
            >>> sqrtm([[1,0],[0,1]])
            [1.0  0.0]
            [0.0  1.0]
            >>> sqrtm([[-1,0],[0,1]])
            [(0.0 - 1.0j)           0.0]
            [         0.0  (1.0 + 0.0j)]
            >>> sqrtm([[j,0],[0,j]])
            [(0.707106781186547 + 0.707106781186547j)                                       0.0]
            [                                     0.0  (0.707106781186547 + 0.707106781186547j)]

        A square root of a rotation matrix, giving the corresponding
        half-angle rotation matrix::

            >>> t1 = 0.75
            >>> t2 = t1 * 0.5
            >>> A1 = matrix([[cos(t1), -sin(t1)], [sin(t1), cos(t1)]])
            >>> A2 = matrix([[cos(t2), -sin(t2)], [sin(t2), cos(t2)]])
            >>> sqrtm(A1)
            [0.930507621912314  -0.366272529086048]
            [0.366272529086048   0.930507621912314]
            >>> A2
            [0.930507621912314  -0.366272529086048]
            [0.366272529086048   0.930507621912314]

        The identity `(A^2)^{1/2} = A` does not necessarily hold::

            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> sqrtm(A**2)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> sqrtm(A)**2
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> A = matrix([[-4,1,4],[7,-8,9],[10,2,11]])
            >>> sqrtm(A**2)
            [  7.43715112194995  -0.324127569985474   1.8481718827526]
            [-0.251549715716942    9.32699765900402  2.48221180985147]
            [  4.11609388833616   0.775751877098258   13.017955697342]
            >>> chop(sqrtm(A)**2)
            [-4.0   1.0   4.0]
            [ 7.0  -8.0   9.0]
            [10.0   2.0  11.0]

        For some matrices, a square root does not exist::

            >>> sqrtm([[0,1], [0,0]])
            Traceback (most recent call last):
              ...
            ZeroDivisionError: matrix is numerically singular

        Two examples from the documentation for Matlab's ``sqrtm``::

            >>> mp.dps = 15; mp.pretty = True
            >>> sqrtm([[7,10],[15,22]])
            [1.56669890360128  1.74077655955698]
            [2.61116483933547  4.17786374293675]
            >>>
            >>> X = matrix(\
            ...   [[5,-4,1,0,0],
            ...   [-4,6,-4,1,0],
            ...   [1,-4,6,-4,1],
            ...   [0,1,-4,6,-4],
            ...   [0,0,1,-4,5]])
            >>> Y = matrix(\
            ...   [[2,-1,-0,-0,-0],
            ...   [-1,2,-1,0,-0],
            ...   [0,-1,2,-1,0],
            ...   [-0,0,-1,2,-1],
            ...   [-0,-0,-0,-1,2]])
            >>> mnorm(sqrtm(X) - Y)
            4.53155328326114e-19

        """
        A = ctx.matrix(A)
        # Trivial
        if A*0 == A:
            return A
        prec = ctx.prec
        if _may_rotate:
            d = ctx.det(A)
            if abs(ctx.im(d)) < 16*ctx.eps and ctx.re(d) < 0:
                return ctx._sqrtm_rot(A, _may_rotate-1)
        try:
            ctx.prec += 10
            tol = ctx.eps * 128
            Y = A
            Z = I = A**0
            k = 0
            # Denman-Beavers iteration
            while 1:
                Yprev = Y
                try:
                    Y, Z = 0.5*(Y+ctx.inverse(Z)), 0.5*(Z+ctx.inverse(Y))
                except ZeroDivisionError:
                    if _may_rotate:
                        Y = ctx._sqrtm_rot(A, _may_rotate-1)
                        break
                    else:
                        raise
                mag1 = ctx.mnorm(Y-Yprev, 'inf')
                mag2 = ctx.mnorm(Y, 'inf')
                if mag1 <= mag2*tol:
                    break
                if _may_rotate and k > 6 and not mag1 < mag2 * 0.001:
                    return ctx._sqrtm_rot(A, _may_rotate-1)
                k += 1
                if k > ctx.prec:
                    raise ctx.NoConvergence
        finally:
            ctx.prec = prec
        Y *= 1
        return Y

    def logm(ctx, A):
        r"""
        Computes a logarithm of the square matrix `A`, i.e. returns
        a matrix `B = \log(A)` such that `\exp(B) = A`. The logarithm
        of a matrix, if it exists, is not unique.

        **Examples**

        Logarithms of some simple matrices::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> logm(X)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            >>> logm(2*X)
            [0.693147180559945                0.0                0.0]
            [              0.0  0.693147180559945                0.0]
            [              0.0                0.0  0.693147180559945]
            >>> logm(expm(X))
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]

        A logarithm of a complex matrix::

            >>> X = matrix([[2+j, 1, 3], [1-j, 1-2*j, 1], [-4, -5, j]])
            >>> B = logm(X)
            >>> nprint(B)
            [ (0.808757 + 0.107759j)    (2.20752 + 0.202762j)   (1.07376 - 0.773874j)]
            [ (0.905709 - 0.107795j)  (0.0287395 - 0.824993j)  (0.111619 + 0.514272j)]
            [(-0.930151 + 0.399512j)   (-2.06266 - 0.674397j)  (0.791552 + 0.519839j)]
            >>> chop(expm(B))
            [(2.0 + 1.0j)           1.0           3.0]
            [(1.0 - 1.0j)  (1.0 - 2.0j)           1.0]
            [        -4.0          -5.0  (0.0 + 1.0j)]

        A matrix `X` close to the identity matrix, for which
        `\log(\exp(X)) = \exp(\log(X)) = X` holds::

            >>> X = eye(3) + hilbert(3)/4
            >>> X
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]
            >>> logm(expm(X))
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]
            >>> expm(logm(X))
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]

        A logarithm of a rotation matrix, giving back the angle of
        the rotation::

            >>> t = 3.7
            >>> A = matrix([[cos(t),sin(t)],[-sin(t),cos(t)]])
            >>> chop(logm(A))
            [             0.0  -2.58318530717959]
            [2.58318530717959                0.0]
            >>> (2*pi-t)
            2.58318530717959

        For some matrices, a logarithm does not exist::

            >>> logm([[1,0], [0,0]])
            Traceback (most recent call last):
              ...
            ZeroDivisionError: matrix is numerically singular

        Logarithm of a matrix with large entries::

            >>> logm(hilbert(3) * 10**20).apply(re)
            [ 45.5597513593433  1.27721006042799  0.317662687717978]
            [ 1.27721006042799  42.5222778973542   2.24003708791604]
            [0.317662687717978  2.24003708791604    42.395212822267]

        """
        A = ctx.matrix(A)
        prec = ctx.prec
        try:
            ctx.prec += 10
            tol = ctx.eps * 128
            I = A**0
            B = A
            n = 0
            while 1:
                B = ctx.sqrtm(B)
                n += 1
                if ctx.mnorm(B-I, 'inf') < 0.125:
                    break
            T = X = B-I
            L = X*0
            k = 1
            while 1:
                if k & 1:
                    L += T / k
                else:
                    L -= T / k
                T *= X
                if ctx.mnorm(T, 'inf') < tol:
                    break
                k += 1
                if k > ctx.prec:
                    raise ctx.NoConvergence
        finally:
            ctx.prec = prec
        L *= 2**n
        return L

    def powm(ctx, A, r):
        r"""
        Computes `A^r = \exp(A \log r)` for a matrix `A` and complex
        number `r`.

        **Examples**

        Powers and inverse powers of a matrix::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> powm(A, 2)
            [ 63.0  20.0   69.0]
            [174.0  89.0  199.0]
            [164.0  48.0  179.0]
            >>> chop(powm(powm(A, 4), 1/4.))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(20)(powm)(A, -4), -1/4.)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> chop(powm(powm(A, 1+0.5j), 1/(1+0.5j)))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(5)(powm)(A, -1.5), -1/(1.5))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]

        A Fibonacci-generating matrix::

            >>> powm([[1,1],[1,0]], 10)
            [89.0  55.0]
            [55.0  34.0]
            >>> fib(10)
            55.0
            >>> powm([[1,1],[1,0]], 6.5)
            [(16.5166626964253 - 0.0121089837381789j)  (10.2078589271083 + 0.0195927472575932j)]
            [(10.2078589271083 + 0.0195927472575932j)  (6.30880376931698 - 0.0317017309957721j)]
            >>> (phi**6.5 - (1-phi)**6.5)/sqrt(5)
            (10.2078589271083 - 0.0195927472575932j)
            >>> powm([[1,1],[1,0]], 6.2)
            [ (14.3076953002666 - 0.008222855781077j)  (8.81733464837593 + 0.0133048601383712j)]
            [(8.81733464837593 + 0.0133048601383712j)  (5.49036065189071 - 0.0215277159194482j)]
            >>> (phi**6.2 - (1-phi)**6.2)/sqrt(5)
            (8.81733464837593 - 0.0133048601383712j)

        """
        A = ctx.matrix(A)
        r = ctx.convert(r)
        prec = ctx.prec
        try:
            ctx.prec += 10
            if ctx.isint(r):
                v = A ** int(r)
            elif ctx.isint(r*2):
                y = int(r*2)
                v = ctx.sqrtm(A) ** y
            else:
                v = ctx.expm(r*ctx.logm(A))
        finally:
            ctx.prec = prec
        v *= 1
        return v

# === atelier-kyo-manager/pricing_calculator.py ===

import math
import requests

def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """
    外部APIからリアルタイムの為替レートを取得します。
    ここでは、簡略化のため固定値を返すか、無料の公開APIを想定します。
    実際の運用では、APIキーが必要な場合や、より信頼性の高いサービスを利用することを推奨します。
    """
    # 例: ExchangeRate-API (無料プランで利用可能なAPI)
    # APIキーが必要な場合は 'YOUR_API_KEY' を置き換えてください
    # APIのURLは変更される可能性があるため、最新のドキュメントを確認してください
    api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status() # HTTPエラーがあれば例外を発生させる
        data = response.json()
        rate = data['rates'].get(to_currency)
        if rate:
            print(f"DEBUG: Fetched exchange rate {from_currency} to {to_currency}: {rate}")
            return rate
        else:
            print(f"WARNING: Could not find exchange rate for {to_currency} in API response. Using default.")
            return 150.0 # フォールバック値
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch exchange rate from API: {e}. Using default.")
        return 150.0 # API呼び出し失敗時のフォールバック値
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while fetching exchange rate: {e}. Using default.")
        return 150.0 # その他のエラー時のフォールバック値


def calculate_profit(
    buyma_price: float,
    source_price: float,
    shipping_cost: float,
    customs_duty_rate: float = 0.1, # 例: 関税率10%
    buyma_commission_rate: float = 0.073, # 例: BUYMA手数料7.3%
    buyma_system_fee: float = 200, # 例: BUYMAシステム利用料200円
    source_currency: str = "USD" # 仕入れ元の通貨を追加
) -> dict:
    """
    BUYMA出品における推定利益と利益率を計算します。

    Args:
        buyma_price (float): BUYMAでの目標販売価格 (日本円)。
        source_price (float): 海外仕入れ先からの商品価格 (元の通貨、例: USD/EUR)。
        shipping_cost (float): 仕入れ先からの送料 (元の通貨)。
        customs_duty_rate (float): 推定関税率 (例: 0.1 は10%)。
        buyma_commission_rate (float): BUYMA手数料率 (例: 0.073 は7.3%)。
        buyma_system_fee (float): 取引ごとの固定BUYMAシステム利用料。
        source_currency (str): 仕入れ元の通貨コード (例: "USD", "EUR")。

    Returns:
        dict: 'profit_estimate' (推定利益) と 'profit_rate' (利益率) を含む辞書。
    """
    # 為替レートを動的に取得
    exchange_rate = get_exchange_rate(source_currency, "JPY")

    # 仕入れ価格と送料を日本円に換算
    source_price_jpy = source_price * exchange_rate
    shipping_cost_jpy = shipping_cost * exchange_rate

    # 関税を含む総コストを計算
    # 関税は通常 (商品価格 + 送料) に対して計算されます
    cost_before_customs = source_price_jpy + shipping_cost_jpy
    customs_duty = cost_before_customs * customs_duty_rate
    total_cost_jpy = cost_before_customs + customs_duty

    # BUYMA手数料を計算
    buyma_commission = buyma_price * buyma_commission_rate
    total_buyma_fees = buyma_commission + buyma_system_fee

    # BUYMAからの純収益を計算
    net_revenue = buyma_price - total_buyma_fees

    # 利益を計算
    profit_estimate = net_revenue - total_cost_jpy

    # 利益率を計算
    profit_rate = (profit_estimate / buyma_price) * 100 if buyma_price > 0 else 0

    return {
        "profit_estimate": round(profit_estimate, 2),
        "profit_rate": round(profit_rate, 2)
    }

# テスト用の使用例 (このファイルが直接実行された場合のみ)
if __name__ == "__main__":
    # 例: BUYMA価格 26800円, SSENSE価格 120 USD, 送料 20 USD
    result = calculate_profit(
        buyma_price=26800,
        source_price=120,
        shipping_cost=20,
        source_currency="USD"
    )
    print(f"推定利益: {result['profit_estimate']} 円")
    print(f"利益率: {result['profit_rate']}%何度も

    # 例2: 利益が出ないケース
    result_low = calculate_profit(
        buyma_price=15000,
        source_price=120,
        shipping_cost=20,
        source_currency="USD"
    )
    print(f"推定利益 (低): {result_low['profit_estimate']} 円")
    print(f"利益率 (低): {result_low['profit_rate']}%何度も

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arraylike.py ===
"""
Methods that can be shared by many array-like classes or subclasses:
    Series
    Index
    ExtensionArray
"""
from __future__ import annotations

import operator
from typing import Any

import numpy as np

from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

from pandas.core.dtypes.generic import ABCNDFrame

from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer

REDUCTION_ALIASES = {
    "maximum": "max",
    "minimum": "min",
    "add": "sum",
    "multiply": "prod",
}


class OpsMixin:
    # -------------------------------------------------------------
    # Comparisons

    def _cmp_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    # -------------------------------------------------------------
    # Logical Methods

    def _logical_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__and__")
    def __and__(self, other):
        return self._logical_method(other, operator.and_)

    @unpack_zerodim_and_defer("__rand__")
    def __rand__(self, other):
        return self._logical_method(other, roperator.rand_)

    @unpack_zerodim_and_defer("__or__")
    def __or__(self, other):
        return self._logical_method(other, operator.or_)

    @unpack_zerodim_and_defer("__ror__")
    def __ror__(self, other):
        return self._logical_method(other, roperator.ror_)

    @unpack_zerodim_and_defer("__xor__")
    def __xor__(self, other):
        return self._logical_method(other, operator.xor)

    @unpack_zerodim_and_defer("__rxor__")
    def __rxor__(self, other):
        return self._logical_method(other, roperator.rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        """
        Get Addition of DataFrame and other, column-wise.

        Equivalent to ``DataFrame.add(other)``.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Object to be added to the DataFrame.

        Returns
        -------
        DataFrame
            The result of adding ``other`` to DataFrame.

        See Also
        --------
        DataFrame.add : Add a DataFrame and another object, with option for index-
            or column-oriented addition.

        Examples
        --------
        >>> df = pd.DataFrame({'height': [1.5, 2.6], 'weight': [500, 800]},
        ...                   index=['elk', 'moose'])
        >>> df
               height  weight
        elk       1.5     500
        moose     2.6     800

        Adding a scalar affects all rows and columns.

        >>> df[['height', 'weight']] + 1.5
               height  weight
        elk       3.0   501.5
        moose     4.1   801.5

        Each element of a list is added to a column of the DataFrame, in order.

        >>> df[['height', 'weight']] + [0.5, 1.5]
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        Keys of a dictionary are aligned to the DataFrame, based on column names;
        each value in the dictionary is added to the corresponding column.

        >>> df[['height', 'weight']] + {'height': 0.5, 'weight': 1.5}
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        When `other` is a :class:`Series`, the index of `other` is aligned with the
        columns of the DataFrame.

        >>> s1 = pd.Series([0.5, 1.5], index=['weight', 'height'])
        >>> df[['height', 'weight']] + s1
               height  weight
        elk       3.0   500.5
        moose     4.1   800.5

        Even when the index of `other` is the same as the index of the DataFrame,
        the :class:`Series` will not be reoriented. If index-wise alignment is desired,
        :meth:`DataFrame.add` should be used with `axis='index'`.

        >>> s2 = pd.Series([0.5, 1.5], index=['elk', 'moose'])
        >>> df[['height', 'weight']] + s2
               elk  height  moose  weight
        elk    NaN     NaN    NaN     NaN
        moose  NaN     NaN    NaN     NaN

        >>> df[['height', 'weight']].add(s2, axis='index')
               height  weight
        elk       2.0   500.5
        moose     4.1   801.5

        When `other` is a :class:`DataFrame`, both columns names and the
        index are aligned.

        >>> other = pd.DataFrame({'height': [0.2, 0.4, 0.6]},
        ...                      index=['elk', 'moose', 'deer'])
        >>> df[['height', 'weight']] + other
               height  weight
        deer      NaN     NaN
        elk       1.7     NaN
        moose     3.0     NaN
        """
        return self._arith_method(other, operator.add)

    @unpack_zerodim_and_defer("__radd__")
    def __radd__(self, other):
        return self._arith_method(other, roperator.radd)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    @unpack_zerodim_and_defer("__rsub__")
    def __rsub__(self, other):
        return self._arith_method(other, roperator.rsub)

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    @unpack_zerodim_and_defer("__rmul__")
    def __rmul__(self, other):
        return self._arith_method(other, roperator.rmul)

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        return self._arith_method(other, roperator.rtruediv)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    @unpack_zerodim_and_defer("__rfloordiv")
    def __rfloordiv__(self, other):
        return self._arith_method(other, roperator.rfloordiv)

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        return self._arith_method(other, roperator.rmod)

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        return self._arith_method(other, divmod)

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        return self._arith_method(other, roperator.rdivmod)

    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    @unpack_zerodim_and_defer("__rpow__")
    def __rpow__(self, other):
        return self._arith_method(other, roperator.rpow)


# -----------------------------------------------------------------------------
# Helpers to implement __array_ufunc__


def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    from pandas.core.frame import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.internals import (
        ArrayManager,
        BlockManager,
    )

    cls = type(self)

    kwargs = _standardize_out_kwarg(**kwargs)

    # for binary ops, use our custom dunder methods
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result

    # Determine if we should defer.
    no_defer = (
        np.ndarray.__array_ufunc__,
        cls.__array_ufunc__,
    )

    for item in inputs:
        higher_priority = (
            hasattr(item, "__array_priority__")
            and item.__array_priority__ > self.__array_priority__
        )
        has_array_ufunc = (
            hasattr(item, "__array_ufunc__")
            and type(item).__array_ufunc__ not in no_defer
            and not isinstance(item, self._HANDLED_TYPES)
        )
        if higher_priority or has_array_ufunc:
            return NotImplemented

    # align all the inputs.
    types = tuple(type(x) for x in inputs)
    alignable = [x for x, t in zip(inputs, types) if issubclass(t, NDFrame)]

    if len(alignable) > 1:
        # This triggers alignment.
        # At the moment, there aren't any ufuncs with more than two inputs
        # so this ends up just being x1.index | x2.index, but we write
        # it to handle *args.
        set_types = set(types)
        if len(set_types) > 1 and {DataFrame, Series}.issubset(set_types):
            # We currently don't handle ufunc(DataFrame, Series)
            # well. Previously this raised an internal ValueError. We might
            # support it someday, so raise a NotImplementedError.
            raise NotImplementedError(
                f"Cannot apply ufunc {ufunc} to mixed DataFrame and Series inputs."
            )
        axes = self.axes
        for obj in alignable[1:]:
            # this relies on the fact that we aren't handling mixed
            # series / frame ufuncs.
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes)):
                axes[i] = ax1.union(ax2)

        reconstruct_axes = dict(zip(self._AXIS_ORDERS, axes))
        inputs = tuple(
            x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x
            for x, t in zip(inputs, types)
        )
    else:
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes))

    if self.ndim == 1:
        names = [getattr(x, "name") for x in inputs if hasattr(x, "name")]
        name = names[0] if len(set(names)) == 1 else None
        reconstruct_kwargs = {"name": name}
    else:
        reconstruct_kwargs = {}

    def reconstruct(result):
        if ufunc.nout > 1:
            # np.modf, np.frexp, np.divmod
            return tuple(_reconstruct(x) for x in result)

        return _reconstruct(result)

    def _reconstruct(result):
        if lib.is_scalar(result):
            return result

        if result.ndim != self.ndim:
            if method == "outer":
                raise NotImplementedError
            return result
        if isinstance(result, (BlockManager, ArrayManager)):
            # we went through BlockManager.apply e.g. np.sqrt
            result = self._constructor_from_mgr(result, axes=result.axes)
        else:
            # we converted an array, lost our axes
            result = self._constructor(
                result, **reconstruct_axes, **reconstruct_kwargs, copy=False
            )
        # TODO: When we support multiple values in __finalize__, this
        # should pass alignable to `__finalize__` instead of self.
        # Then `np.add(a, b)` would consider attrs from both a and b
        # when a and b are NDFrames.
        if len(alignable) == 1:
            result = result.__finalize__(self)
        return result

    if "out" in kwargs:
        # e.g. test_multiindex_get_loc
        result = dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        return reconstruct(result)

    if method == "reduce":
        # e.g. test.series.test_ufunc.test_reduce
        result = dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

    # We still get here with kwargs `axis` for e.g. np.maximum.accumulate
    #  and `dtype` and `keepdims` for np.ptp

    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        # Just give up on preserving types in the complex case.
        # In theory we could preserve them for them.
        # * nout>1 is doable if BlockManager.apply took nout and
        #   returned a Tuple[BlockManager].
        # * len(inputs) > 1 is doable when we know that we have
        #   aligned blocks / dtypes.

        # e.g. my_ufunc, modf, logaddexp, heaviside, subtract, add
        inputs = tuple(np.asarray(x) for x in inputs)
        # Note: we can't use default_array_ufunc here bc reindexing means
        #  that `self` may not be among `inputs`
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        # ufunc(series, ...)
        inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
    else:
        # ufunc(dataframe)
        if method == "__call__" and not kwargs:
            # for np.<ufunc>(..) calls
            # kwargs cannot necessarily be handled block-by-block, so only
            # take this path if there are no kwargs
            mgr = inputs[0]._mgr
            result = mgr.apply(getattr(ufunc, method))
        else:
            # otherwise specific ufunc methods (eg np.<ufunc>.accumulate(..))
            # Those can have an axis keyword and thus can't be called block-by-block
            result = default_array_ufunc(inputs[0], ufunc, method, *inputs, **kwargs)
            # e.g. np.negative (only one reached), with "where" and "out" in kwargs

    result = reconstruct(result)
    return result


def _standardize_out_kwarg(**kwargs) -> dict:
    """
    If kwargs contain "out1" and "out2", replace that with a tuple "out"

    np.divmod, np.modf, np.frexp can have either `out=(out1, out2)` or
    `out1=out1, out2=out2)`
    """
    if "out" not in kwargs and "out1" in kwargs and "out2" in kwargs:
        out1 = kwargs.pop("out1")
        out2 = kwargs.pop("out2")
        out = (out1, out2)
        kwargs["out"] = out
    return kwargs


def dispatch_ufunc_with_out(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    If we have an `out` keyword, then call the ufunc without `out` and then
    set the result into the given `out`.
    """

    # Note: we assume _standardize_out_kwarg has already been called.
    out = kwargs.pop("out")
    where = kwargs.pop("where", None)

    result = getattr(ufunc, method)(*inputs, **kwargs)

    if result is NotImplemented:
        return NotImplemented

    if isinstance(result, tuple):
        # i.e. np.divmod, np.modf, np.frexp
        if not isinstance(out, tuple) or len(out) != len(result):
            raise NotImplementedError

        for arr, res in zip(out, result):
            _assign_where(arr, res, where)

        return out

    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            raise NotImplementedError

    _assign_where(out, result, where)
    return out


def _assign_where(out, result, where) -> None:
    """
    Set a ufunc result into 'out', masking with a 'where' argument if necessary.
    """
    if where is None:
        # no 'where' arg passed to ufunc
        out[:] = result
    else:
        np.putmask(out, where, result)


def default_array_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Fallback to the behavior we would get if we did not define __array_ufunc__.

    Notes
    -----
    We are assuming that `self` is among `inputs`.
    """
    if not any(x is self for x in inputs):
        raise NotImplementedError

    new_inputs = [x if x is not self else np.asarray(x) for x in inputs]

    return getattr(ufunc, method)(*new_inputs, **kwargs)


def dispatch_reduction_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Dispatch ufunc reductions to self's reduction methods.
    """
    assert method == "reduce"

    if len(inputs) != 1 or inputs[0] is not self:
        return NotImplemented

    if ufunc.__name__ not in REDUCTION_ALIASES:
        return NotImplemented

    method_name = REDUCTION_ALIASES[ufunc.__name__]

    # NB: we are assuming that min/max represent minimum/maximum methods,
    #  which would not be accurate for e.g. Timestamp.min
    if not hasattr(self, method_name):
        return NotImplemented

    if self.ndim > 1:
        if isinstance(self, ABCNDFrame):
            # TODO: test cases where this doesn't hold, i.e. 2D DTA/TDA
            kwargs["numeric_only"] = False

        if "axis" not in kwargs:
            # For DataFrame reductions we don't want the default axis=0
            # Note: np.min is not a ufunc, but uses array_function_dispatch,
            #  so calls DataFrame.min (without ever getting here) with the np.min
            #  default of axis=None, which DataFrame.min catches and changes to axis=0.
            # np.minimum.reduce(df) gets here bc axis is not in kwargs,
            #  so we set axis=0 to match the behaviorof np.minimum.reduce(df.values)
            kwargs["axis"] = 0

    # By default, numpy's reductions do not skip NaNs, so we have to
    #  pass skipna=False
    return getattr(self, method_name)(skipna=False, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\intfunc.py ===
"""
The routines here were removed from numbers.py, power.py,
digits.py and factor_.py so they could be imported into core
without raising circular import errors.

Although the name 'intfunc' was chosen to represent functions that
work with integers, it can also be thought of as containing
internal/core functions that are needed by the classes of the core.
"""

import math
import sys
from functools import lru_cache

from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import (gcd as number_gcd, lcm as number_lcm, sqrt,
                                 iroot, bit_scan1, gcdext)
from sympy.utilities.misc import as_int, filldedent


def num_digits(n, base=10):
    """Return the number of digits needed to express n in give base.

    Examples
    ========

    >>> from sympy.core.intfunc import num_digits
    >>> num_digits(10)
    2
    >>> num_digits(10, 2)  # 1010 -> 4 digits
    4
    >>> num_digits(-100, 16)  # -64 -> 2 digits
    2


    Parameters
    ==========

    n: integer
        The number whose digits are counted.

    b: integer
        The base in which digits are computed.

    See Also
    ========
    sympy.ntheory.digits.digits, sympy.ntheory.digits.count_digits
    """
    if base < 0:
        raise ValueError('base must be int greater than 1')
    if not n:
        return 1
    e, t = integer_log(abs(n), base)
    return 1 + e


def integer_log(n, b):
    r"""
    Returns ``(e, bool)`` where e is the largest nonnegative integer
    such that :math:`|n| \geq |b^e|` and ``bool`` is True if $n = b^e$.

    Examples
    ========

    >>> from sympy import integer_log
    >>> integer_log(125, 5)
    (3, True)
    >>> integer_log(17, 9)
    (1, False)

    If the base is positive and the number negative the
    return value will always be the same except for 2:

    >>> integer_log(-4, 2)
    (2, False)
    >>> integer_log(-16, 4)
    (0, False)

    When the base is negative, the returned value
    will only be True if the parity of the exponent is
    correct for the sign of the base:

    >>> integer_log(4, -2)
    (2, True)
    >>> integer_log(8, -2)
    (3, False)
    >>> integer_log(-8, -2)
    (3, True)
    >>> integer_log(-4, -2)
    (2, False)

    See Also
    ========
    integer_nthroot
    sympy.ntheory.primetest.is_square
    sympy.ntheory.factor_.multiplicity
    sympy.ntheory.factor_.perfect_power
    """
    n = as_int(n)
    b = as_int(b)

    if b < 0:
        e, t = integer_log(abs(n), -b)
        # (-2)**3 == -8
        # (-2)**2 = 4
        t = t and e % 2 == (n < 0)
        return e, t
    if b <= 1:
        raise ValueError('base must be 2 or more')
    if n < 0:
        if b != 2:
            return 0, False
        e, t = integer_log(-n, b)
        return e, False
    if n == 0:
        raise ValueError('n cannot be 0')

    if n < b:
        return 0, n == 1
    if b == 2:
        e = n.bit_length() - 1
        return e, trailing(n) == e
    t = trailing(b)
    if 2**t == b:
        e = int(n.bit_length() - 1)//t
        n_ = 1 << (t*e)
        return e, n_ == n

    d = math.floor(math.log10(n) / math.log10(b))
    n_ = b ** d
    while n_ <= n:  # this will iterate 0, 1 or 2 times
        d += 1
        n_ *= b
    return d - (n_ > n), (n_ == n or n_//b == n)


def trailing(n):
    """Count the number of trailing zero digits in the binary
    representation of n, i.e. determine the largest power of 2
    that divides n.

    Examples
    ========

    >>> from sympy import trailing
    >>> trailing(128)
    7
    >>> trailing(63)
    0

    See Also
    ========
    sympy.ntheory.factor_.multiplicity

    """
    if not n:
        return 0
    return bit_scan1(int(n))


@lru_cache(1024)
def igcd(*args):
    """Computes nonnegative integer greatest common divisor.

    Explanation
    ===========

    The algorithm is based on the well known Euclid's algorithm [1]_. To
    improve speed, ``igcd()`` has its own caching mechanism.
    If you do not need the cache mechanism, using ``sympy.external.gmpy.gcd``.

    Examples
    ========

    >>> from sympy import igcd
    >>> igcd(2, 4)
    2
    >>> igcd(5, 10, 15)
    5

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm

    """
    if len(args) < 2:
        raise TypeError("igcd() takes at least 2 arguments (%s given)" % len(args))
    return int(number_gcd(*map(as_int, args)))


igcd2 = math.gcd


def igcd_lehmer(a, b):
    r"""Computes greatest common divisor of two integers.

    Explanation
    ===========

    Euclid's algorithm for the computation of the greatest
    common divisor ``gcd(a, b)``  of two (positive) integers
    $a$ and $b$ is based on the division identity
    $$ a = q \times b + r$$,
    where the quotient  $q$  and the remainder  $r$  are integers
    and  $0 \le r < b$. Then each common divisor of  $a$  and  $b$
    divides  $r$, and it follows that  ``gcd(a, b) == gcd(b, r)``.
    The algorithm works by constructing the sequence
    r0, r1, r2, ..., where  r0 = a, r1 = b,  and each  rn
    is the remainder from the division of the two preceding
    elements.

    In Python, ``q = a // b``  and  ``r = a % b``  are obtained by the
    floor division and the remainder operations, respectively.
    These are the most expensive arithmetic operations, especially
    for large  a  and  b.

    Lehmer's algorithm [1]_ is based on the observation that the quotients
    ``qn = r(n-1) // rn``  are in general small integers even
    when  a  and  b  are very large. Hence the quotients can be
    usually determined from a relatively small number of most
    significant bits.

    The efficiency of the algorithm is further enhanced by not
    computing each long remainder in Euclid's sequence. The remainders
    are linear combinations of  a  and  b  with integer coefficients
    derived from the quotients. The coefficients can be computed
    as far as the quotients can be determined from the chosen
    most significant parts of  a  and  b. Only then a new pair of
    consecutive remainders is computed and the algorithm starts
    anew with this pair.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lehmer%27s_GCD_algorithm

    """
    a, b = abs(as_int(a)), abs(as_int(b))
    if a < b:
        a, b = b, a

    # The algorithm works by using one or two digit division
    # whenever possible. The outer loop will replace the
    # pair (a, b) with a pair of shorter consecutive elements
    # of the Euclidean gcd sequence until a and b
    # fit into two Python (long) int digits.
    nbits = 2 * sys.int_info.bits_per_digit

    while a.bit_length() > nbits and b != 0:
        # Quotients are mostly small integers that can
        # be determined from most significant bits.
        n = a.bit_length() - nbits
        x, y = int(a >> n), int(b >> n)  # most significant bits

        # Elements of the Euclidean gcd sequence are linear
        # combinations of a and b with integer coefficients.
        # Compute the coefficients of consecutive pairs
        #     a' = A*a + B*b, b' = C*a + D*b
        # using small integer arithmetic as far as possible.
        A, B, C, D = 1, 0, 0, 1  # initial values

        while True:
            # The coefficients alternate in sign while looping.
            # The inner loop combines two steps to keep track
            # of the signs.

            # At this point we have
            #   A > 0, B <= 0, C <= 0, D > 0,
            #   x' = x + B <= x < x" = x + A,
            #   y' = y + C <= y < y" = y + D,
            # and
            #   x'*N <= a' < x"*N, y'*N <= b' < y"*N,
            # where N = 2**n.

            # Now, if y' > 0, and x"//y' and x'//y" agree,
            # then their common value is equal to  q = a'//b'.
            # In addition,
            #   x'%y" = x' - q*y" < x" - q*y' = x"%y',
            # and
            #   (x'%y")*N < a'%b' < (x"%y')*N.

            # On the other hand, we also have  x//y == q,
            # and therefore
            #   x'%y" = x + B - q*(y + D) = x%y + B',
            #   x"%y' = x + A - q*(y + C) = x%y + A',
            # where
            #    B' = B - q*D < 0, A' = A - q*C > 0.

            if y + C <= 0:
                break
            q = (x + A) // (y + C)

            # Now  x'//y" <= q, and equality holds if
            #   x' - q*y" = (x - q*y) + (B - q*D) >= 0.
            # This is a minor optimization to avoid division.
            x_qy, B_qD = x - q * y, B - q * D
            if x_qy + B_qD < 0:
                break

            # Next step in the Euclidean sequence.
            x, y = y, x_qy
            A, B, C, D = C, D, A - q * C, B_qD

            # At this point the signs of the coefficients
            # change and their roles are interchanged.
            #   A <= 0, B > 0, C > 0, D < 0,
            #   x' = x + A <= x < x" = x + B,
            #   y' = y + D < y < y" = y + C.

            if y + D <= 0:
                break
            q = (x + B) // (y + D)
            x_qy, A_qC = x - q * y, A - q * C
            if x_qy + A_qC < 0:
                break

            x, y = y, x_qy
            A, B, C, D = C, D, A_qC, B - q * D
            # Now the conditions on top of the loop
            # are again satisfied.
            #   A > 0, B < 0, C < 0, D > 0.

        if B == 0:
            # This can only happen when y == 0 in the beginning
            # and the inner loop does nothing.
            # Long division is forced.
            a, b = b, a % b
            continue

        # Compute new long arguments using the coefficients.
        a, b = A * a + B * b, C * a + D * b

    # Small divisors. Finish with the standard algorithm.
    while b:
        a, b = b, a % b

    return a


def ilcm(*args):
    """Computes integer least common multiple.

    Examples
    ========

    >>> from sympy import ilcm
    >>> ilcm(5, 10)
    10
    >>> ilcm(7, 3)
    21
    >>> ilcm(5, 10, 15)
    30

    """
    if len(args) < 2:
        raise TypeError("ilcm() takes at least 2 arguments (%s given)" % len(args))
    return int(number_lcm(*map(as_int, args)))


def igcdex(a, b):
    """Returns x, y, g such that g = x*a + y*b = gcd(a, b).

    Examples
    ========

    >>> from sympy.core.intfunc import igcdex
    >>> igcdex(2, 3)
    (-1, 1, 1)
    >>> igcdex(10, 12)
    (-1, 1, 2)

    >>> x, y, g = igcdex(100, 2004)
    >>> x, y, g
    (-20, 1, 4)
    >>> x*100 + y*2004
    4

    """
    g, x, y = gcdext(int(a), int(b))
    return x, y, g


def mod_inverse(a, m):
    r"""
    Return the number $c$ such that, $a \times c = 1 \pmod{m}$
    where $c$ has the same sign as $m$. If no such value exists,
    a ValueError is raised.

    Examples
    ========

    >>> from sympy import mod_inverse, S

    Suppose we wish to find multiplicative inverse $x$ of
    3 modulo 11. This is the same as finding $x$ such
    that $3x = 1 \pmod{11}$. One value of x that satisfies
    this congruence is 4. Because $3 \times 4 = 12$ and $12 = 1 \pmod{11}$.
    This is the value returned by ``mod_inverse``:

    >>> mod_inverse(3, 11)
    4
    >>> mod_inverse(-3, 11)
    7

    When there is a common factor between the numerators of
    `a` and `m` the inverse does not exist:

    >>> mod_inverse(2, 4)
    Traceback (most recent call last):
    ...
    ValueError: inverse of 2 mod 4 does not exist

    >>> mod_inverse(S(2)/7, S(5)/2)
    7/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    .. [2] https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    """
    c = None
    try:
        a, m = as_int(a), as_int(m)
        if m != 1 and m != -1:
            x, _, g = igcdex(a, m)
            if g == 1:
                c = x % m
    except ValueError:
        a, m = sympify(a), sympify(m)
        if not (a.is_number and m.is_number):
            raise TypeError(
                filldedent(
                    """
                Expected numbers for arguments; symbolic `mod_inverse`
                is not implemented
                but symbolic expressions can be handled with the
                similar function,
                sympy.polys.polytools.invert"""
                )
            )
        big = m > 1
        if big not in (S.true, S.false):
            raise ValueError("m > 1 did not evaluate; try to simplify %s" % m)
        elif big:
            c = 1 / a
    if c is None:
        raise ValueError("inverse of %s (mod %s) does not exist" % (a, m))
    return c


def isqrt(n):
    r""" Return the largest integer less than or equal to `\sqrt{n}`.

    Parameters
    ==========

    n : non-negative integer

    Returns
    =======

    int : `\left\lfloor\sqrt{n}\right\rfloor`

    Raises
    ======

    ValueError
        If n is negative.
    TypeError
        If n is of a type that cannot be compared to ``int``.
        Therefore, a TypeError is raised for ``str``, but not for ``float``.

    Examples
    ========

    >>> from sympy.core.intfunc import isqrt
    >>> isqrt(0)
    0
    >>> isqrt(9)
    3
    >>> isqrt(10)
    3
    >>> isqrt("30")
    Traceback (most recent call last):
        ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    >>> from sympy.core.numbers import Rational
    >>> isqrt(Rational(-1, 2))
    Traceback (most recent call last):
        ...
    ValueError: n must be nonnegative

    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    return int(sqrt(int(n)))


def integer_nthroot(y, n):
    """
    Return a tuple containing x = floor(y**(1/n))
    and a boolean indicating whether the result is exact (that is,
    whether x**n == y).

    Examples
    ========

    >>> from sympy import integer_nthroot
    >>> integer_nthroot(16, 2)
    (4, True)
    >>> integer_nthroot(26, 2)
    (5, False)

    To simply determine if a number is a perfect square, the is_square
    function should be used:

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(26)
    False

    See Also
    ========
    sympy.ntheory.primetest.is_square
    integer_log
    """
    x, b = iroot(as_int(y), as_int(n))
    return int(x), b

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\arrayop.py ===
import itertools
from collections.abc import Iterable

from sympy.core._print_helpers import Printable
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.core.sympify import _sympify

from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray


def _arrayfy(a):
    from sympy.matrices import MatrixBase

    if isinstance(a, NDimArray):
        return a
    if isinstance(a, (MatrixBase, list, tuple, Tuple)):
        return ImmutableDenseNDimArray(a)
    return a


def tensorproduct(*args):
    """
    Tensor product among scalars or array-like objects.

    The equivalent operator for array expressions is ``ArrayTensorProduct``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.tensor.array import tensorproduct, Array
    >>> from sympy.abc import x, y, z, t
    >>> A = Array([[1, 2], [3, 4]])
    >>> B = Array([x, y])
    >>> tensorproduct(A, B)
    [[[x, y], [2*x, 2*y]], [[3*x, 3*y], [4*x, 4*y]]]
    >>> tensorproduct(A, x)
    [[x, 2*x], [3*x, 4*x]]
    >>> tensorproduct(A, B, B)
    [[[[x**2, x*y], [x*y, y**2]], [[2*x**2, 2*x*y], [2*x*y, 2*y**2]]], [[[3*x**2, 3*x*y], [3*x*y, 3*y**2]], [[4*x**2, 4*x*y], [4*x*y, 4*y**2]]]]

    Applying this function on two matrices will result in a rank 4 array.

    >>> from sympy import Matrix, eye
    >>> m = Matrix([[x, y], [z, t]])
    >>> p = tensorproduct(eye(3), m)
    >>> p
    [[[[x, y], [z, t]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[x, y], [z, t]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[x, y], [z, t]]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayTensorProduct

    """
    from sympy.tensor.array import SparseNDimArray, ImmutableSparseNDimArray

    if len(args) == 0:
        return S.One
    if len(args) == 1:
        return _arrayfy(args[0])
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if any(isinstance(arg, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)) for arg in args):
        return ArrayTensorProduct(*args)
    if len(args) > 2:
        return tensorproduct(tensorproduct(args[0], args[1]), *args[2:])

    # length of args is 2:
    a, b = map(_arrayfy, args)

    if not isinstance(a, NDimArray) or not isinstance(b, NDimArray):
        return a*b

    if isinstance(a, SparseNDimArray) and isinstance(b, SparseNDimArray):
        lp = len(b)
        new_array = {k1*lp + k2: v1*v2 for k1, v1 in a._sparse_array.items() for k2, v2 in b._sparse_array.items()}
        return ImmutableSparseNDimArray(new_array, a.shape + b.shape)

    product_list = [i*j for i in Flatten(a) for j in Flatten(b)]
    return ImmutableDenseNDimArray(product_list, a.shape + b.shape)


def _util_contraction_diagonal(array, *contraction_or_diagonal_axes):
    array = _arrayfy(array)

    # Verify contraction_axes:
    taken_dims = set()
    for axes_group in contraction_or_diagonal_axes:
        if not isinstance(axes_group, Iterable):
            raise ValueError("collections of contraction/diagonal axes expected")

        dim = array.shape[axes_group[0]]

        for d in axes_group:
            if d in taken_dims:
                raise ValueError("dimension specified more than once")
            if dim != array.shape[d]:
                raise ValueError("cannot contract or diagonalize between axes of different dimension")
            taken_dims.add(d)

    rank = array.rank()

    remaining_shape = [dim for i, dim in enumerate(array.shape) if i not in taken_dims]
    cum_shape = [0]*rank
    _cumul = 1
    for i in range(rank):
        cum_shape[rank - i - 1] = _cumul
        _cumul *= int(array.shape[rank - i - 1])

    # DEFINITION: by absolute position it is meant the position along the one
    # dimensional array containing all the tensor components.

    # Possible future work on this module: move computation of absolute
    # positions to a class method.

    # Determine absolute positions of the uncontracted indices:
    remaining_indices = [[cum_shape[i]*j for j in range(array.shape[i])]
                         for i in range(rank) if i not in taken_dims]

    # Determine absolute positions of the contracted indices:
    summed_deltas = []
    for axes_group in contraction_or_diagonal_axes:
        lidx = []
        for js in range(array.shape[axes_group[0]]):
            lidx.append(sum(cum_shape[ig] * js for ig in axes_group))
        summed_deltas.append(lidx)

    return array, remaining_indices, remaining_shape, summed_deltas


def tensorcontraction(array, *contraction_axes):
    """
    Contraction of an array-like object on the specified axes.

    The equivalent operator for array expressions is ``ArrayContraction``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy import Array, tensorcontraction
    >>> from sympy import Matrix, eye
    >>> tensorcontraction(eye(3), (0, 1))
    3
    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensorcontraction(A, (0, 2))
    [21, 30]

    Matrix multiplication may be emulated with a proper combination of
    ``tensorcontraction`` and ``tensorproduct``

    >>> from sympy import tensorproduct
    >>> from sympy.abc import a,b,c,d,e,f,g,h
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> m2 = Matrix([[e, f], [g, h]])
    >>> p = tensorproduct(m1, m2)
    >>> p
    [[[[a*e, a*f], [a*g, a*h]], [[b*e, b*f], [b*g, b*h]]], [[[c*e, c*f], [c*g, c*h]], [[d*e, d*f], [d*g, d*h]]]]
    >>> tensorcontraction(p, (1, 2))
    [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    >>> m1*m2
    Matrix([
    [a*e + b*g, a*f + b*h],
    [c*e + d*g, c*f + d*h]])

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayContraction

    """
    from sympy.tensor.array.expressions.array_expressions import _array_contraction
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_contraction(array, *contraction_axes)

    array, remaining_indices, remaining_shape, summed_deltas = _util_contraction_diagonal(array, *contraction_axes)

    # Compute the contracted array:
    #
    # 1. external for loops on all uncontracted indices.
    #    Uncontracted indices are determined by the combinatorial product of
    #    the absolute positions of the remaining indices.
    # 2. internal loop on all contracted indices.
    #    It sums the values of the absolute contracted index and the absolute
    #    uncontracted index for the external loop.
    contracted_array = []
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = S.Zero
        for sum_to_index in itertools.product(*summed_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum += array[idx]

        contracted_array.append(isum)

    if len(remaining_indices) == 0:
        assert len(contracted_array) == 1
        return contracted_array[0]

    return type(array)(contracted_array, remaining_shape)


def tensordiagonal(array, *diagonal_axes):
    """
    Diagonalization of an array-like object on the specified axes.

    This is equivalent to multiplying the expression by Kronecker deltas
    uniting the axes.

    The diagonal indices are put at the end of the axes.

    The equivalent operator for array expressions is ``ArrayDiagonal``, which
    can be used to keep the expression unevaluated.

    Examples
    ========

    ``tensordiagonal`` acting on a 2-dimensional array by axes 0 and 1 is
    equivalent to the diagonal of the matrix:

    >>> from sympy import Array, tensordiagonal
    >>> from sympy import Matrix, eye
    >>> tensordiagonal(eye(3), (0, 1))
    [1, 1, 1]

    >>> from sympy.abc import a,b,c,d
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> tensordiagonal(m1, [0, 1])
    [a, d]

    In case of higher dimensional arrays, the diagonalized out dimensions
    are appended removed and appended as a single dimension at the end:

    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensordiagonal(A, (0, 2))
    [[0, 7, 14], [3, 10, 17]]
    >>> from sympy import permutedims
    >>> tensordiagonal(A, (0, 2)) == permutedims(Array([A[0, :, 0], A[1, :, 1], A[2, :, 2]]), [1, 0])
    True

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayDiagonal

    """
    if any(len(i) <= 1 for i in diagonal_axes):
        raise ValueError("need at least two axes to diagonalize")

    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, _array_diagonal
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_diagonal(array, *diagonal_axes)

    ArrayDiagonal._validate(array, *diagonal_axes)

    array, remaining_indices, remaining_shape, diagonal_deltas = _util_contraction_diagonal(array, *diagonal_axes)

    # Compute the diagonalized array:
    #
    # 1. external for loops on all undiagonalized indices.
    #    Undiagonalized indices are determined by the combinatorial product of
    #    the absolute positions of the remaining indices.
    # 2. internal loop on all diagonal indices.
    #    It appends the values of the absolute diagonalized index and the absolute
    #    undiagonalized index for the external loop.
    diagonalized_array = []
    diagonal_shape = [len(i) for i in diagonal_deltas]
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = []
        for sum_to_index in itertools.product(*diagonal_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum.append(array[idx])

        isum = type(array)(isum).reshape(*diagonal_shape)
        diagonalized_array.append(isum)

    return type(array)(diagonalized_array, remaining_shape + diagonal_shape)


def derive_by_array(expr, dx):
    r"""
    Derivative by arrays. Supports both arrays and scalars.

    The equivalent operator for array expressions is ``array_derive``.

    Explanation
    ===========

    Given the array `A_{i_1, \ldots, i_N}` and the array `X_{j_1, \ldots, j_M}`
    this function will return a new array `B` defined by

    `B_{j_1,\ldots,j_M,i_1,\ldots,i_N} := \frac{\partial A_{i_1,\ldots,i_N}}{\partial X_{j_1,\ldots,j_M}}`

    Examples
    ========

    >>> from sympy import derive_by_array
    >>> from sympy.abc import x, y, z, t
    >>> from sympy import cos
    >>> derive_by_array(cos(x*t), x)
    -t*sin(t*x)
    >>> derive_by_array(cos(x*t), [x, y, z, t])
    [-t*sin(t*x), 0, 0, -x*sin(t*x)]
    >>> derive_by_array([x, y**2*z], [[x, y], [z, t]])
    [[[1, 0], [0, 2*y*z]], [[0, y**2], [0, 0]]]

    """
    from sympy.matrices import MatrixBase
    from sympy.tensor.array import SparseNDimArray
    array_types = (Iterable, MatrixBase, NDimArray)

    if isinstance(dx, array_types):
        dx = ImmutableDenseNDimArray(dx)
        for i in dx:
            if not i._diff_wrt:
                raise ValueError("cannot derive by this array")

    if isinstance(expr, array_types):
        if isinstance(expr, NDimArray):
            expr = expr.as_immutable()
        else:
            expr = ImmutableDenseNDimArray(expr)

        if isinstance(dx, array_types):
            if isinstance(expr, SparseNDimArray):
                lp = len(expr)
                new_array = {k + i*lp: v
                             for i, x in enumerate(Flatten(dx))
                             for k, v in expr.diff(x)._sparse_array.items()}
            else:
                new_array = [[y.diff(x) for y in Flatten(expr)] for x in Flatten(dx)]
            return type(expr)(new_array, dx.shape + expr.shape)
        else:
            return expr.diff(dx)
    else:
        expr = _sympify(expr)
        if isinstance(dx, array_types):
            return ImmutableDenseNDimArray([expr.diff(i) for i in Flatten(dx)], dx.shape)
        else:
            dx = _sympify(dx)
            return diff(expr, dx)


def permutedims(expr, perm=None, index_order_old=None, index_order_new=None):
    """
    Permutes the indices of an array.

    Parameter specifies the permutation of the indices.

    The equivalent operator for array expressions is ``PermuteDims``, which can
    be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.abc import x, y, z, t
    >>> from sympy import sin
    >>> from sympy import Array, permutedims
    >>> a = Array([[x, y, z], [t, sin(x), 0]])
    >>> a
    [[x, y, z], [t, sin(x), 0]]
    >>> permutedims(a, (1, 0))
    [[x, t], [y, sin(x)], [z, 0]]

    If the array is of second order, ``transpose`` can be used:

    >>> from sympy import transpose
    >>> transpose(a)
    [[x, t], [y, sin(x)], [z, 0]]

    Examples on higher dimensions:

    >>> b = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> permutedims(b, (2, 1, 0))
    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    >>> permutedims(b, (1, 2, 0))
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    An alternative way to specify the same permutations as in the previous
    lines involves passing the *old* and *new* indices, either as a list or as
    a string:

    >>> permutedims(b, index_order_old="cba", index_order_new="abc")
    [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
    >>> permutedims(b, index_order_old="cab", index_order_new="abc")
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    ``Permutation`` objects are also allowed:

    >>> from sympy.combinatorics import Permutation
    >>> permutedims(b, Permutation([1, 2, 0]))
    [[[1, 5], [2, 6]], [[3, 7], [4, 8]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.PermuteDims

    """
    from sympy.tensor.array import SparseNDimArray

    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _permute_dims
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.tensor.array.expressions import PermuteDims
    from sympy.tensor.array.expressions.array_expressions import get_rank
    perm = PermuteDims._get_permutation_from_arguments(perm, index_order_old, index_order_new, get_rank(expr))
    if isinstance(expr, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _permute_dims(expr, perm)

    if not isinstance(expr, NDimArray):
        expr = ImmutableDenseNDimArray(expr)

    from sympy.combinatorics import Permutation
    if not isinstance(perm, Permutation):
        perm = Permutation(list(perm))

    if perm.size != expr.rank():
        raise ValueError("wrong permutation size")

    # Get the inverse permutation:
    iperm = ~perm
    new_shape = perm(expr.shape)

    if isinstance(expr, SparseNDimArray):
        return type(expr)({tuple(perm(expr._get_tuple_index(k))): v
                           for k, v in expr._sparse_array.items()}, new_shape)

    indices_span = perm([range(i) for i in expr.shape])

    new_array = [None]*len(expr)
    for i, idx in enumerate(itertools.product(*indices_span)):
        t = iperm(idx)
        new_array[i] = expr[t]

    return type(expr)(new_array, new_shape)


class Flatten(Printable):
    """
    Flatten an iterable object to a list in a lazy-evaluation way.

    Notes
    =====

    This class is an iterator with which the memory cost can be economised.
    Optimisation has been considered to ameliorate the performance for some
    specific data types like DenseNDimArray and SparseNDimArray.

    Examples
    ========

    >>> from sympy.tensor.array.arrayop import Flatten
    >>> from sympy.tensor.array import Array
    >>> A = Array(range(6)).reshape(2, 3)
    >>> Flatten(A)
    Flatten([[0, 1, 2], [3, 4, 5]])
    >>> [i for i in Flatten(A)]
    [0, 1, 2, 3, 4, 5]
    """
    def __init__(self, iterable):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import NDimArray

        if not isinstance(iterable, (Iterable, MatrixBase)):
            raise NotImplementedError("Data type not yet supported")

        if isinstance(iterable, list):
            iterable = NDimArray(iterable)

        self._iter = iterable
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        from sympy.matrices.matrixbase import MatrixBase

        if len(self._iter) > self._idx:
            if isinstance(self._iter, DenseNDimArray):
                result = self._iter._array[self._idx]

            elif isinstance(self._iter, SparseNDimArray):
                if self._idx in self._iter._sparse_array:
                    result = self._iter._sparse_array[self._idx]
                else:
                    result = 0

            elif isinstance(self._iter, MatrixBase):
                result = self._iter[self._idx]

            elif hasattr(self._iter, '__next__'):
                result = next(self._iter)

            else:
                result = self._iter[self._idx]

        else:
            raise StopIteration

        self._idx += 1
        return result

    def next(self):
        return self.__next__()

    def _sympystr(self, printer):
        return type(self).__name__ + '(' + printer._print(self._iter) + ')'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\benchmark_all.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import argparse
import datetime
import json
import logging
import os
import subprocess

import librosa
import torch
from benchmark_helper import setup_logger
from metrics import BenchmarkRecord
from transformers import WhisperConfig, WhisperProcessor

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--audio-path",
        type=str,
        required=True,
        help="Path to folder of audio files for E2E evaluation",
    )

    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language of audio file",
    )

    parser.add_argument(
        "-t",
        "--task",
        default=None,
        choices=["transcribe", "translate"],
        help="Task to complete",
    )

    parser.add_argument(
        "-w",
        "--warmup-runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--hf-pt-eager",
        default=False,
        action="store_true",
        help="Benchmark in PyTorch without `torch.compile`",
    )

    parser.add_argument(
        "--hf-pt-compile",
        default=False,
        action="store_true",
        help="Benchmark in PyTorch with `torch.compile`",
    )

    parser.add_argument(
        "--hf-ort-dir-path",
        type=str,
        help="Path to folder containing ONNX models for Optimum + ORT benchmarking",
    )

    parser.add_argument(
        "--ort-model-path",
        type=str,
        help="Path to ONNX model for ORT benchmarking",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name in Hugging Face (e.g. openai/whisper-large-v2)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["int8", "fp16", "fp32"],
        help="Precision to run model",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda", "rocm"],
        help="Device to benchmark models",
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Print detailed logs",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Number of mins to attempt the benchmark before moving on",
    )

    parser.add_argument(
        "--log-folder",
        type=str,
        default=None,
        help="Path to folder to save logs and results",
    )

    parser.add_argument("--tune", default=False, action="store_true")

    args = parser.parse_args()

    setattr(args, "model_size", args.model_name.split("/")[-1].replace(".", "-"))  # noqa: B010
    log_folder_name = f"./{args.model_size}-{args.precision}"
    if not args.log_folder:
        args.log_folder = log_folder_name
    os.makedirs(args.log_folder, exist_ok=True)

    # Convert timeout value to secs
    args.timeout *= 60

    return args


def process_log_file(device_id, log_file, base_results):
    entries = []

    # Detect steps in speech pipeline
    step = None
    load_audio_pattern = "Load audio: "
    feat_ext_pattern = "Feature extraction: "
    pytorch_pattern = "Evaluating PyTorch..."
    onnxruntime_pattern = "Evaluating ONNX Runtime..."

    load_audio_latency_s, load_audio_throughput_s = None, None
    feat_ext_latency_s, feat_ext_throughput_s = None, None
    token_length, latency_s, per_token_latency_s, per_token_latency_ms = None, None, None, None
    throughput, memory = None, None

    # Detect metrics
    latency_pattern = "Latency: "
    throughput_pattern = "Throughput: "
    token_length_pattern = "Generated token length: "
    memory_pattern = "peak="

    with open(log_file) as f:
        for input_line in f:
            line = input_line.replace("\n", "")

            # Get step in speech recognition pipeline
            if load_audio_pattern in line:
                step = "load-audio"
            elif feat_ext_pattern in line:
                step = "feature-extraction"
            elif pytorch_pattern in line or onnxruntime_pattern in line:
                step = "process"

            # Check metrics
            if latency_pattern in line:
                latency_s = float(line[len(latency_pattern) : line.rfind(" ")])
            elif throughput_pattern in line:
                throughput = float(line[len(throughput_pattern) : line.rfind(" ")])
                if step == "load-audio":
                    load_audio_latency_s, load_audio_throughput_s = latency_s, throughput
                    step = None
                if step == "feature-extraction":
                    feat_ext_latency_s, feat_ext_throughput_s = latency_s, throughput
                    step = None
            elif token_length_pattern in line:
                token_length = int(line[len(token_length_pattern) : line.rfind(" ")])
                per_token_latency_s = latency_s / token_length
                per_token_latency_ms = per_token_latency_s * 1000
            elif memory_pattern in line:
                if "CPU" in line:
                    # Example format for log entry:
                    # CPU memory usage: before=1000.0 MB, peak=2000.0 MB
                    memory = float(line[line.rfind("=") + 1 : line.rfind(" MB")]) / 1000
                else:
                    # Example format for log entry:
                    # GPU memory usage: before=[{'device_id': 0, 'name': 'Tesla V100-PCIE-16GB', 'max_used_MB': 1638.875}, {'device_id': 1, 'name': 'Tesla V100-PCIE-16GB', 'max_used_MB': 236.875},  peak=[{'device_id': 0, 'name': 'Tesla V100-PCIE-16GB', 'max_used_MB': 1780.875}, {'device_id': 1, 'name': 'Tesla V100-PCIE-16GB', 'max_used_MB': 236.875}]
                    peak = line[line.find(memory_pattern) + len(memory_pattern) :].replace("'", '"')
                    usage = json.loads(peak)[device_id]["max_used_MB"]
                    memory = float(usage) / 1000

                # Calculate real-time factor (RTF):
                # RTF = total latency / audio duration
                total_latency = (
                    (load_audio_latency_s if load_audio_latency_s else 0)
                    + (feat_ext_latency_s if feat_ext_latency_s else 0)
                    + (latency_s if latency_s else 0)
                )
                audio_duration = base_results[-1]
                rtf = (total_latency / audio_duration) if audio_duration else -1
                logger.info(f"Total latency: {total_latency} s")
                logger.info(f"Audio duration: {audio_duration} s")
                logger.info(f"Real-time factor: {rtf}")

                # Append log entry to list of entries
                entry = base_results + [  # noqa: RUF005
                    token_length,
                    load_audio_latency_s,
                    load_audio_throughput_s,
                    feat_ext_latency_s if feat_ext_latency_s else -1,
                    feat_ext_throughput_s if feat_ext_throughput_s else -1,
                    latency_s,
                    per_token_latency_ms,
                    throughput,
                    memory,
                    rtf,
                ]
                entries.append(entry)

    return entries


def save_results(results, filename):
    import pandas as pd

    df = pd.DataFrame(
        results,
        columns=[
            "Warmup Runs",
            "Measured Runs",
            "Model Name",
            "Engine",
            "Precision",
            "Device",
            "Audio File",
            "Duration (s)",
            "Token Length",
            "Load Audio Latency (s)",
            "Load Audio Throughput (qps)",
            "Feature Extractor Latency (s)",
            "Feature Extractor Throughput (qps)",
            "Latency (s)",
            "Per Token Latency (ms/token)",
            "Throughput (qps)",
            "Memory (GB)",
            "Real Time Factor (RTF)",
        ],
    )

    # Set column types
    df["Warmup Runs"] = df["Warmup Runs"].astype("int")
    df["Measured Runs"] = df["Measured Runs"].astype("int")
    df["Duration (s)"] = df["Duration (s)"].astype("float")
    df["Token Length"] = df["Token Length"].astype("int")
    df["Load Audio Latency (s)"] = df["Load Audio Latency (s)"].astype("float")
    df["Load Audio Throughput (qps)"] = df["Load Audio Throughput (qps)"].astype("float")
    df["Feature Extractor Latency (s)"] = df["Feature Extractor Latency (s)"].astype("float")
    df["Feature Extractor Throughput (qps)"] = df["Feature Extractor Throughput (qps)"].astype("float")
    df["Latency (s)"] = df["Latency (s)"].astype("float")
    df["Per Token Latency (ms/token)"] = df["Per Token Latency (ms/token)"].astype("float")
    df["Throughput (qps)"] = df["Throughput (qps)"].astype("float")
    df["Memory (GB)"] = df["Memory (GB)"].astype("float")
    df["Real Time Factor (RTF)"] = df["Real Time Factor (RTF)"].astype("float")

    # get package name and version
    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [f"{i.key}=={i.version}" for i in installed_packages if i.key in ["onnxruntime", "onnxruntime-gpu"]]
    )
    ort_pkg_name = ""
    ort_pkg_version = ""
    if installed_packages_list:
        ort_pkg_name = installed_packages_list[0].split("==")[0]
        ort_pkg_version = installed_packages_list[0].split("==")[1]

    # Save results to csv with standard format
    records = []
    for _, row in df.iterrows():
        if row["Engine"] == "onnxruntime":
            record = BenchmarkRecord(
                row["Model Name"], row["Precision"], row["Engine"], row["Device"], ort_pkg_name, ort_pkg_version
            )
        else:
            record = BenchmarkRecord(
                row["Model Name"], row["Precision"], row["Engine"], row["Device"], torch.__name__, torch.__version__
            )
        record.config.customized["audio_file"] = row["Audio File"]
        record.config.warmup_runs = row["Warmup Runs"]
        record.config.measured_runs = row["Measured Runs"]

        record.metrics.customized["duration"] = row["Duration (s)"]
        record.metrics.customized["token_length"] = row["Token Length"]
        record.metrics.customized["load_audio_latency"] = row["Load Audio Latency (s)"]
        record.metrics.customized["load_audio_throughput"] = row["Load Audio Throughput (qps)"]
        record.metrics.customized["feature_extractor_latency_s"] = row["Feature Extractor Latency (s)"]
        record.metrics.customized["feature_extractor_throughput_qps"] = row["Feature Extractor Throughput (qps)"]
        record.metrics.customized["per_token_latency_ms"] = row["Per Token Latency (ms/token)"]
        record.metrics.customized["rtf"] = row["Real Time Factor (RTF)"]

        record.metrics.latency_ms_mean = row["Latency (s)"] * 1000
        record.metrics.throughput_qps = row["Throughput (qps)"]
        record.metrics.max_memory_usage_GB = row["Memory (GB)"]

        records.append(record)

    BenchmarkRecord.save_as_csv(filename, records)
    BenchmarkRecord.save_as_json(filename.replace(".csv", ".json"), records)
    logger.info(f"Results saved in {filename}!")


def benchmark(args, benchmark_cmd, engine, audio_file, duration):
    log_filename = f"{engine}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.log"
    log_path = os.path.join(args.log_folder, log_filename)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(benchmark_cmd, stdout=log_file, stderr=log_file)
        try:
            process.wait(args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    # Create entries for csv
    logger.info("Gathering data from log files...")
    base_results = [
        args.warmup_runs,
        args.num_runs,
        args.model_name,
        engine,
        args.precision,
        args.device,
        audio_file,
        duration,
    ]
    results = process_log_file(args.device_id, log_path, base_results)

    return results


def main():
    args = get_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True

    config = WhisperConfig.from_pretrained(args.model_name)
    processor = WhisperProcessor.from_pretrained(args.model_name)

    # Calculate forced decoder input ids
    hf_forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    ort_forced_decoder_ids = [config.decoder_start_token_id] + [token_id[1] for token_id in hf_forced_decoder_ids]
    hf_decoder_input_ids_cmd = (
        ["--decoder-input-ids", str(hf_forced_decoder_ids)] if args.language and args.task else []
    )
    ort_decoder_input_ids_cmd = (
        ["--decoder-input-ids", str(ort_forced_decoder_ids)] if args.language and args.task else []
    )
    ort_tune_cmd = ["--tune"] if args.tune else []

    all_results = []
    for audio_file in os.listdir(args.audio_path):
        audio_path = os.path.join(args.audio_path, audio_file)
        try:
            duration = librosa.get_duration(path=audio_path)
        except Exception as e:
            duration = -1
            logger.warning(f"An error occurred while trying to calculate the audio duration: {e}", exc_info=True)
            logger.warning(
                f"If you get an error that says:\n\tsoundfile.LibsndfileError: Error opening '{audio_file}': File contains data in an unknown format.\nyou may not have installed `ffmpeg` in addition to installing `librosa`."
            )
        logger.info(f"Testing {audio_path}...")

        # Benchmark PyTorch without torch.compile
        if args.hf_pt_eager:
            benchmark_cmd = [  # noqa: RUF005
                "python",
                "-m",
                "models.whisper.benchmark",
                "--audio-path",
                audio_path,
                "--benchmark-type",
                "hf-pt-eager",
                "--model-name",
                args.model_name,
                "--precision",
                args.precision,
                "--device",
                args.device,
                "--device-id",
                str(args.device_id),
                "--warmup-runs",
                str(args.warmup_runs),
                "--num-runs",
                str(args.num_runs),
                "--log-folder",
                args.log_folder,
            ] + hf_decoder_input_ids_cmd
            logger.info("Benchmark PyTorch without torch.compile")
            results = benchmark(args, benchmark_cmd, "pytorch-eager", audio_file, duration)
            all_results.extend(results)

        # Benchmark PyTorch with torch.compile
        if args.hf_pt_compile:
            benchmark_cmd = [  # noqa: RUF005
                "python",
                "-m",
                "models.whisper.benchmark",
                "--audio-path",
                audio_path,
                "--benchmark-type",
                "hf-pt-compile",
                "--model-name",
                args.model_name,
                "--precision",
                args.precision,
                "--device",
                args.device,
                "--device-id",
                str(args.device_id),
                "--warmup-runs",
                str(args.warmup_runs),
                "--num-runs",
                str(args.num_runs),
                "--log-folder",
                args.log_folder,
            ] + hf_decoder_input_ids_cmd
            logger.info("Benchmark PyTorch with torch.compile")
            results = benchmark(args, benchmark_cmd, "pytorch-compile", audio_file, duration)
            all_results.extend(results)

        # Benchmark Optimum + ONNX Runtime
        if args.hf_ort_dir_path:
            benchmark_cmd = [  # noqa: RUF005
                "python",
                "-m",
                "models.whisper.benchmark",
                "--audio-path",
                audio_path,
                "--benchmark-type",
                "hf-ort",
                "--hf-ort-dir-path",
                args.hf_ort_dir_path,
                "--model-name",
                args.model_name,
                "--precision",
                args.precision,
                "--device",
                args.device,
                "--device-id",
                str(args.device_id),
                "--warmup-runs",
                str(args.warmup_runs),
                "--num-runs",
                str(args.num_runs),
                "--log-folder",
                args.log_folder,
            ] + hf_decoder_input_ids_cmd
            logger.info("Benchmark Optimum + ONNX Runtime")
            results = benchmark(args, benchmark_cmd, "optimum-ort", audio_file, duration)
            all_results.extend(results)

        # Benchmark ONNX Runtime
        if args.ort_model_path:
            benchmark_cmd = (
                [  # noqa: RUF005
                    "python",
                    "-m",
                    "models.whisper.benchmark",
                    "--audio-path",
                    audio_path,
                    "--benchmark-type",
                    "ort",
                    "--ort-model-path",
                    args.ort_model_path,
                    "--model-name",
                    args.model_name,
                    "--precision",
                    args.precision,
                    "--device",
                    args.device,
                    "--device-id",
                    str(args.device_id),
                    "--warmup-runs",
                    str(args.warmup_runs),
                    "--num-runs",
                    str(args.num_runs),
                    "--log-folder",
                    args.log_folder,
                ]
                + ort_decoder_input_ids_cmd
                + ort_tune_cmd
            )
            logger.info("Benchmark ONNX Runtime")
            results = benchmark(args, benchmark_cmd, "onnxruntime", audio_file, duration)
            all_results.extend(results)

    csv_file = f"{args.model_size}-{args.precision}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.csv"
    save_results(all_results, os.path.join(args.log_folder, csv_file))


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_typing.py ===
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import (
    date,
    datetime,
    timedelta,
    tzinfo,
)
from os import PathLike
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Type as type_t,
    TypeVar,
    Union,
    overload,
)

import numpy as np

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    import numpy.typing as npt

    from pandas._libs import (
        NaTType,
        Period,
        Timedelta,
        Timestamp,
    )
    from pandas._libs.tslibs import BaseOffset

    from pandas.core.dtypes.dtypes import ExtensionDtype

    from pandas import Interval
    from pandas.arrays import (
        DatetimeArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.base import ExtensionArray
    from pandas.core.frame import DataFrame
    from pandas.core.generic import NDFrame
    from pandas.core.groupby.generic import (
        DataFrameGroupBy,
        GroupBy,
        SeriesGroupBy,
    )
    from pandas.core.indexes.base import Index
    from pandas.core.internals import (
        ArrayManager,
        BlockManager,
        SingleArrayManager,
        SingleBlockManager,
    )
    from pandas.core.resample import Resampler
    from pandas.core.series import Series
    from pandas.core.window.rolling import BaseWindow

    from pandas.io.formats.format import EngFormatter
    from pandas.tseries.holiday import AbstractHolidayCalendar

    ScalarLike_co = Union[
        int,
        float,
        complex,
        str,
        bytes,
        np.generic,
    ]

    # numpy compatible types
    NumpyValueArrayLike = Union[ScalarLike_co, npt.ArrayLike]
    # Name "npt._ArrayLikeInt_co" is not defined  [name-defined]
    NumpySorter = Optional[npt._ArrayLikeInt_co]  # type: ignore[name-defined]

    from typing import SupportsIndex

    if sys.version_info >= (3, 10):
        from typing import TypeGuard  # pyright: ignore[reportUnusedImport]
    else:
        from typing_extensions import TypeGuard  # pyright: ignore[reportUnusedImport]

    if sys.version_info >= (3, 11):
        from typing import Self  # pyright: ignore[reportUnusedImport]
    else:
        from typing_extensions import Self  # pyright: ignore[reportUnusedImport]
else:
    npt: Any = None
    Self: Any = None
    TypeGuard: Any = None

HashableT = TypeVar("HashableT", bound=Hashable)
MutableMappingT = TypeVar("MutableMappingT", bound=MutableMapping)

# array-like

ArrayLike = Union["ExtensionArray", np.ndarray]
AnyArrayLike = Union[ArrayLike, "Index", "Series"]
TimeArrayLike = Union["DatetimeArray", "TimedeltaArray"]

# list-like

# from https://github.com/hauntsaninja/useful_types
# includes Sequence-like objects but excludes str and bytes
_T_co = TypeVar("_T_co", covariant=True)


class SequenceNotStr(Protocol[_T_co]):
    @overload
    def __getitem__(self, index: SupportsIndex, /) -> _T_co:
        ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T_co]:
        ...

    def __contains__(self, value: object, /) -> bool:
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[_T_co]:
        ...

    def index(self, value: Any, /, start: int = 0, stop: int = ...) -> int:
        ...

    def count(self, value: Any, /) -> int:
        ...

    def __reversed__(self) -> Iterator[_T_co]:
        ...


ListLike = Union[AnyArrayLike, SequenceNotStr, range]

# scalars

PythonScalar = Union[str, float, bool]
DatetimeLikeScalar = Union["Period", "Timestamp", "Timedelta"]
PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
Scalar = Union[PythonScalar, PandasScalar, np.datetime64, np.timedelta64, date]
IntStrT = TypeVar("IntStrT", bound=Union[int, str])


# timestamp and timedelta convertible types

TimestampConvertibleTypes = Union[
    "Timestamp", date, np.datetime64, np.int64, float, str
]
TimestampNonexistent = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"], timedelta
]
TimedeltaConvertibleTypes = Union[
    "Timedelta", timedelta, np.timedelta64, np.int64, float, str
]
Timezone = Union[str, tzinfo]

ToTimestampHow = Literal["s", "e", "start", "end"]

# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound="NDFrame")

NumpyIndexT = TypeVar("NumpyIndexT", np.ndarray, "Index")

AxisInt = int
Axis = Union[AxisInt, Literal["index", "columns", "rows"]]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Level = Hashable
Shape = tuple[int, ...]
Suffixes = tuple[Optional[str], Optional[str]]
Ordered = Optional[bool]
JSONSerializable = Optional[Union[PythonScalar, list, dict]]
Frequency = Union[str, "BaseOffset"]
Axes = ListLike

RandomState = Union[
    int,
    np.ndarray,
    np.random.Generator,
    np.random.BitGenerator,
    np.random.RandomState,
]

# dtypes
NpDtype = Union[str, np.dtype, type_t[Union[str, complex, bool, object]]]
Dtype = Union["ExtensionDtype", NpDtype]
AstypeArg = Union["ExtensionDtype", "npt.DTypeLike"]
# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg = Union[Dtype, dict[Hashable, Dtype]]
DtypeObj = Union[np.dtype, "ExtensionDtype"]

# converters
ConvertersArg = dict[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg = Union[
    bool, list[Hashable], list[list[Hashable]], dict[Hashable, list[Hashable]]
]

# For functions like rename that convert one label to another
Renamer = Union[Mapping[Any, Hashable], Callable[[Any], Hashable]]

# to maintain type information across generic functions and parametrization
T = TypeVar("T")

# used in decorators to preserve the signature of the function it decorates
# see https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

# types of vectorized key functions for DataFrame::sort_values and
# DataFrame::sort_index, among others
ValueKeyFunc = Optional[Callable[["Series"], Union["Series", AnyArrayLike]]]
IndexKeyFunc = Optional[Callable[["Index"], Union["Index", AnyArrayLike]]]

# types of `func` kwarg for DataFrame.aggregate and Series.aggregate
AggFuncTypeBase = Union[Callable, str]
AggFuncTypeDict = MutableMapping[
    Hashable, Union[AggFuncTypeBase, list[AggFuncTypeBase]]
]
AggFuncType = Union[
    AggFuncTypeBase,
    list[AggFuncTypeBase],
    AggFuncTypeDict,
]
AggObjType = Union[
    "Series",
    "DataFrame",
    "GroupBy",
    "SeriesGroupBy",
    "DataFrameGroupBy",
    "BaseWindow",
    "Resampler",
]

PythonFuncType = Callable[[Any], Any]

# filenames and file-like-objects
AnyStr_co = TypeVar("AnyStr_co", str, bytes, covariant=True)
AnyStr_contra = TypeVar("AnyStr_contra", str, bytes, contravariant=True)


class BaseBuffer(Protocol):
    @property
    def mode(self) -> str:
        # for _get_filepath_or_buffer
        ...

    def seek(self, __offset: int, __whence: int = ...) -> int:
        # with one argument: gzip.GzipFile, bz2.BZ2File
        # with two arguments: zip.ZipFile, read_sas
        ...

    def seekable(self) -> bool:
        # for bz2.BZ2File
        ...

    def tell(self) -> int:
        # for zip.ZipFile, read_stata, to_stata
        ...


class ReadBuffer(BaseBuffer, Protocol[AnyStr_co]):
    def read(self, __n: int = ...) -> AnyStr_co:
        # for BytesIOWrapper, gzip.GzipFile, bz2.BZ2File
        ...


class WriteBuffer(BaseBuffer, Protocol[AnyStr_contra]):
    def write(self, __b: AnyStr_contra) -> Any:
        # for gzip.GzipFile, bz2.BZ2File
        ...

    def flush(self) -> Any:
        # for gzip.GzipFile, bz2.BZ2File
        ...


class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> bytes:
        ...


class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: int | None = ...) -> int:
        ...


class ReadCsvBuffer(ReadBuffer[AnyStr_co], Protocol):
    def __iter__(self) -> Iterator[AnyStr_co]:
        # for engine=python
        ...

    def fileno(self) -> int:
        # for _MMapWrapper
        ...

    def readline(self) -> AnyStr_co:
        # for engine=python
        ...

    @property
    def closed(self) -> bool:
        # for enine=pyarrow
        ...


FilePath = Union[str, "PathLike[str]"]

# for arbitrary kwargs passed during reading/writing files
StorageOptions = Optional[dict[str, Any]]


# compression keywords and compression
CompressionDict = dict[str, Any]
CompressionOptions = Optional[
    Union[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"], CompressionDict]
]

# types in DataFrameFormatter
FormattersType = Union[
    list[Callable], tuple[Callable, ...], Mapping[Union[str, int], Callable]
]
ColspaceType = Mapping[Hashable, Union[str, int]]
FloatFormatType = Union[str, Callable, "EngFormatter"]
ColspaceArgType = Union[
    str, int, Sequence[Union[str, int]], Mapping[Hashable, Union[str, int]]
]

# Arguments for fillna()
FillnaOptions = Literal["backfill", "bfill", "ffill", "pad"]
InterpolateOptions = Literal[
    "linear",
    "time",
    "index",
    "values",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]

# internals
Manager = Union[
    "ArrayManager", "SingleArrayManager", "BlockManager", "SingleBlockManager"
]
SingleManager = Union["SingleArrayManager", "SingleBlockManager"]
Manager2D = Union["ArrayManager", "BlockManager"]

# indexing
# PositionalIndexer -> valid 1D positional indexer, e.g. can pass
# to ndarray.__getitem__
# ScalarIndexer is for a single value as the index
# SequenceIndexer is for list like or slices (but not tuples)
# PositionalIndexerTuple is extends the PositionalIndexer for 2D arrays
# These are used in various __getitem__ overloads
# TODO(typing#684): add Ellipsis, see
# https://github.com/python/typing/issues/684#issuecomment-548203158
# https://bugs.python.org/issue41810
# Using List[int] here rather than Sequence[int] to disallow tuples.
ScalarIndexer = Union[int, np.integer]
SequenceIndexer = Union[slice, list[int], np.ndarray]
PositionalIndexer = Union[ScalarIndexer, SequenceIndexer]
PositionalIndexerTuple = tuple[PositionalIndexer, PositionalIndexer]
PositionalIndexer2D = Union[PositionalIndexer, PositionalIndexerTuple]
if TYPE_CHECKING:
    TakeIndexer = Union[Sequence[int], Sequence[np.integer], npt.NDArray[np.integer]]
else:
    TakeIndexer = Any

# Shared by functions such as drop and astype
IgnoreRaise = Literal["ignore", "raise"]

# Windowing rank methods
WindowingRankType = Literal["average", "min", "max"]

# read_csv engines
CSVEngine = Literal["c", "python", "pyarrow", "python-fwf"]

# read_json engines
JSONEngine = Literal["ujson", "pyarrow"]

# read_xml parsers
XMLParsers = Literal["lxml", "etree"]

# read_html flavors
HTMLFlavors = Literal["lxml", "html5lib", "bs4"]

# Interval closed type
IntervalLeftRight = Literal["left", "right"]
IntervalClosedType = Union[IntervalLeftRight, Literal["both", "neither"]]

# datetime and NaTType
DatetimeNaTType = Union[datetime, "NaTType"]
DateTimeErrorChoices = Union[IgnoreRaise, Literal["coerce"]]

# sort_index
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition = Literal["first", "last"]

# Arguments for nsmalles and n_largest
NsmallestNlargestKeep = Literal["first", "last", "all"]

# quantile interpolation
QuantileInterpolation = Literal["linear", "lower", "higher", "midpoint", "nearest"]

# plotting
PlottingOrientation = Literal["horizontal", "vertical"]

# dropna
AnyAll = Literal["any", "all"]

# merge
MergeHow = Literal["left", "right", "inner", "outer", "cross"]
MergeValidate = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]

# join
JoinHow = Literal["left", "right", "inner", "outer"]
JoinValidate = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]

# reindex
ReindexMethod = Union[FillnaOptions, Literal["nearest"]]

MatplotlibColor = Union[str, Sequence[float]]
TimeGrouperOrigin = Union[
    "Timestamp", Literal["epoch", "start", "start_day", "end", "end_day"]
]
TimeAmbiguous = Union[Literal["infer", "NaT", "raise"], "npt.NDArray[np.bool_]"]
TimeNonexistent = Union[
    Literal["shift_forward", "shift_backward", "NaT", "raise"], timedelta
]
DropKeep = Literal["first", "last", False]
CorrelationMethod = Union[
    Literal["pearson", "kendall", "spearman"], Callable[[np.ndarray, np.ndarray], float]
]
AlignJoin = Literal["outer", "inner", "left", "right"]
DtypeBackend = Literal["pyarrow", "numpy_nullable"]

TimeUnit = Literal["s", "ms", "us", "ns"]
OpenFileErrors = Literal[
    "strict",
    "ignore",
    "replace",
    "surrogateescape",
    "xmlcharrefreplace",
    "backslashreplace",
    "namereplace",
]

# update
UpdateJoin = Literal["left"]

# applymap
NaAction = Literal["ignore"]

# from_dict
FromDictOrient = Literal["columns", "index", "tight"]

# to_gbc
ToGbqIfexist = Literal["fail", "replace", "append"]

# to_stata
ToStataByteorder = Literal[">", "<", "little", "big"]

# ExcelWriter
ExcelWriterIfSheetExists = Literal["error", "new", "replace", "overlay"]

# Offsets
OffsetCalendar = Union[np.busdaycalendar, "AbstractHolidayCalendar"]

# read_csv: usecols
UsecolsArgType = Union[
    SequenceNotStr[Hashable],
    range,
    AnyArrayLike,
    Callable[[HashableT], bool],
    None,
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\indexed_db.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: IndexedDB (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
from . import storage


@dataclass
class DatabaseWithObjectStores:
    '''
    Database with an array of object stores.
    '''
    #: Database name.
    name: str

    #: Database version (type is not 'integer', as the standard
    #: requires the version number to be 'unsigned long long')
    version: float

    #: Object stores in this database.
    object_stores: typing.List[ObjectStore]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['version'] = self.version
        json['objectStores'] = [i.to_json() for i in self.object_stores]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            version=float(json['version']),
            object_stores=[ObjectStore.from_json(i) for i in json['objectStores']],
        )


@dataclass
class ObjectStore:
    '''
    Object store.
    '''
    #: Object store name.
    name: str

    #: Object store key path.
    key_path: KeyPath

    #: If true, object store has auto increment flag set.
    auto_increment: bool

    #: Indexes in this object store.
    indexes: typing.List[ObjectStoreIndex]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['autoIncrement'] = self.auto_increment
        json['indexes'] = [i.to_json() for i in self.indexes]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            key_path=KeyPath.from_json(json['keyPath']),
            auto_increment=bool(json['autoIncrement']),
            indexes=[ObjectStoreIndex.from_json(i) for i in json['indexes']],
        )


@dataclass
class ObjectStoreIndex:
    '''
    Object store index.
    '''
    #: Index name.
    name: str

    #: Index key path.
    key_path: KeyPath

    #: If true, index is unique.
    unique: bool

    #: If true, index allows multiple entries for a key.
    multi_entry: bool

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['unique'] = self.unique
        json['multiEntry'] = self.multi_entry
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            key_path=KeyPath.from_json(json['keyPath']),
            unique=bool(json['unique']),
            multi_entry=bool(json['multiEntry']),
        )


@dataclass
class Key:
    '''
    Key.
    '''
    #: Key type.
    type_: str

    #: Number value.
    number: typing.Optional[float] = None

    #: String value.
    string: typing.Optional[str] = None

    #: Date value.
    date: typing.Optional[float] = None

    #: Array value.
    array: typing.Optional[typing.List[Key]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.number is not None:
            json['number'] = self.number
        if self.string is not None:
            json['string'] = self.string
        if self.date is not None:
            json['date'] = self.date
        if self.array is not None:
            json['array'] = [i.to_json() for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            number=float(json['number']) if 'number' in json else None,
            string=str(json['string']) if 'string' in json else None,
            date=float(json['date']) if 'date' in json else None,
            array=[Key.from_json(i) for i in json['array']] if 'array' in json else None,
        )


@dataclass
class KeyRange:
    '''
    Key range.
    '''
    #: If true lower bound is open.
    lower_open: bool

    #: If true upper bound is open.
    upper_open: bool

    #: Lower bound.
    lower: typing.Optional[Key] = None

    #: Upper bound.
    upper: typing.Optional[Key] = None

    def to_json(self):
        json = dict()
        json['lowerOpen'] = self.lower_open
        json['upperOpen'] = self.upper_open
        if self.lower is not None:
            json['lower'] = self.lower.to_json()
        if self.upper is not None:
            json['upper'] = self.upper.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            lower_open=bool(json['lowerOpen']),
            upper_open=bool(json['upperOpen']),
            lower=Key.from_json(json['lower']) if 'lower' in json else None,
            upper=Key.from_json(json['upper']) if 'upper' in json else None,
        )


@dataclass
class DataEntry:
    '''
    Data entry.
    '''
    #: Key object.
    key: runtime.RemoteObject

    #: Primary key object.
    primary_key: runtime.RemoteObject

    #: Value object.
    value: runtime.RemoteObject

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['primaryKey'] = self.primary_key.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=runtime.RemoteObject.from_json(json['key']),
            primary_key=runtime.RemoteObject.from_json(json['primaryKey']),
            value=runtime.RemoteObject.from_json(json['value']),
        )


@dataclass
class KeyPath:
    '''
    Key path.
    '''
    #: Key path type.
    type_: str

    #: String value.
    string: typing.Optional[str] = None

    #: Array value.
    array: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.string is not None:
            json['string'] = self.string
        if self.array is not None:
            json['array'] = [i for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            string=str(json['string']) if 'string' in json else None,
            array=[str(i) for i in json['array']] if 'array' in json else None,
        )


def clear_object_store(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all entries from an object store.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.clearObjectStore',
        'params': params,
    }
    json = yield cmd_dict


def delete_database(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a database.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.deleteDatabase',
        'params': params,
    }
    json = yield cmd_dict


def delete_object_store_entries(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None,
        key_range: KeyRange = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Delete a range of entries from an object store

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name:
    :param object_store_name:
    :param key_range: Range of entry keys to delete
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.deleteObjectStoreEntries',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables events from backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables events from backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.enable',
    }
    json = yield cmd_dict


def request_data(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None,
        index_name: str = None,
        skip_count: int = None,
        page_size: int = None,
        key_range: typing.Optional[KeyRange] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DataEntry], bool]]:
    '''
    Requests data from object store or index.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :param index_name: Index name, empty string for object store data requests.
    :param skip_count: Number of records to skip.
    :param page_size: Number of records to fetch.
    :param key_range: *(Optional)* Key range.
    :returns: A tuple with the following items:

        0. **objectStoreDataEntries** - Array of object store data entries.
        1. **hasMore** - If true, there are more entries to fetch in the given range.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['indexName'] = index_name
    params['skipCount'] = skip_count
    params['pageSize'] = page_size
    if key_range is not None:
        params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestData',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DataEntry.from_json(i) for i in json['objectStoreDataEntries']],
        bool(json['hasMore'])
    )


def get_metadata(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float]]:
    '''
    Gets metadata of an object store.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :returns: A tuple with the following items:

        0. **entriesCount** - the entries count
        1. **keyGeneratorValue** - the current value of key generator, to become the next inserted key into the object store. Valid if objectStore.autoIncrement is true.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.getMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return (
        float(json['entriesCount']),
        float(json['keyGeneratorValue'])
    )


def request_database(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,DatabaseWithObjectStores]:
    '''
    Requests database with given name in given frame.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :returns: Database with an array of object stores.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestDatabase',
        'params': params,
    }
    json = yield cmd_dict
    return DatabaseWithObjectStores.from_json(json['databaseWithObjectStores'])


def request_database_names(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Requests database names for given security origin.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :returns: Database names for origin.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestDatabaseNames',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['databaseNames']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\indexed_db.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: IndexedDB (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
from . import storage


@dataclass
class DatabaseWithObjectStores:
    '''
    Database with an array of object stores.
    '''
    #: Database name.
    name: str

    #: Database version (type is not 'integer', as the standard
    #: requires the version number to be 'unsigned long long')
    version: float

    #: Object stores in this database.
    object_stores: typing.List[ObjectStore]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['version'] = self.version
        json['objectStores'] = [i.to_json() for i in self.object_stores]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            version=float(json['version']),
            object_stores=[ObjectStore.from_json(i) for i in json['objectStores']],
        )


@dataclass
class ObjectStore:
    '''
    Object store.
    '''
    #: Object store name.
    name: str

    #: Object store key path.
    key_path: KeyPath

    #: If true, object store has auto increment flag set.
    auto_increment: bool

    #: Indexes in this object store.
    indexes: typing.List[ObjectStoreIndex]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['autoIncrement'] = self.auto_increment
        json['indexes'] = [i.to_json() for i in self.indexes]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            key_path=KeyPath.from_json(json['keyPath']),
            auto_increment=bool(json['autoIncrement']),
            indexes=[ObjectStoreIndex.from_json(i) for i in json['indexes']],
        )


@dataclass
class ObjectStoreIndex:
    '''
    Object store index.
    '''
    #: Index name.
    name: str

    #: Index key path.
    key_path: KeyPath

    #: If true, index is unique.
    unique: bool

    #: If true, index allows multiple entries for a key.
    multi_entry: bool

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['unique'] = self.unique
        json['multiEntry'] = self.multi_entry
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            key_path=KeyPath.from_json(json['keyPath']),
            unique=bool(json['unique']),
            multi_entry=bool(json['multiEntry']),
        )


@dataclass
class Key:
    '''
    Key.
    '''
    #: Key type.
    type_: str

    #: Number value.
    number: typing.Optional[float] = None

    #: String value.
    string: typing.Optional[str] = None

    #: Date value.
    date: typing.Optional[float] = None

    #: Array value.
    array: typing.Optional[typing.List[Key]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.number is not None:
            json['number'] = self.number
        if self.string is not None:
            json['string'] = self.string
        if self.date is not None:
            json['date'] = self.date
        if self.array is not None:
            json['array'] = [i.to_json() for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            number=float(json['number']) if 'number' in json else None,
            string=str(json['string']) if 'string' in json else None,
            date=float(json['date']) if 'date' in json else None,
            array=[Key.from_json(i) for i in json['array']] if 'array' in json else None,
        )


@dataclass
class KeyRange:
    '''
    Key range.
    '''
    #: If true lower bound is open.
    lower_open: bool

    #: If true upper bound is open.
    upper_open: bool

    #: Lower bound.
    lower: typing.Optional[Key] = None

    #: Upper bound.
    upper: typing.Optional[Key] = None

    def to_json(self):
        json = dict()
        json['lowerOpen'] = self.lower_open
        json['upperOpen'] = self.upper_open
        if self.lower is not None:
            json['lower'] = self.lower.to_json()
        if self.upper is not None:
            json['upper'] = self.upper.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            lower_open=bool(json['lowerOpen']),
            upper_open=bool(json['upperOpen']),
            lower=Key.from_json(json['lower']) if 'lower' in json else None,
            upper=Key.from_json(json['upper']) if 'upper' in json else None,
        )


@dataclass
class DataEntry:
    '''
    Data entry.
    '''
    #: Key object.
    key: runtime.RemoteObject

    #: Primary key object.
    primary_key: runtime.RemoteObject

    #: Value object.
    value: runtime.RemoteObject

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['primaryKey'] = self.primary_key.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=runtime.RemoteObject.from_json(json['key']),
            primary_key=runtime.RemoteObject.from_json(json['primaryKey']),
            value=runtime.RemoteObject.from_json(json['value']),
        )


@dataclass
class KeyPath:
    '''
    Key path.
    '''
    #: Key path type.
    type_: str

    #: String value.
    string: typing.Optional[str] = None

    #: Array value.
    array: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.string is not None:
            json['string'] = self.string
        if self.array is not None:
            json['array'] = [i for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            string=str(json['string']) if 'string' in json else None,
            array=[str(i) for i in json['array']] if 'array' in json else None,
        )


def clear_object_store(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all entries from an object store.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.clearObjectStore',
        'params': params,
    }
    json = yield cmd_dict


def delete_database(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a database.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.deleteDatabase',
        'params': params,
    }
    json = yield cmd_dict


def delete_object_store_entries(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None,
        key_range: KeyRange = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Delete a range of entries from an object store

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name:
    :param object_store_name:
    :param key_range: Range of entry keys to delete
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.deleteObjectStoreEntries',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables events from backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables events from backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.enable',
    }
    json = yield cmd_dict


def request_data(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None,
        index_name: str = None,
        skip_count: int = None,
        page_size: int = None,
        key_range: typing.Optional[KeyRange] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DataEntry], bool]]:
    '''
    Requests data from object store or index.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :param index_name: Index name, empty string for object store data requests.
    :param skip_count: Number of records to skip.
    :param page_size: Number of records to fetch.
    :param key_range: *(Optional)* Key range.
    :returns: A tuple with the following items:

        0. **objectStoreDataEntries** - Array of object store data entries.
        1. **hasMore** - If true, there are more entries to fetch in the given range.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['indexName'] = index_name
    params['skipCount'] = skip_count
    params['pageSize'] = page_size
    if key_range is not None:
        params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestData',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DataEntry.from_json(i) for i in json['objectStoreDataEntries']],
        bool(json['hasMore'])
    )


def get_metadata(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float]]:
    '''
    Gets metadata of an object store.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :returns: A tuple with the following items:

        0. **entriesCount** - the entries count
        1. **keyGeneratorValue** - the current value of key generator, to become the next inserted key into the object store. Valid if objectStore.autoIncrement is true.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.getMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return (
        float(json['entriesCount']),
        float(json['keyGeneratorValue'])
    )


def request_database(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,DatabaseWithObjectStores]:
    '''
    Requests database with given name in given frame.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :returns: Database with an array of object stores.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestDatabase',
        'params': params,
    }
    json = yield cmd_dict
    return DatabaseWithObjectStores.from_json(json['databaseWithObjectStores'])


def request_database_names(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Requests database names for given security origin.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :returns: Database names for origin.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestDatabaseNames',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['databaseNames']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\indexed_db.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: IndexedDB (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
from . import storage


@dataclass
class DatabaseWithObjectStores:
    '''
    Database with an array of object stores.
    '''
    #: Database name.
    name: str

    #: Database version (type is not 'integer', as the standard
    #: requires the version number to be 'unsigned long long')
    version: float

    #: Object stores in this database.
    object_stores: typing.List[ObjectStore]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['version'] = self.version
        json['objectStores'] = [i.to_json() for i in self.object_stores]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            version=float(json['version']),
            object_stores=[ObjectStore.from_json(i) for i in json['objectStores']],
        )


@dataclass
class ObjectStore:
    '''
    Object store.
    '''
    #: Object store name.
    name: str

    #: Object store key path.
    key_path: KeyPath

    #: If true, object store has auto increment flag set.
    auto_increment: bool

    #: Indexes in this object store.
    indexes: typing.List[ObjectStoreIndex]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['autoIncrement'] = self.auto_increment
        json['indexes'] = [i.to_json() for i in self.indexes]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            key_path=KeyPath.from_json(json['keyPath']),
            auto_increment=bool(json['autoIncrement']),
            indexes=[ObjectStoreIndex.from_json(i) for i in json['indexes']],
        )


@dataclass
class ObjectStoreIndex:
    '''
    Object store index.
    '''
    #: Index name.
    name: str

    #: Index key path.
    key_path: KeyPath

    #: If true, index is unique.
    unique: bool

    #: If true, index allows multiple entries for a key.
    multi_entry: bool

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['unique'] = self.unique
        json['multiEntry'] = self.multi_entry
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            key_path=KeyPath.from_json(json['keyPath']),
            unique=bool(json['unique']),
            multi_entry=bool(json['multiEntry']),
        )


@dataclass
class Key:
    '''
    Key.
    '''
    #: Key type.
    type_: str

    #: Number value.
    number: typing.Optional[float] = None

    #: String value.
    string: typing.Optional[str] = None

    #: Date value.
    date: typing.Optional[float] = None

    #: Array value.
    array: typing.Optional[typing.List[Key]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.number is not None:
            json['number'] = self.number
        if self.string is not None:
            json['string'] = self.string
        if self.date is not None:
            json['date'] = self.date
        if self.array is not None:
            json['array'] = [i.to_json() for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            number=float(json['number']) if 'number' in json else None,
            string=str(json['string']) if 'string' in json else None,
            date=float(json['date']) if 'date' in json else None,
            array=[Key.from_json(i) for i in json['array']] if 'array' in json else None,
        )


@dataclass
class KeyRange:
    '''
    Key range.
    '''
    #: If true lower bound is open.
    lower_open: bool

    #: If true upper bound is open.
    upper_open: bool

    #: Lower bound.
    lower: typing.Optional[Key] = None

    #: Upper bound.
    upper: typing.Optional[Key] = None

    def to_json(self):
        json = dict()
        json['lowerOpen'] = self.lower_open
        json['upperOpen'] = self.upper_open
        if self.lower is not None:
            json['lower'] = self.lower.to_json()
        if self.upper is not None:
            json['upper'] = self.upper.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            lower_open=bool(json['lowerOpen']),
            upper_open=bool(json['upperOpen']),
            lower=Key.from_json(json['lower']) if 'lower' in json else None,
            upper=Key.from_json(json['upper']) if 'upper' in json else None,
        )


@dataclass
class DataEntry:
    '''
    Data entry.
    '''
    #: Key object.
    key: runtime.RemoteObject

    #: Primary key object.
    primary_key: runtime.RemoteObject

    #: Value object.
    value: runtime.RemoteObject

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['primaryKey'] = self.primary_key.to_json()
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=runtime.RemoteObject.from_json(json['key']),
            primary_key=runtime.RemoteObject.from_json(json['primaryKey']),
            value=runtime.RemoteObject.from_json(json['value']),
        )


@dataclass
class KeyPath:
    '''
    Key path.
    '''
    #: Key path type.
    type_: str

    #: String value.
    string: typing.Optional[str] = None

    #: Array value.
    array: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.string is not None:
            json['string'] = self.string
        if self.array is not None:
            json['array'] = [i for i in self.array]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            string=str(json['string']) if 'string' in json else None,
            array=[str(i) for i in json['array']] if 'array' in json else None,
        )


def clear_object_store(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all entries from an object store.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.clearObjectStore',
        'params': params,
    }
    json = yield cmd_dict


def delete_database(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes a database.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.deleteDatabase',
        'params': params,
    }
    json = yield cmd_dict


def delete_object_store_entries(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None,
        key_range: KeyRange = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Delete a range of entries from an object store

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name:
    :param object_store_name:
    :param key_range: Range of entry keys to delete
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.deleteObjectStoreEntries',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables events from backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables events from backend.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.enable',
    }
    json = yield cmd_dict


def request_data(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None,
        index_name: str = None,
        skip_count: int = None,
        page_size: int = None,
        key_range: typing.Optional[KeyRange] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[typing.List[DataEntry], bool]]:
    '''
    Requests data from object store or index.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :param index_name: Index name, empty string for object store data requests.
    :param skip_count: Number of records to skip.
    :param page_size: Number of records to fetch.
    :param key_range: *(Optional)* Key range.
    :returns: A tuple with the following items:

        0. **objectStoreDataEntries** - Array of object store data entries.
        1. **hasMore** - If true, there are more entries to fetch in the given range.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    params['indexName'] = index_name
    params['skipCount'] = skip_count
    params['pageSize'] = page_size
    if key_range is not None:
        params['keyRange'] = key_range.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestData',
        'params': params,
    }
    json = yield cmd_dict
    return (
        [DataEntry.from_json(i) for i in json['objectStoreDataEntries']],
        bool(json['hasMore'])
    )


def get_metadata(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None,
        object_store_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float]]:
    '''
    Gets metadata of an object store.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :param object_store_name: Object store name.
    :returns: A tuple with the following items:

        0. **entriesCount** - the entries count
        1. **keyGeneratorValue** - the current value of key generator, to become the next inserted key into the object store. Valid if objectStore.autoIncrement is true.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    params['objectStoreName'] = object_store_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.getMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return (
        float(json['entriesCount']),
        float(json['keyGeneratorValue'])
    )


def request_database(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None,
        database_name: str = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,DatabaseWithObjectStores]:
    '''
    Requests database with given name in given frame.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :param database_name: Database name.
    :returns: Database with an array of object stores.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    params['databaseName'] = database_name
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestDatabase',
        'params': params,
    }
    json = yield cmd_dict
    return DatabaseWithObjectStores.from_json(json['databaseWithObjectStores'])


def request_database_names(
        security_origin: typing.Optional[str] = None,
        storage_key: typing.Optional[str] = None,
        storage_bucket: typing.Optional[storage.StorageBucket] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Requests database names for given security origin.

    :param security_origin: *(Optional)* At least and at most one of securityOrigin, storageKey, or storageBucket must be specified. Security origin.
    :param storage_key: *(Optional)* Storage key.
    :param storage_bucket: *(Optional)* Storage bucket. If not specified, it uses the default bucket.
    :returns: Database names for origin.
    '''
    params: T_JSON_DICT = dict()
    if security_origin is not None:
        params['securityOrigin'] = security_origin
    if storage_key is not None:
        params['storageKey'] = storage_key
    if storage_bucket is not None:
        params['storageBucket'] = storage_bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'IndexedDB.requestDatabaseNames',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['databaseNames']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\inverse.py ===
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.domains import EX

from .exceptions import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero


def _pinv_full_rank(M):
    """Subroutine for full row or column rank matrices.

    For full row rank matrices, inverse of ``A * A.H`` Exists.
    For full column rank matrices, inverse of ``A.H * A`` Exists.

    This routine can apply for both cases by checking the shape
    and have small decision.
    """

    if M.is_zero_matrix:
        return M.H

    if M.rows >= M.cols:
        return M.H.multiply(M).inv().multiply(M.H)
    else:
        return M.H.multiply(M.multiply(M.H).inv())

def _pinv_rank_decomposition(M):
    """Subroutine for rank decomposition

    With rank decompositions, `A` can be decomposed into two full-
    rank matrices, and each matrix can take pseudoinverse
    individually.
    """

    if M.is_zero_matrix:
        return M.H

    B, C = M.rank_decomposition()

    Bp = _pinv_full_rank(B)
    Cp = _pinv_full_rank(C)

    return Cp.multiply(Bp)

def _pinv_diagonalization(M):
    """Subroutine using diagonalization

    This routine can sometimes fail if SymPy's eigenvalue
    computation is not reliable.
    """

    if M.is_zero_matrix:
        return M.H

    A  = M
    AH = M.H

    try:
        if M.rows >= M.cols:
            P, D   = AH.multiply(A).diagonalize(normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)

            return P.multiply(D_pinv).multiply(P.H).multiply(AH)

        else:
            P, D   = A.multiply(AH).diagonalize(
                        normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)

            return AH.multiply(P).multiply(D_pinv).multiply(P.H)

    except MatrixError:
        raise NotImplementedError(
            'pinv for rank-deficient matrices where '
            'diagonalization of A.H*A fails is not supported yet.')

def _pinv(M, method='RD'):
    """Calculate the Moore-Penrose pseudoinverse of the matrix.

    The Moore-Penrose pseudoinverse exists and is unique for any matrix.
    If the matrix is invertible, the pseudoinverse is the same as the
    inverse.

    Parameters
    ==========

    method : String, optional
        Specifies the method for computing the pseudoinverse.

        If ``'RD'``, Rank-Decomposition will be used.

        If ``'ED'``, Diagonalization will be used.

    Examples
    ========

    Computing pseudoinverse by rank decomposition :

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> A.pinv()
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    Computing pseudoinverse by diagonalization :

    >>> B = A.pinv(method='ED')
    >>> B.simplify()
    >>> B
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    See Also
    ========

    inv
    pinv_solve

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse

    """

    # Trivial case: pseudoinverse of all-zero matrix is its transpose.
    if M.is_zero_matrix:
        return M.H

    if method == 'RD':
        return _pinv_rank_decomposition(M)
    elif method == 'ED':
        return _pinv_diagonalization(M)
    else:
        raise ValueError('invalid pinv method %s' % repr(method))


def _verify_invertible(M, iszerofunc=_iszero):
    """Initial check to see if a matrix is invertible. Raises or returns
    determinant for use in _inv_ADJ."""

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    d    = M.det(method='berkowitz')
    zero = d.equals(0)

    if zero is None: # if equals() can't decide, will rref be able to?
        ok   = M.rref(simplify=True)[0]
        zero = any(iszerofunc(ok[j, j]) for j in range(ok.rows))

    if zero:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    return d

def _inv_ADJ(M, iszerofunc=_iszero):
    """Calculates the inverse using the adjugate matrix and a determinant.

    See Also
    ========

    inv
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    d = _verify_invertible(M, iszerofunc=iszerofunc)

    return M.adjugate() / d

def _inv_GE(M, iszerofunc=_iszero):
    """Calculates the inverse using Gaussian elimination.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    from .dense import Matrix

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    big = Matrix.hstack(M.as_mutable(), Matrix.eye(M.rows))
    red = big.rref(iszerofunc=iszerofunc, simplify=True)[0]

    if any(iszerofunc(red[j, j]) for j in range(red.rows)):
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    return M._new(red[:, big.rows:])

def _inv_LU(M, iszerofunc=_iszero):
    """Calculates the inverse using LU decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")
    if M.free_symbols:
        _verify_invertible(M, iszerofunc=iszerofunc)

    return M.LUsolve(M.eye(M.rows), iszerofunc=_iszero)

def _inv_CH(M, iszerofunc=_iszero):
    """Calculates the inverse using cholesky decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_LDL
    """

    _verify_invertible(M, iszerofunc=iszerofunc)

    return M.cholesky_solve(M.eye(M.rows))

def _inv_LDL(M, iszerofunc=_iszero):
    """Calculates the inverse using LDL decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    """

    _verify_invertible(M, iszerofunc=iszerofunc)

    return M.LDLsolve(M.eye(M.rows))

def _inv_QR(M, iszerofunc=_iszero):
    """Calculates the inverse using QR decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """

    _verify_invertible(M, iszerofunc=iszerofunc)

    return M.QRsolve(M.eye(M.rows))

def _try_DM(M, use_EX=False):
    """Try to convert a matrix to a ``DomainMatrix``."""
    dM = M.to_DM()
    K = dM.domain

    # Return DomainMatrix if a domain is found. Only use EX if use_EX=True.
    if not use_EX and K.is_EXRAW:
        return None
    elif K.is_EXRAW:
        return dM.convert_to(EX)
    else:
        return dM


def _use_exact_domain(dom):
    """Check whether to convert to an exact domain."""
    # DomainMatrix can handle RR and CC with partial pivoting. Other inexact
    # domains like RR[a,b,...] can only be handled by converting to an exact
    # domain like QQ[a,b,...]
    if dom.is_RR or dom.is_CC:
        return False
    else:
        return not dom.is_Exact


def _inv_DM(dM, cancel=True):
    """Calculates the inverse using ``DomainMatrix``.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    sympy.polys.matrices.domainmatrix.DomainMatrix.inv
    """
    m, n = dM.shape
    dom = dM.domain

    if m != n:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    # Convert RR[a,b,...] to QQ[a,b,...]
    use_exact = _use_exact_domain(dom)

    if use_exact:
        dom_exact = dom.get_exact()
        dM = dM.convert_to(dom_exact)

    try:
        dMi, den = dM.inv_den()
    except DMNonInvertibleMatrixError:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    if use_exact:
        dMi = dMi.convert_to(dom)
        den = dom.convert_from(den, dom_exact)

    if cancel:
        # Convert to field and cancel with the denominator.
        if not dMi.domain.is_Field:
            dMi = dMi.to_field()
        Mi = (dMi / den).to_Matrix()
    else:
        # Convert to Matrix and divide without cancelling
        Mi = dMi.to_Matrix() / dMi.domain.to_sympy(den)

    return Mi

def _inv_block(M, iszerofunc=_iszero):
    """Calculates the inverse using BLOCKWISE inversion.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    i = M.shape[0]
    if i <= 20 :
        return M.inv(method="LU", iszerofunc=_iszero)
    A = M[:i // 2, :i //2]
    B = M[:i // 2, i // 2:]
    C = M[i // 2:, :i // 2]
    D = M[i // 2:, i // 2:]
    try:
        D_inv = _inv_block(D)
    except NonInvertibleMatrixError:
        return M.inv(method="LU", iszerofunc=_iszero)
    B_D_i = B*D_inv
    BDC = B_D_i*C
    A_n = A - BDC
    try:
        A_n = _inv_block(A_n)
    except NonInvertibleMatrixError:
        return M.inv(method="LU", iszerofunc=_iszero)
    B_n = -A_n*B_D_i
    dc = D_inv*C
    C_n = -dc*A_n
    D_n = D_inv + dc*-B_n
    nn = BlockMatrix([[A_n, B_n], [C_n, D_n]]).as_explicit()
    return nn

def _inv(M, method=None, iszerofunc=_iszero, try_block_diag=False):
    """
    Return the inverse of a matrix using the method indicated. The default
    is DM if a suitable domain is found or otherwise GE for dense matrices
    LDL for sparse matrices.

    Parameters
    ==========

    method : ('DM', 'DMNC', 'GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR')

    iszerofunc : function, optional
        Zero-testing function to use.

    try_block_diag : bool, optional
        If True then will try to form block diagonal matrices using the
        method get_diag_blocks(), invert these individually, and then
        reconstruct the full inverse matrix.

    Examples
    ========

    >>> from sympy import SparseMatrix, Matrix
    >>> A = SparseMatrix([
    ... [ 2, -1,  0],
    ... [-1,  2, -1],
    ... [ 0,  0,  2]])
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv(method='LDL') # use of 'method=' is optional
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A * _
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> A = Matrix(A)
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv('ADJ') == A.inv('GE') == A.inv('LU') == A.inv('CH') == A.inv('LDL') == A.inv('QR')
    True

    Notes
    =====

    According to the ``method`` keyword, it calls the appropriate method:

        DM .... Use DomainMatrix ``inv_den`` method
        DMNC .... Use DomainMatrix ``inv_den`` method without cancellation
        GE .... inverse_GE(); default for dense matrices
        LU .... inverse_LU()
        ADJ ... inverse_ADJ()
        CH ... inverse_CH()
        LDL ... inverse_LDL(); default for sparse matrices
        QR ... inverse_QR()

    Note, the GE and LU methods may require the matrix to be simplified
    before it is inverted in order to properly detect zeros during
    pivoting. In difficult cases a custom zero detection function can
    be provided by setting the ``iszerofunc`` argument to a function that
    should return True if its argument is zero. The ADJ routine computes
    the determinant and uses that to detect singular matrices in addition
    to testing for zeros on the diagonal.

    See Also
    ========

    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL

    Raises
    ======

    ValueError
        If the determinant of the matrix is zero.
    """

    from sympy.matrices import diag, SparseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    if try_block_diag:
        blocks = M.get_diag_blocks()
        r      = []

        for block in blocks:
            r.append(block.inv(method=method, iszerofunc=iszerofunc))

        return diag(*r)

    # Default: Use DomainMatrix if the domain is not EX.
    # If DM is requested explicitly then use it even if the domain is EX.
    if method is None and iszerofunc is _iszero:
        dM = _try_DM(M, use_EX=False)
        if dM is not None:
            method = 'DM'
    elif method in ("DM", "DMNC"):
        dM = _try_DM(M, use_EX=True)

    # A suitable domain was not found, fall back to GE for dense matrices
    # and LDL for sparse matrices.
    if method is None:
        if isinstance(M, SparseMatrix):
            method = 'LDL'
        else:
            method = 'GE'

    if method == "DM":
        rv = _inv_DM(dM)
    elif method == "DMNC":
        rv = _inv_DM(dM, cancel=False)
    elif method == "GE":
        rv = M.inverse_GE(iszerofunc=iszerofunc)
    elif method == "LU":
        rv = M.inverse_LU(iszerofunc=iszerofunc)
    elif method == "ADJ":
        rv = M.inverse_ADJ(iszerofunc=iszerofunc)
    elif method == "CH":
        rv = M.inverse_CH(iszerofunc=iszerofunc)
    elif method == "LDL":
        rv = M.inverse_LDL(iszerofunc=iszerofunc)
    elif method == "QR":
        rv = M.inverse_QR(iszerofunc=iszerofunc)
    elif method == "BLOCK":
        rv = M.inverse_BLOCK(iszerofunc=iszerofunc)
    else:
        raise ValueError("Inversion method unrecognized")

    return M._new(rv)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\ssl_.py ===
from __future__ import annotations

import hashlib
import hmac
import os
import socket
import sys
import typing
import warnings
from binascii import unhexlify

from ..exceptions import ProxySchemeUnsupported, SSLError
from .url import _BRACELESS_IPV6_ADDRZ_RE, _IPV4_RE

SSLContext = None
SSLTransport = None
HAS_NEVER_CHECK_COMMON_NAME = False
IS_PYOPENSSL = False
ALPN_PROTOCOLS = ["http/1.1"]

_TYPE_VERSION_INFO = tuple[int, int, int, str, int]

# Maps the length of a digest to a possible hash function producing this digest
HASHFUNC_MAP = {
    length: getattr(hashlib, algorithm, None)
    for length, algorithm in ((32, "md5"), (40, "sha1"), (64, "sha256"))
}


def _is_bpo_43522_fixed(
    implementation_name: str,
    version_info: _TYPE_VERSION_INFO,
    pypy_version_info: _TYPE_VERSION_INFO | None,
) -> bool:
    """Return True for CPython 3.9.3+ or 3.10+ and PyPy 7.3.8+ where
    setting SSLContext.hostname_checks_common_name to False works.

    Outside of CPython and PyPy we don't know which implementations work
    or not so we conservatively use our hostname matching as we know that works
    on all implementations.

    https://github.com/urllib3/urllib3/issues/2192#issuecomment-821832963
    https://foss.heptapod.net/pypy/pypy/-/issues/3539
    """
    if implementation_name == "pypy":
        # https://foss.heptapod.net/pypy/pypy/-/issues/3129
        return pypy_version_info >= (7, 3, 8)  # type: ignore[operator]
    elif implementation_name == "cpython":
        major_minor = version_info[:2]
        micro = version_info[2]
        return (major_minor == (3, 9) and micro >= 3) or major_minor >= (3, 10)
    else:  # Defensive:
        return False


def _is_has_never_check_common_name_reliable(
    openssl_version: str,
    openssl_version_number: int,
    implementation_name: str,
    version_info: _TYPE_VERSION_INFO,
    pypy_version_info: _TYPE_VERSION_INFO | None,
) -> bool:
    # As of May 2023, all released versions of LibreSSL fail to reject certificates with
    # only common names, see https://github.com/urllib3/urllib3/pull/3024
    is_openssl = openssl_version.startswith("OpenSSL ")
    # Before fixing OpenSSL issue #14579, the SSL_new() API was not copying hostflags
    # like X509_CHECK_FLAG_NEVER_CHECK_SUBJECT, which tripped up CPython.
    # https://github.com/openssl/openssl/issues/14579
    # This was released in OpenSSL 1.1.1l+ (>=0x101010cf)
    is_openssl_issue_14579_fixed = openssl_version_number >= 0x101010CF

    return is_openssl and (
        is_openssl_issue_14579_fixed
        or _is_bpo_43522_fixed(implementation_name, version_info, pypy_version_info)
    )


if typing.TYPE_CHECKING:
    from ssl import VerifyMode
    from typing import TypedDict

    from .ssltransport import SSLTransport as SSLTransportType

    class _TYPE_PEER_CERT_RET_DICT(TypedDict, total=False):
        subjectAltName: tuple[tuple[str, str], ...]
        subject: tuple[tuple[tuple[str, str], ...], ...]
        serialNumber: str


# Mapping from 'ssl.PROTOCOL_TLSX' to 'TLSVersion.X'
_SSL_VERSION_TO_TLS_VERSION: dict[int, int] = {}

try:  # Do we have ssl at all?
    import ssl
    from ssl import (  # type: ignore[assignment]
        CERT_REQUIRED,
        HAS_NEVER_CHECK_COMMON_NAME,
        OP_NO_COMPRESSION,
        OP_NO_TICKET,
        OPENSSL_VERSION,
        OPENSSL_VERSION_NUMBER,
        PROTOCOL_TLS,
        PROTOCOL_TLS_CLIENT,
        VERIFY_X509_STRICT,
        OP_NO_SSLv2,
        OP_NO_SSLv3,
        SSLContext,
        TLSVersion,
    )

    PROTOCOL_SSLv23 = PROTOCOL_TLS

    # Needed for Python 3.9 which does not define this
    VERIFY_X509_PARTIAL_CHAIN = getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0x80000)

    # Setting SSLContext.hostname_checks_common_name = False didn't work before CPython
    # 3.9.3, and 3.10 (but OK on PyPy) or OpenSSL 1.1.1l+
    if HAS_NEVER_CHECK_COMMON_NAME and not _is_has_never_check_common_name_reliable(
        OPENSSL_VERSION,
        OPENSSL_VERSION_NUMBER,
        sys.implementation.name,
        sys.version_info,
        sys.pypy_version_info if sys.implementation.name == "pypy" else None,  # type: ignore[attr-defined]
    ):  # Defensive: for Python < 3.9.3
        HAS_NEVER_CHECK_COMMON_NAME = False

    # Need to be careful here in case old TLS versions get
    # removed in future 'ssl' module implementations.
    for attr in ("TLSv1", "TLSv1_1", "TLSv1_2"):
        try:
            _SSL_VERSION_TO_TLS_VERSION[getattr(ssl, f"PROTOCOL_{attr}")] = getattr(
                TLSVersion, attr
            )
        except AttributeError:  # Defensive:
            continue

    from .ssltransport import SSLTransport  # type: ignore[assignment]
except ImportError:
    OP_NO_COMPRESSION = 0x20000  # type: ignore[assignment]
    OP_NO_TICKET = 0x4000  # type: ignore[assignment]
    OP_NO_SSLv2 = 0x1000000  # type: ignore[assignment]
    OP_NO_SSLv3 = 0x2000000  # type: ignore[assignment]
    PROTOCOL_SSLv23 = PROTOCOL_TLS = 2  # type: ignore[assignment]
    PROTOCOL_TLS_CLIENT = 16  # type: ignore[assignment]
    VERIFY_X509_PARTIAL_CHAIN = 0x80000
    VERIFY_X509_STRICT = 0x20  # type: ignore[assignment]


_TYPE_PEER_CERT_RET = typing.Union["_TYPE_PEER_CERT_RET_DICT", bytes, None]


def assert_fingerprint(cert: bytes | None, fingerprint: str) -> None:
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """

    if cert is None:
        raise SSLError("No certificate for the peer.")

    fingerprint = fingerprint.replace(":", "").lower()
    digest_length = len(fingerprint)
    if digest_length not in HASHFUNC_MAP:
        raise SSLError(f"Fingerprint of invalid length: {fingerprint}")
    hashfunc = HASHFUNC_MAP.get(digest_length)
    if hashfunc is None:
        raise SSLError(
            f"Hash function implementation unavailable for fingerprint length: {digest_length}"
        )

    # We need encode() here for py32; works on py2 and p33.
    fingerprint_bytes = unhexlify(fingerprint.encode())

    cert_digest = hashfunc(cert).digest()

    if not hmac.compare_digest(cert_digest, fingerprint_bytes):
        raise SSLError(
            f'Fingerprints did not match. Expected "{fingerprint}", got "{cert_digest.hex()}"'
        )


def resolve_cert_reqs(candidate: None | int | str) -> VerifyMode:
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_REQUIRED`.
    If given a string it is assumed to be the name of the constant in the
    :mod:`ssl` module or its abbreviation.
    (So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
    If it's neither `None` nor a string we assume it is already the numeric
    constant which can directly be passed to wrap_socket.
    """
    if candidate is None:
        return CERT_REQUIRED

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "CERT_" + candidate)
        return res  # type: ignore[no-any-return]

    return candidate  # type: ignore[return-value]


def resolve_ssl_version(candidate: None | int | str) -> int:
    """
    like resolve_cert_reqs
    """
    if candidate is None:
        return PROTOCOL_TLS

    if isinstance(candidate, str):
        res = getattr(ssl, candidate, None)
        if res is None:
            res = getattr(ssl, "PROTOCOL_" + candidate)
        return typing.cast(int, res)

    return candidate


def create_urllib3_context(
    ssl_version: int | None = None,
    cert_reqs: int | None = None,
    options: int | None = None,
    ciphers: str | None = None,
    ssl_minimum_version: int | None = None,
    ssl_maximum_version: int | None = None,
    verify_flags: int | None = None,
) -> ssl.SSLContext:
    """Creates and configures an :class:`ssl.SSLContext` instance for use with urllib3.

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.

        This parameter is deprecated instead use 'ssl_minimum_version'.
    :param ssl_minimum_version:
        The minimum version of TLS to be used. Use the 'ssl.TLSVersion' enum for specifying the value.
    :param ssl_maximum_version:
        The maximum version of TLS to be used. Use the 'ssl.TLSVersion' enum for specifying the value.
        Not recommended to set to anything other than 'ssl.TLSVersion.MAXIMUM_SUPPORTED' which is the
        default value.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``, and ``ssl.OP_NO_TICKET``.
    :param ciphers:
        Which cipher suites to allow the server to select. Defaults to either system configured
        ciphers if OpenSSL 1.1.1+, otherwise uses a secure default set of ciphers.
    :param verify_flags:
        The flags for certificate verification operations. These default to
        ``ssl.VERIFY_X509_PARTIAL_CHAIN`` and ``ssl.VERIFY_X509_STRICT`` for Python 3.13+.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    if SSLContext is None:
        raise TypeError("Can't create an SSLContext object without an ssl module")

    # This means 'ssl_version' was specified as an exact value.
    if ssl_version not in (None, PROTOCOL_TLS, PROTOCOL_TLS_CLIENT):
        # Disallow setting 'ssl_version' and 'ssl_minimum|maximum_version'
        # to avoid conflicts.
        if ssl_minimum_version is not None or ssl_maximum_version is not None:
            raise ValueError(
                "Can't specify both 'ssl_version' and either "
                "'ssl_minimum_version' or 'ssl_maximum_version'"
            )

        # 'ssl_version' is deprecated and will be removed in the future.
        else:
            # Use 'ssl_minimum_version' and 'ssl_maximum_version' instead.
            ssl_minimum_version = _SSL_VERSION_TO_TLS_VERSION.get(
                ssl_version, TLSVersion.MINIMUM_SUPPORTED
            )
            ssl_maximum_version = _SSL_VERSION_TO_TLS_VERSION.get(
                ssl_version, TLSVersion.MAXIMUM_SUPPORTED
            )

            # This warning message is pushing users to use 'ssl_minimum_version'
            # instead of both min/max. Best practice is to only set the minimum version and
            # keep the maximum version to be it's default value: 'TLSVersion.MAXIMUM_SUPPORTED'
            warnings.warn(
                "'ssl_version' option is deprecated and will be "
                "removed in urllib3 v2.1.0. Instead use 'ssl_minimum_version'",
                category=DeprecationWarning,
                stacklevel=2,
            )

    # PROTOCOL_TLS is deprecated in Python 3.10 so we always use PROTOCOL_TLS_CLIENT
    context = SSLContext(PROTOCOL_TLS_CLIENT)

    if ssl_minimum_version is not None:
        context.minimum_version = ssl_minimum_version
    else:  # Python <3.10 defaults to 'MINIMUM_SUPPORTED' so explicitly set TLSv1.2 here
        context.minimum_version = TLSVersion.TLSv1_2

    if ssl_maximum_version is not None:
        context.maximum_version = ssl_maximum_version

    # Unless we're given ciphers defer to either system ciphers in
    # the case of OpenSSL 1.1.1+ or use our own secure default ciphers.
    if ciphers:
        context.set_ciphers(ciphers)

    # Setting the default here, as we may have no ssl module on import
    cert_reqs = ssl.CERT_REQUIRED if cert_reqs is None else cert_reqs

    if options is None:
        options = 0
        # SSLv2 is easily broken and is considered harmful and dangerous
        options |= OP_NO_SSLv2
        # SSLv3 has several problems and is now dangerous
        options |= OP_NO_SSLv3
        # Disable compression to prevent CRIME attacks for OpenSSL 1.0+
        # (issue #309)
        options |= OP_NO_COMPRESSION
        # TLSv1.2 only. Unless set explicitly, do not request tickets.
        # This may save some bandwidth on wire, and although the ticket is encrypted,
        # there is a risk associated with it being on wire,
        # if the server is not rotating its ticketing keys properly.
        options |= OP_NO_TICKET

    context.options |= options

    if verify_flags is None:
        verify_flags = 0
        # In Python 3.13+ ssl.create_default_context() sets VERIFY_X509_PARTIAL_CHAIN
        # and VERIFY_X509_STRICT so we do the same
        if sys.version_info >= (3, 13):
            verify_flags |= VERIFY_X509_PARTIAL_CHAIN
            verify_flags |= VERIFY_X509_STRICT

    context.verify_flags |= verify_flags

    # Enable post-handshake authentication for TLS 1.3, see GH #1634. PHA is
    # necessary for conditional client cert authentication with TLS 1.3.
    # The attribute is None for OpenSSL <= 1.1.0 or does not exist when using
    # an SSLContext created by pyOpenSSL.
    if getattr(context, "post_handshake_auth", None) is not None:
        context.post_handshake_auth = True

    # The order of the below lines setting verify_mode and check_hostname
    # matter due to safe-guards SSLContext has to prevent an SSLContext with
    # check_hostname=True, verify_mode=NONE/OPTIONAL.
    # We always set 'check_hostname=False' for pyOpenSSL so we rely on our own
    # 'ssl.match_hostname()' implementation.
    if cert_reqs == ssl.CERT_REQUIRED and not IS_PYOPENSSL:
        context.verify_mode = cert_reqs
        context.check_hostname = True
    else:
        context.check_hostname = False
        context.verify_mode = cert_reqs

    try:
        context.hostname_checks_common_name = False
    except AttributeError:  # Defensive: for CPython < 3.9.3; for PyPy < 7.3.8
        pass

    sslkeylogfile = os.environ.get("SSLKEYLOGFILE")
    if sslkeylogfile:
        context.keylog_filename = sslkeylogfile

    return context


@typing.overload
def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: str | None = ...,
    certfile: str | None = ...,
    cert_reqs: int | None = ...,
    ca_certs: str | None = ...,
    server_hostname: str | None = ...,
    ssl_version: int | None = ...,
    ciphers: str | None = ...,
    ssl_context: ssl.SSLContext | None = ...,
    ca_cert_dir: str | None = ...,
    key_password: str | None = ...,
    ca_cert_data: None | str | bytes = ...,
    tls_in_tls: typing.Literal[False] = ...,
) -> ssl.SSLSocket: ...


@typing.overload
def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: str | None = ...,
    certfile: str | None = ...,
    cert_reqs: int | None = ...,
    ca_certs: str | None = ...,
    server_hostname: str | None = ...,
    ssl_version: int | None = ...,
    ciphers: str | None = ...,
    ssl_context: ssl.SSLContext | None = ...,
    ca_cert_dir: str | None = ...,
    key_password: str | None = ...,
    ca_cert_data: None | str | bytes = ...,
    tls_in_tls: bool = ...,
) -> ssl.SSLSocket | SSLTransportType: ...


def ssl_wrap_socket(
    sock: socket.socket,
    keyfile: str | None = None,
    certfile: str | None = None,
    cert_reqs: int | None = None,
    ca_certs: str | None = None,
    server_hostname: str | None = None,
    ssl_version: int | None = None,
    ciphers: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
    ca_cert_dir: str | None = None,
    key_password: str | None = None,
    ca_cert_data: None | str | bytes = None,
    tls_in_tls: bool = False,
) -> ssl.SSLSocket | SSLTransportType:
    """
    All arguments except for server_hostname, ssl_context, tls_in_tls, ca_cert_data and
    ca_cert_dir have the same meaning as they do when using
    :func:`ssl.create_default_context`, :meth:`ssl.SSLContext.load_cert_chain`,
    :meth:`ssl.SSLContext.set_ciphers` and :meth:`ssl.SSLContext.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    :param key_password:
        Optional password if the keyfile is encrypted.
    :param ca_cert_data:
        Optional string containing CA certificates in PEM format suitable for
        passing as the cadata parameter to SSLContext.load_verify_locations()
    :param tls_in_tls:
        Use SSLTransport to wrap the existing socket.
    """
    context = ssl_context
    if context is None:
        # Note: This branch of code and all the variables in it are only used in tests.
        # We should consider deprecating and removing this code.
        context = create_urllib3_context(ssl_version, cert_reqs, ciphers=ciphers)

    if ca_certs or ca_cert_dir or ca_cert_data:
        try:
            context.load_verify_locations(ca_certs, ca_cert_dir, ca_cert_data)
        except OSError as e:
            raise SSLError(e) from e

    elif ssl_context is None and hasattr(context, "load_default_certs"):
        # try to load OS default certs; works well on Windows.
        context.load_default_certs()

    # Attempt to detect if we get the goofy behavior of the
    # keyfile being encrypted and OpenSSL asking for the
    # passphrase via the terminal and instead error out.
    if keyfile and key_password is None and _is_key_file_encrypted(keyfile):
        raise SSLError("Client private key is encrypted, password is required")

    if certfile:
        if key_password is None:
            context.load_cert_chain(certfile, keyfile)
        else:
            context.load_cert_chain(certfile, keyfile, key_password)

    context.set_alpn_protocols(ALPN_PROTOCOLS)

    ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)
    return ssl_sock


def is_ipaddress(hostname: str | bytes) -> bool:
    """Detects whether the hostname given is an IPv4 or IPv6 address.
    Also detects IPv6 addresses with Zone IDs.

    :param str hostname: Hostname to examine.
    :return: True if the hostname is an IP address, False otherwise.
    """
    if isinstance(hostname, bytes):
        # IDN A-label bytes are ASCII compatible.
        hostname = hostname.decode("ascii")
    return bool(_IPV4_RE.match(hostname) or _BRACELESS_IPV6_ADDRZ_RE.match(hostname))


def _is_key_file_encrypted(key_file: str) -> bool:
    """Detects if a key file is encrypted or not."""
    with open(key_file) as f:
        for line in f:
            # Look for Proc-Type: 4,ENCRYPTED
            if "ENCRYPTED" in line:
                return True

    return False


def _ssl_wrap_socket_impl(
    sock: socket.socket,
    ssl_context: ssl.SSLContext,
    tls_in_tls: bool,
    server_hostname: str | None = None,
) -> ssl.SSLSocket | SSLTransportType:
    if tls_in_tls:
        if not SSLTransport:
            # Import error, ssl is not available.
            raise ProxySchemeUnsupported(
                "TLS in TLS requires support for the 'ssl' module"
            )

        SSLTransport._validate_ssl_context_for_tls_in_tls(ssl_context)
        return SSLTransport(sock, ssl_context, server_hostname)

    return ssl_context.wrap_socket(sock, server_hostname=server_hostname)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\session.py ===
"""PipSession and supporting code, containing all pip-specific
network request configuration and behavior.
"""

import email.utils
import functools
import io
import ipaddress
import json
import logging
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import urllib.parse
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pip._vendor import requests, urllib3
from pip._vendor.cachecontrol import CacheControlAdapter as _BaseCacheControlAdapter
from pip._vendor.requests.adapters import DEFAULT_POOLBLOCK, BaseAdapter
from pip._vendor.requests.adapters import HTTPAdapter as _BaseHTTPAdapter
from pip._vendor.requests.models import PreparedRequest, Response
from pip._vendor.requests.structures import CaseInsensitiveDict
from pip._vendor.urllib3.connectionpool import ConnectionPool
from pip._vendor.urllib3.exceptions import InsecureRequestWarning

from pip import __version__
from pip._internal.metadata import get_default_environment
from pip._internal.models.link import Link
from pip._internal.network.auth import MultiDomainBasicAuth
from pip._internal.network.cache import SafeFileCache

# Import ssl from compat so the initial import occurs in only one place.
from pip._internal.utils.compat import has_tls
from pip._internal.utils.glibc import libc_ver
from pip._internal.utils.misc import build_url_from_netloc, parse_netloc
from pip._internal.utils.urls import url_to_path

if TYPE_CHECKING:
    from ssl import SSLContext

    from pip._vendor.urllib3.poolmanager import PoolManager


logger = logging.getLogger(__name__)

SecureOrigin = Tuple[str, str, Optional[Union[int, str]]]


# Ignore warning raised when using --trusted-host.
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


SECURE_ORIGINS: List[SecureOrigin] = [
    # protocol, hostname, port
    # Taken from Chrome's list of secure origins (See: http://bit.ly/1qrySKC)
    ("https", "*", "*"),
    ("*", "localhost", "*"),
    ("*", "127.0.0.0/8", "*"),
    ("*", "::1/128", "*"),
    ("file", "*", None),
    # ssh is always secure.
    ("ssh", "*", "*"),
]


# These are environment variables present when running under various
# CI systems.  For each variable, some CI systems that use the variable
# are indicated.  The collection was chosen so that for each of a number
# of popular systems, at least one of the environment variables is used.
# This list is used to provide some indication of and lower bound for
# CI traffic to PyPI.  Thus, it is okay if the list is not comprehensive.
# For more background, see: https://github.com/pypa/pip/issues/5499
CI_ENVIRONMENT_VARIABLES = (
    # Azure Pipelines
    "BUILD_BUILDID",
    # Jenkins
    "BUILD_ID",
    # AppVeyor, CircleCI, Codeship, Gitlab CI, Shippable, Travis CI
    "CI",
    # Explicit environment variable.
    "PIP_IS_CI",
)


def looks_like_ci() -> bool:
    """
    Return whether it looks like pip is running under CI.
    """
    # We don't use the method of checking for a tty (e.g. using isatty())
    # because some CI systems mimic a tty (e.g. Travis CI).  Thus that
    # method doesn't provide definitive information in either direction.
    return any(name in os.environ for name in CI_ENVIRONMENT_VARIABLES)


@functools.lru_cache(maxsize=1)
def user_agent() -> str:
    """
    Return a string representing the user agent.
    """
    data: Dict[str, Any] = {
        "installer": {"name": "pip", "version": __version__},
        "python": platform.python_version(),
        "implementation": {
            "name": platform.python_implementation(),
        },
    }

    if data["implementation"]["name"] == "CPython":
        data["implementation"]["version"] = platform.python_version()
    elif data["implementation"]["name"] == "PyPy":
        pypy_version_info = sys.pypy_version_info  # type: ignore
        if pypy_version_info.releaselevel == "final":
            pypy_version_info = pypy_version_info[:3]
        data["implementation"]["version"] = ".".join(
            [str(x) for x in pypy_version_info]
        )
    elif data["implementation"]["name"] == "Jython":
        # Complete Guess
        data["implementation"]["version"] = platform.python_version()
    elif data["implementation"]["name"] == "IronPython":
        # Complete Guess
        data["implementation"]["version"] = platform.python_version()

    if sys.platform.startswith("linux"):
        from pip._vendor import distro

        linux_distribution = distro.name(), distro.version(), distro.codename()
        distro_infos: Dict[str, Any] = dict(
            filter(
                lambda x: x[1],
                zip(["name", "version", "id"], linux_distribution),
            )
        )
        libc = dict(
            filter(
                lambda x: x[1],
                zip(["lib", "version"], libc_ver()),
            )
        )
        if libc:
            distro_infos["libc"] = libc
        if distro_infos:
            data["distro"] = distro_infos

    if sys.platform.startswith("darwin") and platform.mac_ver()[0]:
        data["distro"] = {"name": "macOS", "version": platform.mac_ver()[0]}

    if platform.system():
        data.setdefault("system", {})["name"] = platform.system()

    if platform.release():
        data.setdefault("system", {})["release"] = platform.release()

    if platform.machine():
        data["cpu"] = platform.machine()

    if has_tls():
        import _ssl as ssl

        data["openssl_version"] = ssl.OPENSSL_VERSION

    setuptools_dist = get_default_environment().get_distribution("setuptools")
    if setuptools_dist is not None:
        data["setuptools_version"] = str(setuptools_dist.version)

    if shutil.which("rustc") is not None:
        # If for any reason `rustc --version` fails, silently ignore it
        try:
            rustc_output = subprocess.check_output(
                ["rustc", "--version"], stderr=subprocess.STDOUT, timeout=0.5
            )
        except Exception:
            pass
        else:
            if rustc_output.startswith(b"rustc "):
                # The format of `rustc --version` is:
                # `b'rustc 1.52.1 (9bc8c42bb 2021-05-09)\n'`
                # We extract just the middle (1.52.1) part
                data["rustc_version"] = rustc_output.split(b" ")[1].decode()

    # Use None rather than False so as not to give the impression that
    # pip knows it is not being run under CI.  Rather, it is a null or
    # inconclusive result.  Also, we include some value rather than no
    # value to make it easier to know that the check has been run.
    data["ci"] = True if looks_like_ci() else None

    user_data = os.environ.get("PIP_USER_AGENT_USER_DATA")
    if user_data is not None:
        data["user_data"] = user_data

    return "{data[installer][name]}/{data[installer][version]} {json}".format(
        data=data,
        json=json.dumps(data, separators=(",", ":"), sort_keys=True),
    )


class LocalFSAdapter(BaseAdapter):
    def send(
        self,
        request: PreparedRequest,
        stream: bool = False,
        timeout: Optional[Union[float, Tuple[float, float]]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        proxies: Optional[Mapping[str, str]] = None,
    ) -> Response:
        pathname = url_to_path(request.url)

        resp = Response()
        resp.status_code = 200
        resp.url = request.url

        try:
            stats = os.stat(pathname)
        except OSError as exc:
            # format the exception raised as a io.BytesIO object,
            # to return a better error message:
            resp.status_code = 404
            resp.reason = type(exc).__name__
            resp.raw = io.BytesIO(f"{resp.reason}: {exc}".encode())
        else:
            modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
            content_type = mimetypes.guess_type(pathname)[0] or "text/plain"
            resp.headers = CaseInsensitiveDict(
                {
                    "Content-Type": content_type,
                    "Content-Length": stats.st_size,
                    "Last-Modified": modified,
                }
            )

            resp.raw = open(pathname, "rb")
            resp.close = resp.raw.close

        return resp

    def close(self) -> None:
        pass


class _SSLContextAdapterMixin:
    """Mixin to add the ``ssl_context`` constructor argument to HTTP adapters.

    The additional argument is forwarded directly to the pool manager. This allows us
    to dynamically decide what SSL store to use at runtime, which is used to implement
    the optional ``truststore`` backend.
    """

    def __init__(
        self,
        *,
        ssl_context: Optional["SSLContext"] = None,
        **kwargs: Any,
    ) -> None:
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = DEFAULT_POOLBLOCK,
        **pool_kwargs: Any,
    ) -> "PoolManager":
        if self._ssl_context is not None:
            pool_kwargs.setdefault("ssl_context", self._ssl_context)
        return super().init_poolmanager(  # type: ignore[misc]
            connections=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


class HTTPAdapter(_SSLContextAdapterMixin, _BaseHTTPAdapter):
    pass


class CacheControlAdapter(_SSLContextAdapterMixin, _BaseCacheControlAdapter):
    pass


class InsecureHTTPAdapter(HTTPAdapter):
    def cert_verify(
        self,
        conn: ConnectionPool,
        url: str,
        verify: Union[bool, str],
        cert: Optional[Union[str, Tuple[str, str]]],
    ) -> None:
        super().cert_verify(conn=conn, url=url, verify=False, cert=cert)


class InsecureCacheControlAdapter(CacheControlAdapter):
    def cert_verify(
        self,
        conn: ConnectionPool,
        url: str,
        verify: Union[bool, str],
        cert: Optional[Union[str, Tuple[str, str]]],
    ) -> None:
        super().cert_verify(conn=conn, url=url, verify=False, cert=cert)


class PipSession(requests.Session):
    timeout: Optional[int] = None

    def __init__(
        self,
        *args: Any,
        retries: int = 0,
        cache: Optional[str] = None,
        trusted_hosts: Sequence[str] = (),
        index_urls: Optional[List[str]] = None,
        ssl_context: Optional["SSLContext"] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param trusted_hosts: Domains not to emit warnings for when not using
            HTTPS.
        """
        super().__init__(*args, **kwargs)

        # Namespace the attribute with "pip_" just in case to prevent
        # possible conflicts with the base class.
        self.pip_trusted_origins: List[Tuple[str, Optional[int]]] = []
        self.pip_proxy = None

        # Attach our User Agent to the request
        self.headers["User-Agent"] = user_agent()

        # Attach our Authentication handler to the session
        self.auth = MultiDomainBasicAuth(index_urls=index_urls)

        # Create our urllib3.Retry instance which will allow us to customize
        # how we handle retries.
        retries = urllib3.Retry(
            # Set the total number of retries that a particular request can
            # have.
            total=retries,
            # A 503 error from PyPI typically means that the Fastly -> Origin
            # connection got interrupted in some way. A 503 error in general
            # is typically considered a transient error so we'll go ahead and
            # retry it.
            # A 500 may indicate transient error in Amazon S3
            # A 502 may be a transient error from a CDN like CloudFlare or CloudFront
            # A 520 or 527 - may indicate transient error in CloudFlare
            status_forcelist=[500, 502, 503, 520, 527],
            # Add a small amount of back off between failed requests in
            # order to prevent hammering the service.
            backoff_factor=0.25,
        )  # type: ignore

        # Our Insecure HTTPAdapter disables HTTPS validation. It does not
        # support caching so we'll use it for all http:// URLs.
        # If caching is disabled, we will also use it for
        # https:// hosts that we've marked as ignoring
        # TLS errors for (trusted-hosts).
        insecure_adapter = InsecureHTTPAdapter(max_retries=retries)

        # We want to _only_ cache responses on securely fetched origins or when
        # the host is specified as trusted. We do this because
        # we can't validate the response of an insecurely/untrusted fetched
        # origin, and we don't want someone to be able to poison the cache and
        # require manual eviction from the cache to fix it.
        if cache:
            secure_adapter = CacheControlAdapter(
                cache=SafeFileCache(cache),
                max_retries=retries,
                ssl_context=ssl_context,
            )
            self._trusted_host_adapter = InsecureCacheControlAdapter(
                cache=SafeFileCache(cache),
                max_retries=retries,
            )
        else:
            secure_adapter = HTTPAdapter(max_retries=retries, ssl_context=ssl_context)
            self._trusted_host_adapter = insecure_adapter

        self.mount("https://", secure_adapter)
        self.mount("http://", insecure_adapter)

        # Enable file:// urls
        self.mount("file://", LocalFSAdapter())

        for host in trusted_hosts:
            self.add_trusted_host(host, suppress_logging=True)

    def update_index_urls(self, new_index_urls: List[str]) -> None:
        """
        :param new_index_urls: New index urls to update the authentication
            handler with.
        """
        self.auth.index_urls = new_index_urls

    def add_trusted_host(
        self, host: str, source: Optional[str] = None, suppress_logging: bool = False
    ) -> None:
        """
        :param host: It is okay to provide a host that has previously been
            added.
        :param source: An optional source string, for logging where the host
            string came from.
        """
        if not suppress_logging:
            msg = f"adding trusted host: {host!r}"
            if source is not None:
                msg += f" (from {source})"
            logger.info(msg)

        parsed_host, parsed_port = parse_netloc(host)
        if parsed_host is None:
            raise ValueError(f"Trusted host URL must include a host part: {host!r}")
        if (parsed_host, parsed_port) not in self.pip_trusted_origins:
            self.pip_trusted_origins.append((parsed_host, parsed_port))

        self.mount(
            build_url_from_netloc(host, scheme="http") + "/", self._trusted_host_adapter
        )
        self.mount(build_url_from_netloc(host) + "/", self._trusted_host_adapter)
        if not parsed_port:
            self.mount(
                build_url_from_netloc(host, scheme="http") + ":",
                self._trusted_host_adapter,
            )
            # Mount wildcard ports for the same host.
            self.mount(build_url_from_netloc(host) + ":", self._trusted_host_adapter)

    def iter_secure_origins(self) -> Generator[SecureOrigin, None, None]:
        yield from SECURE_ORIGINS
        for host, port in self.pip_trusted_origins:
            yield ("*", host, "*" if port is None else port)

    def is_secure_origin(self, location: Link) -> bool:
        # Determine if this url used a secure transport mechanism
        parsed = urllib.parse.urlparse(str(location))
        origin_protocol, origin_host, origin_port = (
            parsed.scheme,
            parsed.hostname,
            parsed.port,
        )

        # The protocol to use to see if the protocol matches.
        # Don't count the repository type as part of the protocol: in
        # cases such as "git+ssh", only use "ssh". (I.e., Only verify against
        # the last scheme.)
        origin_protocol = origin_protocol.rsplit("+", 1)[-1]

        # Determine if our origin is a secure origin by looking through our
        # hardcoded list of secure origins, as well as any additional ones
        # configured on this PackageFinder instance.
        for secure_origin in self.iter_secure_origins():
            secure_protocol, secure_host, secure_port = secure_origin
            if origin_protocol != secure_protocol and secure_protocol != "*":
                continue

            try:
                addr = ipaddress.ip_address(origin_host or "")
                network = ipaddress.ip_network(secure_host)
            except ValueError:
                # We don't have both a valid address or a valid network, so
                # we'll check this origin against hostnames.
                if (
                    origin_host
                    and origin_host.lower() != secure_host.lower()
                    and secure_host != "*"
                ):
                    continue
            else:
                # We have a valid address and network, so see if the address
                # is contained within the network.
                if addr not in network:
                    continue

            # Check to see if the port matches.
            if (
                origin_port != secure_port
                and secure_port != "*"
                and secure_port is not None
            ):
                continue

            # If we've gotten here, then this origin matches the current
            # secure origin and we should return True
            return True

        # If we've gotten to this point, then the origin isn't secure and we
        # will not accept it as a valid location to search. We will however
        # log a warning that we are ignoring it.
        logger.warning(
            "The repository located at %s is not a trusted or secure host and "
            "is being ignored. If this repository is available via HTTPS we "
            "recommend you use HTTPS instead, otherwise you may silence "
            "this warning and allow it anyway with '--trusted-host %s'.",
            origin_host,
            origin_host,
        )

        return False

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Response:
        # Allow setting a default timeout on a session
        kwargs.setdefault("timeout", self.timeout)
        # Allow setting a default proxies on a session
        kwargs.setdefault("proxies", self.proxies)

        # Dispatch the actual request
        return super().request(method, url, *args, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\order.py ===
from sympy.core import S, sympify, Expr, Dummy, Add, Mul
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.function import Function, PoleError, expand_power_base, expand_log
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log
from sympy.sets.sets import Complement
from sympy.utilities.iterables import uniq, is_sequence


class Order(Expr):
    r""" Represents the limiting behavior of some function.

    Explanation
    ===========

    The order of a function characterizes the function based on the limiting
    behavior of the function as it goes to some limit. Only taking the limit
    point to be a number is currently supported. This is expressed in
    big O notation [1]_.

    The formal definition for the order of a function `g(x)` about a point `a`
    is such that `g(x) = O(f(x))` as `x \rightarrow a` if and only if there
    exists a `\delta > 0` and an `M > 0` such that `|g(x)| \leq M|f(x)|` for
    `|x-a| < \delta`.  This is equivalent to `\limsup_{x \rightarrow a}
    |g(x)/f(x)| < \infty`.

    Let's illustrate it on the following example by taking the expansion of
    `\sin(x)` about 0:

    .. math ::
        \sin(x) = x - x^3/3! + O(x^5)

    where in this case `O(x^5) = x^5/5! - x^7/7! + \cdots`. By the definition
    of `O`, there is a `\delta > 0` and an `M` such that:

    .. math ::
        |x^5/5! - x^7/7! + ....| <= M|x^5| \text{ for } |x| < \delta

    or by the alternate definition:

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| < \infty

    which surely is true, because

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| = 1/5!


    As it is usually used, the order of a function can be intuitively thought
    of representing all terms of powers greater than the one specified. For
    example, `O(x^3)` corresponds to any terms proportional to `x^3,
    x^4,\ldots` and any higher power. For a polynomial, this leaves terms
    proportional to `x^2`, `x` and constants.

    Examples
    ========

    >>> from sympy import O, oo, cos, pi
    >>> from sympy.abc import x, y

    >>> O(x + x**2)
    O(x)
    >>> O(x + x**2, (x, 0))
    O(x)
    >>> O(x + x**2, (x, oo))
    O(x**2, (x, oo))

    >>> O(1 + x*y)
    O(1, x, y)
    >>> O(1 + x*y, (x, 0), (y, 0))
    O(1, x, y)
    >>> O(1 + x*y, (x, oo), (y, oo))
    O(x*y, (x, oo), (y, oo))

    >>> O(1) in O(1, x)
    True
    >>> O(1, x) in O(1)
    False
    >>> O(x) in O(1, x)
    True
    >>> O(x**2) in O(x)
    True

    >>> O(x)*x
    O(x**2)
    >>> O(x) - O(x)
    O(x)
    >>> O(cos(x))
    O(1)
    >>> O(cos(x), (x, pi/2))
    O(x - pi/2, (x, pi/2))

    References
    ==========

    .. [1] `Big O notation <https://en.wikipedia.org/wiki/Big_O_notation>`_

    Notes
    =====

    In ``O(f(x), x)`` the expression ``f(x)`` is assumed to have a leading
    term.  ``O(f(x), x)`` is automatically transformed to
    ``O(f(x).as_leading_term(x),x)``.

        ``O(expr*f(x), x)`` is ``O(f(x), x)``

        ``O(expr, x)`` is ``O(1)``

        ``O(0, x)`` is 0.

    Multivariate O is also supported:

        ``O(f(x, y), x, y)`` is transformed to
        ``O(f(x, y).as_leading_term(x,y).as_leading_term(y), x, y)``

    In the multivariate case, it is assumed the limits w.r.t. the various
    symbols commute.

    If no symbols are passed then all symbols in the expression are used
    and the limit point is assumed to be zero.

    """

    is_Order = True

    __slots__ = ()

    @cacheit
    def __new__(cls, expr, *args, **kwargs):
        expr = sympify(expr)

        if not args:
            if expr.is_Order:
                variables = expr.variables
                point = expr.point
            else:
                variables = list(expr.free_symbols)
                point = [S.Zero]*len(variables)
        else:
            args = list(args if is_sequence(args) else [args])
            variables, point = [], []
            if is_sequence(args[0]):
                for a in args:
                    v, p = list(map(sympify, a))
                    variables.append(v)
                    point.append(p)
            else:
                variables = list(map(sympify, args))
                point = [S.Zero]*len(variables)

        if not all(v.is_symbol for v in variables):
            raise TypeError('Variables are not symbols, got %s' % variables)

        if len(list(uniq(variables))) != len(variables):
            raise ValueError('Variables are supposed to be unique symbols, got %s' % variables)

        if expr.is_Order:
            expr_vp = dict(expr.args[1:])
            new_vp = dict(expr_vp)
            vp = dict(zip(variables, point))
            for v, p in vp.items():
                if v in new_vp.keys():
                    if p != new_vp[v]:
                        raise NotImplementedError(
                            "Mixing Order at different points is not supported.")
                else:
                    new_vp[v] = p
            if set(expr_vp.keys()) == set(new_vp.keys()):
                return expr
            else:
                variables = list(new_vp.keys())
                point = [new_vp[v] for v in variables]

        if expr is S.NaN:
            return S.NaN

        if any(x in p.free_symbols for x in variables for p in point):
            raise ValueError('Got %s as a point.' % point)

        if variables:
            if any(p != point[0] for p in point):
                raise NotImplementedError(
                    "Multivariable orders at different points are not supported.")
            if point[0] in (S.Infinity, S.Infinity*S.ImaginaryUnit):
                s = {k: 1/Dummy() for k in variables}
                rs = {1/v: 1/k for k, v in s.items()}
                ps = [S.Zero for p in point]
            elif point[0] in (S.NegativeInfinity, S.NegativeInfinity*S.ImaginaryUnit):
                s = {k: -1/Dummy() for k in variables}
                rs = {-1/v: -1/k for k, v in s.items()}
                ps = [S.Zero for p in point]
            elif point[0] is not S.Zero:
                s = {k: Dummy() + point[0] for k in variables}
                rs = {(v - point[0]).together(): k - point[0] for k, v in s.items()}
                ps = [S.Zero for p in point]
            else:
                s = ()
                rs = ()
                ps = list(point)

            expr = expr.subs(s)

            if expr.is_Add:
                expr = expr.factor()

            if s:
                args = tuple([r[0] for r in rs.items()])
            else:
                args = tuple(variables)

            if len(variables) > 1:
                # XXX: better way?  We need this expand() to
                # workaround e.g: expr = x*(x + y).
                # (x*(x + y)).as_leading_term(x, y) currently returns
                # x*y (wrong order term!).  That's why we want to deal with
                # expand()'ed expr (handled in "if expr.is_Add" branch below).
                expr = expr.expand()

            old_expr = None
            while old_expr != expr:
                old_expr = expr
                if expr.is_Add:
                    lst = expr.extract_leading_order(args)
                    expr = Add(*[f.expr for (e, f) in lst])

                elif expr:
                    try:
                        expr = expr.as_leading_term(*args)
                    except PoleError:
                        if isinstance(expr, Function) or\
                                all(isinstance(arg, Function) for arg in expr.args):
                            # It is not possible to simplify an expression
                            # containing only functions (which raise error on
                            # call to leading term) further
                            pass
                        else:
                            orders = []
                            pts = tuple(zip(args, ps))
                            for arg in expr.args:
                                try:
                                    lt = arg.as_leading_term(*args)
                                except PoleError:
                                    lt = arg
                                if lt not in args:
                                    order = Order(lt)
                                else:
                                    order = Order(lt, *pts)
                                orders.append(order)
                            if expr.is_Add:
                                new_expr = Order(Add(*orders), *pts)
                                if new_expr.is_Add:
                                    new_expr = Order(Add(*[a.expr for a in new_expr.args]), *pts)
                                expr = new_expr.expr
                            elif expr.is_Mul:
                                expr = Mul(*[a.expr for a in orders])
                            elif expr.is_Pow:
                                e = expr.exp
                                b = expr.base
                                expr = exp(e * log(b))

                    # It would probably be better to handle this somewhere
                    # else. This is needed for a testcase in which there is a
                    # symbol with the assumptions zero=True.
                    if expr.is_zero:
                        expr = S.Zero
                    else:
                        expr = expr.as_independent(*args, as_Add=False)[1]

                    expr = expand_power_base(expr)
                    expr = expand_log(expr)

                    if len(args) == 1:
                        # The definition of O(f(x)) symbol explicitly stated that
                        # the argument of f(x) is irrelevant.  That's why we can
                        # combine some power exponents (only "on top" of the
                        # expression tree for f(x)), e.g.:
                        # x**p * (-x)**q -> x**(p+q) for real p, q.
                        x = args[0]
                        margs = list(Mul.make_args(
                            expr.as_independent(x, as_Add=False)[1]))

                        for i, t in enumerate(margs):
                            if t.is_Pow:
                                b, q = t.args
                                if b in (x, -x) and q.is_real and not q.has(x):
                                    margs[i] = x**q
                                elif b.is_Pow and not b.exp.has(x):
                                    b, r = b.args
                                    if b in (x, -x) and r.is_real:
                                        margs[i] = x**(r*q)
                                elif b.is_Mul and b.args[0] is S.NegativeOne:
                                    b = -b
                                    if b.is_Pow and not b.exp.has(x):
                                        b, r = b.args
                                        if b in (x, -x) and r.is_real:
                                            margs[i] = x**(r*q)

                        expr = Mul(*margs)

            expr = expr.subs(rs)

        if expr.is_Order:
            expr = expr.expr

        if not expr.has(*variables) and not expr.is_zero:
            expr = S.One

        # create Order instance:
        vp = dict(zip(variables, point))
        variables.sort(key=default_sort_key)
        point = [vp[v] for v in variables]
        args = (expr,) + Tuple(*zip(variables, point))
        obj = Expr.__new__(cls, *args)
        return obj

    def _eval_nseries(self, x, n, logx, cdir=0):
        return self

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        if self.args[1:]:
            return tuple(x[0] for x in self.args[1:])
        else:
            return ()

    @property
    def point(self):
        if self.args[1:]:
            return tuple(x[1] for x in self.args[1:])
        else:
            return ()

    @property
    def free_symbols(self):
        return self.expr.free_symbols | set(self.variables)

    def _eval_power(b, e):
        if e.is_Number and e.is_nonnegative:
            return b.func(b.expr ** e, *b.args[1:])
        if e == O(1):
            return b
        return

    def as_expr_variables(self, order_symbols):
        if order_symbols is None:
            order_symbols = self.args[1:]
        else:
            if (not all(o[1] == order_symbols[0][1] for o in order_symbols) and
                    not all(p == self.point[0] for p in self.point)):  # pragma: no cover
                raise NotImplementedError('Order at points other than 0 '
                    'or oo not supported, got %s as a point.' % self.point)
            if order_symbols and order_symbols[0][1] != self.point[0]:
                raise NotImplementedError(
                        "Multiplying Order at different points is not supported.")
            order_symbols = dict(order_symbols)
            for s, p in dict(self.args[1:]).items():
                if s not in order_symbols.keys():
                    order_symbols[s] = p
            order_symbols = sorted(order_symbols.items(), key=lambda x: default_sort_key(x[0]))
        return self.expr, tuple(order_symbols)

    def removeO(self):
        return S.Zero

    def getO(self):
        return self

    @cacheit
    def contains(self, expr):
        r"""
        Return True if expr belongs to Order(self.expr, \*self.variables).
        Return False if self belongs to expr.
        Return None if the inclusion relation cannot be determined
        (e.g. when self and expr have different symbols).
        """
        expr = sympify(expr)
        if expr.is_zero:
            return True
        if expr is S.NaN:
            return False
        point = self.point[0] if self.point else S.Zero
        if expr.is_Order:
            if (any(p != point for p in expr.point) or
                   any(p != point for p in self.point)):
                return None
            if expr.expr == self.expr:
                # O(1) + O(1), O(1) + O(1, x), etc.
                return all(x in self.args[1:] for x in expr.args[1:])
            if expr.expr.is_Add:
                return all(self.contains(x) for x in expr.expr.args)
            if self.expr.is_Add and point.is_zero:
                return any(self.func(x, *self.args[1:]).contains(expr)
                            for x in self.expr.args)
            if self.variables and expr.variables:
                common_symbols = tuple(
                    [s for s in self.variables if s in expr.variables])
            elif self.variables:
                common_symbols = self.variables
            else:
                common_symbols = expr.variables
            if not common_symbols:
                return None
            if (self.expr.is_Pow and len(self.variables) == 1
                and self.variables == expr.variables):
                    symbol = self.variables[0]
                    other = expr.expr.as_independent(symbol, as_Add=False)[1]
                    if (other.is_Pow and other.base == symbol and
                        self.expr.base == symbol):
                            if point.is_zero:
                                rv = (self.expr.exp - other.exp).is_nonpositive
                            if point.is_infinite:
                                rv = (self.expr.exp - other.exp).is_nonnegative
                            if rv is not None:
                                return rv

            from sympy.simplify.powsimp import powsimp
            r = None
            ratio = self.expr/expr.expr
            ratio = powsimp(ratio, deep=True, combine='exp')
            for s in common_symbols:
                from sympy.series.limits import Limit
                l = Limit(ratio, s, point).doit(heuristics=False)
                if not isinstance(l, Limit):
                    l = l != 0
                else:
                    l = None
                if r is None:
                    r = l
                else:
                    if r != l:
                        return
            return r

        if self.expr.is_Pow and len(self.variables) == 1:
            symbol = self.variables[0]
            other = expr.as_independent(symbol, as_Add=False)[1]
            if (other.is_Pow and other.base == symbol and
                self.expr.base == symbol):
                    if point.is_zero:
                        rv = (self.expr.exp - other.exp).is_nonpositive
                    if point.is_infinite:
                        rv = (self.expr.exp - other.exp).is_nonnegative
                    if rv is not None:
                        return rv

        obj = self.func(expr, *self.args[1:])
        return self.contains(obj)

    def __contains__(self, other):
        result = self.contains(other)
        if result is None:
            raise TypeError('contains did not evaluate to a bool')
        return result

    def _eval_subs(self, old, new):
        if old in self.variables:
            newexpr = self.expr.subs(old, new)
            i = self.variables.index(old)
            newvars = list(self.variables)
            newpt = list(self.point)
            if new.is_symbol:
                newvars[i] = new
            else:
                syms = new.free_symbols
                if len(syms) == 1 or old in syms:
                    if old in syms:
                        var = self.variables[i]
                    else:
                        var = syms.pop()
                    # First, try to substitute self.point in the "new"
                    # expr to see if this is a fixed point.
                    # E.g.  O(y).subs(y, sin(x))
                    from sympy import limit
                    if new.has(Order) and limit(new.getO().expr, var, new.getO().point[0]) == self.point[i]:
                        point = new.getO().point[0]
                        return Order(newexpr, *zip([var], [point]))
                    else:
                        point = new.subs(var, self.point[i])
                    if point != self.point[i]:
                        from sympy.solvers.solveset import solveset
                        d = Dummy()
                        sol = solveset(old - new.subs(var, d), d)
                        if isinstance(sol, Complement):
                            e1 = sol.args[0]
                            e2 = sol.args[1]
                            sol = set(e1) - set(e2)
                        res = [dict(zip((d, ), sol))]
                        point = d.subs(res[0]).limit(old, self.point[i])
                    newvars[i] = var
                    newpt[i] = point
                elif old not in syms:
                    del newvars[i], newpt[i]
                    if not syms and new == self.point[i]:
                        newvars.extend(syms)
                        newpt.extend([S.Zero]*len(syms))
                else:
                    return
            return Order(newexpr, *zip(newvars, newpt))

    def _eval_conjugate(self):
        expr = self.expr._eval_conjugate()
        if expr is not None:
            return self.func(expr, *self.args[1:])

    def _eval_derivative(self, x):
        return self.func(self.expr.diff(x), *self.args[1:]) or self

    def _eval_transpose(self):
        expr = self.expr._eval_transpose()
        if expr is not None:
            return self.func(expr, *self.args[1:])

    def __neg__(self):
        return self

O = Order

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_io.py ===
from __future__ import annotations

import random
import select
import socket as stdlib_socket
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TYPE_CHECKING, TypeVar

import pytest

import trio

from ... import _core
from ...testing import assert_checkpoints, wait_all_tasks_blocked

# Cross-platform tests for IO handling

if TYPE_CHECKING:
    from collections.abc import Generator

    from typing_extensions import ParamSpec

    ArgsT = ParamSpec("ArgsT")


def fill_socket(sock: stdlib_socket.socket) -> None:
    try:
        while True:
            sock.send(b"x" * 65536)
    except BlockingIOError:
        pass


def drain_socket(sock: stdlib_socket.socket) -> None:
    try:
        while True:
            sock.recv(65536)
    except BlockingIOError:
        pass


WaitSocket = Callable[[stdlib_socket.socket], Awaitable[object]]
SocketPair = tuple[stdlib_socket.socket, stdlib_socket.socket]
RetT = TypeVar("RetT")


@pytest.fixture
def socketpair() -> Generator[SocketPair, None, None]:
    pair = stdlib_socket.socketpair()
    for sock in pair:
        sock.setblocking(False)
    yield pair
    for sock in pair:
        sock.close()


def also_using_fileno(
    fn: Callable[[stdlib_socket.socket | int], RetT],
) -> list[Callable[[stdlib_socket.socket], RetT]]:
    def fileno_wrapper(fileobj: stdlib_socket.socket) -> RetT:
        return fn(fileobj.fileno())

    name = f"<{fn.__name__} on fileno>"
    fileno_wrapper.__name__ = fileno_wrapper.__qualname__ = name
    return [fn, fileno_wrapper]


# Decorators that feed in different settings for wait_readable / wait_writable
# / notify_closing.
# Note that if you use all three decorators on the same test, it will run all
# N**3 *combinations*
read_socket_test = pytest.mark.parametrize(
    "wait_readable",
    also_using_fileno(trio.lowlevel.wait_readable),
    ids=lambda fn: fn.__name__,
)
write_socket_test = pytest.mark.parametrize(
    "wait_writable",
    also_using_fileno(trio.lowlevel.wait_writable),
    ids=lambda fn: fn.__name__,
)
notify_closing_test = pytest.mark.parametrize(
    "notify_closing",
    also_using_fileno(trio.lowlevel.notify_closing),
    ids=lambda fn: fn.__name__,
)


# XX These tests are all a bit dicey because they can't distinguish between
# wait_on_{read,writ}able blocking the way it should, versus blocking
# momentarily and then immediately resuming.
@read_socket_test
@write_socket_test
async def test_wait_basic(
    socketpair: SocketPair,
    wait_readable: WaitSocket,
    wait_writable: WaitSocket,
) -> None:
    a, b = socketpair

    # They start out writable()
    with assert_checkpoints():
        await wait_writable(a)

    # But readable() blocks until data arrives
    record = []

    async def block_on_read() -> None:
        try:
            with assert_checkpoints():
                await wait_readable(a)
        except _core.Cancelled:
            record.append("cancelled")
        else:
            record.append("readable")
            assert a.recv(10) == b"x"

    async with _core.open_nursery() as nursery:
        nursery.start_soon(block_on_read)
        await wait_all_tasks_blocked()
        assert record == []
        b.send(b"x")

    fill_socket(a)

    # Now writable will block, but readable won't
    with assert_checkpoints():
        await wait_readable(b)
    record = []

    async def block_on_write() -> None:
        try:
            with assert_checkpoints():
                await wait_writable(a)
        except _core.Cancelled:
            record.append("cancelled")
        else:
            record.append("writable")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(block_on_write)
        await wait_all_tasks_blocked()
        assert record == []
        drain_socket(b)

    # check cancellation
    record = []
    async with _core.open_nursery() as nursery:
        nursery.start_soon(block_on_read)
        await wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()
    assert record == ["cancelled"]

    fill_socket(a)
    record = []
    async with _core.open_nursery() as nursery:
        nursery.start_soon(block_on_write)
        await wait_all_tasks_blocked()
        nursery.cancel_scope.cancel()
    assert record == ["cancelled"]


@read_socket_test
async def test_double_read(socketpair: SocketPair, wait_readable: WaitSocket) -> None:
    a, _b = socketpair

    # You can't have two tasks trying to read from a socket at the same time
    async with _core.open_nursery() as nursery:
        nursery.start_soon(wait_readable, a)
        await wait_all_tasks_blocked()
        with pytest.raises(_core.BusyResourceError):
            await wait_readable(a)
        nursery.cancel_scope.cancel()


@write_socket_test
async def test_double_write(socketpair: SocketPair, wait_writable: WaitSocket) -> None:
    a, _b = socketpair

    # You can't have two tasks trying to write to a socket at the same time
    fill_socket(a)
    async with _core.open_nursery() as nursery:
        nursery.start_soon(wait_writable, a)
        await wait_all_tasks_blocked()
        with pytest.raises(_core.BusyResourceError):
            await wait_writable(a)
        nursery.cancel_scope.cancel()


@read_socket_test
@write_socket_test
@notify_closing_test
async def test_interrupted_by_close(
    socketpair: SocketPair,
    wait_readable: WaitSocket,
    wait_writable: WaitSocket,
    notify_closing: Callable[[stdlib_socket.socket], object],
) -> None:
    a, _b = socketpair

    async def reader() -> None:
        with pytest.raises(_core.ClosedResourceError):
            await wait_readable(a)

    async def writer() -> None:
        with pytest.raises(_core.ClosedResourceError):
            await wait_writable(a)

    fill_socket(a)

    async with _core.open_nursery() as nursery:
        nursery.start_soon(reader)
        nursery.start_soon(writer)
        await wait_all_tasks_blocked()
        notify_closing(a)


@read_socket_test
@write_socket_test
async def test_socket_simultaneous_read_write(
    socketpair: SocketPair,
    wait_readable: WaitSocket,
    wait_writable: WaitSocket,
) -> None:
    record: list[str] = []

    async def r_task(sock: stdlib_socket.socket) -> None:
        await wait_readable(sock)
        record.append("r_task")

    async def w_task(sock: stdlib_socket.socket) -> None:
        await wait_writable(sock)
        record.append("w_task")

    a, b = socketpair
    fill_socket(a)
    async with _core.open_nursery() as nursery:
        nursery.start_soon(r_task, a)
        nursery.start_soon(w_task, a)
        await wait_all_tasks_blocked()
        assert record == []
        b.send(b"x")
        await wait_all_tasks_blocked()
        assert record == ["r_task"]
        drain_socket(b)
        await wait_all_tasks_blocked()
        assert record == ["r_task", "w_task"]


@read_socket_test
@write_socket_test
async def test_socket_actual_streaming(
    socketpair: SocketPair,
    wait_readable: WaitSocket,
    wait_writable: WaitSocket,
) -> None:
    a, b = socketpair

    # Use a small send buffer on one of the sockets to increase the chance of
    # getting partial writes
    a.setsockopt(stdlib_socket.SOL_SOCKET, stdlib_socket.SO_SNDBUF, 10000)

    N = 1000000  # 1 megabyte
    MAX_CHUNK = 65536

    results: dict[str, int] = {}

    async def sender(sock: stdlib_socket.socket, seed: int, key: str) -> None:
        r = random.Random(seed)
        sent = 0
        while sent < N:
            print("sent", sent)
            chunk = bytearray(r.randrange(MAX_CHUNK))
            while chunk:
                with assert_checkpoints():
                    await wait_writable(sock)
                this_chunk_size = sock.send(chunk)
                sent += this_chunk_size
                del chunk[:this_chunk_size]
        sock.shutdown(stdlib_socket.SHUT_WR)
        results[key] = sent

    async def receiver(sock: stdlib_socket.socket, key: str) -> None:
        received = 0
        while True:
            print("received", received)
            with assert_checkpoints():
                await wait_readable(sock)
            this_chunk_size = len(sock.recv(MAX_CHUNK))
            if not this_chunk_size:
                break
            received += this_chunk_size
        results[key] = received

    async with _core.open_nursery() as nursery:
        nursery.start_soon(sender, a, 0, "send_a")
        nursery.start_soon(sender, b, 1, "send_b")
        nursery.start_soon(receiver, a, "recv_a")
        nursery.start_soon(receiver, b, "recv_b")

    assert results["send_a"] == results["recv_b"]
    assert results["send_b"] == results["recv_a"]


async def test_notify_closing_on_invalid_object() -> None:
    # It should either be a no-op (generally on Unix, where we don't know
    # which fds are valid), or an OSError (on Windows, where we currently only
    # support sockets, so we have to do some validation to figure out whether
    # it's a socket or a regular handle).
    got_oserror = False
    got_no_error = False
    try:
        trio.lowlevel.notify_closing(-1)
    except OSError:
        got_oserror = True
    else:
        got_no_error = True
    assert got_oserror or got_no_error


async def test_wait_on_invalid_object() -> None:
    # We definitely want to raise an error everywhere if you pass in an
    # invalid fd to wait_*
    for wait in [trio.lowlevel.wait_readable, trio.lowlevel.wait_writable]:
        with stdlib_socket.socket() as s:
            fileno = s.fileno()
        # We just closed the socket and don't do anything else in between, so
        # we can be confident that the fileno hasn't be reassigned.
        with pytest.raises(
            OSError,
            match=r"^\[\w+ \d+] (Bad file descriptor|An operation was attempted on something that is not a socket)$",
        ):
            await wait(fileno)


async def test_io_manager_statistics() -> None:
    def check(*, expected_readers: int, expected_writers: int) -> None:
        statistics = _core.current_statistics()
        print(statistics)
        iostats = statistics.io_statistics
        if iostats.backend == "epoll" or iostats.backend == "windows":
            assert iostats.tasks_waiting_read == expected_readers
            assert iostats.tasks_waiting_write == expected_writers
        else:
            assert iostats.backend == "kqueue"
            assert iostats.monitors == 0
            assert iostats.tasks_waiting == expected_readers + expected_writers

    a1, b1 = stdlib_socket.socketpair()
    a2, b2 = stdlib_socket.socketpair()
    a3, b3 = stdlib_socket.socketpair()
    for sock in [a1, b1, a2, b2, a3, b3]:
        sock.setblocking(False)
    with a1, b1, a2, b2, a3, b3:
        # let the call_soon_task settle down
        await wait_all_tasks_blocked()

        # 1 for call_soon_task
        check(expected_readers=1, expected_writers=0)

        # We want:
        # - one socket with a writer blocked
        # - two sockets with a reader blocked
        # - a socket with both blocked
        fill_socket(a1)
        fill_socket(a3)
        async with _core.open_nursery() as nursery:
            nursery.start_soon(_core.wait_writable, a1)
            nursery.start_soon(_core.wait_readable, a2)
            nursery.start_soon(_core.wait_readable, b2)
            nursery.start_soon(_core.wait_writable, a3)
            nursery.start_soon(_core.wait_readable, a3)

            await wait_all_tasks_blocked()

            # +1 for call_soon_task
            check(expected_readers=3 + 1, expected_writers=2)

            nursery.cancel_scope.cancel()

        # 1 for call_soon_task
        check(expected_readers=1, expected_writers=0)


@pytest.mark.filterwarnings("ignore:.*UnboundedQueue:trio.TrioDeprecationWarning")
async def test_io_manager_kqueue_monitors_statistics() -> None:
    def check(
        *,
        expected_monitors: int,
        expected_readers: int,
        expected_writers: int,
    ) -> None:
        statistics = _core.current_statistics()
        print(statistics)
        iostats = statistics.io_statistics
        assert iostats.backend == "kqueue"
        assert iostats.monitors == expected_monitors
        assert iostats.tasks_waiting == expected_readers + expected_writers

    a1, b1 = stdlib_socket.socketpair()
    for sock in [a1, b1]:
        sock.setblocking(False)

    with a1, b1:
        # let the call_soon_task settle down
        await wait_all_tasks_blocked()

        if sys.platform != "win32" and sys.platform != "linux":
            # 1 for call_soon_task
            check(expected_monitors=0, expected_readers=1, expected_writers=0)

            with _core.monitor_kevent(a1.fileno(), select.KQ_FILTER_READ):
                with (
                    pytest.raises(_core.BusyResourceError),
                    _core.monitor_kevent(a1.fileno(), select.KQ_FILTER_READ),
                ):
                    pass  # pragma: no cover
                check(expected_monitors=1, expected_readers=1, expected_writers=0)

            check(expected_monitors=0, expected_readers=1, expected_writers=0)


async def test_can_survive_unnotified_close() -> None:
    # An "unnotified" close is when the user closes an fd/socket/handle
    # directly, without calling notify_closing first. This should never happen
    # -- users should call notify_closing before closing things. But, just in
    # case they don't, we would still like to avoid exploding.
    #
    # Acceptable behaviors:
    # - wait_* never return, but can be cancelled cleanly
    # - wait_* exit cleanly
    # - wait_* raise an OSError
    #
    # Not acceptable:
    # - getting stuck in an uncancellable state
    # - TrioInternalError blowing up the whole run
    #
    # This test exercises some tricky "unnotified close" scenarios, to make
    # sure we get the "acceptable" behaviors.

    async def allow_OSError(
        async_func: Callable[ArgsT, Awaitable[object]],
        *args: ArgsT.args,
        **kwargs: ArgsT.kwargs,
    ) -> None:
        with suppress(OSError):
            await async_func(*args, **kwargs)

    with stdlib_socket.socket() as s:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(allow_OSError, trio.lowlevel.wait_readable, s)
            await wait_all_tasks_blocked()
            s.close()
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()

    # We hit different paths on Windows depending on whether we close the last
    # handle to the object (which produces a LOCAL_CLOSE notification and
    # wakes up wait_readable), or only close one of the handles (which leaves
    # wait_readable pending until cancelled).
    with stdlib_socket.socket() as s, s.dup() as s2:  # noqa: F841
        async with trio.open_nursery() as nursery:
            nursery.start_soon(allow_OSError, trio.lowlevel.wait_readable, s)
            await wait_all_tasks_blocked()
            s.close()
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()

    # A more elaborate case, with two tasks waiting. On windows and epoll,
    # the two tasks get muxed together onto a single underlying wait
    # operation. So when they're cancelled, there's a brief moment where one
    # of the tasks is cancelled but the other isn't, so we try to re-issue the
    # underlying wait operation. But here, the handle we were going to use to
    # do that has been pulled out from under our feet... so test that we can
    # survive this.
    a, b = stdlib_socket.socketpair()
    with a, b, a.dup() as a2:
        a.setblocking(False)
        b.setblocking(False)
        fill_socket(a)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(allow_OSError, trio.lowlevel.wait_readable, a)
            nursery.start_soon(allow_OSError, trio.lowlevel.wait_writable, a)
            await wait_all_tasks_blocked()
            a.close()
            nursery.cancel_scope.cancel()

    # A similar case, but now the single-task-wakeup happens due to I/O
    # arriving, not a cancellation, so the operation gets re-issued from
    # handle_io context rather than abort context.
    a, b = stdlib_socket.socketpair()
    with a, b, a.dup() as a2:
        print(f"a={a.fileno()}, b={b.fileno()}, a2={a2.fileno()}")
        a.setblocking(False)
        b.setblocking(False)
        fill_socket(a)
        e = trio.Event()

        # We want to wait for the kernel to process the wakeup on 'a', if any.
        # But depending on the platform, we might not get a wakeup on 'a'. So
        # we put one task to sleep waiting on 'a', and we put a second task to
        # sleep waiting on 'a2', with the idea that the 'a2' notification will
        # definitely arrive, and when it does then we can assume that whatever
        # notification was going to arrive for 'a' has also arrived.
        async def wait_readable_a2_then_set() -> None:
            await trio.lowlevel.wait_readable(a2)
            e.set()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(allow_OSError, trio.lowlevel.wait_readable, a)
            nursery.start_soon(allow_OSError, trio.lowlevel.wait_writable, a)
            nursery.start_soon(wait_readable_a2_then_set)
            await wait_all_tasks_blocked()
            a.close()
            b.send(b"x")
            # Make sure that the wakeup has been received and everything has
            # settled before cancelling the wait_writable.
            await e.wait()
            await wait_all_tasks_blocked()
            nursery.cancel_scope.cancel()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\tensor_quant_overrides.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import onnx

from .quant_utils import QuantType


@dataclass
class QuantTypeInfo:
    """
    The quantization type information for a tensor override.
    """

    quant_type: QuantType
    symmetric: bool | None = None  # If None, assumes default is used.
    reduce_range: bool | None = None  # If None, assumes default is used.
    axis: int | None = None  # If None, assumes per-tensor quantization

    def __eq__(self, other: object):
        if isinstance(other, QuantTypeInfo):
            return (
                self.quant_type == other.quant_type
                and (self.symmetric is None or other.symmetric is None or self.symmetric == other.symmetric)
                and (self.reduce_range is None or other.reduce_range is None or self.reduce_range == other.reduce_range)
                and (self.axis == other.axis)
            )
        return NotImplemented

    @staticmethod
    def load_from_dict(
        raw_dict: dict[str, Any],
        default_qtype: QuantType | None = None,
        default_symmetric: bool | None = None,
        default_reduce_range: bool | None = None,
    ) -> QuantTypeInfo:
        return QuantTypeInfo(
            raw_dict.get("quant_type", default_qtype),
            raw_dict.get("symmetric", default_symmetric),
            raw_dict.get("reduce_range", default_reduce_range),
            raw_dict.get("axis"),
        )

    def save_to_dict(self, raw_dict: dict[str, Any]):
        raw_dict["quant_type"] = self.quant_type
        if self.symmetric is not None:
            raw_dict["symmetric"] = self.symmetric
        if self.reduce_range is not None:
            raw_dict["reduce_range"] = self.reduce_range
        if self.axis is not None:
            raw_dict["axis"] = self.axis


class TensorQuantOverridesHelper(MutableMapping):
    """
    Utility wrapper over the tensor quantization overrides passed via extra_options.
    """

    def __init__(self, raw_overrides: dict[str, list[dict[str, Any]]]):
        self.overrides = raw_overrides
        self.quant_types = None
        self.keys_unsupported_with_scale_zp = {"symmetric", "reduce_range", "rmax", "rmin"}

    def has_per_tensor_overrides(self, tensor_name: str) -> bool:
        overrides_list = self.overrides.get(tensor_name)
        return overrides_list and "axis" not in overrides_list[0]

    def has_per_channel_overrides(self, tensor_name: str) -> bool:
        overrides_list = self.overrides.get(tensor_name)
        return overrides_list and "axis" in overrides_list[0]

    def overrides_scale_zp(self, tensor_name: str) -> bool:
        overrides_list = self.overrides.get(tensor_name)
        return overrides_list and ("scale" in overrides_list[0]) and ("zero_point" in overrides_list[0])

    def get_per_tensor_overrides(
        self,
        tensor_name: str,
        default_val: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        default_list_val = [default_val] if default_val is not None else None
        overrides_list = self.overrides.get(tensor_name, default_list_val)
        if overrides_list and "axis" in overrides_list[0]:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to use per-tensor quantization overrides, "
                f"but found per-channel overrides."
            )

        return overrides_list[0] if overrides_list else None

    def get_per_channel_overrides(
        self,
        tensor_name: str,
        default_val: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        overrides_list = self.overrides.get(tensor_name, default_val)

        if not overrides_list:
            return None

        if "axis" not in overrides_list[0]:
            raise ValueError(
                f"Expected tensor '{tensor_name}' to have per-channel quantization overrides (axis value is missing).",
            )

        return overrides_list

    def get_quant_types(self) -> set[QuantType]:
        if self.quant_types is not None:
            return self.quant_types

        self.quant_types = set()

        if self.overrides:
            for quant_overrides_list in self.overrides.values():
                for quant_overrides in quant_overrides_list:
                    if "quant_type" in quant_overrides:
                        self.quant_types.add(quant_overrides["quant_type"])

                    if "convert" in quant_overrides and "quant_type" in quant_overrides["convert"]:
                        self.quant_types.add(quant_overrides["convert"]["quant_type"])

        return self.quant_types

    def _is_valid_per_tensor(
        self,
        initializers,
        default_activation_qtype,
        tensor_name: str,
        quant_overrides: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if not isinstance(quant_overrides, dict):
            return (
                False,
                f"Tensor quantization overrides for '{tensor_name}' are not in a dict",
            )

        is_initializer = tensor_name in initializers

        quant_type = quant_overrides.get("quant_type")
        if quant_type:
            self.quant_types.add(quant_type)

        has_scale = "scale" in quant_overrides
        has_zero_point = "zero_point" in quant_overrides

        if (has_scale and not has_zero_point) or (has_zero_point and not has_scale):
            return (
                False,
                "Must provide both 'scale' and 'zero_point' if one of the overrides is provided",
            )

        if has_scale:
            keys = self.keys_unsupported_with_scale_zp.intersection(set(quant_overrides))
            if keys:
                return (
                    False,
                    f"Tensor override option(s) [{', '.join(keys)}] are invalid with 'scale' and 'zero_point'",
                )

        if "reduce_range" in quant_overrides and not is_initializer:
            return (
                False,
                f"Option 'reduce_range' is only supported for initializers, not for activation {tensor_name}",
            )

        if "convert" in quant_overrides:
            if is_initializer:
                return False, "Cannot use 'convert' override for initializers"

            if "quant_type" not in quant_overrides["convert"]:
                return False, f"'convert' options (tensor '{tensor_name}') must specify a 'quant_type'"

            if "reduce_range" in quant_overrides["convert"]:
                return (
                    False,
                    f"Option 'reduce_range' is only supported for initializers, not for activation {tensor_name}",
                )

            convert_quant_type = quant_overrides["convert"]["quant_type"]
            original_quant_type = quant_type if quant_type is not None else default_activation_qtype
            if convert_quant_type == original_quant_type:
                return (
                    False,
                    f"'convert' quant_type must differ from original quant_type (tensor '{tensor_name}')",
                )

            convert_has_scale = "scale" in quant_overrides["convert"]
            convert_has_zero_point = "zero_point" in quant_overrides["convert"]

            if (convert_has_scale and not convert_has_zero_point) or (convert_has_zero_point and not convert_has_scale):
                return (
                    False,
                    f"Must provide both 'scale' and 'zero_point' if one of the overrides is provided (tensor '{tensor_name}')",
                )

            if convert_has_scale:
                keys = self.keys_unsupported_with_scale_zp.intersection(set(quant_overrides["convert"]))
                if keys:
                    return (
                        False,
                        f"Tensor override option(s) [{', '.join(keys)}] are invalid with 'scale' and 'zero_point' "
                        f"(tensor '{tensor_name}')",
                    )

            self.quant_types.add(convert_quant_type)

        return True, None

    def _is_valid_per_channel(
        self,
        initializers,
        tensor_name: str,
        quant_overrides_list: list[dict[str, Any]],
    ) -> tuple[bool, str | None]:
        is_initializer = tensor_name in initializers

        if not is_initializer:
            return (
                False,
                f"Tensor '{tensor_name}' has per-channel overrides, but is not an initializer",
            )

        axis = quant_overrides_list[0].get("axis")

        if axis is None:
            return (
                False,
                f"Per-channel overrides for tensor {tensor_name} is missing an 'axis' value in "
                "the first channel dictionary.",
            )

        weight_shape = list(initializers[tensor_name].dims)
        weight_rank = len(weight_shape)
        norm_axis = axis
        if norm_axis < 0:
            norm_axis += weight_rank

        if norm_axis < 0 or norm_axis >= len(weight_shape):
            return (
                False,
                f"Axis override value is out-of-bounds for tensor {tensor_name} (rank {len(weight_shape)})",
            )

        if len(quant_overrides_list) > 1 and len(quant_overrides_list) != weight_shape[norm_axis]:
            return (
                False,
                f"Incorrect number of channel overrides for tensor {tensor_name} (axis {axis}), "
                f"expected {weight_shape[axis]}, but found {len(quant_overrides_list)}.",
            )

        if "convert" in quant_overrides_list[0]:
            return False, f"Cannot use 'convert' override for initializers, such as {tensor_name}."

        quant_type = quant_overrides_list[0].get("quant_type")
        if quant_type:
            self.quant_types.add(quant_type)

        symmetric = quant_overrides_list[0].get("symmetric")
        reduce_range = quant_overrides_list[0].get("reduce_range")

        has_scale = "scale" in quant_overrides_list[0]
        has_zero_point = "zero_point" in quant_overrides_list[0]
        has_scale_zp = has_scale and has_zero_point

        if (has_scale and not has_zero_point) or (has_zero_point and not has_scale):
            return (
                False,
                "Must provide both 'scale' and 'zero_point' if one of the overrides is provided",
            )

        if has_scale_zp:
            keys = self.keys_unsupported_with_scale_zp.intersection(set(quant_overrides_list[0]))
            if keys:
                return (
                    False,
                    f"Tensor override option(s) [{', '.join(keys)}] are invalid with 'scale' and 'zero_point'",
                )

        has_rmin = "rmin" in quant_overrides_list[0]
        has_rmax = "rmax" in quant_overrides_list[0]
        has_rmin_rmax = has_rmin and has_rmax
        if (has_rmin and not has_rmax) or (not has_rmin and has_rmax):
            return (
                False,
                "Must provide both 'rmin' and 'rmax' if one is provided",
            )

        for index, quant_overrides in enumerate(quant_overrides_list[1:]):
            if not isinstance(quant_overrides, dict):
                return (
                    False,
                    f"Tensor quantization overrides at index {index} for '{tensor_name}' are not in a dict",
                )

            if "convert" in quant_overrides:
                return False, f"Cannot use 'convert' override for initializers, such as {tensor_name}."

            # For per-channel quantization, all channels must use the same quantization type, axis, symmetric
            # and reduce_range values. And, if specified, they must be present in the first channel dict
            # (i.e., quant_overrides_list[0]).
            if "quant_type" in quant_overrides and quant_type != quant_overrides["quant_type"]:
                return (
                    False,
                    "Channel quantization types for tensor '{tensor_name}' do not match at index {index}.",
                )
            if "axis" in quant_overrides and axis != quant_overrides["axis"] and norm_axis != quant_overrides["axis"]:
                return (
                    False,
                    "Channel axis for tensor '{tensor_name}' does not match at index {index}.",
                )
            if "symmetric" in quant_overrides and symmetric != quant_overrides["symmetric"]:
                return (
                    False,
                    "Channel symmetric value for tensor '{tensor_name}' does not match at index {index}.",
                )
            if "reduce_range" in quant_overrides and reduce_range != quant_overrides["reduce_range"]:
                return (
                    False,
                    "Channel reduce_range value for tensor '{tensor_name}' does not match at index {index}.",
                )

            # If override scale/zp, must do so for all channels.
            chan_has_scale_zp = "scale" in quant_overrides and "zero_point" in quant_overrides

            if has_scale_zp and not chan_has_scale_zp:
                return (
                    False,
                    "Per-channel overrides that specify scale/zero_point must do so for all channels, "
                    f"but tensor '{tensor_name}' is missing them at index {index}.",
                )

            if chan_has_scale_zp:
                keys = self.keys_unsupported_with_scale_zp.intersection(set(quant_overrides))
                if keys:
                    return (
                        False,
                        f"Tensor override option(s) [{', '.join(keys)}] are invalid with 'scale' and 'zero_point'",
                    )

            # If override rmin/rmax, must do so for all channels.
            chan_has_rmin_rmax = "rmin" in quant_overrides and "rmax" in quant_overrides
            if has_rmin_rmax and not chan_has_rmin_rmax:
                return (
                    False,
                    "Per-channel overrides that specify rmin/rmax must do so for all channels, "
                    f"but tensor '{tensor_name}' is missing them at index {index}.",
                )

        return True, None

    def is_valid(
        self,
        initializers: dict[str, onnx.TensorProto],
        activation_names: set[str],
        default_activation_qtype,
    ) -> tuple[bool, str | None]:
        self.quant_types = set()

        # Validate that compatible/valid overrides are provided.
        if self.overrides:
            for tensor_name, quant_overrides_list in self.overrides.items():
                if tensor_name not in initializers and tensor_name not in activation_names:
                    return False, f"Tensor '{tensor_name}' in TensorQuantOverrides is not present in the model"

                if not isinstance(quant_overrides_list, list):
                    return False, f"Tensor quantization overrides for '{tensor_name}' are not in a list"

                if not quant_overrides_list:
                    continue

                if not isinstance(quant_overrides_list[0], dict):
                    return False, f"Tensor quantization overrides at index 0 for '{tensor_name}' are not in a dict"

                if not quant_overrides_list[0]:
                    continue

                axis = quant_overrides_list[0].get("axis")
                is_per_channel = len(quant_overrides_list) > 1 or axis is not None

                if is_per_channel:
                    return self._is_valid_per_channel(initializers, tensor_name, quant_overrides_list)

                return self._is_valid_per_tensor(
                    initializers, default_activation_qtype, tensor_name, quant_overrides_list[0]
                )

        return True, None

    def update_tensor_overrides(
        self,
        tensor_name: str,
        new_vals: dict[str, Any],
        channels: list[int] | None = None,
        overwrite: bool = True,
    ) -> bool:
        if not new_vals:
            return False

        channels = set(channels) if channels is not None else None
        have_overrides = self.overrides.get(tensor_name)

        # If `overwrite` is False, check if we would overwrite anything.
        do_update = True
        if not overwrite and have_overrides:
            for channel, overrides in enumerate(self.overrides[tensor_name]):
                if channels is not None and channel not in channels:
                    continue
                if set(new_vals).intersection(set(overrides)):
                    do_update = False
                    break

        # Do the update if `overwrite` is True or if nothing is overwritten (do not want partial overwrites).
        if do_update:
            if not have_overrides:
                self.overrides[tensor_name] = [{}]

            for channel, overrides in enumerate(self.overrides[tensor_name]):
                if channels is not None and channel not in channels:
                    continue
                overrides.update(new_vals)

        return do_update

    def get_node_output_qtype_info(
        self,
        output_name: str,
        default_qtype: QuantType | None,
        default_symmetric: bool | None = None,
    ) -> QuantTypeInfo:
        # Outputs are activations, which do not support 'reduce_range' or 'axis'
        if output_name not in self.overrides:
            return QuantTypeInfo(default_qtype, default_symmetric)

        tensor_overrides = self.overrides[output_name][0]

        return QuantTypeInfo(
            tensor_overrides.get("quant_type", default_qtype),
            tensor_overrides.get("symmetric", default_symmetric),
        )

    def get_node_input_qtype_info(
        self,
        input_name: str,
        node_name: str,
        default_qtype: QuantType | None,
        default_symmetric: bool | None = None,
        default_reduce_range: bool | None = None,
    ) -> QuantTypeInfo:
        if input_name not in self.overrides or not self.overrides[input_name]:
            return QuantTypeInfo(default_qtype, default_symmetric, default_reduce_range)

        # Get the first overrides dict in the list. This works for both per-tensor and per-channel
        # quantization because all channels must use the same quant type.
        tensor_overrides = self.overrides[input_name][0]
        producer_type = tensor_overrides.get("quant_type", default_qtype)

        if "convert" not in tensor_overrides:
            return QuantTypeInfo(
                producer_type,
                tensor_overrides.get("symmetric", default_symmetric),
                tensor_overrides.get("reduce_range", default_reduce_range),
                tensor_overrides.get("axis"),
            )

        # This tensor is converted. Check if the node gets the original qtype or the converted qtype.
        convert_dict = tensor_overrides["convert"]
        qtype_info = QuantTypeInfo(
            producer_type,
            convert_dict.get("symmetric", default_symmetric),
            # Converted tensors are not initializers, so do not have 'axis' or 'reduce_range'.
        )

        # Check if all nodes receive the converted type (i.e., recv_nodes is None) or this node
        # is in the list of consumers (recv_nodes).
        if ("recv_nodes" not in convert_dict) or (node_name in convert_dict["recv_nodes"]):
            qtype_info.quant_type = convert_dict["quant_type"]

        return qtype_info

    def pprint_str(self, indent=None) -> str:
        return json.dumps(self.overrides, default=str, indent=indent)

    def empty(self) -> bool:
        return not self.overrides

    def get_dict(self) -> dict[str, list[dict[str, Any]]]:
        return self.overrides

    # Required implementations of abstract methods in collections.abc.MutableMapping
    # so that this class can be used like a dict.
    def __setitem__(self, key: str, value: list[dict]):
        self.overrides[key] = value

    def __getitem__(self, key: str) -> list[dict]:
        return self.overrides[key]

    def __delitem__(self, key: str):
        del self.overrides[key]

    def __iter__(self):
        return iter(self.overrides)

    def __len__(self):
        return len(self.overrides)

    def __str__(self) -> str:
        return str(self.overrides)

    def __repr__(self) -> str:
        return f"{super().__repr__()}, TensorQuantOverridesHelper({self.overrides})"