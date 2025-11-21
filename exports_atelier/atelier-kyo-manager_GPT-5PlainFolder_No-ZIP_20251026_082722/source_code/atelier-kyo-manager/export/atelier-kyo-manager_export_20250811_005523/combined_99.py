
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\exceptions.py ===
"""Implements a number of Python exceptions which can be raised from within
a view to trigger a standard HTTP non-200 response.

Usage Example
-------------

.. code-block:: python

    from werkzeug.wrappers.request import Request
    from werkzeug.exceptions import HTTPException, NotFound

    def view(request):
        raise NotFound()

    @Request.application
    def application(request):
        try:
            return view(request)
        except HTTPException as e:
            return e

As you can see from this example those exceptions are callable WSGI
applications. However, they are not Werkzeug response objects. You
can get a response object by calling ``get_response()`` on a HTTP
exception.

Keep in mind that you may have to pass an environ (WSGI) or scope
(ASGI) to ``get_response()`` because some errors fetch additional
information relating to the request.

If you want to hook in a different exception page to say, a 404 status
code, you can add a second except for a specific subclass of an error:

.. code-block:: python

    @Request.application
    def application(request):
        try:
            return view(request)
        except NotFound as e:
            return not_found(request)
        except HTTPException as e:
            return e

"""

from __future__ import annotations

import typing as t
from datetime import datetime

from markupsafe import escape
from markupsafe import Markup

from ._internal import _get_environ

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIEnvironment

    from .datastructures import WWWAuthenticate
    from .sansio.response import Response
    from .wrappers.request import Request as WSGIRequest
    from .wrappers.response import Response as WSGIResponse


class HTTPException(Exception):
    """The base class for all HTTP exceptions. This exception can be called as a WSGI
    application to render a default error page or you can catch the subclasses
    of it independently and render nicer error messages.

    .. versionchanged:: 2.1
        Removed the ``wrap`` class method.
    """

    code: int | None = None
    description: str | None = None

    def __init__(
        self,
        description: str | None = None,
        response: Response | None = None,
    ) -> None:
        super().__init__()
        if description is not None:
            self.description = description
        self.response = response

    @property
    def name(self) -> str:
        """The status name."""
        from .http import HTTP_STATUS_CODES

        return HTTP_STATUS_CODES.get(self.code, "Unknown Error")  # type: ignore

    def get_description(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> str:
        """Get the description."""
        if self.description is None:
            description = ""
        else:
            description = self.description

        description = escape(description).replace("\n", Markup("<br>"))
        return f"<p>{description}</p>"

    def get_body(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> str:
        """Get the HTML body."""
        return (
            "<!doctype html>\n"
            "<html lang=en>\n"
            f"<title>{self.code} {escape(self.name)}</title>\n"
            f"<h1>{escape(self.name)}</h1>\n"
            f"{self.get_description(environ)}\n"
        )

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        """Get a list of headers."""
        return [("Content-Type", "text/html; charset=utf-8")]

    def get_response(
        self,
        environ: WSGIEnvironment | WSGIRequest | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> Response:
        """Get a response object.  If one was passed to the exception
        it's returned directly.

        :param environ: the optional environ for the request.  This
                        can be used to modify the response depending
                        on how the request looked like.
        :return: a :class:`Response` object or a subclass thereof.
        """
        from .wrappers.response import Response as WSGIResponse  # noqa: F811

        if self.response is not None:
            return self.response
        if environ is not None:
            environ = _get_environ(environ)
        headers = self.get_headers(environ, scope)
        return WSGIResponse(self.get_body(environ, scope), self.code, headers)

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        """Call the exception as WSGI application.

        :param environ: the WSGI environment.
        :param start_response: the response callable provided by the WSGI
                               server.
        """
        response = t.cast("WSGIResponse", self.get_response(environ))
        return response(environ, start_response)

    def __str__(self) -> str:
        code = self.code if self.code is not None else "???"
        return f"{code} {self.name}: {self.description}"

    def __repr__(self) -> str:
        code = self.code if self.code is not None else "???"
        return f"<{type(self).__name__} '{code}: {self.name}'>"


class BadRequest(HTTPException):
    """*400* `Bad Request`

    Raise if the browser sends something to the application the application
    or server cannot handle.
    """

    code = 400
    description = (
        "The browser (or proxy) sent a request that this server could "
        "not understand."
    )


class BadRequestKeyError(BadRequest, KeyError):
    """An exception that is used to signal both a :exc:`KeyError` and a
    :exc:`BadRequest`. Used by many of the datastructures.
    """

    _description = BadRequest.description
    #: Show the KeyError along with the HTTP error message in the
    #: response. This should be disabled in production, but can be
    #: useful in a debug mode.
    show_exception = False

    def __init__(self, arg: object | None = None, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)

        if arg is None:
            KeyError.__init__(self)
        else:
            KeyError.__init__(self, arg)

    @property  # type: ignore
    def description(self) -> str:
        if self.show_exception:
            return (
                f"{self._description}\n"
                f"{KeyError.__name__}: {KeyError.__str__(self)}"
            )

        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = value


class ClientDisconnected(BadRequest):
    """Internal exception that is raised if Werkzeug detects a disconnected
    client.  Since the client is already gone at that point attempting to
    send the error message to the client might not work and might ultimately
    result in another exception in the server.  Mainly this is here so that
    it is silenced by default as far as Werkzeug is concerned.

    Since disconnections cannot be reliably detected and are unspecified
    by WSGI to a large extent this might or might not be raised if a client
    is gone.

    .. versionadded:: 0.8
    """


class SecurityError(BadRequest):
    """Raised if something triggers a security error.  This is otherwise
    exactly like a bad request error.

    .. versionadded:: 0.9
    """


class BadHost(BadRequest):
    """Raised if the submitted host is badly formatted.

    .. versionadded:: 0.11.2
    """


class Unauthorized(HTTPException):
    """*401* ``Unauthorized``

    Raise if the user is not authorized to access a resource.

    The ``www_authenticate`` argument should be used to set the
    ``WWW-Authenticate`` header. This is used for HTTP basic auth and
    other schemes. Use :class:`~werkzeug.datastructures.WWWAuthenticate`
    to create correctly formatted values. Strictly speaking a 401
    response is invalid if it doesn't provide at least one value for
    this header, although real clients typically don't care.

    :param description: Override the default message used for the body
        of the response.
    :param www-authenticate: A single value, or list of values, for the
        WWW-Authenticate header(s).

    .. versionchanged:: 2.0
        Serialize multiple ``www_authenticate`` items into multiple
        ``WWW-Authenticate`` headers, rather than joining them
        into a single value, for better interoperability.

    .. versionchanged:: 0.15.3
        If the ``www_authenticate`` argument is not set, the
        ``WWW-Authenticate`` header is not set.

    .. versionchanged:: 0.15.3
        The ``response`` argument was restored.

    .. versionchanged:: 0.15.1
        ``description`` was moved back as the first argument, restoring
         its previous position.

    .. versionchanged:: 0.15.0
        ``www_authenticate`` was added as the first argument, ahead of
        ``description``.
    """

    code = 401
    description = (
        "The server could not verify that you are authorized to access"
        " the URL requested. You either supplied the wrong credentials"
        " (e.g. a bad password), or your browser doesn't understand"
        " how to supply the credentials required."
    )

    def __init__(
        self,
        description: str | None = None,
        response: Response | None = None,
        www_authenticate: None | (WWWAuthenticate | t.Iterable[WWWAuthenticate]) = None,
    ) -> None:
        super().__init__(description, response)

        from .datastructures import WWWAuthenticate

        if isinstance(www_authenticate, WWWAuthenticate):
            www_authenticate = (www_authenticate,)

        self.www_authenticate = www_authenticate

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.www_authenticate:
            headers.extend(("WWW-Authenticate", str(x)) for x in self.www_authenticate)
        return headers


class Forbidden(HTTPException):
    """*403* `Forbidden`

    Raise if the user doesn't have the permission for the requested resource
    but was authenticated.
    """

    code = 403
    description = (
        "You don't have the permission to access the requested"
        " resource. It is either read-protected or not readable by the"
        " server."
    )


class NotFound(HTTPException):
    """*404* `Not Found`

    Raise if a resource does not exist and never existed.
    """

    code = 404
    description = (
        "The requested URL was not found on the server. If you entered"
        " the URL manually please check your spelling and try again."
    )


class MethodNotAllowed(HTTPException):
    """*405* `Method Not Allowed`

    Raise if the server used a method the resource does not handle.  For
    example `POST` if the resource is view only.  Especially useful for REST.

    The first argument for this exception should be a list of allowed methods.
    Strictly speaking the response would be invalid if you don't provide valid
    methods in the header which you can do with that list.
    """

    code = 405
    description = "The method is not allowed for the requested URL."

    def __init__(
        self,
        valid_methods: t.Iterable[str] | None = None,
        description: str | None = None,
        response: Response | None = None,
    ) -> None:
        """Takes an optional list of valid http methods
        starting with werkzeug 0.3 the list will be mandatory."""
        super().__init__(description=description, response=response)
        self.valid_methods = valid_methods

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.valid_methods:
            headers.append(("Allow", ", ".join(self.valid_methods)))
        return headers


class NotAcceptable(HTTPException):
    """*406* `Not Acceptable`

    Raise if the server can't return any content conforming to the
    `Accept` headers of the client.
    """

    code = 406
    description = (
        "The resource identified by the request is only capable of"
        " generating response entities which have content"
        " characteristics not acceptable according to the accept"
        " headers sent in the request."
    )


class RequestTimeout(HTTPException):
    """*408* `Request Timeout`

    Raise to signalize a timeout.
    """

    code = 408
    description = (
        "The server closed the network connection because the browser"
        " didn't finish the request within the specified time."
    )


class Conflict(HTTPException):
    """*409* `Conflict`

    Raise to signal that a request cannot be completed because it conflicts
    with the current state on the server.

    .. versionadded:: 0.7
    """

    code = 409
    description = (
        "A conflict happened while processing the request. The"
        " resource might have been modified while the request was being"
        " processed."
    )


class Gone(HTTPException):
    """*410* `Gone`

    Raise if a resource existed previously and went away without new location.
    """

    code = 410
    description = (
        "The requested URL is no longer available on this server and"
        " there is no forwarding address. If you followed a link from a"
        " foreign page, please contact the author of this page."
    )


class LengthRequired(HTTPException):
    """*411* `Length Required`

    Raise if the browser submitted data but no ``Content-Length`` header which
    is required for the kind of processing the server does.
    """

    code = 411
    description = (
        "A request with this method requires a valid <code>Content-"
        "Length</code> header."
    )


class PreconditionFailed(HTTPException):
    """*412* `Precondition Failed`

    Status code used in combination with ``If-Match``, ``If-None-Match``, or
    ``If-Unmodified-Since``.
    """

    code = 412
    description = (
        "The precondition on the request for the URL failed positive evaluation."
    )


class RequestEntityTooLarge(HTTPException):
    """*413* `Request Entity Too Large`

    The status code one should return if the data submitted exceeded a given
    limit.
    """

    code = 413
    description = "The data value transmitted exceeds the capacity limit."


class RequestURITooLarge(HTTPException):
    """*414* `Request URI Too Large`

    Like *413* but for too long URLs.
    """

    code = 414
    description = (
        "The length of the requested URL exceeds the capacity limit for"
        " this server. The request cannot be processed."
    )


class UnsupportedMediaType(HTTPException):
    """*415* `Unsupported Media Type`

    The status code returned if the server is unable to handle the media type
    the client transmitted.
    """

    code = 415
    description = (
        "The server does not support the media type transmitted in the request."
    )


class RequestedRangeNotSatisfiable(HTTPException):
    """*416* `Requested Range Not Satisfiable`

    The client asked for an invalid part of the file.

    .. versionadded:: 0.7
    """

    code = 416
    description = "The server cannot provide the requested range."

    def __init__(
        self,
        length: int | None = None,
        units: str = "bytes",
        description: str | None = None,
        response: Response | None = None,
    ) -> None:
        """Takes an optional `Content-Range` header value based on ``length``
        parameter.
        """
        super().__init__(description=description, response=response)
        self.length = length
        self.units = units

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)
        if self.length is not None:
            headers.append(("Content-Range", f"{self.units} */{self.length}"))
        return headers


class ExpectationFailed(HTTPException):
    """*417* `Expectation Failed`

    The server cannot meet the requirements of the Expect request-header.

    .. versionadded:: 0.7
    """

    code = 417
    description = "The server could not meet the requirements of the Expect header"


class ImATeapot(HTTPException):
    """*418* `I'm a teapot`

    The server should return this if it is a teapot and someone attempted
    to brew coffee with it.

    .. versionadded:: 0.7
    """

    code = 418
    description = "This server is a teapot, not a coffee machine"


class MisdirectedRequest(HTTPException):
    """421 Misdirected Request

    Indicates that the request was directed to a server that is not able to
    produce a response.

    .. versionadded:: 3.1
    """

    code = 421
    description = "The server is not able to produce a response."


class UnprocessableEntity(HTTPException):
    """*422* `Unprocessable Entity`

    Used if the request is well formed, but the instructions are otherwise
    incorrect.
    """

    code = 422
    description = (
        "The request was well-formed but was unable to be followed due"
        " to semantic errors."
    )


class Locked(HTTPException):
    """*423* `Locked`

    Used if the resource that is being accessed is locked.
    """

    code = 423
    description = "The resource that is being accessed is locked."


class FailedDependency(HTTPException):
    """*424* `Failed Dependency`

    Used if the method could not be performed on the resource
    because the requested action depended on another action and that action failed.
    """

    code = 424
    description = (
        "The method could not be performed on the resource because the"
        " requested action depended on another action and that action"
        " failed."
    )


class PreconditionRequired(HTTPException):
    """*428* `Precondition Required`

    The server requires this request to be conditional, typically to prevent
    the lost update problem, which is a race condition between two or more
    clients attempting to update a resource through PUT or DELETE. By requiring
    each client to include a conditional header ("If-Match" or "If-Unmodified-
    Since") with the proper value retained from a recent GET request, the
    server ensures that each client has at least seen the previous revision of
    the resource.
    """

    code = 428
    description = (
        "This request is required to be conditional; try using"
        ' "If-Match" or "If-Unmodified-Since".'
    )


class _RetryAfter(HTTPException):
    """Adds an optional ``retry_after`` parameter which will set the
    ``Retry-After`` header. May be an :class:`int` number of seconds or
    a :class:`~datetime.datetime`.
    """

    def __init__(
        self,
        description: str | None = None,
        response: Response | None = None,
        retry_after: datetime | int | None = None,
    ) -> None:
        super().__init__(description, response)
        self.retry_after = retry_after

    def get_headers(
        self,
        environ: WSGIEnvironment | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> list[tuple[str, str]]:
        headers = super().get_headers(environ, scope)

        if self.retry_after:
            if isinstance(self.retry_after, datetime):
                from .http import http_date

                value = http_date(self.retry_after)
            else:
                value = str(self.retry_after)

            headers.append(("Retry-After", value))

        return headers


class TooManyRequests(_RetryAfter):
    """*429* `Too Many Requests`

    The server is limiting the rate at which this user receives
    responses, and this request exceeds that rate. (The server may use
    any convenient method to identify users and their request rates).
    The server may include a "Retry-After" header to indicate how long
    the user should wait before retrying.

    :param retry_after: If given, set the ``Retry-After`` header to this
        value. May be an :class:`int` number of seconds or a
        :class:`~datetime.datetime`.

    .. versionchanged:: 1.0
        Added ``retry_after`` parameter.
    """

    code = 429
    description = "This user has exceeded an allotted request count. Try again later."


class RequestHeaderFieldsTooLarge(HTTPException):
    """*431* `Request Header Fields Too Large`

    The server refuses to process the request because the header fields are too
    large. One or more individual fields may be too large, or the set of all
    headers is too large.
    """

    code = 431
    description = "One or more header fields exceeds the maximum size."


class UnavailableForLegalReasons(HTTPException):
    """*451* `Unavailable For Legal Reasons`

    This status code indicates that the server is denying access to the
    resource as a consequence of a legal demand.
    """

    code = 451
    description = "Unavailable for legal reasons."


class InternalServerError(HTTPException):
    """*500* `Internal Server Error`

    Raise if an internal server error occurred.  This is a good fallback if an
    unknown error occurred in the dispatcher.

    .. versionchanged:: 1.0.0
        Added the :attr:`original_exception` attribute.
    """

    code = 500
    description = (
        "The server encountered an internal error and was unable to"
        " complete your request. Either the server is overloaded or"
        " there is an error in the application."
    )

    def __init__(
        self,
        description: str | None = None,
        response: Response | None = None,
        original_exception: BaseException | None = None,
    ) -> None:
        #: The original exception that caused this 500 error. Can be
        #: used by frameworks to provide context when handling
        #: unexpected errors.
        self.original_exception = original_exception
        super().__init__(description=description, response=response)


class NotImplemented(HTTPException):
    """*501* `Not Implemented`

    Raise if the application does not support the action requested by the
    browser.
    """

    code = 501
    description = "The server does not support the action requested by the browser."


class BadGateway(HTTPException):
    """*502* `Bad Gateway`

    If you do proxying in your application you should return this status code
    if you received an invalid response from the upstream server it accessed
    in attempting to fulfill the request.
    """

    code = 502
    description = (
        "The proxy server received an invalid response from an upstream server."
    )


class ServiceUnavailable(_RetryAfter):
    """*503* `Service Unavailable`

    Status code you should return if a service is temporarily
    unavailable.

    :param retry_after: If given, set the ``Retry-After`` header to this
        value. May be an :class:`int` number of seconds or a
        :class:`~datetime.datetime`.

    .. versionchanged:: 1.0
        Added ``retry_after`` parameter.
    """

    code = 503
    description = (
        "The server is temporarily unable to service your request due"
        " to maintenance downtime or capacity problems. Please try"
        " again later."
    )


class GatewayTimeout(HTTPException):
    """*504* `Gateway Timeout`

    Status code you should return if a connection to an upstream server
    times out.
    """

    code = 504
    description = "The connection to an upstream server timed out."


class HTTPVersionNotSupported(HTTPException):
    """*505* `HTTP Version Not Supported`

    The server does not support the HTTP protocol version used in the request.
    """

    code = 505
    description = (
        "The server does not support the HTTP protocol version used in the request."
    )


default_exceptions: dict[int, type[HTTPException]] = {}


def _find_exceptions() -> None:
    for obj in globals().values():
        try:
            is_http_exception = issubclass(obj, HTTPException)
        except TypeError:
            is_http_exception = False
        if not is_http_exception or obj.code is None:
            continue
        old_obj = default_exceptions.get(obj.code, None)
        if old_obj is not None and issubclass(obj, old_obj):
            continue
        default_exceptions[obj.code] = obj


_find_exceptions()
del _find_exceptions


class Aborter:
    """When passed a dict of code -> exception items it can be used as
    callable that raises exceptions.  If the first argument to the
    callable is an integer it will be looked up in the mapping, if it's
    a WSGI application it will be raised in a proxy exception.

    The rest of the arguments are forwarded to the exception constructor.
    """

    def __init__(
        self,
        mapping: dict[int, type[HTTPException]] | None = None,
        extra: dict[int, type[HTTPException]] | None = None,
    ) -> None:
        if mapping is None:
            mapping = default_exceptions
        self.mapping = dict(mapping)
        if extra is not None:
            self.mapping.update(extra)

    def __call__(
        self, code: int | Response, *args: t.Any, **kwargs: t.Any
    ) -> t.NoReturn:
        from .sansio.response import Response

        if isinstance(code, Response):
            raise HTTPException(response=code)

        if code not in self.mapping:
            raise LookupError(f"no exception for {code!r}")

        raise self.mapping[code](*args, **kwargs)


def abort(status: int | Response, *args: t.Any, **kwargs: t.Any) -> t.NoReturn:
    """Raises an :py:exc:`HTTPException` for the given status code or WSGI
    application.

    If a status code is given, it will be looked up in the list of
    exceptions and will raise that exception.  If passed a WSGI application,
    it will wrap it in a proxy WSGI exception and raise that::

       abort(404)  # 404 Not Found
       abort(Response('Hello World'))

    """
    _aborter(status, *args, **kwargs)


_aborter: Aborter = Aborter()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\riccati.py ===
r"""
This module contains :py:meth:`~sympy.solvers.ode.riccati.solve_riccati`,
a function which gives all rational particular solutions to first order
Riccati ODEs. A general first order Riccati ODE is given by -

.. math:: y' = b_0(x) + b_1(x)w + b_2(x)w^2

where `b_0, b_1` and `b_2` can be arbitrary rational functions of `x`
with `b_2 \ne 0`. When `b_2 = 0`, the equation is not a Riccati ODE
anymore and becomes a Linear ODE. Similarly, when `b_0 = 0`, the equation
is a Bernoulli ODE. The algorithm presented below can find rational
solution(s) to all ODEs with `b_2 \ne 0` that have a rational solution,
or prove that no rational solution exists for the equation.

Background
==========

A Riccati equation can be transformed to its normal form

.. math:: y' + y^2 = a(x)

using the transformation

.. math:: y = -b_2(x) - \frac{b'_2(x)}{2 b_2(x)} - \frac{b_1(x)}{2}

where `a(x)` is given by

.. math:: a(x) = \frac{1}{4}\left(\frac{b_2'}{b_2} + b_1\right)^2 - \frac{1}{2}\left(\frac{b_2'}{b_2} + b_1\right)' - b_0 b_2

Thus, we can develop an algorithm to solve for the Riccati equation
in its normal form, which would in turn give us the solution for
the original Riccati equation.

Algorithm
=========

The algorithm implemented here is presented in the Ph.D thesis
"Rational and Algebraic Solutions of First-Order Algebraic ODEs"
by N. Thieu Vo. The entire thesis can be found here -
https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf

We have only implemented the Rational Riccati solver (Algorithm 11,
Pg 78-82 in Thesis). Before we proceed towards the implementation
of the algorithm, a few definitions to understand are -

1. Valuation of a Rational Function at `\infty`:
    The valuation of a rational function `p(x)` at `\infty` is equal
    to the difference between the degree of the denominator and the
    numerator of `p(x)`.

    NOTE: A general definition of valuation of a rational function
    at any value of `x` can be found in Pg 63 of the thesis, but
    is not of any interest for this algorithm.

2. Zeros and Poles of a Rational Function:
    Let `a(x) = \frac{S(x)}{T(x)}, T \ne 0` be a rational function
    of `x`. Then -

    a. The Zeros of `a(x)` are the roots of `S(x)`.
    b. The Poles of `a(x)` are the roots of `T(x)`. However, `\infty`
    can also be a pole of a(x). We say that `a(x)` has a pole at
    `\infty` if `a(\frac{1}{x})` has a pole at 0.

Every pole is associated with an order that is equal to the multiplicity
of its appearance as a root of `T(x)`. A pole is called a simple pole if
it has an order 1. Similarly, a pole is called a multiple pole if it has
an order `\ge` 2.

Necessary Conditions
====================

For a Riccati equation in its normal form,

.. math:: y' + y^2 = a(x)

we can define

a. A pole is called a movable pole if it is a pole of `y(x)` and is not
a pole of `a(x)`.
b. Similarly, a pole is called a non-movable pole if it is a pole of both
`y(x)` and `a(x)`.

Then, the algorithm states that a rational solution exists only if -

a. Every pole of `a(x)` must be either a simple pole or a multiple pole
of even order.
b. The valuation of `a(x)` at `\infty` must be even or be `\ge` 2.

This algorithm finds all possible rational solutions for the Riccati ODE.
If no rational solutions are found, it means that no rational solutions
exist.

The algorithm works for Riccati ODEs where the coefficients are rational
functions in the independent variable `x` with rational number coefficients
i.e. in `Q(x)`. The coefficients in the rational function cannot be floats,
irrational numbers, symbols or any other kind of expression. The reasons
for this are -

1. When using symbols, different symbols could take the same value and this
would affect the multiplicity of poles if symbols are present here.

2. An integer degree bound is required to calculate a polynomial solution
to an auxiliary differential equation, which in turn gives the particular
solution for the original ODE. If symbols/floats/irrational numbers are
present, we cannot determine if the expression for the degree bound is an
integer or not.

Solution
========

With these definitions, we can state a general form for the solution of
the equation. `y(x)` must have the form -

.. math:: y(x) = \sum_{i=1}^{n} \sum_{j=1}^{r_i} \frac{c_{ij}}{(x - x_i)^j} + \sum_{i=1}^{m} \frac{1}{x - \chi_i} + \sum_{i=0}^{N} d_i x^i

where `x_1, x_2, \dots, x_n` are non-movable poles of `a(x)`,
`\chi_1, \chi_2, \dots, \chi_m` are movable poles of `a(x)`, and the values
of `N, n, r_1, r_2, \dots, r_n` can be determined from `a(x)`. The
coefficient vectors `(d_0, d_1, \dots, d_N)` and `(c_{i1}, c_{i2}, \dots, c_{i r_i})`
can be determined from `a(x)`. We will have 2 choices each of these vectors
and part of the procedure is figuring out which of the 2 should be used
to get the solution correctly.

Implementation
==============

In this implementation, we use ``Poly`` to represent a rational function
rather than using ``Expr`` since ``Poly`` is much faster. Since we cannot
represent rational functions directly using ``Poly``, we instead represent
a rational function with 2 ``Poly`` objects - one for its numerator and
the other for its denominator.

The code is written to match the steps given in the thesis (Pg 82)

Step 0 : Match the equation -
Find `b_0, b_1` and `b_2`. If `b_2 = 0` or no such functions exist, raise
an error

Step 1 : Transform the equation to its normal form as explained in the
theory section.

Step 2 : Initialize an empty set of solutions, ``sol``.

Step 3 : If `a(x) = 0`, append `\frac{1}/{(x - C1)}` to ``sol``.

Step 4 : If `a(x)` is a rational non-zero number, append `\pm \sqrt{a}`
to ``sol``.

Step 5 : Find the poles and their multiplicities of `a(x)`. Let
the number of poles be `n`. Also find the valuation of `a(x)` at
`\infty` using ``val_at_inf``.

NOTE: Although the algorithm considers `\infty` as a pole, it is
not mentioned if it a part of the set of finite poles. `\infty`
is NOT a part of the set of finite poles. If a pole exists at
`\infty`, we use its multiplicity to find the laurent series of
`a(x)` about `\infty`.

Step 6 : Find `n` c-vectors (one for each pole) and 1 d-vector using
``construct_c`` and ``construct_d``. Now, determine all the ``2**(n + 1)``
combinations of choosing between 2 choices for each of the `n` c-vectors
and 1 d-vector.

NOTE: The equation for `d_{-1}` in Case 4 (Pg 80) has a printinig
mistake. The term `- d_N` must be replaced with `-N d_N`. The same
has been explained in the code as well.

For each of these above combinations, do

Step 8 : Compute `m` in ``compute_m_ybar``. `m` is the degree bound of
the polynomial solution we must find for the auxiliary equation.

Step 9 : In ``compute_m_ybar``, compute ybar as well where ``ybar`` is
one part of y(x) -

.. math:: \overline{y}(x) = \sum_{i=1}^{n} \sum_{j=1}^{r_i} \frac{c_{ij}}{(x - x_i)^j} + \sum_{i=0}^{N} d_i x^i

Step 10 : If `m` is a non-negative integer -

Step 11: Find a polynomial solution of degree `m` for the auxiliary equation.

There are 2 cases possible -

    a. `m` is a non-negative integer: We can solve for the coefficients
    in `p(x)` using Undetermined Coefficients.

    b. `m` is not a non-negative integer: In this case, we cannot find
    a polynomial solution to the auxiliary equation, and hence, we ignore
    this value of `m`.

Step 12 : For each `p(x)` that exists, append `ybar + \frac{p'(x)}{p(x)}`
to ``sol``.

Step 13 : For each solution in ``sol``, apply an inverse transformation,
so that the solutions of the original equation are found using the
solutions of the equation in its normal form.
"""


from itertools import product
from sympy.core import S
from sympy.core.add import Add
from sympy.core.numbers import oo, Float
from sympy.core.function import count_ops
from sympy.core.relational import Eq
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions import sqrt, exp
from sympy.functions.elementary.complexes import sign
from sympy.integrals.integrals import Integral
from sympy.polys.domains import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyroots import roots
from sympy.solvers.solveset import linsolve


def riccati_normal(w, x, b1, b2):
    """
    Given a solution `w(x)` to the equation

    .. math:: w'(x) = b_0(x) + b_1(x)*w(x) + b_2(x)*w(x)^2

    and rational function coefficients `b_1(x)` and
    `b_2(x)`, this function transforms the solution to
    give a solution `y(x)` for its corresponding normal
    Riccati ODE

    .. math:: y'(x) + y(x)^2 = a(x)

    using the transformation

    .. math:: y(x) = -b_2(x)*w(x) - b'_2(x)/(2*b_2(x)) - b_1(x)/2
    """
    return -b2*w - b2.diff(x)/(2*b2) - b1/2


def riccati_inverse_normal(y, x, b1, b2, bp=None):
    """
    Inverse transforming the solution to the normal
    Riccati ODE to get the solution to the Riccati ODE.
    """
    # bp is the expression which is independent of the solution
    # and hence, it need not be computed again
    if bp is None:
        bp = -b2.diff(x)/(2*b2**2) - b1/(2*b2)
    # w(x) = -y(x)/b2(x) - b2'(x)/(2*b2(x)^2) - b1(x)/(2*b2(x))
    return -y/b2 + bp


def riccati_reduced(eq, f, x):
    """
    Convert a Riccati ODE into its corresponding
    normal Riccati ODE.
    """
    match, funcs = match_riccati(eq, f, x)
    # If equation is not a Riccati ODE, exit
    if not match:
        return False
    # Using the rational functions, find the expression for a(x)
    b0, b1, b2 = funcs
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - \
        b2.diff(x, 2)/(2*b2)
    # Normal form of Riccati ODE is f'(x) + f(x)^2 = a(x)
    return f(x).diff(x) + f(x)**2 - a

def linsolve_dict(eq, syms):
    """
    Get the output of linsolve as a dict
    """
    # Convert tuple type return value of linsolve
    # to a dictionary for ease of use
    sol = linsolve(eq, syms)
    if not sol:
        return {}
    return dict(zip(syms, list(sol)[0]))


def match_riccati(eq, f, x):
    """
    A function that matches and returns the coefficients
    if an equation is a Riccati ODE

    Parameters
    ==========

    eq: Equation to be matched
    f: Dependent variable
    x: Independent variable

    Returns
    =======

    match: True if equation is a Riccati ODE, False otherwise
    funcs: [b0, b1, b2] if match is True, [] otherwise. Here,
    b0, b1 and b2 are rational functions which match the equation.
    """
    # Group terms based on f(x)
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs
    eq = eq.expand().collect(f(x))
    cf = eq.coeff(f(x).diff(x))

    # There must be an f(x).diff(x) term.
    # eq must be an Add object since we are using the expanded
    # equation and it must have atleast 2 terms (b2 != 0)
    if cf != 0 and isinstance(eq, Add):

        # Divide all coefficients by the coefficient of f(x).diff(x)
        # and add the terms again to get the same equation
        eq = Add(*((x/cf).cancel() for x in eq.args)).collect(f(x))

        # Match the equation with the pattern
        b1 = -eq.coeff(f(x))
        b2 = -eq.coeff(f(x)**2)
        b0 = (f(x).diff(x) - b1*f(x) - b2*f(x)**2 - eq).expand()
        funcs = [b0, b1, b2]

        # Check if coefficients are not symbols and floats
        if any(len(x.atoms(Symbol)) > 1 or len(x.atoms(Float)) for x in funcs):
            return False, []

        # If b_0(x) contains f(x), it is not a Riccati ODE
        if len(b0.atoms(f)) or not all((b2 != 0, b0.is_rational_function(x),
            b1.is_rational_function(x), b2.is_rational_function(x))):
            return False, []
        return True, funcs
    return False, []


def val_at_inf(num, den, x):
    # Valuation of a rational function at oo = deg(denom) - deg(numer)
    return den.degree(x) - num.degree(x)


def check_necessary_conds(val_inf, muls):
    """
    The necessary conditions for a rational solution
    to exist are as follows -

    i) Every pole of a(x) must be either a simple pole
    or a multiple pole of even order.

    ii) The valuation of a(x) at infinity must be even
    or be greater than or equal to 2.

    Here, a simple pole is a pole with multiplicity 1
    and a multiple pole is a pole with multiplicity
    greater than 1.
    """
    return (val_inf >= 2 or (val_inf <= 0 and val_inf%2 == 0)) and \
        all(mul == 1 or (mul%2 == 0 and mul >= 2) for mul in muls)


def inverse_transform_poly(num, den, x):
    """
    A function to make the substitution
    x -> 1/x in a rational function that
    is represented using Poly objects for
    numerator and denominator.
    """
    # Declare for reuse
    one = Poly(1, x)
    xpoly = Poly(x, x)

    # Check if degree of numerator is same as denominator
    pwr = val_at_inf(num, den, x)
    if pwr >= 0:
        # Denominator has greater degree. Substituting x with
        # 1/x would make the extra power go to the numerator
        if num.expr != 0:
            num = num.transform(one, xpoly) * x**pwr
            den = den.transform(one, xpoly)
    else:
        # Numerator has greater degree. Substituting x with
        # 1/x would make the extra power go to the denominator
        num = num.transform(one, xpoly)
        den = den.transform(one, xpoly) * x**(-pwr)
    return num.cancel(den, include=True)


def limit_at_inf(num, den, x):
    """
    Find the limit of a rational function
    at oo
    """
    # pwr = degree(num) - degree(den)
    pwr = -val_at_inf(num, den, x)
    # Numerator has a greater degree than denominator
    # Limit at infinity would depend on the sign of the
    # leading coefficients of numerator and denominator
    if pwr > 0:
        return oo*sign(num.LC()/den.LC())
    # Degree of numerator is equal to that of denominator
    # Limit at infinity is just the ratio of leading coeffs
    elif pwr == 0:
        return num.LC()/den.LC()
    # Degree of numerator is less than that of denominator
    # Limit at infinity is just 0
    else:
        return 0


def construct_c_case_1(num, den, x, pole):
    # Find the coefficient of 1/(x - pole)**2 in the
    # Laurent series expansion of a(x) about pole.
    num1, den1 = (num*Poly((x - pole)**2, x, extension=True)).cancel(den, include=True)
    r = (num1.subs(x, pole))/(den1.subs(x, pole))

    # If multiplicity is 2, the coefficient to be added
    # in the c-vector is c = (1 +- sqrt(1 + 4*r))/2
    if r != -S(1)/4:
        return [[(1 + sqrt(1 + 4*r))/2], [(1 - sqrt(1 + 4*r))/2]]
    return [[S.Half]]


def construct_c_case_2(num, den, x, pole, mul):
    # Generate the coefficients using the recurrence
    # relation mentioned in (5.14) in the thesis (Pg 80)

    # r_i = mul/2
    ri = mul//2

    # Find the Laurent series coefficients about the pole
    ser = rational_laurent_series(num, den, x, pole, mul, 6)

    # Start with an empty memo to store the coefficients
    # This is for the plus case
    cplus = [0 for i in range(ri)]

    # Base Case
    cplus[ri-1] = sqrt(ser[2*ri])

    # Iterate backwards to find all coefficients
    s = ri - 1
    sm = 0
    for s in range(ri-1, 0, -1):
        sm = 0
        for j in range(s+1, ri):
            sm += cplus[j-1]*cplus[ri+s-j-1]
        if s!= 1:
            cplus[s-1] = (ser[ri+s] - sm)/(2*cplus[ri-1])

    # Memo for the minus case
    cminus = [-x for x in cplus]

    # Find the 0th coefficient in the recurrence
    cplus[0] = (ser[ri+s] - sm - ri*cplus[ri-1])/(2*cplus[ri-1])
    cminus[0] = (ser[ri+s] - sm  - ri*cminus[ri-1])/(2*cminus[ri-1])

    # Add both the plus and minus cases' coefficients
    if cplus != cminus:
        return [cplus, cminus]
    return cplus


def construct_c_case_3():
    # If multiplicity is 1, the coefficient to be added
    # in the c-vector is 1 (no choice)
    return [[1]]


def construct_c(num, den, x, poles, muls):
    """
    Helper function to calculate the coefficients
    in the c-vector for each pole.
    """
    c = []
    for pole, mul in zip(poles, muls):
        c.append([])

        # Case 3
        if mul == 1:
            # Add the coefficients from Case 3
            c[-1].extend(construct_c_case_3())

        # Case 1
        elif mul == 2:
            # Add the coefficients from Case 1
            c[-1].extend(construct_c_case_1(num, den, x, pole))

        # Case 2
        else:
            # Add the coefficients from Case 2
            c[-1].extend(construct_c_case_2(num, den, x, pole, mul))

    return c


def construct_d_case_4(ser, N):
    # Initialize an empty vector
    dplus = [0 for i in range(N+2)]
    # d_N = sqrt(a_{2*N})
    dplus[N] = sqrt(ser[2*N])

    # Use the recurrence relations to find
    # the value of d_s
    for s in range(N-1, -2, -1):
        sm = 0
        for j in range(s+1, N):
            sm += dplus[j]*dplus[N+s-j]
        if s != -1:
            dplus[s] = (ser[N+s] - sm)/(2*dplus[N])

    # Coefficients for the case of d_N = -sqrt(a_{2*N})
    dminus = [-x for x in dplus]

    # The third equation in Eq 5.15 of the thesis is WRONG!
    # d_N must be replaced with N*d_N in that equation.
    dplus[-1] = (ser[N+s] - N*dplus[N] - sm)/(2*dplus[N])
    dminus[-1] = (ser[N+s] - N*dminus[N] - sm)/(2*dminus[N])

    if dplus != dminus:
        return [dplus, dminus]
    return dplus


def construct_d_case_5(ser):
    # List to store coefficients for plus case
    dplus = [0, 0]

    # d_0  = sqrt(a_0)
    dplus[0] = sqrt(ser[0])

    # d_(-1) = a_(-1)/(2*d_0)
    dplus[-1] = ser[-1]/(2*dplus[0])

    # Coefficients for the minus case are just the negative
    # of the coefficients for the positive case.
    dminus = [-x for x in dplus]

    if dplus != dminus:
        return [dplus, dminus]
    return dplus


def construct_d_case_6(num, den, x):
    # s_oo = lim x->0 1/x**2 * a(1/x) which is equivalent to
    # s_oo = lim x->oo x**2 * a(x)
    s_inf = limit_at_inf(Poly(x**2, x)*num, den, x)

    # d_(-1) = (1 +- sqrt(1 + 4*s_oo))/2
    if s_inf != -S(1)/4:
        return [[(1 + sqrt(1 + 4*s_inf))/2], [(1 - sqrt(1 + 4*s_inf))/2]]
    return [[S.Half]]


def construct_d(num, den, x, val_inf):
    """
    Helper function to calculate the coefficients
    in the d-vector based on the valuation of the
    function at oo.
    """
    N = -val_inf//2
    # Multiplicity of oo as a pole
    mul = -val_inf if val_inf < 0 else 0
    ser = rational_laurent_series(num, den, x, oo, mul, 1)

    # Case 4
    if val_inf < 0:
        d = construct_d_case_4(ser, N)

    # Case 5
    elif val_inf == 0:
        d = construct_d_case_5(ser)

    # Case 6
    else:
        d = construct_d_case_6(num, den, x)

    return d


def rational_laurent_series(num, den, x, r, m, n):
    r"""
    The function computes the Laurent series coefficients
    of a rational function.

    Parameters
    ==========

    num: A Poly object that is the numerator of `f(x)`.
    den: A Poly object that is the denominator of `f(x)`.
    x: The variable of expansion of the series.
    r: The point of expansion of the series.
    m: Multiplicity of r if r is a pole of `f(x)`. Should
    be zero otherwise.
    n: Order of the term upto which the series is expanded.

    Returns
    =======

    series: A dictionary that has power of the term as key
    and coefficient of that term as value.

    Below is a basic outline of how the Laurent series of a
    rational function `f(x)` about `x_0` is being calculated -

    1. Substitute `x + x_0` in place of `x`. If `x_0`
    is a pole of `f(x)`, multiply the expression by `x^m`
    where `m` is the multiplicity of `x_0`. Denote the
    the resulting expression as g(x). We do this substitution
    so that we can now find the Laurent series of g(x) about
    `x = 0`.

    2. We can then assume that the Laurent series of `g(x)`
    takes the following form -

    .. math:: g(x) = \frac{num(x)}{den(x)} = \sum_{m = 0}^{\infty} a_m x^m

    where `a_m` denotes the Laurent series coefficients.

    3. Multiply the denominator to the RHS of the equation
    and form a recurrence relation for the coefficients `a_m`.
    """
    one = Poly(1, x, extension=True)

    if r == oo:
        # Series at x = oo is equal to first transforming
        # the function from x -> 1/x and finding the
        # series at x = 0
        num, den = inverse_transform_poly(num, den, x)
        r = S(0)

    if r:
        # For an expansion about a non-zero point, a
        # transformation from x -> x + r must be made
        num = num.transform(Poly(x + r, x, extension=True), one)
        den = den.transform(Poly(x + r, x, extension=True), one)

    # Remove the pole from the denominator if the series
    # expansion is about one of the poles
    num, den = (num*x**m).cancel(den, include=True)

    # Equate coefficients for the first terms (base case)
    maxdegree = 1 + max(num.degree(), den.degree())
    syms = symbols(f'a:{maxdegree}', cls=Dummy)
    diff = num - den * Poly(syms[::-1], x)
    coeff_diffs = diff.all_coeffs()[::-1][:maxdegree]
    (coeffs, ) = linsolve(coeff_diffs, syms)

    # Use the recursion relation for the rest
    recursion = den.all_coeffs()[::-1]
    div, rec_rhs = recursion[0], recursion[1:]
    series = list(coeffs)
    while len(series) < n:
        next_coeff = Add(*(c*series[-1-n] for n, c in enumerate(rec_rhs))) / div
        series.append(-next_coeff)
    series = {m - i: val for i, val in enumerate(series)}
    return series

def compute_m_ybar(x, poles, choice, N):
    """
    Helper function to calculate -

    1. m - The degree bound for the polynomial
    solution that must be found for the auxiliary
    differential equation.

    2. ybar - Part of the solution which can be
    computed using the poles, c and d vectors.
    """
    ybar = 0
    m = Poly(choice[-1][-1], x, extension=True)

    # Calculate the first (nested) summation for ybar
    # as given in Step 9 of the Thesis (Pg 82)
    dybar = []
    for i, polei in enumerate(poles):
        for j, cij in enumerate(choice[i]):
            dybar.append(cij/(x - polei)**(j + 1))
        m -=Poly(choice[i][0], x, extension=True)  # can't accumulate Poly and use with Add
    ybar += Add(*dybar)

    # Calculate the second summation for ybar
    for i in range(N+1):
        ybar += choice[-1][i]*x**i
    return (m.expr, ybar)


def solve_aux_eq(numa, dena, numy, deny, x, m):
    """
    Helper function to find a polynomial solution
    of degree m for the auxiliary differential
    equation.
    """
    # Assume that the solution is of the type
    # p(x) = C_0 + C_1*x + ... + C_{m-1}*x**(m-1) + x**m
    psyms = symbols(f'C0:{m}', cls=Dummy)
    K = ZZ[psyms]
    psol = Poly(K.gens, x, domain=K) + Poly(x**m, x, domain=K)

    # Eq (5.16) in Thesis - Pg 81
    auxeq = (dena*(numy.diff(x)*deny - numy*deny.diff(x) + numy**2) - numa*deny**2)*psol
    if m >= 1:
        px = psol.diff(x)
        auxeq += px*(2*numy*deny*dena)
    if m >= 2:
        auxeq += px.diff(x)*(deny**2*dena)
    if m != 0:
        # m is a non-zero integer. Find the constant terms using undetermined coefficients
        return psol, linsolve_dict(auxeq.all_coeffs(), psyms), True
    else:
        # m == 0 . Check if 1 (x**0) is a solution to the auxiliary equation
        return S.One, auxeq, auxeq == 0


def remove_redundant_sols(sol1, sol2, x):
    """
    Helper function to remove redundant
    solutions to the differential equation.
    """
    # If y1 and y2 are redundant solutions, there is
    # some value of the arbitrary constant for which
    # they will be equal

    syms1 = sol1.atoms(Symbol, Dummy)
    syms2 = sol2.atoms(Symbol, Dummy)
    num1, den1 = [Poly(e, x, extension=True) for e in sol1.together().as_numer_denom()]
    num2, den2 = [Poly(e, x, extension=True) for e in sol2.together().as_numer_denom()]
    # Cross multiply
    e = num1*den2 - den1*num2
    # Check if there are any constants
    syms = list(e.atoms(Symbol, Dummy))
    if len(syms):
        # Find values of constants for which solutions are equal
        redn = linsolve(e.all_coeffs(), syms)
        if len(redn):
            # Return the general solution over a particular solution
            if len(syms1) > len(syms2):
                return sol2
            # If both have constants, return the lesser complex solution
            elif len(syms1) == len(syms2):
                return sol1 if count_ops(syms1) >= count_ops(syms2) else sol2
            else:
                return sol1


def get_gen_sol_from_part_sol(part_sols, a, x):
    """"
    Helper function which computes the general
    solution for a Riccati ODE from its particular
    solutions.

    There are 3 cases to find the general solution
    from the particular solutions for a Riccati ODE
    depending on the number of particular solution(s)
    we have - 1, 2 or 3.

    For more information, see Section 6 of
    "Methods of Solution of the Riccati Differential Equation"
    by D. R. Haaheim and F. M. Stein
    """

    # If no particular solutions are found, a general
    # solution cannot be found
    if len(part_sols) == 0:
        return []

    # In case of a single particular solution, the general
    # solution can be found by using the substitution
    # y = y1 + 1/z and solving a Bernoulli ODE to find z.
    elif len(part_sols) == 1:
        y1 = part_sols[0]
        i = exp(Integral(2*y1, x))
        z = i * Integral(a/i, x)
        z = z.doit()
        if a == 0 or z == 0:
            return y1
        return y1 + 1/z

    # In case of 2 particular solutions, the general solution
    # can be found by solving a separable equation. This is
    # the most common case, i.e. most Riccati ODEs have 2
    # rational particular solutions.
    elif len(part_sols) == 2:
        y1, y2 = part_sols
        # One of them already has a constant
        if len(y1.atoms(Dummy)) + len(y2.atoms(Dummy)) > 0:
            u = exp(Integral(y2 - y1, x)).doit()
        # Introduce a constant
        else:
            C1 = Dummy('C1')
            u = C1*exp(Integral(y2 - y1, x)).doit()
        if u == 1:
            return y2
        return (y2*u - y1)/(u - 1)

    # In case of 3 particular solutions, a closed form
    # of the general solution can be obtained directly
    else:
        y1, y2, y3 = part_sols[:3]
        C1 = Dummy('C1')
        return (C1 + 1)*y2*(y1 - y3)/(C1*y1 + y2 - (C1 + 1)*y3)


def solve_riccati(fx, x, b0, b1, b2, gensol=False):
    """
    The main function that gives particular/general
    solutions to Riccati ODEs that have atleast 1
    rational particular solution.
    """
    # Step 1 : Convert to Normal Form
    a = -b0*b2 + b1**2/4 - b1.diff(x)/2 + 3*b2.diff(x)**2/(4*b2**2) + b1*b2.diff(x)/(2*b2) - \
        b2.diff(x, 2)/(2*b2)
    a_t = a.together()
    num, den = [Poly(e, x, extension=True) for e in a_t.as_numer_denom()]
    num, den = num.cancel(den, include=True)

    # Step 2
    presol = []

    # Step 3 : a(x) is 0
    if num == 0:
        presol.append(1/(x + Dummy('C1')))

    # Step 4 : a(x) is a non-zero constant
    elif x not in num.free_symbols.union(den.free_symbols):
        presol.extend([sqrt(a), -sqrt(a)])

    # Step 5 : Find poles and valuation at infinity
    poles = roots(den, x)
    poles, muls = list(poles.keys()), list(poles.values())
    val_inf = val_at_inf(num, den, x)

    if len(poles):
        # Check necessary conditions (outlined in the module docstring)
        if not check_necessary_conds(val_inf, muls):
            raise ValueError("Rational Solution doesn't exist")

        # Step 6
        # Construct c-vectors for each singular point
        c = construct_c(num, den, x, poles, muls)

        # Construct d vectors for each singular point
        d = construct_d(num, den, x, val_inf)

        # Step 7 : Iterate over all possible combinations and return solutions
        # For each possible combination, generate an array of 0's and 1's
        # where 0 means pick 1st choice and 1 means pick the second choice.

        # NOTE: We could exit from the loop if we find 3 particular solutions,
        # but it is not implemented here as -
        #   a. Finding 3 particular solutions is very rare. Most of the time,
        #      only 2 particular solutions are found.
        #   b. In case we exit after finding 3 particular solutions, it might
        #      happen that 1 or 2 of them are redundant solutions. So, instead of
        #      spending some more time in computing the particular solutions,
        #      we will end up computing the general solution from a single
        #      particular solution which is usually slower than computing the
        #      general solution from 2 or 3 particular solutions.
        c.append(d)
        choices = product(*c)
        for choice in choices:
            m, ybar = compute_m_ybar(x, poles, choice, -val_inf//2)
            numy, deny = [Poly(e, x, extension=True) for e in ybar.together().as_numer_denom()]
            # Step 10 : Check if a valid solution exists. If yes, also check
            # if m is a non-negative integer
            if m.is_nonnegative == True and m.is_integer == True:

                # Step 11 : Find polynomial solutions of degree m for the auxiliary equation
                psol, coeffs, exists = solve_aux_eq(num, den, numy, deny, x, m)

                # Step 12 : If valid polynomial solution exists, append solution.
                if exists:
                    # m == 0 case
                    if psol == 1 and coeffs == 0:
                        # p(x) = 1, so p'(x)/p(x) term need not be added
                        presol.append(ybar)
                    # m is a positive integer and there are valid coefficients
                    elif len(coeffs):
                        # Substitute the valid coefficients to get p(x)
                        psol = psol.xreplace(coeffs)
                        # y(x) = ybar(x) + p'(x)/p(x)
                        presol.append(ybar + psol.diff(x)/psol)

    # Remove redundant solutions from the list of existing solutions
    remove = set()
    for i in range(len(presol)):
        for j in range(i+1, len(presol)):
            rem = remove_redundant_sols(presol[i], presol[j], x)
            if rem is not None:
                remove.add(rem)
    sols = [x for x in presol if x not in remove]

    # Step 15 : Inverse transform the solutions of the equation in normal form
    bp = -b2.diff(x)/(2*b2**2) - b1/(2*b2)

    # If general solution is required, compute it from the particular solutions
    if gensol:
        sols = [get_gen_sol_from_part_sol(sols, a, x)]

    # Inverse transform the particular solutions
    presol = [Eq(fx, riccati_inverse_normal(y, x, b1, b2, bp).cancel(extension=True)) for y in sols]
    return presol

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\psycopg2.py ===
# dialects/postgresql/psycopg2.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: postgresql+psycopg2
    :name: psycopg2
    :dbapi: psycopg2
    :connectstring: postgresql+psycopg2://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://pypi.org/project/psycopg2/

.. _psycopg2_toplevel:

psycopg2 Connect Arguments
--------------------------

Keyword arguments that are specific to the SQLAlchemy psycopg2 dialect
may be passed to :func:`_sa.create_engine()`, and include the following:


* ``isolation_level``: This option, available for all PostgreSQL dialects,
  includes the ``AUTOCOMMIT`` isolation level when using the psycopg2
  dialect.   This option sets the **default** isolation level for the
  connection that is set immediately upon connection to the database before
  the connection is pooled.  This option is generally superseded by the more
  modern :paramref:`_engine.Connection.execution_options.isolation_level`
  execution option, detailed at :ref:`dbapi_autocommit`.

  .. seealso::

    :ref:`psycopg2_isolation_level`

    :ref:`dbapi_autocommit`


* ``client_encoding``: sets the client encoding in a libpq-agnostic way,
  using psycopg2's ``set_client_encoding()`` method.

  .. seealso::

    :ref:`psycopg2_unicode`


* ``executemany_mode``, ``executemany_batch_page_size``,
  ``executemany_values_page_size``: Allows use of psycopg2
  extensions for optimizing "executemany"-style queries.  See the referenced
  section below for details.

  .. seealso::

    :ref:`psycopg2_executemany_mode`

.. tip::

    The above keyword arguments are **dialect** keyword arguments, meaning
    that they are passed as explicit keyword arguments to :func:`_sa.create_engine()`::

        engine = create_engine(
            "postgresql+psycopg2://scott:tiger@localhost/test",
            isolation_level="SERIALIZABLE",
        )

    These should not be confused with **DBAPI** connect arguments, which
    are passed as part of the :paramref:`_sa.create_engine.connect_args`
    dictionary and/or are passed in the URL query string, as detailed in
    the section :ref:`custom_dbapi_args`.

.. _psycopg2_ssl:

SSL Connections
---------------

The psycopg2 module has a connection argument named ``sslmode`` for
controlling its behavior regarding secure (SSL) connections. The default is
``sslmode=prefer``; it will attempt an SSL connection and if that fails it
will fall back to an unencrypted connection. ``sslmode=require`` may be used
to ensure that only secure connections are established.  Consult the
psycopg2 / libpq documentation for further options that are available.

Note that ``sslmode`` is specific to psycopg2 so it is included in the
connection URI::

    engine = sa.create_engine(
        "postgresql+psycopg2://scott:tiger@192.168.0.199:5432/test?sslmode=require"
    )

Unix Domain Connections
------------------------

psycopg2 supports connecting via Unix domain connections.   When the ``host``
portion of the URL is omitted, SQLAlchemy passes ``None`` to psycopg2,
which specifies Unix-domain communication rather than TCP/IP communication::

    create_engine("postgresql+psycopg2://user:password@/dbname")

By default, the socket file used is to connect to a Unix-domain socket
in ``/tmp``, or whatever socket directory was specified when PostgreSQL
was built.  This value can be overridden by passing a pathname to psycopg2,
using ``host`` as an additional keyword argument::

    create_engine(
        "postgresql+psycopg2://user:password@/dbname?host=/var/lib/postgresql"
    )

.. warning::  The format accepted here allows for a hostname in the main URL
   in addition to the "host" query string argument.  **When using this URL
   format, the initial host is silently ignored**.  That is, this URL::

        engine = create_engine(
            "postgresql+psycopg2://user:password@myhost1/dbname?host=myhost2"
        )

   Above, the hostname ``myhost1`` is **silently ignored and discarded.**  The
   host which is connected is the ``myhost2`` host.

   This is to maintain some degree of compatibility with PostgreSQL's own URL
   format which has been tested to behave the same way and for which tools like
   PifPaf hardcode two hostnames.

.. seealso::

    `PQconnectdbParams \
    <https://www.postgresql.org/docs/current/static/libpq-connect.html#LIBPQ-PQCONNECTDBPARAMS>`_

.. _psycopg2_multi_host:

Specifying multiple fallback hosts
-----------------------------------

psycopg2 supports multiple connection points in the connection string.
When the ``host`` parameter is used multiple times in the query section of
the URL, SQLAlchemy will create a single string of the host and port
information provided to make the connections.  Tokens may consist of
``host::port`` or just ``host``; in the latter case, the default port
is selected by libpq.  In the example below, three host connections
are specified, for ``HostA::PortA``, ``HostB`` connecting to the default port,
and ``HostC::PortC``::

    create_engine(
        "postgresql+psycopg2://user:password@/dbname?host=HostA:PortA&host=HostB&host=HostC:PortC"
    )

As an alternative, libpq query string format also may be used; this specifies
``host`` and ``port`` as single query string arguments with comma-separated
lists - the default port can be chosen by indicating an empty value
in the comma separated list::

    create_engine(
        "postgresql+psycopg2://user:password@/dbname?host=HostA,HostB,HostC&port=PortA,,PortC"
    )

With either URL style, connections to each host is attempted based on a
configurable strategy, which may be configured using the libpq
``target_session_attrs`` parameter.  Per libpq this defaults to ``any``
which indicates a connection to each host is then attempted until a connection is successful.
Other strategies include ``primary``, ``prefer-standby``, etc.  The complete
list is documented by PostgreSQL at
`libpq connection strings <https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING>`_.

For example, to indicate two hosts using the ``primary`` strategy::

    create_engine(
        "postgresql+psycopg2://user:password@/dbname?host=HostA:PortA&host=HostB&host=HostC:PortC&target_session_attrs=primary"
    )

.. versionchanged:: 1.4.40 Port specification in psycopg2 multiple host format
   is repaired, previously ports were not correctly interpreted in this context.
   libpq comma-separated format is also now supported.

.. versionadded:: 1.3.20 Support for multiple hosts in PostgreSQL connection
   string.

.. seealso::

    `libpq connection strings <https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING>`_ - please refer
    to this section in the libpq documentation for complete background on multiple host support.


Empty DSN Connections / Environment Variable Connections
---------------------------------------------------------

The psycopg2 DBAPI can connect to PostgreSQL by passing an empty DSN to the
libpq client library, which by default indicates to connect to a localhost
PostgreSQL database that is open for "trust" connections.  This behavior can be
further tailored using a particular set of environment variables which are
prefixed with ``PG_...``, which are  consumed by ``libpq`` to take the place of
any or all elements of the connection string.

For this form, the URL can be passed without any elements other than the
initial scheme::

    engine = create_engine("postgresql+psycopg2://")

In the above form, a blank "dsn" string is passed to the ``psycopg2.connect()``
function which in turn represents an empty DSN passed to libpq.

.. versionadded:: 1.3.2 support for parameter-less connections with psycopg2.

.. seealso::

    `Environment Variables\
    <https://www.postgresql.org/docs/current/libpq-envars.html>`_ -
    PostgreSQL documentation on how to use ``PG_...``
    environment variables for connections.

.. _psycopg2_execution_options:

Per-Statement/Connection Execution Options
-------------------------------------------

The following DBAPI-specific options are respected when used with
:meth:`_engine.Connection.execution_options`,
:meth:`.Executable.execution_options`,
:meth:`_query.Query.execution_options`,
in addition to those not specific to DBAPIs:

* ``isolation_level`` - Set the transaction isolation level for the lifespan
  of a :class:`_engine.Connection` (can only be set on a connection,
  not a statement
  or query).   See :ref:`psycopg2_isolation_level`.

* ``stream_results`` - Enable or disable usage of psycopg2 server side
  cursors - this feature makes use of "named" cursors in combination with
  special result handling methods so that result rows are not fully buffered.
  Defaults to False, meaning cursors are buffered by default.

* ``max_row_buffer`` - when using ``stream_results``, an integer value that
  specifies the maximum number of rows to buffer at a time.  This is
  interpreted by the :class:`.BufferedRowCursorResult`, and if omitted the
  buffer will grow to ultimately store 1000 rows at a time.

  .. versionchanged:: 1.4  The ``max_row_buffer`` size can now be greater than
     1000, and the buffer will grow to that size.

.. _psycopg2_batch_mode:

.. _psycopg2_executemany_mode:

Psycopg2 Fast Execution Helpers
-------------------------------

Modern versions of psycopg2 include a feature known as
`Fast Execution Helpers \
<https://www.psycopg.org/docs/extras.html#fast-execution-helpers>`_, which
have been shown in benchmarking to improve psycopg2's executemany()
performance, primarily with INSERT statements, by at least
an order of magnitude.

SQLAlchemy implements a native form of the "insert many values"
handler that will rewrite a single-row INSERT statement to accommodate for
many values at once within an extended VALUES clause; this handler is
equivalent to psycopg2's ``execute_values()`` handler; an overview of this
feature and its configuration are at :ref:`engine_insertmanyvalues`.

.. versionadded:: 2.0 Replaced psycopg2's ``execute_values()`` fast execution
   helper with a native SQLAlchemy mechanism known as
   :ref:`insertmanyvalues <engine_insertmanyvalues>`.

The psycopg2 dialect retains the ability to use the psycopg2-specific
``execute_batch()`` feature, although it is not expected that this is a widely
used feature.  The use of this extension may be enabled using the
``executemany_mode`` flag which may be passed to :func:`_sa.create_engine`::

    engine = create_engine(
        "postgresql+psycopg2://scott:tiger@host/dbname",
        executemany_mode="values_plus_batch",
    )

Possible options for ``executemany_mode`` include:

* ``values_only`` - this is the default value.  SQLAlchemy's native
  :ref:`insertmanyvalues <engine_insertmanyvalues>` handler is used for qualifying
  INSERT statements, assuming
  :paramref:`_sa.create_engine.use_insertmanyvalues` is left at
  its default value of ``True``.  This handler rewrites simple
  INSERT statements to include multiple VALUES clauses so that many
  parameter sets can be inserted with one statement.

* ``'values_plus_batch'``- SQLAlchemy's native
  :ref:`insertmanyvalues <engine_insertmanyvalues>` handler is used for qualifying
  INSERT statements, assuming
  :paramref:`_sa.create_engine.use_insertmanyvalues` is left at its default
  value of ``True``. Then, psycopg2's ``execute_batch()`` handler is used for
  qualifying UPDATE and DELETE statements when executed with multiple parameter
  sets. When using this mode, the :attr:`_engine.CursorResult.rowcount`
  attribute will not contain a value for executemany-style executions against
  UPDATE and DELETE statements.

.. versionchanged:: 2.0 Removed the ``'batch'`` and ``'None'`` options
   from psycopg2 ``executemany_mode``.  Control over batching for INSERT
   statements is now configured via the
   :paramref:`_sa.create_engine.use_insertmanyvalues` engine-level parameter.

The term "qualifying statements" refers to the statement being executed
being a Core :func:`_expression.insert`, :func:`_expression.update`
or :func:`_expression.delete` construct, and **not** a plain textual SQL
string or one constructed using :func:`_expression.text`.  It also may **not** be
a special "extension" statement such as an "ON CONFLICT" "upsert" statement.
When using the ORM, all insert/update/delete statements used by the ORM flush process
are qualifying.

The "page size" for the psycopg2 "batch" strategy can be affected
by using the ``executemany_batch_page_size`` parameter, which defaults to
100.

For the "insertmanyvalues" feature, the page size can be controlled using the
:paramref:`_sa.create_engine.insertmanyvalues_page_size` parameter,
which defaults to 1000.  An example of modifying both parameters
is below::

    engine = create_engine(
        "postgresql+psycopg2://scott:tiger@host/dbname",
        executemany_mode="values_plus_batch",
        insertmanyvalues_page_size=5000,
        executemany_batch_page_size=500,
    )

.. seealso::

    :ref:`engine_insertmanyvalues` - background on "insertmanyvalues"

    :ref:`tutorial_multiple_parameters` - General information on using the
    :class:`_engine.Connection`
    object to execute statements in such a way as to make
    use of the DBAPI ``.executemany()`` method.


.. _psycopg2_unicode:

Unicode with Psycopg2
----------------------

The psycopg2 DBAPI driver supports Unicode data transparently.

The client character encoding can be controlled for the psycopg2 dialect
in the following ways:

* For PostgreSQL 9.1 and above, the ``client_encoding`` parameter may be
  passed in the database URL; this parameter is consumed by the underlying
  ``libpq`` PostgreSQL client library::

    engine = create_engine(
        "postgresql+psycopg2://user:pass@host/dbname?client_encoding=utf8"
    )

  Alternatively, the above ``client_encoding`` value may be passed using
  :paramref:`_sa.create_engine.connect_args` for programmatic establishment with
  ``libpq``::

    engine = create_engine(
        "postgresql+psycopg2://user:pass@host/dbname",
        connect_args={"client_encoding": "utf8"},
    )

* For all PostgreSQL versions, psycopg2 supports a client-side encoding
  value that will be passed to database connections when they are first
  established.  The SQLAlchemy psycopg2 dialect supports this using the
  ``client_encoding`` parameter passed to :func:`_sa.create_engine`::

      engine = create_engine(
          "postgresql+psycopg2://user:pass@host/dbname", client_encoding="utf8"
      )

  .. tip:: The above ``client_encoding`` parameter admittedly is very similar
      in appearance to usage of the parameter within the
      :paramref:`_sa.create_engine.connect_args` dictionary; the difference
      above is that the parameter is consumed by psycopg2 and is
      passed to the database connection using ``SET client_encoding TO
      'utf8'``; in the previously mentioned style, the parameter is instead
      passed through psycopg2 and consumed by the ``libpq`` library.

* A common way to set up client encoding with PostgreSQL databases is to
  ensure it is configured within the server-side postgresql.conf file;
  this is the recommended way to set encoding for a server that is
  consistently of one encoding in all databases::

    # postgresql.conf file

    # client_encoding = sql_ascii # actually, defaults to database
    # encoding
    client_encoding = utf8

Transactions
------------

The psycopg2 dialect fully supports SAVEPOINT and two-phase commit operations.

.. _psycopg2_isolation_level:

Psycopg2 Transaction Isolation Level
-------------------------------------

As discussed in :ref:`postgresql_isolation_level`,
all PostgreSQL dialects support setting of transaction isolation level
both via the ``isolation_level`` parameter passed to :func:`_sa.create_engine`
,
as well as the ``isolation_level`` argument used by
:meth:`_engine.Connection.execution_options`.  When using the psycopg2 dialect
, these
options make use of psycopg2's ``set_isolation_level()`` connection method,
rather than emitting a PostgreSQL directive; this is because psycopg2's
API-level setting is always emitted at the start of each transaction in any
case.

The psycopg2 dialect supports these constants for isolation level:

* ``READ COMMITTED``
* ``READ UNCOMMITTED``
* ``REPEATABLE READ``
* ``SERIALIZABLE``
* ``AUTOCOMMIT``

.. seealso::

    :ref:`postgresql_isolation_level`

    :ref:`pg8000_isolation_level`


NOTICE logging
---------------

The psycopg2 dialect will log PostgreSQL NOTICE messages
via the ``sqlalchemy.dialects.postgresql`` logger.  When this logger
is set to the ``logging.INFO`` level, notice messages will be logged::

    import logging

    logging.getLogger("sqlalchemy.dialects.postgresql").setLevel(logging.INFO)

Above, it is assumed that logging is configured externally.  If this is not
the case, configuration such as ``logging.basicConfig()`` must be utilized::

    import logging

    logging.basicConfig()  # log messages to stdout
    logging.getLogger("sqlalchemy.dialects.postgresql").setLevel(logging.INFO)

.. seealso::

    `Logging HOWTO <https://docs.python.org/3/howto/logging.html>`_ - on the python.org website

.. _psycopg2_hstore:

HSTORE type
------------

The ``psycopg2`` DBAPI includes an extension to natively handle marshalling of
the HSTORE type.   The SQLAlchemy psycopg2 dialect will enable this extension
by default when psycopg2 version 2.4 or greater is used, and
it is detected that the target database has the HSTORE type set up for use.
In other words, when the dialect makes the first
connection, a sequence like the following is performed:

1. Request the available HSTORE oids using
   ``psycopg2.extras.HstoreAdapter.get_oids()``.
   If this function returns a list of HSTORE identifiers, we then determine
   that the ``HSTORE`` extension is present.
   This function is **skipped** if the version of psycopg2 installed is
   less than version 2.4.

2. If the ``use_native_hstore`` flag is at its default of ``True``, and
   we've detected that ``HSTORE`` oids are available, the
   ``psycopg2.extensions.register_hstore()`` extension is invoked for all
   connections.

The ``register_hstore()`` extension has the effect of **all Python
dictionaries being accepted as parameters regardless of the type of target
column in SQL**. The dictionaries are converted by this extension into a
textual HSTORE expression.  If this behavior is not desired, disable the
use of the hstore extension by setting ``use_native_hstore`` to ``False`` as
follows::

    engine = create_engine(
        "postgresql+psycopg2://scott:tiger@localhost/test",
        use_native_hstore=False,
    )

The ``HSTORE`` type is **still supported** when the
``psycopg2.extensions.register_hstore()`` extension is not used.  It merely
means that the coercion between Python dictionaries and the HSTORE
string format, on both the parameter side and the result side, will take
place within SQLAlchemy's own marshalling logic, and not that of ``psycopg2``
which may be more performant.

"""  # noqa
from __future__ import annotations

import collections.abc as collections_abc
import logging
import re
from typing import cast

from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from ... import types as sqltypes
from ... import util
from ...util import FastIntFlag
from ...util import parse_user_argument_for_enum

logger = logging.getLogger("sqlalchemy.dialects.postgresql")


class _PGJSON(JSON):
    def result_processor(self, dialect, coltype):
        return None


class _PGJSONB(JSONB):
    def result_processor(self, dialect, coltype):
        return None


class _Psycopg2Range(ranges.AbstractSingleRangeImpl):
    _psycopg2_range_cls = "none"

    def bind_processor(self, dialect):
        psycopg2_Range = getattr(
            cast(PGDialect_psycopg2, dialect)._psycopg2_extras,
            self._psycopg2_range_cls,
        )

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = psycopg2_Range(
                    value.lower, value.upper, value.bounds, value.empty
                )
            return value

        return to_range

    def result_processor(self, dialect, coltype):
        def to_range(value):
            if value is not None:
                value = ranges.Range(
                    value._lower,
                    value._upper,
                    bounds=value._bounds if value._bounds else "[)",
                    empty=not value._bounds,
                )
            return value

        return to_range


class _Psycopg2NumericRange(_Psycopg2Range):
    _psycopg2_range_cls = "NumericRange"


class _Psycopg2DateRange(_Psycopg2Range):
    _psycopg2_range_cls = "DateRange"


class _Psycopg2DateTimeRange(_Psycopg2Range):
    _psycopg2_range_cls = "DateTimeRange"


class _Psycopg2DateTimeTZRange(_Psycopg2Range):
    _psycopg2_range_cls = "DateTimeTZRange"


class PGExecutionContext_psycopg2(_PGExecutionContext_common_psycopg):
    _psycopg2_fetched_rows = None

    def post_exec(self):
        self._log_notices(self.cursor)

    def _log_notices(self, cursor):
        # check also that notices is an iterable, after it's already
        # established that we will be iterating through it.  This is to get
        # around test suites such as SQLAlchemy's using a Mock object for
        # cursor
        if not cursor.connection.notices or not isinstance(
            cursor.connection.notices, collections_abc.Iterable
        ):
            return

        for notice in cursor.connection.notices:
            # NOTICE messages have a
            # newline character at the end
            logger.info(notice.rstrip())

        cursor.connection.notices[:] = []


class PGIdentifierPreparer_psycopg2(PGIdentifierPreparer):
    pass


class ExecutemanyMode(FastIntFlag):
    EXECUTEMANY_VALUES = 0
    EXECUTEMANY_VALUES_PLUS_BATCH = 1


(
    EXECUTEMANY_VALUES,
    EXECUTEMANY_VALUES_PLUS_BATCH,
) = ExecutemanyMode.__members__.values()


class PGDialect_psycopg2(_PGDialect_common_psycopg):
    driver = "psycopg2"

    supports_statement_cache = True
    supports_server_side_cursors = True

    default_paramstyle = "pyformat"
    # set to true based on psycopg2 version
    supports_sane_multi_rowcount = False
    execution_ctx_cls = PGExecutionContext_psycopg2
    preparer = PGIdentifierPreparer_psycopg2
    psycopg2_version = (0, 0)
    use_insertmanyvalues_wo_returning = True

    returns_native_bytes = False

    _has_native_hstore = True

    colspecs = util.update_copy(
        _PGDialect_common_psycopg.colspecs,
        {
            JSON: _PGJSON,
            sqltypes.JSON: _PGJSON,
            JSONB: _PGJSONB,
            ranges.INT4RANGE: _Psycopg2NumericRange,
            ranges.INT8RANGE: _Psycopg2NumericRange,
            ranges.NUMRANGE: _Psycopg2NumericRange,
            ranges.DATERANGE: _Psycopg2DateRange,
            ranges.TSRANGE: _Psycopg2DateTimeRange,
            ranges.TSTZRANGE: _Psycopg2DateTimeTZRange,
        },
    )

    def __init__(
        self,
        executemany_mode="values_only",
        executemany_batch_page_size=100,
        **kwargs,
    ):
        _PGDialect_common_psycopg.__init__(self, **kwargs)

        if self._native_inet_types:
            raise NotImplementedError(
                "The psycopg2 dialect does not implement "
                "ipaddress type handling; native_inet_types cannot be set "
                "to ``True`` when using this dialect."
            )

        # Parse executemany_mode argument, allowing it to be only one of the
        # symbol names
        self.executemany_mode = parse_user_argument_for_enum(
            executemany_mode,
            {
                EXECUTEMANY_VALUES: ["values_only"],
                EXECUTEMANY_VALUES_PLUS_BATCH: ["values_plus_batch"],
            },
            "executemany_mode",
        )

        self.executemany_batch_page_size = executemany_batch_page_size

        if self.dbapi and hasattr(self.dbapi, "__version__"):
            m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", self.dbapi.__version__)
            if m:
                self.psycopg2_version = tuple(
                    int(x) for x in m.group(1, 2, 3) if x is not None
                )

            if self.psycopg2_version < (2, 7):
                raise ImportError(
                    "psycopg2 version 2.7 or higher is required."
                )

    def initialize(self, connection):
        super().initialize(connection)
        self._has_native_hstore = (
            self.use_native_hstore
            and self._hstore_oids(connection.connection.dbapi_connection)
            is not None
        )

        self.supports_sane_multi_rowcount = (
            self.executemany_mode is not EXECUTEMANY_VALUES_PLUS_BATCH
        )

    @classmethod
    def import_dbapi(cls):
        import psycopg2

        return psycopg2

    @util.memoized_property
    def _psycopg2_extensions(cls):
        from psycopg2 import extensions

        return extensions

    @util.memoized_property
    def _psycopg2_extras(cls):
        from psycopg2 import extras

        return extras

    @util.memoized_property
    def _isolation_lookup(self):
        extensions = self._psycopg2_extensions
        return {
            "AUTOCOMMIT": extensions.ISOLATION_LEVEL_AUTOCOMMIT,
            "READ COMMITTED": extensions.ISOLATION_LEVEL_READ_COMMITTED,
            "READ UNCOMMITTED": extensions.ISOLATION_LEVEL_READ_UNCOMMITTED,
            "REPEATABLE READ": extensions.ISOLATION_LEVEL_REPEATABLE_READ,
            "SERIALIZABLE": extensions.ISOLATION_LEVEL_SERIALIZABLE,
        }

    def set_isolation_level(self, dbapi_connection, level):
        dbapi_connection.set_isolation_level(self._isolation_lookup[level])

    def set_readonly(self, connection, value):
        connection.readonly = value

    def get_readonly(self, connection):
        return connection.readonly

    def set_deferrable(self, connection, value):
        connection.deferrable = value

    def get_deferrable(self, connection):
        return connection.deferrable

    def on_connect(self):
        extras = self._psycopg2_extras

        fns = []
        if self.client_encoding is not None:

            def on_connect(dbapi_conn):
                dbapi_conn.set_client_encoding(self.client_encoding)

            fns.append(on_connect)

        if self.dbapi:

            def on_connect(dbapi_conn):
                extras.register_uuid(None, dbapi_conn)

            fns.append(on_connect)

        if self.dbapi and self.use_native_hstore:

            def on_connect(dbapi_conn):
                hstore_oids = self._hstore_oids(dbapi_conn)
                if hstore_oids is not None:
                    oid, array_oid = hstore_oids
                    kw = {"oid": oid}
                    kw["array_oid"] = array_oid
                    extras.register_hstore(dbapi_conn, **kw)

            fns.append(on_connect)

        if self.dbapi and self._json_deserializer:

            def on_connect(dbapi_conn):
                extras.register_default_json(
                    dbapi_conn, loads=self._json_deserializer
                )
                extras.register_default_jsonb(
                    dbapi_conn, loads=self._json_deserializer
                )

            fns.append(on_connect)

        if fns:

            def on_connect(dbapi_conn):
                for fn in fns:
                    fn(dbapi_conn)

            return on_connect
        else:
            return None

    def do_executemany(self, cursor, statement, parameters, context=None):
        if self.executemany_mode is EXECUTEMANY_VALUES_PLUS_BATCH:
            if self.executemany_batch_page_size:
                kwargs = {"page_size": self.executemany_batch_page_size}
            else:
                kwargs = {}
            self._psycopg2_extras.execute_batch(
                cursor, statement, parameters, **kwargs
            )
        else:
            cursor.executemany(statement, parameters)

    def do_begin_twophase(self, connection, xid):
        connection.connection.tpc_begin(xid)

    def do_prepare_twophase(self, connection, xid):
        connection.connection.tpc_prepare()

    def _do_twophase(self, dbapi_conn, operation, xid, recover=False):
        if recover:
            if dbapi_conn.status != self._psycopg2_extensions.STATUS_READY:
                dbapi_conn.rollback()
            operation(xid)
        else:
            operation()

    def do_rollback_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        dbapi_conn = connection.connection.dbapi_connection
        self._do_twophase(
            dbapi_conn, dbapi_conn.tpc_rollback, xid, recover=recover
        )

    def do_commit_twophase(
        self, connection, xid, is_prepared=True, recover=False
    ):
        dbapi_conn = connection.connection.dbapi_connection
        self._do_twophase(
            dbapi_conn, dbapi_conn.tpc_commit, xid, recover=recover
        )

    @util.memoized_instancemethod
    def _hstore_oids(self, dbapi_connection):
        extras = self._psycopg2_extras
        oids = extras.HstoreAdapter.get_oids(dbapi_connection)
        if oids is not None and oids[0]:
            return oids[0:2]
        else:
            return None

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.Error):
            # check the "closed" flag.  this might not be
            # present on old psycopg2 versions.   Also,
            # this flag doesn't actually help in a lot of disconnect
            # situations, so don't rely on it.
            if getattr(connection, "closed", False):
                return True

            # checks based on strings.  in the case that .closed
            # didn't cut it, fall back onto these.
            str_e = str(e).partition("\n")[0]
            for msg in self._is_disconnect_messages:
                idx = str_e.find(msg)
                if idx >= 0 and '"' not in str_e[:idx]:
                    return True
        return False

    @util.memoized_property
    def _is_disconnect_messages(self):
        return (
            # these error messages from libpq: interfaces/libpq/fe-misc.c
            # and interfaces/libpq/fe-secure.c.
            "terminating connection",
            "closed the connection",
            "connection not open",
            "could not receive data from server",
            "could not send data to server",
            # psycopg2 client errors, psycopg2/connection.h,
            # psycopg2/cursor.h
            "connection already closed",
            "cursor already closed",
            # not sure where this path is originally from, it may
            # be obsolete.   It really says "losed", not "closed".
            "losed the connection unexpectedly",
            # these can occur in newer SSL
            "connection has been closed unexpectedly",
            "SSL error: decryption failed or bad record mac",
            "SSL SYSCALL error: Bad file descriptor",
            "SSL SYSCALL error: EOF detected",
            "SSL SYSCALL error: Operation timed out",
            "SSL SYSCALL error: Bad address",
            # This can occur in OpenSSL 1 when an unexpected EOF occurs.
            # https://www.openssl.org/docs/man1.1.1/man3/SSL_get_error.html#BUGS
            # It may also occur in newer OpenSSL for a non-recoverable I/O
            # error as a result of a system call that does not set 'errno'
            # in libc.
            "SSL SYSCALL error: Success",
        )


dialect = PGDialect_psycopg2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\matexpr.py ===
from __future__ import annotations
from functools import wraps

from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.kind import MatrixKind
from sympy.matrices.matrixbase import MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent


def _sympifyit(arg, retval=None):
    # This version of _sympifyit sympifies MutableMatrix objects
    def deco(func):
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                b = _sympify(b)
                return func(a, b)
            except SympifyError:
                return retval

        return __sympifyit_wrapper

    return deco


class MatrixExpr(Expr):
    """Superclass for Matrix Expressions

    MatrixExprs represent abstract matrices, linear transformations represented
    within a particular basis.

    Examples
    ========

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 3, 3)
    >>> y = MatrixSymbol('y', 3, 1)
    >>> x = (A.T*A).I * A * y

    See Also
    ========

    MatrixSymbol, MatAdd, MatMul, Transpose, Inverse
    """
    __slots__: tuple[str, ...] = ()

    # Should not be considered iterable by the
    # sympy.utilities.iterables.iterable function. Subclass that actually are
    # iterable (i.e., explicit matrices) should set this to True.
    _iterable = False

    _op_priority = 11.0

    is_Matrix: bool = True
    is_MatrixExpr: bool = True
    is_Identity: FuzzyBool = None
    is_Inverse = False
    is_Transpose = False
    is_ZeroMatrix = False
    is_MatAdd = False
    is_MatMul = False

    is_commutative = False
    is_number = False
    is_symbol = False
    is_scalar = False

    kind: MatrixKind = MatrixKind()

    def __new__(cls, *args, **kwargs):
        args = map(_sympify, args)
        return Basic.__new__(cls, *args, **kwargs)

    # The following is adapted from the core Expr object

    @property
    def shape(self) -> tuple[Expr, Expr]:
        raise NotImplementedError

    @property
    def _add_handler(self):
        return MatAdd

    @property
    def _mul_handler(self):
        return MatMul

    def __neg__(self):
        return MatMul(S.NegativeOne, self).doit()

    def __abs__(self):
        raise NotImplementedError

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return MatAdd(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return MatAdd(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return MatAdd(self, -other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return MatAdd(other, -self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __matmul__(self, other):
        return MatMul(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmatmul__(self, other):
        return MatMul(other, self).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return MatPow(self, other).doit()

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        raise NotImplementedError("Matrix Power not defined")

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self * other**S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        raise NotImplementedError()
        #return MatMul(other, Pow(self, S.NegativeOne))

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def is_square(self) -> bool | None:
        rows, cols = self.shape
        if isinstance(rows, Integer) and isinstance(cols, Integer):
            return rows == cols
        if rows == cols:
            return True
        return None

    def _eval_conjugate(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(Transpose(self))

    def as_real_imag(self, deep=True, **hints):
        return self._eval_as_real_imag()

    def _eval_as_real_imag(self):
        real = S.Half * (self + self._eval_conjugate())
        im = (self - self._eval_conjugate())/(2*S.ImaginaryUnit)
        return (real, im)

    def _eval_inverse(self):
        return Inverse(self)

    def _eval_determinant(self):
        return Determinant(self)

    def _eval_transpose(self):
        return Transpose(self)

    def _eval_trace(self):
        return None

    def _eval_power(self, exp):
        """
        Override this in sub-classes to implement simplification of powers.  The cases where the exponent
        is -1, 0, 1 are already covered in MatPow.doit(), so implementations can exclude these cases.
        """
        return MatPow(self, exp)

    def _eval_simplify(self, **kwargs):
        if self.is_Atom:
            return self
        else:
            from sympy.simplify import simplify
            return self.func(*[simplify(x, **kwargs) for x in self.args])

    def _eval_adjoint(self):
        from sympy.matrices.expressions.adjoint import Adjoint
        return Adjoint(self)

    def _eval_derivative_n_times(self, x, n):
        return Basic._eval_derivative_n_times(self, x, n)

    def _eval_derivative(self, x):
        # `x` is a scalar:
        if self.has(x):
            # See if there are other methods using it:
            return super()._eval_derivative(x)
        else:
            return ZeroMatrix(*self.shape)

    @classmethod
    def _check_dim(cls, dim):
        """Helper function to check invalid matrix dimensions"""
        ok = not dim.is_Float and check_assumptions(
            dim, integer=True, nonnegative=True)
        if ok is False:
            raise ValueError(
                "The dimension specification {} should be "
                "a nonnegative integer.".format(dim))


    def _entry(self, i, j, **kwargs):
        raise NotImplementedError(
            "Indexing not implemented for %s" % self.__class__.__name__)

    def adjoint(self):
        return adjoint(self)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return S.One, self

    def conjugate(self):
        return conjugate(self)

    def transpose(self):
        from sympy.matrices.expressions.transpose import transpose
        return transpose(self)

    @property
    def T(self):
        '''Matrix transposition'''
        return self.transpose()

    def inverse(self):
        if self.is_square is False:
            raise NonSquareMatrixError('Inverse of non-square matrix')
        return self._eval_inverse()

    def inv(self):
        return self.inverse()

    def det(self):
        from sympy.matrices.expressions.determinant import det
        return det(self)

    @property
    def I(self):
        return self.inverse()

    def valid_index(self, i, j):
        def is_valid(idx):
            return isinstance(idx, (int, Integer, Symbol, Expr))
        return (is_valid(i) and is_valid(j) and
                (self.rows is None or
                (i >= -self.rows) != False and (i < self.rows) != False) and
                (j >= -self.cols) != False and (j < self.cols) != False)

    def __getitem__(self, key):
        if not isinstance(key, tuple) and isinstance(key, slice):
            from sympy.matrices.expressions.slice import MatrixSlice
            return MatrixSlice(self, key, (0, None, 1))
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
            if isinstance(i, slice) or isinstance(j, slice):
                from sympy.matrices.expressions.slice import MatrixSlice
                return MatrixSlice(self, i, j)
            i, j = _sympify(i), _sympify(j)
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid indices (%s, %s)" % (i, j))
        elif isinstance(key, (SYMPY_INTS, Integer)):
            # row-wise decomposition of matrix
            rows, cols = self.shape
            # allow single indexing if number of columns is known
            if not isinstance(cols, Integer):
                raise IndexError(filldedent('''
                    Single indexing is only supported when the number
                    of columns is known.'''))
            key = _sympify(key)
            i = key // cols
            j = key % cols
            if self.valid_index(i, j) != False:
                return self._entry(i, j)
            else:
                raise IndexError("Invalid index %s" % key)
        elif isinstance(key, (Symbol, Expr)):
            raise IndexError(filldedent('''
                Only integers may be used when addressing the matrix
                with a single index.'''))
        raise IndexError("Invalid index, wanted %s[i,j]" % self)

    def _is_shape_symbolic(self) -> bool:
        return (not isinstance(self.rows, (SYMPY_INTS, Integer))
            or not isinstance(self.cols, (SYMPY_INTS, Integer)))

    def as_explicit(self):
        """
        Returns a dense Matrix with elements represented explicitly

        Returns an object of type ImmutableDenseMatrix.

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.as_explicit()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_mutable: returns mutable Matrix type

        """
        if self._is_shape_symbolic():
            raise ValueError(
                'Matrix with symbolic shape '
                'cannot be represented explicitly.')
        from sympy.matrices.immutable import ImmutableDenseMatrix
        return ImmutableDenseMatrix([[self[i, j]
                            for j in range(self.cols)]
                            for i in range(self.rows)])

    def as_mutable(self):
        """
        Returns a dense, mutable matrix with elements represented explicitly

        Examples
        ========

        >>> from sympy import Identity
        >>> I = Identity(3)
        >>> I
        I
        >>> I.shape
        (3, 3)
        >>> I.as_mutable()
        Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

        See Also
        ========
        as_explicit: returns ImmutableDenseMatrix
        """
        return self.as_explicit().as_mutable()

    def __array__(self, dtype=object, copy=None):
        if copy is not None and not copy:
            raise TypeError("Cannot implement copy=False when converting Matrix to ndarray")
        from numpy import empty
        a = empty(self.shape, dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                a[i, j] = self[i, j]
        return a

    def equals(self, other):
        """
        Test elementwise equality between matrices, potentially of different
        types

        >>> from sympy import Identity, eye
        >>> Identity(3).equals(eye(3))
        True
        """
        return self.as_explicit().equals(other)

    def canonicalize(self):
        return self

    def as_coeff_mmul(self):
        return S.One, MatMul(self)

    @staticmethod
    def from_index_summation(expr, first_index=None, last_index=None, dimensions=None):
        r"""
        Parse expression of matrices with explicitly summed indices into a
        matrix expression without indices, if possible.

        This transformation expressed in mathematical notation:

        `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

        Optional parameter ``first_index``: specify which free index to use as
        the index starting the expression.

        Examples
        ========

        >>> from sympy import MatrixSymbol, MatrixExpr, Sum
        >>> from sympy.abc import i, j, k, l, N
        >>> A = MatrixSymbol("A", N, N)
        >>> B = MatrixSymbol("B", N, N)
        >>> expr = Sum(A[i, j]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B

        Transposition is detected:

        >>> expr = Sum(A[j, i]*B[j, k], (j, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A.T*B

        Detect the trace:

        >>> expr = Sum(A[i, i], (i, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        Trace(A)

        More complicated expressions:

        >>> expr = Sum(A[i, j]*B[k, j]*A[l, k], (j, 0, N-1), (k, 0, N-1))
        >>> MatrixExpr.from_index_summation(expr)
        A*B.T*A.T
        """
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
        first_indices = []
        if first_index is not None:
            first_indices.append(first_index)
        if last_index is not None:
            first_indices.append(last_index)
        arr = convert_indexed_to_array(expr, first_indices=first_indices)
        return convert_array_to_matrix(arr)

    def applyfunc(self, func):
        from .applyfunc import ElementwiseApplyFunction
        return ElementwiseApplyFunction(func, self)


@dispatch(MatrixExpr, Expr)
def _eval_is_eq(lhs, rhs): # noqa:F811
    return False

@dispatch(MatrixExpr, MatrixExpr)  # type: ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    if lhs.shape != rhs.shape:
        return False
    if (lhs - rhs).is_ZeroMatrix:
        return True

def get_postprocessor(cls):
    def _postprocessor(expr):
        # To avoid circular imports, we can't have MatMul/MatAdd on the top level
        mat_class = {Mul: MatMul, Add: MatAdd}[cls]
        nonmatrices = []
        matrices = []
        for term in expr.args:
            if isinstance(term, MatrixExpr):
                matrices.append(term)
            else:
                nonmatrices.append(term)

        if not matrices:
            return cls._from_args(nonmatrices)

        if nonmatrices:
            if cls == Mul:
                for i in range(len(matrices)):
                    if not matrices[i].is_MatrixExpr:
                        # If one of the matrices explicit, absorb the scalar into it
                        # (doit will combine all explicit matrices into one, so it
                        # doesn't matter which)
                        matrices[i] = matrices[i].__mul__(cls._from_args(nonmatrices))
                        nonmatrices = []
                        break

            else:
                # Maintain the ability to create Add(scalar, matrix) without
                # raising an exception. That way different algorithms can
                # replace matrix expressions with non-commutative symbols to
                # manipulate them like non-commutative scalars.
                return cls._from_args(nonmatrices + [mat_class(*matrices).doit(deep=False)])

        if mat_class == MatAdd:
            return mat_class(*matrices).doit(deep=False)
        return mat_class(cls._from_args(nonmatrices), *matrices).doit(deep=False)
    return _postprocessor


Basic._constructor_postprocessor_mapping[MatrixExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
}


def _matrix_derivative(expr, x, old_algorithm=False):

    if isinstance(expr, MatrixBase) or isinstance(x, MatrixBase):
        # Do not use array expressions for explicit matrices:
        old_algorithm = True

    if old_algorithm:
        return _matrix_derivative_old_algorithm(expr, x)

    from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
    from sympy.tensor.array.expressions.arrayexpr_derivatives import array_derive
    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix

    array_expr = convert_matrix_to_array(expr)
    diff_array_expr = array_derive(array_expr, x)
    diff_matrix_expr = convert_array_to_matrix(diff_array_expr)
    return diff_matrix_expr


def _matrix_derivative_old_algorithm(expr, x):
    from sympy.tensor.array.array_derivatives import ArrayDerivative
    lines = expr._eval_derivative_matrix_lines(x)

    parts = [i.build() for i in lines]

    from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix

    parts = [[convert_array_to_matrix(j) for j in i] for i in parts]

    def _get_shape(elem):
        if isinstance(elem, MatrixExpr):
            return elem.shape
        return 1, 1

    def get_rank(parts):
        return sum(j not in (1, None) for i in parts for j in _get_shape(i))

    ranks = [get_rank(i) for i in parts]
    rank = ranks[0]

    def contract_one_dims(parts):
        if len(parts) == 1:
            return parts[0]
        else:
            p1, p2 = parts[:2]
            if p2.is_Matrix:
                p2 = p2.T
            if p1 == Identity(1):
                pbase = p2
            elif p2 == Identity(1):
                pbase = p1
            else:
                pbase = p1*p2
            if len(parts) == 2:
                return pbase
            else:  # len(parts) > 2
                if pbase.is_Matrix:
                    raise ValueError("")
                return pbase*Mul.fromiter(parts[2:])

    if rank <= 2:
        return Add.fromiter([contract_one_dims(i) for i in parts])

    return ArrayDerivative(expr, x)


class MatrixElement(Expr):
    parent = property(lambda self: self.args[0])
    i = property(lambda self: self.args[1])
    j = property(lambda self: self.args[2])
    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, n, m):
        n, m = map(_sympify, (n, m))
        if isinstance(name, str):
            name = Symbol(name)
        else:
            if isinstance(name, MatrixBase):
                if n.is_Integer and m.is_Integer:
                    return name[n, m]
                name = _sympify(name)  # change mutable into immutable
            else:
                name = _sympify(name)
                if not isinstance(name.kind, MatrixKind):
                    raise TypeError("First argument of MatrixElement should be a matrix")
            if not getattr(name, 'valid_index', lambda n, m: True)(n, m):
                raise IndexError('indices out of range')
        obj = Expr.__new__(cls, name, n, m)
        return obj

    @property
    def symbol(self):
        return self.args[0]

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return args[0][args[1], args[2]]

    @property
    def indices(self):
        return self.args[1:]

    def _eval_derivative(self, v):

        if not isinstance(v, MatrixElement):
            return self.parent.diff(v)[self.i, self.j]

        M = self.args[0]

        m, n = self.parent.shape

        if M == v.args[0]:
            return KroneckerDelta(self.args[1], v.args[1], (0, m-1)) * \
                   KroneckerDelta(self.args[2], v.args[2], (0, n-1))

        if isinstance(M, Inverse):
            from sympy.concrete.summations import Sum
            i, j = self.args[1:]
            i1, i2 = symbols("z1, z2", cls=Dummy)
            Y = M.args[0]
            r1, r2 = Y.shape
            return -Sum(M[i, i1]*Y[i1, i2].diff(v)*M[i2, j], (i1, 0, r1-1), (i2, 0, r2-1))

        if self.has(v.args[0]):
            return None

        return S.Zero


class MatrixSymbol(MatrixExpr):
    """Symbolic representation of a Matrix object

    Creates a SymPy Symbol to represent a Matrix. This matrix has a shape and
    can be included in Matrix Expressions

    Examples
    ========

    >>> from sympy import MatrixSymbol, Identity
    >>> A = MatrixSymbol('A', 3, 4) # A 3 by 4 Matrix
    >>> B = MatrixSymbol('B', 4, 3) # A 4 by 3 Matrix
    >>> A.shape
    (3, 4)
    >>> 2*A*B + Identity(3)
    I + 2*A*B
    """
    is_commutative = False
    is_symbol = True
    _diff_wrt = True

    def __new__(cls, name, n, m):
        n, m = _sympify(n), _sympify(m)

        cls._check_dim(m)
        cls._check_dim(n)

        if isinstance(name, str):
            name = Str(name)
        obj = Basic.__new__(cls, name, n, m)
        return obj

    @property
    def shape(self):
        return self.args[1], self.args[2]

    @property
    def name(self):
        return self.args[0].name

    def _entry(self, i, j, **kwargs):
        return MatrixElement(self, i, j)

    @property
    def free_symbols(self):
        return {self}

    def _eval_simplify(self, **kwargs):
        return self

    def _eval_derivative(self, x):
        # x is a scalar:
        return ZeroMatrix(self.shape[0], self.shape[1])

    def _eval_derivative_matrix_lines(self, x):
        if self != x:
            first = ZeroMatrix(x.shape[0], self.shape[0]) if self.shape[0] != 1 else S.Zero
            second = ZeroMatrix(x.shape[1], self.shape[1]) if self.shape[1] != 1 else S.Zero
            return [_LeftRightArgs(
                [first, second],
            )]
        else:
            first = Identity(self.shape[0]) if self.shape[0] != 1 else S.One
            second = Identity(self.shape[1]) if self.shape[1] != 1 else S.One
            return [_LeftRightArgs(
                [first, second],
            )]


def matrix_symbols(expr):
    return [sym for sym in expr.free_symbols if sym.is_Matrix]


class _LeftRightArgs:
    r"""
    Helper class to compute matrix derivatives.

    The logic: when an expression is derived by a matrix `X_{mn}`, two lines of
    matrix multiplications are created: the one contracted to `m` (first line),
    and the one contracted to `n` (second line).

    Transposition flips the side by which new matrices are connected to the
    lines.

    The trace connects the end of the two lines.
    """

    def __init__(self, lines, higher=S.One):
        self._lines = list(lines)
        self._first_pointer_parent = self._lines
        self._first_pointer_index = 0
        self._first_line_index = 0
        self._second_pointer_parent = self._lines
        self._second_pointer_index = 1
        self._second_line_index = 1
        self.higher = higher

    @property
    def first_pointer(self):
        return self._first_pointer_parent[self._first_pointer_index]

    @first_pointer.setter
    def first_pointer(self, value):
        self._first_pointer_parent[self._first_pointer_index] = value

    @property
    def second_pointer(self):
        return self._second_pointer_parent[self._second_pointer_index]

    @second_pointer.setter
    def second_pointer(self, value):
        self._second_pointer_parent[self._second_pointer_index] = value

    def __repr__(self):
        built = [self._build(i) for i in self._lines]
        return "_LeftRightArgs(lines=%s, higher=%s)" % (
            built,
            self.higher,
        )

    def transpose(self):
        self._first_pointer_parent, self._second_pointer_parent = self._second_pointer_parent, self._first_pointer_parent
        self._first_pointer_index, self._second_pointer_index = self._second_pointer_index, self._first_pointer_index
        self._first_line_index, self._second_line_index = self._second_line_index, self._first_line_index
        return self

    @staticmethod
    def _build(expr):
        if isinstance(expr, ExprBuilder):
            return expr.build()
        if isinstance(expr, list):
            if len(expr) == 1:
                return expr[0]
            else:
                return expr[0](*[_LeftRightArgs._build(i) for i in expr[1]])
        else:
            return expr

    def build(self):
        data = [self._build(i) for i in self._lines]
        if self.higher != 1:
            data += [self._build(self.higher)]
        data = list(data)
        return data

    def matrix_form(self):
        if self.first != 1 and self.higher != 1:
            raise ValueError("higher dimensional array cannot be represented")

        def _get_shape(elem):
            if isinstance(elem, MatrixExpr):
                return elem.shape
            return (None, None)

        if _get_shape(self.first)[1] != _get_shape(self.second)[1]:
            # Remove one-dimensional identity matrices:
            # (this is needed by `a.diff(a)` where `a` is a vector)
            if _get_shape(self.second) == (1, 1):
                return self.first*self.second[0, 0]
            if _get_shape(self.first) == (1, 1):
                return self.first[1, 1]*self.second.T
            raise ValueError("incompatible shapes")
        if self.first != 1:
            return self.first*self.second.T
        else:
            return self.higher

    def rank(self):
        """
        Number of dimensions different from trivial (warning: not related to
        matrix rank).
        """
        rank = 0
        if self.first != 1:
            rank += sum(i != 1 for i in self.first.shape)
        if self.second != 1:
            rank += sum(i != 1 for i in self.second.shape)
        if self.higher != 1:
            rank += 2
        return rank

    def _multiply_pointer(self, pointer, other):
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from ...tensor.array.expressions.array_expressions import ArrayContraction

        subexpr = ExprBuilder(
            ArrayContraction,
            [
                ExprBuilder(
                    ArrayTensorProduct,
                    [
                        pointer,
                        other
                    ]
                ),
                (1, 2)
            ],
            validator=ArrayContraction._validate
        )

        return subexpr

    def append_first(self, other):
        self.first_pointer *= other

    def append_second(self, other):
        self.second_pointer *= other


def _make_matrix(x):
    from sympy.matrices.immutable import ImmutableDenseMatrix
    if isinstance(x, MatrixExpr):
        return x
    return ImmutableDenseMatrix([[x]])


from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\traceback.py ===
import inspect
import linecache
import os
import sys
from dataclasses import dataclass, field
from itertools import islice
from traceback import walk_tb
from types import ModuleType, TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pip._vendor.pygments.lexers import guess_lexer_for_filename
from pip._vendor.pygments.token import Comment, Keyword, Name, Number, Operator, String
from pip._vendor.pygments.token import Text as TextToken
from pip._vendor.pygments.token import Token
from pip._vendor.pygments.util import ClassNotFound

from . import pretty
from ._loop import loop_first_last, loop_last
from .columns import Columns
from .console import (
    Console,
    ConsoleOptions,
    ConsoleRenderable,
    Group,
    RenderResult,
    group,
)
from .constrain import Constrain
from .highlighter import RegexHighlighter, ReprHighlighter
from .panel import Panel
from .scope import render_scope
from .style import Style
from .syntax import Syntax, SyntaxPosition
from .text import Text
from .theme import Theme

WINDOWS = sys.platform == "win32"

LOCALS_MAX_LENGTH = 10
LOCALS_MAX_STRING = 80


def _iter_syntax_lines(
    start: SyntaxPosition, end: SyntaxPosition
) -> Iterable[Tuple[int, int, int]]:
    """Yield start and end positions per line.

    Args:
        start: Start position.
        end: End position.

    Returns:
        Iterable of (LINE, COLUMN1, COLUMN2).
    """

    line1, column1 = start
    line2, column2 = end

    if line1 == line2:
        yield line1, column1, column2
    else:
        for first, last, line_no in loop_first_last(range(line1, line2 + 1)):
            if first:
                yield line_no, column1, -1
            elif last:
                yield line_no, 0, column2
            else:
                yield line_no, 0, -1


def install(
    *,
    console: Optional[Console] = None,
    width: Optional[int] = 100,
    code_width: Optional[int] = 88,
    extra_lines: int = 3,
    theme: Optional[str] = None,
    word_wrap: bool = False,
    show_locals: bool = False,
    locals_max_length: int = LOCALS_MAX_LENGTH,
    locals_max_string: int = LOCALS_MAX_STRING,
    locals_hide_dunder: bool = True,
    locals_hide_sunder: Optional[bool] = None,
    indent_guides: bool = True,
    suppress: Iterable[Union[str, ModuleType]] = (),
    max_frames: int = 100,
) -> Callable[[Type[BaseException], BaseException, Optional[TracebackType]], Any]:
    """Install a rich traceback handler.

    Once installed, any tracebacks will be printed with syntax highlighting and rich formatting.


    Args:
        console (Optional[Console], optional): Console to write exception to. Default uses internal Console instance.
        width (Optional[int], optional): Width (in characters) of traceback. Defaults to 100.
        code_width (Optional[int], optional): Code width (in characters) of traceback. Defaults to 88.
        extra_lines (int, optional): Extra lines of code. Defaults to 3.
        theme (Optional[str], optional): Pygments theme to use in traceback. Defaults to ``None`` which will pick
            a theme appropriate for the platform.
        word_wrap (bool, optional): Enable word wrapping of long lines. Defaults to False.
        show_locals (bool, optional): Enable display of local variables. Defaults to False.
        locals_max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to 10.
        locals_max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to 80.
        locals_hide_dunder (bool, optional): Hide locals prefixed with double underscore. Defaults to True.
        locals_hide_sunder (bool, optional): Hide locals prefixed with single underscore. Defaults to False.
        indent_guides (bool, optional): Enable indent guides in code and locals. Defaults to True.
        suppress (Sequence[Union[str, ModuleType]]): Optional sequence of modules or paths to exclude from traceback.

    Returns:
        Callable: The previous exception handler that was replaced.

    """
    traceback_console = Console(stderr=True) if console is None else console

    locals_hide_sunder = (
        True
        if (traceback_console.is_jupyter and locals_hide_sunder is None)
        else locals_hide_sunder
    )

    def excepthook(
        type_: Type[BaseException],
        value: BaseException,
        traceback: Optional[TracebackType],
    ) -> None:
        exception_traceback = Traceback.from_exception(
            type_,
            value,
            traceback,
            width=width,
            code_width=code_width,
            extra_lines=extra_lines,
            theme=theme,
            word_wrap=word_wrap,
            show_locals=show_locals,
            locals_max_length=locals_max_length,
            locals_max_string=locals_max_string,
            locals_hide_dunder=locals_hide_dunder,
            locals_hide_sunder=bool(locals_hide_sunder),
            indent_guides=indent_guides,
            suppress=suppress,
            max_frames=max_frames,
        )
        traceback_console.print(exception_traceback)

    def ipy_excepthook_closure(ip: Any) -> None:  # pragma: no cover
        tb_data = {}  # store information about showtraceback call
        default_showtraceback = ip.showtraceback  # keep reference of default traceback

        def ipy_show_traceback(*args: Any, **kwargs: Any) -> None:
            """wrap the default ip.showtraceback to store info for ip._showtraceback"""
            nonlocal tb_data
            tb_data = kwargs
            default_showtraceback(*args, **kwargs)

        def ipy_display_traceback(
            *args: Any, is_syntax: bool = False, **kwargs: Any
        ) -> None:
            """Internally called traceback from ip._showtraceback"""
            nonlocal tb_data
            exc_tuple = ip._get_exc_info()

            # do not display trace on syntax error
            tb: Optional[TracebackType] = None if is_syntax else exc_tuple[2]

            # determine correct tb_offset
            compiled = tb_data.get("running_compiled_code", False)
            tb_offset = tb_data.get("tb_offset", 1 if compiled else 0)
            # remove ipython internal frames from trace with tb_offset
            for _ in range(tb_offset):
                if tb is None:
                    break
                tb = tb.tb_next

            excepthook(exc_tuple[0], exc_tuple[1], tb)
            tb_data = {}  # clear data upon usage

        # replace _showtraceback instead of showtraceback to allow ipython features such as debugging to work
        # this is also what the ipython docs recommends to modify when subclassing InteractiveShell
        ip._showtraceback = ipy_display_traceback
        # add wrapper to capture tb_data
        ip.showtraceback = ipy_show_traceback
        ip.showsyntaxerror = lambda *args, **kwargs: ipy_display_traceback(
            *args, is_syntax=True, **kwargs
        )

    try:  # pragma: no cover
        # if within ipython, use customized traceback
        ip = get_ipython()  # type: ignore[name-defined]
        ipy_excepthook_closure(ip)
        return sys.excepthook
    except Exception:
        # otherwise use default system hook
        old_excepthook = sys.excepthook
        sys.excepthook = excepthook
        return old_excepthook


@dataclass
class Frame:
    filename: str
    lineno: int
    name: str
    line: str = ""
    locals: Optional[Dict[str, pretty.Node]] = None
    last_instruction: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None


@dataclass
class _SyntaxError:
    offset: int
    filename: str
    line: str
    lineno: int
    msg: str
    notes: List[str] = field(default_factory=list)


@dataclass
class Stack:
    exc_type: str
    exc_value: str
    syntax_error: Optional[_SyntaxError] = None
    is_cause: bool = False
    frames: List[Frame] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    is_group: bool = False
    exceptions: List["Trace"] = field(default_factory=list)


@dataclass
class Trace:
    stacks: List[Stack]


class PathHighlighter(RegexHighlighter):
    highlights = [r"(?P<dim>.*/)(?P<bold>.+)"]


class Traceback:
    """A Console renderable that renders a traceback.

    Args:
        trace (Trace, optional): A `Trace` object produced from `extract`. Defaults to None, which uses
            the last exception.
        width (Optional[int], optional): Number of characters used to traceback. Defaults to 100.
        code_width (Optional[int], optional): Number of code characters used to traceback. Defaults to 88.
        extra_lines (int, optional): Additional lines of code to render. Defaults to 3.
        theme (str, optional): Override pygments theme used in traceback.
        word_wrap (bool, optional): Enable word wrapping of long lines. Defaults to False.
        show_locals (bool, optional): Enable display of local variables. Defaults to False.
        indent_guides (bool, optional): Enable indent guides in code and locals. Defaults to True.
        locals_max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
            Defaults to 10.
        locals_max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to 80.
        locals_hide_dunder (bool, optional): Hide locals prefixed with double underscore. Defaults to True.
        locals_hide_sunder (bool, optional): Hide locals prefixed with single underscore. Defaults to False.
        suppress (Sequence[Union[str, ModuleType]]): Optional sequence of modules or paths to exclude from traceback.
        max_frames (int): Maximum number of frames to show in a traceback, 0 for no maximum. Defaults to 100.

    """

    LEXERS = {
        "": "text",
        ".py": "python",
        ".pxd": "cython",
        ".pyx": "cython",
        ".pxi": "pyrex",
    }

    def __init__(
        self,
        trace: Optional[Trace] = None,
        *,
        width: Optional[int] = 100,
        code_width: Optional[int] = 88,
        extra_lines: int = 3,
        theme: Optional[str] = None,
        word_wrap: bool = False,
        show_locals: bool = False,
        locals_max_length: int = LOCALS_MAX_LENGTH,
        locals_max_string: int = LOCALS_MAX_STRING,
        locals_hide_dunder: bool = True,
        locals_hide_sunder: bool = False,
        indent_guides: bool = True,
        suppress: Iterable[Union[str, ModuleType]] = (),
        max_frames: int = 100,
    ):
        if trace is None:
            exc_type, exc_value, traceback = sys.exc_info()
            if exc_type is None or exc_value is None or traceback is None:
                raise ValueError(
                    "Value for 'trace' required if not called in except: block"
                )
            trace = self.extract(
                exc_type, exc_value, traceback, show_locals=show_locals
            )
        self.trace = trace
        self.width = width
        self.code_width = code_width
        self.extra_lines = extra_lines
        self.theme = Syntax.get_theme(theme or "ansi_dark")
        self.word_wrap = word_wrap
        self.show_locals = show_locals
        self.indent_guides = indent_guides
        self.locals_max_length = locals_max_length
        self.locals_max_string = locals_max_string
        self.locals_hide_dunder = locals_hide_dunder
        self.locals_hide_sunder = locals_hide_sunder

        self.suppress: Sequence[str] = []
        for suppress_entity in suppress:
            if not isinstance(suppress_entity, str):
                assert (
                    suppress_entity.__file__ is not None
                ), f"{suppress_entity!r} must be a module with '__file__' attribute"
                path = os.path.dirname(suppress_entity.__file__)
            else:
                path = suppress_entity
            path = os.path.normpath(os.path.abspath(path))
            self.suppress.append(path)
        self.max_frames = max(4, max_frames) if max_frames > 0 else 0

    @classmethod
    def from_exception(
        cls,
        exc_type: Type[Any],
        exc_value: BaseException,
        traceback: Optional[TracebackType],
        *,
        width: Optional[int] = 100,
        code_width: Optional[int] = 88,
        extra_lines: int = 3,
        theme: Optional[str] = None,
        word_wrap: bool = False,
        show_locals: bool = False,
        locals_max_length: int = LOCALS_MAX_LENGTH,
        locals_max_string: int = LOCALS_MAX_STRING,
        locals_hide_dunder: bool = True,
        locals_hide_sunder: bool = False,
        indent_guides: bool = True,
        suppress: Iterable[Union[str, ModuleType]] = (),
        max_frames: int = 100,
    ) -> "Traceback":
        """Create a traceback from exception info

        Args:
            exc_type (Type[BaseException]): Exception type.
            exc_value (BaseException): Exception value.
            traceback (TracebackType): Python Traceback object.
            width (Optional[int], optional): Number of characters used to traceback. Defaults to 100.
            code_width (Optional[int], optional): Number of code characters used to traceback. Defaults to 88.
            extra_lines (int, optional): Additional lines of code to render. Defaults to 3.
            theme (str, optional): Override pygments theme used in traceback.
            word_wrap (bool, optional): Enable word wrapping of long lines. Defaults to False.
            show_locals (bool, optional): Enable display of local variables. Defaults to False.
            indent_guides (bool, optional): Enable indent guides in code and locals. Defaults to True.
            locals_max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
                Defaults to 10.
            locals_max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to 80.
            locals_hide_dunder (bool, optional): Hide locals prefixed with double underscore. Defaults to True.
            locals_hide_sunder (bool, optional): Hide locals prefixed with single underscore. Defaults to False.
            suppress (Iterable[Union[str, ModuleType]]): Optional sequence of modules or paths to exclude from traceback.
            max_frames (int): Maximum number of frames to show in a traceback, 0 for no maximum. Defaults to 100.

        Returns:
            Traceback: A Traceback instance that may be printed.
        """
        rich_traceback = cls.extract(
            exc_type,
            exc_value,
            traceback,
            show_locals=show_locals,
            locals_max_length=locals_max_length,
            locals_max_string=locals_max_string,
            locals_hide_dunder=locals_hide_dunder,
            locals_hide_sunder=locals_hide_sunder,
        )

        return cls(
            rich_traceback,
            width=width,
            code_width=code_width,
            extra_lines=extra_lines,
            theme=theme,
            word_wrap=word_wrap,
            show_locals=show_locals,
            indent_guides=indent_guides,
            locals_max_length=locals_max_length,
            locals_max_string=locals_max_string,
            locals_hide_dunder=locals_hide_dunder,
            locals_hide_sunder=locals_hide_sunder,
            suppress=suppress,
            max_frames=max_frames,
        )

    @classmethod
    def extract(
        cls,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        traceback: Optional[TracebackType],
        *,
        show_locals: bool = False,
        locals_max_length: int = LOCALS_MAX_LENGTH,
        locals_max_string: int = LOCALS_MAX_STRING,
        locals_hide_dunder: bool = True,
        locals_hide_sunder: bool = False,
    ) -> Trace:
        """Extract traceback information.

        Args:
            exc_type (Type[BaseException]): Exception type.
            exc_value (BaseException): Exception value.
            traceback (TracebackType): Python Traceback object.
            show_locals (bool, optional): Enable display of local variables. Defaults to False.
            locals_max_length (int, optional): Maximum length of containers before abbreviating, or None for no abbreviation.
                Defaults to 10.
            locals_max_string (int, optional): Maximum length of string before truncating, or None to disable. Defaults to 80.
            locals_hide_dunder (bool, optional): Hide locals prefixed with double underscore. Defaults to True.
            locals_hide_sunder (bool, optional): Hide locals prefixed with single underscore. Defaults to False.

        Returns:
            Trace: A Trace instance which you can use to construct a `Traceback`.
        """

        stacks: List[Stack] = []
        is_cause = False

        from pip._vendor.rich import _IMPORT_CWD

        notes: List[str] = getattr(exc_value, "__notes__", None) or []

        def safe_str(_object: Any) -> str:
            """Don't allow exceptions from __str__ to propagate."""
            try:
                return str(_object)
            except Exception:
                return "<exception str() failed>"

        while True:
            stack = Stack(
                exc_type=safe_str(exc_type.__name__),
                exc_value=safe_str(exc_value),
                is_cause=is_cause,
                notes=notes,
            )

            if sys.version_info >= (3, 11):
                if isinstance(exc_value, (BaseExceptionGroup, ExceptionGroup)):
                    stack.is_group = True
                    for exception in exc_value.exceptions:
                        stack.exceptions.append(
                            Traceback.extract(
                                type(exception),
                                exception,
                                exception.__traceback__,
                                show_locals=show_locals,
                                locals_max_length=locals_max_length,
                                locals_hide_dunder=locals_hide_dunder,
                                locals_hide_sunder=locals_hide_sunder,
                            )
                        )

            if isinstance(exc_value, SyntaxError):
                stack.syntax_error = _SyntaxError(
                    offset=exc_value.offset or 0,
                    filename=exc_value.filename or "?",
                    lineno=exc_value.lineno or 0,
                    line=exc_value.text or "",
                    msg=exc_value.msg,
                    notes=notes,
                )

            stacks.append(stack)
            append = stack.frames.append

            def get_locals(
                iter_locals: Iterable[Tuple[str, object]],
            ) -> Iterable[Tuple[str, object]]:
                """Extract locals from an iterator of key pairs."""
                if not (locals_hide_dunder or locals_hide_sunder):
                    yield from iter_locals
                    return
                for key, value in iter_locals:
                    if locals_hide_dunder and key.startswith("__"):
                        continue
                    if locals_hide_sunder and key.startswith("_"):
                        continue
                    yield key, value

            for frame_summary, line_no in walk_tb(traceback):
                filename = frame_summary.f_code.co_filename

                last_instruction: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
                last_instruction = None
                if sys.version_info >= (3, 11):
                    instruction_index = frame_summary.f_lasti // 2
                    instruction_position = next(
                        islice(
                            frame_summary.f_code.co_positions(),
                            instruction_index,
                            instruction_index + 1,
                        )
                    )
                    (
                        start_line,
                        end_line,
                        start_column,
                        end_column,
                    ) = instruction_position
                    if (
                        start_line is not None
                        and end_line is not None
                        and start_column is not None
                        and end_column is not None
                    ):
                        last_instruction = (
                            (start_line, start_column),
                            (end_line, end_column),
                        )

                if filename and not filename.startswith("<"):
                    if not os.path.isabs(filename):
                        filename = os.path.join(_IMPORT_CWD, filename)
                if frame_summary.f_locals.get("_rich_traceback_omit", False):
                    continue

                frame = Frame(
                    filename=filename or "?",
                    lineno=line_no,
                    name=frame_summary.f_code.co_name,
                    locals=(
                        {
                            key: pretty.traverse(
                                value,
                                max_length=locals_max_length,
                                max_string=locals_max_string,
                            )
                            for key, value in get_locals(frame_summary.f_locals.items())
                            if not (inspect.isfunction(value) or inspect.isclass(value))
                        }
                        if show_locals
                        else None
                    ),
                    last_instruction=last_instruction,
                )
                append(frame)
                if frame_summary.f_locals.get("_rich_traceback_guard", False):
                    del stack.frames[:]

            cause = getattr(exc_value, "__cause__", None)
            if cause:
                exc_type = cause.__class__
                exc_value = cause
                # __traceback__ can be None, e.g. for exceptions raised by the
                # 'multiprocessing' module
                traceback = cause.__traceback__
                is_cause = True
                continue

            cause = exc_value.__context__
            if cause and not getattr(exc_value, "__suppress_context__", False):
                exc_type = cause.__class__
                exc_value = cause
                traceback = cause.__traceback__
                is_cause = False
                continue
            # No cover, code is reached but coverage doesn't recognize it.
            break  # pragma: no cover

        trace = Trace(stacks=stacks)

        return trace

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        theme = self.theme
        background_style = theme.get_background_style()
        token_style = theme.get_style_for_token

        traceback_theme = Theme(
            {
                "pretty": token_style(TextToken),
                "pygments.text": token_style(Token),
                "pygments.string": token_style(String),
                "pygments.function": token_style(Name.Function),
                "pygments.number": token_style(Number),
                "repr.indent": token_style(Comment) + Style(dim=True),
                "repr.str": token_style(String),
                "repr.brace": token_style(TextToken) + Style(bold=True),
                "repr.number": token_style(Number),
                "repr.bool_true": token_style(Keyword.Constant),
                "repr.bool_false": token_style(Keyword.Constant),
                "repr.none": token_style(Keyword.Constant),
                "scope.border": token_style(String.Delimiter),
                "scope.equals": token_style(Operator),
                "scope.key": token_style(Name),
                "scope.key.special": token_style(Name.Constant) + Style(dim=True),
            },
            inherit=False,
        )

        highlighter = ReprHighlighter()

        @group()
        def render_stack(stack: Stack, last: bool) -> RenderResult:
            if stack.frames:
                stack_renderable: ConsoleRenderable = Panel(
                    self._render_stack(stack),
                    title="[traceback.title]Traceback [dim](most recent call last)",
                    style=background_style,
                    border_style="traceback.border",
                    expand=True,
                    padding=(0, 1),
                )
                stack_renderable = Constrain(stack_renderable, self.width)
                with console.use_theme(traceback_theme):
                    yield stack_renderable

            if stack.syntax_error is not None:
                with console.use_theme(traceback_theme):
                    yield Constrain(
                        Panel(
                            self._render_syntax_error(stack.syntax_error),
                            style=background_style,
                            border_style="traceback.border.syntax_error",
                            expand=True,
                            padding=(0, 1),
                            width=self.width,
                        ),
                        self.width,
                    )
                yield Text.assemble(
                    (f"{stack.exc_type}: ", "traceback.exc_type"),
                    highlighter(stack.syntax_error.msg),
                )
            elif stack.exc_value:
                yield Text.assemble(
                    (f"{stack.exc_type}: ", "traceback.exc_type"),
                    highlighter(stack.exc_value),
                )
            else:
                yield Text.assemble((f"{stack.exc_type}", "traceback.exc_type"))

            for note in stack.notes:
                yield Text.assemble(("[NOTE] ", "traceback.note"), highlighter(note))

            if stack.is_group:
                for group_no, group_exception in enumerate(stack.exceptions, 1):
                    grouped_exceptions: List[Group] = []
                    for group_last, group_stack in loop_last(group_exception.stacks):
                        grouped_exceptions.append(render_stack(group_stack, group_last))
                    yield ""
                    yield Constrain(
                        Panel(
                            Group(*grouped_exceptions),
                            title=f"Sub-exception #{group_no}",
                            border_style="traceback.group.border",
                        ),
                        self.width,
                    )

            if not last:
                if stack.is_cause:
                    yield Text.from_markup(
                        "\n[i]The above exception was the direct cause of the following exception:\n",
                    )
                else:
                    yield Text.from_markup(
                        "\n[i]During handling of the above exception, another exception occurred:\n",
                    )

        for last, stack in loop_last(reversed(self.trace.stacks)):
            yield render_stack(stack, last)

    @group()
    def _render_syntax_error(self, syntax_error: _SyntaxError) -> RenderResult:
        highlighter = ReprHighlighter()
        path_highlighter = PathHighlighter()
        if syntax_error.filename != "<stdin>":
            if os.path.exists(syntax_error.filename):
                text = Text.assemble(
                    (f" {syntax_error.filename}", "pygments.string"),
                    (":", "pygments.text"),
                    (str(syntax_error.lineno), "pygments.number"),
                    style="pygments.text",
                )
                yield path_highlighter(text)
        syntax_error_text = highlighter(syntax_error.line.rstrip())
        syntax_error_text.no_wrap = True
        offset = min(syntax_error.offset - 1, len(syntax_error_text))
        syntax_error_text.stylize("bold underline", offset, offset)
        syntax_error_text += Text.from_markup(
            "\n" + " " * offset + "[traceback.offset]▲[/]",
            style="pygments.text",
        )
        yield syntax_error_text

    @classmethod
    def _guess_lexer(cls, filename: str, code: str) -> str:
        ext = os.path.splitext(filename)[-1]
        if not ext:
            # No extension, look at first line to see if it is a hashbang
            # Note, this is an educated guess and not a guarantee
            # If it fails, the only downside is that the code is highlighted strangely
            new_line_index = code.index("\n")
            first_line = code[:new_line_index] if new_line_index != -1 else code
            if first_line.startswith("#!") and "python" in first_line.lower():
                return "python"
        try:
            return cls.LEXERS.get(ext) or guess_lexer_for_filename(filename, code).name
        except ClassNotFound:
            return "text"

    @group()
    def _render_stack(self, stack: Stack) -> RenderResult:
        path_highlighter = PathHighlighter()
        theme = self.theme

        def render_locals(frame: Frame) -> Iterable[ConsoleRenderable]:
            if frame.locals:
                yield render_scope(
                    frame.locals,
                    title="locals",
                    indent_guides=self.indent_guides,
                    max_length=self.locals_max_length,
                    max_string=self.locals_max_string,
                )

        exclude_frames: Optional[range] = None
        if self.max_frames != 0:
            exclude_frames = range(
                self.max_frames // 2,
                len(stack.frames) - self.max_frames // 2,
            )

        excluded = False
        for frame_index, frame in enumerate(stack.frames):
            if exclude_frames and frame_index in exclude_frames:
                excluded = True
                continue

            if excluded:
                assert exclude_frames is not None
                yield Text(
                    f"\n... {len(exclude_frames)} frames hidden ...",
                    justify="center",
                    style="traceback.error",
                )
                excluded = False

            first = frame_index == 0
            frame_filename = frame.filename
            suppressed = any(frame_filename.startswith(path) for path in self.suppress)

            if os.path.exists(frame.filename):
                text = Text.assemble(
                    path_highlighter(Text(frame.filename, style="pygments.string")),
                    (":", "pygments.text"),
                    (str(frame.lineno), "pygments.number"),
                    " in ",
                    (frame.name, "pygments.function"),
                    style="pygments.text",
                )
            else:
                text = Text.assemble(
                    "in ",
                    (frame.name, "pygments.function"),
                    (":", "pygments.text"),
                    (str(frame.lineno), "pygments.number"),
                    style="pygments.text",
                )
            if not frame.filename.startswith("<") and not first:
                yield ""
            yield text
            if frame.filename.startswith("<"):
                yield from render_locals(frame)
                continue
            if not suppressed:
                try:
                    code_lines = linecache.getlines(frame.filename)
                    code = "".join(code_lines)
                    if not code:
                        # code may be an empty string if the file doesn't exist, OR
                        # if the traceback filename is generated dynamically
                        continue
                    lexer_name = self._guess_lexer(frame.filename, code)
                    syntax = Syntax(
                        code,
                        lexer_name,
                        theme=theme,
                        line_numbers=True,
                        line_range=(
                            frame.lineno - self.extra_lines,
                            frame.lineno + self.extra_lines,
                        ),
                        highlight_lines={frame.lineno},
                        word_wrap=self.word_wrap,
                        code_width=self.code_width,
                        indent_guides=self.indent_guides,
                        dedent=False,
                    )
                    yield ""
                except Exception as error:
                    yield Text.assemble(
                        (f"\n{error}", "traceback.error"),
                    )
                else:
                    if frame.last_instruction is not None:
                        start, end = frame.last_instruction

                        # Stylize a line at a time
                        # So that indentation isn't underlined (which looks bad)
                        for line1, column1, column2 in _iter_syntax_lines(start, end):
                            try:
                                if column1 == 0:
                                    line = code_lines[line1 - 1]
                                    column1 = len(line) - len(line.lstrip())
                                if column2 == -1:
                                    column2 = len(code_lines[line1 - 1])
                            except IndexError:
                                # Being defensive here
                                # If last_instruction reports a line out-of-bounds, we don't want to crash
                                continue

                            syntax.stylize_range(
                                style="traceback.error_range",
                                start=(line1, column1),
                                end=(line1, column2),
                            )
                    yield (
                        Columns(
                            [
                                syntax,
                                *render_locals(frame),
                            ],
                            padding=1,
                        )
                        if frame.locals
                        else syntax
                    )


if __name__ == "__main__":  # pragma: no cover
    install(show_locals=True)
    import sys

    def bar(
        a: Any,
    ) -> None:  # 这是对亚洲语言支持的测试。面对模棱两可的想法，拒绝猜测的诱惑
        one = 1
        print(one / a)

    def foo(a: Any) -> None:
        _rich_traceback_guard = True
        zed = {
            "characters": {
                "Paul Atreides",
                "Vladimir Harkonnen",
                "Thufir Hawat",
                "Duncan Idaho",
            },
            "atomic_types": (None, False, True),
        }
        bar(a)

    def error() -> None:
        foo(0)

    error()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\properties.py ===
# orm/properties.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""MapperProperty implementations.

This is a private module which defines the behavior of individual ORM-
mapped attributes.

"""

from __future__ import annotations

from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import attributes
from . import exc as orm_exc
from . import strategy_options
from .base import _DeclarativeMapped
from .base import class_mapper
from .descriptor_props import CompositeProperty
from .descriptor_props import ConcreteInheritedProperty
from .descriptor_props import SynonymProperty
from .interfaces import _AttributeOptions
from .interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .interfaces import StrategizedProperty
from .relationships import RelationshipProperty
from .util import de_stringify_annotation
from .. import exc as sa_exc
from .. import ForeignKey
from .. import log
from .. import util
from ..sql import coercions
from ..sql import roles
from ..sql.base import _NoArg
from ..sql.schema import Column
from ..sql.schema import SchemaConst
from ..sql.type_api import TypeEngine
from ..util.typing import de_optionalize_union_types
from ..util.typing import get_args
from ..util.typing import includes_none
from ..util.typing import is_a_type
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import is_pep695
from ..util.typing import Self

if TYPE_CHECKING:
    from ._typing import _IdentityKeyType
    from ._typing import _InstanceDict
    from ._typing import _ORMColumnExprArgument
    from ._typing import _RegistryType
    from .base import Mapped
    from .decl_base import _ClassScanMapperConfig
    from .mapper import Mapper
    from .session import Session
    from .state import _InstallLoaderCallableProto
    from .state import InstanceState
    from ..sql._typing import _InfoType
    from ..sql.elements import ColumnElement
    from ..sql.elements import NamedColumn
    from ..sql.operators import OperatorType
    from ..util.typing import _AnnotationScanType
    from ..util.typing import RODescriptorReference

_T = TypeVar("_T", bound=Any)
_PT = TypeVar("_PT", bound=Any)
_NC = TypeVar("_NC", bound="NamedColumn[Any]")

__all__ = [
    "ColumnProperty",
    "CompositeProperty",
    "ConcreteInheritedProperty",
    "RelationshipProperty",
    "SynonymProperty",
]


@log.class_logger
class ColumnProperty(
    _MapsColumns[_T],
    StrategizedProperty[_T],
    _IntrospectsAnnotations,
    log.Identified,
):
    """Describes an object attribute that corresponds to a table column
    or other column expression.

    Public constructor is the :func:`_orm.column_property` function.

    """

    strategy_wildcard_key = strategy_options._COLUMN_TOKEN
    inherit_cache = True
    """:meta private:"""

    _links_to_entity = False

    columns: List[NamedColumn[Any]]

    _is_polymorphic_discriminator: bool

    _mapped_by_synonym: Optional[str]

    comparator_factory: Type[PropComparator[_T]]

    __slots__ = (
        "columns",
        "group",
        "deferred",
        "instrument",
        "comparator_factory",
        "active_history",
        "expire_on_flush",
        "_creation_order",
        "_is_polymorphic_discriminator",
        "_mapped_by_synonym",
        "_deferred_column_loader",
        "_raise_column_loader",
        "_renders_in_subqueries",
        "raiseload",
    )

    def __init__(
        self,
        column: _ORMColumnExprArgument[_T],
        *additional_columns: _ORMColumnExprArgument[Any],
        attribute_options: Optional[_AttributeOptions] = None,
        group: Optional[str] = None,
        deferred: bool = False,
        raiseload: bool = False,
        comparator_factory: Optional[Type[PropComparator[_T]]] = None,
        active_history: bool = False,
        expire_on_flush: bool = True,
        info: Optional[_InfoType] = None,
        doc: Optional[str] = None,
        _instrument: bool = True,
        _assume_readonly_dc_attributes: bool = False,
    ):
        super().__init__(
            attribute_options=attribute_options,
            _assume_readonly_dc_attributes=_assume_readonly_dc_attributes,
        )
        columns = (column,) + additional_columns
        self.columns = [
            coercions.expect(roles.LabeledColumnExprRole, c) for c in columns
        ]
        self.group = group
        self.deferred = deferred
        self.raiseload = raiseload
        self.instrument = _instrument
        self.comparator_factory = (
            comparator_factory
            if comparator_factory is not None
            else self.__class__.Comparator
        )
        self.active_history = active_history
        self.expire_on_flush = expire_on_flush

        if info is not None:
            self.info.update(info)

        if doc is not None:
            self.doc = doc
        else:
            for col in reversed(self.columns):
                doc = getattr(col, "doc", None)
                if doc is not None:
                    self.doc = doc
                    break
            else:
                self.doc = None

        util.set_creation_order(self)

        self.strategy_key = (
            ("deferred", self.deferred),
            ("instrument", self.instrument),
        )
        if self.raiseload:
            self.strategy_key += (("raiseload", True),)

    def declarative_scan(
        self,
        decl_scan: _ClassScanMapperConfig,
        registry: _RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
        mapped_container: Optional[Type[Mapped[Any]]],
        annotation: Optional[_AnnotationScanType],
        extracted_mapped_annotation: Optional[_AnnotationScanType],
        is_dataclass_field: bool,
    ) -> None:
        column = self.columns[0]
        if column.key is None:
            column.key = key
        if column.name is None:
            column.name = key

    @property
    def mapper_property_to_assign(self) -> Optional[MapperProperty[_T]]:
        return self

    @property
    def columns_to_assign(self) -> List[Tuple[Column[Any], int]]:
        # mypy doesn't care about the isinstance here
        return [
            (c, 0)  # type: ignore
            for c in self.columns
            if isinstance(c, Column) and c.table is None
        ]

    def _memoized_attr__renders_in_subqueries(self) -> bool:
        if ("query_expression", True) in self.strategy_key:
            return self.strategy._have_default_expression  # type: ignore

        return ("deferred", True) not in self.strategy_key or (
            self not in self.parent._readonly_props  # type: ignore
        )

    @util.preload_module("sqlalchemy.orm.state", "sqlalchemy.orm.strategies")
    def _memoized_attr__deferred_column_loader(
        self,
    ) -> _InstallLoaderCallableProto[Any]:
        state = util.preloaded.orm_state
        strategies = util.preloaded.orm_strategies
        return state.InstanceState._instance_level_callable_processor(
            self.parent.class_manager,
            strategies.LoadDeferredColumns(self.key),
            self.key,
        )

    @util.preload_module("sqlalchemy.orm.state", "sqlalchemy.orm.strategies")
    def _memoized_attr__raise_column_loader(
        self,
    ) -> _InstallLoaderCallableProto[Any]:
        state = util.preloaded.orm_state
        strategies = util.preloaded.orm_strategies
        return state.InstanceState._instance_level_callable_processor(
            self.parent.class_manager,
            strategies.LoadDeferredColumns(self.key, True),
            self.key,
        )

    def __clause_element__(self) -> roles.ColumnsClauseRole:
        """Allow the ColumnProperty to work in expression before it is turned
        into an instrumented attribute.
        """

        return self.expression

    @property
    def expression(self) -> roles.ColumnsClauseRole:
        """Return the primary column or expression for this ColumnProperty.

        E.g.::


            class File(Base):
                # ...

                name = Column(String(64))
                extension = Column(String(8))
                filename = column_property(name + "." + extension)
                path = column_property("C:/" + filename.expression)

        .. seealso::

            :ref:`mapper_column_property_sql_expressions_composed`

        """
        return self.columns[0]

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        if not self.instrument:
            return

        attributes.register_descriptor(
            mapper.class_,
            self.key,
            comparator=self.comparator_factory(self, mapper),
            parententity=mapper,
            doc=self.doc,
        )

    def do_init(self) -> None:
        super().do_init()

        if len(self.columns) > 1 and set(self.parent.primary_key).issuperset(
            self.columns
        ):
            util.warn(
                (
                    "On mapper %s, primary key column '%s' is being combined "
                    "with distinct primary key column '%s' in attribute '%s'. "
                    "Use explicit properties to give each column its own "
                    "mapped attribute name."
                )
                % (self.parent, self.columns[1], self.columns[0], self.key)
            )

    def copy(self) -> ColumnProperty[_T]:
        return ColumnProperty(
            *self.columns,
            deferred=self.deferred,
            group=self.group,
            active_history=self.active_history,
        )

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
        if not self.instrument:
            return
        elif self.key in source_dict:
            value = source_dict[self.key]

            if not load:
                dest_dict[self.key] = value
            else:
                impl = dest_state.get_impl(self.key)
                impl.set(dest_state, dest_dict, value, None)
        elif dest_state.has_identity and self.key not in dest_dict:
            dest_state._expire_attributes(
                dest_dict, [self.key], no_loader=True
            )

    class Comparator(util.MemoizedSlots, PropComparator[_PT]):
        """Produce boolean, comparison, and other operators for
        :class:`.ColumnProperty` attributes.

        See the documentation for :class:`.PropComparator` for a brief
        overview.

        .. seealso::

            :class:`.PropComparator`

            :class:`.ColumnOperators`

            :ref:`types_operators`

            :attr:`.TypeEngine.comparator_factory`

        """

        if not TYPE_CHECKING:
            # prevent pylance from being clever about slots
            __slots__ = "__clause_element__", "info", "expressions"

        prop: RODescriptorReference[ColumnProperty[_PT]]

        expressions: Sequence[NamedColumn[Any]]
        """The full sequence of columns referenced by this
         attribute, adjusted for any aliasing in progress.

        .. versionadded:: 1.3.17

        .. seealso::

           :ref:`maptojoin` - usage example
        """

        def _orm_annotate_column(self, column: _NC) -> _NC:
            """annotate and possibly adapt a column to be returned
            as the mapped-attribute exposed version of the column.

            The column in this context needs to act as much like the
            column in an ORM mapped context as possible, so includes
            annotations to give hints to various ORM functions as to
            the source entity of this column.   It also adapts it
            to the mapper's with_polymorphic selectable if one is
            present.

            """

            pe = self._parententity
            annotations: Dict[str, Any] = {
                "entity_namespace": pe,
                "parententity": pe,
                "parentmapper": pe,
                "proxy_key": self.prop.key,
            }

            col = column

            # for a mapper with polymorphic_on and an adapter, return
            # the column against the polymorphic selectable.
            # see also orm.util._orm_downgrade_polymorphic_columns
            # for the reverse operation.
            if self._parentmapper._polymorphic_adapter:
                mapper_local_col = col
                col = self._parentmapper._polymorphic_adapter.traverse(col)

                # this is a clue to the ORM Query etc. that this column
                # was adapted to the mapper's polymorphic_adapter.  the
                # ORM uses this hint to know which column its adapting.
                annotations["adapt_column"] = mapper_local_col

            return col._annotate(annotations)._set_propagate_attrs(
                {"compile_state_plugin": "orm", "plugin_subject": pe}
            )

        if TYPE_CHECKING:

            def __clause_element__(self) -> NamedColumn[_PT]: ...

        def _memoized_method___clause_element__(
            self,
        ) -> NamedColumn[_PT]:
            if self.adapter:
                return self.adapter(self.prop.columns[0], self.prop.key)
            else:
                return self._orm_annotate_column(self.prop.columns[0])

        def _memoized_attr_info(self) -> _InfoType:
            """The .info dictionary for this attribute."""

            ce = self.__clause_element__()
            try:
                return ce.info  # type: ignore
            except AttributeError:
                return self.prop.info

        def _memoized_attr_expressions(self) -> Sequence[NamedColumn[Any]]:
            """The full sequence of columns referenced by this
            attribute, adjusted for any aliasing in progress.

            .. versionadded:: 1.3.17

            """
            if self.adapter:
                return [
                    self.adapter(col, self.prop.key)
                    for col in self.prop.columns
                ]
            else:
                return [
                    self._orm_annotate_column(col) for col in self.prop.columns
                ]

        def _fallback_getattr(self, key: str) -> Any:
            """proxy attribute access down to the mapped column.

            this allows user-defined comparison methods to be accessed.
            """
            return getattr(self.__clause_element__(), key)

        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnElement[Any]:
            return op(self.__clause_element__(), *other, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

        def reverse_operate(
            self, op: OperatorType, other: Any, **kwargs: Any
        ) -> ColumnElement[Any]:
            col = self.__clause_element__()
            return op(col._bind_param(op, other), col, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def __str__(self) -> str:
        if not self.parent or not self.key:
            return object.__repr__(self)
        return str(self.parent.class_.__name__) + "." + self.key


class MappedSQLExpression(ColumnProperty[_T], _DeclarativeMapped[_T]):
    """Declarative front-end for the :class:`.ColumnProperty` class.

    Public constructor is the :func:`_orm.column_property` function.

    .. versionchanged:: 2.0 Added :class:`_orm.MappedSQLExpression` as
       a Declarative compatible subclass for :class:`_orm.ColumnProperty`.

    .. seealso::

        :class:`.MappedColumn`

    """

    inherit_cache = True
    """:meta private:"""


class MappedColumn(
    _IntrospectsAnnotations,
    _MapsColumns[_T],
    _DeclarativeMapped[_T],
):
    """Maps a single :class:`_schema.Column` on a class.

    :class:`_orm.MappedColumn` is a specialization of the
    :class:`_orm.ColumnProperty` class and is oriented towards declarative
    configuration.

    To construct :class:`_orm.MappedColumn` objects, use the
    :func:`_orm.mapped_column` constructor function.

    .. versionadded:: 2.0


    """

    __slots__ = (
        "column",
        "_creation_order",
        "_sort_order",
        "foreign_keys",
        "_has_nullable",
        "_has_insert_default",
        "deferred",
        "deferred_group",
        "deferred_raiseload",
        "active_history",
        "_attribute_options",
        "_has_dataclass_arguments",
        "_use_existing_column",
    )

    deferred: Union[_NoArg, bool]
    deferred_raiseload: bool
    deferred_group: Optional[str]

    column: Column[_T]
    foreign_keys: Optional[Set[ForeignKey]]
    _attribute_options: _AttributeOptions

    def __init__(self, *arg: Any, **kw: Any):
        self._attribute_options = attr_opts = kw.pop(
            "attribute_options", _DEFAULT_ATTRIBUTE_OPTIONS
        )

        self._use_existing_column = kw.pop("use_existing_column", False)

        self._has_dataclass_arguments = (
            attr_opts is not None
            and attr_opts != _DEFAULT_ATTRIBUTE_OPTIONS
            and any(
                attr_opts[i] is not _NoArg.NO_ARG
                for i, attr in enumerate(attr_opts._fields)
                if attr != "dataclasses_default"
            )
        )

        insert_default = kw.pop("insert_default", _NoArg.NO_ARG)
        self._has_insert_default = insert_default is not _NoArg.NO_ARG

        if self._has_insert_default:
            kw["default"] = insert_default
        elif attr_opts.dataclasses_default is not _NoArg.NO_ARG:
            kw["default"] = attr_opts.dataclasses_default

        self.deferred_group = kw.pop("deferred_group", None)
        self.deferred_raiseload = kw.pop("deferred_raiseload", None)
        self.deferred = kw.pop("deferred", _NoArg.NO_ARG)
        self.active_history = kw.pop("active_history", False)

        self._sort_order = kw.pop("sort_order", _NoArg.NO_ARG)
        self.column = cast("Column[_T]", Column(*arg, **kw))
        self.foreign_keys = self.column.foreign_keys
        self._has_nullable = "nullable" in kw and kw.get("nullable") not in (
            None,
            SchemaConst.NULL_UNSPECIFIED,
        )
        util.set_creation_order(self)

    def _copy(self, **kw: Any) -> Self:
        new = self.__class__.__new__(self.__class__)
        new.column = self.column._copy(**kw)
        new.deferred = self.deferred
        new.deferred_group = self.deferred_group
        new.deferred_raiseload = self.deferred_raiseload
        new.foreign_keys = new.column.foreign_keys
        new.active_history = self.active_history
        new._has_nullable = self._has_nullable
        new._attribute_options = self._attribute_options
        new._has_insert_default = self._has_insert_default
        new._has_dataclass_arguments = self._has_dataclass_arguments
        new._use_existing_column = self._use_existing_column
        new._sort_order = self._sort_order
        util.set_creation_order(new)
        return new

    @property
    def name(self) -> str:
        return self.column.name

    @property
    def mapper_property_to_assign(self) -> Optional[MapperProperty[_T]]:
        effective_deferred = self.deferred
        if effective_deferred is _NoArg.NO_ARG:
            effective_deferred = bool(
                self.deferred_group or self.deferred_raiseload
            )

        if effective_deferred or self.active_history:
            return ColumnProperty(
                self.column,
                deferred=effective_deferred,
                group=self.deferred_group,
                raiseload=self.deferred_raiseload,
                attribute_options=self._attribute_options,
                active_history=self.active_history,
            )
        else:
            return None

    @property
    def columns_to_assign(self) -> List[Tuple[Column[Any], int]]:
        return [
            (
                self.column,
                (
                    self._sort_order
                    if self._sort_order is not _NoArg.NO_ARG
                    else 0
                ),
            )
        ]

    def __clause_element__(self) -> Column[_T]:
        return self.column

    def operate(
        self, op: OperatorType, *other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        return op(self.__clause_element__(), *other, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def reverse_operate(
        self, op: OperatorType, other: Any, **kwargs: Any
    ) -> ColumnElement[Any]:
        col = self.__clause_element__()
        return op(col._bind_param(op, other), col, **kwargs)  # type: ignore[no-any-return]  # noqa: E501

    def found_in_pep593_annotated(self) -> Any:
        # return a blank mapped_column().  This mapped_column()'s
        # Column will be merged into it in _init_column_for_annotation().
        return MappedColumn()

    def declarative_scan(
        self,
        decl_scan: _ClassScanMapperConfig,
        registry: _RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
        mapped_container: Optional[Type[Mapped[Any]]],
        annotation: Optional[_AnnotationScanType],
        extracted_mapped_annotation: Optional[_AnnotationScanType],
        is_dataclass_field: bool,
    ) -> None:
        column = self.column

        if (
            self._use_existing_column
            and decl_scan.inherits
            and decl_scan.single
        ):
            if decl_scan.is_deferred:
                raise sa_exc.ArgumentError(
                    "Can't use use_existing_column with deferred mappers"
                )
            supercls_mapper = class_mapper(decl_scan.inherits, False)

            colname = column.name if column.name is not None else key
            column = self.column = supercls_mapper.local_table.c.get(  # type: ignore[assignment] # noqa: E501
                colname, column
            )

        if column.key is None:
            column.key = key
        if column.name is None:
            column.name = key

        sqltype = column.type

        if extracted_mapped_annotation is None:
            if sqltype._isnull and not self.column.foreign_keys:
                self._raise_for_required(key, cls)
            else:
                return

        self._init_column_for_annotation(
            cls,
            registry,
            extracted_mapped_annotation,
            originating_module,
        )

    @util.preload_module("sqlalchemy.orm.decl_base")
    def declarative_scan_for_composite(
        self,
        registry: _RegistryType,
        cls: Type[Any],
        originating_module: Optional[str],
        key: str,
        param_name: str,
        param_annotation: _AnnotationScanType,
    ) -> None:
        decl_base = util.preloaded.orm_decl_base
        decl_base._undefer_column_name(param_name, self.column)
        self._init_column_for_annotation(
            cls, registry, param_annotation, originating_module
        )

    def _init_column_for_annotation(
        self,
        cls: Type[Any],
        registry: _RegistryType,
        argument: _AnnotationScanType,
        originating_module: Optional[str],
    ) -> None:
        sqltype = self.column.type

        if is_fwd_ref(
            argument, check_generic=True, check_for_plain_string=True
        ):
            assert originating_module is not None
            argument = de_stringify_annotation(
                cls, argument, originating_module, include_generic=True
            )

        nullable = includes_none(argument)

        if not self._has_nullable:
            self.column.nullable = nullable

        our_type = de_optionalize_union_types(argument)

        find_mapped_in: Tuple[Any, ...] = ()
        our_type_is_pep593 = False
        raw_pep_593_type = None

        if is_pep593(our_type):
            our_type_is_pep593 = True

            pep_593_components = get_args(our_type)
            raw_pep_593_type = pep_593_components[0]
            if nullable:
                raw_pep_593_type = de_optionalize_union_types(raw_pep_593_type)
            find_mapped_in = pep_593_components[1:]
        elif is_pep695(argument) and is_pep593(argument.__value__):
            # do not support nested annotation inside unions ets
            find_mapped_in = get_args(argument.__value__)[1:]

        use_args_from: Optional[MappedColumn[Any]]
        for elem in find_mapped_in:
            if isinstance(elem, MappedColumn):
                use_args_from = elem
                break
        else:
            use_args_from = None

        if use_args_from is not None:
            if (
                not self._has_insert_default
                and use_args_from.column.default is not None
            ):
                self.column.default = None

            use_args_from.column._merge(self.column)
            sqltype = self.column.type

            if (
                use_args_from.deferred is not _NoArg.NO_ARG
                and self.deferred is _NoArg.NO_ARG
            ):
                self.deferred = use_args_from.deferred

            if (
                use_args_from.deferred_group is not None
                and self.deferred_group is None
            ):
                self.deferred_group = use_args_from.deferred_group

            if (
                use_args_from.deferred_raiseload is not None
                and self.deferred_raiseload is None
            ):
                self.deferred_raiseload = use_args_from.deferred_raiseload

            if (
                use_args_from._use_existing_column
                and not self._use_existing_column
            ):
                self._use_existing_column = True

            if use_args_from.active_history:
                self.active_history = use_args_from.active_history

            if (
                use_args_from._sort_order is not None
                and self._sort_order is _NoArg.NO_ARG
            ):
                self._sort_order = use_args_from._sort_order

            if (
                use_args_from.column.key is not None
                or use_args_from.column.name is not None
            ):
                util.warn_deprecated(
                    "Can't use the 'key' or 'name' arguments in "
                    "Annotated with mapped_column(); this will be ignored",
                    "2.0.22",
                )

            if use_args_from._has_dataclass_arguments:
                for idx, arg in enumerate(
                    use_args_from._attribute_options._fields
                ):
                    if (
                        use_args_from._attribute_options[idx]
                        is not _NoArg.NO_ARG
                    ):
                        arg = arg.replace("dataclasses_", "")
                        util.warn_deprecated(
                            f"Argument '{arg}' is a dataclass argument and "
                            "cannot be specified within a mapped_column() "
                            "bundled inside of an Annotated object",
                            "2.0.22",
                        )

        if sqltype._isnull and not self.column.foreign_keys:
            checks: List[Any]
            if our_type_is_pep593:
                checks = [our_type, raw_pep_593_type]
            else:
                checks = [our_type]

            for check_type in checks:
                new_sqltype = registry._resolve_type(check_type)
                if new_sqltype is not None:
                    break
            else:
                if isinstance(our_type, TypeEngine) or (
                    isinstance(our_type, type)
                    and issubclass(our_type, TypeEngine)
                ):
                    raise orm_exc.MappedAnnotationError(
                        f"The type provided inside the {self.column.key!r} "
                        "attribute Mapped annotation is the SQLAlchemy type "
                        f"{our_type}. Expected a Python type instead"
                    )
                elif is_a_type(our_type):
                    raise orm_exc.MappedAnnotationError(
                        "Could not locate SQLAlchemy Core type for Python "
                        f"type {our_type} inside the {self.column.key!r} "
                        "attribute Mapped annotation"
                    )
                else:
                    raise orm_exc.MappedAnnotationError(
                        f"The object provided inside the {self.column.key!r} "
                        "attribute Mapped annotation is not a Python type, "
                        f"it's the object {our_type!r}. Expected a Python "
                        "type."
                    )

            self.column._set_type(new_sqltype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\minpoly.py ===
"""Minimal polynomials for algebraic numbers."""

from functools import reduce

from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets

from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
    NotAlgebraic,
    GeneratorsError,
)
from sympy.polys.polytools import (
    Poly, PurePoly, invert, factor_list, groebner, resultant,
    degree, poly_from_expr, parallel_poly_from_expr, lcm
)
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
    numbered_symbols, public, sift
)


def _choose_factor(factors, x, v, dom=QQ, prec=200, bound=5):
    """
    Return a factor having root ``v``
    It is assumed that one of the factors has root ``v``.
    """

    if isinstance(factors[0], tuple):
        factors = [f[0] for f in factors]
    if len(factors) == 1:
        return factors[0]

    prec1 = 10
    points = {}
    symbols = dom.symbols if hasattr(dom, 'symbols') else []
    while prec1 <= prec:
        # when dealing with non-Rational numbers we usually evaluate
        # with `subs` argument but we only need a ballpark evaluation
        fe = [f.as_expr().xreplace({x:v}) for f in factors]
        if v.is_number:
            fe = [f.n(prec) for f in fe]

        # assign integers [0, n) to symbols (if any)
        for n in subsets(range(bound), k=len(symbols), repetition=True):
            for s, i in zip(symbols, n):
                points[s] = i

            # evaluate the expression at these points
            candidates = [(abs(f.subs(points).n(prec1)), i)
                for i,f in enumerate(fe)]

            # if we get invalid numbers (e.g. from division by zero)
            # we try again
            if any(i in _illegal for i, _ in candidates):
                continue

            # find the smallest two -- if they differ significantly
            # then we assume we have found the factor that becomes
            # 0 when v is substituted into it
            can = sorted(candidates)
            (a, ix), (b, _) = can[:2]
            if b > a * 10**6:  # XXX what to use?
                return factors[ix]

        prec1 *= 2

    raise NotImplementedError("multiple candidates for the minimal polynomial of %s" % v)


def _is_sum_surds(p):
    return all(f.is_Rational or f.is_Pow and
        f.base.is_Rational and (2*f.exp).is_Integer and f.is_extended_real
        for t in Add.make_args(p) for f in Mul.make_args(t))


def _separate_sq(p):
    """
    helper function for ``_minimal_polynomial_sq``

    It selects a rational ``g`` such that the polynomial ``p``
    consists of a sum of terms whose surds squared have gcd equal to ``g``
    and a sum of terms with surds squared prime with ``g``;
    then it takes the field norm to eliminate ``sqrt(g)``

    See simplify.simplify.split_surds and polytools.sqf_norm.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.minpoly import _separate_sq
    >>> p= -x + sqrt(2) + sqrt(3) + sqrt(7)
    >>> p = _separate_sq(p); p
    -x**2 + 2*sqrt(3)*x + 2*sqrt(7)*x - 2*sqrt(21) - 8
    >>> p = _separate_sq(p); p
    -x**4 + 4*sqrt(7)*x**3 - 32*x**2 + 8*sqrt(7)*x + 20
    >>> p = _separate_sq(p); p
    -x**8 + 48*x**6 - 536*x**4 + 1728*x**2 - 400

    """
    def is_sqrt(expr):
        return expr.is_Pow and expr.exp is S.Half
    # p = c1*sqrt(q1) + ... + cn*sqrt(qn) -> a = [(c1, q1), .., (cn, qn)]
    a = []
    for y in p.args:
        if not y.is_Mul:
            if is_sqrt(y):
                a.append((S.One, y**2))
            elif y.is_Atom:
                a.append((y, S.One))
            elif y.is_Pow and y.exp.is_integer:
                a.append((y, S.One))
            else:
                raise NotImplementedError
        else:
            T, F = sift(y.args, is_sqrt, binary=True)
            a.append((Mul(*F), Mul(*T)**2))
    a.sort(key=lambda z: z[1])
    if a[-1][1] is S.One:
        # there are no surds
        return p
    surds = [z for y, z in a]
    for i in range(len(surds)):
        if surds[i] != 1:
            break
    from sympy.simplify.radsimp import _split_gcd
    g, b1, b2 = _split_gcd(*surds[i:])
    a1 = []
    a2 = []
    for y, z in a:
        if z in b1:
            a1.append(y*z**S.Half)
        else:
            a2.append(y*z**S.Half)
    p1 = Add(*a1)
    p2 = Add(*a2)
    p = _mexpand(p1**2) - _mexpand(p2**2)
    return p

def _minimal_polynomial_sq(p, n, x):
    """
    Returns the minimal polynomial for the ``nth-root`` of a sum of surds
    or ``None`` if it fails.

    Parameters
    ==========

    p : sum of surds
    n : positive integer
    x : variable of the returned polynomial

    Examples
    ========

    >>> from sympy.polys.numberfields.minpoly import _minimal_polynomial_sq
    >>> from sympy import sqrt
    >>> from sympy.abc import x
    >>> q = 1 + sqrt(2) + sqrt(3)
    >>> _minimal_polynomial_sq(q, 3, x)
    x**12 - 4*x**9 - 4*x**6 + 16*x**3 - 8

    """
    p = sympify(p)
    n = sympify(n)
    if not n.is_Integer or not n > 0 or not _is_sum_surds(p):
        return None
    pn = p**Rational(1, n)
    # eliminate the square roots
    p -= x
    while 1:
        p1 = _separate_sq(p)
        if p1 is p:
            p = p1.subs({x:x**n})
            break
        else:
            p = p1

    # _separate_sq eliminates field extensions in a minimal way, so that
    # if n = 1 then `p = constant*(minimal_polynomial(p))`
    # if n > 1 it contains the minimal polynomial as a factor.
    if n == 1:
        p1 = Poly(p)
        if p.coeff(x**p1.degree(x)) < 0:
            p = -p
        p = p.primitive()[1]
        return p
    # by construction `p` has root `pn`
    # the minimal polynomial is the factor vanishing in x = pn
    factors = factor_list(p)[1]

    result = _choose_factor(factors, x, pn)
    return result

def _minpoly_op_algebraic_element(op, ex1, ex2, x, dom, mp1=None, mp2=None):
    """
    return the minimal polynomial for ``op(ex1, ex2)``

    Parameters
    ==========

    op : operation ``Add`` or ``Mul``
    ex1, ex2 : expressions for the algebraic elements
    x : indeterminate of the polynomials
    dom: ground domain
    mp1, mp2 : minimal polynomials for ``ex1`` and ``ex2`` or None

    Examples
    ========

    >>> from sympy import sqrt, Add, Mul, QQ
    >>> from sympy.polys.numberfields.minpoly import _minpoly_op_algebraic_element
    >>> from sympy.abc import x, y
    >>> p1 = sqrt(sqrt(2) + 1)
    >>> p2 = sqrt(sqrt(2) - 1)
    >>> _minpoly_op_algebraic_element(Mul, p1, p2, x, QQ)
    x - 1
    >>> q1 = sqrt(y)
    >>> q2 = 1 / y
    >>> _minpoly_op_algebraic_element(Add, q1, q2, x, QQ.frac_field(y))
    x**2*y**2 - 2*x*y - y**3 + 1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Resultant
    .. [2] I.M. Isaacs, Proc. Amer. Math. Soc. 25 (1970), 638
           "Degrees of sums in a separable field extension".

    """
    y = Dummy(str(x))
    if mp1 is None:
        mp1 = _minpoly_compose(ex1, x, dom)
    if mp2 is None:
        mp2 = _minpoly_compose(ex2, y, dom)
    else:
        mp2 = mp2.subs({x: y})

    if op is Add:
        # mp1a = mp1.subs({x: x - y})
        if dom == QQ:
            R, X = ring('X', QQ)
            p1 = R(dict_from_expr(mp1)[0])
            p2 = R(dict_from_expr(mp2)[0])
        else:
            (p1, p2), _ = parallel_poly_from_expr((mp1, x - y), x, y)
            r = p1.compose(p2)
            mp1a = r.as_expr()

    elif op is Mul:
        mp1a = _muly(mp1, x, y)
    else:
        raise NotImplementedError('option not available')

    if op is Mul or dom != QQ:
        r = resultant(mp1a, mp2, gens=[y, x])
    else:
        r = rs_compose_add(p1, p2)
        r = expr_from_dict(r.as_expr_dict(), x)

    deg1 = degree(mp1, x)
    deg2 = degree(mp2, y)
    if op is Mul and deg1 == 1 or deg2 == 1:
        # if deg1 = 1, then mp1 = x - a; mp1a = x - y - a;
        # r = mp2(x - a), so that `r` is irreducible
        return r

    r = Poly(r, x, domain=dom)
    _, factors = r.factor_list()
    res = _choose_factor(factors, x, op(ex1, ex2), dom)
    return res.as_expr()


def _invertx(p, x):
    """
    Returns ``expand_mul(x**degree(p, x)*p.subs(x, 1/x))``
    """
    p1 = poly_from_expr(p, x)[0]

    n = degree(p1)
    a = [c * x**(n - i) for (i,), c in p1.terms()]
    return Add(*a)


def _muly(p, x, y):
    """
    Returns ``_mexpand(y**deg*p.subs({x:x / y}))``
    """
    p1 = poly_from_expr(p, x)[0]

    n = degree(p1)
    a = [c * x**i * y**(n - i) for (i,), c in p1.terms()]
    return Add(*a)


def _minpoly_pow(ex, pw, x, dom, mp=None):
    """
    Returns ``minpoly(ex**pw, x)``

    Parameters
    ==========

    ex : algebraic element
    pw : rational number
    x : indeterminate of the polynomial
    dom: ground domain
    mp : minimal polynomial of ``p``

    Examples
    ========

    >>> from sympy import sqrt, QQ, Rational
    >>> from sympy.polys.numberfields.minpoly import _minpoly_pow, minpoly
    >>> from sympy.abc import x, y
    >>> p = sqrt(1 + sqrt(2))
    >>> _minpoly_pow(p, 2, x, QQ)
    x**2 - 2*x - 1
    >>> minpoly(p**2, x)
    x**2 - 2*x - 1
    >>> _minpoly_pow(y, Rational(1, 3), x, QQ.frac_field(y))
    x**3 - y
    >>> minpoly(y**Rational(1, 3), x)
    x**3 - y

    """
    pw = sympify(pw)
    if not mp:
        mp = _minpoly_compose(ex, x, dom)
    if not pw.is_rational:
        raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    if pw < 0:
        if mp == x:
            raise ZeroDivisionError('%s is zero' % ex)
        mp = _invertx(mp, x)
        if pw == -1:
            return mp
        pw = -pw
        ex = 1/ex

    y = Dummy(str(x))
    mp = mp.subs({x: y})
    n, d = pw.as_numer_denom()
    res = Poly(resultant(mp, x**d - y**n, gens=[y]), x, domain=dom)
    _, factors = res.factor_list()
    res = _choose_factor(factors, x, ex**pw, dom)
    return res.as_expr()


def _minpoly_add(x, dom, *a):
    """
    returns ``minpoly(Add(*a), dom, x)``
    """
    mp = _minpoly_op_algebraic_element(Add, a[0], a[1], x, dom)
    p = a[0] + a[1]
    for px in a[2:]:
        mp = _minpoly_op_algebraic_element(Add, p, px, x, dom, mp1=mp)
        p = p + px
    return mp


def _minpoly_mul(x, dom, *a):
    """
    returns ``minpoly(Mul(*a), dom, x)``
    """
    mp = _minpoly_op_algebraic_element(Mul, a[0], a[1], x, dom)
    p = a[0] * a[1]
    for px in a[2:]:
        mp = _minpoly_op_algebraic_element(Mul, p, px, x, dom, mp1=mp)
        p = p * px
    return mp


def _minpoly_sin(ex, x):
    """
    Returns the minimal polynomial of ``sin(ex)``
    see https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            n = c.q
            q = sympify(n)
            if q.is_prime:
                # for a = pi*p/q with q odd prime, using chebyshevt
                # write sin(q*a) = mp(sin(a))*sin(a);
                # the roots of mp(x) are sin(pi*p/q) for p = 1,..., q - 1
                a = dup_chebyshevt(n, ZZ)
                return Add(*[x**(n - i - 1)*a[i] for i in range(n)])
            if c.p == 1:
                if q == 9:
                    return 64*x**6 - 96*x**4 + 36*x**2 - 3

            if n % 2 == 1:
                # for a = pi*p/q with q odd, use
                # sin(q*a) = 0 to see that the minimal polynomial must be
                # a factor of dup_chebyshevt(n, ZZ)
                a = dup_chebyshevt(n, ZZ)
                a = [x**(n - i)*a[i] for i in range(n + 1)]
                r = Add(*a)
                _, factors = factor_list(r)
                res = _choose_factor(factors, x, ex)
                return res

            expr = ((1 - cos(2*c*pi))/2)**S.Half
            res = _minpoly_compose(expr, x, QQ)
            return res

    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_cos(ex, x):
    """
    Returns the minimal polynomial of ``cos(ex)``
    see https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            if c.p == 1:
                if c.q == 7:
                    return 8*x**3 - 4*x**2 - 4*x + 1
                if c.q == 9:
                    return 8*x**3 - 6*x - 1
            elif c.p == 2:
                q = sympify(c.q)
                if q.is_prime:
                    s = _minpoly_sin(ex, x)
                    return _mexpand(s.subs({x:sqrt((1 - x)/2)}))

            # for a = pi*p/q, cos(q*a) =T_q(cos(a)) = (-1)**p
            n = int(c.q)
            a = dup_chebyshevt(n, ZZ)
            a = [x**(n - i)*a[i] for i in range(n + 1)]
            r = Add(*a) - (-1)**c.p
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res

    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_tan(ex, x):
    """
    Returns the minimal polynomial of ``tan(ex)``
    see https://github.com/sympy/sympy/issues/21430
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            c = c * 2
            n = int(c.q)
            a = n if c.p % 2 == 0 else 1
            terms = []
            for k in range((c.p+1)%2, n+1, 2):
                terms.append(a*x**k)
                a = -(a*(n-k-1)*(n-k)) // ((k+1)*(k+2))

            r = Add(*terms)
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res

    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_exp(ex, x):
    """
    Returns the minimal polynomial of ``exp(ex)``
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a == I*pi:
        if c.is_rational:
            q = sympify(c.q)
            if c.p == 1 or c.p == -1:
                if q == 3:
                    return x**2 - x + 1
                if q == 4:
                    return x**4 + 1
                if q == 6:
                    return x**4 - x**2 + 1
                if q == 8:
                    return x**8 + 1
                if q == 9:
                    return x**6 - x**3 + 1
                if q == 10:
                    return x**8 - x**6 + x**4 - x**2 + 1
                if q.is_prime:
                    s = 0
                    for i in range(q):
                        s += (-x)**i
                    return s

            # x**(2*q) = product(factors)
            factors = [cyclotomic_poly(i, x) for i in divisors(2*q)]
            mp = _choose_factor(factors, x, ex)
            return mp
        else:
            raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)


def _minpoly_rootof(ex, x):
    """
    Returns the minimal polynomial of a ``CRootOf`` object.
    """
    p = ex.expr
    p = p.subs({ex.poly.gens[0]:x})
    _, factors = factor_list(p, x)
    result = _choose_factor(factors, x, ex)
    return result


def _minpoly_compose(ex, x, dom):
    """
    Computes the minimal polynomial of an algebraic element
    using operations on minimal polynomials

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, Rational
    >>> from sympy.abc import x, y
    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=True)
    x**2 - 2*x - 1
    >>> minimal_polynomial(sqrt(y) + 1/y, x, compose=True)
    x**2*y**2 - 2*x*y - y**3 + 1

    """
    if ex.is_Rational:
        return ex.q*x - ex.p
    if ex is I:
        _, factors = factor_list(x**2 + 1, x, domain=dom)
        return x**2 + 1 if len(factors) == 1 else x - I

    if ex is S.GoldenRatio:
        _, factors = factor_list(x**2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x**2 - x - 1
        else:
            return _choose_factor(factors, x, (1 + sqrt(5))/2, dom=dom)

    if ex is S.TribonacciConstant:
        _, factors = factor_list(x**3 - x**2 - x - 1, x, domain=dom)
        if len(factors) == 1:
            return x**3 - x**2 - x - 1
        else:
            fac = (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3
            return _choose_factor(factors, x, fac, dom=dom)

    if hasattr(dom, 'symbols') and ex in dom.symbols:
        return x - ex

    if dom.is_QQ and _is_sum_surds(ex):
        # eliminate the square roots
        v = ex
        ex -= x
        while 1:
            ex1 = _separate_sq(ex)
            if ex1 is ex:
                return _choose_factor(factor_list(ex)[1], x, v)
            else:
                ex = ex1

    if ex.is_Add:
        res = _minpoly_add(x, dom, *ex.args)
    elif ex.is_Mul:
        f = Factors(ex).factors
        r = sift(f.items(), lambda itx: itx[0].is_Rational and itx[1].is_Rational)
        if r[True] and dom == QQ:
            ex1 = Mul(*[bx**ex for bx, ex in r[False] + r[None]])
            r1 = dict(r[True])
            dens = [y.q for y in r1.values()]
            lcmdens = reduce(lcm, dens, 1)
            neg1 = S.NegativeOne
            expn1 = r1.pop(neg1, S.Zero)
            nums = [base**(y.p*lcmdens // y.q) for base, y in r1.items()]
            ex2 = Mul(*nums)
            mp1 = minimal_polynomial(ex1, x)
            # use the fact that in SymPy canonicalization products of integers
            # raised to rational powers are organized in relatively prime
            # bases, and that in ``base**(n/d)`` a perfect power is
            # simplified with the root
            # Powers of -1 have to be treated separately to preserve sign.
            mp2 = ex2.q*x**lcmdens - ex2.p*neg1**(expn1*lcmdens)
            ex2 = neg1**expn1 * ex2**Rational(1, lcmdens)
            res = _minpoly_op_algebraic_element(Mul, ex1, ex2, x, dom, mp1=mp1, mp2=mp2)
        else:
            res = _minpoly_mul(x, dom, *ex.args)
    elif ex.is_Pow:
        res = _minpoly_pow(ex.base, ex.exp, x, dom)
    elif ex.__class__ is sin:
        res = _minpoly_sin(ex, x)
    elif ex.__class__ is cos:
        res = _minpoly_cos(ex, x)
    elif ex.__class__ is tan:
        res = _minpoly_tan(ex, x)
    elif ex.__class__ is exp:
        res = _minpoly_exp(ex, x)
    elif ex.__class__ is CRootOf:
        res = _minpoly_rootof(ex, x)
    else:
        raise NotAlgebraic("%s does not seem to be an algebraic element" % ex)
    return res


@public
def minimal_polynomial(ex, x=None, compose=True, polys=False, domain=None):
    """
    Computes the minimal polynomial of an algebraic element.

    Parameters
    ==========

    ex : Expr
        Element or expression whose minimal polynomial is to be calculated.

    x : Symbol, optional
        Independent variable of the minimal polynomial

    compose : boolean, optional (default=True)
        Method to use for computing minimal polynomial. If ``compose=True``
        (default) then ``_minpoly_compose`` is used, if ``compose=False`` then
        groebner bases are used.

    polys : boolean, optional (default=False)
        If ``True`` returns a ``Poly`` object else an ``Expr`` object.

    domain : Domain, optional
        Ground domain

    Notes
    =====

    By default ``compose=True``, the minimal polynomial of the subexpressions of ``ex``
    are computed, then the arithmetic operations on them are performed using the resultant
    and factorization.
    If ``compose=False``, a bottom-up algorithm is used with ``groebner``.
    The default algorithm stalls less frequently.

    If no ground domain is given, it will be generated automatically from the expression.

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, solve, QQ
    >>> from sympy.abc import x, y

    >>> minimal_polynomial(sqrt(2), x)
    x**2 - 2
    >>> minimal_polynomial(sqrt(2), x, domain=QQ.algebraic_field(sqrt(2)))
    x - sqrt(2)
    >>> minimal_polynomial(sqrt(2) + sqrt(3), x)
    x**4 - 10*x**2 + 1
    >>> minimal_polynomial(solve(x**3 + x + 3)[0], x)
    x**3 + x + 3
    >>> minimal_polynomial(sqrt(y), x)
    x**2 - y

    """

    ex = sympify(ex)
    if ex.is_number:
        # not sure if it's always needed but try it for numbers (issue 8354)
        ex = _mexpand(ex, recursive=True)
    for expr in preorder_traversal(ex):
        if expr.is_AlgebraicNumber:
            compose = False
            break

    if x is not None:
        x, cls = sympify(x), Poly
    else:
        x, cls = Dummy('x'), PurePoly

    if not domain:
        if ex.free_symbols:
            domain = FractionField(QQ, list(ex.free_symbols))
        else:
            domain = QQ
    if hasattr(domain, 'symbols') and x in domain.symbols:
        raise GeneratorsError("the variable %s is an element of the ground "
                              "domain %s" % (x, domain))

    if compose:
        result = _minpoly_compose(ex, x, domain)
        result = result.primitive()[1]
        c = result.coeff(x**degree(result, x))
        if c.is_negative:
            result = expand_mul(-result)
        return cls(result, x, field=True) if polys else result.collect(x)

    if not domain.is_QQ:
        raise NotImplementedError("groebner method only works for QQ")

    result = _minpoly_groebner(ex, x, cls)
    return cls(result, x, field=True) if polys else result.collect(x)


def _minpoly_groebner(ex, x, cls):
    """
    Computes the minimal polynomial of an algebraic number
    using Groebner bases

    Examples
    ========

    >>> from sympy import minimal_polynomial, sqrt, Rational
    >>> from sympy.abc import x
    >>> minimal_polynomial(sqrt(2) + 3*Rational(1, 3), x, compose=False)
    x**2 - 2*x - 1

    """

    generator = numbered_symbols('a', cls=Dummy)
    mapping, symbols = {}, {}

    def update_mapping(ex, exp, base=None):
        a = next(generator)
        symbols[ex] = a

        if base is not None:
            mapping[ex] = a**exp + base
        else:
            mapping[ex] = exp.as_expr(a)

        return a

    def bottom_up_scan(ex):
        """
        Transform a given algebraic expression *ex* into a multivariate
        polynomial, by introducing fresh variables with defining equations.

        Explanation
        ===========

        The critical elements of the algebraic expression *ex* are root
        extractions, instances of :py:class:`~.AlgebraicNumber`, and negative
        powers.

        When we encounter a root extraction or an :py:class:`~.AlgebraicNumber`
        we replace this expression with a fresh variable ``a_i``, and record
        the defining polynomial for ``a_i``. For example, if ``a_0**(1/3)``
        occurs, we will replace it with ``a_1``, and record the new defining
        polynomial ``a_1**3 - a_0``.

        When we encounter a negative power we transform it into a positive
        power by algebraically inverting the base. This means computing the
        minimal polynomial in ``x`` for the base, inverting ``x`` modulo this
        poly (which generates a new polynomial) and then substituting the
        original base expression for ``x`` in this last polynomial.

        We return the transformed expression, and we record the defining
        equations for new symbols using the ``update_mapping()`` function.

        """
        if ex.is_Atom:
            if ex is S.ImaginaryUnit:
                if ex not in mapping:
                    return update_mapping(ex, 2, 1)
                else:
                    return symbols[ex]
            elif ex.is_Rational:
                return ex
        elif ex.is_Add:
            return Add(*[ bottom_up_scan(g) for g in ex.args ])
        elif ex.is_Mul:
            return Mul(*[ bottom_up_scan(g) for g in ex.args ])
        elif ex.is_Pow:
            if ex.exp.is_Rational:
                if ex.exp < 0:
                    minpoly_base = _minpoly_groebner(ex.base, x, cls)
                    inverse = invert(x, minpoly_base).as_expr()
                    base_inv = inverse.subs(x, ex.base).expand()

                    if ex.exp == -1:
                        return bottom_up_scan(base_inv)
                    else:
                        ex = base_inv**(-ex.exp)
                if not ex.exp.is_Integer:
                    base, exp = (
                        ex.base**ex.exp.p).expand(), Rational(1, ex.exp.q)
                else:
                    base, exp = ex.base, ex.exp
                base = bottom_up_scan(base)
                expr = base**exp

                if expr not in mapping:
                    if exp.is_Integer:
                        return expr.expand()
                    else:
                        return update_mapping(expr, 1 / exp, -base)
                else:
                    return symbols[expr]
        elif ex.is_AlgebraicNumber:
            if ex not in mapping:
                return update_mapping(ex, ex.minpoly_of_element())
            else:
                return symbols[ex]

        raise NotAlgebraic("%s does not seem to be an algebraic number" % ex)

    def simpler_inverse(ex):
        """
        Returns True if it is more likely that the minimal polynomial
        algorithm works better with the inverse
        """
        if ex.is_Pow:
            if (1/ex.exp).is_integer and ex.exp < 0:
                if ex.base.is_Add:
                    return True
        if ex.is_Mul:
            hit = True
            for p in ex.args:
                if p.is_Add:
                    return False
                if p.is_Pow:
                    if p.base.is_Add and p.exp > 0:
                        return False

            if hit:
                return True
        return False

    inverted = False
    ex = expand_multinomial(ex)
    if ex.is_AlgebraicNumber:
        return ex.minpoly_of_element().as_expr(x)
    elif ex.is_Rational:
        result = ex.q*x - ex.p
    else:
        inverted = simpler_inverse(ex)
        if inverted:
            ex = ex**-1
        res = None
        if ex.is_Pow and (1/ex.exp).is_Integer:
            n = 1/ex.exp
            res = _minimal_polynomial_sq(ex.base, n, x)

        elif _is_sum_surds(ex):
            res = _minimal_polynomial_sq(ex, S.One, x)

        if res is not None:
            result = res

        if res is None:
            bus = bottom_up_scan(ex)
            F = [x - bus] + list(mapping.values())
            G = groebner(F, list(symbols.values()) + [x], order='lex')

            _, factors = factor_list(G[-1])
            # by construction G[-1] has root `ex`
            result = _choose_factor(factors, x, ex)
    if inverted:
        result = _invertx(result, x)
        if result.coeff(x**degree(result, x)) < 0:
            result = expand_mul(-result)

    return result


@public
def minpoly(ex, x=None, compose=True, polys=False, domain=None):
    """This is a synonym for :py:func:`~.minimal_polynomial`."""
    return minimal_polynomial(ex, x=x, compose=compose, polys=polys, domain=domain)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\response.py ===
from __future__ import absolute_import

import io
import logging
import sys
import warnings
import zlib
from contextlib import contextmanager
from socket import error as SocketError
from socket import timeout as SocketTimeout

brotli = None

from . import util
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPException
from .exceptions import (
    BodyNotHttplibCompatible,
    DecodeError,
    HTTPError,
    IncompleteRead,
    InvalidChunkLength,
    InvalidHeader,
    ProtocolError,
    ReadTimeoutError,
    ResponseNotChunked,
    SSLError,
)
from .packages import six
from .util.response import is_fp_closed, is_response_to_head

log = logging.getLogger(__name__)


class DeflateDecoder(object):
    def __init__(self):
        self._first_try = True
        self._data = b""
        self._obj = zlib.decompressobj()

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
        if not data:
            return data

        if not self._first_try:
            return self._obj.decompress(data)

        self._data += data
        try:
            decompressed = self._obj.decompress(data)
            if decompressed:
                self._first_try = False
                self._data = None
            return decompressed
        except zlib.error:
            self._first_try = False
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                return self.decompress(self._data)
            finally:
                self._data = None


class GzipDecoderState(object):

    FIRST_MEMBER = 0
    OTHER_MEMBERS = 1
    SWALLOW_DATA = 2


class GzipDecoder(object):
    def __init__(self):
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._state = GzipDecoderState.FIRST_MEMBER

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def decompress(self, data):
        ret = bytearray()
        if self._state == GzipDecoderState.SWALLOW_DATA or not data:
            return bytes(ret)
        while True:
            try:
                ret += self._obj.decompress(data)
            except zlib.error:
                previous_state = self._state
                # Ignore data after the first error
                self._state = GzipDecoderState.SWALLOW_DATA
                if previous_state == GzipDecoderState.OTHER_MEMBERS:
                    # Allow trailing garbage acceptable in other gzip clients
                    return bytes(ret)
                raise
            data = self._obj.unused_data
            if not data:
                return bytes(ret)
            self._state = GzipDecoderState.OTHER_MEMBERS
            self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)


if brotli is not None:

    class BrotliDecoder(object):
        # Supports both 'brotlipy' and 'Brotli' packages
        # since they share an import name. The top branches
        # are for 'brotlipy' and bottom branches for 'Brotli'
        def __init__(self):
            self._obj = brotli.Decompressor()
            if hasattr(self._obj, "decompress"):
                self.decompress = self._obj.decompress
            else:
                self.decompress = self._obj.process

        def flush(self):
            if hasattr(self._obj, "flush"):
                return self._obj.flush()
            return b""


class MultiDecoder(object):
    """
    From RFC7231:
        If one or more encodings have been applied to a representation, the
        sender that applied the encodings MUST generate a Content-Encoding
        header field that lists the content codings in the order in which
        they were applied.
    """

    def __init__(self, modes):
        self._decoders = [_get_decoder(m.strip()) for m in modes.split(",")]

    def flush(self):
        return self._decoders[0].flush()

    def decompress(self, data):
        for d in reversed(self._decoders):
            data = d.decompress(data)
        return data


def _get_decoder(mode):
    if "," in mode:
        return MultiDecoder(mode)

    if mode == "gzip":
        return GzipDecoder()

    if brotli is not None and mode == "br":
        return BrotliDecoder()

    return DeflateDecoder()


class HTTPResponse(io.IOBase):
    """
    HTTP Response container.

    Backwards-compatible with :class:`http.client.HTTPResponse` but the response ``body`` is
    loaded and decoded on-demand when the ``data`` property is accessed.  This
    class is also compatible with the Python standard library's :mod:`io`
    module, and can hence be treated as a readable object in the context of that
    framework.

    Extra parameters for behaviour not present in :class:`http.client.HTTPResponse`:

    :param preload_content:
        If True, the response's body will be preloaded during construction.

    :param decode_content:
        If True, will attempt to decode the body based on the
        'content-encoding' header.

    :param original_response:
        When this HTTPResponse wrapper is generated from an :class:`http.client.HTTPResponse`
        object, it's convenient to include the original for debug purposes. It's
        otherwise unused.

    :param retries:
        The retries contains the last :class:`~urllib3.util.retry.Retry` that
        was used during the request.

    :param enforce_content_length:
        Enforce content length checking. Body returned by server must match
        value of Content-Length header, if present. Otherwise, raise error.
    """

    CONTENT_DECODERS = ["gzip", "deflate"]
    if brotli is not None:
        CONTENT_DECODERS += ["br"]
    REDIRECT_STATUSES = [301, 302, 303, 307, 308]

    def __init__(
        self,
        body="",
        headers=None,
        status=0,
        version=0,
        reason=None,
        strict=0,
        preload_content=True,
        decode_content=True,
        original_response=None,
        pool=None,
        connection=None,
        msg=None,
        retries=None,
        enforce_content_length=False,
        request_method=None,
        request_url=None,
        auto_close=True,
    ):

        if isinstance(headers, HTTPHeaderDict):
            self.headers = headers
        else:
            self.headers = HTTPHeaderDict(headers)
        self.status = status
        self.version = version
        self.reason = reason
        self.strict = strict
        self.decode_content = decode_content
        self.retries = retries
        self.enforce_content_length = enforce_content_length
        self.auto_close = auto_close

        self._decoder = None
        self._body = None
        self._fp = None
        self._original_response = original_response
        self._fp_bytes_read = 0
        self.msg = msg
        self._request_url = request_url

        if body and isinstance(body, (six.string_types, bytes)):
            self._body = body

        self._pool = pool
        self._connection = connection

        if hasattr(body, "read"):
            self._fp = body

        # Are we using the chunked-style of transfer encoding?
        self.chunked = False
        self.chunk_left = None
        tr_enc = self.headers.get("transfer-encoding", "").lower()
        # Don't incur the penalty of creating a list and then discarding it
        encodings = (enc.strip() for enc in tr_enc.split(","))
        if "chunked" in encodings:
            self.chunked = True

        # Determine length of response
        self.length_remaining = self._init_length(request_method)

        # If requested, preload the body.
        if preload_content and not self._body:
            self._body = self.read(decode_content=decode_content)

    def get_redirect_location(self):
        """
        Should we redirect and where to?

        :returns: Truthy redirect location string if we got a redirect status
            code and valid location. ``None`` if redirect status and no
            location. ``False`` if not a redirect status code.
        """
        if self.status in self.REDIRECT_STATUSES:
            return self.headers.get("location")

        return False

    def release_conn(self):
        if not self._pool or not self._connection:
            return

        self._pool._put_conn(self._connection)
        self._connection = None

    def drain_conn(self):
        """
        Read and discard any remaining HTTP response data in the response connection.

        Unread data in the HTTPResponse connection blocks the connection from being released back to the pool.
        """
        try:
            self.read()
        except (HTTPError, SocketError, BaseSSLError, HTTPException):
            pass

    @property
    def data(self):
        # For backwards-compat with earlier urllib3 0.4 and earlier.
        if self._body:
            return self._body

        if self._fp:
            return self.read(cache_content=True)

    @property
    def connection(self):
        return self._connection

    def isclosed(self):
        return is_fp_closed(self._fp)

    def tell(self):
        """
        Obtain the number of bytes pulled over the wire so far. May differ from
        the amount of content returned by :meth:``urllib3.response.HTTPResponse.read``
        if bytes are encoded on the wire (e.g, compressed).
        """
        return self._fp_bytes_read

    def _init_length(self, request_method):
        """
        Set initial length value for Response content if available.
        """
        length = self.headers.get("content-length")

        if length is not None:
            if self.chunked:
                # This Response will fail with an IncompleteRead if it can't be
                # received as chunked. This method falls back to attempt reading
                # the response before raising an exception.
                log.warning(
                    "Received response with both Content-Length and "
                    "Transfer-Encoding set. This is expressly forbidden "
                    "by RFC 7230 sec 3.3.2. Ignoring Content-Length and "
                    "attempting to process response as Transfer-Encoding: "
                    "chunked."
                )
                return None

            try:
                # RFC 7230 section 3.3.2 specifies multiple content lengths can
                # be sent in a single Content-Length header
                # (e.g. Content-Length: 42, 42). This line ensures the values
                # are all valid ints and that as long as the `set` length is 1,
                # all values are the same. Otherwise, the header is invalid.
                lengths = set([int(val) for val in length.split(",")])
                if len(lengths) > 1:
                    raise InvalidHeader(
                        "Content-Length contained multiple "
                        "unmatching values (%s)" % length
                    )
                length = lengths.pop()
            except ValueError:
                length = None
            else:
                if length < 0:
                    length = None

        # Convert status to int for comparison
        # In some cases, httplib returns a status of "_UNKNOWN"
        try:
            status = int(self.status)
        except ValueError:
            status = 0

        # Check for responses that shouldn't include a body
        if status in (204, 304) or 100 <= status < 200 or request_method == "HEAD":
            length = 0

        return length

    def _init_decoder(self):
        """
        Set-up the _decoder attribute if necessary.
        """
        # Note: content-encoding value should be case-insensitive, per RFC 7230
        # Section 3.2
        content_encoding = self.headers.get("content-encoding", "").lower()
        if self._decoder is None:
            if content_encoding in self.CONTENT_DECODERS:
                self._decoder = _get_decoder(content_encoding)
            elif "," in content_encoding:
                encodings = [
                    e.strip()
                    for e in content_encoding.split(",")
                    if e.strip() in self.CONTENT_DECODERS
                ]
                if len(encodings):
                    self._decoder = _get_decoder(content_encoding)

    DECODER_ERROR_CLASSES = (IOError, zlib.error)
    if brotli is not None:
        DECODER_ERROR_CLASSES += (brotli.error,)

    def _decode(self, data, decode_content, flush_decoder):
        """
        Decode the data passed in and potentially flush the decoder.
        """
        if not decode_content:
            return data

        try:
            if self._decoder:
                data = self._decoder.decompress(data)
        except self.DECODER_ERROR_CLASSES as e:
            content_encoding = self.headers.get("content-encoding", "").lower()
            raise DecodeError(
                "Received response with content-encoding: %s, but "
                "failed to decode it." % content_encoding,
                e,
            )
        if flush_decoder:
            data += self._flush_decoder()

        return data

    def _flush_decoder(self):
        """
        Flushes the decoder. Should only be called if the decoder is actually
        being used.
        """
        if self._decoder:
            buf = self._decoder.decompress(b"")
            return buf + self._decoder.flush()

        return b""

    @contextmanager
    def _error_catcher(self):
        """
        Catch low-level python exceptions, instead re-raising urllib3
        variants, so that low-level exceptions are not leaked in the
        high-level api.

        On exit, release the connection back to the pool.
        """
        clean_exit = False

        try:
            try:
                yield

            except SocketTimeout:
                # FIXME: Ideally we'd like to include the url in the ReadTimeoutError but
                # there is yet no clean way to get at it from this context.
                raise ReadTimeoutError(self._pool, None, "Read timed out.")

            except BaseSSLError as e:
                # FIXME: Is there a better way to differentiate between SSLErrors?
                if "read operation timed out" not in str(e):
                    # SSL errors related to framing/MAC get wrapped and reraised here
                    raise SSLError(e)

                raise ReadTimeoutError(self._pool, None, "Read timed out.")

            except (HTTPException, SocketError) as e:
                # This includes IncompleteRead.
                raise ProtocolError("Connection broken: %r" % e, e)

            # If no exception is thrown, we should avoid cleaning up
            # unnecessarily.
            clean_exit = True
        finally:
            # If we didn't terminate cleanly, we need to throw away our
            # connection.
            if not clean_exit:
                # The response may not be closed but we're not going to use it
                # anymore so close it now to ensure that the connection is
                # released back to the pool.
                if self._original_response:
                    self._original_response.close()

                # Closing the response may not actually be sufficient to close
                # everything, so if we have a hold of the connection close that
                # too.
                if self._connection:
                    self._connection.close()

            # If we hold the original response but it's closed now, we should
            # return the connection back to the pool.
            if self._original_response and self._original_response.isclosed():
                self.release_conn()

    def _fp_read(self, amt):
        """
        Read a response with the thought that reading the number of bytes
        larger than can fit in a 32-bit int at a time via SSL in some
        known cases leads to an overflow error that has to be prevented
        if `amt` or `self.length_remaining` indicate that a problem may
        happen.

        The known cases:
          * 3.8 <= CPython < 3.9.7 because of a bug
            https://github.com/urllib3/urllib3/issues/2513#issuecomment-1152559900.
          * urllib3 injected with pyOpenSSL-backed SSL-support.
          * CPython < 3.10 only when `amt` does not fit 32-bit int.
        """
        assert self._fp
        c_int_max = 2 ** 31 - 1
        if (
            (
                (amt and amt > c_int_max)
                or (self.length_remaining and self.length_remaining > c_int_max)
            )
            and not util.IS_SECURETRANSPORT
            and (util.IS_PYOPENSSL or sys.version_info < (3, 10))
        ):
            buffer = io.BytesIO()
            # Besides `max_chunk_amt` being a maximum chunk size, it
            # affects memory overhead of reading a response by this
            # method in CPython.
            # `c_int_max` equal to 2 GiB - 1 byte is the actual maximum
            # chunk size that does not lead to an overflow error, but
            # 256 MiB is a compromise.
            max_chunk_amt = 2 ** 28
            while amt is None or amt != 0:
                if amt is not None:
                    chunk_amt = min(amt, max_chunk_amt)
                    amt -= chunk_amt
                else:
                    chunk_amt = max_chunk_amt
                data = self._fp.read(chunk_amt)
                if not data:
                    break
                buffer.write(data)
                del data  # to reduce peak memory usage by `max_chunk_amt`.
            return buffer.getvalue()
        else:
            # StringIO doesn't like amt=None
            return self._fp.read(amt) if amt is not None else self._fp.read()

    def read(self, amt=None, decode_content=None, cache_content=False):
        """
        Similar to :meth:`http.client.HTTPResponse.read`, but with two additional
        parameters: ``decode_content`` and ``cache_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.

        :param cache_content:
            If True, will save the returned data such that the same result is
            returned despite of the state of the underlying file object. This
            is useful if you want the ``.data`` property to continue working
            after having ``.read()`` the file object. (Overridden if ``amt`` is
            set.)
        """
        self._init_decoder()
        if decode_content is None:
            decode_content = self.decode_content

        if self._fp is None:
            return

        flush_decoder = False
        fp_closed = getattr(self._fp, "closed", False)

        with self._error_catcher():
            data = self._fp_read(amt) if not fp_closed else b""
            if amt is None:
                flush_decoder = True
            else:
                cache_content = False
                if (
                    amt != 0 and not data
                ):  # Platform-specific: Buggy versions of Python.
                    # Close the connection when no data is returned
                    #
                    # This is redundant to what httplib/http.client _should_
                    # already do.  However, versions of python released before
                    # December 15, 2012 (http://bugs.python.org/issue16298) do
                    # not properly close the connection in all cases. There is
                    # no harm in redundantly calling close.
                    self._fp.close()
                    flush_decoder = True
                    if self.enforce_content_length and self.length_remaining not in (
                        0,
                        None,
                    ):
                        # This is an edge case that httplib failed to cover due
                        # to concerns of backward compatibility. We're
                        # addressing it here to make sure IncompleteRead is
                        # raised during streaming, so all calls with incorrect
                        # Content-Length are caught.
                        raise IncompleteRead(self._fp_bytes_read, self.length_remaining)

        if data:
            self._fp_bytes_read += len(data)
            if self.length_remaining is not None:
                self.length_remaining -= len(data)

            data = self._decode(data, decode_content, flush_decoder)

            if cache_content:
                self._body = data

        return data

    def stream(self, amt=2 ** 16, decode_content=None):
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
        if self.chunked and self.supports_chunked_reads():
            for line in self.read_chunked(amt, decode_content=decode_content):
                yield line
        else:
            while not is_fp_closed(self._fp):
                data = self.read(amt=amt, decode_content=decode_content)

                if data:
                    yield data

    @classmethod
    def from_httplib(ResponseCls, r, **response_kw):
        """
        Given an :class:`http.client.HTTPResponse` instance ``r``, return a
        corresponding :class:`urllib3.response.HTTPResponse` object.

        Remaining parameters are passed to the HTTPResponse constructor, along
        with ``original_response=r``.
        """
        headers = r.msg

        if not isinstance(headers, HTTPHeaderDict):
            if six.PY2:
                # Python 2.7
                headers = HTTPHeaderDict.from_httplib(headers)
            else:
                headers = HTTPHeaderDict(headers.items())

        # HTTPResponse objects in Python 3 don't have a .strict attribute
        strict = getattr(r, "strict", 0)
        resp = ResponseCls(
            body=r,
            headers=headers,
            status=r.status,
            version=r.version,
            reason=r.reason,
            strict=strict,
            original_response=r,
            **response_kw
        )
        return resp

    # Backwards-compatibility methods for http.client.HTTPResponse
    def getheaders(self):
        warnings.warn(
            "HTTPResponse.getheaders() is deprecated and will be removed "
            "in urllib3 v2.1.0. Instead access HTTPResponse.headers directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.headers

    def getheader(self, name, default=None):
        warnings.warn(
            "HTTPResponse.getheader() is deprecated and will be removed "
            "in urllib3 v2.1.0. Instead use HTTPResponse.headers.get(name, default).",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.headers.get(name, default)

    # Backwards compatibility for http.cookiejar
    def info(self):
        return self.headers

    # Overrides from io.IOBase
    def close(self):
        if not self.closed:
            self._fp.close()

        if self._connection:
            self._connection.close()

        if not self.auto_close:
            io.IOBase.close(self)

    @property
    def closed(self):
        if not self.auto_close:
            return io.IOBase.closed.__get__(self)
        elif self._fp is None:
            return True
        elif hasattr(self._fp, "isclosed"):
            return self._fp.isclosed()
        elif hasattr(self._fp, "closed"):
            return self._fp.closed
        else:
            return True

    def fileno(self):
        if self._fp is None:
            raise IOError("HTTPResponse has no file to get a fileno from")
        elif hasattr(self._fp, "fileno"):
            return self._fp.fileno()
        else:
            raise IOError(
                "The file-like object this HTTPResponse is wrapped "
                "around has no file descriptor"
            )

    def flush(self):
        if (
            self._fp is not None
            and hasattr(self._fp, "flush")
            and not getattr(self._fp, "closed", False)
        ):
            return self._fp.flush()

    def readable(self):
        # This method is required for `io` module compatibility.
        return True

    def readinto(self, b):
        # This method is required for `io` module compatibility.
        temp = self.read(len(b))
        if len(temp) == 0:
            return 0
        else:
            b[: len(temp)] = temp
            return len(temp)

    def supports_chunked_reads(self):
        """
        Checks if the underlying file-like object looks like a
        :class:`http.client.HTTPResponse` object. We do this by testing for
        the fp attribute. If it is present we assume it returns raw chunks as
        processed by read_chunked().
        """
        return hasattr(self._fp, "fp")

    def _update_chunk_length(self):
        # First, we'll figure out length of a chunk and then
        # we'll try to read it from socket.
        if self.chunk_left is not None:
            return
        line = self._fp.fp.readline()
        line = line.split(b";", 1)[0]
        try:
            self.chunk_left = int(line, 16)
        except ValueError:
            # Invalid chunked protocol response, abort.
            self.close()
            raise InvalidChunkLength(self, line)

    def _handle_chunk(self, amt):
        returned_chunk = None
        if amt is None:
            chunk = self._fp._safe_read(self.chunk_left)
            returned_chunk = chunk
            self._fp._safe_read(2)  # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
        elif amt < self.chunk_left:
            value = self._fp._safe_read(amt)
            self.chunk_left = self.chunk_left - amt
            returned_chunk = value
        elif amt == self.chunk_left:
            value = self._fp._safe_read(amt)
            self._fp._safe_read(2)  # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
            returned_chunk = value
        else:  # amt > self.chunk_left
            returned_chunk = self._fp._safe_read(self.chunk_left)
            self._fp._safe_read(2)  # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
        return returned_chunk

    def read_chunked(self, amt=None, decode_content=None):
        """
        Similar to :meth:`HTTPResponse.read`, but with an additional
        parameter: ``decode_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        self._init_decoder()
        # FIXME: Rewrite this method and make it a class with a better structured logic.
        if not self.chunked:
            raise ResponseNotChunked(
                "Response is not chunked. "
                "Header 'transfer-encoding: chunked' is missing."
            )
        if not self.supports_chunked_reads():
            raise BodyNotHttplibCompatible(
                "Body should be http.client.HTTPResponse like. "
                "It should have have an fp attribute which returns raw chunks."
            )

        with self._error_catcher():
            # Don't bother reading the body of a HEAD request.
            if self._original_response and is_response_to_head(self._original_response):
                self._original_response.close()
                return

            # If a response is already read and closed
            # then return immediately.
            if self._fp.fp is None:
                return

            while True:
                self._update_chunk_length()
                if self.chunk_left == 0:
                    break
                chunk = self._handle_chunk(amt)
                decoded = self._decode(
                    chunk, decode_content=decode_content, flush_decoder=False
                )
                if decoded:
                    yield decoded

            if decode_content:
                # On CPython and PyPy, we should never need to flush the
                # decoder. However, on Jython we *might* need to, so
                # lets defensively do it anyway.
                decoded = self._flush_decoder()
                if decoded:  # Platform-specific: Jython.
                    yield decoded

            # Chunk content ends with \r\n: discard it.
            while True:
                line = self._fp.fp.readline()
                if not line:
                    # Some sites may not end with '\r\n'.
                    break
                if line == b"\r\n":
                    break

            # We read everything; close the "file".
            if self._original_response:
                self._original_response.close()

    def geturl(self):
        """
        Returns the URL that was the source of this response.
        If the request that generated this response redirected, this method
        will return the final redirect location.
        """
        if self.retries is not None and len(self.retries.history):
            return self.retries.history[-1].redirect_location
        else:
            return self._request_url

    def __iter__(self):
        buffer = []
        for chunk in self.stream(decode_content=True):
            if b"\n" in chunk:
                chunk = chunk.split(b"\n")
                yield b"".join(buffer) + chunk[0] + b"\n"
                for x in chunk[1:-1]:
                    yield x + b"\n"
                if chunk[-1]:
                    buffer = [chunk[-1]]
                else:
                    buffer = []
            else:
                buffer.append(chunk)
        if buffer:
            yield b"".join(buffer)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\create.py ===
# engine/create.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import inspect
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Type
from typing import Union

from . import base
from . import url as _url
from .interfaces import DBAPIConnection
from .mock import create_mock_engine
from .. import event
from .. import exc
from .. import util
from ..pool import _AdhocProxiedConnection
from ..pool import ConnectionPoolEntry
from ..sql import compiler
from ..util import immutabledict

if typing.TYPE_CHECKING:
    from .base import Engine
    from .interfaces import _ExecuteOptions
    from .interfaces import _ParamStyle
    from .interfaces import IsolationLevel
    from .url import URL
    from ..log import _EchoFlagType
    from ..pool import _CreatorFnType
    from ..pool import _CreatorWRecFnType
    from ..pool import _ResetStyleArgType
    from ..pool import Pool
    from ..util.typing import Literal


@overload
def create_engine(
    url: Union[str, URL],
    *,
    connect_args: Dict[Any, Any] = ...,
    convert_unicode: bool = ...,
    creator: Union[_CreatorFnType, _CreatorWRecFnType] = ...,
    echo: _EchoFlagType = ...,
    echo_pool: _EchoFlagType = ...,
    enable_from_linting: bool = ...,
    execution_options: _ExecuteOptions = ...,
    future: Literal[True],
    hide_parameters: bool = ...,
    implicit_returning: Literal[True] = ...,
    insertmanyvalues_page_size: int = ...,
    isolation_level: IsolationLevel = ...,
    json_deserializer: Callable[..., Any] = ...,
    json_serializer: Callable[..., Any] = ...,
    label_length: Optional[int] = ...,
    logging_name: str = ...,
    max_identifier_length: Optional[int] = ...,
    max_overflow: int = ...,
    module: Optional[Any] = ...,
    paramstyle: Optional[_ParamStyle] = ...,
    pool: Optional[Pool] = ...,
    poolclass: Optional[Type[Pool]] = ...,
    pool_logging_name: str = ...,
    pool_pre_ping: bool = ...,
    pool_size: int = ...,
    pool_recycle: int = ...,
    pool_reset_on_return: Optional[_ResetStyleArgType] = ...,
    pool_timeout: float = ...,
    pool_use_lifo: bool = ...,
    plugins: List[str] = ...,
    query_cache_size: int = ...,
    use_insertmanyvalues: bool = ...,
    **kwargs: Any,
) -> Engine: ...


@overload
def create_engine(url: Union[str, URL], **kwargs: Any) -> Engine: ...


@util.deprecated_params(
    strategy=(
        "1.4",
        "The :paramref:`_sa.create_engine.strategy` keyword is deprecated, "
        "and the only argument accepted is 'mock'; please use "
        ":func:`.create_mock_engine` going forward.  For general "
        "customization of create_engine which may have been accomplished "
        "using strategies, see :class:`.CreateEnginePlugin`.",
    ),
    empty_in_strategy=(
        "1.4",
        "The :paramref:`_sa.create_engine.empty_in_strategy` keyword is "
        "deprecated, and no longer has any effect.  All IN expressions "
        "are now rendered using "
        'the "expanding parameter" strategy which renders a set of bound'
        'expressions, or an "empty set" SELECT, at statement execution'
        "time.",
    ),
    implicit_returning=(
        "2.0",
        "The :paramref:`_sa.create_engine.implicit_returning` parameter "
        "is deprecated and will be removed in a future release. ",
    ),
)
def create_engine(url: Union[str, _url.URL], **kwargs: Any) -> Engine:
    """Create a new :class:`_engine.Engine` instance.

    The standard calling form is to send the :ref:`URL <database_urls>` as the
    first positional argument, usually a string
    that indicates database dialect and connection arguments::

        engine = create_engine("postgresql+psycopg2://scott:tiger@localhost/test")

    .. note::

        Please review :ref:`database_urls` for general guidelines in composing
        URL strings.  In particular, special characters, such as those often
        part of passwords, must be URL encoded to be properly parsed.

    Additional keyword arguments may then follow it which
    establish various options on the resulting :class:`_engine.Engine`
    and its underlying :class:`.Dialect` and :class:`_pool.Pool`
    constructs::

        engine = create_engine(
            "mysql+mysqldb://scott:tiger@hostname/dbname",
            pool_recycle=3600,
            echo=True,
        )

    The string form of the URL is
    ``dialect[+driver]://user:password@host/dbname[?key=value..]``, where
    ``dialect`` is a database name such as ``mysql``, ``oracle``,
    ``postgresql``, etc., and ``driver`` the name of a DBAPI, such as
    ``psycopg2``, ``pyodbc``, ``cx_oracle``, etc.  Alternatively,
    the URL can be an instance of :class:`~sqlalchemy.engine.url.URL`.

    ``**kwargs`` takes a wide variety of options which are routed
    towards their appropriate components.  Arguments may be specific to
    the :class:`_engine.Engine`, the underlying :class:`.Dialect`,
    as well as the
    :class:`_pool.Pool`.  Specific dialects also accept keyword arguments that
    are unique to that dialect.   Here, we describe the parameters
    that are common to most :func:`_sa.create_engine()` usage.

    Once established, the newly resulting :class:`_engine.Engine` will
    request a connection from the underlying :class:`_pool.Pool` once
    :meth:`_engine.Engine.connect` is called, or a method which depends on it
    such as :meth:`_engine.Engine.execute` is invoked.   The
    :class:`_pool.Pool` in turn
    will establish the first actual DBAPI connection when this request
    is received.   The :func:`_sa.create_engine` call itself does **not**
    establish any actual DBAPI connections directly.

    .. seealso::

        :doc:`/core/engines`

        :doc:`/dialects/index`

        :ref:`connections_toplevel`

    :param connect_args: a dictionary of options which will be
        passed directly to the DBAPI's ``connect()`` method as
        additional keyword arguments.  See the example
        at :ref:`custom_dbapi_args`.

    :param creator: a callable which returns a DBAPI connection.
        This creation function will be passed to the underlying
        connection pool and will be used to create all new database
        connections. Usage of this function causes connection
        parameters specified in the URL argument to be bypassed.

        This hook is not as flexible as the newer
        :meth:`_events.DialectEvents.do_connect` hook which allows complete
        control over how a connection is made to the database, given the full
        set of URL arguments and state beforehand.

        .. seealso::

            :meth:`_events.DialectEvents.do_connect` - event hook that allows
            full control over DBAPI connection mechanics.

            :ref:`custom_dbapi_args`

    :param echo=False: if True, the Engine will log all statements
        as well as a ``repr()`` of their parameter lists to the default log
        handler, which defaults to ``sys.stdout`` for output.   If set to the
        string ``"debug"``, result rows will be printed to the standard output
        as well. The ``echo`` attribute of ``Engine`` can be modified at any
        time to turn logging on and off; direct control of logging is also
        available using the standard Python ``logging`` module.

        .. seealso::

            :ref:`dbengine_logging` - further detail on how to configure
            logging.


    :param echo_pool=False: if True, the connection pool will log
        informational output such as when connections are invalidated
        as well as when connections are recycled to the default log handler,
        which defaults to ``sys.stdout`` for output.   If set to the string
        ``"debug"``, the logging will include pool checkouts and checkins.
        Direct control of logging is also available using the standard Python
        ``logging`` module.

        .. seealso::

            :ref:`dbengine_logging` - further detail on how to configure
            logging.


    :param empty_in_strategy:   No longer used; SQLAlchemy now uses
        "empty set" behavior for IN in all cases.

    :param enable_from_linting: defaults to True.  Will emit a warning
        if a given SELECT statement is found to have un-linked FROM elements
        which would cause a cartesian product.

        .. versionadded:: 1.4

        .. seealso::

            :ref:`change_4737`

    :param execution_options: Dictionary execution options which will
        be applied to all connections.  See
        :meth:`~sqlalchemy.engine.Connection.execution_options`

    :param future: Use the 2.0 style :class:`_engine.Engine` and
        :class:`_engine.Connection` API.

        As of SQLAlchemy 2.0, this parameter is present for backwards
        compatibility only and must remain at its default value of ``True``.

        The :paramref:`_sa.create_engine.future` parameter will be
        deprecated in a subsequent 2.x release and eventually removed.

        .. versionadded:: 1.4

        .. versionchanged:: 2.0 All :class:`_engine.Engine` objects are
           "future" style engines and there is no longer a ``future=False``
           mode of operation.

        .. seealso::

            :ref:`migration_20_toplevel`

    :param hide_parameters: Boolean, when set to True, SQL statement parameters
        will not be displayed in INFO logging nor will they be formatted into
        the string representation of :class:`.StatementError` objects.

        .. versionadded:: 1.3.8

        .. seealso::

            :ref:`dbengine_logging` - further detail on how to configure
            logging.

    :param implicit_returning=True:  Legacy parameter that may only be set
        to True. In SQLAlchemy 2.0, this parameter does nothing. In order to
        disable "implicit returning" for statements invoked by the ORM,
        configure this on a per-table basis using the
        :paramref:`.Table.implicit_returning` parameter.


    :param insertmanyvalues_page_size: number of rows to format into an
     INSERT statement when the statement uses "insertmanyvalues" mode, which is
     a paged form of bulk insert that is used for many backends when using
     :term:`executemany` execution typically in conjunction with RETURNING.
     Defaults to 1000, but may also be subject to dialect-specific limiting
     factors which may override this value on a per-statement basis.

     .. versionadded:: 2.0

     .. seealso::

        :ref:`engine_insertmanyvalues`

        :ref:`engine_insertmanyvalues_page_size`

        :paramref:`_engine.Connection.execution_options.insertmanyvalues_page_size`

    :param isolation_level: optional string name of an isolation level
        which will be set on all new connections unconditionally.
        Isolation levels are typically some subset of the string names
        ``"SERIALIZABLE"``, ``"REPEATABLE READ"``,
        ``"READ COMMITTED"``, ``"READ UNCOMMITTED"`` and ``"AUTOCOMMIT"``
        based on backend.

        The :paramref:`_sa.create_engine.isolation_level` parameter is
        in contrast to the
        :paramref:`.Connection.execution_options.isolation_level`
        execution option, which may be set on an individual
        :class:`.Connection`, as well as the same parameter passed to
        :meth:`.Engine.execution_options`, where it may be used to create
        multiple engines with different isolation levels that share a common
        connection pool and dialect.

        .. versionchanged:: 2.0 The
           :paramref:`_sa.create_engine.isolation_level`
           parameter has been generalized to work on all dialects which support
           the concept of isolation level, and is provided as a more succinct,
           up front configuration switch in contrast to the execution option
           which is more of an ad-hoc programmatic option.

        .. seealso::

            :ref:`dbapi_autocommit`

    :param json_deserializer: for dialects that support the
        :class:`_types.JSON`
        datatype, this is a Python callable that will convert a JSON string
        to a Python object.  By default, the Python ``json.loads`` function is
        used.

        .. versionchanged:: 1.3.7  The SQLite dialect renamed this from
           ``_json_deserializer``.

    :param json_serializer: for dialects that support the :class:`_types.JSON`
        datatype, this is a Python callable that will render a given object
        as JSON.   By default, the Python ``json.dumps`` function is used.

        .. versionchanged:: 1.3.7  The SQLite dialect renamed this from
           ``_json_serializer``.


    :param label_length=None: optional integer value which limits
        the size of dynamically generated column labels to that many
        characters. If less than 6, labels are generated as
        "_(counter)". If ``None``, the value of
        ``dialect.max_identifier_length``, which may be affected via the
        :paramref:`_sa.create_engine.max_identifier_length` parameter,
        is used instead.   The value of
        :paramref:`_sa.create_engine.label_length`
        may not be larger than that of
        :paramref:`_sa.create_engine.max_identfier_length`.

        .. seealso::

            :paramref:`_sa.create_engine.max_identifier_length`

    :param logging_name:  String identifier which will be used within
        the "name" field of logging records generated within the
        "sqlalchemy.engine" logger. Defaults to a hexstring of the
        object's id.

        .. seealso::

            :ref:`dbengine_logging` - further detail on how to configure
            logging.

            :paramref:`_engine.Connection.execution_options.logging_token`

    :param max_identifier_length: integer; override the max_identifier_length
        determined by the dialect.  if ``None`` or zero, has no effect.  This
        is the database's configured maximum number of characters that may be
        used in a SQL identifier such as a table name, column name, or label
        name. All dialects determine this value automatically, however in the
        case of a new database version for which this value has changed but
        SQLAlchemy's dialect has not been adjusted, the value may be passed
        here.

        .. versionadded:: 1.3.9

        .. seealso::

            :paramref:`_sa.create_engine.label_length`

    :param max_overflow=10: the number of connections to allow in
        connection pool "overflow", that is connections that can be
        opened above and beyond the pool_size setting, which defaults
        to five. this is only used with :class:`~sqlalchemy.pool.QueuePool`.

    :param module=None: reference to a Python module object (the module
        itself, not its string name).  Specifies an alternate DBAPI module to
        be used by the engine's dialect.  Each sub-dialect references a
        specific DBAPI which will be imported before first connect.  This
        parameter causes the import to be bypassed, and the given module to
        be used instead. Can be used for testing of DBAPIs as well as to
        inject "mock" DBAPI implementations into the :class:`_engine.Engine`.

    :param paramstyle=None: The `paramstyle <https://legacy.python.org/dev/peps/pep-0249/#paramstyle>`_
        to use when rendering bound parameters.  This style defaults to the
        one recommended by the DBAPI itself, which is retrieved from the
        ``.paramstyle`` attribute of the DBAPI.  However, most DBAPIs accept
        more than one paramstyle, and in particular it may be desirable
        to change a "named" paramstyle into a "positional" one, or vice versa.
        When this attribute is passed, it should be one of the values
        ``"qmark"``, ``"numeric"``, ``"named"``, ``"format"`` or
        ``"pyformat"``, and should correspond to a parameter style known
        to be supported by the DBAPI in use.

    :param pool=None: an already-constructed instance of
        :class:`~sqlalchemy.pool.Pool`, such as a
        :class:`~sqlalchemy.pool.QueuePool` instance. If non-None, this
        pool will be used directly as the underlying connection pool
        for the engine, bypassing whatever connection parameters are
        present in the URL argument. For information on constructing
        connection pools manually, see :ref:`pooling_toplevel`.

    :param poolclass=None: a :class:`~sqlalchemy.pool.Pool`
        subclass, which will be used to create a connection pool
        instance using the connection parameters given in the URL. Note
        this differs from ``pool`` in that you don't actually
        instantiate the pool in this case, you just indicate what type
        of pool to be used.

    :param pool_logging_name:  String identifier which will be used within
       the "name" field of logging records generated within the
       "sqlalchemy.pool" logger. Defaults to a hexstring of the object's
       id.

       .. seealso::

            :ref:`dbengine_logging` - further detail on how to configure
            logging.

    :param pool_pre_ping: boolean, if True will enable the connection pool
        "pre-ping" feature that tests connections for liveness upon
        each checkout.

        .. versionadded:: 1.2

        .. seealso::

            :ref:`pool_disconnects_pessimistic`

    :param pool_size=5: the number of connections to keep open
        inside the connection pool. This used with
        :class:`~sqlalchemy.pool.QueuePool` as
        well as :class:`~sqlalchemy.pool.SingletonThreadPool`.  With
        :class:`~sqlalchemy.pool.QueuePool`, a ``pool_size`` setting
        of 0 indicates no limit; to disable pooling, set ``poolclass`` to
        :class:`~sqlalchemy.pool.NullPool` instead.

    :param pool_recycle=-1: this setting causes the pool to recycle
        connections after the given number of seconds has passed. It
        defaults to -1, or no timeout. For example, setting to 3600
        means connections will be recycled after one hour. Note that
        MySQL in particular will disconnect automatically if no
        activity is detected on a connection for eight hours (although
        this is configurable with the MySQLDB connection itself and the
        server configuration as well).

        .. seealso::

            :ref:`pool_setting_recycle`

    :param pool_reset_on_return='rollback': set the
        :paramref:`_pool.Pool.reset_on_return` parameter of the underlying
        :class:`_pool.Pool` object, which can be set to the values
        ``"rollback"``, ``"commit"``, or ``None``.

        .. seealso::

            :ref:`pool_reset_on_return`

    :param pool_timeout=30: number of seconds to wait before giving
        up on getting a connection from the pool. This is only used
        with :class:`~sqlalchemy.pool.QueuePool`. This can be a float but is
        subject to the limitations of Python time functions which may not be
        reliable in the tens of milliseconds.

        .. note: don't use 30.0 above, it seems to break with the :param tag

    :param pool_use_lifo=False: use LIFO (last-in-first-out) when retrieving
        connections from :class:`.QueuePool` instead of FIFO
        (first-in-first-out). Using LIFO, a server-side timeout scheme can
        reduce the number of connections used during non- peak   periods of
        use.   When planning for server-side timeouts, ensure that a recycle or
        pre-ping strategy is in use to gracefully   handle stale connections.

          .. versionadded:: 1.3

          .. seealso::

            :ref:`pool_use_lifo`

            :ref:`pool_disconnects`

    :param plugins: string list of plugin names to load.  See
        :class:`.CreateEnginePlugin` for background.

        .. versionadded:: 1.2.3

    :param query_cache_size: size of the cache used to cache the SQL string
     form of queries.  Set to zero to disable caching.

     The cache is pruned of its least recently used items when its size reaches
     N * 1.5.  Defaults to 500, meaning the cache will always store at least
     500 SQL statements when filled, and will grow up to 750 items at which
     point it is pruned back down to 500 by removing the 250 least recently
     used items.

     Caching is accomplished on a per-statement basis by generating a
     cache key that represents the statement's structure, then generating
     string SQL for the current dialect only if that key is not present
     in the cache.   All statements support caching, however some features
     such as an INSERT with a large set of parameters will intentionally
     bypass the cache.   SQL logging will indicate statistics for each
     statement whether or not it were pull from the cache.

     .. note:: some ORM functions related to unit-of-work persistence as well
        as some attribute loading strategies will make use of individual
        per-mapper caches outside of the main cache.


     .. seealso::

        :ref:`sql_caching`

     .. versionadded:: 1.4

    :param use_insertmanyvalues: True by default, use the "insertmanyvalues"
     execution style for INSERT..RETURNING statements by default.

     .. versionadded:: 2.0

     .. seealso::

        :ref:`engine_insertmanyvalues`

    """  # noqa

    if "strategy" in kwargs:
        strat = kwargs.pop("strategy")
        if strat == "mock":
            # this case is deprecated
            return create_mock_engine(url, **kwargs)  # type: ignore
        else:
            raise exc.ArgumentError("unknown strategy: %r" % strat)

    kwargs.pop("empty_in_strategy", None)

    # create url.URL object
    u = _url.make_url(url)

    u, plugins, kwargs = u._instantiate_plugins(kwargs)

    entrypoint = u._get_entrypoint()
    _is_async = kwargs.pop("_is_async", False)
    if _is_async:
        dialect_cls = entrypoint.get_async_dialect_cls(u)
    else:
        dialect_cls = entrypoint.get_dialect_cls(u)

    if kwargs.pop("_coerce_config", False):

        def pop_kwarg(key: str, default: Optional[Any] = None) -> Any:
            value = kwargs.pop(key, default)
            if key in dialect_cls.engine_config_types:
                value = dialect_cls.engine_config_types[key](value)
            return value

    else:
        pop_kwarg = kwargs.pop  # type: ignore

    dialect_args = {}
    # consume dialect arguments from kwargs
    for k in util.get_cls_kwargs(dialect_cls):
        if k in kwargs:
            dialect_args[k] = pop_kwarg(k)

    dbapi = kwargs.pop("module", None)
    if dbapi is None:
        dbapi_args = {}

        if "import_dbapi" in dialect_cls.__dict__:
            dbapi_meth = dialect_cls.import_dbapi

        elif hasattr(dialect_cls, "dbapi") and inspect.ismethod(
            dialect_cls.dbapi
        ):
            util.warn_deprecated(
                "The dbapi() classmethod on dialect classes has been "
                "renamed to import_dbapi().  Implement an import_dbapi() "
                f"classmethod directly on class {dialect_cls} to remove this "
                "warning; the old .dbapi() classmethod may be maintained for "
                "backwards compatibility.",
                "2.0",
            )
            dbapi_meth = dialect_cls.dbapi
        else:
            dbapi_meth = dialect_cls.import_dbapi

        for k in util.get_func_kwargs(dbapi_meth):
            if k in kwargs:
                dbapi_args[k] = pop_kwarg(k)
        dbapi = dbapi_meth(**dbapi_args)

    dialect_args["dbapi"] = dbapi

    dialect_args.setdefault("compiler_linting", compiler.NO_LINTING)
    enable_from_linting = kwargs.pop("enable_from_linting", True)
    if enable_from_linting:
        dialect_args["compiler_linting"] ^= compiler.COLLECT_CARTESIAN_PRODUCTS

    for plugin in plugins:
        plugin.handle_dialect_kwargs(dialect_cls, dialect_args)

    # create dialect
    dialect = dialect_cls(**dialect_args)

    # assemble connection arguments
    (cargs_tup, cparams) = dialect.create_connect_args(u)
    cparams.update(pop_kwarg("connect_args", {}))

    if "async_fallback" in cparams and util.asbool(cparams["async_fallback"]):
        util.warn_deprecated(
            "The async_fallback dialect argument is deprecated and will be "
            "removed in SQLAlchemy 2.1.",
            "2.0",
        )

    cargs = list(cargs_tup)  # allow mutability

    # look for existing pool or create
    pool = pop_kwarg("pool", None)
    if pool is None:

        def connect(
            connection_record: Optional[ConnectionPoolEntry] = None,
        ) -> DBAPIConnection:
            if dialect._has_events:
                for fn in dialect.dispatch.do_connect:
                    connection = cast(
                        DBAPIConnection,
                        fn(dialect, connection_record, cargs, cparams),
                    )
                    if connection is not None:
                        return connection

            return dialect.connect(*cargs, **cparams)

        creator = pop_kwarg("creator", connect)

        poolclass = pop_kwarg("poolclass", None)
        if poolclass is None:
            poolclass = dialect.get_dialect_pool_class(u)
        pool_args = {"dialect": dialect}

        # consume pool arguments from kwargs, translating a few of
        # the arguments
        for k in util.get_cls_kwargs(poolclass):
            tk = _pool_translate_kwargs.get(k, k)
            if tk in kwargs:
                pool_args[k] = pop_kwarg(tk)

        for plugin in plugins:
            plugin.handle_pool_kwargs(poolclass, pool_args)

        pool = poolclass(creator, **pool_args)
    else:
        pool._dialect = dialect

    if (
        hasattr(pool, "_is_asyncio")
        and pool._is_asyncio is not dialect.is_async
    ):
        raise exc.ArgumentError(
            f"Pool class {pool.__class__.__name__} cannot be "
            f"used with {'non-' if not dialect.is_async else ''}"
            "asyncio engine",
            code="pcls",
        )

    # create engine.
    if not pop_kwarg("future", True):
        raise exc.ArgumentError(
            "The 'future' parameter passed to "
            "create_engine() may only be set to True."
        )

    engineclass = base.Engine

    engine_args = {}
    for k in util.get_cls_kwargs(engineclass):
        if k in kwargs:
            engine_args[k] = pop_kwarg(k)

    # internal flags used by the test suite for instrumenting / proxying
    # engines with mocks etc.
    _initialize = kwargs.pop("_initialize", True)

    # all kwargs should be consumed
    if kwargs:
        raise TypeError(
            "Invalid argument(s) %s sent to create_engine(), "
            "using configuration %s/%s/%s.  Please check that the "
            "keyword arguments are appropriate for this combination "
            "of components."
            % (
                ",".join("'%s'" % k for k in kwargs),
                dialect.__class__.__name__,
                pool.__class__.__name__,
                engineclass.__name__,
            )
        )

    engine = engineclass(pool, dialect, u, **engine_args)

    if _initialize:
        do_on_connect = dialect.on_connect_url(u)
        if do_on_connect:

            def on_connect(
                dbapi_connection: DBAPIConnection,
                connection_record: ConnectionPoolEntry,
            ) -> None:
                assert do_on_connect is not None
                do_on_connect(dbapi_connection)

            event.listen(pool, "connect", on_connect)

        builtin_on_connect = dialect._builtin_onconnect()
        if builtin_on_connect:
            event.listen(pool, "connect", builtin_on_connect)

        def first_connect(
            dbapi_connection: DBAPIConnection,
            connection_record: ConnectionPoolEntry,
        ) -> None:
            c = base.Connection(
                engine,
                connection=_AdhocProxiedConnection(
                    dbapi_connection, connection_record
                ),
                _has_events=False,
                # reconnecting will be a reentrant condition, so if the
                # connection goes away, Connection is then closed
                _allow_revalidate=False,
                # dont trigger the autobegin sequence
                # within the up front dialect checks
                _allow_autobegin=False,
            )
            c._execution_options = util.EMPTY_DICT

            try:
                dialect.initialize(c)
            finally:
                # note that "invalidated" and "closed" are mutually
                # exclusive in 1.4 Connection.
                if not c.invalidated and not c.closed:
                    # transaction is rolled back otherwise, tested by
                    # test/dialect/postgresql/test_dialect.py
                    # ::MiscBackendTest::test_initial_transaction_state
                    dialect.do_rollback(c.connection)

        # previously, the "first_connect" event was used here, which was then
        # scaled back if the "on_connect" handler were present.  now,
        # since "on_connect" is virtually always present, just use
        # "connect" event with once_unless_exception in all cases so that
        # the connection event flow is consistent in all cases.
        event.listen(
            pool, "connect", first_connect, _once_unless_exception=True
        )

    dialect_cls.engine_created(engine)
    if entrypoint is not dialect_cls:
        entrypoint.engine_created(engine)

    for plugin in plugins:
        plugin.engine_created(engine)

    return engine


def engine_from_config(
    configuration: Dict[str, Any], prefix: str = "sqlalchemy.", **kwargs: Any
) -> Engine:
    """Create a new Engine instance using a configuration dictionary.

    The dictionary is typically produced from a config file.

    The keys of interest to ``engine_from_config()`` should be prefixed, e.g.
    ``sqlalchemy.url``, ``sqlalchemy.echo``, etc.  The 'prefix' argument
    indicates the prefix to be searched for.  Each matching key (after the
    prefix is stripped) is treated as though it were the corresponding keyword
    argument to a :func:`_sa.create_engine` call.

    The only required key is (assuming the default prefix) ``sqlalchemy.url``,
    which provides the :ref:`database URL <database_urls>`.

    A select set of keyword arguments will be "coerced" to their
    expected type based on string values.    The set of arguments
    is extensible per-dialect using the ``engine_config_types`` accessor.

    :param configuration: A dictionary (typically produced from a config file,
        but this is not a requirement).  Items whose keys start with the value
        of 'prefix' will have that prefix stripped, and will then be passed to
        :func:`_sa.create_engine`.

    :param prefix: Prefix to match and then strip from keys
        in 'configuration'.

    :param kwargs: Each keyword argument to ``engine_from_config()`` itself
        overrides the corresponding item taken from the 'configuration'
        dictionary.  Keyword arguments should *not* be prefixed.

    """

    options = {
        key[len(prefix) :]: configuration[key]
        for key in configuration
        if key.startswith(prefix)
    }
    options["_coerce_config"] = True
    options.update(kwargs)
    url = options.pop("url")
    return create_engine(url, **options)


@overload
def create_pool_from_url(
    url: Union[str, URL],
    *,
    poolclass: Optional[Type[Pool]] = ...,
    logging_name: str = ...,
    pre_ping: bool = ...,
    size: int = ...,
    recycle: int = ...,
    reset_on_return: Optional[_ResetStyleArgType] = ...,
    timeout: float = ...,
    use_lifo: bool = ...,
    **kwargs: Any,
) -> Pool: ...


@overload
def create_pool_from_url(url: Union[str, URL], **kwargs: Any) -> Pool: ...


def create_pool_from_url(url: Union[str, URL], **kwargs: Any) -> Pool:
    """Create a pool instance from the given url.

    If ``poolclass`` is not provided the pool class used
    is selected using the dialect specified in the URL.

    The arguments passed to :func:`_sa.create_pool_from_url` are
    identical to the pool argument passed to the :func:`_sa.create_engine`
    function.

    .. versionadded:: 2.0.10
    """

    for key in _pool_translate_kwargs:
        if key in kwargs:
            kwargs[_pool_translate_kwargs[key]] = kwargs.pop(key)

    engine = create_engine(url, **kwargs, _initialize=False)
    return engine.pool


_pool_translate_kwargs = immutabledict(
    {
        "logging_name": "pool_logging_name",
        "echo": "echo_pool",
        "timeout": "pool_timeout",
        "recycle": "pool_recycle",
        "events": "pool_events",  # deprecated
        "reset_on_return": "pool_reset_on_return",
        "pre_ping": "pool_pre_ping",
        "use_lifo": "pool_use_lifo",
    }
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\plane.py ===
"""Geometrical Planes.

Contains
========
Plane

"""

from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from .entity import GeometryEntity
from .line import (Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D,
                   Ray3D, Segment3D)
from .point import Point, Point3D
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable

from mpmath.libmp.libmpf import prec_to_dps

import random


x, y, z, t = [Dummy('plane_dummy') for i in range(4)]


class Plane(GeometryEntity):
    """
    A plane is a flat, two-dimensional surface. A plane is the two-dimensional
    analogue of a point (zero-dimensions), a line (one-dimension) and a solid
    (three-dimensions). A plane can generally be constructed by two types of
    inputs. They are:
    - three non-collinear points
    - a point and the plane's normal vector

    Attributes
    ==========

    p1
    normal_vector

    Examples
    ========

    >>> from sympy import Plane, Point3D
    >>> Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane((1, 1, 1), (2, 3, 4), (2, 2, 2))
    Plane(Point3D(1, 1, 1), (-1, 2, -1))
    >>> Plane(Point3D(1, 1, 1), normal_vector=(1,4,7))
    Plane(Point3D(1, 1, 1), (1, 4, 7))

    """
    def __new__(cls, p1, a=None, b=None, **kwargs):
        p1 = Point3D(p1, dim=3)
        if a and b:
            p2 = Point(a, dim=3)
            p3 = Point(b, dim=3)
            if Point3D.are_collinear(p1, p2, p3):
                raise ValueError('Enter three non-collinear points')
            a = p1.direction_ratio(p2)
            b = p1.direction_ratio(p3)
            normal_vector = tuple(Matrix(a).cross(Matrix(b)))
        else:
            a = kwargs.pop('normal_vector', a)
            evaluate = kwargs.get('evaluate', True)
            if is_sequence(a) and len(a) == 3:
                normal_vector = Point3D(a).args if evaluate else a
            else:
                raise ValueError(filldedent('''
                    Either provide 3 3D points or a point with a
                    normal vector expressed as a sequence of length 3'''))
            if all(coord.is_zero for coord in normal_vector):
                raise ValueError('Normal vector cannot be zero vector')
        return GeometryEntity.__new__(cls, p1, normal_vector, **kwargs)

    def __contains__(self, o):
        k = self.equation(x, y, z)
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            d = Point3D(o.arbitrary_point(t))
            e = k.subs([(x, d.x), (y, d.y), (z, d.z)])
            return e.equals(0)
        try:
            o = Point(o, dim=3, strict=True)
            d = k.xreplace(dict(zip((x, y, z), o.args)))
            return d.equals(0)
        except TypeError:
            return False

    def _eval_evalf(self, prec=15, **options):
        pt, tup = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        tup = tuple([i.evalf(n=dps, **options) for i in tup])
        return self.func(pt, normal_vector=tup, evaluate=False)

    def angle_between(self, o):
        """Angle between the plane and other geometric entity.

        Parameters
        ==========

        LinearEntity3D, Plane.

        Returns
        =======

        angle : angle in radians

        Notes
        =====

        This method accepts only 3D entities as it's parameter, but if you want
        to calculate the angle between a 2D entity and a plane you should
        first convert to a 3D entity by projecting onto a desired plane and
        then proceed to calculate the angle.

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 2, 2), normal_vector=(1, 2, 3))
        >>> b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))
        >>> a.angle_between(b)
        -asin(sqrt(21)/6)

        """
        if isinstance(o, LinearEntity3D):
            a = Matrix(self.normal_vector)
            b = Matrix(o.direction_ratio)
            c = a.dot(b)
            d = sqrt(sum(i**2 for i in self.normal_vector))
            e = sqrt(sum(i**2 for i in o.direction_ratio))
            return asin(c/(d*e))
        if isinstance(o, Plane):
            a = Matrix(self.normal_vector)
            b = Matrix(o.normal_vector)
            c = a.dot(b)
            d = sqrt(sum(i**2 for i in self.normal_vector))
            e = sqrt(sum(i**2 for i in o.normal_vector))
            return acos(c/(d*e))


    def arbitrary_point(self, u=None, v=None):
        """ Returns an arbitrary point on the Plane. If given two
        parameters, the point ranges over the entire plane. If given 1
        or no parameters, returns a point with one parameter which,
        when varying from 0 to 2*pi, moves the point in a circle of
        radius 1 about p1 of the Plane.

        Examples
        ========

        >>> from sympy import Plane, Ray
        >>> from sympy.abc import u, v, t, r
        >>> p = Plane((1, 1, 1), normal_vector=(1, 0, 0))
        >>> p.arbitrary_point(u, v)
        Point3D(1, u + 1, v + 1)
        >>> p.arbitrary_point(t)
        Point3D(1, cos(t) + 1, sin(t) + 1)

        While arbitrary values of u and v can move the point anywhere in
        the plane, the single-parameter point can be used to construct a
        ray whose arbitrary point can be located at angle t and radius
        r from p.p1:

        >>> Ray(p.p1, _).arbitrary_point(r)
        Point3D(1, r*cos(t) + 1, r*sin(t) + 1)

        Returns
        =======

        Point3D

        """
        circle = v is None
        if circle:
            u = _symbol(u or 't', real=True)
        else:
            u = _symbol(u or 'u', real=True)
            v = _symbol(v or 'v', real=True)
        x, y, z = self.normal_vector
        a, b, c = self.p1.args
        # x1, y1, z1 is a nonzero vector parallel to the plane
        if x.is_zero and y.is_zero:
            x1, y1, z1 = S.One, S.Zero, S.Zero
        else:
            x1, y1, z1 = -y, x, S.Zero
        # x2, y2, z2 is also parallel to the plane, and orthogonal to x1, y1, z1
        x2, y2, z2 = tuple(Matrix((x, y, z)).cross(Matrix((x1, y1, z1))))
        if circle:
            x1, y1, z1 = (w/sqrt(x1**2 + y1**2 + z1**2) for w in (x1, y1, z1))
            x2, y2, z2 = (w/sqrt(x2**2 + y2**2 + z2**2) for w in (x2, y2, z2))
            p = Point3D(a + x1*cos(u) + x2*sin(u), \
                        b + y1*cos(u) + y2*sin(u), \
                        c + z1*cos(u) + z2*sin(u))
        else:
            p = Point3D(a + x1*u + x2*v, b + y1*u + y2*v, c + z1*u + z2*v)
        return p


    @staticmethod
    def are_concurrent(*planes):
        """Is a sequence of Planes concurrent?

        Two or more Planes are concurrent if their intersections
        are a common line.

        Parameters
        ==========

        planes: list

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(5, 0, 0), normal_vector=(1, -1, 1))
        >>> b = Plane(Point3D(0, -2, 0), normal_vector=(3, 1, 1))
        >>> c = Plane(Point3D(0, -1, 0), normal_vector=(5, -1, 9))
        >>> Plane.are_concurrent(a, b)
        True
        >>> Plane.are_concurrent(a, b, c)
        False

        """
        planes = list(uniq(planes))
        for i in planes:
            if not isinstance(i, Plane):
                raise ValueError('All objects should be Planes but got %s' % i.func)
        if len(planes) < 2:
            return False
        planes = list(planes)
        first = planes.pop(0)
        sol = first.intersection(planes[0])
        if sol == []:
            return False
        else:
            line = sol[0]
            for i in planes[1:]:
                l = first.intersection(i)
                if not l or l[0] not in line:
                    return False
            return True


    def distance(self, o):
        """Distance between the plane and another geometric entity.

        Parameters
        ==========

        Point3D, LinearEntity3D, Plane.

        Returns
        =======

        distance

        Notes
        =====

        This method accepts only 3D entities as it's parameter, but if you want
        to calculate the distance between a 2D entity and a plane you should
        first convert to a 3D entity by projecting onto a desired plane and
        then proceed to calculate the distance.

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.distance(b)
        sqrt(3)
        >>> c = Line3D(Point3D(2, 3, 1), Point3D(1, 2, 2))
        >>> a.distance(c)
        0

        """
        if self.intersection(o) != []:
            return S.Zero

        if isinstance(o, (Segment3D, Ray3D)):
            a, b = o.p1, o.p2
            pi, = self.intersection(Line3D(a, b))
            if pi in o:
                return self.distance(pi)
            elif a in Segment3D(pi, b):
                return self.distance(a)
            else:
                assert isinstance(o, Segment3D) is True
                return self.distance(b)

        # following code handles `Point3D`, `LinearEntity3D`, `Plane`
        a = o if isinstance(o, Point3D) else o.p1
        n = Point3D(self.normal_vector).unit
        d = (a - self.p1).dot(n)
        return abs(d)


    def equals(self, o):
        """
        Returns True if self and o are the same mathematical entities.

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
        >>> b = Plane(Point3D(1, 2, 3), normal_vector=(2, 2, 2))
        >>> c = Plane(Point3D(1, 2, 3), normal_vector=(-1, 4, 6))
        >>> a.equals(a)
        True
        >>> a.equals(b)
        True
        >>> a.equals(c)
        False
        """
        if isinstance(o, Plane):
            a = self.equation()
            b = o.equation()
            return cancel(a/b).is_constant()
        else:
            return False


    def equation(self, x=None, y=None, z=None):
        """The equation of the Plane.

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 2), Point3D(2, 4, 7), Point3D(3, 5, 1))
        >>> a.equation()
        -23*x + 11*y - 2*z + 16
        >>> a = Plane(Point3D(1, 4, 2), normal_vector=(6, 6, 6))
        >>> a.equation()
        6*x + 6*y + 6*z - 42

        """
        x, y, z = [i if i else Symbol(j, real=True) for i, j in zip((x, y, z), 'xyz')]
        a = Point3D(x, y, z)
        b = self.p1.direction_ratio(a)
        c = self.normal_vector
        return (sum(i*j for i, j in zip(b, c)))


    def intersection(self, o):
        """ The intersection with other geometrical entity.

        Parameters
        ==========

        Point, Point3D, LinearEntity, LinearEntity3D, Plane

        Returns
        =======

        List

        Examples
        ========

        >>> from sympy import Point3D, Line3D, Plane
        >>> a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
        >>> b = Point3D(1, 2, 3)
        >>> a.intersection(b)
        [Point3D(1, 2, 3)]
        >>> c = Line3D(Point3D(1, 4, 7), Point3D(2, 2, 2))
        >>> a.intersection(c)
        [Point3D(2, 2, 2)]
        >>> d = Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
        >>> e = Plane(Point3D(2, 0, 0), normal_vector=(3, 4, -3))
        >>> d.intersection(e)
        [Line3D(Point3D(78/23, -24/23, 0), Point3D(147/23, 321/23, 23))]

        """
        if not isinstance(o, GeometryEntity):
            o = Point(o, dim=3)
        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []
        if isinstance(o, (LinearEntity, LinearEntity3D)):
            # recast to 3D
            p1, p2 = o.p1, o.p2
            if isinstance(o, Segment):
                o = Segment3D(p1, p2)
            elif isinstance(o, Ray):
                o = Ray3D(p1, p2)
            elif isinstance(o, Line):
                o = Line3D(p1, p2)
            else:
                raise ValueError('unhandled linear entity: %s' % o.func)
            if o in self:
                return [o]
            else:
                a = Point3D(o.arbitrary_point(t))
                p1, n = self.p1, Point3D(self.normal_vector)

                # TODO: Replace solve with solveset, when this line is tested
                c = solve((a - p1).dot(n), t)
                if not c:
                    return []
                else:
                    c = [i for i in c if i.is_real is not False]
                    if len(c) > 1:
                        c = [i for i in c if i.is_real]
                    if len(c) != 1:
                        raise Undecidable("not sure which point is real")
                    p = a.subs(t, c[0])
                    if p not in o:
                        return []  # e.g. a segment might not intersect a plane
                    return [p]
        if isinstance(o, Plane):
            if self.equals(o):
                return [self]
            if self.is_parallel(o):
                return []
            else:
                x, y, z = map(Dummy, 'xyz')
                a, b = Matrix([self.normal_vector]), Matrix([o.normal_vector])
                c = list(a.cross(b))
                d = self.equation(x, y, z)
                e = o.equation(x, y, z)
                result = list(linsolve([d, e], x, y, z))[0]
                for i in (x, y, z): result = result.subs(i, 0)
                return [Line3D(Point3D(result), direction_ratio=c)]


    def is_coplanar(self, o):
        """ Returns True if `o` is coplanar with self, else False.

        Examples
        ========

        >>> from sympy import Plane
        >>> o = (0, 0, 0)
        >>> p = Plane(o, (1, 1, 1))
        >>> p2 = Plane(o, (2, 2, 2))
        >>> p == p2
        False
        >>> p.is_coplanar(p2)
        True
        """
        if isinstance(o, Plane):
            return not cancel(self.equation(x, y, z)/o.equation(x, y, z)).has(x, y, z)
        if isinstance(o, Point3D):
            return o in self
        elif isinstance(o, LinearEntity3D):
            return all(i in self for i in self)
        elif isinstance(o, GeometryEntity):  # XXX should only be handling 2D objects now
            return all(i == 0 for i in self.normal_vector[:2])


    def is_parallel(self, l):
        """Is the given geometric entity parallel to the plane?

        Parameters
        ==========

        LinearEntity3D or Plane

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> b = Plane(Point3D(3,1,3), normal_vector=(4, 8, 12))
        >>> a.is_parallel(b)
        True

        """
        if isinstance(l, LinearEntity3D):
            a = l.direction_ratio
            b = self.normal_vector
            return sum(i*j for i, j in zip(a, b)) == 0
        if isinstance(l, Plane):
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            return bool(a.cross(b).is_zero_matrix)


    def is_perpendicular(self, l):
        """Is the given geometric entity perpendicualar to the given plane?

        Parameters
        ==========

        LinearEntity3D or Plane

        Returns
        =======

        Boolean

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> b = Plane(Point3D(2, 2, 2), normal_vector=(-1, 2, -1))
        >>> a.is_perpendicular(b)
        True

        """
        if isinstance(l, LinearEntity3D):
            a = Matrix(l.direction_ratio)
            b = Matrix(self.normal_vector)
            if a.cross(b).is_zero_matrix:
                return True
            else:
                return False
        elif isinstance(l, Plane):
            a = Matrix(l.normal_vector)
            b = Matrix(self.normal_vector)
            if a.dot(b) == 0:
                return True
            else:
                return False
        else:
            return False

    @property
    def normal_vector(self):
        """Normal vector of the given plane.

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        >>> a.normal_vector
        (-1, 2, -1)
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 4, 7))
        >>> a.normal_vector
        (1, 4, 7)

        """
        return self.args[1]

    @property
    def p1(self):
        """The only defining point of the plane. Others can be obtained from the
        arbitrary_point method.

        See Also
        ========

        sympy.geometry.point.Point3D

        Examples
        ========

        >>> from sympy import Point3D, Plane
        >>> a = Plane(Point3D(1, 1, 1), Point3D(2, 3, 4), Point3D(2, 2, 2))
        >>> a.p1
        Point3D(1, 1, 1)

        """
        return self.args[0]

    def parallel_plane(self, pt):
        """
        Plane parallel to the given plane and passing through the point pt.

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1, 4, 6), normal_vector=(2, 4, 6))
        >>> a.parallel_plane(Point3D(2, 3, 5))
        Plane(Point3D(2, 3, 5), (2, 4, 6))

        """
        a = self.normal_vector
        return Plane(pt, normal_vector=a)

    def perpendicular_line(self, pt):
        """A line perpendicular to the given plane.

        Parameters
        ==========

        pt: Point3D

        Returns
        =======

        Line3D

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a = Plane(Point3D(1,4,6), normal_vector=(2, 4, 6))
        >>> a.perpendicular_line(Point3D(9, 8, 7))
        Line3D(Point3D(9, 8, 7), Point3D(11, 12, 13))

        """
        a = self.normal_vector
        return Line3D(pt, direction_ratio=a)

    def perpendicular_plane(self, *pts):
        """
        Return a perpendicular passing through the given points. If the
        direction ratio between the points is the same as the Plane's normal
        vector then, to select from the infinite number of possible planes,
        a third point will be chosen on the z-axis (or the y-axis
        if the normal vector is already parallel to the z-axis). If less than
        two points are given they will be supplied as follows: if no point is
        given then pt1 will be self.p1; if a second point is not given it will
        be a point through pt1 on a line parallel to the z-axis (if the normal
        is not already the z-axis, otherwise on the line parallel to the
        y-axis).

        Parameters
        ==========

        pts: 0, 1 or 2 Point3D

        Returns
        =======

        Plane

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)
        >>> Z = (0, 0, 1)
        >>> p = Plane(a, normal_vector=Z)
        >>> p.perpendicular_plane(a, b)
        Plane(Point3D(0, 0, 0), (1, 0, 0))
        """
        if len(pts) > 2:
            raise ValueError('No more than 2 pts should be provided.')

        pts = list(pts)
        if len(pts) == 0:
            pts.append(self.p1)
        if len(pts) == 1:
            x, y, z = self.normal_vector
            if x == y == 0:
                dir = (0, 1, 0)
            else:
                dir = (0, 0, 1)
            pts.append(pts[0] + Point3D(*dir))

        p1, p2 = [Point(i, dim=3) for i in pts]
        l = Line3D(p1, p2)
        n = Line3D(p1, direction_ratio=self.normal_vector)
        if l in n:  # XXX should an error be raised instead?
            # there are infinitely many perpendicular planes;
            x, y, z = self.normal_vector
            if x == y == 0:
                # the z axis is the normal so pick a pt on the y-axis
                p3 = Point3D(0, 1, 0)  # case 1
            else:
                # else pick a pt on the z axis
                p3 = Point3D(0, 0, 1)  # case 2
            # in case that point is already given, move it a bit
            if p3 in l:
                p3 *= 2  # case 3
        else:
            p3 = p1 + Point3D(*self.normal_vector)  # case 4
        return Plane(p1, p2, p3)

    def projection_line(self, line):
        """Project the given line onto the plane through the normal plane
        containing the line.

        Parameters
        ==========

        LinearEntity or LinearEntity3D

        Returns
        =======

        Point3D, Line3D, Ray3D or Segment3D

        Notes
        =====

        For the interaction between 2D and 3D lines(segments, rays), you should
        convert the line to 3D by using this method. For example for finding the
        intersection between a 2D and a 3D line, convert the 2D line to a 3D line
        by projecting it on a required plane and then proceed to find the
        intersection between those lines.

        Examples
        ========

        >>> from sympy import Plane, Line, Line3D, Point3D
        >>> a = Plane(Point3D(1, 1, 1), normal_vector=(1, 1, 1))
        >>> b = Line(Point3D(1, 1), Point3D(2, 2))
        >>> a.projection_line(b)
        Line3D(Point3D(4/3, 4/3, 1/3), Point3D(5/3, 5/3, -1/3))
        >>> c = Line3D(Point3D(1, 1, 1), Point3D(2, 2, 2))
        >>> a.projection_line(c)
        Point3D(1, 1, 1)

        """
        if not isinstance(line, (LinearEntity, LinearEntity3D)):
            raise NotImplementedError('Enter a linear entity only')
        a, b = self.projection(line.p1), self.projection(line.p2)
        if a == b:
            # projection does not imply intersection so for
            # this case (line parallel to plane's normal) we
            # return the projection point
            return a
        if isinstance(line, (Line, Line3D)):
            return Line3D(a, b)
        if isinstance(line, (Ray, Ray3D)):
            return Ray3D(a, b)
        if isinstance(line, (Segment, Segment3D)):
            return Segment3D(a, b)

    def projection(self, pt):
        """Project the given point onto the plane along the plane normal.

        Parameters
        ==========

        Point or Point3D

        Returns
        =======

        Point3D

        Examples
        ========

        >>> from sympy import Plane, Point3D
        >>> A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))

        The projection is along the normal vector direction, not the z
        axis, so (1, 1) does not project to (1, 1, 2) on the plane A:

        >>> b = Point3D(1, 1)
        >>> A.projection(b)
        Point3D(5/3, 5/3, 2/3)
        >>> _ in A
        True

        But the point (1, 1, 2) projects to (1, 1) on the XY-plane:

        >>> XY = Plane((0, 0, 0), (0, 0, 1))
        >>> XY.projection((1, 1, 2))
        Point3D(1, 1, 0)
        """
        rv = Point(pt, dim=3)
        if rv in self:
            return rv
        return self.intersection(Line3D(rv, rv + Point3D(self.normal_vector)))[0]

    def random_point(self, seed=None):
        """ Returns a random point on the Plane.

        Returns
        =======

        Point3D

        Examples
        ========

        >>> from sympy import Plane
        >>> p = Plane((1, 0, 0), normal_vector=(0, 1, 0))
        >>> r = p.random_point(seed=42)  # seed value is optional
        >>> r.n(3)
        Point3D(2.29, 0, -1.35)

        The random point can be moved to lie on the circle of radius
        1 centered on p1:

        >>> c = p.p1 + (r - p.p1).unit
        >>> c.distance(p.p1).equals(1)
        True
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        params = {
            x: 2*Rational(rng.gauss(0, 1)) - 1,
            y: 2*Rational(rng.gauss(0, 1)) - 1}
        return self.arbitrary_point(x, y).subs(params)

    def parameter_value(self, other, u, v=None):
        """Return the parameter(s) corresponding to the given point.

        Examples
        ========

        >>> from sympy import pi, Plane
        >>> from sympy.abc import t, u, v
        >>> p = Plane((2, 0, 0), (0, 0, 1), (0, 1, 0))

        By default, the parameter value returned defines a point
        that is a distance of 1 from the Plane's p1 value and
        in line with the given point:

        >>> on_circle = p.arbitrary_point(t).subs(t, pi/4)
        >>> on_circle.distance(p.p1)
        1
        >>> p.parameter_value(on_circle, t)
        {t: pi/4}

        Moving the point twice as far from p1 does not change
        the parameter value:

        >>> off_circle = p.p1 + (on_circle - p.p1)*2
        >>> off_circle.distance(p.p1)
        2
        >>> p.parameter_value(off_circle, t)
        {t: pi/4}

        If the 2-value parameter is desired, supply the two
        parameter symbols and a replacement dictionary will
        be returned:

        >>> p.parameter_value(on_circle, u, v)
        {u: sqrt(10)/10, v: sqrt(10)/30}
        >>> p.parameter_value(off_circle, u, v)
        {u: sqrt(10)/5, v: sqrt(10)/15}
        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other, Point):
            raise ValueError("other must be a point")
        if other == self.p1:
            return other
        if isinstance(u, Symbol) and v is None:
            delta = self.arbitrary_point(u) - self.p1
            eq = delta - (other - self.p1).unit
            sol = solve(eq, u, dict=True)
        elif isinstance(u, Symbol) and isinstance(v, Symbol):
            pt = self.arbitrary_point(u, v)
            sol = solve(pt - other, (u, v), dict=True)
        else:
            raise ValueError('expecting 1 or 2 symbols')
        if not sol:
            raise ValueError("Given point is not on %s" % func_name(self))
        return sol[0]  # {t: tval} or {u: uval, v: vval}

    @property
    def ambient_dimension(self):
        return self.p1.ambient_dimension

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\termui.py ===
from __future__ import annotations

import collections.abc as cabc
import inspect
import io
import itertools
import sys
import typing as t
from contextlib import AbstractContextManager
from gettext import gettext as _

from ._compat import isatty
from ._compat import strip_ansi
from .exceptions import Abort
from .exceptions import UsageError
from .globals import resolve_color_default
from .types import Choice
from .types import convert_type
from .types import ParamType
from .utils import echo
from .utils import LazyFile

if t.TYPE_CHECKING:
    from ._termui_impl import ProgressBar

V = t.TypeVar("V")

# The prompt functions to use.  The doc tools currently override these
# functions to customize how they work.
visible_prompt_func: t.Callable[[str], str] = input

_ansi_colors = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "reset": 39,
    "bright_black": 90,
    "bright_red": 91,
    "bright_green": 92,
    "bright_yellow": 93,
    "bright_blue": 94,
    "bright_magenta": 95,
    "bright_cyan": 96,
    "bright_white": 97,
}
_ansi_reset_all = "\033[0m"


def hidden_prompt_func(prompt: str) -> str:
    import getpass

    return getpass.getpass(prompt)


def _build_prompt(
    text: str,
    suffix: str,
    show_default: bool = False,
    default: t.Any | None = None,
    show_choices: bool = True,
    type: ParamType | None = None,
) -> str:
    prompt = text
    if type is not None and show_choices and isinstance(type, Choice):
        prompt += f" ({', '.join(map(str, type.choices))})"
    if default is not None and show_default:
        prompt = f"{prompt} [{_format_default(default)}]"
    return f"{prompt}{suffix}"


def _format_default(default: t.Any) -> t.Any:
    if isinstance(default, (io.IOBase, LazyFile)) and hasattr(default, "name"):
        return default.name

    return default


def prompt(
    text: str,
    default: t.Any | None = None,
    hide_input: bool = False,
    confirmation_prompt: bool | str = False,
    type: ParamType | t.Any | None = None,
    value_proc: t.Callable[[str], t.Any] | None = None,
    prompt_suffix: str = ": ",
    show_default: bool = True,
    err: bool = False,
    show_choices: bool = True,
) -> t.Any:
    """Prompts a user for input.  This is a convenience function that can
    be used to prompt a user for input later.

    If the user aborts the input by sending an interrupt signal, this
    function will catch it and raise a :exc:`Abort` exception.

    :param text: the text to show for the prompt.
    :param default: the default value to use if no input happens.  If this
                    is not given it will prompt until it's aborted.
    :param hide_input: if this is set to true then the input value will
                       be hidden.
    :param confirmation_prompt: Prompt a second time to confirm the
        value. Can be set to a string instead of ``True`` to customize
        the message.
    :param type: the type to use to check the value against.
    :param value_proc: if this parameter is provided it's a function that
                       is invoked instead of the type conversion to
                       convert a value.
    :param prompt_suffix: a suffix that should be added to the prompt.
    :param show_default: shows or hides the default value in the prompt.
    :param err: if set to true the file defaults to ``stderr`` instead of
                ``stdout``, the same as with echo.
    :param show_choices: Show or hide choices if the passed type is a Choice.
                         For example if type is a Choice of either day or week,
                         show_choices is true and text is "Group by" then the
                         prompt will be "Group by (day, week): ".

    .. versionadded:: 8.0
        ``confirmation_prompt`` can be a custom string.

    .. versionadded:: 7.0
        Added the ``show_choices`` parameter.

    .. versionadded:: 6.0
        Added unicode support for cmd.exe on Windows.

    .. versionadded:: 4.0
        Added the `err` parameter.

    """

    def prompt_func(text: str) -> str:
        f = hidden_prompt_func if hide_input else visible_prompt_func
        try:
            # Write the prompt separately so that we get nice
            # coloring through colorama on Windows
            echo(text.rstrip(" "), nl=False, err=err)
            # Echo a space to stdout to work around an issue where
            # readline causes backspace to clear the whole line.
            return f(" ")
        except (KeyboardInterrupt, EOFError):
            # getpass doesn't print a newline if the user aborts input with ^C.
            # Allegedly this behavior is inherited from getpass(3).
            # A doc bug has been filed at https://bugs.python.org/issue24711
            if hide_input:
                echo(None, err=err)
            raise Abort() from None

    if value_proc is None:
        value_proc = convert_type(type, default)

    prompt = _build_prompt(
        text, prompt_suffix, show_default, default, show_choices, type
    )

    if confirmation_prompt:
        if confirmation_prompt is True:
            confirmation_prompt = _("Repeat for confirmation")

        confirmation_prompt = _build_prompt(confirmation_prompt, prompt_suffix)

    while True:
        while True:
            value = prompt_func(prompt)
            if value:
                break
            elif default is not None:
                value = default
                break
        try:
            result = value_proc(value)
        except UsageError as e:
            if hide_input:
                echo(_("Error: The value you entered was invalid."), err=err)
            else:
                echo(_("Error: {e.message}").format(e=e), err=err)
            continue
        if not confirmation_prompt:
            return result
        while True:
            value2 = prompt_func(confirmation_prompt)
            is_empty = not value and not value2
            if value2 or is_empty:
                break
        if value == value2:
            return result
        echo(_("Error: The two entered values do not match."), err=err)


def confirm(
    text: str,
    default: bool | None = False,
    abort: bool = False,
    prompt_suffix: str = ": ",
    show_default: bool = True,
    err: bool = False,
) -> bool:
    """Prompts for confirmation (yes/no question).

    If the user aborts the input by sending a interrupt signal this
    function will catch it and raise a :exc:`Abort` exception.

    :param text: the question to ask.
    :param default: The default value to use when no input is given. If
        ``None``, repeat until input is given.
    :param abort: if this is set to `True` a negative answer aborts the
                  exception by raising :exc:`Abort`.
    :param prompt_suffix: a suffix that should be added to the prompt.
    :param show_default: shows or hides the default value in the prompt.
    :param err: if set to true the file defaults to ``stderr`` instead of
                ``stdout``, the same as with echo.

    .. versionchanged:: 8.0
        Repeat until input is given if ``default`` is ``None``.

    .. versionadded:: 4.0
        Added the ``err`` parameter.
    """
    prompt = _build_prompt(
        text,
        prompt_suffix,
        show_default,
        "y/n" if default is None else ("Y/n" if default else "y/N"),
    )

    while True:
        try:
            # Write the prompt separately so that we get nice
            # coloring through colorama on Windows
            echo(prompt.rstrip(" "), nl=False, err=err)
            # Echo a space to stdout to work around an issue where
            # readline causes backspace to clear the whole line.
            value = visible_prompt_func(" ").lower().strip()
        except (KeyboardInterrupt, EOFError):
            raise Abort() from None
        if value in ("y", "yes"):
            rv = True
        elif value in ("n", "no"):
            rv = False
        elif default is not None and value == "":
            rv = default
        else:
            echo(_("Error: invalid input"), err=err)
            continue
        break
    if abort and not rv:
        raise Abort()
    return rv


def echo_via_pager(
    text_or_generator: cabc.Iterable[str] | t.Callable[[], cabc.Iterable[str]] | str,
    color: bool | None = None,
) -> None:
    """This function takes a text and shows it via an environment specific
    pager on stdout.

    .. versionchanged:: 3.0
       Added the `color` flag.

    :param text_or_generator: the text to page, or alternatively, a
                              generator emitting the text to page.
    :param color: controls if the pager supports ANSI colors or not.  The
                  default is autodetection.
    """
    color = resolve_color_default(color)

    if inspect.isgeneratorfunction(text_or_generator):
        i = t.cast("t.Callable[[], cabc.Iterable[str]]", text_or_generator)()
    elif isinstance(text_or_generator, str):
        i = [text_or_generator]
    else:
        i = iter(t.cast("cabc.Iterable[str]", text_or_generator))

    # convert every element of i to a text type if necessary
    text_generator = (el if isinstance(el, str) else str(el) for el in i)

    from ._termui_impl import pager

    return pager(itertools.chain(text_generator, "\n"), color)


@t.overload
def progressbar(
    *,
    length: int,
    label: str | None = None,
    hidden: bool = False,
    show_eta: bool = True,
    show_percent: bool | None = None,
    show_pos: bool = False,
    fill_char: str = "#",
    empty_char: str = "-",
    bar_template: str = "%(label)s  [%(bar)s]  %(info)s",
    info_sep: str = "  ",
    width: int = 36,
    file: t.TextIO | None = None,
    color: bool | None = None,
    update_min_steps: int = 1,
) -> ProgressBar[int]: ...


@t.overload
def progressbar(
    iterable: cabc.Iterable[V] | None = None,
    length: int | None = None,
    label: str | None = None,
    hidden: bool = False,
    show_eta: bool = True,
    show_percent: bool | None = None,
    show_pos: bool = False,
    item_show_func: t.Callable[[V | None], str | None] | None = None,
    fill_char: str = "#",
    empty_char: str = "-",
    bar_template: str = "%(label)s  [%(bar)s]  %(info)s",
    info_sep: str = "  ",
    width: int = 36,
    file: t.TextIO | None = None,
    color: bool | None = None,
    update_min_steps: int = 1,
) -> ProgressBar[V]: ...


def progressbar(
    iterable: cabc.Iterable[V] | None = None,
    length: int | None = None,
    label: str | None = None,
    hidden: bool = False,
    show_eta: bool = True,
    show_percent: bool | None = None,
    show_pos: bool = False,
    item_show_func: t.Callable[[V | None], str | None] | None = None,
    fill_char: str = "#",
    empty_char: str = "-",
    bar_template: str = "%(label)s  [%(bar)s]  %(info)s",
    info_sep: str = "  ",
    width: int = 36,
    file: t.TextIO | None = None,
    color: bool | None = None,
    update_min_steps: int = 1,
) -> ProgressBar[V]:
    """This function creates an iterable context manager that can be used
    to iterate over something while showing a progress bar.  It will
    either iterate over the `iterable` or `length` items (that are counted
    up).  While iteration happens, this function will print a rendered
    progress bar to the given `file` (defaults to stdout) and will attempt
    to calculate remaining time and more.  By default, this progress bar
    will not be rendered if the file is not a terminal.

    The context manager creates the progress bar.  When the context
    manager is entered the progress bar is already created.  With every
    iteration over the progress bar, the iterable passed to the bar is
    advanced and the bar is updated.  When the context manager exits,
    a newline is printed and the progress bar is finalized on screen.

    Note: The progress bar is currently designed for use cases where the
    total progress can be expected to take at least several seconds.
    Because of this, the ProgressBar class object won't display
    progress that is considered too fast, and progress where the time
    between steps is less than a second.

    No printing must happen or the progress bar will be unintentionally
    destroyed.

    Example usage::

        with progressbar(items) as bar:
            for item in bar:
                do_something_with(item)

    Alternatively, if no iterable is specified, one can manually update the
    progress bar through the `update()` method instead of directly
    iterating over the progress bar.  The update method accepts the number
    of steps to increment the bar with::

        with progressbar(length=chunks.total_bytes) as bar:
            for chunk in chunks:
                process_chunk(chunk)
                bar.update(chunks.bytes)

    The ``update()`` method also takes an optional value specifying the
    ``current_item`` at the new position. This is useful when used
    together with ``item_show_func`` to customize the output for each
    manual step::

        with click.progressbar(
            length=total_size,
            label='Unzipping archive',
            item_show_func=lambda a: a.filename
        ) as bar:
            for archive in zip_file:
                archive.extract()
                bar.update(archive.size, archive)

    :param iterable: an iterable to iterate over.  If not provided the length
                     is required.
    :param length: the number of items to iterate over.  By default the
                   progressbar will attempt to ask the iterator about its
                   length, which might or might not work.  If an iterable is
                   also provided this parameter can be used to override the
                   length.  If an iterable is not provided the progress bar
                   will iterate over a range of that length.
    :param label: the label to show next to the progress bar.
    :param hidden: hide the progressbar. Defaults to ``False``. When no tty is
        detected, it will only print the progressbar label. Setting this to
        ``False`` also disables that.
    :param show_eta: enables or disables the estimated time display.  This is
                     automatically disabled if the length cannot be
                     determined.
    :param show_percent: enables or disables the percentage display.  The
                         default is `True` if the iterable has a length or
                         `False` if not.
    :param show_pos: enables or disables the absolute position display.  The
                     default is `False`.
    :param item_show_func: A function called with the current item which
        can return a string to show next to the progress bar. If the
        function returns ``None`` nothing is shown. The current item can
        be ``None``, such as when entering and exiting the bar.
    :param fill_char: the character to use to show the filled part of the
                      progress bar.
    :param empty_char: the character to use to show the non-filled part of
                       the progress bar.
    :param bar_template: the format string to use as template for the bar.
                         The parameters in it are ``label`` for the label,
                         ``bar`` for the progress bar and ``info`` for the
                         info section.
    :param info_sep: the separator between multiple info items (eta etc.)
    :param width: the width of the progress bar in characters, 0 means full
                  terminal width
    :param file: The file to write to. If this is not a terminal then
        only the label is printed.
    :param color: controls if the terminal supports ANSI colors or not.  The
                  default is autodetection.  This is only needed if ANSI
                  codes are included anywhere in the progress bar output
                  which is not the case by default.
    :param update_min_steps: Render only when this many updates have
        completed. This allows tuning for very fast iterators.

    .. versionadded:: 8.2
        The ``hidden`` argument.

    .. versionchanged:: 8.0
        Output is shown even if execution time is less than 0.5 seconds.

    .. versionchanged:: 8.0
        ``item_show_func`` shows the current item, not the previous one.

    .. versionchanged:: 8.0
        Labels are echoed if the output is not a TTY. Reverts a change
        in 7.0 that removed all output.

    .. versionadded:: 8.0
       The ``update_min_steps`` parameter.

    .. versionadded:: 4.0
        The ``color`` parameter and ``update`` method.

    .. versionadded:: 2.0
    """
    from ._termui_impl import ProgressBar

    color = resolve_color_default(color)
    return ProgressBar(
        iterable=iterable,
        length=length,
        hidden=hidden,
        show_eta=show_eta,
        show_percent=show_percent,
        show_pos=show_pos,
        item_show_func=item_show_func,
        fill_char=fill_char,
        empty_char=empty_char,
        bar_template=bar_template,
        info_sep=info_sep,
        file=file,
        label=label,
        width=width,
        color=color,
        update_min_steps=update_min_steps,
    )


def clear() -> None:
    """Clears the terminal screen.  This will have the effect of clearing
    the whole visible space of the terminal and moving the cursor to the
    top left.  This does not do anything if not connected to a terminal.

    .. versionadded:: 2.0
    """
    if not isatty(sys.stdout):
        return

    # ANSI escape \033[2J clears the screen, \033[1;1H moves the cursor
    echo("\033[2J\033[1;1H", nl=False)


def _interpret_color(color: int | tuple[int, int, int] | str, offset: int = 0) -> str:
    if isinstance(color, int):
        return f"{38 + offset};5;{color:d}"

    if isinstance(color, (tuple, list)):
        r, g, b = color
        return f"{38 + offset};2;{r:d};{g:d};{b:d}"

    return str(_ansi_colors[color] + offset)


def style(
    text: t.Any,
    fg: int | tuple[int, int, int] | str | None = None,
    bg: int | tuple[int, int, int] | str | None = None,
    bold: bool | None = None,
    dim: bool | None = None,
    underline: bool | None = None,
    overline: bool | None = None,
    italic: bool | None = None,
    blink: bool | None = None,
    reverse: bool | None = None,
    strikethrough: bool | None = None,
    reset: bool = True,
) -> str:
    """Styles a text with ANSI styles and returns the new string.  By
    default the styling is self contained which means that at the end
    of the string a reset code is issued.  This can be prevented by
    passing ``reset=False``.

    Examples::

        click.echo(click.style('Hello World!', fg='green'))
        click.echo(click.style('ATTENTION!', blink=True))
        click.echo(click.style('Some things', reverse=True, fg='cyan'))
        click.echo(click.style('More colors', fg=(255, 12, 128), bg=117))

    Supported color names:

    * ``black`` (might be a gray)
    * ``red``
    * ``green``
    * ``yellow`` (might be an orange)
    * ``blue``
    * ``magenta``
    * ``cyan``
    * ``white`` (might be light gray)
    * ``bright_black``
    * ``bright_red``
    * ``bright_green``
    * ``bright_yellow``
    * ``bright_blue``
    * ``bright_magenta``
    * ``bright_cyan``
    * ``bright_white``
    * ``reset`` (reset the color code only)

    If the terminal supports it, color may also be specified as:

    -   An integer in the interval [0, 255]. The terminal must support
        8-bit/256-color mode.
    -   An RGB tuple of three integers in [0, 255]. The terminal must
        support 24-bit/true-color mode.

    See https://en.wikipedia.org/wiki/ANSI_color and
    https://gist.github.com/XVilka/8346728 for more information.

    :param text: the string to style with ansi codes.
    :param fg: if provided this will become the foreground color.
    :param bg: if provided this will become the background color.
    :param bold: if provided this will enable or disable bold mode.
    :param dim: if provided this will enable or disable dim mode.  This is
                badly supported.
    :param underline: if provided this will enable or disable underline.
    :param overline: if provided this will enable or disable overline.
    :param italic: if provided this will enable or disable italic.
    :param blink: if provided this will enable or disable blinking.
    :param reverse: if provided this will enable or disable inverse
                    rendering (foreground becomes background and the
                    other way round).
    :param strikethrough: if provided this will enable or disable
        striking through text.
    :param reset: by default a reset-all code is added at the end of the
                  string which means that styles do not carry over.  This
                  can be disabled to compose styles.

    .. versionchanged:: 8.0
        A non-string ``message`` is converted to a string.

    .. versionchanged:: 8.0
       Added support for 256 and RGB color codes.

    .. versionchanged:: 8.0
        Added the ``strikethrough``, ``italic``, and ``overline``
        parameters.

    .. versionchanged:: 7.0
        Added support for bright colors.

    .. versionadded:: 2.0
    """
    if not isinstance(text, str):
        text = str(text)

    bits = []

    if fg:
        try:
            bits.append(f"\033[{_interpret_color(fg)}m")
        except KeyError:
            raise TypeError(f"Unknown color {fg!r}") from None

    if bg:
        try:
            bits.append(f"\033[{_interpret_color(bg, 10)}m")
        except KeyError:
            raise TypeError(f"Unknown color {bg!r}") from None

    if bold is not None:
        bits.append(f"\033[{1 if bold else 22}m")
    if dim is not None:
        bits.append(f"\033[{2 if dim else 22}m")
    if underline is not None:
        bits.append(f"\033[{4 if underline else 24}m")
    if overline is not None:
        bits.append(f"\033[{53 if overline else 55}m")
    if italic is not None:
        bits.append(f"\033[{3 if italic else 23}m")
    if blink is not None:
        bits.append(f"\033[{5 if blink else 25}m")
    if reverse is not None:
        bits.append(f"\033[{7 if reverse else 27}m")
    if strikethrough is not None:
        bits.append(f"\033[{9 if strikethrough else 29}m")
    bits.append(text)
    if reset:
        bits.append(_ansi_reset_all)
    return "".join(bits)


def unstyle(text: str) -> str:
    """Removes ANSI styling information from a string.  Usually it's not
    necessary to use this function as Click's echo function will
    automatically remove styling if necessary.

    .. versionadded:: 2.0

    :param text: the text to remove style information from.
    """
    return strip_ansi(text)


def secho(
    message: t.Any | None = None,
    file: t.IO[t.AnyStr] | None = None,
    nl: bool = True,
    err: bool = False,
    color: bool | None = None,
    **styles: t.Any,
) -> None:
    """This function combines :func:`echo` and :func:`style` into one
    call.  As such the following two calls are the same::

        click.secho('Hello World!', fg='green')
        click.echo(click.style('Hello World!', fg='green'))

    All keyword arguments are forwarded to the underlying functions
    depending on which one they go with.

    Non-string types will be converted to :class:`str`. However,
    :class:`bytes` are passed directly to :meth:`echo` without applying
    style. If you want to style bytes that represent text, call
    :meth:`bytes.decode` first.

    .. versionchanged:: 8.0
        A non-string ``message`` is converted to a string. Bytes are
        passed through without style applied.

    .. versionadded:: 2.0
    """
    if message is not None and not isinstance(message, (bytes, bytearray)):
        message = style(message, **styles)

    return echo(message, file=file, nl=nl, err=err, color=color)


@t.overload
def edit(
    text: bytes | bytearray,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = False,
    extension: str = ".txt",
) -> bytes | None: ...


@t.overload
def edit(
    text: str,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = True,
    extension: str = ".txt",
) -> str | None: ...


@t.overload
def edit(
    text: None = None,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = True,
    extension: str = ".txt",
    filename: str | cabc.Iterable[str] | None = None,
) -> None: ...


def edit(
    text: str | bytes | bytearray | None = None,
    editor: str | None = None,
    env: cabc.Mapping[str, str] | None = None,
    require_save: bool = True,
    extension: str = ".txt",
    filename: str | cabc.Iterable[str] | None = None,
) -> str | bytes | bytearray | None:
    r"""Edits the given text in the defined editor.  If an editor is given
    (should be the full path to the executable but the regular operating
    system search path is used for finding the executable) it overrides
    the detected editor.  Optionally, some environment variables can be
    used.  If the editor is closed without changes, `None` is returned.  In
    case a file is edited directly the return value is always `None` and
    `require_save` and `extension` are ignored.

    If the editor cannot be opened a :exc:`UsageError` is raised.

    Note for Windows: to simplify cross-platform usage, the newlines are
    automatically converted from POSIX to Windows and vice versa.  As such,
    the message here will have ``\n`` as newline markers.

    :param text: the text to edit.
    :param editor: optionally the editor to use.  Defaults to automatic
                   detection.
    :param env: environment variables to forward to the editor.
    :param require_save: if this is true, then not saving in the editor
                         will make the return value become `None`.
    :param extension: the extension to tell the editor about.  This defaults
                      to `.txt` but changing this might change syntax
                      highlighting.
    :param filename: if provided it will edit this file instead of the
                     provided text contents.  It will not use a temporary
                     file as an indirection in that case. If the editor supports
                     editing multiple files at once, a sequence of files may be
                     passed as well. Invoke `click.file` once per file instead
                     if multiple files cannot be managed at once or editing the
                     files serially is desired.

    .. versionchanged:: 8.2.0
        ``filename`` now accepts any ``Iterable[str]`` in addition to a ``str``
        if the ``editor`` supports editing multiple files at once.

    """
    from ._termui_impl import Editor

    ed = Editor(editor=editor, env=env, require_save=require_save, extension=extension)

    if filename is None:
        return ed.edit(text)

    if isinstance(filename, str):
        filename = (filename,)

    ed.edit_files(filenames=filename)
    return None


def launch(url: str, wait: bool = False, locate: bool = False) -> int:
    """This function launches the given URL (or filename) in the default
    viewer application for this file type.  If this is an executable, it
    might launch the executable in a new session.  The return value is
    the exit code of the launched application.  Usually, ``0`` indicates
    success.

    Examples::

        click.launch('https://click.palletsprojects.com/')
        click.launch('/my/downloaded/file', locate=True)

    .. versionadded:: 2.0

    :param url: URL or filename of the thing to launch.
    :param wait: Wait for the program to exit before returning. This
        only works if the launched program blocks. In particular,
        ``xdg-open`` on Linux does not block.
    :param locate: if this is set to `True` then instead of launching the
                   application associated with the URL it will attempt to
                   launch a file manager with the file located.  This
                   might have weird effects if the URL does not point to
                   the filesystem.
    """
    from ._termui_impl import open_url

    return open_url(url, wait=wait, locate=locate)


# If this is provided, getchar() calls into this instead.  This is used
# for unittesting purposes.
_getchar: t.Callable[[bool], str] | None = None


def getchar(echo: bool = False) -> str:
    """Fetches a single character from the terminal and returns it.  This
    will always return a unicode character and under certain rare
    circumstances this might return more than one character.  The
    situations which more than one character is returned is when for
    whatever reason multiple characters end up in the terminal buffer or
    standard input was not actually a terminal.

    Note that this will always read from the terminal, even if something
    is piped into the standard input.

    Note for Windows: in rare cases when typing non-ASCII characters, this
    function might wait for a second character and then return both at once.
    This is because certain Unicode characters look like special-key markers.

    .. versionadded:: 2.0

    :param echo: if set to `True`, the character read will also show up on
                 the terminal.  The default is to not show it.
    """
    global _getchar

    if _getchar is None:
        from ._termui_impl import getchar as f

        _getchar = f

    return _getchar(echo)


def raw_terminal() -> AbstractContextManager[int]:
    from ._termui_impl import raw_terminal as f

    return f()


def pause(info: str | None = None, err: bool = False) -> None:
    """This command stops execution and waits for the user to press any
    key to continue.  This is similar to the Windows batch "pause"
    command.  If the program is not run through a terminal, this command
    will instead do nothing.

    .. versionadded:: 2.0

    .. versionadded:: 4.0
       Added the `err` parameter.

    :param info: The message to print before pausing. Defaults to
        ``"Press any key to continue..."``.
    :param err: if set to message goes to ``stderr`` instead of
                ``stdout``, the same as with echo.
    """
    if not isatty(sys.stdin) or not isatty(sys.stdout):
        return

    if info is None:
        info = _("Press any key to continue...")

    try:
        if info:
            echo(info, nl=False, err=err)
        try:
            getchar()
        except (KeyboardInterrupt, EOFError):
            pass
    finally:
        if info:
            echo(err=err)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\matrices\eigen.py ===
#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################
#     module for the eigenvalue problem
#       Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
#
# todo:
#  - implement balancing
#  - agressive early deflation
#
##################################################################################################

"""
The eigenvalue problem
----------------------

This file contains routines for the eigenvalue problem.

high level routines:

  hessenberg : reduction of a real or complex square matrix to upper Hessenberg form
  schur : reduction of a real or complex square matrix to upper Schur form
  eig : eigenvalues and eigenvectors of a real or complex square matrix

low level routines:

  hessenberg_reduce_0 : reduction of a real or complex square matrix to upper Hessenberg form
  hessenberg_reduce_1 : auxiliary routine to hessenberg_reduce_0
  qr_step : a single implicitly shifted QR step for an upper Hessenberg matrix
  hessenberg_qr : Schur decomposition of an upper Hessenberg matrix
  eig_tr_r : right eigenvectors of an upper triangular matrix
  eig_tr_l : left  eigenvectors of an upper triangular matrix
"""

from ..libmp.backend import xrange

class Eigen(object):
    pass

def defun(f):
    setattr(Eigen, f.__name__, f)
    return f

def hessenberg_reduce_0(ctx, A, T):
    """
    This routine computes the (upper) Hessenberg decomposition of a square matrix A.
    Given A, an unitary matrix Q is calculated such that

               Q' A Q = H              and             Q' Q = Q Q' = 1

    where H is an upper Hessenberg matrix, meaning that it only contains zeros
    below the first subdiagonal. Here ' denotes the hermitian transpose (i.e.
    transposition and conjugation).

    parameters:
      A         (input/output) On input, A contains the square matrix A of
                dimension (n,n). On output, A contains a compressed representation
                of Q and H.
      T         (output) An array of length n containing the first elements of
                the Householder reflectors.
    """

    # internally we work with householder reflections from the right.
    # let u be a row vector (i.e. u[i]=A[i,:i]). then
    # Q is build up by reflectors of the type (1-v'v) where v is a suitable
    # modification of u. these reflectors are applyed to A from the right.
    # because we work with reflectors from the right we have to start with
    # the bottom row of A and work then upwards (this corresponds to
    # some kind of RQ decomposition).
    # the first part of the vectors v (i.e. A[i,:(i-1)]) are stored as row vectors
    # in the lower left part of A (excluding the diagonal and subdiagonal).
    # the last entry of v is stored in T.
    # the upper right part of A (including diagonal and subdiagonal) becomes H.


    n = A.rows
    if n <= 2: return

    for i in xrange(n-1, 1, -1):

        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[i,k])) + abs(ctx.im(A[i,k]))

        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale

        if scale == 0 or ctx.isinf(scale_inv):
            # sadly there are floating point numbers not equal to zero whose reciprocal is infinity
            T[i] = 0
            A[i,i-1] = 0
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[i,k] *= scale_inv
            rr = ctx.re(A[i,k])
            ii = ctx.im(A[i,k])
            H += rr * rr + ii * ii

        F = A[i,i-1]
        f = abs(F)
        G = ctx.sqrt(H)
        A[i,i-1] = - G * scale

        if f == 0:
            T[i] = G
        else:
            ff = F / f
            T[i] = F + G * ff
            A[i,i-1] *= ff

        H += G * f
        H = 1 / ctx.sqrt(H)

        T[i] *= H
        for k in xrange(0, i - 1):
            A[i,k] *= H

        for j in xrange(0, i):
            # apply housholder transformation (from right)

            G = ctx.conj(T[i]) * A[j,i-1]
            for k in xrange(0, i-1):
                G += ctx.conj(A[i,k]) * A[j,k]

            A[j,i-1] -= G * T[i]
            for k in xrange(0, i-1):
                A[j,k] -= G * A[i,k]

        for j in xrange(0, n):
            # apply housholder transformation (from left)

            G = T[i] * A[i-1,j]
            for k in xrange(0, i-1):
                G += A[i,k] * A[k,j]

            A[i-1,j] -= G * ctx.conj(T[i])
            for k in xrange(0, i-1):
                A[k,j] -= G * ctx.conj(A[i,k])



def hessenberg_reduce_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in hessenberg_reduce_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           hessenberg_reduce_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by hessenberg_reduce_0.
    """

    n = A.rows

    if n == 1:
        A[0,0] = 1
        return

    A[0,0] = A[1,1] = 1
    A[0,1] = A[1,0] = 0

    for i in xrange(2, n):
        if T[i] != 0:

            for j in xrange(0, i):
                G = T[i] * A[i-1,j]
                for k in xrange(0, i-1):
                    G += A[i,k] * A[k,j]

                A[i-1,j] -= G * ctx.conj(T[i])
                for k in xrange(0, i-1):
                    A[k,j] -= G * ctx.conj(A[i,k])

        A[i,i] = 1
        for j in xrange(0, i):
            A[j,i] = A[i,j] = 0



@defun
def hessenberg(ctx, A, overwrite_a = False):
    """
    This routine computes the Hessenberg decomposition of a square matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = H                and               Q' Q = Q Q' = 1

    where H is an upper right Hessenberg matrix. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation).

    input:
      A            : a real or complex square matrix
      overwrite_a  : if true, allows modification of A which may improve
                     performance. if false, A is not modified.

    output:
      Q : an unitary matrix
      H : an upper right Hessenberg matrix

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> Q, H = mp.hessenberg(A)
      >>> mp.nprint(H, 3) # doctest:+SKIP
      [  3.15  2.23  4.44]
      [-0.769  4.85  3.05]
      [   0.0  3.61   7.0]
      >>> print(mp.chop(A - Q * H * Q.transpose_conj()))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    return value:   (Q, H)
    """

    n = A.rows

    if n == 1:
        return (ctx.matrix([[1]]), A)

    if not overwrite_a:
        A = A.copy()

    T = ctx.matrix(n, 1)

    hessenberg_reduce_0(ctx, A, T)
    Q = A.copy()
    hessenberg_reduce_1(ctx, Q, T)

    for x in xrange(n):
        for y in xrange(x+2, n):
            A[y,x] = 0

    return Q, A


###########################################################################


def qr_step(ctx, n0, n1, A, Q, shift):
    """
    This subroutine executes a single implicitly shifted QR step applied to an
    upper Hessenberg matrix A. Given A and shift as input, first an QR
    decomposition is calculated:

      Q R = A - shift * 1 .

    The output is then following matrix:

      R Q + shift * 1

    parameters:
      n0, n1    (input) Two integers which specify the submatrix A[n0:n1,n0:n1]
                on which this subroutine operators. The subdiagonal elements
                to the left and below this submatrix must be deflated (i.e. zero).
                following restriction is imposed: n1>=n0+2
      A         (input/output) On input, A is an upper Hessenberg matrix.
                On output, A is replaced by "R Q + shift * 1"
      Q         (input/output) The parameter Q is multiplied by the unitary matrix
                Q arising from the QR decomposition. Q can also be false, in which
                case the unitary matrix Q is not computated.
      shift     (input) a complex number specifying the shift. idealy close to an
                eigenvalue of the bottemmost part of the submatrix A[n0:n1,n0:n1].

    references:
      Stoer, Bulirsch - Introduction to Numerical Analysis.
      Kresser : Numerical Methods for General and Structured Eigenvalue Problems
    """

    # implicitly shifted and bulge chasing is explained at p.398/399 in "Stoer, Bulirsch - Introduction to Numerical Analysis"
    # for bulge chasing see also "Watkins - The Matrix Eigenvalue Problem" sec.4.5,p.173

    # the Givens rotation we used is determined as follows: let c,s be two complex
    # numbers. then we have following relation:
    #
    #     v = sqrt(|c|^2 + |s|^2)
    #
    #     1/v [ c~  s~]  [c] = [v]
    #         [-s   c ]  [s]   [0]
    #
    # the matrix on the left is our Givens rotation.

    n = A.rows

    # first step

    # calculate givens rotation
    c = A[n0  ,n0] - shift
    s = A[n0+1,n0]

    v = ctx.hypot(ctx.hypot(ctx.re(c), ctx.im(c)), ctx.hypot(ctx.re(s), ctx.im(s)))

    if v == 0:
        v = 1
        c = 1
        s = 0
    else:
        c /= v
        s /= v

    cc = ctx.conj(c)
    cs = ctx.conj(s)

    for k in xrange(n0, n):
        # apply givens rotation from the left
        x = A[n0  ,k]
        y = A[n0+1,k]
        A[n0  ,k] = cc * x + cs * y
        A[n0+1,k] = c * y - s * x

    for k in xrange(min(n1, n0+3)):
        # apply givens rotation from the right
        x = A[k,n0  ]
        y = A[k,n0+1]
        A[k,n0  ] = c * x + s * y
        A[k,n0+1] = cc * y - cs * x

    if not isinstance(Q, bool):
        for k in xrange(n):
            # eigenvectors
            x = Q[k,n0  ]
            y = Q[k,n0+1]
            Q[k,n0  ] = c * x + s * y
            Q[k,n0+1] = cc * y - cs * x

    # chase the bulge

    for j in xrange(n0, n1 - 2):
        # calculate givens rotation

        c = A[j+1,j]
        s = A[j+2,j]

        v = ctx.hypot(ctx.hypot(ctx.re(c), ctx.im(c)), ctx.hypot(ctx.re(s), ctx.im(s)))

        if v == 0:
            A[j+1,j] = 0
            v = 1
            c = 1
            s = 0
        else:
            A[j+1,j] = v
            c /= v
            s /= v

        A[j+2,j] = 0

        cc = ctx.conj(c)
        cs = ctx.conj(s)

        for k in xrange(j+1, n):
            # apply givens rotation from the left
            x = A[j+1,k]
            y = A[j+2,k]
            A[j+1,k] = cc * x + cs * y
            A[j+2,k] = c * y - s * x

        for k in xrange(0, min(n1, j+4)):
            # apply givens rotation from the right
            x = A[k,j+1]
            y = A[k,j+2]
            A[k,j+1] = c * x + s * y
            A[k,j+2] = cc * y - cs * x

        if not isinstance(Q, bool):
            for k in xrange(0, n):
                # eigenvectors
                x = Q[k,j+1]
                y = Q[k,j+2]
                Q[k,j+1] = c * x + s * y
                Q[k,j+2] = cc * y - cs * x



def hessenberg_qr(ctx, A, Q):
    """
    This routine computes the Schur decomposition of an upper Hessenberg matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = R                   and                  Q' Q = Q Q' = 1

    where R is an upper right triangular matrix. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation).

    parameters:
      A         (input/output) On input, A contains an upper Hessenberg matrix.
                On output, A is replace by the upper right triangluar matrix R.

      Q         (input/output) The parameter Q is multiplied by the unitary
                matrix Q arising from the Schur decomposition. Q can also be
                false, in which case the unitary matrix Q is not computated.
    """

    n = A.rows

    norm = 0
    for x in xrange(n):
        for y in xrange(min(x+2, n)):
            norm += ctx.re(A[y,x]) ** 2 + ctx.im(A[y,x]) ** 2
    norm = ctx.sqrt(norm) / n

    if norm == 0:
        return

    n0 = 0
    n1 = n

    eps = ctx.eps / (100 * n)
    maxits = ctx.dps * 4

    its = totalits = 0

    while 1:
        # kressner p.32 algo 3
        # the active submatrix is A[n0:n1,n0:n1]

        k = n0

        while k + 1 < n1:
            s = abs(ctx.re(A[k,k])) + abs(ctx.im(A[k,k])) + abs(ctx.re(A[k+1,k+1])) + abs(ctx.im(A[k+1,k+1]))
            if s < eps * norm:
                s = norm
            if abs(A[k+1,k]) < eps * s:
                break
            k += 1

        if k + 1 < n1:
            # deflation found at position (k+1, k)

            A[k+1,k] = 0
            n0 = k + 1

            its = 0

            if n0 + 1 >= n1:
                # block of size at most two has converged
                n0 = 0
                n1 = k + 1
                if n1 < 2:
                    # QR algorithm has converged
                    return
        else:
            if (its % 30) == 10:
                # exceptional shift
                shift = A[n1-1,n1-2]
            elif (its % 30) == 20:
                # exceptional shift
                shift = abs(A[n1-1,n1-2])
            elif (its % 30) == 29:
                # exceptional shift
                shift = norm
            else:
                #    A = [ a b ]       det(x-A)=x*x-x*tr(A)+det(A)
                #        [ c d ]
                #
                # eigenvalues bad:   (tr(A)+sqrt((tr(A))**2-4*det(A)))/2
                #     bad because of cancellation if |c| is small and |a-d| is small, too.
                #
                # eigenvalues good:     (a+d+sqrt((a-d)**2+4*b*c))/2

                t =  A[n1-2,n1-2] + A[n1-1,n1-1]
                s = (A[n1-1,n1-1] - A[n1-2,n1-2]) ** 2 + 4 * A[n1-1,n1-2] * A[n1-2,n1-1]
                if ctx.re(s) > 0:
                    s = ctx.sqrt(s)
                else:
                    s = ctx.sqrt(-s) * 1j
                a = (t + s) / 2
                b = (t - s) / 2
                if abs(A[n1-1,n1-1] - a) > abs(A[n1-1,n1-1] - b):
                    shift = b
                else:
                    shift = a

            its += 1
            totalits += 1

            qr_step(ctx, n0, n1, A, Q, shift)

            if its > maxits:
                raise RuntimeError("qr: failed to converge after %d steps" % its)


@defun
def schur(ctx, A, overwrite_a = False):
    """
    This routine computes the Schur decomposition of a square matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = R                and               Q' Q = Q Q' = 1

    where R is an upper right triangular matrix. Here ' denotes the
    hermitian transpose (i.e. transposition and conjugation).

    input:
      A            : a real or complex square matrix
      overwrite_a  : if true, allows modification of A which may improve
                     performance. if false, A is not modified.

    output:
      Q : an unitary matrix
      R : an upper right triangular matrix

    return value:   (Q, R)

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> Q, R = mp.schur(A)
      >>> mp.nprint(R, 3) # doctest:+SKIP
      [2.0  0.417  -2.53]
      [0.0    4.0  -4.74]
      [0.0    0.0    9.0]
      >>> print(mp.chop(A - Q * R * Q.transpose_conj()))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    warning: The Schur decomposition is not unique.
    """

    n = A.rows

    if n == 1:
        return (ctx.matrix([[1]]), A)

    if not overwrite_a:
        A = A.copy()

    T = ctx.matrix(n, 1)

    hessenberg_reduce_0(ctx, A, T)
    Q = A.copy()
    hessenberg_reduce_1(ctx, Q, T)

    for x in xrange(n):
        for y in xrange(x + 2, n):
            A[y,x] = 0

    hessenberg_qr(ctx, A, Q)

    return Q, A


def eig_tr_r(ctx, A):
    """
    This routine calculates the right eigenvectors of an upper right triangular matrix.

    input:
      A      an upper right triangular matrix

    output:
      ER     a matrix whose columns form the right eigenvectors of A

    return value: ER
    """

    # this subroutine is inspired by the lapack routines ctrevc.f,clatrs.f

    n = A.rows

    ER = ctx.eye(n)

    eps = ctx.eps

    unfl = ctx.ldexp(ctx.one, -ctx.prec * 30)
    # since mpmath effectively has no limits on the exponent, we simply scale doubles up
    # original double has prec*20

    smlnum = unfl * (n / eps)
    simin = 1 / ctx.sqrt(eps)

    rmax = 1

    for i in xrange(1, n):
        s = A[i,i]

        smin = max(eps * abs(s), smlnum)

        for j in xrange(i - 1, -1, -1):

            r = 0
            for k in xrange(j + 1, i + 1):
                r += A[j,k] * ER[k,i]

            t = A[j,j] - s
            if abs(t) < smin:
                t = smin

            r = -r / t
            ER[j,i] = r

            rmax = max(rmax, abs(r))
            if rmax > simin:
                for k in xrange(j, i+1):
                    ER[k,i] /= rmax
                rmax = 1

        if rmax != 1:
            for k in xrange(0, i + 1):
                ER[k,i] /= rmax

    return ER

def eig_tr_l(ctx, A):
    """
    This routine calculates the left eigenvectors of an upper right triangular matrix.

    input:
      A      an upper right triangular matrix

    output:
      EL     a matrix whose rows form the left eigenvectors of A

    return value:  EL
    """

    n = A.rows

    EL = ctx.eye(n)

    eps = ctx.eps

    unfl = ctx.ldexp(ctx.one, -ctx.prec * 30)
    # since mpmath effectively has no limits on the exponent, we simply scale doubles up
    # original double has prec*20

    smlnum = unfl * (n / eps)
    simin = 1 / ctx.sqrt(eps)

    rmax = 1

    for i in xrange(0, n - 1):
        s = A[i,i]

        smin = max(eps * abs(s), smlnum)

        for j in xrange(i + 1, n):

            r = 0
            for k in xrange(i, j):
                r += EL[i,k] * A[k,j]

            t = A[j,j] - s
            if abs(t) < smin:
                t = smin

            r = -r / t
            EL[i,j] = r

            rmax = max(rmax, abs(r))
            if rmax > simin:
                for k in xrange(i, j + 1):
                    EL[i,k] /= rmax
                rmax = 1

        if rmax != 1:
            for k in xrange(i, n):
                EL[i,k] /= rmax

    return EL

@defun
def eig(ctx, A, left = False, right = True, overwrite_a = False):
    """
    This routine computes the eigenvalues and optionally the left and right
    eigenvectors of a square matrix A. Given A, a vector E and matrices ER
    and EL are calculated such that

                        A ER[:,i] =         E[i] ER[:,i]
                EL[i,:] A         = EL[i,:] E[i]

    E contains the eigenvalues of A. The columns of ER contain the right eigenvectors
    of A whereas the rows of EL contain the left eigenvectors.


    input:
      A           : a real or complex square matrix of shape (n, n)
      left        : if true, the left eigenvectors are calculated.
      right       : if true, the right eigenvectors are calculated.
      overwrite_a : if true, allows modification of A which may improve
                    performance. if false, A is not modified.

    output:
      E    : a list of length n containing the eigenvalues of A.
      ER   : a matrix whose columns contain the right eigenvectors of A.
      EL   : a matrix whose rows contain the left eigenvectors of A.

    return values:
       E            if left and right are both false.
      (E, ER)       if right is true and left is false.
      (E, EL)       if left is true and right is false.
      (E, EL, ER)   if left and right are true.


    examples:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> E, ER = mp.eig(A)
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]

      >>> E, EL, ER = mp.eig(A,left = True, right = True)
      >>> E, EL, ER = mp.eig_sort(E, EL, ER)
      >>> mp.nprint(E)
      [2.0, 4.0, 9.0]
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]
      >>> print(mp.chop( EL[0,:] * A - EL[0,:] * E[0]))
      [0.0  0.0  0.0]

    warning:
     - If there are multiple eigenvalues, the eigenvectors do not necessarily
       span the whole vectorspace, i.e. ER and EL may have not full rank.
       Furthermore in that case the eigenvectors are numerical ill-conditioned.
     - In the general case the eigenvalues have no natural order.

    see also:
      - eigh (or eigsy, eighe) for the symmetric eigenvalue problem.
      - eig_sort for sorting of eigenvalues and eigenvectors
    """

    n = A.rows

    if n == 1:
        if left and (not right):
            return ([A[0]], ctx.matrix([[1]]))

        if right and (not left):
            return ([A[0]], ctx.matrix([[1]]))

        return ([A[0]], ctx.matrix([[1]]), ctx.matrix([[1]]))

    if not overwrite_a:
        A = A.copy()

    T = ctx.zeros(n, 1)

    hessenberg_reduce_0(ctx, A, T)

    if left or right:
        Q = A.copy()
        hessenberg_reduce_1(ctx, Q, T)
    else:
        Q = False

    for x in xrange(n):
        for y in xrange(x + 2, n):
            A[y,x] = 0

    hessenberg_qr(ctx, A, Q)

    E = [0 for i in xrange(n)]
    for i in xrange(n):
        E[i] = A[i,i]

    if not (left or right):
        return E

    if left:
        EL = eig_tr_l(ctx, A)
        EL = EL * Q.transpose_conj()

    if right:
        ER = eig_tr_r(ctx, A)
        ER = Q * ER

    if left and (not right):
        return (E, EL)

    if right and (not left):
        return (E, ER)

    return (E, EL, ER)

@defun
def eig_sort(ctx, E, EL = False, ER = False, f = "real"):
    """
    This routine sorts the eigenvalues and eigenvectors delivered by ``eig``.

    parameters:
      E  : the eigenvalues as delivered by eig
      EL : the left  eigenvectors as delivered by eig, or false
      ER : the right eigenvectors as delivered by eig, or false
      f  : either a string ("real" sort by increasing real part, "imag" sort by
           increasing imag part, "abs" sort by absolute value) or a function
           mapping complexs to the reals, i.e. ``f = lambda x: -mp.re(x) ``
           would sort the eigenvalues by decreasing real part.

    return values:
       E            if EL and ER are both false.
      (E, ER)       if ER is not false and left is false.
      (E, EL)       if EL is not false and right is false.
      (E, EL, ER)   if EL and ER are not false.

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> E, EL, ER = mp.eig(A,left = True, right = True)
      >>> E, EL, ER = mp.eig_sort(E, EL, ER)
      >>> mp.nprint(E)
      [2.0, 4.0, 9.0]
      >>> E, EL, ER = mp.eig_sort(E, EL, ER,f = lambda x: -mp.re(x))
      >>> mp.nprint(E)
      [9.0, 4.0, 2.0]
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]
      >>> print(mp.chop( EL[0,:] * A - EL[0,:] * E[0]))
      [0.0  0.0  0.0]
    """

    if isinstance(f, str):
        if f == "real":
            f = ctx.re
        elif f == "imag":
            f = ctx.im
        elif f == "abs":
            f = abs
        else:
            raise RuntimeError("unknown function %s" % f)

    n = len(E)

    # Sort eigenvalues (bubble-sort)

    for i in xrange(n):
        imax = i
        s = f(E[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = f(E[j])
            if c < s:
                s = c
                imax = j

        if imax != i:
            # swap eigenvalues

            z = E[i]
            E[i] = E[imax]
            E[imax] = z

            if not isinstance(EL, bool):
                for j in xrange(n):
                    z = EL[i,j]
                    EL[i,j] = EL[imax,j]
                    EL[imax,j] = z

            if not isinstance(ER, bool):
                for j in xrange(n):
                    z = ER[j,i]
                    ER[j,i] = ER[j,imax]
                    ER[j,imax] = z

    if isinstance(EL, bool) and isinstance(ER, bool):
        return E

    if isinstance(EL, bool) and not(isinstance(ER, bool)):
        return (E, ER)

    if isinstance(ER, bool) and not(isinstance(EL, bool)):
        return (E, EL)

    return (E, EL, ER)