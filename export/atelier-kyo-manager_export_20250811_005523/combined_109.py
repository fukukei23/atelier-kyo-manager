
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\adapters.py ===
"""
requests.adapters
~~~~~~~~~~~~~~~~~

This module contains the transport adapters that Requests uses to define
and maintain connections.
"""

import os.path
import socket  # noqa: F401
import typing
import warnings

from urllib3.exceptions import ClosedPoolError, ConnectTimeoutError
from urllib3.exceptions import HTTPError as _HTTPError
from urllib3.exceptions import InvalidHeader as _InvalidHeader
from urllib3.exceptions import (
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
)
from urllib3.exceptions import ProxyError as _ProxyError
from urllib3.exceptions import ReadTimeoutError, ResponseError
from urllib3.exceptions import SSLError as _SSLError
from urllib3.poolmanager import PoolManager, proxy_from_url
from urllib3.util import Timeout as TimeoutSauce
from urllib3.util import parse_url
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context

from .auth import _basic_auth_str
from .compat import basestring, urlparse
from .cookies import extract_cookies_to_jar
from .exceptions import (
    ConnectionError,
    ConnectTimeout,
    InvalidHeader,
    InvalidProxyURL,
    InvalidSchema,
    InvalidURL,
    ProxyError,
    ReadTimeout,
    RetryError,
    SSLError,
)
from .models import Response
from .structures import CaseInsensitiveDict
from .utils import (
    DEFAULT_CA_BUNDLE_PATH,
    extract_zipped_paths,
    get_auth_from_url,
    get_encoding_from_headers,
    prepend_scheme_if_needed,
    select_proxy,
    urldefragauth,
)

try:
    from urllib3.contrib.socks import SOCKSProxyManager
except ImportError:

    def SOCKSProxyManager(*args, **kwargs):
        raise InvalidSchema("Missing dependencies for SOCKS support.")


if typing.TYPE_CHECKING:
    from .models import PreparedRequest


DEFAULT_POOLBLOCK = False
DEFAULT_POOLSIZE = 10
DEFAULT_RETRIES = 0
DEFAULT_POOL_TIMEOUT = None


try:
    import ssl  # noqa: F401

    _preloaded_ssl_context = create_urllib3_context()
    _preloaded_ssl_context.load_verify_locations(
        extract_zipped_paths(DEFAULT_CA_BUNDLE_PATH)
    )
except ImportError:
    # Bypass default SSLContext creation when Python
    # interpreter isn't built with the ssl module.
    _preloaded_ssl_context = None


def _urllib3_request_context(
    request: "PreparedRequest",
    verify: "bool | str | None",
    client_cert: "typing.Tuple[str, str] | str | None",
    poolmanager: "PoolManager",
) -> "(typing.Dict[str, typing.Any], typing.Dict[str, typing.Any])":
    host_params = {}
    pool_kwargs = {}
    parsed_request_url = urlparse(request.url)
    scheme = parsed_request_url.scheme.lower()
    port = parsed_request_url.port

    # Determine if we have and should use our default SSLContext
    # to optimize performance on standard requests.
    poolmanager_kwargs = getattr(poolmanager, "connection_pool_kw", {})
    has_poolmanager_ssl_context = poolmanager_kwargs.get("ssl_context")
    should_use_default_ssl_context = (
        _preloaded_ssl_context is not None and not has_poolmanager_ssl_context
    )

    cert_reqs = "CERT_REQUIRED"
    if verify is False:
        cert_reqs = "CERT_NONE"
    elif verify is True and should_use_default_ssl_context:
        pool_kwargs["ssl_context"] = _preloaded_ssl_context
    elif isinstance(verify, str):
        if not os.path.isdir(verify):
            pool_kwargs["ca_certs"] = verify
        else:
            pool_kwargs["ca_cert_dir"] = verify
    pool_kwargs["cert_reqs"] = cert_reqs
    if client_cert is not None:
        if isinstance(client_cert, tuple) and len(client_cert) == 2:
            pool_kwargs["cert_file"] = client_cert[0]
            pool_kwargs["key_file"] = client_cert[1]
        else:
            # According to our docs, we allow users to specify just the client
            # cert path
            pool_kwargs["cert_file"] = client_cert
    host_params = {
        "scheme": scheme,
        "host": parsed_request_url.hostname,
        "port": port,
    }
    return host_params, pool_kwargs


class BaseAdapter:
    """The Base Transport Adapter"""

    def __init__(self):
        super().__init__()

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        """
        raise NotImplementedError

    def close(self):
        """Cleans up adapter specific items."""
        raise NotImplementedError


class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3.

    Provides a general-case interface for Requests sessions to contact HTTP and
    HTTPS urls by implementing the Transport Adapter interface. This class will
    usually be created by the :class:`Session <Session>` class under the
    covers.

    :param pool_connections: The number of urllib3 connection pools to cache.
    :param pool_maxsize: The maximum number of connections to save in the pool.
    :param max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups, socket
        connections and connection timeouts, never to requests where data has
        made it to the server. By default, Requests does not retry failed
        connections. If you need granular control over the conditions under
        which we retry a request, import urllib3's ``Retry`` class and pass
        that instead.
    :param pool_block: Whether the connection pool should block for connections.

    Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> a = requests.adapters.HTTPAdapter(max_retries=3)
      >>> s.mount('http://', a)
    """

    __attrs__ = [
        "max_retries",
        "config",
        "_pool_connections",
        "_pool_maxsize",
        "_pool_block",
    ]

    def __init__(
        self,
        pool_connections=DEFAULT_POOLSIZE,
        pool_maxsize=DEFAULT_POOLSIZE,
        max_retries=DEFAULT_RETRIES,
        pool_block=DEFAULT_POOLBLOCK,
    ):
        if max_retries == DEFAULT_RETRIES:
            self.max_retries = Retry(0, read=False)
        else:
            self.max_retries = Retry.from_int(max_retries)
        self.config = {}
        self.proxy_manager = {}

        super().__init__()

        self._pool_connections = pool_connections
        self._pool_maxsize = pool_maxsize
        self._pool_block = pool_block

        self.init_poolmanager(pool_connections, pool_maxsize, block=pool_block)

    def __getstate__(self):
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        # Can't handle by adding 'proxy_manager' to self.__attrs__ because
        # self.poolmanager uses a lambda function, which isn't pickleable.
        self.proxy_manager = {}
        self.config = {}

        for attr, value in state.items():
            setattr(self, attr, value)

        self.init_poolmanager(
            self._pool_connections, self._pool_maxsize, block=self._pool_block
        )

    def init_poolmanager(
        self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs
    ):
        """Initializes a urllib3 PoolManager.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param connections: The number of urllib3 connection pools to cache.
        :param maxsize: The maximum number of connections to save in the pool.
        :param block: Block when no free connections are available.
        :param pool_kwargs: Extra keyword arguments used to initialize the Pool Manager.
        """
        # save these values for pickling
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Return urllib3 ProxyManager for the given proxy.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The proxy to return a urllib3 ProxyManager for.
        :param proxy_kwargs: Extra keyword arguments used to configure the Proxy Manager.
        :returns: ProxyManager
        :rtype: urllib3.ProxyManager
        """
        if proxy in self.proxy_manager:
            manager = self.proxy_manager[proxy]
        elif proxy.lower().startswith("socks"):
            username, password = get_auth_from_url(proxy)
            manager = self.proxy_manager[proxy] = SOCKSProxyManager(
                proxy,
                username=username,
                password=password,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )
        else:
            proxy_headers = self.proxy_headers(proxy)
            manager = self.proxy_manager[proxy] = proxy_from_url(
                proxy,
                proxy_headers=proxy_headers,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )

        return manager

    def cert_verify(self, conn, url, verify, cert):
        """Verify a SSL certificate. This method should not be called from user
        code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param conn: The urllib3 connection object associated with the cert.
        :param url: The requested URL.
        :param verify: Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: The SSL certificate to verify.
        """
        if url.lower().startswith("https") and verify:
            conn.cert_reqs = "CERT_REQUIRED"

            # Only load the CA certificates if 'verify' is a string indicating the CA bundle to use.
            # Otherwise, if verify is a boolean, we don't load anything since
            # the connection will be using a context with the default certificates already loaded,
            # and this avoids a call to the slow load_verify_locations()
            if verify is not True:
                # `verify` must be a str with a path then
                cert_loc = verify

                if not os.path.exists(cert_loc):
                    raise OSError(
                        f"Could not find a suitable TLS CA certificate bundle, "
                        f"invalid path: {cert_loc}"
                    )

                if not os.path.isdir(cert_loc):
                    conn.ca_certs = cert_loc
                else:
                    conn.ca_cert_dir = cert_loc
        else:
            conn.cert_reqs = "CERT_NONE"
            conn.ca_certs = None
            conn.ca_cert_dir = None

        if cert:
            if not isinstance(cert, basestring):
                conn.cert_file = cert[0]
                conn.key_file = cert[1]
            else:
                conn.cert_file = cert
                conn.key_file = None
            if conn.cert_file and not os.path.exists(conn.cert_file):
                raise OSError(
                    f"Could not find the TLS certificate file, "
                    f"invalid path: {conn.cert_file}"
                )
            if conn.key_file and not os.path.exists(conn.key_file):
                raise OSError(
                    f"Could not find the TLS key file, invalid path: {conn.key_file}"
                )

    def build_response(self, req, resp):
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response. This should not be called from user code, and is only exposed
        for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`

        :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
        :param resp: The urllib3 response object.
        :rtype: requests.Response
        """
        response = Response()

        # Fallback to None if there's no status_code, for whatever reason.
        response.status_code = getattr(resp, "status", None)

        # Make headers case-insensitive.
        response.headers = CaseInsensitiveDict(getattr(resp, "headers", {}))

        # Set encoding.
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp
        response.reason = response.raw.reason

        if isinstance(req.url, bytes):
            response.url = req.url.decode("utf-8")
        else:
            response.url = req.url

        # Add new cookies from the server.
        extract_cookies_to_jar(response.cookies, req, resp)

        # Give the Response some context.
        response.request = req
        response.connection = self

        return response

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        """Build the PoolKey attributes used by urllib3 to return a connection.

        This looks at the PreparedRequest, the user-specified verify value,
        and the value of the cert parameter to determine what PoolKey values
        to use to select a connection from a given urllib3 Connection Pool.

        The SSL related pool key arguments are not consistently set. As of
        this writing, use the following to determine what keys may be in that
        dictionary:

        * If ``verify`` is ``True``, ``"ssl_context"`` will be set and will be the
          default Requests SSL Context
        * If ``verify`` is ``False``, ``"ssl_context"`` will not be set but
          ``"cert_reqs"`` will be set
        * If ``verify`` is a string, (i.e., it is a user-specified trust bundle)
          ``"ca_certs"`` will be set if the string is not a directory recognized
          by :py:func:`os.path.isdir`, otherwise ``"ca_certs_dir"`` will be
          set.
        * If ``"cert"`` is specified, ``"cert_file"`` will always be set. If
          ``"cert"`` is a tuple with a second item, ``"key_file"`` will also
          be present

        To override these settings, one may subclass this class, call this
        method and use the above logic to change parameters as desired. For
        example, if one wishes to use a custom :py:class:`ssl.SSLContext` one
        must both set ``"ssl_context"`` and based on what else they require,
        alter the other keys to ensure the desired behaviour.

        :param request:
            The PreparedReqest being sent over the connection.
        :type request:
            :class:`~requests.models.PreparedRequest`
        :param verify:
            Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use.
        :param cert:
            (optional) Any user-provided SSL certificate for client
            authentication (a.k.a., mTLS). This may be a string (i.e., just
            the path to a file which holds both certificate and key) or a
            tuple of length 2 with the certificate file path and key file
            path.
        :returns:
            A tuple of two dictionaries. The first is the "host parameters"
            portion of the Pool Key including scheme, hostname, and port. The
            second is a dictionary of SSLContext related parameters.
        """
        return _urllib3_request_context(request, verify, cert, self.poolmanager)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        """Returns a urllib3 connection for the given request and TLS settings.
        This should not be called from user code, and is only exposed for use
        when subclassing the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request:
            The :class:`PreparedRequest <PreparedRequest>` object to be sent
            over the connection.
        :param verify:
            Either a boolean, in which case it controls whether we verify the
            server's TLS certificate, or a string, in which case it must be a
            path to a CA bundle to use.
        :param proxies:
            (optional) The proxies dictionary to apply to the request.
        :param cert:
            (optional) Any user-provided SSL certificate to be used for client
            authentication (a.k.a., mTLS).
        :rtype:
            urllib3.ConnectionPool
        """
        proxy = select_proxy(request.url, proxies)
        try:
            host_params, pool_kwargs = self.build_connection_pool_key_attributes(
                request,
                verify,
                cert,
            )
        except ValueError as e:
            raise InvalidURL(e, request=request)
        if proxy:
            proxy = prepend_scheme_if_needed(proxy, "http")
            proxy_url = parse_url(proxy)
            if not proxy_url.host:
                raise InvalidProxyURL(
                    "Please check proxy URL. It is malformed "
                    "and could be missing the host."
                )
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_host(
                **host_params, pool_kwargs=pool_kwargs
            )
        else:
            # Only scheme should be lower case
            conn = self.poolmanager.connection_from_host(
                **host_params, pool_kwargs=pool_kwargs
            )

        return conn

    def get_connection(self, url, proxies=None):
        """DEPRECATED: Users should move to `get_connection_with_tls_context`
        for all subclasses of HTTPAdapter using Requests>=2.32.2.

        Returns a urllib3 connection for the given URL. This should not be
        called from user code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param url: The URL to connect to.
        :param proxies: (optional) A Requests-style dictionary of proxies used on this request.
        :rtype: urllib3.ConnectionPool
        """
        warnings.warn(
            (
                "`get_connection` has been deprecated in favor of "
                "`get_connection_with_tls_context`. Custom HTTPAdapter subclasses "
                "will need to migrate for Requests>=2.32.2. Please see "
                "https://github.com/psf/requests/pull/6710 for more details."
            ),
            DeprecationWarning,
        )
        proxy = select_proxy(url, proxies)

        if proxy:
            proxy = prepend_scheme_if_needed(proxy, "http")
            proxy_url = parse_url(proxy)
            if not proxy_url.host:
                raise InvalidProxyURL(
                    "Please check proxy URL. It is malformed "
                    "and could be missing the host."
                )
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_url(url)
        else:
            # Only scheme should be lower case
            parsed = urlparse(url)
            url = parsed.geturl()
            conn = self.poolmanager.connection_from_url(url)

        return conn

    def close(self):
        """Disposes of any internal state.

        Currently, this closes the PoolManager and any active ProxyManager,
        which closes any pooled connections.
        """
        self.poolmanager.clear()
        for proxy in self.proxy_manager.values():
            proxy.clear()

    def request_url(self, request, proxies):
        """Obtain the url to use when making the final request.

        If the message is being sent through a HTTP proxy, the full URL has to
        be used. Otherwise, we should only use the path portion of the URL.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs.
        :rtype: str
        """
        proxy = select_proxy(request.url, proxies)
        scheme = urlparse(request.url).scheme

        is_proxied_http_request = proxy and scheme != "https"
        using_socks_proxy = False
        if proxy:
            proxy_scheme = urlparse(proxy).scheme.lower()
            using_socks_proxy = proxy_scheme.startswith("socks")

        url = request.path_url
        if url.startswith("//"):  # Don't confuse urllib3
            url = f"/{url.lstrip('/')}"

        if is_proxied_http_request and not using_socks_proxy:
            url = urldefragauth(request.url)

        return url

    def add_headers(self, request, **kwargs):
        """Add any headers needed by the connection. As of v2.0 this does
        nothing by default, but is left for overriding by users that subclass
        the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` to add headers to.
        :param kwargs: The keyword arguments from the call to send().
        """
        pass

    def proxy_headers(self, proxy):
        """Returns a dictionary of the headers to add to any request sent
        through a proxy. This works with urllib3 magic to ensure that they are
        correctly sent to the proxy, rather than in a tunnelled request if
        CONNECT is being used.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The url of the proxy being used for this request.
        :rtype: dict
        """
        headers = {}
        username, password = get_auth_from_url(proxy)

        if username:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return headers

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple or urllib3 Timeout object
        :param verify: (optional) Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        :rtype: requests.Response
        """

        try:
            conn = self.get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
        except LocationValueError as e:
            raise InvalidURL(e, request=request)

        self.cert_verify(conn, request.url, verify, cert)
        url = self.request_url(request, proxies)
        self.add_headers(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )

        chunked = not (request.body is None or "Content-Length" in request.headers)

        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
                timeout = TimeoutSauce(connect=connect, read=read)
            except ValueError:
                raise ValueError(
                    f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "
                    f"or a single float to set both timeouts to the same value."
                )
        elif isinstance(timeout, TimeoutSauce):
            pass
        else:
            timeout = TimeoutSauce(connect=timeout, read=timeout)

        try:
            resp = conn.urlopen(
                method=request.method,
                url=url,
                body=request.body,
                headers=request.headers,
                redirect=False,
                assert_same_host=False,
                preload_content=False,
                decode_content=False,
                retries=self.max_retries,
                timeout=timeout,
                chunked=chunked,
            )

        except (ProtocolError, OSError) as err:
            raise ConnectionError(err, request=request)

        except MaxRetryError as e:
            if isinstance(e.reason, ConnectTimeoutError):
                # TODO: Remove this in 3.0.0: see #2811
                if not isinstance(e.reason, NewConnectionError):
                    raise ConnectTimeout(e, request=request)

            if isinstance(e.reason, ResponseError):
                raise RetryError(e, request=request)

            if isinstance(e.reason, _ProxyError):
                raise ProxyError(e, request=request)

            if isinstance(e.reason, _SSLError):
                # This branch is for urllib3 v1.22 and later.
                raise SSLError(e, request=request)

            raise ConnectionError(e, request=request)

        except ClosedPoolError as e:
            raise ConnectionError(e, request=request)

        except _ProxyError as e:
            raise ProxyError(e)

        except (_SSLError, _HTTPError) as e:
            if isinstance(e, _SSLError):
                # This branch is for urllib3 versions earlier than v1.22
                raise SSLError(e, request=request)
            elif isinstance(e, ReadTimeoutError):
                raise ReadTimeout(e, request=request)
            elif isinstance(e, _InvalidHeader):
                raise InvalidHeader(e, request=request)
            else:
                raise

        return self.build_response(request, resp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\powsimp.py ===
from collections import defaultdict
from functools import reduce
from math import prod

from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational, equal_valued
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity



def powsimp(expr, deep=False, combine='all', force=False, measure=count_ops):
    """
    Reduce expression by combining powers with similar bases and exponents.

    Explanation
    ===========

    If ``deep`` is ``True`` then powsimp() will also simplify arguments of
    functions. By default ``deep`` is set to ``False``.

    If ``force`` is ``True`` then bases will be combined without checking for
    assumptions, e.g. sqrt(x)*sqrt(y) -> sqrt(x*y) which is not true
    if x and y are both negative.

    You can make powsimp() only combine bases or only combine exponents by
    changing combine='base' or combine='exp'.  By default, combine='all',
    which does both.  combine='base' will only combine::

         a   a          a                          2x      x
        x * y  =>  (x*y)   as well as things like 2   =>  4

    and combine='exp' will only combine
    ::

         a   b      (a + b)
        x * x  =>  x

    combine='exp' will strictly only combine exponents in the way that used
    to be automatic.  Also use deep=True if you need the old behavior.

    When combine='all', 'exp' is evaluated first.  Consider the first
    example below for when there could be an ambiguity relating to this.
    This is done so things like the second example can be completely
    combined.  If you want 'base' combined first, do something like
    powsimp(powsimp(expr, combine='base'), combine='exp').

    Examples
    ========

    >>> from sympy import powsimp, exp, log, symbols
    >>> from sympy.abc import x, y, z, n
    >>> powsimp(x**y*x**z*y**z, combine='all')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='exp')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='base', force=True)
    x**y*(x*y)**z

    >>> powsimp(x**z*x**y*n**z*n**y, combine='all', force=True)
    (n*x)**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='exp')
    n**(y + z)*x**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='base', force=True)
    (n*x)**y*(n*x)**z

    >>> x, y = symbols('x y', positive=True)
    >>> powsimp(log(exp(x)*exp(y)))
    log(exp(x)*exp(y))
    >>> powsimp(log(exp(x)*exp(y)), deep=True)
    x + y

    Radicals with Mul bases will be combined if combine='exp'

    >>> from sympy import sqrt
    >>> x, y = symbols('x y')

    Two radicals are automatically joined through Mul:

    >>> a=sqrt(x*sqrt(y))
    >>> a*a**3 == a**4
    True

    But if an integer power of that radical has been
    autoexpanded then Mul does not join the resulting factors:

    >>> a**4 # auto expands to a Mul, no longer a Pow
    x**2*y
    >>> _*a # so Mul doesn't combine them
    x**2*y*sqrt(x*sqrt(y))
    >>> powsimp(_) # but powsimp will
    (x*sqrt(y))**(5/2)
    >>> powsimp(x*y*a) # but won't when doing so would violate assumptions
    x*y*sqrt(x*sqrt(y))

    """
    def recurse(arg, **kwargs):
        _deep = kwargs.get('deep', deep)
        _combine = kwargs.get('combine', combine)
        _force = kwargs.get('force', force)
        _measure = kwargs.get('measure', measure)
        return powsimp(arg, _deep, _combine, _force, _measure)

    expr = sympify(expr)

    if (not isinstance(expr, Basic) or isinstance(expr, MatrixSymbol) or (
            expr.is_Atom or expr in (exp_polar(0), exp_polar(1)))):
        return expr

    if deep or expr.is_Add or expr.is_Mul and _y not in expr.args:
        expr = expr.func(*[recurse(w) for w in expr.args])

    if expr.is_Pow:
        return recurse(expr*_y, deep=False)/_y

    if not expr.is_Mul:
        return expr

    # handle the Mul
    if combine in ('exp', 'all'):
        # Collect base/exp data, while maintaining order in the
        # non-commutative parts of the product
        c_powers = defaultdict(list)
        nc_part = []
        newexpr = []
        coeff = S.One
        for term in expr.args:
            if term.is_Rational:
                coeff *= term
                continue
            if term.is_Pow:
                term = _denest_pow(term)
            if term.is_commutative:
                b, e = term.as_base_exp()
                if deep:
                    b, e = [recurse(i) for i in [b, e]]
                if b.is_Pow or isinstance(b, exp):
                    # don't let smthg like sqrt(x**a) split into x**a, 1/2
                    # or else it will be joined as x**(a/2) later
                    b, e = b**e, S.One
                c_powers[b].append(e)
            else:
                # This is the logic that combines exponents for equal,
                # but non-commutative bases: A**x*A**y == A**(x+y).
                if nc_part:
                    b1, e1 = nc_part[-1].as_base_exp()
                    b2, e2 = term.as_base_exp()
                    if (b1 == b2 and
                            e1.is_commutative and e2.is_commutative):
                        nc_part[-1] = Pow(b1, Add(e1, e2))
                        continue
                nc_part.append(term)

        # add up exponents of common bases
        for b, e in ordered(iter(c_powers.items())):
            # allow 2**x/4 -> 2**(x - 2); don't do this when b and e are
            # Numbers since autoevaluation will undo it, e.g.
            # 2**(1/3)/4 -> 2**(1/3 - 2) -> 2**(1/3)/4
            if (b and b.is_Rational and not all(ei.is_Number for ei in e) and \
                    coeff is not S.One and
                    b not in (S.One, S.NegativeOne)):
                m = multiplicity(abs(b), abs(coeff))
                if m:
                    e.append(m)
                    coeff /= b**m
            c_powers[b] = Add(*e)
        if coeff is not S.One:
            if coeff in c_powers:
                c_powers[coeff] += S.One
            else:
                c_powers[coeff] = S.One

        # convert to plain dictionary
        c_powers = dict(c_powers)

        # check for base and inverted base pairs
        be = list(c_powers.items())
        skip = set()  # skip if we already saw them
        for b, e in be:
            if b in skip:
                continue
            bpos = b.is_positive or b.is_polar
            if bpos:
                binv = 1/b
                #Special case for float 1
                if b.is_Float and equal_valued(b, 1):
                    c_powers[b] = S.One
                    continue
                if b != binv and binv in c_powers:
                    if b.as_numer_denom()[0] is S.One:
                        c_powers.pop(b)
                        c_powers[binv] -= e
                    else:
                        skip.add(binv)
                        e = c_powers.pop(binv)
                        c_powers[b] -= e

        # check for base and negated base pairs
        be = list(c_powers.items())
        _n = S.NegativeOne
        for b, e in be:
            if (b.is_Symbol or b.is_Add) and -b in c_powers and b in c_powers:
                if (b.is_positive is not None or e.is_integer):
                    if e.is_integer or b.is_negative:
                        c_powers[-b] += c_powers.pop(b)
                    else:  # (-b).is_positive so use its e
                        e = c_powers.pop(-b)
                        c_powers[b] += e
                    if _n in c_powers:
                        c_powers[_n] += e
                    else:
                        c_powers[_n] = e

        # filter c_powers and convert to a list
        c_powers = [(b, e) for b, e in c_powers.items() if e]

        # ==============================================================
        # check for Mul bases of Rational powers that can be combined with
        # separated bases, e.g. x*sqrt(x*y)*sqrt(x*sqrt(x*y)) ->
        # (x*sqrt(x*y))**(3/2)
        # ---------------- helper functions

        def ratq(x):
            '''Return Rational part of x's exponent as it appears in the bkey.
            '''
            return bkey(x)[0][1]

        def bkey(b, e=None):
            '''Return (b**s, c.q), c.p where e -> c*s. If e is not given then
            it will be taken by using as_base_exp() on the input b.
            e.g.
                x**3/2 -> (x, 2), 3
                x**y -> (x**y, 1), 1
                x**(2*y/3) -> (x**y, 3), 2
                exp(x/2) -> (exp(a), 2), 1

            '''
            if e is not None:  # coming from c_powers or from below
                if e.is_Integer:
                    return (b, S.One), e
                elif e.is_Rational:
                    return (b, Integer(e.q)), Integer(e.p)
                else:
                    c, m = e.as_coeff_Mul(rational=True)
                    if c is not S.One:
                        if m.is_integer:
                            return (b, Integer(c.q)), m*Integer(c.p)
                        return (b**m, Integer(c.q)), Integer(c.p)
                    else:
                        return (b**e, S.One), S.One
            else:
                return bkey(*b.as_base_exp())

        def update(b):
            '''Decide what to do with base, b. If its exponent is now an
            integer multiple of the Rational denominator, then remove it
            and put the factors of its base in the common_b dictionary or
            update the existing bases if necessary. If it has been zeroed
            out, simply remove the base.
            '''
            newe, r = divmod(common_b[b], b[1])
            if not r:
                common_b.pop(b)
                if newe:
                    for m in Mul.make_args(b[0]**newe):
                        b, e = bkey(m)
                        if b not in common_b:
                            common_b[b] = 0
                        common_b[b] += e
                        if b[1] != 1:
                            bases.append(b)
        # ---------------- end of helper functions

        # assemble a dictionary of the factors having a Rational power
        common_b = {}
        done = []
        bases = []
        for b, e in c_powers:
            b, e = bkey(b, e)
            if b in common_b:
                common_b[b] = common_b[b] + e
            else:
                common_b[b] = e
            if b[1] != 1 and b[0].is_Mul:
                bases.append(b)
        bases.sort(key=default_sort_key)  # this makes tie-breaking canonical
        bases.sort(key=measure, reverse=True)  # handle longest first
        for base in bases:
            if base not in common_b:  # it may have been removed already
                continue
            b, exponent = base
            last = False  # True when no factor of base is a radical
            qlcm = 1  # the lcm of the radical denominators
            while True:
                bstart = b
                qstart = qlcm

                bb = []  # list of factors
                ee = []  # (factor's expo. and it's current value in common_b)
                for bi in Mul.make_args(b):
                    bib, bie = bkey(bi)
                    if bib not in common_b or common_b[bib] < bie:
                        ee = bb = []  # failed
                        break
                    ee.append([bie, common_b[bib]])
                    bb.append(bib)
                if ee:
                    # find the number of integral extractions possible
                    # e.g. [(1, 2), (2, 2)] -> min(2/1, 2/2) -> 1
                    min1 = ee[0][1]//ee[0][0]
                    for i in range(1, len(ee)):
                        rat = ee[i][1]//ee[i][0]
                        if rat < 1:
                            break
                        min1 = min(min1, rat)
                    else:
                        # update base factor counts
                        # e.g. if ee = [(2, 5), (3, 6)] then min1 = 2
                        # and the new base counts will be 5-2*2 and 6-2*3
                        for i in range(len(bb)):
                            common_b[bb[i]] -= min1*ee[i][0]
                            update(bb[i])
                        # update the count of the base
                        # e.g. x**2*y*sqrt(x*sqrt(y)) the count of x*sqrt(y)
                        # will increase by 4 to give bkey (x*sqrt(y), 2, 5)
                        common_b[base] += min1*qstart*exponent
                if (last  # no more radicals in base
                    or len(common_b) == 1  # nothing left to join with
                    or all(k[1] == 1 for k in common_b)  # no rad's in common_b
                        ):
                    break
                # see what we can exponentiate base by to remove any radicals
                # so we know what to search for
                # e.g. if base were x**(1/2)*y**(1/3) then we should
                # exponentiate by 6 and look for powers of x and y in the ratio
                # of 2 to 3
                qlcm = lcm([ratq(bi) for bi in Mul.make_args(bstart)])
                if qlcm == 1:
                    break  # we are done
                b = bstart**qlcm
                qlcm *= qstart
                if all(ratq(bi) == 1 for bi in Mul.make_args(b)):
                    last = True  # we are going to be done after this next pass
            # this base no longer can find anything to join with and
            # since it was longer than any other we are done with it
            b, q = base
            done.append((b, common_b.pop(base)*Rational(1, q)))

        # update c_powers and get ready to continue with powsimp
        c_powers = done
        # there may be terms still in common_b that were bases that were
        # identified as needing processing, so remove those, too
        for (b, q), e in common_b.items():
            if (b.is_Pow or isinstance(b, exp)) and \
                    q is not S.One and not b.exp.is_Rational:
                b, be = b.as_base_exp()
                b = b**(be/q)
            else:
                b = root(b, q)
            c_powers.append((b, e))
        check = len(c_powers)
        c_powers = dict(c_powers)
        assert len(c_powers) == check  # there should have been no duplicates
        # ==============================================================

        # rebuild the expression
        newexpr = expr.func(*(newexpr + [Pow(b, e) for b, e in c_powers.items()]))
        if combine == 'exp':
            return expr.func(newexpr, expr.func(*nc_part))
        else:
            return recurse(expr.func(*nc_part), combine='base') * \
                recurse(newexpr, combine='base')

    elif combine == 'base':

        # Build c_powers and nc_part.  These must both be lists not
        # dicts because exp's are not combined.
        c_powers = []
        nc_part = []
        for term in expr.args:
            if term.is_commutative:
                c_powers.append(list(term.as_base_exp()))
            else:
                nc_part.append(term)

        # Pull out numerical coefficients from exponent if assumptions allow
        # e.g., 2**(2*x) => 4**x
        for i in range(len(c_powers)):
            b, e = c_powers[i]
            if not (all(x.is_nonnegative for x in b.as_numer_denom()) or e.is_integer or force or b.is_polar):
                continue
            exp_c, exp_t = e.as_coeff_Mul(rational=True)
            if exp_c is not S.One and exp_t is not S.One:
                c_powers[i] = [Pow(b, exp_c), exp_t]

        # Combine bases whenever they have the same exponent and
        # assumptions allow
        # first gather the potential bases under the common exponent
        c_exp = defaultdict(list)
        for b, e in c_powers:
            if deep:
                e = recurse(e)
            if e.is_Add and (b.is_positive or e.is_integer):
                e = factor_terms(e)
                if _coeff_isneg(e):
                    e = -e
                    b = 1/b
            c_exp[e].append(b)
        del c_powers

        # Merge back in the results of the above to form a new product
        c_powers = defaultdict(list)
        for e in c_exp:
            bases = c_exp[e]

            # calculate the new base for e

            if len(bases) == 1:
                new_base = bases[0]
            elif e.is_integer or force:
                new_base = expr.func(*bases)
            else:
                # see which ones can be joined
                unk = []
                nonneg = []
                neg = []
                for bi in bases:
                    if bi.is_negative:
                        neg.append(bi)
                    elif bi.is_nonnegative:
                        nonneg.append(bi)
                    elif bi.is_polar:
                        nonneg.append(
                            bi)  # polar can be treated like non-negative
                    else:
                        unk.append(bi)
                if len(unk) == 1 and not neg or len(neg) == 1 and not unk:
                    # a single neg or a single unk can join the rest
                    nonneg.extend(unk + neg)
                    unk = neg = []
                elif neg:
                    # their negative signs cancel in groups of 2*q if we know
                    # that e = p/q else we have to treat them as unknown
                    israt = False
                    if e.is_Rational:
                        israt = True
                    else:
                        p, d = e.as_numer_denom()
                        if p.is_integer and d.is_integer:
                            israt = True
                    if israt:
                        neg = [-w for w in neg]
                        unk.extend([S.NegativeOne]*len(neg))
                    else:
                        unk.extend(neg)
                        neg = []
                    del israt

                # these shouldn't be joined
                for b in unk:
                    c_powers[b].append(e)
                # here is a new joined base
                new_base = expr.func(*(nonneg + neg))
                # if there are positive parts they will just get separated
                # again unless some change is made

                def _terms(e):
                    # return the number of terms of this expression
                    # when multiplied out -- assuming no joining of terms
                    if e.is_Add:
                        return sum(_terms(ai) for ai in e.args)
                    if e.is_Mul:
                        return prod([_terms(mi) for mi in e.args])
                    return 1
                xnew_base = expand_mul(new_base, deep=False)
                if len(Add.make_args(xnew_base)) < _terms(new_base):
                    new_base = factor_terms(xnew_base)

            c_powers[new_base].append(e)

        # break out the powers from c_powers now
        c_part = [Pow(b, ei) for b, e in c_powers.items() for ei in e]

        # we're done
        return expr.func(*(c_part + nc_part))

    else:
        raise ValueError("combine must be one of ('all', 'exp', 'base').")


def powdenest(eq, force=False, polar=False):
    r"""
    Collect exponents on powers as assumptions allow.

    Explanation
    ===========

    Given ``(bb**be)**e``, this can be simplified as follows:
        * if ``bb`` is positive, or
        * ``e`` is an integer, or
        * ``|be| < 1`` then this simplifies to ``bb**(be*e)``

    Given a product of powers raised to a power, ``(bb1**be1 *
    bb2**be2...)**e``, simplification can be done as follows:

    - if e is positive, the gcd of all bei can be joined with e;
    - all non-negative bb can be separated from those that are negative
      and their gcd can be joined with e; autosimplification already
      handles this separation.
    - integer factors from powers that have integers in the denominator
      of the exponent can be removed from any term and the gcd of such
      integers can be joined with e

    Setting ``force`` to ``True`` will make symbols that are not explicitly
    negative behave as though they are positive, resulting in more
    denesting.

    Setting ``polar`` to ``True`` will do simplifications on the Riemann surface of
    the logarithm, also resulting in more denestings.

    When there are sums of logs in exp() then a product of powers may be
    obtained e.g. ``exp(3*(log(a) + 2*log(b)))`` - > ``a**3*b**6``.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y, z
    >>> from sympy import Symbol, exp, log, sqrt, symbols, powdenest

    >>> powdenest((x**(2*a/3))**(3*x))
    (x**(2*a/3))**(3*x)
    >>> powdenest(exp(3*x*log(2)))
    2**(3*x)

    Assumptions may prevent expansion:

    >>> powdenest(sqrt(x**2))
    sqrt(x**2)

    >>> p = symbols('p', positive=True)
    >>> powdenest(sqrt(p**2))
    p

    No other expansion is done.

    >>> i, j = symbols('i,j', integer=True)
    >>> powdenest((x**x)**(i + j)) # -X-> (x**x)**i*(x**x)**j
    x**(x*(i + j))

    But exp() will be denested by moving all non-log terms outside of
    the function; this may result in the collapsing of the exp to a power
    with a different base:

    >>> powdenest(exp(3*y*log(x)))
    x**(3*y)
    >>> powdenest(exp(y*(log(a) + log(b))))
    (a*b)**y
    >>> powdenest(exp(3*(log(a) + log(b))))
    a**3*b**3

    If assumptions allow, symbols can also be moved to the outermost exponent:

    >>> i = Symbol('i', integer=True)
    >>> powdenest(((x**(2*i))**(3*y))**x)
    ((x**(2*i))**(3*y))**x
    >>> powdenest(((x**(2*i))**(3*y))**x, force=True)
    x**(6*i*x*y)

    >>> powdenest(((x**(2*a/3))**(3*y/i))**x)
    ((x**(2*a/3))**(3*y/i))**x
    >>> powdenest((x**(2*i)*y**(4*i))**z, force=True)
    (x*y**2)**(2*i*z)

    >>> n = Symbol('n', negative=True)

    >>> powdenest((x**i)**y, force=True)
    x**(i*y)
    >>> powdenest((n**i)**x, force=True)
    (n**i)**x

    """
    from sympy.simplify.simplify import posify

    if force:
        def _denest(b, e):
            if not isinstance(b, (Pow, exp)):
                return b.is_positive, Pow(b, e, evaluate=False)
            return _denest(b.base, b.exp*e)
        reps = []
        for p in eq.atoms(Pow, exp):
            if isinstance(p.base, (Pow, exp)):
                ok, dp = _denest(*p.args)
                if ok is not False:
                    reps.append((p, dp))
        if reps:
            eq = eq.subs(reps)
        eq, reps = posify(eq)
        return powdenest(eq, force=False, polar=polar).xreplace(reps)

    if polar:
        eq, rep = polarify(eq)
        return unpolarify(powdenest(unpolarify(eq, exponents_only=True)), rep)

    new = powsimp(eq)
    return new.xreplace(Transform(
        _denest_pow, filter=lambda m: m.is_Pow or isinstance(m, exp)))

_y = Dummy('y')


def _denest_pow(eq):
    """
    Denest powers.

    This is a helper function for powdenest that performs the actual
    transformation.
    """
    from sympy.simplify.simplify import logcombine

    b, e = eq.as_base_exp()
    if b.is_Pow or isinstance(b, exp) and e != 1:
        new = b._eval_power(e)
        if new is not None:
            eq = new
            b, e = new.as_base_exp()

    # denest exp with log terms in exponent
    if b is S.Exp1 and e.is_Mul:
        logs = []
        other = []
        for ei in e.args:
            if any(isinstance(ai, log) for ai in Add.make_args(ei)):
                logs.append(ei)
            else:
                other.append(ei)
        logs = logcombine(Mul(*logs))
        return Pow(exp(logs), Mul(*other))

    _, be = b.as_base_exp()
    if be is S.One and not (b.is_Mul or
                            b.is_Rational and b.q != 1 or
                            b.is_positive):
        return eq

    # denest eq which is either pos**e or Pow**e or Mul**e or
    # Mul(b1**e1, b2**e2)

    # handle polar numbers specially
    polars, nonpolars = [], []
    for bb in Mul.make_args(b):
        if bb.is_polar:
            polars.append(bb.as_base_exp())
        else:
            nonpolars.append(bb)
    if len(polars) == 1 and not polars[0][0].is_Mul:
        return Pow(polars[0][0], polars[0][1]*e)*powdenest(Mul(*nonpolars)**e)
    elif polars:
        return Mul(*[powdenest(bb**(ee*e)) for (bb, ee) in polars]) \
            *powdenest(Mul(*nonpolars)**e)

    if b.is_Integer:
        # use log to see if there is a power here
        logb = expand_log(log(b))
        if logb.is_Mul:
            c, logb = logb.args
            e *= c
            base = logb.args[0]
            return Pow(base, e)

    # if b is not a Mul or any factor is an atom then there is nothing to do
    if not b.is_Mul or any(s.is_Atom for s in Mul.make_args(b)):
        return eq

    # let log handle the case of the base of the argument being a Mul, e.g.
    # sqrt(x**(2*i)*y**(6*i)) -> x**i*y**(3**i) if x and y are positive; we
    # will take the log, expand it, and then factor out the common powers that
    # now appear as coefficient. We do this manually since terms_gcd pulls out
    # fractions, terms_gcd(x+x*y/2) -> x*(y + 2)/2 and we don't want the 1/2;
    # gcd won't pull out numerators from a fraction: gcd(3*x, 9*x/2) -> x but
    # we want 3*x. Neither work with noncommutatives.

    def nc_gcd(aa, bb):
        a, b = [i.as_coeff_Mul() for i in [aa, bb]]
        c = gcd(a[0], b[0]).as_numer_denom()[0]
        g = Mul(*(a[1].args_cnc(cset=True)[0] & b[1].args_cnc(cset=True)[0]))
        return _keep_coeff(c, g)

    glogb = expand_log(log(b))
    if glogb.is_Add:
        args = glogb.args
        g = reduce(nc_gcd, args)
        if g != 1:
            cg, rg = g.as_coeff_Mul()
            glogb = _keep_coeff(cg, rg*Add(*[a/g for a in args]))

    # now put the log back together again
    if isinstance(glogb, log) or not glogb.is_Mul:
        if glogb.args[0].is_Pow or isinstance(glogb.args[0], exp):
            glogb = _denest_pow(glogb.args[0])
            if (abs(glogb.exp) < 1) == True:
                return Pow(glogb.base, glogb.exp*e)
        return eq

    # the log(b) was a Mul so join any adds with logcombine
    add = []
    other = []
    for a in glogb.args:
        if a.is_Add:
            add.append(a)
        else:
            other.append(a)
    return Pow(exp(logcombine(Mul(*add))), e*Mul(*other))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\text.py ===
# Copyright (c) 2010-2024 openpyxl


from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Alias,
    Typed,
    Set,
    NoneSet,
    Sequence,
    String,
    Bool,
    MinMax,
    Integer
)
from openpyxl.descriptors.excel import (
    HexBinary,
    Coordinate,
    Relation,
)
from openpyxl.descriptors.nested import (
    NestedInteger,
    NestedText,
    NestedValue,
    EmptyTag
)
from openpyxl.xml.constants import DRAWING_NS


from .colors import ColorChoiceDescriptor
from .effect import (
    EffectList,
    EffectContainer,
)
from .fill import(
    GradientFillProperties,
    BlipFillProperties,
    PatternFillProperties,
    Blip
)
from .geometry import (
    LineProperties,
    Color,
    Scene3D
)

from openpyxl.descriptors.excel import ExtensionList as OfficeArtExtensionList
from openpyxl.descriptors.nested import NestedBool


class EmbeddedWAVAudioFile(Serialisable):

    name = String(allow_none=True)

    def __init__(self,
                 name=None,
                ):
        self.name = name


class Hyperlink(Serialisable):

    tagname = "hlinkClick"
    namespace = DRAWING_NS

    invalidUrl = String(allow_none=True)
    action = String(allow_none=True)
    tgtFrame = String(allow_none=True)
    tooltip = String(allow_none=True)
    history = Bool(allow_none=True)
    highlightClick = Bool(allow_none=True)
    endSnd = Bool(allow_none=True)
    snd = Typed(expected_type=EmbeddedWAVAudioFile, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)
    id = Relation(allow_none=True)

    __elements__ = ('snd',)

    def __init__(self,
                 invalidUrl=None,
                 action=None,
                 tgtFrame=None,
                 tooltip=None,
                 history=None,
                 highlightClick=None,
                 endSnd=None,
                 snd=None,
                 extLst=None,
                 id=None,
                ):
        self.invalidUrl = invalidUrl
        self.action = action
        self.tgtFrame = tgtFrame
        self.tooltip = tooltip
        self.history = history
        self.highlightClick = highlightClick
        self.endSnd = endSnd
        self.snd = snd
        self.id = id


class Font(Serialisable):

    tagname = "latin"
    namespace = DRAWING_NS

    typeface = String()
    panose = HexBinary(allow_none=True)
    pitchFamily = MinMax(min=0, max=52, allow_none=True)
    charset = Integer(allow_none=True)

    def __init__(self,
                 typeface=None,
                 panose=None,
                 pitchFamily=None,
                 charset=None,
                ):
        self.typeface = typeface
        self.panose = panose
        self.pitchFamily = pitchFamily
        self.charset = charset


class CharacterProperties(Serialisable):

    tagname = "defRPr"
    namespace = DRAWING_NS

    kumimoji = Bool(allow_none=True)
    lang = String(allow_none=True)
    altLang = String(allow_none=True)
    sz = MinMax(allow_none=True, min=100, max=400000) # 100ths of a point
    b = Bool(allow_none=True)
    i = Bool(allow_none=True)
    u = NoneSet(values=(['words', 'sng', 'dbl', 'heavy', 'dotted',
                         'dottedHeavy', 'dash', 'dashHeavy', 'dashLong', 'dashLongHeavy',
                         'dotDash', 'dotDashHeavy', 'dotDotDash', 'dotDotDashHeavy', 'wavy',
                         'wavyHeavy', 'wavyDbl']))
    strike = NoneSet(values=(['noStrike', 'sngStrike', 'dblStrike']))
    kern = Integer(allow_none=True)
    cap = NoneSet(values=(['small', 'all']))
    spc = Integer(allow_none=True)
    normalizeH = Bool(allow_none=True)
    baseline = Integer(allow_none=True)
    noProof = Bool(allow_none=True)
    dirty = Bool(allow_none=True)
    err = Bool(allow_none=True)
    smtClean = Bool(allow_none=True)
    smtId = Integer(allow_none=True)
    bmk = String(allow_none=True)
    ln = Typed(expected_type=LineProperties, allow_none=True)
    highlight = Typed(expected_type=Color, allow_none=True)
    latin = Typed(expected_type=Font, allow_none=True)
    ea = Typed(expected_type=Font, allow_none=True)
    cs = Typed(expected_type=Font, allow_none=True)
    sym = Typed(expected_type=Font, allow_none=True)
    hlinkClick = Typed(expected_type=Hyperlink, allow_none=True)
    hlinkMouseOver = Typed(expected_type=Hyperlink, allow_none=True)
    rtl = NestedBool(allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)
    # uses element group EG_FillProperties
    noFill = EmptyTag(namespace=DRAWING_NS)
    solidFill = ColorChoiceDescriptor()
    gradFill = Typed(expected_type=GradientFillProperties, allow_none=True)
    blipFill = Typed(expected_type=BlipFillProperties, allow_none=True)
    pattFill = Typed(expected_type=PatternFillProperties, allow_none=True)
    grpFill = EmptyTag(namespace=DRAWING_NS)
    # uses element group EG_EffectProperties
    effectLst = Typed(expected_type=EffectList, allow_none=True)
    effectDag = Typed(expected_type=EffectContainer, allow_none=True)
    # uses element group EG_TextUnderlineLine
    uLnTx = EmptyTag()
    uLn = Typed(expected_type=LineProperties, allow_none=True)
    # uses element group EG_TextUnderlineFill
    uFillTx = EmptyTag()
    uFill = EmptyTag()

    __elements__ = ('ln', 'noFill', 'solidFill', 'gradFill', 'blipFill',
                    'pattFill', 'grpFill', 'effectLst', 'effectDag', 'highlight','uLnTx',
                    'uLn', 'uFillTx', 'uFill', 'latin', 'ea', 'cs', 'sym', 'hlinkClick',
                    'hlinkMouseOver', 'rtl', )

    def __init__(self,
                 kumimoji=None,
                 lang=None,
                 altLang=None,
                 sz=None,
                 b=None,
                 i=None,
                 u=None,
                 strike=None,
                 kern=None,
                 cap=None,
                 spc=None,
                 normalizeH=None,
                 baseline=None,
                 noProof=None,
                 dirty=None,
                 err=None,
                 smtClean=None,
                 smtId=None,
                 bmk=None,
                 ln=None,
                 highlight=None,
                 latin=None,
                 ea=None,
                 cs=None,
                 sym=None,
                 hlinkClick=None,
                 hlinkMouseOver=None,
                 rtl=None,
                 extLst=None,
                 noFill=None,
                 solidFill=None,
                 gradFill=None,
                 blipFill=None,
                 pattFill=None,
                 grpFill=None,
                 effectLst=None,
                 effectDag=None,
                 uLnTx=None,
                 uLn=None,
                 uFillTx=None,
                 uFill=None,
                ):
        self.kumimoji = kumimoji
        self.lang = lang
        self.altLang = altLang
        self.sz = sz
        self.b = b
        self.i = i
        self.u = u
        self.strike = strike
        self.kern = kern
        self.cap = cap
        self.spc = spc
        self.normalizeH = normalizeH
        self.baseline = baseline
        self.noProof = noProof
        self.dirty = dirty
        self.err = err
        self.smtClean = smtClean
        self.smtId = smtId
        self.bmk = bmk
        self.ln = ln
        self.highlight = highlight
        self.latin = latin
        self.ea = ea
        self.cs = cs
        self.sym = sym
        self.hlinkClick = hlinkClick
        self.hlinkMouseOver = hlinkMouseOver
        self.rtl = rtl
        self.noFill = noFill
        self.solidFill = solidFill
        self.gradFill = gradFill
        self.blipFill = blipFill
        self.pattFill = pattFill
        self.grpFill = grpFill
        self.effectLst = effectLst
        self.effectDag = effectDag
        self.uLnTx = uLnTx
        self.uLn = uLn
        self.uFillTx = uFillTx
        self.uFill = uFill


class TabStop(Serialisable):

    pos = Typed(expected_type=Coordinate, allow_none=True)
    algn = Typed(expected_type=Set(values=(['l', 'ctr', 'r', 'dec'])))

    def __init__(self,
                 pos=None,
                 algn=None,
                ):
        self.pos = pos
        self.algn = algn


class TabStopList(Serialisable):

    tab = Typed(expected_type=TabStop, allow_none=True)

    def __init__(self,
                 tab=None,
                ):
        self.tab = tab


class Spacing(Serialisable):

    spcPct = NestedInteger(allow_none=True)
    spcPts = NestedInteger(allow_none=True)

    __elements__ = ('spcPct', 'spcPts')

    def __init__(self,
                 spcPct=None,
                 spcPts=None,
                 ):
        self.spcPct = spcPct
        self.spcPts = spcPts


class AutonumberBullet(Serialisable):

    type = Set(values=(['alphaLcParenBoth', 'alphaUcParenBoth',
                        'alphaLcParenR', 'alphaUcParenR', 'alphaLcPeriod', 'alphaUcPeriod',
                        'arabicParenBoth', 'arabicParenR', 'arabicPeriod', 'arabicPlain',
                        'romanLcParenBoth', 'romanUcParenBoth', 'romanLcParenR', 'romanUcParenR',
                        'romanLcPeriod', 'romanUcPeriod', 'circleNumDbPlain',
                        'circleNumWdBlackPlain', 'circleNumWdWhitePlain', 'arabicDbPeriod',
                        'arabicDbPlain', 'ea1ChsPeriod', 'ea1ChsPlain', 'ea1ChtPeriod',
                        'ea1ChtPlain', 'ea1JpnChsDbPeriod', 'ea1JpnKorPlain', 'ea1JpnKorPeriod',
                        'arabic1Minus', 'arabic2Minus', 'hebrew2Minus', 'thaiAlphaPeriod',
                        'thaiAlphaParenR', 'thaiAlphaParenBoth', 'thaiNumPeriod',
                        'thaiNumParenR', 'thaiNumParenBoth', 'hindiAlphaPeriod',
                        'hindiNumPeriod', 'hindiNumParenR', 'hindiAlpha1Period']))
    startAt = Integer()

    def __init__(self,
                 type=None,
                 startAt=None,
                ):
        self.type = type
        self.startAt = startAt


class ParagraphProperties(Serialisable):

    tagname = "pPr"
    namespace = DRAWING_NS

    marL = Integer(allow_none=True)
    marR = Integer(allow_none=True)
    lvl = Integer(allow_none=True)
    indent = Integer(allow_none=True)
    algn = NoneSet(values=(['l', 'ctr', 'r', 'just', 'justLow', 'dist', 'thaiDist']))
    defTabSz = Integer(allow_none=True)
    rtl = Bool(allow_none=True)
    eaLnBrk = Bool(allow_none=True)
    fontAlgn = NoneSet(values=(['auto', 't', 'ctr', 'base', 'b']))
    latinLnBrk = Bool(allow_none=True)
    hangingPunct = Bool(allow_none=True)

    # uses element group EG_TextBulletColor
    # uses element group EG_TextBulletSize
    # uses element group EG_TextBulletTypeface
    # uses element group EG_TextBullet
    lnSpc = Typed(expected_type=Spacing, allow_none=True)
    spcBef = Typed(expected_type=Spacing, allow_none=True)
    spcAft = Typed(expected_type=Spacing, allow_none=True)
    tabLst = Typed(expected_type=TabStopList, allow_none=True)
    defRPr = Typed(expected_type=CharacterProperties, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)
    buClrTx = EmptyTag()
    buClr = Typed(expected_type=Color, allow_none=True)
    buSzTx = EmptyTag()
    buSzPct = NestedInteger(allow_none=True)
    buSzPts = NestedInteger(allow_none=True)
    buFontTx = EmptyTag()
    buFont = Typed(expected_type=Font, allow_none=True)
    buNone = EmptyTag()
    buAutoNum = EmptyTag()
    buChar = NestedValue(expected_type=str, attribute="char", allow_none=True)
    buBlip = NestedValue(expected_type=Blip, attribute="blip", allow_none=True)

    __elements__ = ('lnSpc', 'spcBef', 'spcAft', 'tabLst', 'defRPr',
                    'buClrTx', 'buClr', 'buSzTx', 'buSzPct', 'buSzPts', 'buFontTx', 'buFont',
                    'buNone', 'buAutoNum', 'buChar', 'buBlip')

    def __init__(self,
                 marL=None,
                 marR=None,
                 lvl=None,
                 indent=None,
                 algn=None,
                 defTabSz=None,
                 rtl=None,
                 eaLnBrk=None,
                 fontAlgn=None,
                 latinLnBrk=None,
                 hangingPunct=None,
                 lnSpc=None,
                 spcBef=None,
                 spcAft=None,
                 tabLst=None,
                 defRPr=None,
                 extLst=None,
                 buClrTx=None,
                 buClr=None,
                 buSzTx=None,
                 buSzPct=None,
                 buSzPts=None,
                 buFontTx=None,
                 buFont=None,
                 buNone=None,
                 buAutoNum=None,
                 buChar=None,
                 buBlip=None,
                 ):
        self.marL = marL
        self.marR = marR
        self.lvl = lvl
        self.indent = indent
        self.algn = algn
        self.defTabSz = defTabSz
        self.rtl = rtl
        self.eaLnBrk = eaLnBrk
        self.fontAlgn = fontAlgn
        self.latinLnBrk = latinLnBrk
        self.hangingPunct = hangingPunct
        self.lnSpc = lnSpc
        self.spcBef = spcBef
        self.spcAft = spcAft
        self.tabLst = tabLst
        self.defRPr = defRPr
        self.buClrTx = buClrTx
        self.buClr = buClr
        self.buSzTx = buSzTx
        self.buSzPct = buSzPct
        self.buSzPts = buSzPts
        self.buFontTx = buFontTx
        self.buFont = buFont
        self.buNone = buNone
        self.buAutoNum = buAutoNum
        self.buChar = buChar
        self.buBlip = buBlip
        self.defRPr = defRPr


class ListStyle(Serialisable):

    tagname = "lstStyle"
    namespace = DRAWING_NS

    defPPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl1pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl2pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl3pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl4pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl5pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl6pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl7pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl8pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    lvl9pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)

    __elements__ = ("defPPr", "lvl1pPr", "lvl2pPr", "lvl3pPr", "lvl4pPr",
                    "lvl5pPr", "lvl6pPr", "lvl7pPr", "lvl8pPr", "lvl9pPr")

    def __init__(self,
                 defPPr=None,
                 lvl1pPr=None,
                 lvl2pPr=None,
                 lvl3pPr=None,
                 lvl4pPr=None,
                 lvl5pPr=None,
                 lvl6pPr=None,
                 lvl7pPr=None,
                 lvl8pPr=None,
                 lvl9pPr=None,
                 extLst=None,
                ):
        self.defPPr = defPPr
        self.lvl1pPr = lvl1pPr
        self.lvl2pPr = lvl2pPr
        self.lvl3pPr = lvl3pPr
        self.lvl4pPr = lvl4pPr
        self.lvl5pPr = lvl5pPr
        self.lvl6pPr = lvl6pPr
        self.lvl7pPr = lvl7pPr
        self.lvl8pPr = lvl8pPr
        self.lvl9pPr = lvl9pPr


class RegularTextRun(Serialisable):

    tagname = "r"
    namespace = DRAWING_NS

    rPr = Typed(expected_type=CharacterProperties, allow_none=True)
    properties = Alias("rPr")
    t = NestedText(expected_type=str)
    value = Alias("t")

    __elements__ = ('rPr', 't')

    def __init__(self,
                 rPr=None,
                 t="",
                ):
        self.rPr = rPr
        self.t = t


class LineBreak(Serialisable):

    tagname = "br"
    namespace = DRAWING_NS

    rPr = Typed(expected_type=CharacterProperties, allow_none=True)

    __elements__ = ('rPr',)

    def __init__(self,
                 rPr=None,
                ):
        self.rPr = rPr


class TextField(Serialisable):

    id = String()
    type = String(allow_none=True)
    rPr = Typed(expected_type=CharacterProperties, allow_none=True)
    pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    t = String(allow_none=True)

    __elements__ = ('rPr', 'pPr')

    def __init__(self,
                 id=None,
                 type=None,
                 rPr=None,
                 pPr=None,
                 t=None,
                ):
        self.id = id
        self.type = type
        self.rPr = rPr
        self.pPr = pPr
        self.t = t


class Paragraph(Serialisable):

    tagname = "p"
    namespace = DRAWING_NS

    # uses element group EG_TextRun
    pPr = Typed(expected_type=ParagraphProperties, allow_none=True)
    properties = Alias("pPr")
    endParaRPr = Typed(expected_type=CharacterProperties, allow_none=True)
    r = Sequence(expected_type=RegularTextRun)
    text = Alias('r')
    br = Typed(expected_type=LineBreak, allow_none=True)
    fld = Typed(expected_type=TextField, allow_none=True)

    __elements__ = ('pPr', 'r', 'br', 'fld', 'endParaRPr')

    def __init__(self,
                 pPr=None,
                 endParaRPr=None,
                 r=None,
                 br=None,
                 fld=None,
                 ):
        self.pPr = pPr
        self.endParaRPr = endParaRPr
        if r is None:
            r = [RegularTextRun()]
        self.r = r
        self.br = br
        self.fld = fld


class GeomGuide(Serialisable):

    name = String(())
    fmla = String(())

    def __init__(self,
                 name=None,
                 fmla=None,
                ):
        self.name = name
        self.fmla = fmla


class GeomGuideList(Serialisable):

    gd = Sequence(expected_type=GeomGuide, allow_none=True)

    def __init__(self,
                 gd=None,
                ):
        self.gd = gd


class PresetTextShape(Serialisable):

    prst = Typed(expected_type=Set(values=(
        ['textNoShape', 'textPlain','textStop', 'textTriangle', 'textTriangleInverted', 'textChevron',
         'textChevronInverted', 'textRingInside', 'textRingOutside', 'textArchUp',
         'textArchDown', 'textCircle', 'textButton', 'textArchUpPour',
         'textArchDownPour', 'textCirclePour', 'textButtonPour', 'textCurveUp',
         'textCurveDown', 'textCanUp', 'textCanDown', 'textWave1', 'textWave2',
         'textDoubleWave1', 'textWave4', 'textInflate', 'textDeflate',
         'textInflateBottom', 'textDeflateBottom', 'textInflateTop',
         'textDeflateTop', 'textDeflateInflate', 'textDeflateInflateDeflate',
         'textFadeRight', 'textFadeLeft', 'textFadeUp', 'textFadeDown',
         'textSlantUp', 'textSlantDown', 'textCascadeUp', 'textCascadeDown'
         ]
    )))
    avLst = Typed(expected_type=GeomGuideList, allow_none=True)

    def __init__(self,
                 prst=None,
                 avLst=None,
                ):
        self.prst = prst
        self.avLst = avLst


class TextNormalAutofit(Serialisable):

    fontScale = Integer()
    lnSpcReduction = Integer()

    def __init__(self,
                 fontScale=None,
                 lnSpcReduction=None,
                ):
        self.fontScale = fontScale
        self.lnSpcReduction = lnSpcReduction


class RichTextProperties(Serialisable):

    tagname = "bodyPr"
    namespace = DRAWING_NS

    rot = Integer(allow_none=True)
    spcFirstLastPara = Bool(allow_none=True)
    vertOverflow = NoneSet(values=(['overflow', 'ellipsis', 'clip']))
    horzOverflow = NoneSet(values=(['overflow', 'clip']))
    vert = NoneSet(values=(['horz', 'vert', 'vert270', 'wordArtVert',
                            'eaVert', 'mongolianVert', 'wordArtVertRtl']))
    wrap = NoneSet(values=(['none', 'square']))
    lIns = Integer(allow_none=True)
    tIns = Integer(allow_none=True)
    rIns = Integer(allow_none=True)
    bIns = Integer(allow_none=True)
    numCol = Integer(allow_none=True)
    spcCol = Integer(allow_none=True)
    rtlCol = Bool(allow_none=True)
    fromWordArt = Bool(allow_none=True)
    anchor = NoneSet(values=(['t', 'ctr', 'b', 'just', 'dist']))
    anchorCtr = Bool(allow_none=True)
    forceAA = Bool(allow_none=True)
    upright = Bool(allow_none=True)
    compatLnSpc = Bool(allow_none=True)
    prstTxWarp = Typed(expected_type=PresetTextShape, allow_none=True)
    scene3d = Typed(expected_type=Scene3D, allow_none=True)
    extLst = Typed(expected_type=OfficeArtExtensionList, allow_none=True)
    noAutofit = EmptyTag()
    normAutofit = EmptyTag()
    spAutoFit = EmptyTag()
    flatTx = NestedInteger(attribute="z", allow_none=True)

    __elements__ = ('prstTxWarp', 'scene3d', 'noAutofit', 'normAutofit', 'spAutoFit')

    def __init__(self,
                 rot=None,
                 spcFirstLastPara=None,
                 vertOverflow=None,
                 horzOverflow=None,
                 vert=None,
                 wrap=None,
                 lIns=None,
                 tIns=None,
                 rIns=None,
                 bIns=None,
                 numCol=None,
                 spcCol=None,
                 rtlCol=None,
                 fromWordArt=None,
                 anchor=None,
                 anchorCtr=None,
                 forceAA=None,
                 upright=None,
                 compatLnSpc=None,
                 prstTxWarp=None,
                 scene3d=None,
                 extLst=None,
                 noAutofit=None,
                 normAutofit=None,
                 spAutoFit=None,
                 flatTx=None,
                ):
        self.rot = rot
        self.spcFirstLastPara = spcFirstLastPara
        self.vertOverflow = vertOverflow
        self.horzOverflow = horzOverflow
        self.vert = vert
        self.wrap = wrap
        self.lIns = lIns
        self.tIns = tIns
        self.rIns = rIns
        self.bIns = bIns
        self.numCol = numCol
        self.spcCol = spcCol
        self.rtlCol = rtlCol
        self.fromWordArt = fromWordArt
        self.anchor = anchor
        self.anchorCtr = anchorCtr
        self.forceAA = forceAA
        self.upright = upright
        self.compatLnSpc = compatLnSpc
        self.prstTxWarp = prstTxWarp
        self.scene3d = scene3d
        self.noAutofit = noAutofit
        self.normAutofit = normAutofit
        self.spAutoFit = spAutoFit
        self.flatTx = flatTx

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\_collections.py ===
# util/_collections.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Collection classes and helpers."""
from __future__ import annotations

import operator
import threading
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Container
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import ValuesView
import weakref

from ._has_cy import HAS_CYEXTENSION
from .typing import is_non_string_iterable
from .typing import Literal
from .typing import Protocol

if typing.TYPE_CHECKING or not HAS_CYEXTENSION:
    from ._py_collections import immutabledict as immutabledict
    from ._py_collections import IdentitySet as IdentitySet
    from ._py_collections import ReadOnlyContainer as ReadOnlyContainer
    from ._py_collections import ImmutableDictBase as ImmutableDictBase
    from ._py_collections import OrderedSet as OrderedSet
    from ._py_collections import unique_list as unique_list
else:
    from sqlalchemy.cyextension.immutabledict import (
        ReadOnlyContainer as ReadOnlyContainer,
    )
    from sqlalchemy.cyextension.immutabledict import (
        ImmutableDictBase as ImmutableDictBase,
    )
    from sqlalchemy.cyextension.immutabledict import (
        immutabledict as immutabledict,
    )
    from sqlalchemy.cyextension.collections import IdentitySet as IdentitySet
    from sqlalchemy.cyextension.collections import OrderedSet as OrderedSet
    from sqlalchemy.cyextension.collections import (  # noqa
        unique_list as unique_list,
    )


_T = TypeVar("_T", bound=Any)
_KT = TypeVar("_KT", bound=Any)
_VT = TypeVar("_VT", bound=Any)
_T_co = TypeVar("_T_co", covariant=True)

EMPTY_SET: FrozenSet[Any] = frozenset()
NONE_SET: FrozenSet[Any] = frozenset([None])


def merge_lists_w_ordering(a: List[Any], b: List[Any]) -> List[Any]:
    """merge two lists, maintaining ordering as much as possible.

    this is to reconcile vars(cls) with cls.__annotations__.

    Example::

        >>> a = ["__tablename__", "id", "x", "created_at"]
        >>> b = ["id", "name", "data", "y", "created_at"]
        >>> merge_lists_w_ordering(a, b)
        ['__tablename__', 'id', 'name', 'data', 'y', 'x', 'created_at']

    This is not necessarily the ordering that things had on the class,
    in this case the class is::

        class User(Base):
            __tablename__ = "users"

            id: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str]
            data: Mapped[Optional[str]]
            x = Column(Integer)
            y: Mapped[int]
            created_at: Mapped[datetime.datetime] = mapped_column()

    But things are *mostly* ordered.

    The algorithm could also be done by creating a partial ordering for
    all items in both lists and then using topological_sort(), but that
    is too much overhead.

    Background on how I came up with this is at:
    https://gist.github.com/zzzeek/89de958cf0803d148e74861bd682ebae

    """
    overlap = set(a).intersection(b)

    result = []

    current, other = iter(a), iter(b)

    while True:
        for element in current:
            if element in overlap:
                overlap.discard(element)
                other, current = current, other
                break

            result.append(element)
        else:
            result.extend(other)
            break

    return result


def coerce_to_immutabledict(d: Mapping[_KT, _VT]) -> immutabledict[_KT, _VT]:
    if not d:
        return EMPTY_DICT
    elif isinstance(d, immutabledict):
        return d
    else:
        return immutabledict(d)


EMPTY_DICT: immutabledict[Any, Any] = immutabledict()


class FacadeDict(ImmutableDictBase[_KT, _VT]):
    """A dictionary that is not publicly mutable."""

    def __new__(cls, *args: Any) -> FacadeDict[Any, Any]:
        new = ImmutableDictBase.__new__(cls)
        return new

    def copy(self) -> NoReturn:
        raise NotImplementedError(
            "an immutabledict shouldn't need to be copied.  use dict(d) "
            "if you need a mutable dictionary."
        )

    def __reduce__(self) -> Any:
        return FacadeDict, (dict(self),)

    def _insert_item(self, key: _KT, value: _VT) -> None:
        """insert an item into the dictionary directly."""
        dict.__setitem__(self, key, value)

    def __repr__(self) -> str:
        return "FacadeDict(%s)" % dict.__repr__(self)


_DT = TypeVar("_DT", bound=Any)

_F = TypeVar("_F", bound=Any)


class Properties(Generic[_T]):
    """Provide a __getattr__/__setattr__ interface over a dict."""

    __slots__ = ("_data",)

    _data: Dict[str, _T]

    def __init__(self, data: Dict[str, _T]):
        object.__setattr__(self, "_data", data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[_T]:
        return iter(list(self._data.values()))

    def __dir__(self) -> List[str]:
        return dir(super()) + [str(k) for k in self._data.keys()]

    def __add__(self, other: Properties[_F]) -> List[Union[_T, _F]]:
        return list(self) + list(other)

    def __setitem__(self, key: str, obj: _T) -> None:
        self._data[key] = obj

    def __getitem__(self, key: str) -> _T:
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __setattr__(self, key: str, obj: _T) -> None:
        self._data[key] = obj

    def __getstate__(self) -> Dict[str, Any]:
        return {"_data": self._data}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", state["_data"])

    def __getattr__(self, key: str) -> _T:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def as_readonly(self) -> ReadOnlyProperties[_T]:
        """Return an immutable proxy for this :class:`.Properties`."""

        return ReadOnlyProperties(self._data)

    def update(self, value: Dict[str, _T]) -> None:
        self._data.update(value)

    @overload
    def get(self, key: str) -> Optional[_T]: ...

    @overload
    def get(self, key: str, default: Union[_DT, _T]) -> Union[_DT, _T]: ...

    def get(
        self, key: str, default: Optional[Union[_DT, _T]] = None
    ) -> Optional[Union[_T, _DT]]:
        if key in self:
            return self[key]
        else:
            return default

    def keys(self) -> List[str]:
        return list(self._data)

    def values(self) -> List[_T]:
        return list(self._data.values())

    def items(self) -> List[Tuple[str, _T]]:
        return list(self._data.items())

    def has_key(self, key: str) -> bool:
        return key in self._data

    def clear(self) -> None:
        self._data.clear()


class OrderedProperties(Properties[_T]):
    """Provide a __getattr__/__setattr__ interface with an OrderedDict
    as backing store."""

    __slots__ = ()

    def __init__(self):
        Properties.__init__(self, OrderedDict())


class ReadOnlyProperties(ReadOnlyContainer, Properties[_T]):
    """Provide immutable dict/object attribute to an underlying dictionary."""

    __slots__ = ()


def _ordered_dictionary_sort(d, key=None):
    """Sort an OrderedDict in-place."""

    items = [(k, d[k]) for k in sorted(d, key=key)]

    d.clear()

    d.update(items)


OrderedDict = dict
sort_dictionary = _ordered_dictionary_sort


class WeakSequence(Sequence[_T]):
    def __init__(self, __elements: Sequence[_T] = ()):
        # adapted from weakref.WeakKeyDictionary, prevent reference
        # cycles in the collection itself
        def _remove(item, selfref=weakref.ref(self)):
            self = selfref()
            if self is not None:
                self._storage.remove(item)

        self._remove = _remove
        self._storage = [
            weakref.ref(element, _remove) for element in __elements
        ]

    def append(self, item):
        self._storage.append(weakref.ref(item, self._remove))

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return (
            obj for obj in (ref() for ref in self._storage) if obj is not None
        )

    def __getitem__(self, index):
        try:
            obj = self._storage[index]
        except KeyError:
            raise IndexError("Index %s out of range" % index)
        else:
            return obj()


class OrderedIdentitySet(IdentitySet):
    def __init__(self, iterable: Optional[Iterable[Any]] = None):
        IdentitySet.__init__(self)
        self._members = OrderedDict()
        if iterable:
            for o in iterable:
                self.add(o)


class PopulateDict(Dict[_KT, _VT]):
    """A dict which populates missing values via a creation function.

    Note the creation function takes a key, unlike
    collections.defaultdict.

    """

    def __init__(self, creator: Callable[[_KT], _VT]):
        self.creator = creator

    def __missing__(self, key: Any) -> Any:
        self[key] = val = self.creator(key)
        return val


class WeakPopulateDict(Dict[_KT, _VT]):
    """Like PopulateDict, but assumes a self + a method and does not create
    a reference cycle.

    """

    def __init__(self, creator_method: types.MethodType):
        self.creator = creator_method.__func__
        weakself = creator_method.__self__
        self.weakself = weakref.ref(weakself)

    def __missing__(self, key: Any) -> Any:
        self[key] = val = self.creator(self.weakself(), key)
        return val


# Define collections that are capable of storing
# ColumnElement objects as hashable keys/elements.
# At this point, these are mostly historical, things
# used to be more complicated.
column_set = set
column_dict = dict
ordered_column_set = OrderedSet


class UniqueAppender(Generic[_T]):
    """Appends items to a collection ensuring uniqueness.

    Additional appends() of the same object are ignored.  Membership is
    determined by identity (``is a``) not equality (``==``).
    """

    __slots__ = "data", "_data_appender", "_unique"

    data: Union[Iterable[_T], Set[_T], List[_T]]
    _data_appender: Callable[[_T], None]
    _unique: Dict[int, Literal[True]]

    def __init__(
        self,
        data: Union[Iterable[_T], Set[_T], List[_T]],
        via: Optional[str] = None,
    ):
        self.data = data
        self._unique = {}
        if via:
            self._data_appender = getattr(data, via)
        elif hasattr(data, "append"):
            self._data_appender = cast("List[_T]", data).append
        elif hasattr(data, "add"):
            self._data_appender = cast("Set[_T]", data).add

    def append(self, item: _T) -> None:
        id_ = id(item)
        if id_ not in self._unique:
            self._data_appender(item)
            self._unique[id_] = True

    def __iter__(self) -> Iterator[_T]:
        return iter(self.data)


def coerce_generator_arg(arg: Any) -> List[Any]:
    if len(arg) == 1 and isinstance(arg[0], types.GeneratorType):
        return list(arg[0])
    else:
        return cast("List[Any]", arg)


def to_list(x: Any, default: Optional[List[Any]] = None) -> List[Any]:
    if x is None:
        return default  # type: ignore
    if not is_non_string_iterable(x):
        return [x]
    elif isinstance(x, list):
        return x
    else:
        return list(x)


def has_intersection(set_: Container[Any], iterable: Iterable[Any]) -> bool:
    r"""return True if any items of set\_ are present in iterable.

    Goes through special effort to ensure __hash__ is not called
    on items in iterable that don't support it.

    """
    return any(i in set_ for i in iterable if i.__hash__)


def to_set(x):
    if x is None:
        return set()
    if not isinstance(x, set):
        return set(to_list(x))
    else:
        return x


def to_column_set(x: Any) -> Set[Any]:
    if x is None:
        return column_set()
    if not isinstance(x, column_set):
        return column_set(to_list(x))
    else:
        return x


def update_copy(
    d: Dict[Any, Any], _new: Optional[Dict[Any, Any]] = None, **kw: Any
) -> Dict[Any, Any]:
    """Copy the given dict and update with the given values."""

    d = d.copy()
    if _new:
        d.update(_new)
    d.update(**kw)
    return d


def flatten_iterator(x: Iterable[_T]) -> Iterator[_T]:
    """Given an iterator of which further sub-elements may also be
    iterators, flatten the sub-elements into a single iterator.

    """
    elem: _T
    for elem in x:
        if not isinstance(elem, str) and hasattr(elem, "__iter__"):
            yield from flatten_iterator(elem)
        else:
            yield elem


class LRUCache(typing.MutableMapping[_KT, _VT]):
    """Dictionary with 'squishy' removal of least
    recently used items.

    Note that either get() or [] should be used here, but
    generally its not safe to do an "in" check first as the dictionary
    can change subsequent to that call.

    """

    __slots__ = (
        "capacity",
        "threshold",
        "size_alert",
        "_data",
        "_counter",
        "_mutex",
    )

    capacity: int
    threshold: float
    size_alert: Optional[Callable[[LRUCache[_KT, _VT]], None]]

    def __init__(
        self,
        capacity: int = 100,
        threshold: float = 0.5,
        size_alert: Optional[Callable[..., None]] = None,
    ):
        self.capacity = capacity
        self.threshold = threshold
        self.size_alert = size_alert
        self._counter = 0
        self._mutex = threading.Lock()
        self._data: Dict[_KT, Tuple[_KT, _VT, List[int]]] = {}

    def _inc_counter(self):
        self._counter += 1
        return self._counter

    @overload
    def get(self, key: _KT) -> Optional[_VT]: ...

    @overload
    def get(self, key: _KT, default: Union[_VT, _T]) -> Union[_VT, _T]: ...

    def get(
        self, key: _KT, default: Optional[Union[_VT, _T]] = None
    ) -> Optional[Union[_VT, _T]]:
        item = self._data.get(key)
        if item is not None:
            item[2][0] = self._inc_counter()
            return item[1]
        else:
            return default

    def __getitem__(self, key: _KT) -> _VT:
        item = self._data[key]
        item[2][0] = self._inc_counter()
        return item[1]

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def values(self) -> ValuesView[_VT]:
        return typing.ValuesView({k: i[1] for k, i in self._data.items()})

    def __setitem__(self, key: _KT, value: _VT) -> None:
        self._data[key] = (key, value, [self._inc_counter()])
        self._manage_size()

    def __delitem__(self, __v: _KT) -> None:
        del self._data[__v]

    @property
    def size_threshold(self) -> float:
        return self.capacity + self.capacity * self.threshold

    def _manage_size(self) -> None:
        if not self._mutex.acquire(False):
            return
        try:
            size_alert = bool(self.size_alert)
            while len(self) > self.capacity + self.capacity * self.threshold:
                if size_alert:
                    size_alert = False
                    self.size_alert(self)  # type: ignore
                by_counter = sorted(
                    self._data.values(),
                    key=operator.itemgetter(2),
                    reverse=True,
                )
                for item in by_counter[self.capacity :]:
                    try:
                        del self._data[item[0]]
                    except KeyError:
                        # deleted elsewhere; skip
                        continue
        finally:
            self._mutex.release()


class _CreateFuncType(Protocol[_T_co]):
    def __call__(self) -> _T_co: ...


class _ScopeFuncType(Protocol):
    def __call__(self) -> Any: ...


class ScopedRegistry(Generic[_T]):
    """A Registry that can store one or multiple instances of a single
    class on the basis of a "scope" function.

    The object implements ``__call__`` as the "getter", so by
    calling ``myregistry()`` the contained object is returned
    for the current scope.

    :param createfunc:
      a callable that returns a new object to be placed in the registry

    :param scopefunc:
      a callable that will return a key to store/retrieve an object.
    """

    __slots__ = "createfunc", "scopefunc", "registry"

    createfunc: _CreateFuncType[_T]
    scopefunc: _ScopeFuncType
    registry: Any

    def __init__(
        self, createfunc: Callable[[], _T], scopefunc: Callable[[], Any]
    ):
        """Construct a new :class:`.ScopedRegistry`.

        :param createfunc:  A creation function that will generate
          a new value for the current scope, if none is present.

        :param scopefunc:  A function that returns a hashable
          token representing the current scope (such as, current
          thread identifier).

        """
        self.createfunc = createfunc
        self.scopefunc = scopefunc
        self.registry = {}

    def __call__(self) -> _T:
        key = self.scopefunc()
        try:
            return self.registry[key]  # type: ignore[no-any-return]
        except KeyError:
            return self.registry.setdefault(key, self.createfunc())  # type: ignore[no-any-return] # noqa: E501

    def has(self) -> bool:
        """Return True if an object is present in the current scope."""

        return self.scopefunc() in self.registry

    def set(self, obj: _T) -> None:
        """Set the value for the current scope."""

        self.registry[self.scopefunc()] = obj

    def clear(self) -> None:
        """Clear the current scope, if any."""

        try:
            del self.registry[self.scopefunc()]
        except KeyError:
            pass


class ThreadLocalRegistry(ScopedRegistry[_T]):
    """A :class:`.ScopedRegistry` that uses a ``threading.local()``
    variable for storage.

    """

    def __init__(self, createfunc: Callable[[], _T]):
        self.createfunc = createfunc
        self.registry = threading.local()

    def __call__(self) -> _T:
        try:
            return self.registry.value  # type: ignore[no-any-return]
        except AttributeError:
            val = self.registry.value = self.createfunc()
            return val

    def has(self) -> bool:
        return hasattr(self.registry, "value")

    def set(self, obj: _T) -> None:
        self.registry.value = obj

    def clear(self) -> None:
        try:
            del self.registry.value
        except AttributeError:
            pass


def has_dupes(sequence, target):
    """Given a sequence and search object, return True if there's more
    than one, False if zero or one of them.


    """
    # compare to .index version below, this version introduces less function
    # overhead and is usually the same speed.  At 15000 items (way bigger than
    # a relationship-bound collection in memory usually is) it begins to
    # fall behind the other version only by microseconds.
    c = 0
    for item in sequence:
        if item is target:
            c += 1
            if c > 1:
                return True
    return False


# .index version.  the two __contains__ calls as well
# as .index() and isinstance() slow this down.
# def has_dupes(sequence, target):
#    if target not in sequence:
#        return False
#    elif not isinstance(sequence, collections_abc.Sequence):
#        return False
#
#    idx = sequence.index(target)
#    return target in sequence[idx + 1:]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\modes\emacs.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import pyreadline3.lineeditor.lineobj as lineobj
from pyreadline3.lineeditor.lineobj import Point
from pyreadline3.logger import log

from . import basemode


class IncrementalSearchPromptMode(object):
    def __init__(self, rlobj):
        pass

    def _process_incremental_search_keyevent(self, keyinfo):
        log("_process_incremental_search_keyevent")
        keytuple = keyinfo.tuple()
        # dispatch_func = self.key_dispatch.get(keytuple, default)
        revtuples = []
        fwdtuples = []
        for ktuple, func in self.key_dispatch.items():
            if func == self.reverse_search_history:
                revtuples.append(ktuple)
            elif func == self.forward_search_history:
                fwdtuples.append(ktuple)

        log("IncrementalSearchPromptMode %s %s" % (keyinfo, keytuple))
        if keyinfo.keyname == "backspace":
            self.subsearch_query = self.subsearch_query[:-1]
            if len(self.subsearch_query) > 0:
                self.line = self.subsearch_fun(self.subsearch_query)
            else:
                self._bell()
                self.line = ""  # empty query means no search result
        elif keyinfo.keyname in ["return", "escape"]:
            self._bell()
            self.prompt = self.subsearch_oldprompt
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            self._history.history_cursor = len(self._history.history)
            if keyinfo.keyname == "escape":
                self.l_buffer.set_line(self.subsearch_old_line)
            return True
        elif keyinfo.keyname:
            pass
        elif keytuple in revtuples:
            self.subsearch_fun = self._history.reverse_search_history
            self.subsearch_prompt = "reverse-i-search%d`%s': "
            self.line = self.subsearch_fun(self.subsearch_query)
        elif keytuple in fwdtuples:
            self.subsearch_fun = self._history.forward_search_history
            self.subsearch_prompt = "forward-i-search%d`%s': "
            self.line = self.subsearch_fun(self.subsearch_query)
        elif keyinfo.control == False and keyinfo.meta == False:
            self.subsearch_query += keyinfo.char
            self.line = self.subsearch_fun(self.subsearch_query)
        else:
            pass
        self.prompt = self.subsearch_prompt % (
            self._history.history_cursor,
            self.subsearch_query,
        )
        self.l_buffer.set_line(self.line)

    def _init_incremental_search(self, searchfun, init_event):
        """Initialize search prompt"""
        log("init_incremental_search")
        self.subsearch_query = ""
        self.subsearch_fun = searchfun
        self.subsearch_old_line = self.l_buffer.get_line_text()

        queue = self.process_keyevent_queue
        queue.append(self._process_incremental_search_keyevent)

        self.subsearch_oldprompt = self.prompt

        if (
            self.previous_func != self.reverse_search_history
            and self.previous_func != self.forward_search_history
        ):
            self.subsearch_query = self.l_buffer[0:Point].get_line_text()

        if self.subsearch_fun == self.reverse_search_history:
            self.subsearch_prompt = "reverse-i-search%d`%s': "
        else:
            self.subsearch_prompt = "forward-i-search%d`%s': "

        self.prompt = self.subsearch_prompt % (self._history.history_cursor, "")

        if self.subsearch_query:
            self.line = self._process_incremental_search_keyevent(init_event)
        else:
            self.line = ""


class SearchPromptMode(object):
    def __init__(self, rlobj):
        pass

    def _process_non_incremental_search_keyevent(self, keyinfo):
        keytuple = keyinfo.tuple()
        log("SearchPromptMode %s %s" % (keyinfo, keytuple))
        history = self._history

        if keyinfo.keyname == "backspace":
            self.non_inc_query = self.non_inc_query[:-1]
        elif keyinfo.keyname in ["return", "escape"]:
            if self.non_inc_query:
                if self.non_inc_direction == -1:
                    res = history.reverse_search_history(self.non_inc_query)
                else:
                    res = history.forward_search_history(self.non_inc_query)

            self._bell()
            self.prompt = self.non_inc_oldprompt
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            self._history.history_cursor = len(self._history.history)
            if keyinfo.keyname == "escape":
                self.l_buffer = self.non_inc_oldline
            else:
                self.l_buffer.set_line(res)
            return False
        elif keyinfo.keyname:
            pass
        elif keyinfo.control == False and keyinfo.meta == False:
            self.non_inc_query += keyinfo.char
        else:
            pass
        self.prompt = self.non_inc_oldprompt + ":" + self.non_inc_query

    def _init_non_i_search(self, direction):
        self.non_inc_direction = direction
        self.non_inc_query = ""
        self.non_inc_oldprompt = self.prompt
        self.non_inc_oldline = self.l_buffer.copy()
        self.l_buffer.reset_line()
        self.prompt = self.non_inc_oldprompt + ":"
        queue = self.process_keyevent_queue
        queue.append(self._process_non_incremental_search_keyevent)

    def non_incremental_reverse_search_history(self, e):  # (M-p)
        """Search backward starting at the current line and moving up
        through the history as necessary using a non-incremental search for
        a string supplied by the user."""
        return self._init_non_i_search(-1)

    def non_incremental_forward_search_history(self, e):  # (M-n)
        """Search forward starting at the current line and moving down
        through the the history as necessary using a non-incremental search
        for a string supplied by the user."""
        return self._init_non_i_search(1)


class LeaveModeTryNext(Exception):
    pass


class DigitArgumentMode(object):
    def __init__(self, rlobj):
        pass

    def _process_digit_argument_keyevent(self, keyinfo):
        log("DigitArgumentMode.keyinfo %s" % keyinfo)
        keytuple = keyinfo.tuple()
        log("DigitArgumentMode.keytuple %s %s" % (keyinfo, keytuple))
        if keyinfo.keyname in ["return"]:
            self.prompt = self._digit_argument_oldprompt
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            return True
        elif keyinfo.keyname:
            pass
        elif (
            keyinfo.char in "0123456789"
            and keyinfo.control == False
            and keyinfo.meta == False
        ):
            log("arg %s %s" % (self.argument, keyinfo.char))
            self.argument = self.argument * 10 + int(keyinfo.char)
        else:
            self.prompt = self._digit_argument_oldprompt
            raise LeaveModeTryNext
        self.prompt = "(arg: %s) " % self.argument

    def _init_digit_argument(self, keyinfo):
        """Initialize search prompt"""
        c = self.console
        line = self.l_buffer.get_line_text()
        self._digit_argument_oldprompt = self.prompt
        queue = self.process_keyevent_queue
        queue = self.process_keyevent_queue
        queue.append(self._process_digit_argument_keyevent)

        if keyinfo.char == "-":
            self.argument = -1
        elif keyinfo.char in "0123456789":
            self.argument = int(keyinfo.char)
        log("<%s> %s" % (self.argument, type(self.argument)))
        self.prompt = "(arg: %s) " % self.argument
        log("arg-init %s %s" % (self.argument, keyinfo.char))


class EmacsMode(
    DigitArgumentMode, IncrementalSearchPromptMode, SearchPromptMode, basemode.BaseMode
):
    mode = "emacs"

    def __init__(self, rlobj):
        basemode.BaseMode.__init__(self, rlobj)
        IncrementalSearchPromptMode.__init__(self, rlobj)
        SearchPromptMode.__init__(self, rlobj)
        DigitArgumentMode.__init__(self, rlobj)
        self._keylog = lambda x, y: None
        self.previous_func = None
        self.prompt = ">>> "
        self._insert_verbatim = False
        self.next_meta = False  # True to force meta on next character

        self.process_keyevent_queue = [self._process_keyevent]

    def __repr__(self):
        return "<EmacsMode>"

    def add_key_logger(self, logfun):
        """logfun should be function that takes disp_fun and line_""" """buffer object """
        self._keylog = logfun

    def process_keyevent(self, keyinfo):
        try:
            r = self.process_keyevent_queue[-1](keyinfo)
        except LeaveModeTryNext:
            self.process_keyevent_queue = self.process_keyevent_queue[:-1]
            r = self.process_keyevent(keyinfo)
        if r:
            self.add_history(self.l_buffer.copy())
            return True
        return False

    def _process_keyevent(self, keyinfo):
        """return True when line is final"""
        # Process exit keys. Only exit on empty line
        log("_process_keyevent <%s>" % keyinfo)

        def nop(e):
            pass

        if self.next_meta:
            self.next_meta = False
            keyinfo.meta = True
        keytuple = keyinfo.tuple()

        if self._insert_verbatim:
            self.insert_text(keyinfo)
            self._insert_verbatim = False
            self.argument = 0
            return False

        if keytuple in self.exit_dispatch:
            pars = (self.l_buffer, lineobj.EndOfLine(self.l_buffer))
            log("exit_dispatch:<%s, %s>" % pars)
            if lineobj.EndOfLine(self.l_buffer) == 0:
                raise EOFError
        if keyinfo.keyname or keyinfo.control or keyinfo.meta:
            default = nop
        else:
            default = self.self_insert
        dispatch_func = self.key_dispatch.get(keytuple, default)

        log("readline from keyboard:<%s,%s>" % (keytuple, dispatch_func))

        r = None
        if dispatch_func:
            r = dispatch_func(keyinfo)
            self._keylog(dispatch_func, self.l_buffer)
            self.l_buffer.push_undo()

        self.previous_func = dispatch_func
        return r

    # History commands
    def previous_history(self, e):  # (C-p)
        """Move back through the history list, fetching the previous
        command."""
        self._history.previous_history(self.l_buffer)
        self.l_buffer.point = lineobj.EndOfLine
        self.finalize()

    def next_history(self, e):  # (C-n)
        """Move forward through the history list, fetching the next
        command."""
        self._history.next_history(self.l_buffer)
        self.finalize()

    def beginning_of_history(self, e):  # (M-<)
        """Move to the first line in the history."""
        self._history.beginning_of_history()
        self.finalize()

    def end_of_history(self, e):  # (M->)
        """Move to the end of the input history, i.e., the line currently
        being entered."""
        self._history.end_of_history(self.l_buffer)
        self.finalize()

    def reverse_search_history(self, e):  # (C-r)
        """Search backward starting at the current line and moving up
        through the history as necessary. This is an incremental search."""
        log("rev_search_history")
        self._init_incremental_search(self._history.reverse_search_history, e)
        self.finalize()

    def forward_search_history(self, e):  # (C-s)
        """Search forward starting at the current line and moving down
        through the the history as necessary. This is an incremental
        search."""
        log("fwd_search_history")
        self._init_incremental_search(self._history.forward_search_history, e)
        self.finalize()

    def history_search_forward(self, e):  # ()
        """Search forward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        if self.previous_func and hasattr(self._history, self.previous_func.__name__):
            self._history.lastcommand = getattr(
                self._history, self.previous_func.__name__
            )
        else:
            self._history.lastcommand = None
        q = self._history.history_search_forward(self.l_buffer)
        self.l_buffer = q
        self.l_buffer.point = q.point
        self.finalize()

    def history_search_backward(self, e):  # ()
        """Search backward through the history for the string of characters
        between the start of the current line and the point. This is a
        non-incremental search. By default, this command is unbound."""
        if self.previous_func and hasattr(self._history, self.previous_func.__name__):
            self._history.lastcommand = getattr(
                self._history, self.previous_func.__name__
            )
        else:
            self._history.lastcommand = None
        q = self._history.history_search_backward(self.l_buffer)
        self.l_buffer = q
        self.l_buffer.point = q.point
        self.finalize()

    def yank_nth_arg(self, e):  # (M-C-y)
        """Insert the first argument to the previous command (usually the
        second word on the previous line) at point. With an argument n,
        insert the nth word from the previous command (the words in the
        previous command begin with word 0). A negative argument inserts the
        nth word from the end of the previous command."""
        self.finalize()

    def yank_last_arg(self, e):  # (M-. or M-_)
        """Insert last argument to the previous command (the last word of
        the previous history entry). With an argument, behave exactly like
        yank-nth-arg. Successive calls to yank-last-arg move back through
        the history list, inserting the last argument of each line in turn."""
        self.finalize()

    def forward_backward_delete_char(self, e):  # ()
        """Delete the character under the cursor, unless the cursor is at
        the end of the line, in which case the character behind the cursor
        is deleted. By default, this is not bound to a key."""
        self.finalize()

    def quoted_insert(self, e):  # (C-q or C-v)
        """Add the next character typed to the line verbatim. This is how to
        insert key sequences like C-q, for example."""
        self._insert_verbatim = True
        self.finalize()

    def tab_insert(self, e):  # (M-TAB)
        """Insert a tab character."""
        cursor = min(self.l_buffer.point, len(self.l_buffer.line_buffer))
        ws = " " * (self.tabstop - (cursor % self.tabstop))
        self.insert_text(ws)
        self.finalize()

    def transpose_chars(self, e):  # (C-t)
        """Drag the character before the cursor forward over the character
        at the cursor, moving the cursor forward as well. If the insertion
        point is at the end of the line, then this transposes the last two
        characters of the line. Negative arguments have no effect."""
        self.l_buffer.transpose_chars()
        self.finalize()

    def transpose_words(self, e):  # (M-t)
        """Drag the word before point past the word after point, moving
        point past that word as well. If the insertion point is at the end
        of the line, this transposes the last two words on the line."""
        self.l_buffer.transpose_words()
        self.finalize()

    def overwrite_mode(self, e):  # ()
        """Toggle overwrite mode. With an explicit positive numeric
        argument, switches to overwrite mode. With an explicit non-positive
        numeric argument, switches to insert mode. This command affects only
        emacs mode; vi mode does overwrite differently. Each call to
        readline() starts in insert mode. In overwrite mode, characters
        bound to self-insert replace the text at point rather than pushing
        the text to the right. Characters bound to backward-delete-char
        replace the character before point with a space."""
        self.finalize()

    def kill_line(self, e):  # (C-k)
        """Kill the text from point to the end of the line."""
        self.l_buffer.kill_line()
        self.finalize()

    def backward_kill_line(self, e):  # (C-x Rubout)
        """Kill backward to the beginning of the line."""
        self.l_buffer.backward_kill_line()
        self.finalize()

    def unix_line_discard(self, e):  # (C-u)
        """Kill backward from the cursor to the beginning of the current
        line."""
        # how is this different from backward_kill_line?
        self.l_buffer.unix_line_discard()
        self.finalize()

    def kill_whole_line(self, e):  # ()
        """Kill all characters on the current line, no matter where point
        is. By default, this is unbound."""
        self.l_buffer.kill_whole_line()
        self.finalize()

    def kill_word(self, e):  # (M-d)
        """Kill from point to the end of the current word, or if between
        words, to the end of the next word. Word boundaries are the same as
        forward-word."""
        self.l_buffer.kill_word()
        self.finalize()

    forward_kill_word = kill_word

    def backward_kill_word(self, e):  # (M-DEL)
        """Kill the word behind point. Word boundaries are the same as
        backward-word."""
        self.l_buffer.backward_kill_word()
        self.finalize()

    def unix_word_rubout(self, e):  # (C-w)
        """Kill the word behind point, using white space as a word
        boundary. The killed text is saved on the kill-ring."""
        self.l_buffer.unix_word_rubout()
        self.finalize()

    def kill_region(self, e):  # ()
        """Kill the text in the current region. By default, this command is
        unbound."""
        self.finalize()

    def copy_region_as_kill(self, e):  # ()
        """Copy the text in the region to the kill buffer, so it can be
        yanked right away. By default, this command is unbound."""
        self.finalize()

    def copy_backward_word(self, e):  # ()
        """Copy the word before point to the kill buffer. The word
        boundaries are the same as backward-word. By default, this command
        is unbound."""
        self.finalize()

    def copy_forward_word(self, e):  # ()
        """Copy the word following point to the kill buffer. The word
        boundaries are the same as forward-word. By default, this command is
        unbound."""
        self.finalize()

    def yank(self, e):  # (C-y)
        """Yank the top of the kill ring into the buffer at point."""
        self.l_buffer.yank()
        self.finalize()

    def yank_pop(self, e):  # (M-y)
        """Rotate the kill-ring, and yank the new top. You can only do this
        if the prior command is yank or yank-pop."""
        self.l_buffer.yank_pop()
        self.finalize()

    def delete_char_or_list(self, e):  # ()
        """Deletes the character under the cursor if not at the beginning or
        end of the line (like delete-char). If at the end of the line,
        behaves identically to possible-completions. This command is unbound
        by default."""
        self.finalize()

    def start_kbd_macro(self, e):  # (C-x ()
        """Begin saving the characters typed into the current keyboard
        macro."""
        self.finalize()

    def end_kbd_macro(self, e):  # (C-x ))
        """Stop saving the characters typed into the current keyboard macro
        and save the definition."""
        self.finalize()

    def call_last_kbd_macro(self, e):  # (C-x e)
        """Re-execute the last keyboard macro defined, by making the
        characters in the macro appear as if typed at the keyboard."""
        self.finalize()

    def re_read_init_file(self, e):  # (C-x C-r)
        """Read in the contents of the inputrc file, and incorporate any
        bindings or variable assignments found there."""
        self.finalize()

    def abort(self, e):  # (C-g)
        """Abort the current editing command and ring the terminals bell
        (subject to the setting of bell-style)."""
        self._bell()
        self.finalize()

    def do_uppercase_version(self, e):  # (M-a, M-b, M-x, ...)
        """If the metafied character x is lowercase, run the command that is
        bound to the corresponding uppercase character."""
        self.finalize()

    def prefix_meta(self, e):  # (ESC)
        """Metafy the next character typed. This is for keyboards without a
        meta key. Typing ESC f is equivalent to typing M-f."""
        self.next_meta = True
        self.finalize()

    def undo(self, e):  # (C-_ or C-x C-u)
        """Incremental undo, separately remembered for each line."""
        self.l_buffer.pop_undo()
        self.finalize()

    def revert_line(self, e):  # (M-r)
        """Undo all changes made to this line. This is like executing the
        undo command enough times to get back to the beginning."""
        self.finalize()

    def tilde_expand(self, e):  # (M-~)
        """Perform tilde expansion on the current word."""
        self.finalize()

    def set_mark(self, e):  # (C-@)
        """Set the mark to the point. If a numeric argument is supplied, the
        mark is set to that position."""
        self.l_buffer.set_mark()
        self.finalize()

    def exchange_point_and_mark(self, e):  # (C-x C-x)
        """Swap the point with the mark. The current cursor position is set
        to the saved position, and the old cursor position is saved as the
        mark."""
        self.finalize()

    def character_search(self, e):  # (C-])
        """A character is read and point is moved to the next occurrence of
        that character. A negative count searches for previous occurrences."""
        self.finalize()

    def character_search_backward(self, e):  # (M-C-])
        """A character is read and point is moved to the previous occurrence
        of that character. A negative count searches for subsequent
        occurrences."""
        self.finalize()

    def insert_comment(self, e):  # (M-#)
        """Without a numeric argument, the value of the comment-begin
        variable is inserted at the beginning of the current line. If a
        numeric argument is supplied, this command acts as a toggle: if the
        characters at the beginning of the line do not match the value of
        comment-begin, the value is inserted, otherwise the characters in
        comment-begin are deleted from the beginning of the line. In either
        case, the line is accepted as if a newline had been typed."""
        self.finalize()

    def dump_variables(self, e):  # ()
        """Print all of the settable variables and their values to the
        Readline output stream. If a numeric argument is supplied, the
        output is formatted in such a way that it can be made part of an
        inputrc file. This command is unbound by default."""
        self.finalize()

    def dump_macros(self, e):  # ()
        """Print all of the Readline key sequences bound to macros and the
        strings they output. If a numeric argument is supplied, the output
        is formatted in such a way that it can be made part of an inputrc
        file. This command is unbound by default."""
        self.finalize()

    def digit_argument(self, e):  # (M-0, M-1, ... M--)
        """Add this digit to the argument already accumulating, or start a
        new argument. M-- starts a negative argument."""
        self._init_digit_argument(e)
        # Should not finalize

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
        # Should not finalize

    # Create key bindings:
    def init_editing_mode(self, e):  # (C-e)
        """When in vi command mode, this causes a switch to emacs editing
        mode."""
        self._bind_exit_key("Control-d")
        self._bind_exit_key("Control-z")

        # I often accidentally hold the shift or control while typing space
        self._bind_key("space", self.self_insert)
        self._bind_key("Shift-space", self.self_insert)
        self._bind_key("Control-space", self.self_insert)
        self._bind_key("Return", self.accept_line)
        self._bind_key("Left", self.backward_char)
        self._bind_key("Control-b", self.backward_char)
        self._bind_key("Right", self.forward_char)
        self._bind_key("Control-f", self.forward_char)
        self._bind_key("Control-h", self.backward_delete_char)
        self._bind_key("BackSpace", self.backward_delete_char)
        self._bind_key("Control-BackSpace", self.backward_delete_word)

        self._bind_key("Home", self.beginning_of_line)
        self._bind_key("End", self.end_of_line)
        self._bind_key("Delete", self.delete_char)
        self._bind_key("Control-d", self.delete_char)
        self._bind_key("Clear", self.clear_screen)
        self._bind_key("Alt-f", self.forward_word)
        self._bind_key("Alt-b", self.backward_word)
        self._bind_key("Control-l", self.clear_screen)
        self._bind_key("Control-p", self.previous_history)
        self._bind_key("Up", self.history_search_backward)
        self._bind_key("Control-n", self.next_history)
        self._bind_key("Down", self.history_search_forward)
        self._bind_key("Control-a", self.beginning_of_line)
        self._bind_key("Control-e", self.end_of_line)
        self._bind_key("Alt-<", self.beginning_of_history)
        self._bind_key("Alt->", self.end_of_history)
        self._bind_key("Control-r", self.reverse_search_history)
        self._bind_key("Control-s", self.forward_search_history)
        self._bind_key("Control-Shift-r", self.forward_search_history)
        self._bind_key("Alt-p", self.non_incremental_reverse_search_history)
        self._bind_key("Alt-n", self.non_incremental_forward_search_history)
        self._bind_key("Control-z", self.undo)
        self._bind_key("Control-_", self.undo)
        self._bind_key("Escape", self.kill_whole_line)
        self._bind_key("Meta-d", self.kill_word)
        self._bind_key("Control-Delete", self.forward_delete_word)
        self._bind_key("Control-w", self.unix_word_rubout)
        # self._bind_key('Control-Shift-v',   self.quoted_insert)
        self._bind_key("Control-v", self.paste)
        self._bind_key("Alt-v", self.ipython_paste)
        self._bind_key("Control-y", self.yank)
        self._bind_key("Control-k", self.kill_line)
        self._bind_key("Control-m", self.set_mark)
        self._bind_key("Control-q", self.copy_region_to_clipboard)
        #        self._bind_key('Control-shift-k',  self.kill_whole_line)
        self._bind_key("Control-Shift-v", self.paste_mulitline_code)
        self._bind_key("Control-Right", self.forward_word_end)
        self._bind_key("Control-Left", self.backward_word)
        self._bind_key("Shift-Right", self.forward_char_extend_selection)
        self._bind_key("Shift-Left", self.backward_char_extend_selection)
        self._bind_key("Shift-Control-Right", self.forward_word_end_extend_selection)
        self._bind_key("Shift-Control-Left", self.backward_word_extend_selection)
        self._bind_key("Shift-Home", self.beginning_of_line_extend_selection)
        self._bind_key("Shift-End", self.end_of_line_extend_selection)
        self._bind_key("numpad0", self.self_insert)
        self._bind_key("numpad1", self.self_insert)
        self._bind_key("numpad2", self.self_insert)
        self._bind_key("numpad3", self.self_insert)
        self._bind_key("numpad4", self.self_insert)
        self._bind_key("numpad5", self.self_insert)
        self._bind_key("numpad6", self.self_insert)
        self._bind_key("numpad7", self.self_insert)
        self._bind_key("numpad8", self.self_insert)
        self._bind_key("numpad9", self.self_insert)
        self._bind_key("add", self.self_insert)
        self._bind_key("subtract", self.self_insert)
        self._bind_key("multiply", self.self_insert)
        self._bind_key("divide", self.self_insert)
        self._bind_key("vk_decimal", self.self_insert)
        log("RUNNING INIT EMACS")
        for i in range(0, 10):
            self._bind_key("alt-%d" % i, self.digit_argument)
        self._bind_key("alt--", self.digit_argument)


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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\handlers\matrices.py ===
"""
This module contains query handlers responsible for Matrices queries:
Square, Symmetric, Invertible etc.
"""

from sympy.logic.boolalg import conjuncts
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import test_closed_group
from sympy.matrices import MatrixBase
from sympy.matrices.expressions import (BlockMatrix, BlockDiagMatrix, Determinant,
    DiagMatrix, DiagonalMatrix, HadamardProduct, Identity, Inverse, MatAdd, MatMul,
    MatPow, MatrixExpr, MatrixSlice, MatrixSymbol, OneMatrix, Trace, Transpose,
    ZeroMatrix)
from sympy.matrices.expressions.blockmatrix import reblock_2x2
from sympy.matrices.expressions.factorizations import Factorization
from sympy.matrices.expressions.fourier import DFT
from sympy.core.logic import fuzzy_and
from sympy.utilities.iterables import sift
from sympy.core import Basic

from ..predicates.matrices import (SquarePredicate, SymmetricPredicate,
    InvertiblePredicate, OrthogonalPredicate, UnitaryPredicate,
    FullRankPredicate, PositiveDefinitePredicate, UpperTriangularPredicate,
    LowerTriangularPredicate, DiagonalPredicate, IntegerElementsPredicate,
    RealElementsPredicate, ComplexElementsPredicate)


def _Factorization(predicate, expr, assumptions):
    if predicate in expr.predicates:
        return True


# SquarePredicate

@SquarePredicate.register(MatrixExpr)
def _(expr, assumptions):
    return expr.shape[0] == expr.shape[1]


# SymmetricPredicate

@SymmetricPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if all(ask(Q.symmetric(arg), assumptions) for arg in mmul.args):
        return True
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if len(mmul.args) >= 2 and mmul.args[0] == mmul.args[-1].T:
        if len(mmul.args) == 2:
            return True
        return ask(Q.symmetric(MatMul(*mmul.args[1:-1])), assumptions)

@SymmetricPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.symmetric(base), assumptions)
    return None

@SymmetricPredicate.register(MatAdd)
def _(expr, assumptions):
    return all(ask(Q.symmetric(arg), assumptions) for arg in expr.args)

@SymmetricPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if Q.symmetric(expr) in conjuncts(assumptions):
        return True

@SymmetricPredicate.register_many(OneMatrix, ZeroMatrix)
def _(expr, assumptions):
    return ask(Q.square(expr), assumptions)

@SymmetricPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.symmetric(expr.arg), assumptions)

@SymmetricPredicate.register(MatrixSlice)
def _(expr, assumptions):
    # TODO: implement sathandlers system for the matrices.
    # Now it duplicates the general fact: Implies(Q.diagonal, Q.symmetric).
    if ask(Q.diagonal(expr), assumptions):
        return True
    if not expr.on_diag:
        return None
    else:
        return ask(Q.symmetric(expr.parent), assumptions)

@SymmetricPredicate.register(Identity)
def _(expr, assumptions):
    return True


# InvertiblePredicate

@InvertiblePredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if all(ask(Q.invertible(arg), assumptions) for arg in mmul.args):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@InvertiblePredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    if exp.is_negative == False:
        return ask(Q.invertible(base), assumptions)
    return None

@InvertiblePredicate.register(MatAdd)
def _(expr, assumptions):
    return None

@InvertiblePredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if Q.invertible(expr) in conjuncts(assumptions):
        return True

@InvertiblePredicate.register_many(Identity, Inverse)
def _(expr, assumptions):
    return True

@InvertiblePredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@InvertiblePredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@InvertiblePredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.invertible(expr.arg), assumptions)

@InvertiblePredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.invertible(expr.parent), assumptions)

@InvertiblePredicate.register(MatrixBase)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    return expr.rank() == expr.rows

@InvertiblePredicate.register(MatrixExpr)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    return None

@InvertiblePredicate.register(BlockMatrix)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if expr.blockshape == (1, 1):
        return ask(Q.invertible(expr.blocks[0, 0]), assumptions)
    expr = reblock_2x2(expr)
    if expr.blockshape == (2, 2):
        [[A, B], [C, D]] = expr.blocks.tolist()
        if ask(Q.invertible(A), assumptions) == True:
            invertible = ask(Q.invertible(D - C * A.I * B), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(B), assumptions) == True:
            invertible = ask(Q.invertible(C - D * B.I * A), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(C), assumptions) == True:
            invertible = ask(Q.invertible(B - A * C.I * D), assumptions)
            if invertible is not None:
                return invertible
        if ask(Q.invertible(D), assumptions) == True:
            invertible = ask(Q.invertible(A - B * D.I * C), assumptions)
            if invertible is not None:
                return invertible
    return None

@InvertiblePredicate.register(BlockDiagMatrix)
def _(expr, assumptions):
    if expr.rowblocksizes != expr.colblocksizes:
        return None
    return fuzzy_and([ask(Q.invertible(a), assumptions) for a in expr.diag])


# OrthogonalPredicate

@OrthogonalPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.orthogonal(arg), assumptions) for arg in mmul.args) and
            factor == 1):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@OrthogonalPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp:
        return ask(Q.orthogonal(base), assumptions)
    return None

@OrthogonalPredicate.register(MatAdd)
def _(expr, assumptions):
    if (len(expr.args) == 1 and
            ask(Q.orthogonal(expr.args[0]), assumptions)):
        return True

@OrthogonalPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if (not expr.is_square or
                    ask(Q.invertible(expr), assumptions) is False):
        return False
    if Q.orthogonal(expr) in conjuncts(assumptions):
        return True

@OrthogonalPredicate.register(Identity)
def _(expr, assumptions):
    return True

@OrthogonalPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@OrthogonalPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.orthogonal(expr.arg), assumptions)

@OrthogonalPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.orthogonal(expr.parent), assumptions)

@OrthogonalPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.orthogonal, expr, assumptions)


# UnitaryPredicate

@UnitaryPredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.unitary(arg), assumptions) for arg in mmul.args) and
            abs(factor) == 1):
        return True
    if any(ask(Q.invertible(arg), assumptions) is False
            for arg in mmul.args):
        return False

@UnitaryPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp:
        return ask(Q.unitary(base), assumptions)
    return None

@UnitaryPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if (not expr.is_square or
                    ask(Q.invertible(expr), assumptions) is False):
        return False
    if Q.unitary(expr) in conjuncts(assumptions):
        return True

@UnitaryPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.unitary(expr.arg), assumptions)

@UnitaryPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.unitary(expr.parent), assumptions)

@UnitaryPredicate.register_many(DFT, Identity)
def _(expr, assumptions):
    return True

@UnitaryPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@UnitaryPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.unitary, expr, assumptions)


# FullRankPredicate

@FullRankPredicate.register(MatMul)
def _(expr, assumptions):
    if all(ask(Q.fullrank(arg), assumptions) for arg in expr.args):
        return True

@FullRankPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if int_exp and ask(~Q.negative(exp), assumptions):
        return ask(Q.fullrank(base), assumptions)
    return None

@FullRankPredicate.register(Identity)
def _(expr, assumptions):
    return True

@FullRankPredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@FullRankPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@FullRankPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.fullrank(expr.arg), assumptions)

@FullRankPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if ask(Q.orthogonal(expr.parent), assumptions):
        return True


# PositiveDefinitePredicate

@PositiveDefinitePredicate.register(MatMul)
def _(expr, assumptions):
    factor, mmul = expr.as_coeff_mmul()
    if (all(ask(Q.positive_definite(arg), assumptions)
            for arg in mmul.args) and factor > 0):
        return True
    if (len(mmul.args) >= 2
            and mmul.args[0] == mmul.args[-1].T
            and ask(Q.fullrank(mmul.args[0]), assumptions)):
        return ask(Q.positive_definite(
            MatMul(*mmul.args[1:-1])), assumptions)

@PositiveDefinitePredicate.register(MatPow)
def _(expr, assumptions):
    # a power of a positive definite matrix is positive definite
    if ask(Q.positive_definite(expr.args[0]), assumptions):
        return True

@PositiveDefinitePredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.positive_definite(arg), assumptions)
            for arg in expr.args):
        return True

@PositiveDefinitePredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if not expr.is_square:
        return False
    if Q.positive_definite(expr) in conjuncts(assumptions):
        return True

@PositiveDefinitePredicate.register(Identity)
def _(expr, assumptions):
    return True

@PositiveDefinitePredicate.register(ZeroMatrix)
def _(expr, assumptions):
    return False

@PositiveDefinitePredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@PositiveDefinitePredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.positive_definite(expr.arg), assumptions)

@PositiveDefinitePredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.positive_definite(expr.parent), assumptions)


# UpperTriangularPredicate

@UpperTriangularPredicate.register(MatMul)
def _(expr, assumptions):
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.upper_triangular(m), assumptions) for m in matrices):
        return True

@UpperTriangularPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.upper_triangular(arg), assumptions) for arg in expr.args):
        return True

@UpperTriangularPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.upper_triangular(base), assumptions)
    return None

@UpperTriangularPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if Q.upper_triangular(expr) in conjuncts(assumptions):
        return True

@UpperTriangularPredicate.register_many(Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@UpperTriangularPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@UpperTriangularPredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.lower_triangular(expr.arg), assumptions)

@UpperTriangularPredicate.register(Inverse)
def _(expr, assumptions):
    return ask(Q.upper_triangular(expr.arg), assumptions)

@UpperTriangularPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.upper_triangular(expr.parent), assumptions)

@UpperTriangularPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.upper_triangular, expr, assumptions)

# LowerTriangularPredicate

@LowerTriangularPredicate.register(MatMul)
def _(expr, assumptions):
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.lower_triangular(m), assumptions) for m in matrices):
        return True

@LowerTriangularPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.lower_triangular(arg), assumptions) for arg in expr.args):
        return True

@LowerTriangularPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.lower_triangular(base), assumptions)
    return None

@LowerTriangularPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if Q.lower_triangular(expr) in conjuncts(assumptions):
        return True

@LowerTriangularPredicate.register_many(Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@LowerTriangularPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@LowerTriangularPredicate.register(Transpose)
def _(expr, assumptions):
    return ask(Q.upper_triangular(expr.arg), assumptions)

@LowerTriangularPredicate.register(Inverse)
def _(expr, assumptions):
    return ask(Q.lower_triangular(expr.arg), assumptions)

@LowerTriangularPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if not expr.on_diag:
        return None
    else:
        return ask(Q.lower_triangular(expr.parent), assumptions)

@LowerTriangularPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.lower_triangular, expr, assumptions)


# DiagonalPredicate

def _is_empty_or_1x1(expr):
    return expr.shape in ((0, 0), (1, 1))

@DiagonalPredicate.register(MatMul)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    factor, matrices = expr.as_coeff_matrices()
    if all(ask(Q.diagonal(m), assumptions) for m in matrices):
        return True

@DiagonalPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.diagonal(base), assumptions)
    return None

@DiagonalPredicate.register(MatAdd)
def _(expr, assumptions):
    if all(ask(Q.diagonal(arg), assumptions) for arg in expr.args):
        return True

@DiagonalPredicate.register(MatrixSymbol)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    if Q.diagonal(expr) in conjuncts(assumptions):
        return True

@DiagonalPredicate.register(OneMatrix)
def _(expr, assumptions):
    return expr.shape[0] == 1 and expr.shape[1] == 1

@DiagonalPredicate.register_many(Inverse, Transpose)
def _(expr, assumptions):
    return ask(Q.diagonal(expr.arg), assumptions)

@DiagonalPredicate.register(MatrixSlice)
def _(expr, assumptions):
    if _is_empty_or_1x1(expr):
        return True
    if not expr.on_diag:
        return None
    else:
        return ask(Q.diagonal(expr.parent), assumptions)

@DiagonalPredicate.register_many(DiagonalMatrix, DiagMatrix, Identity, ZeroMatrix)
def _(expr, assumptions):
    return True

@DiagonalPredicate.register(Factorization)
def _(expr, assumptions):
    return _Factorization(Q.diagonal, expr, assumptions)


# IntegerElementsPredicate

def BM_elements(predicate, expr, assumptions):
    """ Block Matrix elements. """
    return all(ask(predicate(b), assumptions) for b in expr.blocks)

def MS_elements(predicate, expr, assumptions):
    """ Matrix Slice elements. """
    return ask(predicate(expr.parent), assumptions)

def MatMul_elements(matrix_predicate, scalar_predicate, expr, assumptions):
    d = sift(expr.args, lambda x: isinstance(x, MatrixExpr))
    factors, matrices = d[False], d[True]
    return fuzzy_and([
        test_closed_group(Basic(*factors), assumptions, scalar_predicate),
        test_closed_group(Basic(*matrices), assumptions, matrix_predicate)])


@IntegerElementsPredicate.register_many(Determinant, HadamardProduct, MatAdd,
    Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.integer_elements)

@IntegerElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    if exp.is_negative == False:
        return ask(Q.integer_elements(base), assumptions)
    return None

@IntegerElementsPredicate.register_many(Identity, OneMatrix, ZeroMatrix)
def _(expr, assumptions):
    return True

@IntegerElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.integer_elements, Q.integer, expr, assumptions)

@IntegerElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.integer_elements, expr, assumptions)

@IntegerElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.integer_elements, expr, assumptions)


# RealElementsPredicate

@RealElementsPredicate.register_many(Determinant, Factorization, HadamardProduct,
    MatAdd, Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.real_elements)

@RealElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.real_elements(base), assumptions)
    return None

@RealElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.real_elements, Q.real, expr, assumptions)

@RealElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.real_elements, expr, assumptions)

@RealElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.real_elements, expr, assumptions)


# ComplexElementsPredicate

@ComplexElementsPredicate.register_many(Determinant, Factorization, HadamardProduct,
    Inverse, MatAdd, Trace, Transpose)
def _(expr, assumptions):
    return test_closed_group(expr, assumptions, Q.complex_elements)

@ComplexElementsPredicate.register(MatPow)
def _(expr, assumptions):
    # only for integer powers
    base, exp = expr.args
    int_exp = ask(Q.integer(exp), assumptions)
    if not int_exp:
        return None
    non_negative = ask(~Q.negative(exp), assumptions)
    if (non_negative or non_negative == False
                        and ask(Q.invertible(base), assumptions)):
        return ask(Q.complex_elements(base), assumptions)
    return None

@ComplexElementsPredicate.register(MatMul)
def _(expr, assumptions):
    return MatMul_elements(Q.complex_elements, Q.complex, expr, assumptions)

@ComplexElementsPredicate.register(MatrixSlice)
def _(expr, assumptions):
    return MS_elements(Q.complex_elements, expr, assumptions)

@ComplexElementsPredicate.register(BlockMatrix)
def _(expr, assumptions):
    return BM_elements(Q.complex_elements, expr, assumptions)

@ComplexElementsPredicate.register(DFT)
def _(expr, assumptions):
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\hep\gamma_matrices.py ===
"""
    Module to handle gamma matrices expressed as tensor objects.

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i = tensor_indices('i', LorentzIndex)
    >>> G(i)
    GammaMatrix(i)

    Note that there is already an instance of GammaMatrixHead in four dimensions:
    GammaMatrix, which is simply declare as

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i = tensor_indices('i', LorentzIndex)
    >>> GammaMatrix(i)
    GammaMatrix(i)

    To access the metric tensor

    >>> LorentzIndex.metric
    metric(LorentzIndex,LorentzIndex)

"""
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.dense import eye
from sympy.matrices.expressions.trace import trace
from sympy.tensor.tensor import TensorIndexType, TensorIndex,\
    TensMul, TensAdd, tensor_mul, Tensor, TensorHead, TensorSymmetry


# DiracSpinorIndex = TensorIndexType('DiracSpinorIndex', dim=4, dummy_name="S")


LorentzIndex = TensorIndexType('LorentzIndex', dim=4, dummy_name="L")


GammaMatrix = TensorHead("GammaMatrix", [LorentzIndex],
                         TensorSymmetry.no_symmetry(1), comm=None)


def extract_type_tens(expression, component):
    """
    Extract from a ``TensExpr`` all tensors with `component`.

    Returns two tensor expressions:

    * the first contains all ``Tensor`` of having `component`.
    * the second contains all remaining.


    """
    if isinstance(expression, Tensor):
        sp = [expression]
    elif isinstance(expression, TensMul):
        sp = expression.args
    else:
        raise ValueError('wrong type')

    # Collect all gamma matrices of the same dimension
    new_expr = S.One
    residual_expr = S.One
    for i in sp:
        if isinstance(i, Tensor) and i.component == component:
            new_expr *= i
        else:
            residual_expr *= i
    return new_expr, residual_expr


def simplify_gamma_expression(expression):
    extracted_expr, residual_expr = extract_type_tens(expression, GammaMatrix)
    res_expr = _simplify_single_line(extracted_expr)
    return res_expr * residual_expr


def simplify_gpgp(ex, sort=True):
    """
    simplify products ``G(i)*p(-i)*G(j)*p(-j) -> p(i)*p(-i)``

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, simplify_gpgp
    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads
    >>> p, q = tensor_heads('p, q', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> ps = p(i0)*G(-i0)
    >>> qs = q(i0)*G(-i0)
    >>> simplify_gpgp(ps*qs*qs)
    GammaMatrix(-L_0)*p(L_0)*q(L_1)*q(-L_1)
    """
    def _simplify_gpgp(ex):
        components = ex.components
        a = []
        comp_map = []
        for i, comp in enumerate(components):
            comp_map.extend([i]*comp.rank)
        dum = [(i[0], i[1], comp_map[i[0]], comp_map[i[1]]) for i in ex.dum]
        for i in range(len(components)):
            if components[i] != GammaMatrix:
                continue
            for dx in dum:
                if dx[2] == i:
                    p_pos1 = dx[3]
                elif dx[3] == i:
                    p_pos1 = dx[2]
                else:
                    continue
                comp1 = components[p_pos1]
                if comp1.comm == 0 and comp1.rank == 1:
                    a.append((i, p_pos1))
        if not a:
            return ex
        elim = set()
        tv = []
        hit = True
        coeff = S.One
        ta = None
        while hit:
            hit = False
            for i, ai in enumerate(a[:-1]):
                if ai[0] in elim:
                    continue
                if ai[0] != a[i + 1][0] - 1:
                    continue
                if components[ai[1]] != components[a[i + 1][1]]:
                    continue
                elim.add(ai[0])
                elim.add(ai[1])
                elim.add(a[i + 1][0])
                elim.add(a[i + 1][1])
                if not ta:
                    ta = ex.split()
                    mu = TensorIndex('mu', LorentzIndex)
                hit = True
                if i == 0:
                    coeff = ex.coeff
                tx = components[ai[1]](mu)*components[ai[1]](-mu)
                if len(a) == 2:
                    tx *= 4  # eye(4)
                tv.append(tx)
                break

        if tv:
            a = [x for j, x in enumerate(ta) if j not in elim]
            a.extend(tv)
            t = tensor_mul(*a)*coeff
            # t = t.replace(lambda x: x.is_Matrix, lambda x: 1)
            return t
        else:
            return ex

    if sort:
        ex = ex.sorted_components()
    # this would be better off with pattern matching
    while 1:
        t = _simplify_gpgp(ex)
        if t != ex:
            ex = t
        else:
            return t


def gamma_trace(t):
    """
    trace of a single line of gamma matrices

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        gamma_trace, LorentzIndex
    >>> from sympy.tensor.tensor import tensor_indices, tensor_heads
    >>> p, q = tensor_heads('p, q', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> ps = p(i0)*G(-i0)
    >>> qs = q(i0)*G(-i0)
    >>> gamma_trace(G(i0)*G(i1))
    4*metric(i0, i1)
    >>> gamma_trace(ps*ps) - 4*p(i0)*p(-i0)
    0
    >>> gamma_trace(ps*qs + ps*ps) - 4*p(i0)*p(-i0) - 4*p(i0)*q(-i0)
    0

    """
    if isinstance(t, TensAdd):
        res = TensAdd(*[gamma_trace(x) for x in t.args])
        return res
    t = _simplify_single_line(t)
    res = _trace_single_line(t)
    return res


def _simplify_single_line(expression):
    """
    Simplify single-line product of gamma matrices.

    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, _simplify_single_line
    >>> from sympy.tensor.tensor import tensor_indices, TensorHead
    >>> p = TensorHead('p', [LorentzIndex])
    >>> i0,i1 = tensor_indices('i0:2', LorentzIndex)
    >>> _simplify_single_line(G(i0)*G(i1)*p(-i1)*G(-i0)) + 2*G(i0)*p(-i0)
    0

    """
    t1, t2 = extract_type_tens(expression, GammaMatrix)
    if t1 != 1:
        t1 = kahane_simplify(t1)
    res = t1*t2
    return res


def _trace_single_line(t):
    """
    Evaluate the trace of a single gamma matrix line inside a ``TensExpr``.

    Notes
    =====

    If there are ``DiracSpinorIndex.auto_left`` and ``DiracSpinorIndex.auto_right``
    indices trace over them; otherwise traces are not implied (explain)


    Examples
    ========

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, \
        LorentzIndex, _trace_single_line
    >>> from sympy.tensor.tensor import tensor_indices, TensorHead
    >>> p = TensorHead('p', [LorentzIndex])
    >>> i0,i1,i2,i3,i4,i5 = tensor_indices('i0:6', LorentzIndex)
    >>> _trace_single_line(G(i0)*G(i1))
    4*metric(i0, i1)
    >>> _trace_single_line(G(i0)*p(-i0)*G(i1)*p(-i1)) - 4*p(i0)*p(-i0)
    0

    """
    def _trace_single_line1(t):
        t = t.sorted_components()
        components = t.components
        ncomps = len(components)
        g = LorentzIndex.metric
        # gamma matirices are in a[i:j]
        hit = 0
        for i in range(ncomps):
            if components[i] == GammaMatrix:
                hit = 1
                break

        for j in range(i + hit, ncomps):
            if components[j] != GammaMatrix:
                break
        else:
            j = ncomps
        numG = j - i
        if numG == 0:
            tcoeff = t.coeff
            return t.nocoeff if tcoeff else t
        if numG % 2 == 1:
            return TensMul.from_data(S.Zero, [], [], [])
        elif numG > 4:
            # find the open matrix indices and connect them:
            a = t.split()
            ind1 = a[i].get_indices()[0]
            ind2 = a[i + 1].get_indices()[0]
            aa = a[:i] + a[i + 2:]
            t1 = tensor_mul(*aa)*g(ind1, ind2)
            t1 = t1.contract_metric(g)
            args = [t1]
            sign = 1
            for k in range(i + 2, j):
                sign = -sign
                ind2 = a[k].get_indices()[0]
                aa = a[:i] + a[i + 1:k] + a[k + 1:]
                t2 = sign*tensor_mul(*aa)*g(ind1, ind2)
                t2 = t2.contract_metric(g)
                t2 = simplify_gpgp(t2, False)
                args.append(t2)
            t3 = TensAdd(*args)
            t3 = _trace_single_line(t3)
            return t3
        else:
            a = t.split()
            t1 = _gamma_trace1(*a[i:j])
            a2 = a[:i] + a[j:]
            t2 = tensor_mul(*a2)
            t3 = t1*t2
            if not t3:
                return t3
            t3 = t3.contract_metric(g)
            return t3

    t = t.expand()
    if isinstance(t, TensAdd):
        a = [_trace_single_line1(x)*x.coeff for x in t.args]
        return TensAdd(*a)
    elif isinstance(t, (Tensor, TensMul)):
        r = t.coeff*_trace_single_line1(t)
        return r
    else:
        return trace(t)


def _gamma_trace1(*a):
    gctr = 4  # FIXME specific for d=4
    g = LorentzIndex.metric
    if not a:
        return gctr
    n = len(a)
    if n%2 == 1:
        #return TensMul.from_data(S.Zero, [], [], [])
        return S.Zero
    if n == 2:
        ind0 = a[0].get_indices()[0]
        ind1 = a[1].get_indices()[0]
        return gctr*g(ind0, ind1)
    if n == 4:
        ind0 = a[0].get_indices()[0]
        ind1 = a[1].get_indices()[0]
        ind2 = a[2].get_indices()[0]
        ind3 = a[3].get_indices()[0]

        return gctr*(g(ind0, ind1)*g(ind2, ind3) - \
           g(ind0, ind2)*g(ind1, ind3) + g(ind0, ind3)*g(ind1, ind2))


def kahane_simplify(expression):
    r"""
    This function cancels contracted elements in a product of four
    dimensional gamma matrices, resulting in an expression equal to the given
    one, without the contracted gamma matrices.

    Parameters
    ==========

    `expression`    the tensor expression containing the gamma matrices to simplify.

    Notes
    =====

    If spinor indices are given, the matrices must be given in
    the order given in the product.

    Algorithm
    =========

    The idea behind the algorithm is to use some well-known identities,
    i.e., for contractions enclosing an even number of `\gamma` matrices

    `\gamma^\mu \gamma_{a_1} \cdots \gamma_{a_{2N}} \gamma_\mu = 2 (\gamma_{a_{2N}} \gamma_{a_1} \cdots \gamma_{a_{2N-1}} + \gamma_{a_{2N-1}} \cdots \gamma_{a_1} \gamma_{a_{2N}} )`

    for an odd number of `\gamma` matrices

    `\gamma^\mu \gamma_{a_1} \cdots \gamma_{a_{2N+1}} \gamma_\mu = -2 \gamma_{a_{2N+1}} \gamma_{a_{2N}} \cdots \gamma_{a_{1}}`

    Instead of repeatedly applying these identities to cancel out all contracted indices,
    it is possible to recognize the links that would result from such an operation,
    the problem is thus reduced to a simple rearrangement of free gamma matrices.

    Examples
    ========

    When using, always remember that the original expression coefficient
    has to be handled separately

    >>> from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    >>> from sympy.physics.hep.gamma_matrices import kahane_simplify
    >>> from sympy.tensor.tensor import tensor_indices
    >>> i0, i1, i2 = tensor_indices('i0:3', LorentzIndex)
    >>> ta = G(i0)*G(-i0)
    >>> kahane_simplify(ta)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> tb = G(i0)*G(i1)*G(-i0)
    >>> kahane_simplify(tb)
    -2*GammaMatrix(i1)
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])
    >>> t = G(i0)*G(-i0)
    >>> kahane_simplify(t)
    Matrix([
    [4, 0, 0, 0],
    [0, 4, 0, 0],
    [0, 0, 4, 0],
    [0, 0, 0, 4]])

    If there are no contractions, the same expression is returned

    >>> tc = G(i0)*G(i1)
    >>> kahane_simplify(tc)
    GammaMatrix(i0)*GammaMatrix(i1)

    References
    ==========

    [1] Algorithm for Reducing Contracted Products of gamma Matrices,
    Joseph Kahane, Journal of Mathematical Physics, Vol. 9, No. 10, October 1968.
    """

    if isinstance(expression, Mul):
        return expression
    if isinstance(expression, TensAdd):
        return TensAdd(*[kahane_simplify(arg) for arg in expression.args])

    if isinstance(expression, Tensor):
        return expression

    assert isinstance(expression, TensMul)

    gammas = expression.args

    for gamma in gammas:
        assert gamma.component == GammaMatrix

    free = expression.free
    # spinor_free = [_ for _ in expression.free_in_args if _[1] != 0]

    # if len(spinor_free) == 2:
    #     spinor_free.sort(key=lambda x: x[2])
    #     assert spinor_free[0][1] == 1 and spinor_free[-1][1] == 2
    #     assert spinor_free[0][2] == 0
    # elif spinor_free:
    #     raise ValueError('spinor indices do not match')

    dum = []
    for dum_pair in expression.dum:
        if expression.index_types[dum_pair[0]] == LorentzIndex:
            dum.append((dum_pair[0], dum_pair[1]))

    dum = sorted(dum)

    if len(dum) == 0:  # or GammaMatrixHead:
        # no contractions in `expression`, just return it.
        return expression

    # find the `first_dum_pos`, i.e. the position of the first contracted
    # gamma matrix, Kahane's algorithm as described in his paper requires the
    # gamma matrix expression to start with a contracted gamma matrix, this is
    # a workaround which ignores possible initial free indices, and re-adds
    # them later.

    first_dum_pos = min(map(min, dum))

    # for p1, p2, a1, a2 in expression.dum_in_args:
    #     if p1 != 0 or p2 != 0:
    #         # only Lorentz indices, skip Dirac indices:
    #         continue
    #     first_dum_pos = min(p1, p2)
    #     break

    total_number = len(free) + len(dum)*2
    number_of_contractions = len(dum)

    free_pos = [None]*total_number
    for i in free:
        free_pos[i[1]] = i[0]

    # `index_is_free` is a list of booleans, to identify index position
    # and whether that index is free or dummy.
    index_is_free = [False]*total_number

    for i, indx in enumerate(free):
        index_is_free[indx[1]] = True

    # `links` is a dictionary containing the graph described in Kahane's paper,
    # to every key correspond one or two values, representing the linked indices.
    # All values in `links` are integers, negative numbers are used in the case
    # where it is necessary to insert gamma matrices between free indices, in
    # order to make Kahane's algorithm work (see paper).
    links = {i: [] for i in range(first_dum_pos, total_number)}

    # `cum_sign` is a step variable to mark the sign of every index, see paper.
    cum_sign = -1
    # `cum_sign_list` keeps storage for all `cum_sign` (every index).
    cum_sign_list = [None]*total_number
    block_free_count = 0

    # multiply `resulting_coeff` by the coefficient parameter, the rest
    # of the algorithm ignores a scalar coefficient.
    resulting_coeff = S.One

    # initialize a list of lists of indices. The outer list will contain all
    # additive tensor expressions, while the inner list will contain the
    # free indices (rearranged according to the algorithm).
    resulting_indices = [[]]

    # start to count the `connected_components`, which together with the number
    # of contractions, determines a -1 or +1 factor to be multiplied.
    connected_components = 1

    # First loop: here we fill `cum_sign_list`, and draw the links
    # among consecutive indices (they are stored in `links`). Links among
    # non-consecutive indices will be drawn later.
    for i, is_free in enumerate(index_is_free):
        # if `expression` starts with free indices, they are ignored here;
        # they are later added as they are to the beginning of all
        # `resulting_indices` list of lists of indices.
        if i < first_dum_pos:
            continue

        if is_free:
            block_free_count += 1
            # if previous index was free as well, draw an arch in `links`.
            if block_free_count > 1:
                links[i - 1].append(i)
                links[i].append(i - 1)
        else:
            # Change the sign of the index (`cum_sign`) if the number of free
            # indices preceding it is even.
            cum_sign *= 1 if (block_free_count % 2) else -1
            if block_free_count == 0 and i != first_dum_pos:
                # check if there are two consecutive dummy indices:
                # in this case create virtual indices with negative position,
                # these "virtual" indices represent the insertion of two
                # gamma^0 matrices to separate consecutive dummy indices, as
                # Kahane's algorithm requires dummy indices to be separated by
                # free indices. The product of two gamma^0 matrices is unity,
                # so the new expression being examined is the same as the
                # original one.
                if cum_sign == -1:
                    links[-1-i] = [-1-i+1]
                    links[-1-i+1] = [-1-i]
            if (i - cum_sign) in links:
                if i != first_dum_pos:
                    links[i].append(i - cum_sign)
                if block_free_count != 0:
                    if i - cum_sign < len(index_is_free):
                        if index_is_free[i - cum_sign]:
                            links[i - cum_sign].append(i)
            block_free_count = 0

        cum_sign_list[i] = cum_sign

    # The previous loop has only created links between consecutive free indices,
    # it is necessary to properly create links among dummy (contracted) indices,
    # according to the rules described in Kahane's paper. There is only one exception
    # to Kahane's rules: the negative indices, which handle the case of some
    # consecutive free indices (Kahane's paper just describes dummy indices
    # separated by free indices, hinting that free indices can be added without
    # altering the expression result).
    for i in dum:
        # get the positions of the two contracted indices:
        pos1 = i[0]
        pos2 = i[1]

        # create Kahane's upper links, i.e. the upper arcs between dummy
        # (i.e. contracted) indices:
        links[pos1].append(pos2)
        links[pos2].append(pos1)

        # create Kahane's lower links, this corresponds to the arcs below
        # the line described in the paper:

        # first we move `pos1` and `pos2` according to the sign of the indices:
        linkpos1 = pos1 + cum_sign_list[pos1]
        linkpos2 = pos2 + cum_sign_list[pos2]

        # otherwise, perform some checks before creating the lower arcs:

        # make sure we are not exceeding the total number of indices:
        if linkpos1 >= total_number:
            continue
        if linkpos2 >= total_number:
            continue

        # make sure we are not below the first dummy index in `expression`:
        if linkpos1 < first_dum_pos:
            continue
        if linkpos2 < first_dum_pos:
            continue

        # check if the previous loop created "virtual" indices between dummy
        # indices, in such a case relink `linkpos1` and `linkpos2`:
        if (-1-linkpos1) in links:
            linkpos1 = -1-linkpos1
        if (-1-linkpos2) in links:
            linkpos2 = -1-linkpos2

        # move only if not next to free index:
        if linkpos1 >= 0 and not index_is_free[linkpos1]:
            linkpos1 = pos1

        if linkpos2 >=0 and not index_is_free[linkpos2]:
            linkpos2 = pos2

        # create the lower arcs:
        if linkpos2 not in links[linkpos1]:
            links[linkpos1].append(linkpos2)
        if linkpos1 not in links[linkpos2]:
            links[linkpos2].append(linkpos1)

    # This loop starts from the `first_dum_pos` index (first dummy index)
    # walks through the graph deleting the visited indices from `links`,
    # it adds a gamma matrix for every free index in encounters, while it
    # completely ignores dummy indices and virtual indices.
    pointer = first_dum_pos
    previous_pointer = 0
    while True:
        if pointer in links:
            next_ones = links.pop(pointer)
        else:
            break

        if previous_pointer in next_ones:
            next_ones.remove(previous_pointer)

        previous_pointer = pointer

        if next_ones:
            pointer = next_ones[0]
        else:
            break

        if pointer == previous_pointer:
            break
        if pointer >=0 and free_pos[pointer] is not None:
            for ri in resulting_indices:
                ri.append(free_pos[pointer])

    # The following loop removes the remaining connected components in `links`.
    # If there are free indices inside a connected component, it gives a
    # contribution to the resulting expression given by the factor
    # `gamma_a gamma_b ... gamma_z + gamma_z ... gamma_b gamma_a`, in Kahanes's
    # paper represented as  {gamma_a, gamma_b, ... , gamma_z},
    # virtual indices are ignored. The variable `connected_components` is
    # increased by one for every connected component this loop encounters.

    # If the connected component has virtual and dummy indices only
    # (no free indices), it contributes to `resulting_indices` by a factor of two.
    # The multiplication by two is a result of the
    # factor {gamma^0, gamma^0} = 2 I, as it appears in Kahane's paper.
    # Note: curly brackets are meant as in the paper, as a generalized
    # multi-element anticommutator!

    while links:
        connected_components += 1
        pointer = min(links.keys())
        previous_pointer = pointer
        # the inner loop erases the visited indices from `links`, and it adds
        # all free indices to `prepend_indices` list, virtual indices are
        # ignored.
        prepend_indices = []
        while True:
            if pointer in links:
                next_ones = links.pop(pointer)
            else:
                break

            if previous_pointer in next_ones:
                if len(next_ones) > 1:
                    next_ones.remove(previous_pointer)

            previous_pointer = pointer

            if next_ones:
                pointer = next_ones[0]

            if pointer >= first_dum_pos and free_pos[pointer] is not None:
                prepend_indices.insert(0, free_pos[pointer])
        # if `prepend_indices` is void, it means there are no free indices
        # in the loop (and it can be shown that there must be a virtual index),
        # loops of virtual indices only contribute by a factor of two:
        if len(prepend_indices) == 0:
            resulting_coeff *= 2
        # otherwise, add the free indices in `prepend_indices` to
        # the `resulting_indices`:
        else:
            expr1 = prepend_indices
            expr2 = list(reversed(prepend_indices))
            resulting_indices = [expri + ri for ri in resulting_indices for expri in (expr1, expr2)]

    # sign correction, as described in Kahane's paper:
    resulting_coeff *= -1 if (number_of_contractions - connected_components + 1) % 2 else 1
    # power of two factor, as described in Kahane's paper:
    resulting_coeff *= 2**(number_of_contractions)

    # If `first_dum_pos` is not zero, it means that there are trailing free gamma
    # matrices in front of `expression`, so multiply by them:
    resulting_indices = [ free_pos[0:first_dum_pos] + ri for ri in resulting_indices ]

    resulting_expr = S.Zero
    for i in resulting_indices:
        temp_expr = S.One
        for j in i:
            temp_expr *= GammaMatrix(j)
        resulting_expr += temp_expr

    t = resulting_coeff * resulting_expr
    t1 = None
    if isinstance(t, TensAdd):
        t1 = t.args[0]
    elif isinstance(t, TensMul):
        t1 = t
    if t1:
        pass
    else:
        t = eye(4)*t
    return t

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\_selectable_constructors.py ===
# sql/_selectable_constructors.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from . import coercions
from . import roles
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from .elements import ColumnClause
from .selectable import Alias
from .selectable import CompoundSelect
from .selectable import Exists
from .selectable import FromClause
from .selectable import Join
from .selectable import Lateral
from .selectable import LateralFromClause
from .selectable import NamedFromClause
from .selectable import Select
from .selectable import TableClause
from .selectable import TableSample
from .selectable import Values

if TYPE_CHECKING:
    from ._typing import _FromClauseArgument
    from ._typing import _OnClauseArgument
    from ._typing import _SelectStatementForCompoundArgument
    from ._typing import _T0
    from ._typing import _T1
    from ._typing import _T2
    from ._typing import _T3
    from ._typing import _T4
    from ._typing import _T5
    from ._typing import _T6
    from ._typing import _T7
    from ._typing import _T8
    from ._typing import _T9
    from ._typing import _TP
    from ._typing import _TypedColumnClauseArgument as _TCCA
    from .functions import Function
    from .selectable import CTE
    from .selectable import HasCTE
    from .selectable import ScalarSelect
    from .selectable import SelectBase


def alias(
    selectable: FromClause, name: Optional[str] = None, flat: bool = False
) -> NamedFromClause:
    """Return a named alias of the given :class:`.FromClause`.

    For :class:`.Table` and :class:`.Join` objects, the return type is the
    :class:`_expression.Alias` object. Other kinds of :class:`.NamedFromClause`
    objects may be returned for other kinds of :class:`.FromClause` objects.

    The named alias represents any :class:`_expression.FromClause` with an
    alternate name assigned within SQL, typically using the ``AS`` clause when
    generated, e.g. ``SELECT * FROM table AS aliasname``.

    Equivalent functionality is available via the
    :meth:`_expression.FromClause.alias`
    method available on all :class:`_expression.FromClause` objects.

    :param selectable: any :class:`_expression.FromClause` subclass,
        such as a table, select statement, etc.

    :param name: string name to be assigned as the alias.
        If ``None``, a name will be deterministically generated at compile
        time. Deterministic means the name is guaranteed to be unique against
        other constructs used in the same statement, and will also be the same
        name for each successive compilation of the same statement object.

    :param flat: Will be passed through to if the given selectable
     is an instance of :class:`_expression.Join` - see
     :meth:`_expression.Join.alias` for details.

    """
    return Alias._factory(selectable, name=name, flat=flat)


def cte(
    selectable: HasCTE, name: Optional[str] = None, recursive: bool = False
) -> CTE:
    r"""Return a new :class:`_expression.CTE`,
    or Common Table Expression instance.

    Please see :meth:`_expression.HasCTE.cte` for detail on CTE usage.

    """
    return coercions.expect(roles.HasCTERole, selectable).cte(
        name=name, recursive=recursive
    )


# TODO: mypy requires the _TypedSelectable overloads in all compound select
# constructors since _SelectStatementForCompoundArgument includes
# untyped args that make it return CompoundSelect[Unpack[tuple[Never, ...]]]
# pyright does not have this issue
_TypedSelectable = Union["Select[_TP]", "CompoundSelect[_TP]"]


@overload
def except_(
    *selects: _TypedSelectable[_TP],
) -> CompoundSelect[_TP]: ...


@overload
def except_(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]: ...


def except_(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]:
    r"""Return an ``EXCEPT`` of multiple selectables.

    The returned object is an instance of
    :class:`_expression.CompoundSelect`.

    :param \*selects:
      a list of :class:`_expression.Select` instances.

    """
    return CompoundSelect._create_except(*selects)


@overload
def except_all(
    *selects: _TypedSelectable[_TP],
) -> CompoundSelect[_TP]: ...


@overload
def except_all(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]: ...


def except_all(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]:
    r"""Return an ``EXCEPT ALL`` of multiple selectables.

    The returned object is an instance of
    :class:`_expression.CompoundSelect`.

    :param \*selects:
      a list of :class:`_expression.Select` instances.

    """
    return CompoundSelect._create_except_all(*selects)


def exists(
    __argument: Optional[
        Union[_ColumnsClauseArgument[Any], SelectBase, ScalarSelect[Any]]
    ] = None,
) -> Exists:
    """Construct a new :class:`_expression.Exists` construct.

    The :func:`_sql.exists` can be invoked by itself to produce an
    :class:`_sql.Exists` construct, which will accept simple WHERE
    criteria::

        exists_criteria = exists().where(table1.c.col1 == table2.c.col2)

    However, for greater flexibility in constructing the SELECT, an
    existing :class:`_sql.Select` construct may be converted to an
    :class:`_sql.Exists`, most conveniently by making use of the
    :meth:`_sql.SelectBase.exists` method::

        exists_criteria = (
            select(table2.c.col2).where(table1.c.col1 == table2.c.col2).exists()
        )

    The EXISTS criteria is then used inside of an enclosing SELECT::

        stmt = select(table1.c.col1).where(exists_criteria)

    The above statement will then be of the form:

    .. sourcecode:: sql

        SELECT col1 FROM table1 WHERE EXISTS
        (SELECT table2.col2 FROM table2 WHERE table2.col2 = table1.col1)

    .. seealso::

        :ref:`tutorial_exists` - in the :term:`2.0 style` tutorial.

        :meth:`_sql.SelectBase.exists` - method to transform a ``SELECT`` to an
        ``EXISTS`` clause.

    """  # noqa: E501

    return Exists(__argument)


@overload
def intersect(
    *selects: _TypedSelectable[_TP],
) -> CompoundSelect[_TP]: ...


@overload
def intersect(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]: ...


def intersect(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]:
    r"""Return an ``INTERSECT`` of multiple selectables.

    The returned object is an instance of
    :class:`_expression.CompoundSelect`.

    :param \*selects:
      a list of :class:`_expression.Select` instances.

    """
    return CompoundSelect._create_intersect(*selects)


@overload
def intersect_all(
    *selects: _TypedSelectable[_TP],
) -> CompoundSelect[_TP]: ...


@overload
def intersect_all(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]: ...


def intersect_all(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]:
    r"""Return an ``INTERSECT ALL`` of multiple selectables.

    The returned object is an instance of
    :class:`_expression.CompoundSelect`.

    :param \*selects:
      a list of :class:`_expression.Select` instances.


    """
    return CompoundSelect._create_intersect_all(*selects)


def join(
    left: _FromClauseArgument,
    right: _FromClauseArgument,
    onclause: Optional[_OnClauseArgument] = None,
    isouter: bool = False,
    full: bool = False,
) -> Join:
    """Produce a :class:`_expression.Join` object, given two
    :class:`_expression.FromClause`
    expressions.

    E.g.::

        j = join(
            user_table, address_table, user_table.c.id == address_table.c.user_id
        )
        stmt = select(user_table).select_from(j)

    would emit SQL along the lines of:

    .. sourcecode:: sql

        SELECT user.id, user.name FROM user
        JOIN address ON user.id = address.user_id

    Similar functionality is available given any
    :class:`_expression.FromClause` object (e.g. such as a
    :class:`_schema.Table`) using
    the :meth:`_expression.FromClause.join` method.

    :param left: The left side of the join.

    :param right: the right side of the join; this is any
     :class:`_expression.FromClause` object such as a
     :class:`_schema.Table` object, and
     may also be a selectable-compatible object such as an ORM-mapped
     class.

    :param onclause: a SQL expression representing the ON clause of the
     join.  If left at ``None``, :meth:`_expression.FromClause.join`
     will attempt to
     join the two tables based on a foreign key relationship.

    :param isouter: if True, render a LEFT OUTER JOIN, instead of JOIN.

    :param full: if True, render a FULL OUTER JOIN, instead of JOIN.

    .. seealso::

        :meth:`_expression.FromClause.join` - method form,
        based on a given left side.

        :class:`_expression.Join` - the type of object produced.

    """  # noqa: E501

    return Join(left, right, onclause, isouter, full)


def lateral(
    selectable: Union[SelectBase, _FromClauseArgument],
    name: Optional[str] = None,
) -> LateralFromClause:
    """Return a :class:`_expression.Lateral` object.

    :class:`_expression.Lateral` is an :class:`_expression.Alias`
    subclass that represents
    a subquery with the LATERAL keyword applied to it.

    The special behavior of a LATERAL subquery is that it appears in the
    FROM clause of an enclosing SELECT, but may correlate to other
    FROM clauses of that SELECT.   It is a special case of subquery
    only supported by a small number of backends, currently more recent
    PostgreSQL versions.

    .. seealso::

        :ref:`tutorial_lateral_correlation` -  overview of usage.

    """
    return Lateral._factory(selectable, name=name)


def outerjoin(
    left: _FromClauseArgument,
    right: _FromClauseArgument,
    onclause: Optional[_OnClauseArgument] = None,
    full: bool = False,
) -> Join:
    """Return an ``OUTER JOIN`` clause element.

    The returned object is an instance of :class:`_expression.Join`.

    Similar functionality is also available via the
    :meth:`_expression.FromClause.outerjoin` method on any
    :class:`_expression.FromClause`.

    :param left: The left side of the join.

    :param right: The right side of the join.

    :param onclause:  Optional criterion for the ``ON`` clause, is
      derived from foreign key relationships established between
      left and right otherwise.

    To chain joins together, use the :meth:`_expression.FromClause.join`
    or
    :meth:`_expression.FromClause.outerjoin` methods on the resulting
    :class:`_expression.Join` object.

    """
    return Join(left, right, onclause, isouter=True, full=full)


# START OVERLOADED FUNCTIONS select Select 1-10

# code within this block is **programmatically,
# statically generated** by tools/generate_tuple_map_overloads.py


@overload
def select(__ent0: _TCCA[_T0]) -> Select[Tuple[_T0]]: ...


@overload
def select(
    __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]
) -> Select[Tuple[_T0, _T1]]: ...


@overload
def select(
    __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]
) -> Select[Tuple[_T0, _T1, _T2]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
) -> Select[Tuple[_T0, _T1, _T2, _T3]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
    __ent4: _TCCA[_T4],
) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
    __ent4: _TCCA[_T4],
    __ent5: _TCCA[_T5],
) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
    __ent4: _TCCA[_T4],
    __ent5: _TCCA[_T5],
    __ent6: _TCCA[_T6],
) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
    __ent4: _TCCA[_T4],
    __ent5: _TCCA[_T5],
    __ent6: _TCCA[_T6],
    __ent7: _TCCA[_T7],
) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
    __ent4: _TCCA[_T4],
    __ent5: _TCCA[_T5],
    __ent6: _TCCA[_T6],
    __ent7: _TCCA[_T7],
    __ent8: _TCCA[_T8],
) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8]]: ...


@overload
def select(
    __ent0: _TCCA[_T0],
    __ent1: _TCCA[_T1],
    __ent2: _TCCA[_T2],
    __ent3: _TCCA[_T3],
    __ent4: _TCCA[_T4],
    __ent5: _TCCA[_T5],
    __ent6: _TCCA[_T6],
    __ent7: _TCCA[_T7],
    __ent8: _TCCA[_T8],
    __ent9: _TCCA[_T9],
) -> Select[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7, _T8, _T9]]: ...


# END OVERLOADED FUNCTIONS select


@overload
def select(
    *entities: _ColumnsClauseArgument[Any], **__kw: Any
) -> Select[Any]: ...


def select(*entities: _ColumnsClauseArgument[Any], **__kw: Any) -> Select[Any]:
    r"""Construct a new :class:`_expression.Select`.


    .. versionadded:: 1.4 - The :func:`_sql.select` function now accepts
       column arguments positionally.   The top-level :func:`_sql.select`
       function will automatically use the 1.x or 2.x style API based on
       the incoming arguments; using :func:`_sql.select` from the
       ``sqlalchemy.future`` module will enforce that only the 2.x style
       constructor is used.

    Similar functionality is also available via the
    :meth:`_expression.FromClause.select` method on any
    :class:`_expression.FromClause`.

    .. seealso::

        :ref:`tutorial_selecting_data` - in the :ref:`unified_tutorial`

    :param \*entities:
      Entities to SELECT from.  For Core usage, this is typically a series
      of :class:`_expression.ColumnElement` and / or
      :class:`_expression.FromClause`
      objects which will form the columns clause of the resulting
      statement.   For those objects that are instances of
      :class:`_expression.FromClause` (typically :class:`_schema.Table`
      or :class:`_expression.Alias`
      objects), the :attr:`_expression.FromClause.c`
      collection is extracted
      to form a collection of :class:`_expression.ColumnElement` objects.

      This parameter will also accept :class:`_expression.TextClause`
      constructs as
      given, as well as ORM-mapped classes.

    """
    # the keyword args are a necessary element in order for the typing
    # to work out w/ the varargs vs. having named "keyword" arguments that
    # aren't always present.
    if __kw:
        raise _no_kw()
    return Select(*entities)


def table(name: str, *columns: ColumnClause[Any], **kw: Any) -> TableClause:
    """Produce a new :class:`_expression.TableClause`.

    The object returned is an instance of
    :class:`_expression.TableClause`, which
    represents the "syntactical" portion of the schema-level
    :class:`_schema.Table` object.
    It may be used to construct lightweight table constructs.

    :param name: Name of the table.

    :param columns: A collection of :func:`_expression.column` constructs.

    :param schema: The schema name for this table.

        .. versionadded:: 1.3.18 :func:`_expression.table` can now
           accept a ``schema`` argument.
    """

    return TableClause(name, *columns, **kw)


def tablesample(
    selectable: _FromClauseArgument,
    sampling: Union[float, Function[Any]],
    name: Optional[str] = None,
    seed: Optional[roles.ExpressionElementRole[Any]] = None,
) -> TableSample:
    """Return a :class:`_expression.TableSample` object.

    :class:`_expression.TableSample` is an :class:`_expression.Alias`
    subclass that represents
    a table with the TABLESAMPLE clause applied to it.
    :func:`_expression.tablesample`
    is also available from the :class:`_expression.FromClause`
    class via the
    :meth:`_expression.FromClause.tablesample` method.

    The TABLESAMPLE clause allows selecting a randomly selected approximate
    percentage of rows from a table. It supports multiple sampling methods,
    most commonly BERNOULLI and SYSTEM.

    e.g.::

        from sqlalchemy import func

        selectable = people.tablesample(
            func.bernoulli(1), name="alias", seed=func.random()
        )
        stmt = select(selectable.c.people_id)

    Assuming ``people`` with a column ``people_id``, the above
    statement would render as:

    .. sourcecode:: sql

        SELECT alias.people_id FROM
        people AS alias TABLESAMPLE bernoulli(:bernoulli_1)
        REPEATABLE (random())

    :param sampling: a ``float`` percentage between 0 and 100 or
        :class:`_functions.Function`.

    :param name: optional alias name

    :param seed: any real-valued SQL expression.  When specified, the
     REPEATABLE sub-clause is also rendered.

    """
    return TableSample._factory(selectable, sampling, name=name, seed=seed)


@overload
def union(
    *selects: _TypedSelectable[_TP],
) -> CompoundSelect[_TP]: ...


@overload
def union(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]: ...


def union(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]:
    r"""Return a ``UNION`` of multiple selectables.

    The returned object is an instance of
    :class:`_expression.CompoundSelect`.

    A similar :func:`union()` method is available on all
    :class:`_expression.FromClause` subclasses.

    :param \*selects:
      a list of :class:`_expression.Select` instances.

    :param \**kwargs:
      available keyword arguments are the same as those of
      :func:`select`.

    """
    return CompoundSelect._create_union(*selects)


@overload
def union_all(
    *selects: _TypedSelectable[_TP],
) -> CompoundSelect[_TP]: ...


@overload
def union_all(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]: ...


def union_all(
    *selects: _SelectStatementForCompoundArgument[_TP],
) -> CompoundSelect[_TP]:
    r"""Return a ``UNION ALL`` of multiple selectables.

    The returned object is an instance of
    :class:`_expression.CompoundSelect`.

    A similar :func:`union_all()` method is available on all
    :class:`_expression.FromClause` subclasses.

    :param \*selects:
      a list of :class:`_expression.Select` instances.

    """
    return CompoundSelect._create_union_all(*selects)


def values(
    *columns: ColumnClause[Any],
    name: Optional[str] = None,
    literal_binds: bool = False,
) -> Values:
    r"""Construct a :class:`_expression.Values` construct.

    The column expressions and the actual data for
    :class:`_expression.Values` are given in two separate steps.  The
    constructor receives the column expressions typically as
    :func:`_expression.column` constructs,
    and the data is then passed via the
    :meth:`_expression.Values.data` method as a list,
    which can be called multiple
    times to add more data, e.g.::

        from sqlalchemy import column
        from sqlalchemy import values
        from sqlalchemy import Integer
        from sqlalchemy import String

        value_expr = values(
            column("id", Integer),
            column("name", String),
            name="my_values",
        ).data([(1, "name1"), (2, "name2"), (3, "name3")])

    :param \*columns: column expressions, typically composed using
     :func:`_expression.column` objects.

    :param name: the name for this VALUES construct.  If omitted, the
     VALUES construct will be unnamed in a SQL expression.   Different
     backends may have different requirements here.

    :param literal_binds: Defaults to False.  Whether or not to render
     the data values inline in the SQL output, rather than using bound
     parameters.

    """
    return Values(*columns, literal_binds=literal_binds, name=name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\vector.py ===
from __future__ import annotations
from itertools import product

from sympy.core import Add, Basic
from sympy.core.assumptions import StdFactKB
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.basisdependent import (BasisDependentZero,
    BasisDependent, BasisDependentMul, BasisDependentAdd)
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.dyadic import Dyadic, BaseDyadic, DyadicAdd
from sympy.vector.kind import VectorKind


class Vector(BasisDependent):
    """
    Super class for all Vector classes.
    Ideally, neither this class nor any of its subclasses should be
    instantiated by the user.
    """

    is_scalar = False
    is_Vector = True
    _op_priority = 12.0

    _expr_type: type[Vector]
    _mul_func: type[Vector]
    _add_func: type[Vector]
    _zero_func: type[Vector]
    _base_func: type[Vector]
    zero: VectorZero

    kind: VectorKind = VectorKind()

    @property
    def components(self):
        """
        Returns the components of this vector in the form of a
        Python dictionary mapping BaseVector instances to the
        corresponding measure numbers.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.components
        {C.i: 3, C.j: 4, C.k: 5}

        """
        # The '_components' attribute is defined according to the
        # subclass of Vector the instance belongs to.
        return self._components

    def magnitude(self):
        """
        Returns the magnitude of this vector.
        """
        return sqrt(self & self)

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self / self.magnitude()

    def equals(self, other):
        """
        Check if ``self`` and ``other`` are identically equal vectors.

        Explanation
        ===========

        Checks if two vector expressions are equal for all possible values of
        the symbols present in the expressions.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy.abc import x, y
        >>> from sympy import pi
        >>> C = CoordSys3D('C')

        Compare vectors that are equal or not:

        >>> C.i.equals(C.j)
        False
        >>> C.i.equals(C.i)
        True

        These two vectors are equal if `x = y` but are not identically equal
        as expressions since for some values of `x` and `y` they are unequal:

        >>> v1 = x*C.i + C.j
        >>> v2 = y*C.i + C.j
        >>> v1.equals(v1)
        True
        >>> v1.equals(v2)
        False

        Vectors from different coordinate systems can be compared:

        >>> D = C.orient_new_axis('D', pi/2, C.i)
        >>> D.j.equals(C.j)
        False
        >>> D.j.equals(C.k)
        True

        Parameters
        ==========

        other: Vector
            The other vector expression to compare with.

        Returns
        =======

        ``True``, ``False`` or ``None``. A return value of ``True`` indicates
        that the two vectors are identically equal. A return value of ``False``
        indicates that they are not. In some cases it is not possible to
        determine if the two vectors are identically equal and ``None`` is
        returned.

        See Also
        ========

        sympy.core.expr.Expr.equals
        """
        diff = self - other
        diff_mag2 = diff.dot(diff)
        return diff_mag2.equals(0)

    def dot(self, other):
        """
        Returns the dot product of this Vector, either with another
        Vector, or a Dyadic, or a Del operator.
        If 'other' is a Vector, returns the dot product scalar (SymPy
        expression).
        If 'other' is a Dyadic, the dot product is returned as a Vector.
        If 'other' is an instance of Del, returns the directional
        derivative operator as a Python function. If this function is
        applied to a scalar expression, it returns the directional
        derivative of the scalar field wrt this Vector.

        Parameters
        ==========

        other: Vector/Dyadic/Del
            The Vector or Dyadic we are dotting with, or a Del operator .

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Del
        >>> C = CoordSys3D('C')
        >>> delop = Del()
        >>> C.i.dot(C.j)
        0
        >>> C.i & C.i
        1
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v.dot(C.k)
        5
        >>> (C.i & delop)(C.x*C.y*C.z)
        C.y*C.z
        >>> d = C.i.outer(C.i)
        >>> C.i.dot(d)
        C.i

        """

        # Check special cases
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Vector.zero
            outvec = Vector.zero
            for k, v in other.components.items():
                vect_dot = k.args[0].dot(self)
                outvec += vect_dot * v * k.args[1]
            return outvec
        from sympy.vector.deloperator import Del
        if not isinstance(other, (Del, Vector)):
            raise TypeError(str(other) + " is not a vector, dyadic or " +
                            "del operator")

        # Check if the other is a del operator
        if isinstance(other, Del):
            def directional_derivative(field):
                from sympy.vector.functions import directional_derivative
                return directional_derivative(field, self)
            return directional_derivative

        return dot(self, other)

    def __and__(self, other):
        return self.dot(other)

    __and__.__doc__ = dot.__doc__

    def cross(self, other):
        """
        Returns the cross product of this Vector with another Vector or
        Dyadic instance.
        The cross product is a Vector, if 'other' is a Vector. If 'other'
        is a Dyadic, this returns a Dyadic instance.

        Parameters
        ==========

        other: Vector/Dyadic
            The Vector or Dyadic we are crossing with.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> C.i.cross(C.j)
        C.k
        >>> C.i ^ C.i
        0
        >>> v = 3*C.i + 4*C.j + 5*C.k
        >>> v ^ C.i
        5*C.j + (-4)*C.k
        >>> d = C.i.outer(C.i)
        >>> C.j.cross(d)
        (-1)*(C.k|C.i)

        """

        # Check special cases
        if isinstance(other, Dyadic):
            if isinstance(self, VectorZero):
                return Dyadic.zero
            outdyad = Dyadic.zero
            for k, v in other.components.items():
                cross_product = self.cross(k.args[0])
                outer = cross_product.outer(k.args[1])
                outdyad += v * outer
            return outdyad

        return cross(self, other)

    def __xor__(self, other):
        return self.cross(other)

    __xor__.__doc__ = cross.__doc__

    def outer(self, other):
        """
        Returns the outer product of this vector with another, in the
        form of a Dyadic instance.

        Parameters
        ==========

        other : Vector
            The Vector with respect to which the outer product is to
            be computed.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> N.i.outer(N.j)
        (N.i|N.j)

        """

        # Handle the special cases
        if not isinstance(other, Vector):
            raise TypeError("Invalid operand for outer product")
        elif (isinstance(self, VectorZero) or
                isinstance(other, VectorZero)):
            return Dyadic.zero

        # Iterate over components of both the vectors to generate
        # the required Dyadic instance
        args = [(v1 * v2) * BaseDyadic(k1, k2) for (k1, v1), (k2, v2)
                in product(self.components.items(), other.components.items())]

        return DyadicAdd(*args)

    def projection(self, other, scalar=False):
        """
        Returns the vector or scalar projection of the 'other' on 'self'.

        Examples
        ========

        >>> from sympy.vector.coordsysrect import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> i, j, k = C.base_vectors()
        >>> v1 = i + j + k
        >>> v2 = 3*i + 4*j
        >>> v1.projection(v2)
        7/3*C.i + 7/3*C.j + 7/3*C.k
        >>> v1.projection(v2, scalar=True)
        7/3

        """
        if self.equals(Vector.zero):
            return S.Zero if scalar else Vector.zero

        if scalar:
            return self.dot(other) / self.dot(self)
        else:
            return self.dot(other) / self.dot(self) * self

    @property
    def _projections(self):
        """
        Returns the components of this vector but the output includes
        also zero values components.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, Vector
        >>> C = CoordSys3D('C')
        >>> v1 = 3*C.i + 4*C.j + 5*C.k
        >>> v1._projections
        (3, 4, 5)
        >>> v2 = C.x*C.y*C.z*C.i
        >>> v2._projections
        (C.x*C.y*C.z, 0, 0)
        >>> v3 = Vector.zero
        >>> v3._projections
        (0, 0, 0)
        """

        from sympy.vector.operators import _get_coord_systems
        if isinstance(self, VectorZero):
            return (S.Zero, S.Zero, S.Zero)
        base_vec = next(iter(_get_coord_systems(self))).base_vectors()
        return tuple([self.dot(i) for i in base_vec])

    def __or__(self, other):
        return self.outer(other)

    __or__.__doc__ = outer.__doc__

    def to_matrix(self, system):
        """
        Returns the matrix form of this vector with respect to the
        specified coordinate system.

        Parameters
        ==========

        system : CoordSys3D
            The system wrt which the matrix form is to be computed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> C = CoordSys3D('C')
        >>> from sympy.abc import a, b, c
        >>> v = a*C.i + b*C.j + c*C.k
        >>> v.to_matrix(C)
        Matrix([
        [a],
        [b],
        [c]])

        """

        return Matrix([self.dot(unit_vec) for unit_vec in
                       system.base_vectors()])

    def separate(self):
        """
        The constituents of this vector in different coordinate systems,
        as per its definition.

        Returns a dict mapping each CoordSys3D to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> R1 = CoordSys3D('R1')
        >>> R2 = CoordSys3D('R2')
        >>> v = R1.i + R2.i
        >>> v.separate() == {R1: R1.i, R2: R2.i}
        True

        """

        parts = {}
        for vect, measure in self.components.items():
            parts[vect.system] = (parts.get(vect.system, Vector.zero) +
                                  vect * measure)
        return parts

    def _div_helper(one, other):
        """ Helper for division involving vectors. """
        if isinstance(one, Vector) and isinstance(other, Vector):
            raise TypeError("Cannot divide two vectors")
        elif isinstance(one, Vector):
            if other == S.Zero:
                raise ValueError("Cannot divide a vector by zero")
            return VectorMul(one, Pow(other, S.NegativeOne))
        else:
            raise TypeError("Invalid division involving a vector")

# The following is adapted from the matrices.expressions.matexpr file

def get_postprocessor(cls):
    def _postprocessor(expr):
        vec_class = {Add: VectorAdd}[cls]
        vectors = []
        for term in expr.args:
            if isinstance(term.kind, VectorKind):
                vectors.append(term)

        if vec_class == VectorAdd:
            return VectorAdd(*vectors).doit(deep=False)
    return _postprocessor


Basic._constructor_postprocessor_mapping[Vector] = {
    "Add": [get_postprocessor(Add)],
}

class BaseVector(Vector, AtomicExpr):
    """
    Class to denote a base vector.

    """

    def __new__(cls, index, system, pretty_str=None, latex_str=None):
        if pretty_str is None:
            pretty_str = "x{}".format(index)
        if latex_str is None:
            latex_str = "x_{}".format(index)
        pretty_str = str(pretty_str)
        latex_str = str(latex_str)
        # Verify arguments
        if index not in range(0, 3):
            raise ValueError("index must be 0, 1 or 2")
        if not isinstance(system, CoordSys3D):
            raise TypeError("system should be a CoordSys3D")
        name = system._vector_names[index]
        # Initialize an object
        obj = super().__new__(cls, S(index), system)
        # Assign important attributes
        obj._base_instance = obj
        obj._components = {obj: S.One}
        obj._measure_number = S.One
        obj._name = system._name + '.' + name
        obj._pretty_form = '' + pretty_str
        obj._latex_form = latex_str
        obj._system = system
        # The _id is used for printing purposes
        obj._id = (index, system)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)

        # This attr is used for re-expression to one of the systems
        # involved in the definition of the Vector. Applies to
        # VectorMul and VectorAdd too.
        obj._sys = system

        return obj

    @property
    def system(self):
        return self._system

    def _sympystr(self, printer):
        return self._name

    def _sympyrepr(self, printer):
        index, system = self._id
        return printer._print(system) + '.' + system._vector_names[index]

    @property
    def free_symbols(self):
        return {self}

    def _eval_conjugate(self):
        return self


class VectorAdd(BasisDependentAdd, Vector):
    """
    Class to denote sum of Vector instances.
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentAdd.__new__(cls, *args, **options)
        return obj

    def _sympystr(self, printer):
        ret_str = ''
        items = list(self.separate().items())
        items.sort(key=lambda x: x[0].__str__())
        for system, vect in items:
            base_vects = system.base_vectors()
            for x in base_vects:
                if x in vect.components:
                    temp_vect = self.components[x] * x
                    ret_str += printer._print(temp_vect) + " + "
        return ret_str[:-3]


class VectorMul(BasisDependentMul, Vector):
    """
    Class to denote products of scalars and BaseVectors.
    """

    def __new__(cls, *args, **options):
        obj = BasisDependentMul.__new__(cls, *args, **options)
        return obj

    @property
    def base_vector(self):
        """ The BaseVector involved in the product. """
        return self._base_instance

    @property
    def measure_number(self):
        """ The scalar expression involved in the definition of
        this VectorMul.
        """
        return self._measure_number


class VectorZero(BasisDependentZero, Vector):
    """
    Class to denote a zero vector
    """

    _op_priority = 12.1
    _pretty_form = '0'
    _latex_form = r'\mathbf{\hat{0}}'

    def __new__(cls):
        obj = BasisDependentZero.__new__(cls)
        return obj


class Cross(Vector):
    """
    Represents unevaluated Cross product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> Cross(v1, v2)
    Cross(R.i + R.j + R.k, R.x*R.i + R.y*R.j + R.z*R.k)
    >>> Cross(v1, v2).doit()
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        if default_sort_key(expr1) > default_sort_key(expr2):
            return -Cross(expr2, expr1)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return cross(self._expr1, self._expr2)


class Dot(Expr):
    """
    Represents unevaluated Dot product.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D, Dot
    >>> from sympy import symbols
    >>> R = CoordSys3D('R')
    >>> a, b, c = symbols('a b c')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = a * R.i + b * R.j + c * R.k
    >>> Dot(v1, v2)
    Dot(R.i + R.j + R.k, a*R.i + b*R.j + c*R.k)
    >>> Dot(v1, v2).doit()
    a + b + c

    """

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)
        expr1, expr2 = sorted([expr1, expr2], key=default_sort_key)
        obj = Expr.__new__(cls, expr1, expr2)
        obj._expr1 = expr1
        obj._expr2 = expr2
        return obj

    def doit(self, **hints):
        return dot(self._expr1, self._expr2)


def cross(vect1, vect2):
    """
    Returns cross product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import cross
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> cross(v1, v2)
    (-R.y + R.z)*R.i + (R.x - R.z)*R.j + (-R.x + R.y)*R.k

    """
    if isinstance(vect1, Add):
        return VectorAdd.fromiter(cross(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return VectorAdd.fromiter(cross(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            n1 = vect1.args[0]
            n2 = vect2.args[0]
            if n1 == n2:
                return Vector.zero
            n3 = ({0,1,2}.difference({n1, n2})).pop()
            sign = 1 if ((n1 + 1) % 3 == n2) else -1
            return sign*vect1._sys.base_vectors()[n3]
        from .functions import express
        try:
            v = express(vect1, vect2._sys)
        except ValueError:
            return Cross(vect1, vect2)
        else:
            return cross(v, vect2)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return Vector.zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1*cross(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2*cross(vect1, v2)

    return Cross(vect1, vect2)


def dot(vect1, vect2):
    """
    Returns dot product of two vectors.

    Examples
    ========

    >>> from sympy.vector import CoordSys3D
    >>> from sympy.vector.vector import dot
    >>> R = CoordSys3D('R')
    >>> v1 = R.i + R.j + R.k
    >>> v2 = R.x * R.i + R.y * R.j + R.z * R.k
    >>> dot(v1, v2)
    R.x + R.y + R.z

    """
    if isinstance(vect1, Add):
        return Add.fromiter(dot(i, vect2) for i in vect1.args)
    if isinstance(vect2, Add):
        return Add.fromiter(dot(vect1, i) for i in vect2.args)
    if isinstance(vect1, BaseVector) and isinstance(vect2, BaseVector):
        if vect1._sys == vect2._sys:
            return S.One if vect1 == vect2 else S.Zero
        from .functions import express
        try:
            v = express(vect2, vect1._sys)
        except ValueError:
            return Dot(vect1, vect2)
        else:
            return dot(vect1, v)
    if isinstance(vect1, VectorZero) or isinstance(vect2, VectorZero):
        return S.Zero
    if isinstance(vect1, VectorMul):
        v1, m1 = next(iter(vect1.components.items()))
        return m1*dot(v1, vect2)
    if isinstance(vect2, VectorMul):
        v2, m2 = next(iter(vect2.components.items()))
        return m2*dot(vect1, v2)

    return Dot(vect1, vect2)


Vector._expr_type = Vector
Vector._mul_func = VectorMul
Vector._add_func = VectorAdd
Vector._zero_func = VectorZero
Vector._base_func = BaseVector
Vector.zero = VectorZero()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_abc.py ===
from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import trio

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    # both of these introduce circular imports if outside a TYPE_CHECKING guard
    from ._socket import SocketType
    from .lowlevel import Task


class Clock(ABC):
    """The interface for custom run loop clocks."""

    __slots__ = ()

    @abstractmethod
    def start_clock(self) -> None:
        """Do any setup this clock might need.

        Called at the beginning of the run.

        """

    @abstractmethod
    def current_time(self) -> float:
        """Return the current time, according to this clock.

        This is used to implement functions like :func:`trio.current_time` and
        :func:`trio.move_on_after`.

        Returns:
            float: The current time.

        """

    @abstractmethod
    def deadline_to_sleep_time(self, deadline: float) -> float:
        """Compute the real time until the given deadline.

        This is called before we enter a system-specific wait function like
        :func:`select.select`, to get the timeout to pass.

        For a clock using wall-time, this should be something like::

           return deadline - self.current_time()

        but of course it may be different if you're implementing some kind of
        virtual clock.

        Args:
            deadline (float): The absolute time of the next deadline,
                according to this clock.

        Returns:
            float: The number of real seconds to sleep until the given
            deadline. May be :data:`math.inf`.

        """


class Instrument(ABC):  # noqa: B024  # conceptually is ABC
    """The interface for run loop instrumentation.

    Instruments don't have to inherit from this abstract base class, and all
    of these methods are optional. This class serves mostly as documentation.

    """

    __slots__ = ()

    def before_run(self) -> None:
        """Called at the beginning of :func:`trio.run`."""
        return

    def after_run(self) -> None:
        """Called just before :func:`trio.run` returns."""
        return

    def task_spawned(self, task: Task) -> None:
        """Called when the given task is created.

        Args:
            task (trio.lowlevel.Task): The new task.

        """
        return

    def task_scheduled(self, task: Task) -> None:
        """Called when the given task becomes runnable.

        It may still be some time before it actually runs, if there are other
        runnable tasks ahead of it.

        Args:
            task (trio.lowlevel.Task): The task that became runnable.

        """
        return

    def before_task_step(self, task: Task) -> None:
        """Called immediately before we resume running the given task.

        Args:
            task (trio.lowlevel.Task): The task that is about to run.

        """
        return

    def after_task_step(self, task: Task) -> None:
        """Called when we return to the main run loop after a task has yielded.

        Args:
            task (trio.lowlevel.Task): The task that just ran.

        """
        return

    def task_exited(self, task: Task) -> None:
        """Called when the given task exits.

        Args:
            task (trio.lowlevel.Task): The finished task.

        """
        return

    def before_io_wait(self, timeout: float) -> None:
        """Called before blocking to wait for I/O readiness.

        Args:
            timeout (float): The number of seconds we are willing to wait.

        """
        return

    def after_io_wait(self, timeout: float) -> None:
        """Called after handling pending I/O.

        Args:
            timeout (float): The number of seconds we were willing to
                wait. This much time may or may not have elapsed, depending on
                whether any I/O was ready.

        """
        return


class HostnameResolver(ABC):
    """If you have a custom hostname resolver, then implementing
    :class:`HostnameResolver` allows you to register this to be used by Trio.

    See :func:`trio.socket.set_custom_hostname_resolver`.

    """

    __slots__ = ()

    @abstractmethod
    async def getaddrinfo(
        self,
        host: bytes | None,
        port: bytes | str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[
        tuple[
            socket.AddressFamily,
            socket.SocketKind,
            int,
            str,
            tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes],
        ]
    ]:
        """A custom implementation of :func:`~trio.socket.getaddrinfo`.

        Called by :func:`trio.socket.getaddrinfo`.

        If ``host`` is given as a numeric IP address, then
        :func:`~trio.socket.getaddrinfo` may handle the request itself rather
        than calling this method.

        Any required IDNA encoding is handled before calling this function;
        your implementation can assume that it will never see U-labels like
        ``"café.com"``, and only needs to handle A-labels like
        ``b"xn--caf-dma.com"``."""  # spellchecker:disable-line

    @abstractmethod
    async def getnameinfo(
        self,
        sockaddr: tuple[str, int] | tuple[str, int, int, int],
        flags: int,
    ) -> tuple[str, str]:
        """A custom implementation of :func:`~trio.socket.getnameinfo`.

        Called by :func:`trio.socket.getnameinfo`.

        """


class SocketFactory(ABC):
    """If you write a custom class implementing the Trio socket interface,
    then you can use a :class:`SocketFactory` to get Trio to use it.

    See :func:`trio.socket.set_custom_socket_factory`.

    """

    __slots__ = ()

    @abstractmethod
    def socket(
        self,
        family: socket.AddressFamily | int = socket.AF_INET,
        type: socket.SocketKind | int = socket.SOCK_STREAM,
        proto: int = 0,
    ) -> SocketType:
        """Create and return a socket object.

        Your socket object must inherit from :class:`trio.socket.SocketType`,
        which is an empty class whose only purpose is to "mark" which classes
        should be considered valid Trio sockets.

        Called by :func:`trio.socket.socket`.

        Note that unlike :func:`trio.socket.socket`, this does not take a
        ``fileno=`` argument. If a ``fileno=`` is specified, then
        :func:`trio.socket.socket` returns a regular Trio socket object
        instead of calling this method.

        """


class AsyncResource(ABC):
    """A standard interface for resources that needs to be cleaned up, and
    where that cleanup may require blocking operations.

    This class distinguishes between "graceful" closes, which may perform I/O
    and thus block, and a "forceful" close, which cannot. For example, cleanly
    shutting down a TLS-encrypted connection requires sending a "goodbye"
    message; but if a peer has become non-responsive, then sending this
    message might block forever, so we may want to just drop the connection
    instead. Therefore the :meth:`aclose` method is unusual in that it
    should always close the connection (or at least make its best attempt)
    *even if it fails*; failure indicates a failure to achieve grace, not a
    failure to close the connection.

    Objects that implement this interface can be used as async context
    managers, i.e., you can write::

      async with create_resource() as some_async_resource:
          ...

    Entering the context manager is synchronous (not a checkpoint); exiting it
    calls :meth:`aclose`. The default implementations of
    ``__aenter__`` and ``__aexit__`` should be adequate for all subclasses.

    """

    __slots__ = ()

    @abstractmethod
    async def aclose(self) -> None:
        """Close this resource, possibly blocking.

        IMPORTANT: This method may block in order to perform a "graceful"
        shutdown. But, if this fails, then it still *must* close any
        underlying resources before returning. An error from this method
        indicates a failure to achieve grace, *not* a failure to close the
        connection.

        For example, suppose we call :meth:`aclose` on a TLS-encrypted
        connection. This requires sending a "goodbye" message; but if the peer
        has become non-responsive, then our attempt to send this message might
        block forever, and eventually time out and be cancelled. In this case
        the :meth:`aclose` method on :class:`~trio.SSLStream` will
        immediately close the underlying transport stream using
        :func:`trio.aclose_forcefully` before raising :exc:`~trio.Cancelled`.

        If the resource is already closed, then this method should silently
        succeed.

        Once this method completes, any other pending or future operations on
        this resource should generally raise :exc:`~trio.ClosedResourceError`,
        unless there's a good reason to do otherwise.

        See also: :func:`trio.aclose_forcefully`.

        """

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


class SendStream(AsyncResource):
    """A standard interface for sending data on a byte stream.

    The underlying stream may be unidirectional, or bidirectional. If it's
    bidirectional, then you probably want to also implement
    :class:`ReceiveStream`, which makes your object a :class:`Stream`.

    :class:`SendStream` objects also implement the :class:`AsyncResource`
    interface, so they can be closed by calling :meth:`~AsyncResource.aclose`
    or using an ``async with`` block.

    If you want to send Python objects rather than raw bytes, see
    :class:`SendChannel`.

    """

    __slots__ = ()

    @abstractmethod
    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        """Sends the given data through the stream, blocking if necessary.

        Args:
          data (bytes, bytearray, or memoryview): The data to send.

        Raises:
          trio.BusyResourceError: if another task is already executing a
              :meth:`send_all`, :meth:`wait_send_all_might_not_block`, or
              :meth:`HalfCloseableStream.send_eof` on this stream.
          trio.BrokenResourceError: if something has gone wrong, and the stream
              is broken.
          trio.ClosedResourceError: if you previously closed this stream
              object, or if another task closes this stream object while
              :meth:`send_all` is running.

        Most low-level operations in Trio provide a guarantee: if they raise
        :exc:`trio.Cancelled`, this means that they had no effect, so the
        system remains in a known state. This is **not true** for
        :meth:`send_all`. If this operation raises :exc:`trio.Cancelled` (or
        any other exception for that matter), then it may have sent some, all,
        or none of the requested data, and there is no way to know which.

        """

    @abstractmethod
    async def wait_send_all_might_not_block(self) -> None:
        """Block until it's possible that :meth:`send_all` might not block.

        This method may return early: it's possible that after it returns,
        :meth:`send_all` will still block. (In the worst case, if no better
        implementation is available, then it might always return immediately
        without blocking. It's nice to do better than that when possible,
        though.)

        This method **must not** return *late*: if it's possible for
        :meth:`send_all` to complete without blocking, then it must
        return. When implementing it, err on the side of returning early.

        Raises:
          trio.BusyResourceError: if another task is already executing a
              :meth:`send_all`, :meth:`wait_send_all_might_not_block`, or
              :meth:`HalfCloseableStream.send_eof` on this stream.
          trio.BrokenResourceError: if something has gone wrong, and the stream
              is broken.
          trio.ClosedResourceError: if you previously closed this stream
              object, or if another task closes this stream object while
              :meth:`wait_send_all_might_not_block` is running.

        Note:

          This method is intended to aid in implementing protocols that want
          to delay choosing which data to send until the last moment. E.g.,
          suppose you're working on an implementation of a remote display server
          like `VNC
          <https://en.wikipedia.org/wiki/Virtual_Network_Computing>`__, and
          the network connection is currently backed up so that if you call
          :meth:`send_all` now then it will sit for 0.5 seconds before actually
          sending anything. In this case it doesn't make sense to take a
          screenshot, then wait 0.5 seconds, and then send it, because the
          screen will keep changing while you wait; it's better to wait 0.5
          seconds, then take the screenshot, and then send it, because this
          way the data you deliver will be more
          up-to-date. Using :meth:`wait_send_all_might_not_block` makes it
          possible to implement the better strategy.

          If you use this method, you might also want to read up on
          ``TCP_NOTSENT_LOWAT``.

          Further reading:

          * `Prioritization Only Works When There's Pending Data to Prioritize
            <https://insouciant.org/tech/prioritization-only-works-when-theres-pending-data-to-prioritize/>`__

          * WWDC 2015: Your App and Next Generation Networks: `slides
            <http://devstreaming.apple.com/videos/wwdc/2015/719ui2k57m/719/719_your_app_and_next_generation_networks.pdf?dl=1>`__,
            `video and transcript
            <https://developer.apple.com/videos/play/wwdc2015/719/>`__

        """


class ReceiveStream(AsyncResource):
    """A standard interface for receiving data on a byte stream.

    The underlying stream may be unidirectional, or bidirectional. If it's
    bidirectional, then you probably want to also implement
    :class:`SendStream`, which makes your object a :class:`Stream`.

    :class:`ReceiveStream` objects also implement the :class:`AsyncResource`
    interface, so they can be closed by calling :meth:`~AsyncResource.aclose`
    or using an ``async with`` block.

    If you want to receive Python objects rather than raw bytes, see
    :class:`ReceiveChannel`.

    `ReceiveStream` objects can be used in ``async for`` loops. Each iteration
    will produce an arbitrary sized chunk of bytes, like calling
    `receive_some` with no arguments. Every chunk will contain at least one
    byte, and the loop automatically exits when reaching end-of-file.

    """

    __slots__ = ()

    @abstractmethod
    async def receive_some(self, max_bytes: int | None = None) -> bytes | bytearray:
        """Wait until there is data available on this stream, and then return
        some of it.

        A return value of ``b""`` (an empty bytestring) indicates that the
        stream has reached end-of-file. Implementations should be careful that
        they return ``b""`` if, and only if, the stream has reached
        end-of-file!

        Args:
          max_bytes (int): The maximum number of bytes to return. Must be
              greater than zero. Optional; if omitted, then the stream object
              is free to pick a reasonable default.

        Returns:
          bytes or bytearray: The data received.

        Raises:
          trio.BusyResourceError: if two tasks attempt to call
              :meth:`receive_some` on the same stream at the same time.
          trio.BrokenResourceError: if something has gone wrong, and the stream
              is broken.
          trio.ClosedResourceError: if you previously closed this stream
              object, or if another task closes this stream object while
              :meth:`receive_some` is running.

        """

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes | bytearray:
        data = await self.receive_some()
        if not data:
            raise StopAsyncIteration
        return data


class Stream(SendStream, ReceiveStream):
    """A standard interface for interacting with bidirectional byte streams.

    A :class:`Stream` is an object that implements both the
    :class:`SendStream` and :class:`ReceiveStream` interfaces.

    If implementing this interface, you should consider whether you can go one
    step further and implement :class:`HalfCloseableStream`.

    """

    __slots__ = ()


class HalfCloseableStream(Stream):
    """This interface extends :class:`Stream` to also allow closing the send
    part of the stream without closing the receive part.

    """

    __slots__ = ()

    @abstractmethod
    async def send_eof(self) -> None:
        """Send an end-of-file indication on this stream, if possible.

        The difference between :meth:`send_eof` and
        :meth:`~AsyncResource.aclose` is that :meth:`send_eof` is a
        *unidirectional* end-of-file indication. After you call this method,
        you shouldn't try sending any more data on this stream, and your
        remote peer should receive an end-of-file indication (eventually,
        after receiving all the data you sent before that). But, they may
        continue to send data to you, and you can continue to receive it by
        calling :meth:`~ReceiveStream.receive_some`. You can think of it as
        calling :meth:`~AsyncResource.aclose` on just the
        :class:`SendStream` "half" of the stream object (and in fact that's
        literally how :class:`trio.StapledStream` implements it).

        Examples:

        * On a socket, this corresponds to ``shutdown(..., SHUT_WR)`` (`man
          page <https://linux.die.net/man/2/shutdown>`__).

        * The SSH protocol provides the ability to multiplex bidirectional
          "channels" on top of a single encrypted connection. A Trio
          implementation of SSH could expose these channels as
          :class:`HalfCloseableStream` objects, and calling :meth:`send_eof`
          would send an ``SSH_MSG_CHANNEL_EOF`` request (see `RFC 4254 §5.3
          <https://tools.ietf.org/html/rfc4254#section-5.3>`__).

        * On an SSL/TLS-encrypted connection, the protocol doesn't provide any
          way to do a unidirectional shutdown without closing the connection
          entirely, so :class:`~trio.SSLStream` implements
          :class:`Stream`, not :class:`HalfCloseableStream`.

        If an EOF has already been sent, then this method should silently
        succeed.

        Raises:
          trio.BusyResourceError: if another task is already executing a
              :meth:`~SendStream.send_all`,
              :meth:`~SendStream.wait_send_all_might_not_block`, or
              :meth:`send_eof` on this stream.
          trio.BrokenResourceError: if something has gone wrong, and the stream
              is broken.
          trio.ClosedResourceError: if you previously closed this stream
              object, or if another task closes this stream object while
              :meth:`send_eof` is running.

        """


# A regular invariant generic type
T = TypeVar("T")

# The type of object produced by a ReceiveChannel (covariant because
# ReceiveChannel[Derived] can be passed to someone expecting
# ReceiveChannel[Base])
ReceiveType = TypeVar("ReceiveType", covariant=True)

# The type of object accepted by a SendChannel (contravariant because
# SendChannel[Base] can be passed to someone expecting
# SendChannel[Derived])
SendType = TypeVar("SendType", contravariant=True)

# The type of object produced by a Listener (covariant plus must be
# an AsyncResource)
T_resource = TypeVar("T_resource", bound=AsyncResource, covariant=True)


class Listener(AsyncResource, Generic[T_resource]):
    """A standard interface for listening for incoming connections.

    :class:`Listener` objects also implement the :class:`AsyncResource`
    interface, so they can be closed by calling :meth:`~AsyncResource.aclose`
    or using an ``async with`` block.

    """

    __slots__ = ()

    @abstractmethod
    async def accept(self) -> T_resource:
        """Wait until an incoming connection arrives, and then return it.

        Returns:
          AsyncResource: An object representing the incoming connection. In
          practice this is generally some kind of :class:`Stream`,
          but in principle you could also define a :class:`Listener` that
          returned, say, channel objects.

        Raises:
          trio.BusyResourceError: if two tasks attempt to call
              :meth:`accept` on the same listener at the same time.
          trio.ClosedResourceError: if you previously closed this listener
              object, or if another task closes this listener object while
              :meth:`accept` is running.

        Listeners don't generally raise :exc:`~trio.BrokenResourceError`,
        because for listeners there is no general condition of "the
        network/remote peer broke the connection" that can be handled in a
        generic way, like there is for streams. Other errors *can* occur and
        be raised from :meth:`accept` – for example, if you run out of file
        descriptors then you might get an :class:`OSError` with its errno set
        to ``EMFILE``.

        """


class SendChannel(AsyncResource, Generic[SendType]):
    """A standard interface for sending Python objects to some receiver.

    `SendChannel` objects also implement the `AsyncResource` interface, so
    they can be closed by calling `~AsyncResource.aclose` or using an ``async
    with`` block.

    If you want to send raw bytes rather than Python objects, see
    `SendStream`.

    """

    __slots__ = ()

    @abstractmethod
    async def send(self, value: SendType) -> None:
        """Attempt to send an object through the channel, blocking if necessary.

        Args:
          value (object): The object to send.

        Raises:
          trio.BrokenResourceError: if something has gone wrong, and the
              channel is broken. For example, you may get this if the receiver
              has already been closed.
          trio.ClosedResourceError: if you previously closed this
              :class:`SendChannel` object, or if another task closes it while
              :meth:`send` is running.
          trio.BusyResourceError: some channels allow multiple tasks to call
              `send` at the same time, but others don't. If you try to call
              `send` simultaneously from multiple tasks on a channel that
              doesn't support it, then you can get `~trio.BusyResourceError`.

        """


class ReceiveChannel(AsyncResource, Generic[ReceiveType]):
    """A standard interface for receiving Python objects from some sender.

    You can iterate over a :class:`ReceiveChannel` using an ``async for``
    loop::

       async for value in receive_channel:
           ...

    This is equivalent to calling :meth:`receive` repeatedly. The loop exits
    without error when `receive` raises `~trio.EndOfChannel`.

    `ReceiveChannel` objects also implement the `AsyncResource` interface, so
    they can be closed by calling `~AsyncResource.aclose` or using an ``async
    with`` block.

    If you want to receive raw bytes rather than Python objects, see
    `ReceiveStream`.

    """

    __slots__ = ()

    @abstractmethod
    async def receive(self) -> ReceiveType:
        """Attempt to receive an incoming object, blocking if necessary.

        Returns:
          object: Whatever object was received.

        Raises:
          trio.EndOfChannel: if the sender has been closed cleanly, and no
              more objects are coming. This is not an error condition.
          trio.ClosedResourceError: if you previously closed this
              :class:`ReceiveChannel` object.
          trio.BrokenResourceError: if something has gone wrong, and the
              channel is broken.
          trio.BusyResourceError: some channels allow multiple tasks to call
              `receive` at the same time, but others don't. If you try to call
              `receive` simultaneously from multiple tasks on a channel that
              doesn't support it, then you can get `~trio.BusyResourceError`.

        """

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ReceiveType:
        try:
            return await self.receive()
        except trio.EndOfChannel:
            raise StopAsyncIteration from None


# these are necessary for Sphinx's :show-inheritance: with type args.
# (this should be removed if possible)
# see: https://github.com/python/cpython/issues/123250
SendChannel.__module__ = SendChannel.__module__.replace("_abc", "abc")
ReceiveChannel.__module__ = ReceiveChannel.__module__.replace("_abc", "abc")
Listener.__module__ = Listener.__module__.replace("_abc", "abc")


class Channel(SendChannel[T], ReceiveChannel[T]):
    """A standard interface for interacting with bidirectional channels.

    A `Channel` is an object that implements both the `SendChannel` and
    `ReceiveChannel` interfaces, so you can both send and receive objects.

    """

    __slots__ = ()


# see above
Channel.__module__ = Channel.__module__.replace("_abc", "abc")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\octave.py ===
"""
Octave (and Matlab) code printer

The `OctaveCodePrinter` converts SymPy expressions into Octave expressions.
It uses a subset of the Octave language for Matlab compatibility.

A complete code generator, which uses `octave_code` extensively, can be found
in `sympy.utilities.codegen`.  The `codegen` module can be used to generate
complete source code files.

"""

from __future__ import annotations
from typing import Any

from sympy.core import Mul, Pow, S, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from re import search

# List of known functions.  First, those that have the same name in
# SymPy and Octave.   This is almost certainly incomplete!
known_fcns_src1 = ["sin", "cos", "tan", "cot", "sec", "csc",
                   "asin", "acos", "acot", "atan", "atan2", "asec", "acsc",
                   "sinh", "cosh", "tanh", "coth", "csch", "sech",
                   "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
                   "erfc", "erfi", "erf", "erfinv", "erfcinv",
                   "besseli", "besselj", "besselk", "bessely",
                   "bernoulli", "beta", "euler", "exp", "factorial", "floor",
                   "fresnelc", "fresnels", "gamma", "harmonic", "log",
                   "polylog", "sign", "zeta", "legendre"]

# These functions have different names ("SymPy": "Octave"), more
# generally a mapping to (argument_conditions, octave_function).
known_fcns_src2 = {
    "Abs": "abs",
    "arg": "angle",  # arg/angle ok in Octave but only angle in Matlab
    "binomial": "bincoeff",
    "ceiling": "ceil",
    "chebyshevu": "chebyshevU",
    "chebyshevt": "chebyshevT",
    "Chi": "coshint",
    "Ci": "cosint",
    "conjugate": "conj",
    "DiracDelta": "dirac",
    "Heaviside": "heaviside",
    "im": "imag",
    "laguerre": "laguerreL",
    "LambertW": "lambertw",
    "li": "logint",
    "loggamma": "gammaln",
    "Max": "max",
    "Min": "min",
    "Mod": "mod",
    "polygamma": "psi",
    "re": "real",
    "RisingFactorial": "pochhammer",
    "Shi": "sinhint",
    "Si": "sinint",
}


class OctaveCodePrinter(CodePrinter):
    """
    A printer to convert expressions to strings of Octave/Matlab code.
    """
    printmethod = "_octave"
    language = "Octave"

    _operators = {
        'and': '&',
        'or': '|',
        'not': '~',
    }

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 17,
        'user_functions': {},
        'contract': True,
        'inline': True,
    })
    # Note: contract is for expressing tensors as loops (if True), or just
    # assignment (if False).  FIXME: this should be looked a more carefully
    # for Octave.


    def __init__(self, settings={}):
        super().__init__(settings)
        self.known_functions = dict(zip(known_fcns_src1, known_fcns_src1))
        self.known_functions.update(dict(known_fcns_src2))
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)


    def _rate_index_position(self, p):
        return p*5


    def _get_statement(self, codestring):
        return "%s;" % codestring


    def _get_comment(self, text):
        return "% {}".format(text)


    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)


    def _format_code(self, lines):
        return self.indent_code(lines)


    def _traverse_matrix_indices(self, mat):
        # Octave uses Fortran order (column-major)
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))


    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        for i in indices:
            # Octave arrays start at 1 and end at dimension
            var, start, stop = map(self._print,
                    [i.label, i.lower + 1, i.upper + 1])
            open_lines.append("for %s = %s:%s" % (var, start, stop))
            close_lines.append("end")
        return open_lines, close_lines


    def _print_Mul(self, expr):
        # print complex numbers nicely in Octave
        if (expr.is_number and expr.is_imaginary and
                (S.ImaginaryUnit*expr).is_Integer):
            return "%si" % self._print(-S.ImaginaryUnit*expr)

        # cribbed from str.py
        prec = precedence(expr)

        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = "-"
        else:
            sign = ""

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        pow_paren = []  # Will collect all pow with more than one base element and exp = -1

        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        # Gather args for numerator/denominator
        for item in args:
            if (item.is_commutative and item.is_Pow and item.exp.is_Rational
                    and item.exp.is_negative):
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, Mul):   # To avoid situations like #14160
                        pow_paren.append(item)
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)

        a = a or [S.One]

        a_str = [self.parenthesize(x, prec) for x in a]
        b_str = [self.parenthesize(x, prec) for x in b]

        # To parenthesize Pow with exp = -1 and having more than one Symbol
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = "(%s)" % b_str[b.index(item.base)]

        # from here it differs from str.py to deal with "*" and ".*"
        def multjoin(a, a_str):
            # here we probably are assuming the constants will come first
            r = a_str[0]
            for i in range(1, len(a)):
                mulsym = '*' if a[i-1].is_number else '.*'
                r = r + mulsym + a_str[i]
            return r

        if not b:
            return sign + multjoin(a, a_str)
        elif len(b) == 1:
            divsym = '/' if b[0].is_number else './'
            return sign + multjoin(a, a_str) + divsym + b_str[0]
        else:
            divsym = '/' if all(bi.is_number for bi in b) else './'
            return (sign + multjoin(a, a_str) +
                    divsym + "(%s)" % multjoin(b, b_str))

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Pow(self, expr):
        powsymbol = '^' if all(x.is_number for x in expr.args) else '.^'

        PREC = precedence(expr)

        if equal_valued(expr.exp, 0.5):
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if equal_valued(expr.exp, -0.5):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "sqrt(%s)" % self._print(expr.base)
            if equal_valued(expr.exp, -1):
                sym = '/' if expr.base.is_number else './'
                return "1" + sym + "%s" % self.parenthesize(expr.base, PREC)

        return '%s%s%s' % (self.parenthesize(expr.base, PREC), powsymbol,
                           self.parenthesize(expr.exp, PREC))


    def _print_MatPow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC),
                          self.parenthesize(expr.exp, PREC))

    def _print_MatrixSolve(self, expr):
        PREC = precedence(expr)
        return "%s \\ %s" % (self.parenthesize(expr.matrix, PREC),
                             self.parenthesize(expr.vector, PREC))

    def _print_Pi(self, expr):
        return 'pi'


    def _print_ImaginaryUnit(self, expr):
        return "1i"


    def _print_Exp1(self, expr):
        return "exp(1)"


    def _print_GoldenRatio(self, expr):
        # FIXME: how to do better, e.g., for octave_code(2*GoldenRatio)?
        #return self._print((1+sqrt(S(5)))/2)
        return "(1+sqrt(5))/2"


    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.tensor.indexed import IndexedBase
        # Copied from codeprinter, but remove special MatrixSymbol treatment
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        if not self._settings["inline"] and isinstance(expr.rhs, Piecewise):
            # Here we modify Piecewise so each expression is now
            # an Assignment, and then continue on the print.
            expressions = []
            conditions = []
            for (e, c) in rhs.args:
                expressions.append(Assignment(lhs, e))
                conditions.append(c)
            temp = Piecewise(*zip(expressions, conditions))
            return self._print(temp)
        if self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))


    def _print_Infinity(self, expr):
        return 'inf'


    def _print_NegativeInfinity(self, expr):
        return '-inf'


    def _print_NaN(self, expr):
        return 'NaN'


    def _print_list(self, expr):
        return '{' + ', '.join(self._print(a) for a in expr) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list
    _print_List = _print_list


    def _print_BooleanTrue(self, expr):
        return "true"


    def _print_BooleanFalse(self, expr):
        return "false"


    def _print_bool(self, expr):
        return str(expr).lower()


    # Could generate quadrature code for definite Integrals?
    #_print_Integral = _print_not_supported


    def _print_MatrixBase(self, A):
        # Handle zero dimensions:
        if (A.rows, A.cols) == (0, 0):
            return '[]'
        elif S.Zero in A.shape:
            return 'zeros(%s, %s)' % (A.rows, A.cols)
        elif (A.rows, A.cols) == (1, 1):
            # Octave does not distinguish between scalars and 1x1 matrices
            return self._print(A[0, 0])
        return "[%s]" % "; ".join(" ".join([self._print(a) for a in A[r, :]])
                                  for r in range(A.rows))


    def _print_SparseRepMatrix(self, A):
        from sympy.matrices import Matrix
        L = A.col_list()
        # make row vectors of the indices and entries
        I = Matrix([[k[0] + 1 for k in L]])
        J = Matrix([[k[1] + 1 for k in L]])
        AIJ = Matrix([[k[2] for k in L]])
        return "sparse(%s, %s, %s, %s, %s)" % (self._print(I), self._print(J),
                                            self._print(AIJ), A.rows, A.cols)


    def _print_MatrixElement(self, expr):
        return self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) \
            + '(%s, %s)' % (expr.i + 1, expr.j + 1)


    def _print_MatrixSlice(self, expr):
        def strslice(x, lim):
            l = x[0] + 1
            h = x[1]
            step = x[2]
            lstr = self._print(l)
            hstr = 'end' if h == lim else self._print(h)
            if step == 1:
                if l == 1 and h == lim:
                    return ':'
                if l == h:
                    return lstr
                else:
                    return lstr + ':' + hstr
            else:
                return ':'.join((lstr, self._print(step), hstr))
        return (self._print(expr.parent) + '(' +
                strslice(expr.rowslice, expr.parent.shape[0]) + ', ' +
                strslice(expr.colslice, expr.parent.shape[1]) + ')')


    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))


    def _print_KroneckerDelta(self, expr):
        prec = PRECEDENCE["Pow"]
        return "double(%s == %s)" % tuple(self.parenthesize(x, prec)
                                          for x in expr.args)

    def _print_HadamardProduct(self, expr):
        return '.*'.join([self.parenthesize(arg, precedence(expr))
                          for arg in expr.args])

    def _print_HadamardPower(self, expr):
        PREC = precedence(expr)
        return '.**'.join([
            self.parenthesize(expr.base, PREC),
            self.parenthesize(expr.exp, PREC)
            ])

    def _print_Identity(self, expr):
        shape = expr.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            shape = [shape[0]]
        s = ", ".join(self._print(n) for n in shape)
        return "eye(" + s + ")"

    def _print_lowergamma(self, expr):
        # Octave implements regularized incomplete gamma function
        return "(gammainc({1}, {0}).*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))


    def _print_uppergamma(self, expr):
        return "(gammainc({1}, {0}, 'upper').*gamma({0}))".format(
            self._print(expr.args[0]), self._print(expr.args[1]))


    def _print_sinc(self, expr):
        #Note: Divide by pi because Octave implements normalized sinc function.
        return "sinc(%s)" % self._print(expr.args[0]/S.Pi)


    def _print_hankel1(self, expr):
        return "besselh(%s, 1, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    def _print_hankel2(self, expr):
        return "besselh(%s, 2, %s)" % (self._print(expr.order),
                                       self._print(expr.argument))


    # Note: as of 2015, Octave doesn't have spherical Bessel functions
    def _print_jn(self, expr):
        from sympy.functions import sqrt, besselj
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*besselj(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_yn(self, expr):
        from sympy.functions import sqrt, bessely
        x = expr.argument
        expr2 = sqrt(S.Pi/(2*x))*bessely(expr.order + S.Half, x)
        return self._print(expr2)


    def _print_airyai(self, expr):
        return "airy(0, %s)" % self._print(expr.args[0])


    def _print_airyaiprime(self, expr):
        return "airy(1, %s)" % self._print(expr.args[0])


    def _print_airybi(self, expr):
        return "airy(2, %s)" % self._print(expr.args[0])


    def _print_airybiprime(self, expr):
        return "airy(3, %s)" % self._print(expr.args[0])


    def _print_expint(self, expr):
        mu, x = expr.args
        if mu != 1:
            return self._print_not_supported(expr)
        return "expint(%s)" % self._print(x)


    def _one_or_two_reversed_args(self, expr):
        assert len(expr.args) <= 2
        return '{name}({args})'.format(
            name=self.known_functions[expr.__class__.__name__],
            args=", ".join([self._print(x) for x in reversed(expr.args)])
        )


    _print_DiracDelta = _print_LambertW = _one_or_two_reversed_args


    def _nested_binary_math_func(self, expr):
        return '{name}({arg1}, {arg2})'.format(
            name=self.known_functions[expr.__class__.__name__],
            arg1=self._print(expr.args[0]),
            arg2=self._print(expr.func(*expr.args[1:]))
            )

    _print_Max = _print_Min = _nested_binary_math_func


    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if self._settings["inline"]:
            # Express each (cond, expr) pair in a nested Horner form:
            #   (condition) .* (expr) + (not cond) .* (<others>)
            # Expressions that result in multiple statements won't work here.
            ecpairs = ["({0}).*({1}) + (~({0})).*(".format
                       (self._print(c), self._print(e))
                       for e, c in expr.args[:-1]]
            elast = "%s" % self._print(expr.args[-1].expr)
            pw = " ...\n".join(ecpairs) + elast + ")"*len(ecpairs)
            # Note: current need these outer brackets for 2*pw.  Would be
            # nicer to teach parenthesize() to do this for us when needed!
            return "(" + pw + ")"
        else:
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s)" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else")
                else:
                    lines.append("elseif (%s)" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                if i == len(expr.args) - 1:
                    lines.append("end")
            return "\n".join(lines)


    def _print_zeta(self, expr):
        if len(expr.args) == 1:
            return "zeta(%s)" % self._print(expr.args[0])
        else:
            # Matlab two argument zeta is not equivalent to SymPy's
            return self._print_not_supported(expr)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        # code mostly copied from ccode
        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "  "
        inc_regex = ('^function ', '^if ', '^elseif ', '^else$', '^for ')
        dec_regex = ('^end$', '^elseif ', '^else$')

        # pre-strip left-space from the code
        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(search(re, line) for re in inc_regex))
                     for line in code ]
        decrease = [ int(any(search(re, line) for re in dec_regex))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty


def octave_code(expr, assign_to=None, **settings):
    r"""Converts `expr` to a string of Octave (or Matlab) code.

    The string uses a subset of the Octave language for Matlab compatibility.

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned.  Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type.  This can be helpful for
        expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi  [default=16].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations.  Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)].  See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols.  If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text).  [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].
    inline: bool, optional
        If True, we try to create single-statement code instead of multiple
        statements.  [default=True].

    Examples
    ========

    >>> from sympy import octave_code, symbols, sin, pi
    >>> x = symbols('x')
    >>> octave_code(sin(x).series(x).removeO())
    'x.^5/120 - x.^3/6 + x'

    >>> from sympy import Rational, ceiling
    >>> x, y, tau = symbols("x, y, tau")
    >>> octave_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau.^(7/2)'

    Note that element-wise (Hadamard) operations are used by default between
    symbols.  This is because its very common in Octave to write "vectorized"
    code.  It is harmless if the values are scalars.

    >>> octave_code(sin(pi*x*y), assign_to="s")
    's = sin(pi*x.*y);'

    If you need a matrix product "*" or matrix power "^", you can specify the
    symbol as a ``MatrixSymbol``.

    >>> from sympy import Symbol, MatrixSymbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> octave_code(3*pi*A**3)
    '(3*pi)*A^3'

    This class uses several rules to decide which symbol to use a product.
    Pure numbers use "*", Symbols use ".*" and MatrixSymbols use "*".
    A HadamardProduct can be used to specify componentwise multiplication ".*"
    of two MatrixSymbols.  There is currently there is no easy way to specify
    scalar symbols, so sometimes the code might have some minor cosmetic
    issues.  For example, suppose x and y are scalars and A is a Matrix, then
    while a human programmer might write "(x^2*y)*A^3", we generate:

    >>> octave_code(x**2*y*A**3)
    '(x.^2.*y)*A^3'

    Matrices are supported using Octave inline notation.  When using
    ``assign_to`` with matrices, the name can be specified either as a string
    or as a ``MatrixSymbol``.  The dimensions must align in the latter case.

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([[x**2, sin(x), ceiling(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 sin(x) ceil(x)];'

    ``Piecewise`` expressions are implemented with logical masking by default.
    Alternatively, you can pass "inline=False" to use if-else conditionals.
    Note that if the ``Piecewise`` lacks a default term, represented by
    ``(expr, True)`` then an error will be thrown.  This is to prevent
    generating an expression that may not evaluate to anything.

    >>> from sympy import Piecewise
    >>> pw = Piecewise((x + 1, x > 0), (x, True))
    >>> octave_code(pw, assign_to=tau)
    'tau = ((x > 0).*(x + 1) + (~(x > 0)).*(x));'

    Note that any expression that can be generated normally can also exist
    inside a Matrix:

    >>> mat = Matrix([[x**2, pw, sin(x)]])
    >>> octave_code(mat, assign_to='A')
    'A = [x.^2 ((x > 0).*(x + 1) + (~(x > 0)).*(x)) sin(x)];'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg.  Alternatively, the
    dictionary value can be a list of tuples i.e., [(argument_test,
    cfunction_string)].  This can be used to call a custom Octave function.

    >>> from sympy import Function
    >>> f = Function('f')
    >>> g = Function('g')
    >>> custom_functions = {
    ...   "f": "existing_octave_fcn",
    ...   "g": [(lambda x: x.is_Matrix, "my_mat_fcn"),
    ...         (lambda x: not x.is_Matrix, "my_fcn")]
    ... }
    >>> mat = Matrix([[1, x]])
    >>> octave_code(f(x) + g(x) + g(mat), user_functions=custom_functions)
    'existing_octave_fcn(x) + my_fcn(x) + my_mat_fcn([1 x])'

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e = Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> octave_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy(i) = (y(i + 1) - y(i))./(t(i + 1) - t(i));'
    """
    return OctaveCodePrinter(settings).doprint(expr, assign_to)


def print_octave_code(expr, **settings):
    """Prints the Octave (or Matlab) representation of the given expression.

    See `octave_code` for the meaning of the optional arguments.
    """
    print(octave_code(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\validators.py ===
# SPDX-License-Identifier: MIT

"""
Commonly useful validators.
"""

import operator
import re

from contextlib import contextmanager
from re import Pattern

from ._config import get_run_validators, set_run_validators
from ._make import _AndValidator, and_, attrib, attrs
from .converters import default_if_none
from .exceptions import NotCallableError


__all__ = [
    "and_",
    "deep_iterable",
    "deep_mapping",
    "disabled",
    "ge",
    "get_disabled",
    "gt",
    "in_",
    "instance_of",
    "is_callable",
    "le",
    "lt",
    "matches_re",
    "max_len",
    "min_len",
    "not_",
    "optional",
    "or_",
    "set_disabled",
]


def set_disabled(disabled):
    """
    Globally disable or enable running validators.

    By default, they are run.

    Args:
        disabled (bool): If `True`, disable running all validators.

    .. warning::

        This function is not thread-safe!

    .. versionadded:: 21.3.0
    """
    set_run_validators(not disabled)


def get_disabled():
    """
    Return a bool indicating whether validators are currently disabled or not.

    Returns:
        bool:`True` if validators are currently disabled.

    .. versionadded:: 21.3.0
    """
    return not get_run_validators()


@contextmanager
def disabled():
    """
    Context manager that disables running validators within its context.

    .. warning::

        This context manager is not thread-safe!

    .. versionadded:: 21.3.0
    """
    set_run_validators(False)
    try:
        yield
    finally:
        set_run_validators(True)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _InstanceOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not isinstance(value, self.type):
            msg = f"'{attr.name}' must be {self.type!r} (got {value!r} that is a {value.__class__!r})."
            raise TypeError(
                msg,
                attr,
                self.type,
                value,
            )

    def __repr__(self):
        return f"<instance_of validator for type {self.type!r}>"


def instance_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called with a
    wrong type for this particular attribute (checks are performed using
    `isinstance` therefore it's also valid to pass a tuple of types).

    Args:
        type (type | tuple[type]): The type to check for.

    Raises:
        TypeError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected type, and the value it got.
    """
    return _InstanceOfValidator(type)


@attrs(repr=False, frozen=True, slots=True)
class _MatchesReValidator:
    pattern = attrib()
    match_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.match_func(value):
            msg = f"'{attr.name}' must match regex {self.pattern.pattern!r} ({value!r} doesn't)"
            raise ValueError(
                msg,
                attr,
                self.pattern,
                value,
            )

    def __repr__(self):
        return f"<matches_re validator for pattern {self.pattern!r}>"


def matches_re(regex, flags=0, func=None):
    r"""
    A validator that raises `ValueError` if the initializer is called with a
    string that doesn't match *regex*.

    Args:
        regex (str, re.Pattern):
            A regex string or precompiled pattern to match against

        flags (int):
            Flags that will be passed to the underlying re function (default 0)

        func (typing.Callable):
            Which underlying `re` function to call. Valid options are
            `re.fullmatch`, `re.search`, and `re.match`; the default `None`
            means `re.fullmatch`. For performance reasons, the pattern is
            always precompiled using `re.compile`.

    .. versionadded:: 19.2.0
    .. versionchanged:: 21.3.0 *regex* can be a pre-compiled pattern.
    """
    valid_funcs = (re.fullmatch, None, re.search, re.match)
    if func not in valid_funcs:
        msg = "'func' must be one of {}.".format(
            ", ".join(
                sorted((e and e.__name__) or "None" for e in set(valid_funcs))
            )
        )
        raise ValueError(msg)

    if isinstance(regex, Pattern):
        if flags:
            msg = "'flags' can only be used with a string pattern; pass flags to re.compile() instead"
            raise TypeError(msg)
        pattern = regex
    else:
        pattern = re.compile(regex, flags)

    if func is re.match:
        match_func = pattern.match
    elif func is re.search:
        match_func = pattern.search
    else:
        match_func = pattern.fullmatch

    return _MatchesReValidator(pattern, match_func)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _OptionalValidator:
    validator = attrib()

    def __call__(self, inst, attr, value):
        if value is None:
            return

        self.validator(inst, attr, value)

    def __repr__(self):
        return f"<optional validator for {self.validator!r} or None>"


def optional(validator):
    """
    A validator that makes an attribute optional.  An optional attribute is one
    which can be set to `None` in addition to satisfying the requirements of
    the sub-validator.

    Args:
        validator
            (typing.Callable | tuple[typing.Callable] | list[typing.Callable]):
            A validator (or validators) that is used for non-`None` values.

    .. versionadded:: 15.1.0
    .. versionchanged:: 17.1.0 *validator* can be a list of validators.
    .. versionchanged:: 23.1.0 *validator* can also be a tuple of validators.
    """
    if isinstance(validator, (list, tuple)):
        return _OptionalValidator(_AndValidator(validator))

    return _OptionalValidator(validator)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _InValidator:
    options = attrib()
    _original_options = attrib(hash=False)

    def __call__(self, inst, attr, value):
        try:
            in_options = value in self.options
        except TypeError:  # e.g. `1 in "abc"`
            in_options = False

        if not in_options:
            msg = f"'{attr.name}' must be in {self._original_options!r} (got {value!r})"
            raise ValueError(
                msg,
                attr,
                self._original_options,
                value,
            )

    def __repr__(self):
        return f"<in_ validator with options {self._original_options!r}>"


def in_(options):
    """
    A validator that raises a `ValueError` if the initializer is called with a
    value that does not belong in the *options* provided.

    The check is performed using ``value in options``, so *options* has to
    support that operation.

    To keep the validator hashable, dicts, lists, and sets are transparently
    transformed into a `tuple`.

    Args:
        options: Allowed options.

    Raises:
        ValueError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected options, and the value it got.

    .. versionadded:: 17.1.0
    .. versionchanged:: 22.1.0
       The ValueError was incomplete until now and only contained the human
       readable error message. Now it contains all the information that has
       been promised since 17.1.0.
    .. versionchanged:: 24.1.0
       *options* that are a list, dict, or a set are now transformed into a
       tuple to keep the validator hashable.
    """
    repr_options = options
    if isinstance(options, (list, dict, set)):
        options = tuple(options)

    return _InValidator(options, repr_options)


@attrs(repr=False, slots=False, unsafe_hash=True)
class _IsCallableValidator:
    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not callable(value):
            message = (
                "'{name}' must be callable "
                "(got {value!r} that is a {actual!r})."
            )
            raise NotCallableError(
                msg=message.format(
                    name=attr.name, value=value, actual=value.__class__
                ),
                value=value,
            )

    def __repr__(self):
        return "<is_callable validator>"


def is_callable():
    """
    A validator that raises a `attrs.exceptions.NotCallableError` if the
    initializer is called with a value for this particular attribute that is
    not callable.

    .. versionadded:: 19.1.0

    Raises:
        attrs.exceptions.NotCallableError:
            With a human readable error message containing the attribute
            (`attrs.Attribute`) name, and the value it got.
    """
    return _IsCallableValidator()


@attrs(repr=False, slots=True, unsafe_hash=True)
class _DeepIterable:
    member_validator = attrib(validator=is_callable())
    iterable_validator = attrib(
        default=None, validator=optional(is_callable())
    )

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.iterable_validator is not None:
            self.iterable_validator(inst, attr, value)

        for member in value:
            self.member_validator(inst, attr, member)

    def __repr__(self):
        iterable_identifier = (
            ""
            if self.iterable_validator is None
            else f" {self.iterable_validator!r}"
        )
        return (
            f"<deep_iterable validator for{iterable_identifier}"
            f" iterables of {self.member_validator!r}>"
        )


def deep_iterable(member_validator, iterable_validator=None):
    """
    A validator that performs deep validation of an iterable.

    Args:
        member_validator: Validator to apply to iterable members.

        iterable_validator:
            Validator to apply to iterable itself (optional).

    Raises
        TypeError: if any sub-validators fail

    .. versionadded:: 19.1.0
    """
    if isinstance(member_validator, (list, tuple)):
        member_validator = and_(*member_validator)
    return _DeepIterable(member_validator, iterable_validator)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _DeepMapping:
    key_validator = attrib(validator=is_callable())
    value_validator = attrib(validator=is_callable())
    mapping_validator = attrib(default=None, validator=optional(is_callable()))

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if self.mapping_validator is not None:
            self.mapping_validator(inst, attr, value)

        for key in value:
            self.key_validator(inst, attr, key)
            self.value_validator(inst, attr, value[key])

    def __repr__(self):
        return f"<deep_mapping validator for objects mapping {self.key_validator!r} to {self.value_validator!r}>"


def deep_mapping(key_validator, value_validator, mapping_validator=None):
    """
    A validator that performs deep validation of a dictionary.

    Args:
        key_validator: Validator to apply to dictionary keys.

        value_validator: Validator to apply to dictionary values.

        mapping_validator:
            Validator to apply to top-level mapping attribute (optional).

    .. versionadded:: 19.1.0

    Raises:
        TypeError: if any sub-validators fail
    """
    return _DeepMapping(key_validator, value_validator, mapping_validator)


@attrs(repr=False, frozen=True, slots=True)
class _NumberValidator:
    bound = attrib()
    compare_op = attrib()
    compare_func = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not self.compare_func(value, self.bound):
            msg = f"'{attr.name}' must be {self.compare_op} {self.bound}: {value}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<Validator for x {self.compare_op} {self.bound}>"


def lt(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number larger or equal to *val*.

    The validator uses `operator.lt` to compare the values.

    Args:
        val: Exclusive upper bound for values.

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, "<", operator.lt)


def le(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number greater than *val*.

    The validator uses `operator.le` to compare the values.

    Args:
        val: Inclusive upper bound for values.

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, "<=", operator.le)


def ge(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number smaller than *val*.

    The validator uses `operator.ge` to compare the values.

    Args:
        val: Inclusive lower bound for values

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, ">=", operator.ge)


def gt(val):
    """
    A validator that raises `ValueError` if the initializer is called with a
    number smaller or equal to *val*.

    The validator uses `operator.ge` to compare the values.

    Args:
       val: Exclusive lower bound for values

    .. versionadded:: 21.3.0
    """
    return _NumberValidator(val, ">", operator.gt)


@attrs(repr=False, frozen=True, slots=True)
class _MaxLengthValidator:
    max_length = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if len(value) > self.max_length:
            msg = f"Length of '{attr.name}' must be <= {self.max_length}: {len(value)}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<max_len validator for {self.max_length}>"


def max_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is longer than *length*.

    Args:
        length (int): Maximum length of the string or iterable

    .. versionadded:: 21.3.0
    """
    return _MaxLengthValidator(length)


@attrs(repr=False, frozen=True, slots=True)
class _MinLengthValidator:
    min_length = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if len(value) < self.min_length:
            msg = f"Length of '{attr.name}' must be >= {self.min_length}: {len(value)}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<min_len validator for {self.min_length}>"


def min_len(length):
    """
    A validator that raises `ValueError` if the initializer is called
    with a string or iterable that is shorter than *length*.

    Args:
        length (int): Minimum length of the string or iterable

    .. versionadded:: 22.1.0
    """
    return _MinLengthValidator(length)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _SubclassOfValidator:
    type = attrib()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        if not issubclass(value, self.type):
            msg = f"'{attr.name}' must be a subclass of {self.type!r} (got {value!r})."
            raise TypeError(
                msg,
                attr,
                self.type,
                value,
            )

    def __repr__(self):
        return f"<subclass_of validator for type {self.type!r}>"


def _subclass_of(type):
    """
    A validator that raises a `TypeError` if the initializer is called with a
    wrong type for this particular attribute (checks are performed using
    `issubclass` therefore it's also valid to pass a tuple of types).

    Args:
        type (type | tuple[type, ...]): The type(s) to check for.

    Raises:
        TypeError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the expected type, and the value it got.
    """
    return _SubclassOfValidator(type)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _NotValidator:
    validator = attrib()
    msg = attrib(
        converter=default_if_none(
            "not_ validator child '{validator!r}' "
            "did not raise a captured error"
        )
    )
    exc_types = attrib(
        validator=deep_iterable(
            member_validator=_subclass_of(Exception),
            iterable_validator=instance_of(tuple),
        ),
    )

    def __call__(self, inst, attr, value):
        try:
            self.validator(inst, attr, value)
        except self.exc_types:
            pass  # suppress error to invert validity
        else:
            raise ValueError(
                self.msg.format(
                    validator=self.validator,
                    exc_types=self.exc_types,
                ),
                attr,
                self.validator,
                value,
                self.exc_types,
            )

    def __repr__(self):
        return f"<not_ validator wrapping {self.validator!r}, capturing {self.exc_types!r}>"


def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
    """
    A validator that wraps and logically 'inverts' the validator passed to it.
    It will raise a `ValueError` if the provided validator *doesn't* raise a
    `ValueError` or `TypeError` (by default), and will suppress the exception
    if the provided validator *does*.

    Intended to be used with existing validators to compose logic without
    needing to create inverted variants, for example, ``not_(in_(...))``.

    Args:
        validator: A validator to be logically inverted.

        msg (str):
            Message to raise if validator fails. Formatted with keys
            ``exc_types`` and ``validator``.

        exc_types (tuple[type, ...]):
            Exception type(s) to capture. Other types raised by child
            validators will not be intercepted and pass through.

    Raises:
        ValueError:
            With a human readable error message, the attribute (of type
            `attrs.Attribute`), the validator that failed to raise an
            exception, the value it got, and the expected exception types.

    .. versionadded:: 22.2.0
    """
    try:
        exc_types = tuple(exc_types)
    except TypeError:
        exc_types = (exc_types,)
    return _NotValidator(validator, msg, exc_types)


@attrs(repr=False, slots=True, unsafe_hash=True)
class _OrValidator:
    validators = attrib()

    def __call__(self, inst, attr, value):
        for v in self.validators:
            try:
                v(inst, attr, value)
            except Exception:  # noqa: BLE001, PERF203, S112
                continue
            else:
                return

        msg = f"None of {self.validators!r} satisfied for value {value!r}"
        raise ValueError(msg)

    def __repr__(self):
        return f"<or validator wrapping {self.validators!r}>"


def or_(*validators):
    """
    A validator that composes multiple validators into one.

    When called on a value, it runs all wrapped validators until one of them is
    satisfied.

    Args:
        validators (~collections.abc.Iterable[typing.Callable]):
            Arbitrary number of validators.

    Raises:
        ValueError:
            If no validator is satisfied. Raised with a human-readable error
            message listing all the wrapped validators and the value that
            failed all of them.

    .. versionadded:: 24.1.0
    """
    vals = []
    for v in validators:
        vals.extend(v.validators if isinstance(v, _OrValidator) else [v])

    return _OrValidator(tuple(vals))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\pc_groups.py ===
from sympy.ntheory.primetest import isprime
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.printing.defaults import DefaultPrinting
from sympy.combinatorics.free_groups import free_group


class PolycyclicGroup(DefaultPrinting):

    is_group = True
    is_solvable = True

    def __init__(self, pc_sequence, pc_series, relative_order, collector=None):
        """

        Parameters
        ==========

        pc_sequence : list
            A sequence of elements whose classes generate the cyclic factor
            groups of pc_series.
        pc_series : list
            A subnormal sequence of subgroups where each factor group is cyclic.
        relative_order : list
            The orders of factor groups of pc_series.
        collector : Collector
            By default, it is None. Collector class provides the
            polycyclic presentation with various other functionalities.

        """
        self.pcgs = pc_sequence
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.collector = Collector(self.pcgs, pc_series, relative_order) if not collector else collector

    def is_prime_order(self):
        return all(isprime(order) for order in self.relative_order)

    def length(self):
        return len(self.pcgs)


class Collector(DefaultPrinting):

    """
    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"
           Section 8.1.3
    """

    def __init__(self, pcgs, pc_series, relative_order, free_group_=None, pc_presentation=None):
        """

        Most of the parameters for the Collector class are the same as for PolycyclicGroup.
        Others are described below.

        Parameters
        ==========

        free_group_ : tuple
            free_group_ provides the mapping of polycyclic generating
            sequence with the free group elements.
        pc_presentation : dict
            Provides the presentation of polycyclic groups with the
            help of power and conjugate relators.

        See Also
        ========

        PolycyclicGroup

        """
        self.pcgs = pcgs
        self.pc_series = pc_series
        self.relative_order = relative_order
        self.free_group = free_group('x:{}'.format(len(pcgs)))[0] if not free_group_ else free_group_
        self.index = {s: i for i, s in enumerate(self.free_group.symbols)}
        self.pc_presentation = self.pc_relators()

    def minimal_uncollected_subword(self, word):
        r"""
        Returns the minimal uncollected subwords.

        Explanation
        ===========

        A word ``v`` defined on generators in ``X`` is a minimal
        uncollected subword of the word ``w`` if ``v`` is a subword
        of ``w`` and it has one of the following form

        * `v = {x_{i+1}}^{a_j}x_i`

        * `v = {x_{i+1}}^{a_j}{x_i}^{-1}`

        * `v = {x_i}^{a_j}`

        for `a_j` not in `\{1, \ldots, s-1\}`. Where, ``s`` is the power
        exponent of the corresponding generator.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> collector.minimal_uncollected_subword(word)
        ((x2, 2),)

        """
        # To handle the case word = <identity>
        if not word:
            return None

        array = word.array_form
        re = self.relative_order
        index = self.index

        for i in range(len(array)):
            s1, e1 = array[i]

            if re[index[s1]] and (e1 < 0 or e1 > re[index[s1]]-1):
                return ((s1, e1), )

        for i in range(len(array)-1):
            s1, e1 = array[i]
            s2, e2 = array[i+1]

            if index[s1] > index[s2]:
                e = 1 if e2 > 0 else -1
                return ((s1, e1), (s2, e))

        return None

    def relations(self):
        """
        Separates the given relators of pc presentation in power and
        conjugate relations.

        Returns
        =======

        (power_rel, conj_rel)
            Separates pc presentation into power and conjugate relations.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> power_rel, conj_rel = collector.relations()
        >>> power_rel
        {x0**2: (), x1**3: ()}
        >>> conj_rel
        {x0**-1*x1*x0: x1**2}

        See Also
        ========

        pc_relators

        """
        power_relators = {}
        conjugate_relators = {}
        for key, value in self.pc_presentation.items():
            if len(key.array_form) == 1:
                power_relators[key] = value
            else:
                conjugate_relators[key] = value
        return power_relators, conjugate_relators

    def subword_index(self, word, w):
        """
        Returns the start and ending index of a given
        subword in a word.

        Parameters
        ==========

        word : FreeGroupElement
            word defined on free group elements for a
            polycyclic group.
        w : FreeGroupElement
            subword of a given word, whose starting and
            ending index to be computed.

        Returns
        =======

        (i, j)
            A tuple containing starting and ending index of ``w``
            in the given word. If not exists, (-1,-1) is returned.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x1, x2 = free_group("x1, x2")
        >>> word = x2**2*x1**7
        >>> w = x2**2*x1
        >>> collector.subword_index(word, w)
        (0, 3)
        >>> w = x1**7
        >>> collector.subword_index(word, w)
        (2, 9)
        >>> w = x1**8
        >>> collector.subword_index(word, w)
        (-1, -1)

        """
        low = -1
        high = -1
        for i in range(len(word)-len(w)+1):
            if word.subword(i, i+len(w)) == w:
                low = i
                high = i+len(w)
                break
        return low, high

    def map_relation(self, w):
        """
        Return a conjugate relation.

        Explanation
        ===========

        Given a word formed by two free group elements, the
        corresponding conjugate relation with those free
        group elements is formed and mapped with the collected
        word in the polycyclic presentation.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1 = free_group("x0, x1")
        >>> w = x1*x0
        >>> collector.map_relation(w)
        x1**2

        See Also
        ========

        pc_presentation

        """
        array = w.array_form
        s1 = array[0][0]
        s2 = array[1][0]
        key = ((s2, -1), (s1, 1), (s2, 1))
        key = self.free_group.dtype(key)
        return self.pc_presentation[key]


    def collected_word(self, word):
        r"""
        Return the collected form of a word.

        Explanation
        ===========

        A word ``w`` is called collected, if `w = {x_{i_1}}^{a_1} * \ldots *
        {x_{i_r}}^{a_r}` with `i_1 < i_2< \ldots < i_r` and `a_j` is in
        `\{1, \ldots, {s_j}-1\}`.

        Otherwise w is uncollected.

        Parameters
        ==========

        word : FreeGroupElement
            An uncollected word.

        Returns
        =======

        word
            A collected word of form `w = {x_{i_1}}^{a_1}, \ldots,
            {x_{i_r}}^{a_r}` with `i_1, i_2, \ldots, i_r` and `a_j \in
            \{1, \ldots, {s_j}-1\}`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics import free_group
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> F, x0, x1, x2, x3 = free_group("x0, x1, x2, x3")
        >>> word = x3*x2*x1*x0
        >>> collected_word = collector.collected_word(word)
        >>> free_to_perm = {}
        >>> free_group = collector.free_group
        >>> for sym, gen in zip(free_group.symbols, collector.pcgs):
        ...     free_to_perm[sym] = gen
        >>> G1 = PermutationGroup()
        >>> for w in word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G1 = PermutationGroup([perm] + G1.generators)
        >>> G2 = PermutationGroup()
        >>> for w in collected_word:
        ...     sym = w[0]
        ...     perm = free_to_perm[sym]
        ...     G2 = PermutationGroup([perm] + G2.generators)

        The two are not identical, but they are equivalent:

        >>> G1.equals(G2), G1 == G2
        (True, False)

        See Also
        ========

        minimal_uncollected_subword

        """
        free_group = self.free_group
        while True:
            w = self.minimal_uncollected_subword(word)
            if not w:
                break

            low, high = self.subword_index(word, free_group.dtype(w))
            if low == -1:
                continue

            s1, e1 = w[0]
            if len(w) == 1:
                re = self.relative_order[self.index[s1]]
                q = e1 // re
                r = e1-q*re

                key = ((w[0][0], re), )
                key = free_group.dtype(key)
                if self.pc_presentation[key]:
                    presentation = self.pc_presentation[key].array_form
                    sym, exp = presentation[0]
                    word_ = ((w[0][0], r), (sym, q*exp))
                    word_ = free_group.dtype(word_)
                else:
                    if r != 0:
                        word_ = ((w[0][0], r), )
                        word_ = free_group.dtype(word_)
                    else:
                        word_ = None
                word = word.eliminate_word(free_group.dtype(w), word_)

            if len(w) == 2 and w[1][1] > 0:
                s2, e2 = w[1]
                s2 = ((s2, 1), )
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2*word_**e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)

            elif len(w) == 2 and w[1][1] < 0:
                s2, e2 = w[1]
                s2 = ((s2, 1), )
                s2 = free_group.dtype(s2)
                word_ = self.map_relation(free_group.dtype(w))
                word_ = s2**-1*word_**e1
                word_ = free_group.dtype(word_)
                word = word.substituted_word(low, high, word_)

        return word


    def pc_relators(self):
        r"""
        Return the polycyclic presentation.

        Explanation
        ===========

        There are two types of relations used in polycyclic
        presentation.

        * Power relations : Power relators are of the form `x_i^{re_i}`,
          where `i \in \{0, \ldots, \mathrm{len(pcgs)}\}`, ``x`` represents polycyclic
          generator and ``re`` is the corresponding relative order.

        * Conjugate relations : Conjugate relators are of the form `x_j^-1x_ix_j`,
          where `j < i \in \{0, \ldots, \mathrm{len(pcgs)}\}`.

        Returns
        =======

        A dictionary with power and conjugate relations as key and
        their collected form as corresponding values.

        Notes
        =====

        Identity Permutation is mapped with empty ``()``.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> S = SymmetricGroup(49).sylow_subgroup(7)
        >>> der = S.derived_series()
        >>> G = der[len(der)-2]
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> len(pcgs)
        6
        >>> free_group = collector.free_group
        >>> pc_resentation = collector.pc_presentation
        >>> free_to_perm = {}
        >>> for s, g in zip(free_group.symbols, pcgs):
        ...     free_to_perm[s] = g

        >>> for k, v in pc_resentation.items():
        ...     k_array = k.array_form
        ...     if v != ():
        ...        v_array = v.array_form
        ...     lhs = Permutation()
        ...     for gen in k_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         lhs = lhs*free_to_perm[s]**e
        ...     if v == ():
        ...         assert lhs.is_identity
        ...         continue
        ...     rhs = Permutation()
        ...     for gen in v_array:
        ...         s = gen[0]
        ...         e = gen[1]
        ...         rhs = rhs*free_to_perm[s]**e
        ...     assert lhs == rhs

        """
        free_group = self.free_group
        rel_order = self.relative_order
        pc_relators = {}
        perm_to_free = {}
        pcgs = self.pcgs

        for gen, s in zip(pcgs, free_group.generators):
            perm_to_free[gen**-1] = s**-1
            perm_to_free[gen] = s

        pcgs = pcgs[::-1]
        series = self.pc_series[::-1]
        rel_order = rel_order[::-1]
        collected_gens = []

        for i, gen in enumerate(pcgs):
            re = rel_order[i]
            relation = perm_to_free[gen]**re
            G = series[i]

            l = G.generator_product(gen**re, original = True)
            l.reverse()

            word = free_group.identity
            for g in l:
                word = word*perm_to_free[g]

            word = self.collected_word(word)
            pc_relators[relation] = word if word else ()
            self.pc_presentation = pc_relators

            collected_gens.append(gen)
            if len(collected_gens) > 1:
                conj = collected_gens[len(collected_gens)-1]
                conjugator = perm_to_free[conj]

                for j in range(len(collected_gens)-1):
                    conjugated = perm_to_free[collected_gens[j]]

                    relation = conjugator**-1*conjugated*conjugator
                    gens = conj**-1*collected_gens[j]*conj

                    l = G.generator_product(gens, original = True)
                    l.reverse()
                    word = free_group.identity
                    for g in l:
                        word = word*perm_to_free[g]

                    word = self.collected_word(word)
                    pc_relators[relation] = word if word else ()
                    self.pc_presentation = pc_relators

        return pc_relators

    def exponent_vector(self, element):
        r"""
        Return the exponent vector of length equal to the
        length of polycyclic generating sequence.

        Explanation
        ===========

        For a given generator/element ``g`` of the polycyclic group,
        it can be represented as `g = {x_1}^{e_1}, \ldots, {x_n}^{e_n}`,
        where `x_i` represents polycyclic generators and ``n`` is
        the number of generators in the free_group equal to the length
        of pcgs.

        Parameters
        ==========

        element : Permutation
            Generator of a polycyclic group.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> G = SymmetricGroup(4)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> pcgs = PcGroup.pcgs
        >>> collector.exponent_vector(G[0])
        [1, 0, 0, 0]
        >>> exp = collector.exponent_vector(G[1])
        >>> g = Permutation()
        >>> for i in range(len(exp)):
        ...     g = g*pcgs[i]**exp[i] if exp[i] else g
        >>> assert g == G[1]

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.4

        """
        free_group = self.free_group
        G = PermutationGroup()
        for g in self.pcgs:
            G = PermutationGroup([g] + G.generators)
        gens = G.generator_product(element, original = True)
        gens.reverse()

        perm_to_free = {}
        for sym, g in zip(free_group.generators, self.pcgs):
            perm_to_free[g**-1] = sym**-1
            perm_to_free[g] = sym
        w = free_group.identity
        for g in gens:
            w = w*perm_to_free[g]

        word = self.collected_word(w)

        index = self.index
        exp_vector = [0]*len(free_group)
        word = word.array_form
        for t in word:
            exp_vector[index[t[0]]] = t[1]
        return exp_vector

    def depth(self, element):
        r"""
        Return the depth of a given element.

        Explanation
        ===========

        The depth of a given element ``g`` is defined by
        `\mathrm{dep}[g] = i` if `e_1 = e_2 = \ldots = e_{i-1} = 0`
        and `e_i != 0`, where ``e`` represents the exponent-vector.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.depth(G[0])
        2
        >>> collector.depth(G[1])
        1

        References
        ==========

        .. [1] Holt, D., Eick, B., O'Brien, E.
               "Handbook of Computational Group Theory"
               Section 8.1.1, Definition 8.5

        """
        exp_vector = self.exponent_vector(element)
        return next((i+1 for i, x in enumerate(exp_vector) if x), len(self.pcgs)+1)

    def leading_exponent(self, element):
        r"""
        Return the leading non-zero exponent.

        Explanation
        ===========

        The leading exponent for a given element `g` is defined
        by `\mathrm{leading\_exponent}[g]` `= e_i`, if `\mathrm{depth}[g] = i`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> G = SymmetricGroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> collector.leading_exponent(G[1])
        1

        """
        exp_vector = self.exponent_vector(element)
        depth = self.depth(element)
        if depth != len(self.pcgs)+1:
            return exp_vector[depth-1]
        return None

    def _sift(self, z, g):
        h = g
        d = self.depth(h)
        while d < len(self.pcgs) and z[d-1] != 1:
            k = z[d-1]
            e = self.leading_exponent(h)*(self.leading_exponent(k))**-1
            e = e % self.relative_order[d-1]
            h = k**-e*h
            d = self.depth(h)
        return h

    def induced_pcgs(self, gens):
        """

        Parameters
        ==========

        gens : list
            A list of generators on which polycyclic subgroup
            is to be defined.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import SymmetricGroup
        >>> S = SymmetricGroup(8)
        >>> G = S.sylow_subgroup(2)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [2, 2, 2]
        >>> G = S.sylow_subgroup(3)
        >>> PcGroup = G.polycyclic_group()
        >>> collector = PcGroup.collector
        >>> gens = [G[0], G[1]]
        >>> ipcgs = collector.induced_pcgs(gens)
        >>> [gen.order() for gen in ipcgs]
        [3]

        """
        z = [1]*len(self.pcgs)
        G = gens
        while G:
            g = G.pop(0)
            h = self._sift(z, g)
            d = self.depth(h)
            if d < len(self.pcgs):
                for gen in z:
                    if gen != 1:
                        G.append(h**-1*gen**-1*h*gen)
                z[d-1] = h
        z = [gen for gen in z if gen != 1]
        return z

    def constructive_membership_test(self, ipcgs, g):
        """
        Return the exponent vector for induced pcgs.
        """
        e = [0]*len(ipcgs)
        h = g
        d = self.depth(h)
        for i, gen in enumerate(ipcgs):
            while self.depth(gen) == d:
                f = self.leading_exponent(h)*self.leading_exponent(gen)
                f = f % self.relative_order[d-1]
                h = gen**(-f)*h
                e[i] = f
                d = self.depth(h)
        if h == 1:
            return e
        return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\integers.py ===
from __future__ import annotations

from sympy.core.basic import Basic
from sympy.core.expr import Expr

from sympy.core import Add, S
from sympy.core.evalf import get_integer_part, PrecisionExhausted
from sympy.core.function import DefinedFunction
from sympy.core.logic import fuzzy_or, fuzzy_and
from sympy.core.numbers import Integer, int_valued
from sympy.core.relational import Gt, Lt, Ge, Le, Relational, is_eq, is_le, is_lt
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import im, re
from sympy.multipledispatch import dispatch

###############################################################################
######################### FLOOR and CEILING FUNCTIONS #########################
###############################################################################


class RoundFunction(DefinedFunction):
    """Abstract base class for rounding functions."""

    args: tuple[Expr]

    @classmethod
    def eval(cls, arg):
        if (v := cls._eval_number(arg)) is not None:
            return v
        if (v := cls._eval_const_number(arg)) is not None:
            return v

        if arg.is_integer or arg.is_finite is False:
            return arg
        if arg.is_imaginary or (S.ImaginaryUnit*arg).is_real:
            i = im(arg)
            if not i.has(S.ImaginaryUnit):
                return cls(i)*S.ImaginaryUnit
            return cls(arg, evaluate=False)

        # Integral, numerical, symbolic part
        ipart = npart = spart = S.Zero

        # Extract integral (or complex integral) terms
        intof = lambda x: int(x) if int_valued(x) else (
            x if x.is_integer else None)
        for t in Add.make_args(arg):
            if t.is_imaginary and (i := intof(im(t))) is not None:
                ipart += i*S.ImaginaryUnit
            elif (i := intof(t)) is not None:
                ipart += i
            elif t.is_number:
                npart += t
            else:
                spart += t

        if not (npart or spart):
            return ipart

        # Evaluate npart numerically if independent of spart
        if npart and (
            not spart or
            npart.is_real and (spart.is_imaginary or (S.ImaginaryUnit*spart).is_real) or
                npart.is_imaginary and spart.is_real):
            try:
                r, i = get_integer_part(
                    npart, cls._dir, {}, return_ints=True)
                ipart += Integer(r) + Integer(i)*S.ImaginaryUnit
                npart = S.Zero
            except (PrecisionExhausted, NotImplementedError):
                pass

        spart += npart
        if not spart:
            return ipart
        elif spart.is_imaginary or (S.ImaginaryUnit*spart).is_real:
            return ipart + cls(im(spart), evaluate=False)*S.ImaginaryUnit
        elif isinstance(spart, (floor, ceiling)):
            return ipart + spart
        else:
            return ipart + cls(spart, evaluate=False)

    @classmethod
    def _eval_number(cls, arg):
        raise NotImplementedError()

    def _eval_is_finite(self):
        return self.args[0].is_finite

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_integer(self):
        return self.args[0].is_real


class floor(RoundFunction):
    """
    Floor is a univariate function which returns the largest integer
    value not greater than its argument. This implementation
    generalizes floor to complex numbers by taking the floor of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import floor, E, I, S, Float, Rational
    >>> floor(17)
    17
    >>> floor(Rational(23, 10))
    2
    >>> floor(2*E)
    5
    >>> floor(-Float(0.567))
    -1
    >>> floor(-I/2)
    -I
    >>> floor(S(5)/2 + 5*I/2)
    2 + 2*I

    See Also
    ========

    sympy.functions.elementary.integers.ceiling

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] https://mathworld.wolfram.com/FloorFunction.html

    """
    _dir = -1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.floor()
        if any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[0]

    @classmethod
    def _eval_const_number(cls, arg):
        if arg.is_real:
            if arg.is_zero:
                return S.Zero
            if arg.is_positive:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # 0 <= num/den < 1 -> 0
                if is_lt(num, den):
                    return S.Zero
                # 1 <= num/den < 2 -> 1
                if fuzzy_and([is_le(den, num), is_lt(num, 2*den)]):
                    return S.One
            if arg.is_negative:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # -1 <= num/den < 0 -> -1
                if is_le(-den, num):
                    return S.NegativeOne
                # -2 <= num/den < -1 -> -2
                if fuzzy_and([is_le(-2*den, num), is_lt(num, -den)]):
                    return Integer(-2)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        if arg0.is_finite:
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
                if ndir.is_negative:
                    return r - 1
                elif ndir.is_positive:
                    return r
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ndir)
            else:
                return r
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = floor(arg0)
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(-1, 0)
            return s + o
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            if ndir.is_negative:
                return r - 1
            elif ndir.is_positive:
                return r
            else:
                raise NotImplementedError("Not sure of sign of %s" % ndir)
        else:
            return r

    def _eval_is_negative(self):
        return self.args[0].is_negative

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return -ceiling(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg - frac(arg)

    def __le__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] < other + 1
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.Infinity and self.is_finite:
            return S.true

        return Le(self, other, evaluate=False)

    def __ge__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] >= other
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] >= other + 1
            if other.is_number and other.is_real:
                return self.args[0] >= ceiling(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Gt(self, other, evaluate=False)

    def __lt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] < other
            if other.is_number and other.is_real:
                return self.args[0] < ceiling(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.true
        if other is S.Infinity and self.is_finite:
            return S.true

        return Lt(self, other, evaluate=False)


@dispatch(floor, Expr)
def _eval_is_eq(lhs, rhs): # noqa:F811
    return is_eq(lhs.rewrite(ceiling), rhs) or \
        is_eq(lhs.rewrite(frac),rhs)


class ceiling(RoundFunction):
    """
    Ceiling is a univariate function which returns the smallest integer
    value not less than its argument. This implementation
    generalizes ceiling to complex numbers by taking the ceiling of the
    real and imaginary parts separately.

    Examples
    ========

    >>> from sympy import ceiling, E, I, S, Float, Rational
    >>> ceiling(17)
    17
    >>> ceiling(Rational(23, 10))
    3
    >>> ceiling(2*E)
    6
    >>> ceiling(-Float(0.567))
    0
    >>> ceiling(I/2)
    I
    >>> ceiling(S(5)/2 + 5*I/2)
    3 + 3*I

    See Also
    ========

    sympy.functions.elementary.integers.floor

    References
    ==========

    .. [1] "Concrete mathematics" by Graham, pp. 87
    .. [2] https://mathworld.wolfram.com/CeilingFunction.html

    """
    _dir = 1

    @classmethod
    def _eval_number(cls, arg):
        if arg.is_Number:
            return arg.ceiling()
        if any(isinstance(i, j)
                for i in (arg, -arg) for j in (floor, ceiling)):
            return arg
        if arg.is_NumberSymbol:
            return arg.approximation_interval(Integer)[1]

    @classmethod
    def _eval_const_number(cls, arg):
        if arg.is_real:
            if arg.is_zero:
                return S.Zero
            if arg.is_positive:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # 0 < num/den <= 1 -> 1
                if is_le(num, den):
                    return S.One
                # 1 < num/den <= 2 -> 2
                if fuzzy_and([is_lt(den, num), is_le(num, 2*den)]):
                    return Integer(2)
            if arg.is_negative:
                num, den = arg.as_numer_denom()
                s = den.is_negative
                if s is None:
                    return None
                if s:
                    num, den = -num, -den
                # -1 < num/den <= 0 -> 0
                if is_lt(-den, num):
                    return S.Zero
                # -2 < num/den <= -1 -> -1
                if fuzzy_and([is_lt(-2*den, num), is_le(num, -den)]):
                    return S.NegativeOne

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN or isinstance(arg0, AccumBounds):
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = ceiling(arg0)
        if arg0.is_finite:
            if arg0 == r:
                ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
                if ndir.is_negative:
                    return r
                elif ndir.is_positive:
                    return r + 1
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ndir)
            else:
                return r
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
            r = ceiling(arg0)
        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            from sympy.series.order import Order
            s = arg._eval_nseries(x, n, logx, cdir)
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1)
            return s + o
        if arg0 == r:
            ndir = arg.dir(x, cdir=cdir if cdir != 0 else 1)
            if ndir.is_negative:
                return r
            elif ndir.is_positive:
                return r + 1
            else:
                raise NotImplementedError("Not sure of sign of %s" % ndir)
        else:
            return r

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return -floor(-arg)

    def _eval_rewrite_as_frac(self, arg, **kwargs):
        return arg + frac(-arg)

    def _eval_is_positive(self):
        return self.args[0].is_positive

    def _eval_is_nonpositive(self):
        return self.args[0].is_nonpositive

    def __lt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] <= other - 1
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        if self.args[0] == other and other.is_real:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Lt(self, other, evaluate=False)

    def __gt__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] > other
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.true
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Gt(self, other, evaluate=False)

    def __ge__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] > other - 1
            if other.is_number and other.is_real:
                return self.args[0] > floor(other)
        if self.args[0] == other and other.is_real:
            return S.true
        if other is S.NegativeInfinity and self.is_finite:
            return S.true

        return Ge(self, other, evaluate=False)

    def __le__(self, other):
        other = S(other)
        if self.args[0].is_real:
            if other.is_integer:
                return self.args[0] <= other
            if other.is_number and other.is_real:
                return self.args[0] <= floor(other)
        if self.args[0] == other and other.is_real and other.is_noninteger:
            return S.false
        if other is S.Infinity and self.is_finite:
            return S.true

        return Le(self, other, evaluate=False)


@dispatch(ceiling, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    return is_eq(lhs.rewrite(floor), rhs) or is_eq(lhs.rewrite(frac),rhs)


class frac(DefinedFunction):
    r"""Represents the fractional part of x

    For real numbers it is defined [1]_ as

    .. math::
        x - \left\lfloor{x}\right\rfloor

    Examples
    ========

    >>> from sympy import Symbol, frac, Rational, floor, I
    >>> frac(Rational(4, 3))
    1/3
    >>> frac(-Rational(4, 3))
    2/3

    returns zero for integer arguments

    >>> n = Symbol('n', integer=True)
    >>> frac(n)
    0

    rewrite as floor

    >>> x = Symbol('x')
    >>> frac(x).rewrite(floor)
    x - floor(x)

    for complex arguments

    >>> r = Symbol('r', real=True)
    >>> t = Symbol('t', real=True)
    >>> frac(t + I*r)
    I*frac(r) + frac(t)

    See Also
    ========

    sympy.functions.elementary.integers.floor
    sympy.functions.elementary.integers.ceiling

    References
    ===========

    .. [1] https://en.wikipedia.org/wiki/Fractional_part
    .. [2] https://mathworld.wolfram.com/FractionalPart.html

    """
    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds

        def _eval(arg):
            if arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(0, 1)
            if arg.is_integer:
                return S.Zero
            if arg.is_number:
                if arg is S.NaN:
                    return S.NaN
                elif arg is S.ComplexInfinity:
                    return S.NaN
                else:
                    return arg - floor(arg)
            return cls(arg, evaluate=False)

        real, imag = S.Zero, S.Zero
        for t in Add.make_args(arg):
            # Two checks are needed for complex arguments
            # see issue-7649 for details
            if t.is_imaginary or (S.ImaginaryUnit*t).is_real:
                i = im(t)
                if not i.has(S.ImaginaryUnit):
                    imag += i
                else:
                    real += t
            else:
                real += t

        real = _eval(real)
        imag = _eval(imag)
        return real + S.ImaginaryUnit*imag

    def _eval_rewrite_as_floor(self, arg, **kwargs):
        return arg - floor(arg)

    def _eval_rewrite_as_ceiling(self, arg, **kwargs):
        return arg + ceiling(-arg)

    def _eval_is_finite(self):
        return True

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def _eval_is_integer(self):
        return self.args[0].is_integer

    def _eval_is_zero(self):
        return fuzzy_or([self.args[0].is_zero, self.args[0].is_integer])

    def _eval_is_negative(self):
        return False

    def __ge__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.true
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
        return Ge(self, other, evaluate=False)

    def __gt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            res = self._value_one_or_more(other)
            if res is not None:
                return not(res)
            # Check if other >= 1
            if other.is_extended_negative:
                return S.true
        return Gt(self, other, evaluate=False)

    def __le__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other < 0
            if other.is_extended_negative:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Le(self, other, evaluate=False)

    def __lt__(self, other):
        if self.is_extended_real:
            other = _sympify(other)
            # Check if other <= 0
            if other.is_extended_nonpositive:
                return S.false
            # Check if other >= 1
            res = self._value_one_or_more(other)
            if res is not None:
                return res
        return Lt(self, other, evaluate=False)

    def _value_one_or_more(self, other):
        if other.is_extended_real:
            if other.is_number:
                res = other >= 1
                if res and not isinstance(res, Relational):
                    return S.true
            if other.is_integer and other.is_positive:
                return S.true

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)

        if arg0.is_finite:
            if r.is_zero:
                ndir = arg.dir(x, cdir=cdir)
                if ndir.is_negative:
                    return S.One
                return (arg - arg0).as_leading_term(x, logx=logx, cdir=cdir)
            else:
                return r
        elif arg0 in (S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
            return AccumBounds(0, 1)
        return arg.as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        arg = self.args[0]
        arg0 = arg.subs(x, 0)
        r = self.subs(x, 0)

        if arg0.is_infinite:
            from sympy.calculus.accumulationbounds import AccumBounds
            o = Order(1, (x, 0)) if n <= 0 else AccumBounds(0, 1) + Order(x**n, (x, 0))
            return o
        else:
            res = (arg - arg0)._eval_nseries(x, n, logx=logx, cdir=cdir)
            if r.is_zero:
                ndir = arg.dir(x, cdir=cdir)
                res += S.One if ndir.is_negative else S.Zero
            else:
                res += r
            return res


@dispatch(frac, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    if (lhs.rewrite(floor) == rhs) or \
        (lhs.rewrite(ceiling) == rhs):
        return True
    # Check if other < 0
    if rhs.is_extended_negative:
        return False
    # Check if other >= 1
    res = lhs._value_one_or_more(rhs)
    if res is not None:
        return False