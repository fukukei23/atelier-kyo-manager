
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\test.py ===
from __future__ import annotations

import dataclasses
import mimetypes
import sys
import typing as t
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from itertools import chain
from random import random
from tempfile import TemporaryFile
from time import time
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from ._internal import _get_environ
from ._internal import _wsgi_decoding_dance
from ._internal import _wsgi_encoding_dance
from .datastructures import Authorization
from .datastructures import CallbackDict
from .datastructures import CombinedMultiDict
from .datastructures import EnvironHeaders
from .datastructures import FileMultiDict
from .datastructures import Headers
from .datastructures import MultiDict
from .http import dump_cookie
from .http import dump_options_header
from .http import parse_cookie
from .http import parse_date
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartEncoder
from .sansio.multipart import Preamble
from .urls import _urlencode
from .urls import iri_to_uri
from .utils import cached_property
from .utils import get_content_type
from .wrappers.request import Request
from .wrappers.response import Response
from .wsgi import ClosingIterator
from .wsgi import get_current_url

if t.TYPE_CHECKING:
    import typing_extensions as te
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


def stream_encode_multipart(
    data: t.Mapping[str, t.Any],
    use_tempfile: bool = True,
    threshold: int = 1024 * 500,
    boundary: str | None = None,
) -> tuple[t.IO[bytes], int, str]:
    """Encode a dict of values (either strings or file descriptors or
    :class:`FileStorage` objects.) into a multipart encoded string stored
    in a file descriptor.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.
    """
    if boundary is None:
        boundary = f"---------------WerkzeugFormPart_{time()}{random()}"

    stream: t.IO[bytes] = BytesIO()
    total_length = 0
    on_disk = False
    write_binary: t.Callable[[bytes], int]

    if use_tempfile:

        def write_binary(s: bytes) -> int:
            nonlocal stream, total_length, on_disk

            if on_disk:
                return stream.write(s)
            else:
                length = len(s)

                if length + total_length <= threshold:
                    stream.write(s)
                else:
                    new_stream = t.cast(t.IO[bytes], TemporaryFile("wb+"))
                    new_stream.write(stream.getvalue())  # type: ignore
                    new_stream.write(s)
                    stream = new_stream
                    on_disk = True

                total_length += length
                return length

    else:
        write_binary = stream.write

    encoder = MultipartEncoder(boundary.encode())
    write_binary(encoder.send_event(Preamble(data=b"")))
    for key, value in _iter_data(data):
        reader = getattr(value, "read", None)
        if reader is not None:
            filename = getattr(value, "filename", getattr(value, "name", None))
            content_type = getattr(value, "content_type", None)
            if content_type is None:
                content_type = (
                    filename
                    and mimetypes.guess_type(filename)[0]
                    or "application/octet-stream"
                )
            headers = value.headers
            headers.update([("Content-Type", content_type)])
            if filename is None:
                write_binary(encoder.send_event(Field(name=key, headers=headers)))
            else:
                write_binary(
                    encoder.send_event(
                        File(name=key, filename=filename, headers=headers)
                    )
                )
            while True:
                chunk = reader(16384)

                if not chunk:
                    write_binary(encoder.send_event(Data(data=chunk, more_data=False)))
                    break

                write_binary(encoder.send_event(Data(data=chunk, more_data=True)))
        else:
            if not isinstance(value, str):
                value = str(value)
            write_binary(encoder.send_event(Field(name=key, headers=Headers())))
            write_binary(encoder.send_event(Data(data=value.encode(), more_data=False)))

    write_binary(encoder.send_event(Epilogue(data=b"")))

    length = stream.tell()
    stream.seek(0)
    return stream, length, boundary


def encode_multipart(
    values: t.Mapping[str, t.Any], boundary: str | None = None
) -> tuple[str, bytes]:
    """Like `stream_encode_multipart` but returns a tuple in the form
    (``boundary``, ``data``) where data is bytes.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.
    """
    stream, length, boundary = stream_encode_multipart(
        values, use_tempfile=False, boundary=boundary
    )
    return boundary, stream.read()


def _iter_data(data: t.Mapping[str, t.Any]) -> t.Iterator[tuple[str, t.Any]]:
    """Iterate over a mapping that might have a list of values, yielding
    all key, value pairs. Almost like iter_multi_items but only allows
    lists, not tuples, of values so tuples can be used for files.
    """
    if isinstance(data, MultiDict):
        yield from data.items(multi=True)
    else:
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    yield key, v
            else:
                yield key, value


_TAnyMultiDict = t.TypeVar("_TAnyMultiDict", bound="MultiDict[t.Any, t.Any]")


class EnvironBuilder:
    """This class can be used to conveniently create a WSGI environment
    for testing purposes.  It can be used to quickly create WSGI environments
    or request objects from arbitrary data.

    The signature of this class is also used in some other places as of
    Werkzeug 0.5 (:func:`create_environ`, :meth:`Response.from_values`,
    :meth:`Client.open`).  Because of this most of the functionality is
    available through the constructor alone.

    Files and regular form data can be manipulated independently of each
    other with the :attr:`form` and :attr:`files` attributes, but are
    passed with the same argument to the constructor: `data`.

    `data` can be any of these values:

    -   a `str` or `bytes` object: The object is converted into an
        :attr:`input_stream`, the :attr:`content_length` is set and you have to
        provide a :attr:`content_type`.
    -   a `dict` or :class:`MultiDict`: The keys have to be strings. The values
        have to be either any of the following objects, or a list of any of the
        following objects:

        -   a :class:`file`-like object:  These are converted into
            :class:`FileStorage` objects automatically.
        -   a `tuple`:  The :meth:`~FileMultiDict.add_file` method is called
            with the key and the unpacked `tuple` items as positional
            arguments.
        -   a `str`:  The string is set as form data for the associated key.
    -   a file-like object: The object content is loaded in memory and then
        handled like a regular `str` or a `bytes`.

    :param path: the path of the request.  In the WSGI environment this will
                 end up as `PATH_INFO`.  If the `query_string` is not defined
                 and there is a question mark in the `path` everything after
                 it is used as query string.
    :param base_url: the base URL is a URL that is used to extract the WSGI
                     URL scheme, host (server name + server port) and the
                     script root (`SCRIPT_NAME`).
    :param query_string: an optional string or dict with URL parameters.
    :param method: the HTTP method to use, defaults to `GET`.
    :param input_stream: an optional input stream.  Do not specify this and
                         `data`.  As soon as an input stream is set you can't
                         modify :attr:`args` and :attr:`files` unless you
                         set the :attr:`input_stream` to `None` again.
    :param content_type: The content type for the request.  As of 0.5 you
                         don't have to provide this when specifying files
                         and form data via `data`.
    :param content_length: The content length for the request.  You don't
                           have to specify this when providing data via
                           `data`.
    :param errors_stream: an optional error stream that is used for
                          `wsgi.errors`.  Defaults to :data:`stderr`.
    :param multithread: controls `wsgi.multithread`.  Defaults to `False`.
    :param multiprocess: controls `wsgi.multiprocess`.  Defaults to `False`.
    :param run_once: controls `wsgi.run_once`.  Defaults to `False`.
    :param headers: an optional list or :class:`Headers` object of headers.
    :param data: a string or dict of form data or a file-object.
                 See explanation above.
    :param json: An object to be serialized and assigned to ``data``.
        Defaults the content type to ``"application/json"``.
        Serialized with the function assigned to :attr:`json_dumps`.
    :param environ_base: an optional dict of environment defaults.
    :param environ_overrides: an optional dict of environment overrides.
    :param auth: An authorization object to use for the
        ``Authorization`` header value. A ``(username, password)`` tuple
        is a shortcut for ``Basic`` authorization.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.

    .. versionchanged:: 2.1
        ``CONTENT_TYPE`` and ``CONTENT_LENGTH`` are not duplicated as
        header keys in the environ.

    .. versionchanged:: 2.0
        ``REQUEST_URI`` and ``RAW_URI`` is the full raw URI including
        the query string, not only the path.

    .. versionchanged:: 2.0
        The default :attr:`request_class` is ``Request`` instead of
        ``BaseRequest``.

    .. versionadded:: 2.0
       Added the ``auth`` parameter.

    .. versionadded:: 0.15
        The ``json`` param and :meth:`json_dumps` method.

    .. versionadded:: 0.15
        The environ has keys ``REQUEST_URI`` and ``RAW_URI`` containing
        the path before percent-decoding. This is not part of the WSGI
        PEP, but many WSGI servers include it.

    .. versionchanged:: 0.6
       ``path`` and ``base_url`` can now be unicode strings that are
       encoded with :func:`iri_to_uri`.
    """

    #: the server protocol to use.  defaults to HTTP/1.1
    server_protocol = "HTTP/1.1"

    #: the wsgi version to use.  defaults to (1, 0)
    wsgi_version = (1, 0)

    #: The default request class used by :meth:`get_request`.
    request_class = Request

    import json

    #: The serialization function used when ``json`` is passed.
    json_dumps = staticmethod(json.dumps)
    del json

    _args: MultiDict[str, str] | None
    _query_string: str | None
    _input_stream: t.IO[bytes] | None
    _form: MultiDict[str, str] | None
    _files: FileMultiDict | None

    def __init__(
        self,
        path: str = "/",
        base_url: str | None = None,
        query_string: t.Mapping[str, str] | str | None = None,
        method: str = "GET",
        input_stream: t.IO[bytes] | None = None,
        content_type: str | None = None,
        content_length: int | None = None,
        errors_stream: t.IO[str] | None = None,
        multithread: bool = False,
        multiprocess: bool = False,
        run_once: bool = False,
        headers: Headers | t.Iterable[tuple[str, str]] | None = None,
        data: None | (t.IO[bytes] | str | bytes | t.Mapping[str, t.Any]) = None,
        environ_base: t.Mapping[str, t.Any] | None = None,
        environ_overrides: t.Mapping[str, t.Any] | None = None,
        mimetype: str | None = None,
        json: t.Mapping[str, t.Any] | None = None,
        auth: Authorization | tuple[str, str] | None = None,
    ) -> None:
        if query_string is not None and "?" in path:
            raise ValueError("Query string is defined in the path and as an argument")
        request_uri = urlsplit(path)
        if query_string is None and "?" in path:
            query_string = request_uri.query

        self.path = iri_to_uri(request_uri.path)
        self.request_uri = path
        if base_url is not None:
            base_url = iri_to_uri(base_url)
        self.base_url = base_url  # type: ignore
        if isinstance(query_string, str):
            self.query_string = query_string
        else:
            if query_string is None:
                query_string = MultiDict()
            elif not isinstance(query_string, MultiDict):
                query_string = MultiDict(query_string)
            self.args = query_string
        self.method = method
        if headers is None:
            headers = Headers()
        elif not isinstance(headers, Headers):
            headers = Headers(headers)
        self.headers = headers
        if content_type is not None:
            self.content_type = content_type
        if errors_stream is None:
            errors_stream = sys.stderr
        self.errors_stream = errors_stream
        self.multithread = multithread
        self.multiprocess = multiprocess
        self.run_once = run_once
        self.environ_base = environ_base
        self.environ_overrides = environ_overrides
        self.input_stream = input_stream
        self.content_length = content_length
        self.closed = False

        if auth is not None:
            if isinstance(auth, tuple):
                auth = Authorization(
                    "basic", {"username": auth[0], "password": auth[1]}
                )

            self.headers.set("Authorization", auth.to_header())

        if json is not None:
            if data is not None:
                raise TypeError("can't provide both json and data")

            data = self.json_dumps(json)

            if self.content_type is None:
                self.content_type = "application/json"

        if data:
            if input_stream is not None:
                raise TypeError("can't provide input stream and data")
            if hasattr(data, "read"):
                data = data.read()
            if isinstance(data, str):
                data = data.encode()
            if isinstance(data, bytes):
                self.input_stream = BytesIO(data)
                if self.content_length is None:
                    self.content_length = len(data)
            else:
                for key, value in _iter_data(data):
                    if isinstance(value, (tuple, dict)) or hasattr(value, "read"):
                        self._add_file_from_data(key, value)
                    else:
                        self.form.setlistdefault(key).append(value)

        if mimetype is not None:
            self.mimetype = mimetype

    @classmethod
    def from_environ(cls, environ: WSGIEnvironment, **kwargs: t.Any) -> EnvironBuilder:
        """Turn an environ dict back into a builder. Any extra kwargs
        override the args extracted from the environ.

        .. versionchanged:: 2.0
            Path and query values are passed through the WSGI decoding
            dance to avoid double encoding.

        .. versionadded:: 0.15
        """
        headers = Headers(EnvironHeaders(environ))
        out = {
            "path": _wsgi_decoding_dance(environ["PATH_INFO"]),
            "base_url": cls._make_base_url(
                environ["wsgi.url_scheme"],
                headers.pop("Host"),
                _wsgi_decoding_dance(environ["SCRIPT_NAME"]),
            ),
            "query_string": _wsgi_decoding_dance(environ["QUERY_STRING"]),
            "method": environ["REQUEST_METHOD"],
            "input_stream": environ["wsgi.input"],
            "content_type": headers.pop("Content-Type", None),
            "content_length": headers.pop("Content-Length", None),
            "errors_stream": environ["wsgi.errors"],
            "multithread": environ["wsgi.multithread"],
            "multiprocess": environ["wsgi.multiprocess"],
            "run_once": environ["wsgi.run_once"],
            "headers": headers,
        }
        out.update(kwargs)
        return cls(**out)

    def _add_file_from_data(
        self,
        key: str,
        value: (t.IO[bytes] | tuple[t.IO[bytes], str] | tuple[t.IO[bytes], str, str]),
    ) -> None:
        """Called in the EnvironBuilder to add files from the data dict."""
        if isinstance(value, tuple):
            self.files.add_file(key, *value)
        else:
            self.files.add_file(key, value)

    @staticmethod
    def _make_base_url(scheme: str, host: str, script_root: str) -> str:
        return urlunsplit((scheme, host, script_root, "", "")).rstrip("/") + "/"

    @property
    def base_url(self) -> str:
        """The base URL is used to extract the URL scheme, host name,
        port, and root path.
        """
        return self._make_base_url(self.url_scheme, self.host, self.script_root)

    @base_url.setter
    def base_url(self, value: str | None) -> None:
        if value is None:
            scheme = "http"
            netloc = "localhost"
            script_root = ""
        else:
            scheme, netloc, script_root, qs, anchor = urlsplit(value)
            if qs or anchor:
                raise ValueError("base url must not contain a query string or fragment")
        self.script_root = script_root.rstrip("/")
        self.host = netloc
        self.url_scheme = scheme

    @property
    def content_type(self) -> str | None:
        """The content type for the request.  Reflected from and to
        the :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        ct = self.headers.get("Content-Type")
        if ct is None and not self._input_stream:
            if self._files:
                return "multipart/form-data"
            if self._form:
                return "application/x-www-form-urlencoded"
            return None
        return ct

    @content_type.setter
    def content_type(self, value: str | None) -> None:
        if value is None:
            self.headers.pop("Content-Type", None)
        else:
            self.headers["Content-Type"] = value

    @property
    def mimetype(self) -> str | None:
        """The mimetype (content type without charset etc.)

        .. versionadded:: 0.14
        """
        ct = self.content_type
        return ct.split(";")[0].strip() if ct else None

    @mimetype.setter
    def mimetype(self, value: str) -> None:
        self.content_type = get_content_type(value, "utf-8")

    @property
    def mimetype_params(self) -> t.Mapping[str, str]:
        """The mimetype parameters as dict.  For example if the
        content type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.14
        """

        def on_update(d: CallbackDict[str, str]) -> None:
            self.headers["Content-Type"] = dump_options_header(self.mimetype, d)

        d = parse_options_header(self.headers.get("content-type", ""))[1]
        return CallbackDict(d, on_update)

    @property
    def content_length(self) -> int | None:
        """The content length as integer.  Reflected from and to the
        :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        return self.headers.get("Content-Length", type=int)

    @content_length.setter
    def content_length(self, value: int | None) -> None:
        if value is None:
            self.headers.pop("Content-Length", None)
        else:
            self.headers["Content-Length"] = str(value)

    def _get_form(self, name: str, storage: type[_TAnyMultiDict]) -> _TAnyMultiDict:
        """Common behavior for getting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param storage: Storage class used for the data.
        """
        if self.input_stream is not None:
            raise AttributeError("an input stream is defined")

        rv = getattr(self, name)

        if rv is None:
            rv = storage()
            setattr(self, name, rv)

        return rv  # type: ignore

    def _set_form(self, name: str, value: MultiDict[str, t.Any]) -> None:
        """Common behavior for setting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param value: Value to assign to the attribute.
        """
        self._input_stream = None
        setattr(self, name, value)

    @property
    def form(self) -> MultiDict[str, str]:
        """A :class:`MultiDict` of form values."""
        return self._get_form("_form", MultiDict)

    @form.setter
    def form(self, value: MultiDict[str, str]) -> None:
        self._set_form("_form", value)

    @property
    def files(self) -> FileMultiDict:
        """A :class:`FileMultiDict` of uploaded files. Use
        :meth:`~FileMultiDict.add_file` to add new files.
        """
        return self._get_form("_files", FileMultiDict)

    @files.setter
    def files(self, value: FileMultiDict) -> None:
        self._set_form("_files", value)

    @property
    def input_stream(self) -> t.IO[bytes] | None:
        """An optional input stream. This is mutually exclusive with
        setting :attr:`form` and :attr:`files`, setting it will clear
        those. Do not provide this if the method is not ``POST`` or
        another method that has a body.
        """
        return self._input_stream

    @input_stream.setter
    def input_stream(self, value: t.IO[bytes] | None) -> None:
        self._input_stream = value
        self._form = None
        self._files = None

    @property
    def query_string(self) -> str:
        """The query string.  If you set this to a string
        :attr:`args` will no longer be available.
        """
        if self._query_string is None:
            if self._args is not None:
                return _urlencode(self._args)
            return ""
        return self._query_string

    @query_string.setter
    def query_string(self, value: str | None) -> None:
        self._query_string = value
        self._args = None

    @property
    def args(self) -> MultiDict[str, str]:
        """The URL arguments as :class:`MultiDict`."""
        if self._query_string is not None:
            raise AttributeError("a query string is defined")
        if self._args is None:
            self._args = MultiDict()
        return self._args

    @args.setter
    def args(self, value: MultiDict[str, str] | None) -> None:
        self._query_string = None
        self._args = value

    @property
    def server_name(self) -> str:
        """The server name (read-only, use :attr:`host` to set)"""
        return self.host.split(":", 1)[0]

    @property
    def server_port(self) -> int:
        """The server port as integer (read-only, use :attr:`host` to set)"""
        pieces = self.host.split(":", 1)

        if len(pieces) == 2:
            try:
                return int(pieces[1])
            except ValueError:
                pass

        if self.url_scheme == "https":
            return 443
        return 80

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Closes all files.  If you put real :class:`file` objects into the
        :attr:`files` dict you can call this method to automatically close
        them all in one go.
        """
        if self.closed:
            return
        try:
            files = self.files.values()
        except AttributeError:
            files = ()
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        self.closed = True

    def get_environ(self) -> WSGIEnvironment:
        """Return the built environ.

        .. versionchanged:: 0.15
            The content type and length headers are set based on
            input stream detection. Previously this only set the WSGI
            keys.
        """
        input_stream = self.input_stream
        content_length = self.content_length

        mimetype = self.mimetype
        content_type = self.content_type

        if input_stream is not None:
            start_pos = input_stream.tell()
            input_stream.seek(0, 2)
            end_pos = input_stream.tell()
            input_stream.seek(start_pos)
            content_length = end_pos - start_pos
        elif mimetype == "multipart/form-data":
            input_stream, content_length, boundary = stream_encode_multipart(
                CombinedMultiDict([self.form, self.files])
            )
            content_type = f'{mimetype}; boundary="{boundary}"'
        elif mimetype == "application/x-www-form-urlencoded":
            form_encoded = _urlencode(self.form).encode("ascii")
            content_length = len(form_encoded)
            input_stream = BytesIO(form_encoded)
        else:
            input_stream = BytesIO()

        result: WSGIEnvironment = {}
        if self.environ_base:
            result.update(self.environ_base)

        def _path_encode(x: str) -> str:
            return _wsgi_encoding_dance(unquote(x))

        raw_uri = _wsgi_encoding_dance(self.request_uri)
        result.update(
            {
                "REQUEST_METHOD": self.method,
                "SCRIPT_NAME": _path_encode(self.script_root),
                "PATH_INFO": _path_encode(self.path),
                "QUERY_STRING": _wsgi_encoding_dance(self.query_string),
                # Non-standard, added by mod_wsgi, uWSGI
                "REQUEST_URI": raw_uri,
                # Non-standard, added by gunicorn
                "RAW_URI": raw_uri,
                "SERVER_NAME": self.server_name,
                "SERVER_PORT": str(self.server_port),
                "HTTP_HOST": self.host,
                "SERVER_PROTOCOL": self.server_protocol,
                "wsgi.version": self.wsgi_version,
                "wsgi.url_scheme": self.url_scheme,
                "wsgi.input": input_stream,
                "wsgi.errors": self.errors_stream,
                "wsgi.multithread": self.multithread,
                "wsgi.multiprocess": self.multiprocess,
                "wsgi.run_once": self.run_once,
            }
        )

        headers = self.headers.copy()
        # Don't send these as headers, they're part of the environ.
        headers.remove("Content-Type")
        headers.remove("Content-Length")

        if content_type is not None:
            result["CONTENT_TYPE"] = content_type

        if content_length is not None:
            result["CONTENT_LENGTH"] = str(content_length)

        combined_headers = defaultdict(list)

        for key, value in headers.to_wsgi_list():
            combined_headers[f"HTTP_{key.upper().replace('-', '_')}"].append(value)

        for key, values in combined_headers.items():
            result[key] = ", ".join(values)

        if self.environ_overrides:
            result.update(self.environ_overrides)

        return result

    def get_request(self, cls: type[Request] | None = None) -> Request:
        """Returns a request with the data.  If the request class is not
        specified :attr:`request_class` is used.

        :param cls: The request wrapper to use.
        """
        if cls is None:
            cls = self.request_class

        return cls(self.get_environ())


class ClientRedirectError(Exception):
    """If a redirect loop is detected when using follow_redirects=True with
    the :cls:`Client`, then this exception is raised.
    """


class Client:
    """Simulate sending requests to a WSGI application without running a WSGI or HTTP
    server.

    :param application: The WSGI application to make requests to.
    :param response_wrapper: A :class:`.Response` class to wrap response data with.
        Defaults to :class:`.TestResponse`. If it's not a subclass of ``TestResponse``,
        one will be created.
    :param use_cookies: Persist cookies from ``Set-Cookie`` response headers to the
        ``Cookie`` header in subsequent requests. Domain and path matching is supported,
        but other cookie parameters are ignored.
    :param allow_subdomain_redirects: Allow requests to follow redirects to subdomains.
        Enable this if the application handles subdomains and redirects between them.

    .. versionchanged:: 2.3
        Simplify cookie implementation, support domain and path matching.

    .. versionchanged:: 2.1
        All data is available as properties on the returned response object. The
        response cannot be returned as a tuple.

    .. versionchanged:: 2.0
        ``response_wrapper`` is always a subclass of :class:``TestResponse``.

    .. versionchanged:: 0.5
        Added the ``use_cookies`` parameter.
    """

    def __init__(
        self,
        application: WSGIApplication,
        response_wrapper: type[Response] | None = None,
        use_cookies: bool = True,
        allow_subdomain_redirects: bool = False,
    ) -> None:
        self.application = application

        if response_wrapper in {None, Response}:
            response_wrapper = TestResponse
        elif response_wrapper is not None and not issubclass(
            response_wrapper, TestResponse
        ):
            response_wrapper = type(
                "WrapperTestResponse",
                (TestResponse, response_wrapper),
                {},
            )

        self.response_wrapper = t.cast(type["TestResponse"], response_wrapper)

        if use_cookies:
            self._cookies: dict[tuple[str, str, str], Cookie] | None = {}
        else:
            self._cookies = None

        self.allow_subdomain_redirects = allow_subdomain_redirects

    def get_cookie(
        self, key: str, domain: str = "localhost", path: str = "/"
    ) -> Cookie | None:
        """Return a :class:`.Cookie` if it exists. Cookies are uniquely identified by
        ``(domain, path, key)``.

        :param key: The decoded form of the key for the cookie.
        :param domain: The domain the cookie was set for.
        :param path: The path the cookie was set for.

        .. versionadded:: 2.3
        """
        if self._cookies is None:
            raise TypeError(
                "Cookies are disabled. Create a client with 'use_cookies=True'."
            )

        return self._cookies.get((domain, path, key))

    def set_cookie(
        self,
        key: str,
        value: str = "",
        *,
        domain: str = "localhost",
        origin_only: bool = True,
        path: str = "/",
        **kwargs: t.Any,
    ) -> None:
        """Set a cookie to be sent in subsequent requests.

        This is a convenience to skip making a test request to a route that would set
        the cookie. To test the cookie, make a test request to a route that uses the
        cookie value.

        The client uses ``domain``, ``origin_only``, and ``path`` to determine which
        cookies to send with a request. It does not use other cookie parameters that
        browsers use, since they're not applicable in tests.

        :param key: The key part of the cookie.
        :param value: The value part of the cookie.
        :param domain: Send this cookie with requests that match this domain. If
            ``origin_only`` is true, it must be an exact match, otherwise it may be a
            suffix match.
        :param origin_only: Whether the domain must be an exact match to the request.
        :param path: Send this cookie with requests that match this path either exactly
            or as a prefix.
        :param kwargs: Passed to :func:`.dump_cookie`.

        .. versionchanged:: 3.0
            The parameter ``server_name`` is removed. The first parameter is
            ``key``. Use the ``domain`` and ``origin_only`` parameters instead.

        .. versionchanged:: 2.3
            The ``origin_only`` parameter was added.

        .. versionchanged:: 2.3
            The ``domain`` parameter defaults to ``localhost``.
        """
        if self._cookies is None:
            raise TypeError(
                "Cookies are disabled. Create a client with 'use_cookies=True'."
            )

        cookie = Cookie._from_response_header(
            domain, "/", dump_cookie(key, value, domain=domain, path=path, **kwargs)
        )
        cookie.origin_only = origin_only

        if cookie._should_delete:
            self._cookies.pop(cookie._storage_key, None)
        else:
            self._cookies[cookie._storage_key] = cookie

    def delete_cookie(
        self,
        key: str,
        *,
        domain: str = "localhost",
        path: str = "/",
    ) -> None:
        """Delete a cookie if it exists. Cookies are uniquely identified by
        ``(domain, path, key)``.

        :param key: The decoded form of the key for the cookie.
        :param domain: The domain the cookie was set for.
        :param path: The path the cookie was set for.

        .. versionchanged:: 3.0
            The ``server_name`` parameter is removed. The first parameter is
            ``key``. Use the ``domain`` parameter instead.

        .. versionchanged:: 3.0
            The ``secure``, ``httponly`` and ``samesite`` parameters are removed.

        .. versionchanged:: 2.3
            The ``domain`` parameter defaults to ``localhost``.
        """
        if self._cookies is None:
            raise TypeError(
                "Cookies are disabled. Create a client with 'use_cookies=True'."
            )

        self._cookies.pop((domain, path, key), None)

    def _add_cookies_to_wsgi(self, environ: WSGIEnvironment) -> None:
        """If cookies are enabled, set the ``Cookie`` header in the environ to the
        cookies that are applicable to the request host and path.

        :meta private:

        .. versionadded:: 2.3
        """
        if self._cookies is None:
            return

        url = urlsplit(get_current_url(environ))
        server_name = url.hostname or "localhost"
        value = "; ".join(
            c._to_request_header()
            for c in self._cookies.values()
            if c._matches_request(server_name, url.path)
        )

        if value:
            environ["HTTP_COOKIE"] = value
        else:
            environ.pop("HTTP_COOKIE", None)

    def _update_cookies_from_response(
        self, server_name: str, path: str, headers: list[str]
    ) -> None:
        """If cookies are enabled, update the stored cookies from any ``Set-Cookie``
        headers in the response.

        :meta private:

        .. versionadded:: 2.3
        """
        if self._cookies is None:
            return

        for header in headers:
            cookie = Cookie._from_response_header(server_name, path, header)

            if cookie._should_delete:
                self._cookies.pop(cookie._storage_key, None)
            else:
                self._cookies[cookie._storage_key] = cookie

    def run_wsgi_app(
        self, environ: WSGIEnvironment, buffered: bool = False
    ) -> tuple[t.Iterable[bytes], str, Headers]:
        """Runs the wrapped WSGI app with the given environment.

        :meta private:
        """
        self._add_cookies_to_wsgi(environ)
        rv = run_wsgi_app(self.application, environ, buffered=buffered)
        url = urlsplit(get_current_url(environ))
        self._update_cookies_from_response(
            url.hostname or "localhost", url.path, rv[2].getlist("Set-Cookie")
        )
        return rv

    def resolve_redirect(
        self, response: TestResponse, buffered: bool = False
    ) -> TestResponse:
        """Perform a new request to the location given by the redirect
        response to the previous request.

        :meta private:
        """
        scheme, netloc, path, qs, anchor = urlsplit(response.location)
        builder = EnvironBuilder.from_environ(
            response.request.environ, path=path, query_string=qs
        )

        to_name_parts = netloc.split(":", 1)[0].split(".")
        from_name_parts = builder.server_name.split(".")

        if to_name_parts != [""]:
            # The new location has a host, use it for the base URL.
            builder.url_scheme = scheme
            builder.host = netloc
        else:
            # A local redirect with autocorrect_location_header=False
            # doesn't have a host, so use the request's host.
            to_name_parts = from_name_parts

        # Explain why a redirect to a different server name won't be followed.
        if to_name_parts != from_name_parts:
            if to_name_parts[-len(from_name_parts) :] == from_name_parts:
                if not self.allow_subdomain_redirects:
                    raise RuntimeError("Following subdomain redirects is not enabled.")
            else:
                raise RuntimeError("Following external redirects is not supported.")

        path_parts = path.split("/")
        root_parts = builder.script_root.split("/")

        if path_parts[: len(root_parts)] == root_parts:
            # Strip the script root from the path.
            builder.path = path[len(builder.script_root) :]
        else:
            # The new location is not under the script root, so use the
            # whole path and clear the previous root.
            builder.path = path
            builder.script_root = ""

        # Only 307 and 308 preserve all of the original request.
        if response.status_code not in {307, 308}:
            # HEAD is preserved, everything else becomes GET.
            if builder.method != "HEAD":
                builder.method = "GET"

            # Clear the body and the headers that describe it.

            if builder.input_stream is not None:
                builder.input_stream.close()
                builder.input_stream = None

            builder.content_type = None
            builder.content_length = None
            builder.headers.pop("Transfer-Encoding", None)

        return self.open(builder, buffered=buffered)

    def open(
        self,
        *args: t.Any,
        buffered: bool = False,
        follow_redirects: bool = False,
        **kwargs: t.Any,
    ) -> TestResponse:
        """Generate an environ dict from the given arguments, make a
        request to the application using it, and return the response.

        :param args: Passed to :class:`EnvironBuilder` to create the
            environ for the request. If a single arg is passed, it can
            be an existing :class:`EnvironBuilder` or an environ dict.
        :param buffered: Convert the iterator returned by the app into
            a list. If the iterator has a ``close()`` method, it is
            called automatically.
        :param follow_redirects: Make additional requests to follow HTTP
            redirects until a non-redirect status is returned.
            :attr:`TestResponse.history` lists the intermediate
            responses.

        .. versionchanged:: 2.1
            Removed the ``as_tuple`` parameter.

        .. versionchanged:: 2.0
            The request input stream is closed when calling
            ``response.close()``. Input streams for redirects are
            automatically closed.

        .. versionchanged:: 0.5
            If a dict is provided as file in the dict for the ``data``
            parameter the content type has to be called ``content_type``
            instead of ``mimetype``. This change was made for
            consistency with :class:`werkzeug.FileWrapper`.

        .. versionchanged:: 0.5
            Added the ``follow_redirects`` parameter.
        """
        request: Request | None = None

        if not kwargs and len(args) == 1:
            arg = args[0]

            if isinstance(arg, EnvironBuilder):
                request = arg.get_request()
            elif isinstance(arg, dict):
                request = EnvironBuilder.from_environ(arg).get_request()
            elif isinstance(arg, Request):
                request = arg

        if request is None:
            builder = EnvironBuilder(*args, **kwargs)

            try:
                request = builder.get_request()
            finally:
                builder.close()

        response_parts = self.run_wsgi_app(request.environ, buffered=buffered)
        response = self.response_wrapper(*response_parts, request=request)

        redirects = set()
        history: list[TestResponse] = []

        if not follow_redirects:
            return response

        while response.status_code in {
            301,
            302,
            303,
            305,
            307,
            308,
        }:
            # Exhaust intermediate response bodies to ensure middleware
            # that returns an iterator runs any cleanup code.
            if not buffered:
                response.make_sequence()
                response.close()

            new_redirect_entry = (response.location, response.status_code)

            if new_redirect_entry in redirects:
                raise ClientRedirectError(
                    f"Loop detected: A {response.status_code} redirect"
                    f" to {response.location} was already made."
                )

            redirects.add(new_redirect_entry)
            response.history = tuple(history)
            history.append(response)
            response = self.resolve_redirect(response, buffered=buffered)
        else:
            # This is the final request after redirects.
            response.history = tuple(history)
            # Close the input stream when closing the response, in case
            # the input is an open temporary file.
            response.call_on_close(request.input_stream.close)
            return response

    def get(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``GET``."""
        kw["method"] = "GET"
        return self.open(*args, **kw)

    def post(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``POST``."""
        kw["method"] = "POST"
        return self.open(*args, **kw)

    def put(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``PUT``."""
        kw["method"] = "PUT"
        return self.open(*args, **kw)

    def delete(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``DELETE``."""
        kw["method"] = "DELETE"
        return self.open(*args, **kw)

    def patch(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``PATCH``."""
        kw["method"] = "PATCH"
        return self.open(*args, **kw)

    def options(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``OPTIONS``."""
        kw["method"] = "OPTIONS"
        return self.open(*args, **kw)

    def head(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``HEAD``."""
        kw["method"] = "HEAD"
        return self.open(*args, **kw)

    def trace(self, *args: t.Any, **kw: t.Any) -> TestResponse:
        """Call :meth:`open` with ``method`` set to ``TRACE``."""
        kw["method"] = "TRACE"
        return self.open(*args, **kw)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.application!r}>"


def create_environ(*args: t.Any, **kwargs: t.Any) -> WSGIEnvironment:
    """Create a new WSGI environ dict based on the values passed.  The first
    parameter should be the path of the request which defaults to '/'.  The
    second one can either be an absolute path (in that case the host is
    localhost:80) or a full path to the request with scheme, netloc port and
    the path to the script.

    This accepts the same arguments as the :class:`EnvironBuilder`
    constructor.

    .. versionchanged:: 0.5
       This function is now a thin wrapper over :class:`EnvironBuilder` which
       was added in 0.5.  The `headers`, `environ_base`, `environ_overrides`
       and `charset` parameters were added.
    """
    builder = EnvironBuilder(*args, **kwargs)

    try:
        return builder.get_environ()
    finally:
        builder.close()


def run_wsgi_app(
    app: WSGIApplication, environ: WSGIEnvironment, buffered: bool = False
) -> tuple[t.Iterable[bytes], str, Headers]:
    """Return a tuple in the form (app_iter, status, headers) of the
    application output.  This works best if you pass it an application that
    returns an iterator all the time.

    Sometimes applications may use the `write()` callable returned
    by the `start_response` function.  This tries to resolve such edge
    cases automatically.  But if you don't get the expected output you
    should set `buffered` to `True` which enforces buffering.

    If passed an invalid WSGI application the behavior of this function is
    undefined.  Never pass non-conforming WSGI applications to this function.

    :param app: the application to execute.
    :param buffered: set to `True` to enforce buffering.
    :return: tuple in the form ``(app_iter, status, headers)``
    """
    # Copy environ to ensure any mutations by the app (ProxyFix, for
    # example) don't affect subsequent requests (such as redirects).
    environ = _get_environ(environ).copy()
    status: str
    response: tuple[str, list[tuple[str, str]]] | None = None
    buffer: list[bytes] = []

    def start_response(status, headers, exc_info=None):  # type: ignore
        nonlocal response

        if exc_info:
            try:
                raise exc_info[1].with_traceback(exc_info[2])
            finally:
                exc_info = None

        response = (status, headers)
        return buffer.append

    app_rv = app(environ, start_response)
    close_func = getattr(app_rv, "close", None)
    app_iter: t.Iterable[bytes] = iter(app_rv)

    # when buffering we emit the close call early and convert the
    # application iterator into a regular list
    if buffered:
        try:
            app_iter = list(app_iter)
        finally:
            if close_func is not None:
                close_func()

    # otherwise we iterate the application iter until we have a response, chain
    # the already received data with the already collected data and wrap it in
    # a new `ClosingIterator` if we need to restore a `close` callable from the
    # original return value.
    else:
        for item in app_iter:
            buffer.append(item)

            if response is not None:
                break

        if buffer:
            app_iter = chain(buffer, app_iter)

        if close_func is not None and app_iter is not app_rv:
            app_iter = ClosingIterator(app_iter, close_func)

    status, headers = response  # type: ignore
    return app_iter, status, Headers(headers)


class TestResponse(Response):
    """:class:`~werkzeug.wrappers.Response` subclass that provides extra
    information about requests made with the test :class:`Client`.

    Test client requests will always return an instance of this class.
    If a custom response class is passed to the client, it is
    subclassed along with this to support test information.

    If the test request included large files, or if the application is
    serving a file, call :meth:`close` to close any open files and
    prevent Python showing a ``ResourceWarning``.

    .. versionchanged:: 2.2
        Set the ``default_mimetype`` to None to prevent a mimetype being
        assumed if missing.

    .. versionchanged:: 2.1
        Response instances cannot be treated as tuples.

    .. versionadded:: 2.0
        Test client methods always return instances of this class.
    """

    default_mimetype = None
    # Don't assume a mimetype, instead use whatever the response provides

    request: Request
    """A request object with the environ used to make the request that
    resulted in this response.
    """

    history: tuple[TestResponse, ...]
    """A list of intermediate responses. Populated when the test request
    is made with ``follow_redirects`` enabled.
    """

    # Tell Pytest to ignore this, it's not a test class.
    __test__ = False

    def __init__(
        self,
        response: t.Iterable[bytes],
        status: str,
        headers: Headers,
        request: Request,
        history: tuple[TestResponse] = (),  # type: ignore
        **kwargs: t.Any,
    ) -> None:
        super().__init__(response, status, headers, **kwargs)
        self.request = request
        self.history = history
        self._compat_tuple = response, status, headers

    @cached_property
    def text(self) -> str:
        """The response data as text. A shortcut for
        ``response.get_data(as_text=True)``.

        .. versionadded:: 2.1
        """
        return self.get_data(as_text=True)


@dataclasses.dataclass
class Cookie:
    """A cookie key, value, and parameters.

    The class itself is not a public API. Its attributes are documented for inspection
    with :meth:`.Client.get_cookie` only.

    .. versionadded:: 2.3
    """

    key: str
    """The cookie key, encoded as a client would see it."""

    value: str
    """The cookie key, encoded as a client would see it."""

    decoded_key: str
    """The cookie key, decoded as the application would set and see it."""

    decoded_value: str
    """The cookie value, decoded as the application would set and see it."""

    expires: datetime | None
    """The time at which the cookie is no longer valid."""

    max_age: int | None
    """The number of seconds from when the cookie was set at which it is
    no longer valid.
    """

    domain: str
    """The domain that the cookie was set for, or the request domain if not set."""

    origin_only: bool
    """Whether the cookie will be sent for exact domain matches only. This is ``True``
    if the ``Domain`` parameter was not present.
    """

    path: str
    """The path that the cookie was set for."""

    secure: bool | None
    """The ``Secure`` parameter."""

    http_only: bool | None
    """The ``HttpOnly`` parameter."""

    same_site: str | None
    """The ``SameSite`` parameter."""

    def _matches_request(self, server_name: str, path: str) -> bool:
        return (
            server_name == self.domain
            or (
                not self.origin_only
                and server_name.endswith(self.domain)
                and server_name[: -len(self.domain)].endswith(".")
            )
        ) and (
            path == self.path
            or (
                path.startswith(self.path)
                and path[len(self.path) - self.path.endswith("/") :].startswith("/")
            )
        )

    def _to_request_header(self) -> str:
        return f"{self.key}={self.value}"

    @classmethod
    def _from_response_header(cls, server_name: str, path: str, header: str) -> te.Self:
        header, _, parameters_str = header.partition(";")
        key, _, value = header.partition("=")
        decoded_key, decoded_value = next(parse_cookie(header).items())  # type: ignore[call-overload]
        params = {}

        for item in parameters_str.split(";"):
            k, sep, v = item.partition("=")
            params[k.strip().lower()] = v.strip() if sep else None

        return cls(
            key=key.strip(),
            value=value.strip(),
            decoded_key=decoded_key,
            decoded_value=decoded_value,
            expires=parse_date(params.get("expires")),
            max_age=int(params["max-age"] or 0) if "max-age" in params else None,
            domain=params.get("domain") or server_name,
            origin_only="domain" not in params,
            path=params.get("path") or path.rpartition("/")[0] or "/",
            secure="secure" in params,
            http_only="httponly" in params,
            same_site=params.get("samesite"),
        )

    @property
    def _storage_key(self) -> tuple[str, str, str]:
        return self.domain, self.path, self.decoded_key

    @property
    def _should_delete(self) -> bool:
        return self.max_age == 0 or (
            self.expires is not None and self.expires.timestamp() == 0
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\parsers\base_parser.py ===
from __future__ import annotations

from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    parsers,
)
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    ParserError,
    ParserWarning,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.missing import isna

from pandas import (
    ArrowDtype,
    DataFrame,
    DatetimeIndex,
    StringDtype,
    concat,
)
from pandas.core import algorithms
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    BooleanArray,
    Categorical,
    ExtensionArray,
    FloatingArray,
    IntegerArray,
)
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
    ensure_index_from_sequences,
)
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools

from pandas.io.common import is_potential_multi_index

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeArg,
        DtypeObj,
        Scalar,
    )


class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2

    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    keep_date_col: bool
    usecols_dtype: str | None

    def __init__(self, kwds) -> None:
        self._implicit_index = False

        self.names = kwds.get("names")
        self.orig_names: Sequence[Hashable] | None = None

        self.index_col = kwds.get("index_col", None)
        self.unnamed_cols: set = set()
        self.index_names: Sequence[Hashable] | None = None
        self.col_names: Sequence[Hashable] | None = None

        self.parse_dates = _validate_parse_dates_arg(kwds.pop("parse_dates", False))
        self._parse_date_cols: Iterable = []
        self.date_parser = kwds.pop("date_parser", lib.no_default)
        self.date_format = kwds.pop("date_format", None)
        self.dayfirst = kwds.pop("dayfirst", False)
        self.keep_date_col = kwds.pop("keep_date_col", False)

        self.na_values = kwds.get("na_values")
        self.na_fvalues = kwds.get("na_fvalues")
        self.na_filter = kwds.get("na_filter", False)
        self.keep_default_na = kwds.get("keep_default_na", True)

        self.dtype = copy(kwds.get("dtype", None))
        self.converters = kwds.get("converters")
        self.dtype_backend = kwds.get("dtype_backend")

        self.true_values = kwds.get("true_values")
        self.false_values = kwds.get("false_values")
        self.cache_dates = kwds.pop("cache_dates", True)

        self._date_conv = _make_date_converter(
            date_parser=self.date_parser,
            date_format=self.date_format,
            dayfirst=self.dayfirst,
            cache_dates=self.cache_dates,
        )

        # validate header options for mi
        self.header = kwds.get("header")
        if is_list_like(self.header, allow_sets=False):
            if kwds.get("usecols"):
                raise ValueError(
                    "cannot specify usecols when specifying a multi-index header"
                )
            if kwds.get("names"):
                raise ValueError(
                    "cannot specify names when specifying a multi-index header"
                )

            # validate index_col that only contains integers
            if self.index_col is not None:
                # In this case we can pin down index_col as list[int]
                if is_integer(self.index_col):
                    self.index_col = [self.index_col]
                elif not (
                    is_list_like(self.index_col, allow_sets=False)
                    and all(map(is_integer, self.index_col))
                ):
                    raise ValueError(
                        "index_col must only contain row numbers "
                        "when specifying a multi-index header"
                    )
                else:
                    self.index_col = list(self.index_col)

        self._name_processed = False

        self._first_chunk = True

        self.usecols, self.usecols_dtype = self._validate_usecols_arg(kwds["usecols"])

        # Fallback to error to pass a sketchy test(test_override_set_noconvert_columns)
        # Normally, this arg would get pre-processed earlier on
        self.on_bad_lines = kwds.get("on_bad_lines", self.BadLineHandleMethod.ERROR)

    def _validate_parse_dates_presence(self, columns: Sequence[Hashable]) -> Iterable:
        """
        Check if parse_dates are in columns.

        If user has provided names for parse_dates, check if those columns
        are available.

        Parameters
        ----------
        columns : list
            List of names of the dataframe.

        Returns
        -------
        The names of the columns which will get parsed later if a dict or list
        is given as specification.

        Raises
        ------
        ValueError
            If column to parse_date is not in dataframe.

        """
        cols_needed: Iterable
        if is_dict_like(self.parse_dates):
            cols_needed = itertools.chain(*self.parse_dates.values())
        elif is_list_like(self.parse_dates):
            # a column in parse_dates could be represented
            # ColReference = Union[int, str]
            # DateGroups = List[ColReference]
            # ParseDates = Union[DateGroups, List[DateGroups],
            #     Dict[ColReference, DateGroups]]
            cols_needed = itertools.chain.from_iterable(
                col if is_list_like(col) and not isinstance(col, tuple) else [col]
                for col in self.parse_dates
            )
        else:
            cols_needed = []

        cols_needed = list(cols_needed)

        # get only columns that are references using names (str), not by index
        missing_cols = ", ".join(
            sorted(
                {
                    col
                    for col in cols_needed
                    if isinstance(col, str) and col not in columns
                }
            )
        )
        if missing_cols:
            raise ValueError(
                f"Missing column provided to 'parse_dates': '{missing_cols}'"
            )
        # Convert positions to actual column names
        return [
            col if (isinstance(col, str) or col in columns) else columns[col]
            for col in cols_needed
        ]

    def close(self) -> None:
        pass

    @final
    @property
    def _has_complex_date_col(self) -> bool:
        return isinstance(self.parse_dates, dict) or (
            isinstance(self.parse_dates, list)
            and len(self.parse_dates) > 0
            and isinstance(self.parse_dates[0], list)
        )

    @final
    def _should_parse_dates(self, i: int) -> bool:
        if lib.is_bool(self.parse_dates):
            return bool(self.parse_dates)
        else:
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]

            return (j in self.parse_dates) or (
                name is not None and name in self.parse_dates
            )

    @final
    def _extract_multi_indexer_columns(
        self,
        header,
        index_names: Sequence[Hashable] | None,
        passed_names: bool = False,
    ) -> tuple[
        Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool
    ]:
        """
        Extract and return the names, index_names, col_names if the column
        names are a MultiIndex.

        Parameters
        ----------
        header: list of lists
            The header rows
        index_names: list, optional
            The names of the future index
        passed_names: bool, default False
            A flag specifying if names where passed

        """
        if len(header) < 2:
            return header[0], index_names, None, passed_names

        # the names are the tuples of the header that are not the index cols
        # 0 is the name of the index, assuming index_col is a list of column
        # numbers
        ic = self.index_col
        if ic is None:
            ic = []

        if not isinstance(ic, (list, tuple, np.ndarray)):
            ic = [ic]
        sic = set(ic)

        # clean the index_names
        index_names = header.pop(-1)
        index_names, _, _ = self._clean_index_names(index_names, self.index_col)

        # extract the columns
        field_count = len(header[0])

        # check if header lengths are equal
        if not all(len(header_iter) == field_count for header_iter in header[1:]):
            raise ParserError("Header rows must have an equal number of columns.")

        def extract(r):
            return tuple(r[i] for i in range(field_count) if i not in sic)

        columns = list(zip(*(extract(r) for r in header)))
        names = columns.copy()
        for single_ic in sorted(ic):
            names.insert(single_ic, single_ic)

        # Clean the column names (if we have an index_col).
        if len(ic):
            col_names = [
                r[ic[0]]
                if ((r[ic[0]] is not None) and r[ic[0]] not in self.unnamed_cols)
                else None
                for r in header
            ]
        else:
            col_names = [None] * len(header)

        passed_names = True

        return names, index_names, col_names, passed_names

    @final
    def _maybe_make_multi_index_columns(
        self,
        columns: Sequence[Hashable],
        col_names: Sequence[Hashable] | None = None,
    ) -> Sequence[Hashable] | MultiIndex:
        # possibly create a column mi here
        if is_potential_multi_index(columns):
            list_columns = cast(list[tuple], columns)
            return MultiIndex.from_tuples(list_columns, names=col_names)
        return columns

    @final
    def _make_index(
        self, data, alldata, columns, indexnamerow: list[Scalar] | None = None
    ) -> tuple[Index | None, Sequence[Hashable] | MultiIndex]:
        index: Index | None
        if not is_index_col(self.index_col) or not self.index_col:
            index = None

        elif not self._has_complex_date_col:
            simple_index = self._get_simple_index(alldata, columns)
            index = self._agg_index(simple_index)
        elif self._has_complex_date_col:
            if not self._name_processed:
                (self.index_names, _, self.index_col) = self._clean_index_names(
                    list(columns), self.index_col
                )
                self._name_processed = True
            date_index = self._get_complex_date_index(data, columns)
            index = self._agg_index(date_index, try_parse_dates=False)

        # add names for the index
        if indexnamerow:
            coffset = len(indexnamerow) - len(columns)
            assert index is not None
            index = index.set_names(indexnamerow[:coffset])

        # maybe create a mi on the columns
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)

        return index, columns

    @final
    def _get_simple_index(self, data, columns):
        def ix(col):
            if not isinstance(col, str):
                return col
            raise ValueError(f"Index {col} invalid")

        to_remove = []
        index = []
        for idx in self.index_col:
            i = ix(idx)
            to_remove.append(i)
            index.append(data[i])

        # remove index items from content and columns, don't pop in
        # loop
        for i in sorted(to_remove, reverse=True):
            data.pop(i)
            if not self._implicit_index:
                columns.pop(i)

        return index

    @final
    def _get_complex_date_index(self, data, col_names):
        def _get_name(icol):
            if isinstance(icol, str):
                return icol

            if col_names is None:
                raise ValueError(f"Must supply column order to use {icol!s} as index")

            for i, c in enumerate(col_names):
                if i == icol:
                    return c

        to_remove = []
        index = []
        for idx in self.index_col:
            name = _get_name(idx)
            to_remove.append(name)
            index.append(data[name])

        # remove index items from content and columns, don't pop in
        # loop
        for c in sorted(to_remove, reverse=True):
            data.pop(c)
            col_names.remove(c)

        return index

    @final
    def _clean_mapping(self, mapping):
        """converts col numbers to names"""
        if not isinstance(mapping, dict):
            return mapping
        clean = {}
        # for mypy
        assert self.orig_names is not None

        for col, v in mapping.items():
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            clean[col] = v
        if isinstance(mapping, defaultdict):
            remaining_cols = set(self.orig_names) - set(clean.keys())
            clean.update({col: mapping[col] for col in remaining_cols})
        return clean

    @final
    def _agg_index(self, index, try_parse_dates: bool = True) -> Index:
        arrays = []
        converters = self._clean_mapping(self.converters)

        if self.index_names is not None:
            names: Iterable = self.index_names
        else:
            names = itertools.cycle([None])
        for i, (arr, name) in enumerate(zip(index, names)):
            if try_parse_dates and self._should_parse_dates(i):
                arr = self._date_conv(
                    arr,
                    col=self.index_names[i] if self.index_names is not None else None,
                )

            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()

            if isinstance(self.na_values, dict):
                assert self.index_names is not None
                col_name = self.index_names[i]
                if col_name is not None:
                    col_na_values, col_na_fvalues = _get_na_values(
                        col_name, self.na_values, self.na_fvalues, self.keep_default_na
                    )

            clean_dtypes = self._clean_mapping(self.dtype)

            cast_type = None
            index_converter = False
            if self.index_names is not None:
                if isinstance(clean_dtypes, dict):
                    cast_type = clean_dtypes.get(self.index_names[i], None)

                if isinstance(converters, dict):
                    index_converter = converters.get(self.index_names[i]) is not None

            try_num_bool = not (
                cast_type and is_string_dtype(cast_type) or index_converter
            )

            arr, _ = self._infer_types(
                arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool
            )
            if cast_type is not None:
                # Don't perform RangeIndex inference
                idx = Index(arr, name=name, dtype=cast_type)
            else:
                idx = ensure_index_from_sequences([arr], [name])
            arrays.append(idx)

        if len(arrays) == 1:
            return arrays[0]
        else:
            return MultiIndex.from_arrays(arrays)

    @final
    def _convert_to_ndarrays(
        self,
        dct: Mapping,
        na_values,
        na_fvalues,
        verbose: bool = False,
        converters=None,
        dtypes=None,
    ):
        result = {}
        for c, values in dct.items():
            conv_f = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                # single dtype or None
                cast_type = dtypes

            if self.na_filter:
                col_na_values, col_na_fvalues = _get_na_values(
                    c, na_values, na_fvalues, self.keep_default_na
                )
            else:
                col_na_values, col_na_fvalues = set(), set()

            if c in self._parse_date_cols:
                # GH#26203 Do not convert columns which get converted to dates
                # but replace nans to ensure to_datetime works
                mask = algorithms.isin(values, set(col_na_values) | col_na_fvalues)
                np.putmask(values, mask, np.nan)
                result[c] = values
                continue

            if conv_f is not None:
                # conv_f applied to data before inference
                if cast_type is not None:
                    warnings.warn(
                        (
                            "Both a converter and dtype were specified "
                            f"for column {c} - only the converter will be used."
                        ),
                        ParserWarning,
                        stacklevel=find_stack_level(),
                    )

                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)

                cvals, na_count = self._infer_types(
                    values,
                    set(col_na_values) | col_na_fvalues,
                    cast_type is None,
                    try_num_bool=False,
                )
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
                # skip inference if specified dtype is object
                # or casting to an EA
                try_num_bool = not (cast_type and is_str_or_ea_dtype)

                # general type inference and conversion
                cvals, na_count = self._infer_types(
                    values,
                    set(col_na_values) | col_na_fvalues,
                    cast_type is None,
                    try_num_bool,
                )

                # type specified in dtype param or cast_type is an EA
                if cast_type is not None:
                    cast_type = pandas_dtype(cast_type)
                if cast_type and (cvals.dtype != cast_type or is_ea):
                    if not is_ea and na_count > 0:
                        if is_bool_dtype(cast_type):
                            raise ValueError(f"Bool column has NA values in column {c}")
                    cvals = self._cast_types(cvals, cast_type, c)

            result[c] = cvals
            if verbose and na_count:
                print(f"Filled {na_count} NA values in column {c!s}")
        return result

    @final
    def _set_noconvert_dtype_columns(
        self, col_indices: list[int], names: Sequence[Hashable]
    ) -> set[int]:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
        usecols: list[int] | list[str] | None
        noconvert_columns = set()
        if self.usecols_dtype == "integer":
            # A set of integers will be converted to a list in
            # the correct order every single time.
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ("empty", None):
            # The names attribute should have the correct columns
            # in the proper order for indexing with parse_dates.
            usecols = col_indices
        else:
            # Usecols is empty.
            usecols = None

        def _set(x) -> int:
            if usecols is not None and is_integer(x):
                x = usecols[x]

            if not is_integer(x):
                x = col_indices[names.index(x)]

            return x

        if isinstance(self.parse_dates, list):
            for val in self.parse_dates:
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))

        elif isinstance(self.parse_dates, dict):
            for val in self.parse_dates.values():
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))

        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))

        return noconvert_columns

    @final
    def _infer_types(
        self, values, na_values, no_dtype_specified, try_num_bool: bool = True
    ) -> tuple[ArrayLike, int]:
        """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        no_dtype_specified: Specifies if we want to cast explicitly
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray or ExtensionArray
        na_count : int
        """
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            # If our array has numeric dtype, we don't have to check for strings in isin
            na_values = np.array([val for val in na_values if not isinstance(val, str)])
            mask = algorithms.isin(values, na_values)
            na_count = mask.astype("uint8", copy=False).sum()
            if na_count > 0:
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return values, na_count

        dtype_backend = self.dtype_backend
        non_default_dtype_backend = (
            no_dtype_specified and dtype_backend is not lib.no_default
        )
        result: ArrayLike

        if try_num_bool and is_object_dtype(values.dtype):
            # exclude e.g DatetimeIndex here
            try:
                result, result_mask = lib.maybe_convert_numeric(
                    values,
                    na_values,
                    False,
                    convert_to_masked_nullable=non_default_dtype_backend,  # type: ignore[arg-type]
                )
            except (ValueError, TypeError):
                # e.g. encountering datetime string gets ValueError
                #  TypeError can be raised in floatify
                na_count = parsers.sanitize_objects(values, na_values)
                result = values
            else:
                if non_default_dtype_backend:
                    if result_mask is None:
                        result_mask = np.zeros(result.shape, dtype=np.bool_)

                    if result_mask.all():
                        result = IntegerArray(
                            np.ones(result_mask.shape, dtype=np.int64), result_mask
                        )
                    elif is_integer_dtype(result):
                        result = IntegerArray(result, result_mask)
                    elif is_bool_dtype(result):
                        result = BooleanArray(result, result_mask)
                    elif is_float_dtype(result):
                        result = FloatingArray(result, result_mask)

                    na_count = result_mask.sum()
                else:
                    na_count = isna(result).sum()
        else:
            result = values
            if values.dtype == np.object_:
                na_count = parsers.sanitize_objects(values, na_values)

        if result.dtype == np.object_ and try_num_bool:
            result, bool_mask = libops.maybe_convert_bool(
                np.asarray(values),
                true_values=self.true_values,
                false_values=self.false_values,
                convert_to_masked_nullable=non_default_dtype_backend,  # type: ignore[arg-type]
            )
            if result.dtype == np.bool_ and non_default_dtype_backend:
                if bool_mask is None:
                    bool_mask = np.zeros(result.shape, dtype=np.bool_)
                result = BooleanArray(result, bool_mask)
            elif result.dtype == np.object_ and non_default_dtype_backend:
                # read_excel sends array of datetime objects
                if not lib.is_datetime_array(result, skipna=True):
                    dtype = StringDtype()
                    cls = dtype.construct_array_type()
                    result = cls._from_sequence(values, dtype=dtype)

        if dtype_backend == "pyarrow":
            pa = import_optional_dependency("pyarrow")
            if isinstance(result, np.ndarray):
                result = ArrowExtensionArray(pa.array(result, from_pandas=True))
            elif isinstance(result, BaseMaskedArray):
                if result._mask.all():
                    # We want an arrow null array here
                    result = ArrowExtensionArray(pa.array([None] * len(result)))
                else:
                    result = ArrowExtensionArray(
                        pa.array(result._data, mask=result._mask)
                    )
            else:
                result = ArrowExtensionArray(
                    pa.array(result.to_numpy(), from_pandas=True)
                )

        return result, na_count

    @final
    def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column) -> ArrayLike:
        """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray or ExtensionArray
        cast_type : np.dtype or ExtensionDtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray or ExtensionArray
        """
        if isinstance(cast_type, CategoricalDtype):
            known_cats = cast_type.categories is not None

            if not is_object_dtype(values.dtype) and not known_cats:
                # TODO: this is for consistency with
                # c-parser which parses all categories
                # as strings
                values = lib.ensure_string_array(
                    values, skipna=False, convert_na_value=False
                )

            cats = Index(values).unique().dropna()
            values = Categorical._from_inferred_categories(
                cats, cats.get_indexer(values), cast_type, true_values=self.true_values
            )

        # use the EA's implementation of casting
        elif isinstance(cast_type, ExtensionDtype):
            array_type = cast_type.construct_array_type()
            try:
                if isinstance(cast_type, BooleanDtype):
                    # error: Unexpected keyword argument "true_values" for
                    # "_from_sequence_of_strings" of "ExtensionArray"
                    return array_type._from_sequence_of_strings(  # type: ignore[call-arg]
                        values,
                        dtype=cast_type,
                        true_values=self.true_values,
                        false_values=self.false_values,
                    )
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Extension Array: {array_type} must implement "
                    "_from_sequence_of_strings in order to be used in parser methods"
                ) from err

        elif isinstance(values, ExtensionArray):
            values = values.astype(cast_type, copy=False)
        elif issubclass(cast_type.type, str):
            # TODO: why skipna=True here and False above? some tests depend
            #  on it here, but nothing fails if we change it above
            #  (as no tests get there as of 2022-12-06)
            values = lib.ensure_string_array(
                values, skipna=True, convert_na_value=False
            )
        else:
            try:
                values = astype_array(values, cast_type, copy=True)
            except ValueError as err:
                raise ValueError(
                    f"Unable to convert column {column} to type {cast_type}"
                ) from err
        return values

    @overload
    def _do_date_conversions(
        self,
        names: Index,
        data: DataFrame,
    ) -> tuple[Sequence[Hashable] | Index, DataFrame]:
        ...

    @overload
    def _do_date_conversions(
        self,
        names: Sequence[Hashable],
        data: Mapping[Hashable, ArrayLike],
    ) -> tuple[Sequence[Hashable], Mapping[Hashable, ArrayLike]]:
        ...

    @final
    def _do_date_conversions(
        self,
        names: Sequence[Hashable] | Index,
        data: Mapping[Hashable, ArrayLike] | DataFrame,
    ) -> tuple[Sequence[Hashable] | Index, Mapping[Hashable, ArrayLike] | DataFrame]:
        # returns data, columns

        if self.parse_dates is not None:
            data, names = _process_date_conversion(
                data,
                self._date_conv,
                self.parse_dates,
                self.index_col,
                self.index_names,
                names,
                keep_date_col=self.keep_date_col,
                dtype_backend=self.dtype_backend,
            )

        return names, data

    @final
    def _check_data_length(
        self,
        columns: Sequence[Hashable],
        data: Sequence[ArrayLike],
    ) -> None:
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ""
            # error: No overload variant of "__ror__" of "ndarray" matches
            # argument type "ExtensionArray"
            empty_str_or_na = empty_str | isna(data[-1])  # type: ignore[operator]
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn(
                "Length of header or names does not match length of data. This leads "
                "to a loss of data with index_col=False.",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    @overload
    def _evaluate_usecols(
        self,
        usecols: set[int] | Callable[[Hashable], object],
        names: Sequence[Hashable],
    ) -> set[int]:
        ...

    @overload
    def _evaluate_usecols(
        self, usecols: set[str], names: Sequence[Hashable]
    ) -> set[str]:
        ...

    @final
    def _evaluate_usecols(
        self,
        usecols: Callable[[Hashable], object] | set[str] | set[int],
        names: Sequence[Hashable],
    ) -> set[str] | set[int]:
        """
        Check whether or not the 'usecols' parameter
        is a callable.  If so, enumerates the 'names'
        parameter and returns a set of indices for
        each entry in 'names' that evaluates to True.
        If not a callable, returns 'usecols'.
        """
        if callable(usecols):
            return {i for i, name in enumerate(names) if usecols(name)}
        return usecols

    @final
    def _validate_usecols_names(self, usecols, names: Sequence):
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(
                f"Usecols do not match columns, columns expected but not found: "
                f"{missing}"
            )

        return usecols

    @final
    def _validate_usecols_arg(self, usecols):
        """
        Validate the 'usecols' parameter.

        Checks whether or not the 'usecols' parameter contains all integers
        (column selection by index), strings (column by name) or is a callable.
        Raises a ValueError if that is not the case.

        Parameters
        ----------
        usecols : list-like, callable, or None
            List of columns to use when parsing or a callable that can be used
            to filter a list of table columns.

        Returns
        -------
        usecols_tuple : tuple
            A tuple of (verified_usecols, usecols_dtype).

            'verified_usecols' is either a set if an array-like is passed in or
            'usecols' if a callable or None is passed in.

            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
            is passed in or None if a callable or None is passed in.
        """
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, "
            "all integers or a callable."
        )
        if usecols is not None:
            if callable(usecols):
                return usecols, None

            if not is_list_like(usecols):
                # see gh-20529
                #
                # Ensure it is iterable container but not string.
                raise ValueError(msg)

            usecols_dtype = lib.infer_dtype(usecols, skipna=False)

            if usecols_dtype not in ("empty", "integer", "string"):
                raise ValueError(msg)

            usecols = set(usecols)

            return usecols, usecols_dtype
        return usecols, None

    @final
    def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]:
        if not is_index_col(index_col):
            return None, columns, index_col

        columns = list(columns)

        # In case of no rows and multiindex columns we have to set index_names to
        # list of Nones GH#38292
        if not columns:
            return [None] * len(index_col), columns, index_col

        cp_cols = list(columns)
        index_names: list[str | int | None] = []

        # don't mutate
        index_col = list(index_col)

        for i, c in enumerate(index_col):
            if isinstance(c, str):
                index_names.append(c)
                for j, name in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        columns.remove(name)
                        break
            else:
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)

        # Only clean index names that were placeholders.
        for i, name in enumerate(index_names):
            if isinstance(name, str) and name in self.unnamed_cols:
                index_names[i] = None

        return index_names, columns, index_col

    @final
    def _get_empty_meta(self, columns, dtype: DtypeArg | None = None):
        columns = list(columns)

        index_col = self.index_col
        index_names = self.index_names

        # Convert `dtype` to a defaultdict of some kind.
        # This will enable us to write `dtype[col_name]`
        # without worrying about KeyError issues later on.
        dtype_dict: defaultdict[Hashable, Any]
        if not is_dict_like(dtype):
            # if dtype == None, default will be object.
            dtype_dict = defaultdict(lambda: dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(
                lambda: None,
                {columns[k] if is_integer(k) else k: v for k, v in dtype.items()},
            )

        # Even though we have no data, the "index" of the empty DataFrame
        # could for example still be an empty MultiIndex. Thus, we need to
        # check whether we have any index columns specified, via either:
        #
        # 1) index_col (column indices)
        # 2) index_names (column names)
        #
        # Both must be non-null to ensure a successful construction. Otherwise,
        # we have to create a generic empty Index.
        index: Index
        if (index_col is None or index_col is False) or index_names is None:
            index = default_index(0)
        else:
            # TODO: We could return default_index(0) if dtype_dict[name] is None
            data = [
                Index([], name=name, dtype=dtype_dict[name]) for name in index_names
            ]
            if len(data) == 1:
                index = data[0]
            else:
                index = MultiIndex.from_arrays(data)
            index_col.sort()

            for i, n in enumerate(index_col):
                columns.pop(n - i)

        col_dict = {
            col_name: Series([], dtype=dtype_dict[col_name]) for col_name in columns
        }

        return index, columns, col_dict


def _make_date_converter(
    date_parser=lib.no_default,
    dayfirst: bool = False,
    cache_dates: bool = True,
    date_format: dict[Hashable, str] | str | None = None,
):
    if date_parser is not lib.no_default:
        warnings.warn(
            "The argument 'date_parser' is deprecated and will "
            "be removed in a future version. "
            "Please use 'date_format' instead, or read your data in as 'object' dtype "
            "and then call 'to_datetime'.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    if date_parser is not lib.no_default and date_format is not None:
        raise TypeError("Cannot use both 'date_parser' and 'date_format'")

    def unpack_if_single_element(arg):
        # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
        if isinstance(arg, np.ndarray) and arg.ndim == 1 and len(arg) == 1:
            return arg[0]
        return arg

    def converter(*date_cols, col: Hashable):
        if len(date_cols) == 1 and date_cols[0].dtype.kind in "Mm":
            return date_cols[0]

        if date_parser is lib.no_default:
            strs = parsing.concat_date_cols(date_cols)
            date_fmt = (
                date_format.get(col) if isinstance(date_format, dict) else date_format
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    ".*parsing datetimes with mixed time zones will raise an error",
                    category=FutureWarning,
                )
                str_objs = ensure_object(strs)
                try:
                    result = tools.to_datetime(
                        str_objs,
                        format=date_fmt,
                        utc=False,
                        dayfirst=dayfirst,
                        cache=cache_dates,
                    )
                except (ValueError, TypeError):
                    # test_usecols_with_parse_dates4
                    return str_objs

            if isinstance(result, DatetimeIndex):
                arr = result.to_numpy()
                arr.flags.writeable = True
                return arr
            return result._values
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*parsing datetimes with mixed time zones "
                        "will raise an error",
                        category=FutureWarning,
                    )
                    pre_parsed = date_parser(
                        *(unpack_if_single_element(arg) for arg in date_cols)
                    )
                    try:
                        result = tools.to_datetime(
                            pre_parsed,
                            cache=cache_dates,
                        )
                    except (ValueError, TypeError):
                        # test_read_csv_with_custom_date_parser
                        result = pre_parsed
                if isinstance(result, datetime.datetime):
                    raise Exception("scalar parser")
                return result
            except Exception:
                # e.g. test_datetime_fractional_seconds
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*parsing datetimes with mixed time zones "
                        "will raise an error",
                        category=FutureWarning,
                    )
                    pre_parsed = parsing.try_parse_dates(
                        parsing.concat_date_cols(date_cols),
                        parser=date_parser,
                    )
                    try:
                        return tools.to_datetime(pre_parsed)
                    except (ValueError, TypeError):
                        # TODO: not reached in tests 2023-10-27; needed?
                        return pre_parsed

    return converter


parser_defaults = {
    "delimiter": None,
    "escapechar": None,
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "doublequote": True,
    "skipinitialspace": False,
    "lineterminator": None,
    "header": "infer",
    "index_col": None,
    "names": None,
    "skiprows": None,
    "skipfooter": 0,
    "nrows": None,
    "na_values": None,
    "keep_default_na": True,
    "true_values": None,
    "false_values": None,
    "converters": None,
    "dtype": None,
    "cache_dates": True,
    "thousands": None,
    "comment": None,
    "decimal": ".",
    # 'engine': 'c',
    "parse_dates": False,
    "keep_date_col": False,
    "dayfirst": False,
    "date_parser": lib.no_default,
    "date_format": None,
    "usecols": None,
    # 'iterator': False,
    "chunksize": None,
    "verbose": False,
    "encoding": None,
    "compression": None,
    "skip_blank_lines": True,
    "encoding_errors": "strict",
    "on_bad_lines": ParserBase.BadLineHandleMethod.ERROR,
    "dtype_backend": lib.no_default,
}


def _process_date_conversion(
    data_dict,
    converter: Callable,
    parse_spec,
    index_col,
    index_names,
    columns,
    keep_date_col: bool = False,
    dtype_backend=lib.no_default,
):
    def _isindex(colspec):
        return (isinstance(index_col, list) and colspec in index_col) or (
            isinstance(index_names, list) and colspec in index_names
        )

    new_cols = []
    new_data = {}

    orig_names = columns
    columns = list(columns)

    date_cols = set()

    if parse_spec is None or isinstance(parse_spec, bool):
        return data_dict, columns

    if isinstance(parse_spec, list):
        # list of column lists
        for colspec in parse_spec:
            if is_scalar(colspec) or isinstance(colspec, tuple):
                if isinstance(colspec, int) and colspec not in data_dict:
                    colspec = orig_names[colspec]
                if _isindex(colspec):
                    continue
                elif dtype_backend == "pyarrow":
                    import pyarrow as pa

                    dtype = data_dict[colspec].dtype
                    if isinstance(dtype, ArrowDtype) and (
                        pa.types.is_timestamp(dtype.pyarrow_dtype)
                        or pa.types.is_date(dtype.pyarrow_dtype)
                    ):
                        continue

                # Pyarrow engine returns Series which we need to convert to
                # numpy array before converter, its a no-op for other parsers
                data_dict[colspec] = converter(
                    np.asarray(data_dict[colspec]), col=colspec
                )
            else:
                new_name, col, old_names = _try_convert_dates(
                    converter, colspec, data_dict, orig_names
                )
                if new_name in data_dict:
                    raise ValueError(f"New date column already in dict {new_name}")
                new_data[new_name] = col
                new_cols.append(new_name)
                date_cols.update(old_names)

    elif isinstance(parse_spec, dict):
        # dict of new name to column list
        for new_name, colspec in parse_spec.items():
            if new_name in data_dict:
                raise ValueError(f"Date column {new_name} already in dict")

            _, col, old_names = _try_convert_dates(
                converter,
                colspec,
                data_dict,
                orig_names,
                target_name=new_name,
            )

            new_data[new_name] = col

            # If original column can be converted to date we keep the converted values
            # This can only happen if values are from single column
            if len(colspec) == 1:
                new_data[colspec[0]] = col

            new_cols.append(new_name)
            date_cols.update(old_names)

    if isinstance(data_dict, DataFrame):
        data_dict = concat([DataFrame(new_data), data_dict], axis=1, copy=False)
    else:
        data_dict.update(new_data)
    new_cols.extend(columns)

    if not keep_date_col:
        for c in list(date_cols):
            data_dict.pop(c)
            new_cols.remove(c)

    return data_dict, new_cols


def _try_convert_dates(
    parser: Callable, colspec, data_dict, columns, target_name: str | None = None
):
    colset = set(columns)
    colnames = []

    for c in colspec:
        if c in colset:
            colnames.append(c)
        elif isinstance(c, int) and c not in columns:
            colnames.append(columns[c])
        else:
            colnames.append(c)

    new_name: tuple | str
    if all(isinstance(x, tuple) for x in colnames):
        new_name = tuple(map("_".join, zip(*colnames)))
    else:
        new_name = "_".join([str(x) for x in colnames])
    to_parse = [np.asarray(data_dict[c]) for c in colnames if c in data_dict]

    new_col = parser(*to_parse, col=new_name if target_name is None else target_name)
    return new_name, new_col, colnames


def _get_na_values(col, na_values, na_fvalues, keep_default_na: bool):
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
    if isinstance(na_values, dict):
        if col in na_values:
            return na_values[col], na_fvalues[col]
        else:
            if keep_default_na:
                return STR_NA_VALUES, set()

            return set(), set()
    else:
        return na_values, na_fvalues


def _validate_parse_dates_arg(parse_dates):
    """
    Check whether or not the 'parse_dates' parameter
    is a non-boolean scalar. Raises a ValueError if
    that is the case.
    """
    msg = (
        "Only booleans, lists, and dictionaries are accepted "
        "for the 'parse_dates' parameter"
    )

    if not (
        parse_dates is None
        or lib.is_bool(parse_dates)
        or isinstance(parse_dates, (list, dict))
    ):
        raise TypeError(msg)

    return parse_dates


def is_index_col(col) -> bool:
    return col is not None and col is not False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_testing\asserters.py ===
from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions

from pandas.core.dtypes.common import (
    is_bool,
    is_float_dtype,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.missing import array_equivalent

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalDtype,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


def assert_almost_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = "equiv",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> None:
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    check_dtype : bool or {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Index` with int64 dtype are also considered
        equivalent when doing type checking.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    """
    if isinstance(left, Index):
        assert_index_equal(
            left,
            right,
            check_exact=False,
            exact=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, Series):
        assert_series_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, DataFrame):
        assert_frame_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    else:
        # Other sequences.
        if check_dtype:
            if is_number(left) and is_number(right):
                # Do not compare numeric classes, like np.float64 and float.
                pass
            elif is_bool(left) and is_bool(right):
                # Do not compare bool classes, like np.bool_ and bool.
                pass
            else:
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                    obj = "numpy array"
                else:
                    obj = "Input"
                assert_class_equal(left, right, obj=obj)

        # if we have "equiv", this becomes True
        _testing.assert_almost_equal(
            left, right, check_dtype=bool(check_dtype), rtol=rtol, atol=atol, **kwargs
        )


def _check_isinstance(left, right, cls) -> None:
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """
    cls_name = cls.__name__

    if not isinstance(left, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(left)} instead"
        )
    if not isinstance(right, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(right)} instead"
        )


def assert_dict_equal(left, right, compare_keys: bool = True) -> None:
    _check_isinstance(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)


def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool | str = "equiv",
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Index",
) -> None:
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
    right : Index
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)
    """
    __tracebackhide__ = True

    def _check_types(left, right, obj: str = "Index") -> None:
        if not exact:
            return

        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal("inferred_type", left, right, obj=obj)

        # Skip exact dtype checking when `check_categorical` is False
        if isinstance(left.dtype, CategoricalDtype) and isinstance(
            right.dtype, CategoricalDtype
        ):
            if check_categorical:
                assert_attr_equal("dtype", left, right, obj=obj)
                assert_index_equal(left.categories, right.categories, exact=exact)
            return

        assert_attr_equal("dtype", left, right, obj=obj)

    # instance validation
    _check_isinstance(left, right, Index)

    # class / dtype comparison
    _check_types(left, right, obj=obj)

    # level comparison
    if left.nlevels != right.nlevels:
        msg1 = f"{obj} levels are different"
        msg2 = f"{left.nlevels}, {left}"
        msg3 = f"{right.nlevels}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{obj} length are different"
        msg2 = f"{len(left)}, {left}"
        msg3 = f"{len(right)}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # If order doesn't matter then sort the index entries
    if not check_order:
        left = safe_sort_index(left)
        right = safe_sort_index(right)

    # MultiIndex special comparison for little-friendly error messages
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)

        for level in range(left.nlevels):
            lobj = f"MultiIndex level [{level}]"
            try:
                # try comparison on levels/codes to avoid densifying MultiIndex
                assert_index_equal(
                    left.levels[level],
                    right.levels[level],
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)

                assert_index_equal(
                    llevel,
                    rlevel,
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            # get_level_values may change dtype
            _check_types(left.levels[level], right.levels[level], obj=obj)

    # skip exact index checking when `check_categorical` is False
    elif check_exact and check_categorical:
        if not left.equals(right):
            mismatch = left._values != right._values

            if not isinstance(mismatch, np.ndarray):
                mismatch = cast("ExtensionArray", mismatch).fillna(True)

            diff = np.sum(mismatch.astype(int)) * 100.0 / len(left)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right)
    else:
        # if we have "equiv", this becomes True
        exact_bool = bool(exact)
        _testing.assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=exact_bool,
            obj=obj,
            lobj=left,
            robj=right,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("names", left, right, obj=obj)
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        assert_attr_equal("dtype", left, right, obj=obj)
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        assert_interval_array_equal(left._values, right._values)

    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            assert_categorical_equal(left._values, right._values, obj=f"{obj} category")


def assert_class_equal(
    left, right, exact: bool | str = True, obj: str = "Input"
) -> None:
    """
    Checks classes are equal.
    """
    __tracebackhide__ = True

    def repr_class(x):
        if isinstance(x, Index):
            # return Index as it is to include values in the error message
            return x

        return type(x).__name__

    def is_class_equiv(idx: Index) -> bool:
        """Classes that are a RangeIndex (sub-)instance or exactly an `Index` .

        This only checks class equivalence. There is a separate check that the
        dtype is int64.
        """
        return type(idx) is Index or isinstance(idx, RangeIndex)

    if type(left) == type(right):
        return

    if exact == "equiv":
        if is_class_equiv(left) and is_class_equiv(right):
            return

    msg = f"{obj} classes are different"
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))


def assert_attr_equal(attr: str, left, right, obj: str = "Attributes") -> None:
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    __tracebackhide__ = True

    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    if left_attr is right_attr or is_matching_na(left_attr, right_attr):
        # e.g. both np.nan, both NaT, both pd.NA, ...
        return None

    try:
        result = left_attr == right_attr
    except TypeError:
        # datetimetz on rhs may raise TypeError
        result = False
    if (left_attr is pd.NA) ^ (right_attr is pd.NA):
        result = False
    elif not isinstance(result, bool):
        result = result.all()

    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None


def assert_is_valid_plot_return_object(objs) -> None:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    if isinstance(objs, (Series, np.ndarray)):
        if isinstance(objs, Series):
            objs = objs._values
        for el in objs.ravel():
            msg = (
                "one of 'objs' is not a matplotlib Axes instance, "
                f"type encountered {repr(type(el).__name__)}"
            )
            assert isinstance(el, (Axes, dict)), msg
    else:
        msg = (
            "objs is neither an ndarray of Artist instances nor a single "
            "ArtistArtist instance, tuple, or dict, 'objs' is a "
            f"{repr(type(objs).__name__)}"
        )
        assert isinstance(objs, (Artist, tuple, dict)), msg


def assert_is_sorted(seq) -> None:
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    # sorting does not change precisions
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])


def assert_categorical_equal(
    left,
    right,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None:
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same.
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    """
    _check_isinstance(left, right, Categorical)

    exact: bool | str
    if isinstance(left.categories, RangeIndex) or isinstance(
        right.categories, RangeIndex
    ):
        exact = "equiv"
    else:
        # We still want to require exact matches for Index
        exact = True

    if check_category_order:
        assert_index_equal(
            left.categories, right.categories, obj=f"{obj}.categories", exact=exact
        )
        assert_numpy_array_equal(
            left.codes, right.codes, check_dtype=check_dtype, obj=f"{obj}.codes"
        )
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            # e.g. '<' not supported between instances of 'int' and 'str'
            lc, rc = left.categories, right.categories
        assert_index_equal(lc, rc, obj=f"{obj}.categories", exact=exact)
        assert_index_equal(
            left.categories.take(left.codes),
            right.categories.take(right.codes),
            obj=f"{obj}.values",
            exact=exact,
        )

    assert_attr_equal("ordered", left, right, obj=obj)


def assert_interval_array_equal(
    left, right, exact: bool | Literal["equiv"] = "equiv", obj: str = "IntervalArray"
) -> None:
    """
    Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    _check_isinstance(left, right, IntervalArray)

    kwargs = {}
    if left._left.dtype.kind in "mM":
        # We have a DatetimeArray or TimedeltaArray
        kwargs["check_freq"] = False

    assert_equal(left._left, right._left, obj=f"{obj}.left", **kwargs)
    assert_equal(left._right, right._right, obj=f"{obj}.left", **kwargs)

    assert_attr_equal("closed", left, right, obj=obj)


def assert_period_array_equal(left, right, obj: str = "PeriodArray") -> None:
    _check_isinstance(left, right, PeriodArray)

    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    assert_attr_equal("dtype", left, right, obj=obj)


def assert_datetime_array_equal(
    left, right, obj: str = "DatetimeArray", check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    _check_isinstance(left, right, DatetimeArray)

    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)
    assert_attr_equal("tz", left, right, obj=obj)


def assert_timedelta_array_equal(
    left, right, obj: str = "TimedeltaArray", check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    _check_isinstance(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)


def raise_assert_detail(
    obj, message, left, right, diff=None, first_diff=None, index_values=None
) -> NoReturn:
    __tracebackhide__ = True

    msg = f"""{obj} are different

{message}"""

    if isinstance(index_values, Index):
        index_values = np.asarray(index_values)

    if isinstance(index_values, np.ndarray):
        msg += f"\n[index]: {pprint_thing(index_values)}"

    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (CategoricalDtype, NumpyEADtype)):
        left = repr(left)
    elif isinstance(left, StringDtype):
        # TODO(infer_string) this special case could be avoided if we have
        # a more informative repr https://github.com/pandas-dev/pandas/issues/59342
        left = f"StringDtype(storage={left.storage}, na_value={left.na_value})"

    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (CategoricalDtype, NumpyEADtype)):
        right = repr(right)
    elif isinstance(right, StringDtype):
        right = f"StringDtype(storage={right.storage}, na_value={right.na_value})"

    msg += f"""
[left]:  {left}
[right]: {right}"""

    if diff is not None:
        msg += f"\n[diff]: {diff}"

    if first_diff is not None:
        msg += f"\n{first_diff}"

    raise AssertionError(msg)


def assert_numpy_array_equal(
    left,
    right,
    strict_nan: bool = False,
    check_dtype: bool | Literal["equiv"] = True,
    err_msg=None,
    check_same=None,
    obj: str = "numpy array",
    index_values=None,
) -> None:
    """
    Check that 'np.ndarray' is equivalent.

    Parameters
    ----------
    left, right : numpy.ndarray or iterable
        The two arrays to be compared.
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype : bool, default True
        Check dtype if both a and b are np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | numpy.ndarray, default None
        optional index (shared by both left and right), used in output.
    """
    __tracebackhide__ = True

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an np.ndarray
    _check_isinstance(left, right, np.ndarray)

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

    def _raise(left, right, err_msg) -> NoReturn:
        if err_msg is None:
            if left.shape != right.shape:
                raise_assert_detail(
                    obj, f"{obj} shapes are different", left.shape, right.shape
                )

            diff = 0
            for left_arr, right_arr in zip(left, right):
                # count up differences
                if not array_equivalent(left_arr, right_arr, strict_nan=strict_nan):
                    diff += 1

            diff = diff * 100.0 / left.size
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right, index_values=index_values)

        raise AssertionError(err_msg)

    # compare shape and values
    if not array_equivalent(left, right, strict_nan=strict_nan):
        _raise(left, right, err_msg)

    if check_dtype:
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            assert_attr_equal("dtype", left, right, obj=obj)


def assert_extension_array_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    index_values=None,
    check_exact: bool | lib.NoDefault = lib.no_default,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "ExtensionArray",
) -> None:
    """
    Check that left and right ExtensionArrays are equal.

    Parameters
    ----------
    left, right : ExtensionArray
        The two arrays to compare.
    check_dtype : bool, default True
        Whether to check if the ExtensionArray dtypes are identical.
    index_values : Index | numpy.ndarray, default None
        Optional index (shared by both left and right), used in output.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'ExtensionArray'
        Specify object name being compared, internally used to show appropriate
        assertion message.

        .. versionadded:: 2.0.0

    Notes
    -----
    Missing values are checked separately from valid values.
    A mask of missing values is computed for each and checked to match.
    The remaining all-valid values are cast to object dtype and checked.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b, c = a.array, a.array
    >>> tm.assert_extension_array_equal(b, c)
    """
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype)
            and not is_float_dtype(left.dtype)
            or is_numeric_dtype(right.dtype)
            and not is_float_dtype(right.dtype)
        )
    elif check_exact is lib.no_default:
        check_exact = False

    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    assert isinstance(left, ExtensionArray), "left is not an ExtensionArray"
    assert isinstance(right, ExtensionArray), "right is not an ExtensionArray"
    if check_dtype:
        assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")

    if (
        isinstance(left, DatetimeLikeArrayMixin)
        and isinstance(right, DatetimeLikeArrayMixin)
        and type(right) == type(left)
    ):
        # GH 52449
        if not check_dtype and left.dtype.kind in "mM":
            if not isinstance(left.dtype, np.dtype):
                l_unit = cast(DatetimeTZDtype, left.dtype).unit
            else:
                l_unit = np.datetime_data(left.dtype)[0]
            if not isinstance(right.dtype, np.dtype):
                r_unit = cast(DatetimeTZDtype, right.dtype).unit
            else:
                r_unit = np.datetime_data(right.dtype)[0]
            if (
                l_unit != r_unit
                and compare_mismatched_resolutions(
                    left._ndarray, right._ndarray, operator.eq
                ).all()
            ):
                return
        # Avoid slow object-dtype comparisons
        # np.asarray for case where we have a np.MaskedArray
        assert_numpy_array_equal(
            np.asarray(left.asi8),
            np.asarray(right.asi8),
            index_values=index_values,
            obj=obj,
        )
        return

    left_na = np.asarray(left.isna())
    right_na = np.asarray(right.isna())
    assert_numpy_array_equal(
        left_na, right_na, obj=f"{obj} NA mask", index_values=index_values
    )

    # Specifically for StringArrayNumpySemantics, validate here we have a valid array
    if (
        isinstance(left.dtype, StringDtype)
        and left.dtype.storage == "python"
        and left.dtype.na_value is np.nan
    ):
        assert np.all(
            [np.isnan(val) for val in left._ndarray[left_na]]  # type: ignore[attr-defined]
        ), "wrong missing value sentinels"
    if (
        isinstance(right.dtype, StringDtype)
        and right.dtype.storage == "python"
        and right.dtype.na_value is np.nan
    ):
        assert np.all(
            [np.isnan(val) for val in right._ndarray[right_na]]  # type: ignore[attr-defined]
        ), "wrong missing value sentinels"

    left_valid = left[~left_na].to_numpy(dtype=object)
    right_valid = right[~right_na].to_numpy(dtype=object)
    if check_exact:
        assert_numpy_array_equal(
            left_valid, right_valid, obj=obj, index_values=index_values
        )
    else:
        _testing.assert_almost_equal(
            left_valid,
            right_valid,
            check_dtype=bool(check_dtype),
            rtol=rtol,
            atol=atol,
            obj=obj,
            index_values=index_values,
        )


# This could be refactored to use the NDFrame.equals method
def assert_series_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "Series",
    *,
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    """
    Check that left and right Series are equal.

    Parameters
    ----------
    left : Series
    right : Series
    check_dtype : bool, default True
        Whether to check the Series dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_series_type : bool, default True
         Whether to check the Series class is identical.
    check_names : bool, default True
        Whether to check the Series and Index names attribute.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_category_order : bool, default True
        Whether to compare category order of internal Categoricals.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Series'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    check_index : bool, default True
        Whether to check index equivalence. If False, then compare only values.

        .. versionadded:: 1.3.0
    check_like : bool, default False
        If True, ignore the order of the index. Must be False if check_index is False.
        Note: same labels must be with the same data.

        .. versionadded:: 1.5.0

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Series([1, 2, 3, 4])
    >>> b = pd.Series([1, 2, 3, 4])
    >>> tm.assert_series_equal(a, b)
    """
    __tracebackhide__ = True
    check_exact_index = False if check_exact is lib.no_default else check_exact
    if (
        check_exact is lib.no_default
        and rtol is lib.no_default
        and atol is lib.no_default
    ):
        check_exact = (
            is_numeric_dtype(left.dtype)
            and not is_float_dtype(left.dtype)
            or is_numeric_dtype(right.dtype)
            and not is_float_dtype(right.dtype)
        )
    elif check_exact is lib.no_default:
        check_exact = False

    rtol = rtol if rtol is not lib.no_default else 1.0e-5
    atol = atol if atol is not lib.no_default else 1.0e-8

    if not check_index and check_like:
        raise ValueError("check_like must be False if check_index is False")

    # instance validation
    _check_isinstance(left, right, Series)

    if check_series_type:
        assert_class_equal(left, right, obj=obj)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{len(left)}, {left.index}"
        msg2 = f"{len(right)}, {right.index}"
        raise_assert_detail(obj, "Series length are different", msg1, msg2)

    if check_flags:
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"

    if check_index:
        # GH #38183
        assert_index_equal(
            left.index,
            right.index,
            exact=check_index_type,
            check_names=check_names,
            check_exact=check_exact_index,
            check_categorical=check_categorical,
            check_order=not check_like,
            rtol=rtol,
            atol=atol,
            obj=f"{obj}.index",
        )

    if check_like:
        left = left.reindex_like(right)

    if check_freq and isinstance(left.index, (DatetimeIndex, TimedeltaIndex)):
        lidx = left.index
        ridx = right.index
        assert lidx.freq == ridx.freq, (lidx.freq, ridx.freq)

    if check_dtype:
        # We want to skip exact dtype checking when `check_categorical`
        # is False. We'll still raise if only one is a `Categorical`,
        # regardless of `check_categorical`
        if (
            isinstance(left.dtype, CategoricalDtype)
            and isinstance(right.dtype, CategoricalDtype)
            and not check_categorical
        ):
            pass
        else:
            assert_attr_equal("dtype", left, right, obj=f"Attributes of {obj}")
    if check_exact:
        left_values = left._values
        right_values = right._values
        # Only check exact if dtype is numeric
        if isinstance(left_values, ExtensionArray) and isinstance(
            right_values, ExtensionArray
        ):
            assert_extension_array_equal(
                left_values,
                right_values,
                check_dtype=check_dtype,
                index_values=left.index,
                obj=str(obj),
            )
        else:
            # convert both to NumPy if not, check_dtype would raise earlier
            lv, rv = left_values, right_values
            if isinstance(left_values, ExtensionArray):
                lv = left_values.to_numpy()
            if isinstance(right_values, ExtensionArray):
                rv = right_values.to_numpy()
            assert_numpy_array_equal(
                lv,
                rv,
                check_dtype=check_dtype,
                obj=str(obj),
                index_values=left.index,
            )
    elif check_datetimelike_compat and (
        needs_i8_conversion(left.dtype) or needs_i8_conversion(right.dtype)
    ):
        # we want to check only if we have compat dtypes
        # e.g. integer and M|m are NOT compat, but we can simply check
        # the values in that case

        # datetimelike may have different objects (e.g. datetime.datetime
        # vs Timestamp) but will compare equal
        if not Index(left._values).equals(Index(right._values)):
            msg = (
                f"[datetimelike_compat=True] {left._values} "
                f"is not equal to {right._values}."
            )
            raise AssertionError(msg)
    elif isinstance(left.dtype, IntervalDtype) and isinstance(
        right.dtype, IntervalDtype
    ):
        assert_interval_array_equal(left.array, right.array)
    elif isinstance(left.dtype, CategoricalDtype) or isinstance(
        right.dtype, CategoricalDtype
    ):
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )
    elif isinstance(left.dtype, ExtensionDtype) and isinstance(
        right.dtype, ExtensionDtype
    ):
        assert_extension_array_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    elif is_extension_array_dtype_and_needs_i8_conversion(
        left.dtype, right.dtype
    ) or is_extension_array_dtype_and_needs_i8_conversion(right.dtype, left.dtype):
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    elif needs_i8_conversion(left.dtype) and needs_i8_conversion(right.dtype):
        # DatetimeArray or TimedeltaArray
        assert_extension_array_equal(
            left._values,
            right._values,
            check_dtype=check_dtype,
            index_values=left.index,
            obj=str(obj),
        )
    else:
        _testing.assert_almost_equal(
            left._values,
            right._values,
            rtol=rtol,
            atol=atol,
            check_dtype=bool(check_dtype),
            obj=str(obj),
            index_values=left.index,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("name", left, right, obj=obj)

    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            assert_categorical_equal(
                left._values,
                right._values,
                obj=f"{obj} category",
                check_category_order=check_category_order,
            )


# This could be refactored to use the NDFrame.equals method
def assert_frame_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = True,
    check_index_type: bool | Literal["equiv"] = "equiv",
    check_column_type: bool | Literal["equiv"] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool | lib.NoDefault = lib.no_default,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float | lib.NoDefault = lib.no_default,
    atol: float | lib.NoDefault = lib.no_default,
    obj: str = "DataFrame",
) -> None:
    """
    Check that left and right DataFrame are equal.

    This function is intended to compare two DataFrames and output any
    differences. It is mostly intended for use in unit tests.
    Additional parameters allow varying the strictness of the
    equality checks performed.

    Parameters
    ----------
    left : DataFrame
        First DataFrame to compare.
    right : DataFrame
        Second DataFrame to compare.
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical.
    check_column_type : bool or {'equiv'}, default 'equiv'
        Whether to check the columns class, dtype and inferred_type
        are identical. Is passed as the ``exact`` argument of
        :func:`assert_index_equal`.
    check_frame_type : bool, default True
        Whether to check the DataFrame class is identical.
    check_names : bool, default True
        Whether to check that the `names` attribute for both the `index`
        and `column` attributes of the DataFrame is identical.
    by_blocks : bool, default False
        Specify how to compare internal data. If False, compare by columns.
        If True, compare by blocks.
    check_exact : bool, default False
        Whether to compare number exactly.

        .. versionchanged:: 2.2.0

            Defaults to True for integer dtypes if none of
            ``check_exact``, ``rtol`` and ``atol`` are specified.
    check_datetimelike_compat : bool, default False
        Compare datetime-like which is comparable ignoring dtype.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_like : bool, default False
        If True, ignore the order of index & columns.
        Note: index labels must match their respective rows
        (same as in columns) - same labels must be with the same data.
    check_freq : bool, default True
        Whether to check the `freq` attribute on a DatetimeIndex or TimedeltaIndex.
    check_flags : bool, default True
        Whether to check the `flags` attribute.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'DataFrame'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    See Also
    --------
    assert_series_equal : Equivalent method for asserting Series equality.
    DataFrame.equals : Check DataFrame equality.

    Examples
    --------
    This example shows comparing two DataFrames that are equal
    but with columns of differing dtypes.

    >>> from pandas.testing import assert_frame_equal
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})

    df1 equals itself.

    >>> assert_frame_equal(df1, df1)

    df1 differs from df2 as column 'b' is of a different type.

    >>> assert_frame_equal(df1, df2)
    Traceback (most recent call last):
    ...
    AssertionError: Attributes of DataFrame.iloc[:, 1] (column name="b") are different

    Attribute "dtype" are different
    [left]:  int64
    [right]: float64

    Ignore differing dtypes in columns with check_dtype.

    >>> assert_frame_equal(df1, df2, check_dtype=False)
    """
    __tracebackhide__ = True
    _rtol = rtol if rtol is not lib.no_default else 1.0e-5
    _atol = atol if atol is not lib.no_default else 1.0e-8
    _check_exact = check_exact if check_exact is not lib.no_default else False

    # instance validation
    _check_isinstance(left, right, DataFrame)

    if check_frame_type:
        assert isinstance(left, type(right))
        # assert_class_equal(left, right, obj=obj)

    # shape comparison
    if left.shape != right.shape:
        raise_assert_detail(
            obj, f"{obj} shape mismatch", f"{repr(left.shape)}", f"{repr(right.shape)}"
        )

    if check_flags:
        assert left.flags == right.flags, f"{repr(left.flags)} != {repr(right.flags)}"

    # index comparison
    assert_index_equal(
        left.index,
        right.index,
        exact=check_index_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.index",
    )

    # column comparison
    assert_index_equal(
        left.columns,
        right.columns,
        exact=check_column_type,
        check_names=check_names,
        check_exact=_check_exact,
        check_categorical=check_categorical,
        check_order=not check_like,
        rtol=_rtol,
        atol=_atol,
        obj=f"{obj}.columns",
    )

    if check_like:
        left = left.reindex_like(right)

    # compare by blocks
    if by_blocks:
        rblocks = right._to_dict_of_blocks()
        lblocks = left._to_dict_of_blocks()
        for dtype in list(set(list(lblocks.keys()) + list(rblocks.keys()))):
            assert dtype in lblocks
            assert dtype in rblocks
            assert_frame_equal(
                lblocks[dtype], rblocks[dtype], check_dtype=check_dtype, obj=obj
            )

    # compare by columns
    else:
        for i, col in enumerate(left.columns):
            # We have already checked that columns match, so we can do
            #  fast location-based lookups
            lcol = left._ixs(i, axis=1)
            rcol = right._ixs(i, axis=1)

            # GH #38183
            # use check_index=False, because we do not want to run
            # assert_index_equal for each column,
            # as we already checked it for the whole dataframe before.
            assert_series_equal(
                lcol,
                rcol,
                check_dtype=check_dtype,
                check_index_type=check_index_type,
                check_exact=check_exact,
                check_names=check_names,
                check_datetimelike_compat=check_datetimelike_compat,
                check_categorical=check_categorical,
                check_freq=check_freq,
                obj=f'{obj}.iloc[:, {i}] (column name="{col}")',
                rtol=rtol,
                atol=atol,
                check_index=False,
                check_flags=False,
            )


def assert_equal(left, right, **kwargs) -> None:
    """
    Wrapper for tm.assert_*_equal to dispatch to the appropriate test function.

    Parameters
    ----------
    left, right : Index, Series, DataFrame, ExtensionArray, or np.ndarray
        The two items to be compared.
    **kwargs
        All keyword arguments are passed through to the underlying assert method.
    """
    __tracebackhide__ = True

    if isinstance(left, Index):
        assert_index_equal(left, right, **kwargs)
        if isinstance(left, (DatetimeIndex, TimedeltaIndex)):
            assert left.freq == right.freq, (left.freq, right.freq)
    elif isinstance(left, Series):
        assert_series_equal(left, right, **kwargs)
    elif isinstance(left, DataFrame):
        assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, IntervalArray):
        assert_interval_array_equal(left, right, **kwargs)
    elif isinstance(left, PeriodArray):
        assert_period_array_equal(left, right, **kwargs)
    elif isinstance(left, DatetimeArray):
        assert_datetime_array_equal(left, right, **kwargs)
    elif isinstance(left, TimedeltaArray):
        assert_timedelta_array_equal(left, right, **kwargs)
    elif isinstance(left, ExtensionArray):
        assert_extension_array_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray):
        assert_numpy_array_equal(left, right, **kwargs)
    elif isinstance(left, str):
        assert kwargs == {}
        assert left == right
    else:
        assert kwargs == {}
        assert_almost_equal(left, right)


def assert_sp_array_equal(left, right) -> None:
    """
    Check that the left and right SparseArray are equal.

    Parameters
    ----------
    left : SparseArray
    right : SparseArray
    """
    _check_isinstance(left, right, pd.arrays.SparseArray)

    assert_numpy_array_equal(left.sp_values, right.sp_values)

    # SparseIndex comparison
    assert isinstance(left.sp_index, SparseIndex)
    assert isinstance(right.sp_index, SparseIndex)

    left_index = left.sp_index
    right_index = right.sp_index

    if not left_index.equals(right_index):
        raise_assert_detail(
            "SparseArray.index", "index are not equal", left_index, right_index
        )
    else:
        # Just ensure a
        pass

    assert_attr_equal("fill_value", left, right)
    assert_attr_equal("dtype", left, right)
    assert_numpy_array_equal(left.to_dense(), right.to_dense())


def assert_contains_all(iterable, dic) -> None:
    for k in iterable:
        assert k in dic, f"Did not contain item: {repr(k)}"


def assert_copy(iter1, iter2, **eql_kwargs) -> None:
    """
    iter1, iter2: iterables that produce elements
    comparable with assert_almost_equal

    Checks that the elements are equal, but not
    the same object. (Does not check that items
    in sequences are also not the same object)
    """
    for elem1, elem2 in zip(iter1, iter2):
        assert_almost_equal(elem1, elem2, **eql_kwargs)
        msg = (
            f"Expected object {repr(type(elem1))} and object {repr(type(elem2))} to be "
            "different objects, but they were the same object."
        )
        assert elem1 is not elem2, msg


def is_extension_array_dtype_and_needs_i8_conversion(
    left_dtype: DtypeObj, right_dtype: DtypeObj
) -> bool:
    """
    Checks that we have the combination of an ExtensionArraydtype and
    a dtype that should be converted to int64

    Returns
    -------
    bool

    Related to issue #37609
    """
    return isinstance(left_dtype, ExtensionDtype) and needs_i8_conversion(right_dtype)


def assert_indexing_slices_equivalent(ser: Series, l_slc: slice, i_slc: slice) -> None:
    """
    Check that ser.iloc[i_slc] matches ser.loc[l_slc] and, if applicable,
    ser[l_slc].
    """
    expected = ser.iloc[i_slc]

    assert_series_equal(ser.loc[l_slc], expected)

    if not is_integer_dtype(ser.index):
        # For integer indices, .loc and plain getitem are position-based.
        assert_series_equal(ser[l_slc], expected)


def assert_metadata_equivalent(
    left: DataFrame | Series, right: DataFrame | Series | None = None
) -> None:
    """
    Check that ._metadata attributes are equivalent.
    """
    for attr in left._metadata:
        val = getattr(left, attr, None)
        if right is None:
            assert val is None
        else:
            assert val == getattr(right, attr, None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\polynomials.py ===
"""
This module mainly implements special orthogonal polynomials.

See also functions.combinatorial.numbers which contains some
combinatorial polynomials.

"""

from sympy.core import Rational
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial, factorial, RisingFactorial
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sec
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import (chebyshevt_poly, chebyshevu_poly,
                                    gegenbauer_poly, hermite_poly, hermite_prob_poly,
                                    jacobi_poly, laguerre_poly, legendre_poly)

_x = Dummy('x')


class OrthogonalPolynomial(DefinedFunction):
    """Base class for orthogonal polynomials.
    """

    @classmethod
    def _eval_at_order(cls, n, x):
        if n.is_integer and n >= 0:
            return cls._ortho_poly(int(n), _x).subs(_x, x)

    def _eval_conjugate(self):
        return self.func(self.args[0], self.args[1].conjugate())

#----------------------------------------------------------------------------
# Jacobi polynomials
#


class jacobi(OrthogonalPolynomial):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi(n, alpha, beta, x)`` gives the $n$th Jacobi polynomial
    in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    Examples
    ========

    >>> from sympy import jacobi, S, conjugate, diff
    >>> from sympy.abc import a, b, n, x

    >>> jacobi(0, a, b, x)
    1
    >>> jacobi(1, a, b, x)
    a/2 - b/2 + x*(a/2 + b/2 + 1)
    >>> jacobi(2, a, b, x)
    a**2/8 - a*b/4 - a/8 + b**2/8 - b/8 + x**2*(a**2/8 + a*b/4 + 7*a/8 + b**2/8 + 7*b/8 + 3/2) + x*(a**2/4 + 3*a/4 - b**2/4 - 3*b/4) - 1/2

    >>> jacobi(n, a, b, x)
    jacobi(n, a, b, x)

    >>> jacobi(n, a, a, x)
    RisingFactorial(a + 1, n)*gegenbauer(n,
        a + 1/2, x)/RisingFactorial(2*a + 1, n)

    >>> jacobi(n, 0, 0, x)
    legendre(n, x)

    >>> jacobi(n, S(1)/2, S(1)/2, x)
    RisingFactorial(3/2, n)*chebyshevu(n, x)/factorial(n + 1)

    >>> jacobi(n, -S(1)/2, -S(1)/2, x)
    RisingFactorial(1/2, n)*chebyshevt(n, x)/factorial(n)

    >>> jacobi(n, a, b, -x)
    (-1)**n*jacobi(n, b, a, x)

    >>> jacobi(n, a, b, 0)
    gamma(a + n + 1)*hyper((-n, -b - n), (a + 1,), -1)/(2**n*factorial(n)*gamma(a + 1))
    >>> jacobi(n, a, b, 1)
    RisingFactorial(a + 1, n)/factorial(n)

    >>> conjugate(jacobi(n, a, b, x))
    jacobi(n, conjugate(a), conjugate(b), conjugate(x))

    >>> diff(jacobi(n,a,b,x), x)
    (a/2 + b/2 + n/2 + 1/2)*jacobi(n - 1, a + 1, b + 1, x)

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """

    @classmethod
    def eval(cls, n, a, b, x):
        # Simplify to other polynomials
        # P^{a, a}_n(x)
        if a == b:
            if a == Rational(-1, 2):
                return RisingFactorial(S.Half, n) / factorial(n) * chebyshevt(n, x)
            elif a.is_zero:
                return legendre(n, x)
            elif a == S.Half:
                return RisingFactorial(3*S.Half, n) / factorial(n + 1) * chebyshevu(n, x)
            else:
                return RisingFactorial(a + 1, n) / RisingFactorial(2*a + 1, n) * gegenbauer(n, a + S.Half, x)
        elif b == -a:
            # P^{a, -a}_n(x)
            return gamma(n + a + 1) / gamma(n + 1) * (1 + x)**(a/2) / (1 - x)**(a/2) * assoc_legendre(n, -a, x)

        if not n.is_Number:
            # Symbolic result P^{a,b}_n(x)
            # P^{a,b}_n(-x)  --->  (-1)**n * P^{b,a}_n(-x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * jacobi(n, b, a, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return (2**(-n) * gamma(a + n + 1) / (gamma(a + 1) * factorial(n)) *
                        hyper([-b - n, -n], [a + 1], -1))
            if x == S.One:
                return RisingFactorial(a + 1, n) / factorial(n)
            elif x is S.Infinity:
                if n.is_positive:
                    # Make sure a+b+2*n \notin Z
                    if (a + b + 2*n).is_integer:
                        raise ValueError("Error. a + b + 2*n should not be an integer.")
                    return RisingFactorial(a + b + n + 1, n) * S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            return jacobi_poly(n, a, b, x)

    def fdiff(self, argindex=4):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt a
            n, a, b, x = self.args
            k = Dummy("k")
            f1 = 1 / (a + b + n + k + 1)
            f2 = ((a + b + 2*k + 1) * RisingFactorial(b + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt b
            n, a, b, x = self.args
            k = Dummy("k")
            f1 = 1 / (a + b + n + k + 1)
            f2 = (-1)**(n - k) * ((a + b + 2*k + 1) * RisingFactorial(a + k + 1, n - k) /
                  ((n - k) * RisingFactorial(a + b + k + 1, n - k)))
            return Sum(f1 * (jacobi(n, a, b, x) + f2*jacobi(k, a, b, x)), (k, 0, n - 1))
        elif argindex == 4:
            # Diff wrt x
            n, a, b, x = self.args
            return S.Half * (a + b + n + 1) * jacobi(n - 1, a + 1, b + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, b, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = (RisingFactorial(-n, k) * RisingFactorial(a + b + n + 1, k) * RisingFactorial(a + k + 1, n - k) /
                factorial(k) * ((1 - x)/2)**k)
        return 1 / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, a, b, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, a, b, x, **kwargs)

    def _eval_conjugate(self):
        n, a, b, x = self.args
        return self.func(n, a.conjugate(), b.conjugate(), x.conjugate())


def jacobi_normalized(n, a, b, x):
    r"""
    Jacobi polynomial $P_n^{\left(\alpha, \beta\right)}(x)$.

    Explanation
    ===========

    ``jacobi_normalized(n, alpha, beta, x)`` gives the $n$th
    Jacobi polynomial in $x$, $P_n^{\left(\alpha, \beta\right)}(x)$.

    The Jacobi polynomials are orthogonal on $[-1, 1]$ with respect
    to the weight $\left(1-x\right)^\alpha \left(1+x\right)^\beta$.

    This functions returns the polynomials normilzed:

    .. math::

        \int_{-1}^{1}
          P_m^{\left(\alpha, \beta\right)}(x)
          P_n^{\left(\alpha, \beta\right)}(x)
          (1-x)^{\alpha} (1+x)^{\beta} \mathrm{d}x
        = \delta_{m,n}

    Examples
    ========

    >>> from sympy import jacobi_normalized
    >>> from sympy.abc import n,a,b,x

    >>> jacobi_normalized(n, a, b, x)
    jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)/((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1)))

    Parameters
    ==========

    n : integer degree of polynomial

    a : alpha value

    b : beta value

    x : symbol

    See Also
    ========

    gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly,
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Jacobi_polynomials
    .. [2] https://mathworld.wolfram.com/JacobiPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/JacobiP/

    """
    nfactor = (S(2)**(a + b + 1) * (gamma(n + a + 1) * gamma(n + b + 1))
               / (2*n + a + b + 1) / (factorial(n) * gamma(n + a + b + 1)))

    return jacobi(n, a, b, x) / sqrt(nfactor)


#----------------------------------------------------------------------------
# Gegenbauer polynomials
#


class gegenbauer(OrthogonalPolynomial):
    r"""
    Gegenbauer polynomial $C_n^{\left(\alpha\right)}(x)$.

    Explanation
    ===========

    ``gegenbauer(n, alpha, x)`` gives the $n$th Gegenbauer polynomial
    in $x$, $C_n^{\left(\alpha\right)}(x)$.

    The Gegenbauer polynomials are orthogonal on $[-1, 1]$ with
    respect to the weight $\left(1-x^2\right)^{\alpha-\frac{1}{2}}$.

    Examples
    ========

    >>> from sympy import gegenbauer, conjugate, diff
    >>> from sympy.abc import n,a,x
    >>> gegenbauer(0, a, x)
    1
    >>> gegenbauer(1, a, x)
    2*a*x
    >>> gegenbauer(2, a, x)
    -a + x**2*(2*a**2 + 2*a)
    >>> gegenbauer(3, a, x)
    x**3*(4*a**3/3 + 4*a**2 + 8*a/3) + x*(-2*a**2 - 2*a)

    >>> gegenbauer(n, a, x)
    gegenbauer(n, a, x)
    >>> gegenbauer(n, a, -x)
    (-1)**n*gegenbauer(n, a, x)

    >>> gegenbauer(n, a, 0)
    2**n*sqrt(pi)*gamma(a + n/2)/(gamma(a)*gamma(1/2 - n/2)*gamma(n + 1))
    >>> gegenbauer(n, a, 1)
    gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    >>> conjugate(gegenbauer(n, a, x))
    gegenbauer(n, conjugate(a), conjugate(x))

    >>> diff(gegenbauer(n, a, x), x)
    2*a*gegenbauer(n - 1, a + 1, x)

    See Also
    ========

    jacobi,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gegenbauer_polynomials
    .. [2] https://mathworld.wolfram.com/GegenbauerPolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/GegenbauerC3/

    """

    @classmethod
    def eval(cls, n, a, x):
        # For negative n the polynomials vanish
        # See https://functions.wolfram.com/Polynomials/GegenbauerC3/03/01/03/0012/
        if n.is_negative:
            return S.Zero

        # Some special values for fixed a
        if a == S.Half:
            return legendre(n, x)
        elif a == S.One:
            return chebyshevu(n, x)
        elif a == S.NegativeOne:
            return S.Zero

        if not n.is_Number:
            # Handle this before the general sign extraction rule
            if x == S.NegativeOne:
                if (re(a) > S.Half) == True:
                    return S.ComplexInfinity
                else:
                    return (cos(S.Pi*(a+n)) * sec(S.Pi*a) * gamma(2*a+n) /
                                (gamma(2*a) * gamma(n+1)))

            # Symbolic result C^a_n(x)
            # C^a_n(-x)  --->  (-1)**n * C^a_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * gegenbauer(n, a, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return (2**n * sqrt(S.Pi) * gamma(a + S.Half*n) /
                        (gamma((1 - n)/2) * gamma(n + 1) * gamma(a)) )
            if x == S.One:
                return gamma(2*a + n) / (gamma(2*a) * gamma(n + 1))
            elif x is S.Infinity:
                if n.is_positive:
                    return RisingFactorial(a, n) * S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            return gegenbauer_poly(n, a, x)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt a
            n, a, x = self.args
            k = Dummy("k")
            factor1 = 2 * (1 + (-1)**(n - k)) * (k + a) / ((k +
                           n + 2*a) * (n - k))
            factor2 = 2*(k + 1) / ((k + 2*a) * (2*k + 2*a + 1)) + \
                2 / (k + n + 2*a)
            kern = factor1*gegenbauer(k, a, x) + factor2*gegenbauer(n, a, x)
            return Sum(kern, (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt x
            n, a, x = self.args
            return 2*a*gegenbauer(n - 1, a + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, a, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = ((-1)**k * RisingFactorial(a, n - k) * (2*x)**(n - 2*k) /
                (factorial(k) * factorial(n - 2*k)))
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, a, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, a, x, **kwargs)

    def _eval_conjugate(self):
        n, a, x = self.args
        return self.func(n, a.conjugate(), x.conjugate())

#----------------------------------------------------------------------------
# Chebyshev polynomials of first and second kind
#


class chebyshevt(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the first kind, $T_n(x)$.

    Explanation
    ===========

    ``chebyshevt(n, x)`` gives the $n$th Chebyshev polynomial (of the first
    kind) in $x$, $T_n(x)$.

    The Chebyshev polynomials of the first kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\frac{1}{\sqrt{1-x^2}}$.

    Examples
    ========

    >>> from sympy import chebyshevt, diff
    >>> from sympy.abc import n,x
    >>> chebyshevt(0, x)
    1
    >>> chebyshevt(1, x)
    x
    >>> chebyshevt(2, x)
    2*x**2 - 1

    >>> chebyshevt(n, x)
    chebyshevt(n, x)
    >>> chebyshevt(n, -x)
    (-1)**n*chebyshevt(n, x)
    >>> chebyshevt(-n, x)
    chebyshevt(n, x)

    >>> chebyshevt(n, 0)
    cos(pi*n/2)
    >>> chebyshevt(n, -1)
    (-1)**n

    >>> diff(chebyshevt(n, x), x)
    n*chebyshevu(n - 1, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    _ortho_poly = staticmethod(chebyshevt_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result T_n(x)
            # T_n(-x)  --->  (-1)**n * T_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevt(n, -x)
            # T_{-n}(x)  --->  T_n(x)
            if n.could_extract_minus_sign():
                return chebyshevt(-n, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                # T_{-n}(x) == T_n(x)
                return cls._eval_at_order(-n, x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return n * chebyshevu(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = binomial(n, 2*k) * (x**2 - 1)**k * x**(n - 2*k)
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class chebyshevu(OrthogonalPolynomial):
    r"""
    Chebyshev polynomial of the second kind, $U_n(x)$.

    Explanation
    ===========

    ``chebyshevu(n, x)`` gives the $n$th Chebyshev polynomial of the second
    kind in x, $U_n(x)$.

    The Chebyshev polynomials of the second kind are orthogonal on
    $[-1, 1]$ with respect to the weight $\sqrt{1-x^2}$.

    Examples
    ========

    >>> from sympy import chebyshevu, diff
    >>> from sympy.abc import n,x
    >>> chebyshevu(0, x)
    1
    >>> chebyshevu(1, x)
    2*x
    >>> chebyshevu(2, x)
    4*x**2 - 1

    >>> chebyshevu(n, x)
    chebyshevu(n, x)
    >>> chebyshevu(n, -x)
    (-1)**n*chebyshevu(n, x)
    >>> chebyshevu(-n, x)
    -chebyshevu(n - 2, x)

    >>> chebyshevu(n, 0)
    cos(pi*n/2)
    >>> chebyshevu(n, 1)
    n + 1

    >>> diff(chebyshevu(n, x), x)
    (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev_polynomial
    .. [2] https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
    .. [3] https://mathworld.wolfram.com/ChebyshevPolynomialoftheSecondKind.html
    .. [4] https://functions.wolfram.com/Polynomials/ChebyshevT/
    .. [5] https://functions.wolfram.com/Polynomials/ChebyshevU/

    """

    _ortho_poly = staticmethod(chebyshevu_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result U_n(x)
            # U_n(-x)  --->  (-1)**n * U_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * chebyshevu(n, -x)
            # U_{-n}(x)  --->  -U_{n-2}(x)
            if n.could_extract_minus_sign():
                if n == S.NegativeOne:
                    # n can not be -1 here
                    return S.Zero
                elif not (-n - 2).could_extract_minus_sign():
                    return -chebyshevu(-n - 2, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return cos(S.Half * S.Pi * n)
            if x == S.One:
                return S.One + n
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                # U_{-n}(x)  --->  -U_{n-2}(x)
                if n == S.NegativeOne:
                    return S.Zero
                else:
                    return -cls._eval_at_order(-n - 2, x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return ((n + 1) * chebyshevt(n + 1, x) - x * chebyshevu(n, x)) / (x**2 - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k * factorial(
            n - k) * (2*x)**(n - 2*k) / (factorial(k) * factorial(n - 2*k))
        return Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class chebyshevt_root(DefinedFunction):
    r"""
    ``chebyshev_root(n, k)`` returns the $k$th root (indexed from zero) of
    the $n$th Chebyshev polynomial of the first kind; that is, if
    $0 \le k < n$, ``chebyshevt(n, chebyshevt_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevt, chebyshevt_root
    >>> chebyshevt_root(3, 2)
    -sqrt(3)/2
    >>> chebyshevt(3, chebyshevt_root(3, 2))
    0

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """

    @classmethod
    def eval(cls, n, k):
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        return cos(S.Pi*(2*k + 1)/(2*n))


class chebyshevu_root(DefinedFunction):
    r"""
    ``chebyshevu_root(n, k)`` returns the $k$th root (indexed from zero) of the
    $n$th Chebyshev polynomial of the second kind; that is, if $0 \le k < n$,
    ``chebyshevu(n, chebyshevu_root(n, k)) == 0``.

    Examples
    ========

    >>> from sympy import chebyshevu, chebyshevu_root
    >>> chebyshevu_root(3, 2)
    -sqrt(2)/2
    >>> chebyshevu(3, chebyshevu_root(3, 2))
    0

    See Also
    ========

    chebyshevt, chebyshevt_root, chebyshevu,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly
    """


    @classmethod
    def eval(cls, n, k):
        if not ((0 <= k) and (k < n)):
            raise ValueError("must have 0 <= k < n, "
                "got k = %s and n = %s" % (k, n))
        return cos(S.Pi*(k + 1)/(n + 1))

#----------------------------------------------------------------------------
# Legendre polynomials and Associated Legendre polynomials
#


class legendre(OrthogonalPolynomial):
    r"""
    ``legendre(n, x)`` gives the $n$th Legendre polynomial of $x$, $P_n(x)$

    Explanation
    ===========

    The Legendre polynomials are orthogonal on $[-1, 1]$ with respect to
    the constant weight 1. They satisfy $P_n(1) = 1$ for all $n$; further,
    $P_n$ is odd for odd $n$ and even for even $n$.

    Examples
    ========

    >>> from sympy import legendre, diff
    >>> from sympy.abc import x, n
    >>> legendre(0, x)
    1
    >>> legendre(1, x)
    x
    >>> legendre(2, x)
    3*x**2/2 - 1/2
    >>> legendre(n, x)
    legendre(n, x)
    >>> diff(legendre(n,x), x)
    n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    assoc_legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Legendre_polynomial
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    _ortho_poly = staticmethod(legendre_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_n(-x)  --->  (-1)**n * L_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * legendre(n, -x)
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return legendre(-n - S.One, x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return sqrt(S.Pi)/(gamma(S.Half - n/2)*gamma(S.One + n/2))
            elif x == S.One:
                return S.One
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial;
            # L_{-n}(x)  --->  L_{n-1}(x)
            if n.is_negative:
                n = -n - S.One
            return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = +/-1
            # https://www.autodiff.org/ad16/Oral/Buecker_Legendre.pdf says
            # at x = 1:
            #    n*(n + 1)/2            , m = 0
            #    oo                     , m = 1
            #    -(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                      , m = 3, 4, ..., n
            #
            # at x = -1
            #    (-1)**(n+1)*n*(n + 1)/2       , m = 0
            #    (-1)**n*oo                    , m = 1
            #    (-1)**n*(n-1)*n*(n+1)*(n+2)/4 , m = 2
            #    0                             , m = 3, 4, ..., n
            n, x = self.args
            return n/(x**2 - 1)*(x*legendre(n, x) - legendre(n - 1, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k*binomial(n, k)**2*((1 + x)/2)**(n - k)*((1 - x)/2)**k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class assoc_legendre(DefinedFunction):
    r"""
    ``assoc_legendre(n, m, x)`` gives $P_n^m(x)$, where $n$ and $m$ are
    the degree and order or an expression which is related to the nth
    order Legendre polynomial, $P_n(x)$ in the following manner:

    .. math::
        P_n^m(x) = (-1)^m (1 - x^2)^{\frac{m}{2}}
                   \frac{\mathrm{d}^m P_n(x)}{\mathrm{d} x^m}

    Explanation
    ===========

    Associated Legendre polynomials are orthogonal on $[-1, 1]$ with:

    - weight $= 1$            for the same $m$ and different $n$.
    - weight $= \frac{1}{1-x^2}$   for the same $n$ and different $m$.

    Examples
    ========

    >>> from sympy import assoc_legendre
    >>> from sympy.abc import x, m, n
    >>> assoc_legendre(0,0, x)
    1
    >>> assoc_legendre(1,0, x)
    x
    >>> assoc_legendre(1,1, x)
    -sqrt(1 - x**2)
    >>> assoc_legendre(n,m,x)
    assoc_legendre(n, m, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre,
    hermite, hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Associated_Legendre_polynomials
    .. [2] https://mathworld.wolfram.com/LegendrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LegendreP/
    .. [4] https://functions.wolfram.com/Polynomials/LegendreP2/

    """

    @classmethod
    def _eval_at_order(cls, n, m):
        P = legendre_poly(n, _x, polys=True).diff((_x, m))
        return S.NegativeOne**m * (1 - _x**2)**Rational(m, 2) * P.as_expr()

    @classmethod
    def eval(cls, n, m, x):
        if m.could_extract_minus_sign():
            # P^{-m}_n  --->  F * P^m_n
            return S.NegativeOne**(-m) * (factorial(m + n)/factorial(n - m)) * assoc_legendre(n, -m, x)
        if m == 0:
            # P^0_n  --->  L_n
            return legendre(n, x)
        if x == 0:
            return 2**m*sqrt(S.Pi) / (gamma((1 - m - n)/2)*gamma(1 - (m - n)/2))
        if n.is_Number and m.is_Number and n.is_integer and m.is_integer:
            if n.is_negative:
                raise ValueError("%s : 1st index must be nonnegative integer (got %r)" % (cls, n))
            if abs(m) > n:
                raise ValueError("%s : abs('2nd index') must be <= '1st index' (got %r, %r)" % (cls, n, m))
            return cls._eval_at_order(int(n), abs(int(m))).subs(_x, x)

    def fdiff(self, argindex=3):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt m
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            # Diff wrt x
            # Find better formula, this is unsuitable for x = 1
            n, m, x = self.args
            return 1/(x**2 - 1)*(x*n*assoc_legendre(n, m, x) - (m + n)*assoc_legendre(n - 1, m, x))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, m, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = factorial(2*n - 2*k)/(2**n*factorial(n - k)*factorial(
            k)*factorial(n - 2*k - m))*S.NegativeOne**k*x**(n - m - 2*k)
        return (1 - x**2)**(m/2) * Sum(kern, (k, 0, floor((n - m)*S.Half)))

    def _eval_rewrite_as_polynomial(self, n, m, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, m, x, **kwargs)

    def _eval_conjugate(self):
        n, m, x = self.args
        return self.func(n, m.conjugate(), x.conjugate())

#----------------------------------------------------------------------------
# Hermite polynomials
#


class hermite(OrthogonalPolynomial):
    r"""
    ``hermite(n, x)`` gives the $n$th Hermite polynomial in $x$, $H_n(x)$.

    Explanation
    ===========

    The Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-x^2\right)$.

    Examples
    ========

    >>> from sympy import hermite, diff
    >>> from sympy.abc import x, n
    >>> hermite(0, x)
    1
    >>> hermite(1, x)
    2*x
    >>> hermite(2, x)
    4*x**2 - 2
    >>> hermite(n, x)
    hermite(n, x)
    >>> diff(hermite(n,x), x)
    2*n*hermite(n - 1, x)
    >>> hermite(n, -x)
    (-1)**n*hermite(n, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/HermiteH/

    """

    _ortho_poly = staticmethod(hermite_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result H_n(x)
            # H_n(-x)  --->  (-1)**n * H_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite(n, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return 2**n * sqrt(S.Pi) / gamma((S.One - n)/2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return 2*n*hermite(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k / (factorial(k)*factorial(n - 2*k)) * (2*x)**(n - 2*k)
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    def _eval_rewrite_as_hermite_prob(self, n, x, **kwargs):
        return sqrt(2)**n * hermite_prob(n, x*sqrt(2))


class hermite_prob(OrthogonalPolynomial):
    r"""
    ``hermite_prob(n, x)`` gives the $n$th probabilist's Hermite polynomial
    in $x$, $He_n(x)$.

    Explanation
    ===========

    The probabilist's Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-\frac{x^2}{2}\right)$. They are monic
    polynomials, related to the plain Hermite polynomials (:py:class:`~.hermite`) by

    .. math :: He_n(x) = 2^{-n/2} H_n(x/\sqrt{2})

    Examples
    ========

    >>> from sympy import hermite_prob, diff, I
    >>> from sympy.abc import x, n
    >>> hermite_prob(1, x)
    x
    >>> hermite_prob(5, x)
    x**5 - 10*x**3 + 15*x
    >>> diff(hermite_prob(n,x), x)
    n*hermite_prob(n - 1, x)
    >>> hermite_prob(n, -x)
    (-1)**n*hermite_prob(n, x)

    The sum of absolute values of coefficients of $He_n(x)$ is the number of
    matchings in the complete graph $K_n$ or telephone number, A000085 in the OEIS:

    >>> [hermite_prob(n,I) / I**n for n in range(11)]
    [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    """

    _ortho_poly = staticmethod(hermite_prob_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite_prob(n, -x)
            if x.is_zero:
                return sqrt(S.Pi) / gamma((S.One-n) / 2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            if n.is_negative:
                ValueError("n must be a nonnegative integer, not %r" % n)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 2:
            n, x = self.args
            return n*hermite_prob(n-1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = (-S.Half)**k * x**(n-2*k) / (factorial(k) * factorial(n-2*k))
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    def _eval_rewrite_as_hermite(self, n, x, **kwargs):
        return sqrt(2)**(-n) * hermite(n, x/sqrt(2))


#----------------------------------------------------------------------------
# Laguerre polynomials
#


class laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import laguerre, diff
    >>> from sympy.abc import x, n
    >>> laguerre(0, x)
    1
    >>> laguerre(1, x)
    1 - x
    >>> laguerre(2, x)
    x**2/2 - 2*x + 1
    >>> laguerre(3, x)
    -x**3/6 + 3*x**2/2 - 3*x + 1

    >>> laguerre(n, x)
    laguerre(n, x)

    >>> diff(laguerre(n, x), x)
    -assoc_laguerre(n - 1, 1, x)

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial
    .. [2] https://mathworld.wolfram.com/LaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    _ortho_poly = staticmethod(laguerre_poly)

    @classmethod
    def eval(cls, n, x):
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_{n}(-x)  --->  exp(-x) * L_{-n-1}(x)
            # L_{-n}(x)  --->  exp(x) * L_{n-1}(-x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return exp(x)*laguerre(-n - 1, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return S.One
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif x is S.Infinity:
                return S.NegativeOne**n * S.Infinity
        else:
            if n.is_negative:
                return exp(x)*laguerre(-n - 1, -x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return -assoc_laguerre(n - 1, 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N_0
        if n.is_negative:
            return exp(x) * self._eval_rewrite_as_Sum(-n - 1, -x, **kwargs)
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        k = Dummy("k")
        kern = RisingFactorial(-n, k) / factorial(k)**2 * x**k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class assoc_laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th generalized Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import assoc_laguerre, diff
    >>> from sympy.abc import x, n, a
    >>> assoc_laguerre(0, a, x)
    1
    >>> assoc_laguerre(1, a, x)
    a - x + 1
    >>> assoc_laguerre(2, a, x)
    a**2/2 + 3*a/2 + x**2/2 + x*(-a - 2) + 1
    >>> assoc_laguerre(3, a, x)
    a**3/6 + a**2 + 11*a/6 - x**3/6 + x**2*(a/2 + 3/2) +
        x*(-a**2/2 - 5*a/2 - 3) + 1

    >>> assoc_laguerre(n, a, 0)
    binomial(a + n, a)

    >>> assoc_laguerre(n, a, x)
    assoc_laguerre(n, a, x)

    >>> assoc_laguerre(n, 0, x)
    laguerre(n, x)

    >>> diff(assoc_laguerre(n, a, x), x)
    -assoc_laguerre(n - 1, a + 1, x)

    >>> diff(assoc_laguerre(n, a, x), a)
    Sum(assoc_laguerre(_k, a, x)/(-a + n), (_k, 0, n - 1))

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    alpha : Expr
        Arbitrary expression. For ``alpha=0`` regular Laguerre
        polynomials will be generated.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial#Generalized_Laguerre_polynomials
    .. [2] https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    @classmethod
    def eval(cls, n, alpha, x):
        # L_{n}^{0}(x)  --->  L_{n}(x)
        if alpha.is_zero:
            return laguerre(n, x)

        if not n.is_Number:
            # We can evaluate for some special values of x
            if x.is_zero:
                return binomial(n + alpha, alpha)
            elif x is S.Infinity and n > 0:
                return S.NegativeOne**n * S.Infinity
            elif x is S.NegativeInfinity and n > 0:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return laguerre_poly(n, x, alpha)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt alpha
            n, alpha, x = self.args
            k = Dummy("k")
            return Sum(assoc_laguerre(k, alpha, x) / (n - alpha), (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt x
            n, alpha, x = self.args
            return -assoc_laguerre(n - 1, alpha + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, alpha, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N_0
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = RisingFactorial(
            -n, k) / (gamma(k + alpha + 1) * factorial(k)) * x**k
        return gamma(n + alpha + 1) / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, alpha, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, alpha, x, **kwargs)

    def _eval_conjugate(self):
        n, alpha, x = self.args
        return self.func(n, alpha.conjugate(), x.conjugate())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\ddl.py ===
# sql/ddl.py
# Copyright (C) 2009-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""
Provides the hierarchy of DDL-defining schema items as well as routines
to invoke them for a create/drop call.

"""
from __future__ import annotations

import contextlib
import typing
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence as typing_Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

from . import roles
from .base import _generative
from .base import Executable
from .base import SchemaVisitor
from .elements import ClauseElement
from .. import exc
from .. import util
from ..util import topological
from ..util.typing import Protocol
from ..util.typing import Self

if typing.TYPE_CHECKING:
    from .compiler import Compiled
    from .compiler import DDLCompiler
    from .elements import BindParameter
    from .schema import Column
    from .schema import Constraint
    from .schema import ForeignKeyConstraint
    from .schema import Index
    from .schema import SchemaItem
    from .schema import Sequence as Sequence  # noqa: F401
    from .schema import Table
    from .selectable import TableClause
    from ..engine.base import Connection
    from ..engine.interfaces import CacheStats
    from ..engine.interfaces import CompiledCacheType
    from ..engine.interfaces import Dialect
    from ..engine.interfaces import SchemaTranslateMapType

_SI = TypeVar("_SI", bound=Union["SchemaItem", str])


class BaseDDLElement(ClauseElement):
    """The root of DDL constructs, including those that are sub-elements
    within the "create table" and other processes.

    .. versionadded:: 2.0

    """

    _hierarchy_supports_caching = False
    """disable cache warnings for all _DDLCompiles subclasses. """

    def _compiler(self, dialect, **kw):
        """Return a compiler appropriate for this ClauseElement, given a
        Dialect."""

        return dialect.ddl_compiler(dialect, self, **kw)

    def _compile_w_cache(
        self,
        dialect: Dialect,
        *,
        compiled_cache: Optional[CompiledCacheType],
        column_keys: List[str],
        for_executemany: bool = False,
        schema_translate_map: Optional[SchemaTranslateMapType] = None,
        **kw: Any,
    ) -> Tuple[
        Compiled, Optional[typing_Sequence[BindParameter[Any]]], CacheStats
    ]:
        raise NotImplementedError()


class DDLIfCallable(Protocol):
    def __call__(
        self,
        ddl: BaseDDLElement,
        target: Union[SchemaItem, str],
        bind: Optional[Connection],
        tables: Optional[List[Table]] = None,
        state: Optional[Any] = None,
        *,
        dialect: Dialect,
        compiler: Optional[DDLCompiler] = ...,
        checkfirst: bool,
    ) -> bool: ...


class DDLIf(typing.NamedTuple):
    dialect: Optional[str]
    callable_: Optional[DDLIfCallable]
    state: Optional[Any]

    def _should_execute(
        self,
        ddl: BaseDDLElement,
        target: Union[SchemaItem, str],
        bind: Optional[Connection],
        compiler: Optional[DDLCompiler] = None,
        **kw: Any,
    ) -> bool:
        if bind is not None:
            dialect = bind.dialect
        elif compiler is not None:
            dialect = compiler.dialect
        else:
            assert False, "compiler or dialect is required"

        if isinstance(self.dialect, str):
            if self.dialect != dialect.name:
                return False
        elif isinstance(self.dialect, (tuple, list, set)):
            if dialect.name not in self.dialect:
                return False
        if self.callable_ is not None and not self.callable_(
            ddl,
            target,
            bind,
            state=self.state,
            dialect=dialect,
            compiler=compiler,
            **kw,
        ):
            return False

        return True


class ExecutableDDLElement(roles.DDLRole, Executable, BaseDDLElement):
    """Base class for standalone executable DDL expression constructs.

    This class is the base for the general purpose :class:`.DDL` class,
    as well as the various create/drop clause constructs such as
    :class:`.CreateTable`, :class:`.DropTable`, :class:`.AddConstraint`,
    etc.

    .. versionchanged:: 2.0  :class:`.ExecutableDDLElement` is renamed from
       :class:`.DDLElement`, which still exists for backwards compatibility.

    :class:`.ExecutableDDLElement` integrates closely with SQLAlchemy events,
    introduced in :ref:`event_toplevel`.  An instance of one is
    itself an event receiving callable::

        event.listen(
            users,
            "after_create",
            AddConstraint(constraint).execute_if(dialect="postgresql"),
        )

    .. seealso::

        :class:`.DDL`

        :class:`.DDLEvents`

        :ref:`event_toplevel`

        :ref:`schema_ddl_sequences`

    """

    _ddl_if: Optional[DDLIf] = None
    target: Union[SchemaItem, str, None] = None

    def _execute_on_connection(
        self, connection, distilled_params, execution_options
    ):
        return connection._execute_ddl(
            self, distilled_params, execution_options
        )

    @_generative
    def against(self, target: SchemaItem) -> Self:
        """Return a copy of this :class:`_schema.ExecutableDDLElement` which
        will include the given target.

        This essentially applies the given item to the ``.target`` attribute of
        the returned :class:`_schema.ExecutableDDLElement` object. This target
        is then usable by event handlers and compilation routines in order to
        provide services such as tokenization of a DDL string in terms of a
        particular :class:`_schema.Table`.

        When a :class:`_schema.ExecutableDDLElement` object is established as
        an event handler for the :meth:`_events.DDLEvents.before_create` or
        :meth:`_events.DDLEvents.after_create` events, and the event then
        occurs for a given target such as a :class:`_schema.Constraint` or
        :class:`_schema.Table`, that target is established with a copy of the
        :class:`_schema.ExecutableDDLElement` object using this method, which
        then proceeds to the :meth:`_schema.ExecutableDDLElement.execute`
        method in order to invoke the actual DDL instruction.

        :param target: a :class:`_schema.SchemaItem` that will be the subject
         of a DDL operation.

        :return: a copy of this :class:`_schema.ExecutableDDLElement` with the
         ``.target`` attribute assigned to the given
         :class:`_schema.SchemaItem`.

        .. seealso::

            :class:`_schema.DDL` - uses tokenization against the "target" when
            processing the DDL string.

        """
        self.target = target
        return self

    @_generative
    def execute_if(
        self,
        dialect: Optional[str] = None,
        callable_: Optional[DDLIfCallable] = None,
        state: Optional[Any] = None,
    ) -> Self:
        r"""Return a callable that will execute this
        :class:`_ddl.ExecutableDDLElement` conditionally within an event
        handler.

        Used to provide a wrapper for event listening::

            event.listen(
                metadata,
                "before_create",
                DDL("my_ddl").execute_if(dialect="postgresql"),
            )

        :param dialect: May be a string or tuple of strings.
          If a string, it will be compared to the name of the
          executing database dialect::

            DDL("something").execute_if(dialect="postgresql")

          If a tuple, specifies multiple dialect names::

            DDL("something").execute_if(dialect=("postgresql", "mysql"))

        :param callable\_: A callable, which will be invoked with
          three positional arguments as well as optional keyword
          arguments:

            :ddl:
              This DDL element.

            :target:
              The :class:`_schema.Table` or :class:`_schema.MetaData`
              object which is the
              target of this event. May be None if the DDL is executed
              explicitly.

            :bind:
              The :class:`_engine.Connection` being used for DDL execution.
              May be None if this construct is being created inline within
              a table, in which case ``compiler`` will be present.

            :tables:
              Optional keyword argument - a list of Table objects which are to
              be created/ dropped within a MetaData.create_all() or drop_all()
              method call.

            :dialect: keyword argument, but always present - the
              :class:`.Dialect` involved in the operation.

            :compiler: keyword argument.  Will be ``None`` for an engine
              level DDL invocation, but will refer to a :class:`.DDLCompiler`
              if this DDL element is being created inline within a table.

            :state:
              Optional keyword argument - will be the ``state`` argument
              passed to this function.

            :checkfirst:
             Keyword argument, will be True if the 'checkfirst' flag was
             set during the call to ``create()``, ``create_all()``,
             ``drop()``, ``drop_all()``.

          If the callable returns a True value, the DDL statement will be
          executed.

        :param state: any value which will be passed to the callable\_
          as the ``state`` keyword argument.

        .. seealso::

            :meth:`.SchemaItem.ddl_if`

            :class:`.DDLEvents`

            :ref:`event_toplevel`

        """
        self._ddl_if = DDLIf(dialect, callable_, state)
        return self

    def _should_execute(self, target, bind, **kw):
        if self._ddl_if is None:
            return True
        else:
            return self._ddl_if._should_execute(self, target, bind, **kw)

    def _invoke_with(self, bind):
        if self._should_execute(self.target, bind):
            return bind.execute(self)

    def __call__(self, target, bind, **kw):
        """Execute the DDL as a ddl_listener."""

        self.against(target)._invoke_with(bind)

    def _generate(self):
        s = self.__class__.__new__(self.__class__)
        s.__dict__ = self.__dict__.copy()
        return s


DDLElement = ExecutableDDLElement
""":class:`.DDLElement` is renamed to :class:`.ExecutableDDLElement`."""


class DDL(ExecutableDDLElement):
    """A literal DDL statement.

    Specifies literal SQL DDL to be executed by the database.  DDL objects
    function as DDL event listeners, and can be subscribed to those events
    listed in :class:`.DDLEvents`, using either :class:`_schema.Table` or
    :class:`_schema.MetaData` objects as targets.
    Basic templating support allows
    a single DDL instance to handle repetitive tasks for multiple tables.

    Examples::

      from sqlalchemy import event, DDL

      tbl = Table("users", metadata, Column("uid", Integer))
      event.listen(tbl, "before_create", DDL("DROP TRIGGER users_trigger"))

      spow = DDL("ALTER TABLE %(table)s SET secretpowers TRUE")
      event.listen(tbl, "after_create", spow.execute_if(dialect="somedb"))

      drop_spow = DDL("ALTER TABLE users SET secretpowers FALSE")
      connection.execute(drop_spow)

    When operating on Table events, the following ``statement``
    string substitutions are available:

    .. sourcecode:: text

      %(table)s  - the Table name, with any required quoting applied
      %(schema)s - the schema name, with any required quoting applied
      %(fullname)s - the Table name including schema, quoted if needed

    The DDL's "context", if any, will be combined with the standard
    substitutions noted above.  Keys present in the context will override
    the standard substitutions.

    """

    __visit_name__ = "ddl"

    def __init__(self, statement, context=None):
        """Create a DDL statement.

        :param statement:
          A string or unicode string to be executed.  Statements will be
          processed with Python's string formatting operator using
          a fixed set of string substitutions, as well as additional
          substitutions provided by the optional :paramref:`.DDL.context`
          parameter.

          A literal '%' in a statement must be escaped as '%%'.

          SQL bind parameters are not available in DDL statements.

        :param context:
          Optional dictionary, defaults to None.  These values will be
          available for use in string substitutions on the DDL statement.

        .. seealso::

            :class:`.DDLEvents`

            :ref:`event_toplevel`

        """

        if not isinstance(statement, str):
            raise exc.ArgumentError(
                "Expected a string or unicode SQL statement, got '%r'"
                % statement
            )

        self.statement = statement
        self.context = context or {}

    def __repr__(self):
        parts = [repr(self.statement)]
        if self.context:
            parts.append(f"context={self.context}")

        return "<%s@%s; %s>" % (
            type(self).__name__,
            id(self),
            ", ".join(parts),
        )


class _CreateDropBase(ExecutableDDLElement, Generic[_SI]):
    """Base class for DDL constructs that represent CREATE and DROP or
    equivalents.

    The common theme of _CreateDropBase is a single
    ``element`` attribute which refers to the element
    to be created or dropped.

    """

    def __init__(self, element: _SI) -> None:
        self.element = self.target = element
        self._ddl_if = getattr(element, "_ddl_if", None)

    @property
    def stringify_dialect(self):
        assert not isinstance(self.element, str)
        return self.element.create_drop_stringify_dialect

    def _create_rule_disable(self, compiler):
        """Allow disable of _create_rule using a callable.

        Pass to _create_rule using
        util.portable_instancemethod(self._create_rule_disable)
        to retain serializability.

        """
        return False


class _CreateBase(_CreateDropBase[_SI]):
    def __init__(self, element: _SI, if_not_exists: bool = False) -> None:
        super().__init__(element)
        self.if_not_exists = if_not_exists


class _DropBase(_CreateDropBase[_SI]):
    def __init__(self, element: _SI, if_exists: bool = False) -> None:
        super().__init__(element)
        self.if_exists = if_exists


class CreateSchema(_CreateBase[str]):
    """Represent a CREATE SCHEMA statement.

    The argument here is the string name of the schema.

    """

    __visit_name__ = "create_schema"

    stringify_dialect = "default"

    def __init__(
        self,
        name: str,
        if_not_exists: bool = False,
    ) -> None:
        """Create a new :class:`.CreateSchema` construct."""

        super().__init__(element=name, if_not_exists=if_not_exists)


class DropSchema(_DropBase[str]):
    """Represent a DROP SCHEMA statement.

    The argument here is the string name of the schema.

    """

    __visit_name__ = "drop_schema"

    stringify_dialect = "default"

    def __init__(
        self,
        name: str,
        cascade: bool = False,
        if_exists: bool = False,
    ) -> None:
        """Create a new :class:`.DropSchema` construct."""

        super().__init__(element=name, if_exists=if_exists)
        self.cascade = cascade


class CreateTable(_CreateBase["Table"]):
    """Represent a CREATE TABLE statement."""

    __visit_name__ = "create_table"

    def __init__(
        self,
        element: Table,
        include_foreign_key_constraints: Optional[
            typing_Sequence[ForeignKeyConstraint]
        ] = None,
        if_not_exists: bool = False,
    ) -> None:
        """Create a :class:`.CreateTable` construct.

        :param element: a :class:`_schema.Table` that's the subject
         of the CREATE
        :param on: See the description for 'on' in :class:`.DDL`.
        :param include_foreign_key_constraints: optional sequence of
         :class:`_schema.ForeignKeyConstraint` objects that will be included
         inline within the CREATE construct; if omitted, all foreign key
         constraints that do not specify use_alter=True are included.

        :param if_not_exists: if True, an IF NOT EXISTS operator will be
         applied to the construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_not_exists=if_not_exists)
        self.columns = [CreateColumn(column) for column in element.columns]
        self.include_foreign_key_constraints = include_foreign_key_constraints


class _DropView(_DropBase["Table"]):
    """Semi-public 'DROP VIEW' construct.

    Used by the test suite for dialect-agnostic drops of views.
    This object will eventually be part of a public "view" API.

    """

    __visit_name__ = "drop_view"


class CreateConstraint(BaseDDLElement):
    element: Constraint

    def __init__(self, element: Constraint) -> None:
        self.element = element


class CreateColumn(BaseDDLElement):
    """Represent a :class:`_schema.Column`
    as rendered in a CREATE TABLE statement,
    via the :class:`.CreateTable` construct.

    This is provided to support custom column DDL within the generation
    of CREATE TABLE statements, by using the
    compiler extension documented in :ref:`sqlalchemy.ext.compiler_toplevel`
    to extend :class:`.CreateColumn`.

    Typical integration is to examine the incoming :class:`_schema.Column`
    object, and to redirect compilation if a particular flag or condition
    is found::

        from sqlalchemy import schema
        from sqlalchemy.ext.compiler import compiles


        @compiles(schema.CreateColumn)
        def compile(element, compiler, **kw):
            column = element.element

            if "special" not in column.info:
                return compiler.visit_create_column(element, **kw)

            text = "%s SPECIAL DIRECTIVE %s" % (
                column.name,
                compiler.type_compiler.process(column.type),
            )
            default = compiler.get_column_default_string(column)
            if default is not None:
                text += " DEFAULT " + default

            if not column.nullable:
                text += " NOT NULL"

            if column.constraints:
                text += " ".join(
                    compiler.process(const) for const in column.constraints
                )
            return text

    The above construct can be applied to a :class:`_schema.Table`
    as follows::

        from sqlalchemy import Table, Metadata, Column, Integer, String
        from sqlalchemy import schema

        metadata = MetaData()

        table = Table(
            "mytable",
            MetaData(),
            Column("x", Integer, info={"special": True}, primary_key=True),
            Column("y", String(50)),
            Column("z", String(20), info={"special": True}),
        )

        metadata.create_all(conn)

    Above, the directives we've added to the :attr:`_schema.Column.info`
    collection
    will be detected by our custom compilation scheme:

    .. sourcecode:: sql

        CREATE TABLE mytable (
                x SPECIAL DIRECTIVE INTEGER NOT NULL,
                y VARCHAR(50),
                z SPECIAL DIRECTIVE VARCHAR(20),
            PRIMARY KEY (x)
        )

    The :class:`.CreateColumn` construct can also be used to skip certain
    columns when producing a ``CREATE TABLE``.  This is accomplished by
    creating a compilation rule that conditionally returns ``None``.
    This is essentially how to produce the same effect as using the
    ``system=True`` argument on :class:`_schema.Column`, which marks a column
    as an implicitly-present "system" column.

    For example, suppose we wish to produce a :class:`_schema.Table`
    which skips
    rendering of the PostgreSQL ``xmin`` column against the PostgreSQL
    backend, but on other backends does render it, in anticipation of a
    triggered rule.  A conditional compilation rule could skip this name only
    on PostgreSQL::

        from sqlalchemy.schema import CreateColumn


        @compiles(CreateColumn, "postgresql")
        def skip_xmin(element, compiler, **kw):
            if element.element.name == "xmin":
                return None
            else:
                return compiler.visit_create_column(element, **kw)


        my_table = Table(
            "mytable",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("xmin", Integer),
        )

    Above, a :class:`.CreateTable` construct will generate a ``CREATE TABLE``
    which only includes the ``id`` column in the string; the ``xmin`` column
    will be omitted, but only against the PostgreSQL backend.

    """

    __visit_name__ = "create_column"

    element: Column[Any]

    def __init__(self, element: Column[Any]) -> None:
        self.element = element


class DropTable(_DropBase["Table"]):
    """Represent a DROP TABLE statement."""

    __visit_name__ = "drop_table"

    def __init__(self, element: Table, if_exists: bool = False) -> None:
        """Create a :class:`.DropTable` construct.

        :param element: a :class:`_schema.Table` that's the subject
         of the DROP.
        :param on: See the description for 'on' in :class:`.DDL`.
        :param if_exists: if True, an IF EXISTS operator will be applied to the
         construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_exists=if_exists)


class CreateSequence(_CreateBase["Sequence"]):
    """Represent a CREATE SEQUENCE statement."""

    __visit_name__ = "create_sequence"


class DropSequence(_DropBase["Sequence"]):
    """Represent a DROP SEQUENCE statement."""

    __visit_name__ = "drop_sequence"


class CreateIndex(_CreateBase["Index"]):
    """Represent a CREATE INDEX statement."""

    __visit_name__ = "create_index"

    def __init__(self, element: Index, if_not_exists: bool = False) -> None:
        """Create a :class:`.Createindex` construct.

        :param element: a :class:`_schema.Index` that's the subject
         of the CREATE.
        :param if_not_exists: if True, an IF NOT EXISTS operator will be
         applied to the construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_not_exists=if_not_exists)


class DropIndex(_DropBase["Index"]):
    """Represent a DROP INDEX statement."""

    __visit_name__ = "drop_index"

    def __init__(self, element: Index, if_exists: bool = False) -> None:
        """Create a :class:`.DropIndex` construct.

        :param element: a :class:`_schema.Index` that's the subject
         of the DROP.
        :param if_exists: if True, an IF EXISTS operator will be applied to the
         construct.

         .. versionadded:: 1.4.0b2

        """
        super().__init__(element, if_exists=if_exists)


class AddConstraint(_CreateBase["Constraint"]):
    """Represent an ALTER TABLE ADD CONSTRAINT statement."""

    __visit_name__ = "add_constraint"

    def __init__(
        self,
        element: Constraint,
        *,
        isolate_from_table: bool = True,
    ) -> None:
        """Construct a new :class:`.AddConstraint` construct.

        :param element: a :class:`.Constraint` object

        :param isolate_from_table: optional boolean, defaults to True.  Has
         the effect of the incoming constraint being isolated from being
         included in a CREATE TABLE sequence when associated with a
         :class:`.Table`.

         .. versionadded:: 2.0.39 - added
            :paramref:`.AddConstraint.isolate_from_table`, defaulting
            to True.  Previously, the behavior of this parameter was implicitly
            turned on in all cases.

        """
        super().__init__(element)

        if isolate_from_table:
            element._create_rule = util.portable_instancemethod(
                self._create_rule_disable
            )


class DropConstraint(_DropBase["Constraint"]):
    """Represent an ALTER TABLE DROP CONSTRAINT statement."""

    __visit_name__ = "drop_constraint"

    def __init__(
        self,
        element: Constraint,
        *,
        cascade: bool = False,
        if_exists: bool = False,
        isolate_from_table: bool = True,
        **kw: Any,
    ) -> None:
        """Construct a new :class:`.DropConstraint` construct.

        :param element: a :class:`.Constraint` object
        :param cascade: optional boolean, indicates backend-specific
         "CASCADE CONSTRAINT" directive should be rendered if available
        :param if_exists: optional boolean, indicates backend-specific
         "IF EXISTS" directive should be rendered if available
        :param isolate_from_table: optional boolean, defaults to True.  Has
         the effect of the incoming constraint being isolated from being
         included in a CREATE TABLE sequence when associated with a
         :class:`.Table`.

         .. versionadded:: 2.0.39 - added
            :paramref:`.DropConstraint.isolate_from_table`, defaulting
            to True.  Previously, the behavior of this parameter was implicitly
            turned on in all cases.

        """
        self.cascade = cascade
        super().__init__(element, if_exists=if_exists, **kw)

        if isolate_from_table:
            element._create_rule = util.portable_instancemethod(
                self._create_rule_disable
            )


class SetTableComment(_CreateDropBase["Table"]):
    """Represent a COMMENT ON TABLE IS statement."""

    __visit_name__ = "set_table_comment"


class DropTableComment(_CreateDropBase["Table"]):
    """Represent a COMMENT ON TABLE '' statement.

    Note this varies a lot across database backends.

    """

    __visit_name__ = "drop_table_comment"


class SetColumnComment(_CreateDropBase["Column[Any]"]):
    """Represent a COMMENT ON COLUMN IS statement."""

    __visit_name__ = "set_column_comment"


class DropColumnComment(_CreateDropBase["Column[Any]"]):
    """Represent a COMMENT ON COLUMN IS NULL statement."""

    __visit_name__ = "drop_column_comment"


class SetConstraintComment(_CreateDropBase["Constraint"]):
    """Represent a COMMENT ON CONSTRAINT IS statement."""

    __visit_name__ = "set_constraint_comment"


class DropConstraintComment(_CreateDropBase["Constraint"]):
    """Represent a COMMENT ON CONSTRAINT IS NULL statement."""

    __visit_name__ = "drop_constraint_comment"


class InvokeDDLBase(SchemaVisitor):
    def __init__(self, connection, **kw):
        self.connection = connection
        assert not kw, f"Unexpected keywords: {kw.keys()}"

    @contextlib.contextmanager
    def with_ddl_events(self, target, **kw):
        """helper context manager that will apply appropriate DDL events
        to a CREATE or DROP operation."""

        raise NotImplementedError()


class InvokeCreateDDLBase(InvokeDDLBase):
    @contextlib.contextmanager
    def with_ddl_events(self, target, **kw):
        """helper context manager that will apply appropriate DDL events
        to a CREATE or DROP operation."""

        target.dispatch.before_create(
            target, self.connection, _ddl_runner=self, **kw
        )
        yield
        target.dispatch.after_create(
            target, self.connection, _ddl_runner=self, **kw
        )


class InvokeDropDDLBase(InvokeDDLBase):
    @contextlib.contextmanager
    def with_ddl_events(self, target, **kw):
        """helper context manager that will apply appropriate DDL events
        to a CREATE or DROP operation."""

        target.dispatch.before_drop(
            target, self.connection, _ddl_runner=self, **kw
        )
        yield
        target.dispatch.after_drop(
            target, self.connection, _ddl_runner=self, **kw
        )


class SchemaGenerator(InvokeCreateDDLBase):
    def __init__(
        self, dialect, connection, checkfirst=False, tables=None, **kwargs
    ):
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst
        self.tables = tables
        self.preparer = dialect.identifier_preparer
        self.dialect = dialect
        self.memo = {}

    def _can_create_table(self, table):
        self.dialect.validate_identifier(table.name)
        effective_schema = self.connection.schema_for_object(table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or not self.dialect.has_table(
            self.connection, table.name, schema=effective_schema
        )

    def _can_create_index(self, index):
        effective_schema = self.connection.schema_for_object(index.table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or not self.dialect.has_index(
            self.connection,
            index.table.name,
            index.name,
            schema=effective_schema,
        )

    def _can_create_sequence(self, sequence):
        effective_schema = self.connection.schema_for_object(sequence)

        return self.dialect.supports_sequences and (
            (not self.dialect.sequences_optional or not sequence.optional)
            and (
                not self.checkfirst
                or not self.dialect.has_sequence(
                    self.connection, sequence.name, schema=effective_schema
                )
            )
        )

    def visit_metadata(self, metadata):
        if self.tables is not None:
            tables = self.tables
        else:
            tables = list(metadata.tables.values())

        collection = sort_tables_and_constraints(
            [t for t in tables if self._can_create_table(t)]
        )

        seq_coll = [
            s
            for s in metadata._sequences.values()
            if s.column is None and self._can_create_sequence(s)
        ]

        event_collection = [t for (t, fks) in collection if t is not None]

        with self.with_ddl_events(
            metadata,
            tables=event_collection,
            checkfirst=self.checkfirst,
        ):
            for seq in seq_coll:
                self.traverse_single(seq, create_ok=True)

            for table, fkcs in collection:
                if table is not None:
                    self.traverse_single(
                        table,
                        create_ok=True,
                        include_foreign_key_constraints=fkcs,
                        _is_metadata_operation=True,
                    )
                else:
                    for fkc in fkcs:
                        self.traverse_single(fkc)

    def visit_table(
        self,
        table,
        create_ok=False,
        include_foreign_key_constraints=None,
        _is_metadata_operation=False,
    ):
        if not create_ok and not self._can_create_table(table):
            return

        with self.with_ddl_events(
            table,
            checkfirst=self.checkfirst,
            _is_metadata_operation=_is_metadata_operation,
        ):
            for column in table.columns:
                if column.default is not None:
                    self.traverse_single(column.default)

            if not self.dialect.supports_alter:
                # e.g., don't omit any foreign key constraints
                include_foreign_key_constraints = None

            CreateTable(
                table,
                include_foreign_key_constraints=(
                    include_foreign_key_constraints
                ),
            )._invoke_with(self.connection)

            if hasattr(table, "indexes"):
                for index in table.indexes:
                    self.traverse_single(index, create_ok=True)

            if (
                self.dialect.supports_comments
                and not self.dialect.inline_comments
            ):
                if table.comment is not None:
                    SetTableComment(table)._invoke_with(self.connection)

                for column in table.columns:
                    if column.comment is not None:
                        SetColumnComment(column)._invoke_with(self.connection)

                if self.dialect.supports_constraint_comments:
                    for constraint in table.constraints:
                        if constraint.comment is not None:
                            self.connection.execute(
                                SetConstraintComment(constraint)
                            )

    def visit_foreign_key_constraint(self, constraint):
        if not self.dialect.supports_alter:
            return

        with self.with_ddl_events(constraint):
            AddConstraint(constraint)._invoke_with(self.connection)

    def visit_sequence(self, sequence, create_ok=False):
        if not create_ok and not self._can_create_sequence(sequence):
            return
        with self.with_ddl_events(sequence):
            CreateSequence(sequence)._invoke_with(self.connection)

    def visit_index(self, index, create_ok=False):
        if not create_ok and not self._can_create_index(index):
            return
        with self.with_ddl_events(index):
            CreateIndex(index)._invoke_with(self.connection)


class SchemaDropper(InvokeDropDDLBase):
    def __init__(
        self, dialect, connection, checkfirst=False, tables=None, **kwargs
    ):
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst
        self.tables = tables
        self.preparer = dialect.identifier_preparer
        self.dialect = dialect
        self.memo = {}

    def visit_metadata(self, metadata):
        if self.tables is not None:
            tables = self.tables
        else:
            tables = list(metadata.tables.values())

        try:
            unsorted_tables = [t for t in tables if self._can_drop_table(t)]
            collection = list(
                reversed(
                    sort_tables_and_constraints(
                        unsorted_tables,
                        filter_fn=lambda constraint: (
                            False
                            if not self.dialect.supports_alter
                            or constraint.name is None
                            else None
                        ),
                    )
                )
            )
        except exc.CircularDependencyError as err2:
            if not self.dialect.supports_alter:
                util.warn(
                    "Can't sort tables for DROP; an "
                    "unresolvable foreign key "
                    "dependency exists between tables: %s; and backend does "
                    "not support ALTER.  To restore at least a partial sort, "
                    "apply use_alter=True to ForeignKey and "
                    "ForeignKeyConstraint "
                    "objects involved in the cycle to mark these as known "
                    "cycles that will be ignored."
                    % (", ".join(sorted([t.fullname for t in err2.cycles])))
                )
                collection = [(t, ()) for t in unsorted_tables]
            else:
                raise exc.CircularDependencyError(
                    err2.args[0],
                    err2.cycles,
                    err2.edges,
                    msg="Can't sort tables for DROP; an "
                    "unresolvable foreign key "
                    "dependency exists between tables: %s.  Please ensure "
                    "that the ForeignKey and ForeignKeyConstraint objects "
                    "involved in the cycle have "
                    "names so that they can be dropped using "
                    "DROP CONSTRAINT."
                    % (", ".join(sorted([t.fullname for t in err2.cycles]))),
                ) from err2

        seq_coll = [
            s
            for s in metadata._sequences.values()
            if self._can_drop_sequence(s)
        ]

        event_collection = [t for (t, fks) in collection if t is not None]

        with self.with_ddl_events(
            metadata,
            tables=event_collection,
            checkfirst=self.checkfirst,
        ):
            for table, fkcs in collection:
                if table is not None:
                    self.traverse_single(
                        table,
                        drop_ok=True,
                        _is_metadata_operation=True,
                        _ignore_sequences=seq_coll,
                    )
                else:
                    for fkc in fkcs:
                        self.traverse_single(fkc)

            for seq in seq_coll:
                self.traverse_single(seq, drop_ok=seq.column is None)

    def _can_drop_table(self, table):
        self.dialect.validate_identifier(table.name)
        effective_schema = self.connection.schema_for_object(table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or self.dialect.has_table(
            self.connection, table.name, schema=effective_schema
        )

    def _can_drop_index(self, index):
        effective_schema = self.connection.schema_for_object(index.table)
        if effective_schema:
            self.dialect.validate_identifier(effective_schema)
        return not self.checkfirst or self.dialect.has_index(
            self.connection,
            index.table.name,
            index.name,
            schema=effective_schema,
        )

    def _can_drop_sequence(self, sequence):
        effective_schema = self.connection.schema_for_object(sequence)
        return self.dialect.supports_sequences and (
            (not self.dialect.sequences_optional or not sequence.optional)
            and (
                not self.checkfirst
                or self.dialect.has_sequence(
                    self.connection, sequence.name, schema=effective_schema
                )
            )
        )

    def visit_index(self, index, drop_ok=False):
        if not drop_ok and not self._can_drop_index(index):
            return

        with self.with_ddl_events(index):
            DropIndex(index)(index, self.connection)

    def visit_table(
        self,
        table,
        drop_ok=False,
        _is_metadata_operation=False,
        _ignore_sequences=(),
    ):
        if not drop_ok and not self._can_drop_table(table):
            return

        with self.with_ddl_events(
            table,
            checkfirst=self.checkfirst,
            _is_metadata_operation=_is_metadata_operation,
        ):
            DropTable(table)._invoke_with(self.connection)

            # traverse client side defaults which may refer to server-side
            # sequences. noting that some of these client side defaults may
            # also be set up as server side defaults
            # (see https://docs.sqlalchemy.org/en/
            # latest/core/defaults.html
            # #associating-a-sequence-as-the-server-side-
            # default), so have to be dropped after the table is dropped.
            for column in table.columns:
                if (
                    column.default is not None
                    and column.default not in _ignore_sequences
                ):
                    self.traverse_single(column.default)

    def visit_foreign_key_constraint(self, constraint):
        if not self.dialect.supports_alter:
            return
        with self.with_ddl_events(constraint):
            DropConstraint(constraint)._invoke_with(self.connection)

    def visit_sequence(self, sequence, drop_ok=False):
        if not drop_ok and not self._can_drop_sequence(sequence):
            return
        with self.with_ddl_events(sequence):
            DropSequence(sequence)._invoke_with(self.connection)


def sort_tables(
    tables: Iterable[TableClause],
    skip_fn: Optional[Callable[[ForeignKeyConstraint], bool]] = None,
    extra_dependencies: Optional[
        typing_Sequence[Tuple[TableClause, TableClause]]
    ] = None,
) -> List[Table]:
    """Sort a collection of :class:`_schema.Table` objects based on
    dependency.

    This is a dependency-ordered sort which will emit :class:`_schema.Table`
    objects such that they will follow their dependent :class:`_schema.Table`
    objects.
    Tables are dependent on another based on the presence of
    :class:`_schema.ForeignKeyConstraint`
    objects as well as explicit dependencies
    added by :meth:`_schema.Table.add_is_dependent_on`.

    .. warning::

        The :func:`._schema.sort_tables` function cannot by itself
        accommodate automatic resolution of dependency cycles between
        tables, which are usually caused by mutually dependent foreign key
        constraints. When these cycles are detected, the foreign keys
        of these tables are omitted from consideration in the sort.
        A warning is emitted when this condition occurs, which will be an
        exception raise in a future release.   Tables which are not part
        of the cycle will still be returned in dependency order.

        To resolve these cycles, the
        :paramref:`_schema.ForeignKeyConstraint.use_alter` parameter may be
        applied to those constraints which create a cycle.  Alternatively,
        the :func:`_schema.sort_tables_and_constraints` function will
        automatically return foreign key constraints in a separate
        collection when cycles are detected so that they may be applied
        to a schema separately.

        .. versionchanged:: 1.3.17 - a warning is emitted when
           :func:`_schema.sort_tables` cannot perform a proper sort due to
           cyclical dependencies.  This will be an exception in a future
           release.  Additionally, the sort will continue to return
           other tables not involved in the cycle in dependency order
           which was not the case previously.

    :param tables: a sequence of :class:`_schema.Table` objects.

    :param skip_fn: optional callable which will be passed a
     :class:`_schema.ForeignKeyConstraint` object; if it returns True, this
     constraint will not be considered as a dependency.  Note this is
     **different** from the same parameter in
     :func:`.sort_tables_and_constraints`, which is
     instead passed the owning :class:`_schema.ForeignKeyConstraint` object.

    :param extra_dependencies: a sequence of 2-tuples of tables which will
     also be considered as dependent on each other.

    .. seealso::

        :func:`.sort_tables_and_constraints`

        :attr:`_schema.MetaData.sorted_tables` - uses this function to sort


    """

    if skip_fn is not None:
        fixed_skip_fn = skip_fn

        def _skip_fn(fkc):
            for fk in fkc.elements:
                if fixed_skip_fn(fk):
                    return True
            else:
                return None

    else:
        _skip_fn = None  # type: ignore

    return [
        t
        for (t, fkcs) in sort_tables_and_constraints(
            tables,
            filter_fn=_skip_fn,
            extra_dependencies=extra_dependencies,
            _warn_for_cycles=True,
        )
        if t is not None
    ]


def sort_tables_and_constraints(
    tables, filter_fn=None, extra_dependencies=None, _warn_for_cycles=False
):
    """Sort a collection of :class:`_schema.Table`  /
    :class:`_schema.ForeignKeyConstraint`
    objects.

    This is a dependency-ordered sort which will emit tuples of
    ``(Table, [ForeignKeyConstraint, ...])`` such that each
    :class:`_schema.Table` follows its dependent :class:`_schema.Table`
    objects.
    Remaining :class:`_schema.ForeignKeyConstraint`
    objects that are separate due to
    dependency rules not satisfied by the sort are emitted afterwards
    as ``(None, [ForeignKeyConstraint ...])``.

    Tables are dependent on another based on the presence of
    :class:`_schema.ForeignKeyConstraint` objects, explicit dependencies
    added by :meth:`_schema.Table.add_is_dependent_on`,
    as well as dependencies
    stated here using the :paramref:`~.sort_tables_and_constraints.skip_fn`
    and/or :paramref:`~.sort_tables_and_constraints.extra_dependencies`
    parameters.

    :param tables: a sequence of :class:`_schema.Table` objects.

    :param filter_fn: optional callable which will be passed a
     :class:`_schema.ForeignKeyConstraint` object,
     and returns a value based on
     whether this constraint should definitely be included or excluded as
     an inline constraint, or neither.   If it returns False, the constraint
     will definitely be included as a dependency that cannot be subject
     to ALTER; if True, it will **only** be included as an ALTER result at
     the end.   Returning None means the constraint is included in the
     table-based result unless it is detected as part of a dependency cycle.

    :param extra_dependencies: a sequence of 2-tuples of tables which will
     also be considered as dependent on each other.

    .. seealso::

        :func:`.sort_tables`


    """

    fixed_dependencies = set()
    mutable_dependencies = set()

    if extra_dependencies is not None:
        fixed_dependencies.update(extra_dependencies)

    remaining_fkcs = set()
    for table in tables:
        for fkc in table.foreign_key_constraints:
            if fkc.use_alter is True:
                remaining_fkcs.add(fkc)
                continue

            if filter_fn:
                filtered = filter_fn(fkc)

                if filtered is True:
                    remaining_fkcs.add(fkc)
                    continue

            dependent_on = fkc.referred_table
            if dependent_on is not table:
                mutable_dependencies.add((dependent_on, table))

        fixed_dependencies.update(
            (parent, table) for parent in table._extra_dependencies
        )

    try:
        candidate_sort = list(
            topological.sort(
                fixed_dependencies.union(mutable_dependencies),
                tables,
            )
        )
    except exc.CircularDependencyError as err:
        if _warn_for_cycles:
            util.warn(
                "Cannot correctly sort tables; there are unresolvable cycles "
                'between tables "%s", which is usually caused by mutually '
                "dependent foreign key constraints.  Foreign key constraints "
                "involving these tables will not be considered; this warning "
                "may raise an error in a future release."
                % (", ".join(sorted(t.fullname for t in err.cycles)),)
            )
        for edge in err.edges:
            if edge in mutable_dependencies:
                table = edge[1]
                if table not in err.cycles:
                    continue
                can_remove = [
                    fkc
                    for fkc in table.foreign_key_constraints
                    if filter_fn is None or filter_fn(fkc) is not False
                ]
                remaining_fkcs.update(can_remove)
                for fkc in can_remove:
                    dependent_on = fkc.referred_table
                    if dependent_on is not table:
                        mutable_dependencies.discard((dependent_on, table))
        candidate_sort = list(
            topological.sort(
                fixed_dependencies.union(mutable_dependencies),
                tables,
            )
        )

    return [
        (table, table.foreign_key_constraints.difference(remaining_fkcs))
        for table in candidate_sort
    ] + [(None, list(remaining_fkcs))]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\lambdas.py ===
# sql/lambdas.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal


if TYPE_CHECKING:
    from .elements import BindParameter
    from .elements import ClauseElement
    from .roles import SQLRole
    from .visitors import _CloneCallableType

_LambdaCacheType = MutableMapping[
    Tuple[Any, ...], Union["NonAnalyzedFunction", "AnalyzedFunction"]
]
_BoundParameterGetter = Callable[..., Any]

_closure_per_cache_key: _LambdaCacheType = util.LRUCache(1000)


_LambdaType = Callable[[], Any]

_AnyLambdaType = Callable[..., Any]

_StmtLambdaType = Callable[[], Any]

_E = TypeVar("_E", bound=Executable)
_StmtLambdaElementType = Callable[[_E], Any]


class LambdaOptions(Options):
    enable_tracking = True
    track_closure_variables = True
    track_on: Optional[object] = None
    global_track_bound_values = True
    track_bound_values = True
    lambda_cache: Optional[_LambdaCacheType] = None


def lambda_stmt(
    lmb: _StmtLambdaType,
    enable_tracking: bool = True,
    track_closure_variables: bool = True,
    track_on: Optional[object] = None,
    global_track_bound_values: bool = True,
    track_bound_values: bool = True,
    lambda_cache: Optional[_LambdaCacheType] = None,
) -> StatementLambdaElement:
    """Produce a SQL statement that is cached as a lambda.

    The Python code object within the lambda is scanned for both Python
    literals that will become bound parameters as well as closure variables
    that refer to Core or ORM constructs that may vary.   The lambda itself
    will be invoked only once per particular set of constructs detected.

    E.g.::

        from sqlalchemy import lambda_stmt

        stmt = lambda_stmt(lambda: table.select())
        stmt += lambda s: s.where(table.c.id == 5)

        result = connection.execute(stmt)

    The object returned is an instance of :class:`_sql.StatementLambdaElement`.

    .. versionadded:: 1.4

    :param lmb: a Python function, typically a lambda, which takes no arguments
     and returns a SQL expression construct
    :param enable_tracking: when False, all scanning of the given lambda for
     changes in closure variables or bound parameters is disabled.  Use for
     a lambda that produces the identical results in all cases with no
     parameterization.
    :param track_closure_variables: when False, changes in closure variables
     within the lambda will not be scanned.   Use for a lambda where the
     state of its closure variables will never change the SQL structure
     returned by the lambda.
    :param track_bound_values: when False, bound parameter tracking will
     be disabled for the given lambda.  Use for a lambda that either does
     not produce any bound values, or where the initial bound values never
     change.
    :param global_track_bound_values: when False, bound parameter tracking
     will be disabled for the entire statement including additional links
     added via the :meth:`_sql.StatementLambdaElement.add_criteria` method.
    :param lambda_cache: a dictionary or other mapping-like object where
     information about the lambda's Python code as well as the tracked closure
     variables in the lambda itself will be stored.   Defaults
     to a global LRU cache.  This cache is independent of the "compiled_cache"
     used by the :class:`_engine.Connection` object.

    .. seealso::

        :ref:`engine_lambda_caching`


    """

    return StatementLambdaElement(
        lmb,
        roles.StatementRole,
        LambdaOptions(
            enable_tracking=enable_tracking,
            track_on=track_on,
            track_closure_variables=track_closure_variables,
            global_track_bound_values=global_track_bound_values,
            track_bound_values=track_bound_values,
            lambda_cache=lambda_cache,
        ),
    )


class LambdaElement(elements.ClauseElement):
    """A SQL construct where the state is stored as an un-invoked lambda.

    The :class:`_sql.LambdaElement` is produced transparently whenever
    passing lambda expressions into SQL constructs, such as::

        stmt = select(table).where(lambda: table.c.col == parameter)

    The :class:`_sql.LambdaElement` is the base of the
    :class:`_sql.StatementLambdaElement` which represents a full statement
    within a lambda.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`engine_lambda_caching`

    """

    __visit_name__ = "lambda_element"

    _is_lambda_element = True

    _traverse_internals = [
        ("_resolved", visitors.InternalTraversal.dp_clauseelement)
    ]

    _transforms: Tuple[_CloneCallableType, ...] = ()

    _resolved_bindparams: List[BindParameter[Any]]
    parent_lambda: Optional[StatementLambdaElement] = None
    closure_cache_key: Union[Tuple[Any, ...], Literal[CacheConst.NO_CACHE]]
    role: Type[SQLRole]
    _rec: Union[AnalyzedFunction, NonAnalyzedFunction]
    fn: _AnyLambdaType
    tracker_key: Tuple[CodeType, ...]

    def __repr__(self):
        return "%s(%r)" % (
            self.__class__.__name__,
            self.fn.__code__,
        )

    def __init__(
        self,
        fn: _LambdaType,
        role: Type[SQLRole],
        opts: Union[Type[LambdaOptions], LambdaOptions] = LambdaOptions,
        apply_propagate_attrs: Optional[ClauseElement] = None,
    ):
        self.fn = fn
        self.role = role
        self.tracker_key = (fn.__code__,)
        self.opts = opts

        if apply_propagate_attrs is None and (role is roles.StatementRole):
            apply_propagate_attrs = self

        rec = self._retrieve_tracker_rec(fn, apply_propagate_attrs, opts)

        if apply_propagate_attrs is not None:
            propagate_attrs = rec.propagate_attrs
            if propagate_attrs:
                apply_propagate_attrs._propagate_attrs = propagate_attrs

    def _retrieve_tracker_rec(self, fn, apply_propagate_attrs, opts):
        lambda_cache = opts.lambda_cache
        if lambda_cache is None:
            lambda_cache = _closure_per_cache_key

        tracker_key = self.tracker_key

        fn = self.fn
        closure = fn.__closure__
        tracker = AnalyzedCode.get(
            fn,
            self,
            opts,
        )

        bindparams: List[BindParameter[Any]]
        self._resolved_bindparams = bindparams = []

        if self.parent_lambda is not None:
            parent_closure_cache_key = self.parent_lambda.closure_cache_key
        else:
            parent_closure_cache_key = ()

        cache_key: Union[Tuple[Any, ...], Literal[CacheConst.NO_CACHE]]

        if parent_closure_cache_key is not _cache_key.NO_CACHE:
            anon_map = visitors.anon_map()
            cache_key = tuple(
                [
                    getter(closure, opts, anon_map, bindparams)
                    for getter in tracker.closure_trackers
                ]
            )

            if _cache_key.NO_CACHE not in anon_map:
                cache_key = parent_closure_cache_key + cache_key

                self.closure_cache_key = cache_key

                rec = lambda_cache.get(tracker_key + cache_key)
            else:
                cache_key = _cache_key.NO_CACHE
                rec = None

        else:
            cache_key = _cache_key.NO_CACHE
            rec = None

        self.closure_cache_key = cache_key

        if rec is None:
            if cache_key is not _cache_key.NO_CACHE:
                with AnalyzedCode._generation_mutex:
                    key = tracker_key + cache_key
                    if key not in lambda_cache:
                        rec = AnalyzedFunction(
                            tracker, self, apply_propagate_attrs, fn
                        )
                        rec.closure_bindparams = list(bindparams)
                        lambda_cache[key] = rec
                    else:
                        rec = lambda_cache[key]
            else:
                rec = NonAnalyzedFunction(self._invoke_user_fn(fn))

        else:
            bindparams[:] = [
                orig_bind._with_value(new_bind.value, maintain_key=True)
                for orig_bind, new_bind in zip(
                    rec.closure_bindparams, bindparams
                )
            ]

        self._rec = rec

        if cache_key is not _cache_key.NO_CACHE:
            if self.parent_lambda is not None:
                bindparams[:0] = self.parent_lambda._resolved_bindparams

            lambda_element: Optional[LambdaElement] = self
            while lambda_element is not None:
                rec = lambda_element._rec
                if rec.bindparam_trackers:
                    tracker_instrumented_fn = rec.tracker_instrumented_fn
                    for tracker in rec.bindparam_trackers:
                        tracker(
                            lambda_element.fn,
                            tracker_instrumented_fn,
                            bindparams,
                        )
                lambda_element = lambda_element.parent_lambda

        return rec

    def __getattr__(self, key):
        return getattr(self._rec.expected_expr, key)

    @property
    def _is_sequence(self):
        return self._rec.is_sequence

    @property
    def _select_iterable(self):
        if self._is_sequence:
            return itertools.chain.from_iterable(
                [element._select_iterable for element in self._resolved]
            )

        else:
            return self._resolved._select_iterable

    @property
    def _from_objects(self):
        if self._is_sequence:
            return itertools.chain.from_iterable(
                [element._from_objects for element in self._resolved]
            )

        else:
            return self._resolved._from_objects

    def _param_dict(self):
        return {b.key: b.value for b in self._resolved_bindparams}

    def _setup_binds_for_tracked_expr(self, expr):
        bindparam_lookup = {b.key: b for b in self._resolved_bindparams}

        def replace(
            element: Optional[visitors.ExternallyTraversible], **kw: Any
        ) -> Optional[visitors.ExternallyTraversible]:
            if isinstance(element, elements.BindParameter):
                if element.key in bindparam_lookup:
                    bind = bindparam_lookup[element.key]
                    if element.expanding:
                        bind.expanding = True
                        bind.expand_op = element.expand_op
                        bind.type = element.type
                    return bind

            return None

        if self._rec.is_sequence:
            expr = [
                visitors.replacement_traverse(sub_expr, {}, replace)
                for sub_expr in expr
            ]
        elif getattr(expr, "is_clause_element", False):
            expr = visitors.replacement_traverse(expr, {}, replace)

        return expr

    def _copy_internals(
        self,
        clone: _CloneCallableType = _clone,
        deferred_copy_internals: Optional[_CloneCallableType] = None,
        **kw: Any,
    ) -> None:
        # TODO: this needs A LOT of tests
        self._resolved = clone(
            self._resolved,
            deferred_copy_internals=deferred_copy_internals,
            **kw,
        )

    @util.memoized_property
    def _resolved(self):
        expr = self._rec.expected_expr

        if self._resolved_bindparams:
            expr = self._setup_binds_for_tracked_expr(expr)

        return expr

    def _gen_cache_key(self, anon_map, bindparams):
        if self.closure_cache_key is _cache_key.NO_CACHE:
            anon_map[_cache_key.NO_CACHE] = True
            return None

        cache_key = (
            self.fn.__code__,
            self.__class__,
        ) + self.closure_cache_key

        parent = self.parent_lambda

        while parent is not None:
            assert parent.closure_cache_key is not CacheConst.NO_CACHE
            parent_closure_cache_key: Tuple[Any, ...] = (
                parent.closure_cache_key
            )

            cache_key = (
                (parent.fn.__code__,) + parent_closure_cache_key + cache_key
            )

            parent = parent.parent_lambda

        if self._resolved_bindparams:
            bindparams.extend(self._resolved_bindparams)
        return cache_key

    def _invoke_user_fn(self, fn: _AnyLambdaType, *arg: Any) -> ClauseElement:
        return fn()  # type: ignore[no-any-return]


class DeferredLambdaElement(LambdaElement):
    """A LambdaElement where the lambda accepts arguments and is
    invoked within the compile phase with special context.

    This lambda doesn't normally produce its real SQL expression outside of the
    compile phase.  It is passed a fixed set of initial arguments
    so that it can generate a sample expression.

    """

    def __init__(
        self,
        fn: _AnyLambdaType,
        role: Type[roles.SQLRole],
        opts: Union[Type[LambdaOptions], LambdaOptions] = LambdaOptions,
        lambda_args: Tuple[Any, ...] = (),
    ):
        self.lambda_args = lambda_args
        super().__init__(fn, role, opts)

    def _invoke_user_fn(self, fn, *arg):
        return fn(*self.lambda_args)

    def _resolve_with_args(self, *lambda_args: Any) -> ClauseElement:
        assert isinstance(self._rec, AnalyzedFunction)
        tracker_fn = self._rec.tracker_instrumented_fn
        expr = tracker_fn(*lambda_args)

        expr = coercions.expect(self.role, expr)

        expr = self._setup_binds_for_tracked_expr(expr)

        # this validation is getting very close, but not quite, to achieving
        # #5767.  The problem is if the base lambda uses an unnamed column
        # as is very common with mixins, the parameter name is different
        # and it produces a false positive; that is, for the documented case
        # that is exactly what people will be doing, it doesn't work, so
        # I'm not really sure how to handle this right now.
        # expected_binds = [
        #    b._orig_key
        #    for b in self._rec.expr._generate_cache_key()[1]
        #    if b.required
        # ]
        # got_binds = [
        #    b._orig_key for b in expr._generate_cache_key()[1] if b.required
        # ]
        # if expected_binds != got_binds:
        #    raise exc.InvalidRequestError(
        #        "Lambda callable at %s produced a different set of bound "
        #        "parameters than its original run: %s"
        #        % (self.fn.__code__, ", ".join(got_binds))
        #    )

        # TODO: TEST TEST TEST, this is very out there
        for deferred_copy_internals in self._transforms:
            expr = deferred_copy_internals(expr)

        return expr  # type: ignore

    def _copy_internals(
        self, clone=_clone, deferred_copy_internals=None, **kw
    ):
        super()._copy_internals(
            clone=clone,
            deferred_copy_internals=deferred_copy_internals,  # **kw
            opts=kw,
        )

        # TODO: A LOT A LOT of tests.   for _resolve_with_args, we don't know
        # our expression yet.   so hold onto the replacement
        if deferred_copy_internals:
            self._transforms += (deferred_copy_internals,)


class StatementLambdaElement(
    roles.AllowsLambdaRole, LambdaElement, Executable
):
    """Represent a composable SQL statement as a :class:`_sql.LambdaElement`.

    The :class:`_sql.StatementLambdaElement` is constructed using the
    :func:`_sql.lambda_stmt` function::


        from sqlalchemy import lambda_stmt

        stmt = lambda_stmt(lambda: select(table))

    Once constructed, additional criteria can be built onto the statement
    by adding subsequent lambdas, which accept the existing statement
    object as a single parameter::

        stmt += lambda s: s.where(table.c.col == parameter)

    .. versionadded:: 1.4

    .. seealso::

        :ref:`engine_lambda_caching`

    """

    if TYPE_CHECKING:

        def __init__(
            self,
            fn: _StmtLambdaType,
            role: Type[SQLRole],
            opts: Union[Type[LambdaOptions], LambdaOptions] = LambdaOptions,
            apply_propagate_attrs: Optional[ClauseElement] = None,
        ): ...

    def __add__(
        self, other: _StmtLambdaElementType[Any]
    ) -> StatementLambdaElement:
        return self.add_criteria(other)

    def add_criteria(
        self,
        other: _StmtLambdaElementType[Any],
        enable_tracking: bool = True,
        track_on: Optional[Any] = None,
        track_closure_variables: bool = True,
        track_bound_values: bool = True,
    ) -> StatementLambdaElement:
        """Add new criteria to this :class:`_sql.StatementLambdaElement`.

        E.g.::

            >>> def my_stmt(parameter):
            ...     stmt = lambda_stmt(
            ...         lambda: select(table.c.x, table.c.y),
            ...     )
            ...     stmt = stmt.add_criteria(lambda: table.c.x > parameter)
            ...     return stmt

        The :meth:`_sql.StatementLambdaElement.add_criteria` method is
        equivalent to using the Python addition operator to add a new
        lambda, except that additional arguments may be added including
        ``track_closure_values`` and ``track_on``::

            >>> def my_stmt(self, foo):
            ...     stmt = lambda_stmt(
            ...         lambda: select(func.max(foo.x, foo.y)),
            ...         track_closure_variables=False,
            ...     )
            ...     stmt = stmt.add_criteria(lambda: self.where_criteria, track_on=[self])
            ...     return stmt

        See :func:`_sql.lambda_stmt` for a description of the parameters
        accepted.

        """  # noqa: E501

        opts = self.opts + dict(
            enable_tracking=enable_tracking,
            track_closure_variables=track_closure_variables,
            global_track_bound_values=self.opts.global_track_bound_values,
            track_on=track_on,
            track_bound_values=track_bound_values,
        )

        return LinkedLambdaElement(other, parent_lambda=self, opts=opts)

    def _execute_on_connection(
        self, connection, distilled_params, execution_options
    ):
        if TYPE_CHECKING:
            assert isinstance(self._rec.expected_expr, ClauseElement)
        if self._rec.expected_expr.supports_execution:
            return connection._execute_clauseelement(
                self, distilled_params, execution_options
            )
        else:
            raise exc.ObjectNotExecutableError(self)

    @property
    def _proxied(self) -> Any:
        return self._rec_expected_expr

    @property
    def _with_options(self):
        return self._proxied._with_options

    @property
    def _effective_plugin_target(self):
        return self._proxied._effective_plugin_target

    @property
    def _execution_options(self):
        return self._proxied._execution_options

    @property
    def _all_selected_columns(self):
        return self._proxied._all_selected_columns

    @property
    def is_select(self):
        return self._proxied.is_select

    @property
    def is_update(self):
        return self._proxied.is_update

    @property
    def is_insert(self):
        return self._proxied.is_insert

    @property
    def is_text(self):
        return self._proxied.is_text

    @property
    def is_delete(self):
        return self._proxied.is_delete

    @property
    def is_dml(self):
        return self._proxied.is_dml

    def spoil(self) -> NullLambdaStatement:
        """Return a new :class:`.StatementLambdaElement` that will run
        all lambdas unconditionally each time.

        """
        return NullLambdaStatement(self.fn())


class NullLambdaStatement(roles.AllowsLambdaRole, elements.ClauseElement):
    """Provides the :class:`.StatementLambdaElement` API but does not
    cache or analyze lambdas.

    the lambdas are instead invoked immediately.

    The intended use is to isolate issues that may arise when using
    lambda statements.

    """

    __visit_name__ = "lambda_element"

    _is_lambda_element = True

    _traverse_internals = [
        ("_resolved", visitors.InternalTraversal.dp_clauseelement)
    ]

    def __init__(self, statement):
        self._resolved = statement
        self._propagate_attrs = statement._propagate_attrs

    def __getattr__(self, key):
        return getattr(self._resolved, key)

    def __add__(self, other):
        statement = other(self._resolved)

        return NullLambdaStatement(statement)

    def add_criteria(self, other, **kw):
        statement = other(self._resolved)

        return NullLambdaStatement(statement)

    def _execute_on_connection(
        self, connection, distilled_params, execution_options
    ):
        if self._resolved.supports_execution:
            return connection._execute_clauseelement(
                self, distilled_params, execution_options
            )
        else:
            raise exc.ObjectNotExecutableError(self)


class LinkedLambdaElement(StatementLambdaElement):
    """Represent subsequent links of a :class:`.StatementLambdaElement`."""

    parent_lambda: StatementLambdaElement

    def __init__(
        self,
        fn: _StmtLambdaElementType[Any],
        parent_lambda: StatementLambdaElement,
        opts: Union[Type[LambdaOptions], LambdaOptions],
    ):
        self.opts = opts
        self.fn = fn
        self.parent_lambda = parent_lambda

        self.tracker_key = parent_lambda.tracker_key + (fn.__code__,)
        self._retrieve_tracker_rec(fn, self, opts)
        self._propagate_attrs = parent_lambda._propagate_attrs

    def _invoke_user_fn(self, fn, *arg):
        return fn(self.parent_lambda._resolved)


class AnalyzedCode:
    __slots__ = (
        "track_closure_variables",
        "track_bound_values",
        "bindparam_trackers",
        "closure_trackers",
        "build_py_wrappers",
    )
    _fns: weakref.WeakKeyDictionary[CodeType, AnalyzedCode] = (
        weakref.WeakKeyDictionary()
    )

    _generation_mutex = threading.RLock()

    @classmethod
    def get(cls, fn, lambda_element, lambda_kw, **kw):
        try:
            # TODO: validate kw haven't changed?
            return cls._fns[fn.__code__]
        except KeyError:
            pass

        with cls._generation_mutex:
            # check for other thread already created object
            if fn.__code__ in cls._fns:
                return cls._fns[fn.__code__]

            analyzed: AnalyzedCode
            cls._fns[fn.__code__] = analyzed = AnalyzedCode(
                fn, lambda_element, lambda_kw, **kw
            )
            return analyzed

    def __init__(self, fn, lambda_element, opts):
        if inspect.ismethod(fn):
            raise exc.ArgumentError(
                "Method %s may not be passed as a SQL expression" % fn
            )
        closure = fn.__closure__

        self.track_bound_values = (
            opts.track_bound_values and opts.global_track_bound_values
        )
        enable_tracking = opts.enable_tracking
        track_on = opts.track_on
        track_closure_variables = opts.track_closure_variables

        self.track_closure_variables = track_closure_variables and not track_on

        # a list of callables generated from _bound_parameter_getter_*
        # functions.  Each of these uses a PyWrapper object to retrieve
        # a parameter value
        self.bindparam_trackers = []

        # a list of callables generated from _cache_key_getter_* functions
        # these callables work to generate a cache key for the lambda
        # based on what's inside its closure variables.
        self.closure_trackers = []

        self.build_py_wrappers = []

        if enable_tracking:
            if track_on:
                self._init_track_on(track_on)

            self._init_globals(fn)

            if closure:
                self._init_closure(fn)

        self._setup_additional_closure_trackers(fn, lambda_element, opts)

    def _init_track_on(self, track_on):
        self.closure_trackers.extend(
            self._cache_key_getter_track_on(idx, elem)
            for idx, elem in enumerate(track_on)
        )

    def _init_globals(self, fn):
        build_py_wrappers = self.build_py_wrappers
        bindparam_trackers = self.bindparam_trackers
        track_bound_values = self.track_bound_values

        for name in fn.__code__.co_names:
            if name not in fn.__globals__:
                continue

            _bound_value = self._roll_down_to_literal(fn.__globals__[name])

            if coercions._deep_is_literal(_bound_value):
                build_py_wrappers.append((name, None))
                if track_bound_values:
                    bindparam_trackers.append(
                        self._bound_parameter_getter_func_globals(name)
                    )

    def _init_closure(self, fn):
        build_py_wrappers = self.build_py_wrappers
        closure = fn.__closure__

        track_bound_values = self.track_bound_values
        track_closure_variables = self.track_closure_variables
        bindparam_trackers = self.bindparam_trackers
        closure_trackers = self.closure_trackers

        for closure_index, (fv, cell) in enumerate(
            zip(fn.__code__.co_freevars, closure)
        ):
            _bound_value = self._roll_down_to_literal(cell.cell_contents)

            if coercions._deep_is_literal(_bound_value):
                build_py_wrappers.append((fv, closure_index))
                if track_bound_values:
                    bindparam_trackers.append(
                        self._bound_parameter_getter_func_closure(
                            fv, closure_index
                        )
                    )
            else:
                # for normal cell contents, add them to a list that
                # we can compare later when we get new lambdas.  if
                # any identities have changed, then we will
                # recalculate the whole lambda and run it again.

                if track_closure_variables:
                    closure_trackers.append(
                        self._cache_key_getter_closure_variable(
                            fn, fv, closure_index, cell.cell_contents
                        )
                    )

    def _setup_additional_closure_trackers(self, fn, lambda_element, opts):
        # an additional step is to actually run the function, then
        # go through the PyWrapper objects that were set up to catch a bound
        # parameter.   then if they *didn't* make a param, oh they're another
        # object in the closure we have to track for our cache key.  so
        # create trackers to catch those.

        analyzed_function = AnalyzedFunction(
            self,
            lambda_element,
            None,
            fn,
        )

        closure_trackers = self.closure_trackers

        for pywrapper in analyzed_function.closure_pywrappers:
            if not pywrapper._sa__has_param:
                closure_trackers.append(
                    self._cache_key_getter_tracked_literal(fn, pywrapper)
                )

    @classmethod
    def _roll_down_to_literal(cls, element):
        is_clause_element = hasattr(element, "__clause_element__")

        if is_clause_element:
            while not isinstance(
                element, (elements.ClauseElement, schema.SchemaItem, type)
            ):
                try:
                    element = element.__clause_element__()
                except AttributeError:
                    break

        if not is_clause_element:
            insp = inspection.inspect(element, raiseerr=False)
            if insp is not None:
                try:
                    return insp.__clause_element__()
                except AttributeError:
                    return insp

            # TODO: should we coerce consts None/True/False here?
            return element
        else:
            return element

    def _bound_parameter_getter_func_globals(self, name):
        """Return a getter that will extend a list of bound parameters
        with new entries from the ``__globals__`` collection of a particular
        lambda.

        """

        def extract_parameter_value(
            current_fn, tracker_instrumented_fn, result
        ):
            wrapper = tracker_instrumented_fn.__globals__[name]
            object.__getattribute__(wrapper, "_extract_bound_parameters")(
                current_fn.__globals__[name], result
            )

        return extract_parameter_value

    def _bound_parameter_getter_func_closure(self, name, closure_index):
        """Return a getter that will extend a list of bound parameters
        with new entries from the ``__closure__`` collection of a particular
        lambda.

        """

        def extract_parameter_value(
            current_fn, tracker_instrumented_fn, result
        ):
            wrapper = tracker_instrumented_fn.__closure__[
                closure_index
            ].cell_contents
            object.__getattribute__(wrapper, "_extract_bound_parameters")(
                current_fn.__closure__[closure_index].cell_contents, result
            )

        return extract_parameter_value

    def _cache_key_getter_track_on(self, idx, elem):
        """Return a getter that will extend a cache key with new entries
        from the "track_on" parameter passed to a :class:`.LambdaElement`.

        """

        if isinstance(elem, tuple):
            # tuple must contain hascachekey elements
            def get(closure, opts, anon_map, bindparams):
                return tuple(
                    tup_elem._gen_cache_key(anon_map, bindparams)
                    for tup_elem in opts.track_on[idx]
                )

        elif isinstance(elem, _cache_key.HasCacheKey):

            def get(closure, opts, anon_map, bindparams):
                return opts.track_on[idx]._gen_cache_key(anon_map, bindparams)

        else:

            def get(closure, opts, anon_map, bindparams):
                return opts.track_on[idx]

        return get

    def _cache_key_getter_closure_variable(
        self,
        fn,
        variable_name,
        idx,
        cell_contents,
        use_clause_element=False,
        use_inspect=False,
    ):
        """Return a getter that will extend a cache key with new entries
        from the ``__closure__`` collection of a particular lambda.

        """

        if isinstance(cell_contents, _cache_key.HasCacheKey):

            def get(closure, opts, anon_map, bindparams):
                obj = closure[idx].cell_contents
                if use_inspect:
                    obj = inspection.inspect(obj)
                elif use_clause_element:
                    while hasattr(obj, "__clause_element__"):
                        if not getattr(obj, "is_clause_element", False):
                            obj = obj.__clause_element__()

                return obj._gen_cache_key(anon_map, bindparams)

        elif isinstance(cell_contents, types.FunctionType):

            def get(closure, opts, anon_map, bindparams):
                return closure[idx].cell_contents.__code__

        elif isinstance(cell_contents, collections_abc.Sequence):

            def get(closure, opts, anon_map, bindparams):
                contents = closure[idx].cell_contents

                try:
                    return tuple(
                        elem._gen_cache_key(anon_map, bindparams)
                        for elem in contents
                    )
                except AttributeError as ae:
                    self._raise_for_uncacheable_closure_variable(
                        variable_name, fn, from_=ae
                    )

        else:
            # if the object is a mapped class or aliased class, or some
            # other object in the ORM realm of things like that, imitate
            # the logic used in coercions.expect() to roll it down to the
            # SQL element
            element = cell_contents
            is_clause_element = False
            while hasattr(element, "__clause_element__"):
                is_clause_element = True
                if not getattr(element, "is_clause_element", False):
                    element = element.__clause_element__()
                else:
                    break

            if not is_clause_element:
                insp = inspection.inspect(element, raiseerr=False)
                if insp is not None:
                    return self._cache_key_getter_closure_variable(
                        fn, variable_name, idx, insp, use_inspect=True
                    )
            else:
                return self._cache_key_getter_closure_variable(
                    fn, variable_name, idx, element, use_clause_element=True
                )

            self._raise_for_uncacheable_closure_variable(variable_name, fn)

        return get

    def _raise_for_uncacheable_closure_variable(
        self, variable_name, fn, from_=None
    ):
        raise exc.InvalidRequestError(
            "Closure variable named '%s' inside of lambda callable %s "
            "does not refer to a cacheable SQL element, and also does not "
            "appear to be serving as a SQL literal bound value based on "
            "the default "
            "SQL expression returned by the function.   This variable "
            "needs to remain outside the scope of a SQL-generating lambda "
            "so that a proper cache key may be generated from the "
            "lambda's state.  Evaluate this variable outside of the "
            "lambda, set track_on=[<elements>] to explicitly select "
            "closure elements to track, or set "
            "track_closure_variables=False to exclude "
            "closure variables from being part of the cache key."
            % (variable_name, fn.__code__),
        ) from from_

    def _cache_key_getter_tracked_literal(self, fn, pytracker):
        """Return a getter that will extend a cache key with new entries
        from the ``__closure__`` collection of a particular lambda.

        this getter differs from _cache_key_getter_closure_variable
        in that these are detected after the function is run, and PyWrapper
        objects have recorded that a particular literal value is in fact
        not being interpreted as a bound parameter.

        """

        elem = pytracker._sa__to_evaluate
        closure_index = pytracker._sa__closure_index
        variable_name = pytracker._sa__name

        return self._cache_key_getter_closure_variable(
            fn, variable_name, closure_index, elem
        )


class NonAnalyzedFunction:
    __slots__ = ("expr",)

    closure_bindparams: Optional[List[BindParameter[Any]]] = None
    bindparam_trackers: Optional[List[_BoundParameterGetter]] = None

    is_sequence = False

    expr: ClauseElement

    def __init__(self, expr: ClauseElement):
        self.expr = expr

    @property
    def expected_expr(self) -> ClauseElement:
        return self.expr


class AnalyzedFunction:
    __slots__ = (
        "analyzed_code",
        "fn",
        "closure_pywrappers",
        "tracker_instrumented_fn",
        "expr",
        "bindparam_trackers",
        "expected_expr",
        "is_sequence",
        "propagate_attrs",
        "closure_bindparams",
    )

    closure_bindparams: Optional[List[BindParameter[Any]]]
    expected_expr: Union[ClauseElement, List[ClauseElement]]
    bindparam_trackers: Optional[List[_BoundParameterGetter]]

    def __init__(
        self,
        analyzed_code,
        lambda_element,
        apply_propagate_attrs,
        fn,
    ):
        self.analyzed_code = analyzed_code
        self.fn = fn

        self.bindparam_trackers = analyzed_code.bindparam_trackers

        self._instrument_and_run_function(lambda_element)

        self._coerce_expression(lambda_element, apply_propagate_attrs)

    def _instrument_and_run_function(self, lambda_element):
        analyzed_code = self.analyzed_code

        fn = self.fn
        self.closure_pywrappers = closure_pywrappers = []

        build_py_wrappers = analyzed_code.build_py_wrappers

        if not build_py_wrappers:
            self.tracker_instrumented_fn = tracker_instrumented_fn = fn
            self.expr = lambda_element._invoke_user_fn(tracker_instrumented_fn)
        else:
            track_closure_variables = analyzed_code.track_closure_variables
            closure = fn.__closure__

            # will form the __closure__ of the function when we rebuild it
            if closure:
                new_closure = {
                    fv: cell.cell_contents
                    for fv, cell in zip(fn.__code__.co_freevars, closure)
                }
            else:
                new_closure = {}

            # will form the __globals__ of the function when we rebuild it
            new_globals = fn.__globals__.copy()

            for name, closure_index in build_py_wrappers:
                if closure_index is not None:
                    value = closure[closure_index].cell_contents
                    new_closure[name] = bind = PyWrapper(
                        fn,
                        name,
                        value,
                        closure_index=closure_index,
                        track_bound_values=(
                            self.analyzed_code.track_bound_values
                        ),
                    )
                    if track_closure_variables:
                        closure_pywrappers.append(bind)
                else:
                    value = fn.__globals__[name]
                    new_globals[name] = PyWrapper(fn, name, value)

            # rewrite the original fn.   things that look like they will
            # become bound parameters are wrapped in a PyWrapper.
            self.tracker_instrumented_fn = tracker_instrumented_fn = (
                self._rewrite_code_obj(
                    fn,
                    [new_closure[name] for name in fn.__code__.co_freevars],
                    new_globals,
                )
            )

            # now invoke the function.  This will give us a new SQL
            # expression, but all the places that there would be a bound
            # parameter, the PyWrapper in its place will give us a bind
            # with a predictable name we can match up later.

            # additionally, each PyWrapper will log that it did in fact
            # create a parameter, otherwise, it's some kind of Python
            # object in the closure and we want to track that, to make
            # sure it doesn't change to something else, or if it does,
            # that we create a different tracked function with that
            # variable.
            self.expr = lambda_element._invoke_user_fn(tracker_instrumented_fn)

    def _coerce_expression(self, lambda_element, apply_propagate_attrs):
        """Run the tracker-generated expression through coercion rules.

        After the user-defined lambda has been invoked to produce a statement
        for re-use, run it through coercion rules to both check that it's the
        correct type of object and also to coerce it to its useful form.

        """

        parent_lambda = lambda_element.parent_lambda
        expr = self.expr

        if parent_lambda is None:
            if isinstance(expr, collections_abc.Sequence):
                self.expected_expr = [
                    cast(
                        "ClauseElement",
                        coercions.expect(
                            lambda_element.role,
                            sub_expr,
                            apply_propagate_attrs=apply_propagate_attrs,
                        ),
                    )
                    for sub_expr in expr
                ]
                self.is_sequence = True
            else:
                self.expected_expr = cast(
                    "ClauseElement",
                    coercions.expect(
                        lambda_element.role,
                        expr,
                        apply_propagate_attrs=apply_propagate_attrs,
                    ),
                )
                self.is_sequence = False
        else:
            self.expected_expr = expr
            self.is_sequence = False

        if apply_propagate_attrs is not None:
            self.propagate_attrs = apply_propagate_attrs._propagate_attrs
        else:
            self.propagate_attrs = util.EMPTY_DICT

    def _rewrite_code_obj(self, f, cell_values, globals_):
        """Return a copy of f, with a new closure and new globals

        yes it works in pypy :P

        """

        argrange = range(len(cell_values))

        code = "def make_cells():\n"
        if cell_values:
            code += "    (%s) = (%s)\n" % (
                ", ".join("i%d" % i for i in argrange),
                ", ".join("o%d" % i for i in argrange),
            )
        code += "    def closure():\n"
        code += "        return %s\n" % ", ".join("i%d" % i for i in argrange)
        code += "    return closure.__closure__"
        vars_ = {"o%d" % i: cell_values[i] for i in argrange}
        exec(code, vars_, vars_)
        closure = vars_["make_cells"]()

        func = type(f)(
            f.__code__, globals_, f.__name__, f.__defaults__, closure
        )
        func.__annotations__ = f.__annotations__
        func.__kwdefaults__ = f.__kwdefaults__
        func.__doc__ = f.__doc__
        func.__module__ = f.__module__

        return func


class PyWrapper(ColumnOperators):
    """A wrapper object that is injected into the ``__globals__`` and
    ``__closure__`` of a Python function.

    When the function is instrumented with :class:`.PyWrapper` objects, it is
    then invoked just once in order to set up the wrappers.  We look through
    all the :class:`.PyWrapper` objects we made to find the ones that generated
    a :class:`.BindParameter` object, e.g. the expression system interpreted
    something as a literal.   Those positions in the globals/closure are then
    ones that we will look at, each time a new lambda comes in that refers to
    the same ``__code__`` object.   In this way, we keep a single version of
    the SQL expression that this lambda produced, without calling upon the
    Python function that created it more than once, unless its other closure
    variables have changed.   The expression is then transformed to have the
    new bound values embedded into it.

    """

    def __init__(
        self,
        fn,
        name,
        to_evaluate,
        closure_index=None,
        getter=None,
        track_bound_values=True,
    ):
        self.fn = fn
        self._name = name
        self._to_evaluate = to_evaluate
        self._param = None
        self._has_param = False
        self._bind_paths = {}
        self._getter = getter
        self._closure_index = closure_index
        self.track_bound_values = track_bound_values

    def __call__(self, *arg, **kw):
        elem = object.__getattribute__(self, "_to_evaluate")
        value = elem(*arg, **kw)
        if (
            self._sa_track_bound_values
            and coercions._deep_is_literal(value)
            and not isinstance(
                # TODO: coverage where an ORM option or similar is here
                value,
                _cache_key.HasCacheKey,
            )
        ):
            name = object.__getattribute__(self, "_name")
            raise exc.InvalidRequestError(
                "Can't invoke Python callable %s() inside of lambda "
                "expression argument at %s; lambda SQL constructs should "
                "not invoke functions from closure variables to produce "
                "literal values since the "
                "lambda SQL system normally extracts bound values without "
                "actually "
                "invoking the lambda or any functions within it.  Call the "
                "function outside of the "
                "lambda and assign to a local variable that is used in the "
                "lambda as a closure variable, or set "
                "track_bound_values=False if the return value of this "
                "function is used in some other way other than a SQL bound "
                "value." % (name, self._sa_fn.__code__)
            )
        else:
            return value

    def operate(self, op, *other, **kwargs):
        elem = object.__getattribute__(self, "_py_wrapper_literal")()
        return op(elem, *other, **kwargs)

    def reverse_operate(self, op, other, **kwargs):
        elem = object.__getattribute__(self, "_py_wrapper_literal")()
        return op(other, elem, **kwargs)

    def _extract_bound_parameters(self, starting_point, result_list):
        param = object.__getattribute__(self, "_param")
        if param is not None:
            param = param._with_value(starting_point, maintain_key=True)
            result_list.append(param)
        for pywrapper in object.__getattribute__(self, "_bind_paths").values():
            getter = object.__getattribute__(pywrapper, "_getter")
            element = getter(starting_point)
            pywrapper._sa__extract_bound_parameters(element, result_list)

    def _py_wrapper_literal(self, expr=None, operator=None, **kw):
        param = object.__getattribute__(self, "_param")
        to_evaluate = object.__getattribute__(self, "_to_evaluate")
        if param is None:
            name = object.__getattribute__(self, "_name")
            self._param = param = elements.BindParameter(
                name,
                required=False,
                unique=True,
                _compared_to_operator=operator,
                _compared_to_type=expr.type if expr is not None else None,
            )
            self._has_param = True
        return param._with_value(to_evaluate, maintain_key=True)

    def __bool__(self):
        to_evaluate = object.__getattribute__(self, "_to_evaluate")
        return bool(to_evaluate)

    def __getattribute__(self, key):
        if key.startswith("_sa_"):
            return object.__getattribute__(self, key[4:])
        elif key in (
            "__clause_element__",
            "operate",
            "reverse_operate",
            "_py_wrapper_literal",
            "__class__",
            "__dict__",
        ):
            return object.__getattribute__(self, key)

        if key.startswith("__"):
            elem = object.__getattribute__(self, "_to_evaluate")
            return getattr(elem, key)
        else:
            return self._sa__add_getter(key, operator.attrgetter)

    def __iter__(self):
        elem = object.__getattribute__(self, "_to_evaluate")
        return iter(elem)

    def __getitem__(self, key):
        elem = object.__getattribute__(self, "_to_evaluate")
        if not hasattr(elem, "__getitem__"):
            raise AttributeError("__getitem__")

        if isinstance(key, PyWrapper):
            # TODO: coverage
            raise exc.InvalidRequestError(
                "Dictionary keys / list indexes inside of a cached "
                "lambda must be Python literals only"
            )
        return self._sa__add_getter(key, operator.itemgetter)

    def _add_getter(self, key, getter_fn):
        bind_paths = object.__getattribute__(self, "_bind_paths")

        bind_path_key = (key, getter_fn)
        if bind_path_key in bind_paths:
            return bind_paths[bind_path_key]

        getter = getter_fn(key)
        elem = object.__getattribute__(self, "_to_evaluate")
        value = getter(elem)

        rolled_down_value = AnalyzedCode._roll_down_to_literal(value)

        if coercions._deep_is_literal(rolled_down_value):
            wrapper = PyWrapper(self._sa_fn, key, value, getter=getter)
            bind_paths[bind_path_key] = wrapper
            return wrapper
        else:
            return value


@inspection._inspects(LambdaElement)
def insp(lmb):
    return inspection.inspect(lmb._resolved)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\densetools.py ===
"""Advanced tools for dense recursive polynomials in ``K[x]`` or ``K[X]``. """


from sympy.polys.densearith import (
    dup_add_term, dmp_add_term,
    dup_lshift,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr,
    dup_div,
    dup_rem, dmp_rem,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground,
    dup_exquo_ground, dmp_exquo_ground,
)
from sympy.polys.densebasic import (
    dup_strip, dmp_strip,
    dup_convert, dmp_convert,
    dup_degree, dmp_degree,
    dmp_to_dict,
    dmp_from_dict,
    dup_LC, dmp_LC, dmp_ground_LC,
    dup_TC, dmp_TC,
    dmp_zero, dmp_ground,
    dmp_zero_p,
    dup_to_raw_dict, dup_from_raw_dict,
    dmp_zeros,
    dmp_include,
)
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    DomainError
)

from math import ceil as _ceil, log2 as _log2


def dup_integrate(f, m, K):
    """
    Computes the indefinite integral of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_integrate(x**2 + 2*x, 1)
    1/3*x**3 + x**2
    >>> R.dup_integrate(x**2 + 2*x, 2)
    1/12*x**4 + 1/3*x**3

    """
    if m <= 0 or not f:
        return f

    g = [K.zero]*m

    for i, c in enumerate(reversed(f)):
        n = i + 1

        for j in range(1, m):
            n *= i + j + 1

        g.insert(0, K.exquo(c, K(n)))

    return g


def dmp_integrate(f, m, u, K):
    """
    Computes the indefinite integral of ``f`` in ``x_0`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate(x + 2*y, 1)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate(x + 2*y, 2)
    1/6*x**3 + x**2*y

    """
    if not u:
        return dup_integrate(f, m, K)

    if m <= 0 or dmp_zero_p(f, u):
        return f

    g, v = dmp_zeros(m, u - 1, K), u - 1

    for i, c in enumerate(reversed(f)):
        n = i + 1

        for j in range(1, m):
            n *= i + j + 1

        g.insert(0, dmp_quo_ground(c, K(n), v, K))

    return g


def _rec_integrate_in(g, m, v, i, j, K):
    """Recursive helper for :func:`dmp_integrate_in`."""
    if i == j:
        return dmp_integrate(g, m, v, K)

    w, i = v - 1, i + 1

    return dmp_strip([ _rec_integrate_in(c, m, w, i, j, K) for c in g ], v)


def dmp_integrate_in(f, m, j, u, K):
    """
    Computes the indefinite integral of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_integrate_in(x + 2*y, 1, 0)
    1/2*x**2 + 2*x*y
    >>> R.dmp_integrate_in(x + 2*y, 1, 1)
    x*y + y**2

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= u expected, got u = %d, j = %d" % (u, j))

    return _rec_integrate_in(f, m, u, 0, j, K)


def dup_diff(f, m, K):
    """
    ``m``-th order derivative of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 1)
    3*x**2 + 4*x + 3
    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 2)
    6*x + 4

    """
    if m <= 0:
        return f

    n = dup_degree(f)

    if n < m:
        return []

    deriv = []

    if m == 1:
        for coeff in f[:-m]:
            deriv.append(K(n)*coeff)
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n

            for i in range(n - 1, n - m, -1):
                k *= i

            deriv.append(K(k)*coeff)
            n -= 1

    return dup_strip(deriv)


def dmp_diff(f, m, u, K):
    """
    ``m``-th order derivative in ``x_0`` of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff(f, 1)
    y**2 + 2*y + 3
    >>> R.dmp_diff(f, 2)
    0

    """
    if not u:
        return dup_diff(f, m, K)
    if m <= 0:
        return f

    n = dmp_degree(f, u)

    if n < m:
        return dmp_zero(u)

    deriv, v = [], u - 1

    if m == 1:
        for coeff in f[:-m]:
            deriv.append(dmp_mul_ground(coeff, K(n), v, K))
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n

            for i in range(n - 1, n - m, -1):
                k *= i

            deriv.append(dmp_mul_ground(coeff, K(k), v, K))
            n -= 1

    return dmp_strip(deriv, u)


def _rec_diff_in(g, m, v, i, j, K):
    """Recursive helper for :func:`dmp_diff_in`."""
    if i == j:
        return dmp_diff(g, m, v, K)

    w, i = v - 1, i + 1

    return dmp_strip([ _rec_diff_in(c, m, w, i, j, K) for c in g ], v)


def dmp_diff_in(f, m, j, u, K):
    """
    ``m``-th order derivative in ``x_j`` of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff_in(f, 1, 0)
    y**2 + 2*y + 3
    >>> R.dmp_diff_in(f, 1, 1)
    2*x*y + 2*x + 4*y + 3

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    return _rec_diff_in(f, m, u, 0, j, K)


def dup_eval(f, a, K):
    """
    Evaluate a polynomial at ``x = a`` in ``K[x]`` using Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_eval(x**2 + 2*x + 3, 2)
    11

    """
    if not a:
        return K.convert(dup_TC(f, K))

    result = K.zero

    for c in f:
        result *= a
        result += c

    return result


def dmp_eval(f, a, u, K):
    """
    Evaluate a polynomial at ``x_0 = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_eval(2*x*y + 3*x + y + 2, 2)
    5*y + 8

    """
    if not u:
        return dup_eval(f, a, K)

    if not a:
        return dmp_TC(f, K)

    result, v = dmp_LC(f, K), u - 1

    for coeff in f[1:]:
        result = dmp_mul_ground(result, a, v, K)
        result = dmp_add(result, coeff, v, K)

    return result


def _rec_eval_in(g, a, v, i, j, K):
    """Recursive helper for :func:`dmp_eval_in`."""
    if i == j:
        return dmp_eval(g, a, v, K)

    v, i = v - 1, i + 1

    return dmp_strip([ _rec_eval_in(c, a, v, i, j, K) for c in g ], v)


def dmp_eval_in(f, a, j, u, K):
    """
    Evaluate a polynomial at ``x_j = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_in(f, 2, 0)
    5*y + 8
    >>> R.dmp_eval_in(f, 2, 1)
    7*x + 4

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    return _rec_eval_in(f, a, u, 0, j, K)


def _rec_eval_tail(g, i, A, u, K):
    """Recursive helper for :func:`dmp_eval_tail`."""
    if i == u:
        return dup_eval(g, A[-1], K)
    else:
        h = [ _rec_eval_tail(c, i + 1, A, u, K) for c in g ]

        if i < u - len(A) + 1:
            return h
        else:
            return dup_eval(h, A[-u + i - 1], K)


def dmp_eval_tail(f, A, u, K):
    """
    Evaluate a polynomial at ``x_j = a_j, ...`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_tail(f, [2])
    7*x + 4
    >>> R.dmp_eval_tail(f, [2, 2])
    18

    """
    if not A:
        return f

    if dmp_zero_p(f, u):
        return dmp_zero(u - len(A))

    e = _rec_eval_tail(f, 0, A, u, K)

    if u == len(A) - 1:
        return e
    else:
        return dmp_strip(e, u - len(A))


def _rec_diff_eval(g, m, a, v, i, j, K):
    """Recursive helper for :func:`dmp_diff_eval`."""
    if i == j:
        return dmp_eval(dmp_diff(g, m, v, K), a, v, K)

    v, i = v - 1, i + 1

    return dmp_strip([ _rec_diff_eval(c, m, a, v, i, j, K) for c in g ], v)


def dmp_diff_eval_in(f, m, a, j, u, K):
    """
    Differentiate and evaluate a polynomial in ``x_j`` at ``a`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff_eval_in(f, 1, 2, 0)
    y**2 + 2*y + 3
    >>> R.dmp_diff_eval_in(f, 1, 2, 1)
    6*x + 11

    """
    if j > u:
        raise IndexError("-%s <= j < %s expected, got %s" % (u, u, j))
    if not j:
        return dmp_eval(dmp_diff(f, m, u, K), a, u, K)

    return _rec_diff_eval(f, m, a, u, 0, j, K)


def dup_trunc(f, p, K):
    """
    Reduce a ``K[x]`` polynomial modulo a constant ``p`` in ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_trunc(2*x**3 + 3*x**2 + 5*x + 7, ZZ(3))
    -x**3 - x + 1

    """
    if K.is_ZZ:
        g = []

        for c in f:
            c = c % p

            if c > p // 2:
                g.append(c - p)
            else:
                g.append(c)
    elif K.is_FiniteField:
        # XXX: python-flint's nmod does not support %
        pi = int(p)
        g = [ K(int(c) % pi) for c in f ]
    else:
        g = [ c % p for c in f ]

    return dup_strip(g)


def dmp_trunc(f, p, u, K):
    """
    Reduce a ``K[X]`` polynomial modulo a polynomial ``p`` in ``K[Y]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y + 8*x**2 + 5*x*y + 6*x + 2*y + 3
    >>> g = (y - 1).drop(x)

    >>> R.dmp_trunc(f, g)
    11*x**2 + 11*x + 5

    """
    return dmp_strip([ dmp_rem(c, p, u - 1, K) for c in f ], u)


def dmp_ground_trunc(f, p, u, K):
    """
    Reduce a ``K[X]`` polynomial modulo a constant ``p`` in ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y + 8*x**2 + 5*x*y + 6*x + 2*y + 3

    >>> R.dmp_ground_trunc(f, ZZ(3))
    -x**2 - x*y - y

    """
    if not u:
        return dup_trunc(f, p, K)

    v = u - 1

    return dmp_strip([ dmp_ground_trunc(c, p, v, K) for c in f ], u)


def dup_monic(f, K):
    """
    Divide all coefficients by ``LC(f)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_monic(3*x**2 + 6*x + 9)
    x**2 + 2*x + 3

    >>> R, x = ring("x", QQ)
    >>> R.dup_monic(3*x**2 + 4*x + 2)
    x**2 + 4/3*x + 2/3

    """
    if not f:
        return f

    lc = dup_LC(f, K)

    if K.is_one(lc):
        return f
    else:
        return dup_exquo_ground(f, lc, K)


def dmp_ground_monic(f, u, K):
    """
    Divide all coefficients by ``LC(f)`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 3*x**2*y + 6*x**2 + 3*x*y + 9*y + 3

    >>> R.dmp_ground_monic(f)
    x**2*y + 2*x**2 + x*y + 3*y + 1

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 3*x**2*y + 8*x**2 + 5*x*y + 6*x + 2*y + 3

    >>> R.dmp_ground_monic(f)
    x**2*y + 8/3*x**2 + 5/3*x*y + 2*x + 2/3*y + 1

    """
    if not u:
        return dup_monic(f, K)

    if dmp_zero_p(f, u):
        return f

    lc = dmp_ground_LC(f, u, K)

    if K.is_one(lc):
        return f
    else:
        return dmp_exquo_ground(f, lc, u, K)


def dup_content(f, K):
    """
    Compute the GCD of coefficients of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_content(f)
    2

    >>> R, x = ring("x", QQ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_content(f)
    2

    """
    from sympy.polys.domains import QQ

    if not f:
        return K.zero

    cont = K.zero

    if K == QQ:
        for c in f:
            cont = K.gcd(cont, c)
    else:
        for c in f:
            cont = K.gcd(cont, c)

            if K.is_one(cont):
                break

    return cont


def dmp_ground_content(f, u, K):
    """
    Compute the GCD of coefficients of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    """
    from sympy.polys.domains import QQ

    if not u:
        return dup_content(f, K)

    if dmp_zero_p(f, u):
        return K.zero

    cont, v = K.zero, u - 1

    if K == QQ:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))
    else:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))

            if K.is_one(cont):
                break

    return cont


def dup_primitive(f, K):
    """
    Compute content and the primitive form of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_primitive(f)
    (2, 3*x**2 + 4*x + 6)

    >>> R, x = ring("x", QQ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_primitive(f)
    (2, 3*x**2 + 4*x + 6)

    """
    if not f:
        return K.zero, f

    cont = dup_content(f, K)

    if K.is_one(cont):
        return cont, f
    else:
        return cont, dup_quo_ground(f, cont, K)


def dmp_ground_primitive(f, u, K):
    """
    Compute content and the primitive form of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_primitive(f)
    (2, x*y + 3*x + 2*y + 6)

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_primitive(f)
    (2, x*y + 3*x + 2*y + 6)

    """
    if not u:
        return dup_primitive(f, K)

    if dmp_zero_p(f, u):
        return K.zero, f

    cont = dmp_ground_content(f, u, K)

    if K.is_one(cont):
        return cont, f
    else:
        return cont, dmp_quo_ground(f, cont, u, K)


def dup_extract(f, g, K):
    """
    Extract common content from a pair of polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_extract(6*x**2 + 12*x + 18, 4*x**2 + 8*x + 12)
    (2, 3*x**2 + 6*x + 9, 2*x**2 + 4*x + 6)

    """
    fc = dup_content(f, K)
    gc = dup_content(g, K)

    gcd = K.gcd(fc, gc)

    if not K.is_one(gcd):
        f = dup_quo_ground(f, gcd, K)
        g = dup_quo_ground(g, gcd, K)

    return gcd, f, g


def dmp_ground_extract(f, g, u, K):
    """
    Extract common content from a pair of polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_ground_extract(6*x*y + 12*x + 18, 4*x*y + 8*x + 12)
    (2, 3*x*y + 6*x + 9, 2*x*y + 4*x + 6)

    """
    fc = dmp_ground_content(f, u, K)
    gc = dmp_ground_content(g, u, K)

    gcd = K.gcd(fc, gc)

    if not K.is_one(gcd):
        f = dmp_quo_ground(f, gcd, u, K)
        g = dmp_quo_ground(g, gcd, u, K)

    return gcd, f, g


def dup_real_imag(f, K):
    """
    Find ``f1`` and ``f2``, such that ``f(x+I*y) = f1(x,y) + f2(x,y)*I``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dup_real_imag(x**3 + x**2 + x + 1)
    (x**3 + x**2 - 3*x*y**2 + x - y**2 + 1, 3*x**2*y + 2*x*y - y**3 + y)

    >>> from sympy.abc import x, y, z
    >>> from sympy import I
    >>> (z**3 + z**2 + z + 1).subs(z, x+I*y).expand().collect(I)
    x**3 + x**2 - 3*x*y**2 + x - y**2 + I*(3*x**2*y + 2*x*y - y**3 + y) + 1

    """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("computing real and imaginary parts is not supported over %s" % K)

    f1 = dmp_zero(1)
    f2 = dmp_zero(1)

    if not f:
        return f1, f2

    g = [[[K.one, K.zero]], [[K.one], []]]
    h = dmp_ground(f[0], 2)

    for c in f[1:]:
        h = dmp_mul(h, g, 2, K)
        h = dmp_add_term(h, dmp_ground(c, 1), 0, 2, K)

    H = dup_to_raw_dict(h)

    for k, h in H.items():
        m = k % 4

        if not m:
            f1 = dmp_add(f1, h, 1, K)
        elif m == 1:
            f2 = dmp_add(f2, h, 1, K)
        elif m == 2:
            f1 = dmp_sub(f1, h, 1, K)
        else:
            f2 = dmp_sub(f2, h, 1, K)

    return f1, f2


def dup_mirror(f, K):
    """
    Evaluate efficiently the composition ``f(-x)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mirror(x**3 + 2*x**2 - 4*x + 2)
    -x**3 + 2*x**2 + 4*x + 2

    """
    f = list(f)

    for i in range(len(f) - 2, -1, -2):
        f[i] = -f[i]

    return f


def dup_scale(f, a, K):
    """
    Evaluate efficiently composition ``f(a*x)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_scale(x**2 - 2*x + 1, ZZ(2))
    4*x**2 - 4*x + 1

    """
    f, n, b = list(f), len(f) - 1, a

    for i in range(n - 1, -1, -1):
        f[i], b = b*f[i], b*a

    return f


def dup_shift(f, a, K):
    """
    Evaluate efficiently Taylor shift ``f(x + a)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_shift(x**2 - 2*x + 1, ZZ(2))
    x**2 + 2*x + 1

    """
    f, n = list(f), len(f) - 1

    for i in range(n, 0, -1):
        for j in range(0, i):
            f[j + 1] += a*f[j]

    return f


def dmp_shift(f, a, u, K):
    """
    Evaluate efficiently Taylor shift ``f(X + A)`` in ``K[X]``.

    Examples
    ========

    >>> from sympy import symbols, ring, ZZ
    >>> x, y = symbols('x y')
    >>> R, _, _ = ring([x, y], ZZ)

    >>> p = x**2*y + 2*x*y + 3*x + 4*y + 5

    >>> R.dmp_shift(R(p), [ZZ(1), ZZ(2)])
    x**2*y + 2*x**2 + 4*x*y + 11*x + 7*y + 22

    >>> p.subs({x: x + 1, y: y + 2}).expand()
    x**2*y + 2*x**2 + 4*x*y + 11*x + 7*y + 22
    """
    if not u:
        return dup_shift(f, a[0], K)

    if dmp_zero_p(f, u):
        return f

    a0, a1 = a[0], a[1:]

    if any(a1):
        f = [ dmp_shift(c, a1, u-1, K) for c in f ]
    else:
        f = list(f)

    if a0:
        n = len(f) - 1

        for i in range(n, 0, -1):
            for j in range(0, i):
                afj = dmp_mul_ground(f[j], a0, u-1, K)
                f[j + 1] = dmp_add(f[j + 1], afj, u-1, K)

    return dmp_strip(f, u)


def dup_transform(f, p, q, K):
    """
    Evaluate functional transformation ``q**n * f(p/q)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_transform(x**2 - 2*x + 1, x**2 + 1, x - 1)
    x**4 - 2*x**3 + 5*x**2 - 4*x + 4

    """
    if not f:
        return []

    n = len(f) - 1
    h, Q = [f[0]], [[K.one]]

    for i in range(0, n):
        Q.append(dup_mul(Q[-1], q, K))

    for c, q in zip(f[1:], Q[1:]):
        h = dup_mul(h, p, K)
        q = dup_mul_ground(q, c, K)
        h = dup_add(h, q, K)

    return h


def dup_compose(f, g, K):
    """
    Evaluate functional composition ``f(g)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_compose(x**2 + x, x - 1)
    x**2 - x

    """
    if len(g) <= 1:
        return dup_strip([dup_eval(f, dup_LC(g, K), K)])

    if not f:
        return []

    h = [f[0]]

    for c in f[1:]:
        h = dup_mul(h, g, K)
        h = dup_add_term(h, c, 0, K)

    return h


def dmp_compose(f, g, u, K):
    """
    Evaluate functional composition ``f(g)`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_compose(x*y + 2*x + y, y)
    y**2 + 3*y

    """
    if not u:
        return dup_compose(f, g, K)

    if dmp_zero_p(f, u):
        return f

    h = [f[0]]

    for c in f[1:]:
        h = dmp_mul(h, g, u, K)
        h = dmp_add_term(h, c, 0, u, K)

    return h


def _dup_right_decompose(f, s, K):
    """Helper function for :func:`_dup_decompose`."""
    n = len(f) - 1
    lc = dup_LC(f, K)

    f = dup_to_raw_dict(f)
    g = { s: K.one }

    r = n // s

    for i in range(1, s):
        coeff = K.zero

        for j in range(0, i):
            if not n + j - i in f:
                continue

            if not s - j in g:
                continue

            fc, gc = f[n + j - i], g[s - j]
            coeff += (i - r*j)*fc*gc

        g[s - i] = K.quo(coeff, i*r*lc)

    return dup_from_raw_dict(g, K)


def _dup_left_decompose(f, h, K):
    """Helper function for :func:`_dup_decompose`."""
    g, i = {}, 0

    while f:
        q, r = dup_div(f, h, K)

        if dup_degree(r) > 0:
            return None
        else:
            g[i] = dup_LC(r, K)
            f, i = q, i + 1

    return dup_from_raw_dict(g, K)


def _dup_decompose(f, K):
    """Helper function for :func:`dup_decompose`."""
    df = len(f) - 1

    for s in range(2, df):
        if df % s != 0:
            continue

        h = _dup_right_decompose(f, s, K)

        if h is not None:
            g = _dup_left_decompose(f, h, K)

            if g is not None:
                return g, h

    return None


def dup_decompose(f, K):
    """
    Computes functional decomposition of ``f`` in ``K[x]``.

    Given a univariate polynomial ``f`` with coefficients in a field of
    characteristic zero, returns list ``[f_1, f_2, ..., f_n]``, where::

              f = f_1 o f_2 o ... f_n = f_1(f_2(... f_n))

    and ``f_2, ..., f_n`` are monic and homogeneous polynomials of at
    least second degree.

    Unlike factorization, complete functional decompositions of
    polynomials are not unique, consider examples:

    1. ``f o g = f(x + b) o (g - b)``
    2. ``x**n o x**m = x**m o x**n``
    3. ``T_n o T_m = T_m o T_n``

    where ``T_n`` and ``T_m`` are Chebyshev polynomials.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_decompose(x**4 - 2*x**3 + x**2)
    [x**2, x**2 - x]

    References
    ==========

    .. [1] [Kozen89]_

    """
    F = []

    while True:
        result = _dup_decompose(f, K)

        if result is not None:
            f, h = result
            F = [h] + F
        else:
            break

    return [f] + F


def dmp_alg_inject(f, u, K):
    """
    Convert polynomial from ``K(a)[X]`` to ``K[a,X]``.

    Examples
    ========

    >>> from sympy.polys.densetools import dmp_alg_inject
    >>> from sympy import QQ, sqrt

    >>> K = QQ.algebraic_field(sqrt(2))

    >>> p = [K.from_sympy(sqrt(2)), K.zero, K.one]
    >>> P, lev, dom = dmp_alg_inject(p, 0, K)
    >>> P
    [[1, 0, 0], [1]]
    >>> lev
    1
    >>> dom
    QQ

    """
    if K.is_GaussianRing or K.is_GaussianField:
        return _dmp_alg_inject_gaussian(f, u, K)
    elif K.is_Algebraic:
        return _dmp_alg_inject_alg(f, u, K)
    else:
        raise DomainError('computation can be done only in an algebraic domain')


def _dmp_alg_inject_gaussian(f, u, K):
    """Helper function for :func:`dmp_alg_inject`."""
    f, h = dmp_to_dict(f, u), {}

    for f_monom, g in f.items():
        x, y = g.x, g.y
        if x:
            h[(0,) + f_monom] = x
        if y:
            h[(1,) + f_monom] = y

    F = dmp_from_dict(h, u + 1, K.dom)

    return F, u + 1, K.dom


def _dmp_alg_inject_alg(f, u, K):
    """Helper function for :func:`dmp_alg_inject`."""
    f, h = dmp_to_dict(f, u), {}

    for f_monom, g in f.items():
        for g_monom, c in g.to_dict().items():
            h[g_monom + f_monom] = c

    F = dmp_from_dict(h, u + 1, K.dom)

    return F, u + 1, K.dom


def dmp_lift(f, u, K):
    """
    Convert algebraic coefficients to integers in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x = ring("x", K)

    >>> f = x**2 + K([QQ(1), QQ(0)])*x + K([QQ(2), QQ(0)])

    >>> R.dmp_lift(f)
    x**4 + x**2 + 4*x + 4

    """
    # Circular import. Probably dmp_lift should be moved to euclidtools
    from .euclidtools import dmp_resultant

    F, v, K2 = dmp_alg_inject(f, u, K)

    p_a = K.mod.to_list()
    P_A = dmp_include(p_a, list(range(1, v + 1)), 0, K2)

    return dmp_resultant(F, P_A, v, K2)


def dup_sign_variations(f, K):
    """
    Compute the number of sign variations of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sign_variations(x**4 - x**2 - x + 1)
    2

    """
    def is_negative_sympy(a):
        if not a:
            # XXX: requires zero equivalence testing in the domain
            return False
        else:
            # XXX: This is inefficient. It should not be necessary to use a
            # symbolic expression here at least for algebraic fields. If the
            # domain elements can be numerically evaluated to real values with
            # precision then this should work. We first need to rule out zero
            # elements though.
            return bool(K.to_sympy(a) < 0)

    # XXX: There should be a way to check for real numeric domains and
    # Domain.is_negative should be fixed to handle all real numeric domains.
    # It should not be necessary to special case all these different domains
    # in this otherwise generic function.
    if K.is_ZZ or K.is_QQ or K.is_RR:
        is_negative = K.is_negative
    elif K.is_AlgebraicField and K.ext.is_comparable:
        is_negative = is_negative_sympy
    elif ((K.is_PolynomialRing or K.is_FractionField) and len(K.symbols) == 1 and
          (K.dom.is_ZZ or K.dom.is_QQ or K.is_AlgebraicField) and
          K.symbols[0].is_transcendental and K.symbols[0].is_comparable):
        # We can handle a polynomial ring like QQ[E] if there is a single
        # transcendental generator because then zero equivalence is assured.
        is_negative = is_negative_sympy
    else:
        raise DomainError("sign variation counting not supported over %s" % K)

    prev, k = K.zero, 0

    for coeff in f:
        if is_negative(coeff*prev):
            k += 1

        if coeff:
            prev = coeff

    return k


def dup_clear_denoms(f, K0, K1=None, convert=False):
    """
    Clear denominators, i.e. transform ``K_0`` to ``K_1``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x + QQ(1,3)

    >>> R.dup_clear_denoms(f, convert=False)
    (6, 3*x + 2)
    >>> R.dup_clear_denoms(f, convert=True)
    (6, 3*x + 2)

    """
    if K1 is None:
        if K0.has_assoc_Ring:
            K1 = K0.get_ring()
        else:
            K1 = K0

    common = K1.one

    for c in f:
        common = K1.lcm(common, K0.denom(c))

    if K1.is_one(common):
        if not convert:
            return common, f
        else:
            return common, dup_convert(f, K0, K1)

    # Use quo rather than exquo to handle inexact domains by discarding the
    # remainder.
    f = [K0.numer(c)*K1.quo(common, K0.denom(c)) for c in f]

    if not convert:
        return common, dup_convert(f, K1, K0)
    else:
        return common, f


def _rec_clear_denoms(g, v, K0, K1):
    """Recursive helper for :func:`dmp_clear_denoms`."""
    common = K1.one

    if not v:
        for c in g:
            common = K1.lcm(common, K0.denom(c))
    else:
        w = v - 1

        for c in g:
            common = K1.lcm(common, _rec_clear_denoms(c, w, K0, K1))

    return common


def dmp_clear_denoms(f, u, K0, K1=None, convert=False):
    """
    Clear denominators, i.e. transform ``K_0`` to ``K_1``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> f = QQ(1,2)*x + QQ(1,3)*y + 1

    >>> R.dmp_clear_denoms(f, convert=False)
    (6, 3*x + 2*y + 6)
    >>> R.dmp_clear_denoms(f, convert=True)
    (6, 3*x + 2*y + 6)

    """
    if not u:
        return dup_clear_denoms(f, K0, K1, convert=convert)

    if K1 is None:
        if K0.has_assoc_Ring:
            K1 = K0.get_ring()
        else:
            K1 = K0

    common = _rec_clear_denoms(f, u, K0, K1)

    if not K1.is_one(common):
        f = dmp_mul_ground(f, common, u, K0)

    if not convert:
        return common, f
    else:
        return common, dmp_convert(f, u, K0, K1)


def dup_revert(f, n, K):
    """
    Compute ``f**(-1)`` mod ``x**n`` using Newton iteration.

    This function computes first ``2**n`` terms of a polynomial that
    is a result of inversion of a polynomial modulo ``x**n``. This is
    useful to efficiently compute series expansion of ``1/f``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = -QQ(1,720)*x**6 + QQ(1,24)*x**4 - QQ(1,2)*x**2 + 1

    >>> R.dup_revert(f, 8)
    61/720*x**6 + 5/24*x**4 + 1/2*x**2 + 1

    """
    g = [K.revert(dup_TC(f, K))]
    h = [K.one, K.zero, K.zero]

    N = int(_ceil(_log2(n)))

    for i in range(1, N + 1):
        a = dup_mul_ground(g, K(2), K)
        b = dup_mul(f, dup_sqr(g, K), K)
        g = dup_rem(dup_sub(a, b, K), h, K)
        h = dup_lshift(h, dup_degree(h), K)

    return g


def dmp_revert(f, g, u, K):
    """
    Compute ``f**(-1)`` mod ``x**n`` using Newton iteration.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    """
    if not u:
        return dup_revert(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)