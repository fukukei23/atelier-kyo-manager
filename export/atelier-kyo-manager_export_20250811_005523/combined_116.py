
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\helpers.py ===
from __future__ import annotations

import importlib.util
import os
import sys
import typing as t
from datetime import datetime
from functools import cache
from functools import update_wrapper

import werkzeug.utils
from werkzeug.exceptions import abort as _wz_abort
from werkzeug.utils import redirect as _wz_redirect
from werkzeug.wrappers import Response as BaseResponse

from .globals import _cv_request
from .globals import current_app
from .globals import request
from .globals import request_ctx
from .globals import session
from .signals import message_flashed

if t.TYPE_CHECKING:  # pragma: no cover
    from .wrappers import Response


def get_debug_flag() -> bool:
    """Get whether debug mode should be enabled for the app, indicated by the
    :envvar:`FLASK_DEBUG` environment variable. The default is ``False``.
    """
    val = os.environ.get("FLASK_DEBUG")
    return bool(val and val.lower() not in {"0", "false", "no"})


def get_load_dotenv(default: bool = True) -> bool:
    """Get whether the user has disabled loading default dotenv files by
    setting :envvar:`FLASK_SKIP_DOTENV`. The default is ``True``, load
    the files.

    :param default: What to return if the env var isn't set.
    """
    val = os.environ.get("FLASK_SKIP_DOTENV")

    if not val:
        return default

    return val.lower() in ("0", "false", "no")


@t.overload
def stream_with_context(
    generator_or_function: t.Iterator[t.AnyStr],
) -> t.Iterator[t.AnyStr]: ...


@t.overload
def stream_with_context(
    generator_or_function: t.Callable[..., t.Iterator[t.AnyStr]],
) -> t.Callable[[t.Iterator[t.AnyStr]], t.Iterator[t.AnyStr]]: ...


def stream_with_context(
    generator_or_function: t.Iterator[t.AnyStr] | t.Callable[..., t.Iterator[t.AnyStr]],
) -> t.Iterator[t.AnyStr] | t.Callable[[t.Iterator[t.AnyStr]], t.Iterator[t.AnyStr]]:
    """Request contexts disappear when the response is started on the server.
    This is done for efficiency reasons and to make it less likely to encounter
    memory leaks with badly written WSGI middlewares.  The downside is that if
    you are using streamed responses, the generator cannot access request bound
    information any more.

    This function however can help you keep the context around for longer::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            @stream_with_context
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(generate())

    Alternatively it can also be used around a specific generator::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(stream_with_context(generate()))

    .. versionadded:: 0.9
    """
    try:
        gen = iter(generator_or_function)  # type: ignore[arg-type]
    except TypeError:

        def decorator(*args: t.Any, **kwargs: t.Any) -> t.Any:
            gen = generator_or_function(*args, **kwargs)  # type: ignore[operator]
            return stream_with_context(gen)

        return update_wrapper(decorator, generator_or_function)  # type: ignore[arg-type]

    def generator() -> t.Iterator[t.AnyStr | None]:
        ctx = _cv_request.get(None)
        if ctx is None:
            raise RuntimeError(
                "'stream_with_context' can only be used when a request"
                " context is active, such as in a view function."
            )
        with ctx:
            # Dummy sentinel.  Has to be inside the context block or we're
            # not actually keeping the context around.
            yield None

            # The try/finally is here so that if someone passes a WSGI level
            # iterator in we're still running the cleanup logic.  Generators
            # don't need that because they are closed on their destruction
            # automatically.
            try:
                yield from gen
            finally:
                if hasattr(gen, "close"):
                    gen.close()

    # The trick is to start the generator.  Then the code execution runs until
    # the first dummy None is yielded at which point the context was already
    # pushed.  This item is discarded.  Then when the iteration continues the
    # real generator is executed.
    wrapped_g = generator()
    next(wrapped_g)
    return wrapped_g  # type: ignore[return-value]


def make_response(*args: t.Any) -> Response:
    """Sometimes it is necessary to set additional headers in a view.  Because
    views do not have to return response objects but can return a value that
    is converted into a response object by Flask itself, it becomes tricky to
    add headers to it.  This function can be called instead of using a return
    and you will get a response object which you can use to attach headers.

    If view looked like this and you want to add a new header::

        def index():
            return render_template('index.html', foo=42)

    You can now do something like this::

        def index():
            response = make_response(render_template('index.html', foo=42))
            response.headers['X-Parachutes'] = 'parachutes are cool'
            return response

    This function accepts the very same arguments you can return from a
    view function.  This for example creates a response with a 404 error
    code::

        response = make_response(render_template('not_found.html'), 404)

    The other use case of this function is to force the return value of a
    view function into a response which is helpful with view
    decorators::

        response = make_response(view_function())
        response.headers['X-Parachutes'] = 'parachutes are cool'

    Internally this function does the following things:

    -   if no arguments are passed, it creates a new response argument
    -   if one argument is passed, :meth:`flask.Flask.make_response`
        is invoked with it.
    -   if more than one argument is passed, the arguments are passed
        to the :meth:`flask.Flask.make_response` function as tuple.

    .. versionadded:: 0.6
    """
    if not args:
        return current_app.response_class()
    if len(args) == 1:
        args = args[0]
    return current_app.make_response(args)


def url_for(
    endpoint: str,
    *,
    _anchor: str | None = None,
    _method: str | None = None,
    _scheme: str | None = None,
    _external: bool | None = None,
    **values: t.Any,
) -> str:
    """Generate a URL to the given endpoint with the given values.

    This requires an active request or application context, and calls
    :meth:`current_app.url_for() <flask.Flask.url_for>`. See that method
    for full documentation.

    :param endpoint: The endpoint name associated with the URL to
        generate. If this starts with a ``.``, the current blueprint
        name (if any) will be used.
    :param _anchor: If given, append this as ``#anchor`` to the URL.
    :param _method: If given, generate the URL associated with this
        method for the endpoint.
    :param _scheme: If given, the URL will have this scheme if it is
        external.
    :param _external: If given, prefer the URL to be internal (False) or
        require it to be external (True). External URLs include the
        scheme and domain. When not in an active request, URLs are
        external by default.
    :param values: Values to use for the variable parts of the URL rule.
        Unknown keys are appended as query string arguments, like
        ``?a=b&c=d``.

    .. versionchanged:: 2.2
        Calls ``current_app.url_for``, allowing an app to override the
        behavior.

    .. versionchanged:: 0.10
       The ``_scheme`` parameter was added.

    .. versionchanged:: 0.9
       The ``_anchor`` and ``_method`` parameters were added.

    .. versionchanged:: 0.9
       Calls ``app.handle_url_build_error`` on build errors.
    """
    return current_app.url_for(
        endpoint,
        _anchor=_anchor,
        _method=_method,
        _scheme=_scheme,
        _external=_external,
        **values,
    )


def redirect(
    location: str, code: int = 302, Response: type[BaseResponse] | None = None
) -> BaseResponse:
    """Create a redirect response object.

    If :data:`~flask.current_app` is available, it will use its
    :meth:`~flask.Flask.redirect` method, otherwise it will use
    :func:`werkzeug.utils.redirect`.

    :param location: The URL to redirect to.
    :param code: The status code for the redirect.
    :param Response: The response class to use. Not used when
        ``current_app`` is active, which uses ``app.response_class``.

    .. versionadded:: 2.2
        Calls ``current_app.redirect`` if available instead of always
        using Werkzeug's default ``redirect``.
    """
    if current_app:
        return current_app.redirect(location, code=code)

    return _wz_redirect(location, code=code, Response=Response)


def abort(code: int | BaseResponse, *args: t.Any, **kwargs: t.Any) -> t.NoReturn:
    """Raise an :exc:`~werkzeug.exceptions.HTTPException` for the given
    status code.

    If :data:`~flask.current_app` is available, it will call its
    :attr:`~flask.Flask.aborter` object, otherwise it will use
    :func:`werkzeug.exceptions.abort`.

    :param code: The status code for the exception, which must be
        registered in ``app.aborter``.
    :param args: Passed to the exception.
    :param kwargs: Passed to the exception.

    .. versionadded:: 2.2
        Calls ``current_app.aborter`` if available instead of always
        using Werkzeug's default ``abort``.
    """
    if current_app:
        current_app.aborter(code, *args, **kwargs)

    _wz_abort(code, *args, **kwargs)


def get_template_attribute(template_name: str, attribute: str) -> t.Any:
    """Loads a macro (or variable) a template exports.  This can be used to
    invoke a macro from within Python code.  If you for example have a
    template named :file:`_cider.html` with the following contents:

    .. sourcecode:: html+jinja

       {% macro hello(name) %}Hello {{ name }}!{% endmacro %}

    You can access this from Python code like this::

        hello = get_template_attribute('_cider.html', 'hello')
        return hello('World')

    .. versionadded:: 0.2

    :param template_name: the name of the template
    :param attribute: the name of the variable of macro to access
    """
    return getattr(current_app.jinja_env.get_template(template_name).module, attribute)


def flash(message: str, category: str = "message") -> None:
    """Flashes a message to the next request.  In order to remove the
    flashed message from the session and to display it to the user,
    the template has to call :func:`get_flashed_messages`.

    .. versionchanged:: 0.3
       `category` parameter added.

    :param message: the message to be flashed.
    :param category: the category for the message.  The following values
                     are recommended: ``'message'`` for any kind of message,
                     ``'error'`` for errors, ``'info'`` for information
                     messages and ``'warning'`` for warnings.  However any
                     kind of string can be used as category.
    """
    # Original implementation:
    #
    #     session.setdefault('_flashes', []).append((category, message))
    #
    # This assumed that changes made to mutable structures in the session are
    # always in sync with the session object, which is not true for session
    # implementations that use external storage for keeping their keys/values.
    flashes = session.get("_flashes", [])
    flashes.append((category, message))
    session["_flashes"] = flashes
    app = current_app._get_current_object()  # type: ignore
    message_flashed.send(
        app,
        _async_wrapper=app.ensure_sync,
        message=message,
        category=category,
    )


def get_flashed_messages(
    with_categories: bool = False, category_filter: t.Iterable[str] = ()
) -> list[str] | list[tuple[str, str]]:
    """Pulls all flashed messages from the session and returns them.
    Further calls in the same request to the function will return
    the same messages.  By default just the messages are returned,
    but when `with_categories` is set to ``True``, the return value will
    be a list of tuples in the form ``(category, message)`` instead.

    Filter the flashed messages to one or more categories by providing those
    categories in `category_filter`.  This allows rendering categories in
    separate html blocks.  The `with_categories` and `category_filter`
    arguments are distinct:

    * `with_categories` controls whether categories are returned with message
      text (``True`` gives a tuple, where ``False`` gives just the message text).
    * `category_filter` filters the messages down to only those matching the
      provided categories.

    See :doc:`/patterns/flashing` for examples.

    .. versionchanged:: 0.3
       `with_categories` parameter added.

    .. versionchanged:: 0.9
        `category_filter` parameter added.

    :param with_categories: set to ``True`` to also receive categories.
    :param category_filter: filter of categories to limit return values.  Only
                            categories in the list will be returned.
    """
    flashes = request_ctx.flashes
    if flashes is None:
        flashes = session.pop("_flashes") if "_flashes" in session else []
        request_ctx.flashes = flashes
    if category_filter:
        flashes = list(filter(lambda f: f[0] in category_filter, flashes))
    if not with_categories:
        return [x[1] for x in flashes]
    return flashes


def _prepare_send_file_kwargs(**kwargs: t.Any) -> dict[str, t.Any]:
    if kwargs.get("max_age") is None:
        kwargs["max_age"] = current_app.get_send_file_max_age

    kwargs.update(
        environ=request.environ,
        use_x_sendfile=current_app.config["USE_X_SENDFILE"],
        response_class=current_app.response_class,
        _root_path=current_app.root_path,  # type: ignore
    )
    return kwargs


def send_file(
    path_or_file: os.PathLike[t.AnyStr] | str | t.BinaryIO,
    mimetype: str | None = None,
    as_attachment: bool = False,
    download_name: str | None = None,
    conditional: bool = True,
    etag: bool | str = True,
    last_modified: datetime | int | float | None = None,
    max_age: None | (int | t.Callable[[str | None], int | None]) = None,
) -> Response:
    """Send the contents of a file to the client.

    The first argument can be a file path or a file-like object. Paths
    are preferred in most cases because Werkzeug can manage the file and
    get extra information from the path. Passing a file-like object
    requires that the file is opened in binary mode, and is mostly
    useful when building a file in memory with :class:`io.BytesIO`.

    Never pass file paths provided by a user. The path is assumed to be
    trusted, so a user could craft a path to access a file you didn't
    intend. Use :func:`send_from_directory` to safely serve
    user-requested paths from within a directory.

    If the WSGI server sets a ``file_wrapper`` in ``environ``, it is
    used, otherwise Werkzeug's built-in wrapper is used. Alternatively,
    if the HTTP server supports ``X-Sendfile``, configuring Flask with
    ``USE_X_SENDFILE = True`` will tell the server to send the given
    path, which is much more efficient than reading it in Python.

    :param path_or_file: The path to the file to send, relative to the
        current working directory if a relative path is given.
        Alternatively, a file-like object opened in binary mode. Make
        sure the file pointer is seeked to the start of the data.
    :param mimetype: The MIME type to send for the file. If not
        provided, it will try to detect it from the file name.
    :param as_attachment: Indicate to a browser that it should offer to
        save the file instead of displaying it.
    :param download_name: The default name browsers will use when saving
        the file. Defaults to the passed file name.
    :param conditional: Enable conditional and range responses based on
        request headers. Requires passing a file path and ``environ``.
    :param etag: Calculate an ETag for the file, which requires passing
        a file path. Can also be a string to use instead.
    :param last_modified: The last modified time to send for the file,
        in seconds. If not provided, it will try to detect it from the
        file path.
    :param max_age: How long the client should cache the file, in
        seconds. If set, ``Cache-Control`` will be ``public``, otherwise
        it will be ``no-cache`` to prefer conditional caching.

    .. versionchanged:: 2.0
        ``download_name`` replaces the ``attachment_filename``
        parameter. If ``as_attachment=False``, it is passed with
        ``Content-Disposition: inline`` instead.

    .. versionchanged:: 2.0
        ``max_age`` replaces the ``cache_timeout`` parameter.
        ``conditional`` is enabled and ``max_age`` is not set by
        default.

    .. versionchanged:: 2.0
        ``etag`` replaces the ``add_etags`` parameter. It can be a
        string to use instead of generating one.

    .. versionchanged:: 2.0
        Passing a file-like object that inherits from
        :class:`~io.TextIOBase` will raise a :exc:`ValueError` rather
        than sending an empty file.

    .. versionadded:: 2.0
        Moved the implementation to Werkzeug. This is now a wrapper to
        pass some Flask-specific arguments.

    .. versionchanged:: 1.1
        ``filename`` may be a :class:`~os.PathLike` object.

    .. versionchanged:: 1.1
        Passing a :class:`~io.BytesIO` object supports range requests.

    .. versionchanged:: 1.0.3
        Filenames are encoded with ASCII instead of Latin-1 for broader
        compatibility with WSGI servers.

    .. versionchanged:: 1.0
        UTF-8 filenames as specified in :rfc:`2231` are supported.

    .. versionchanged:: 0.12
        The filename is no longer automatically inferred from file
        objects. If you want to use automatic MIME and etag support,
        pass a filename via ``filename_or_fp`` or
        ``attachment_filename``.

    .. versionchanged:: 0.12
        ``attachment_filename`` is preferred over ``filename`` for MIME
        detection.

    .. versionchanged:: 0.9
        ``cache_timeout`` defaults to
        :meth:`Flask.get_send_file_max_age`.

    .. versionchanged:: 0.7
        MIME guessing and etag support for file-like objects was
        removed because it was unreliable. Pass a filename if you are
        able to, otherwise attach an etag yourself.

    .. versionchanged:: 0.5
        The ``add_etags``, ``cache_timeout`` and ``conditional``
        parameters were added. The default behavior is to add etags.

    .. versionadded:: 0.2
    """
    return werkzeug.utils.send_file(  # type: ignore[return-value]
        **_prepare_send_file_kwargs(
            path_or_file=path_or_file,
            environ=request.environ,
            mimetype=mimetype,
            as_attachment=as_attachment,
            download_name=download_name,
            conditional=conditional,
            etag=etag,
            last_modified=last_modified,
            max_age=max_age,
        )
    )


def send_from_directory(
    directory: os.PathLike[str] | str,
    path: os.PathLike[str] | str,
    **kwargs: t.Any,
) -> Response:
    """Send a file from within a directory using :func:`send_file`.

    .. code-block:: python

        @app.route("/uploads/<path:name>")
        def download_file(name):
            return send_from_directory(
                app.config['UPLOAD_FOLDER'], name, as_attachment=True
            )

    This is a secure way to serve files from a folder, such as static
    files or uploads. Uses :func:`~werkzeug.security.safe_join` to
    ensure the path coming from the client is not maliciously crafted to
    point outside the specified directory.

    If the final path does not point to an existing regular file,
    raises a 404 :exc:`~werkzeug.exceptions.NotFound` error.

    :param directory: The directory that ``path`` must be located under,
        relative to the current application's root path. This *must not*
        be a value provided by the client, otherwise it becomes insecure.
    :param path: The path to the file to send, relative to
        ``directory``.
    :param kwargs: Arguments to pass to :func:`send_file`.

    .. versionchanged:: 2.0
        ``path`` replaces the ``filename`` parameter.

    .. versionadded:: 2.0
        Moved the implementation to Werkzeug. This is now a wrapper to
        pass some Flask-specific arguments.

    .. versionadded:: 0.5
    """
    return werkzeug.utils.send_from_directory(  # type: ignore[return-value]
        directory, path, **_prepare_send_file_kwargs(**kwargs)
    )


def get_root_path(import_name: str) -> str:
    """Find the root path of a package, or the path that contains a
    module. If it cannot be found, returns the current working
    directory.

    Not to be confused with the value returned by :func:`find_package`.

    :meta private:
    """
    # Module already imported and has a file attribute. Use that first.
    mod = sys.modules.get(import_name)

    if mod is not None and hasattr(mod, "__file__") and mod.__file__ is not None:
        return os.path.dirname(os.path.abspath(mod.__file__))

    # Next attempt: check the loader.
    try:
        spec = importlib.util.find_spec(import_name)

        if spec is None:
            raise ValueError
    except (ImportError, ValueError):
        loader = None
    else:
        loader = spec.loader

    # Loader does not exist or we're referring to an unloaded main
    # module or a main module without path (interactive sessions), go
    # with the current working directory.
    if loader is None:
        return os.getcwd()

    if hasattr(loader, "get_filename"):
        filepath = loader.get_filename(import_name)  # pyright: ignore
    else:
        # Fall back to imports.
        __import__(import_name)
        mod = sys.modules[import_name]
        filepath = getattr(mod, "__file__", None)

        # If we don't have a file path it might be because it is a
        # namespace package. In this case pick the root path from the
        # first module that is contained in the package.
        if filepath is None:
            raise RuntimeError(
                "No root path can be found for the provided module"
                f" {import_name!r}. This can happen because the module"
                " came from an import hook that does not provide file"
                " name information or because it's a namespace package."
                " In this case the root path needs to be explicitly"
                " provided."
            )

    # filepath is import_name.py for a module, or __init__.py for a package.
    return os.path.dirname(os.path.abspath(filepath))  # type: ignore[no-any-return]


@cache
def _split_blueprint_path(name: str) -> list[str]:
    out: list[str] = [name]

    if "." in name:
        out.extend(_split_blueprint_path(name.rpartition(".")[0]))

    return out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tseries\holiday.py ===
from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
import warnings

from dateutil.relativedelta import (
    FR,
    MO,
    SA,
    SU,
    TH,
    TU,
    WE,
)
import numpy as np

from pandas.errors import PerformanceWarning

from pandas import (
    DateOffset,
    DatetimeIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)

from pandas.tseries.offsets import (
    Day,
    Easter,
)


def next_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday, use Monday instead
    """
    if dt.weekday() == 5:
        return dt + timedelta(2)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def next_monday_or_tuesday(dt: datetime) -> datetime:
    """
    For second holiday of two adjacent ones!
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday or Monday, use following Tuesday instead
    (because Monday is already taken by adjacent holiday on the day before)
    """
    dow = dt.weekday()
    if dow in (5, 6):
        return dt + timedelta(2)
    if dow == 0:
        return dt + timedelta(1)
    return dt


def previous_friday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt


def sunday_to_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def weekend_to_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Sunday or Saturday,
    use day thereafter (Monday) instead.
    Needed for holidays such as Christmas observation in Europe
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    elif dt.weekday() == 5:
        return dt + timedelta(2)
    return dt


def nearest_workday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday, use day before (Friday) instead;
    if holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def next_workday(dt: datetime) -> datetime:
    """
    returns next weekday used for observances
    """
    dt += timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt += timedelta(days=1)
    return dt


def previous_workday(dt: datetime) -> datetime:
    """
    returns previous weekday used for observances
    """
    dt -= timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt -= timedelta(days=1)
    return dt


def before_nearest_workday(dt: datetime) -> datetime:
    """
    returns previous workday after nearest workday
    """
    return previous_workday(nearest_workday(dt))


def after_nearest_workday(dt: datetime) -> datetime:
    """
    returns next workday after nearest workday
    needed for Boxing day or multiple holidays in a series
    """
    return next_workday(nearest_workday(dt))


class Holiday:
    """
    Class that defines a holiday with start/end dates and rules
    for observance.
    """

    start_date: Timestamp | None
    end_date: Timestamp | None
    days_of_week: tuple[int, ...] | None

    def __init__(
        self,
        name: str,
        year=None,
        month=None,
        day=None,
        offset=None,
        observance=None,
        start_date=None,
        end_date=None,
        days_of_week=None,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the holiday , defaults to class name
        offset : array of pandas.tseries.offsets or
                class from pandas.tseries.offsets
            computes offset from date
        observance: function
            computes when holiday is given a pandas Timestamp
        days_of_week:
            provide a tuple of days e.g  (0,1,2,3,) for Monday Through Thursday
            Monday=0,..,Sunday=6

        Examples
        --------
        >>> from dateutil.relativedelta import MO

        >>> USMemorialDay = pd.tseries.holiday.Holiday(
        ...     "Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))
        ... )
        >>> USMemorialDay
        Holiday: Memorial Day (month=5, day=31, offset=<DateOffset: weekday=MO(-1)>)

        >>> USLaborDay = pd.tseries.holiday.Holiday(
        ...     "Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))
        ... )
        >>> USLaborDay
        Holiday: Labor Day (month=9, day=1, offset=<DateOffset: weekday=MO(+1)>)

        >>> July3rd = pd.tseries.holiday.Holiday("July 3rd", month=7, day=3)
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )

        >>> NewYears = pd.tseries.holiday.Holiday(
        ...     "New Years Day", month=1,  day=1,
        ...      observance=pd.tseries.holiday.nearest_workday
        ... )
        >>> NewYears  # doctest: +SKIP
        Holiday: New Years Day (
            month=1, day=1, observance=<function nearest_workday at 0x66545e9bc440>
        )

        >>> July3rd = pd.tseries.holiday.Holiday(
        ...     "July 3rd", month=7, day=3,
        ...     days_of_week=(0, 1, 2, 3)
        ... )
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )
        """
        if offset is not None and observance is not None:
            raise NotImplementedError("Cannot use both offset and observance.")

        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = (
            Timestamp(start_date) if start_date is not None else start_date
        )
        self.end_date = Timestamp(end_date) if end_date is not None else end_date
        self.observance = observance
        assert days_of_week is None or type(days_of_week) == tuple
        self.days_of_week = days_of_week

    def __repr__(self) -> str:
        info = ""
        if self.year is not None:
            info += f"year={self.year}, "
        info += f"month={self.month}, day={self.day}, "

        if self.offset is not None:
            info += f"offset={self.offset}"

        if self.observance is not None:
            info += f"observance={self.observance}"

        repr = f"Holiday: {self.name} ({info})"
        return repr

    def dates(
        self, start_date, end_date, return_name: bool = False
    ) -> Series | DatetimeIndex:
        """
        Calculate holidays observed between start date and end date

        Parameters
        ----------
        start_date : starting date, datetime-like, optional
        end_date : ending date, datetime-like, optional
        return_name : bool, optional, default=False
            If True, return a series that has dates and holiday names.
            False will only return dates.

        Returns
        -------
        Series or DatetimeIndex
            Series if return_name is True
        """
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)

        filter_start_date = start_date
        filter_end_date = end_date

        if self.year is not None:
            dt = Timestamp(datetime(self.year, self.month, self.day))
            dti = DatetimeIndex([dt])
            if return_name:
                return Series(self.name, index=dti)
            else:
                return dti

        dates = self._reference_dates(start_date, end_date)
        holiday_dates = self._apply_rule(dates)
        if self.days_of_week is not None:
            holiday_dates = holiday_dates[
                np.isin(
                    # error: "DatetimeIndex" has no attribute "dayofweek"
                    holiday_dates.dayofweek,  # type: ignore[attr-defined]
                    self.days_of_week,
                ).ravel()
            ]

        if self.start_date is not None:
            filter_start_date = max(
                self.start_date.tz_localize(filter_start_date.tz), filter_start_date
            )
        if self.end_date is not None:
            filter_end_date = min(
                self.end_date.tz_localize(filter_end_date.tz), filter_end_date
            )
        holiday_dates = holiday_dates[
            (holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date)
        ]
        if return_name:
            return Series(self.name, index=holiday_dates)
        return holiday_dates

    def _reference_dates(
        self, start_date: Timestamp, end_date: Timestamp
    ) -> DatetimeIndex:
        """
        Get reference dates for the holiday.

        Return reference dates for the holiday also returning the year
        prior to the start_date and year following the end_date.  This ensures
        that any offsets to be applied will yield the holidays within
        the passed in dates.
        """
        if self.start_date is not None:
            start_date = self.start_date.tz_localize(start_date.tz)

        if self.end_date is not None:
            end_date = self.end_date.tz_localize(start_date.tz)

        year_offset = DateOffset(years=1)
        reference_start_date = Timestamp(
            datetime(start_date.year - 1, self.month, self.day)
        )

        reference_end_date = Timestamp(
            datetime(end_date.year + 1, self.month, self.day)
        )
        # Don't process unnecessary holidays
        dates = date_range(
            start=reference_start_date,
            end=reference_end_date,
            freq=year_offset,
            tz=start_date.tz,
        )

        return dates

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        """
        Apply the given offset/observance to a DatetimeIndex of dates.

        Parameters
        ----------
        dates : DatetimeIndex
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates with rules applied
        """
        if dates.empty:
            return dates.copy()

        if self.observance is not None:
            return dates.map(lambda d: self.observance(d))

        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                # if we are adding a non-vectorized value
                # ignore the PerformanceWarnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PerformanceWarning)
                    dates += offset
        return dates


holiday_calendars = {}


def register(cls) -> None:
    try:
        name = cls.name
    except AttributeError:
        name = cls.__name__
    holiday_calendars[name] = cls


def get_calendar(name: str):
    """
    Return an instance of a calendar based on its name.

    Parameters
    ----------
    name : str
        Calendar name to return an instance of
    """
    return holiday_calendars[name]()


class HolidayCalendarMetaClass(type):
    def __new__(cls, clsname: str, bases, attrs):
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class


class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    """
    Abstract interface to create holidays following certain rules.
    """

    rules: list[Holiday] = []
    start_date = Timestamp(datetime(1970, 1, 1))
    end_date = Timestamp(datetime(2200, 12, 31))
    _cache = None

    def __init__(self, name: str = "", rules=None) -> None:
        """
        Initializes holiday object with a given set a rules.  Normally
        classes just have the rules defined within them.

        Parameters
        ----------
        name : str
            Name of the holiday calendar, defaults to class name
        rules : array of Holiday objects
            A set of rules used to create the holidays.
        """
        super().__init__()
        if not name:
            name = type(self).__name__
        self.name = name

        if rules is not None:
            self.rules = rules

    def rule_from_name(self, name: str):
        for rule in self.rules:
            if rule.name == name:
                return rule

        return None

    def holidays(self, start=None, end=None, return_name: bool = False):
        """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
            DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(
                f"Holiday Calendar {self.name} does not have any rules specified"
            )

        if start is None:
            start = AbstractHolidayCalendar.start_date

        if end is None:
            end = AbstractHolidayCalendar.end_date

        start = Timestamp(start)
        end = Timestamp(end)

        # If we don't have a cache or the dates are outside the prior cache, we
        # get them again
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            pre_holidays = [
                rule.dates(start, end, return_name=True) for rule in self.rules
            ]
            if pre_holidays:
                # error: Argument 1 to "concat" has incompatible type
                # "List[Union[Series, DatetimeIndex]]"; expected
                # "Union[Iterable[DataFrame], Mapping[<nothing>, DataFrame]]"
                holidays = concat(pre_holidays)  # type: ignore[arg-type]
            else:
                # error: Incompatible types in assignment (expression has type
                # "Series", variable has type "DataFrame")
                holidays = Series(
                    index=DatetimeIndex([]), dtype=object
                )  # type: ignore[assignment]

            self._cache = (start, end, holidays.sort_index())

        holidays = self._cache[2]
        holidays = holidays[start:end]

        if return_name:
            return holidays
        else:
            return holidays.index

    @staticmethod
    def merge_class(base, other):
        """
        Merge holiday calendars together. The base calendar
        will take precedence to other. The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        base : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        other : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        """
        try:
            other = other.rules
        except AttributeError:
            pass

        if not isinstance(other, list):
            other = [other]
        other_holidays = {holiday.name: holiday for holiday in other}

        try:
            base = base.rules
        except AttributeError:
            pass

        if not isinstance(base, list):
            base = [base]
        base_holidays = {holiday.name: holiday for holiday in base}

        other_holidays.update(base_holidays)
        return list(other_holidays.values())

    def merge(self, other, inplace: bool = False):
        """
        Merge holiday calendars together.  The caller's class
        rules take precedence.  The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        other : holiday calendar
        inplace : bool (default=False)
            If True set rule_table to holidays, else return array of Holidays
        """
        holidays = self.merge_class(self, other)
        if inplace:
            self.rules = holidays
        else:
            return holidays


USMemorialDay = Holiday(
    "Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))
)
USLaborDay = Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))
USColumbusDay = Holiday(
    "Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))
)
USThanksgivingDay = Holiday(
    "Thanksgiving Day", month=11, day=1, offset=DateOffset(weekday=TH(4))
)
USMartinLutherKingJr = Holiday(
    "Birthday of Martin Luther King, Jr.",
    start_date=datetime(1986, 1, 1),
    month=1,
    day=1,
    offset=DateOffset(weekday=MO(3)),
)
USPresidentsDay = Holiday(
    "Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))
)
GoodFriday = Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)])

EasterMonday = Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)])


class USFederalHolidayCalendar(AbstractHolidayCalendar):
    """
    US Federal Government Holiday Calendar based on rules specified by:
    https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def HolidayCalendarFactory(name: str, base, other, base_class=AbstractHolidayCalendar):
    rules = AbstractHolidayCalendar.merge_class(base, other)
    calendar_class = type(name, (base_class,), {"rules": rules, "name": name})
    return calendar_class


__all__ = [
    "after_nearest_workday",
    "before_nearest_workday",
    "FR",
    "get_calendar",
    "HolidayCalendarFactory",
    "MO",
    "nearest_workday",
    "next_monday",
    "next_monday_or_tuesday",
    "next_workday",
    "previous_friday",
    "previous_workday",
    "register",
    "SA",
    "SU",
    "sunday_to_monday",
    "TH",
    "TU",
    "WE",
    "weekend_to_monday",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\facts.py ===
r"""This is rule-based deduction system for SymPy

The whole thing is split into two parts

 - rules compilation and preparation of tables
 - runtime inference

For rule-based inference engines, the classical work is RETE algorithm [1],
[2] Although we are not implementing it in full (or even significantly)
it's still worth a read to understand the underlying ideas.

In short, every rule in a system of rules is one of two forms:

 - atom                     -> ...      (alpha rule)
 - And(atom1, atom2, ...)   -> ...      (beta rule)


The major complexity is in efficient beta-rules processing and usually for an
expert system a lot of effort goes into code that operates on beta-rules.


Here we take minimalistic approach to get something usable first.

 - (preparation)    of alpha- and beta- networks, everything except
 - (runtime)        FactRules.deduce_all_facts

             _____________________________________
            ( Kirr: I've never thought that doing )
            ( logic stuff is that difficult...    )
             -------------------------------------
                    o   ^__^
                     o  (oo)\_______
                        (__)\       )\/\
                            ||----w |
                            ||     ||


Some references on the topic
----------------------------

[1] https://en.wikipedia.org/wiki/Rete_algorithm
[2] http://reports-archive.adm.cs.cmu.edu/anon/1995/CMU-CS-95-113.pdf

https://en.wikipedia.org/wiki/Propositional_formula
https://en.wikipedia.org/wiki/Inference_rule
https://en.wikipedia.org/wiki/List_of_rules_of_inference
"""

from collections import defaultdict
from typing import Iterator

from .logic import Logic, And, Or, Not


def _base_fact(atom):
    """Return the literal fact of an atom.

    Effectively, this merely strips the Not around a fact.
    """
    if isinstance(atom, Not):
        return atom.arg
    else:
        return atom


def _as_pair(atom):
    if isinstance(atom, Not):
        return (atom.arg, False)
    else:
        return (atom, True)

# XXX this prepares forward-chaining rules for alpha-network


def transitive_closure(implications):
    """
    Computes the transitive closure of a list of implications

    Uses Warshall's algorithm, as described at
    http://www.cs.hope.edu/~cusack/Notes/Notes/DiscreteMath/Warshall.pdf.
    """
    full_implications = set(implications)
    literals = set().union(*map(set, full_implications))

    for k in literals:
        for i in literals:
            if (i, k) in full_implications:
                for j in literals:
                    if (k, j) in full_implications:
                        full_implications.add((i, j))

    return full_implications


def deduce_alpha_implications(implications):
    """deduce all implications

       Description by example
       ----------------------

       given set of logic rules:

         a -> b
         b -> c

       we deduce all possible rules:

         a -> b, c
         b -> c


       implications: [] of (a,b)
       return:       {} of a -> set([b, c, ...])
    """
    implications = implications + [(Not(j), Not(i)) for (i, j) in implications]
    res = defaultdict(set)
    full_implications = transitive_closure(implications)
    for a, b in full_implications:
        if a == b:
            continue    # skip a->a cyclic input

        res[a].add(b)

    # Clean up tautologies and check consistency
    for a, impl in res.items():
        impl.discard(a)
        na = Not(a)
        if na in impl:
            raise ValueError(
                'implications are inconsistent: %s -> %s %s' % (a, na, impl))

    return res


def apply_beta_to_alpha_route(alpha_implications, beta_rules):
    """apply additional beta-rules (And conditions) to already-built
    alpha implication tables

       TODO: write about

       - static extension of alpha-chains
       - attaching refs to beta-nodes to alpha chains


       e.g.

       alpha_implications:

       a  ->  [b, !c, d]
       b  ->  [d]
       ...


       beta_rules:

       &(b,d) -> e


       then we'll extend a's rule to the following

       a  ->  [b, !c, d, e]
    """
    x_impl = {}
    for x in alpha_implications.keys():
        x_impl[x] = (set(alpha_implications[x]), [])
    for bcond, bimpl in beta_rules:
        for bk in bcond.args:
            if bk in x_impl:
                continue
            x_impl[bk] = (set(), [])

    # static extensions to alpha rules:
    # A: x -> a,b   B: &(a,b) -> c  ==>  A: x -> a,b,c
    seen_static_extension = True
    while seen_static_extension:
        seen_static_extension = False

        for bcond, bimpl in beta_rules:
            if not isinstance(bcond, And):
                raise TypeError("Cond is not And")
            bargs = set(bcond.args)
            for x, (ximpls, bb) in x_impl.items():
                x_all = ximpls | {x}
                # A: ... -> a   B: &(...) -> a  is non-informative
                if bimpl not in x_all and bargs.issubset(x_all):
                    ximpls.add(bimpl)

                    # we introduced new implication - now we have to restore
                    # completeness of the whole set.
                    bimpl_impl = x_impl.get(bimpl)
                    if bimpl_impl is not None:
                        ximpls |= bimpl_impl[0]
                    seen_static_extension = True

    # attach beta-nodes which can be possibly triggered by an alpha-chain
    for bidx, (bcond, bimpl) in enumerate(beta_rules):
        bargs = set(bcond.args)
        for x, (ximpls, bb) in x_impl.items():
            x_all = ximpls | {x}
            # A: ... -> a   B: &(...) -> a      (non-informative)
            if bimpl in x_all:
                continue
            # A: x -> a...  B: &(!a,...) -> ... (will never trigger)
            # A: x -> a...  B: &(...) -> !a     (will never trigger)
            if any(Not(xi) in bargs or Not(xi) == bimpl for xi in x_all):
                continue

            if bargs & x_all:
                bb.append(bidx)

    return x_impl


def rules_2prereq(rules):
    """build prerequisites table from rules

       Description by example
       ----------------------

       given set of logic rules:

         a -> b, c
         b -> c

       we build prerequisites (from what points something can be deduced):

         b <- a
         c <- a, b

       rules:   {} of a -> [b, c, ...]
       return:  {} of c <- [a, b, ...]

       Note however, that this prerequisites may be *not* enough to prove a
       fact. An example is 'a -> b' rule, where prereq(a) is b, and prereq(b)
       is a. That's because a=T -> b=T, and b=F -> a=F, but a=F -> b=?
    """
    prereq = defaultdict(set)
    for (a, _), impl in rules.items():
        if isinstance(a, Not):
            a = a.args[0]
        for (i, _) in impl:
            if isinstance(i, Not):
                i = i.args[0]
            prereq[i].add(a)
    return prereq

################
# RULES PROVER #
################


class TautologyDetected(Exception):
    """(internal) Prover uses it for reporting detected tautology"""
    pass


class Prover:
    """ai - prover of logic rules

       given a set of initial rules, Prover tries to prove all possible rules
       which follow from given premises.

       As a result proved_rules are always either in one of two forms: alpha or
       beta:

       Alpha rules
       -----------

       This are rules of the form::

         a -> b & c & d & ...


       Beta rules
       ----------

       This are rules of the form::

         &(a,b,...) -> c & d & ...


       i.e. beta rules are join conditions that say that something follows when
       *several* facts are true at the same time.
    """

    def __init__(self):
        self.proved_rules = []
        self._rules_seen = set()

    def split_alpha_beta(self):
        """split proved rules into alpha and beta chains"""
        rules_alpha = []    # a      -> b
        rules_beta = []     # &(...) -> b
        for a, b in self.proved_rules:
            if isinstance(a, And):
                rules_beta.append((a, b))
            else:
                rules_alpha.append((a, b))
        return rules_alpha, rules_beta

    @property
    def rules_alpha(self):
        return self.split_alpha_beta()[0]

    @property
    def rules_beta(self):
        return self.split_alpha_beta()[1]

    def process_rule(self, a, b):
        """process a -> b rule"""   # TODO write more?
        if (not a) or isinstance(b, bool):
            return
        if isinstance(a, bool):
            return
        if (a, b) in self._rules_seen:
            return
        else:
            self._rules_seen.add((a, b))

        # this is the core of processing
        try:
            self._process_rule(a, b)
        except TautologyDetected:
            pass

    def _process_rule(self, a, b):
        # right part first

        # a -> b & c    -->  a -> b  ;  a -> c
        # (?) FIXME this is only correct when b & c != null !

        if isinstance(b, And):
            sorted_bargs = sorted(b.args, key=str)
            for barg in sorted_bargs:
                self.process_rule(a, barg)

        # a -> b | c    -->  !b & !c -> !a
        #               -->   a & !b -> c
        #               -->   a & !c -> b
        elif isinstance(b, Or):
            sorted_bargs = sorted(b.args, key=str)
            # detect tautology first
            if not isinstance(a, Logic):    # Atom
                # tautology:  a -> a|c|...
                if a in sorted_bargs:
                    raise TautologyDetected(a, b, 'a -> a|c|...')
            self.process_rule(And(*[Not(barg) for barg in b.args]), Not(a))

            for bidx in range(len(sorted_bargs)):
                barg = sorted_bargs[bidx]
                brest = sorted_bargs[:bidx] + sorted_bargs[bidx + 1:]
                self.process_rule(And(a, Not(barg)), Or(*brest))

        # left part

        # a & b -> c    -->  IRREDUCIBLE CASE -- WE STORE IT AS IS
        #                    (this will be the basis of beta-network)
        elif isinstance(a, And):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a & b -> a')
            self.proved_rules.append((a, b))
            # XXX NOTE at present we ignore  !c -> !a | !b

        elif isinstance(a, Or):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a | b -> a')
            for aarg in sorted_aargs:
                self.process_rule(aarg, b)

        else:
            # both `a` and `b` are atoms
            self.proved_rules.append((a, b))             # a  -> b
            self.proved_rules.append((Not(b), Not(a)))   # !b -> !a

########################################


class FactRules:
    """Rules that describe how to deduce facts in logic space

       When defined, these rules allow implications to quickly be determined
       for a set of facts. For this precomputed deduction tables are used.
       see `deduce_all_facts`   (forward-chaining)

       Also it is possible to gather prerequisites for a fact, which is tried
       to be proven.    (backward-chaining)


       Definition Syntax
       -----------------

       a -> b       -- a=T -> b=T  (and automatically b=F -> a=F)
       a -> !b      -- a=T -> b=F
       a == b       -- a -> b & b -> a
       a -> b & c   -- a=T -> b=T & c=T
       # TODO b | c


       Internals
       ---------

       .full_implications[k, v]: all the implications of fact k=v
       .beta_triggers[k, v]: beta rules that might be triggered when k=v
       .prereq  -- {} k <- [] of k's prerequisites

       .defined_facts -- set of defined fact names
    """

    def __init__(self, rules):
        """Compile rules into internal lookup tables"""

        if isinstance(rules, str):
            rules = rules.splitlines()

        # --- parse and process rules ---
        P = Prover()

        for rule in rules:
            # XXX `a` is hardcoded to be always atom
            a, op, b = rule.split(None, 2)

            a = Logic.fromstring(a)
            b = Logic.fromstring(b)

            if op == '->':
                P.process_rule(a, b)
            elif op == '==':
                P.process_rule(a, b)
                P.process_rule(b, a)
            else:
                raise ValueError('unknown op %r' % op)

        # --- build deduction networks ---
        self.beta_rules = []
        for bcond, bimpl in P.rules_beta:
            self.beta_rules.append(
                ({_as_pair(a) for a in bcond.args}, _as_pair(bimpl)))

        # deduce alpha implications
        impl_a = deduce_alpha_implications(P.rules_alpha)

        # now:
        # - apply beta rules to alpha chains  (static extension), and
        # - further associate beta rules to alpha chain (for inference
        # at runtime)
        impl_ab = apply_beta_to_alpha_route(impl_a, P.rules_beta)

        # extract defined fact names
        self.defined_facts = {_base_fact(k) for k in impl_ab.keys()}

        # build rels (forward chains)
        full_implications = defaultdict(set)
        beta_triggers = defaultdict(set)
        for k, (impl, betaidxs) in impl_ab.items():
            full_implications[_as_pair(k)] = {_as_pair(i) for i in impl}
            beta_triggers[_as_pair(k)] = betaidxs

        self.full_implications = full_implications
        self.beta_triggers = beta_triggers

        # build prereq (backward chains)
        prereq = defaultdict(set)
        rel_prereq = rules_2prereq(full_implications)
        for k, pitems in rel_prereq.items():
            prereq[k] |= pitems
        self.prereq = prereq

    def _to_python(self) -> str:
        """ Generate a string with plain python representation of the instance """
        return '\n'.join(self.print_rules())

    @classmethod
    def _from_python(cls, data : dict):
        """ Generate an instance from the plain python representation """
        self = cls('')
        for key in ['full_implications', 'beta_triggers', 'prereq']:
            d=defaultdict(set)
            d.update(data[key])
            setattr(self, key, d)
        self.beta_rules = data['beta_rules']
        self.defined_facts = set(data['defined_facts'])

        return self

    def _defined_facts_lines(self):
        yield 'defined_facts = ['
        for fact in sorted(self.defined_facts):
            yield f'    {fact!r},'
        yield '] # defined_facts'

    def _full_implications_lines(self):
        yield 'full_implications = dict( ['
        for fact in sorted(self.defined_facts):
            for value in (True, False):
                yield f'    # Implications of {fact} = {value}:'
                yield f'    (({fact!r}, {value!r}), set( ('
                implications = self.full_implications[(fact, value)]
                for implied in sorted(implications):
                    yield f'        {implied!r},'
                yield '       ) ),'
                yield '     ),'
        yield ' ] ) # full_implications'

    def _prereq_lines(self):
        yield 'prereq = {'
        yield ''
        for fact in sorted(self.prereq):
            yield f'    # facts that could determine the value of {fact}'
            yield f'    {fact!r}: {{'
            for pfact in sorted(self.prereq[fact]):
                yield f'        {pfact!r},'
            yield '    },'
            yield ''
        yield '} # prereq'

    def _beta_rules_lines(self):
        reverse_implications = defaultdict(list)
        for n, (pre, implied) in enumerate(self.beta_rules):
            reverse_implications[implied].append((pre, n))

        yield '# Note: the order of the beta rules is used in the beta_triggers'
        yield 'beta_rules = ['
        yield ''
        m = 0
        indices = {}
        for implied in sorted(reverse_implications):
            fact, value = implied
            yield f'    # Rules implying {fact} = {value}'
            for pre, n in reverse_implications[implied]:
                indices[n] = m
                m += 1
                setstr = ", ".join(map(str, sorted(pre)))
                yield f'    ({{{setstr}}},'
                yield f'        {implied!r}),'
            yield ''
        yield '] # beta_rules'

        yield 'beta_triggers = {'
        for query in sorted(self.beta_triggers):
            fact, value = query
            triggers = [indices[n] for n in self.beta_triggers[query]]
            yield f'    {query!r}: {triggers!r},'
        yield '} # beta_triggers'

    def print_rules(self) -> Iterator[str]:
        """ Returns a generator with lines to represent the facts and rules """
        yield from self._defined_facts_lines()
        yield ''
        yield ''
        yield from self._full_implications_lines()
        yield ''
        yield ''
        yield from self._prereq_lines()
        yield ''
        yield ''
        yield from self._beta_rules_lines()
        yield ''
        yield ''
        yield "generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,"
        yield "               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}"


class InconsistentAssumptions(ValueError):
    def __str__(self):
        kb, fact, value = self.args
        return "%s, %s=%s" % (kb, fact, value)


class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """
    def __str__(self):
        return '{\n%s}' % ',\n'.join(
            ["\t%s: %s" % i for i in sorted(self.items())])

    def __init__(self, rules):
        self.rules = rules

    def _tell(self, k, v):
        """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
        if k in self and self[k] is not None:
            if self[k] == v:
                return False
            else:
                raise InconsistentAssumptions(self, k, v)
        else:
            self[k] = v
            return True

    # *********************************************
    # * This is the workhorse, so keep it *fast*. *
    # *********************************************
    def deduce_all_facts(self, facts):
        """
        Update the KB with all the implications of a list of facts.

        Facts can be specified as a dictionary or as a list of (key, value)
        pairs.
        """
        # keep frequently used attributes locally, so we'll avoid extra
        # attribute access overhead
        full_implications = self.rules.full_implications
        beta_triggers = self.rules.beta_triggers
        beta_rules = self.rules.beta_rules

        if isinstance(facts, dict):
            facts = facts.items()

        while facts:
            beta_maytrigger = set()

            # --- alpha chains ---
            for k, v in facts:
                if not self._tell(k, v) or v is None:
                    continue

                # lookup routing tables
                for key, value in full_implications[k, v]:
                    self._tell(key, value)

                beta_maytrigger.update(beta_triggers[k, v])

            # --- beta chains ---
            facts = []
            for bidx in beta_maytrigger:
                bcond, bimpl = beta_rules[bidx]
                if all(self.get(k) is v for k, v in bcond):
                    facts.append(bimpl)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\numerictypes.py ===
"""
numerictypes: Define the numeric type objects

This module is designed so "from numerictypes import \\*" is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    sctypeDict

  Type objects (not all will be available, depends on platform):
      see variable sctypes for which ones you have

    Bit-width names

    int8 int16 int32 int64
    uint8 uint16 uint32 uint64
    float16 float32 float64 float96 float128
    complex64 complex128 complex192 complex256
    datetime64 timedelta64

    c-based names

    bool

    object_

    void, str_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    double, cdouble,
    longdouble, clongdouble,

   As part of the type-hierarchy:    xx -- is bit-width

   generic
     +-> bool                                   (kind=b)
     +-> number
     |   +-> integer
     |   |   +-> signedinteger     (intxx)      (kind=i)
     |   |   |     byte
     |   |   |     short
     |   |   |     intc
     |   |   |     intp
     |   |   |     int_
     |   |   |     longlong
     |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
     |   |         ubyte
     |   |         ushort
     |   |         uintc
     |   |         uintp
     |   |         uint
     |   |         ulonglong
     |   +-> inexact
     |       +-> floating          (floatxx)    (kind=f)
     |       |     half
     |       |     single
     |       |     double
     |       |     longdouble
     |       \\-> complexfloating  (complexxx)  (kind=c)
     |             csingle
     |             cdouble
     |             clongdouble
     +-> flexible
     |   +-> character
     |   |     bytes_                           (kind=S)
     |   |     str_                             (kind=U)
     |   |
     |   \\-> void                              (kind=V)
     \\-> object_ (not used much)               (kind=O)

"""
import numbers
import warnings

from numpy._utils import set_module

from . import multiarray as ma
from .multiarray import (
    busday_count,
    busday_offset,
    busdaycalendar,
    datetime_as_string,
    datetime_data,
    dtype,
    is_busday,
    ndarray,
)

# we add more at the bottom
__all__ = [
    'ScalarType', 'typecodes', 'issubdtype', 'datetime_data',
    'datetime_as_string', 'busday_offset', 'busday_count',
    'is_busday', 'busdaycalendar', 'isdtype'
]

# we don't need all these imports, but we need to keep them for compatibility
# for users using np._core.numerictypes.UPPER_TABLE
# we don't export these for import *, but we do want them accessible
# as numerictypes.bool, etc.
from builtins import bool, bytes, complex, float, int, object, str  # noqa: F401, UP029

from ._dtype import _kind_name
from ._string_helpers import (  # noqa: F401
    LOWER_TABLE,
    UPPER_TABLE,
    english_capitalize,
    english_lower,
    english_upper,
)
from ._type_aliases import allTypes, sctypeDict, sctypes

# We use this later
generic = allTypes['generic']

genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16',
                   'int32', 'uint32', 'int64', 'uint64',
                   'float16', 'float32', 'float64', 'float96', 'float128',
                   'complex64', 'complex128', 'complex192', 'complex256',
                   'object']

@set_module('numpy')
def maximum_sctype(t):
    """
    Return the scalar type of highest precision of the same kind as the input.

    .. deprecated:: 2.0
        Use an explicit dtype like int64 or float64 instead.

    Parameters
    ----------
    t : dtype or dtype specifier
        The input data type. This can be a `dtype` object or an object that
        is convertible to a `dtype`.

    Returns
    -------
    out : dtype
        The highest precision data type of the same kind (`dtype.kind`) as `t`.

    See Also
    --------
    obj2sctype, mintypecode, sctype2char
    dtype

    Examples
    --------
    >>> from numpy._core.numerictypes import maximum_sctype
    >>> maximum_sctype(int)
    <class 'numpy.int64'>
    >>> maximum_sctype(np.uint8)
    <class 'numpy.uint64'>
    >>> maximum_sctype(complex)
    <class 'numpy.complex256'> # may vary

    >>> maximum_sctype(str)
    <class 'numpy.str_'>

    >>> maximum_sctype('i2')
    <class 'numpy.int64'>
    >>> maximum_sctype('f4')
    <class 'numpy.float128'> # may vary

    """

    # Deprecated in NumPy 2.0, 2023-07-11
    warnings.warn(
        "`maximum_sctype` is deprecated. Use an explicit dtype like int64 "
        "or float64 instead. (deprecated in NumPy 2.0)",
        DeprecationWarning,
        stacklevel=2
    )

    g = obj2sctype(t)
    if g is None:
        return t
    t = g
    base = _kind_name(dtype(t))
    if base in sctypes:
        return sctypes[base][-1]
    else:
        return t


@set_module('numpy')
def issctype(rep):
    """
    Determines whether the given object represents a scalar data-type.

    Parameters
    ----------
    rep : any
        If `rep` is an instance of a scalar dtype, True is returned. If not,
        False is returned.

    Returns
    -------
    out : bool
        Boolean result of check whether `rep` is a scalar dtype.

    See Also
    --------
    issubsctype, issubdtype, obj2sctype, sctype2char

    Examples
    --------
    >>> from numpy._core.numerictypes import issctype
    >>> issctype(np.int32)
    True
    >>> issctype(list)
    False
    >>> issctype(1.1)
    False

    Strings are also a scalar type:

    >>> issctype(np.dtype('str'))
    True

    """
    if not isinstance(rep, (type, dtype)):
        return False
    try:
        res = obj2sctype(rep)
        if res and res != object_:
            return True
        else:
            return False
    except Exception:
        return False


def obj2sctype(rep, default=None):
    """
    Return the scalar dtype or NumPy equivalent of Python type of an object.

    Parameters
    ----------
    rep : any
        The object of which the type is returned.
    default : any, optional
        If given, this is returned for objects whose types can not be
        determined. If not given, None is returned for those objects.

    Returns
    -------
    dtype : dtype or Python type
        The data type of `rep`.

    See Also
    --------
    sctype2char, issctype, issubsctype, issubdtype

    Examples
    --------
    >>> from numpy._core.numerictypes import obj2sctype
    >>> obj2sctype(np.int32)
    <class 'numpy.int32'>
    >>> obj2sctype(np.array([1., 2.]))
    <class 'numpy.float64'>
    >>> obj2sctype(np.array([1.j]))
    <class 'numpy.complex128'>

    >>> obj2sctype(dict)
    <class 'numpy.object_'>
    >>> obj2sctype('string')

    >>> obj2sctype(1, default=list)
    <class 'list'>

    """
    # prevent abstract classes being upcast
    if isinstance(rep, type) and issubclass(rep, generic):
        return rep
    # extract dtype from arrays
    if isinstance(rep, ndarray):
        return rep.dtype.type
    # fall back on dtype to convert
    try:
        res = dtype(rep)
    except Exception:
        return default
    else:
        return res.type


@set_module('numpy')
def issubclass_(arg1, arg2):
    """
    Determine if a class is a subclass of a second class.

    `issubclass_` is equivalent to the Python built-in ``issubclass``,
    except that it returns False instead of raising a TypeError if one
    of the arguments is not a class.

    Parameters
    ----------
    arg1 : class
        Input class. True is returned if `arg1` is a subclass of `arg2`.
    arg2 : class or tuple of classes.
        Input class. If a tuple of classes, True is returned if `arg1` is a
        subclass of any of the tuple elements.

    Returns
    -------
    out : bool
        Whether `arg1` is a subclass of `arg2` or not.

    See Also
    --------
    issubsctype, issubdtype, issctype

    Examples
    --------
    >>> np.issubclass_(np.int32, int)
    False
    >>> np.issubclass_(np.int32, float)
    False
    >>> np.issubclass_(np.float64, float)
    True

    """
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False


@set_module('numpy')
def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    Parameters
    ----------
    arg1, arg2 : dtype or dtype specifier
        Data-types.

    Returns
    -------
    out : bool
        The result.

    See Also
    --------
    issctype, issubdtype, obj2sctype

    Examples
    --------
    >>> from numpy._core import issubsctype
    >>> issubsctype('S8', str)
    False
    >>> issubsctype(np.array([1]), int)
    True
    >>> issubsctype(np.array([1]), float)
    False

    """
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))


class _PreprocessDTypeError(Exception):
    pass


def _preprocess_dtype(dtype):
    """
    Preprocess dtype argument by:
      1. fetching type from a data type
      2. verifying that types are built-in NumPy dtypes
    """
    if isinstance(dtype, ma.dtype):
        dtype = dtype.type
    if isinstance(dtype, ndarray) or dtype not in allTypes.values():
        raise _PreprocessDTypeError
    return dtype


@set_module('numpy')
def isdtype(dtype, kind):
    """
    Determine if a provided dtype is of a specified data type ``kind``.

    This function only supports built-in NumPy's data types.
    Third-party dtypes are not yet supported.

    Parameters
    ----------
    dtype : dtype
        The input dtype.
    kind : dtype or str or tuple of dtypes/strs.
        dtype or dtype kind. Allowed dtype kinds are:
        * ``'bool'`` : boolean kind
        * ``'signed integer'`` : signed integer data types
        * ``'unsigned integer'`` : unsigned integer data types
        * ``'integral'`` : integer data types
        * ``'real floating'`` : real-valued floating-point data types
        * ``'complex floating'`` : complex floating-point data types
        * ``'numeric'`` : numeric data types

    Returns
    -------
    out : bool

    See Also
    --------
    issubdtype

    Examples
    --------
    >>> import numpy as np
    >>> np.isdtype(np.float32, np.float64)
    False
    >>> np.isdtype(np.float32, "real floating")
    True
    >>> np.isdtype(np.complex128, ("real floating", "complex floating"))
    True

    """
    try:
        dtype = _preprocess_dtype(dtype)
    except _PreprocessDTypeError:
        raise TypeError(
            "dtype argument must be a NumPy dtype, "
            f"but it is a {type(dtype)}."
        ) from None

    input_kinds = kind if isinstance(kind, tuple) else (kind,)

    processed_kinds = set()

    for kind in input_kinds:
        if kind == "bool":
            processed_kinds.add(allTypes["bool"])
        elif kind == "signed integer":
            processed_kinds.update(sctypes["int"])
        elif kind == "unsigned integer":
            processed_kinds.update(sctypes["uint"])
        elif kind == "integral":
            processed_kinds.update(sctypes["int"] + sctypes["uint"])
        elif kind == "real floating":
            processed_kinds.update(sctypes["float"])
        elif kind == "complex floating":
            processed_kinds.update(sctypes["complex"])
        elif kind == "numeric":
            processed_kinds.update(
                sctypes["int"] + sctypes["uint"] +
                sctypes["float"] + sctypes["complex"]
            )
        elif isinstance(kind, str):
            raise ValueError(
                "kind argument is a string, but"
                f" {kind!r} is not a known kind name."
            )
        else:
            try:
                kind = _preprocess_dtype(kind)
            except _PreprocessDTypeError:
                raise TypeError(
                    "kind argument must be comprised of "
                    "NumPy dtypes or strings only, "
                    f"but is a {type(kind)}."
                ) from None
            processed_kinds.add(kind)

    return dtype in processed_kinds


@set_module('numpy')
def issubdtype(arg1, arg2):
    r"""
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    This is like the builtin :func:`issubclass`, but for `dtype`\ s.

    Parameters
    ----------
    arg1, arg2 : dtype_like
        `dtype` or object coercible to one

    Returns
    -------
    out : bool

    See Also
    --------
    :ref:`arrays.scalars` : Overview of the numpy type hierarchy.

    Examples
    --------
    `issubdtype` can be used to check the type of arrays:

    >>> ints = np.array([1, 2, 3], dtype=np.int32)
    >>> np.issubdtype(ints.dtype, np.integer)
    True
    >>> np.issubdtype(ints.dtype, np.floating)
    False

    >>> floats = np.array([1, 2, 3], dtype=np.float32)
    >>> np.issubdtype(floats.dtype, np.integer)
    False
    >>> np.issubdtype(floats.dtype, np.floating)
    True

    Similar types of different sizes are not subdtypes of each other:

    >>> np.issubdtype(np.float64, np.float32)
    False
    >>> np.issubdtype(np.float32, np.float64)
    False

    but both are subtypes of `floating`:

    >>> np.issubdtype(np.float64, np.floating)
    True
    >>> np.issubdtype(np.float32, np.floating)
    True

    For convenience, dtype-like objects are allowed too:

    >>> np.issubdtype('S1', np.bytes_)
    True
    >>> np.issubdtype('i4', np.signedinteger)
    True

    """
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type

    return issubclass(arg1, arg2)


@set_module('numpy')
def sctype2char(sctype):
    """
    Return the string representation of a scalar dtype.

    Parameters
    ----------
    sctype : scalar dtype or object
        If a scalar dtype, the corresponding string character is
        returned. If an object, `sctype2char` tries to infer its scalar type
        and then return the corresponding string character.

    Returns
    -------
    typechar : str
        The string character corresponding to the scalar type.

    Raises
    ------
    ValueError
        If `sctype` is an object for which the type can not be inferred.

    See Also
    --------
    obj2sctype, issctype, issubsctype, mintypecode

    Examples
    --------
    >>> from numpy._core.numerictypes import sctype2char
    >>> for sctype in [np.int32, np.double, np.cdouble, np.bytes_, np.ndarray]:
    ...     print(sctype2char(sctype))
    l # may vary
    d
    D
    S
    O

    >>> x = np.array([1., 2-1.j])
    >>> sctype2char(x)
    'D'
    >>> sctype2char(list)
    'O'

    """
    sctype = obj2sctype(sctype)
    if sctype is None:
        raise ValueError("unrecognized type")
    if sctype not in sctypeDict.values():
        # for compatibility
        raise KeyError(sctype)
    return dtype(sctype).char


def _scalar_type_key(typ):
    """A ``key`` function for `sorted`."""
    dt = dtype(typ)
    return (dt.kind.lower(), dt.itemsize)


ScalarType = [int, float, complex, bool, bytes, str, memoryview]
ScalarType += sorted(set(sctypeDict.values()), key=_scalar_type_key)
ScalarType = tuple(ScalarType)


# Now add the types we've determined to this module
for key in allTypes:
    globals()[key] = allTypes[key]
    __all__.append(key)

del key

typecodes = {'Character': 'c',
             'Integer': 'bhilqnp',
             'UnsignedInteger': 'BHILQNP',
             'Float': 'efdg',
             'Complex': 'FDG',
             'AllInteger': 'bBhHiIlLqQnNpP',
             'AllFloat': 'efdgFDG',
             'Datetime': 'Mm',
             'All': '?bhilqnpBHILQNPefdgFDGSUVOMm'}

# backwards compatibility --- deprecated name
# Formal deprecation: Numpy 1.20.0, 2020-10-19 (see numpy/__init__.py)
typeDict = sctypeDict

def _register_types():
    numbers.Integral.register(integer)
    numbers.Complex.register(inexact)
    numbers.Real.register(floating)
    numbers.Number.register(number)


_register_types()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\bert_perf_test.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This tool measures the inference performance of onnxruntime on BERT-like model with inputs like input_ids,
# token_type_ids (optional), and attention_mask (optional).
#
# If the model does not have exactly three inputs like above, you might need specify names of inputs with
# --input_ids_name, --segment_ids_name and --input_mask_name

# Example command to run test on batch_size 1 and 2 for a model on GPU:
#   python bert_perf_test.py --model bert.onnx --batch_size 1 2 --sequence_length 128 --use_gpu --samples 1000 --test_times 1

import argparse
import csv
import json
import multiprocessing
import os
import random
import statistics
import timeit
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
from bert_test_data import generate_test_data, get_bert_inputs


@dataclass
class TestSetting:
    batch_size: int
    sequence_length: int
    test_cases: int
    test_times: int
    use_gpu: bool
    use_io_binding: bool
    provider: str
    intra_op_num_threads: int
    seed: int
    verbose: bool
    log_severity: int
    average_sequence_length: int
    random_sequence_length: bool


@dataclass
class ModelSetting:
    model_path: str
    input_ids_name: str
    segment_ids_name: str
    input_mask_name: str
    opt_level: int
    input_tuning_results: str | None
    output_tuning_results: str | None
    mask_type: int


def create_session(
    model_path,
    use_gpu,
    provider,
    intra_op_num_threads,
    graph_optimization_level=None,
    log_severity=2,
    tuning_results_path=None,
):
    import onnxruntime

    onnxruntime.set_default_logger_severity(log_severity)

    if use_gpu and ("CUDAExecutionProvider" not in onnxruntime.get_available_providers()):
        print(
            "Warning: Please install onnxruntime-gpu package instead of onnxruntime, and use a machine with GPU for testing gpu performance."
        )

    if use_gpu:
        if provider == "dml":
            execution_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif provider == "rocm":
            execution_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        elif provider == "migraphx":
            execution_providers = [
                "MIGraphXExecutionProvider",
                "ROCMExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif provider == "cuda":
            execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "tensorrt":
            execution_providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        execution_providers = ["CPUExecutionProvider"]

    sess_options = onnxruntime.SessionOptions()
    sess_options.log_severity_level = log_severity
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    if graph_optimization_level is None:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    elif graph_optimization_level == 0:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif graph_optimization_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif graph_optimization_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    elif graph_optimization_level == 99:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    else:
        sess_options.graph_optimization_level = graph_optimization_level

    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = intra_op_num_threads

    session = onnxruntime.InferenceSession(model_path, sess_options, providers=execution_providers)

    if use_gpu:
        if provider == "dml":
            assert "DmlExecutionProvider" in session.get_providers()
        elif provider == "rocm":
            assert "ROCMExecutionProvider" in session.get_providers()
        elif provider == "migraphx":
            assert "MIGraphXExecutionProvider" in session.get_providers()
            assert "ROCMExecutionProvider" in session.get_providers()
        elif provider == "cuda":
            assert "CUDAExecutionProvider" in session.get_providers()
        elif provider == "tensorrt":
            assert "TensorrtExecutionProvider" in session.get_providers()
            assert "CUDAExecutionProvider" in session.get_providers()
        else:
            assert "CUDAExecutionProvider" in session.get_providers()
    else:
        assert "CPUExecutionProvider" in session.get_providers()

    if tuning_results_path is not None:
        with open(tuning_results_path) as f:
            session.set_tuning_results(json.load(f))

    return session


def numpy_type(torch_type):
    type_map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
    }
    return type_map[torch_type]


def create_input_output_tensors(inputs, outputs, device):
    input_tensors = {name: torch.from_numpy(array).to(device) for name, array in inputs.items()}
    output_tensors = {name: torch.from_numpy(array).to(device) for name, array in outputs.items()}
    return input_tensors, output_tensors


def create_io_binding(sess, input_tensors, output_tensors):
    io_binding = sess.io_binding()
    for name, tensor in input_tensors.items():
        io_binding.bind_input(
            name,
            tensor.device.type,
            0,
            numpy_type(tensor.dtype),
            tensor.shape,
            tensor.data_ptr(),
        )
    for name, tensor in output_tensors.items():
        io_binding.bind_output(
            name,
            tensor.device.type,
            0,
            numpy_type(tensor.dtype),
            tensor.shape,
            tensor.data_ptr(),
        )
    return io_binding


def onnxruntime_inference_with_io_binding(session, all_inputs, output_names, test_setting):
    results = []
    latency_list = []
    device = "cuda" if test_setting.use_gpu else "cpu"
    for _test_case_id, inputs in enumerate(all_inputs):
        result = session.run(output_names, inputs)
        results.append(result)
        outputs = {}
        for i in range(len(output_names)):
            outputs[output_names[i]] = result[i]

        input_tensors, output_tensors = create_input_output_tensors(inputs, outputs, device)
        io_binding = create_io_binding(session, input_tensors, output_tensors)

        # warm up once
        session.run_with_iobinding(io_binding)

        start_time = timeit.default_timer()
        session.run_with_iobinding(io_binding)
        latency = timeit.default_timer() - start_time
        latency_list.append(latency)

    return results, latency_list


def onnxruntime_inference(session, all_inputs, output_names):
    if len(all_inputs) > 0:
        # Use a random input as warm up.
        session.run(output_names, random.choice(all_inputs))

    results = []
    latency_list = []
    for _test_case_id, inputs in enumerate(all_inputs):
        start_time = timeit.default_timer()
        result = session.run(output_names, inputs)
        latency = timeit.default_timer() - start_time
        results.append(result)
        latency_list.append(latency)
    return results, latency_list


def to_string(model_path, session, test_setting):
    sess_options = session.get_session_options()
    option = f"model={os.path.basename(model_path)},"
    option += f"graph_optimization_level={sess_options.graph_optimization_level},intra_op_num_threads={sess_options.intra_op_num_threads},".replace(
        "GraphOptimizationLevel.ORT_", ""
    )

    option += f"batch_size={test_setting.batch_size},sequence_length={test_setting.sequence_length},"
    option += f"test_cases={test_setting.test_cases},test_times={test_setting.test_times},"
    option += f"use_gpu={test_setting.use_gpu},use_io_binding={test_setting.use_io_binding},"
    option += f"average_sequence_length={test_setting.average_sequence_length},"
    option += f"random_sequence_length={test_setting.random_sequence_length}"
    return option


def run_one_test(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads):
    session = create_session(
        model_setting.model_path,
        test_setting.use_gpu,
        test_setting.provider,
        intra_op_num_threads,
        model_setting.opt_level,
        log_severity=test_setting.log_severity,
        tuning_results_path=model_setting.input_tuning_results,
    )
    output_names = [output.name for output in session.get_outputs()]

    key = to_string(model_setting.model_path, session, test_setting)
    if key in perf_results:
        print("skip duplicated test:", key)
        return

    print("Running test:", key)

    all_latency_list = []
    if test_setting.use_io_binding:
        for _i in range(test_setting.test_times):
            results, latency_list = onnxruntime_inference_with_io_binding(
                session, all_inputs, output_names, test_setting
            )
            all_latency_list.extend(latency_list)
    else:
        for _i in range(test_setting.test_times):
            results, latency_list = onnxruntime_inference(session, all_inputs, output_names)
            all_latency_list.extend(latency_list)

    # latency in milliseconds
    latency_ms = np.array(all_latency_list) * 1000

    average_latency = statistics.mean(latency_ms)
    latency_50 = np.percentile(latency_ms, 50)
    latency_75 = np.percentile(latency_ms, 75)
    latency_90 = np.percentile(latency_ms, 90)
    latency_95 = np.percentile(latency_ms, 95)
    latency_99 = np.percentile(latency_ms, 99)
    throughput = test_setting.batch_size * (1000.0 / average_latency)

    perf_results[key] = (
        average_latency,
        latency_50,
        latency_75,
        latency_90,
        latency_95,
        latency_99,
        throughput,
    )

    print(
        "Average latency = {} ms, Throughput = {} QPS".format(format(average_latency, ".2f"), format(throughput, ".2f"))
    )

    if model_setting.output_tuning_results:
        output_path = os.path.abspath(model_setting.output_tuning_results)
        if os.path.exists(output_path):
            old_output_path = output_path
            output_path = f"""{output_path.rsplit(".json", 1)[0]}.{datetime.now().timestamp()}.json"""
            print("WARNING:", old_output_path, "exists, will write to", output_path, "instead.")

        trs = session.get_tuning_results()
        with open(output_path, "w") as f:
            json.dump(trs, f)
        print("Tuning results is saved to", output_path)


def launch_test(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads):
    process = multiprocessing.Process(
        target=run_one_test,
        args=(
            model_setting,
            test_setting,
            perf_results,
            all_inputs,
            intra_op_num_threads,
        ),
    )
    process.start()
    process.join()


def run_perf_tests(model_setting, test_setting, perf_results, all_inputs):
    if test_setting.intra_op_num_threads is not None:
        launch_test(
            model_setting,
            test_setting,
            perf_results,
            all_inputs,
            test_setting.intra_op_num_threads,
        )
        return

    cpu_count = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    candidate_threads = list({logical_cores, cpu_count})
    for i in range(1, min(16, logical_cores)):
        if i not in candidate_threads:
            candidate_threads.append(i)
    candidate_threads.sort(reverse=True)

    for intra_op_num_threads in candidate_threads:
        launch_test(model_setting, test_setting, perf_results, all_inputs, intra_op_num_threads)


def run_performance(model_setting, test_setting, perf_results):
    input_ids, segment_ids, input_mask = get_bert_inputs(
        model_setting.model_path,
        model_setting.input_ids_name,
        model_setting.segment_ids_name,
        model_setting.input_mask_name,
    )

    # Do not generate random mask for performance test.
    print(
        f"Generating {test_setting.test_cases} samples for batch_size={test_setting.batch_size} sequence_length={test_setting.sequence_length}"
    )
    all_inputs = generate_test_data(
        test_setting.batch_size,
        test_setting.sequence_length,
        test_setting.test_cases,
        test_setting.seed,
        test_setting.verbose,
        input_ids,
        segment_ids,
        input_mask,
        test_setting.average_sequence_length,
        test_setting.random_sequence_length,
        mask_type=model_setting.mask_type,
    )

    run_perf_tests(model_setting, test_setting, perf_results, all_inputs)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="bert onnx model path")

    parser.add_argument(
        "-b",
        "--batch_size",
        required=True,
        type=int,
        nargs="+",
        help="batch size of input. Allow one or multiple values in the range of [1, 128].",
    )

    parser.add_argument(
        "-s",
        "--sequence_length",
        required=True,
        type=int,
        help="maximum sequence length of input",
    )

    parser.add_argument(
        "--samples",
        required=False,
        type=int,
        default=10,
        help="number of samples to be generated",
    )

    parser.add_argument(
        "-t",
        "--test_times",
        required=False,
        type=int,
        default=0,
        help="number of times to run per sample. By default, the value is 1000 / samples",
    )

    parser.add_argument(
        "--opt_level",
        required=False,
        type=int,
        choices=[0, 1, 2, 99],
        default=99,
        help="onnxruntime optimization level: 0 - disable all, 1 - basic, 2 - extended, 99 - enable all.",
    )

    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=3,
        help="random seed. Use the same seed to make sure test data is same in multiple tests.",
    )

    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="print verbose information",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--log_severity",
        required=False,
        type=int,
        default=2,
        choices=[0, 1, 2, 3, 4],
        help="0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal",
    )

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument("--use_io_binding", required=False, action="store_true", help="use io_binding")
    parser.set_defaults(use_io_binding=False)

    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        default=None,
        help="Execution provider to use",
    )

    parser.add_argument(
        "-n",
        "--intra_op_num_threads",
        required=False,
        type=int,
        default=None,
        help=">=0, set intra_op_num_threads",
    )

    parser.add_argument(
        "--input_ids_name",
        required=False,
        type=str,
        default=None,
        help="input name for input ids",
    )

    parser.add_argument(
        "--segment_ids_name",
        required=False,
        type=str,
        default=None,
        help="input name for segment ids",
    )

    parser.add_argument(
        "--input_mask_name",
        required=False,
        type=str,
        default=None,
        help="input name for attention mask",
    )

    parser.add_argument(
        "--input_tuning_results",
        default=None,
        type=str,
        help="tuning results (json) to be loaded before benchmark",
    )

    parser.add_argument(
        "--output_tuning_results",
        default=None,
        type=str,
        help="tuning results (json) to be saved after benchmark",
    )

    parser.add_argument(
        "-a",
        "--average_sequence_length",
        default=-1,
        type=int,
        help="average sequence length excluding padding",
    )

    parser.add_argument(
        "-r",
        "--random_sequence_length",
        required=False,
        action="store_true",
        help="use uniform random instead of fixed sequence length",
    )
    parser.set_defaults(random_sequence_length=False)

    parser.add_argument(
        "--mask_type",
        required=False,
        type=int,
        default=2,
        help="mask type: (1: mask index or sequence length, 2: raw 2D mask, 3: key len, cumulated lengths of query and key)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.test_times == 0:
        args.test_times = max(1, int(1000 / args.samples))

    if args.average_sequence_length <= 0:
        args.average_sequence_length = args.sequence_length

    manager = multiprocessing.Manager()
    perf_results = manager.dict()

    batch_size_set = set(args.batch_size)
    if not (min(batch_size_set) >= 1 and max(batch_size_set) <= 128):
        raise Exception("batch_size not in range [1, 128]")

    model_setting = ModelSetting(
        args.model,
        args.input_ids_name,
        args.segment_ids_name,
        args.input_mask_name,
        args.opt_level,
        args.input_tuning_results,
        args.output_tuning_results,
        args.mask_type,
    )

    for batch_size in batch_size_set:
        test_setting = TestSetting(
            batch_size,
            args.sequence_length,
            args.samples,
            args.test_times,
            args.use_gpu,
            args.use_io_binding,
            args.provider,
            args.intra_op_num_threads,
            args.seed,
            args.verbose,
            args.log_severity,
            args.average_sequence_length,
            args.random_sequence_length,
        )

        print("test setting", test_setting)
        run_performance(model_setting, test_setting, perf_results)

    # Sort the results so that the first one has smallest latency.
    sorted_results = sorted(perf_results.items(), reverse=False, key=lambda x: x[1])

    summary_file = os.path.join(
        Path(args.model).parent,
        "perf_results_{}_B{}_S{}_{}.txt".format(
            "GPU" if args.use_gpu else "CPU",
            "-".join([str(x) for x in sorted(batch_size_set)]),
            args.sequence_length,
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        ),
    )
    with open(summary_file, "w+", newline="") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
        headers = None
        for key, perf_result in sorted_results:
            params = key.split(",")
            if headers is None:
                headers = [
                    "Latency(ms)",
                    "Latency_P50",
                    "Latency_P75",
                    "Latency_P90",
                    "Latency_P95",
                    "Latency_P99",
                    "Throughput(QPS)",
                ]
                headers.extend([x.split("=")[0] for x in params])
                tsv_writer.writerow(headers)

            values = [format(x, ".2f") for x in perf_result]
            values.extend([x.split("=")[1] for x in params])
            tsv_writer.writerow(values)

    print("Test summary is saved to", summary_file)


if __name__ == "__main__":
    # work around for AnaConda Jupyter. See https://stackoverflow.com/questions/45720153/python-multiprocessing-error-attributeerror-module-main-has-no-attribute
    __spec__ = None

    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\sansio\blueprints.py ===
from __future__ import annotations

import os
import typing as t
from collections import defaultdict
from functools import update_wrapper

from .. import typing as ft
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .scaffold import setupmethod

if t.TYPE_CHECKING:  # pragma: no cover
    from .app import App

DeferredSetupFunction = t.Callable[["BlueprintSetupState"], None]
T_after_request = t.TypeVar("T_after_request", bound=ft.AfterRequestCallable[t.Any])
T_before_request = t.TypeVar("T_before_request", bound=ft.BeforeRequestCallable)
T_error_handler = t.TypeVar("T_error_handler", bound=ft.ErrorHandlerCallable)
T_teardown = t.TypeVar("T_teardown", bound=ft.TeardownCallable)
T_template_context_processor = t.TypeVar(
    "T_template_context_processor", bound=ft.TemplateContextProcessorCallable
)
T_template_filter = t.TypeVar("T_template_filter", bound=ft.TemplateFilterCallable)
T_template_global = t.TypeVar("T_template_global", bound=ft.TemplateGlobalCallable)
T_template_test = t.TypeVar("T_template_test", bound=ft.TemplateTestCallable)
T_url_defaults = t.TypeVar("T_url_defaults", bound=ft.URLDefaultCallable)
T_url_value_preprocessor = t.TypeVar(
    "T_url_value_preprocessor", bound=ft.URLValuePreprocessorCallable
)


class BlueprintSetupState:
    """Temporary holder object for registering a blueprint with the
    application.  An instance of this class is created by the
    :meth:`~flask.Blueprint.make_setup_state` method and later passed
    to all register callback functions.
    """

    def __init__(
        self,
        blueprint: Blueprint,
        app: App,
        options: t.Any,
        first_registration: bool,
    ) -> None:
        #: a reference to the current application
        self.app = app

        #: a reference to the blueprint that created this setup state.
        self.blueprint = blueprint

        #: a dictionary with all options that were passed to the
        #: :meth:`~flask.Flask.register_blueprint` method.
        self.options = options

        #: as blueprints can be registered multiple times with the
        #: application and not everything wants to be registered
        #: multiple times on it, this attribute can be used to figure
        #: out if the blueprint was registered in the past already.
        self.first_registration = first_registration

        subdomain = self.options.get("subdomain")
        if subdomain is None:
            subdomain = self.blueprint.subdomain

        #: The subdomain that the blueprint should be active for, ``None``
        #: otherwise.
        self.subdomain = subdomain

        url_prefix = self.options.get("url_prefix")
        if url_prefix is None:
            url_prefix = self.blueprint.url_prefix
        #: The prefix that should be used for all URLs defined on the
        #: blueprint.
        self.url_prefix = url_prefix

        self.name = self.options.get("name", blueprint.name)
        self.name_prefix = self.options.get("name_prefix", "")

        #: A dictionary with URL defaults that is added to each and every
        #: URL that was defined with the blueprint.
        self.url_defaults = dict(self.blueprint.url_values_defaults)
        self.url_defaults.update(self.options.get("url_defaults", ()))

    def add_url_rule(
        self,
        rule: str,
        endpoint: str | None = None,
        view_func: ft.RouteCallable | None = None,
        **options: t.Any,
    ) -> None:
        """A helper method to register a rule (and optionally a view function)
        to the application.  The endpoint is automatically prefixed with the
        blueprint's name.
        """
        if self.url_prefix is not None:
            if rule:
                rule = "/".join((self.url_prefix.rstrip("/"), rule.lstrip("/")))
            else:
                rule = self.url_prefix
        options.setdefault("subdomain", self.subdomain)
        if endpoint is None:
            endpoint = _endpoint_from_view_func(view_func)  # type: ignore
        defaults = self.url_defaults
        if "defaults" in options:
            defaults = dict(defaults, **options.pop("defaults"))

        self.app.add_url_rule(
            rule,
            f"{self.name_prefix}.{self.name}.{endpoint}".lstrip("."),
            view_func,
            defaults=defaults,
            **options,
        )


class Blueprint(Scaffold):
    """Represents a blueprint, a collection of routes and other
    app-related functions that can be registered on a real application
    later.

    A blueprint is an object that allows defining application functions
    without requiring an application object ahead of time. It uses the
    same decorators as :class:`~flask.Flask`, but defers the need for an
    application by recording them for later registration.

    Decorating a function with a blueprint creates a deferred function
    that is called with :class:`~flask.blueprints.BlueprintSetupState`
    when the blueprint is registered on an application.

    See :doc:`/blueprints` for more information.

    :param name: The name of the blueprint. Will be prepended to each
        endpoint name.
    :param import_name: The name of the blueprint package, usually
        ``__name__``. This helps locate the ``root_path`` for the
        blueprint.
    :param static_folder: A folder with static files that should be
        served by the blueprint's static route. The path is relative to
        the blueprint's root path. Blueprint static files are disabled
        by default.
    :param static_url_path: The url to serve static files from.
        Defaults to ``static_folder``. If the blueprint does not have
        a ``url_prefix``, the app's static route will take precedence,
        and the blueprint's static files won't be accessible.
    :param template_folder: A folder with templates that should be added
        to the app's template search path. The path is relative to the
        blueprint's root path. Blueprint templates are disabled by
        default. Blueprint templates have a lower precedence than those
        in the app's templates folder.
    :param url_prefix: A path to prepend to all of the blueprint's URLs,
        to make them distinct from the rest of the app's routes.
    :param subdomain: A subdomain that blueprint routes will match on by
        default.
    :param url_defaults: A dict of default values that blueprint routes
        will receive by default.
    :param root_path: By default, the blueprint will automatically set
        this based on ``import_name``. In certain situations this
        automatic detection can fail, so the path can be specified
        manually instead.

    .. versionchanged:: 1.1.0
        Blueprints have a ``cli`` group to register nested CLI commands.
        The ``cli_group`` parameter controls the name of the group under
        the ``flask`` command.

    .. versionadded:: 0.7
    """

    _got_registered_once = False

    def __init__(
        self,
        name: str,
        import_name: str,
        static_folder: str | os.PathLike[str] | None = None,
        static_url_path: str | None = None,
        template_folder: str | os.PathLike[str] | None = None,
        url_prefix: str | None = None,
        subdomain: str | None = None,
        url_defaults: dict[str, t.Any] | None = None,
        root_path: str | None = None,
        cli_group: str | None = _sentinel,  # type: ignore[assignment]
    ):
        super().__init__(
            import_name=import_name,
            static_folder=static_folder,
            static_url_path=static_url_path,
            template_folder=template_folder,
            root_path=root_path,
        )

        if not name:
            raise ValueError("'name' may not be empty.")

        if "." in name:
            raise ValueError("'name' may not contain a dot '.' character.")

        self.name = name
        self.url_prefix = url_prefix
        self.subdomain = subdomain
        self.deferred_functions: list[DeferredSetupFunction] = []

        if url_defaults is None:
            url_defaults = {}

        self.url_values_defaults = url_defaults
        self.cli_group = cli_group
        self._blueprints: list[tuple[Blueprint, dict[str, t.Any]]] = []

    def _check_setup_finished(self, f_name: str) -> None:
        if self._got_registered_once:
            raise AssertionError(
                f"The setup method '{f_name}' can no longer be called on the blueprint"
                f" '{self.name}'. It has already been registered at least once, any"
                " changes will not be applied consistently.\n"
                "Make sure all imports, decorators, functions, etc. needed to set up"
                " the blueprint are done before registering it."
            )

    @setupmethod
    def record(self, func: DeferredSetupFunction) -> None:
        """Registers a function that is called when the blueprint is
        registered on the application.  This function is called with the
        state as argument as returned by the :meth:`make_setup_state`
        method.
        """
        self.deferred_functions.append(func)

    @setupmethod
    def record_once(self, func: DeferredSetupFunction) -> None:
        """Works like :meth:`record` but wraps the function in another
        function that will ensure the function is only called once.  If the
        blueprint is registered a second time on the application, the
        function passed is not called.
        """

        def wrapper(state: BlueprintSetupState) -> None:
            if state.first_registration:
                func(state)

        self.record(update_wrapper(wrapper, func))

    def make_setup_state(
        self, app: App, options: dict[str, t.Any], first_registration: bool = False
    ) -> BlueprintSetupState:
        """Creates an instance of :meth:`~flask.blueprints.BlueprintSetupState`
        object that is later passed to the register callback functions.
        Subclasses can override this to return a subclass of the setup state.
        """
        return BlueprintSetupState(self, app, options, first_registration)

    @setupmethod
    def register_blueprint(self, blueprint: Blueprint, **options: t.Any) -> None:
        """Register a :class:`~flask.Blueprint` on this blueprint. Keyword
        arguments passed to this method will override the defaults set
        on the blueprint.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.

        .. versionadded:: 2.0
        """
        if blueprint is self:
            raise ValueError("Cannot register a blueprint on itself")
        self._blueprints.append((blueprint, options))

    def register(self, app: App, options: dict[str, t.Any]) -> None:
        """Called by :meth:`Flask.register_blueprint` to register all
        views and callbacks registered on the blueprint with the
        application. Creates a :class:`.BlueprintSetupState` and calls
        each :meth:`record` callback with it.

        :param app: The application this blueprint is being registered
            with.
        :param options: Keyword arguments forwarded from
            :meth:`~Flask.register_blueprint`.

        .. versionchanged:: 2.3
            Nested blueprints now correctly apply subdomains.

        .. versionchanged:: 2.1
            Registering the same blueprint with the same name multiple
            times is an error.

        .. versionchanged:: 2.0.1
            Nested blueprints are registered with their dotted name.
            This allows different blueprints with the same name to be
            nested at different locations.

        .. versionchanged:: 2.0.1
            The ``name`` option can be used to change the (pre-dotted)
            name the blueprint is registered with. This allows the same
            blueprint to be registered multiple times with unique names
            for ``url_for``.
        """
        name_prefix = options.get("name_prefix", "")
        self_name = options.get("name", self.name)
        name = f"{name_prefix}.{self_name}".lstrip(".")

        if name in app.blueprints:
            bp_desc = "this" if app.blueprints[name] is self else "a different"
            existing_at = f" '{name}'" if self_name != name else ""

            raise ValueError(
                f"The name '{self_name}' is already registered for"
                f" {bp_desc} blueprint{existing_at}. Use 'name=' to"
                f" provide a unique name."
            )

        first_bp_registration = not any(bp is self for bp in app.blueprints.values())
        first_name_registration = name not in app.blueprints

        app.blueprints[name] = self
        self._got_registered_once = True
        state = self.make_setup_state(app, options, first_bp_registration)

        if self.has_static_folder:
            state.add_url_rule(
                f"{self.static_url_path}/<path:filename>",
                view_func=self.send_static_file,  # type: ignore[attr-defined]
                endpoint="static",
            )

        # Merge blueprint data into parent.
        if first_bp_registration or first_name_registration:
            self._merge_blueprint_funcs(app, name)

        for deferred in self.deferred_functions:
            deferred(state)

        cli_resolved_group = options.get("cli_group", self.cli_group)

        if self.cli.commands:
            if cli_resolved_group is None:
                app.cli.commands.update(self.cli.commands)
            elif cli_resolved_group is _sentinel:
                self.cli.name = name
                app.cli.add_command(self.cli)
            else:
                self.cli.name = cli_resolved_group
                app.cli.add_command(self.cli)

        for blueprint, bp_options in self._blueprints:
            bp_options = bp_options.copy()
            bp_url_prefix = bp_options.get("url_prefix")
            bp_subdomain = bp_options.get("subdomain")

            if bp_subdomain is None:
                bp_subdomain = blueprint.subdomain

            if state.subdomain is not None and bp_subdomain is not None:
                bp_options["subdomain"] = bp_subdomain + "." + state.subdomain
            elif bp_subdomain is not None:
                bp_options["subdomain"] = bp_subdomain
            elif state.subdomain is not None:
                bp_options["subdomain"] = state.subdomain

            if bp_url_prefix is None:
                bp_url_prefix = blueprint.url_prefix

            if state.url_prefix is not None and bp_url_prefix is not None:
                bp_options["url_prefix"] = (
                    state.url_prefix.rstrip("/") + "/" + bp_url_prefix.lstrip("/")
                )
            elif bp_url_prefix is not None:
                bp_options["url_prefix"] = bp_url_prefix
            elif state.url_prefix is not None:
                bp_options["url_prefix"] = state.url_prefix

            bp_options["name_prefix"] = name
            blueprint.register(app, bp_options)

    def _merge_blueprint_funcs(self, app: App, name: str) -> None:
        def extend(
            bp_dict: dict[ft.AppOrBlueprintKey, list[t.Any]],
            parent_dict: dict[ft.AppOrBlueprintKey, list[t.Any]],
        ) -> None:
            for key, values in bp_dict.items():
                key = name if key is None else f"{name}.{key}"
                parent_dict[key].extend(values)

        for key, value in self.error_handler_spec.items():
            key = name if key is None else f"{name}.{key}"
            value = defaultdict(
                dict,
                {
                    code: {exc_class: func for exc_class, func in code_values.items()}
                    for code, code_values in value.items()
                },
            )
            app.error_handler_spec[key] = value

        for endpoint, func in self.view_functions.items():
            app.view_functions[endpoint] = func

        extend(self.before_request_funcs, app.before_request_funcs)
        extend(self.after_request_funcs, app.after_request_funcs)
        extend(
            self.teardown_request_funcs,
            app.teardown_request_funcs,
        )
        extend(self.url_default_functions, app.url_default_functions)
        extend(self.url_value_preprocessors, app.url_value_preprocessors)
        extend(self.template_context_processors, app.template_context_processors)

    @setupmethod
    def add_url_rule(
        self,
        rule: str,
        endpoint: str | None = None,
        view_func: ft.RouteCallable | None = None,
        provide_automatic_options: bool | None = None,
        **options: t.Any,
    ) -> None:
        """Register a URL rule with the blueprint. See :meth:`.Flask.add_url_rule` for
        full documentation.

        The URL rule is prefixed with the blueprint's URL prefix. The endpoint name,
        used with :func:`url_for`, is prefixed with the blueprint's name.
        """
        if endpoint and "." in endpoint:
            raise ValueError("'endpoint' may not contain a dot '.' character.")

        if view_func and hasattr(view_func, "__name__") and "." in view_func.__name__:
            raise ValueError("'view_func' name may not contain a dot '.' character.")

        self.record(
            lambda s: s.add_url_rule(
                rule,
                endpoint,
                view_func,
                provide_automatic_options=provide_automatic_options,
                **options,
            )
        )

    @setupmethod
    def app_template_filter(
        self, name: str | None = None
    ) -> t.Callable[[T_template_filter], T_template_filter]:
        """Register a template filter, available in any template rendered by the
        application. Equivalent to :meth:`.Flask.template_filter`.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_filter) -> T_template_filter:
            self.add_app_template_filter(f, name=name)
            return f

        return decorator

    @setupmethod
    def add_app_template_filter(
        self, f: ft.TemplateFilterCallable, name: str | None = None
    ) -> None:
        """Register a template filter, available in any template rendered by the
        application. Works like the :meth:`app_template_filter` decorator. Equivalent to
        :meth:`.Flask.add_template_filter`.

        :param name: the optional name of the filter, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.filters[name or f.__name__] = f

        self.record_once(register_template)

    @setupmethod
    def app_template_test(
        self, name: str | None = None
    ) -> t.Callable[[T_template_test], T_template_test]:
        """Register a template test, available in any template rendered by the
        application. Equivalent to :meth:`.Flask.template_test`.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_test) -> T_template_test:
            self.add_app_template_test(f, name=name)
            return f

        return decorator

    @setupmethod
    def add_app_template_test(
        self, f: ft.TemplateTestCallable, name: str | None = None
    ) -> None:
        """Register a template test, available in any template rendered by the
        application. Works like the :meth:`app_template_test` decorator. Equivalent to
        :meth:`.Flask.add_template_test`.

        .. versionadded:: 0.10

        :param name: the optional name of the test, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.tests[name or f.__name__] = f

        self.record_once(register_template)

    @setupmethod
    def app_template_global(
        self, name: str | None = None
    ) -> t.Callable[[T_template_global], T_template_global]:
        """Register a template global, available in any template rendered by the
        application. Equivalent to :meth:`.Flask.template_global`.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def decorator(f: T_template_global) -> T_template_global:
            self.add_app_template_global(f, name=name)
            return f

        return decorator

    @setupmethod
    def add_app_template_global(
        self, f: ft.TemplateGlobalCallable, name: str | None = None
    ) -> None:
        """Register a template global, available in any template rendered by the
        application. Works like the :meth:`app_template_global` decorator. Equivalent to
        :meth:`.Flask.add_template_global`.

        .. versionadded:: 0.10

        :param name: the optional name of the global, otherwise the
                     function name will be used.
        """

        def register_template(state: BlueprintSetupState) -> None:
            state.app.jinja_env.globals[name or f.__name__] = f

        self.record_once(register_template)

    @setupmethod
    def before_app_request(self, f: T_before_request) -> T_before_request:
        """Like :meth:`before_request`, but before every request, not only those handled
        by the blueprint. Equivalent to :meth:`.Flask.before_request`.
        """
        self.record_once(
            lambda s: s.app.before_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def after_app_request(self, f: T_after_request) -> T_after_request:
        """Like :meth:`after_request`, but after every request, not only those handled
        by the blueprint. Equivalent to :meth:`.Flask.after_request`.
        """
        self.record_once(
            lambda s: s.app.after_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def teardown_app_request(self, f: T_teardown) -> T_teardown:
        """Like :meth:`teardown_request`, but after every request, not only those
        handled by the blueprint. Equivalent to :meth:`.Flask.teardown_request`.
        """
        self.record_once(
            lambda s: s.app.teardown_request_funcs.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_context_processor(
        self, f: T_template_context_processor
    ) -> T_template_context_processor:
        """Like :meth:`context_processor`, but for templates rendered by every view, not
        only by the blueprint. Equivalent to :meth:`.Flask.context_processor`.
        """
        self.record_once(
            lambda s: s.app.template_context_processors.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_errorhandler(
        self, code: type[Exception] | int
    ) -> t.Callable[[T_error_handler], T_error_handler]:
        """Like :meth:`errorhandler`, but for every request, not only those handled by
        the blueprint. Equivalent to :meth:`.Flask.errorhandler`.
        """

        def decorator(f: T_error_handler) -> T_error_handler:
            def from_blueprint(state: BlueprintSetupState) -> None:
                state.app.errorhandler(code)(f)

            self.record_once(from_blueprint)
            return f

        return decorator

    @setupmethod
    def app_url_value_preprocessor(
        self, f: T_url_value_preprocessor
    ) -> T_url_value_preprocessor:
        """Like :meth:`url_value_preprocessor`, but for every request, not only those
        handled by the blueprint. Equivalent to :meth:`.Flask.url_value_preprocessor`.
        """
        self.record_once(
            lambda s: s.app.url_value_preprocessors.setdefault(None, []).append(f)
        )
        return f

    @setupmethod
    def app_url_defaults(self, f: T_url_defaults) -> T_url_defaults:
        """Like :meth:`url_defaults`, but for every request, not only those handled by
        the blueprint. Equivalent to :meth:`.Flask.url_defaults`.
        """
        self.record_once(
            lambda s: s.app.url_default_functions.setdefault(None, []).append(f)
        )
        return f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\rlmain.py ===
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
import re
import sys
import time
from glob import glob

import pyreadline3
import pyreadline3.clipboard as clipboard
import pyreadline3.console as console
import pyreadline3.lineeditor.history as history
import pyreadline3.lineeditor.lineobj as lineobj
import pyreadline3.logger as logger
from pyreadline3.keysyms.common import make_KeyPress_from_keydescr
from pyreadline3.py3k_compat import is_ironpython
from pyreadline3.unicode_helper import ensure_str, ensure_unicode

from .error import GetSetError, ReadlineError
from .logger import log
from .modes import editingmodes
from .py3k_compat import execfile, is_callable

# an attempt to implement readline for Python in Python using ctypes


if is_ironpython:  # ironpython does not provide a prompt string to readline
    import System

    default_prompt = ">>> "
else:
    default_prompt = ""


class MockConsoleError(Exception):
    pass


class MockConsole(object):
    """object used during refactoring. Should raise errors when someone tries
    to use it.
    """

    def __setattr__(self, _name, _value):
        raise MockConsoleError("Should not try to get attributes from MockConsole")

    def cursor(self, size=50):
        pass


class BaseReadline(object):
    def __init__(self):
        self.allow_ctrl_c = False
        self.ctrl_c_tap_time_interval = 0.3

        self.debug = False
        self.bell_style = "none"
        self.mark = -1
        self.console = MockConsole()
        self.disable_readline = False
        # this code needs to follow l_buffer and history creation
        self.editingmodes = [mode(self) for mode in editingmodes]
        for mode in self.editingmodes:
            mode.init_editing_mode(None)
        self.mode = self.editingmodes[0]

        self.read_inputrc()
        log("\n".join(self.mode.rl_settings_to_string()))

        self.callback = None

    def parse_and_bind(self, string):
        """Parse and execute single line of a readline init file."""
        try:
            log('parse_and_bind("%s")' % string)
            if string.startswith("#"):
                return
            if string.startswith("set"):
                m = re.compile(r"set\s+([-a-zA-Z0-9]+)\s+(.+)\s*$").match(string)
                if m:
                    var_name = m.group(1)
                    val = m.group(2)
                    try:
                        setattr(self.mode, var_name.replace("-", "_"), val)
                    except AttributeError:
                        log('unknown var="%s" val="%s"' % (var_name, val))
                else:
                    log('bad set "%s"' % string)
                return
            m = re.compile(r"\s*(\S+)\s*:\s*([-a-zA-Z]+)\s*$").match(string)
            if m:
                key = m.group(1)
                func_name = m.group(2)
                py_name = func_name.replace("-", "_")
                try:
                    func = getattr(self.mode, py_name)
                except AttributeError:
                    log('unknown func key="%s" func="%s"' % (key, func_name))
                    if self.debug:
                        print(
                            "pyreadline3 parse_and_bind error, unknown "
                            'function to bind: "%s"' % func_name
                        )
                    return
                self.mode._bind_key(key, func)
        except BaseException:
            log("error")
            raise

    def _set_prompt(self, prompt):
        self.mode.prompt = prompt

    def _get_prompt(self):
        return self.mode.prompt

    prompt = property(_get_prompt, _set_prompt)

    def get_line_buffer(self):
        """Return the current contents of the line buffer."""
        return self.mode.l_buffer.get_line_text()

    def insert_text(self, string):
        """Insert text into the command line."""
        self.mode.insert_text(string)

    def read_init_file(self, filename=None):
        """Parse a readline initialization file. The default filename is the last filename used."""
        log('read_init_file("%s")' % filename)

    # History file book keeping methods (non-bindable)

    def add_history(self, line):
        """Append a line to the history buffer, as if it was the last line typed."""
        self.mode._history.add_history(line)

    def get_current_history_length(self):
        """Return the number of lines currently in the history.
        (This is different from get_history_length(), which returns
        the maximum number of lines that will be written to a history file.)"""
        return self.mode._history.get_current_history_length()

    def get_history_length(self):
        """Return the desired length of the history file.

        Negative values imply unlimited history file size."""
        return self.mode._history.get_history_length()

    def set_history_length(self, length):
        """Set the number of lines to save in the history file.

        write_history_file() uses this value to truncate the history file
        when saving. Negative values imply unlimited history file size.
        """
        self.mode._history.set_history_length(length)

    def get_history_item(self, index):
        """Return the current contents of history item at index."""
        return self.mode._history.get_history_item(index)

    def clear_history(self):
        """Clear readline history"""
        self.mode._history.clear_history()

    def read_history_file(self, filename=None):
        """Load a readline history file. The default filename is ~/.history."""
        if filename is None:
            filename = self.mode._history.history_filename
        log("read_history_file from %s" % ensure_unicode(filename))
        self.mode._history.read_history_file(filename)

    def write_history_file(self, filename=None):
        """Save a readline history file. The default filename is ~/.history."""
        self.mode._history.write_history_file(filename)

    # Completer functions

    def set_completer(self, function=None):
        """Set or remove the completer function.

        If function is specified, it will be used as the new completer
        function; if omitted or None, any completer function already
        installed is removed. The completer function is called as
        function(text, state), for state in 0, 1, 2, ..., until it returns a
        non-string value. It should return the next possible completion
        starting with text.
        """
        log("set_completer")
        self.mode.completer = function

    def get_completer(self):
        """Get the completer function."""
        log("get_completer")
        return self.mode.completer

    def get_begidx(self):
        """Get the beginning index of the readline tab-completion scope."""
        return self.mode.begidx

    def get_endidx(self):
        """Get the ending index of the readline tab-completion scope."""
        return self.mode.endidx

    def set_completer_delims(self, string):
        """Set the readline word delimiters for tab-completion."""
        self.mode.completer_delims = string

    def get_completer_delims(self):
        """Get the readline word delimiters for tab-completion."""
        return self.mode.completer_delims

    def set_startup_hook(self, function=None):
        """Set or remove the startup_hook function.

        If function is specified, it will be used as the new startup_hook
        function; if omitted or None, any hook function already installed is
        removed. The startup_hook function is called with no arguments just
        before readline prints the first prompt.

        """
        self.mode.startup_hook = function

    def set_pre_input_hook(self, function=None):
        """Set or remove the pre_input_hook function.

        If function is specified, it will be used as the new pre_input_hook
        function; if omitted or None, any hook function already installed is
        removed. The pre_input_hook function is called with no arguments
        after the first prompt has been printed and just before readline
        starts reading input characters.

        """
        self.mode.pre_input_hook = function

    # Functions that are not relevant for all Readlines but should at least
    # have a NOP

    def _bell(self):
        pass

    #
    # Standard call, not available for all implementations
    #

    def readline(self, prompt=""):
        raise NotImplementedError

    #
    # Callback interface
    #
    def process_keyevent(self, keyinfo):
        return self.mode.process_keyevent(keyinfo)

    def readline_setup(self, prompt=""):
        return self.mode.readline_setup(prompt)

    def keyboard_poll(self):
        return self.mode._readline_from_keyboard_poll()

    def callback_handler_install(self, prompt, callback):
        """bool readline_callback_handler_install ( string prompt, callback callback)
        Initializes the readline callback interface and terminal, prints the prompt and returns immediately
        """
        self.callback = callback
        self.readline_setup(prompt)

    def callback_handler_remove(self):
        """Removes a previously installed callback handler and restores terminal settings"""
        self.callback = None

    def callback_read_char(self):
        """Reads a character and informs the readline callback interface when a line is received"""
        if self.keyboard_poll():
            line = self.get_line_buffer() + "\n"
            # however there is another newline added by
            # self.mode.readline_setup(prompt) which is called by callback_handler_install
            # this differs from GNU readline
            self.add_history(self.mode.l_buffer)
            # TADA:
            self.callback(line)

    def read_inputrc(
        self,  # in 2.4 we cannot call expanduser with unicode string
        inputrcpath=os.path.expanduser(ensure_str("~/pyreadlineconfig.ini")),
    ):
        modes = dict([(x.mode, x) for x in self.editingmodes])
        mode = self.editingmodes[0].mode

        def setmode(name):
            self.mode = modes[name]

        def bind_key(key, name):
            import types

            if is_callable(name):
                modes[mode]._bind_key(key, types.MethodType(name, modes[mode]))
            elif hasattr(modes[mode], name):
                modes[mode]._bind_key(key, getattr(modes[mode], name))
            else:
                print("Trying to bind unknown command '%s' to key '%s'" % (name, key))

        def un_bind_key(key):
            keyinfo = make_KeyPress_from_keydescr(key).tuple()
            if keyinfo in modes[mode].key_dispatch:
                del modes[mode].key_dispatch[keyinfo]

        def bind_exit_key(key):
            modes[mode]._bind_exit_key(key)

        def un_bind_exit_key(key):
            keyinfo = make_KeyPress_from_keydescr(key).tuple()
            if keyinfo in modes[mode].exit_dispatch:
                del modes[mode].exit_dispatch[keyinfo]

        def setkill_ring_to_clipboard(killring):
            import pyreadline3.lineeditor.lineobj

            pyreadline3.lineeditor.lineobj.kill_ring_to_clipboard = killring

        def sethistoryfilename(filename):
            self.mode._history.history_filename = os.path.expanduser(
                ensure_str(filename)
            )

        def setbellstyle(mode):
            self.bell_style = mode

        def disable_readline(mode):
            self.disable_readline = mode

        def sethistorylength(length):
            self.mode._history.history_length = int(length)

        def allow_ctrl_c(mode):
            log("allow_ctrl_c:%s:%s" % (self.allow_ctrl_c, mode))
            self.allow_ctrl_c = mode

        def setbellstyle(mode):
            self.bell_style = mode

        def show_all_if_ambiguous(mode):
            self.mode.show_all_if_ambiguous = mode

        def ctrl_c_tap_time_interval(mode):
            self.ctrl_c_tap_time_interval = mode

        def mark_directories(mode):
            self.mode.mark_directories = mode

        def completer_delims(delims):
            self.mode.completer_delims = delims

        def complete_filesystem(delims):
            self.mode.complete_filesystem = delims.lower()

        def enable_ipython_paste_for_paths(boolean):
            self.mode.enable_ipython_paste_for_paths = boolean

        def debug_output(
            on, filename="pyreadline_debug_log.txt"
        ):  # Not implemented yet
            if on in ["on", "on_nologfile"]:
                self.debug = True

            if on == "on":
                logger.start_file_log(filename)
                logger.start_socket_log()
                logger.log("STARTING LOG")
            elif on == "on_nologfile":
                logger.start_socket_log()
                logger.log("STARTING LOG")
            else:
                logger.log("STOPING LOG")
                logger.stop_file_log()
                logger.stop_socket_log()

        _color_trtable = {
            "black": 0,
            "darkred": 4,
            "darkgreen": 2,
            "darkyellow": 6,
            "darkblue": 1,
            "darkmagenta": 5,
            "darkcyan": 3,
            "gray": 7,
            "red": 4 + 8,
            "green": 2 + 8,
            "yellow": 6 + 8,
            "blue": 1 + 8,
            "magenta": 5 + 8,
            "cyan": 3 + 8,
            "white": 7 + 8,
        }

        def set_prompt_color(color):
            self.prompt_color = self._color_trtable.get(color.lower(), 7)

        def set_input_color(color):
            self.command_color = self._color_trtable.get(color.lower(), 7)

        loc = {
            "version": pyreadline3.__version__,
            "mode": mode,
            "modes": modes,
            "set_mode": setmode,
            "bind_key": bind_key,
            "disable_readline": disable_readline,
            "bind_exit_key": bind_exit_key,
            "un_bind_key": un_bind_key,
            "un_bind_exit_key": un_bind_exit_key,
            "bell_style": setbellstyle,
            "mark_directories": mark_directories,
            "show_all_if_ambiguous": show_all_if_ambiguous,
            "completer_delims": completer_delims,
            "complete_filesystem": complete_filesystem,
            "debug_output": debug_output,
            "history_filename": sethistoryfilename,
            "history_length": sethistorylength,
            "set_prompt_color": set_prompt_color,
            "set_input_color": set_input_color,
            "allow_ctrl_c": allow_ctrl_c,
            "ctrl_c_tap_time_interval": ctrl_c_tap_time_interval,
            "kill_ring_to_clipboard": setkill_ring_to_clipboard,
            "enable_ipython_paste_for_paths": enable_ipython_paste_for_paths,
        }
        if os.path.isfile(inputrcpath):
            try:
                execfile(inputrcpath, loc, loc)
            except Exception as x:
                raise
                import traceback

                print("Error reading .pyinputrc", file=sys.stderr)
                filepath, lineno = traceback.extract_tb(sys.exc_info()[2])[1][:2]
                print("Line: %s in file %s" % (lineno, filepath), file=sys.stderr)
                print(x, file=sys.stderr)
                raise ReadlineError("Error reading .pyinputrc")

    def redisplay(self):
        pass


class Readline(BaseReadline):
    """Baseclass for readline based on a console"""

    def __init__(self):
        super().__init__()

        self.console = console.Console()
        self.selection_color = self.console.saveattr << 4
        self.command_color = None
        self.prompt_color = None
        self.size = self.console.size()

        # variables you can control with parse_and_bind

    #  To export as readline interface

    # Internal functions

    def _bell(self):
        """ring the bell if requested."""
        if self.bell_style == "none":
            pass
        elif self.bell_style == "visible":
            raise NotImplementedError("Bellstyle visible is not implemented yet.")
        elif self.bell_style == "audible":
            self.console.bell()
        else:
            raise ReadlineError("Bellstyle %s unknown." % self.bell_style)

    def _clear_after(self):
        c = self.console
        x, y = c.pos()
        w, h = c.size()
        c.rectangle((x, y, w + 1, y + 1))
        c.rectangle((0, y + 1, w, min(y + 3, h)))

    def _set_cursor(self):
        c = self.console
        xc, yc = self.prompt_end_pos
        w, h = c.size()
        xc += self.mode.l_buffer.visible_line_width()
        while xc >= w:
            xc -= w
            yc += 1
        c.pos(xc, yc)

    def _print_prompt(self):
        c = self.console
        x, y = c.pos()

        n = c.write_scrolling(self.prompt, self.prompt_color)
        self.prompt_begin_pos = (x, y - n)
        self.prompt_end_pos = c.pos()
        self.size = c.size()

    def _update_prompt_pos(self, n):
        if n != 0:
            bx, by = self.prompt_begin_pos
            ex, ey = self.prompt_end_pos
            self.prompt_begin_pos = (bx, by - n)
            self.prompt_end_pos = (ex, ey - n)

    def _update_line(self):
        c = self.console
        l_buffer = self.mode.l_buffer
        c.cursor(0)  # Hide cursor avoiding flicking
        c.pos(*self.prompt_begin_pos)
        self._print_prompt()
        ltext = l_buffer.quoted_text()
        if l_buffer.enable_selection and (l_buffer.selection_mark >= 0):
            start = len(l_buffer[: l_buffer.selection_mark].quoted_text())
            stop = len(l_buffer[: l_buffer.point].quoted_text())
            if start > stop:
                stop, start = start, stop
            n = c.write_scrolling(ltext[:start], self.command_color)
            n = c.write_scrolling(ltext[start:stop], self.selection_color)
            n = c.write_scrolling(ltext[stop:], self.command_color)
        else:
            n = c.write_scrolling(ltext, self.command_color)

        x, y = c.pos()  # Preserve one line for Asian IME(Input Method Editor) statusbar
        w, h = c.size()
        if (y >= h - 1) or (n > 0):
            c.scroll_window(-1)
            c.scroll((0, 0, w, h), 0, -1)
            n += 1

        self._update_prompt_pos(n)
        if hasattr(
            c, "clear_to_end_of_window"
        ):  # Work around function for ironpython due
            c.clear_to_end_of_window()  # to System.Console's lack of FillFunction
        else:
            self._clear_after()

        # Show cursor, set size vi mode changes size in insert/overwrite mode
        c.cursor(1, size=self.mode.cursor_size)
        self._set_cursor()

    def callback_read_char(self):
        # Override base to get automatic newline
        """Reads a character and informs the readline callback interface when a line is received"""
        if self.keyboard_poll():
            line = self.get_line_buffer() + "\n"
            self.console.write("\r\n")
            # however there is another newline added by
            # self.mode.readline_setup(prompt) which is called by callback_handler_install
            # this differs from GNU readline
            self.add_history(self.mode.l_buffer)
            # TADA:
            self.callback(line)

    def event_available(self):
        return self.console.peek() or (len(self.paste_line_buffer) > 0)

    def _readline_from_keyboard(self):
        while True:
            if self._readline_from_keyboard_poll():
                break

    def _readline_from_keyboard_poll(self):
        pastebuffer = self.mode.paste_line_buffer
        if len(pastebuffer) > 0:
            # paste first line in multiline paste buffer
            self.l_buffer = lineobj.ReadLineTextBuffer(pastebuffer[0])
            self._update_line()
            self.mode.paste_line_buffer = pastebuffer[1:]
            return True

        c = self.console

        def nop(e):
            pass

        try:
            event = c.getkeypress()
        except KeyboardInterrupt:
            event = self.handle_ctrl_c()
        try:
            result = self.mode.process_keyevent(event.keyinfo)
        except EOFError:
            logger.stop_logging()
            raise
        self._update_line()
        return result

    def readline_setup(self, prompt=""):
        BaseReadline.readline_setup(self, prompt)
        self._print_prompt()
        self._update_line()

    def readline(self, prompt=""):
        self.readline_setup(prompt)
        self.ctrl_c_timeout = time.time()
        self._readline_from_keyboard()
        self.console.write("\r\n")
        log("returning(%s)" % self.get_line_buffer())
        return self.get_line_buffer() + "\n"

    def handle_ctrl_c(self):
        from pyreadline3.console.event import Event
        from pyreadline3.keysyms.common import KeyPress

        log("KBDIRQ")
        event = Event(0, 0)
        event.char = "c"
        event.keyinfo = KeyPress(
            "c", shift=False, control=True, meta=False, keyname=None
        )
        if self.allow_ctrl_c:
            now = time.time()
            if (now - self.ctrl_c_timeout) < self.ctrl_c_tap_time_interval:
                log("Raise KeyboardInterrupt")
                raise KeyboardInterrupt
            else:
                self.ctrl_c_timeout = now
        else:
            raise KeyboardInterrupt
        return event

    def redisplay(self):
        BaseReadline.redisplay(self)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\platformdirs\__init__.py ===
"""
Utilities for determining application-specific dirs.

See <https://github.com/platformdirs/platformdirs> for details and usage.

"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from .api import PlatformDirsABC
from .version import __version__
from .version import __version_tuple__ as __version_info__

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

if sys.platform == "win32":
    from pip._vendor.platformdirs.windows import Windows as _Result
elif sys.platform == "darwin":
    from pip._vendor.platformdirs.macos import MacOS as _Result
else:
    from pip._vendor.platformdirs.unix import Unix as _Result


def _set_platform_dir_class() -> type[PlatformDirsABC]:
    if os.getenv("ANDROID_DATA") == "/data" and os.getenv("ANDROID_ROOT") == "/system":
        if os.getenv("SHELL") or os.getenv("PREFIX"):
            return _Result

        from pip._vendor.platformdirs.android import _android_folder  # noqa: PLC0415

        if _android_folder() is not None:
            from pip._vendor.platformdirs.android import Android  # noqa: PLC0415

            return Android  # return to avoid redefinition of a result

    return _Result


if TYPE_CHECKING:
    # Work around mypy issue: https://github.com/python/mypy/issues/10962
    PlatformDirs = _Result
else:
    PlatformDirs = _set_platform_dir_class()  #: Currently active platform
AppDirs = PlatformDirs  #: Backwards compatibility with appdirs


def user_data_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.roaming>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: data directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
    ).user_data_dir


def site_data_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    multipath: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `roaming <platformdirs.api.PlatformDirsABC.multipath>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: data directory shared by users
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        multipath=multipath,
        ensure_exists=ensure_exists,
    ).site_data_dir


def user_config_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.roaming>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: config directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
    ).user_config_dir


def site_config_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    multipath: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `roaming <platformdirs.api.PlatformDirsABC.multipath>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: config directory shared by the users
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        multipath=multipath,
        ensure_exists=ensure_exists,
    ).site_config_dir


def user_cache_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: cache directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).user_cache_dir


def site_cache_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: cache directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).site_cache_dir


def user_state_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.roaming>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: state directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
    ).user_state_dir


def user_log_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: log directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).user_log_dir


def user_documents_dir() -> str:
    """:returns: documents directory tied to the user"""
    return PlatformDirs().user_documents_dir


def user_downloads_dir() -> str:
    """:returns: downloads directory tied to the user"""
    return PlatformDirs().user_downloads_dir


def user_pictures_dir() -> str:
    """:returns: pictures directory tied to the user"""
    return PlatformDirs().user_pictures_dir


def user_videos_dir() -> str:
    """:returns: videos directory tied to the user"""
    return PlatformDirs().user_videos_dir


def user_music_dir() -> str:
    """:returns: music directory tied to the user"""
    return PlatformDirs().user_music_dir


def user_desktop_dir() -> str:
    """:returns: desktop directory tied to the user"""
    return PlatformDirs().user_desktop_dir


def user_runtime_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: runtime directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).user_runtime_dir


def site_runtime_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: runtime directory shared by users
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).site_runtime_dir


def user_data_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.roaming>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: data path tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
    ).user_data_path


def site_data_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    multipath: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `multipath <platformdirs.api.PlatformDirsABC.multipath>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: data path shared by users
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        multipath=multipath,
        ensure_exists=ensure_exists,
    ).site_data_path


def user_config_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.roaming>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: config path tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
    ).user_config_path


def site_config_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    multipath: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param multipath: See `roaming <platformdirs.api.PlatformDirsABC.multipath>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: config path shared by the users
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        multipath=multipath,
        ensure_exists=ensure_exists,
    ).site_config_path


def site_cache_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: cache directory tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).site_cache_path


def user_cache_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: cache path tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).user_cache_path


def user_state_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param roaming: See `roaming <platformdirs.api.PlatformDirsABC.roaming>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: state path tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        roaming=roaming,
        ensure_exists=ensure_exists,
    ).user_state_path


def user_log_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `roaming <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: log path tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).user_log_path


def user_documents_path() -> Path:
    """:returns: documents a path tied to the user"""
    return PlatformDirs().user_documents_path


def user_downloads_path() -> Path:
    """:returns: downloads path tied to the user"""
    return PlatformDirs().user_downloads_path


def user_pictures_path() -> Path:
    """:returns: pictures path tied to the user"""
    return PlatformDirs().user_pictures_path


def user_videos_path() -> Path:
    """:returns: videos path tied to the user"""
    return PlatformDirs().user_videos_path


def user_music_path() -> Path:
    """:returns: music path tied to the user"""
    return PlatformDirs().user_music_path


def user_desktop_path() -> Path:
    """:returns: desktop path tied to the user"""
    return PlatformDirs().user_desktop_path


def user_runtime_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: runtime path tied to the user
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).user_runtime_path


def site_runtime_path(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,  # noqa: FBT001, FBT002
    ensure_exists: bool = False,  # noqa: FBT001, FBT002
) -> Path:
    """
    :param appname: See `appname <platformdirs.api.PlatformDirsABC.appname>`.
    :param appauthor: See `appauthor <platformdirs.api.PlatformDirsABC.appauthor>`.
    :param version: See `version <platformdirs.api.PlatformDirsABC.version>`.
    :param opinion: See `opinion <platformdirs.api.PlatformDirsABC.opinion>`.
    :param ensure_exists: See `ensure_exists <platformdirs.api.PlatformDirsABC.ensure_exists>`.
    :returns: runtime path shared by users
    """
    return PlatformDirs(
        appname=appname,
        appauthor=appauthor,
        version=version,
        opinion=opinion,
        ensure_exists=ensure_exists,
    ).site_runtime_path


__all__ = [
    "AppDirs",
    "PlatformDirs",
    "PlatformDirsABC",
    "__version__",
    "__version_info__",
    "site_cache_dir",
    "site_cache_path",
    "site_config_dir",
    "site_config_path",
    "site_data_dir",
    "site_data_path",
    "site_runtime_dir",
    "site_runtime_path",
    "user_cache_dir",
    "user_cache_path",
    "user_config_dir",
    "user_config_path",
    "user_data_dir",
    "user_data_path",
    "user_desktop_dir",
    "user_desktop_path",
    "user_documents_dir",
    "user_documents_path",
    "user_downloads_dir",
    "user_downloads_path",
    "user_log_dir",
    "user_log_path",
    "user_music_dir",
    "user_music_path",
    "user_pictures_dir",
    "user_pictures_path",
    "user_runtime_dir",
    "user_runtime_path",
    "user_state_dir",
    "user_state_path",
    "user_videos_dir",
    "user_videos_path",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\monomials.py ===
"""Tools and arithmetics for monomials of distributed polynomials. """


from itertools import combinations_with_replacement, product
from textwrap import dedent

from sympy.core.cache import cacheit
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable

@public
def itermonomials(variables, max_degrees, min_degrees=None):
    r"""
    ``max_degrees`` and ``min_degrees`` are either both integers or both lists.
    Unless otherwise specified, ``min_degrees`` is either ``0`` or
    ``[0, ..., 0]``.

    A generator of all monomials ``monom`` is returned, such that
    either
    ``min_degree <= total_degree(monom) <= max_degree``,
    or
    ``min_degrees[i] <= degree_list(monom)[i] <= max_degrees[i]``,
    for all ``i``.

    Case I. ``max_degrees`` and ``min_degrees`` are both integers
    =============================================================

    Given a set of variables $V$ and a min_degree $N$ and a max_degree $M$
    generate a set of monomials of degree less than or equal to $N$ and greater
    than or equal to $M$. The total number of monomials in commutative
    variables is huge and is given by the following formula if $M = 0$:

        .. math::
            \frac{(\#V + N)!}{\#V! N!}

    For example if we would like to generate a dense polynomial of
    a total degree $N = 50$ and $M = 0$, which is the worst case, in 5
    variables, assuming that exponents and all of coefficients are 32-bit long
    and stored in an array we would need almost 80 GiB of memory! Fortunately
    most polynomials, that we will encounter, are sparse.

    Consider monomials in commutative variables $x$ and $y$
    and non-commutative variables $a$ and $b$::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], 2), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2]

        >>> sorted(itermonomials([x, y], 3), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]

        >>> a, b = symbols('a, b', commutative=False)
        >>> set(itermonomials([a, b, x], 2))
        {1, a, a**2, b, b**2, x, x**2, a*b, b*a, x*a, x*b}

        >>> sorted(itermonomials([x, y], 2, 1), key=monomial_key('grlex', [y, x]))
        [x, y, x**2, x*y, y**2]

    Case II. ``max_degrees`` and ``min_degrees`` are both lists
    ===========================================================

    If ``max_degrees = [d_1, ..., d_n]`` and
    ``min_degrees = [e_1, ..., e_n]``, the number of monomials generated
    is:

    .. math::
        (d_1 - e_1 + 1) (d_2 - e_2 + 1) \cdots (d_n - e_n + 1)

    Let us generate all monomials ``monom`` in variables $x$ and $y$
    such that ``[1, 2][i] <= degree_list(monom)[i] <= [2, 4][i]``,
    ``i = 0, 1`` ::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], [2, 4], [1, 2]), reverse=True, key=monomial_key('lex', [x, y]))
        [x**2*y**4, x**2*y**3, x**2*y**2, x*y**4, x*y**3, x*y**2]
    """
    if is_sequence(max_degrees):
        n = len(variables)
        if len(max_degrees) != n:
            raise ValueError('Argument sizes do not match')
        if min_degrees is None:
            min_degrees = [0]*n
        elif not is_sequence(min_degrees):
            raise ValueError('min_degrees is not a list')
        else:
            if len(min_degrees) != n:
                raise ValueError('Argument sizes do not match')
            if any(i < 0 for i in min_degrees):
                raise ValueError("min_degrees cannot contain negative numbers")
        if any(min_degrees[i] > max_degrees[i] for i in range(n)):
            raise ValueError('min_degrees[i] must be <= max_degrees[i] for all i')
        power_lists = []
        for var, min_d, max_d in zip(variables, min_degrees, max_degrees):
            power_lists.append([var**i for i in range(min_d, max_d + 1)])
        for powers in product(*power_lists):
            yield Mul(*powers)
    else:
        max_degree = max_degrees
        if max_degree < 0:
            raise ValueError("max_degrees cannot be negative")
        if min_degrees is None:
            min_degree = 0
        else:
            if min_degrees < 0:
                raise ValueError("min_degrees cannot be negative")
            min_degree = min_degrees
        if min_degree > max_degree:
            return
        if not variables or max_degree == 0:
            yield S.One
            return
        # Force to list in case of passed tuple or other incompatible collection
        variables = list(variables) + [S.One]
        if all(variable.is_commutative for variable in variables):
            it = combinations_with_replacement(variables, max_degree)
        else:
            it = product(variables, repeat=max_degree)
        monomials_set = set()
        d = max_degree - min_degree
        for item in it:
            count = 0
            for variable in item:
                if variable == 1:
                    count += 1
                    if d < count:
                        break
            else:
                monomials_set.add(Mul(*item))
        yield from monomials_set

def monomial_count(V, N):
    r"""
    Computes the number of monomials.

    The number of monomials is given by the following formula:

    .. math::

        \frac{(\#V + N)!}{\#V! N!}

    where `N` is a total degree and `V` is a set of variables.

    Examples
    ========

    >>> from sympy.polys.monomials import itermonomials, monomial_count
    >>> from sympy.polys.orderings import monomial_key
    >>> from sympy.abc import x, y

    >>> monomial_count(2, 2)
    6

    >>> M = list(itermonomials([x, y], 2))

    >>> sorted(M, key=monomial_key('grlex', [y, x]))
    [1, x, y, x**2, x*y, y**2]
    >>> len(M)
    6

    """
    from sympy.functions.combinatorial.factorials import factorial
    return factorial(V + N) / factorial(V) / factorial(N)

def monomial_mul(A, B):
    """
    Multiplication of tuples representing monomials.

    Examples
    ========

    Lets multiply `x**3*y**4*z` with `x*y**2`::

        >>> from sympy.polys.monomials import monomial_mul

        >>> monomial_mul((3, 4, 1), (1, 2, 0))
        (4, 6, 1)

    which gives `x**4*y**5*z`.

    """
    return tuple([ a + b for a, b in zip(A, B) ])

def monomial_div(A, B):
    """
    Division of tuples representing monomials.

    Examples
    ========

    Lets divide `x**3*y**4*z` by `x*y**2`::

        >>> from sympy.polys.monomials import monomial_div

        >>> monomial_div((3, 4, 1), (1, 2, 0))
        (2, 2, 1)

    which gives `x**2*y**2*z`. However::

        >>> monomial_div((3, 4, 1), (1, 2, 2)) is None
        True

    `x*y**2*z**2` does not divide `x**3*y**4*z`.

    """
    C = monomial_ldiv(A, B)

    if all(c >= 0 for c in C):
        return tuple(C)
    else:
        return None

def monomial_ldiv(A, B):
    """
    Division of tuples representing monomials.

    Examples
    ========

    Lets divide `x**3*y**4*z` by `x*y**2`::

        >>> from sympy.polys.monomials import monomial_ldiv

        >>> monomial_ldiv((3, 4, 1), (1, 2, 0))
        (2, 2, 1)

    which gives `x**2*y**2*z`.

        >>> monomial_ldiv((3, 4, 1), (1, 2, 2))
        (2, 2, -1)

    which gives `x**2*y**2*z**-1`.

    """
    return tuple([ a - b for a, b in zip(A, B) ])

def monomial_pow(A, n):
    """Return the n-th pow of the monomial. """
    return tuple([ a*n for a in A ])

def monomial_gcd(A, B):
    """
    Greatest common divisor of tuples representing monomials.

    Examples
    ========

    Lets compute GCD of `x*y**4*z` and `x**3*y**2`::

        >>> from sympy.polys.monomials import monomial_gcd

        >>> monomial_gcd((1, 4, 1), (3, 2, 0))
        (1, 2, 0)

    which gives `x*y**2`.

    """
    return tuple([ min(a, b) for a, b in zip(A, B) ])

def monomial_lcm(A, B):
    """
    Least common multiple of tuples representing monomials.

    Examples
    ========

    Lets compute LCM of `x*y**4*z` and `x**3*y**2`::

        >>> from sympy.polys.monomials import monomial_lcm

        >>> monomial_lcm((1, 4, 1), (3, 2, 0))
        (3, 4, 1)

    which gives `x**3*y**4*z`.

    """
    return tuple([ max(a, b) for a, b in zip(A, B) ])

def monomial_divides(A, B):
    """
    Does there exist a monomial X such that XA == B?

    Examples
    ========

    >>> from sympy.polys.monomials import monomial_divides
    >>> monomial_divides((1, 2), (3, 4))
    True
    >>> monomial_divides((1, 2), (0, 2))
    False
    """
    return all(a <= b for a, b in zip(A, B))

def monomial_max(*monoms):
    """
    Returns maximal degree for each variable in a set of monomials.

    Examples
    ========

    Consider monomials `x**3*y**4*z**5`, `y**5*z` and `x**6*y**3*z**9`.
    We wish to find out what is the maximal degree for each of `x`, `y`
    and `z` variables::

        >>> from sympy.polys.monomials import monomial_max

        >>> monomial_max((3,4,5), (0,5,1), (6,3,9))
        (6, 5, 9)

    """
    M = list(monoms[0])

    for N in monoms[1:]:
        for i, n in enumerate(N):
            M[i] = max(M[i], n)

    return tuple(M)

def monomial_min(*monoms):
    """
    Returns minimal degree for each variable in a set of monomials.

    Examples
    ========

    Consider monomials `x**3*y**4*z**5`, `y**5*z` and `x**6*y**3*z**9`.
    We wish to find out what is the minimal degree for each of `x`, `y`
    and `z` variables::

        >>> from sympy.polys.monomials import monomial_min

        >>> monomial_min((3,4,5), (0,5,1), (6,3,9))
        (0, 3, 1)

    """
    M = list(monoms[0])

    for N in monoms[1:]:
        for i, n in enumerate(N):
            M[i] = min(M[i], n)

    return tuple(M)

def monomial_deg(M):
    """
    Returns the total degree of a monomial.

    Examples
    ========

    The total degree of `xy^2` is 3:

    >>> from sympy.polys.monomials import monomial_deg
    >>> monomial_deg((1, 2))
    3
    """
    return sum(M)

def term_div(a, b, domain):
    """Division of two terms in over a ring/field. """
    a_lm, a_lc = a
    b_lm, b_lc = b

    monom = monomial_div(a_lm, b_lm)

    if domain.is_Field:
        if monom is not None:
            return monom, domain.quo(a_lc, b_lc)
        else:
            return None
    else:
        if not (monom is None or a_lc % b_lc):
            return monom, domain.quo(a_lc, b_lc)
        else:
            return None

class MonomialOps:
    """Code generator of fast monomial arithmetic functions. """

    @cacheit
    def __new__(cls, ngens):
        obj = super().__new__(cls)
        obj.ngens = ngens
        return obj

    def __getnewargs__(self):
        return (self.ngens,)

    def _build(self, code, name):
        ns = {}
        exec(code, ns)
        return ns[name]

    def _vars(self, name):
        return [ "%s%s" % (name, i) for i in range(self.ngens) ]

    @cacheit
    def mul(self):
        name = "monomial_mul"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s + %s" % (a, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

    @cacheit
    def pow(self):
        name = "monomial_pow"
        template = dedent("""\
        def %(name)s(A, k):
            (%(A)s,) = A
            return (%(Ak)s,)
        """)
        A = self._vars("a")
        Ak = [ "%s*k" % a for a in A ]
        code = template % {"name": name, "A": ", ".join(A), "Ak": ", ".join(Ak)}
        return self._build(code, name)

    @cacheit
    def mulpow(self):
        name = "monomial_mulpow"
        template = dedent("""\
        def %(name)s(A, B, k):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(ABk)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        ABk = [ "%s + %s*k" % (a, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "ABk": ", ".join(ABk)}
        return self._build(code, name)

    @cacheit
    def ldiv(self):
        name = "monomial_ldiv"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s - %s" % (a, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

    @cacheit
    def div(self):
        name = "monomial_div"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            %(RAB)s
            return (%(R)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        RAB = [ "r%(i)s = a%(i)s - b%(i)s\n    if r%(i)s < 0: return None" % {"i": i} for i in range(self.ngens) ]
        R = self._vars("r")
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "RAB": "\n    ".join(RAB), "R": ", ".join(R)}
        return self._build(code, name)

    @cacheit
    def lcm(self):
        name = "monomial_lcm"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s if %s >= %s else %s" % (a, a, b, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

    @cacheit
    def gcd(self):
        name = "monomial_gcd"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s if %s <= %s else %s" % (a, a, b, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

@public
class Monomial(PicklableWithSlots):
    """Class representing a monomial, i.e. a product of powers. """

    __slots__ = ('exponents', 'gens')

    def __init__(self, monom, gens=None):
        if not iterable(monom):
            rep, gens = dict_from_expr(sympify(monom), gens=gens)
            if len(rep) == 1 and list(rep.values())[0] == 1:
                monom = list(rep.keys())[0]
            else:
                raise ValueError("Expected a monomial got {}".format(monom))

        self.exponents = tuple(map(int, monom))
        self.gens = gens

    def rebuild(self, exponents, gens=None):
        return self.__class__(exponents, gens or self.gens)

    def __len__(self):
        return len(self.exponents)

    def __iter__(self):
        return iter(self.exponents)

    def __getitem__(self, item):
        return self.exponents[item]

    def __hash__(self):
        return hash((self.__class__.__name__, self.exponents, self.gens))

    def __str__(self):
        if self.gens:
            return "*".join([ "%s**%s" % (gen, exp) for gen, exp in zip(self.gens, self.exponents) ])
        else:
            return "%s(%s)" % (self.__class__.__name__, self.exponents)

    def as_expr(self, *gens):
        """Convert a monomial instance to a SymPy expression. """
        gens = gens or self.gens

        if not gens:
            raise ValueError(
                "Cannot convert %s to an expression without generators" % self)

        return Mul(*[ gen**exp for gen, exp in zip(gens, self.exponents) ])

    def __eq__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            return False

        return self.exponents == exponents

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError

        return self.rebuild(monomial_mul(self.exponents, exponents))

    def __truediv__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError

        result = monomial_div(self.exponents, exponents)

        if result is not None:
            return self.rebuild(result)
        else:
            raise ExactQuotientFailed(self, Monomial(other))

    __floordiv__ = __truediv__

    def __pow__(self, other):
        n = int(other)
        if n < 0:
            raise ValueError("a non-negative integer expected, got %s" % other)
        return self.rebuild(monomial_pow(self.exponents, n))

    def gcd(self, other):
        """Greatest common divisor of monomials. """
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError(
                "an instance of Monomial class expected, got %s" % other)

        return self.rebuild(monomial_gcd(self.exponents, exponents))

    def lcm(self, other):
        """Least common multiple of monomials. """
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError(
                "an instance of Monomial class expected, got %s" % other)

        return self.rebuild(monomial_lcm(self.exponents, exponents))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\utils.py ===
from __future__ import annotations

import collections.abc as cabc
import os
import re
import sys
import typing as t
from functools import update_wrapper
from types import ModuleType
from types import TracebackType

from ._compat import _default_text_stderr
from ._compat import _default_text_stdout
from ._compat import _find_binary_writer
from ._compat import auto_wrap_for_ansi
from ._compat import binary_streams
from ._compat import open_stream
from ._compat import should_strip_ansi
from ._compat import strip_ansi
from ._compat import text_streams
from ._compat import WIN
from .globals import resolve_color_default

if t.TYPE_CHECKING:
    import typing_extensions as te

    P = te.ParamSpec("P")

R = t.TypeVar("R")


def _posixify(name: str) -> str:
    return "-".join(name.split()).lower()


def safecall(func: t.Callable[P, R]) -> t.Callable[P, R | None]:
    """Wraps a function so that it swallows exceptions."""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
        return None

    return update_wrapper(wrapper, func)


def make_str(value: t.Any) -> str:
    """Converts a value into a valid string."""
    if isinstance(value, bytes):
        try:
            return value.decode(sys.getfilesystemencoding())
        except UnicodeError:
            return value.decode("utf-8", "replace")
    return str(value)


def make_default_short_help(help: str, max_length: int = 45) -> str:
    """Returns a condensed version of help string."""
    # Consider only the first paragraph.
    paragraph_end = help.find("\n\n")

    if paragraph_end != -1:
        help = help[:paragraph_end]

    # Collapse newlines, tabs, and spaces.
    words = help.split()

    if not words:
        return ""

    # The first paragraph started with a "no rewrap" marker, ignore it.
    if words[0] == "\b":
        words = words[1:]

    total_length = 0
    last_index = len(words) - 1

    for i, word in enumerate(words):
        total_length += len(word) + (i > 0)

        if total_length > max_length:  # too long, truncate
            break

        if word[-1] == ".":  # sentence end, truncate without "..."
            return " ".join(words[: i + 1])

        if total_length == max_length and i != last_index:
            break  # not at sentence end, truncate with "..."
    else:
        return " ".join(words)  # no truncation needed

    # Account for the length of the suffix.
    total_length += len("...")

    # remove words until the length is short enough
    while i > 0:
        total_length -= len(words[i]) + (i > 0)

        if total_length <= max_length:
            break

        i -= 1

    return " ".join(words[:i]) + "..."


class LazyFile:
    """A lazy file works like a regular file but it does not fully open
    the file but it does perform some basic checks early to see if the
    filename parameter does make sense.  This is useful for safely opening
    files for writing.
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        mode: str = "r",
        encoding: str | None = None,
        errors: str | None = "strict",
        atomic: bool = False,
    ):
        self.name: str = os.fspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.errors = errors
        self.atomic = atomic
        self._f: t.IO[t.Any] | None
        self.should_close: bool

        if self.name == "-":
            self._f, self.should_close = open_stream(filename, mode, encoding, errors)
        else:
            if "r" in mode:
                # Open and close the file in case we're opening it for
                # reading so that we can catch at least some errors in
                # some cases early.
                open(filename, mode).close()
            self._f = None
            self.should_close = True

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self.open(), name)

    def __repr__(self) -> str:
        if self._f is not None:
            return repr(self._f)
        return f"<unopened file '{format_filename(self.name)}' {self.mode}>"

    def open(self) -> t.IO[t.Any]:
        """Opens the file if it's not yet open.  This call might fail with
        a :exc:`FileError`.  Not handling this error will produce an error
        that Click shows.
        """
        if self._f is not None:
            return self._f
        try:
            rv, self.should_close = open_stream(
                self.name, self.mode, self.encoding, self.errors, atomic=self.atomic
            )
        except OSError as e:
            from .exceptions import FileError

            raise FileError(self.name, hint=e.strerror) from e
        self._f = rv
        return rv

    def close(self) -> None:
        """Closes the underlying file, no matter what."""
        if self._f is not None:
            self._f.close()

    def close_intelligently(self) -> None:
        """This function only closes the file if it was opened by the lazy
        file wrapper.  For instance this will never close stdin.
        """
        if self.should_close:
            self.close()

    def __enter__(self) -> LazyFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close_intelligently()

    def __iter__(self) -> cabc.Iterator[t.AnyStr]:
        self.open()
        return iter(self._f)  # type: ignore


class KeepOpenFile:
    def __init__(self, file: t.IO[t.Any]) -> None:
        self._file: t.IO[t.Any] = file

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._file, name)

    def __enter__(self) -> KeepOpenFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass

    def __repr__(self) -> str:
        return repr(self._file)

    def __iter__(self) -> cabc.Iterator[t.AnyStr]:
        return iter(self._file)


def echo(
    message: t.Any | None = None,
    file: t.IO[t.Any] | None = None,
    nl: bool = True,
    err: bool = False,
    color: bool | None = None,
) -> None:
    """Print a message and newline to stdout or a file. This should be
    used instead of :func:`print` because it provides better support
    for different data, files, and environments.

    Compared to :func:`print`, this does the following:

    -   Ensures that the output encoding is not misconfigured on Linux.
    -   Supports Unicode in the Windows console.
    -   Supports writing to binary outputs, and supports writing bytes
        to text outputs.
    -   Supports colors and styles on Windows.
    -   Removes ANSI color and style codes if the output does not look
        like an interactive terminal.
    -   Always flushes the output.

    :param message: The string or bytes to output. Other objects are
        converted to strings.
    :param file: The file to write to. Defaults to ``stdout``.
    :param err: Write to ``stderr`` instead of ``stdout``.
    :param nl: Print a newline after the message. Enabled by default.
    :param color: Force showing or hiding colors and other styles. By
        default Click will remove color if the output does not look like
        an interactive terminal.

    .. versionchanged:: 6.0
        Support Unicode output on the Windows console. Click does not
        modify ``sys.stdout``, so ``sys.stdout.write()`` and ``print()``
        will still not support Unicode.

    .. versionchanged:: 4.0
        Added the ``color`` parameter.

    .. versionadded:: 3.0
        Added the ``err`` parameter.

    .. versionchanged:: 2.0
        Support colors on Windows if colorama is installed.
    """
    if file is None:
        if err:
            file = _default_text_stderr()
        else:
            file = _default_text_stdout()

        # There are no standard streams attached to write to. For example,
        # pythonw on Windows.
        if file is None:
            return

    # Convert non bytes/text into the native string type.
    if message is not None and not isinstance(message, (str, bytes, bytearray)):
        out: str | bytes | None = str(message)
    else:
        out = message

    if nl:
        out = out or ""
        if isinstance(out, str):
            out += "\n"
        else:
            out += b"\n"

    if not out:
        file.flush()
        return

    # If there is a message and the value looks like bytes, we manually
    # need to find the binary stream and write the message in there.
    # This is done separately so that most stream types will work as you
    # would expect. Eg: you can write to StringIO for other cases.
    if isinstance(out, (bytes, bytearray)):
        binary_file = _find_binary_writer(file)

        if binary_file is not None:
            file.flush()
            binary_file.write(out)
            binary_file.flush()
            return

    # ANSI style code support. For no message or bytes, nothing happens.
    # When outputting to a file instead of a terminal, strip codes.
    else:
        color = resolve_color_default(color)

        if should_strip_ansi(file, color):
            out = strip_ansi(out)
        elif WIN:
            if auto_wrap_for_ansi is not None:
                file = auto_wrap_for_ansi(file, color)  # type: ignore
            elif not color:
                out = strip_ansi(out)

    file.write(out)  # type: ignore
    file.flush()


def get_binary_stream(name: t.Literal["stdin", "stdout", "stderr"]) -> t.BinaryIO:
    """Returns a system stream for byte processing.

    :param name: the name of the stream to open.  Valid names are ``'stdin'``,
                 ``'stdout'`` and ``'stderr'``
    """
    opener = binary_streams.get(name)
    if opener is None:
        raise TypeError(f"Unknown standard stream '{name}'")
    return opener()


def get_text_stream(
    name: t.Literal["stdin", "stdout", "stderr"],
    encoding: str | None = None,
    errors: str | None = "strict",
) -> t.TextIO:
    """Returns a system stream for text processing.  This usually returns
    a wrapped stream around a binary stream returned from
    :func:`get_binary_stream` but it also can take shortcuts for already
    correctly configured streams.

    :param name: the name of the stream to open.  Valid names are ``'stdin'``,
                 ``'stdout'`` and ``'stderr'``
    :param encoding: overrides the detected default encoding.
    :param errors: overrides the default error mode.
    """
    opener = text_streams.get(name)
    if opener is None:
        raise TypeError(f"Unknown standard stream '{name}'")
    return opener(encoding, errors)


def open_file(
    filename: str | os.PathLike[str],
    mode: str = "r",
    encoding: str | None = None,
    errors: str | None = "strict",
    lazy: bool = False,
    atomic: bool = False,
) -> t.IO[t.Any]:
    """Open a file, with extra behavior to handle ``'-'`` to indicate
    a standard stream, lazy open on write, and atomic write. Similar to
    the behavior of the :class:`~click.File` param type.

    If ``'-'`` is given to open ``stdout`` or ``stdin``, the stream is
    wrapped so that using it in a context manager will not close it.
    This makes it possible to use the function without accidentally
    closing a standard stream:

    .. code-block:: python

        with open_file(filename) as f:
            ...

    :param filename: The name or Path of the file to open, or ``'-'`` for
        ``stdin``/``stdout``.
    :param mode: The mode in which to open the file.
    :param encoding: The encoding to decode or encode a file opened in
        text mode.
    :param errors: The error handling mode.
    :param lazy: Wait to open the file until it is accessed. For read
        mode, the file is temporarily opened to raise access errors
        early, then closed until it is read again.
    :param atomic: Write to a temporary file and replace the given file
        on close.

    .. versionadded:: 3.0
    """
    if lazy:
        return t.cast(
            "t.IO[t.Any]", LazyFile(filename, mode, encoding, errors, atomic=atomic)
        )

    f, should_close = open_stream(filename, mode, encoding, errors, atomic=atomic)

    if not should_close:
        f = t.cast("t.IO[t.Any]", KeepOpenFile(f))

    return f


def format_filename(
    filename: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    shorten: bool = False,
) -> str:
    """Format a filename as a string for display. Ensures the filename can be
    displayed by replacing any invalid bytes or surrogate escapes in the name
    with the replacement character ``�``.

    Invalid bytes or surrogate escapes will raise an error when written to a
    stream with ``errors="strict"``. This will typically happen with ``stdout``
    when the locale is something like ``en_GB.UTF-8``.

    Many scenarios *are* safe to write surrogates though, due to PEP 538 and
    PEP 540, including:

    -   Writing to ``stderr``, which uses ``errors="backslashreplace"``.
    -   The system has ``LANG=C.UTF-8``, ``C``, or ``POSIX``. Python opens
        stdout and stderr with ``errors="surrogateescape"``.
    -   None of ``LANG/LC_*`` are set. Python assumes ``LANG=C.UTF-8``.
    -   Python is started in UTF-8 mode  with  ``PYTHONUTF8=1`` or ``-X utf8``.
        Python opens stdout and stderr with ``errors="surrogateescape"``.

    :param filename: formats a filename for UI display.  This will also convert
                     the filename into unicode without failing.
    :param shorten: this optionally shortens the filename to strip of the
                    path that leads up to it.
    """
    if shorten:
        filename = os.path.basename(filename)
    else:
        filename = os.fspath(filename)

    if isinstance(filename, bytes):
        filename = filename.decode(sys.getfilesystemencoding(), "replace")
    else:
        filename = filename.encode("utf-8", "surrogateescape").decode(
            "utf-8", "replace"
        )

    return filename


def get_app_dir(app_name: str, roaming: bool = True, force_posix: bool = False) -> str:
    r"""Returns the config folder for the application.  The default behavior
    is to return whatever is most appropriate for the operating system.

    To give you an idea, for an app called ``"Foo Bar"``, something like
    the following folders could be returned:

    Mac OS X:
      ``~/Library/Application Support/Foo Bar``
    Mac OS X (POSIX):
      ``~/.foo-bar``
    Unix:
      ``~/.config/foo-bar``
    Unix (POSIX):
      ``~/.foo-bar``
    Windows (roaming):
      ``C:\Users\<user>\AppData\Roaming\Foo Bar``
    Windows (not roaming):
      ``C:\Users\<user>\AppData\Local\Foo Bar``

    .. versionadded:: 2.0

    :param app_name: the application name.  This should be properly capitalized
                     and can contain whitespace.
    :param roaming: controls if the folder should be roaming or not on Windows.
                    Has no effect otherwise.
    :param force_posix: if this is set to `True` then on any POSIX system the
                        folder will be stored in the home folder with a leading
                        dot instead of the XDG config home or darwin's
                        application support folder.
    """
    if WIN:
        key = "APPDATA" if roaming else "LOCALAPPDATA"
        folder = os.environ.get(key)
        if folder is None:
            folder = os.path.expanduser("~")
        return os.path.join(folder, app_name)
    if force_posix:
        return os.path.join(os.path.expanduser(f"~/.{_posixify(app_name)}"))
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~/Library/Application Support"), app_name
        )
    return os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
        _posixify(app_name),
    )


class PacifyFlushWrapper:
    """This wrapper is used to catch and suppress BrokenPipeErrors resulting
    from ``.flush()`` being called on broken pipe during the shutdown/final-GC
    of the Python interpreter. Notably ``.flush()`` is always called on
    ``sys.stdout`` and ``sys.stderr``. So as to have minimal impact on any
    other cleanup code, and the case where the underlying file is not a broken
    pipe, all calls and attributes are proxied.
    """

    def __init__(self, wrapped: t.IO[t.Any]) -> None:
        self.wrapped = wrapped

    def flush(self) -> None:
        try:
            self.wrapped.flush()
        except OSError as e:
            import errno

            if e.errno != errno.EPIPE:
                raise

    def __getattr__(self, attr: str) -> t.Any:
        return getattr(self.wrapped, attr)


def _detect_program_name(
    path: str | None = None, _main: ModuleType | None = None
) -> str:
    """Determine the command used to run the program, for use in help
    text. If a file or entry point was executed, the file name is
    returned. If ``python -m`` was used to execute a module or package,
    ``python -m name`` is returned.

    This doesn't try to be too precise, the goal is to give a concise
    name for help text. Files are only shown as their name without the
    path. ``python`` is only shown for modules, and the full path to
    ``sys.executable`` is not shown.

    :param path: The Python file being executed. Python puts this in
        ``sys.argv[0]``, which is used by default.
    :param _main: The ``__main__`` module. This should only be passed
        during internal testing.

    .. versionadded:: 8.0
        Based on command args detection in the Werkzeug reloader.

    :meta private:
    """
    if _main is None:
        _main = sys.modules["__main__"]

    if not path:
        path = sys.argv[0]

    # The value of __package__ indicates how Python was called. It may
    # not exist if a setuptools script is installed as an egg. It may be
    # set incorrectly for entry points created with pip on Windows.
    # It is set to "" inside a Shiv or PEX zipapp.
    if getattr(_main, "__package__", None) in {None, ""} or (
        os.name == "nt"
        and _main.__package__ == ""
        and not os.path.exists(path)
        and os.path.exists(f"{path}.exe")
    ):
        # Executed a file, like "python app.py".
        return os.path.basename(path)

    # Executed a module, like "python -m example".
    # Rewritten by Python from "-m script" to "/path/to/script.py".
    # Need to look at main module to determine how it was executed.
    py_module = t.cast(str, _main.__package__)
    name = os.path.splitext(os.path.basename(path))[0]

    # A submodule like "example.cli".
    if name != "__main__":
        py_module = f"{py_module}.{name}"

    return f"python -m {py_module.lstrip('.')}"


def _expand_args(
    args: cabc.Iterable[str],
    *,
    user: bool = True,
    env: bool = True,
    glob_recursive: bool = True,
) -> list[str]:
    """Simulate Unix shell expansion with Python functions.

    See :func:`glob.glob`, :func:`os.path.expanduser`, and
    :func:`os.path.expandvars`.

    This is intended for use on Windows, where the shell does not do any
    expansion. It may not exactly match what a Unix shell would do.

    :param args: List of command line arguments to expand.
    :param user: Expand user home directory.
    :param env: Expand environment variables.
    :param glob_recursive: ``**`` matches directories recursively.

    .. versionchanged:: 8.1
        Invalid glob patterns are treated as empty expansions rather
        than raising an error.

    .. versionadded:: 8.0

    :meta private:
    """
    from glob import glob

    out = []

    for arg in args:
        if user:
            arg = os.path.expanduser(arg)

        if env:
            arg = os.path.expandvars(arg)

        try:
            matches = glob(arg, recursive=glob_recursive)
        except re.error:
            matches = []

        if not matches:
            out.append(arg)
        else:
            out.extend(matches)

    return out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_channel.py ===
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Union

import pytest

import trio
from trio import EndOfChannel, as_safe_channel, open_memory_channel

from ..testing import Matcher, RaisesGroup, assert_checkpoints, wait_all_tasks_blocked

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


async def test_channel() -> None:
    with pytest.raises(TypeError):
        open_memory_channel(1.0)
    with pytest.raises(ValueError, match=r"^max_buffer_size must be >= 0$"):
        open_memory_channel(-1)

    s, r = open_memory_channel[Union[int, str, None]](2)
    repr(s)  # smoke test
    repr(r)  # smoke test

    s.send_nowait(1)
    with assert_checkpoints():
        await s.send(2)
    with pytest.raises(trio.WouldBlock):
        s.send_nowait(None)

    with assert_checkpoints():
        assert await r.receive() == 1
    assert r.receive_nowait() == 2
    with pytest.raises(trio.WouldBlock):
        r.receive_nowait()

    s.send_nowait("last")
    await s.aclose()
    with pytest.raises(trio.ClosedResourceError):
        await s.send("too late")
    with pytest.raises(trio.ClosedResourceError):
        s.send_nowait("too late")
    with pytest.raises(trio.ClosedResourceError):
        s.clone()
    await s.aclose()

    assert r.receive_nowait() == "last"
    with pytest.raises(EndOfChannel):
        await r.receive()
    await r.aclose()
    with pytest.raises(trio.ClosedResourceError):
        await r.receive()
    with pytest.raises(trio.ClosedResourceError):
        r.receive_nowait()
    await r.aclose()


async def test_553(autojump_clock: trio.abc.Clock) -> None:
    s, r = open_memory_channel[str](1)
    with trio.move_on_after(10) as timeout_scope:
        await r.receive()
    assert timeout_scope.cancelled_caught
    await s.send("Test for PR #553")


async def test_channel_multiple_producers() -> None:
    async def producer(send_channel: trio.MemorySendChannel[int], i: int) -> None:
        # We close our handle when we're done with it
        async with send_channel:
            for j in range(3 * i, 3 * (i + 1)):
                await send_channel.send(j)

    send_channel, receive_channel = open_memory_channel[int](0)
    async with trio.open_nursery() as nursery:
        # We hand out clones to all the new producers, and then close the
        # original.
        async with send_channel:
            for i in range(10):
                nursery.start_soon(producer, send_channel.clone(), i)

        got = [value async for value in receive_channel]

        got.sort()
        assert got == list(range(30))


async def test_channel_multiple_consumers() -> None:
    successful_receivers = set()
    received = []

    async def consumer(receive_channel: trio.MemoryReceiveChannel[int], i: int) -> None:
        async for value in receive_channel:
            successful_receivers.add(i)
            received.append(value)

    async with trio.open_nursery() as nursery:
        send_channel, receive_channel = trio.open_memory_channel[int](1)
        async with send_channel:
            for i in range(5):
                nursery.start_soon(consumer, receive_channel, i)
            await wait_all_tasks_blocked()
            for i in range(10):
                await send_channel.send(i)

    assert successful_receivers == set(range(5))
    assert len(received) == 10
    assert set(received) == set(range(10))


async def test_close_basics() -> None:
    async def send_block(
        s: trio.MemorySendChannel[None],
        expect: type[BaseException],
    ) -> None:
        with pytest.raises(expect):
            await s.send(None)

    # closing send -> other send gets ClosedResourceError
    s, r = open_memory_channel[None](0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(send_block, s, trio.ClosedResourceError)
        await wait_all_tasks_blocked()
        await s.aclose()

    # and it's persistent
    with pytest.raises(trio.ClosedResourceError):
        s.send_nowait(None)
    with pytest.raises(trio.ClosedResourceError):
        await s.send(None)

    # and receive gets EndOfChannel
    with pytest.raises(EndOfChannel):
        r.receive_nowait()
    with pytest.raises(EndOfChannel):
        await r.receive()

    # closing receive -> send gets BrokenResourceError
    s, r = open_memory_channel[None](0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(send_block, s, trio.BrokenResourceError)
        await wait_all_tasks_blocked()
        await r.aclose()

    # and it's persistent
    with pytest.raises(trio.BrokenResourceError):
        s.send_nowait(None)
    with pytest.raises(trio.BrokenResourceError):
        await s.send(None)

    # closing receive -> other receive gets ClosedResourceError
    async def receive_block(r: trio.MemoryReceiveChannel[int]) -> None:
        with pytest.raises(trio.ClosedResourceError):
            await r.receive()

    _s2, r2 = open_memory_channel[int](0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(receive_block, r2)
        await wait_all_tasks_blocked()
        await r2.aclose()

    # and it's persistent
    with pytest.raises(trio.ClosedResourceError):
        r2.receive_nowait()
    with pytest.raises(trio.ClosedResourceError):
        await r2.receive()


async def test_close_sync() -> None:
    async def send_block(
        s: trio.MemorySendChannel[None],
        expect: type[BaseException],
    ) -> None:
        with pytest.raises(expect):
            await s.send(None)

    # closing send -> other send gets ClosedResourceError
    s, r = open_memory_channel[None](0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(send_block, s, trio.ClosedResourceError)
        await wait_all_tasks_blocked()
        s.close()

    # and it's persistent
    with pytest.raises(trio.ClosedResourceError):
        s.send_nowait(None)
    with pytest.raises(trio.ClosedResourceError):
        await s.send(None)

    # and receive gets EndOfChannel
    with pytest.raises(EndOfChannel):
        r.receive_nowait()
    with pytest.raises(EndOfChannel):
        await r.receive()

    # closing receive -> send gets BrokenResourceError
    s, r = open_memory_channel[None](0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(send_block, s, trio.BrokenResourceError)
        await wait_all_tasks_blocked()
        r.close()

    # and it's persistent
    with pytest.raises(trio.BrokenResourceError):
        s.send_nowait(None)
    with pytest.raises(trio.BrokenResourceError):
        await s.send(None)

    # closing receive -> other receive gets ClosedResourceError
    async def receive_block(r: trio.MemoryReceiveChannel[None]) -> None:
        with pytest.raises(trio.ClosedResourceError):
            await r.receive()

    s, r = open_memory_channel[None](0)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(receive_block, r)
        await wait_all_tasks_blocked()
        r.close()

    # and it's persistent
    with pytest.raises(trio.ClosedResourceError):
        r.receive_nowait()
    with pytest.raises(trio.ClosedResourceError):
        await r.receive()


async def test_receive_channel_clone_and_close() -> None:
    s, r = open_memory_channel[None](10)

    r2 = r.clone()
    r3 = r.clone()

    s.send_nowait(None)
    await r.aclose()
    with r2:
        pass

    with pytest.raises(trio.ClosedResourceError):
        r.clone()

    with pytest.raises(trio.ClosedResourceError):
        r2.clone()

    # Can still send, r3 is still open
    s.send_nowait(None)

    await r3.aclose()

    # But now the receiver is really closed
    with pytest.raises(trio.BrokenResourceError):
        s.send_nowait(None)


async def test_close_multiple_send_handles() -> None:
    # With multiple send handles, closing one handle only wakes senders on
    # that handle, but others can continue just fine
    s1, r = open_memory_channel[str](0)
    s2 = s1.clone()

    async def send_will_close() -> None:
        with pytest.raises(trio.ClosedResourceError):
            await s1.send("nope")

    async def send_will_succeed() -> None:
        await s2.send("ok")

    async with trio.open_nursery() as nursery:
        nursery.start_soon(send_will_close)
        nursery.start_soon(send_will_succeed)
        await wait_all_tasks_blocked()
        await s1.aclose()
        assert await r.receive() == "ok"


async def test_close_multiple_receive_handles() -> None:
    # With multiple receive handles, closing one handle only wakes receivers on
    # that handle, but others can continue just fine
    s, r1 = open_memory_channel[str](0)
    r2 = r1.clone()

    async def receive_will_close() -> None:
        with pytest.raises(trio.ClosedResourceError):
            await r1.receive()

    async def receive_will_succeed() -> None:
        assert await r2.receive() == "ok"

    async with trio.open_nursery() as nursery:
        nursery.start_soon(receive_will_close)
        nursery.start_soon(receive_will_succeed)
        await wait_all_tasks_blocked()
        await r1.aclose()
        await s.send("ok")


async def test_inf_capacity() -> None:
    send, receive = open_memory_channel[int](float("inf"))

    # It's accepted, and we can send all day without blocking
    with send:
        for i in range(10):
            send.send_nowait(i)

    got = [i async for i in receive]
    assert got == list(range(10))


async def test_statistics() -> None:
    s, r = open_memory_channel[None](2)

    assert s.statistics() == r.statistics()
    stats = s.statistics()
    assert stats.current_buffer_used == 0
    assert stats.max_buffer_size == 2
    assert stats.open_send_channels == 1
    assert stats.open_receive_channels == 1
    assert stats.tasks_waiting_send == 0
    assert stats.tasks_waiting_receive == 0

    s.send_nowait(None)
    assert s.statistics().current_buffer_used == 1

    s2 = s.clone()
    assert s.statistics().open_send_channels == 2
    await s.aclose()
    assert s2.statistics().open_send_channels == 1

    r2 = r.clone()
    assert s2.statistics().open_receive_channels == 2
    await r2.aclose()
    assert s2.statistics().open_receive_channels == 1

    async with trio.open_nursery() as nursery:
        s2.send_nowait(None)  # fill up the buffer
        assert s.statistics().current_buffer_used == 2
        nursery.start_soon(s2.send, None)
        nursery.start_soon(s2.send, None)
        await wait_all_tasks_blocked()
        assert s.statistics().tasks_waiting_send == 2
        nursery.cancel_scope.cancel()
    assert s.statistics().tasks_waiting_send == 0

    # empty out the buffer again
    try:
        while True:
            r.receive_nowait()
    except trio.WouldBlock:
        pass

    async with trio.open_nursery() as nursery:
        nursery.start_soon(r.receive)
        await wait_all_tasks_blocked()
        assert s.statistics().tasks_waiting_receive == 1
        nursery.cancel_scope.cancel()
    assert s.statistics().tasks_waiting_receive == 0


async def test_channel_fairness() -> None:
    # We can remove an item we just sent, and send an item back in after, if
    # no-one else is waiting.
    s, r = open_memory_channel[Union[int, None]](1)
    s.send_nowait(1)
    assert r.receive_nowait() == 1
    s.send_nowait(2)
    assert r.receive_nowait() == 2

    # But if someone else is waiting to receive, then they "own" the item we
    # send, so we can't receive it (even though we run first):

    result: int | None = None

    async def do_receive(r: trio.MemoryReceiveChannel[int | None]) -> None:
        nonlocal result
        result = await r.receive()

    async with trio.open_nursery() as nursery:
        nursery.start_soon(do_receive, r)
        await wait_all_tasks_blocked()
        s.send_nowait(2)
        with pytest.raises(trio.WouldBlock):
            r.receive_nowait()
    assert result == 2

    # And the analogous situation for send: if we free up a space, we can't
    # immediately send something in it if someone is already waiting to do
    # that
    s, r = open_memory_channel[Union[int, None]](1)
    s.send_nowait(1)
    with pytest.raises(trio.WouldBlock):
        s.send_nowait(None)
    async with trio.open_nursery() as nursery:
        nursery.start_soon(s.send, 2)
        await wait_all_tasks_blocked()
        assert r.receive_nowait() == 1
        with pytest.raises(trio.WouldBlock):
            s.send_nowait(3)
        assert (await r.receive()) == 2


async def test_unbuffered() -> None:
    s, r = open_memory_channel[int](0)
    with pytest.raises(trio.WouldBlock):
        r.receive_nowait()
    with pytest.raises(trio.WouldBlock):
        s.send_nowait(1)

    async def do_send(s: trio.MemorySendChannel[int], v: int) -> None:
        with assert_checkpoints():
            await s.send(v)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(do_send, s, 1)
        with assert_checkpoints():
            assert await r.receive() == 1
    with pytest.raises(trio.WouldBlock):
        r.receive_nowait()


async def test_as_safe_channel_exhaust() -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        yield 1

    async with agen() as recv_chan:
        async for x in recv_chan:
            assert x == 1


async def test_as_safe_channel_broken_resource() -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        yield 1
        yield 2

    async with agen() as recv_chan:
        assert await recv_chan.__anext__() == 1

        # close the receiving channel
        await recv_chan.aclose()

        # trying to get the next element errors
        with pytest.raises(trio.ClosedResourceError):
            await recv_chan.__anext__()

        # but we don't get an error on exit of the cm


async def test_as_safe_channel_cancelled() -> None:
    with trio.CancelScope() as cs:

        @as_safe_channel
        async def agen() -> AsyncGenerator[None]:  # pragma: no cover
            raise AssertionError(
                "cancel before consumption means generator should not be iterated"
            )
            yield  # indicate that we're an iterator

        async with agen():
            cs.cancel()


async def test_as_safe_channel_no_race() -> None:
    # this previously led to a race condition due to
    # https://github.com/python-trio/trio/issues/1559
    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        yield 1
        raise ValueError("oae")

    with pytest.raises(ValueError, match=r"^oae$"):
        async with agen() as recv_chan:
            async for x in recv_chan:
                assert x == 1


async def test_as_safe_channel_buffer_size_too_small(
    autojump_clock: trio.testing.MockClock,
) -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        yield 1
        raise AssertionError(
            "buffer size 0 means we shouldn't be asked for another value"
        )  # pragma: no cover

    with trio.move_on_after(5):
        async with agen() as recv_chan:
            async for x in recv_chan:  # pragma: no branch
                assert x == 1
                await trio.sleep_forever()


async def test_as_safe_channel_no_interleave() -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        yield 1
        raise AssertionError  # pragma: no cover

    async with agen() as recv_chan:
        assert await recv_chan.__anext__() == 1
        await trio.lowlevel.checkpoint()


async def test_as_safe_channel_genexit_finally() -> None:
    @as_safe_channel
    async def agen(events: list[str]) -> AsyncGenerator[int]:
        try:
            yield 1
        except BaseException as e:
            events.append(repr(e))
            raise
        finally:
            events.append("finally")
            raise ValueError("agen")

    events: list[str] = []
    with RaisesGroup(
        RaisesGroup(
            Matcher(ValueError, match="^agen$"),
            Matcher(TypeError, match="^iterator$"),
        ),
        match=r"^Encountered exception during cleanup of generator object, as well as exception in the contextmanager body - unable to unwrap.$",
    ):
        async with agen(events) as recv_chan:
            async for i in recv_chan:  # pragma: no branch
                assert i == 1
                raise TypeError("iterator")

    assert events == ["GeneratorExit()", "finally"]


async def test_as_safe_channel_nested_loop() -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        for i in range(2):
            yield i

    ii = 0
    async with agen() as recv_chan1:
        async for i in recv_chan1:
            async with agen() as recv_chan:
                jj = 0
                async for j in recv_chan:
                    assert (i, j) == (ii, jj)
                    jj += 1
            ii += 1


async def test_as_safe_channel_doesnt_leak_cancellation() -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[None]:
        with trio.CancelScope() as cscope:
            cscope.cancel()
            yield

    with pytest.raises(AssertionError):
        async with agen() as recv_chan:
            async for _ in recv_chan:
                pass
        raise AssertionError("should be reachable")


async def test_as_safe_channel_dont_unwrap_user_exceptiongroup() -> None:
    @as_safe_channel
    async def agen() -> AsyncGenerator[None]:
        raise NotImplementedError("not entered")
        yield  # pragma: no cover

    with RaisesGroup(Matcher(ValueError, match="bar"), match="foo"):
        async with agen() as _:
            raise ExceptionGroup("foo", [ValueError("bar")])


async def test_as_safe_channel_multiple_receiver() -> None:
    event = trio.Event()

    @as_safe_channel
    async def agen() -> AsyncGenerator[int]:
        await event.wait()
        yield 0
        yield 1

    async def handle_value(
        recv_chan: trio.abc.ReceiveChannel[int],
        value: int,
        task_status: trio.TaskStatus,
    ) -> None:
        task_status.started()
        assert await recv_chan.receive() == value

    async with agen() as recv_chan:
        async with trio.open_nursery() as nursery:
            await nursery.start(handle_value, recv_chan, 0)
            await nursery.start(handle_value, recv_chan, 1)
            event.set()


async def test_as_safe_channel_multi_cancel() -> None:
    @as_safe_channel
    async def agen(events: list[str]) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            # this will give a warning of ASYNC120, although it's not technically a
            # problem of swallowing existing exceptions
            try:
                await trio.lowlevel.checkpoint()
            except trio.Cancelled:
                events.append("agen cancel")
                raise

    events: list[str] = []
    with trio.CancelScope() as cs:
        with pytest.raises(trio.Cancelled):
            async with agen(events) as recv_chan:
                async for _ in recv_chan:  # pragma: no branch
                    cs.cancel()
                    try:
                        await trio.lowlevel.checkpoint()
                    except trio.Cancelled:
                        events.append("body cancel")
                        raise
    assert events == ["body cancel", "agen cancel"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\_next_gen.py ===
# SPDX-License-Identifier: MIT

"""
These are keyword-only APIs that call `attr.s` and `attr.ib` with different
default values.
"""

from functools import partial

from . import setters
from ._funcs import asdict as _asdict
from ._funcs import astuple as _astuple
from ._make import (
    _DEFAULT_ON_SETATTR,
    NOTHING,
    _frozen_setattrs,
    attrib,
    attrs,
)
from .exceptions import UnannotatedAttributeError


def define(
    maybe_cls=None,
    *,
    these=None,
    repr=None,
    unsafe_hash=None,
    hash=None,
    init=None,
    slots=True,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=None,
    kw_only=False,
    cache_hash=False,
    auto_exc=True,
    eq=None,
    order=False,
    auto_detect=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
):
    r"""
    A class decorator that adds :term:`dunder methods` according to
    :term:`fields <field>` specified using :doc:`type annotations <types>`,
    `field()` calls, or the *these* argument.

    Since *attrs* patches or replaces an existing class, you cannot use
    `object.__init_subclass__` with *attrs* classes, because it runs too early.
    As a replacement, you can define ``__attrs_init_subclass__`` on your class.
    It will be called by *attrs* classes that subclass it after they're
    created. See also :ref:`init-subclass`.

    Args:
        slots (bool):
            Create a :term:`slotted class <slotted classes>` that's more
            memory-efficient. Slotted classes are generally superior to the
            default dict classes, but have some gotchas you should know about,
            so we encourage you to read the :term:`glossary entry <slotted
            classes>`.

        auto_detect (bool):
            Instead of setting the *init*, *repr*, *eq*, and *hash* arguments
            explicitly, assume they are set to True **unless any** of the
            involved methods for one of the arguments is implemented in the
            *current* class (meaning, it is *not* inherited from some base
            class).

            So, for example by implementing ``__eq__`` on a class yourself,
            *attrs* will deduce ``eq=False`` and will create *neither*
            ``__eq__`` *nor* ``__ne__`` (but Python classes come with a
            sensible ``__ne__`` by default, so it *should* be enough to only
            implement ``__eq__`` in most cases).

            Passing True or False` to *init*, *repr*, *eq*, or *hash*
            overrides whatever *auto_detect* would determine.

        auto_exc (bool):
            If the class subclasses `BaseException` (which implicitly includes
            any subclass of any exception), the following happens to behave
            like a well-behaved Python exception class:

            - the values for *eq*, *order*, and *hash* are ignored and the
              instances compare and hash by the instance's ids [#]_ ,
            - all attributes that are either passed into ``__init__`` or have a
              default value are additionally available as a tuple in the
              ``args`` attribute,
            - the value of *str* is ignored leaving ``__str__`` to base
              classes.

            .. [#]
               Note that *attrs* will *not* remove existing implementations of
               ``__hash__`` or the equality methods. It just won't add own
               ones.

        on_setattr (~typing.Callable | list[~typing.Callable] | None | ~typing.Literal[attrs.setters.NO_OP]):
            A callable that is run whenever the user attempts to set an
            attribute (either by assignment like ``i.x = 42`` or by using
            `setattr` like ``setattr(i, "x", 42)``). It receives the same
            arguments as validators: the instance, the attribute that is being
            modified, and the new value.

            If no exception is raised, the attribute is set to the return value
            of the callable.

            If a list of callables is passed, they're automatically wrapped in
            an `attrs.setters.pipe`.

            If left None, the default behavior is to run converters and
            validators whenever an attribute is set.

        init (bool):
            Create a ``__init__`` method that initializes the *attrs*
            attributes. Leading underscores are stripped for the argument name,
            unless an alias is set on the attribute.

            .. seealso::
                `init` shows advanced ways to customize the generated
                ``__init__`` method, including executing code before and after.

        repr(bool):
            Create a ``__repr__`` method with a human readable representation
            of *attrs* attributes.

        str (bool):
            Create a ``__str__`` method that is identical to ``__repr__``. This
            is usually not necessary except for `Exception`\ s.

        eq (bool | None):
            If True or None (default), add ``__eq__`` and ``__ne__`` methods
            that check two instances for equality.

            .. seealso::
                `comparison` describes how to customize the comparison behavior
                going as far comparing NumPy arrays.

        order (bool | None):
            If True, add ``__lt__``, ``__le__``, ``__gt__``, and ``__ge__``
            methods that behave like *eq* above and allow instances to be
            ordered.

            They compare the instances as if they were tuples of their *attrs*
            attributes if and only if the types of both classes are
            *identical*.

            If `None` mirror value of *eq*.

            .. seealso:: `comparison`

        unsafe_hash (bool | None):
            If None (default), the ``__hash__`` method is generated according
            how *eq* and *frozen* are set.

            1. If *both* are True, *attrs* will generate a ``__hash__`` for
               you.
            2. If *eq* is True and *frozen* is False, ``__hash__`` will be set
               to None, marking it unhashable (which it is).
            3. If *eq* is False, ``__hash__`` will be left untouched meaning
               the ``__hash__`` method of the base class will be used. If the
               base class is `object`, this means it will fall back to id-based
               hashing.

            Although not recommended, you can decide for yourself and force
            *attrs* to create one (for example, if the class is immutable even
            though you didn't freeze it programmatically) by passing True or
            not.  Both of these cases are rather special and should be used
            carefully.

            .. seealso::

                - Our documentation on `hashing`,
                - Python's documentation on `object.__hash__`,
                - and the `GitHub issue that led to the default \ behavior
                  <https://github.com/python-attrs/attrs/issues/136>`_ for more
                  details.

        hash (bool | None):
            Deprecated alias for *unsafe_hash*. *unsafe_hash* takes precedence.

        cache_hash (bool):
            Ensure that the object's hash code is computed only once and stored
            on the object.  If this is set to True, hashing must be either
            explicitly or implicitly enabled for this class.  If the hash code
            is cached, avoid any reassignments of fields involved in hash code
            computation or mutations of the objects those fields point to after
            object creation.  If such changes occur, the behavior of the
            object's hash code is undefined.

        frozen (bool):
            Make instances immutable after initialization.  If someone attempts
            to modify a frozen instance, `attrs.exceptions.FrozenInstanceError`
            is raised.

            .. note::

                1. This is achieved by installing a custom ``__setattr__``
                   method on your class, so you can't implement your own.

                2. True immutability is impossible in Python.

                3. This *does* have a minor a runtime performance `impact
                   <how-frozen>` when initializing new instances.  In other
                   words: ``__init__`` is slightly slower with ``frozen=True``.

                4. If a class is frozen, you cannot modify ``self`` in
                   ``__attrs_post_init__`` or a self-written ``__init__``. You
                   can circumvent that limitation by using
                   ``object.__setattr__(self, "attribute_name", value)``.

                5. Subclasses of a frozen class are frozen too.

        kw_only (bool):
            Make all attributes keyword-only in the generated ``__init__`` (if
            *init* is False, this parameter is ignored).

        weakref_slot (bool):
            Make instances weak-referenceable.  This has no effect unless
            *slots* is True.

        field_transformer (~typing.Callable | None):
            A function that is called with the original class object and all
            fields right before *attrs* finalizes the class.  You can use this,
            for example, to automatically add converters or validators to
            fields based on their types.

            .. seealso:: `transform-fields`

        match_args (bool):
            If True (default), set ``__match_args__`` on the class to support
            :pep:`634` (*Structural Pattern Matching*). It is a tuple of all
            non-keyword-only ``__init__`` parameter names on Python 3.10 and
            later. Ignored on older Python versions.

        collect_by_mro (bool):
            If True, *attrs* collects attributes from base classes correctly
            according to the `method resolution order
            <https://docs.python.org/3/howto/mro.html>`_. If False, *attrs*
            will mimic the (wrong) behavior of `dataclasses` and :pep:`681`.

            See also `issue #428
            <https://github.com/python-attrs/attrs/issues/428>`_.

        getstate_setstate (bool | None):
            .. note::

                This is usually only interesting for slotted classes and you
                should probably just set *auto_detect* to True.

            If True, ``__getstate__`` and ``__setstate__`` are generated and
            attached to the class. This is necessary for slotted classes to be
            pickleable. If left None, it's True by default for slotted classes
            and False for dict classes.

            If *auto_detect* is True, and *getstate_setstate* is left None, and
            **either** ``__getstate__`` or ``__setstate__`` is detected
            directly on the class (meaning: not inherited), it is set to False
            (this is usually what you want).

        auto_attribs (bool | None):
            If True, look at type annotations to determine which attributes to
            use, like `dataclasses`. If False, it will only look for explicit
            :func:`field` class attributes, like classic *attrs*.

            If left None, it will guess:

            1. If any attributes are annotated and no unannotated
               `attrs.field`\ s are found, it assumes *auto_attribs=True*.
            2. Otherwise it assumes *auto_attribs=False* and tries to collect
               `attrs.field`\ s.

            If *attrs* decides to look at type annotations, **all** fields
            **must** be annotated. If *attrs* encounters a field that is set to
            a :func:`field` / `attr.ib` but lacks a type annotation, an
            `attrs.exceptions.UnannotatedAttributeError` is raised.  Use
            ``field_name: typing.Any = field(...)`` if you don't want to set a
            type.

            .. warning::

                For features that use the attribute name to create decorators
                (for example, :ref:`validators <validators>`), you still *must*
                assign :func:`field` / `attr.ib` to them. Otherwise Python will
                either not find the name or try to use the default value to
                call, for example, ``validator`` on it.

            Attributes annotated as `typing.ClassVar`, and attributes that are
            neither annotated nor set to an `field()` are **ignored**.

        these (dict[str, object]):
            A dictionary of name to the (private) return value of `field()`
            mappings. This is useful to avoid the definition of your attributes
            within the class body because you can't (for example, if you want
            to add ``__repr__`` methods to Django models) or don't want to.

            If *these* is not `None`, *attrs* will *not* search the class body
            for attributes and will *not* remove any attributes from it.

            The order is deduced from the order of the attributes inside
            *these*.

            Arguably, this is a rather obscure feature.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.3.0 Converters are also run ``on_setattr``.
    .. versionadded:: 22.2.0
       *unsafe_hash* as an alias for *hash* (for :pep:`681` compliance).
    .. versionchanged:: 24.1.0
       Instances are not compared as tuples of attributes anymore, but using a
       big ``and`` condition. This is faster and has more correct behavior for
       uncomparable values like `math.nan`.
    .. versionadded:: 24.1.0
       If a class has an *inherited* classmethod called
       ``__attrs_init_subclass__``, it is executed after the class is created.
    .. deprecated:: 24.1.0 *hash* is deprecated in favor of *unsafe_hash*.
    .. versionadded:: 24.3.0
       Unless already present, a ``__replace__`` method is automatically
       created for `copy.replace` (Python 3.13+ only).

    .. note::

        The main differences to the classic `attr.s` are:

        - Automatically detect whether or not *auto_attribs* should be `True`
          (c.f. *auto_attribs* parameter).
        - Converters and validators run when attributes are set by default --
          if *frozen* is `False`.
        - *slots=True*

          Usually, this has only upsides and few visible effects in everyday
          programming. But it *can* lead to some surprising behaviors, so
          please make sure to read :term:`slotted classes`.

        - *auto_exc=True*
        - *auto_detect=True*
        - *order=False*
        - Some options that were only relevant on Python 2 or were kept around
          for backwards-compatibility have been removed.

    """

    def do_it(cls, auto_attribs):
        return attrs(
            maybe_cls=cls,
            these=these,
            repr=repr,
            hash=hash,
            unsafe_hash=unsafe_hash,
            init=init,
            slots=slots,
            frozen=frozen,
            weakref_slot=weakref_slot,
            str=str,
            auto_attribs=auto_attribs,
            kw_only=kw_only,
            cache_hash=cache_hash,
            auto_exc=auto_exc,
            eq=eq,
            order=order,
            auto_detect=auto_detect,
            collect_by_mro=True,
            getstate_setstate=getstate_setstate,
            on_setattr=on_setattr,
            field_transformer=field_transformer,
            match_args=match_args,
        )

    def wrap(cls):
        """
        Making this a wrapper ensures this code runs during class creation.

        We also ensure that frozen-ness of classes is inherited.
        """
        nonlocal frozen, on_setattr

        had_on_setattr = on_setattr not in (None, setters.NO_OP)

        # By default, mutable classes convert & validate on setattr.
        if frozen is False and on_setattr is None:
            on_setattr = _DEFAULT_ON_SETATTR

        # However, if we subclass a frozen class, we inherit the immutability
        # and disable on_setattr.
        for base_cls in cls.__bases__:
            if base_cls.__setattr__ is _frozen_setattrs:
                if had_on_setattr:
                    msg = "Frozen classes can't use on_setattr (frozen-ness was inherited)."
                    raise ValueError(msg)

                on_setattr = setters.NO_OP
                break

        if auto_attribs is not None:
            return do_it(cls, auto_attribs)

        try:
            return do_it(cls, True)
        except UnannotatedAttributeError:
            return do_it(cls, False)

    # maybe_cls's type depends on the usage of the decorator.  It's a class
    # if it's used as `@attrs` but `None` if used as `@attrs()`.
    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


mutable = define
frozen = partial(define, frozen=True, on_setattr=None)


def field(
    *,
    default=NOTHING,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    type=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
):
    """
    Create a new :term:`field` / :term:`attribute` on a class.

    ..  warning::

        Does **nothing** unless the class is also decorated with
        `attrs.define` (or similar)!

    Args:
        default:
            A value that is used if an *attrs*-generated ``__init__`` is used
            and no value is passed while instantiating or the attribute is
            excluded using ``init=False``.

            If the value is an instance of `attrs.Factory`, its callable will
            be used to construct a new value (useful for mutable data types
            like lists or dicts).

            If a default is not set (or set manually to `attrs.NOTHING`), a
            value *must* be supplied when instantiating; otherwise a
            `TypeError` will be raised.

            .. seealso:: `defaults`

        factory (~typing.Callable):
            Syntactic sugar for ``default=attr.Factory(factory)``.

        validator (~typing.Callable | list[~typing.Callable]):
            Callable that is called by *attrs*-generated ``__init__`` methods
            after the instance has been initialized.  They receive the
            initialized instance, the :func:`~attrs.Attribute`, and the passed
            value.

            The return value is *not* inspected so the validator has to throw
            an exception itself.

            If a `list` is passed, its items are treated as validators and must
            all pass.

            Validators can be globally disabled and re-enabled using
            `attrs.validators.get_disabled` / `attrs.validators.set_disabled`.

            The validator can also be set using decorator notation as shown
            below.

            .. seealso:: :ref:`validators`

        repr (bool | ~typing.Callable):
            Include this attribute in the generated ``__repr__`` method. If
            True, include the attribute; if False, omit it. By default, the
            built-in ``repr()`` function is used. To override how the attribute
            value is formatted, pass a ``callable`` that takes a single value
            and returns a string. Note that the resulting string is used as-is,
            which means it will be used directly *instead* of calling
            ``repr()`` (the default).

        eq (bool | ~typing.Callable):
            If True (default), include this attribute in the generated
            ``__eq__`` and ``__ne__`` methods that check two instances for
            equality. To override how the attribute value is compared, pass a
            callable that takes a single value and returns the value to be
            compared.

            .. seealso:: `comparison`

        order (bool | ~typing.Callable):
            If True (default), include this attributes in the generated
            ``__lt__``, ``__le__``, ``__gt__`` and ``__ge__`` methods. To
            override how the attribute value is ordered, pass a callable that
            takes a single value and returns the value to be ordered.

            .. seealso:: `comparison`

        hash (bool | None):
            Include this attribute in the generated ``__hash__`` method.  If
            None (default), mirror *eq*'s value.  This is the correct behavior
            according the Python spec.  Setting this value to anything else
            than None is *discouraged*.

            .. seealso:: `hashing`

        init (bool):
            Include this attribute in the generated ``__init__`` method.

            It is possible to set this to False and set a default value. In
            that case this attributed is unconditionally initialized with the
            specified default value or factory.

            .. seealso:: `init`

        converter (typing.Callable | Converter):
            A callable that is called by *attrs*-generated ``__init__`` methods
            to convert attribute's value to the desired format.

            If a vanilla callable is passed, it is given the passed-in value as
            the only positional argument. It is possible to receive additional
            arguments by wrapping the callable in a `Converter`.

            Either way, the returned value will be used as the new value of the
            attribute.  The value is converted before being passed to the
            validator, if any.

            .. seealso:: :ref:`converters`

        metadata (dict | None):
            An arbitrary mapping, to be used by third-party code.

            .. seealso:: `extending-metadata`.

        type (type):
            The type of the attribute. Nowadays, the preferred method to
            specify the type is using a variable annotation (see :pep:`526`).
            This argument is provided for backwards-compatibility and for usage
            with `make_class`. Regardless of the approach used, the type will
            be stored on ``Attribute.type``.

            Please note that *attrs* doesn't do anything with this metadata by
            itself. You can use it as part of your own code or for `static type
            checking <types>`.

        kw_only (bool):
            Make this attribute keyword-only in the generated ``__init__`` (if
            ``init`` is False, this parameter is ignored).

        on_setattr (~typing.Callable | list[~typing.Callable] | None | ~typing.Literal[attrs.setters.NO_OP]):
            Allows to overwrite the *on_setattr* setting from `attr.s`. If left
            None, the *on_setattr* value from `attr.s` is used. Set to
            `attrs.setters.NO_OP` to run **no** `setattr` hooks for this
            attribute -- regardless of the setting in `define()`.

        alias (str | None):
            Override this attribute's parameter name in the generated
            ``__init__`` method. If left None, default to ``name`` stripped
            of leading underscores. See `private-attributes`.

    .. versionadded:: 20.1.0
    .. versionchanged:: 21.1.0
       *eq*, *order*, and *cmp* also accept a custom callable
    .. versionadded:: 22.2.0 *alias*
    .. versionadded:: 23.1.0
       The *type* parameter has been re-added; mostly for `attrs.make_class`.
       Please note that type checkers ignore this metadata.

    .. seealso::

       `attr.ib`
    """
    return attrib(
        default=default,
        validator=validator,
        repr=repr,
        hash=hash,
        init=init,
        metadata=metadata,
        type=type,
        converter=converter,
        factory=factory,
        kw_only=kw_only,
        eq=eq,
        order=order,
        on_setattr=on_setattr,
        alias=alias,
    )


def asdict(inst, *, recurse=True, filter=None, value_serializer=None):
    """
    Same as `attr.asdict`, except that collections types are always retained
    and dict is always used as *dict_factory*.

    .. versionadded:: 21.3.0
    """
    return _asdict(
        inst=inst,
        recurse=recurse,
        filter=filter,
        value_serializer=value_serializer,
        retain_collection_types=True,
    )


def astuple(inst, *, recurse=True, filter=None):
    """
    Same as `attr.astuple`, except that collections types are always retained
    and `tuple` is always used as the *tuple_factory*.

    .. versionadded:: 21.3.0
    """
    return _astuple(
        inst=inst, recurse=recurse, filter=filter, retain_collection_types=True
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\optimizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.
#
# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to check optimizations from OnnxRuntime.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
#
# This script is retained for experiment purpose. Useful scenarios like the following:
#  (1) Change model from fp32 to fp16 for mixed precision inference in GPU with Tensor Core.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.

import argparse
import logging
import os
import tempfile
from pathlib import Path

import coloredlogs
from fusion_options import FusionOptions
from onnx import ModelProto, load_model
from onnx_model import OnnxModel
from onnx_model_bart import BartOnnxModel
from onnx_model_bert import BertOnnxModel
from onnx_model_bert_keras import BertOnnxModelKeras
from onnx_model_bert_tf import BertOnnxModelTF
from onnx_model_clip import ClipOnnxModel
from onnx_model_conformer import ConformerOnnxModel
from onnx_model_gpt2 import Gpt2OnnxModel
from onnx_model_mmdit import MmditOnnxModel
from onnx_model_phi import PhiOnnxModel
from onnx_model_sam2 import Sam2OnnxModel
from onnx_model_t5 import T5OnnxModel
from onnx_model_tnlr import TnlrOnnxModel
from onnx_model_unet import UnetOnnxModel
from onnx_model_vae import VaeOnnxModel
from onnx_utils import extract_raw_data_from_model, has_external_data

import onnxruntime

logger = logging.getLogger(__name__)

# Map model type to tuple: optimizer class, export tools (pytorch, tf2onnx, keras2onnx), and default opt_level
MODEL_TYPES = {
    "bart": (BartOnnxModel, "pytorch", 1),
    "bert": (BertOnnxModel, "pytorch", 1),
    "bert_tf": (BertOnnxModelTF, "tf2onnx", 0),
    "bert_keras": (BertOnnxModelKeras, "keras2onnx", 0),
    "clip": (ClipOnnxModel, "pytorch", 1),  # Clip in Stable Diffusion
    "conformer": (ConformerOnnxModel, "pytorch", 1),
    "gpt2": (Gpt2OnnxModel, "pytorch", 1),
    "gpt2_tf": (Gpt2OnnxModel, "tf2onnx", 0),  # might add a class for GPT2OnnxModel for TF later.
    "gpt_neox": (BertOnnxModel, "pytorch", 0),  # GPT-NeoX
    "phi": (PhiOnnxModel, "pytorch", 0),
    "sam2": (Sam2OnnxModel, "pytorch", 1),
    "swin": (BertOnnxModel, "pytorch", 1),
    "tnlr": (TnlrOnnxModel, "pytorch", 1),
    "t5": (T5OnnxModel, "pytorch", 2),
    "unet": (UnetOnnxModel, "pytorch", 1),  # UNet in Stable Diffusion
    "vae": (VaeOnnxModel, "pytorch", 1),  # UAE in Stable Diffusion
    "vit": (BertOnnxModel, "pytorch", 1),
    "mmdit": (MmditOnnxModel, "pytorch", 1),
}


def optimize_by_onnxruntime(
    onnx_model: str | ModelProto | None = None,
    use_gpu: bool = False,
    optimized_model_path: str | None = None,
    opt_level: int | None = 99,
    disabled_optimizers: list[str] = [],  # noqa: B006
    verbose: bool = False,
    save_as_external_data: bool = False,
    external_data_filename: str = "",
    external_data_file_threshold: int = 1024,
    *,
    provider: str | None = None,
    **deprecated_kwargs,
) -> str:
    """
    Use onnxruntime to optimize model.

    Args:
        onnx_model (str | ModelProto): the path of input onnx model or ModelProto.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
        opt_level (int): graph optimization level.
        disabled_optimizers (List[str]): a list of names of disabled optimizers
        save_as_external_data (bool): whether to save external data outside of ONNX model
        external_data_filename (str): name of external data file. If not provided, name is automatically created from ONNX model.
        external_data_file_threshold (int): threshold to decide whether to save tensor in ONNX model or in external data file
        provider (str or None): execution provider to use if use_gpu
    Returns:
        optimized_model_path (str): the path of optimized model
    """
    assert opt_level in [1, 2, 99]
    from torch import version as torch_version

    if onnx_model is None:
        onnx_model = deprecated_kwargs.pop("onnx_model_path", None)
    assert onnx_model is not None

    if (
        use_gpu
        and provider is None
        and set(onnxruntime.get_available_providers()).isdisjoint(
            ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider"]
        )
    ):
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model

    model = (
        OnnxModel(load_model(onnx_model, load_external_data=False))
        if isinstance(onnx_model, str)
        else OnnxModel(onnx_model)
    )
    if model.use_float16() and not use_gpu:
        logger.warning(
            "This model uses float16 in the graph, use_gpu=False might cause extra Cast nodes. "
            "Most operators have no float16 implementation in CPU, so Cast nodes are added to compute them in float32. "
            "If the model is intended to use in GPU, please set use_gpu=True. "
            "Otherwise, consider exporting onnx in float32 and optional int8 quantization for better performance. "
        )

    sess_options = onnxruntime.SessionOptions()
    if opt_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if optimized_model_path is None:
        if isinstance(onnx_model, str):
            path_prefix = str(Path(onnx_model).with_suffix(""))  # remove .onnx suffix
        else:
            path_prefix = "optimized_model"
        optimized_model_path = "{}_o{}_{}.onnx".format(path_prefix, opt_level, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path
    if save_as_external_data:
        if len(external_data_filename) == 0:
            # Set external data filename to model_name.onnx.data
            external_data_filename = os.path.basename(optimized_model_path) + ".data"
        sess_options.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", external_data_filename
        )
        sess_options.add_session_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes", str(external_data_file_threshold)
        )

    if verbose:
        print("Using onnxruntime to optimize model - Debug level Set to verbose")
        sess_options.log_severity_level = 0

    kwargs = {}
    if disabled_optimizers:
        kwargs["disabled_optimizers"] = disabled_optimizers

    if not use_gpu:
        providers = ["CPUExecutionProvider"]
    elif provider is not None:
        if provider == "dml":
            providers = ["DmlExecutionProvider"]
        elif provider == "rocm":
            providers = ["ROCMExecutionProvider"]
        elif provider == "migraphx":
            providers = ["MIGraphXExecutionProvider", "ROCMExecutionProvider"]
        elif provider == "cuda":
            providers = ["CUDAExecutionProvider"]
        elif provider == "tensorrt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider"]

        providers.append("CPUExecutionProvider")
    else:
        providers = []

        if torch_version.hip:
            providers.append("MIGraphXExecutionProvider")
            providers.append("ROCMExecutionProvider")
        else:
            providers.append("CUDAExecutionProvider")

    # For large model, extract external data from model and add to session options
    if isinstance(onnx_model, ModelProto):
        if has_external_data(onnx_model):
            raise ValueError(
                "ModelProto has external data not loaded into memory, ORT cannot create session. "
                "Please load external data before calling this function. "
                "See https://onnx.ai/onnx/repo-docs/ExternalData.html for more information."
            )
        external_names, external_values = extract_raw_data_from_model(onnx_model)
        sess_options.add_external_initializers(list(external_names), list(external_values))

    # Inference session is only used to optimize the model.
    onnx_model = onnx_model.SerializeToString() if isinstance(onnx_model, ModelProto) else onnx_model
    onnxruntime.InferenceSession(onnx_model, sess_options, providers=providers, **kwargs)

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.debug("Save optimized model by onnxruntime to %s", optimized_model_path)
    return optimized_model_path


def optimize_by_fusion(
    model: ModelProto,
    model_type: str = "bert",
    num_heads: int = 0,
    hidden_size: int = 0,
    optimization_options: FusionOptions | None = None,
) -> OnnxModel:
    """Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need to specify these parameters.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically.
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically.
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions.
                                                        Defaults to None.

     Returns:
        object of an optimizer class.
    """
    if model_type not in ["bert", "t5", "swin", "unet", "vae", "clip", "sam2", "mmdit"] and (
        num_heads == 0 or hidden_size == 0
    ):
        logger.warning(f"Please specify parameters of num_heads and hidden_size for model_type {model_type}")

    if model_type not in MODEL_TYPES:
        logger.warning(f"Unsupported model type: {model_type} for graph fusion, directly return model.")
        return OnnxModel(model)

    (optimizer_class, producer, _) = MODEL_TYPES[model_type]

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f'Model producer not matched: Expected "{producer}", Got "{model.producer_name}".'
            "Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = optimizer_class(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    optimizer.model.producer_name = "onnxruntime.transformers"
    from onnxruntime import __version__ as onnxruntime_version

    optimizer.model.producer_version = onnxruntime_version

    return optimizer


def optimize_model(
    input: str | ModelProto,
    model_type: str = "bert",
    num_heads: int = 0,
    hidden_size: int = 0,
    optimization_options: FusionOptions | None = None,
    opt_level: int | None = None,
    use_gpu: bool = False,
    only_onnxruntime: bool = False,
    verbose: bool = False,
    *,
    provider: str | None = None,
) -> OnnxModel:
    """Optimize Model by OnnxRuntime and/or python fusion logic.

    ONNX Runtime has graph optimizations (https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html).
    However, the coverage is limited. We also have graph fusions that implemented in Python to improve the coverage.
    They can combined: ONNX Runtime will run first when opt_level > 0, then graph fusions in Python will be applied.

    To use ONNX Runtime only and no Python fusion logic, use only_onnxruntime flag and a positive opt_level like
        optimize_model(input, opt_level=1, use_gpu=False, only_onnxruntime=True)

    When opt_level is None, we will choose default optimization level according to model type.

    When opt_level is 0 and only_onnxruntime is False, only python fusion logic is used and onnxruntime is disabled.

    When opt_level > 1, use_gpu shall set properly
    since the optimized graph might contain operators for GPU or CPU only.

    If your model is intended for GPU inference only (especially float16 or mixed precision model), it is recommended to
    set use_gpu to be True, otherwise the model is not optimized for GPU inference.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.

    Args:
        input (str | ModelProto): input model path or ModelProto.
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
            0 allows detect the parameter from graph automatically.
        hidden_size (int, optional): hidden size. Defaults to 0.
            0 allows detect the parameter from graph automatically.
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions.
            Defaults to None.
        opt_level (int, optional): onnxruntime graph optimization level (0, 1, 2 or 99) or None. Defaults to None.
            When the value is None, default value (1 for bert and gpt2, 0 for other model types) will be used.
            When the level > 0, onnxruntime will be used to optimize model first.
        use_gpu (bool, optional): use gpu or not for onnxruntime. Defaults to False.
        only_onnxruntime (bool, optional): only use onnxruntime to optimize model, and no python fusion.
            Defaults to False.
        provider (str, optional): execution provider to use if use_gpu. Defaults to None.

     Returns:
        object of an optimizer class.
    """
    assert opt_level is None or opt_level in [0, 1, 2, 99]

    if model_type not in MODEL_TYPES:
        logger.warning(f"Unsupported model type: {model_type} for optimization, directly return model.")
        return OnnxModel(load_model(input)) if isinstance(input, str) else OnnxModel(input)

    (optimizer_class, _, default_opt_level) = MODEL_TYPES[model_type]

    if opt_level is None:
        opt_level = default_opt_level

    # Disable constant sharing to avoid model proto str mismatch in test. Ideally the optimizer should not
    # affect other fusions. We can update the expected model proto once the ConstantSharing optimizer logic becomes
    # stable.
    disabled_optimizers = ["ConstantSharing"]
    temp_model_path = None
    temp_dir = tempfile.TemporaryDirectory()
    optimized_model_name = "model_o{}_{}.onnx".format(opt_level, "gpu" if use_gpu else "cpu")
    optimized_model_path = os.path.join(temp_dir.name, optimized_model_name)

    # Auto detect if input model has external data
    has_external_data_file = False
    original_model = load_model(input, load_external_data=False) if isinstance(input, str) else input
    if has_external_data(original_model):
        has_external_data_file = True
    del original_model

    if opt_level > 1:
        # Disable some optimizers that might cause failure in symbolic shape inference or attention fusion.
        disabled_optimizers += (
            []
            if only_onnxruntime
            else [
                "MatMulScaleFusion",
                "MatMulAddFusion",
                "MatmulTransposeFusion",
                "GemmActivationFusion",
                "BiasSoftmaxFusion",
            ]
        )
        temp_model_path = optimize_by_onnxruntime(
            input,
            use_gpu=use_gpu,
            provider=provider,
            optimized_model_path=optimized_model_path,
            opt_level=opt_level,
            disabled_optimizers=disabled_optimizers,
            verbose=verbose,
            save_as_external_data=has_external_data_file,
        )
    elif opt_level == 1:
        # basic optimizations (like constant folding and cast elimination) are not specified to execution provider.
        # Note that use_gpu=False might cause extra Cast nodes for float16 model since most operators does not support float16 in CPU.
        # Sometime, use_gpu=True might cause extra memory copy nodes when some operators are supported only in CPU.
        # We might need remove GPU memory copy nodes as preprocess of optimize_by_fusion if they cause no matching in fusion.
        temp_model_path = optimize_by_onnxruntime(
            input,
            use_gpu=use_gpu,
            provider=provider,
            optimized_model_path=optimized_model_path,
            opt_level=1,
            disabled_optimizers=disabled_optimizers,
            verbose=verbose,
            save_as_external_data=has_external_data_file,
        )

    if only_onnxruntime and not temp_model_path:
        logger.warning("Please specify a positive value for opt_level when only_onnxruntime is True")

    if temp_model_path is not None:
        model = load_model(temp_model_path)
    elif isinstance(input, str):
        model = load_model(input)
    else:
        model = input

    if only_onnxruntime:
        optimizer = optimizer_class(model, num_heads, hidden_size)
    else:
        optimizer = optimize_by_fusion(model, model_type, num_heads, hidden_size, optimization_options)

    # remove the temporary directory
    temp_dir.cleanup()

    return optimizer


def get_fusion_statistics(optimized_model_path: str) -> dict[str, int]:
    """
    Get counter of fused operators in optimized model.

    Args:
        optimized_model_path (str): the path of onnx model.

    Returns:
        A dictionary with operator type as key, and count as value
    """
    model = load_model(optimized_model_path, format=None, load_external_data=True)
    optimizer = BertOnnxModel(model)
    return optimizer.get_fused_operator_statistics()


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Graph optimization tool for ONNX Runtime."
        "It transforms ONNX graph to use optimized operators for Transformer models."
    )
    parser.add_argument("--input", required=True, type=str, help="input onnx model path")

    parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")

    parser.add_argument(
        "--model_type",
        required=False,
        type=str.lower,
        default="bert",
        choices=list(MODEL_TYPES.keys()),
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES.keys()),
    )

    parser.add_argument(
        "--num_heads",
        required=False,
        type=int,
        default=0,
        help="number of attention heads like 12 for bert-base and 16 for bert-large. "
        "Default is 0 to detect automatically for BERT."
        "For other model type, this parameter need specify correctly.",
    )

    parser.add_argument(
        "--hidden_size",
        required=False,
        type=int,
        default=0,
        help="hidden size like 768 for bert-base and 1024 for bert-large. "
        "Default is 0 to detect automatically for BERT. "
        "For other model type, this parameter need specify correctly.",
    )

    parser.add_argument(
        "--input_int32",
        required=False,
        action="store_true",
        help="Use int32 (instead of int64) inputs. "
        "It could avoid unnecessary data cast when EmbedLayerNormalization is fused for BERT.",
    )
    parser.set_defaults(input_int32=False)

    parser.add_argument(
        "--float16",
        required=False,
        action="store_true",
        help="Convert all weights and nodes in float32 to float16. "
        "It has potential loss in precision compared to mixed precision conversion.",
    )
    parser.set_defaults(float16=False)

    FusionOptions.add_arguments(parser)

    parser.add_argument("--verbose", required=False, action="store_true", help="show debug information.")
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="Use GPU for inference. Set this flag if your model is intended for GPU when opt_level > 1.",
    )
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "--provider",
        required=False,
        type=str,
        default=None,
        help="Execution provider to use if use_gpu",
    )

    parser.add_argument(
        "--only_onnxruntime",
        required=False,
        action="store_true",
        help="optimized by onnxruntime only, and no graph fusion in Python",
    )
    parser.set_defaults(only_onnxruntime=False)

    parser.add_argument(
        "--opt_level",
        required=False,
        type=int,
        choices=[0, 1, 2, 99],
        default=None,
        help="onnxruntime optimization level. 0 will disable onnxruntime graph optimization. "
        "The recommended value is 1. When opt_level > 1 is used, optimized model for GPU might not run in CPU. "
        "Level 2 and 99 are intended for --only_onnxruntime.",
    )

    parser.add_argument(
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="use external data format to store large model (>2GB)",
    )
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument(
        "--disable_symbolic_shape_infer",
        required=False,
        action="store_true",
        help="disable symbolic shape inference",
    )
    parser.set_defaults(disable_symbolic_shape_infer=False)

    parser.add_argument(
        "--convert_to_packing_mode",
        required=False,
        action="store_true",
        help="convert the model to packing mode. Only available for BERT like model",
    )
    parser.set_defaults(convert_to_packing_mode=False)

    parser.add_argument(
        "--convert_attribute",
        required=False,
        action="store_true",
        help="convert attributes when using a rewritten ONNX model (e.g. Dynamo-exported model from ONNX Script)",
    )
    parser.set_defaults(convert_attribute=False)

    args = parser.parse_args()

    return args


def _setup_logger(verbose):
    if verbose:
        coloredlogs.install(
            level="DEBUG",
            fmt="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s",
        )
    else:
        coloredlogs.install(fmt="%(funcName)20s: %(message)s")


def main():
    args = _parse_arguments()

    _setup_logger(args.verbose)

    logger.debug(f"arguments:{args}")

    if os.path.realpath(args.input) == os.path.realpath(args.output):
        logger.warning("Specified the same input and output path. Note that this may overwrite the original model")

    optimization_options = FusionOptions.parse(args)

    optimizer = optimize_model(
        args.input,
        args.model_type,
        args.num_heads,
        args.hidden_size,
        opt_level=args.opt_level,
        optimization_options=optimization_options,
        use_gpu=args.use_gpu,
        provider=args.provider,
        only_onnxruntime=args.only_onnxruntime,
    )

    if args.float16:
        optimizer.convert_float_to_float16(keep_io_types=True)

    if args.input_int32:
        optimizer.change_graph_inputs_to_int32()

    # Print the operator statistics might help end user.
    optimizer.get_operator_statistics()

    fused_op_count = optimizer.get_fused_operator_statistics()
    if "bert" in args.model_type and optimizer.is_fully_optimized(fused_op_count):
        logger.info("The model has been fully optimized.")
    else:
        logger.info("The model has been optimized.")

    if args.convert_to_packing_mode:
        if args.model_type == "bert":
            optimizer.convert_to_packing_mode(not args.disable_symbolic_shape_infer)
        else:
            logger.warning("Packing mode only supports BERT like models")

    optimizer.save_model_to_file(args.output, args.use_external_data_format, convert_attribute=args.convert_attribute)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\req\req_file.py ===
"""
Requirements file parsing
"""

import codecs
import locale
import logging
import optparse
import os
import re
import shlex
import sys
import urllib.parse
from dataclasses import dataclass
from optparse import Values
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    NoReturn,
    Optional,
    Tuple,
)

from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.network.session import PipSession

__all__ = ["parse_requirements"]

ReqFileLines = Iterable[Tuple[int, str]]

LineParser = Callable[[str], Tuple[str, Values]]

SCHEME_RE = re.compile(r"^(http|https|file):", re.I)
COMMENT_RE = re.compile(r"(^|\s+)#.*$")

# Matches environment variable-style values in '${MY_VARIABLE_1}' with the
# variable name consisting of only uppercase letters, digits or the '_'
# (underscore). This follows the POSIX standard defined in IEEE Std 1003.1,
# 2013 Edition.
ENV_VAR_RE = re.compile(r"(?P<var>\$\{(?P<name>[A-Z0-9_]+)\})")

SUPPORTED_OPTIONS: List[Callable[..., optparse.Option]] = [
    cmdoptions.index_url,
    cmdoptions.extra_index_url,
    cmdoptions.no_index,
    cmdoptions.constraints,
    cmdoptions.requirements,
    cmdoptions.editable,
    cmdoptions.find_links,
    cmdoptions.no_binary,
    cmdoptions.only_binary,
    cmdoptions.prefer_binary,
    cmdoptions.require_hashes,
    cmdoptions.pre,
    cmdoptions.trusted_host,
    cmdoptions.use_new_feature,
]

# options to be passed to requirements
SUPPORTED_OPTIONS_REQ: List[Callable[..., optparse.Option]] = [
    cmdoptions.global_options,
    cmdoptions.hash,
    cmdoptions.config_settings,
]

SUPPORTED_OPTIONS_EDITABLE_REQ: List[Callable[..., optparse.Option]] = [
    cmdoptions.config_settings,
]


# the 'dest' string values
SUPPORTED_OPTIONS_REQ_DEST = [str(o().dest) for o in SUPPORTED_OPTIONS_REQ]
SUPPORTED_OPTIONS_EDITABLE_REQ_DEST = [
    str(o().dest) for o in SUPPORTED_OPTIONS_EDITABLE_REQ
]

# order of BOMS is important: codecs.BOM_UTF16_LE is a prefix of codecs.BOM_UTF32_LE
# so data.startswith(BOM_UTF16_LE) would be true for UTF32_LE data
BOMS: List[Tuple[bytes, str]] = [
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF32, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF16, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
]

PEP263_ENCODING_RE = re.compile(rb"coding[:=]\s*([-\w.]+)")
DEFAULT_ENCODING = "utf-8"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedRequirement:
    # TODO: replace this with slots=True when dropping Python 3.9 support.
    __slots__ = (
        "requirement",
        "is_editable",
        "comes_from",
        "constraint",
        "options",
        "line_source",
    )

    requirement: str
    is_editable: bool
    comes_from: str
    constraint: bool
    options: Optional[Dict[str, Any]]
    line_source: Optional[str]


@dataclass(frozen=True)
class ParsedLine:
    __slots__ = ("filename", "lineno", "args", "opts", "constraint")

    filename: str
    lineno: int
    args: str
    opts: Values
    constraint: bool

    @property
    def is_editable(self) -> bool:
        return bool(self.opts.editables)

    @property
    def requirement(self) -> Optional[str]:
        if self.args:
            return self.args
        elif self.is_editable:
            # We don't support multiple -e on one line
            return self.opts.editables[0]
        return None


def parse_requirements(
    filename: str,
    session: "PipSession",
    finder: Optional["PackageFinder"] = None,
    options: Optional[optparse.Values] = None,
    constraint: bool = False,
) -> Generator[ParsedRequirement, None, None]:
    """Parse a requirements file and yield ParsedRequirement instances.

    :param filename:    Path or url of requirements file.
    :param session:     PipSession instance.
    :param finder:      Instance of pip.index.PackageFinder.
    :param options:     cli options.
    :param constraint:  If true, parsing a constraint file rather than
        requirements file.
    """
    line_parser = get_line_parser(finder)
    parser = RequirementsFileParser(session, line_parser)

    for parsed_line in parser.parse(filename, constraint):
        parsed_req = handle_line(
            parsed_line, options=options, finder=finder, session=session
        )
        if parsed_req is not None:
            yield parsed_req


def preprocess(content: str) -> ReqFileLines:
    """Split, filter, and join lines, and return a line iterator

    :param content: the content of the requirements file
    """
    lines_enum: ReqFileLines = enumerate(content.splitlines(), start=1)
    lines_enum = join_lines(lines_enum)
    lines_enum = ignore_comments(lines_enum)
    lines_enum = expand_env_variables(lines_enum)
    return lines_enum


def handle_requirement_line(
    line: ParsedLine,
    options: Optional[optparse.Values] = None,
) -> ParsedRequirement:
    # preserve for the nested code path
    line_comes_from = "{} {} (line {})".format(
        "-c" if line.constraint else "-r",
        line.filename,
        line.lineno,
    )

    assert line.requirement is not None

    # get the options that apply to requirements
    if line.is_editable:
        supported_dest = SUPPORTED_OPTIONS_EDITABLE_REQ_DEST
    else:
        supported_dest = SUPPORTED_OPTIONS_REQ_DEST
    req_options = {}
    for dest in supported_dest:
        if dest in line.opts.__dict__ and line.opts.__dict__[dest]:
            req_options[dest] = line.opts.__dict__[dest]

    line_source = f"line {line.lineno} of {line.filename}"
    return ParsedRequirement(
        requirement=line.requirement,
        is_editable=line.is_editable,
        comes_from=line_comes_from,
        constraint=line.constraint,
        options=req_options,
        line_source=line_source,
    )


def handle_option_line(
    opts: Values,
    filename: str,
    lineno: int,
    finder: Optional["PackageFinder"] = None,
    options: Optional[optparse.Values] = None,
    session: Optional["PipSession"] = None,
) -> None:
    if opts.hashes:
        logger.warning(
            "%s line %s has --hash but no requirement, and will be ignored.",
            filename,
            lineno,
        )

    if options:
        # percolate options upward
        if opts.require_hashes:
            options.require_hashes = opts.require_hashes
        if opts.features_enabled:
            options.features_enabled.extend(
                f for f in opts.features_enabled if f not in options.features_enabled
            )

    # set finder options
    if finder:
        find_links = finder.find_links
        index_urls = finder.index_urls
        no_index = finder.search_scope.no_index
        if opts.no_index is True:
            no_index = True
            index_urls = []
        if opts.index_url and not no_index:
            index_urls = [opts.index_url]
        if opts.extra_index_urls and not no_index:
            index_urls.extend(opts.extra_index_urls)
        if opts.find_links:
            # FIXME: it would be nice to keep track of the source
            # of the find_links: support a find-links local path
            # relative to a requirements file.
            value = opts.find_links[0]
            req_dir = os.path.dirname(os.path.abspath(filename))
            relative_to_reqs_file = os.path.join(req_dir, value)
            if os.path.exists(relative_to_reqs_file):
                value = relative_to_reqs_file
            find_links.append(value)

        if session:
            # We need to update the auth urls in session
            session.update_index_urls(index_urls)

        search_scope = SearchScope(
            find_links=find_links,
            index_urls=index_urls,
            no_index=no_index,
        )
        finder.search_scope = search_scope

        if opts.pre:
            finder.set_allow_all_prereleases()

        if opts.prefer_binary:
            finder.set_prefer_binary()

        if session:
            for host in opts.trusted_hosts or []:
                source = f"line {lineno} of {filename}"
                session.add_trusted_host(host, source=source)


def handle_line(
    line: ParsedLine,
    options: Optional[optparse.Values] = None,
    finder: Optional["PackageFinder"] = None,
    session: Optional["PipSession"] = None,
) -> Optional[ParsedRequirement]:
    """Handle a single parsed requirements line; This can result in
    creating/yielding requirements, or updating the finder.

    :param line:        The parsed line to be processed.
    :param options:     CLI options.
    :param finder:      The finder - updated by non-requirement lines.
    :param session:     The session - updated by non-requirement lines.

    Returns a ParsedRequirement object if the line is a requirement line,
    otherwise returns None.

    For lines that contain requirements, the only options that have an effect
    are from SUPPORTED_OPTIONS_REQ, and they are scoped to the
    requirement. Other options from SUPPORTED_OPTIONS may be present, but are
    ignored.

    For lines that do not contain requirements, the only options that have an
    effect are from SUPPORTED_OPTIONS. Options from SUPPORTED_OPTIONS_REQ may
    be present, but are ignored. These lines may contain multiple options
    (although our docs imply only one is supported), and all our parsed and
    affect the finder.
    """

    if line.requirement is not None:
        parsed_req = handle_requirement_line(line, options)
        return parsed_req
    else:
        handle_option_line(
            line.opts,
            line.filename,
            line.lineno,
            finder,
            options,
            session,
        )
        return None


class RequirementsFileParser:
    def __init__(
        self,
        session: "PipSession",
        line_parser: LineParser,
    ) -> None:
        self._session = session
        self._line_parser = line_parser

    def parse(
        self, filename: str, constraint: bool
    ) -> Generator[ParsedLine, None, None]:
        """Parse a given file, yielding parsed lines."""
        yield from self._parse_and_recurse(
            filename, constraint, [{os.path.abspath(filename): None}]
        )

    def _parse_and_recurse(
        self,
        filename: str,
        constraint: bool,
        parsed_files_stack: List[Dict[str, Optional[str]]],
    ) -> Generator[ParsedLine, None, None]:
        for line in self._parse_file(filename, constraint):
            if line.requirement is None and (
                line.opts.requirements or line.opts.constraints
            ):
                # parse a nested requirements file
                if line.opts.requirements:
                    req_path = line.opts.requirements[0]
                    nested_constraint = False
                else:
                    req_path = line.opts.constraints[0]
                    nested_constraint = True

                # original file is over http
                if SCHEME_RE.search(filename):
                    # do a url join so relative paths work
                    req_path = urllib.parse.urljoin(filename, req_path)
                # original file and nested file are paths
                elif not SCHEME_RE.search(req_path):
                    # do a join so relative paths work
                    # and then abspath so that we can identify recursive references
                    req_path = os.path.abspath(
                        os.path.join(
                            os.path.dirname(filename),
                            req_path,
                        )
                    )
                parsed_files = parsed_files_stack[0]
                if req_path in parsed_files:
                    initial_file = parsed_files[req_path]
                    tail = (
                        f" and again in {initial_file}"
                        if initial_file is not None
                        else ""
                    )
                    raise RequirementsFileParseError(
                        f"{req_path} recursively references itself in {filename}{tail}"
                    )
                # Keeping a track where was each file first included in
                new_parsed_files = parsed_files.copy()
                new_parsed_files[req_path] = filename
                yield from self._parse_and_recurse(
                    req_path, nested_constraint, [new_parsed_files, *parsed_files_stack]
                )
            else:
                yield line

    def _parse_file(
        self, filename: str, constraint: bool
    ) -> Generator[ParsedLine, None, None]:
        _, content = get_file_content(filename, self._session)

        lines_enum = preprocess(content)

        for line_number, line in lines_enum:
            try:
                args_str, opts = self._line_parser(line)
            except OptionParsingError as e:
                # add offending line
                msg = f"Invalid requirement: {line}\n{e.msg}"
                raise RequirementsFileParseError(msg)

            yield ParsedLine(
                filename,
                line_number,
                args_str,
                opts,
                constraint,
            )


def get_line_parser(finder: Optional["PackageFinder"]) -> LineParser:
    def parse_line(line: str) -> Tuple[str, Values]:
        # Build new parser for each line since it accumulates appendable
        # options.
        parser = build_parser()
        defaults = parser.get_default_values()
        defaults.index_url = None
        if finder:
            defaults.format_control = finder.format_control

        args_str, options_str = break_args_options(line)

        try:
            options = shlex.split(options_str)
        except ValueError as e:
            raise OptionParsingError(f"Could not split options: {options_str}") from e

        opts, _ = parser.parse_args(options, defaults)

        return args_str, opts

    return parse_line


def break_args_options(line: str) -> Tuple[str, str]:
    """Break up the line into an args and options string.  We only want to shlex
    (and then optparse) the options, not the args.  args can contain markers
    which are corrupted by shlex.
    """
    tokens = line.split(" ")
    args = []
    options = tokens[:]
    for token in tokens:
        if token.startswith("-") or token.startswith("--"):
            break
        else:
            args.append(token)
            options.pop(0)
    return " ".join(args), " ".join(options)


class OptionParsingError(Exception):
    def __init__(self, msg: str) -> None:
        self.msg = msg


def build_parser() -> optparse.OptionParser:
    """
    Return a parser for parsing requirement lines
    """
    parser = optparse.OptionParser(add_help_option=False)

    option_factories = SUPPORTED_OPTIONS + SUPPORTED_OPTIONS_REQ
    for option_factory in option_factories:
        option = option_factory()
        parser.add_option(option)

    # By default optparse sys.exits on parsing errors. We want to wrap
    # that in our own exception.
    def parser_exit(self: Any, msg: str) -> "NoReturn":
        raise OptionParsingError(msg)

    # NOTE: mypy disallows assigning to a method
    #       https://github.com/python/mypy/issues/2427
    parser.exit = parser_exit  # type: ignore

    return parser


def join_lines(lines_enum: ReqFileLines) -> ReqFileLines:
    """Joins a line ending in '\' with the previous line (except when following
    comments).  The joined line takes on the index of the first line.
    """
    primary_line_number = None
    new_line: List[str] = []
    for line_number, line in lines_enum:
        if not line.endswith("\\") or COMMENT_RE.match(line):
            if COMMENT_RE.match(line):
                # this ensures comments are always matched later
                line = " " + line
            if new_line:
                new_line.append(line)
                assert primary_line_number is not None
                yield primary_line_number, "".join(new_line)
                new_line = []
            else:
                yield line_number, line
        else:
            if not new_line:
                primary_line_number = line_number
            new_line.append(line.strip("\\"))

    # last line contains \
    if new_line:
        assert primary_line_number is not None
        yield primary_line_number, "".join(new_line)

    # TODO: handle space after '\'.


def ignore_comments(lines_enum: ReqFileLines) -> ReqFileLines:
    """
    Strips comments and filter empty lines.
    """
    for line_number, line in lines_enum:
        line = COMMENT_RE.sub("", line)
        line = line.strip()
        if line:
            yield line_number, line


def expand_env_variables(lines_enum: ReqFileLines) -> ReqFileLines:
    """Replace all environment variables that can be retrieved via `os.getenv`.

    The only allowed format for environment variables defined in the
    requirement file is `${MY_VARIABLE_1}` to ensure two things:

    1. Strings that contain a `$` aren't accidentally (partially) expanded.
    2. Ensure consistency across platforms for requirement files.

    These points are the result of a discussion on the `github pull
    request #3514 <https://github.com/pypa/pip/pull/3514>`_.

    Valid characters in variable names follow the `POSIX standard
    <http://pubs.opengroup.org/onlinepubs/9699919799/>`_ and are limited
    to uppercase letter, digits and the `_` (underscore).
    """
    for line_number, line in lines_enum:
        for env_var, var_name in ENV_VAR_RE.findall(line):
            value = os.getenv(var_name)
            if not value:
                continue

            line = line.replace(env_var, value)

        yield line_number, line


def get_file_content(url: str, session: "PipSession") -> Tuple[str, str]:
    """Gets the content of a file; it may be a filename, file: URL, or
    http: URL.  Returns (location, content).  Content is unicode.
    Respects # -*- coding: declarations on the retrieved files.

    :param url:         File path or url.
    :param session:     PipSession instance.
    """
    scheme = urllib.parse.urlsplit(url).scheme
    # Pip has special support for file:// URLs (LocalFSAdapter).
    if scheme in ["http", "https", "file"]:
        # Delay importing heavy network modules until absolutely necessary.
        from pip._internal.network.utils import raise_for_status

        resp = session.get(url)
        raise_for_status(resp)
        return resp.url, resp.text

    # Assume this is a bare path.
    try:
        with open(url, "rb") as f:
            raw_content = f.read()
    except OSError as exc:
        raise InstallationError(f"Could not open requirements file: {exc}")

    content = _decode_req_file(raw_content, url)

    return url, content


def _decode_req_file(data: bytes, url: str) -> str:
    for bom, encoding in BOMS:
        if data.startswith(bom):
            return data[len(bom) :].decode(encoding)

    for line in data.split(b"\n")[:2]:
        if line[0:1] == b"#":
            result = PEP263_ENCODING_RE.search(line)
            if result is not None:
                encoding = result.groups()[0].decode("ascii")
                return data.decode(encoding)

    try:
        return data.decode(DEFAULT_ENCODING)
    except UnicodeDecodeError:
        locale_encoding = locale.getpreferredencoding(False) or sys.getdefaultencoding()
        logging.warning(
            "unable to decode data from %s with default encoding %s, "
            "falling back to encoding from locale: %s. "
            "If this is intentional you should specify the encoding with a "
            "PEP-263 style comment, e.g. '# -*- coding: %s -*-'",
            url,
            DEFAULT_ENCODING,
            locale_encoding,
            locale_encoding,
        )
        return data.decode(locale_encoding)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\galoisgroups.py ===
"""
Compute Galois groups of polynomials.

We use algorithms from [1], with some modifications to use lookup tables for
resolvents.

References
==========

.. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*.

"""

from collections import defaultdict
import random

from sympy.core.symbol import Dummy, symbols
from sympy.ntheory.primetest import is_square
from sympy.polys.domains import ZZ
from sympy.polys.densebasic import dup_random
from sympy.polys.densetools import dup_eval
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p
from sympy.polys.numberfields.galois_resolvents import (
    GaloisGroupException, get_resolvent_by_lookup, define_resolvents,
    Resolvent,
)
from sympy.polys.numberfields.utilities import coeff_search
from sympy.polys.polytools import (Poly, poly_from_expr,
                                   PolificationFailed, ComputationFailed)
from sympy.polys.sqfreetools import dup_sqf_p
from sympy.utilities import public


class MaxTriesException(GaloisGroupException):
    ...


def tschirnhausen_transformation(T, max_coeff=10, max_tries=30, history=None,
                                 fixed_order=True):
    r"""
    Given a univariate, monic, irreducible polynomial over the integers, find
    another such polynomial defining the same number field.

    Explanation
    ===========

    See Alg 6.3.4 of [1].

    Parameters
    ==========

    T : Poly
        The given polynomial
    max_coeff : int
        When choosing a transformation as part of the process,
        keep the coeffs between plus and minus this.
    max_tries : int
        Consider at most this many transformations.
    history : set, None, optional (default=None)
        Pass a set of ``Poly.rep``'s in order to prevent any of these
        polynomials from being returned as the polynomial ``U`` i.e. the
        transformation of the given polynomial *T*. The given poly *T* will
        automatically be added to this set, before we try to find a new one.
    fixed_order : bool, default True
        If ``True``, work through candidate transformations A(x) in a fixed
        order, from small coeffs to large, resulting in deterministic behavior.
        If ``False``, the A(x) are chosen randomly, while still working our way
        up from small coefficients to larger ones.

    Returns
    =======

    Pair ``(A, U)``

        ``A`` and ``U`` are ``Poly``, ``A`` is the
        transformation, and ``U`` is the transformed polynomial that defines
        the same number field as *T*. The polynomial ``A`` maps the roots of
        *T* to the roots of ``U``.

    Raises
    ======

    MaxTriesException
        if could not find a polynomial before exceeding *max_tries*.

    """
    X = Dummy('X')
    n = T.degree()
    if history is None:
        history = set()
    history.add(T.rep)

    if fixed_order:
        coeff_generators = {}
        deg_coeff_sum = 3
        current_degree = 2

    def get_coeff_generator(degree):
        gen = coeff_generators.get(degree, coeff_search(degree, 1))
        coeff_generators[degree] = gen
        return gen

    for i in range(max_tries):

        # We never use linear A(x), since applying a fixed linear transformation
        # to all roots will only multiply the discriminant of T by a square
        # integer. This will change nothing important. In particular, if disc(T)
        # was zero before, it will still be zero now, and typically we apply
        # the transformation in hopes of replacing T by a squarefree poly.

        if fixed_order:
            # If d is degree and c max coeff, we move through the dc-space
            # along lines of constant sum. First d + c = 3 with (d, c) = (2, 1).
            # Then d + c = 4 with (d, c) = (3, 1), (2, 2). Then d + c = 5 with
            # (d, c) = (4, 1), (3, 2), (2, 3), and so forth. For a given (d, c)
            # we go though all sets of coeffs where max = c, before moving on.
            gen = get_coeff_generator(current_degree)
            coeffs = next(gen)
            m = max(abs(c) for c in coeffs)
            if current_degree + m > deg_coeff_sum:
                if current_degree == 2:
                    deg_coeff_sum += 1
                    current_degree = deg_coeff_sum - 1
                else:
                    current_degree -= 1
                gen = get_coeff_generator(current_degree)
                coeffs = next(gen)
            a = [ZZ(1)] + [ZZ(c) for c in coeffs]

        else:
            # We use a progressive coeff bound, up to the max specified, since it
            # is preferable to succeed with smaller coeffs.
            # Give each coeff bound five tries, before incrementing.
            C = min(i//5 + 1, max_coeff)
            d = random.randint(2, n - 1)
            a = dup_random(d, -C, C, ZZ)

        A = Poly(a, T.gen)
        U = Poly(T.resultant(X - A), X)
        if U.rep not in history and dup_sqf_p(U.rep.to_list(), ZZ):
            return A, U
    raise MaxTriesException


def has_square_disc(T):
    """Convenience to check if a Poly or dup has square discriminant. """
    d = T.discriminant() if isinstance(T, Poly) else dup_discriminant(T, ZZ)
    return is_square(d)


def _galois_group_degree_3(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 3.

    Explanation
    ===========

    Uses Prop 6.3.5 of [1].

    """
    from sympy.combinatorics.galois import S3TransitiveSubgroups
    return ((S3TransitiveSubgroups.A3, True) if has_square_disc(T)
            else (S3TransitiveSubgroups.S3, False))


def _galois_group_degree_4_root_approx(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Follows Alg 6.3.7 of [1], using a pure root approximation approach.

    """
    from sympy.combinatorics.permutations import Permutation
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    X = symbols('X0 X1 X2 X3')
    # We start by considering the resolvent for the form
    #   F = X0*X2 + X1*X3
    # and the group G = S4. In this case, the stabilizer H is D4 = < (0123), (02) >,
    # and a set of representatives of G/H is {I, (01), (03)}
    F1 = X[0]*X[2] + X[1]*X[3]
    s1 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]
    R1 = Resolvent(F1, X, s1)

    # In the second half of the algorithm (if we reach it), we use another
    # form and set of coset representatives. However, we may need to permute
    # them first, so cannot form their resolvent now.
    F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s2_pre = [
        Permutation(3),
        Permutation(3)(0, 2)
    ]

    history = set()
    for i in range(max_tries):
        if i > 0:
            # If we're retrying, need a new polynomial T.
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)

        R_dup, _, i0 = R1.eval_for_poly(T, find_integer_root=True)
        # If R is not squarefree, must retry.
        if not dup_sqf_p(R_dup, ZZ):
            continue

        # By Prop 6.3.1 of [1], Gal(T) is contained in A4 iff disc(T) is square.
        sq_disc = has_square_disc(T)

        if i0 is None:
            # By Thm 6.3.3 of [1], Gal(T) is not conjugate to any subgroup of the
            # stabilizer H = D4 that we chose. This means Gal(T) is either A4 or S4.
            return ((S4TransitiveSubgroups.A4, True) if sq_disc
                    else (S4TransitiveSubgroups.S4, False))

        # Gal(T) is conjugate to a subgroup of H = D4, so it is either V, C4
        # or D4 itself.

        if sq_disc:
            # Neither C4 nor D4 is contained in A4, so Gal(T) must be V.
            return (S4TransitiveSubgroups.V, True)

        # Gal(T) can only be D4 or C4.
        # We will now use our second resolvent, with G being that conjugate of D4 that
        # Gal(T) is contained in. To determine the right conjugate, we will need
        # the permutation corresponding to the integer root we found.
        sigma = s1[i0]
        # Applying sigma means permuting the args of F, and
        # conjugating the set of coset representatives.
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma*tau*sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)
        # If d is zero (R has a repeated root), must retry.
        if d == 0:
            continue
        if is_square(d):
            return (S4TransitiveSubgroups.C4, False)
        else:
            return (S4TransitiveSubgroups.D4, False)

    raise MaxTriesException


def _galois_group_degree_4_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Based on Alg 6.3.6 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S4TransitiveSubgroups

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 0)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    # Compute list L of degrees of irreducible factors of R, in increasing order:
    fl = dup_factor_list(R_dup, ZZ)
    L = sorted(sum([
        [len(r) - 1] * e for r, e in fl[1]
    ], []))

    if L == [6]:
        return ((S4TransitiveSubgroups.A4, True) if has_square_disc(T)
            else (S4TransitiveSubgroups.S4, False))

    if L == [1, 1, 4]:
        return (S4TransitiveSubgroups.C4, False)

    if L == [2, 2, 2]:
        return (S4TransitiveSubgroups.V, True)

    assert L == [2, 4]
    return (S4TransitiveSubgroups.D4, False)


def _galois_group_degree_5_hybrid(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses a hybrid approach, combining resolvent
    coeff lookup, with root approximation.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups
    from sympy.combinatorics.permutations import Permutation

    X5 = symbols("X0,X1,X2,X3,X4")
    res = define_resolvents()
    F51, _, s51 = res[(5, 1)]
    F51 = F51.as_expr(*X5)
    R51 = Resolvent(F51, X5, s51)

    history = set()
    reached_second_stage = False
    for i in range(max_tries):
        if i > 0:
            _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                                history=history,
                                                fixed_order=not randomize)
        R51_dup = get_resolvent_by_lookup(T, 1)
        if not dup_sqf_p(R51_dup, ZZ):
            continue

        # First stage
        # If we have not yet reached the second stage, then the group still
        # might be S5, A5, or M20, so must test for that.
        if not reached_second_stage:
            sq_disc = has_square_disc(T)

            if dup_irreducible_p(R51_dup, ZZ):
                return ((S5TransitiveSubgroups.A5, True) if sq_disc
                        else (S5TransitiveSubgroups.S5, False))

            if not sq_disc:
                return (S5TransitiveSubgroups.M20, False)

        # Second stage
        reached_second_stage = True
        # R51 must have an integer root for T.
        # To choose our second resolvent, we need to know which conjugate of
        # F51 is a root.
        rounded_roots = R51.round_roots_to_integers_for_poly(T)
        # These are integers, and candidates to be roots of R51.
        # We find the first one that actually is a root.
        for permutation_index, candidate_root in rounded_roots.items():
            if not dup_eval(R51_dup, candidate_root, ZZ):
                break

        X = X5
        F2_pre = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[4]**2 + X[4]*X[0]**2
        s2_pre = [
            Permutation(4),
            Permutation(4)(0, 1)(2, 4)
        ]

        i0 = permutation_index
        sigma = s51[i0]
        F2 = F2_pre.subs(zip(X, sigma(X)), simultaneous=True)
        s2 = [sigma*tau*sigma for tau in s2_pre]
        R2 = Resolvent(F2, X, s2)
        R_dup, _, _ = R2.eval_for_poly(T)
        d = dup_discriminant(R_dup, ZZ)

        if d == 0:
            continue
        if is_square(d):
            return (S5TransitiveSubgroups.C5, True)
        else:
            return (S5TransitiveSubgroups.D5, True)

    raise MaxTriesException


def _galois_group_degree_5_lookup_ext_factor(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 5.

    Explanation
    ===========

    Based on Alg 6.3.9 of [1], but uses resolvent coeff lookup, plus
    factorization over an algebraic extension.

    """
    from sympy.combinatorics.galois import S5TransitiveSubgroups

    _T = T

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    sq_disc = has_square_disc(T)

    if dup_irreducible_p(R_dup, ZZ):
        return ((S5TransitiveSubgroups.A5, True) if sq_disc
                else (S5TransitiveSubgroups.S5, False))

    if not sq_disc:
        return (S5TransitiveSubgroups.M20, False)

    # If we get this far, Gal(T) can only be D5 or C5.
    # But for Gal(T) to have order 5, T must already split completely in
    # the extension field obtained by adjoining a single one of its roots.
    fl = Poly(_T, domain=ZZ.alg_field_from_poly(_T)).factor_list()[1]
    if len(fl) == 5:
        return (S5TransitiveSubgroups.C5, True)
    else:
        return (S5TransitiveSubgroups.D5, True)


def _galois_group_degree_6_lookup(T, max_tries=30, randomize=False):
    r"""
    Compute the Galois group of a polynomial of degree 6.

    Explanation
    ===========

    Based on Alg 6.3.10 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S6TransitiveSubgroups

    # First resolvent:

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    fl = dup_factor_list(R_dup, ZZ)

    # Group the factors by degree.
    factors_by_deg = defaultdict(list)
    for r, _ in fl[1]:
        factors_by_deg[len(r) - 1].append(r)

    L = sorted(sum([
        [d] * len(ff) for d, ff in factors_by_deg.items()
    ], []))

    T_has_sq_disc = has_square_disc(T)

    if L == [1, 2, 3]:
        f1 = factors_by_deg[3][0]
        return ((S6TransitiveSubgroups.C6, False) if has_square_disc(f1)
                else (S6TransitiveSubgroups.D6, False))

    elif L == [3, 3]:
        f1, f2 = factors_by_deg[3]
        any_square = has_square_disc(f1) or has_square_disc(f2)
        return ((S6TransitiveSubgroups.G18, False) if any_square
                else (S6TransitiveSubgroups.G36m, False))

    elif L == [2, 4]:
        if T_has_sq_disc:
            return (S6TransitiveSubgroups.S4p, True)
        else:
            f1 = factors_by_deg[4][0]
            return ((S6TransitiveSubgroups.A4xC2, False) if has_square_disc(f1)
                    else (S6TransitiveSubgroups.S4xC2, False))

    elif L == [1, 1, 4]:
        return ((S6TransitiveSubgroups.A4, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S4m, False))

    elif L == [1, 5]:
        return ((S6TransitiveSubgroups.PSL2F5, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.PGL2F5, False))

    elif L == [1, 1, 1, 3]:
        return (S6TransitiveSubgroups.S3, False)

    assert L == [6]

    # Second resolvent:

    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 2)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries,
                                            history=history,
                                            fixed_order=not randomize)
    else:
        raise MaxTriesException

    T_has_sq_disc = has_square_disc(T)

    if dup_irreducible_p(R_dup, ZZ):
        return ((S6TransitiveSubgroups.A6, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.S6, False))
    else:
        return ((S6TransitiveSubgroups.G36p, True) if T_has_sq_disc
                else (S6TransitiveSubgroups.G72, False))


@public
def galois_group(f, *gens, by_name=False, max_tries=30, randomize=False, **args):
    r"""
    Compute the Galois group for polynomials *f* up to degree 6.

    Examples
    ========

    >>> from sympy import galois_group
    >>> from sympy.abc import x
    >>> f = x**4 + 1
    >>> G, alt = galois_group(f)
    >>> print(G)
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    The group is returned along with a boolean, indicating whether it is
    contained in the alternating group $A_n$, where $n$ is the degree of *T*.
    Along with other group properties, this can help determine which group it
    is:

    >>> alt
    True
    >>> G.order()
    4

    Alternatively, the group can be returned by name:

    >>> G_name, _ = galois_group(f, by_name=True)
    >>> print(G_name)
    S4TransitiveSubgroups.V

    The group itself can then be obtained by calling the name's
    ``get_perm_group()`` method:

    >>> G_name.get_perm_group()
    PermutationGroup([
    (0 1)(2 3),
    (0 2)(1 3)])

    Group names are values of the enum classes
    :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`,
    :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`,
    etc.

    Parameters
    ==========

    f : Expr
        Irreducible polynomial over :ref:`ZZ` or :ref:`QQ`, whose Galois group
        is to be determined.
    gens : optional list of symbols
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.
    by_name : bool, default False
        If ``True``, the Galois group will be returned by name.
        Otherwise it will be returned as a :py:class:`~.PermutationGroup`.
    max_tries : int, default 30
        Make at most this many attempts in those steps that involve
        generating Tschirnhausen transformations.
    randomize : bool, default False
        If ``True``, then use random coefficients when generating Tschirnhausen
        transformations. Otherwise try transformations in a fixed order. Both
        approaches start with small coefficients and degrees and work upward.
    args : optional
        For converting *f* to Poly, and will be passed on to the
        :py:func:`~.poly_from_expr` function.

    Returns
    =======

    Pair ``(G, alt)``
        The first element ``G`` indicates the Galois group. It is an instance
        of one of the :py:class:`sympy.combinatorics.galois.S1TransitiveSubgroups`
        :py:class:`sympy.combinatorics.galois.S2TransitiveSubgroups`, etc. enum
        classes if *by_name* was ``True``, and a :py:class:`~.PermutationGroup`
        if ``False``.

        The second element is a boolean, saying whether the group is contained
        in the alternating group $A_n$ ($n$ the degree of *T*).

    Raises
    ======

    ValueError
        if *f* is of an unsupported degree.

    MaxTriesException
        if could not complete before exceeding *max_tries* in those steps
        that involve generating Tschirnhausen transformations.

    See Also
    ========

    .Poly.galois_group

    """
    gens = gens or []
    args = args or {}

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('galois_group', 1, exc)

    return F.galois_group(by_name=by_name, max_tries=max_tries,
                          randomize=randomize)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\_compat.py ===
from __future__ import annotations

import codecs
import collections.abc as cabc
import io
import os
import re
import sys
import typing as t
from types import TracebackType
from weakref import WeakKeyDictionary

CYGWIN = sys.platform.startswith("cygwin")
WIN = sys.platform.startswith("win")
auto_wrap_for_ansi: t.Callable[[t.TextIO], t.TextIO] | None = None
_ansi_re = re.compile(r"\033\[[;?0-9]*[a-zA-Z]")


def _make_text_stream(
    stream: t.BinaryIO,
    encoding: str | None,
    errors: str | None,
    force_readable: bool = False,
    force_writable: bool = False,
) -> t.TextIO:
    if encoding is None:
        encoding = get_best_encoding(stream)
    if errors is None:
        errors = "replace"
    return _NonClosingTextIOWrapper(
        stream,
        encoding,
        errors,
        line_buffering=True,
        force_readable=force_readable,
        force_writable=force_writable,
    )


def is_ascii_encoding(encoding: str) -> bool:
    """Checks if a given encoding is ascii."""
    try:
        return codecs.lookup(encoding).name == "ascii"
    except LookupError:
        return False


def get_best_encoding(stream: t.IO[t.Any]) -> str:
    """Returns the default stream encoding if not found."""
    rv = getattr(stream, "encoding", None) or sys.getdefaultencoding()
    if is_ascii_encoding(rv):
        return "utf-8"
    return rv


class _NonClosingTextIOWrapper(io.TextIOWrapper):
    def __init__(
        self,
        stream: t.BinaryIO,
        encoding: str | None,
        errors: str | None,
        force_readable: bool = False,
        force_writable: bool = False,
        **extra: t.Any,
    ) -> None:
        self._stream = stream = t.cast(
            t.BinaryIO, _FixupStream(stream, force_readable, force_writable)
        )
        super().__init__(stream, encoding, errors, **extra)

    def __del__(self) -> None:
        try:
            self.detach()
        except Exception:
            pass

    def isatty(self) -> bool:
        # https://bitbucket.org/pypy/pypy/issue/1803
        return self._stream.isatty()


class _FixupStream:
    """The new io interface needs more from streams than streams
    traditionally implement.  As such, this fix-up code is necessary in
    some circumstances.

    The forcing of readable and writable flags are there because some tools
    put badly patched objects on sys (one such offender are certain version
    of jupyter notebook).
    """

    def __init__(
        self,
        stream: t.BinaryIO,
        force_readable: bool = False,
        force_writable: bool = False,
    ):
        self._stream = stream
        self._force_readable = force_readable
        self._force_writable = force_writable

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._stream, name)

    def read1(self, size: int) -> bytes:
        f = getattr(self._stream, "read1", None)

        if f is not None:
            return t.cast(bytes, f(size))

        return self._stream.read(size)

    def readable(self) -> bool:
        if self._force_readable:
            return True
        x = getattr(self._stream, "readable", None)
        if x is not None:
            return t.cast(bool, x())
        try:
            self._stream.read(0)
        except Exception:
            return False
        return True

    def writable(self) -> bool:
        if self._force_writable:
            return True
        x = getattr(self._stream, "writable", None)
        if x is not None:
            return t.cast(bool, x())
        try:
            self._stream.write(b"")
        except Exception:
            try:
                self._stream.write(b"")
            except Exception:
                return False
        return True

    def seekable(self) -> bool:
        x = getattr(self._stream, "seekable", None)
        if x is not None:
            return t.cast(bool, x())
        try:
            self._stream.seek(self._stream.tell())
        except Exception:
            return False
        return True


def _is_binary_reader(stream: t.IO[t.Any], default: bool = False) -> bool:
    try:
        return isinstance(stream.read(0), bytes)
    except Exception:
        return default
        # This happens in some cases where the stream was already
        # closed.  In this case, we assume the default.


def _is_binary_writer(stream: t.IO[t.Any], default: bool = False) -> bool:
    try:
        stream.write(b"")
    except Exception:
        try:
            stream.write("")
            return False
        except Exception:
            pass
        return default
    return True


def _find_binary_reader(stream: t.IO[t.Any]) -> t.BinaryIO | None:
    # We need to figure out if the given stream is already binary.
    # This can happen because the official docs recommend detaching
    # the streams to get binary streams.  Some code might do this, so
    # we need to deal with this case explicitly.
    if _is_binary_reader(stream, False):
        return t.cast(t.BinaryIO, stream)

    buf = getattr(stream, "buffer", None)

    # Same situation here; this time we assume that the buffer is
    # actually binary in case it's closed.
    if buf is not None and _is_binary_reader(buf, True):
        return t.cast(t.BinaryIO, buf)

    return None


def _find_binary_writer(stream: t.IO[t.Any]) -> t.BinaryIO | None:
    # We need to figure out if the given stream is already binary.
    # This can happen because the official docs recommend detaching
    # the streams to get binary streams.  Some code might do this, so
    # we need to deal with this case explicitly.
    if _is_binary_writer(stream, False):
        return t.cast(t.BinaryIO, stream)

    buf = getattr(stream, "buffer", None)

    # Same situation here; this time we assume that the buffer is
    # actually binary in case it's closed.
    if buf is not None and _is_binary_writer(buf, True):
        return t.cast(t.BinaryIO, buf)

    return None


def _stream_is_misconfigured(stream: t.TextIO) -> bool:
    """A stream is misconfigured if its encoding is ASCII."""
    # If the stream does not have an encoding set, we assume it's set
    # to ASCII.  This appears to happen in certain unittest
    # environments.  It's not quite clear what the correct behavior is
    # but this at least will force Click to recover somehow.
    return is_ascii_encoding(getattr(stream, "encoding", None) or "ascii")


def _is_compat_stream_attr(stream: t.TextIO, attr: str, value: str | None) -> bool:
    """A stream attribute is compatible if it is equal to the
    desired value or the desired value is unset and the attribute
    has a value.
    """
    stream_value = getattr(stream, attr, None)
    return stream_value == value or (value is None and stream_value is not None)


def _is_compatible_text_stream(
    stream: t.TextIO, encoding: str | None, errors: str | None
) -> bool:
    """Check if a stream's encoding and errors attributes are
    compatible with the desired values.
    """
    return _is_compat_stream_attr(
        stream, "encoding", encoding
    ) and _is_compat_stream_attr(stream, "errors", errors)


def _force_correct_text_stream(
    text_stream: t.IO[t.Any],
    encoding: str | None,
    errors: str | None,
    is_binary: t.Callable[[t.IO[t.Any], bool], bool],
    find_binary: t.Callable[[t.IO[t.Any]], t.BinaryIO | None],
    force_readable: bool = False,
    force_writable: bool = False,
) -> t.TextIO:
    if is_binary(text_stream, False):
        binary_reader = t.cast(t.BinaryIO, text_stream)
    else:
        text_stream = t.cast(t.TextIO, text_stream)
        # If the stream looks compatible, and won't default to a
        # misconfigured ascii encoding, return it as-is.
        if _is_compatible_text_stream(text_stream, encoding, errors) and not (
            encoding is None and _stream_is_misconfigured(text_stream)
        ):
            return text_stream

        # Otherwise, get the underlying binary reader.
        possible_binary_reader = find_binary(text_stream)

        # If that's not possible, silently use the original reader
        # and get mojibake instead of exceptions.
        if possible_binary_reader is None:
            return text_stream

        binary_reader = possible_binary_reader

    # Default errors to replace instead of strict in order to get
    # something that works.
    if errors is None:
        errors = "replace"

    # Wrap the binary stream in a text stream with the correct
    # encoding parameters.
    return _make_text_stream(
        binary_reader,
        encoding,
        errors,
        force_readable=force_readable,
        force_writable=force_writable,
    )


def _force_correct_text_reader(
    text_reader: t.IO[t.Any],
    encoding: str | None,
    errors: str | None,
    force_readable: bool = False,
) -> t.TextIO:
    return _force_correct_text_stream(
        text_reader,
        encoding,
        errors,
        _is_binary_reader,
        _find_binary_reader,
        force_readable=force_readable,
    )


def _force_correct_text_writer(
    text_writer: t.IO[t.Any],
    encoding: str | None,
    errors: str | None,
    force_writable: bool = False,
) -> t.TextIO:
    return _force_correct_text_stream(
        text_writer,
        encoding,
        errors,
        _is_binary_writer,
        _find_binary_writer,
        force_writable=force_writable,
    )


def get_binary_stdin() -> t.BinaryIO:
    reader = _find_binary_reader(sys.stdin)
    if reader is None:
        raise RuntimeError("Was not able to determine binary stream for sys.stdin.")
    return reader


def get_binary_stdout() -> t.BinaryIO:
    writer = _find_binary_writer(sys.stdout)
    if writer is None:
        raise RuntimeError("Was not able to determine binary stream for sys.stdout.")
    return writer


def get_binary_stderr() -> t.BinaryIO:
    writer = _find_binary_writer(sys.stderr)
    if writer is None:
        raise RuntimeError("Was not able to determine binary stream for sys.stderr.")
    return writer


def get_text_stdin(encoding: str | None = None, errors: str | None = None) -> t.TextIO:
    rv = _get_windows_console_stream(sys.stdin, encoding, errors)
    if rv is not None:
        return rv
    return _force_correct_text_reader(sys.stdin, encoding, errors, force_readable=True)


def get_text_stdout(encoding: str | None = None, errors: str | None = None) -> t.TextIO:
    rv = _get_windows_console_stream(sys.stdout, encoding, errors)
    if rv is not None:
        return rv
    return _force_correct_text_writer(sys.stdout, encoding, errors, force_writable=True)


def get_text_stderr(encoding: str | None = None, errors: str | None = None) -> t.TextIO:
    rv = _get_windows_console_stream(sys.stderr, encoding, errors)
    if rv is not None:
        return rv
    return _force_correct_text_writer(sys.stderr, encoding, errors, force_writable=True)


def _wrap_io_open(
    file: str | os.PathLike[str] | int,
    mode: str,
    encoding: str | None,
    errors: str | None,
) -> t.IO[t.Any]:
    """Handles not passing ``encoding`` and ``errors`` in binary mode."""
    if "b" in mode:
        return open(file, mode)

    return open(file, mode, encoding=encoding, errors=errors)


def open_stream(
    filename: str | os.PathLike[str],
    mode: str = "r",
    encoding: str | None = None,
    errors: str | None = "strict",
    atomic: bool = False,
) -> tuple[t.IO[t.Any], bool]:
    binary = "b" in mode
    filename = os.fspath(filename)

    # Standard streams first. These are simple because they ignore the
    # atomic flag. Use fsdecode to handle Path("-").
    if os.fsdecode(filename) == "-":
        if any(m in mode for m in ["w", "a", "x"]):
            if binary:
                return get_binary_stdout(), False
            return get_text_stdout(encoding=encoding, errors=errors), False
        if binary:
            return get_binary_stdin(), False
        return get_text_stdin(encoding=encoding, errors=errors), False

    # Non-atomic writes directly go out through the regular open functions.
    if not atomic:
        return _wrap_io_open(filename, mode, encoding, errors), True

    # Some usability stuff for atomic writes
    if "a" in mode:
        raise ValueError(
            "Appending to an existing file is not supported, because that"
            " would involve an expensive `copy`-operation to a temporary"
            " file. Open the file in normal `w`-mode and copy explicitly"
            " if that's what you're after."
        )
    if "x" in mode:
        raise ValueError("Use the `overwrite`-parameter instead.")
    if "w" not in mode:
        raise ValueError("Atomic writes only make sense with `w`-mode.")

    # Atomic writes are more complicated.  They work by opening a file
    # as a proxy in the same folder and then using the fdopen
    # functionality to wrap it in a Python file.  Then we wrap it in an
    # atomic file that moves the file over on close.
    import errno
    import random

    try:
        perm: int | None = os.stat(filename).st_mode
    except OSError:
        perm = None

    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL

    if binary:
        flags |= getattr(os, "O_BINARY", 0)

    while True:
        tmp_filename = os.path.join(
            os.path.dirname(filename),
            f".__atomic-write{random.randrange(1 << 32):08x}",
        )
        try:
            fd = os.open(tmp_filename, flags, 0o666 if perm is None else perm)
            break
        except OSError as e:
            if e.errno == errno.EEXIST or (
                os.name == "nt"
                and e.errno == errno.EACCES
                and os.path.isdir(e.filename)
                and os.access(e.filename, os.W_OK)
            ):
                continue
            raise

    if perm is not None:
        os.chmod(tmp_filename, perm)  # in case perm includes bits in umask

    f = _wrap_io_open(fd, mode, encoding, errors)
    af = _AtomicFile(f, tmp_filename, os.path.realpath(filename))
    return t.cast(t.IO[t.Any], af), True


class _AtomicFile:
    def __init__(self, f: t.IO[t.Any], tmp_filename: str, real_filename: str) -> None:
        self._f = f
        self._tmp_filename = tmp_filename
        self._real_filename = real_filename
        self.closed = False

    @property
    def name(self) -> str:
        return self._real_filename

    def close(self, delete: bool = False) -> None:
        if self.closed:
            return
        self._f.close()
        os.replace(self._tmp_filename, self._real_filename)
        self.closed = True

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._f, name)

    def __enter__(self) -> _AtomicFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close(delete=exc_type is not None)

    def __repr__(self) -> str:
        return repr(self._f)


def strip_ansi(value: str) -> str:
    return _ansi_re.sub("", value)


def _is_jupyter_kernel_output(stream: t.IO[t.Any]) -> bool:
    while isinstance(stream, (_FixupStream, _NonClosingTextIOWrapper)):
        stream = stream._stream

    return stream.__class__.__module__.startswith("ipykernel.")


def should_strip_ansi(
    stream: t.IO[t.Any] | None = None, color: bool | None = None
) -> bool:
    if color is None:
        if stream is None:
            stream = sys.stdin
        return not isatty(stream) and not _is_jupyter_kernel_output(stream)
    return not color


# On Windows, wrap the output streams with colorama to support ANSI
# color codes.
# NOTE: double check is needed so mypy does not analyze this on Linux
if sys.platform.startswith("win") and WIN:
    from ._winconsole import _get_windows_console_stream

    def _get_argv_encoding() -> str:
        import locale

        return locale.getpreferredencoding()

    _ansi_stream_wrappers: cabc.MutableMapping[t.TextIO, t.TextIO] = WeakKeyDictionary()

    def auto_wrap_for_ansi(stream: t.TextIO, color: bool | None = None) -> t.TextIO:
        """Support ANSI color and style codes on Windows by wrapping a
        stream with colorama.
        """
        try:
            cached = _ansi_stream_wrappers.get(stream)
        except Exception:
            cached = None

        if cached is not None:
            return cached

        import colorama

        strip = should_strip_ansi(stream, color)
        ansi_wrapper = colorama.AnsiToWin32(stream, strip=strip)
        rv = t.cast(t.TextIO, ansi_wrapper.stream)
        _write = rv.write

        def _safe_write(s: str) -> int:
            try:
                return _write(s)
            except BaseException:
                ansi_wrapper.reset_all()
                raise

        rv.write = _safe_write  # type: ignore[method-assign]

        try:
            _ansi_stream_wrappers[stream] = rv
        except Exception:
            pass

        return rv

else:

    def _get_argv_encoding() -> str:
        return getattr(sys.stdin, "encoding", None) or sys.getfilesystemencoding()

    def _get_windows_console_stream(
        f: t.TextIO, encoding: str | None, errors: str | None
    ) -> t.TextIO | None:
        return None


def term_len(x: str) -> int:
    return len(strip_ansi(x))


def isatty(stream: t.IO[t.Any]) -> bool:
    try:
        return stream.isatty()
    except Exception:
        return False


def _make_cached_stream_func(
    src_func: t.Callable[[], t.TextIO | None],
    wrapper_func: t.Callable[[], t.TextIO],
) -> t.Callable[[], t.TextIO | None]:
    cache: cabc.MutableMapping[t.TextIO, t.TextIO] = WeakKeyDictionary()

    def func() -> t.TextIO | None:
        stream = src_func()

        if stream is None:
            return None

        try:
            rv = cache.get(stream)
        except Exception:
            rv = None
        if rv is not None:
            return rv
        rv = wrapper_func()
        try:
            cache[stream] = rv
        except Exception:
            pass
        return rv

    return func


_default_text_stdin = _make_cached_stream_func(lambda: sys.stdin, get_text_stdin)
_default_text_stdout = _make_cached_stream_func(lambda: sys.stdout, get_text_stdout)
_default_text_stderr = _make_cached_stream_func(lambda: sys.stderr, get_text_stderr)


binary_streams: cabc.Mapping[str, t.Callable[[], t.BinaryIO]] = {
    "stdin": get_binary_stdin,
    "stdout": get_binary_stdout,
    "stderr": get_binary_stderr,
}

text_streams: cabc.Mapping[str, t.Callable[[str | None, str | None], t.TextIO]] = {
    "stdin": get_text_stdin,
    "stdout": get_text_stdout,
    "stderr": get_text_stderr,
}