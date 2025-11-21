
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\locations\base.py ===
import functools
import os
import site
import sys
import sysconfig
import typing

from pip._internal.exceptions import InstallationError
from pip._internal.utils import appdirs
from pip._internal.utils.virtualenv import running_under_virtualenv

# Application Directories
USER_CACHE_DIR = appdirs.user_cache_dir("pip")

# FIXME doesn't account for venv linked to global site-packages
site_packages: str = sysconfig.get_path("purelib")


def get_major_minor_version() -> str:
    """
    Return the major-minor version of the current Python as a string, e.g.
    "3.7" or "3.10".
    """
    return "{}.{}".format(*sys.version_info)


def change_root(new_root: str, pathname: str) -> str:
    """Return 'pathname' with 'new_root' prepended.

    If 'pathname' is relative, this is equivalent to os.path.join(new_root, pathname).
    Otherwise, it requires making 'pathname' relative and then joining the
    two, which is tricky on DOS/Windows and Mac OS.

    This is borrowed from Python's standard library's distutils module.
    """
    if os.name == "posix":
        if not os.path.isabs(pathname):
            return os.path.join(new_root, pathname)
        else:
            return os.path.join(new_root, pathname[1:])

    elif os.name == "nt":
        (drive, path) = os.path.splitdrive(pathname)
        if path[0] == "\\":
            path = path[1:]
        return os.path.join(new_root, path)

    else:
        raise InstallationError(
            f"Unknown platform: {os.name}\n"
            "Can not change root path prefix on unknown platform."
        )


def get_src_prefix() -> str:
    if running_under_virtualenv():
        src_prefix = os.path.join(sys.prefix, "src")
    else:
        # FIXME: keep src in cwd for now (it is not a temporary folder)
        try:
            src_prefix = os.path.join(os.getcwd(), "src")
        except OSError:
            # In case the current working directory has been renamed or deleted
            sys.exit("The folder you are executing pip from can no longer be found.")

    # under macOS + virtualenv sys.prefix is not properly resolved
    # it is something like /path/to/python/bin/..
    return os.path.abspath(src_prefix)


try:
    # Use getusersitepackages if this is present, as it ensures that the
    # value is initialised properly.
    user_site: typing.Optional[str] = site.getusersitepackages()
except AttributeError:
    user_site = site.USER_SITE


@functools.lru_cache(maxsize=None)
def is_osx_framework() -> bool:
    return bool(sysconfig.get_config_var("PYTHONFRAMEWORK"))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\mobile.py ===
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

from .command import Command


class _ConnectionType:
    def __init__(self, mask):
        self.mask = mask

    @property
    def airplane_mode(self):
        return self.mask % 2 == 1

    @property
    def wifi(self):
        return (self.mask / 2) % 2 == 1

    @property
    def data(self):
        return (self.mask / 4) > 0


class Mobile:
    ConnectionType = _ConnectionType
    ALL_NETWORK = ConnectionType(6)
    WIFI_NETWORK = ConnectionType(2)
    DATA_NETWORK = ConnectionType(4)
    AIRPLANE_MODE = ConnectionType(1)

    def __init__(self, driver):
        import weakref

        self._driver = weakref.proxy(driver)

    @property
    def network_connection(self):
        return self.ConnectionType(self._driver.execute(Command.GET_NETWORK_CONNECTION)["value"])

    def set_network_connection(self, network):
        """Set the network connection for the remote device.

        Example of setting airplane mode::

            driver.mobile.set_network_connection(driver.mobile.AIRPLANE_MODE)
        """
        mode = network.mask if isinstance(network, self.ConnectionType) else network
        return self.ConnectionType(
            self._driver.execute(
                Command.SET_NETWORK_CONNECTION, {"name": "network_connection", "parameters": {"type": mode}}
            )["value"]
        )

    @property
    def context(self):
        """Returns the current context (Native or WebView)."""
        return self._driver.execute(Command.CURRENT_CONTEXT_HANDLE)

    @context.setter
    def context(self, new_context) -> None:
        """Sets the current context."""
        self._driver.execute(Command.SWITCH_TO_CONTEXT, {"name": new_context})

    @property
    def contexts(self):
        """Returns a list of available contexts."""
        return self._driver.execute(Command.CONTEXT_HANDLES)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\json.py ===
# dialects/mysql/json.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from ... import types as sqltypes


class JSON(sqltypes.JSON):
    """MySQL JSON type.

    MySQL supports JSON as of version 5.7.
    MariaDB supports JSON (as an alias for LONGTEXT) as of version 10.2.

    :class:`_mysql.JSON` is used automatically whenever the base
    :class:`_types.JSON` datatype is used against a MySQL or MariaDB backend.

    .. seealso::

        :class:`_types.JSON` - main documentation for the generic
        cross-platform JSON datatype.

    The :class:`.mysql.JSON` type supports persistence of JSON values
    as well as the core index operations provided by :class:`_types.JSON`
    datatype, by adapting the operations to render the ``JSON_EXTRACT``
    function at the database level.

    """

    pass


class _FormatTypeMixin:
    def _format_value(self, value):
        raise NotImplementedError()

    def bind_processor(self, dialect):
        super_proc = self.string_bind_processor(dialect)

        def process(value):
            value = self._format_value(value)
            if super_proc:
                value = super_proc(value)
            return value

        return process

    def literal_processor(self, dialect):
        super_proc = self.string_literal_processor(dialect)

        def process(value):
            value = self._format_value(value)
            if super_proc:
                value = super_proc(value)
            return value

        return process


class JSONIndexType(_FormatTypeMixin, sqltypes.JSON.JSONIndexType):
    def _format_value(self, value):
        if isinstance(value, int):
            value = "$[%s]" % value
        else:
            value = '$."%s"' % value
        return value


class JSONPathType(_FormatTypeMixin, sqltypes.JSON.JSONPathType):
    def _format_value(self, value):
        return "$%s" % (
            "".join(
                [
                    "[%s]" % elem if isinstance(elem, int) else '."%s"' % elem
                    for elem in value
                ]
            )
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\predicates\common.py ===
from sympy.assumptions import Predicate, AppliedPredicate, Q
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.multipledispatch import Dispatcher


class CommutativePredicate(Predicate):
    """
    Commutative predicate.

    Explanation
    ===========

    ``ask(Q.commutative(x))`` is true iff ``x`` commutes with any other
    object with respect to multiplication operation.

    """
    # TODO: Add examples
    name = 'commutative'
    handler = Dispatcher("CommutativeHandler", doc="Handler for key 'commutative'.")


binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}

class IsTruePredicate(Predicate):
    """
    Generic predicate.

    Explanation
    ===========

    ``ask(Q.is_true(x))`` is true iff ``x`` is true. This only makes
    sense if ``x`` is a boolean object.

    Examples
    ========

    >>> from sympy import ask, Q
    >>> from sympy.abc import x, y
    >>> ask(Q.is_true(True))
    True

    Wrapping another applied predicate just returns the applied predicate.

    >>> Q.is_true(Q.even(x))
    Q.even(x)

    Wrapping binary relation classes in SymPy core returns applied binary
    relational predicates.

    >>> from sympy import Eq, Gt
    >>> Q.is_true(Eq(x, y))
    Q.eq(x, y)
    >>> Q.is_true(Gt(x, y))
    Q.gt(x, y)

    Notes
    =====

    This class is designed to wrap the boolean objects so that they can
    behave as if they are applied predicates. Consequently, wrapping another
    applied predicate is unnecessary and thus it just returns the argument.
    Also, binary relation classes in SymPy core have binary predicates to
    represent themselves and thus wrapping them with ``Q.is_true`` converts them
    to these applied predicates.

    """
    name = 'is_true'
    handler = Dispatcher(
        "IsTrueHandler",
        doc="Wrapper allowing to query the truth value of a boolean expression."
    )

    def __call__(self, arg):
        # No need to wrap another predicate
        if isinstance(arg, AppliedPredicate):
            return arg
        # Convert relational predicates instead of wrapping them
        if getattr(arg, "is_Relational", False):
            pred = binrelpreds[type(arg)]
            return pred(*arg.args)
        return super().__call__(arg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\middleware\dispatcher.py ===
"""
Application Dispatcher
======================

This middleware creates a single WSGI application that dispatches to
multiple other WSGI applications mounted at different URL paths.

A common example is writing a Single Page Application, where you have a
backend API and a frontend written in JavaScript that does the routing
in the browser rather than requesting different pages from the server.
The frontend is a single HTML and JS file that should be served for any
path besides "/api".

This example dispatches to an API app under "/api", an admin app
under "/admin", and an app that serves frontend files for all other
requests::

    app = DispatcherMiddleware(serve_frontend, {
        '/api': api_app,
        '/admin': admin_app,
    })

In production, you might instead handle this at the HTTP server level,
serving files or proxying to application servers based on location. The
API and admin apps would each be deployed with a separate WSGI server,
and the static files would be served directly by the HTTP server.

.. autoclass:: DispatcherMiddleware

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class DispatcherMiddleware:
    """Combine multiple applications as a single WSGI application.
    Requests are dispatched to an application based on the path it is
    mounted under.

    :param app: The WSGI application to dispatch to if the request
        doesn't match a mounted path.
    :param mounts: Maps path prefixes to applications for dispatching.
    """

    def __init__(
        self,
        app: WSGIApplication,
        mounts: dict[str, WSGIApplication] | None = None,
    ) -> None:
        self.app = app
        self.mounts = mounts or {}

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        script = environ.get("PATH_INFO", "")
        path_info = ""

        while "/" in script:
            if script in self.mounts:
                app = self.mounts[script]
                break

            script, last_item = script.rsplit("/", 1)
            path_info = f"/{last_item}{path_info}"
        else:
            app = self.mounts.get(script, self.app)

        original_script_name = environ.get("SCRIPT_NAME", "")
        environ["SCRIPT_NAME"] = original_script_name + script
        environ["PATH_INFO"] = path_info
        return app(environ, start_response)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\_deprecation.py ===
"""Helper functions for deprecation.

This interface is itself unstable and may change without warning. Do
not use these functions yourself, even as a joke. The underscores are
there for a reason. No support will be given.

In particular, most of this will go away without warning once
Beautiful Soup drops support for Python 3.11, since Python 3.12
defines a `@typing.deprecated()
decorator. <https://peps.python.org/pep-0702/>`_
"""

import functools
import warnings

from typing import (
    Any,
    Callable,
)


def _deprecated_alias(old_name: str, new_name: str, version: str):
    """Alias one attribute name to another for backward compatibility

    :meta private:
    """

    @property
    def alias(self) -> Any:
        ":meta private:"
        warnings.warn(
            f"Access to deprecated property {old_name}. (Replaced by {new_name}) -- Deprecated since version {version}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)

    @alias.setter
    def alias(self, value: str) -> None:
        ":meta private:"
        warnings.warn(
            f"Write to deprecated property {old_name}. (Replaced by {new_name}) -- Deprecated since version {version}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return setattr(self, new_name, value)

    return alias


def _deprecated_function_alias(
    old_name: str, new_name: str, version: str
) -> Callable[[Any], Any]:
    def alias(self, *args: Any, **kwargs: Any) -> Any:
        ":meta private:"
        warnings.warn(
            f"Call to deprecated method {old_name}. (Replaced by {new_name}) -- Deprecated since version {version}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)(*args, **kwargs)

    return alias


def _deprecated(replaced_by: str, version: str) -> Callable:
    def deprecate(func: Callable) -> Callable:
        @functools.wraps(func)
        def with_warning(*args: Any, **kwargs: Any) -> Any:
            ":meta private:"
            warnings.warn(
                f"Call to deprecated method {func.__name__}. (Replaced by {replaced_by}) -- Deprecated since version {version}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return with_warning

    return deprecate

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\tz\_factories.py ===
from datetime import timedelta
import weakref
from collections import OrderedDict

from six.moves import _thread


class _TzSingleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super(_TzSingleton, cls).__init__(*args, **kwargs)

    def __call__(cls):
        if cls.__instance is None:
            cls.__instance = super(_TzSingleton, cls).__call__()
        return cls.__instance


class _TzFactory(type):
    def instance(cls, *args, **kwargs):
        """Alternate constructor that returns a fresh instance"""
        return type.__call__(cls, *args, **kwargs)


class _TzOffsetFactory(_TzFactory):
    def __init__(cls, *args, **kwargs):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__strong_cache = OrderedDict()
        cls.__strong_cache_size = 8

        cls._cache_lock = _thread.allocate_lock()

    def __call__(cls, name, offset):
        if isinstance(offset, timedelta):
            key = (name, offset.total_seconds())
        else:
            key = (name, offset)

        instance = cls.__instances.get(key, None)
        if instance is None:
            instance = cls.__instances.setdefault(key,
                                                  cls.instance(name, offset))

        # This lock may not be necessary in Python 3. See GH issue #901
        with cls._cache_lock:
            cls.__strong_cache[key] = cls.__strong_cache.pop(key, instance)

            # Remove an item if the strong cache is overpopulated
            if len(cls.__strong_cache) > cls.__strong_cache_size:
                cls.__strong_cache.popitem(last=False)

        return instance


class _TzStrFactory(_TzFactory):
    def __init__(cls, *args, **kwargs):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__strong_cache = OrderedDict()
        cls.__strong_cache_size = 8

        cls.__cache_lock = _thread.allocate_lock()

    def __call__(cls, s, posix_offset=False):
        key = (s, posix_offset)
        instance = cls.__instances.get(key, None)

        if instance is None:
            instance = cls.__instances.setdefault(key,
                cls.instance(s, posix_offset))

        # This lock may not be necessary in Python 3. See GH issue #901
        with cls.__cache_lock:
            cls.__strong_cache[key] = cls.__strong_cache.pop(key, instance)

            # Remove an item if the strong cache is overpopulated
            if len(cls.__strong_cache) > cls.__strong_cache_size:
                cls.__strong_cache.popitem(last=False)

        return instance


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\DimensionValue.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DimensionValue(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DimensionValue()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDimensionValue(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DimensionValueBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # DimensionValue
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DimensionValue
    def DimType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DimensionValue
    def DimValue(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # DimensionValue
    def DimParam(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def DimensionValueStart(builder):
    builder.StartObject(3)

def Start(builder):
    DimensionValueStart(builder)

def DimensionValueAddDimType(builder, dimType):
    builder.PrependInt8Slot(0, dimType, 0)

def AddDimType(builder, dimType):
    DimensionValueAddDimType(builder, dimType)

def DimensionValueAddDimValue(builder, dimValue):
    builder.PrependInt64Slot(1, dimValue, 0)

def AddDimValue(builder, dimValue):
    DimensionValueAddDimValue(builder, dimValue)

def DimensionValueAddDimParam(builder, dimParam):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(dimParam), 0)

def AddDimParam(builder, dimParam):
    DimensionValueAddDimParam(builder, dimParam)

def DimensionValueEnd(builder):
    return builder.EndObject()

def End(builder):
    return DimensionValueEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\egg_link.py ===
import os
import re
import sys
from typing import List, Optional

from pip._internal.locations import site_packages, user_site
from pip._internal.utils.virtualenv import (
    running_under_virtualenv,
    virtualenv_no_global,
)

__all__ = [
    "egg_link_path_from_sys_path",
    "egg_link_path_from_location",
]


def _egg_link_names(raw_name: str) -> List[str]:
    """
    Convert a Name metadata value to a .egg-link name, by applying
    the same substitution as pkg_resources's safe_name function.
    Note: we cannot use canonicalize_name because it has a different logic.

    We also look for the raw name (without normalization) as setuptools 69 changed
    the way it names .egg-link files (https://github.com/pypa/setuptools/issues/4167).
    """
    return [
        re.sub("[^A-Za-z0-9.]+", "-", raw_name) + ".egg-link",
        f"{raw_name}.egg-link",
    ]


def egg_link_path_from_sys_path(raw_name: str) -> Optional[str]:
    """
    Look for a .egg-link file for project name, by walking sys.path.
    """
    egg_link_names = _egg_link_names(raw_name)
    for path_item in sys.path:
        for egg_link_name in egg_link_names:
            egg_link = os.path.join(path_item, egg_link_name)
            if os.path.isfile(egg_link):
                return egg_link
    return None


def egg_link_path_from_location(raw_name: str) -> Optional[str]:
    """
    Return the path for the .egg-link file if it exists, otherwise, None.

    There's 3 scenarios:
    1) not in a virtualenv
       try to find in site.USER_SITE, then site_packages
    2) in a no-global virtualenv
       try to find in site_packages
    3) in a yes-global virtualenv
       try to find in site_packages, then site.USER_SITE
       (don't look in global location)

    For #1 and #3, there could be odd cases, where there's an egg-link in 2
    locations.

    This method will just return the first one found.
    """
    sites: List[str] = []
    if running_under_virtualenv():
        sites.append(site_packages)
        if not virtualenv_no_global() and user_site:
            sites.append(user_site)
    else:
        if user_site:
            sites.append(user_site)
        sites.append(site_packages)

    egg_link_names = _egg_link_names(raw_name)
    for site in sites:
        for egg_link_name in egg_link_names:
            egglink = os.path.join(site, egg_link_name)
            if os.path.isfile(egglink):
                return egglink
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\alert.py ===
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
"""The Alert implementation."""

from selenium.webdriver.common.utils import keys_to_typing
from selenium.webdriver.remote.command import Command


class Alert:
    """Allows to work with alerts.

    Use this class to interact with alert prompts.  It contains methods for dismissing,
    accepting, inputting, and getting text from alert prompts.

    Accepting / Dismissing alert prompts::

        Alert(driver).accept()
        Alert(driver).dismiss()

    Inputting a value into an alert prompt::

        name_prompt = Alert(driver)
        name_prompt.send_keys("Willian Shakesphere")
        name_prompt.accept()


    Reading a the text of a prompt for verification::

        alert_text = Alert(driver).text
        self.assertEqual("Do you wish to quit?", alert_text)
    """

    def __init__(self, driver) -> None:
        """Creates a new Alert.

        :Args:
         - driver: The WebDriver instance which performs user actions.
        """
        self.driver = driver

    @property
    def text(self) -> str:
        """Gets the text of the Alert."""
        return self.driver.execute(Command.W3C_GET_ALERT_TEXT)["value"]

    def dismiss(self) -> None:
        """Dismisses the alert available."""
        self.driver.execute(Command.W3C_DISMISS_ALERT)

    def accept(self) -> None:
        """Accepts the alert available.

        :Usage:
            ::

                Alert(driver).accept()  # Confirm a alert dialog.
        """
        self.driver.execute(Command.W3C_ACCEPT_ALERT)

    def send_keys(self, keysToSend: str) -> None:
        """Send Keys to the Alert.

        :Args:
         - keysToSend: The text to be sent to Alert.
        """
        self.driver.execute(Command.W3C_SET_ALERT_VALUE, {"value": keys_to_typing(keysToSend), "text": keysToSend})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\pointer_input.py ===
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

from typing import Any, Dict, Optional, Union

from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.remote.webelement import WebElement

from .input_device import InputDevice
from .interaction import POINTER, POINTER_KINDS


class PointerInput(InputDevice):
    DEFAULT_MOVE_DURATION = 250

    def __init__(self, kind, name):
        super().__init__()
        if kind not in POINTER_KINDS:
            raise InvalidArgumentException(f"Invalid PointerInput kind '{kind}'")
        self.type = POINTER
        self.kind = kind
        self.name = name

    def create_pointer_move(
        self,
        duration=DEFAULT_MOVE_DURATION,
        x: float = 0,
        y: float = 0,
        origin: Optional[WebElement] = None,
        **kwargs,
    ):
        action = {"type": "pointerMove", "duration": duration, "x": x, "y": y, **kwargs}
        if isinstance(origin, WebElement):
            action["origin"] = {"element-6066-11e4-a52e-4f735466cecf": origin.id}
        elif origin is not None:
            action["origin"] = origin
        self.add_action(self._convert_keys(action))

    def create_pointer_down(self, **kwargs):
        data = {"type": "pointerDown", "duration": 0, **kwargs}
        self.add_action(self._convert_keys(data))

    def create_pointer_up(self, button):
        self.add_action({"type": "pointerUp", "duration": 0, "button": button})

    def create_pointer_cancel(self):
        self.add_action({"type": "pointerCancel"})

    def create_pause(self, pause_duration: Union[int, float] = 0) -> None:
        self.add_action({"type": "pause", "duration": int(pause_duration * 1000)})

    def encode(self):
        return {"type": self.type, "parameters": {"pointerType": self.kind}, "id": self.name, "actions": self.actions}

    def _convert_keys(self, actions: Dict[str, Any]):
        out = {}
        for k, v in actions.items():
            if v is None:
                continue
            if k in ("x", "y"):
                out[k] = int(v)
                continue
            splits = k.split("_")
            new_key = splits[0] + "".join(v.title() for v in splits[1:])
            out[new_key] = v
        return out

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\setters.py ===
# SPDX-License-Identifier: MIT

"""
Commonly used hooks for on_setattr.
"""

from . import _config
from .exceptions import FrozenAttributeError


def pipe(*setters):
    """
    Run all *setters* and return the return value of the last one.

    .. versionadded:: 20.1.0
    """

    def wrapped_pipe(instance, attrib, new_value):
        rv = new_value

        for setter in setters:
            rv = setter(instance, attrib, rv)

        return rv

    return wrapped_pipe


def frozen(_, __, ___):
    """
    Prevent an attribute to be modified.

    .. versionadded:: 20.1.0
    """
    raise FrozenAttributeError


def validate(instance, attrib, new_value):
    """
    Run *attrib*'s validator on *new_value* if it has one.

    .. versionadded:: 20.1.0
    """
    if _config._run_validators is False:
        return new_value

    v = attrib.validator
    if not v:
        return new_value

    v(instance, attrib, new_value)

    return new_value


def convert(instance, attrib, new_value):
    """
    Run *attrib*'s converter -- if it has one -- on *new_value* and return the
    result.

    .. versionadded:: 20.1.0
    """
    c = attrib.converter
    if c:
        # This can be removed once we drop 3.8 and use attrs.Converter instead.
        from ._make import Converter

        if not isinstance(c, Converter):
            return c(new_value)

        return c(new_value, instance, attrib)

    return new_value


# Sentinel for disabling class-wide *on_setattr* hooks for certain attributes.
# Sphinx's autodata stopped working, so the docstring is inlined in the API
# docs.
NO_OP = object()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\logging.py ===
from __future__ import annotations

import logging
import sys
import typing as t

from werkzeug.local import LocalProxy

from .globals import request

if t.TYPE_CHECKING:  # pragma: no cover
    from .sansio.app import App


@LocalProxy
def wsgi_errors_stream() -> t.TextIO:
    """Find the most appropriate error stream for the application. If a request
    is active, log to ``wsgi.errors``, otherwise use ``sys.stderr``.

    If you configure your own :class:`logging.StreamHandler`, you may want to
    use this for the stream. If you are using file or dict configuration and
    can't import this directly, you can refer to it as
    ``ext://flask.logging.wsgi_errors_stream``.
    """
    if request:
        return request.environ["wsgi.errors"]  # type: ignore[no-any-return]

    return sys.stderr


def has_level_handler(logger: logging.Logger) -> bool:
    """Check if there is a handler in the logging chain that will handle the
    given logger's :meth:`effective level <~logging.Logger.getEffectiveLevel>`.
    """
    level = logger.getEffectiveLevel()
    current = logger

    while current:
        if any(handler.level <= level for handler in current.handlers):
            return True

        if not current.propagate:
            break

        current = current.parent  # type: ignore

    return False


#: Log messages to :func:`~flask.logging.wsgi_errors_stream` with the format
#: ``[%(asctime)s] %(levelname)s in %(module)s: %(message)s``.
default_handler = logging.StreamHandler(wsgi_errors_stream)  # type: ignore
default_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
)


def create_logger(app: App) -> logging.Logger:
    """Get the Flask app's logger and configure it if needed.

    The logger name will be the same as
    :attr:`app.import_name <flask.Flask.name>`.

    When :attr:`~flask.Flask.debug` is enabled, set the logger level to
    :data:`logging.DEBUG` if it is not set.

    If there is no handler for the logger's effective level, add a
    :class:`~logging.StreamHandler` for
    :func:`~flask.logging.wsgi_errors_stream` with a basic format.
    """
    logger = logging.getLogger(app.name)

    if app.debug and not logger.level:
        logger.setLevel(logging.DEBUG)

    if not has_level_handler(logger):
        logger.addHandler(default_handler)

    return logger

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_expired_attrs_2_0.py ===
"""
Dict of expired attributes that are discontinued since 2.0 release.
Each item is associated with a migration note.
"""

__expired_attributes__ = {
    "geterrobj": "Use the np.errstate context manager instead.",
    "seterrobj": "Use the np.errstate context manager instead.",
    "cast": "Use `np.asarray(arr, dtype=dtype)` instead.",
    "source": "Use `inspect.getsource` instead.",
    "lookfor":  "Search NumPy's documentation directly.",
    "who": "Use an IDE variable explorer or `locals()` instead.",
    "fastCopyAndTranspose": "Use `arr.T.copy()` instead.",
    "set_numeric_ops":
        "For the general case, use `PyUFunc_ReplaceLoopBySignature`. "
        "For ndarray subclasses, define the ``__array_ufunc__`` method "
        "and override the relevant ufunc.",
    "NINF": "Use `-np.inf` instead.",
    "PINF": "Use `np.inf` instead.",
    "NZERO": "Use `-0.0` instead.",
    "PZERO": "Use `0.0` instead.",
    "add_newdoc":
        "It's still available as `np.lib.add_newdoc`.",
    "add_docstring":
        "It's still available as `np.lib.add_docstring`.",
    "add_newdoc_ufunc":
        "It's an internal function and doesn't have a replacement.",
    "safe_eval": "Use `ast.literal_eval` instead.",
    "float_": "Use `np.float64` instead.",
    "complex_": "Use `np.complex128` instead.",
    "longfloat": "Use `np.longdouble` instead.",
    "singlecomplex": "Use `np.complex64` instead.",
    "cfloat": "Use `np.complex128` instead.",
    "longcomplex": "Use `np.clongdouble` instead.",
    "clongfloat": "Use `np.clongdouble` instead.",
    "string_": "Use `np.bytes_` instead.",
    "unicode_": "Use `np.str_` instead.",
    "Inf": "Use `np.inf` instead.",
    "Infinity": "Use `np.inf` instead.",
    "NaN": "Use `np.nan` instead.",
    "infty": "Use `np.inf` instead.",
    "issctype": "Use `issubclass(rep, np.generic)` instead.",
    "maximum_sctype":
        "Use a specific dtype instead. You should avoid relying "
        "on any implicit mechanism and select the largest dtype of "
        "a kind explicitly in the code.",
    "obj2sctype": "Use `np.dtype(obj).type` instead.",
    "sctype2char": "Use `np.dtype(obj).char` instead.",
    "sctypes": "Access dtypes explicitly instead.",
    "issubsctype": "Use `np.issubdtype` instead.",
    "set_string_function":
        "Use `np.set_printoptions` instead with a formatter for "
        "custom printing of NumPy objects.",
    "asfarray": "Use `np.asarray` with a proper dtype instead.",
    "issubclass_": "Use `issubclass` builtin instead.",
    "tracemalloc_domain": "It's now available from `np.lib`.",
    "mat": "Use `np.asmatrix` instead.",
    "recfromcsv": "Use `np.genfromtxt` with comma delimiter instead.",
    "recfromtxt": "Use `np.genfromtxt` instead.",
    "deprecate": "Emit `DeprecationWarning` with `warnings.warn` directly, "
        "or use `typing.deprecated`.",
    "deprecate_with_doc": "Emit `DeprecationWarning` with `warnings.warn` "
        "directly, or use `typing.deprecated`.",
    "disp": "Use your own printing function instead.",
    "find_common_type":
        "Use `numpy.promote_types` or `numpy.result_type` instead. "
        "To achieve semantics for the `scalar_types` argument, use "
        "`numpy.result_type` and pass the Python values `0`, `0.0`, or `0j`.",
    "round_": "Use `np.round` instead.",
    "get_array_wrap": "",
    "DataSource": "It's still available as `np.lib.npyio.DataSource`.",
    "nbytes": "Use `np.dtype(<dtype>).itemsize` instead.",
    "byte_bounds": "Now it's available under `np.lib.array_utils.byte_bounds`",
    "compare_chararrays":
        "It's still available as `np.char.compare_chararrays`.",
    "format_parser": "It's still available as `np.rec.format_parser`.",
    "alltrue": "Use `np.all` instead.",
    "sometrue": "Use `np.any` instead.",
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_nested_sequence.py ===
"""A module containing the `_NestedSequence` protocol."""

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["_NestedSequence"]

_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class _NestedSequence(Protocol[_T_co]):
    """A protocol for representing nested sequences.

    Warning
    -------
    `_NestedSequence` currently does not work in combination with typevars,
    *e.g.* ``def func(a: _NestedSequnce[T]) -> T: ...``.

    See Also
    --------
    collections.abc.Sequence
        ABCs for read-only and mutable :term:`sequences`.

    Examples
    --------
    .. code-block:: python

        >>> from typing import TYPE_CHECKING
        >>> import numpy as np
        >>> from numpy._typing import _NestedSequence

        >>> def get_dtype(seq: _NestedSequence[float]) -> np.dtype[np.float64]:
        ...     return np.asarray(seq).dtype

        >>> a = get_dtype([1.0])
        >>> b = get_dtype([[1.0]])
        >>> c = get_dtype([[[1.0]]])
        >>> d = get_dtype([[[[1.0]]]])

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
        ...     # note:     b: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
        ...     # note:     c: numpy.dtype[numpy.floating[numpy._typing._64Bit]]
        ...     # note:     d: numpy.dtype[numpy.floating[numpy._typing._64Bit]]

    """

    def __len__(self, /) -> int:
        """Implement ``len(self)``."""
        raise NotImplementedError

    def __getitem__(self, index: int, /) -> "_T_co | _NestedSequence[_T_co]":
        """Implement ``self[x]``."""
        raise NotImplementedError

    def __contains__(self, x: object, /) -> bool:
        """Implement ``x in self``."""
        raise NotImplementedError

    def __iter__(self, /) -> "Iterator[_T_co | _NestedSequence[_T_co]]":
        """Implement ``iter(self)``."""
        raise NotImplementedError

    def __reversed__(self, /) -> "Iterator[_T_co | _NestedSequence[_T_co]]":
        """Implement ``reversed(self)``."""
        raise NotImplementedError

    def count(self, value: Any, /) -> int:
        """Return the number of occurrences of `value`."""
        raise NotImplementedError

    def index(self, value: Any, /) -> int:
        """Return the first index of `value`."""
        raise NotImplementedError

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\RuntimeOptimizations.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RuntimeOptimizations(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RuntimeOptimizations()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRuntimeOptimizations(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def RuntimeOptimizationsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # RuntimeOptimizations
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # mapping from optimizer name to [RuntimeOptimizationRecord]
    # RuntimeOptimizations
    def Records(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.RuntimeOptimizationRecordContainerEntry import RuntimeOptimizationRecordContainerEntry
            obj = RuntimeOptimizationRecordContainerEntry()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RuntimeOptimizations
    def RecordsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RuntimeOptimizations
    def RecordsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def RuntimeOptimizationsStart(builder):
    builder.StartObject(1)

def Start(builder):
    RuntimeOptimizationsStart(builder)

def RuntimeOptimizationsAddRecords(builder, records):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(records), 0)

def AddRecords(builder, records):
    RuntimeOptimizationsAddRecords(builder, records)

def RuntimeOptimizationsStartRecordsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartRecordsVector(builder, numElems: int) -> int:
    return RuntimeOptimizationsStartRecordsVector(builder, numElems)

def RuntimeOptimizationsEnd(builder):
    return builder.EndObject()

def End(builder):
    return RuntimeOptimizationsEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\main.py ===
"""Primary application entrypoint."""

import locale
import logging
import os
import sys
import warnings
from typing import List, Optional

from pip._internal.cli.autocompletion import autocomplete
from pip._internal.cli.main_parser import parse_command
from pip._internal.commands import create_command
from pip._internal.exceptions import PipError
from pip._internal.utils import deprecation

logger = logging.getLogger(__name__)


# Do not import and use main() directly! Using it directly is actively
# discouraged by pip's maintainers. The name, location and behavior of
# this function is subject to change, so calling it directly is not
# portable across different pip versions.

# In addition, running pip in-process is unsupported and unsafe. This is
# elaborated in detail at
# https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program.
# That document also provides suggestions that should work for nearly
# all users that are considering importing and using main() directly.

# However, we know that certain users will still want to invoke pip
# in-process. If you understand and accept the implications of using pip
# in an unsupported manner, the best approach is to use runpy to avoid
# depending on the exact location of this entry point.

# The following example shows how to use runpy to invoke pip in that
# case:
#
#     sys.argv = ["pip", your, args, here]
#     runpy.run_module("pip", run_name="__main__")
#
# Note that this will exit the process after running, unlike a direct
# call to main. As it is not safe to do any processing after calling
# main, this should not be an issue in practice.


def main(args: Optional[List[str]] = None) -> int:
    if args is None:
        args = sys.argv[1:]

    # Suppress the pkg_resources deprecation warning
    # Note - we use a module of .*pkg_resources to cover
    # the normal case (pip._vendor.pkg_resources) and the
    # devendored case (a bare pkg_resources)
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module=".*pkg_resources"
    )

    # Configure our deprecation warnings to be sent through loggers
    deprecation.install_warning_logger()

    autocomplete()

    try:
        cmd_name, cmd_args = parse_command(args)
    except PipError as exc:
        sys.stderr.write(f"ERROR: {exc}")
        sys.stderr.write(os.linesep)
        sys.exit(1)

    # Needed for locale.getpreferredencoding(False) to work
    # in pip._internal.utils.encoding.auto_decode
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error as e:
        # setlocale can apparently crash if locale are uninitialized
        logger.debug("Ignoring error %s when setting locale", e)
    command = create_command(cmd_name, isolated=("--isolated" in cmd_args))

    return command.main(cmd_args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\req\req_dependency_group.py ===
import sys
from typing import Any, Dict, Iterable, Iterator, List, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    from pip._vendor import tomli as tomllib

from pip._vendor.dependency_groups import DependencyGroupResolver

from pip._internal.exceptions import InstallationError


def parse_dependency_groups(groups: List[Tuple[str, str]]) -> List[str]:
    """
    Parse dependency groups data as provided via the CLI, in a `[path:]group` syntax.

    Raises InstallationErrors if anything goes wrong.
    """
    resolvers = _build_resolvers(path for (path, _) in groups)
    return list(_resolve_all_groups(resolvers, groups))


def _resolve_all_groups(
    resolvers: Dict[str, DependencyGroupResolver], groups: List[Tuple[str, str]]
) -> Iterator[str]:
    """
    Run all resolution, converting any error from `DependencyGroupResolver` into
    an InstallationError.
    """
    for path, groupname in groups:
        resolver = resolvers[path]
        try:
            yield from (str(req) for req in resolver.resolve(groupname))
        except (ValueError, TypeError, LookupError) as e:
            raise InstallationError(
                f"[dependency-groups] resolution failed for '{groupname}' "
                f"from '{path}': {e}"
            ) from e


def _build_resolvers(paths: Iterable[str]) -> Dict[str, Any]:
    resolvers = {}
    for path in paths:
        if path in resolvers:
            continue

        pyproject = _load_pyproject(path)
        if "dependency-groups" not in pyproject:
            raise InstallationError(
                f"[dependency-groups] table was missing from '{path}'. "
                "Cannot resolve '--group' option."
            )
        raw_dependency_groups = pyproject["dependency-groups"]
        if not isinstance(raw_dependency_groups, dict):
            raise InstallationError(
                f"[dependency-groups] table was malformed in {path}. "
                "Cannot resolve '--group' option."
            )

        resolvers[path] = DependencyGroupResolver(raw_dependency_groups)
    return resolvers


def _load_pyproject(path: str) -> Dict[str, Any]:
    """
    This helper loads a pyproject.toml as TOML.

    It raises an InstallationError if the operation fails.
    """
    try:
        with open(path, "rb") as fp:
            return tomllib.load(fp)
    except FileNotFoundError:
        raise InstallationError(f"{path} not found. Cannot resolve '--group' option.")
    except tomllib.TOMLDecodeError as e:
        raise InstallationError(f"Error parsing {path}: {e}") from e
    except OSError as e:
        raise InstallationError(f"Error reading {path}: {e}") from e

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\compat.py ===
"""Stuff that differs in different Python versions and platform
distributions."""

import importlib.resources
import logging
import os
import sys
from typing import IO

__all__ = ["get_path_uid", "stdlib_pkgs", "WINDOWS"]


logger = logging.getLogger(__name__)


def has_tls() -> bool:
    try:
        import _ssl  # noqa: F401  # ignore unused

        return True
    except ImportError:
        pass

    from pip._vendor.urllib3.util import IS_PYOPENSSL

    return IS_PYOPENSSL


def get_path_uid(path: str) -> int:
    """
    Return path's uid.

    Does not follow symlinks:
        https://github.com/pypa/pip/pull/935#discussion_r5307003

    Placed this function in compat due to differences on AIX and
    Jython, that should eventually go away.

    :raises OSError: When path is a symlink or can't be read.
    """
    if hasattr(os, "O_NOFOLLOW"):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        file_uid = os.fstat(fd).st_uid
        os.close(fd)
    else:  # AIX and Jython
        # WARNING: time of check vulnerability, but best we can do w/o NOFOLLOW
        if not os.path.islink(path):
            # older versions of Jython don't have `os.fstat`
            file_uid = os.stat(path).st_uid
        else:
            # raise OSError for parity with os.O_NOFOLLOW above
            raise OSError(f"{path} is a symlink; Will not return uid for symlinks")
    return file_uid


# The importlib.resources.open_text function was deprecated in 3.11 with suggested
# replacement we use below.
if sys.version_info < (3, 11):
    open_text_resource = importlib.resources.open_text
else:

    def open_text_resource(
        package: str, resource: str, encoding: str = "utf-8", errors: str = "strict"
    ) -> IO[str]:
        return (importlib.resources.files(package) / resource).open(
            "r", encoding=encoding, errors=errors
        )


# packages in the stdlib that may have installation metadata, but should not be
# considered 'installed'.  this theoretically could be determined based on
# dist.location (py27:`sysconfig.get_paths()['stdlib']`,
# py26:sysconfig.get_config_vars('LIBDEST')), but fear platform variation may
# make this ineffective, so hard-coding
stdlib_pkgs = {"python", "wsgiref", "argparse"}


# windows detection, covers cpython and ironpython
WINDOWS = sys.platform.startswith("win") or (sys.platform == "cli" and os.name == "nt")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\__init__.py ===
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

from .chrome.options import Options as ChromeOptions  # noqa
from .chrome.service import Service as ChromeService  # noqa
from .chrome.webdriver import WebDriver as Chrome  # noqa
from .common.action_chains import ActionChains  # noqa
from .common.desired_capabilities import DesiredCapabilities  # noqa
from .common.keys import Keys  # noqa
from .common.proxy import Proxy  # noqa
from .edge.options import Options as EdgeOptions  # noqa
from .edge.service import Service as EdgeService  # noqa
from .edge.webdriver import WebDriver as ChromiumEdge  # noqa
from .edge.webdriver import WebDriver as Edge  # noqa
from .firefox.firefox_profile import FirefoxProfile  # noqa
from .firefox.options import Options as FirefoxOptions  # noqa
from .firefox.service import Service as FirefoxService  # noqa
from .firefox.webdriver import WebDriver as Firefox  # noqa
from .ie.options import Options as IeOptions  # noqa
from .ie.service import Service as IeService  # noqa
from .ie.webdriver import WebDriver as Ie  # noqa
from .remote.webdriver import WebDriver as Remote  # noqa
from .safari.options import Options as SafariOptions
from .safari.service import Service as SafariService  # noqa
from .safari.webdriver import WebDriver as Safari  # noqa
from .webkitgtk.options import Options as WebKitGTKOptions  # noqa
from .webkitgtk.service import Service as WebKitGTKService  # noqa
from .webkitgtk.webdriver import WebDriver as WebKitGTK  # noqa
from .wpewebkit.options import Options as WPEWebKitOptions  # noqa
from .wpewebkit.service import Service as WPEWebKitService  # noqa
from .wpewebkit.webdriver import WebDriver as WPEWebKit  # noqa

__version__ = "4.33.0"

# We need an explicit __all__ because the above won't otherwise be exported.
__all__ = [
    "ActionChains",
    "Chrome",
    "ChromeOptions",
    "ChromeService",
    "ChromiumEdge",
    "DesiredCapabilities",
    "Edge",
    "EdgeOptions",
    "EdgeService",
    "Firefox",
    "FirefoxOptions",
    "FirefoxProfile",
    "FirefoxService",
    "Ie",
    "IeOptions",
    "IeService",
    "Keys",
    "Proxy",
    "Remote",
    "Safari",
    "SafariOptions",
    "SafariService",
    "WPEWebKit",
    "WPEWebKitOptions",
    "WPEWebKitService",
    "WebKitGTK",
    "WebKitGTKOptions",
    "WebKitGTKService",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\log.py ===
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

from dataclasses import dataclass
from typing import List


class LogEntryAdded:
    event_class = "log.entryAdded"

    @classmethod
    def from_json(cls, json):
        if json["type"] == "console":
            return ConsoleLogEntry.from_json(json)
        elif json["type"] == "javascript":
            return JavaScriptLogEntry.from_json(json)


@dataclass
class ConsoleLogEntry:
    level: str
    text: str
    timestamp: str
    method: str
    args: List[dict]
    type_: str

    @classmethod
    def from_json(cls, json):
        return cls(
            level=json["level"],
            text=json["text"],
            timestamp=json["timestamp"],
            method=json["method"],
            args=json["args"],
            type_=json["type"],
        )


@dataclass
class JavaScriptLogEntry:
    level: str
    text: str
    timestamp: str
    stacktrace: dict
    type_: str

    @classmethod
    def from_json(cls, json):
        return cls(
            level=json["level"],
            text=json["text"],
            timestamp=json["timestamp"],
            stacktrace=json["stackTrace"],
            type_=json["type"],
        )


class LogLevel:
    """Represents log level."""

    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\safari\service.py ===
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


from typing import List, Mapping, Optional

from selenium.webdriver.common import service


class Service(service.Service):
    """A Service class that is responsible for the starting and stopping of
    `safaridriver`  This is only supported on MAC OSX.

    :param executable_path: install path of the safaridriver executable, defaults to `/usr/bin/safaridriver`.
    :param port: Port for the service to run on, defaults to 0 where the operating system will decide.
    :param service_args: (Optional) List of args to be passed to the subprocess when launching the executable.
    :param env: (Optional) Mapping of environment variables for the new process, defaults to `os.environ`.
    :param enable_logging: (Optional) Enable logging of the service. Logs can be located at
        `~/Library/Logs/com.apple.WebDriver/`
    :param driver_path_env_key: (Optional) Environment variable to use to get the path to the driver executable.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        port: int = 0,
        service_args: Optional[List[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        reuse_service=False,
        enable_logging: bool = False,
        driver_path_env_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.service_args = service_args or []
        driver_path_env_key = driver_path_env_key or "SE_SAFARIDRIVER"

        if enable_logging:
            self.service_args.append("--diagnose")

        self.reuse_service = reuse_service
        super().__init__(
            executable_path=executable_path,
            port=port,
            env=env,
            driver_path_env_key=driver_path_env_key,
            **kwargs,
        )

    def command_line_args(self) -> List[str]:
        return ["-p", f"{self.port}"] + self.service_args

    @property
    def service_url(self) -> str:
        """Gets the url of the SafariDriver Service."""
        return f"http://localhost:{self.port}"

    @property
    def reuse_service(self) -> bool:
        return self._reuse_service

    @reuse_service.setter
    def reuse_service(self, reuse: bool) -> None:
        if not isinstance(reuse, bool):
            raise TypeError("reuse must be a boolean")
        self._reuse_service = reuse

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\scipy_nodes.py ===
from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import cos, sin


def _cosm1(x, *, evaluate=True):
    return Add(cos(x, evaluate=evaluate), -S.One, evaluate=evaluate)


class cosm1(Function):
    """ Minus one plus cosine of x, i.e. cos(x) - 1. For use when x is close to zero.

    Helper class for use with e.g. scipy.special.cosm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosm1.html
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return -sin(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return _cosm1(x)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(cos).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        x, = self.args
        candidate = _cosm1(x.simplify(**kwargs))
        if candidate != _cosm1(x, evaluate=False):
            return candidate
        else:
            return cosm1(x)


def _powm1(x, y, *, evaluate=True):
    return Add(Pow(x, y, evaluate=evaluate), -S.One, evaluate=evaluate)


class powm1(Function):
    """ Minus one plus x to the power of y, i.e. x**y - 1. For use when x is close to one or y is close to zero.

    Helper class for use with e.g. scipy.special.powm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html
    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], self.args[1])*self.args[1]/self.args[0]
        elif argindex == 2:
            return log(self.args[0])*Pow(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, x, y, **kwargs):
        return _powm1(x, y)

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(Pow).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        x, y = self.args
        candidate = _powm1(x.simplify(**kwargs), y.simplify(**kwargs))
        if candidate != _powm1(x, y, evaluate=False):
            return candidate
        else:
            return powm1(x, y)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\ast_parser.py ===
"""
This module implements the functionality to take any Python expression as a
string and fix all numbers and other things before evaluating it,
thus

1/2

returns

Integer(1)/Integer(2)

We use the ast module for this. It is well documented at docs.python.org.

Some tips to understand how this works: use dump() to get a nice
representation of any node. Then write a string of what you want to get,
e.g. "Integer(1)", parse it, dump it and you'll see that you need to do
"Call(Name('Integer', Load()), [node], [], None, None)". You do not need
to bother with lineno and col_offset, just call fix_missing_locations()
before returning the node.
"""

from sympy.core.basic import Basic
from sympy.core.sympify import SympifyError

from ast import parse, NodeTransformer, Call, Name, Load, \
    fix_missing_locations, Constant, Tuple

class Transform(NodeTransformer):

    def __init__(self, local_dict, global_dict):
        NodeTransformer.__init__(self)
        self.local_dict = local_dict
        self.global_dict = global_dict

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return fix_missing_locations(Call(func=Name('Integer', Load()),
                    args=[node], keywords=[]))
        elif isinstance(node.value, float):
            return fix_missing_locations(Call(func=Name('Float', Load()),
                    args=[node], keywords=[]))
        return node

    def visit_Name(self, node):
        if node.id in self.local_dict:
            return node
        elif node.id in self.global_dict:
            name_obj = self.global_dict[node.id]

            if isinstance(name_obj, (Basic, type)) or callable(name_obj):
                return node
        elif node.id in ['True', 'False']:
            return node
        return fix_missing_locations(Call(func=Name('Symbol', Load()),
                args=[Constant(node.id)], keywords=[]))

    def visit_Lambda(self, node):
        args = [self.visit(arg) for arg in node.args.args]
        body = self.visit(node.body)
        n = Call(func=Name('Lambda', Load()),
            args=[Tuple(args, Load()), body], keywords=[])
        return fix_missing_locations(n)

def parse_expr(s, local_dict):
    """
    Converts the string "s" to a SymPy expression, in local_dict.

    It converts all numbers to Integers before feeding it to Python and
    automatically creates Symbols.
    """
    global_dict = {}
    exec('from sympy import *', global_dict)
    try:
        a = parse(s.strip(), mode="eval")
    except SyntaxError:
        raise SympifyError("Cannot parse %s." % repr(s))
    a = Transform(local_dict, global_dict).visit(a)
    e = compile(a, "<string>", "eval")
    return eval(e, global_dict, local_dict)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\handlers\add.py ===
from sympy.core.numbers import oo, Infinity, NegativeInfinity
from sympy.core.singleton import S
from sympy.core import Basic, Expr
from sympy.multipledispatch import Dispatcher
from sympy.sets import Interval, FiniteSet



# XXX: The functions in this module are clearly not tested and are broken in a
# number of ways.

_set_add = Dispatcher('_set_add')
_set_sub = Dispatcher('_set_sub')


@_set_add.register(Basic, Basic)
def _(x, y):
    return None


@_set_add.register(Expr, Expr)
def _(x, y):
    return x+y


@_set_add.register(Interval, Interval)
def _(x, y):
    """
    Additions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    return Interval(x.start + y.start, x.end + y.end,
                    x.left_open or y.left_open, x.right_open or y.right_open)


@_set_add.register(Interval, Infinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet({S.Infinity})

@_set_add.register(Interval, NegativeInfinity)
def _(x, y):
    if x.end is S.Infinity:
        return Interval(-oo, oo)
    return FiniteSet({S.NegativeInfinity})


@_set_sub.register(Basic, Basic)
def _(x, y):
    return None


@_set_sub.register(Expr, Expr)
def _(x, y):
    return x-y


@_set_sub.register(Interval, Interval)
def _(x, y):
    """
    Subtractions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    return Interval(x.start - y.end, x.end - y.start,
                    x.left_open or y.right_open, x.right_open or y.left_open)


@_set_sub.register(Interval, Infinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet(-oo)

@_set_sub.register(Interval, NegativeInfinity)
def _(x, y):
    if x.start is S.NegativeInfinity:
        return Interval(-oo, oo)
    return FiniteSet(-oo)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\handlers\mul.py ===
from sympy.core import Basic, Expr
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.multipledispatch import Dispatcher
from sympy.sets.setexpr import set_mul
from sympy.sets.sets import Interval, Set


_x, _y = symbols("x y")


_set_mul = Dispatcher('_set_mul')
_set_div = Dispatcher('_set_div')


@_set_mul.register(Basic, Basic)
def _(x, y):
    return None

@_set_mul.register(Set, Set)
def _(x, y):
    return None

@_set_mul.register(Expr, Expr)
def _(x, y):
    return x*y

@_set_mul.register(Interval, Interval)
def _(x, y):
    """
    Multiplications in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    # TODO: some intervals containing 0 and oo will fail as 0*oo returns nan.
    comvals = (
        (x.start * y.start, bool(x.left_open or y.left_open)),
        (x.start * y.end, bool(x.left_open or y.right_open)),
        (x.end * y.start, bool(x.right_open or y.left_open)),
        (x.end * y.end, bool(x.right_open or y.right_open)),
    )
    # TODO: handle symbolic intervals
    minval, minopen = min(comvals)
    maxval, maxopen = max(comvals)
    return Interval(
        minval,
        maxval,
        minopen,
        maxopen
    )

@_set_div.register(Basic, Basic)
def _(x, y):
    return None

@_set_div.register(Expr, Expr)
def _(x, y):
    return x/y

@_set_div.register(Set, Set)
def _(x, y):
    return None

@_set_div.register(Interval, Interval)
def _(x, y):
    """
    Divisions in interval arithmetic
    https://en.wikipedia.org/wiki/Interval_arithmetic
    """
    if (y.start*y.end).is_negative:
        return Interval(-oo, oo)
    if y.start == 0:
        s2 = oo
    else:
        s2 = 1/y.start
    if y.end == 0:
        s1 = -oo
    else:
        s1 = 1/y.end
    return set_mul(x, Interval(s1, s2, y.right_open, y.left_open))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\type_tests\nursery_start.py ===
"""Test variadic generic typing for Nursery.start[_soon]()."""

from typing import TYPE_CHECKING

from trio import TASK_STATUS_IGNORED, Nursery, TaskStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def task_0() -> None: ...


async def task_1a(value: int) -> None: ...


async def task_1b(value: str) -> None: ...


async def task_2a(a: int, b: str) -> None: ...


async def task_2b(a: str, b: int) -> None: ...


async def task_2c(a: str, b: int, optional: bool = False) -> None: ...


async def task_requires_kw(a: int, *, b: bool) -> None: ...


async def task_startable_1(
    a: str,
    *,
    task_status: TaskStatus[bool] = TASK_STATUS_IGNORED,
) -> None: ...


async def task_startable_2(
    a: str,
    b: float,
    *,
    task_status: TaskStatus[bool] = TASK_STATUS_IGNORED,
) -> None: ...


async def task_requires_start(*, task_status: TaskStatus[str]) -> None:
    """Check a function requiring start() to be used."""


async def task_pos_or_kw(value: str, task_status: TaskStatus[int]) -> None:
    """Check a function which doesn't use the *-syntax works."""


def check_start_soon(nursery: Nursery) -> None:
    """start_soon() functionality."""
    nursery.start_soon(task_0)
    nursery.start_soon(task_1a)  # type: ignore
    nursery.start_soon(task_2b)  # type: ignore

    nursery.start_soon(task_0, 45)  # type: ignore
    nursery.start_soon(task_1a, 32)
    nursery.start_soon(task_1b, 32)  # type: ignore
    nursery.start_soon(task_1a, "abc")  # type: ignore
    nursery.start_soon(task_1b, "abc")

    nursery.start_soon(task_2b, "abc")  # type: ignore
    nursery.start_soon(task_2a, 38, "46")
    nursery.start_soon(task_2c, "abc", 12, True)

    nursery.start_soon(task_2c, "abc", 12)
    task_2c_cast: Callable[[str, int], Awaitable[object]] = (
        task_2c  # The assignment makes it work.
    )
    nursery.start_soon(task_2c_cast, "abc", 12)

    nursery.start_soon(task_requires_kw, 12, True)  # type: ignore
    # Tasks following the start() API can be made to work.
    nursery.start_soon(task_startable_1, "cdf")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\__init__.py ===
from wtforms import validators
from wtforms import widgets
from wtforms.fields.choices import RadioField
from wtforms.fields.choices import SelectField
from wtforms.fields.choices import SelectFieldBase
from wtforms.fields.choices import SelectMultipleField
from wtforms.fields.core import Field
from wtforms.fields.core import Flags
from wtforms.fields.core import Label
from wtforms.fields.datetime import DateField
from wtforms.fields.datetime import DateTimeField
from wtforms.fields.datetime import DateTimeLocalField
from wtforms.fields.datetime import MonthField
from wtforms.fields.datetime import TimeField
from wtforms.fields.datetime import WeekField
from wtforms.fields.form import FormField
from wtforms.fields.list import FieldList
from wtforms.fields.numeric import DecimalField
from wtforms.fields.numeric import DecimalRangeField
from wtforms.fields.numeric import FloatField
from wtforms.fields.numeric import IntegerField
from wtforms.fields.numeric import IntegerRangeField
from wtforms.fields.simple import BooleanField
from wtforms.fields.simple import ColorField
from wtforms.fields.simple import EmailField
from wtforms.fields.simple import FileField
from wtforms.fields.simple import HiddenField
from wtforms.fields.simple import MultipleFileField
from wtforms.fields.simple import PasswordField
from wtforms.fields.simple import SearchField
from wtforms.fields.simple import StringField
from wtforms.fields.simple import SubmitField
from wtforms.fields.simple import TelField
from wtforms.fields.simple import TextAreaField
from wtforms.fields.simple import URLField
from wtforms.form import Form
from wtforms.validators import ValidationError

__version__ = "3.2.1"

__all__ = [
    "validators",
    "widgets",
    "Form",
    "ValidationError",
    "SelectField",
    "SelectMultipleField",
    "SelectFieldBase",
    "RadioField",
    "Field",
    "Flags",
    "Label",
    "DateTimeField",
    "DateField",
    "TimeField",
    "MonthField",
    "DateTimeLocalField",
    "WeekField",
    "FormField",
    "FieldList",
    "IntegerField",
    "DecimalField",
    "FloatField",
    "IntegerRangeField",
    "DecimalRangeField",
    "BooleanField",
    "TextAreaField",
    "PasswordField",
    "FileField",
    "MultipleFileField",
    "HiddenField",
    "SearchField",
    "SubmitField",
    "StringField",
    "TelField",
    "URLField",
    "EmailField",
    "ColorField",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\tests\test_systems.py ===
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
                                       ODEOrderError, ODENonlinearError, _simpsol,
                                       _is_commutative_anti_derivative, linodesolve,
                                       canonical_odes, dsolve_system, _component_division,
                                       _eqs2dict, _dict2graph)
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import raises, slow, tooslow, XFAIL


C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = symbols('C0:11')
x = symbols('x')
f = Function('f')
g = Function('g')
h = Function('h')


def test_linear_ode_to_matrix():
    f, g, h = symbols("f, g, h", cls=Function)
    t = Symbol("t")
    funcs = [f(t), g(t), h(t)]
    f1 = f(t).diff(t)
    g1 = g(t).diff(t)
    h1 = h(t).diff(t)
    f2 = f(t).diff(t, 2)
    g2 = g(t).diff(t, 2)
    h2 = h(t).diff(t, 2)

    eqs_1 = [Eq(f1, g(t)), Eq(g1, f(t))]
    sol_1 = ([Matrix([[1, 0], [0, 1]]), Matrix([[ 0, 1], [1,  0]])], Matrix([[0],[0]]))
    assert linear_ode_to_matrix(eqs_1, funcs[:-1], t, 1) == sol_1

    eqs_2 = [Eq(f1, f(t) + 2*g(t)), Eq(g1, h(t)), Eq(h1, g(t) + h(t) + f(t))]
    sol_2 = ([Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), Matrix([[1, 2,  0], [ 0,  0, 1], [1, 1, 1]])],
             Matrix([[0], [0], [0]]))
    assert linear_ode_to_matrix(eqs_2, funcs, t, 1) == sol_2

    eqs_3 = [Eq(2*f1 + 3*h1, f(t) + g(t)), Eq(4*h1 + 5*g1, f(t) + h(t)), Eq(5*f1 + 4*g1, g(t) + h(t))]
    sol_3 = ([Matrix([[2, 0, 3], [0, 5, 4], [5, 4, 0]]), Matrix([[1, 1,  0], [1,  0, 1], [0, 1, 1]])],
             Matrix([[0], [0], [0]]))
    assert linear_ode_to_matrix(eqs_3, funcs, t, 1) == sol_3

    eqs_4 = [Eq(f2 + h(t), f1 + g(t)), Eq(2*h2 + g2 + g1 + g(t), 0), Eq(3*h1, 4)]
    sol_4 = ([Matrix([[1, 0, 0], [0, 1, 2], [0, 0, 0]]), Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -3]]),
              Matrix([[0, 1, -1], [0,  -1, 0], [0, 0, 0]])], Matrix([[0], [0], [4]]))
    assert linear_ode_to_matrix(eqs_4, funcs, t, 2) == sol_4

    eqs_5 = [Eq(f2, g(t)), Eq(f1 + g1, f(t))]
    raises(ODEOrderError, lambda: linear_ode_to_matrix(eqs_5, funcs[:-1], t, 1))

    eqs_6 = [Eq(f1, f(t)**2), Eq(g1, f(t) + g(t))]
    raises(ODENonlinearError, lambda: linear_ode_to_matrix(eqs_6, funcs[:-1], t, 1))


def test__classify_linear_system():
    x, y, z, w = symbols('x, y, z, w', cls=Function)
    t, k, l = symbols('t k l')
    x1 = diff(x(t), t)
    y1 = diff(y(t), t)
    z1 = diff(z(t), t)
    w1 = diff(w(t), t)
    x2 = diff(x(t), t, t)
    y2 = diff(y(t), t, t)
    funcs = [x(t), y(t)]
    funcs_2 = funcs + [z(t), w(t)]

    eqs_1 = (5 * x1 + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * t * x(t) + 3 * y(t) + t))
    assert _classify_linear_system(eqs_1, funcs, t) is None

    eqs_2 = (5 * (x1**2) + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * t * x(t) + 3 * y(t) + t))
    sol2 = {'is_implicit': True,
            'canon_eqs': [[Eq(Derivative(x(t), t), -sqrt(-12*x(t)/5 + 6*y(t)/5)),
            Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)],
            [Eq(Derivative(x(t), t), sqrt(-12*x(t)/5 + 6*y(t)/5)),
            Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)]]}
    assert _classify_linear_system(eqs_2, funcs, t) == sol2

    eqs_2_1 = [Eq(Derivative(x(t), t), -sqrt(-12*x(t)/5 + 6*y(t)/5)),
            Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)]
    assert _classify_linear_system(eqs_2_1, funcs, t) is None

    eqs_2_2 = [Eq(Derivative(x(t), t), sqrt(-12*x(t)/5 + 6*y(t)/5)),
            Eq(Derivative(y(t), t), 11*t*x(t)/2 - t/2 - 3*y(t)/2)]
    assert _classify_linear_system(eqs_2_2, funcs, t) is None

    eqs_3 = (5 * x1 + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * x(t) + 3 * y(t)), (5 * w1 + z(t)), (z1 + w(t)))
    answer_3 = {'no_of_equation': 4,
     'eq': (12*x(t) - 6*y(t) + 5*Derivative(x(t), t),
      -11*x(t) + 3*y(t) + 2*Derivative(y(t), t),
      z(t) + 5*Derivative(w(t), t),
      w(t) + Derivative(z(t), t)),
     'func': [x(t), y(t), z(t), w(t)],
     'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1},
     'is_linear': True,
     'is_constant': True,
     'is_homogeneous': True,
     'func_coeff': -Matrix([
     [Rational(12, 5), Rational(-6, 5), 0, 0],
     [Rational(-11, 2),  Rational(3, 2),   0, 0],
     [0, 0, 0, 1],
     [0, 0, Rational(1, 5), 0]]),
     'type_of_equation': 'type1',
     'is_general': True}
    assert _classify_linear_system(eqs_3, funcs_2, t) == answer_3

    eqs_4 = (5 * x1 + 12 * x(t) - 6 * (y(t)), (2 * y1 - 11 * x(t) + 3 * y(t)), (z1 - w(t)), (w1 - z(t)))
    answer_4 = {'no_of_equation': 4,
     'eq': (12 * x(t) - 6 * y(t) + 5 * Derivative(x(t), t),
            -11 * x(t) + 3 * y(t) + 2 * Derivative(y(t), t),
            -w(t) + Derivative(z(t), t),
            -z(t) + Derivative(w(t), t)),
     'func': [x(t), y(t), z(t), w(t)],
     'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1},
     'is_linear': True,
     'is_constant': True,
     'is_homogeneous': True,
     'func_coeff': -Matrix([
         [Rational(12, 5), Rational(-6, 5), 0, 0],
         [Rational(-11, 2), Rational(3, 2), 0, 0],
         [0, 0, 0, -1],
         [0, 0, -1, 0]]),
     'type_of_equation': 'type1',
     'is_general': True}
    assert _classify_linear_system(eqs_4, funcs_2, t) == answer_4

    eqs_5 = (5*x1 + 12*x(t) - 6*(y(t)) + x2, (2*y1 - 11*x(t) + 3*y(t)), (z1 - w(t)), (w1 - z(t)))
    answer_5 = {'no_of_equation': 4, 'eq': (12*x(t) - 6*y(t) + 5*Derivative(x(t), t) + Derivative(x(t), (t, 2)),
                -11*x(t) + 3*y(t) + 2*Derivative(y(t), t), -w(t) + Derivative(z(t), t), -z(t) + Derivative(w(t),
                t)), 'func': [x(t), y(t), z(t), w(t)], 'order': {x(t): 2, y(t): 1, z(t): 1, w(t): 1}, 'is_linear':
                True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type0', 'is_higher_order': True}
    assert _classify_linear_system(eqs_5, funcs_2, t) == answer_5

    eqs_6 = (Eq(x1, 3*y(t) - 11*z(t)), Eq(y1, 7*z(t) - 3*x(t)), Eq(z1, 11*x(t) - 7*y(t)))
    answer_6 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)), Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),
            Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1},
            'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
            'func_coeff': -Matrix([
                         [  0, -3, 11],
                         [  3,  0, -7],
                         [-11,  7,  0]]),
            'type_of_equation': 'type1', 'is_general': True}

    assert _classify_linear_system(eqs_6, funcs_2[:-1], t) == answer_6

    eqs_7 = (Eq(x1, y(t)), Eq(y1, x(t)))
    answer_7 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t))),
                'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True,
                'is_homogeneous': True, 'func_coeff': -Matrix([
                                                        [ 0, -1],
                                                        [-1,  0]]),
                'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_7, funcs, t) == answer_7

    eqs_8 = (Eq(x1, 21*x(t)), Eq(y1, 17*x(t) + 3*y(t)), Eq(z1, 5*x(t) + 7*y(t) + 9*z(t)))
    answer_8 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 21*x(t)), Eq(Derivative(y(t), t), 17*x(t) + 3*y(t)),
            Eq(Derivative(z(t), t), 5*x(t) + 7*y(t) + 9*z(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1},
            'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
            'func_coeff': -Matrix([
                            [-21,  0,  0],
                            [-17, -3,  0],
                            [ -5, -7, -9]]),
            'type_of_equation': 'type1', 'is_general': True}

    assert _classify_linear_system(eqs_8, funcs_2[:-1], t) == answer_8

    eqs_9 = (Eq(x1, 4*x(t) + 5*y(t) + 2*z(t)), Eq(y1, x(t) + 13*y(t) + 9*z(t)), Eq(z1, 32*x(t) + 41*y(t) + 11*z(t)))
    answer_9 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 4*x(t) + 5*y(t) + 2*z(t)),
                Eq(Derivative(y(t), t), x(t) + 13*y(t) + 9*z(t)), Eq(Derivative(z(t), t), 32*x(t) + 41*y(t) + 11*z(t))),
                'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True,
                'is_constant': True, 'is_homogeneous': True,
                'func_coeff': -Matrix([
                            [ -4,  -5,  -2],
                            [ -1, -13,  -9],
                            [-32, -41, -11]]),
                'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_9, funcs_2[:-1], t) == answer_9

    eqs_10 = (Eq(3*x1, 4*5*(y(t) - z(t))), Eq(4*y1, 3*5*(z(t) - x(t))), Eq(5*z1, 3*4*(x(t) - y(t))))
    answer_10 = {'no_of_equation': 3, 'eq': (Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),
                Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)), Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))),
                'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True,
                'is_constant': True, 'is_homogeneous': True,
                'func_coeff': -Matrix([
                                [  0, Rational(-20, 3),  Rational(20, 3)],
                                [Rational(15, 4),     0, Rational(-15, 4)],
                                [Rational(-12, 5), Rational(12, 5),  0]]),
                'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eqs_10, funcs_2[:-1], t) == answer_10

    eq11 = (Eq(x1, 3*y(t) - 11*z(t)), Eq(y1, 7*z(t) - 3*x(t)), Eq(z1, 11*x(t) - 7*y(t)))
    sol11 = {'no_of_equation': 3, 'eq': (Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)), Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),
            Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))), 'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1},
            'is_linear': True, 'is_constant': True, 'is_homogeneous': True, 'func_coeff': -Matrix([
            [  0, -3, 11], [  3,  0, -7], [-11,  7,  0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq11, funcs_2[:-1], t) == sol11

    eq12 = (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t)))
    sol12 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), y(t)), Eq(Derivative(y(t), t), x(t))),
             'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True,
             'is_homogeneous': True, 'func_coeff': -Matrix([
            [0, -1],
            [-1, 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq12, [x(t), y(t)], t) == sol12

    eq13 = (Eq(Derivative(x(t), t), 21*x(t)), Eq(Derivative(y(t), t), 17*x(t) + 3*y(t)),
            Eq(Derivative(z(t), t), 5*x(t) + 7*y(t) + 9*z(t)))
    sol13 = {'no_of_equation': 3, 'eq': (
    Eq(Derivative(x(t), t), 21 * x(t)), Eq(Derivative(y(t), t), 17 * x(t) + 3 * y(t)),
    Eq(Derivative(z(t), t), 5 * x(t) + 7 * y(t) + 9 * z(t))), 'func': [x(t), y(t), z(t)],
             'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
             'func_coeff': -Matrix([
                 [-21, 0, 0],
                 [-17, -3, 0],
                 [-5, -7, -9]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq13, [x(t), y(t), z(t)], t) == sol13

    eq14 = (
        Eq(Derivative(x(t), t), 4*x(t) + 5*y(t) + 2*z(t)), Eq(Derivative(y(t), t), x(t) + 13*y(t) + 9*z(t)),
        Eq(Derivative(z(t), t), 32*x(t) + 41*y(t) + 11*z(t)))
    sol14 = {'no_of_equation': 3, 'eq': (
    Eq(Derivative(x(t), t), 4 * x(t) + 5 * y(t) + 2 * z(t)), Eq(Derivative(y(t), t), x(t) + 13 * y(t) + 9 * z(t)),
    Eq(Derivative(z(t), t), 32 * x(t) + 41 * y(t) + 11 * z(t))), 'func': [x(t), y(t), z(t)],
             'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
             'func_coeff': -Matrix([
                 [-4, -5, -2],
                 [-1, -13, -9],
                 [-32, -41, -11]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq14, [x(t), y(t), z(t)], t) == sol14

    eq15 = (Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)), Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),
            Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t)))
    sol15 = {'no_of_equation': 3, 'eq': (
    Eq(3 * Derivative(x(t), t), 20 * y(t) - 20 * z(t)), Eq(4 * Derivative(y(t), t), -15 * x(t) + 15 * z(t)),
    Eq(5 * Derivative(z(t), t), 12 * x(t) - 12 * y(t))), 'func': [x(t), y(t), z(t)],
             'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': True,
             'func_coeff': -Matrix([
                 [0, Rational(-20, 3), Rational(20, 3)],
                 [Rational(15, 4), 0, Rational(-15, 4)],
                 [Rational(-12, 5), Rational(12, 5), 0]]), 'type_of_equation': 'type1', 'is_general': True}
    assert _classify_linear_system(eq15, [x(t), y(t), z(t)], t) == sol15

    # Constant coefficient homogeneous ODEs
    eq1 = (Eq(diff(x(t), t), x(t) + y(t) + 9), Eq(diff(y(t), t), 2*x(t) + 5*y(t) + 23))
    sol1 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), x(t) + y(t) + 9),
        Eq(Derivative(y(t), t), 2*x(t) + 5*y(t) + 23)), 'func': [x(t), y(t)],
        'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': True, 'is_homogeneous': False, 'is_general': True,
        'func_coeff': -Matrix([[-1, -1], [-2, -5]]), 'rhs': Matrix([[ 9], [23]]), 'type_of_equation': 'type2'}
    assert _classify_linear_system(eq1, funcs, t) == sol1

    # Non constant coefficient homogeneous ODEs
    eq1 = (Eq(diff(x(t), t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t), t), 2*x(t) + 5*t*y(t)))
    sol1 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), t), 5*t*x(t) + 2*y(t)), Eq(Derivative(y(t), t), 5*t*y(t) + 2*x(t))),
            'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True, 'is_constant': False,
            'is_homogeneous': True, 'func_coeff': -Matrix([ [-5*t,   -2], [  -2, -5*t]]), 'commutative_antiderivative': Matrix([
            [5*t**2/2,      2*t], [     2*t, 5*t**2/2]]), 'type_of_equation': 'type3', 'is_general': True}
    assert _classify_linear_system(eq1, funcs, t) == sol1

    # Non constant coefficient non-homogeneous ODEs
    eq1 = [Eq(x1, x(t) + t*y(t) + t), Eq(y1, t*x(t) + y(t))]
    sol1 = {'no_of_equation': 2, 'eq': [Eq(Derivative(x(t), t), t*y(t) + t + x(t)), Eq(Derivative(y(t), t),
            t*x(t) + y(t))], 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True,
            'is_constant': False, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([ [-1, -t],
            [-t, -1]]), 'commutative_antiderivative': Matrix([ [     t, t**2/2], [t**2/2,      t]]), 'rhs':
            Matrix([ [t], [0]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq1, funcs, t) == sol1

    eq2 = [Eq(x1, t*x(t) + t*y(t) + t), Eq(y1, t*x(t) + t*y(t) + cos(t))]
    sol2 = {'no_of_equation': 2, 'eq': [Eq(Derivative(x(t), t), t*x(t) + t*y(t) + t), Eq(Derivative(y(t), t),
            t*x(t) + t*y(t) + cos(t))], 'func': [x(t), y(t)], 'order': {x(t): 1, y(t): 1}, 'is_linear': True,
            'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([ [     t], [cos(t)]]), 'func_coeff':
            Matrix([ [t, t], [t, t]]), 'is_constant': False, 'type_of_equation': 'type4',
            'commutative_antiderivative': Matrix([ [t**2/2, t**2/2], [t**2/2, t**2/2]])}
    assert _classify_linear_system(eq2, funcs, t) == sol2

    eq3 = [Eq(x1, t*(x(t) + y(t) + z(t) + 1)), Eq(y1, t*(x(t) + y(t) + z(t))), Eq(z1, t*(x(t) + y(t) + z(t)))]
    sol3 = {'no_of_equation': 3, 'eq': [Eq(Derivative(x(t), t), t*(x(t) + y(t) + z(t) + 1)),
            Eq(Derivative(y(t), t), t*(x(t) + y(t) + z(t))), Eq(Derivative(z(t), t), t*(x(t) + y(t) + z(t)))],
            'func': [x(t), y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant':
            False, 'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([ [-t, -t, -t], [-t, -t,
            -t], [-t, -t, -t]]), 'commutative_antiderivative': Matrix([ [t**2/2, t**2/2, t**2/2], [t**2/2,
            t**2/2, t**2/2], [t**2/2, t**2/2, t**2/2]]), 'rhs': Matrix([ [t], [0], [0]]), 'type_of_equation':
            'type4'}
    assert _classify_linear_system(eq3, funcs_2[:-1], t) == sol3

    eq4 = [Eq(x1, x(t) + y(t) + t*z(t) + 1), Eq(y1, x(t) + t*y(t) + z(t) + 10), Eq(z1, t*x(t) + y(t) + z(t) + t)]
    sol4 = {'no_of_equation': 3, 'eq': [Eq(Derivative(x(t), t), t*z(t) + x(t) + y(t) + 1), Eq(Derivative(y(t),
            t), t*y(t) + x(t) + z(t) + 10), Eq(Derivative(z(t), t), t*x(t) + t + y(t) + z(t))], 'func': [x(t),
            y(t), z(t)], 'order': {x(t): 1, y(t): 1, z(t): 1}, 'is_linear': True, 'is_constant': False,
            'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([ [-1, -1, -t], [-1, -t, -1], [-t,
            -1, -1]]), 'commutative_antiderivative': Matrix([ [     t,      t, t**2/2], [     t, t**2/2,
            t], [t**2/2,      t,      t]]), 'rhs': Matrix([ [ 1], [10], [ t]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq4, funcs_2[:-1], t) == sol4

    sum_terms = t*(x(t) + y(t) + z(t) + w(t))
    eq5 = [Eq(x1, sum_terms), Eq(y1, sum_terms), Eq(z1, sum_terms + 1), Eq(w1, sum_terms)]
    sol5 = {'no_of_equation': 4, 'eq': [Eq(Derivative(x(t), t), t*(w(t) + x(t) + y(t) + z(t))),
            Eq(Derivative(y(t), t), t*(w(t) + x(t) + y(t) + z(t))), Eq(Derivative(z(t), t), t*(w(t) + x(t) +
            y(t) + z(t)) + 1), Eq(Derivative(w(t), t), t*(w(t) + x(t) + y(t) + z(t)))], 'func': [x(t), y(t),
            z(t), w(t)], 'order': {x(t): 1, y(t): 1, z(t): 1, w(t): 1}, 'is_linear': True, 'is_constant': False,
            'is_homogeneous': False, 'is_general': True, 'func_coeff': -Matrix([ [-t, -t, -t, -t], [-t, -t, -t,
            -t], [-t, -t, -t, -t], [-t, -t, -t, -t]]), 'commutative_antiderivative': Matrix([ [t**2/2, t**2/2,
            t**2/2, t**2/2], [t**2/2, t**2/2, t**2/2, t**2/2], [t**2/2, t**2/2, t**2/2, t**2/2], [t**2/2,
            t**2/2, t**2/2, t**2/2]]), 'rhs': Matrix([ [0], [0], [1], [0]]), 'type_of_equation': 'type4'}
    assert _classify_linear_system(eq5, funcs_2, t) == sol5

    # Second Order
    t_ = symbols("t_")

    eq1 = (Eq(9*x(t) + 7*y(t) + 4*Derivative(x(t), t) + Derivative(x(t), (t, 2)) + 3*Derivative(y(t), t), 11*exp(I*t)),
           Eq(3*x(t) + 12*y(t) + 5*Derivative(x(t), t) + 8*Derivative(y(t), t) + Derivative(y(t), (t, 2)), 2*exp(I*t)))
    sol1 = {'no_of_equation': 2, 'eq': (Eq(9*x(t) + 7*y(t) + 4*Derivative(x(t), t) + Derivative(x(t), (t, 2)) +
            3*Derivative(y(t), t), 11*exp(I*t)), Eq(3*x(t) + 12*y(t) + 5*Derivative(x(t), t) +
            8*Derivative(y(t), t) + Derivative(y(t), (t, 2)), 2*exp(I*t))), 'func': [x(t), y(t)], 'order':
            {x(t): 2, y(t): 2}, 'is_linear': True, 'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([
            [11*exp(I*t)], [ 2*exp(I*t)]]), 'type_of_equation': 'type0', 'is_second_order': True,
            'is_higher_order': True}
    assert _classify_linear_system(eq1, funcs, t) == sol1

    eq2 = (Eq((4*t**2 + 7*t + 1)**2*Derivative(x(t), (t, 2)), 5*x(t) + 35*y(t)),
           Eq((4*t**2 + 7*t + 1)**2*Derivative(y(t), (t, 2)), x(t) + 9*y(t)))
    sol2 = {'no_of_equation': 2, 'eq': (Eq((4*t**2 + 7*t + 1)**2*Derivative(x(t), (t, 2)), 5*x(t) + 35*y(t)),
            Eq((4*t**2 + 7*t + 1)**2*Derivative(y(t), (t, 2)), x(t) + 9*y(t))), 'func': [x(t), y(t)], 'order':
            {x(t): 2, y(t): 2}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True,
            'type_of_equation': 'type2', 'A0': Matrix([ [Rational(53, 4),   35], [   1, Rational(69, 4)]]), 'g(t)': sqrt(4*t**2 + 7*t
            + 1), 'tau': sqrt(33)*log(t - sqrt(33)/8 + Rational(7, 8))/33 - sqrt(33)*log(t + sqrt(33)/8 + Rational(7, 8))/33,
            'is_transformed': True, 't_': t_, 'is_second_order': True, 'is_higher_order': True}
    assert _classify_linear_system(eq2, funcs, t) == sol2

    eq3 = ((t*Derivative(x(t), t) - x(t))*log(t) + (t*Derivative(y(t), t) - y(t))*exp(t) + Derivative(x(t), (t, 2)),
            t**2*(t*Derivative(x(t), t) - x(t)) + t*(t*Derivative(y(t), t) - y(t)) + Derivative(y(t), (t, 2)))
    sol3 = {'no_of_equation': 2, 'eq': ((t*Derivative(x(t), t) - x(t))*log(t) + (t*Derivative(y(t), t) -
            y(t))*exp(t) + Derivative(x(t), (t, 2)), t**2*(t*Derivative(x(t), t) - x(t)) + t*(t*Derivative(y(t),
            t) - y(t)) + Derivative(y(t), (t, 2))), 'func': [x(t), y(t)], 'order': {x(t): 2, y(t): 2},
            'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation': 'type1', 'A1':
            Matrix([ [-t*log(t), -t*exp(t)], [    -t**3,     -t**2]]), 'is_second_order': True,
            'is_higher_order': True}
    assert _classify_linear_system(eq3, funcs, t) == sol3

    eq4 = (Eq(x2, k*x(t) - l*y1), Eq(y2, l*x1 + k*y(t)))
    sol4 = {'no_of_equation': 2, 'eq': (Eq(Derivative(x(t), (t, 2)), k*x(t) - l*Derivative(y(t), t)),
            Eq(Derivative(y(t), (t, 2)), k*y(t) + l*Derivative(x(t), t))), 'func': [x(t), y(t)], 'order': {x(t):
            2, y(t): 2}, 'is_linear': True, 'is_homogeneous': True, 'is_general': True, 'type_of_equation':
            'type0', 'is_second_order': True, 'is_higher_order': True}
    assert _classify_linear_system(eq4, funcs, t) == sol4


    # Multiple matches

    f, g = symbols("f g", cls=Function)
    y, t_ = symbols("y t_")
    funcs = [f(t), g(t)]

    eq1 = [Eq(Derivative(f(t), t)**2 - 2*Derivative(f(t), t) + 1, 4),
            Eq(-y*f(t) + Derivative(g(t), t), 0)]
    sol1 = {'is_implicit': True,
             'canon_eqs': [[Eq(Derivative(f(t), t), -1), Eq(Derivative(g(t), t), y*f(t))],
              [Eq(Derivative(f(t), t), 3), Eq(Derivative(g(t), t), y*f(t))]]}
    assert _classify_linear_system(eq1, funcs, t) == sol1

    raises(ValueError, lambda: _classify_linear_system(eq1, funcs[:1], t))

    eq2 = [Eq(Derivative(f(t), t), (2*f(t) + g(t) + 1)/t), Eq(Derivative(g(t), t), (f(t) + 2*g(t))/t)]
    sol2 = {'no_of_equation': 2, 'eq': [Eq(Derivative(f(t), t), (2*f(t) + g(t) + 1)/t), Eq(Derivative(g(t), t),
            (f(t) + 2*g(t))/t)], 'func': [f(t), g(t)], 'order': {f(t): 1, g(t): 1}, 'is_linear': True,
            'is_homogeneous': False, 'is_general': True, 'rhs': Matrix([ [1], [0]]), 'func_coeff': Matrix([ [2,
            1], [1, 2]]), 'is_constant': False, 'type_of_equation': 'type6', 't_': t_, 'tau': log(t),
            'commutative_antiderivative': Matrix([ [2*log(t),   log(t)], [  log(t), 2*log(t)]])}
    assert _classify_linear_system(eq2, funcs, t) == sol2

    eq3 = [Eq(Derivative(f(t), t), (2*f(t) + g(t))/t), Eq(Derivative(g(t), t), (f(t) + 2*g(t))/t)]
    sol3 = {'no_of_equation': 2, 'eq': [Eq(Derivative(f(t), t), (2*f(t) + g(t))/t), Eq(Derivative(g(t), t),
            (f(t) + 2*g(t))/t)], 'func': [f(t), g(t)], 'order': {f(t): 1, g(t): 1}, 'is_linear': True,
            'is_homogeneous': True, 'is_general': True, 'func_coeff': Matrix([ [2, 1], [1, 2]]), 'is_constant':
            False, 'type_of_equation': 'type5', 't_': t_, 'rhs': Matrix([ [0], [0]]), 'tau': log(t),
            'commutative_antiderivative': Matrix([ [2*log(t),   log(t)], [  log(t), 2*log(t)]])}
    assert _classify_linear_system(eq3, funcs, t) == sol3


def test_matrix_exp():
    from sympy.matrices.dense import Matrix, eye, zeros
    from sympy.solvers.ode.systems import matrix_exp
    t = Symbol('t')

    for n in range(1, 6+1):
        assert matrix_exp(zeros(n), t) == eye(n)

    for n in range(1, 6+1):
        A = eye(n)
        expAt = exp(t) * eye(n)
        assert matrix_exp(A, t) == expAt

    for n in range(1, 6+1):
        A = Matrix(n, n, lambda i,j: i+1 if i==j else 0)
        expAt = Matrix(n, n, lambda i,j: exp((i+1)*t) if i==j else 0)
        assert matrix_exp(A, t) == expAt

    A = Matrix([[0, 1], [-1, 0]])
    expAt = Matrix([[cos(t), sin(t)], [-sin(t), cos(t)]])
    assert matrix_exp(A, t) == expAt

    A = Matrix([[2, -5], [2, -4]])
    expAt = Matrix([
            [3*exp(-t)*sin(t) + exp(-t)*cos(t), -5*exp(-t)*sin(t)],
            [2*exp(-t)*sin(t), -3*exp(-t)*sin(t) + exp(-t)*cos(t)]
            ])
    assert matrix_exp(A, t) == expAt

    A = Matrix([[21, 17, 6], [-5, -1, -6], [4, 4, 16]])
    # TO update this.
    # expAt = Matrix([
    #     [(8*t*exp(12*t) + 5*exp(12*t) - 1)*exp(4*t)/4,
    #      (8*t*exp(12*t) + 5*exp(12*t) - 5)*exp(4*t)/4,
    #      (exp(12*t) - 1)*exp(4*t)/2],
    #     [(-8*t*exp(12*t) - exp(12*t) + 1)*exp(4*t)/4,
    #      (-8*t*exp(12*t) - exp(12*t) + 5)*exp(4*t)/4,
    #      (-exp(12*t) + 1)*exp(4*t)/2],
    #     [4*t*exp(16*t), 4*t*exp(16*t), exp(16*t)]])
    expAt = Matrix([
        [2*t*exp(16*t) + 5*exp(16*t)/4 - exp(4*t)/4, 2*t*exp(16*t) + 5*exp(16*t)/4 - 5*exp(4*t)/4,  exp(16*t)/2 - exp(4*t)/2],
        [ -2*t*exp(16*t) - exp(16*t)/4 + exp(4*t)/4,  -2*t*exp(16*t) - exp(16*t)/4 + 5*exp(4*t)/4, -exp(16*t)/2 + exp(4*t)/2],
        [                             4*t*exp(16*t),                                4*t*exp(16*t),                 exp(16*t)]
        ])
    assert matrix_exp(A, t) == expAt

    A = Matrix([[1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, -S(1)/8],
                [0, 0, S(1)/2, S(1)/2]])
    expAt = Matrix([
        [exp(t), t*exp(t), 4*t*exp(3*t/4) + 8*t*exp(t) + 48*exp(3*t/4) - 48*exp(t),
                            -2*t*exp(3*t/4) - 2*t*exp(t) - 16*exp(3*t/4) + 16*exp(t)],
        [0, exp(t), -t*exp(3*t/4) - 8*exp(3*t/4) + 8*exp(t), t*exp(3*t/4)/2 + 2*exp(3*t/4) - 2*exp(t)],
        [0, 0, t*exp(3*t/4)/4 + exp(3*t/4), -t*exp(3*t/4)/8],
        [0, 0, t*exp(3*t/4)/2, -t*exp(3*t/4)/4 + exp(3*t/4)]
        ])
    assert matrix_exp(A, t) == expAt

    A = Matrix([
    [ 0, 1,  0, 0],
    [-1, 0,  0, 0],
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0]])

    expAt = Matrix([
    [ cos(t), sin(t),         0,        0],
    [-sin(t), cos(t),         0,        0],
    [      0,      0,    cos(t),   sin(t)],
    [      0,      0,   -sin(t),   cos(t)]])
    assert matrix_exp(A, t) == expAt

    A = Matrix([
    [ 0, 1,  1, 0],
    [-1, 0,  0, 1],
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0]])

    expAt = Matrix([
    [ cos(t), sin(t),  t*cos(t), t*sin(t)],
    [-sin(t), cos(t), -t*sin(t), t*cos(t)],
    [      0,      0,    cos(t),   sin(t)],
    [      0,      0,   -sin(t),   cos(t)]])
    assert matrix_exp(A, t) == expAt

    # This case is unacceptably slow right now but should be solvable...
    #a, b, c, d, e, f = symbols('a b c d e f')
    #A = Matrix([
    #[-a,  b,          c,  d],
    #[ a, -b,          e,  0],
    #[ 0,  0, -c - e - f,  0],
    #[ 0,  0,          f, -d]])

    A = Matrix([[0, I], [I, 0]])
    expAt = Matrix([
    [exp(I*t)/2 + exp(-I*t)/2, exp(I*t)/2 - exp(-I*t)/2],
    [exp(I*t)/2 - exp(-I*t)/2, exp(I*t)/2 + exp(-I*t)/2]])
    assert matrix_exp(A, t) == expAt

    # Testing Errors
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 7, 7]])
    M1 = Matrix([[t, 1], [1, 1]])

    raises(ValueError, lambda: matrix_exp(M[:, :2], t))
    raises(ValueError, lambda: matrix_exp(M[:2, :], t))
    raises(ValueError, lambda: matrix_exp(M1, t))
    raises(ValueError, lambda: matrix_exp(M1[:1, :1], t))


def test_canonical_odes():
    f, g, h = symbols('f g h', cls=Function)
    x = symbols('x')
    funcs = [f(x), g(x), h(x)]

    eqs1 = [Eq(f(x).diff(x, x), f(x) + 2*g(x)), Eq(g(x) + 1, g(x).diff(x) + f(x))]
    sol1 = [[Eq(Derivative(f(x), (x, 2)), f(x) + 2*g(x)), Eq(Derivative(g(x), x), -f(x) + g(x) + 1)]]
    assert canonical_odes(eqs1, funcs[:2], x) == sol1

    eqs2 = [Eq(f(x).diff(x), h(x).diff(x) + f(x)), Eq(g(x).diff(x)**2, f(x) + h(x)), Eq(h(x).diff(x), f(x))]
    sol2 = [[Eq(Derivative(f(x), x), 2*f(x)), Eq(Derivative(g(x), x), -sqrt(f(x) + h(x))), Eq(Derivative(h(x), x), f(x))],
            [Eq(Derivative(f(x), x), 2*f(x)), Eq(Derivative(g(x), x), sqrt(f(x) + h(x))), Eq(Derivative(h(x), x), f(x))]]
    assert canonical_odes(eqs2, funcs, x) == sol2


def test_sysode_linear_neq_order1_type1():

    f, g, x, y, h = symbols('f g x y h', cls=Function)
    a, b, c, t = symbols('a b c t')

    eqs1 = [Eq(Derivative(x(t), t), x(t)),
            Eq(Derivative(y(t), t), y(t))]
    sol1 = [Eq(x(t), C1*exp(t)),
            Eq(y(t), C2*exp(t))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    eqs2 = [Eq(Derivative(x(t), t), 2*x(t)),
            Eq(Derivative(y(t), t), 3*y(t))]
    sol2 = [Eq(x(t), C1*exp(2*t)),
            Eq(y(t), C2*exp(3*t))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    eqs3 = [Eq(Derivative(x(t), t), a*x(t)),
            Eq(Derivative(y(t), t), a*y(t))]
    sol3 = [Eq(x(t), C1*exp(a*t)),
            Eq(y(t), C2*exp(a*t))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    # Regression test case for issue #15474
    # https://github.com/sympy/sympy/issues/15474
    eqs4 = [Eq(Derivative(x(t), t), a*x(t)),
            Eq(Derivative(y(t), t), b*y(t))]
    sol4 = [Eq(x(t), C1*exp(a*t)),
            Eq(y(t), C2*exp(b*t))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])

    eqs5 = [Eq(Derivative(x(t), t), -y(t)),
            Eq(Derivative(y(t), t), x(t))]
    sol5 = [Eq(x(t), -C1*sin(t) - C2*cos(t)),
            Eq(y(t), C1*cos(t) - C2*sin(t))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])

    eqs6 = [Eq(Derivative(x(t), t), -2*y(t)),
            Eq(Derivative(y(t), t), 2*x(t))]
    sol6 = [Eq(x(t), -C1*sin(2*t) - C2*cos(2*t)),
            Eq(y(t), C1*cos(2*t) - C2*sin(2*t))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])

    eqs7 = [Eq(Derivative(x(t), t), I*y(t)),
            Eq(Derivative(y(t), t), I*x(t))]
    sol7 = [Eq(x(t), -C1*exp(-I*t) + C2*exp(I*t)),
            Eq(y(t), C1*exp(-I*t) + C2*exp(I*t))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    eqs8 = [Eq(Derivative(x(t), t), -a*y(t)),
            Eq(Derivative(y(t), t), a*x(t))]
    sol8 = [Eq(x(t), -I*C1*exp(-I*a*t) + I*C2*exp(I*a*t)),
            Eq(y(t), C1*exp(-I*a*t) + C2*exp(I*a*t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])

    eqs9 = [Eq(Derivative(x(t), t), x(t) + y(t)),
            Eq(Derivative(y(t), t), x(t) - y(t))]
    sol9 = [Eq(x(t), C1*(1 - sqrt(2))*exp(-sqrt(2)*t) + C2*(1 + sqrt(2))*exp(sqrt(2)*t)),
            Eq(y(t), C1*exp(-sqrt(2)*t) + C2*exp(sqrt(2)*t))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])

    eqs10 = [Eq(Derivative(x(t), t), x(t) + y(t)),
             Eq(Derivative(y(t), t), x(t) + y(t))]
    sol10 = [Eq(x(t), -C1 + C2*exp(2*t)),
             Eq(y(t), C1 + C2*exp(2*t))]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])

    eqs11 = [Eq(Derivative(x(t), t), 2*x(t) + y(t)),
             Eq(Derivative(y(t), t), -x(t) + 2*y(t))]
    sol11 = [Eq(x(t), C1*exp(2*t)*sin(t) + C2*exp(2*t)*cos(t)),
             Eq(y(t), C1*exp(2*t)*cos(t) - C2*exp(2*t)*sin(t))]
    assert dsolve(eqs11) == sol11
    assert checksysodesol(eqs11, sol11) == (True, [0, 0])

    eqs12 = [Eq(Derivative(x(t), t), x(t) + 2*y(t)),
             Eq(Derivative(y(t), t), 2*x(t) + y(t))]
    sol12 = [Eq(x(t), -C1*exp(-t) + C2*exp(3*t)),
             Eq(y(t), C1*exp(-t) + C2*exp(3*t))]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])

    eqs13 = [Eq(Derivative(x(t), t), 4*x(t) + y(t)),
             Eq(Derivative(y(t), t), -x(t) + 2*y(t))]
    sol13 = [Eq(x(t), C2*t*exp(3*t) + (C1 + C2)*exp(3*t)),
             Eq(y(t), -C1*exp(3*t) - C2*t*exp(3*t))]
    assert dsolve(eqs13) == sol13
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])

    eqs14 = [Eq(Derivative(x(t), t), a*y(t)),
             Eq(Derivative(y(t), t), a*x(t))]
    sol14 = [Eq(x(t), -C1*exp(-a*t) + C2*exp(a*t)),
             Eq(y(t), C1*exp(-a*t) + C2*exp(a*t))]
    assert dsolve(eqs14) == sol14
    assert checksysodesol(eqs14, sol14) == (True, [0, 0])

    eqs15 = [Eq(Derivative(x(t), t), a*y(t)),
             Eq(Derivative(y(t), t), b*x(t))]
    sol15 = [Eq(x(t), -C1*a*exp(-t*sqrt(a*b))/sqrt(a*b) + C2*a*exp(t*sqrt(a*b))/sqrt(a*b)),
             Eq(y(t), C1*exp(-t*sqrt(a*b)) + C2*exp(t*sqrt(a*b)))]
    assert dsolve(eqs15) == sol15
    assert checksysodesol(eqs15, sol15) == (True, [0, 0])

    eqs16 = [Eq(Derivative(x(t), t), a*x(t) + b*y(t)),
             Eq(Derivative(y(t), t), c*x(t))]
    sol16 = [Eq(x(t), -2*C1*b*exp(t*(a + sqrt(a**2 + 4*b*c))/2)/(a - sqrt(a**2 + 4*b*c)) - 2*C2*b*exp(t*(a -
              sqrt(a**2 + 4*b*c))/2)/(a + sqrt(a**2 + 4*b*c))),
             Eq(y(t), C1*exp(t*(a + sqrt(a**2 + 4*b*c))/2) + C2*exp(t*(a - sqrt(a**2 + 4*b*c))/2))]
    assert dsolve(eqs16) == sol16
    assert checksysodesol(eqs16, sol16) == (True, [0, 0])

    # Regression test case for issue #18562
    # https://github.com/sympy/sympy/issues/18562
    eqs17 = [Eq(Derivative(x(t), t), a*y(t) + x(t)),
            Eq(Derivative(y(t), t), a*x(t) - y(t))]
    sol17 = [Eq(x(t), C1*a*exp(t*sqrt(a**2 + 1))/(sqrt(a**2 + 1) - 1) - C2*a*exp(-t*sqrt(a**2 + 1))/(sqrt(a**2 +
              1) + 1)),
            Eq(y(t), C1*exp(t*sqrt(a**2 + 1)) + C2*exp(-t*sqrt(a**2 + 1)))]
    assert dsolve(eqs17) == sol17
    assert checksysodesol(eqs17, sol17) == (True, [0, 0])

    eqs18 = [Eq(Derivative(x(t), t), 0),
            Eq(Derivative(y(t), t), 0)]
    sol18 = [Eq(x(t), C1),
            Eq(y(t), C2)]
    assert dsolve(eqs18) == sol18
    assert checksysodesol(eqs18, sol18) == (True, [0, 0])

    eqs19 = [Eq(Derivative(x(t), t), 2*x(t) - y(t)),
            Eq(Derivative(y(t), t), x(t))]
    sol19 = [Eq(x(t), C2*t*exp(t) + (C1 + C2)*exp(t)),
            Eq(y(t), C1*exp(t) + C2*t*exp(t))]
    assert dsolve(eqs19) == sol19
    assert checksysodesol(eqs19, sol19) == (True, [0, 0])

    eqs20 = [Eq(Derivative(x(t), t), x(t)),
            Eq(Derivative(y(t), t), x(t) + y(t))]
    sol20 = [Eq(x(t), C1*exp(t)),
            Eq(y(t), C1*t*exp(t) + C2*exp(t))]
    assert dsolve(eqs20) == sol20
    assert checksysodesol(eqs20, sol20) == (True, [0, 0])

    eqs21 = [Eq(Derivative(x(t), t), 3*x(t)),
            Eq(Derivative(y(t), t), x(t) + y(t))]
    sol21 = [Eq(x(t), 2*C1*exp(3*t)),
            Eq(y(t), C1*exp(3*t) + C2*exp(t))]
    assert dsolve(eqs21) == sol21
    assert checksysodesol(eqs21, sol21) == (True, [0, 0])

    eqs22 = [Eq(Derivative(x(t), t), 3*x(t)),
            Eq(Derivative(y(t), t), y(t))]
    sol22 = [Eq(x(t), C1*exp(3*t)),
            Eq(y(t), C2*exp(t))]
    assert dsolve(eqs22) == sol22
    assert checksysodesol(eqs22, sol22) == (True, [0, 0])


@slow
def test_sysode_linear_neq_order1_type1_slow():

    t = Symbol('t')
    Z0 = Function('Z0')
    Z1 = Function('Z1')
    Z2 = Function('Z2')
    Z3 = Function('Z3')

    k01, k10, k20, k21, k23, k30 = symbols('k01 k10 k20 k21 k23 k30')

    eqs1 = [Eq(Derivative(Z0(t), t), -k01*Z0(t) + k10*Z1(t) + k20*Z2(t) + k30*Z3(t)),
            Eq(Derivative(Z1(t), t), k01*Z0(t) - k10*Z1(t) + k21*Z2(t)),
            Eq(Derivative(Z2(t), t), (-k20 - k21 - k23)*Z2(t)),
            Eq(Derivative(Z3(t), t), k23*Z2(t) - k30*Z3(t))]
    sol1 = [Eq(Z0(t), C1*k10/k01 - C2*(k10 - k30)*exp(-k30*t)/(k01 + k10 - k30) - C3*(k10*(k20 + k21 - k30) -
             k20**2 - k20*(k21 + k23 - k30) + k23*k30)*exp(-t*(k20 + k21 + k23))/(k23*(-k01 - k10 + k20 + k21 +
             k23)) - C4*exp(-t*(k01 + k10))),
            Eq(Z1(t), C1 - C2*k01*exp(-k30*t)/(k01 + k10 - k30) + C3*(-k01*(k20 + k21 - k30) + k20*k21 + k21**2
             + k21*(k23 - k30))*exp(-t*(k20 + k21 + k23))/(k23*(-k01 - k10 + k20 + k21 + k23)) + C4*exp(-t*(k01 +
             k10))),
            Eq(Z2(t), -C3*(k20 + k21 + k23 - k30)*exp(-t*(k20 + k21 + k23))/k23),
            Eq(Z3(t), C2*exp(-k30*t) + C3*exp(-t*(k20 + k21 + k23)))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0, 0, 0])

    x, y, z, u, v, w = symbols('x y z u v w', cls=Function)
    k2, k3 = symbols('k2 k3')
    a_b, a_c = symbols('a_b a_c', real=True)

    eqs2 = [Eq(Derivative(z(t), t), k2*y(t)),
            Eq(Derivative(x(t), t), k3*y(t)),
            Eq(Derivative(y(t), t), (-k2 - k3)*y(t))]
    sol2 = [Eq(z(t), C1 - C2*k2*exp(-t*(k2 + k3))/(k2 + k3)),
            Eq(x(t), -C2*k3*exp(-t*(k2 + k3))/(k2 + k3) + C3),
            Eq(y(t), C2*exp(-t*(k2 + k3)))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0, 0])

    eqs3 = [4*u(t) - v(t) - 2*w(t) + Derivative(u(t), t),
            2*u(t) + v(t) - 2*w(t) + Derivative(v(t), t),
            5*u(t) + v(t) - 3*w(t) + Derivative(w(t), t)]
    sol3 = [Eq(u(t), C3*exp(-2*t) + (C1/2 + sqrt(3)*C2/6)*cos(sqrt(3)*t) + sin(sqrt(3)*t)*(sqrt(3)*C1/6 +
             C2*Rational(-1, 2))),
            Eq(v(t), (C1/2 + sqrt(3)*C2/6)*cos(sqrt(3)*t) + sin(sqrt(3)*t)*(sqrt(3)*C1/6 + C2*Rational(-1, 2))),
            Eq(w(t), C1*cos(sqrt(3)*t) - C2*sin(sqrt(3)*t) + C3*exp(-2*t))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0])

    eqs4 = [Eq(Derivative(x(t), t), w(t)*Rational(-2, 9) + 2*x(t) + y(t) + z(t)*Rational(-8, 9)),
            Eq(Derivative(y(t), t), w(t)*Rational(4, 9) + 2*y(t) + z(t)*Rational(16, 9)),
            Eq(Derivative(z(t), t), w(t)*Rational(-2, 9) + z(t)*Rational(37, 9)),
            Eq(Derivative(w(t), t), w(t)*Rational(44, 9) + z(t)*Rational(-4, 9))]
    sol4 = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t)),
            Eq(y(t), C2*exp(2*t) + 2*C3*exp(4*t)),
            Eq(z(t), 2*C3*exp(4*t) + C4*exp(5*t)*Rational(-1, 4)),
            Eq(w(t), C3*exp(4*t) + C4*exp(5*t))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0, 0])

    # Regression test case for issue #15574
    # https://github.com/sympy/sympy/issues/15574
    eq5 = [Eq(x(t).diff(t), x(t)), Eq(y(t).diff(t), y(t)), Eq(z(t).diff(t), z(t)), Eq(w(t).diff(t), w(t))]
    sol5 = [Eq(x(t), C1*exp(t)), Eq(y(t), C2*exp(t)), Eq(z(t), C3*exp(t)), Eq(w(t), C4*exp(t))]
    assert dsolve(eq5) == sol5
    assert checksysodesol(eq5, sol5) == (True, [0, 0, 0, 0])

    eqs6 = [Eq(Derivative(x(t), t), x(t) + y(t)),
            Eq(Derivative(y(t), t), y(t) + z(t)),
            Eq(Derivative(z(t), t), w(t)*Rational(-1, 8) + z(t)),
            Eq(Derivative(w(t), t), w(t)/2 + z(t)/2)]
    sol6 = [Eq(x(t), C1*exp(t) + C2*t*exp(t) + 4*C4*t*exp(t*Rational(3, 4)) + (4*C3 + 48*C4)*exp(t*Rational(3,
             4))),
            Eq(y(t), C2*exp(t) - C4*t*exp(t*Rational(3, 4)) - (C3 + 8*C4)*exp(t*Rational(3, 4))),
            Eq(z(t), C4*t*exp(t*Rational(3, 4))/4 + (C3/4 + C4)*exp(t*Rational(3, 4))),
            Eq(w(t), C3*exp(t*Rational(3, 4))/2 + C4*t*exp(t*Rational(3, 4))/2)]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])

    # Regression test case for issue #15574
    # https://github.com/sympy/sympy/issues/15574
    eq7 = [Eq(Derivative(x(t), t), x(t)), Eq(Derivative(y(t), t), y(t)), Eq(Derivative(z(t), t), z(t)),
           Eq(Derivative(w(t), t), w(t)), Eq(Derivative(u(t), t), u(t))]
    sol7 = [Eq(x(t), C1*exp(t)), Eq(y(t), C2*exp(t)), Eq(z(t), C3*exp(t)), Eq(w(t), C4*exp(t)),
            Eq(u(t), C5*exp(t))]
    assert dsolve(eq7) == sol7
    assert checksysodesol(eq7, sol7) == (True, [0, 0, 0, 0, 0])

    eqs8 = [Eq(Derivative(x(t), t), 2*x(t) + y(t)),
            Eq(Derivative(y(t), t), 2*y(t)),
            Eq(Derivative(z(t), t), 4*z(t)),
            Eq(Derivative(w(t), t), u(t) + 5*w(t)),
            Eq(Derivative(u(t), t), 5*u(t))]
    sol8 = [Eq(x(t), C1*exp(2*t) + C2*t*exp(2*t)),
            Eq(y(t), C2*exp(2*t)),
            Eq(z(t), C3*exp(4*t)),
            Eq(w(t), C4*exp(5*t) + C5*t*exp(5*t)),
            Eq(u(t), C5*exp(5*t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0, 0, 0, 0])

    # Regression test case for issue #15574
    # https://github.com/sympy/sympy/issues/15574
    eq9 = [Eq(Derivative(x(t), t), x(t)), Eq(Derivative(y(t), t), y(t)), Eq(Derivative(z(t), t), z(t))]
    sol9 = [Eq(x(t), C1*exp(t)), Eq(y(t), C2*exp(t)), Eq(z(t), C3*exp(t))]
    assert dsolve(eq9) == sol9
    assert checksysodesol(eq9, sol9) == (True, [0, 0, 0])

    # Regression test case for issue #15407
    # https://github.com/sympy/sympy/issues/15407
    eqs10 = [Eq(Derivative(x(t), t), (-a_b - a_c)*x(t)),
             Eq(Derivative(y(t), t), a_b*y(t)),
             Eq(Derivative(z(t), t), a_c*x(t))]
    sol10 = [Eq(x(t), -C1*(a_b + a_c)*exp(-t*(a_b + a_c))/a_c),
             Eq(y(t), C2*exp(a_b*t)),
             Eq(z(t), C1*exp(-t*(a_b + a_c)) + C3)]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0, 0])

    # Regression test case for issue #14312
    # https://github.com/sympy/sympy/issues/14312
    eqs11 = [Eq(Derivative(x(t), t), k3*y(t)),
             Eq(Derivative(y(t), t), (-k2 - k3)*y(t)),
             Eq(Derivative(z(t), t), k2*y(t))]
    sol11 = [Eq(x(t), C1 + C2*k3*exp(-t*(k2 + k3))/k2),
             Eq(y(t), -C2*(k2 + k3)*exp(-t*(k2 + k3))/k2),
             Eq(z(t), C2*exp(-t*(k2 + k3)) + C3)]
    assert dsolve(eqs11) == sol11
    assert checksysodesol(eqs11, sol11) == (True, [0, 0, 0])

    # Regression test case for issue #14312
    # https://github.com/sympy/sympy/issues/14312
    eqs12 = [Eq(Derivative(z(t), t), k2*y(t)),
             Eq(Derivative(x(t), t), k3*y(t)),
             Eq(Derivative(y(t), t), (-k2 - k3)*y(t))]
    sol12 = [Eq(z(t), C1 - C2*k2*exp(-t*(k2 + k3))/(k2 + k3)),
             Eq(x(t), -C2*k3*exp(-t*(k2 + k3))/(k2 + k3) + C3),
             Eq(y(t), C2*exp(-t*(k2 + k3)))]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0, 0])

    f, g, h = symbols('f, g, h', cls=Function)
    a, b, c = symbols('a, b, c')

    # Regression test case for issue #15474
    # https://github.com/sympy/sympy/issues/15474
    eqs13 = [Eq(Derivative(f(t), t), 2*f(t) + g(t)),
             Eq(Derivative(g(t), t), a*f(t))]
    sol13 = [Eq(f(t), C1*exp(t*(sqrt(a + 1) + 1))/(sqrt(a + 1) - 1) - C2*exp(-t*(sqrt(a + 1) - 1))/(sqrt(a + 1) +
              1)),
             Eq(g(t), C1*exp(t*(sqrt(a + 1) + 1)) + C2*exp(-t*(sqrt(a + 1) - 1)))]
    assert dsolve(eqs13) == sol13
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])

    eqs14 = [Eq(Derivative(f(t), t), 2*g(t) - 3*h(t)),
             Eq(Derivative(g(t), t), -2*f(t) + 4*h(t)),
             Eq(Derivative(h(t), t), 3*f(t) - 4*g(t))]
    sol14 = [Eq(f(t), 2*C1 - sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(3, 25) + C3*Rational(-8, 25)) -
              cos(sqrt(29)*t)*(C2*Rational(8, 25) + sqrt(29)*C3*Rational(3, 25))),
             Eq(g(t), C1*Rational(3, 2) + sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(4, 25) + C3*Rational(6, 25)) -
              cos(sqrt(29)*t)*(C2*Rational(6, 25) + sqrt(29)*C3*Rational(-4, 25))),
             Eq(h(t), C1 + C2*cos(sqrt(29)*t) - C3*sin(sqrt(29)*t))]
    assert dsolve(eqs14) == sol14
    assert checksysodesol(eqs14, sol14) == (True, [0, 0, 0])

    eqs15 = [Eq(2*Derivative(f(t), t), 12*g(t) - 12*h(t)),
             Eq(3*Derivative(g(t), t), -8*f(t) + 8*h(t)),
             Eq(4*Derivative(h(t), t), 6*f(t) - 6*g(t))]
    sol15 = [Eq(f(t), C1 - sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(6, 13) + C3*Rational(-16, 13)) -
              cos(sqrt(29)*t)*(C2*Rational(16, 13) + sqrt(29)*C3*Rational(6, 13))),
             Eq(g(t), C1 + sin(sqrt(29)*t)*(sqrt(29)*C2*Rational(8, 39) + C3*Rational(16, 13)) -
              cos(sqrt(29)*t)*(C2*Rational(16, 13) + sqrt(29)*C3*Rational(-8, 39))),
             Eq(h(t), C1 + C2*cos(sqrt(29)*t) - C3*sin(sqrt(29)*t))]
    assert dsolve(eqs15) == sol15
    assert checksysodesol(eqs15, sol15) == (True, [0, 0, 0])

    eq16 = (Eq(diff(x(t), t), 21*x(t)), Eq(diff(y(t), t), 17*x(t) + 3*y(t)),
            Eq(diff(z(t), t), 5*x(t) + 7*y(t) + 9*z(t)))
    sol16 = [Eq(x(t), 216*C1*exp(21*t)/209),
             Eq(y(t), 204*C1*exp(21*t)/209 - 6*C2*exp(3*t)/7),
             Eq(z(t), C1*exp(21*t) + C2*exp(3*t) + C3*exp(9*t))]
    assert dsolve(eq16) == sol16
    assert checksysodesol(eq16, sol16) == (True, [0, 0, 0])

    eqs17 = [Eq(Derivative(x(t), t), 3*y(t) - 11*z(t)),
             Eq(Derivative(y(t), t), -3*x(t) + 7*z(t)),
             Eq(Derivative(z(t), t), 11*x(t) - 7*y(t))]
    sol17 = [Eq(x(t), C1*Rational(7, 3) - sin(sqrt(179)*t)*(sqrt(179)*C2*Rational(11, 170) + C3*Rational(-21,
              170)) - cos(sqrt(179)*t)*(C2*Rational(21, 170) + sqrt(179)*C3*Rational(11, 170))),
             Eq(y(t), C1*Rational(11, 3) + sin(sqrt(179)*t)*(sqrt(179)*C2*Rational(7, 170) + C3*Rational(33,
              170)) - cos(sqrt(179)*t)*(C2*Rational(33, 170) + sqrt(179)*C3*Rational(-7, 170))),
             Eq(z(t), C1 + C2*cos(sqrt(179)*t) - C3*sin(sqrt(179)*t))]
    assert dsolve(eqs17) == sol17
    assert checksysodesol(eqs17, sol17) == (True, [0, 0, 0])

    eqs18 = [Eq(3*Derivative(x(t), t), 20*y(t) - 20*z(t)),
             Eq(4*Derivative(y(t), t), -15*x(t) + 15*z(t)),
             Eq(5*Derivative(z(t), t), 12*x(t) - 12*y(t))]
    sol18 = [Eq(x(t), C1 - sin(5*sqrt(2)*t)*(sqrt(2)*C2*Rational(4, 3) - C3) - cos(5*sqrt(2)*t)*(C2 +
              sqrt(2)*C3*Rational(4, 3))),
             Eq(y(t), C1 + sin(5*sqrt(2)*t)*(sqrt(2)*C2*Rational(3, 4) + C3) - cos(5*sqrt(2)*t)*(C2 +
              sqrt(2)*C3*Rational(-3, 4))),
             Eq(z(t), C1 + C2*cos(5*sqrt(2)*t) - C3*sin(5*sqrt(2)*t))]
    assert dsolve(eqs18) == sol18
    assert checksysodesol(eqs18, sol18) == (True, [0, 0, 0])

    eqs19 = [Eq(Derivative(x(t), t), 4*x(t) - z(t)),
             Eq(Derivative(y(t), t), 2*x(t) + 2*y(t) - z(t)),
             Eq(Derivative(z(t), t), 3*x(t) + y(t))]
    sol19 = [Eq(x(t), C2*t**2*exp(2*t)/2 + t*(2*C2 + C3)*exp(2*t) + (C1 + C2 + 2*C3)*exp(2*t)),
             Eq(y(t), C2*t**2*exp(2*t)/2 + t*(2*C2 + C3)*exp(2*t) + (C1 + 2*C3)*exp(2*t)),
             Eq(z(t), C2*t**2*exp(2*t) + t*(3*C2 + 2*C3)*exp(2*t) + (2*C1 + 3*C3)*exp(2*t))]
    assert dsolve(eqs19) == sol19
    assert checksysodesol(eqs19, sol19) == (True, [0, 0, 0])

    eqs20 = [Eq(Derivative(x(t), t), 4*x(t) - y(t) - 2*z(t)),
             Eq(Derivative(y(t), t), 2*x(t) + y(t) - 2*z(t)),
             Eq(Derivative(z(t), t), 5*x(t) - 3*z(t))]
    sol20 = [Eq(x(t), C1*exp(2*t) - sin(t)*(C2*Rational(3, 5) + C3/5) - cos(t)*(C2/5 + C3*Rational(-3, 5))),
             Eq(y(t), -sin(t)*(C2*Rational(3, 5) + C3/5) - cos(t)*(C2/5 + C3*Rational(-3, 5))),
             Eq(z(t), C1*exp(2*t) - C2*sin(t) + C3*cos(t))]
    assert dsolve(eqs20) == sol20
    assert checksysodesol(eqs20, sol20) == (True, [0, 0, 0])

    eq21 = (Eq(diff(x(t), t), 9*y(t)), Eq(diff(y(t), t), 12*x(t)))
    sol21 = [Eq(x(t), -sqrt(3)*C1*exp(-6*sqrt(3)*t)/2 + sqrt(3)*C2*exp(6*sqrt(3)*t)/2),
             Eq(y(t), C1*exp(-6*sqrt(3)*t) + C2*exp(6*sqrt(3)*t))]

    assert dsolve(eq21) == sol21
    assert checksysodesol(eq21, sol21) == (True, [0, 0])

    eqs22 = [Eq(Derivative(x(t), t), 2*x(t) + 4*y(t)),
             Eq(Derivative(y(t), t), 12*x(t) + 41*y(t))]
    sol22 = [Eq(x(t), C1*(39 - sqrt(1713))*exp(t*(sqrt(1713) + 43)/2)*Rational(-1, 24) + C2*(39 +
              sqrt(1713))*exp(t*(43 - sqrt(1713))/2)*Rational(-1, 24)),
             Eq(y(t), C1*exp(t*(sqrt(1713) + 43)/2) + C2*exp(t*(43 - sqrt(1713))/2))]
    assert dsolve(eqs22) == sol22
    assert checksysodesol(eqs22, sol22) == (True, [0, 0])

    eqs23 = [Eq(Derivative(x(t), t), x(t) + y(t)),
             Eq(Derivative(y(t), t), -2*x(t) + 2*y(t))]
    sol23 = [Eq(x(t), (C1/4 + sqrt(7)*C2/4)*cos(sqrt(7)*t/2)*exp(t*Rational(3, 2)) +
              sin(sqrt(7)*t/2)*(sqrt(7)*C1/4 + C2*Rational(-1, 4))*exp(t*Rational(3, 2))),
             Eq(y(t), C1*cos(sqrt(7)*t/2)*exp(t*Rational(3, 2)) - C2*sin(sqrt(7)*t/2)*exp(t*Rational(3, 2)))]
    assert dsolve(eqs23) == sol23
    assert checksysodesol(eqs23, sol23) == (True, [0, 0])

    # Regression test case for issue #15474
    # https://github.com/sympy/sympy/issues/15474
    a = Symbol("a", real=True)
    eq24 = [x(t).diff(t) - a*y(t), y(t).diff(t) + a*x(t)]
    sol24 = [Eq(x(t), C1*sin(a*t) + C2*cos(a*t)), Eq(y(t), C1*cos(a*t) - C2*sin(a*t))]
    assert dsolve(eq24) == sol24
    assert checksysodesol(eq24, sol24) == (True, [0, 0])

    # Regression test case for issue #19150
    # https://github.com/sympy/sympy/issues/19150
    eqs25 = [Eq(Derivative(f(t), t), 0),
             Eq(Derivative(g(t), t), (f(t) - 2*g(t) + x(t))/(b*c)),
             Eq(Derivative(x(t), t), (g(t) - 2*x(t) + y(t))/(b*c)),
             Eq(Derivative(y(t), t), (h(t) + x(t) - 2*y(t))/(b*c)),
             Eq(Derivative(h(t), t), 0)]
    sol25 = [Eq(f(t), -3*C1 + 4*C2),
             Eq(g(t), -2*C1 + 3*C2 - C3*exp(-2*t/(b*c)) + C4*exp(-t*(sqrt(2) + 2)/(b*c)) + C5*exp(-t*(2 -
              sqrt(2))/(b*c))),
             Eq(x(t), -C1 + 2*C2 - sqrt(2)*C4*exp(-t*(sqrt(2) + 2)/(b*c)) + sqrt(2)*C5*exp(-t*(2 -
              sqrt(2))/(b*c))),
             Eq(y(t), C2 + C3*exp(-2*t/(b*c)) + C4*exp(-t*(sqrt(2) + 2)/(b*c)) + C5*exp(-t*(2 - sqrt(2))/(b*c))),
             Eq(h(t), C1)]
    assert dsolve(eqs25) == sol25
    assert checksysodesol(eqs25, sol25) == (True, [0, 0, 0, 0, 0])

    eq26 = [Eq(Derivative(f(t), t), 2*f(t)), Eq(Derivative(g(t), t), 3*f(t) + 7*g(t))]
    sol26 = [Eq(f(t), -5*C1*exp(2*t)/3), Eq(g(t), C1*exp(2*t) + C2*exp(7*t))]
    assert dsolve(eq26) == sol26
    assert checksysodesol(eq26, sol26) == (True, [0, 0])

    eq27 = [Eq(Derivative(f(t), t), -9*I*f(t) - 4*g(t)), Eq(Derivative(g(t), t), -4*I*g(t))]
    sol27 = [Eq(f(t), 4*I*C1*exp(-4*I*t)/5 + C2*exp(-9*I*t)), Eq(g(t), C1*exp(-4*I*t))]
    assert dsolve(eq27) == sol27
    assert checksysodesol(eq27, sol27) == (True, [0, 0])

    eq28 = [Eq(Derivative(f(t), t), -9*I*f(t)), Eq(Derivative(g(t), t), -4*I*g(t))]
    sol28 = [Eq(f(t), C1*exp(-9*I*t)), Eq(g(t), C2*exp(-4*I*t))]
    assert dsolve(eq28) == sol28
    assert checksysodesol(eq28, sol28) == (True, [0, 0])

    eq29 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), 0)]
    sol29 = [Eq(f(t), C1), Eq(g(t), C2)]
    assert dsolve(eq29) == sol29
    assert checksysodesol(eq29, sol29) == (True, [0, 0])

    eq30 = [Eq(Derivative(f(t), t), f(t)), Eq(Derivative(g(t), t), 0)]
    sol30 = [Eq(f(t), C1*exp(t)), Eq(g(t), C2)]
    assert dsolve(eq30) == sol30
    assert checksysodesol(eq30, sol30) == (True, [0, 0])

    eq31 = [Eq(Derivative(f(t), t), g(t)), Eq(Derivative(g(t), t), 0)]
    sol31 = [Eq(f(t), C1 + C2*t), Eq(g(t), C2)]
    assert dsolve(eq31) == sol31
    assert checksysodesol(eq31, sol31) == (True, [0, 0])

    eq32 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), f(t))]
    sol32 = [Eq(f(t), C1), Eq(g(t), C1*t + C2)]
    assert dsolve(eq32) == sol32
    assert checksysodesol(eq32, sol32) == (True, [0, 0])

    eq33 = [Eq(Derivative(f(t), t), 0), Eq(Derivative(g(t), t), g(t))]
    sol33 = [Eq(f(t), C1), Eq(g(t), C2*exp(t))]
    assert dsolve(eq33) == sol33
    assert checksysodesol(eq33, sol33) == (True, [0, 0])

    eq34 = [Eq(Derivative(f(t), t), f(t)), Eq(Derivative(g(t), t), I*g(t))]
    sol34 = [Eq(f(t), C1*exp(t)), Eq(g(t), C2*exp(I*t))]
    assert dsolve(eq34) == sol34
    assert checksysodesol(eq34, sol34) == (True, [0, 0])

    eq35 = [Eq(Derivative(f(t), t), I*f(t)), Eq(Derivative(g(t), t), -I*g(t))]
    sol35 = [Eq(f(t), C1*exp(I*t)), Eq(g(t), C2*exp(-I*t))]
    assert dsolve(eq35) == sol35
    assert checksysodesol(eq35, sol35) == (True, [0, 0])

    eq36 = [Eq(Derivative(f(t), t), I*g(t)), Eq(Derivative(g(t), t), 0)]
    sol36 = [Eq(f(t), I*C1 + I*C2*t), Eq(g(t), C2)]
    assert dsolve(eq36) == sol36
    assert checksysodesol(eq36, sol36) == (True, [0, 0])

    eq37 = [Eq(Derivative(f(t), t), I*g(t)), Eq(Derivative(g(t), t), I*f(t))]
    sol37 = [Eq(f(t), -C1*exp(-I*t) + C2*exp(I*t)), Eq(g(t), C1*exp(-I*t) + C2*exp(I*t))]
    assert dsolve(eq37) == sol37
    assert checksysodesol(eq37, sol37) == (True, [0, 0])

    # Multiple systems
    eq1 = [Eq(Derivative(f(t), t)**2, g(t)**2), Eq(-f(t) + Derivative(g(t), t), 0)]
    sol1 = [[Eq(f(t), -C1*sin(t) - C2*cos(t)),
             Eq(g(t), C1*cos(t) - C2*sin(t))],
            [Eq(f(t), -C1*exp(-t) + C2*exp(t)),
             Eq(g(t), C1*exp(-t) + C2*exp(t))]]
    assert dsolve(eq1) == sol1
    for sol in sol1:
        assert checksysodesol(eq1, sol) == (True, [0, 0])


def test_sysode_linear_neq_order1_type2():

    f, g, h, k = symbols('f g h k', cls=Function)
    x, t, a, b, c, d, y = symbols('x t a b c d y')
    k1, k2 = symbols('k1 k2')


    eqs1 = [Eq(Derivative(f(x), x), f(x) + g(x) + 5),
            Eq(Derivative(g(x), x), -f(x) - g(x) + 7)]
    sol1 = [Eq(f(x), C1 + C2 + 6*x**2 + x*(C2 + 5)),
            Eq(g(x), -C1 - 6*x**2 - x*(C2 - 7))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    eqs2 = [Eq(Derivative(f(x), x), f(x) + g(x) + 5),
            Eq(Derivative(g(x), x), f(x) + g(x) + 7)]
    sol2 = [Eq(f(x), -C1 + C2*exp(2*x) - x - 3),
            Eq(g(x), C1 + C2*exp(2*x) + x - 3)]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    eqs3 = [Eq(Derivative(f(x), x), f(x) + 5),
            Eq(Derivative(g(x), x), f(x) + 7)]
    sol3 = [Eq(f(x), C1*exp(x) - 5),
            Eq(g(x), C1*exp(x) + C2 + 2*x - 5)]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    eqs4 = [Eq(Derivative(f(x), x), f(x) + exp(x)),
            Eq(Derivative(g(x), x), x*exp(x) + f(x) + g(x))]
    sol4 = [Eq(f(x), C1*exp(x) + x*exp(x)),
            Eq(g(x), C1*x*exp(x) + C2*exp(x) + x**2*exp(x))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])

    eqs5 = [Eq(Derivative(f(x), x), 5*x + f(x) + g(x)),
            Eq(Derivative(g(x), x), f(x) - g(x))]
    sol5 = [Eq(f(x), C1*(1 + sqrt(2))*exp(sqrt(2)*x) + C2*(1 - sqrt(2))*exp(-sqrt(2)*x) + x*Rational(-5, 2) +
             Rational(-5, 2)),
            Eq(g(x), C1*exp(sqrt(2)*x) + C2*exp(-sqrt(2)*x) + x*Rational(-5, 2))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])

    eqs6 = [Eq(Derivative(f(x), x), -9*f(x) - 4*g(x)),
            Eq(Derivative(g(x), x), -4*g(x)),
            Eq(Derivative(h(x), x), h(x) + exp(x))]
    sol6 = [Eq(f(x), C2*exp(-4*x)*Rational(-4, 5) + C1*exp(-9*x)),
            Eq(g(x), C2*exp(-4*x)),
            Eq(h(x), C3*exp(x) + x*exp(x))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0])

    # Regression test case for issue #8859
    # https://github.com/sympy/sympy/issues/8859
    eqs7 = [Eq(Derivative(f(t), t), 3*t + f(t)),
            Eq(Derivative(g(t), t), g(t))]
    sol7 = [Eq(f(t), C1*exp(t) - 3*t - 3),
            Eq(g(t), C2*exp(t))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    # Regression test case for issue #8567
    # https://github.com/sympy/sympy/issues/8567
    eqs8 = [Eq(Derivative(f(t), t), f(t) + 2*g(t)),
            Eq(Derivative(g(t), t), -2*f(t) + g(t) + 2*exp(t))]
    sol8 = [Eq(f(t), C1*exp(t)*sin(2*t) + C2*exp(t)*cos(2*t)
                + exp(t)*sin(2*t)**2 + exp(t)*cos(2*t)**2),
            Eq(g(t), C1*exp(t)*cos(2*t) - C2*exp(t)*sin(2*t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])

    # Regression test case for issue #19150
    # https://github.com/sympy/sympy/issues/19150
    eqs9 = [Eq(Derivative(f(t), t), (c - 2*f(t) + g(t))/(a*b)),
            Eq(Derivative(g(t), t), (f(t) - 2*g(t) + h(t))/(a*b)),
            Eq(Derivative(h(t), t), (d + g(t) - 2*h(t))/(a*b))]
    sol9 = [Eq(f(t), -C1*exp(-2*t/(a*b)) + C2*exp(-t*(sqrt(2) + 2)/(a*b)) + C3*exp(-t*(2 - sqrt(2))/(a*b)) +
             Mul(Rational(1, 4), 3*c + d, evaluate=False)),
            Eq(g(t), -sqrt(2)*C2*exp(-t*(sqrt(2) + 2)/(a*b)) + sqrt(2)*C3*exp(-t*(2 - sqrt(2))/(a*b)) +
             Mul(Rational(1, 2), c + d, evaluate=False)),
            Eq(h(t), C1*exp(-2*t/(a*b)) + C2*exp(-t*(sqrt(2) + 2)/(a*b)) + C3*exp(-t*(2 - sqrt(2))/(a*b)) +
             Mul(Rational(1, 4), c + 3*d, evaluate=False))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0, 0])

    # Regression test case for issue #16635
    # https://github.com/sympy/sympy/issues/16635
    eqs10 = [Eq(Derivative(f(t), t), 15*t + f(t) - g(t) - 10),
             Eq(Derivative(g(t), t), -15*t + f(t) - g(t) - 5)]
    sol10 = [Eq(f(t), C1 + C2 + 5*t**3 + 5*t**2 + t*(C2 - 10)),
             Eq(g(t), C1 + 5*t**3 - 10*t**2 + t*(C2 - 5))]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])

    # Multiple solutions
    eqs11 = [Eq(Derivative(f(t), t)**2 - 2*Derivative(f(t), t) + 1, 4),
             Eq(-y*f(t) + Derivative(g(t), t), 0)]
    sol11 = [[Eq(f(t), C1 - t), Eq(g(t), C1*t*y + C2*y + t**2*y*Rational(-1, 2))],
             [Eq(f(t), C1 + 3*t), Eq(g(t), C1*t*y + C2*y + t**2*y*Rational(3, 2))]]
    assert dsolve(eqs11) == sol11
    for s11 in sol11:
        assert checksysodesol(eqs11, s11) == (True, [0, 0])

    # test case for issue #19831
    # https://github.com/sympy/sympy/issues/19831
    n = symbols('n', positive=True)
    x0 = symbols('x_0')
    t0 = symbols('t_0')
    x_0 = symbols('x_0')
    t_0 = symbols('t_0')
    t = symbols('t')
    x = Function('x')
    y = Function('y')
    T = symbols('T')

    eqs12 = [Eq(Derivative(y(t), t), x(t)),
             Eq(Derivative(x(t), t), n*(y(t) + 1))]
    sol12 = [Eq(y(t), C1*exp(sqrt(n)*t)*n**Rational(-1, 2) - C2*exp(-sqrt(n)*t)*n**Rational(-1, 2) - 1),
             Eq(x(t), C1*exp(sqrt(n)*t) + C2*exp(-sqrt(n)*t))]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])

    sol12b = [
        Eq(y(t), (T*exp(-sqrt(n)*t_0)/2 + exp(-sqrt(n)*t_0)/2 +
            x_0*exp(-sqrt(n)*t_0)/(2*sqrt(n)))*exp(sqrt(n)*t) +
            (T*exp(sqrt(n)*t_0)/2 + exp(sqrt(n)*t_0)/2 -
                x_0*exp(sqrt(n)*t_0)/(2*sqrt(n)))*exp(-sqrt(n)*t) - 1),
        Eq(x(t), (T*sqrt(n)*exp(-sqrt(n)*t_0)/2 + sqrt(n)*exp(-sqrt(n)*t_0)/2
            + x_0*exp(-sqrt(n)*t_0)/2)*exp(sqrt(n)*t)
                - (T*sqrt(n)*exp(sqrt(n)*t_0)/2 + sqrt(n)*exp(sqrt(n)*t_0)/2 -
                    x_0*exp(sqrt(n)*t_0)/2)*exp(-sqrt(n)*t))
          ]
    assert dsolve(eqs12, ics={y(t0): T, x(t0): x0}) == sol12b
    assert checksysodesol(eqs12, sol12b) == (True, [0, 0])

    #Test cases added for the issue 19763
    #https://github.com/sympy/sympy/issues/19763

    eq13 = [Eq(Derivative(f(t), t), f(t) + g(t) + 9),
            Eq(Derivative(g(t), t), 2*f(t) + 5*g(t) + 23)]
    sol13 = [Eq(f(t), -C1*(2 + sqrt(6))*exp(t*(3 - sqrt(6)))/2 - C2*(2 - sqrt(6))*exp(t*(sqrt(6) + 3))/2 -
              Rational(22,3)),
             Eq(g(t), C1*exp(t*(3 - sqrt(6))) + C2*exp(t*(sqrt(6) + 3)) - Rational(5,3))]
    assert dsolve(eq13) == sol13
    assert checksysodesol(eq13, sol13) == (True, [0, 0])

    eq14 = [Eq(Derivative(f(t), t), f(t) + g(t) + 81),
            Eq(Derivative(g(t), t), -2*f(t) + g(t) + 23)]
    sol14 = [Eq(f(t), sqrt(2)*C1*exp(t)*sin(sqrt(2)*t)/2
                    + sqrt(2)*C2*exp(t)*cos(sqrt(2)*t)/2
                    - 58*sin(sqrt(2)*t)**2/3 - 58*cos(sqrt(2)*t)**2/3),
             Eq(g(t), C1*exp(t)*cos(sqrt(2)*t) - C2*exp(t)*sin(sqrt(2)*t)
                    - 185*sin(sqrt(2)*t)**2/3 - 185*cos(sqrt(2)*t)**2/3)]
    assert dsolve(eq14) == sol14
    assert checksysodesol(eq14, sol14) == (True, [0,0])

    eq15 = [Eq(Derivative(f(t), t), f(t) + 2*g(t) + k1),
            Eq(Derivative(g(t), t), 3*f(t) + 4*g(t) + k2)]
    sol15 = [Eq(f(t), -C1*(3 - sqrt(33))*exp(t*(5 + sqrt(33))/2)/6 -
              C2*(3 + sqrt(33))*exp(t*(5 - sqrt(33))/2)/6 + 2*k1 - k2),
             Eq(g(t), C1*exp(t*(5 + sqrt(33))/2) + C2*exp(t*(5 - sqrt(33))/2) -
              Mul(Rational(1,2), 3*k1 - k2, evaluate = False))]
    assert dsolve(eq15) == sol15
    assert checksysodesol(eq15, sol15) == (True, [0,0])

    eq16 = [Eq(Derivative(f(t), t), k1),
            Eq(Derivative(g(t), t), k2)]
    sol16 = [Eq(f(t), C1 + k1*t),
             Eq(g(t), C2 + k2*t)]
    assert dsolve(eq16) == sol16
    assert checksysodesol(eq16, sol16) == (True, [0,0])

    eq17 = [Eq(Derivative(f(t), t), 0),
            Eq(Derivative(g(t), t), c*f(t) + k2)]
    sol17 = [Eq(f(t), C1),
             Eq(g(t), C2*c + t*(C1*c + k2))]
    assert dsolve(eq17) == sol17
    assert checksysodesol(eq17 , sol17) == (True , [0,0])

    eq18 = [Eq(Derivative(f(t), t), k1),
            Eq(Derivative(g(t), t), f(t) + k2)]
    sol18 = [Eq(f(t), C1 + k1*t),
             Eq(g(t), C2 + k1*t**2/2 + t*(C1 + k2))]
    assert dsolve(eq18) == sol18
    assert checksysodesol(eq18 , sol18) == (True , [0,0])

    eq19 = [Eq(Derivative(f(t), t), k1),
            Eq(Derivative(g(t), t), f(t) + 2*g(t) + k2)]
    sol19 = [Eq(f(t), -2*C1 + k1*t),
            Eq(g(t), C1 + C2*exp(2*t) - k1*t/2 - Mul(Rational(1,4), k1 + 2*k2 , evaluate = False))]
    assert dsolve(eq19) == sol19
    assert checksysodesol(eq19 , sol19) == (True , [0,0])

    eq20 = [Eq(diff(f(t), t), f(t) + k1),
            Eq(diff(g(t), t), k2)]
    sol20 = [Eq(f(t), C1*exp(t) - k1),
            Eq(g(t), C2 + k2*t)]
    assert dsolve(eq20) == sol20
    assert checksysodesol(eq20 , sol20) == (True , [0,0])

    eq21 = [Eq(diff(f(t), t), g(t) + k1),
            Eq(diff(g(t), t), 0)]
    sol21 = [Eq(f(t), C1 + t*(C2 + k1)),
            Eq(g(t), C2)]
    assert dsolve(eq21) == sol21
    assert checksysodesol(eq21 , sol21) == (True , [0,0])

    eq22 = [Eq(Derivative(f(t), t), f(t) + 2*g(t) + k1),
            Eq(Derivative(g(t), t), k2)]
    sol22 = [Eq(f(t), -2*C1 + C2*exp(t) - k1 - 2*k2*t - 2*k2),
            Eq(g(t), C1 + k2*t)]
    assert dsolve(eq22) == sol22
    assert checksysodesol(eq22 , sol22) == (True , [0,0])

    eq23 = [Eq(Derivative(f(t), t), g(t) + k1),
            Eq(Derivative(g(t), t), 2*g(t) + k2)]
    sol23 = [Eq(f(t), C1 + C2*exp(2*t)/2 - k2/4 + t*(2*k1 - k2)/2),
            Eq(g(t), C2*exp(2*t) - k2/2)]
    assert dsolve(eq23) == sol23
    assert checksysodesol(eq23 , sol23) == (True , [0,0])

    eq24 = [Eq(Derivative(f(t), t), f(t) + k1),
            Eq(Derivative(g(t), t), 2*f(t) + k2)]
    sol24 = [Eq(f(t), C1*exp(t)/2 - k1),
            Eq(g(t), C1*exp(t) + C2 - 2*k1 - t*(2*k1 - k2))]
    assert dsolve(eq24) == sol24
    assert checksysodesol(eq24 , sol24) == (True , [0,0])

    eq25 = [Eq(Derivative(f(t), t), f(t) + 2*g(t) + k1),
            Eq(Derivative(g(t), t), 3*f(t) + 6*g(t) + k2)]
    sol25 = [Eq(f(t), -2*C1 + C2*exp(7*t)/3 + 2*t*(3*k1 - k2)/7 -
              Mul(Rational(1,49), k1 + 2*k2 , evaluate = False)),
             Eq(g(t), C1 + C2*exp(7*t) - t*(3*k1 - k2)/7 -
              Mul(Rational(3,49), k1 + 2*k2 , evaluate = False))]
    assert dsolve(eq25) == sol25
    assert checksysodesol(eq25 , sol25) == (True , [0,0])

    eq26 = [Eq(Derivative(f(t), t), 2*f(t) - g(t) + k1),
            Eq(Derivative(g(t), t), 4*f(t) - 2*g(t) + 2*k1)]
    sol26 = [Eq(f(t), C1 + 2*C2 + t*(2*C1 + k1)),
            Eq(g(t), 4*C2 + t*(4*C1 + 2*k1))]
    assert dsolve(eq26) == sol26
    assert checksysodesol(eq26 , sol26) == (True , [0,0])

    # Test Case added for issue #22715
    # https://github.com/sympy/sympy/issues/22715

    eq27 = [Eq(diff(x(t),t),-1*y(t)+10), Eq(diff(y(t),t),5*x(t)-2*y(t)+3)]
    sol27 = [Eq(x(t), (C1/5 - 2*C2/5)*exp(-t)*cos(2*t)
                    - (2*C1/5 + C2/5)*exp(-t)*sin(2*t)
                    + 17*sin(2*t)**2/5 + 17*cos(2*t)**2/5),
            Eq(y(t), C1*exp(-t)*cos(2*t) - C2*exp(-t)*sin(2*t)
                    + 10*sin(2*t)**2 + 10*cos(2*t)**2)]
    assert dsolve(eq27) == sol27
    assert checksysodesol(eq27 , sol27) == (True , [0,0])


def test_sysode_linear_neq_order1_type3():

    f, g, h, k, x0 , y0 = symbols('f g h k x0 y0', cls=Function)
    x, t, a = symbols('x t a')
    r = symbols('r', real=True)

    eqs1 = [Eq(Derivative(f(r), r), r*g(r) + f(r)),
            Eq(Derivative(g(r), r), -r*f(r) + g(r))]
    sol1 = [Eq(f(r), C1*exp(r)*sin(r**2/2) + C2*exp(r)*cos(r**2/2)),
            Eq(g(r), C1*exp(r)*cos(r**2/2) - C2*exp(r)*sin(r**2/2))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    eqs2 = [Eq(Derivative(f(x), x), x**2*g(x) + x*f(x)),
            Eq(Derivative(g(x), x), 2*x**2*f(x) + (3*x**2 + x)*g(x))]
    sol2 = [Eq(f(x), (sqrt(17)*C1/17 + C2*(17 - 3*sqrt(17))/34)*exp(x**3*(3 + sqrt(17))/6 + x**2/2) -
             exp(x**3*(3 - sqrt(17))/6 + x**2/2)*(sqrt(17)*C1/17 + C2*(3*sqrt(17) + 17)*Rational(-1, 34))),
            Eq(g(x), exp(x**3*(3 - sqrt(17))/6 + x**2/2)*(C1*(17 - 3*sqrt(17))/34 + sqrt(17)*C2*Rational(-2,
             17)) + exp(x**3*(3 + sqrt(17))/6 + x**2/2)*(C1*(3*sqrt(17) + 17)/34 + sqrt(17)*C2*Rational(2, 17)))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    eqs3 = [Eq(f(x).diff(x), x*f(x) + g(x)),
            Eq(g(x).diff(x), -f(x) + x*g(x))]
    sol3 = [Eq(f(x), (C1/2 + I*C2/2)*exp(x**2/2 - I*x) + exp(x**2/2 + I*x)*(C1/2 + I*C2*Rational(-1, 2))),
            Eq(g(x), (I*C1/2 + C2/2)*exp(x**2/2 + I*x) - exp(x**2/2 - I*x)*(I*C1/2 + C2*Rational(-1, 2)))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    eqs4 = [Eq(f(x).diff(x), x*(f(x) + g(x) + h(x))), Eq(g(x).diff(x), x*(f(x) + g(x) + h(x))),
            Eq(h(x).diff(x), x*(f(x) + g(x) + h(x)))]
    sol4 = [Eq(f(x), -C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2)),
            Eq(g(x), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2)),
            Eq(h(x), -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0])

    eqs5 = [Eq(f(x).diff(x), x**2*(f(x) + g(x) + h(x))), Eq(g(x).diff(x), x**2*(f(x) + g(x) + h(x))),
            Eq(h(x).diff(x), x**2*(f(x) + g(x) + h(x)))]
    sol5 = [Eq(f(x), -C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(x**3)),
            Eq(g(x), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(x**3)),
            Eq(h(x), -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(x**3))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0])

    eqs6 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x) + k(x))),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x) + k(x))),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x) + k(x))),
            Eq(Derivative(k(x), x), x*(f(x) + g(x) + h(x) + k(x)))]
    sol6 = [Eq(f(x), -C1/4 - C2/4 - C3/4 + 3*C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2)),
            Eq(g(x), 3*C1/4 - C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2)),
            Eq(h(x), -C1/4 + 3*C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2)),
            Eq(k(x), -C1/4 - C2/4 + 3*C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 + C4/4)*exp(2*x**2))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])

    y = symbols("y", real=True)

    eqs7 = [Eq(Derivative(f(y), y), y*f(y) + g(y)),
            Eq(Derivative(g(y), y), y*g(y) - f(y))]
    sol7 = [Eq(f(y), C1*exp(y**2/2)*sin(y) + C2*exp(y**2/2)*cos(y)),
            Eq(g(y), C1*exp(y**2/2)*cos(y) - C2*exp(y**2/2)*sin(y))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    #Test cases added for the issue 19763
    #https://github.com/sympy/sympy/issues/19763

    eqs8 = [Eq(Derivative(f(t), t), 5*t*f(t) + 2*h(t)),
            Eq(Derivative(h(t), t), 2*f(t) + 5*t*h(t))]
    sol8 = [Eq(f(t), Mul(-1, (C1/2 - C2/2), evaluate = False)*exp(5*t**2/2 - 2*t) + (C1/2 + C2/2)*exp(5*t**2/2 + 2*t)),
            Eq(h(t), (C1/2 - C2/2)*exp(5*t**2/2 - 2*t) + (C1/2 + C2/2)*exp(5*t**2/2 + 2*t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])

    eqs9 = [Eq(diff(f(t), t), 5*t*f(t) + t**2*g(t)),
            Eq(diff(g(t), t), -t**2*f(t) + 5*t*g(t))]
    sol9 = [Eq(f(t), (C1/2 - I*C2/2)*exp(I*t**3/3 + 5*t**2/2) + (C1/2 + I*C2/2)*exp(-I*t**3/3 + 5*t**2/2)),
            Eq(g(t), Mul(-1, (I*C1/2 - C2/2) , evaluate = False)*exp(-I*t**3/3 + 5*t**2/2) + (I*C1/2 + C2/2)*exp(I*t**3/3 + 5*t**2/2))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9 , sol9) == (True , [0,0])

    eqs10 = [Eq(diff(f(t), t), t**2*g(t) + 5*t*f(t)),
             Eq(diff(g(t), t), -t**2*f(t) + (9*t**2 + 5*t)*g(t))]
    sol10 = [Eq(f(t), (C1*(77 - 9*sqrt(77))/154 + sqrt(77)*C2/77)*exp(t**3*(sqrt(77) + 9)/6 + 5*t**2/2) + (C1*(77 + 9*sqrt(77))/154 - sqrt(77)*C2/77)*exp(t**3*(9 - sqrt(77))/6 + 5*t**2/2)),
             Eq(g(t), (sqrt(77)*C1/77 + C2*(77 - 9*sqrt(77))/154)*exp(t**3*(9 - sqrt(77))/6 + 5*t**2/2) - (sqrt(77)*C1/77 - C2*(77 + 9*sqrt(77))/154)*exp(t**3*(sqrt(77) + 9)/6 + 5*t**2/2))]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10 , sol10) == (True , [0,0])

    eqs11 = [Eq(diff(f(t), t), 5*t*f(t) + t**2*g(t)),
             Eq(diff(g(t), t), (1-t**2)*f(t) + (5*t + 9*t**2)*g(t))]
    sol11 = [Eq(f(t), C1*x0(t) + C2*x0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t)),
             Eq(g(t), C1*y0(t) + C2*(y0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t) + exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)))]
    assert dsolve(eqs11) == sol11

@slow
def test_sysode_linear_neq_order1_type4():

    f, g, h, k = symbols('f g h k', cls=Function)
    x, t, a = symbols('x t a')
    r = symbols('r', real=True)

    eqs1 = [Eq(diff(f(r), r), f(r) + r*g(r) + r**2), Eq(diff(g(r), r), -r*f(r) + g(r) + r)]
    sol1 = [Eq(f(r), C1*exp(r)*sin(r**2/2) + C2*exp(r)*cos(r**2/2) + exp(r)*sin(r**2/2)*Integral(r**2*exp(-r)*sin(r**2/2) +
                r*exp(-r)*cos(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r**2*exp(-r)*cos(r**2/2) - r*exp(-r)*sin(r**2/2), r)),
            Eq(g(r), C1*exp(r)*cos(r**2/2) - C2*exp(r)*sin(r**2/2) - exp(r)*sin(r**2/2)*Integral(r**2*exp(-r)*cos(r**2/2) -
                r*exp(-r)*sin(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r**2*exp(-r)*sin(r**2/2) + r*exp(-r)*cos(r**2/2), r))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    eqs2 = [Eq(diff(f(r), r), f(r) + r*g(r) + r), Eq(diff(g(r), r), -r*f(r) + g(r) + log(r))]
    sol2 = [Eq(f(r), C1*exp(r)*sin(r**2/2) + C2*exp(r)*cos(r**2/2) + exp(r)*sin(r**2/2)*Integral(r*exp(-r)*sin(r**2/2) +
                exp(-r)*log(r)*cos(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r*exp(-r)*cos(r**2/2) - exp(-r)*log(r)*sin(
                r**2/2), r)),
            Eq(g(r), C1*exp(r)*cos(r**2/2) - C2*exp(r)*sin(r**2/2) - exp(r)*sin(r**2/2)*Integral(r*exp(-r)*cos(r**2/2) -
                exp(-r)*log(r)*sin(r**2/2), r) + exp(r)*cos(r**2/2)*Integral(r*exp(-r)*sin(r**2/2) + exp(-r)*log(r)*cos(
                r**2/2), r))]
    # XXX: dsolve hangs for this in integration
    assert dsolve_system(eqs2, simplify=False, doit=False) == [sol2]
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    eqs3 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x)) + x),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x)) + x),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x)) + 1)]
    sol3 = [Eq(f(x), C1*Rational(-1, 3) + C2*Rational(-1, 3) + C3*Rational(2, 3) + x**2/6 + x*Rational(-1, 3) +
             (C1/3 + C2/3 + C3/3)*exp(x**2*Rational(3, 2)) +
             sqrt(6)*sqrt(pi)*erf(sqrt(6)*x/2)*exp(x**2*Rational(3, 2))/18 + Rational(-2, 9)),
            Eq(g(x), C1*Rational(2, 3) + C2*Rational(-1, 3) + C3*Rational(-1, 3) + x**2/6 + x*Rational(-1, 3) +
             (C1/3 + C2/3 + C3/3)*exp(x**2*Rational(3, 2)) +
             sqrt(6)*sqrt(pi)*erf(sqrt(6)*x/2)*exp(x**2*Rational(3, 2))/18 + Rational(-2, 9)),
            Eq(h(x), C1*Rational(-1, 3) + C2*Rational(2, 3) + C3*Rational(-1, 3) + x**2*Rational(-1, 3) +
             x*Rational(2, 3) + (C1/3 + C2/3 + C3/3)*exp(x**2*Rational(3, 2)) +
             sqrt(6)*sqrt(pi)*erf(sqrt(6)*x/2)*exp(x**2*Rational(3, 2))/18 + Rational(-2, 9))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0])

    eqs4 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x)) + sin(x)),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x)) + sin(x)),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x)) + sin(x))]
    sol4 = [Eq(f(x), C1*Rational(-1, 3) + C2*Rational(-1, 3) + C3*Rational(2, 3) + (C1/3 + C2/3 +
             C3/3)*exp(x**2*Rational(3, 2)) + Integral(sin(x)*exp(x**2*Rational(-3, 2)), x)*exp(x**2*Rational(3,
             2))),
            Eq(g(x), C1*Rational(2, 3) + C2*Rational(-1, 3) + C3*Rational(-1, 3) + (C1/3 + C2/3 +
             C3/3)*exp(x**2*Rational(3, 2)) + Integral(sin(x)*exp(x**2*Rational(-3, 2)), x)*exp(x**2*Rational(3,
             2))),
            Eq(h(x), C1*Rational(-1, 3) + C2*Rational(2, 3) + C3*Rational(-1, 3) + (C1/3 + C2/3 +
             C3/3)*exp(x**2*Rational(3, 2)) + Integral(sin(x)*exp(x**2*Rational(-3, 2)), x)*exp(x**2*Rational(3,
             2)))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0])

    eqs5 = [Eq(Derivative(f(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(g(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(h(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(k(x), x), x*(f(x) + g(x) + h(x) + k(x) + 1))]
    sol5 = [Eq(f(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(3, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4)),
            Eq(g(x), C1*Rational(3, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4)),
            Eq(h(x), C1*Rational(-1, 4) + C2*Rational(3, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4)),
            Eq(k(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(3, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(2*x**2) + Rational(-1, 4))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0, 0])

    eqs6 = [Eq(Derivative(f(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(g(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(h(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1)),
            Eq(Derivative(k(x), x), x**2*(f(x) + g(x) + h(x) + k(x) + 1))]
    sol6 = [Eq(f(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(3, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4)),
            Eq(g(x), C1*Rational(3, 4) + C2*Rational(-1, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4)),
            Eq(h(x), C1*Rational(-1, 4) + C2*Rational(3, 4) + C3*Rational(-1, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4)),
            Eq(k(x), C1*Rational(-1, 4) + C2*Rational(-1, 4) + C3*Rational(3, 4) + C4*Rational(-1, 4) + (C1/4 +
             C2/4 + C3/4 + C4/4)*exp(x**3*Rational(4, 3)) + Rational(-1, 4))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])

    eqs7 = [Eq(Derivative(f(x), x), (f(x) + g(x) + h(x))*log(x) + sin(x)), Eq(Derivative(g(x), x), (f(x) + g(x)
            + h(x))*log(x) + sin(x)), Eq(Derivative(h(x), x), (f(x) + g(x) + h(x))*log(x) + sin(x))]
    sol7 = [Eq(f(x), -C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 +
                C3/3)*exp(x*(3*log(x) - 3)) + exp(x*(3*log(x) -
                    3))*Integral(exp(3*x)*exp(-3*x*log(x))*sin(x), x)),
            Eq(g(x), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 +
                C3/3)*exp(x*(3*log(x) - 3)) + exp(x*(3*log(x) -
                    3))*Integral(exp(3*x)*exp(-3*x*log(x))*sin(x), x)),
            Eq(h(x), -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 +
                C3/3)*exp(x*(3*log(x) - 3)) + exp(x*(3*log(x) -
                    3))*Integral(exp(3*x)*exp(-3*x*log(x))*sin(x), x))]
    with dotprodsimp(True):
        assert dsolve(eqs7, simplify=False, doit=False) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0, 0])

    eqs8 = [Eq(Derivative(f(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) + sin(x)), Eq(Derivative(g(x), x), (f(x)
            + g(x) + h(x) + k(x))*log(x) + sin(x)), Eq(Derivative(h(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) +
            sin(x)), Eq(Derivative(k(x), x), (f(x) + g(x) + h(x) + k(x))*log(x) + sin(x))]
    sol8 = [Eq(f(x), -C1/4 - C2/4 - C3/4 + 3*C4/4 + (C1/4 + C2/4 + C3/4 +
                C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) -
                    4))*Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x)),
            Eq(g(x), 3*C1/4 - C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 +
                C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) -
                    4))*Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x)),
            Eq(h(x), -C1/4 + 3*C2/4 - C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 +
                C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) -
                    4))*Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x)),
            Eq(k(x), -C1/4 - C2/4 + 3*C3/4 - C4/4 + (C1/4 + C2/4 + C3/4 +
                C4/4)*exp(x*(4*log(x) - 4)) + exp(x*(4*log(x) -
                    4))*Integral(exp(4*x)*exp(-4*x*log(x))*sin(x), x))]
    with dotprodsimp(True):
        assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0, 0, 0])


def test_sysode_linear_neq_order1_type5_type6():
    f, g = symbols("f g", cls=Function)
    x, x_ = symbols("x x_")

    # Type 5
    eqs1 = [Eq(Derivative(f(x), x), (2*f(x) + g(x))/x), Eq(Derivative(g(x), x), (f(x) + 2*g(x))/x)]
    sol1 = [Eq(f(x), -C1*x + C2*x**3), Eq(g(x), C1*x + C2*x**3)]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    # Type 6
    eqs2 = [Eq(Derivative(f(x), x), (2*f(x) + g(x) + 1)/x),
            Eq(Derivative(g(x), x), (x + f(x) + 2*g(x))/x)]
    sol2 = [Eq(f(x), C2*x**3 - x*(C1 + Rational(1, 4)) + x*log(x)*Rational(-1, 2) + Rational(-2, 3)),
            Eq(g(x), C2*x**3 + x*log(x)/2 + x*(C1 + Rational(-1, 4)) + Rational(1, 3))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])


def test_higher_order_to_first_order():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')

    eqs1 = [Eq(Derivative(f(x), (x, 2)), 2*f(x) + g(x)),
            Eq(Derivative(g(x), (x, 2)), -f(x))]
    sol1 = [Eq(f(x), -C2*x*exp(-x) + C3*x*exp(x) - (C1 - C2)*exp(-x) + (C3 + C4)*exp(x)),
            Eq(g(x), C2*x*exp(-x) - C3*x*exp(x) + (C1 + C2)*exp(-x) + (C3 - C4)*exp(x))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])

    eqs2 = [Eq(f(x).diff(x, 2), 0), Eq(g(x).diff(x, 2), f(x))]
    sol2 = [Eq(f(x), C1 + C2*x), Eq(g(x), C1*x**2/2 + C2*x**3/6 + C3 + C4*x)]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    eqs3 = [Eq(Derivative(f(x), (x, 2)), 2*f(x)),
            Eq(Derivative(g(x), (x, 2)), -f(x) + 2*g(x))]
    sol3 = [Eq(f(x), 4*C1*exp(-sqrt(2)*x) + 4*C2*exp(sqrt(2)*x)),
            Eq(g(x), sqrt(2)*C1*x*exp(-sqrt(2)*x) - sqrt(2)*C2*x*exp(sqrt(2)*x) + (C1 +
             sqrt(2)*C4)*exp(-sqrt(2)*x) + (C2 - sqrt(2)*C3)*exp(sqrt(2)*x))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    eqs4 = [Eq(Derivative(f(x), (x, 2)), 2*f(x) + g(x)),
            Eq(Derivative(g(x), (x, 2)), 2*g(x))]
    sol4 = [Eq(f(x), C1*x*exp(sqrt(2)*x)/4 + C3*x*exp(-sqrt(2)*x)/4 + (C2/4 + sqrt(2)*C3/8)*exp(-sqrt(2)*x) -
             exp(sqrt(2)*x)*(sqrt(2)*C1/8 + C4*Rational(-1, 4))),
            Eq(g(x), sqrt(2)*C1*exp(sqrt(2)*x)/2 + sqrt(2)*C3*exp(-sqrt(2)*x)*Rational(-1, 2))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])

    eqs5 = [Eq(f(x).diff(x, 2), f(x)), Eq(g(x).diff(x, 2), f(x))]
    sol5 = [Eq(f(x), -C1*exp(-x) + C2*exp(x)), Eq(g(x), -C1*exp(-x) + C2*exp(x) + C3 + C4*x)]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])

    eqs6 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x)),
            Eq(Derivative(g(x), (x, 2)), -f(x) - g(x))]
    sol6 = [Eq(f(x), C1 + C2*x**2/2 + C2 + C4*x**3/6 + x*(C3 + C4)),
            Eq(g(x), -C1 + C2*x**2*Rational(-1, 2) - C3*x + C4*x**3*Rational(-1, 6))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])

    eqs7 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x) + 1),
            Eq(Derivative(g(x), (x, 2)), f(x) + g(x) + 1)]
    sol7 = [Eq(f(x), -C1 - C2*x + sqrt(2)*C3*exp(sqrt(2)*x)/2 + sqrt(2)*C4*exp(-sqrt(2)*x)*Rational(-1, 2) +
             Rational(-1, 2)),
            Eq(g(x), C1 + C2*x + sqrt(2)*C3*exp(sqrt(2)*x)/2 + sqrt(2)*C4*exp(-sqrt(2)*x)*Rational(-1, 2) +
             Rational(-1, 2))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])

    eqs8 = [Eq(Derivative(f(x), (x, 2)), f(x) + g(x) + 1),
            Eq(Derivative(g(x), (x, 2)), -f(x) - g(x) + 1)]
    sol8 = [Eq(f(x), C1 + C2 + C4*x**3/6 + x**4/12 + x**2*(C2/2 + Rational(1, 2)) + x*(C3 + C4)),
            Eq(g(x), -C1 - C3*x + C4*x**3*Rational(-1, 6) + x**4*Rational(-1, 12) - x**2*(C2/2 + Rational(-1,
             2)))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])

    x, y = symbols('x, y', cls=Function)
    t, l = symbols('t, l')

    eqs10 = [Eq(Derivative(x(t), (t, 2)), 5*x(t) + 43*y(t)),
             Eq(Derivative(y(t), (t, 2)), x(t) + 9*y(t))]
    sol10 = [Eq(x(t), C1*(61 - 9*sqrt(47))*sqrt(sqrt(47) + 7)*exp(-t*sqrt(sqrt(47) + 7))/2 + C2*sqrt(7 -
              sqrt(47))*(61 + 9*sqrt(47))*exp(-t*sqrt(7 - sqrt(47)))/2 + C3*(61 - 9*sqrt(47))*sqrt(sqrt(47) +
              7)*exp(t*sqrt(sqrt(47) + 7))*Rational(-1, 2) + C4*sqrt(7 - sqrt(47))*(61 + 9*sqrt(47))*exp(t*sqrt(7
              - sqrt(47)))*Rational(-1, 2)),
             Eq(y(t), C1*(7 - sqrt(47))*sqrt(sqrt(47) + 7)*exp(-t*sqrt(sqrt(47) + 7))*Rational(-1, 2) + C2*sqrt(7
              - sqrt(47))*(sqrt(47) + 7)*exp(-t*sqrt(7 - sqrt(47)))*Rational(-1, 2) + C3*(7 -
              sqrt(47))*sqrt(sqrt(47) + 7)*exp(t*sqrt(sqrt(47) + 7))/2 + C4*sqrt(7 - sqrt(47))*(sqrt(47) +
              7)*exp(t*sqrt(7 - sqrt(47)))/2)]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])

    eqs11 = [Eq(7*x(t) + Derivative(x(t), (t, 2)) - 9*Derivative(y(t), t), 0),
             Eq(7*y(t) + 9*Derivative(x(t), t) + Derivative(y(t), (t, 2)), 0)]
    sol11 = [Eq(y(t), C1*(9 - sqrt(109))*sin(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)/14 + C2*(9 -
              sqrt(109))*cos(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)*Rational(-1, 14) + C3*(9 +
              sqrt(109))*sin(sqrt(2)*t*sqrt(95 - 9*sqrt(109))/2)/14 + C4*(9 + sqrt(109))*cos(sqrt(2)*t*sqrt(95 -
              9*sqrt(109))/2)*Rational(-1, 14)),
             Eq(x(t), C1*(9 - sqrt(109))*cos(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)*Rational(-1, 14) + C2*(9 -
              sqrt(109))*sin(sqrt(2)*t*sqrt(9*sqrt(109) + 95)/2)*Rational(-1, 14) + C3*(9 +
              sqrt(109))*cos(sqrt(2)*t*sqrt(95 - 9*sqrt(109))/2)/14 + C4*(9 + sqrt(109))*sin(sqrt(2)*t*sqrt(95 -
              9*sqrt(109))/2)/14)]
    assert dsolve(eqs11) == sol11
    assert checksysodesol(eqs11, sol11) == (True, [0, 0])

    # Euler Systems
    # Note: To add examples of euler systems solver with non-homogeneous term.
    eqs13 = [Eq(Derivative(f(t), (t, 2)), Derivative(f(t), t)/t + f(t)/t**2 + g(t)/t**2),
             Eq(Derivative(g(t), (t, 2)), g(t)/t**2)]
    sol13 = [Eq(f(t), C1*(sqrt(5) + 3)*Rational(-1, 2)*t**(Rational(1, 2) +
                sqrt(5)*Rational(-1, 2)) + C2*t**(Rational(1, 2) +
                sqrt(5)/2)*(3 - sqrt(5))*Rational(-1, 2) - C3*t**(1 -
                sqrt(2))*(1 + sqrt(2)) - C4*t**(1 + sqrt(2))*(1 - sqrt(2))),
             Eq(g(t), C1*(1 + sqrt(5))*Rational(-1, 2)*t**(Rational(1, 2) +
                 sqrt(5)*Rational(-1, 2)) + C2*t**(Rational(1, 2) +
                     sqrt(5)/2)*(1 - sqrt(5))*Rational(-1, 2))]
    assert dsolve(eqs13) == sol13
    assert checksysodesol(eqs13, sol13) == (True, [0, 0])

    # Solving systems using dsolve separately
    eqs14 = [Eq(Derivative(f(t), (t, 2)), t*f(t)),
         Eq(Derivative(g(t), (t, 2)), t*g(t))]
    sol14 = [Eq(f(t), C1*airyai(t) + C2*airybi(t)),
         Eq(g(t), C3*airyai(t) + C4*airybi(t))]
    assert dsolve(eqs14) == sol14
    assert checksysodesol(eqs14, sol14) == (True, [0, 0])


    eqs15 = [Eq(Derivative(x(t), (t, 2)), t*(4*Derivative(x(t), t) + 8*Derivative(y(t), t))),
             Eq(Derivative(y(t), (t, 2)), t*(12*Derivative(x(t), t) - 6*Derivative(y(t), t)))]
    sol15 = [Eq(x(t), C1 - erf(sqrt(6)*t)*(sqrt(6)*sqrt(pi)*C2/33 + sqrt(6)*sqrt(pi)*C3*Rational(-1, 44)) +
              erfi(sqrt(5)*t)*(sqrt(5)*sqrt(pi)*C2*Rational(2, 55) + sqrt(5)*sqrt(pi)*C3*Rational(4, 55))),
             Eq(y(t), C4 + erf(sqrt(6)*t)*(sqrt(6)*sqrt(pi)*C2*Rational(2, 33) + sqrt(6)*sqrt(pi)*C3*Rational(-1,
              22)) + erfi(sqrt(5)*t)*(sqrt(5)*sqrt(pi)*C2*Rational(3, 110) + sqrt(5)*sqrt(pi)*C3*Rational(3, 55)))]
    assert dsolve(eqs15) == sol15
    assert checksysodesol(eqs15, sol15) == (True, [0, 0])


@slow
def test_higher_order_to_first_order_9():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')

    eqs9 = [f(x) + g(x) - 2*exp(I*x) + 2*Derivative(f(x), x) + Derivative(f(x), (x, 2)),
            f(x) + g(x) - 2*exp(I*x) + 2*Derivative(g(x), x) + Derivative(g(x), (x, 2))]
    sol9 =  [Eq(f(x), -C1 + C4*exp(-2*x)/2 - (C2/2 - C3/2)*exp(-x)*cos(x)
                    + (C2/2 + C3/2)*exp(-x)*sin(x) + 2*((1 - 2*I)*exp(I*x)*sin(x)**2/5)
                    + 2*((1 - 2*I)*exp(I*x)*cos(x)**2/5)),
            Eq(g(x), C1 - C4*exp(-2*x)/2 - (C2/2 - C3/2)*exp(-x)*cos(x)
                    + (C2/2 + C3/2)*exp(-x)*sin(x) + 2*((1 - 2*I)*exp(I*x)*sin(x)**2/5)
                    + 2*((1 - 2*I)*exp(I*x)*cos(x)**2/5))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])


def test_higher_order_to_first_order_12():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')

    x, y = symbols('x, y', cls=Function)
    t, l = symbols('t, l')

    eqs12 = [Eq(4*x(t) + Derivative(x(t), (t, 2)) + 8*Derivative(y(t), t), 0),
             Eq(4*y(t) - 8*Derivative(x(t), t) + Derivative(y(t), (t, 2)), 0)]
    sol12 = [Eq(y(t), C1*(2 - sqrt(5))*sin(2*t*sqrt(4*sqrt(5) + 9))*Rational(-1, 2) + C2*(2 -
              sqrt(5))*cos(2*t*sqrt(4*sqrt(5) + 9))/2 + C3*(2 + sqrt(5))*sin(2*t*sqrt(9 - 4*sqrt(5)))*Rational(-1,
              2) + C4*(2 + sqrt(5))*cos(2*t*sqrt(9 - 4*sqrt(5)))/2),
             Eq(x(t), C1*(2 - sqrt(5))*cos(2*t*sqrt(4*sqrt(5) + 9))*Rational(-1, 2) + C2*(2 -
              sqrt(5))*sin(2*t*sqrt(4*sqrt(5) + 9))*Rational(-1, 2) + C3*(2 + sqrt(5))*cos(2*t*sqrt(9 -
              4*sqrt(5)))/2 + C4*(2 + sqrt(5))*sin(2*t*sqrt(9 - 4*sqrt(5)))/2)]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])


def test_second_order_to_first_order_2():
    f, g = symbols("f g", cls=Function)
    x, t, x_, t_, d, a, m = symbols("x t x_ t_ d a m")

    eqs2 = [Eq(f(x).diff(x, 2), 2*(x*g(x).diff(x) - g(x))),
            Eq(g(x).diff(x, 2),-2*(x*f(x).diff(x) - f(x)))]
    sol2 = [Eq(f(x), C1*x + x*Integral(C2*exp(-x_)*exp(I*exp(2*x_))/2 + C2*exp(-x_)*exp(-I*exp(2*x_))/2 -
                I*C3*exp(-x_)*exp(I*exp(2*x_))/2 + I*C3*exp(-x_)*exp(-I*exp(2*x_))/2, (x_, log(x)))),
            Eq(g(x), C4*x + x*Integral(I*C2*exp(-x_)*exp(I*exp(2*x_))/2 - I*C2*exp(-x_)*exp(-I*exp(2*x_))/2 +
                C3*exp(-x_)*exp(I*exp(2*x_))/2 + C3*exp(-x_)*exp(-I*exp(2*x_))/2, (x_, log(x))))]
    # XXX: dsolve hangs for this in integration
    assert dsolve_system(eqs2, simplify=False, doit=False) == [sol2]
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])

    eqs3 = (Eq(diff(f(t),t,t), 9*t*diff(g(t),t)-9*g(t)), Eq(diff(g(t),t,t),7*t*diff(f(t),t)-7*f(t)))
    sol3 = [Eq(f(t), C1*t + t*Integral(C2*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/2 + C2*exp(-t_)*
                exp(-3*sqrt(7)*exp(2*t_)/2)/2 + 3*sqrt(7)*C3*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/14 -
                3*sqrt(7)*C3*exp(-t_)*exp(-3*sqrt(7)*exp(2*t_)/2)/14, (t_, log(t)))),
            Eq(g(t), C4*t + t*Integral(sqrt(7)*C2*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/6 - sqrt(7)*C2*exp(-t_)*
                exp(-3*sqrt(7)*exp(2*t_)/2)/6 + C3*exp(-t_)*exp(3*sqrt(7)*exp(2*t_)/2)/2 + C3*exp(-t_)*exp(-3*sqrt(7)*
                exp(2*t_)/2)/2, (t_, log(t))))]
    # XXX: dsolve hangs for this in integration
    assert dsolve_system(eqs3, simplify=False, doit=False) == [sol3]
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])

    # Regression Test case for sympy#19238
    # https://github.com/sympy/sympy/issues/19238
    # Note: When the doit method is removed, these particular types of systems
    # can be divided first so that we have lesser number of big matrices.
    eqs5 = [Eq(Derivative(g(t), (t, 2)), a*m),
            Eq(Derivative(f(t), (t, 2)), 0)]
    sol5 = [Eq(g(t), C1 + C2*t + a*m*t**2/2),
            Eq(f(t), C3 + C4*t)]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])

    # Type 2
    eqs6 = [Eq(Derivative(f(t), (t, 2)), f(t)/t**4),
            Eq(Derivative(g(t), (t, 2)), d*g(t)/t**4)]
    sol6 = [Eq(f(t), C1*sqrt(t**2)*exp(-1/t) - C2*sqrt(t**2)*exp(1/t)),
            Eq(g(t), C3*sqrt(t**2)*exp(-sqrt(d)/t)*d**Rational(-1, 2) -
             C4*sqrt(t**2)*exp(sqrt(d)/t)*d**Rational(-1, 2))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])


@slow
def test_second_order_to_first_order_slow1():
    f, g = symbols("f g", cls=Function)
    x, t, x_, t_, d, a, m = symbols("x t x_ t_ d a m")

    # Type 1

    eqs1 = [Eq(f(x).diff(x, 2), 2/x *(x*g(x).diff(x) - g(x))),
           Eq(g(x).diff(x, 2),-2/x *(x*f(x).diff(x) - f(x)))]
    sol1 = [Eq(f(x), C1*x + 2*C2*x*Ci(2*x) - C2*sin(2*x) - 2*C3*x*Si(2*x) - C3*cos(2*x)),
            Eq(g(x), -2*C2*x*Si(2*x) - C2*cos(2*x) - 2*C3*x*Ci(2*x) + C3*sin(2*x) + C4*x)]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])


def test_second_order_to_first_order_slow4():
    f, g = symbols("f g", cls=Function)
    x, t, x_, t_, d, a, m = symbols("x t x_ t_ d a m")

    eqs4 = [Eq(Derivative(f(t), (t, 2)), t*sin(t)*Derivative(g(t), t) - g(t)*sin(t)),
            Eq(Derivative(g(t), (t, 2)), t*sin(t)*Derivative(f(t), t) - f(t)*sin(t))]
    sol4 = [Eq(f(t), C1*t + t*Integral(C2*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*exp(-sin(exp(t_)))/2 +
                C2*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2 - C3*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*
                exp(-sin(exp(t_)))/2 +
                C3*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2, (t_, log(t)))),
            Eq(g(t), C4*t + t*Integral(-C2*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*exp(-sin(exp(t_)))/2 +
                C2*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2 + C3*exp(-t_)*exp(exp(t_)*cos(exp(t_)))*
                exp(-sin(exp(t_)))/2 + C3*exp(-t_)*exp(-exp(t_)*cos(exp(t_)))*exp(sin(exp(t_)))/2, (t_, log(t))))]
    # XXX: dsolve hangs for this in integration
    assert dsolve_system(eqs4, simplify=False, doit=False) == [sol4]
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])


def test_component_division():
    f, g, h, k = symbols('f g h k', cls=Function)
    x = symbols("x")
    funcs = [f(x), g(x), h(x), k(x)]

    eqs1 = [Eq(Derivative(f(x), x), 2*f(x)),
            Eq(Derivative(g(x), x), f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), h(x)**4 + k(x))]
    sol1 = [Eq(f(x), 2*C1*exp(2*x)),
            Eq(g(x), C1*exp(2*x) + C2),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C3**4*exp(4*x)/3 + C4*exp(x))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0, 0, 0])

    components1 = {((Eq(Derivative(f(x), x), 2*f(x)),), (Eq(Derivative(g(x), x), f(x)),)),
                   ((Eq(Derivative(h(x), x), h(x)),), (Eq(Derivative(k(x), x), h(x)**4 + k(x)),))}
    eqsdict1 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {h(x)}},
                {f(x): Eq(Derivative(f(x), x), 2*f(x)),
                g(x): Eq(Derivative(g(x), x), f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), h(x)**4 + k(x))})
    graph1 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), h(x))}]
    assert {tuple(tuple(scc) for scc in wcc) for wcc in _component_division(eqs1, funcs, x)} == components1
    assert _eqs2dict(eqs1, funcs) == eqsdict1
    assert [set(element) for element in _dict2graph(eqsdict1[0])] == graph1

    eqs2 = [Eq(Derivative(f(x), x), 2*f(x)),
            Eq(Derivative(g(x), x), f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    sol2 = [Eq(f(x), C1*exp(2*x)),
            Eq(g(x), C1*exp(2*x)/2 + C2),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C1**4*exp(8*x)/7 + C4*exp(x))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0, 0, 0])

    components2 = {frozenset([(Eq(Derivative(f(x), x), 2*f(x)),),
                    (Eq(Derivative(g(x), x), f(x)),),
                    (Eq(Derivative(k(x), x), f(x)**4 + k(x)),)]),
                   frozenset([(Eq(Derivative(h(x), x), h(x)),)])}
    eqsdict2 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                 {f(x): Eq(Derivative(f(x), x), 2*f(x)),
                  g(x): Eq(Derivative(g(x), x), f(x)),
                  h(x): Eq(Derivative(h(x), x), h(x)),
                  k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    graph2 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), f(x))}]
    assert {frozenset(tuple(scc) for scc in wcc) for wcc in _component_division(eqs2, funcs, x)} == components2
    assert _eqs2dict(eqs2, funcs) == eqsdict2
    assert [set(element) for element in _dict2graph(eqsdict2[0])] == graph2

    eqs3 = [Eq(Derivative(f(x), x), 2*f(x)),
            Eq(Derivative(g(x), x), x + f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    sol3 = [Eq(f(x), C1*exp(2*x)),
            Eq(g(x), C1*exp(2*x)/2 + C2 + x**2/2),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C1**4*exp(8*x)/7 + C4*exp(x))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0, 0])

    components3 = {frozenset([(Eq(Derivative(f(x), x), 2*f(x)),),
                    (Eq(Derivative(g(x), x), x + f(x)),),
                    (Eq(Derivative(k(x), x), f(x)**4 + k(x)),)]),
                    frozenset([(Eq(Derivative(h(x), x), h(x)),),])}
    eqsdict3 = ({f(x): set(), g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                {f(x): Eq(Derivative(f(x), x), 2*f(x)),
                g(x): Eq(Derivative(g(x), x), x + f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    graph3 = [{f(x), g(x), h(x), k(x)}, {(g(x), f(x)), (k(x), f(x))}]
    assert {frozenset(tuple(scc) for scc in wcc) for wcc in _component_division(eqs3, funcs, x)} == components3
    assert _eqs2dict(eqs3, funcs) == eqsdict3
    assert [set(l) for l in _dict2graph(eqsdict3[0])] == graph3

    # Note: To be uncommented when the default option to call dsolve first for
    # single ODE system can be rearranged. This can be done after the doit
    # option in dsolve is made False by default.

    eqs4 = [Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
            Eq(Derivative(g(x), x), f(x) + x*g(x) + x),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    sol4 = [Eq(f(x), (C1/2 - sqrt(2)*C2/2 - sqrt(2)*Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 +\
                sqrt(2)*x)/2, x)/2 + Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 - sqrt(2)*x*exp(-x**2/2 +\
                sqrt(2)*x)/2, x)/2)*exp(x**2/2 - sqrt(2)*x) + (C1/2 + sqrt(2)*C2/2 + sqrt(2)*Integral(x*exp(-x**2/2
                - sqrt(2)*x)/2 + x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 + Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2
                - sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2)*exp(x**2/2 + sqrt(2)*x)),
            Eq(g(x), (-sqrt(2)*C1/4 + C2/2 + Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 -\
                sqrt(2)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 - sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2,
                x)/4)*exp(x**2/2 - sqrt(2)*x) + (sqrt(2)*C1/4 + C2/2 + Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 +
                x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 + sqrt(2)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 -
                sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2, x)/4)*exp(x**2/2 + sqrt(2)*x)),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C4*exp(x) + exp(x)*Integral((C1*exp(x**2/2 - sqrt(2)*x)/2 + C1*exp(x**2/2 + sqrt(2)*x)/2 -
                sqrt(2)*C2*exp(x**2/2 - sqrt(2)*x)/2 + sqrt(2)*C2*exp(x**2/2 + sqrt(2)*x)/2 - sqrt(2)*exp(x**2/2 -
                sqrt(2)*x)*Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2 + exp(x**2/2 -
                sqrt(2)*x)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 - sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2,
                x)/2 + sqrt(2)*exp(x**2/2 + sqrt(2)*x)*Integral(x*exp(-x**2/2 - sqrt(2)*x)/2 + x*exp(-x**2/2 +
                sqrt(2)*x)/2, x)/2 + exp(x**2/2 + sqrt(2)*x)*Integral(sqrt(2)*x*exp(-x**2/2 - sqrt(2)*x)/2 -
                sqrt(2)*x*exp(-x**2/2 + sqrt(2)*x)/2, x)/2)**4*exp(-x), x))]
    components4 = {(frozenset([Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
                    Eq(Derivative(g(x), x), x*g(x) + x + f(x))]),
                    frozenset([Eq(Derivative(k(x), x), f(x)**4 + k(x)),])),
                    (frozenset([Eq(Derivative(h(x), x), h(x)),]),)}
    eqsdict4 = ({f(x): {g(x)}, g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                {f(x): Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
                g(x): Eq(Derivative(g(x), x), x*g(x) + x + f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    graph4 = [{f(x), g(x), h(x), k(x)}, {(f(x), g(x)), (g(x), f(x)), (k(x), f(x))}]
    assert {tuple(frozenset(scc) for scc in wcc) for wcc in _component_division(eqs4, funcs, x)} == components4
    assert _eqs2dict(eqs4, funcs) == eqsdict4
    assert [set(element) for element in _dict2graph(eqsdict4[0])] == graph4
    # XXX: dsolve hangs in integration here:
    assert dsolve_system(eqs4, simplify=False, doit=False) == [sol4]
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0, 0])

    eqs5 = [Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
            Eq(Derivative(g(x), x), x*g(x) + f(x)),
            Eq(Derivative(h(x), x), h(x)),
            Eq(Derivative(k(x), x), f(x)**4 + k(x))]
    sol5 = [Eq(f(x), (C1/2 - sqrt(2)*C2/2)*exp(x**2/2 - sqrt(2)*x) + (C1/2 + sqrt(2)*C2/2)*exp(x**2/2 + sqrt(2)*x)),
            Eq(g(x), (-sqrt(2)*C1/4 + C2/2)*exp(x**2/2 - sqrt(2)*x) + (sqrt(2)*C1/4 + C2/2)*exp(x**2/2 + sqrt(2)*x)),
            Eq(h(x), C3*exp(x)),
            Eq(k(x), C4*exp(x) + exp(x)*Integral((C1*exp(x**2/2 - sqrt(2)*x)/2 + C1*exp(x**2/2 + sqrt(2)*x)/2 -
                sqrt(2)*C2*exp(x**2/2 - sqrt(2)*x)/2 + sqrt(2)*C2*exp(x**2/2 + sqrt(2)*x)/2)**4*exp(-x), x))]
    components5 = {(frozenset([Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
                    Eq(Derivative(g(x), x), x*g(x) + f(x))]),
                    frozenset([Eq(Derivative(k(x), x), f(x)**4 + k(x)),])),
                    (frozenset([Eq(Derivative(h(x), x), h(x)),]),)}
    eqsdict5 = ({f(x): {g(x)}, g(x): {f(x)}, h(x): set(), k(x): {f(x)}},
                {f(x): Eq(Derivative(f(x), x), x*f(x) + 2*g(x)),
                g(x): Eq(Derivative(g(x), x), x*g(x) + f(x)),
                h(x): Eq(Derivative(h(x), x), h(x)),
                k(x): Eq(Derivative(k(x), x), f(x)**4 + k(x))})
    graph5 = [{f(x), g(x), h(x), k(x)}, {(f(x), g(x)), (g(x), f(x)), (k(x), f(x))}]
    assert {tuple(frozenset(scc) for scc in wcc) for wcc in _component_division(eqs5, funcs, x)} == components5
    assert _eqs2dict(eqs5, funcs) == eqsdict5
    assert [set(element) for element in _dict2graph(eqsdict5[0])] == graph5
    # XXX: dsolve hangs in integration here:
    assert dsolve_system(eqs5, simplify=False, doit=False) == [sol5]
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0, 0])


def test_linodesolve():
    t, x, a = symbols("t x a")
    f, g, h = symbols("f g h", cls=Function)

    # Testing the Errors
    raises(ValueError, lambda: linodesolve(1, t))
    raises(ValueError, lambda: linodesolve(a, t))

    A1 = Matrix([[1, 2], [2, 4], [4, 6]])
    raises(NonSquareMatrixError, lambda: linodesolve(A1, t))

    A2 = Matrix([[1, 2, 1], [3, 1, 2]])
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t))

    # Testing auto functionality
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t) + g(t).diff(t), g(t)), Eq(g(t).diff(t), f(t))]
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    sol = [C1*(-Rational(1, 2) + sqrt(5)/2)*exp(t*(-Rational(1, 2) + sqrt(5)/2)) + C2*(-sqrt(5)/2 - Rational(1, 2))*
           exp(t*(-sqrt(5)/2 - Rational(1, 2))),
           C1*exp(t*(-Rational(1, 2) + sqrt(5)/2)) + C2*exp(t*(-sqrt(5)/2 - Rational(1, 2)))]
    assert constant_renumber(linodesolve(A, t), variables=Tuple(*eq).free_symbols) == sol

    # Testing the Errors
    raises(ValueError, lambda: linodesolve(1, t, b=Matrix([t+1])))
    raises(ValueError, lambda: linodesolve(a, t, b=Matrix([log(t) + sin(t)])))

    raises(ValueError, lambda: linodesolve(Matrix([7]), t, b=t**2))
    raises(ValueError, lambda: linodesolve(Matrix([a+10]), t, b=log(t)*cos(t)))

    raises(ValueError, lambda: linodesolve(7, t, b=t**2))
    raises(ValueError, lambda: linodesolve(a, t, b=log(t) + sin(t)))

    A1 = Matrix([[1, 2], [2, 4], [4, 6]])
    b1 = Matrix([t, 1, t**2])
    raises(NonSquareMatrixError, lambda: linodesolve(A1, t, b=b1))

    A2 = Matrix([[1, 2, 1], [3, 1, 2]])
    b2 = Matrix([t, t**2])
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t, b=b2))

    raises(ValueError, lambda: linodesolve(A1[:2, :], t, b=b1))
    raises(ValueError, lambda: linodesolve(A1[:2, :], t, b=b1[:1]))

    # DOIT check
    A1 = Matrix([[1, -1], [1, -1]])
    b1 = Matrix([15*t - 10, -15*t - 5])
    sol1 = [C1 + C2*t + C2 - 10*t**3 + 10*t**2 + t*(15*t**2 - 5*t) - 10*t,
            C1 + C2*t - 10*t**3 - 5*t**2 + t*(15*t**2 - 5*t) - 5*t]
    assert constant_renumber(linodesolve(A1, t, b=b1, type="type2", doit=True),
                             variables=[t]) == sol1

    # Testing auto functionality
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t) + g(t).diff(t), g(t) + t), Eq(g(t).diff(t), f(t))]
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    sol = [-C1*exp(-t/2 + sqrt(5)*t/2)/2 + sqrt(5)*C1*exp(-t/2 + sqrt(5)*t/2)/2 - sqrt(5)*C2*exp(-sqrt(5)*t/2 -
          t/2)/2 - C2*exp(-sqrt(5)*t/2 - t/2)/2 - exp(-t/2 + sqrt(5)*t/2)*Integral(t*exp(-sqrt(5)*t/2 +
          t/2)/(-5 + sqrt(5)) - sqrt(5)*t*exp(-sqrt(5)*t/2 + t/2)/(-5 + sqrt(5)), t)/2 + sqrt(5)*exp(-t/2 +
          sqrt(5)*t/2)*Integral(t*exp(-sqrt(5)*t/2 + t/2)/(-5 + sqrt(5)) - sqrt(5)*t*exp(-sqrt(5)*t/2 +
          t/2)/(-5 + sqrt(5)), t)/2 - sqrt(5)*exp(-sqrt(5)*t/2 - t/2)*Integral(-sqrt(5)*t*exp(t/2 +
          sqrt(5)*t/2)/5, t)/2 - exp(-sqrt(5)*t/2 - t/2)*Integral(-sqrt(5)*t*exp(t/2 + sqrt(5)*t/2)/5, t)/2,
         C1*exp(-t/2 + sqrt(5)*t/2) + C2*exp(-sqrt(5)*t/2 - t/2) + exp(-t/2 +
          sqrt(5)*t/2)*Integral(t*exp(-sqrt(5)*t/2 + t/2)/(-5 + sqrt(5)) - sqrt(5)*t*exp(-sqrt(5)*t/2 +
          t/2)/(-5 + sqrt(5)), t) + exp(-sqrt(5)*t/2 -
              t/2)*Integral(-sqrt(5)*t*exp(t/2 + sqrt(5)*t/2)/5, t)]
    assert constant_renumber(linodesolve(A, t, b=b), variables=[t]) == sol

    # non-homogeneous term assumed to be 0
    sol1 = [-C1*exp(-t/2 + sqrt(5)*t/2)/2 + sqrt(5)*C1*exp(-t/2 + sqrt(5)*t/2)/2 - sqrt(5)*C2*exp(-sqrt(5)*t/2
                - t/2)/2 - C2*exp(-sqrt(5)*t/2 - t/2)/2,
            C1*exp(-t/2 + sqrt(5)*t/2) + C2*exp(-sqrt(5)*t/2 - t/2)]
    assert constant_renumber(linodesolve(A, t, type="type2"), variables=[t]) == sol1

    # Testing the Errors
    raises(ValueError, lambda: linodesolve(t+10, t))
    raises(ValueError, lambda: linodesolve(a*t, t))

    A1 = Matrix([[1, t], [-t, 1]])
    B1, _ = _is_commutative_anti_derivative(A1, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A1[:, :1], t, B=B1))
    raises(ValueError, lambda: linodesolve(A1, t, B=1))

    A2 = Matrix([[t, t, t], [t, t, t], [t, t, t]])
    B2, _ = _is_commutative_anti_derivative(A2, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t, B=B2[:2, :]))
    raises(ValueError, lambda: linodesolve(A2, t, B=2))
    raises(ValueError, lambda: linodesolve(A2, t, B=B2, type="type31"))

    raises(ValueError, lambda: linodesolve(A1, t, B=B2))
    raises(ValueError, lambda: linodesolve(A2, t, B=B1))

    # Testing auto functionality
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t), f(t) + t*g(t)), Eq(g(t).diff(t), -t*f(t) + g(t))]
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    sol = [(C1/2 - I*C2/2)*exp(I*t**2/2 + t) + (C1/2 + I*C2/2)*exp(-I*t**2/2 + t),
           (-I*C1/2 + C2/2)*exp(-I*t**2/2 + t) + (I*C1/2 + C2/2)*exp(I*t**2/2 + t)]
    assert constant_renumber(linodesolve(A, t), variables=Tuple(*eq).free_symbols) == sol
    assert constant_renumber(linodesolve(A, t, type="type3"), variables=Tuple(*eq).free_symbols) == sol

    A1 = Matrix([[t, 1], [t, -1]])
    raises(NotImplementedError, lambda: linodesolve(A1, t))

    # Testing the Errors
    raises(ValueError, lambda: linodesolve(t+10, t, b=Matrix([t+1])))
    raises(ValueError, lambda: linodesolve(a*t, t, b=Matrix([log(t) + sin(t)])))

    raises(ValueError, lambda: linodesolve(Matrix([7*t]), t, b=t**2))
    raises(ValueError, lambda: linodesolve(Matrix([a + 10*log(t)]), t, b=log(t)*cos(t)))

    raises(ValueError, lambda: linodesolve(7*t, t, b=t**2))
    raises(ValueError, lambda: linodesolve(a*t**2, t, b=log(t) + sin(t)))

    A1 = Matrix([[1, t], [-t, 1]])
    b1 = Matrix([t, t ** 2])
    B1, _ = _is_commutative_anti_derivative(A1, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A1[:, :1], t, b=b1))

    A2 = Matrix([[t, t, t], [t, t, t], [t, t, t]])
    b2 = Matrix([t, 1, t**2])
    B2, _ = _is_commutative_anti_derivative(A2, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A2[:2, :], t, b=b2))

    raises(ValueError, lambda: linodesolve(A1, t, b=b2))
    raises(ValueError, lambda: linodesolve(A2, t, b=b1))

    raises(ValueError, lambda: linodesolve(A1, t, b=b1, B=B2))
    raises(ValueError, lambda: linodesolve(A2, t, b=b2, B=B1))

    # Testing auto functionality
    func = [f(x), g(x), h(x)]
    eq = [Eq(f(x).diff(x), x*(f(x) + g(x) + h(x)) + x),
          Eq(g(x).diff(x), x*(f(x) + g(x) + h(x)) + x),
          Eq(h(x).diff(x), x*(f(x) + g(x) + h(x)) + 1)]
    ceq = canonical_odes(eq, func, x)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, x, 1)
    A = A0
    _x1 = exp(-3*x**2/2)
    _x2 = exp(3*x**2/2)
    _x3 = Integral(2*_x1*x/3 + _x1/3 + x/3 - Rational(1, 3), x)
    _x4 = 2*_x2*_x3/3
    _x5 = Integral(2*_x1*x/3 + _x1/3 - 2*x/3 + Rational(2, 3), x)
    sol = [
        C1*_x2/3 - C1/3 + C2*_x2/3 - C2/3 + C3*_x2/3 + 2*C3/3 + _x2*_x5/3 + _x3/3 + _x4 - _x5/3,
        C1*_x2/3 + 2*C1/3 + C2*_x2/3 - C2/3 + C3*_x2/3 - C3/3 + _x2*_x5/3 + _x3/3 + _x4 - _x5/3,
        C1*_x2/3 - C1/3 + C2*_x2/3 + 2*C2/3 + C3*_x2/3 - C3/3 + _x2*_x5/3 - 2*_x3/3 + _x4 + 2*_x5/3,
    ]
    assert constant_renumber(linodesolve(A, x, b=b), variables=Tuple(*eq).free_symbols) == sol
    assert constant_renumber(linodesolve(A, x, b=b, type="type4"),
                             variables=Tuple(*eq).free_symbols) == sol

    A1 = Matrix([[t, 1], [t, -1]])
    raises(NotImplementedError, lambda: linodesolve(A1, t, b=b1))

    # non-homogeneous term not passed
    sol1 = [-C1/3 - C2/3 + 2*C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2), 2*C1/3 - C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2),
            -C1/3 + 2*C2/3 - C3/3 + (C1/3 + C2/3 + C3/3)*exp(3*x**2/2)]
    assert constant_renumber(linodesolve(A, x, type="type4", doit=True), variables=Tuple(*eq).free_symbols) == sol1


@slow
def test_linear_3eq_order1_type4_slow():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')

    f = t ** 3 + log(t)
    g = t ** 2 + sin(t)
    eq1 = (Eq(diff(x(t), t), (4 * f + g) * x(t) - f * y(t) - 2 * f * z(t)),
                Eq(diff(y(t), t), 2 * f * x(t) + (f + g) * y(t) - 2 * f * z(t)), Eq(diff(z(t), t), 5 * f * x(t) + f * y(
        t) + (-3 * f + g) * z(t)))
    with dotprodsimp(True):
        dsolve(eq1)


@slow
def test_linear_neq_order1_type2_slow1():
    i, r1, c1, r2, c2, t = symbols('i, r1, c1, r2, c2, t')
    x1 = Function('x1')
    x2 = Function('x2')

    eq1 = r1*c1*Derivative(x1(t), t) + x1(t) - x2(t) - r1*i
    eq2 = r2*c1*Derivative(x1(t), t) + r2*c2*Derivative(x2(t), t) + x2(t) - r2*i
    eq = [eq1, eq2]

    # XXX: Solution is too complicated
    [sol] = dsolve_system(eq, simplify=False, doit=False)
    assert checksysodesol(eq, sol) == (True, [0, 0])


# Regression test case for issue #9204
# https://github.com/sympy/sympy/issues/9204
@tooslow
def test_linear_new_order1_type2_de_lorentz_slow_check():
    m = Symbol("m", real=True)
    q = Symbol("q", real=True)
    t = Symbol("t", real=True)

    e1, e2, e3 = symbols("e1:4", real=True)
    b1, b2, b3 = symbols("b1:4", real=True)
    v1, v2, v3 = symbols("v1:4", cls=Function, real=True)

    eqs = [
        -e1*q + m*Derivative(v1(t), t) - q*(-b2*v3(t) + b3*v2(t)),
        -e2*q + m*Derivative(v2(t), t) - q*(b1*v3(t) - b3*v1(t)),
        -e3*q + m*Derivative(v3(t), t) - q*(-b1*v2(t) + b2*v1(t))
    ]
    sol = dsolve(eqs)
    assert checksysodesol(eqs, sol) == (True, [0, 0, 0])


# Regression test case for issue #14001
# https://github.com/sympy/sympy/issues/14001
@slow
def test_linear_neq_order1_type2_slow_check():
    RC, t, C, Vs, L, R1, V0, I0 = symbols("RC t C Vs L R1 V0 I0")
    V = Function("V")
    I = Function("I")
    system = [Eq(V(t).diff(t), -1/RC*V(t) + I(t)/C), Eq(I(t).diff(t), -R1/L*I(t) - 1/L*V(t) + Vs/L)]
    [sol] = dsolve_system(system, simplify=False, doit=False)

    assert checksysodesol(system, sol) == (True, [0, 0])


def _linear_3eq_order1_type4_long():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')

    f = t ** 3 + log(t)
    g = t ** 2 + sin(t)

    eq1 = (Eq(diff(x(t), t), (4*f + g)*x(t) - f*y(t) - 2*f*z(t)),
           Eq(diff(y(t), t), 2*f*x(t) + (f + g)*y(t) - 2*f*z(t)), Eq(diff(z(t), t), 5*f*x(t) + f*y(
        t) + (-3*f + g)*z(t)))

    dsolve_sol = dsolve(eq1)
    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]

    x_1 = sqrt(-t**6 - 8*t**3*log(t) + 8*t**3 - 16*log(t)**2 + 32*log(t) - 16)
    x_2 = sqrt(3)
    x_3 = 8324372644*C1*x_1*x_2 + 4162186322*C2*x_1*x_2 - 8324372644*C3*x_1*x_2
    x_4 = 1 / (1903457163*t**3 + 3825881643*x_1*x_2 + 7613828652*log(t) - 7613828652)
    x_5 = exp(t**3/3 + t*x_1*x_2/4 - cos(t))
    x_6 = exp(t**3/3 - t*x_1*x_2/4 - cos(t))
    x_7 = exp(t**4/2 + t**3/3 + 2*t*log(t) - 2*t - cos(t))
    x_8 = 91238*C1*x_1*x_2 + 91238*C2*x_1*x_2 - 91238*C3*x_1*x_2
    x_9 = 1 / (66049*t**3 - 50629*x_1*x_2 + 264196*log(t) - 264196)
    x_10 = 50629 * C1 / 25189 + 37909*C2/25189 - 50629*C3/25189 - x_3*x_4
    x_11 = -50629*C1/25189 - 12720*C2/25189 + 50629*C3/25189 + x_3*x_4
    sol = [Eq(x(t), x_10*x_5 + x_11*x_6 + x_7*(C1 - C2)), Eq(y(t), x_10*x_5 + x_11*x_6), Eq(z(t), x_5*(
            -424*C1/257 - 167*C2/257 + 424*C3/257 - x_8*x_9) + x_6*(167*C1/257 + 424*C2/257 -
            167*C3/257 + x_8*x_9) + x_7*(C1 - C2))]

    assert dsolve_sol1 == sol
    assert checksysodesol(eq1, dsolve_sol1) == (True, [0, 0, 0])


@slow
def test_neq_order1_type4_slow_check1():
    f, g = symbols("f g", cls=Function)
    x = symbols("x")

    eqs = [Eq(diff(f(x), x), x*f(x) + x**2*g(x) + x),
           Eq(diff(g(x), x), 2*x**2*f(x) + (x + 3*x**2)*g(x) + 1)]
    sol = dsolve(eqs)
    assert checksysodesol(eqs, sol) == (True, [0, 0])


@slow
def test_neq_order1_type4_slow_check2():
    f, g, h = symbols("f, g, h", cls=Function)
    x = Symbol("x")

    eqs = [
        Eq(Derivative(f(x), x), x*h(x) + f(x) + g(x) + 1),
        Eq(Derivative(g(x), x), x*g(x) + f(x) + h(x) + 10),
        Eq(Derivative(h(x), x), x*f(x) + x + g(x) + h(x))
    ]
    with dotprodsimp(True):
        sol = dsolve(eqs)
    assert checksysodesol(eqs, sol) == (True, [0, 0, 0])


def _neq_order1_type4_slow3():
    f, g = symbols("f g", cls=Function)
    x = symbols("x")

    eqs = [
        Eq(Derivative(f(x), x), x*f(x) + g(x) + sin(x)),
        Eq(Derivative(g(x), x), x**2 + x*g(x) - f(x))
    ]
    sol = [
        Eq(f(x), (C1/2 - I*C2/2 - I*Integral(x**2*exp(-x**2/2 - I*x)/2 +
            x**2*exp(-x**2/2 + I*x)/2 + I*exp(-x**2/2 - I*x)*sin(x)/2 -
            I*exp(-x**2/2 + I*x)*sin(x)/2, x)/2 + Integral(-I*x**2*exp(-x**2/2
            - I*x)/2 + I*x**2*exp(-x**2/2 + I*x)/2 + exp(-x**2/2 -
            I*x)*sin(x)/2 + exp(-x**2/2 + I*x)*sin(x)/2, x)/2)*exp(x**2/2 +
            I*x) + (C1/2 + I*C2/2 + I*Integral(x**2*exp(-x**2/2 - I*x)/2 +
            x**2*exp(-x**2/2 + I*x)/2 + I*exp(-x**2/2 - I*x)*sin(x)/2 -
            I*exp(-x**2/2 + I*x)*sin(x)/2, x)/2 + Integral(-I*x**2*exp(-x**2/2
            - I*x)/2 + I*x**2*exp(-x**2/2 + I*x)/2 + exp(-x**2/2 -
            I*x)*sin(x)/2 + exp(-x**2/2 + I*x)*sin(x)/2, x)/2)*exp(x**2/2 -
            I*x)),
        Eq(g(x), (-I*C1/2 + C2/2 + Integral(x**2*exp(-x**2/2 - I*x)/2 +
            x**2*exp(-x**2/2 + I*x)/2 + I*exp(-x**2/2 - I*x)*sin(x)/2 -
            I*exp(-x**2/2 + I*x)*sin(x)/2, x)/2 -
            I*Integral(-I*x**2*exp(-x**2/2 - I*x)/2 + I*x**2*exp(-x**2/2 +
            I*x)/2 + exp(-x**2/2 - I*x)*sin(x)/2 + exp(-x**2/2 +
            I*x)*sin(x)/2, x)/2)*exp(x**2/2 - I*x) + (I*C1/2 + C2/2 +
            Integral(x**2*exp(-x**2/2 - I*x)/2 + x**2*exp(-x**2/2 + I*x)/2 +
            I*exp(-x**2/2 - I*x)*sin(x)/2 - I*exp(-x**2/2 + I*x)*sin(x)/2,
            x)/2 + I*Integral(-I*x**2*exp(-x**2/2 - I*x)/2 +
            I*x**2*exp(-x**2/2 + I*x)/2 + exp(-x**2/2 - I*x)*sin(x)/2 +
            exp(-x**2/2 + I*x)*sin(x)/2, x)/2)*exp(x**2/2 + I*x))
    ]

    return eqs, sol


def test_neq_order1_type4_slow3():
    eqs, sol = _neq_order1_type4_slow3()
    assert dsolve_system(eqs, simplify=False, doit=False) == [sol]
    # XXX: dsolve gives an error in integration:
    #     assert dsolve(eqs) == sol
    # https://github.com/sympy/sympy/issues/20155


@slow
def test_neq_order1_type4_slow_check3():
    eqs, sol = _neq_order1_type4_slow3()
    assert checksysodesol(eqs, sol) == (True, [0, 0])


@tooslow
@XFAIL
def test_linear_3eq_order1_type4_long_dsolve_slow_xfail():
    eq, sol = _linear_3eq_order1_type4_long()

    dsolve_sol = dsolve(eq)
    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]

    assert dsolve_sol1 == sol


@tooslow
def test_linear_3eq_order1_type4_long_dsolve_dotprodsimp():
    eq, sol = _linear_3eq_order1_type4_long()

    # XXX: Only works with dotprodsimp see
    # test_linear_3eq_order1_type4_long_dsolve_slow_xfail which is too slow
    with dotprodsimp(True):
        dsolve_sol = dsolve(eq)

    dsolve_sol1 = [_simpsol(sol) for sol in dsolve_sol]
    assert dsolve_sol1 == sol


@tooslow
def test_linear_3eq_order1_type4_long_check():
    eq, sol = _linear_3eq_order1_type4_long()
    assert checksysodesol(eq, sol) == (True, [0, 0, 0])


def test_dsolve_system():
    f, g = symbols("f g", cls=Function)
    x = symbols("x")
    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    funcs = [f(x), g(x)]

    sol = [[Eq(f(x), -C1 + C2*exp(2*x)), Eq(g(x), C1 + C2*exp(2*x))]]
    assert dsolve_system(eqs, funcs=funcs, t=x, doit=True) == sol

    raises(ValueError, lambda: dsolve_system(1))
    raises(ValueError, lambda: dsolve_system(eqs, 1))
    raises(ValueError, lambda: dsolve_system(eqs, funcs, 1))
    raises(ValueError, lambda: dsolve_system(eqs, funcs[:1], x))

    eq = (Eq(f(x).diff(x), 12 * f(x) - 6 * g(x)), Eq(g(x).diff(x) ** 2, 11 * f(x) + 3 * g(x)))
    raises(NotImplementedError, lambda: dsolve_system(eq) == ([], []))

    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)]) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], t=x) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], t=x, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, t=x, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], ics={f(0): 1, g(0): 1}) == ([], []))

def test_dsolve():

    f, g = symbols('f g', cls=Function)
    x, y = symbols('x y')

    eqs = [f(x).diff(x) - x, f(x).diff(x) + x]
    with raises(ValueError):
        dsolve(eqs)

    eqs = [f(x, y).diff(x)]
    with raises(ValueError):
        dsolve(eqs)

    eqs = [f(x, y).diff(x)+g(x).diff(x), g(x).diff(x)]
    with raises(ValueError):
        dsolve(eqs)


@slow
def test_higher_order1_slow1():
    x, y = symbols("x y", cls=Function)
    t = symbols("t")

    eq = [
        Eq(diff(x(t),t,t), (log(t)+t**2)*diff(x(t),t)+(log(t)+t**2)*3*diff(y(t),t)),
        Eq(diff(y(t),t,t), (log(t)+t**2)*2*diff(x(t),t)+(log(t)+t**2)*9*diff(y(t),t))
    ]
    sol, = dsolve_system(eq, simplify=False, doit=False)
    # The solution is too long to write out explicitly and checkodesol is too
    # slow so we test for particular values of t:
    for e in eq:
        res = (e.lhs - e.rhs).subs({sol[0].lhs:sol[0].rhs, sol[1].lhs:sol[1].rhs})
        res = res.subs({d: d.doit(deep=False) for d in res.atoms(Derivative)})
        assert ratsimp(res.subs(t, 1)) == 0


def test_second_order_type2_slow1():
    x, y, z = symbols('x, y, z', cls=Function)
    t, l = symbols('t, l')

    eqs1 = [Eq(Derivative(x(t), (t, 2)), t*(2*x(t) + y(t))),
            Eq(Derivative(y(t), (t, 2)), t*(-x(t) + 2*y(t)))]
    sol1 = [Eq(x(t), I*C1*airyai(t*(2 - I)**(S(1)/3)) + I*C2*airybi(t*(2 - I)**(S(1)/3)) - I*C3*airyai(t*(2 +
             I)**(S(1)/3)) - I*C4*airybi(t*(2 + I)**(S(1)/3))),
            Eq(y(t), C1*airyai(t*(2 - I)**(S(1)/3)) + C2*airybi(t*(2 - I)**(S(1)/3)) + C3*airyai(t*(2 + I)**(S(1)/3)) +
             C4*airybi(t*(2 + I)**(S(1)/3)))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])


@tooslow
@XFAIL
def test_nonlinear_3eq_order1_type1():
    a, b, c = symbols('a b c')

    eqs = [
        a * f(x).diff(x) - (b - c) * g(x) * h(x),
        b * g(x).diff(x) - (c - a) * h(x) * f(x),
        c * h(x).diff(x) - (a - b) * f(x) * g(x),
    ]

    assert dsolve(eqs) # NotImplementedError


@XFAIL
def test_nonlinear_3eq_order1_type4():
    eqs = [
        Eq(f(x).diff(x), (2*h(x)*g(x) - 3*g(x)*h(x))),
        Eq(g(x).diff(x), (4*f(x)*h(x) - 2*h(x)*f(x))),
        Eq(h(x).diff(x), (3*g(x)*f(x) - 4*f(x)*g(x))),
    ]
    dsolve(eqs)  # KeyError when matching
    # sol = ?
    # assert dsolve_sol == sol
    # assert checksysodesol(eqs, dsolve_sol) == (True, [0, 0, 0])


@tooslow
@XFAIL
def test_nonlinear_3eq_order1_type3():
    eqs = [
        Eq(f(x).diff(x), (2*f(x)**2 - 3        )),
        Eq(g(x).diff(x), (4         - 2*h(x)   )),
        Eq(h(x).diff(x), (3*h(x)    - 4*f(x)**2)),
    ]
    dsolve(eqs)  # Not sure if this finishes...
    # sol = ?
    # assert dsolve_sol == sol
    # assert checksysodesol(eqs, dsolve_sol) == (True, [0, 0, 0])


@XFAIL
def test_nonlinear_3eq_order1_type5():
    eqs = [
        Eq(f(x).diff(x), f(x)*(2*f(x) - 3*g(x))),
        Eq(g(x).diff(x), g(x)*(4*g(x) - 2*h(x))),
        Eq(h(x).diff(x), h(x)*(3*h(x) - 4*f(x))),
    ]
    dsolve(eqs)  # KeyError
    # sol = ?
    # assert dsolve_sol == sol
    # assert checksysodesol(eqs, dsolve_sol) == (True, [0, 0, 0])


def test_linear_2eq_order1():
    x, y, z = symbols('x, y, z', cls=Function)
    k, l, m, n = symbols('k, l, m, n', Integer=True)
    t = Symbol('t')
    x0, y0 = symbols('x0, y0', cls=Function)

    eq1 = (Eq(diff(x(t),t), x(t) + y(t) + 9), Eq(diff(y(t),t), 2*x(t) + 5*y(t) + 23))
    sol1 = [Eq(x(t), C1*exp(t*(sqrt(6) + 3)) + C2*exp(t*(-sqrt(6) + 3)) - Rational(22, 3)), \
    Eq(y(t), C1*(2 + sqrt(6))*exp(t*(sqrt(6) + 3)) + C2*(-sqrt(6) + 2)*exp(t*(-sqrt(6) + 3)) - Rational(5, 3))]
    assert checksysodesol(eq1, sol1) == (True, [0, 0])

    eq2 = (Eq(diff(x(t),t), x(t) + y(t) + 81), Eq(diff(y(t),t), -2*x(t) + y(t) + 23))
    sol2 = [Eq(x(t), (C1*cos(sqrt(2)*t) + C2*sin(sqrt(2)*t))*exp(t) - Rational(58, 3)), \
    Eq(y(t), (-sqrt(2)*C1*sin(sqrt(2)*t) + sqrt(2)*C2*cos(sqrt(2)*t))*exp(t) - Rational(185, 3))]
    assert checksysodesol(eq2, sol2) == (True, [0, 0])

    eq3 = (Eq(diff(x(t),t), 5*t*x(t) + 2*y(t)), Eq(diff(y(t),t), 2*x(t) + 5*t*y(t)))
    sol3 = [Eq(x(t), (C1*exp(2*t) + C2*exp(-2*t))*exp(Rational(5, 2)*t**2)), \
    Eq(y(t), (C1*exp(2*t) - C2*exp(-2*t))*exp(Rational(5, 2)*t**2))]
    assert checksysodesol(eq3, sol3) == (True, [0, 0])

    eq4 = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    sol4 = [Eq(x(t), (C1*cos((t**3)/3) + C2*sin((t**3)/3))*exp(Rational(5, 2)*t**2)), \
    Eq(y(t), (-C1*sin((t**3)/3) + C2*cos((t**3)/3))*exp(Rational(5, 2)*t**2))]
    assert checksysodesol(eq4, sol4) == (True, [0, 0])

    eq5 = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), -t**2*x(t) + (5*t+9*t**2)*y(t)))
    sol5 = [Eq(x(t), (C1*exp((sqrt(77)/2 + Rational(9, 2))*(t**3)/3) + \
    C2*exp((-sqrt(77)/2 + Rational(9, 2))*(t**3)/3))*exp(Rational(5, 2)*t**2)), \
    Eq(y(t), (C1*(sqrt(77)/2 + Rational(9, 2))*exp((sqrt(77)/2 + Rational(9, 2))*(t**3)/3) + \
    C2*(-sqrt(77)/2 + Rational(9, 2))*exp((-sqrt(77)/2 + Rational(9, 2))*(t**3)/3))*exp(Rational(5, 2)*t**2))]
    assert checksysodesol(eq5, sol5) == (True, [0, 0])

    eq6 = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t)), Eq(diff(y(t),t), (1-t**2)*x(t) + (5*t+9*t**2)*y(t)))
    sol6 = [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t)), \
    Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(t**2*exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)**2, t) + \
    exp(Integral(5*t, t))*exp(Integral(9*t**2 + 5*t, t))/x0(t)))]
    s = dsolve(eq6)
    assert s == sol6   # too complicated to test with subs and simplify
    # assert checksysodesol(eq10, sol10) == (True, [0, 0])  # this one fails


def test_nonlinear_2eq_order1():
    x, y, z = symbols('x, y, z', cls=Function)
    t = Symbol('t')
    eq1 = (Eq(diff(x(t),t),x(t)*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    sol1 = [
        Eq(x(t), C1*exp((-1/(4*C2 + 4*t))**(Rational(-1, 4)))),
        Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), C1*exp(-1/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), C1*exp(-I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), C1*exp(I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert dsolve(eq1) == sol1
    assert checksysodesol(eq1, sol1) == (True, [0, 0])

    eq2 = (Eq(diff(x(t),t), exp(3*x(t))*y(t)**3),Eq(diff(y(t),t), y(t)**5))
    sol2 = [
        Eq(x(t), -log(C1 - 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), -log(C1 + 3/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), -log(C1 + 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), -log(C1 - 3*I/(-1/(4*C2 + 4*t))**Rational(1, 4))/3),
        Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert dsolve(eq2) == sol2
    assert checksysodesol(eq2, sol2) == (True, [0, 0])

    eq3 = (Eq(diff(x(t),t), y(t)*x(t)), Eq(diff(y(t),t), x(t)**3))
    tt = Rational(2, 3)
    sol3 = [
        Eq(x(t), 6**tt/(6*(-sinh(sqrt(C1)*(C2 + t)/2)/sqrt(C1))**tt)),
        Eq(y(t), sqrt(C1 + C1/sinh(sqrt(C1)*(C2 + t)/2)**2)/3)]
    assert dsolve(eq3) == sol3
    # FIXME: assert checksysodesol(eq3, sol3) == (True, [0, 0])

    eq4 = (Eq(diff(x(t),t),x(t)*y(t)*sin(t)**2), Eq(diff(y(t),t),y(t)**2*sin(t)**2))
    sol4 = {Eq(x(t), -2*exp(C1)/(C2*exp(C1) + t - sin(2*t)/2)), Eq(y(t), -2/(C1 + t - sin(2*t)/2))}
    assert dsolve(eq4) == sol4
    # FIXME: assert checksysodesol(eq4, sol4) == (True, [0, 0])

    eq5 = (Eq(x(t),t*diff(x(t),t)+diff(x(t),t)*diff(y(t),t)), Eq(y(t),t*diff(y(t),t)+diff(y(t),t)**2))
    sol5 = {Eq(x(t), C1*C2 + C1*t), Eq(y(t), C2**2 + C2*t)}
    assert dsolve(eq5) == sol5
    assert checksysodesol(eq5, sol5) == (True, [0, 0])

    eq6 = (Eq(diff(x(t),t),x(t)**2*y(t)**3), Eq(diff(y(t),t),y(t)**5))
    sol6 = [
        Eq(x(t), 1/(C1 - 1/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), -(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), 1/(C1 + (-1/(4*C2 + 4*t))**(Rational(-1, 4)))),
        Eq(y(t), (-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), 1/(C1 + I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), -I*(-1/(4*C2 + 4*t))**Rational(1, 4)),
        Eq(x(t), 1/(C1 - I/(-1/(4*C2 + 4*t))**Rational(1, 4))),
        Eq(y(t), I*(-1/(4*C2 + 4*t))**Rational(1, 4))]
    assert dsolve(eq6) == sol6
    assert checksysodesol(eq6, sol6) == (True, [0, 0])


@slow
def test_nonlinear_3eq_order1():
    x, y, z = symbols('x, y, z', cls=Function)
    t, u = symbols('t u')
    eq1 = (4*diff(x(t),t) + 2*y(t)*z(t), 3*diff(y(t),t) - z(t)*x(t), 5*diff(z(t),t) - x(t)*y(t))
    sol1 = [Eq(4*Integral(1/(sqrt(-4*u**2 - 3*C1 + C2)*sqrt(-4*u**2 + 5*C1 - C2)), (u, x(t))),
        C3 - sqrt(15)*t/15), Eq(3*Integral(1/(sqrt(-6*u**2 - C1 + 5*C2)*sqrt(3*u**2 + C1 - 4*C2)),
        (u, y(t))), C3 + sqrt(5)*t/10), Eq(5*Integral(1/(sqrt(-10*u**2 - 3*C1 + C2)*
        sqrt(5*u**2 + 4*C1 - C2)), (u, z(t))), C3 + sqrt(3)*t/6)]
    assert [i.dummy_eq(j) for i, j in zip(dsolve(eq1), sol1)]
    # FIXME: assert checksysodesol(eq1, sol1) == (True, [0, 0, 0])

    eq2 = (4*diff(x(t),t) + 2*y(t)*z(t)*sin(t), 3*diff(y(t),t) - z(t)*x(t)*sin(t), 5*diff(z(t),t) - x(t)*y(t)*sin(t))
    sol2 = [Eq(3*Integral(1/(sqrt(-6*u**2 - C1 + 5*C2)*sqrt(3*u**2 + C1 - 4*C2)), (u, x(t))), C3 +
        sqrt(5)*cos(t)/10), Eq(4*Integral(1/(sqrt(-4*u**2 - 3*C1 + C2)*sqrt(-4*u**2 + 5*C1 - C2)),
        (u, y(t))), C3 - sqrt(15)*cos(t)/15), Eq(5*Integral(1/(sqrt(-10*u**2 - 3*C1 + C2)*
        sqrt(5*u**2 + 4*C1 - C2)), (u, z(t))), C3 + sqrt(3)*cos(t)/6)]
    assert [i.dummy_eq(j) for i, j in zip(dsolve(eq2), sol2)]
    # FIXME: assert checksysodesol(eq2, sol2) == (True, [0, 0, 0])


def test_C1_function_9239():
    t = Symbol('t')
    C1 = Function('C1')
    C2 = Function('C2')
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    eq = (Eq(diff(C1(t), t), 9*C2(t)), Eq(diff(C2(t), t), 12*C1(t)))
    sol = [Eq(C1(t), 9*C3*exp(6*sqrt(3)*t) + 9*C4*exp(-6*sqrt(3)*t)),
           Eq(C2(t), 6*sqrt(3)*C3*exp(6*sqrt(3)*t) - 6*sqrt(3)*C4*exp(-6*sqrt(3)*t))]
    assert checksysodesol(eq, sol) == (True, [0, 0])


def test_dsolve_linsystem_symbol():
    eps = Symbol('epsilon', positive=True)
    eq1 = (Eq(diff(f(x), x), -eps*g(x)), Eq(diff(g(x), x), eps*f(x)))
    sol1 = [Eq(f(x), -C1*eps*cos(eps*x) - C2*eps*sin(eps*x)),
            Eq(g(x), -C1*eps*sin(eps*x) + C2*eps*cos(eps*x))]
    assert checksysodesol(eq1, sol1) == (True, [0, 0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\tests\test_query.py ===
from sympy.abc import t, w, x, y, z, n, k, m, p, i
from sympy.assumptions import (ask, AssumptionsContext, Q, register_handler,
        remove_handler)
from sympy.assumptions.assume import assuming, global_assumptions, Predicate
from sympy.assumptions.cnf import CNF, Literal
from sympy.assumptions.facts import (single_fact_lookup,
    get_known_facts, generate_known_facts_dict, get_known_facts_keys)
from sympy.assumptions.handlers import AskHandler
from sympy.assumptions.ask_generated import (get_all_known_facts,
    get_known_facts_dict)
from sympy.core.add import Add
from sympy.core.numbers import (I, Integer, Rational, oo, zoo, pi)
from sympy.core.singleton import S
from sympy.core.power import Pow
from sympy.core.symbol import Str, symbols, Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, sin, tan)
from sympy.logic.boolalg import Equivalent, Implies, Xor, And, to_cnf
from sympy.matrices import Matrix, SparseMatrix
from sympy.testing.pytest import (XFAIL, slow, raises, warns_deprecated_sympy,
    _both_exp_pow)
import math


def test_int_1():
    z = 1
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is True
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_int_11():
    z = 11
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is True
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is True
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_int_12():
    z = 12
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is True
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is True
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_float_1():
    z = 1.0
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is None
    assert ask(Q.rational(z)) is None
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is None
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is None
    assert ask(Q.odd(z)) is None
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is None
    assert ask(Q.composite(z)) is None
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    z = 7.2123
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is None
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is None
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    # test for issue #12168
    assert ask(Q.rational(math.pi)) is None


def test_zero_0():
    z = Integer(0)
    assert ask(Q.nonzero(z)) is False
    assert ask(Q.zero(z)) is True
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is True
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is True


def test_negativeone():
    z = Integer(-1)
    assert ask(Q.nonzero(z)) is True
    assert ask(Q.zero(z)) is False
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is True
    assert ask(Q.rational(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is True
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is True
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_infinity():
    assert ask(Q.commutative(oo)) is True
    assert ask(Q.integer(oo)) is False
    assert ask(Q.rational(oo)) is False
    assert ask(Q.algebraic(oo)) is False
    assert ask(Q.real(oo)) is False
    assert ask(Q.extended_real(oo)) is True
    assert ask(Q.complex(oo)) is False
    assert ask(Q.irrational(oo)) is False
    assert ask(Q.imaginary(oo)) is False
    assert ask(Q.positive(oo)) is False
    assert ask(Q.extended_positive(oo)) is True
    assert ask(Q.negative(oo)) is False
    assert ask(Q.even(oo)) is False
    assert ask(Q.odd(oo)) is False
    assert ask(Q.finite(oo)) is False
    assert ask(Q.infinite(oo)) is True
    assert ask(Q.prime(oo)) is False
    assert ask(Q.composite(oo)) is False
    assert ask(Q.hermitian(oo)) is False
    assert ask(Q.antihermitian(oo)) is False
    assert ask(Q.positive_infinite(oo)) is True
    assert ask(Q.negative_infinite(oo)) is False


def test_neg_infinity():
    mm = S.NegativeInfinity
    assert ask(Q.commutative(mm)) is True
    assert ask(Q.integer(mm)) is False
    assert ask(Q.rational(mm)) is False
    assert ask(Q.algebraic(mm)) is False
    assert ask(Q.real(mm)) is False
    assert ask(Q.extended_real(mm)) is True
    assert ask(Q.complex(mm)) is False
    assert ask(Q.irrational(mm)) is False
    assert ask(Q.imaginary(mm)) is False
    assert ask(Q.positive(mm)) is False
    assert ask(Q.negative(mm)) is False
    assert ask(Q.extended_negative(mm)) is True
    assert ask(Q.even(mm)) is False
    assert ask(Q.odd(mm)) is False
    assert ask(Q.finite(mm)) is False
    assert ask(Q.infinite(oo)) is True
    assert ask(Q.prime(mm)) is False
    assert ask(Q.composite(mm)) is False
    assert ask(Q.hermitian(mm)) is False
    assert ask(Q.antihermitian(mm)) is False
    assert ask(Q.positive_infinite(-oo)) is False
    assert ask(Q.negative_infinite(-oo)) is True


def test_complex_infinity():
    assert ask(Q.commutative(zoo)) is True
    assert ask(Q.integer(zoo)) is False
    assert ask(Q.rational(zoo)) is False
    assert ask(Q.algebraic(zoo)) is False
    assert ask(Q.real(zoo)) is False
    assert ask(Q.extended_real(zoo)) is False
    assert ask(Q.complex(zoo)) is False
    assert ask(Q.irrational(zoo)) is False
    assert ask(Q.imaginary(zoo)) is False
    assert ask(Q.positive(zoo)) is False
    assert ask(Q.negative(zoo)) is False
    assert ask(Q.zero(zoo)) is False
    assert ask(Q.nonzero(zoo)) is False
    assert ask(Q.even(zoo)) is False
    assert ask(Q.odd(zoo)) is False
    assert ask(Q.finite(zoo)) is False
    assert ask(Q.infinite(zoo)) is True
    assert ask(Q.prime(zoo)) is False
    assert ask(Q.composite(zoo)) is False
    assert ask(Q.hermitian(zoo)) is False
    assert ask(Q.antihermitian(zoo)) is False
    assert ask(Q.positive_infinite(zoo)) is False
    assert ask(Q.negative_infinite(zoo)) is False


def test_nan():
    nan = S.NaN
    assert ask(Q.commutative(nan)) is True
    assert ask(Q.integer(nan)) is None
    assert ask(Q.rational(nan)) is None
    assert ask(Q.algebraic(nan)) is None
    assert ask(Q.real(nan)) is None
    assert ask(Q.extended_real(nan)) is None
    assert ask(Q.complex(nan)) is None
    assert ask(Q.irrational(nan)) is None
    assert ask(Q.imaginary(nan)) is None
    assert ask(Q.positive(nan)) is None
    assert ask(Q.nonzero(nan)) is None
    assert ask(Q.zero(nan)) is None
    assert ask(Q.even(nan)) is None
    assert ask(Q.odd(nan)) is None
    assert ask(Q.finite(nan)) is None
    assert ask(Q.infinite(nan)) is None
    assert ask(Q.prime(nan)) is None
    assert ask(Q.composite(nan)) is None
    assert ask(Q.hermitian(nan)) is None
    assert ask(Q.antihermitian(nan)) is None


def test_Rational_number():
    r = Rational(3, 4)
    assert ask(Q.commutative(r)) is True
    assert ask(Q.integer(r)) is False
    assert ask(Q.rational(r)) is True
    assert ask(Q.real(r)) is True
    assert ask(Q.complex(r)) is True
    assert ask(Q.irrational(r)) is False
    assert ask(Q.imaginary(r)) is False
    assert ask(Q.positive(r)) is True
    assert ask(Q.negative(r)) is False
    assert ask(Q.even(r)) is False
    assert ask(Q.odd(r)) is False
    assert ask(Q.finite(r)) is True
    assert ask(Q.prime(r)) is False
    assert ask(Q.composite(r)) is False
    assert ask(Q.hermitian(r)) is True
    assert ask(Q.antihermitian(r)) is False

    r = Rational(1, 4)
    assert ask(Q.positive(r)) is True
    assert ask(Q.negative(r)) is False

    r = Rational(5, 4)
    assert ask(Q.negative(r)) is False
    assert ask(Q.positive(r)) is True

    r = Rational(5, 3)
    assert ask(Q.positive(r)) is True
    assert ask(Q.negative(r)) is False

    r = Rational(-3, 4)
    assert ask(Q.positive(r)) is False
    assert ask(Q.negative(r)) is True

    r = Rational(-1, 4)
    assert ask(Q.positive(r)) is False
    assert ask(Q.negative(r)) is True

    r = Rational(-5, 4)
    assert ask(Q.negative(r)) is True
    assert ask(Q.positive(r)) is False

    r = Rational(-5, 3)
    assert ask(Q.positive(r)) is False
    assert ask(Q.negative(r)) is True


def test_sqrt_2():
    z = sqrt(2)
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_pi():
    z = S.Pi
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is False
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    z = S.Pi + 1
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is False
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    z = 2*S.Pi
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is False
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    z = S.Pi ** 2
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is False
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False

    z = (1 + S.Pi) ** 2
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is None
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_E():
    z = S.Exp1
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is False
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_GoldenRatio():
    z = S.GoldenRatio
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_TribonacciConstant():
    z = S.TribonacciConstant
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is True
    assert ask(Q.real(z)) is True
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is True
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is True
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is True
    assert ask(Q.antihermitian(z)) is False


def test_I():
    z = I
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is True
    assert ask(Q.real(z)) is False
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is True
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is False
    assert ask(Q.antihermitian(z)) is True

    z = 1 + I
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is True
    assert ask(Q.real(z)) is False
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is False
    assert ask(Q.antihermitian(z)) is False

    z = I*(1 + I)
    assert ask(Q.commutative(z)) is True
    assert ask(Q.integer(z)) is False
    assert ask(Q.rational(z)) is False
    assert ask(Q.algebraic(z)) is True
    assert ask(Q.real(z)) is False
    assert ask(Q.complex(z)) is True
    assert ask(Q.irrational(z)) is False
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.positive(z)) is False
    assert ask(Q.negative(z)) is False
    assert ask(Q.even(z)) is False
    assert ask(Q.odd(z)) is False
    assert ask(Q.finite(z)) is True
    assert ask(Q.prime(z)) is False
    assert ask(Q.composite(z)) is False
    assert ask(Q.hermitian(z)) is False
    assert ask(Q.antihermitian(z)) is False

    z = I**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

    z = (-I)**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

    z = (3*I)**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is False

    z = (1)**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

    z = (-1)**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

    z = (1+I)**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is False

    z = (I)**(I+3)
    assert ask(Q.imaginary(z)) is True
    assert ask(Q.real(z)) is False

    z = (I)**(I+2)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

    z = (I)**(2)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

    z = (I)**(3)
    assert ask(Q.imaginary(z)) is True
    assert ask(Q.real(z)) is False

    z = (3)**(I)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is False

    z = (I)**(0)
    assert ask(Q.imaginary(z)) is False
    assert ask(Q.real(z)) is True

def test_bounded():
    x, y, z = symbols('x,y,z')
    a = x + y
    x, y = a.args
    assert ask(Q.finite(a), Q.positive_infinite(y)) is None
    assert ask(Q.finite(x)) is None
    assert ask(Q.finite(x), Q.finite(x)) is True
    assert ask(Q.finite(x), Q.finite(y)) is None
    assert ask(Q.finite(x), Q.complex(x)) is True
    assert ask(Q.finite(x), Q.extended_real(x)) is None

    assert ask(Q.finite(x + 1)) is None
    assert ask(Q.finite(x + 1), Q.finite(x)) is True
    a = x + y
    x, y = a.args
    # B + B
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is True
    assert ask(Q.finite(a), Q.positive(x) & Q.finite(y)) is True
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)) is True
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)) is True
    assert ask(Q.finite(a), Q.positive(x) & Q.finite(y)
        & ~Q.positive(y)) is True
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & Q.positive(y)) is True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y) & ~Q.positive(x)
        & ~Q.positive(y)) is True
    # B + U
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is False
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)) is False
    assert ask(Q.finite(a), Q.finite(x)
        & Q.positive_infinite(y)) is False
    assert ask(Q.finite(a), Q.positive(x)
        & Q.positive_infinite(y)) is False
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & ~Q.positive(y)) is False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & Q.positive_infinite(y)) is False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x) & ~Q.finite(y)
        & ~Q.positive(y)) is False
    # B + ?
    assert ask(Q.finite(a), Q.finite(x)) is None
    assert ask(Q.finite(a), Q.positive(x)) is None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.extended_positive(y)) is None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.extended_positive(y)) is None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.positive(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & Q.extended_positive(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.positive(x)
        & ~Q.positive(y)) is None
    # U + U
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & ~Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.positive_infinite(y)) is False
    assert ask(Q.finite(a), Q.positive_infinite(x) & ~Q.finite(y)
        & ~Q.extended_positive(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.extended_positive(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & ~Q.extended_positive(x) & ~Q.extended_positive(y)) is False
    # U + ?
    assert ask(Q.finite(a), ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.extended_positive(x)
        & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.extended_positive(x)
        & Q.positive_infinite(y)) is False
    assert ask(Q.finite(a), Q.extended_positive(x)
        & ~Q.finite(y) & ~Q.extended_positive(y)) is None
    assert ask(Q.finite(a), ~Q.extended_positive(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), ~Q.extended_positive(x) & ~Q.finite(y)
        & ~Q.extended_positive(y)) is False
    # ? + ?
    assert ask(Q.finite(a)) is None
    assert ask(Q.finite(a), Q.extended_positive(x)) is None
    assert ask(Q.finite(a), Q.extended_positive(y)) is None
    assert ask(Q.finite(a), Q.extended_positive(x)
        & Q.extended_positive(y)) is None
    assert ask(Q.finite(a), Q.extended_positive(x)
        & ~Q.extended_positive(y)) is None
    assert ask(Q.finite(a), ~Q.extended_positive(x)
        & Q.extended_positive(y)) is None
    assert ask(Q.finite(a), ~Q.extended_positive(x)
        & ~Q.extended_positive(y)) is None

    x, y, z = symbols('x,y,z')
    a = x + y + z
    x, y, z = a.args
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.negative(z)) is True
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.finite(z)) is True
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.positive(z)) is True
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & Q.finite(z)) is True
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & Q.positive(z)) is True
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.finite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.positive(z)) is True
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_positive(y)
        & Q.finite(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.extended_negative(z)) is False
    assert ask(Q.finite(a), Q.negative(x)
        & Q.negative_infinite(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.negative_infinite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.positive_infinite(y)
        & Q.negative_infinite(z)) is None
    assert ask(Q.finite(a), Q.negative(x) &
         Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is False
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x)) is None
    assert ask(Q.finite(a), Q.negative(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative(x) & Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.finite(z)) is True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.positive(z)) is True
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.positive(z)) is True
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.extended_negative(z)) is False
    assert ask(Q.finite(a), Q.finite(x)
        & Q.negative_infinite(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.negative_infinite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x)) is None
    assert ask(Q.finite(a), Q.finite(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.positive(z)) is True
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.extended_negative(z)) is False
    assert ask(Q.finite(a), Q.positive(x)
        & Q.negative_infinite(y)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.negative_infinite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.positive(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.positive(x) & Q.positive_infinite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is False
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive(x)) is None
    assert ask(Q.finite(a), Q.positive(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive(x) & Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & Q.negative_infinite(z)) is False
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y)& Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & Q.extended_negative(z)) is False
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.negative_infinite(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & ~Q.finite(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y) & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y) & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.positive_infinite(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_negative(y) & Q.extended_negative(z)) is False
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_negative(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.negative_infinite(x)
        & Q.extended_positive(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(z)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(y)
        & Q.positive_infinite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.positive_infinite(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_negative(y)
        & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_negative(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x)) is None
    assert ask(Q.finite(a), ~Q.finite(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.positive_infinite(y) & Q.positive_infinite(z)) is False
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.positive_infinite(y) & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.positive_infinite(y)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.positive_infinite(y) & Q.extended_positive(z)) is False
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.extended_negative(y) & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.extended_negative(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.positive_infinite(x)
        & Q.extended_positive(y) & Q.extended_positive(z)) is False
    assert ask(Q.finite(a), Q.extended_negative(x)
        & Q.extended_negative(y) & Q.extended_negative(z)) is None
    assert ask(Q.finite(a), Q.extended_negative(x)
        & Q.extended_negative(y)) is None
    assert ask(Q.finite(a), Q.extended_negative(x)
        & Q.extended_negative(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.extended_negative(x)) is None
    assert ask(Q.finite(a), Q.extended_negative(x)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.extended_negative(x)
        & Q.extended_positive(y) & Q.extended_positive(z)) is None
    assert ask(Q.finite(a)) is None
    assert ask(Q.finite(a), Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.extended_positive(y)
        & Q.extended_positive(z)) is None
    assert ask(Q.finite(a), Q.extended_positive(x)
        & Q.extended_positive(y) & Q.extended_positive(z)) is None

    assert ask(Q.finite(2*x)) is None
    assert ask(Q.finite(2*x), Q.finite(x)) is True

    x, y, z = symbols('x,y,z')
    a = x*y
    x, y = a.args
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is True
    assert ask(Q.finite(a), Q.finite(x) & ~Q.zero(x) & ~Q.finite(y)) is False
    assert ask(Q.finite(a), Q.finite(x)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y) &~Q.zero(y)) is False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is False
    assert ask(Q.finite(a), ~Q.finite(x)) is None
    assert ask(Q.finite(a), Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(y)) is None
    assert ask(Q.finite(a)) is None
    a = x*y*z
    x, y, z = a.args
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & Q.finite(z)) is True
    assert ask(Q.finite(a), Q.finite(x) & ~Q.zero(x) & Q.finite(y)
        & ~Q.zero(y) & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.zero(x) & ~Q.finite(y)
        & Q.finite(z) & ~Q.zero(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.zero(x) & ~Q.finite(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(x)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y) & ~Q.zero(y)
        & Q.finite(z) & ~Q.zero(z)) is False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.zero(x) & Q.finite(y)
        & ~Q.zero(y) & ~Q.finite(z)) is False
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.finite(z) & ~Q.zero(z)) is False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is False
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x)) is None
    assert ask(Q.finite(a), Q.finite(y) & Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(y) & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(y) & Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(y) & ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(y)) is None
    assert ask(Q.finite(a), Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(z) & Q.extended_nonzero(x)
        & Q.extended_nonzero(y) & Q.extended_nonzero(z)) is None
    assert ask(Q.finite(a), Q.extended_nonzero(x) & ~Q.finite(y)
        & Q.extended_nonzero(y) & ~Q.finite(z)
        & Q.extended_nonzero(z)) is False

    x, y, z = symbols('x,y,z')
    assert ask(Q.finite(x**2)) is None
    assert ask(Q.finite(2**x)) is None
    assert ask(Q.finite(2**x), Q.finite(x)) is True
    assert ask(Q.finite(x**x)) is None
    assert ask(Q.finite(S.Half ** x)) is None
    assert ask(Q.finite(S.Half ** x), Q.extended_positive(x)) is True
    assert ask(Q.finite(S.Half ** x), Q.extended_negative(x)) is None
    assert ask(Q.finite(2**x), Q.extended_negative(x)) is True
    assert ask(Q.finite(sqrt(x))) is None
    assert ask(Q.finite(2**x), ~Q.finite(x)) is False
    assert ask(Q.finite(x**2), ~Q.finite(x)) is False

    # https://github.com/sympy/sympy/issues/27707
    assert ask(Q.finite(x**y), Q.real(x) & Q.real(y)) is None
    assert ask(Q.finite(x**y), Q.real(x) & Q.negative(y)) is None
    assert ask(Q.finite(x**y), Q.zero(x) & Q.negative(y)) is False
    assert ask(Q.finite(x**y), Q.real(x) & Q.positive(y)) is True
    assert ask(Q.finite(x**y), Q.nonzero(x) & Q.real(y)) is True
    assert ask(Q.finite(x**y), Q.nonzero(x) & Q.negative(y)) is True
    assert ask(Q.finite(x**y), Q.zero(x) & Q.positive(y)) is True

    # sign function
    assert ask(Q.finite(sign(x))) is True
    assert ask(Q.finite(sign(x)), ~Q.finite(x)) is True

    # exponential functions
    assert ask(Q.finite(log(x))) is None
    assert ask(Q.finite(log(x)), Q.finite(x)) is None
    assert ask(Q.finite(log(x)), ~Q.zero(x)) is True
    assert ask(Q.finite(log(x)), Q.infinite(x)) is False
    assert ask(Q.finite(log(x)), Q.zero(x)) is False
    assert ask(Q.finite(exp(x))) is None
    assert ask(Q.finite(exp(x)), Q.finite(x)) is True
    assert ask(Q.finite(exp(2))) is True

    # trigonometric functions
    assert ask(Q.finite(sin(x))) is True
    assert ask(Q.finite(sin(x)), ~Q.finite(x)) is True
    assert ask(Q.finite(cos(x))) is True
    assert ask(Q.finite(cos(x)), ~Q.finite(x)) is True
    assert ask(Q.finite(2*sin(x))) is True
    assert ask(Q.finite(sin(x)**2)) is True
    assert ask(Q.finite(cos(x)**2)) is True
    assert ask(Q.finite(cos(x) + sin(x))) is True


def test_unbounded():
    assert ask(Q.infinite(I * oo)) is True
    assert ask(Q.infinite(1 + I*oo)) is True
    assert ask(Q.infinite(3 * (I * oo))) is True
    assert ask(Q.infinite(-I * oo)) is True
    assert ask(Q.infinite(1 + zoo)) is True
    assert ask(Q.infinite(I * zoo)) is True
    assert ask(Q.infinite(x / y), Q.infinite(x) & Q.finite(y) & ~Q.zero(y)) is True
    assert ask(Q.infinite(I * oo - I * oo)) is None
    assert ask(Q.infinite(x * I * oo)) is None
    assert ask(Q.infinite(1 / x), Q.finite(x) & ~Q.zero(x)) is False
    assert ask(Q.infinite(1 / (I * oo))) is False


def test_issue_27441():
    # https://github.com/sympy/sympy/issues/27441
    assert ask(Q.composite(y), Q.integer(y) & Q.positive(y) & ~Q.prime(y)) is None


def test_issue_27447():
    x,y,z = symbols('x y z')
    a = x*y
    assert ask(Q.finite(a), Q.finite(x)  & ~Q.finite(y)) is None
    assert ask(Q.finite(a), ~Q.finite(x)  & Q.finite(y)) is None

    a = x*y*z
    assert ask(Q.finite(a), Q.finite(x) & Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & Q.finite(z) ) is None
    assert ask(Q.finite(a), Q.finite(x) & ~Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)
        & Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & Q.finite(y)
        & ~Q.finite(z)) is None
    assert ask(Q.finite(a), ~Q.finite(x) & ~Q.finite(y)
        & Q.finite(z)) is None


@XFAIL
def test_issue_27662_xfail():
    assert ask(Q.finite(x*y), ~Q.finite(x)
        & Q.zero(y)) is None


@XFAIL
def test_bounded_xfail():
    """We need to support relations in ask for this to work"""
    assert ask(Q.finite(sin(x)**x)) is True
    assert ask(Q.finite(cos(x)**x)) is True


def test_commutative():
    """By default objects are Q.commutative that is why it returns True
    for both key=True and key=False"""
    assert ask(Q.commutative(x)) is True
    assert ask(Q.commutative(x), ~Q.commutative(x)) is False
    assert ask(Q.commutative(x), Q.complex(x)) is True
    assert ask(Q.commutative(x), Q.imaginary(x)) is True
    assert ask(Q.commutative(x), Q.real(x)) is True
    assert ask(Q.commutative(x), Q.positive(x)) is True
    assert ask(Q.commutative(x), ~Q.commutative(y)) is True

    assert ask(Q.commutative(2*x)) is True
    assert ask(Q.commutative(2*x), ~Q.commutative(x)) is False

    assert ask(Q.commutative(x + 1)) is True
    assert ask(Q.commutative(x + 1), ~Q.commutative(x)) is False

    assert ask(Q.commutative(x**2)) is True
    assert ask(Q.commutative(x**2), ~Q.commutative(x)) is False

    assert ask(Q.commutative(log(x))) is True


@_both_exp_pow
def test_complex():
    assert ask(Q.complex(x)) is None
    assert ask(Q.complex(x), Q.complex(x)) is True
    assert ask(Q.complex(x), Q.complex(y)) is None
    assert ask(Q.complex(x), ~Q.complex(x)) is False
    assert ask(Q.complex(x), Q.real(x)) is True
    assert ask(Q.complex(x), ~Q.real(x)) is None
    assert ask(Q.complex(x), Q.rational(x)) is True
    assert ask(Q.complex(x), Q.irrational(x)) is True
    assert ask(Q.complex(x), Q.positive(x)) is True
    assert ask(Q.complex(x), Q.imaginary(x)) is True
    assert ask(Q.complex(x), Q.algebraic(x)) is True

    # a+b
    assert ask(Q.complex(x + 1), Q.complex(x)) is True
    assert ask(Q.complex(x + 1), Q.real(x)) is True
    assert ask(Q.complex(x + 1), Q.rational(x)) is True
    assert ask(Q.complex(x + 1), Q.irrational(x)) is True
    assert ask(Q.complex(x + 1), Q.imaginary(x)) is True
    assert ask(Q.complex(x + 1), Q.integer(x)) is True
    assert ask(Q.complex(x + 1), Q.even(x)) is True
    assert ask(Q.complex(x + 1), Q.odd(x)) is True
    assert ask(Q.complex(x + y), Q.complex(x) & Q.complex(y)) is True
    assert ask(Q.complex(x + y), Q.real(x) & Q.imaginary(y)) is True

    # a*x +b
    assert ask(Q.complex(2*x + 1), Q.complex(x)) is True
    assert ask(Q.complex(2*x + 1), Q.real(x)) is True
    assert ask(Q.complex(2*x + 1), Q.positive(x)) is True
    assert ask(Q.complex(2*x + 1), Q.rational(x)) is True
    assert ask(Q.complex(2*x + 1), Q.irrational(x)) is True
    assert ask(Q.complex(2*x + 1), Q.imaginary(x)) is True
    assert ask(Q.complex(2*x + 1), Q.integer(x)) is True
    assert ask(Q.complex(2*x + 1), Q.even(x)) is True
    assert ask(Q.complex(2*x + 1), Q.odd(x)) is True

    # x**2
    assert ask(Q.complex(x**2), Q.complex(x)) is True
    assert ask(Q.complex(x**2), Q.real(x)) is True
    assert ask(Q.complex(x**2), Q.positive(x)) is True
    assert ask(Q.complex(x**2), Q.rational(x)) is True
    assert ask(Q.complex(x**2), Q.irrational(x)) is True
    assert ask(Q.complex(x**2), Q.imaginary(x)) is True
    assert ask(Q.complex(x**2), Q.integer(x)) is True
    assert ask(Q.complex(x**2), Q.even(x)) is True
    assert ask(Q.complex(x**2), Q.odd(x)) is True

    # 2**x
    assert ask(Q.complex(2**x), Q.complex(x)) is True
    assert ask(Q.complex(2**x), Q.real(x)) is True
    assert ask(Q.complex(2**x), Q.positive(x)) is True
    assert ask(Q.complex(2**x), Q.rational(x)) is True
    assert ask(Q.complex(2**x), Q.irrational(x)) is True
    assert ask(Q.complex(2**x), Q.imaginary(x)) is True
    assert ask(Q.complex(2**x), Q.integer(x)) is True
    assert ask(Q.complex(2**x), Q.even(x)) is True
    assert ask(Q.complex(2**x), Q.odd(x)) is True
    assert ask(Q.complex(x**y), Q.complex(x) & Q.complex(y)) is True

    # trigonometric expressions
    assert ask(Q.complex(sin(x))) is True
    assert ask(Q.complex(sin(2*x + 1))) is True
    assert ask(Q.complex(cos(x))) is True
    assert ask(Q.complex(cos(2*x + 1))) is True

    # exponential
    assert ask(Q.complex(exp(x))) is True
    assert ask(Q.complex(exp(x))) is True

    # Q.complexes
    assert ask(Q.complex(Abs(x))) is True
    assert ask(Q.complex(re(x))) is True
    assert ask(Q.complex(im(x))) is True


def test_even_query():
    assert ask(Q.even(x)) is None
    assert ask(Q.even(x), Q.integer(x)) is None
    assert ask(Q.even(x), ~Q.integer(x)) is False
    assert ask(Q.even(x), Q.rational(x)) is None
    assert ask(Q.even(x), Q.positive(x)) is None

    assert ask(Q.even(2*x)) is None
    assert ask(Q.even(2*x), Q.integer(x)) is True
    assert ask(Q.even(2*x), Q.even(x)) is True
    assert ask(Q.even(2*x), Q.irrational(x)) is False
    assert ask(Q.even(2*x), Q.odd(x)) is True
    assert ask(Q.even(2*x), ~Q.integer(x)) is None
    assert ask(Q.even(3*x), Q.integer(x)) is None
    assert ask(Q.even(3*x), Q.even(x)) is True
    assert ask(Q.even(3*x), Q.odd(x)) is False

    assert ask(Q.even(x + 1), Q.odd(x)) is True
    assert ask(Q.even(x + 1), Q.even(x)) is False
    assert ask(Q.even(x + 2), Q.odd(x)) is False
    assert ask(Q.even(x + 2), Q.even(x)) is True
    assert ask(Q.even(7 - x), Q.odd(x)) is True
    assert ask(Q.even(7 + x), Q.odd(x)) is True
    assert ask(Q.even(x + y), Q.odd(x) & Q.odd(y)) is True
    assert ask(Q.even(x + y), Q.odd(x) & Q.even(y)) is False
    assert ask(Q.even(x + y), Q.even(x) & Q.even(y)) is True

    assert ask(Q.even(2*x + 1), Q.integer(x)) is False
    assert ask(Q.even(2*x*y), Q.rational(x) & Q.rational(x)) is None
    assert ask(Q.even(2*x*y), Q.irrational(x) & Q.irrational(x)) is None

    assert ask(Q.even(x + y + z), Q.odd(x) & Q.odd(y) & Q.even(z)) is True
    assert ask(Q.even(x + y + z + t),
        Q.odd(x) & Q.odd(y) & Q.even(z) & Q.integer(t)) is None

    assert ask(Q.even(Abs(x)), Q.even(x)) is True
    assert ask(Q.even(Abs(x)), ~Q.even(x)) is None
    assert ask(Q.even(re(x)), Q.even(x)) is True
    assert ask(Q.even(re(x)), ~Q.even(x)) is None
    assert ask(Q.even(im(x)), Q.even(x)) is True
    assert ask(Q.even(im(x)), Q.real(x)) is True

    assert ask(Q.even((-1)**n), Q.integer(n)) is False

    assert ask(Q.even(k**2), Q.even(k)) is True
    assert ask(Q.even(n**2), Q.odd(n)) is False
    assert ask(Q.even(2**k), Q.even(k)) is None
    assert ask(Q.even(x**2)) is None

    assert ask(Q.even(k**m), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.even(n**m), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is False

    assert ask(Q.even(k**p), Q.even(k) & Q.integer(p) & Q.positive(p)) is True
    assert ask(Q.even(n**p), Q.odd(n) & Q.integer(p) & Q.positive(p)) is False

    assert ask(Q.even(m**k), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.even(p**k), Q.even(k) & Q.integer(p) & Q.positive(p)) is None

    assert ask(Q.even(m**n), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.even(p**n), Q.odd(n) & Q.integer(p) & Q.positive(p)) is None

    assert ask(Q.even(k**x), Q.even(k)) is None
    assert ask(Q.even(n**x), Q.odd(n)) is None

    assert ask(Q.even(x*y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.even(x*x), Q.integer(x)) is None
    assert ask(Q.even(x*(x + y)), Q.integer(x) & Q.odd(y)) is True
    assert ask(Q.even(x*(x + y)), Q.integer(x) & Q.even(y)) is None


@XFAIL
def test_evenness_in_ternary_integer_product_with_odd():
    # Tests that oddness inference is independent of term ordering.
    # Term ordering at the point of testing depends on SymPy's symbol order, so
    # we try to force a different order by modifying symbol names.
    assert ask(Q.even(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is True
    assert ask(Q.even(y*x*(x + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is True


def test_evenness_in_ternary_integer_product_with_even():
    assert ask(Q.even(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.even(z)) is None


def test_extended_real():
    assert ask(Q.extended_real(x), Q.positive_infinite(x)) is True
    assert ask(Q.extended_real(x), Q.positive(x)) is True
    assert ask(Q.extended_real(x), Q.zero(x)) is True
    assert ask(Q.extended_real(x), Q.negative(x)) is True
    assert ask(Q.extended_real(x), Q.negative_infinite(x)) is True

    assert ask(Q.extended_real(-x), Q.positive(x)) is True
    assert ask(Q.extended_real(-x), Q.negative(x)) is True

    assert ask(Q.extended_real(x + S.Infinity), Q.real(x)) is True

    assert ask(Q.extended_real(x), Q.infinite(x)) is None


@_both_exp_pow
def test_rational():
    assert ask(Q.rational(x), Q.integer(x)) is True
    assert ask(Q.rational(x), Q.irrational(x)) is False
    assert ask(Q.rational(x), Q.real(x)) is None
    assert ask(Q.rational(x), Q.positive(x)) is None
    assert ask(Q.rational(x), Q.negative(x)) is None
    assert ask(Q.rational(x), Q.nonzero(x)) is None
    assert ask(Q.rational(x), ~Q.algebraic(x)) is False

    assert ask(Q.rational(2*x), Q.rational(x)) is True
    assert ask(Q.rational(2*x), Q.integer(x)) is True
    assert ask(Q.rational(2*x), Q.even(x)) is True
    assert ask(Q.rational(2*x), Q.odd(x)) is True
    assert ask(Q.rational(2*x), Q.irrational(x)) is False

    assert ask(Q.rational(x/2), Q.rational(x)) is True
    assert ask(Q.rational(x/2), Q.integer(x)) is True
    assert ask(Q.rational(x/2), Q.even(x)) is True
    assert ask(Q.rational(x/2), Q.odd(x)) is True
    assert ask(Q.rational(x/2), Q.irrational(x)) is False

    assert ask(Q.rational(1/x), Q.rational(x) & Q.nonzero(x)) is True
    assert ask(Q.rational(1/x), Q.integer(x) & Q.nonzero(x)) is True
    assert ask(Q.rational(1/x), Q.even(x) & Q.nonzero(x)) is True
    assert ask(Q.rational(1/x), Q.odd(x)) is True
    assert ask(Q.rational(1/x), Q.irrational(x)) is False

    assert ask(Q.rational(2/x), Q.rational(x) & Q.nonzero(x)) is True
    assert ask(Q.rational(2/x), Q.integer(x) & Q.nonzero(x)) is True
    assert ask(Q.rational(2/x), Q.even(x) & Q.nonzero(x)) is True
    assert ask(Q.rational(2/x), Q.odd(x)) is True
    assert ask(Q.rational(2/x), Q.irrational(x)) is False

    assert ask(Q.rational(x), ~Q.algebraic(x)) is False

    # with multiple symbols
    assert ask(Q.rational(x*y), Q.irrational(x) & Q.irrational(y)) is None
    assert ask(Q.rational(y/x), Q.rational(x) & Q.rational(y) & Q.nonzero(x)) is True
    assert ask(Q.rational(y/x), Q.integer(x) & Q.rational(y) & Q.nonzero(x)) is True
    assert ask(Q.rational(y/x), Q.even(x) & Q.rational(y) & Q.nonzero(x)) is True
    assert ask(Q.rational(y/x), Q.odd(x) & Q.rational(y)) is True
    assert ask(Q.rational(y/x), Q.irrational(x) & Q.rational(y) & Q.nonzero(y)) is False

    for f in [exp, sin, tan, asin, atan, cos]:
        assert ask(Q.rational(f(7))) is False
        assert ask(Q.rational(f(7, evaluate=False))) is False
        assert ask(Q.rational(f(0, evaluate=False))) is True
        assert ask(Q.rational(f(x)), Q.rational(x)) is None
        assert ask(Q.rational(f(x)), Q.rational(x) & Q.nonzero(x)) is False

    for g in [log, acos]:
        assert ask(Q.rational(g(7))) is False
        assert ask(Q.rational(g(7, evaluate=False))) is False
        assert ask(Q.rational(g(1, evaluate=False))) is True
        assert ask(Q.rational(g(x)), Q.rational(x)) is None
        assert ask(Q.rational(g(x)), Q.rational(x) & Q.nonzero(x - 1)) is False

    for h in [cot, acot]:
        assert ask(Q.rational(h(7))) is False
        assert ask(Q.rational(h(7, evaluate=False))) is False
        assert ask(Q.rational(h(x)), Q.rational(x)) is False

    # https://github.com/sympy/sympy/issues/27442
    assert ask(Q.rational(x**y),Q.irrational(x) & Q.rational(y)) is None
    assert ask(Q.rational(x**y),Q.integer(x) & Q.prime(x) & Q.rational(y)) is None
    assert ask(Q.rational(x**y),Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.rational(x**y),Q.integer(x) & Q.eq(x,0) & Q.integer(y)) is None
    assert ask(Q.rational(x**y),Q.eq(x,1) & Q.rational(y)) is None
    assert ask(Q.rational(x**y),Q.eq(x,-1) & Q.rational(y)) is None
    assert ask(Q.rational(x**y), Q.prime(x) & Q.rational(y)) is None
    assert ask(Q.rational(x**y), ~Q.rational(x) & Q.integer(y) ) is None
    assert ask(Q.rational(Pow(-1, x, evaluate=False), Q.rational(x))) is None
    assert ask(Q.rational(x**y), Q.integer(y) & ~Q. algebraic(x)) is None
    assert ask(Q.rational(x**y), Q.integer(y) & ~Q. algebraic(x) & ~Q.zero(x)) is None
    assert ask(Q.rational(x**y), Q.integer(y) & ~Q.algebraic(x) & Q.complex(x) & ~Q.real(x)) is None
    assert ask(Q.rational(x**y), Q.integer(y) & ~Q.algebraic(x) & Q.complex(x)) is None


def test_hermitian():
    assert ask(Q.hermitian(x)) is None
    assert ask(Q.hermitian(x), Q.antihermitian(x)) is None
    assert ask(Q.hermitian(x), Q.imaginary(x)) is False
    assert ask(Q.hermitian(x), Q.prime(x)) is True
    assert ask(Q.hermitian(x), Q.real(x)) is True
    assert ask(Q.hermitian(x), Q.zero(x)) is True

    assert ask(Q.hermitian(x + 1), Q.antihermitian(x)) is None
    assert ask(Q.hermitian(x + 1), Q.complex(x)) is None
    assert ask(Q.hermitian(x + 1), Q.hermitian(x)) is True
    assert ask(Q.hermitian(x + 1), Q.imaginary(x)) is False
    assert ask(Q.hermitian(x + 1), Q.real(x)) is True
    assert ask(Q.hermitian(x + I), Q.antihermitian(x)) is None
    assert ask(Q.hermitian(x + I), Q.complex(x)) is None
    assert ask(Q.hermitian(x + I), Q.hermitian(x)) is False
    assert ask(Q.hermitian(x + I), Q.imaginary(x)) is None
    assert ask(Q.hermitian(x + I), Q.real(x)) is False
    assert ask(
        Q.hermitian(x + y), Q.antihermitian(x) & Q.antihermitian(y)) is None
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.complex(y)) is None
    assert ask(
        Q.hermitian(x + y), Q.antihermitian(x) & Q.hermitian(y)) is None
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.imaginary(y)) is None
    assert ask(Q.hermitian(x + y), Q.antihermitian(x) & Q.real(y)) is None
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.complex(y)) is None
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.hermitian(y)) is True
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.imaginary(y)) is False
    assert ask(Q.hermitian(x + y), Q.hermitian(x) & Q.real(y)) is True
    assert ask(Q.hermitian(x + y), Q.imaginary(x) & Q.complex(y)) is None
    assert ask(Q.hermitian(x + y), Q.imaginary(x) & Q.imaginary(y)) is None
    assert ask(Q.hermitian(x + y), Q.imaginary(x) & Q.real(y)) is False
    assert ask(Q.hermitian(x + y), Q.real(x) & Q.complex(y)) is None
    assert ask(Q.hermitian(x + y), Q.real(x) & Q.real(y)) is True

    assert ask(Q.hermitian(I*x), Q.antihermitian(x)) is True
    assert ask(Q.hermitian(I*x), Q.complex(x)) is None
    assert ask(Q.hermitian(I*x), Q.hermitian(x)) is False
    assert ask(Q.hermitian(I*x), Q.imaginary(x)) is True
    assert ask(Q.hermitian(I*x), Q.real(x)) is False
    assert ask(Q.hermitian(x*y), Q.hermitian(x) & Q.real(y)) is True

    assert ask(
        Q.hermitian(x + y + z), Q.real(x) & Q.real(y) & Q.real(z)) is True
    assert ask(Q.hermitian(x + y + z),
        Q.real(x) & Q.real(y) & Q.imaginary(z)) is False
    assert ask(Q.hermitian(x + y + z),
        Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is None
    assert ask(Q.hermitian(x + y + z),
        Q.imaginary(x) & Q.imaginary(y) & Q.imaginary(z)) is None

    assert ask(Q.antihermitian(x)) is None
    assert ask(Q.antihermitian(x), Q.real(x)) is False
    assert ask(Q.antihermitian(x), Q.prime(x)) is False

    assert ask(Q.antihermitian(x + 1), Q.antihermitian(x)) is False
    assert ask(Q.antihermitian(x + 1), Q.complex(x)) is None
    assert ask(Q.antihermitian(x + 1), Q.hermitian(x)) is None
    assert ask(Q.antihermitian(x + 1), Q.imaginary(x)) is False
    assert ask(Q.antihermitian(x + 1), Q.real(x)) is None
    assert ask(Q.antihermitian(x + I), Q.antihermitian(x)) is True
    assert ask(Q.antihermitian(x + I), Q.complex(x)) is None
    assert ask(Q.antihermitian(x + I), Q.hermitian(x)) is None
    assert ask(Q.antihermitian(x + I), Q.imaginary(x)) is True
    assert ask(Q.antihermitian(x + I), Q.real(x)) is False
    assert ask(Q.antihermitian(x), Q.zero(x)) is True

    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.antihermitian(y)
    ) is True
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.complex(y)) is None
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.hermitian(y)) is None
    assert ask(
        Q.antihermitian(x + y), Q.antihermitian(x) & Q.imaginary(y)) is True
    assert ask(Q.antihermitian(x + y), Q.antihermitian(x) & Q.real(y)
        ) is False
    assert ask(Q.antihermitian(x + y), Q.hermitian(x) & Q.complex(y)) is None
    assert ask(Q.antihermitian(x + y), Q.hermitian(x) & Q.hermitian(y)
        ) is None
    assert ask(
        Q.antihermitian(x + y), Q.hermitian(x) & Q.imaginary(y)) is None
    assert ask(Q.antihermitian(x + y), Q.hermitian(x) & Q.real(y)) is None
    assert ask(Q.antihermitian(x + y), Q.imaginary(x) & Q.complex(y)) is None
    assert ask(Q.antihermitian(x + y), Q.imaginary(x) & Q.imaginary(y)) is True
    assert ask(Q.antihermitian(x + y), Q.imaginary(x) & Q.real(y)) is False
    assert ask(Q.antihermitian(x + y), Q.real(x) & Q.complex(y)) is None
    assert ask(Q.antihermitian(x + y), Q.real(x) & Q.real(y)) is None

    assert ask(Q.antihermitian(I*x), Q.real(x)) is True
    assert ask(Q.antihermitian(I*x), Q.antihermitian(x)) is False
    assert ask(Q.antihermitian(I*x), Q.complex(x)) is None
    assert ask(Q.antihermitian(x*y), Q.antihermitian(x) & Q.real(y)) is True

    assert ask(Q.antihermitian(x + y + z),
        Q.real(x) & Q.real(y) & Q.real(z)) is None
    assert ask(Q.antihermitian(x + y + z),
        Q.real(x) & Q.real(y) & Q.imaginary(z)) is None
    assert ask(Q.antihermitian(x + y + z),
        Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is False
    assert ask(Q.antihermitian(x + y + z),
        Q.imaginary(x) & Q.imaginary(y) & Q.imaginary(z)) is True


@_both_exp_pow
def test_imaginary():
    assert ask(Q.imaginary(x)) is None
    assert ask(Q.imaginary(x), Q.real(x)) is False
    assert ask(Q.imaginary(x), Q.prime(x)) is False

    assert ask(Q.imaginary(x + 1), Q.real(x)) is False
    assert ask(Q.imaginary(x + 1), Q.imaginary(x)) is False
    assert ask(Q.imaginary(x + I), Q.real(x)) is False
    assert ask(Q.imaginary(x + I), Q.imaginary(x)) is True
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.imaginary(y)) is True
    assert ask(Q.imaginary(x + y), Q.real(x) & Q.real(y)) is False
    assert ask(Q.imaginary(x + y), Q.imaginary(x) & Q.real(y)) is False
    assert ask(Q.imaginary(x + y), Q.complex(x) & Q.real(y)) is None
    assert ask(
        Q.imaginary(x + y + z), Q.real(x) & Q.real(y) & Q.real(z)) is False
    assert ask(Q.imaginary(x + y + z),
        Q.real(x) & Q.real(y) & Q.imaginary(z)) is None
    assert ask(Q.imaginary(x + y + z),
        Q.real(x) & Q.imaginary(y) & Q.imaginary(z)) is False

    assert ask(Q.imaginary(I*x), Q.real(x)) is True
    assert ask(Q.imaginary(I*x), Q.imaginary(x)) is False
    assert ask(Q.imaginary(I*x), Q.complex(x)) is None
    assert ask(Q.imaginary(x*y), Q.imaginary(x) & Q.real(y)) is True
    assert ask(Q.imaginary(x*y), Q.real(x) & Q.real(y)) is False

    assert ask(Q.imaginary(I**x), Q.negative(x)) is None
    assert ask(Q.imaginary(I**x), Q.positive(x)) is None
    assert ask(Q.imaginary(I**x), Q.even(x)) is False
    assert ask(Q.imaginary(I**x), Q.odd(x)) is True
    assert ask(Q.imaginary(I**x), Q.imaginary(x)) is False
    assert ask(Q.imaginary((2*I)**x), Q.imaginary(x)) is False
    assert ask(Q.imaginary(x**0), Q.imaginary(x)) is False
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.integer(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(y) & Q.integer(x)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.odd(y)) is True
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.rational(y)) is None
    assert ask(Q.imaginary(x**y), Q.imaginary(x) & Q.even(y)) is False

    assert ask(Q.imaginary(x**y), Q.real(x) & Q.integer(y)) is False
    assert ask(Q.imaginary(x**y), Q.positive(x) & Q.real(y)) is False
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.real(y)) is None
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.real(y) & ~Q.rational(y)) is False
    assert ask(Q.imaginary(x**y), Q.integer(x) & Q.imaginary(y)) is None
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y) & Q.integer(2*y)) is True
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y) & ~Q.integer(2*y)) is False
    assert ask(Q.imaginary(x**y), Q.negative(x) & Q.rational(y)) is None
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.rational(y) & ~Q.integer(2*y)) is False
    assert ask(Q.imaginary(x**y), Q.real(x) & Q.rational(y) & Q.integer(2*y)) is None

    # logarithm
    assert ask(Q.imaginary(log(I))) is True
    assert ask(Q.imaginary(log(2*I))) is False
    assert ask(Q.imaginary(log(I + 1))) is False
    assert ask(Q.imaginary(log(x)), Q.complex(x)) is None
    assert ask(Q.imaginary(log(x)), Q.imaginary(x)) is None
    assert ask(Q.imaginary(log(x)), Q.positive(x)) is False
    assert ask(Q.imaginary(log(exp(x))), Q.complex(x)) is None
    assert ask(Q.imaginary(log(exp(x))), Q.imaginary(x)) is None  # zoo/I/a+I*b
    assert ask(Q.imaginary(log(exp(I)))) is True

    # exponential
    assert ask(Q.imaginary(exp(x)**x), Q.imaginary(x)) is False
    eq = Pow(exp(pi*I*x, evaluate=False), x, evaluate=False)
    assert ask(Q.imaginary(eq), Q.even(x)) is False
    eq = Pow(exp(pi*I*x/2, evaluate=False), x, evaluate=False)
    assert ask(Q.imaginary(eq), Q.odd(x)) is True
    assert ask(Q.imaginary(exp(3*I*pi*x)**x), Q.integer(x)) is False
    assert ask(Q.imaginary(exp(2*pi*I, evaluate=False))) is False
    assert ask(Q.imaginary(exp(pi*I/2, evaluate=False))) is True

    # issue 7886
    assert ask(Q.imaginary(Pow(x, Rational(1, 4))), Q.real(x) & Q.negative(x)) is False


def test_integer():
    assert ask(Q.integer(x)) is None
    assert ask(Q.integer(x), Q.integer(x)) is True
    assert ask(Q.integer(x), ~Q.integer(x)) is False
    assert ask(Q.integer(x), ~Q.real(x)) is False
    assert ask(Q.integer(x), ~Q.positive(x)) is None
    assert ask(Q.integer(x), Q.even(x) | Q.odd(x)) is True

    assert ask(Q.integer(2*x), Q.integer(x)) is True
    assert ask(Q.integer(2*x), Q.even(x)) is True
    assert ask(Q.integer(2*x), Q.prime(x)) is True
    assert ask(Q.integer(2*x), Q.rational(x)) is None
    assert ask(Q.integer(2*x), Q.real(x)) is None
    assert ask(Q.integer(sqrt(2)*x), Q.integer(x)) is False
    assert ask(Q.integer(sqrt(2)*x), Q.irrational(x)) is None

    assert ask(Q.integer(x/2), Q.odd(x)) is False
    assert ask(Q.integer(x/2), Q.even(x)) is True
    assert ask(Q.integer(x/3), Q.odd(x)) is None
    assert ask(Q.integer(x/3), Q.even(x)) is None

    # https://github.com/sympy/sympy/issues/7286
    assert ask(Q.integer(Abs(x)),Q.integer(x)) is True
    assert ask(Q.integer(Abs(-x)),Q.integer(x)) is True
    assert ask(Q.integer(Abs(x)), ~Q.integer(x)) is None
    assert ask(Q.integer(Abs(x)),Q.complex(x)) is None
    assert ask(Q.integer(Abs(x+I*y)),Q.real(x) & Q.real(y)) is None

    # https://github.com/sympy/sympy/issues/27739
    assert ask(Q.integer(x/y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.integer(1/x), Q.integer(x)) is None
    assert ask(Q.integer(x**y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.integer(sqrt(5))) is False
    assert ask(Q.integer(x**y), Q.nonzero(x) & Q.zero(y)) is True
    assert ask(Q.integer(x**y), Q.integer(x) & Q.integer(y) & Q.positive(y)) is True
    assert ask(Q.integer(-1**x), Q.integer(x)) is True
    assert ask(Q.integer(x**y), Q.integer(x) & Q.integer(y) & Q.positive(y)) is True
    assert ask(Q.integer(x**y), Q.zero(x) & Q.integer(y) & Q.positive(y)) is True
    assert ask(Q.integer(pi**x), Q.zero(x)) is True
    assert ask(Q.integer(x**y), Q.imaginary(x) & Q.zero(y)) is True


def test_negative():
    assert ask(Q.negative(x), Q.negative(x)) is True
    assert ask(Q.negative(x), Q.positive(x)) is False
    assert ask(Q.negative(x), ~Q.real(x)) is False
    assert ask(Q.negative(x), Q.prime(x)) is False
    assert ask(Q.negative(x), ~Q.prime(x)) is None

    assert ask(Q.negative(-x), Q.positive(x)) is True
    assert ask(Q.negative(-x), ~Q.positive(x)) is None
    assert ask(Q.negative(-x), Q.negative(x)) is False
    assert ask(Q.negative(-x), Q.positive(x)) is True

    assert ask(Q.negative(x - 1), Q.negative(x)) is True
    assert ask(Q.negative(x + y)) is None
    assert ask(Q.negative(x + y), Q.negative(x)) is None
    assert ask(Q.negative(x + y), Q.negative(x) & Q.negative(y)) is True
    assert ask(Q.negative(x + y), Q.negative(x) & Q.nonpositive(y)) is True
    assert ask(Q.negative(2 + I)) is False
    # although this could be False, it is representative of expressions
    # that don't evaluate to a zero with precision
    assert ask(Q.negative(cos(I)**2 + sin(I)**2 - 1)) is None
    assert ask(Q.negative(-I + I*(cos(2)**2 + sin(2)**2))) is None

    assert ask(Q.negative(x**2)) is None
    assert ask(Q.negative(x**2), Q.real(x)) is False
    assert ask(Q.negative(x**1.4), Q.real(x)) is None

    assert ask(Q.negative(x**I), Q.positive(x)) is None

    assert ask(Q.negative(x*y)) is None
    assert ask(Q.negative(x*y), Q.positive(x) & Q.positive(y)) is False
    assert ask(Q.negative(x*y), Q.positive(x) & Q.negative(y)) is True
    assert ask(Q.negative(x*y), Q.complex(x) & Q.complex(y)) is None

    assert ask(Q.negative(x**y)) is None
    assert ask(Q.negative(x**y), Q.negative(x) & Q.even(y)) is False
    assert ask(Q.negative(x**y), Q.negative(x) & Q.odd(y)) is True
    assert ask(Q.negative(x**y), Q.positive(x) & Q.integer(y)) is False

    assert ask(Q.negative(Abs(x))) is False


def test_nonzero():
    assert ask(Q.nonzero(x)) is None
    assert ask(Q.nonzero(x), Q.real(x)) is None
    assert ask(Q.nonzero(x), Q.positive(x)) is True
    assert ask(Q.nonzero(x), Q.negative(x)) is True
    assert ask(Q.nonzero(x), Q.negative(x) | Q.positive(x)) is True

    assert ask(Q.nonzero(x + y)) is None
    assert ask(Q.nonzero(x + y), Q.positive(x) & Q.positive(y)) is True
    assert ask(Q.nonzero(x + y), Q.positive(x) & Q.negative(y)) is None
    assert ask(Q.nonzero(x + y), Q.negative(x) & Q.negative(y)) is True

    assert ask(Q.nonzero(2*x)) is None
    assert ask(Q.nonzero(2*x), Q.positive(x)) is True
    assert ask(Q.nonzero(2*x), Q.negative(x)) is True
    assert ask(Q.nonzero(x*y), Q.nonzero(x)) is None
    assert ask(Q.nonzero(x*y), Q.nonzero(x) & Q.nonzero(y)) is True

    assert ask(Q.nonzero(x**y), Q.nonzero(x)) is True

    assert ask(Q.nonzero(Abs(x))) is None
    assert ask(Q.nonzero(Abs(x)), Q.nonzero(x)) is True

    assert ask(Q.nonzero(log(exp(2*I)))) is False
    # although this could be False, it is representative of expressions
    # that don't evaluate to a zero with precision
    assert ask(Q.nonzero(cos(1)**2 + sin(1)**2 - 1)) is None


def test_zero():
    assert ask(Q.zero(x)) is None
    assert ask(Q.zero(x), Q.real(x)) is None
    assert ask(Q.zero(x), Q.positive(x)) is False
    assert ask(Q.zero(x), Q.negative(x)) is False
    assert ask(Q.zero(x), Q.negative(x) | Q.positive(x)) is False

    assert ask(Q.zero(x), Q.nonnegative(x) & Q.nonpositive(x)) is True

    assert ask(Q.zero(x + y)) is None
    assert ask(Q.zero(x + y), Q.positive(x) & Q.positive(y)) is False
    assert ask(Q.zero(x + y), Q.positive(x) & Q.negative(y)) is None
    assert ask(Q.zero(x + y), Q.negative(x) & Q.negative(y)) is False

    assert ask(Q.zero(2*x)) is None
    assert ask(Q.zero(2*x), Q.positive(x)) is False
    assert ask(Q.zero(2*x), Q.negative(x)) is False
    assert ask(Q.zero(x*y), Q.nonzero(x)) is None

    assert ask(Q.zero(Abs(x))) is None
    assert ask(Q.zero(Abs(x)), Q.zero(x)) is True

    assert ask(Q.integer(x), Q.zero(x)) is True
    assert ask(Q.even(x), Q.zero(x)) is True
    assert ask(Q.odd(x), Q.zero(x)) is False
    assert ask(Q.zero(x), Q.even(x)) is None
    assert ask(Q.zero(x), Q.odd(x)) is False
    assert ask(Q.zero(x) | Q.zero(y), Q.zero(x*y)) is True


def test_odd_query():
    assert ask(Q.odd(x)) is None
    assert ask(Q.odd(x), Q.odd(x)) is True
    assert ask(Q.odd(x), Q.integer(x)) is None
    assert ask(Q.odd(x), ~Q.integer(x)) is False
    assert ask(Q.odd(x), Q.rational(x)) is None
    assert ask(Q.odd(x), Q.positive(x)) is None

    assert ask(Q.odd(-x), Q.odd(x)) is True

    assert ask(Q.odd(2*x)) is None
    assert ask(Q.odd(2*x), Q.integer(x)) is False
    assert ask(Q.odd(2*x), Q.odd(x)) is False
    assert ask(Q.odd(2*x), Q.irrational(x)) is False
    assert ask(Q.odd(2*x), ~Q.integer(x)) is None
    assert ask(Q.odd(3*x), Q.integer(x)) is None

    assert ask(Q.odd(x/3), Q.odd(x)) is None
    assert ask(Q.odd(x/3), Q.even(x)) is None

    assert ask(Q.odd(x + 1), Q.even(x)) is True
    assert ask(Q.odd(x + 2), Q.even(x)) is False
    assert ask(Q.odd(x + 2), Q.odd(x)) is True
    assert ask(Q.odd(3 - x), Q.odd(x)) is False
    assert ask(Q.odd(3 - x), Q.even(x)) is True
    assert ask(Q.odd(3 + x), Q.odd(x)) is False
    assert ask(Q.odd(3 + x), Q.even(x)) is True
    assert ask(Q.odd(x + y), Q.odd(x) & Q.odd(y)) is False
    assert ask(Q.odd(x + y), Q.odd(x) & Q.even(y)) is True
    assert ask(Q.odd(x - y), Q.even(x) & Q.odd(y)) is True
    assert ask(Q.odd(x - y), Q.odd(x) & Q.odd(y)) is False

    assert ask(Q.odd(x + y + z), Q.odd(x) & Q.odd(y) & Q.even(z)) is False
    assert ask(Q.odd(x + y + z + t),
        Q.odd(x) & Q.odd(y) & Q.even(z) & Q.integer(t)) is None

    assert ask(Q.odd(2*x + 1), Q.integer(x)) is True
    assert ask(Q.odd(2*x + y), Q.integer(x) & Q.odd(y)) is True
    assert ask(Q.odd(2*x + y), Q.integer(x) & Q.even(y)) is False
    assert ask(Q.odd(2*x + y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.odd(x*y), Q.odd(x) & Q.even(y)) is False
    assert ask(Q.odd(x*y), Q.odd(x) & Q.odd(y)) is True
    assert ask(Q.odd(2*x*y), Q.rational(x) & Q.rational(x)) is None
    assert ask(Q.odd(2*x*y), Q.irrational(x) & Q.irrational(x)) is None

    assert ask(Q.odd(Abs(x)), Q.odd(x)) is True

    assert ask(Q.odd((-1)**n), Q.integer(n)) is True

    assert ask(Q.odd(k**2), Q.even(k)) is False
    assert ask(Q.odd(n**2), Q.odd(n)) is True
    assert ask(Q.odd(3**k), Q.even(k)) is None

    assert ask(Q.odd(k**m), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.odd(n**m), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is True

    assert ask(Q.odd(k**p), Q.even(k) & Q.integer(p) & Q.positive(p)) is False
    assert ask(Q.odd(n**p), Q.odd(n) & Q.integer(p) & Q.positive(p)) is True

    assert ask(Q.odd(m**k), Q.even(k) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.odd(p**k), Q.even(k) & Q.integer(p) & Q.positive(p)) is None

    assert ask(Q.odd(m**n), Q.odd(n) & Q.integer(m) & ~Q.negative(m)) is None
    assert ask(Q.odd(p**n), Q.odd(n) & Q.integer(p) & Q.positive(p)) is None

    assert ask(Q.odd(k**x), Q.even(k)) is None
    assert ask(Q.odd(n**x), Q.odd(n)) is None

    assert ask(Q.odd(x*y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.odd(x*x), Q.integer(x)) is None
    assert ask(Q.odd(x*(x + y)), Q.integer(x) & Q.odd(y)) is False
    assert ask(Q.odd(x*(x + y)), Q.integer(x) & Q.even(y)) is None


@XFAIL
def test_oddness_in_ternary_integer_product_with_odd():
    # Tests that oddness inference is independent of term ordering.
    # Term ordering at the point of testing depends on SymPy's symbol order, so
    # we try to force a different order by modifying symbol names.
    assert ask(Q.odd(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is False
    assert ask(Q.odd(y*x*(x + z)), Q.integer(x) & Q.integer(y) & Q.odd(z)) is False


def test_oddness_in_ternary_integer_product_with_even():
    assert ask(Q.odd(x*y*(y + z)), Q.integer(x) & Q.integer(y) & Q.even(z)) is None


def test_prime():
    assert ask(Q.prime(x), Q.prime(x)) is True
    assert ask(Q.prime(x), ~Q.prime(x)) is False
    assert ask(Q.prime(x), Q.integer(x)) is None
    assert ask(Q.prime(x), ~Q.integer(x)) is False

    assert ask(Q.prime(2*x), Q.integer(x)) is None
    assert ask(Q.prime(x*y)) is None
    assert ask(Q.prime(x*y), Q.prime(x)) is None
    assert ask(Q.prime(x*y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.prime(4*x), Q.integer(x)) is False
    assert ask(Q.prime(4*x)) is None

    assert ask(Q.prime(x**2), Q.integer(x)) is False
    assert ask(Q.prime(x**2), Q.prime(x)) is False

    # https://github.com/sympy/sympy/issues/27446
    assert ask(Q.prime(4**x), Q.integer(x)) is False
    assert ask(Q.prime(p**x), Q.prime(p) & Q.integer(x) & Q.ne(x, 1)) is False
    assert ask(Q.prime(n**x), Q.integer(x) & Q.composite(n)) is False
    assert ask(Q.prime(x**y), Q.integer(x) & Q.integer(y)) is None
    assert ask(Q.prime(2**x), Q.integer(x)) is None
    assert ask(Q.prime(p**x), Q.prime(p) & Q.integer(x)) is None

    # Ideally, these should return True since the base is prime and the exponent is one,
    # but currently, they return None.
    assert ask(Q.prime(x**y), Q.prime(x) & Q.eq(y,1)) is None
    assert ask(Q.prime(x**y), Q.prime(x) & Q.integer(y) & Q.gt(y,0) & Q.lt(y,2)) is None

    assert ask(Q.prime(Pow(x,1, evaluate=False)), Q.prime(x)) is True


@_both_exp_pow
def test_positive():
    assert ask(Q.positive(cos(I) ** 2 + sin(I) ** 2 - 1)) is None
    assert ask(Q.positive(x), Q.positive(x)) is True
    assert ask(Q.positive(x), Q.negative(x)) is False
    assert ask(Q.positive(x), Q.nonzero(x)) is None

    assert ask(Q.positive(-x), Q.positive(x)) is False
    assert ask(Q.positive(-x), Q.negative(x)) is True

    assert ask(Q.positive(x + y), Q.positive(x) & Q.positive(y)) is True
    assert ask(Q.positive(x + y), Q.positive(x) & Q.nonnegative(y)) is True
    assert ask(Q.positive(x + y), Q.positive(x) & Q.negative(y)) is None
    assert ask(Q.positive(x + y), Q.positive(x) & Q.imaginary(y)) is False

    assert ask(Q.positive(2*x), Q.positive(x)) is True
    assumptions = Q.positive(x) & Q.negative(y) & Q.negative(z) & Q.positive(w)
    assert ask(Q.positive(x*y*z)) is None
    assert ask(Q.positive(x*y*z), assumptions) is True
    assert ask(Q.positive(-x*y*z), assumptions) is False

    assert ask(Q.positive(x**I), Q.positive(x)) is None

    assert ask(Q.positive(x**2), Q.positive(x)) is True
    assert ask(Q.positive(x**2), Q.negative(x)) is True
    assert ask(Q.positive(x**3), Q.negative(x)) is False
    assert ask(Q.positive(1/(1 + x**2)), Q.real(x)) is True
    assert ask(Q.positive(2**I)) is False
    assert ask(Q.positive(2 + I)) is False
    # although this could be False, it is representative of expressions
    # that don't evaluate to a zero with precision
    assert ask(Q.positive(cos(I)**2 + sin(I)**2 - 1)) is None
    assert ask(Q.positive(-I + I*(cos(2)**2 + sin(2)**2))) is None

    #exponential
    assert ask(Q.positive(exp(x)), Q.real(x)) is True
    assert ask(~Q.negative(exp(x)), Q.real(x)) is True
    assert ask(Q.positive(x + exp(x)), Q.real(x)) is None
    assert ask(Q.positive(exp(x)), Q.imaginary(x)) is None
    assert ask(Q.positive(exp(2*pi*I, evaluate=False)), Q.imaginary(x)) is True
    assert ask(Q.negative(exp(pi*I, evaluate=False)), Q.imaginary(x)) is True
    assert ask(Q.positive(exp(x*pi*I)), Q.even(x)) is True
    assert ask(Q.positive(exp(x*pi*I)), Q.odd(x)) is False
    assert ask(Q.positive(exp(x*pi*I)), Q.real(x)) is None

    # logarithm
    assert ask(Q.positive(log(x)), Q.imaginary(x)) is False
    assert ask(Q.positive(log(x)), Q.negative(x)) is False
    assert ask(Q.positive(log(x)), Q.positive(x)) is None
    assert ask(Q.positive(log(x + 2)), Q.positive(x)) is True

    # factorial
    assert ask(Q.positive(factorial(x)), Q.integer(x) & Q.positive(x))
    assert ask(Q.positive(factorial(x)), Q.integer(x)) is None

    #absolute value
    assert ask(Q.positive(Abs(x))) is None  # Abs(0) = 0
    assert ask(Q.positive(Abs(x)), Q.positive(x)) is True


def test_nonpositive():
    assert ask(Q.nonpositive(-1))
    assert ask(Q.nonpositive(0))
    assert ask(Q.nonpositive(1)) is False
    assert ask(~Q.positive(x), Q.nonpositive(x))
    assert ask(Q.nonpositive(x), Q.positive(x)) is False
    assert ask(Q.nonpositive(sqrt(-1))) is False
    assert ask(Q.nonpositive(x), Q.imaginary(x)) is False


def test_nonnegative():
    assert ask(Q.nonnegative(-1)) is False
    assert ask(Q.nonnegative(0))
    assert ask(Q.nonnegative(1))
    assert ask(~Q.negative(x), Q.nonnegative(x))
    assert ask(Q.nonnegative(x), Q.negative(x)) is False
    assert ask(Q.nonnegative(sqrt(-1))) is False
    assert ask(Q.nonnegative(x), Q.imaginary(x)) is False

def test_real_basic():
    assert ask(Q.real(x)) is None
    assert ask(Q.real(x), Q.real(x)) is True
    assert ask(Q.real(x), Q.nonzero(x)) is True
    assert ask(Q.real(x), Q.positive(x)) is True
    assert ask(Q.real(x), Q.negative(x)) is True
    assert ask(Q.real(x), Q.integer(x)) is True
    assert ask(Q.real(x), Q.even(x)) is True
    assert ask(Q.real(x), Q.prime(x)) is True

    assert ask(Q.real(x/sqrt(2)), Q.real(x)) is True
    assert ask(Q.real(x/sqrt(-2)), Q.real(x)) is False

    assert ask(Q.real(x + 1), Q.real(x)) is True
    assert ask(Q.real(x + I), Q.real(x)) is False
    assert ask(Q.real(x + I), Q.complex(x)) is None

    assert ask(Q.real(2*x), Q.real(x)) is True
    assert ask(Q.real(I*x), Q.real(x)) is False
    assert ask(Q.real(I*x), Q.imaginary(x)) is True
    assert ask(Q.real(I*x), Q.complex(x)) is None


def test_real_pow():
    assert ask(Q.real(x**2), Q.real(x)) is True
    assert ask(Q.real(sqrt(x)), Q.negative(x)) is False
    assert ask(Q.real(x**y), Q.real(x) & Q.integer(y)) is None
    assert ask(Q.real(x**y), Q.real(x) & Q.real(y)) is None
    assert ask(Q.real(x**y), Q.positive(x) & Q.real(y)) is True
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.imaginary(y)) is None  # I**I or (2*I)**I
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.real(y)) is None  # I**1 or I**0
    assert ask(Q.real(x**y), Q.real(x) & Q.imaginary(y)) is None  # could be exp(2*pi*I) or 2**I
    assert ask(Q.real(x**0), Q.imaginary(x)) is True
    assert ask(Q.real(x**y), Q.positive(x) & Q.real(y)) is True
    assert ask(Q.real(x**y), Q.real(x) & Q.rational(y)) is None
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.integer(y)) is None
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.odd(y)) is False
    assert ask(Q.real(x**y), Q.imaginary(x) & Q.even(y)) is True
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.real(y/z) & Q.rational(y/z) & Q.even(z) & Q.positive(x)) is True
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.rational(y/z) & Q.even(z) & Q.negative(x)) is None
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.integer(y/z)) is None
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.real(y/z) & Q.positive(x)) is True
    assert ask(Q.real(x**(y/z)), Q.real(x) & Q.real(y/z) & Q.negative(x)) is None
    assert ask(Q.real((-I)**i), Q.imaginary(i)) is True
    assert ask(Q.real(I**i), Q.imaginary(i)) is True
    assert ask(Q.real(i**i), Q.imaginary(i)) is None  # i might be 2*I
    assert ask(Q.real(x**i), Q.imaginary(i)) is None  # x could be 0
    assert ask(Q.real(x**(I*pi/log(x))), Q.real(x)) is True

    # https://github.com/sympy/sympy/issues/27485
    assert ask(Q.real(n**p), Q.negative(n) & Q.positive(p)) is None

    # https://github.com/sympy/sympy/issues/16530
    assert ask(Q.real(1/Abs(x))) is None
    assert ask(Q.real(x**y), Q.zero(x) & Q.real(y)) is None
    assert ask(Q.real(x**y), Q.zero(x) & Q.positive(y)) is True


@_both_exp_pow
def test_real_functions():
    # trigonometric functions
    assert ask(Q.real(sin(x))) is None
    assert ask(Q.real(cos(x))) is None
    assert ask(Q.real(sin(x)), Q.real(x)) is True
    assert ask(Q.real(cos(x)), Q.real(x)) is True

    # exponential function
    assert ask(Q.real(exp(x))) is None
    assert ask(Q.real(exp(x)), Q.real(x)) is True
    assert ask(Q.real(x + exp(x)), Q.real(x)) is True
    assert ask(Q.real(exp(2*pi*I, evaluate=False))) is True
    assert ask(Q.real(exp(pi*I, evaluate=False))) is True
    assert ask(Q.real(exp(pi*I/2, evaluate=False))) is False

    # logarithm
    assert ask(Q.real(log(I))) is False
    assert ask(Q.real(log(2*I))) is False
    assert ask(Q.real(log(I + 1))) is False
    assert ask(Q.real(log(x)), Q.complex(x)) is None
    assert ask(Q.real(log(x)), Q.imaginary(x)) is False
    assert ask(Q.real(log(exp(x))), Q.imaginary(x)) is None  # exp(2*pi*I) is 1, log(exp(pi*I)) is pi*I (disregarding periodicity)
    assert ask(Q.real(log(exp(x))), Q.complex(x)) is None
    eq = Pow(exp(2*pi*I*x, evaluate=False), x, evaluate=False)
    assert ask(Q.real(eq), Q.integer(x)) is True
    assert ask(Q.real(exp(x)**x), Q.imaginary(x)) is True
    assert ask(Q.real(exp(x)**x), Q.complex(x)) is None

    # Q.complexes
    assert ask(Q.real(re(x))) is True
    assert ask(Q.real(im(x))) is True


def test_matrix():

    # hermitian
    assert ask(Q.hermitian(Matrix([[2, 2 + I, 4], [2 - I, 3, I], [4, -I, 1]]))) == True
    assert ask(Q.hermitian(Matrix([[2, 2 + I, 4], [2 + I, 3, I], [4, -I, 1]]))) == False
    z = symbols('z', complex=True)
    assert ask(Q.hermitian(Matrix([[2, 2 + I, z], [2 - I, 3, I], [4, -I, 1]]))) == None
    assert ask(Q.hermitian(SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11))))) == True
    assert ask(Q.hermitian(SparseMatrix(((25, 15, -5), (15, I, 0), (-5, 0, 11))))) == False
    assert ask(Q.hermitian(SparseMatrix(((25, 15, -5), (15, z, 0), (-5, 0, 11))))) == None

    # antihermitian
    A = Matrix([[0, -2 - I, 0], [2 - I, 0, -I], [0, -I, 0]])
    B = Matrix([[-I, 2 + I, 0], [-2 + I, 0, 2 + I], [0, -2 + I, -I]])
    assert ask(Q.antihermitian(A)) is True
    assert ask(Q.antihermitian(B)) is True
    assert ask(Q.antihermitian(A**2)) is False
    C = (B**3)
    C.simplify()
    assert ask(Q.antihermitian(C)) is True
    _A = Matrix([[0, -2 - I, 0], [z, 0, -I], [0, -I, 0]])
    assert ask(Q.antihermitian(_A)) is None


@_both_exp_pow
def test_algebraic():
    assert ask(Q.algebraic(x)) is None

    assert ask(Q.algebraic(I)) is True
    assert ask(Q.algebraic(2*I)) is True
    assert ask(Q.algebraic(I/3)) is True

    assert ask(Q.algebraic(sqrt(7))) is True
    assert ask(Q.algebraic(2*sqrt(7))) is True
    assert ask(Q.algebraic(sqrt(7)/3)) is True

    assert ask(Q.algebraic(I*sqrt(3))) is True
    assert ask(Q.algebraic(sqrt(1 + I*sqrt(3)))) is True

    assert ask(Q.algebraic(1 + I*sqrt(3)**Rational(17, 31))) is True
    assert ask(Q.algebraic(1 + I*sqrt(3)**(17/pi))) is None

    for f in [exp, sin, tan, asin, atan, cos]:
        assert ask(Q.algebraic(f(7))) is False
        assert ask(Q.algebraic(f(7, evaluate=False))) is False
        assert ask(Q.algebraic(f(0, evaluate=False))) is True
        assert ask(Q.algebraic(f(x)), Q.algebraic(x)) is None
        assert ask(Q.algebraic(f(x)), Q.algebraic(x) & Q.nonzero(x)) is False

    for g in [log, acos]:
        assert ask(Q.algebraic(g(7))) is False
        assert ask(Q.algebraic(g(7, evaluate=False))) is False
        assert ask(Q.algebraic(g(1, evaluate=False))) is True
        assert ask(Q.algebraic(g(x)), Q.algebraic(x)) is None
        assert ask(Q.algebraic(g(x)), Q.algebraic(x) & Q.nonzero(x - 1)) is False

    for h in [cot, acot]:
        assert ask(Q.algebraic(h(7))) is False
        assert ask(Q.algebraic(h(7, evaluate=False))) is False
        assert ask(Q.algebraic(h(x)), Q.algebraic(x)) is False

    assert ask(Q.algebraic(sqrt(sin(7)))) is None
    assert ask(Q.algebraic(sqrt(y + I*sqrt(7)))) is None

    assert ask(Q.algebraic(2.47)) is True

    assert ask(Q.algebraic(x), Q.transcendental(x)) is False
    assert ask(Q.transcendental(x), Q.algebraic(x)) is False

    #https://github.com/sympy/sympy/issues/27445
    assert ask(Q.algebraic(Pow(1, x, evaluate=False)), Q.algebraic(x)) is None
    assert ask(Q.algebraic(Pow(x, y))) is None
    assert ask(Q.algebraic(Pow(1, x, evaluate=False))) is None
    assert ask(Q.algebraic(x**(pi*I))) is None
    assert ask(Q.algebraic(pi**n),Q.integer(n) & Q.positive(n)) is False
    assert ask(Q.algebraic(x**y),Q.algebraic(x) & Q.rational(y)) is True


def test_global():
    """Test ask with global assumptions"""
    assert ask(Q.integer(x)) is None
    global_assumptions.add(Q.integer(x))
    assert ask(Q.integer(x)) is True
    global_assumptions.clear()
    assert ask(Q.integer(x)) is None


def test_custom_context():
    """Test ask with custom assumptions context"""
    assert ask(Q.integer(x)) is None
    local_context = AssumptionsContext()
    local_context.add(Q.integer(x))
    assert ask(Q.integer(x), context=local_context) is True
    assert ask(Q.integer(x)) is None


def test_functions_in_assumptions():
    assert ask(Q.negative(x), Q.real(x) >> Q.positive(x)) is False
    assert ask(Q.negative(x), Equivalent(Q.real(x), Q.positive(x))) is False
    assert ask(Q.negative(x), Xor(Q.real(x), Q.negative(x))) is False


def test_composite_ask():
    assert ask(Q.negative(x) & Q.integer(x),
        assumptions=Q.real(x) >> Q.positive(x)) is False


def test_composite_proposition():
    assert ask(True) is True
    assert ask(False) is False
    assert ask(~Q.negative(x), Q.positive(x)) is True
    assert ask(~Q.real(x), Q.commutative(x)) is None
    assert ask(Q.negative(x) & Q.integer(x), Q.positive(x)) is False
    assert ask(Q.negative(x) & Q.integer(x)) is None
    assert ask(Q.real(x) | Q.integer(x), Q.positive(x)) is True
    assert ask(Q.real(x) | Q.integer(x)) is None
    assert ask(Q.real(x) >> Q.positive(x), Q.negative(x)) is False
    assert ask(Implies(
        Q.real(x), Q.positive(x), evaluate=False), Q.negative(x)) is False
    assert ask(Implies(Q.real(x), Q.positive(x), evaluate=False)) is None
    assert ask(Equivalent(Q.integer(x), Q.even(x)), Q.even(x)) is True
    assert ask(Equivalent(Q.integer(x), Q.even(x))) is None
    assert ask(Equivalent(Q.positive(x), Q.integer(x)), Q.integer(x)) is None
    assert ask(Q.real(x) | Q.integer(x), Q.real(x) | Q.integer(x)) is True

def test_tautology():
    assert ask(Q.real(x) | ~Q.real(x)) is True
    assert ask(Q.real(x) & ~Q.real(x)) is False

def test_composite_assumptions():
    assert ask(Q.real(x), Q.real(x) & Q.real(y)) is True
    assert ask(Q.positive(x), Q.positive(x) | Q.positive(y)) is None
    assert ask(Q.positive(x), Q.real(x) >> Q.positive(y)) is None
    assert ask(Q.real(x), ~(Q.real(x) >> Q.real(y))) is True

def test_key_extensibility():
    """test that you can add keys to the ask system at runtime"""
    # make sure the key is not defined
    raises(AttributeError, lambda: ask(Q.my_key(x)))

    # Old handler system
    class MyAskHandler(AskHandler):
        @staticmethod
        def Symbol(expr, assumptions):
            return True
    try:
        with warns_deprecated_sympy():
            register_handler('my_key', MyAskHandler)
        with warns_deprecated_sympy():
            assert ask(Q.my_key(x)) is True
        with warns_deprecated_sympy():
            assert ask(Q.my_key(x + 1)) is None
    finally:
        # We have to disable the stacklevel testing here because this raises
        # the warning twice from two different places
        with warns_deprecated_sympy():
            remove_handler('my_key', MyAskHandler)
        del Q.my_key
    raises(AttributeError, lambda: ask(Q.my_key(x)))

    # New handler system
    class MyPredicate(Predicate):
        pass
    try:
        Q.my_key = MyPredicate()
        @Q.my_key.register(Symbol)
        def _(expr, assumptions):
            return True
        assert ask(Q.my_key(x)) is True
        assert ask(Q.my_key(x+1)) is None
    finally:
        del Q.my_key
    raises(AttributeError, lambda: ask(Q.my_key(x)))


def test_type_extensibility():
    """test that new types can be added to the ask system at runtime
    """
    from sympy.core import Basic

    class MyType(Basic):
        pass

    @Q.prime.register(MyType)
    def _(expr, assumptions):
        return True

    assert ask(Q.prime(MyType())) is True


def test_single_fact_lookup():
    known_facts = And(Implies(Q.integer, Q.rational),
                      Implies(Q.rational, Q.real),
                      Implies(Q.real, Q.complex))
    known_facts_keys = {Q.integer, Q.rational, Q.real, Q.complex}

    known_facts_cnf = to_cnf(known_facts)
    mapping = single_fact_lookup(known_facts_keys, known_facts_cnf)

    assert mapping[Q.rational] == {Q.real, Q.rational, Q.complex}


def test_generate_known_facts_dict():
    known_facts = And(Implies(Q.integer(x), Q.rational(x)),
                      Implies(Q.rational(x), Q.real(x)),
                      Implies(Q.real(x), Q.complex(x)))
    known_facts_keys = {Q.integer(x), Q.rational(x), Q.real(x), Q.complex(x)}

    assert generate_known_facts_dict(known_facts_keys, known_facts) == \
        {Q.complex: ({Q.complex}, set()),
         Q.integer: ({Q.complex, Q.integer, Q.rational, Q.real}, set()),
         Q.rational: ({Q.complex, Q.rational, Q.real}, set()),
         Q.real: ({Q.complex, Q.real}, set())}


@slow
def test_known_facts_consistent():
    """"Test that ask_generated.py is up-to-date"""
    x = Symbol('x')
    fact = get_known_facts(x)
    # test cnf clauses of fact between unary predicates
    cnf = CNF.to_CNF(fact)
    clauses = set()
    clauses.update(frozenset(Literal(lit.arg.function, lit.is_Not) for lit in sorted(cl, key=str)) for cl in cnf.clauses)
    assert get_all_known_facts() == clauses
    # test dictionary of fact between unary predicates
    keys = [pred(x) for pred in get_known_facts_keys()]
    mapping = generate_known_facts_dict(keys, fact)
    assert get_known_facts_dict() == mapping


def test_Add_queries():
    assert ask(Q.prime(12345678901234567890 + (cos(1)**2 + sin(1)**2))) is True
    assert ask(Q.even(Add(S(2), S(2), evaluate=False))) is True
    assert ask(Q.prime(Add(S(2), S(2), evaluate=False))) is False
    assert ask(Q.integer(Add(S(2), S(2), evaluate=False))) is True


def test_positive_assuming():
    with assuming(Q.positive(x + 1)):
        assert not ask(Q.positive(x))


def test_issue_5421():
    raises(TypeError, lambda: ask(pi/log(x), Q.real))


def test_issue_3906():
    raises(TypeError, lambda: ask(Q.positive))


def test_issue_5833():
    assert ask(Q.positive(log(x)**2), Q.positive(x)) is None
    assert ask(~Q.negative(log(x)**2), Q.positive(x)) is True


def test_issue_6732():
    raises(ValueError, lambda: ask(Q.positive(x), Q.positive(x) & Q.negative(x)))
    raises(ValueError, lambda: ask(Q.negative(x), Q.positive(x) & Q.negative(x)))


def test_issue_7246():
    assert ask(Q.positive(atan(p)), Q.positive(p)) is True
    assert ask(Q.positive(atan(p)), Q.negative(p)) is False
    assert ask(Q.positive(atan(p)), Q.zero(p)) is False
    assert ask(Q.positive(atan(x))) is None

    assert ask(Q.positive(asin(p)), Q.positive(p)) is None
    assert ask(Q.positive(asin(p)), Q.zero(p)) is None
    assert ask(Q.positive(asin(Rational(1, 7)))) is True
    assert ask(Q.positive(asin(x)), Q.positive(x) & Q.nonpositive(x - 1)) is True
    assert ask(Q.positive(asin(x)), Q.negative(x) & Q.nonnegative(x + 1)) is False

    assert ask(Q.positive(acos(p)), Q.positive(p)) is None
    assert ask(Q.positive(acos(Rational(1, 7)))) is True
    assert ask(Q.positive(acos(x)), Q.nonnegative(x + 1) & Q.nonpositive(x - 1)) is True
    assert ask(Q.positive(acos(x)), Q.nonnegative(x - 1)) is None

    assert ask(Q.positive(acot(x)), Q.positive(x)) is True
    assert ask(Q.positive(acot(x)), Q.real(x)) is True
    assert ask(Q.positive(acot(x)), Q.imaginary(x)) is False
    assert ask(Q.positive(acot(x))) is None


@XFAIL
def test_issue_7246_failing():
    #Move this test to test_issue_7246 once
    #the new assumptions module is improved.
    assert ask(Q.positive(acos(x)), Q.zero(x)) is True


def test_check_old_assumption():
    x = symbols('x', real=True)
    assert ask(Q.real(x)) is True
    assert ask(Q.imaginary(x)) is False
    assert ask(Q.complex(x)) is True

    x = symbols('x', imaginary=True)
    assert ask(Q.real(x)) is False
    assert ask(Q.imaginary(x)) is True
    assert ask(Q.complex(x)) is True

    x = symbols('x', complex=True)
    assert ask(Q.real(x)) is None
    assert ask(Q.complex(x)) is True

    x = symbols('x', positive=True)
    assert ask(Q.positive(x)) is True
    assert ask(Q.negative(x)) is False
    assert ask(Q.real(x)) is True

    x = symbols('x', commutative=False)
    assert ask(Q.commutative(x)) is False

    x = symbols('x', negative=True)
    assert ask(Q.positive(x)) is False
    assert ask(Q.negative(x)) is True

    x = symbols('x', nonnegative=True)
    assert ask(Q.negative(x)) is False
    assert ask(Q.positive(x)) is None
    assert ask(Q.zero(x)) is None

    x = symbols('x', finite=True)
    assert ask(Q.finite(x)) is True

    x = symbols('x', prime=True)
    assert ask(Q.prime(x)) is True
    assert ask(Q.composite(x)) is False

    x = symbols('x', composite=True)
    assert ask(Q.prime(x)) is False
    assert ask(Q.composite(x)) is True

    x = symbols('x', even=True)
    assert ask(Q.even(x)) is True
    assert ask(Q.odd(x)) is False

    x = symbols('x', odd=True)
    assert ask(Q.even(x)) is False
    assert ask(Q.odd(x)) is True

    x = symbols('x', nonzero=True)
    assert ask(Q.nonzero(x)) is True
    assert ask(Q.zero(x)) is False

    x = symbols('x', zero=True)
    assert ask(Q.zero(x)) is True

    x = symbols('x', integer=True)
    assert ask(Q.integer(x)) is True

    x = symbols('x', rational=True)
    assert ask(Q.rational(x)) is True
    assert ask(Q.irrational(x)) is False

    x = symbols('x', irrational=True)
    assert ask(Q.irrational(x)) is True
    assert ask(Q.rational(x)) is False


def test_issue_9636():
    assert ask(Q.integer(1.0)) is None
    assert ask(Q.prime(3.0)) is None
    assert ask(Q.composite(4.0)) is None
    assert ask(Q.even(2.0)) is None
    assert ask(Q.odd(3.0)) is None


def test_autosimp_used_to_fail():
    # See issue #9807
    assert ask(Q.imaginary(0**I)) is None
    assert ask(Q.imaginary(0**(-I))) is None
    assert ask(Q.real(0**I)) is None
    assert ask(Q.real(0**(-I))) is None


def test_custom_AskHandler():
    from sympy.logic.boolalg import conjuncts

    # Old handler system
    class MersenneHandler(AskHandler):
        @staticmethod
        def Integer(expr, assumptions):
            if ask(Q.integer(log(expr + 1, 2))):
                return True
        @staticmethod
        def Symbol(expr, assumptions):
            if expr in conjuncts(assumptions):
                return True
    try:
        with warns_deprecated_sympy():
            register_handler('mersenne', MersenneHandler)
        n = Symbol('n', integer=True)
        with warns_deprecated_sympy():
            assert ask(Q.mersenne(7))
        with warns_deprecated_sympy():
            assert ask(Q.mersenne(n), Q.mersenne(n))
    finally:
        del Q.mersenne

    # New handler system
    class MersennePredicate(Predicate):
        pass
    try:
        Q.mersenne = MersennePredicate()
        @Q.mersenne.register(Integer)
        def _(expr, assumptions):
            if ask(Q.integer(log(expr + 1, 2))):
                return True
        @Q.mersenne.register(Symbol)
        def _(expr, assumptions):
            if expr in conjuncts(assumptions):
                return True
        assert ask(Q.mersenne(7))
        assert ask(Q.mersenne(n), Q.mersenne(n))
    finally:
        del Q.mersenne


def test_polyadic_predicate():

    class SexyPredicate(Predicate):
        pass
    try:
        Q.sexyprime = SexyPredicate()

        @Q.sexyprime.register(Integer, Integer)
        def _(int1, int2, assumptions):
            args = sorted([int1, int2])
            if not all(ask(Q.prime(a), assumptions) for a in args):
                return False
            return args[1] - args[0] == 6

        @Q.sexyprime.register(Integer, Integer, Integer)
        def _(int1, int2, int3, assumptions):
            args = sorted([int1, int2, int3])
            if not all(ask(Q.prime(a), assumptions) for a in args):
                return False
            return args[2] - args[1] == 6 and args[1] - args[0] == 6

        assert ask(Q.sexyprime(5, 11))
        assert ask(Q.sexyprime(7, 13, 19))
    finally:
        del Q.sexyprime


def test_Predicate_handler_is_unique():

    # Undefined predicate does not have a handler
    assert Predicate('mypredicate').handler is None

    # Handler of defined predicate is unique to the class
    class MyPredicate(Predicate):
        pass
    mp1 = MyPredicate(Str('mp1'))
    mp2 = MyPredicate(Str('mp2'))
    assert mp1.handler is mp2.handler


def test_relational():
    assert ask(Q.eq(x, 0), Q.zero(x))
    assert not ask(Q.eq(x, 0), Q.nonzero(x))
    assert not ask(Q.ne(x, 0), Q.zero(x))
    assert ask(Q.ne(x, 0), Q.nonzero(x))


def test_issue_25221():
    assert ask(Q.transcendental(x), Q.algebraic(x) | Q.positive(y,y)) is None
    assert ask(Q.transcendental(x), Q.algebraic(x) | (0 > y)) is None
    assert ask(Q.transcendental(x), Q.algebraic(x) | Q.gt(0,y)) is None


def test_issue_27440():
    nan = S.NaN
    assert ask(Q.negative(nan)) is None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\CalTableFlatBuffers\KeyValue.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: CalTableFlatBuffers

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class KeyValue:
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):  # noqa: N802
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = KeyValue()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsKeyValue(cls, buf, offset=0):  # noqa: N802
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # KeyValue
    def Init(self, buf, pos):  # noqa: N802
        self._tab = flatbuffers.table.Table(buf, pos)

    # KeyValue
    def Key(self):  # noqa: N802
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # KeyValue
    def Value(self):  # noqa: N802
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None


def Start(builder):  # noqa: N802
    builder.StartObject(2)


def KeyValueStart(builder):  # noqa: N802
    """This method is deprecated. Please switch to Start."""
    return Start(builder)


def AddKey(builder, key):  # noqa: N802
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(key), 0)


def KeyValueAddKey(builder, key):  # noqa: N802
    """This method is deprecated. Please switch to AddKey."""
    return AddKey(builder, key)


def AddValue(builder, value):  # noqa: N802
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(value), 0)


def KeyValueAddValue(builder, value):  # noqa: N802
    """This method is deprecated. Please switch to AddValue."""
    return AddValue(builder, value)


def End(builder):  # noqa: N802
    return builder.EndObject()


def KeyValueEnd(builder):  # noqa: N802
    """This method is deprecated. Please switch to End."""
    return End(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\direct_q8.py ===
from ..quant_utils import TENSOR_NAME_QUANT_SUFFIX, QuantizedValue, QuantizedValueType
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


# For operators that support 8bits operations directly, and output could
# reuse input[0]'s type, zeropoint, scale; For example,Transpose, Reshape, etc.
class Direct8BitOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node

        if not self.quantizer.force_quantize_no_input_check:
            # Keep backward compatibility
            # Quantize when input[0] is quantized already. Otherwise keep it.
            quantized_input_value = self.quantizer.find_quantized_value(node.input[0])
            if quantized_input_value is None:
                self.quantizer.new_nodes += [node]
                return

            quantized_output_value = QuantizedValue(
                node.output[0],
                node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
                quantized_input_value.scale_name,
                quantized_input_value.zp_name,
                quantized_input_value.value_type,
            )
            self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

            node.input[0] = quantized_input_value.q_name
            node.output[0] = quantized_output_value.q_name
            self.quantizer.new_nodes += [node]

        else:
            # Force quantize those ops if possible, use exclude node list if this is not you want
            if not self.quantizer.is_valid_quantize_weight(node.input[0]):
                super().quantize()
                return

            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])
            if quantized_input_names is None:
                return super().quantize()

            # Create an entry for output quantized value
            quantized_output_value = QuantizedValue(
                node.output[0],
                node.output[0] + TENSOR_NAME_QUANT_SUFFIX,
                scale_names[0],
                zero_point_names[0],
                QuantizedValueType.Input,
            )
            self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

            node.input[0] = quantized_input_names[0]
            node.output[0] = quantized_output_value.q_name
            nodes.append(node)

            self.quantizer.new_nodes += nodes


class QDQDirect8BitOp(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        if self.quantizer.force_quantize_no_input_check:
            self.quantizer.quantize_activation_tensor(self.node.input[0])
            if not self.disable_qdq_for_node_output:
                self.quantizer.quantize_output_same_as_input(self.node.output[0], self.node.input[0], self.node.name)
        elif self.quantizer.is_tensor_quantized(self.node.input[0]) and not self.disable_qdq_for_node_output:
            self.quantizer.quantize_output_same_as_input(self.node.output[0], self.node.input[0], self.node.name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\KernelTypeStrResolver.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class KernelTypeStrResolver(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = KernelTypeStrResolver()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsKernelTypeStrResolver(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def KernelTypeStrResolverBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # KernelTypeStrResolver
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # KernelTypeStrResolver
    def OpKernelTypeStrArgs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.OpIdKernelTypeStrArgsEntry import OpIdKernelTypeStrArgsEntry
            obj = OpIdKernelTypeStrArgsEntry()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # KernelTypeStrResolver
    def OpKernelTypeStrArgsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # KernelTypeStrResolver
    def OpKernelTypeStrArgsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def KernelTypeStrResolverStart(builder):
    builder.StartObject(1)

def Start(builder):
    KernelTypeStrResolverStart(builder)

def KernelTypeStrResolverAddOpKernelTypeStrArgs(builder, opKernelTypeStrArgs):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(opKernelTypeStrArgs), 0)

def AddOpKernelTypeStrArgs(builder, opKernelTypeStrArgs):
    KernelTypeStrResolverAddOpKernelTypeStrArgs(builder, opKernelTypeStrArgs)

def KernelTypeStrResolverStartOpKernelTypeStrArgsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOpKernelTypeStrArgsVector(builder, numElems: int) -> int:
    return KernelTypeStrResolverStartOpKernelTypeStrArgsVector(builder, numElems)

def KernelTypeStrResolverEnd(builder):
    return builder.EndObject()

def End(builder):
    return KernelTypeStrResolverEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\Shape.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Shape(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Shape()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsShape(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ShapeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Shape
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Shape
    def Dim(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Dimension import Dimension
            obj = Dimension()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Shape
    def DimLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Shape
    def DimIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def ShapeStart(builder):
    builder.StartObject(1)

def Start(builder):
    ShapeStart(builder)

def ShapeAddDim(builder, dim):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(dim), 0)

def AddDim(builder, dim):
    ShapeAddDim(builder, dim)

def ShapeStartDimVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartDimVector(builder, numElems: int) -> int:
    return ShapeStartDimVector(builder, numElems)

def ShapeEnd(builder):
    return builder.EndObject()

def End(builder):
    return ShapeEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\text.py ===
# Copyright (c) 2010-2024 openpyxl
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Alias,
    Sequence,
)


from openpyxl.drawing.text import (
    RichTextProperties,
    ListStyle,
    Paragraph,
)

from .data_source import StrRef


class RichText(Serialisable):

    """
    From the specification: 21.2.2.216

    This element specifies text formatting. The lstStyle element is not supported.
    """

    tagname = "rich"

    bodyPr = Typed(expected_type=RichTextProperties)
    properties = Alias("bodyPr")
    lstStyle = Typed(expected_type=ListStyle, allow_none=True)
    p = Sequence(expected_type=Paragraph)
    paragraphs = Alias('p')

    __elements__ = ("bodyPr", "lstStyle", "p")

    def __init__(self,
                 bodyPr=None,
                 lstStyle=None,
                 p=None,
                ):
        if bodyPr is None:
            bodyPr = RichTextProperties()
        self.bodyPr = bodyPr
        self.lstStyle = lstStyle
        if p is None:
            p = [Paragraph()]
        self.p = p


class Text(Serialisable):

    """
    The value can be either a cell reference or a text element
    If both are present then the reference will be used.
    """

    tagname = "tx"

    strRef = Typed(expected_type=StrRef, allow_none=True)
    rich = Typed(expected_type=RichText, allow_none=True)

    __elements__ = ("strRef", "rich")

    def __init__(self,
                 strRef=None,
                 rich=None
                 ):
        self.strRef = strRef
        if rich is None:
            rich = RichText()
        self.rich = rich


    def to_tree(self, tagname=None, idx=None, namespace=None):
        if self.strRef and self.rich:
            self.rich = None # can only have one
        return super().to_tree(tagname, idx, namespace)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\smart_tag.py ===
#Autogenerated schema
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Bool,
    Integer,
    String,
    Sequence,
)


class CellSmartTagPr(Serialisable):

    tagname = "cellSmartTagPr"

    key = String()
    val = String()

    def __init__(self,
                 key=None,
                 val=None,
                ):
        self.key = key
        self.val = val


class CellSmartTag(Serialisable):

    tagname = "cellSmartTag"

    cellSmartTagPr = Sequence(expected_type=CellSmartTagPr)
    type = Integer()
    deleted = Bool(allow_none=True)
    xmlBased = Bool(allow_none=True)

    __elements__ = ('cellSmartTagPr',)

    def __init__(self,
                 cellSmartTagPr=(),
                 type=None,
                 deleted=False,
                 xmlBased=False,
                ):
        self.cellSmartTagPr = cellSmartTagPr
        self.type = type
        self.deleted = deleted
        self.xmlBased = xmlBased


class CellSmartTags(Serialisable):

    tagname = "cellSmartTags"

    cellSmartTag = Sequence(expected_type=CellSmartTag)
    r = String()

    __elements__ = ('cellSmartTag',)

    def __init__(self,
                 cellSmartTag=(),
                 r=None,
                ):
        self.cellSmartTag = cellSmartTag
        self.r = r


class SmartTags(Serialisable):

    tagname = "smartTags"

    cellSmartTags = Sequence(expected_type=CellSmartTags)

    __elements__ = ('cellSmartTags',)

    def __init__(self,
                 cellSmartTags=(),
                ):
        self.cellSmartTags = cellSmartTags


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\models\format_control.py ===
from typing import FrozenSet, Optional, Set

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.exceptions import CommandError


class FormatControl:
    """Helper for managing formats from which a package can be installed."""

    __slots__ = ["no_binary", "only_binary"]

    def __init__(
        self,
        no_binary: Optional[Set[str]] = None,
        only_binary: Optional[Set[str]] = None,
    ) -> None:
        if no_binary is None:
            no_binary = set()
        if only_binary is None:
            only_binary = set()

        self.no_binary = no_binary
        self.only_binary = only_binary

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.__slots__ != other.__slots__:
            return False

        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.no_binary}, {self.only_binary})"

    @staticmethod
    def handle_mutual_excludes(value: str, target: Set[str], other: Set[str]) -> None:
        if value.startswith("-"):
            raise CommandError(
                "--no-binary / --only-binary option requires 1 argument."
            )
        new = value.split(",")
        while ":all:" in new:
            other.clear()
            target.clear()
            target.add(":all:")
            del new[: new.index(":all:") + 1]
            # Without a none, we want to discard everything as :all: covers it
            if ":none:" not in new:
                return
        for name in new:
            if name == ":none:":
                target.clear()
                continue
            name = canonicalize_name(name)
            other.discard(name)
            target.add(name)

    def get_allowed_formats(self, canonical_name: str) -> FrozenSet[str]:
        result = {"binary", "source"}
        if canonical_name in self.only_binary:
            result.discard("source")
        elif canonical_name in self.no_binary:
            result.discard("binary")
        elif ":all:" in self.only_binary:
            result.discard("source")
        elif ":all:" in self.no_binary:
            result.discard("binary")
        return frozenset(result)

    def disallow_binaries(self) -> None:
        self.handle_mutual_excludes(
            ":all:",
            self.no_binary,
            self.only_binary,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\compat.py ===
"""
requests.compat
~~~~~~~~~~~~~~~

This module previously handled import compatibility issues
between Python 2 and Python 3. It remains for backwards
compatibility until the next major version.
"""

import sys

# -------------------
# Character Detection
# -------------------


def _resolve_char_detection():
    """Find supported character detection libraries."""
    chardet = None
    return chardet


chardet = _resolve_char_detection()

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = _ver[0] == 2

#: Python 3.x?
is_py3 = _ver[0] == 3

# Note: We've patched out simplejson support in pip because it prevents
#       upgrading simplejson on Windows.
import json
from json import JSONDecodeError

# Keep OrderedDict for backwards compatibility.
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from http import cookiejar as cookielib
from http.cookies import Morsel
from io import StringIO

# --------------
# Legacy Imports
# --------------
from urllib.parse import (
    quote,
    quote_plus,
    unquote,
    unquote_plus,
    urldefrag,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunparse,
)
from urllib.request import (
    getproxies,
    getproxies_environment,
    parse_http_list,
    proxy_bypass,
    proxy_bypass_environment,
)

builtin_str = str
str = str
bytes = bytes
basestring = (str, bytes)
numeric_types = (int, float)
integer_types = (int,)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\interval_membership.py ===
from sympy.core.logic import fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_xor


class intervalMembership:
    """Represents a boolean expression returned by the comparison of
    the interval object.

    Parameters
    ==========

    (a, b) : (bool, bool)
        The first value determines the comparison as follows:
        - True: If the comparison is True throughout the intervals.
        - False: If the comparison is False throughout the intervals.
        - None: If the comparison is True for some part of the intervals.

        The second value is determined as follows:
        - True: If both the intervals in comparison are valid.
        - False: If at least one of the intervals is False, else
        - None
    """
    def __init__(self, a, b):
        self._wrapped = (a, b)

    def __getitem__(self, i):
        try:
            return self._wrapped[i]
        except IndexError:
            raise IndexError(
                "{} must be a valid indexing for the 2-tuple."
                .format(i))

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(self._wrapped)

    def __str__(self):
        return "intervalMembership({}, {})".format(*self)
    __repr__ = __str__

    def __and__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_and([a1, a2]), fuzzy_and([b1, b2]))

    def __or__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_or([a1, a2]), fuzzy_and([b1, b2]))

    def __invert__(self):
        a, b = self
        return intervalMembership(fuzzy_not(a), b)

    def __xor__(self, other):
        if not isinstance(other, intervalMembership):
            raise ValueError(
                "The comparison is not supported for {}.".format(other))

        a1, b1 = self
        a2, b2 = other
        return intervalMembership(fuzzy_xor([a1, a2]), fuzzy_and([b1, b2]))

    def __eq__(self, other):
        return self._wrapped == other

    def __ne__(self, other):
        return self._wrapped != other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\libmp\__init__.py ===
from .libmpf import (prec_to_dps, dps_to_prec, repr_dps,
  round_down, round_up, round_floor, round_ceiling, round_nearest,
  to_pickable, from_pickable, ComplexResult,
  fzero, fnzero, fone, fnone, ftwo, ften, fhalf, fnan, finf, fninf,
  math_float_inf, round_int, normalize, normalize1,
  from_man_exp, from_int, to_man_exp, to_int, mpf_ceil, mpf_floor,
  mpf_nint, mpf_frac,
  from_float, from_npfloat, from_Decimal, to_float, from_rational, to_rational, to_fixed,
  mpf_rand, mpf_eq, mpf_hash, mpf_cmp, mpf_lt, mpf_le, mpf_gt, mpf_ge,
  mpf_pos, mpf_neg, mpf_abs, mpf_sign, mpf_add, mpf_sub, mpf_sum,
  mpf_mul, mpf_mul_int, mpf_shift, mpf_frexp,
  mpf_div, mpf_rdiv_int, mpf_mod, mpf_pow_int,
  mpf_perturb,
  to_digits_exp, to_str, str_to_man_exp, from_str, from_bstr, to_bstr,
  mpf_sqrt, mpf_hypot)

from .libmpc import (mpc_one, mpc_zero, mpc_two, mpc_half,
  mpc_is_inf, mpc_is_infnan, mpc_to_str, mpc_to_complex, mpc_hash,
  mpc_conjugate, mpc_is_nonzero, mpc_add, mpc_add_mpf,
  mpc_sub, mpc_sub_mpf, mpc_pos, mpc_neg, mpc_shift, mpc_abs,
  mpc_arg, mpc_floor, mpc_ceil,  mpc_nint, mpc_frac, mpc_mul, mpc_square,
  mpc_mul_mpf, mpc_mul_imag_mpf, mpc_mul_int,
  mpc_div, mpc_div_mpf, mpc_reciprocal, mpc_mpf_div,
  complex_int_pow, mpc_pow, mpc_pow_mpf, mpc_pow_int,
  mpc_sqrt, mpc_nthroot, mpc_cbrt, mpc_exp, mpc_log, mpc_cos, mpc_sin,
  mpc_tan, mpc_cos_pi, mpc_sin_pi, mpc_cosh, mpc_sinh, mpc_tanh,
  mpc_atan, mpc_acos, mpc_asin, mpc_asinh, mpc_acosh, mpc_atanh,
  mpc_fibonacci, mpf_expj, mpf_expjpi, mpc_expj, mpc_expjpi,
  mpc_cos_sin, mpc_cos_sin_pi)

from .libelefun import (ln2_fixed, mpf_ln2, ln10_fixed, mpf_ln10,
  pi_fixed, mpf_pi, e_fixed, mpf_e, phi_fixed, mpf_phi,
  degree_fixed, mpf_degree,
  mpf_pow, mpf_nthroot, mpf_cbrt, log_int_fixed, agm_fixed,
  mpf_log, mpf_log_hypot, mpf_exp, mpf_cos_sin, mpf_cos, mpf_sin, mpf_tan,
  mpf_cos_sin_pi, mpf_cos_pi, mpf_sin_pi, mpf_cosh_sinh,
  mpf_cosh, mpf_sinh, mpf_tanh, mpf_atan, mpf_atan2, mpf_asin,
  mpf_acos, mpf_asinh, mpf_acosh, mpf_atanh, mpf_fibonacci)

from .libhyper import (NoConvergence, make_hyp_summator,
  mpf_erf, mpf_erfc, mpf_ei, mpc_ei, mpf_e1, mpc_e1, mpf_expint,
  mpf_ci_si, mpf_ci, mpf_si, mpc_ci, mpc_si, mpf_besseljn,
  mpc_besseljn, mpf_agm, mpf_agm1, mpc_agm, mpc_agm1,
  mpf_ellipk, mpc_ellipk, mpf_ellipe, mpc_ellipe)

from .gammazeta import (catalan_fixed, mpf_catalan,
  khinchin_fixed, mpf_khinchin, glaisher_fixed, mpf_glaisher,
  apery_fixed, mpf_apery, euler_fixed, mpf_euler, mertens_fixed,
  mpf_mertens, twinprime_fixed, mpf_twinprime,
  mpf_bernoulli, bernfrac, mpf_gamma_int,
  mpf_factorial, mpc_factorial, mpf_gamma, mpc_gamma,
  mpf_loggamma, mpc_loggamma, mpf_rgamma, mpc_rgamma,
  mpf_harmonic, mpc_harmonic, mpf_psi0, mpc_psi0,
  mpf_psi, mpc_psi, mpf_zeta_int, mpf_zeta, mpc_zeta,
  mpf_altzeta, mpc_altzeta, mpf_zetasum, mpc_zetasum)

from .libmpi import (mpi_str,
  mpi_from_str, mpi_to_str,
  mpi_eq, mpi_ne,
  mpi_lt, mpi_le, mpi_gt, mpi_ge,
  mpi_add, mpi_sub, mpi_delta, mpi_mid,
  mpi_pos, mpi_neg, mpi_abs, mpi_mul, mpi_div, mpi_exp,
  mpi_log, mpi_sqrt, mpi_pow_int, mpi_pow, mpi_cos_sin,
  mpi_cos, mpi_sin, mpi_tan, mpi_cot,
  mpi_atan, mpi_atan2,
  mpci_pos, mpci_neg, mpci_add, mpci_sub, mpci_mul, mpci_div, mpci_pow,
  mpci_abs, mpci_pow, mpci_exp, mpci_log, mpci_cos, mpci_sin,
  mpi_gamma, mpci_gamma, mpi_loggamma, mpci_loggamma,
  mpi_rgamma, mpci_rgamma, mpi_factorial, mpci_factorial)

from .libintmath import (trailing, bitcount, numeral, bin_to_radix,
  isqrt, isqrt_small, isqrt_fast, sqrt_fixed, sqrtrem, ifib, ifac,
  list_primes, isprime, moebius, gcd, eulernum, stirling1, stirling2)

from .backend import (gmpy, sage, BACKEND, STRICT, MPZ, MPZ_TYPE,
  MPZ_ZERO, MPZ_ONE, MPZ_TWO, MPZ_THREE, MPZ_FIVE, int_types,
  HASH_MODULUS, HASH_BITS)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\compat\compressors.py ===
"""
Patched ``BZ2File`` and ``LZMAFile`` to handle pickle protocol 5.
"""

from __future__ import annotations

from pickle import PickleBuffer

from pandas.compat._constants import PY310

try:
    import bz2

    has_bz2 = True
except ImportError:
    has_bz2 = False

try:
    import lzma

    has_lzma = True
except ImportError:
    has_lzma = False


def flatten_buffer(
    b: bytes | bytearray | memoryview | PickleBuffer,
) -> bytes | bytearray | memoryview:
    """
    Return some 1-D `uint8` typed buffer.

    Coerces anything that does not match that description to one that does
    without copying if possible (otherwise will copy).
    """

    if isinstance(b, (bytes, bytearray)):
        return b

    if not isinstance(b, PickleBuffer):
        b = PickleBuffer(b)

    try:
        # coerce to 1-D `uint8` C-contiguous `memoryview` zero-copy
        return b.raw()
    except BufferError:
        # perform in-memory copy if buffer is not contiguous
        return memoryview(b).tobytes("A")


if has_bz2:

    class BZ2File(bz2.BZ2File):
        if not PY310:

            def write(self, b) -> int:
                # Workaround issue where `bz2.BZ2File` expects `len`
                # to return the number of bytes in `b` by converting
                # `b` into something that meets that constraint with
                # minimal copying.
                #
                # Note: This is fixed in Python 3.10.
                return super().write(flatten_buffer(b))


if has_lzma:

    class LZMAFile(lzma.LZMAFile):
        if not PY310:

            def write(self, b) -> int:
                # Workaround issue where `lzma.LZMAFile` expects `len`
                # to return the number of bytes in `b` by converting
                # `b` into something that meets that constraint with
                # minimal copying.
                #
                # Note: This is fixed in Python 3.10.
                return super().write(flatten_buffer(b))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\actions\wheel_input.py ===
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
from typing import Union

from selenium.webdriver.remote.webelement import WebElement

from . import interaction
from .input_device import InputDevice


class ScrollOrigin:
    def __init__(self, origin: Union[str, WebElement], x_offset: int, y_offset: int) -> None:
        self._origin = origin
        self._x_offset = x_offset
        self._y_offset = y_offset

    @classmethod
    def from_element(cls, element: WebElement, x_offset: int = 0, y_offset: int = 0):
        return cls(element, x_offset, y_offset)

    @classmethod
    def from_viewport(cls, x_offset: int = 0, y_offset: int = 0):
        return cls("viewport", x_offset, y_offset)

    @property
    def origin(self) -> Union[str, WebElement]:
        return self._origin

    @property
    def x_offset(self) -> int:
        return self._x_offset

    @property
    def y_offset(self) -> int:
        return self._y_offset


class WheelInput(InputDevice):
    def __init__(self, name) -> None:
        super().__init__(name=name)
        self.name = name
        self.type = interaction.WHEEL

    def encode(self) -> dict:
        return {"type": self.type, "id": self.name, "actions": self.actions}

    def create_scroll(self, x: int, y: int, delta_x: int, delta_y: int, duration: int, origin) -> None:
        if isinstance(origin, WebElement):
            origin = {"element-6066-11e4-a52e-4f735466cecf": origin.id}
        self.add_action(
            {
                "type": "scroll",
                "x": x,
                "y": y,
                "deltaX": delta_x,
                "deltaY": delta_y,
                "duration": duration,
                "origin": origin,
            }
        )

    def create_pause(self, pause_duration: Union[int, float] = 0) -> None:
        self.add_action({"type": "pause", "duration": int(pause_duration * 1000)})

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\abstract_event_listener.py ===
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


class AbstractEventListener:
    """Event listener must subclass and implement this fully or partially."""

    def before_navigate_to(self, url: str, driver) -> None:
        pass

    def after_navigate_to(self, url: str, driver) -> None:
        pass

    def before_navigate_back(self, driver) -> None:
        pass

    def after_navigate_back(self, driver) -> None:
        pass

    def before_navigate_forward(self, driver) -> None:
        pass

    def after_navigate_forward(self, driver) -> None:
        pass

    def before_find(self, by, value, driver) -> None:
        pass

    def after_find(self, by, value, driver) -> None:
        pass

    def before_click(self, element, driver) -> None:
        pass

    def after_click(self, element, driver) -> None:
        pass

    def before_change_value_of(self, element, driver) -> None:
        pass

    def after_change_value_of(self, element, driver) -> None:
        pass

    def before_execute_script(self, script, driver) -> None:
        pass

    def after_execute_script(self, script, driver) -> None:
        pass

    def before_close(self, driver) -> None:
        pass

    def after_close(self, driver) -> None:
        pass

    def before_quit(self, driver) -> None:
        pass

    def after_quit(self, driver) -> None:
        pass

    def on_exception(self, exception, driver) -> None:
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\webkitgtk\options.py ===
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

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions


class Options(ArgOptions):
    KEY = "webkitgtk:browserOptions"

    def __init__(self) -> None:
        super().__init__()
        self._binary_location = ""
        self._overlay_scrollbars_enabled = True

    @property
    def binary_location(self) -> str:
        """:Returns: The location of the browser binary otherwise an empty
        string."""
        return self._binary_location

    @binary_location.setter
    def binary_location(self, value: str) -> None:
        """Allows you to set the browser binary to launch.

        :Args:
         - value : path to the browser binary
        """
        self._binary_location = value

    @property
    def overlay_scrollbars_enabled(self):
        """:Returns: Whether overlay scrollbars should be enabled."""
        return self._overlay_scrollbars_enabled

    @overlay_scrollbars_enabled.setter
    def overlay_scrollbars_enabled(self, value) -> None:
        """Allows you to enable or disable overlay scrollbars.

        :Args:
         - value : True or False
        """
        self._overlay_scrollbars_enabled = value

    def to_capabilities(self):
        """Creates a capabilities with all the options that have been set and
        returns a dictionary with everything."""
        caps = self._caps

        browser_options = {}
        if self.binary_location:
            browser_options["binary"] = self.binary_location
        if self.arguments:
            browser_options["args"] = self.arguments
        browser_options["useOverlayScrollbars"] = self.overlay_scrollbars_enabled

        caps[Options.KEY] = browser_options

        return caps

    @property
    def default_capabilities(self):
        return DesiredCapabilities.WEBKITGTK.copy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\__init__.py ===
# dialects/oracle/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors
from types import ModuleType

from . import base  # noqa
from . import cx_oracle  # noqa
from . import oracledb  # noqa
from .base import BFILE
from .base import BINARY_DOUBLE
from .base import BINARY_FLOAT
from .base import BLOB
from .base import CHAR
from .base import CLOB
from .base import DATE
from .base import DOUBLE_PRECISION
from .base import FLOAT
from .base import INTERVAL
from .base import LONG
from .base import NCHAR
from .base import NCLOB
from .base import NUMBER
from .base import NVARCHAR
from .base import NVARCHAR2
from .base import RAW
from .base import REAL
from .base import ROWID
from .base import TIMESTAMP
from .base import VARCHAR
from .base import VARCHAR2
from .base import VECTOR
from .base import VectorIndexConfig
from .base import VectorIndexType
from .vector import VectorDistanceType
from .vector import VectorStorageFormat

# Alias oracledb also as oracledb_async
oracledb_async = type(
    "oracledb_async", (ModuleType,), {"dialect": oracledb.dialect_async}
)

base.dialect = dialect = cx_oracle.dialect

__all__ = (
    "VARCHAR",
    "NVARCHAR",
    "CHAR",
    "NCHAR",
    "DATE",
    "NUMBER",
    "BLOB",
    "BFILE",
    "CLOB",
    "NCLOB",
    "TIMESTAMP",
    "RAW",
    "FLOAT",
    "DOUBLE_PRECISION",
    "BINARY_DOUBLE",
    "BINARY_FLOAT",
    "LONG",
    "dialect",
    "INTERVAL",
    "VARCHAR2",
    "NVARCHAR2",
    "ROWID",
    "REAL",
    "VECTOR",
    "VectorDistanceType",
    "VectorIndexType",
    "VectorIndexConfig",
    "VectorStorageFormat",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\_compilation\availability.py ===
import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError

def has_fortran():
    if not hasattr(has_fortran, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings(
                [('main.f90', (
                    'program foo\n'
                    'print *, "hello world"\n'
                    'end program'
                ))], clean=True
            )
        except CompilerNotFoundError:
            has_fortran.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                has_fortran.result = False
            else:
                has_fortran.result = True
    return has_fortran.result


def has_c():
    if not hasattr(has_c, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings(
                [('main.c', (
                    '#include <stdio.h>\n'
                    'int main(){\n'
                    'printf("hello world\\n");\n'
                    'return 0;\n'
                    '}'
                ))], clean=True
            )
        except CompilerNotFoundError:
            has_c.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                has_c.result = False
            else:
                has_c.result = True
    return has_c.result


def has_cxx():
    if not hasattr(has_cxx, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings(
                [('main.cxx', (
                    '#include <iostream>\n'
                    'int main(){\n'
                    'std::cout << "hello world" << std::endl;\n'
                    '}'
                ))], clean=True
            )
        except CompilerNotFoundError:
            has_cxx.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                has_cxx.result = False
            else:
                has_cxx.result = True
    return has_cxx.result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\driver.py ===
from webdriver_manager.core.logger import log
from webdriver_manager.core.config import gh_token
from webdriver_manager.core.os_manager import OperationSystemManager


class Driver(object):
    def __init__(
            self,
            name,
            driver_version_to_download,
            url,
            latest_release_url,
            http_client,
            os_system_manager):
        self._name = name
        self._url = url
        self._latest_release_url = latest_release_url
        self._http_client = http_client
        self._browser_version = None
        self._driver_version_to_download = driver_version_to_download
        self._os_system_manager = os_system_manager
        if not self._os_system_manager:
            self._os_system_manager = OperationSystemManager()

    @property
    def auth_header(self):
        token = gh_token()
        if token:
            log("GH_TOKEN will be used to perform requests")
            return {"Authorization": f"token {token}"}
        return None

    def get_name(self):
        return self._name

    def get_driver_download_url(self, os_type):
        return f"{self._url}/{self.get_driver_version_to_download()}/{self._name}_{os_type}.zip"

    def get_driver_version_to_download(self):
        """
        Downloads version from parameter if version not None or "latest".
        Downloads latest, if version is "latest" or browser could not been determined.
        Downloads determined browser version driver in all other ways as a bonus fallback for lazy users.
        """
        if self._driver_version_to_download:
            return self._driver_version_to_download

        return self.get_latest_release_version()

    def get_latest_release_version(self):
        # type: () -> str
        raise NotImplementedError("Please implement this method")

    def get_browser_version_from_os(self):
        """
        Use-cases:
        - for key in metadata;
        - for printing nice logs;
        - for fallback if version was not set at all.
        Note: the fallback may have collisions in user cases when previous browser was not uninstalled properly.
        """
        if self._browser_version is None:
            self._browser_version = self._os_system_manager.get_browser_version_from_os(self.get_browser_type())
        return self._browser_version

    def get_browser_type(self):
        raise NotImplementedError("Please implement this method")

    def get_binary_name(self, os_type):
        driver_name = self.get_name()
        driver_binary_name = (
            "msedgedriver" if driver_name == "edgedriver" else driver_name
        )
        driver_binary_name = (
            f"{driver_binary_name}.exe" if "win" in os_type else driver_binary_name
        )
        return driver_binary_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\tests\test_arit.py ===
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
    oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet

from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin

from itertools import product

a, c, x, y, z = symbols('a,c,x,y,z')
b = Symbol("b", positive=True)


def same_and_same_prec(a, b):
    # stricter matching for Floats
    return a == b and a._prec == b._prec


def test_bug1():
    assert re(x) != x
    x.series(x, 0, 1)
    assert re(x) != x


def test_Symbol():
    e = a*b
    assert e == a*b
    assert a*b*b == a*b**2
    assert a*b*b + c == c + a*b**2
    assert a*b*b - c == -c + a*b**2

    x = Symbol('x', complex=True, real=False)
    assert x.is_imaginary is None  # could be I or 1 + I
    x = Symbol('x', complex=True, imaginary=False)
    assert x.is_real is None  # could be 1 or 1 + I
    x = Symbol('x', real=True)
    assert x.is_complex
    x = Symbol('x', imaginary=True)
    assert x.is_complex
    x = Symbol('x', real=False, imaginary=False)
    assert x.is_complex is None  # might be a non-number


def test_arit0():
    p = Rational(5)
    e = a*b
    assert e == a*b
    e = a*b + b*a
    assert e == 2*a*b
    e = a*b + b*a + a*b + p*b*a
    assert e == 8*a*b
    e = a*b + b*a + a*b + p*b*a + a
    assert e == a + 8*a*b
    e = a + a
    assert e == 2*a
    e = a + b + a
    assert e == b + 2*a
    e = a + b*b + a + b*b
    assert e == 2*a + 2*b**2
    e = a + Rational(2) + b*b + a + b*b + p
    assert e == 7 + 2*a + 2*b**2
    e = (a + b*b + a + b*b)*p
    assert e == 5*(2*a + 2*b**2)
    e = (a*b*c + c*b*a + b*a*c)*p
    assert e == 15*a*b*c
    e = (a*b*c + c*b*a + b*a*c)*p - Rational(15)*a*b*c
    assert e == Rational(0)
    e = Rational(50)*(a - a)
    assert e == Rational(0)
    e = b*a - b - a*b + b
    assert e == Rational(0)
    e = a*b + c**p
    assert e == a*b + c**5
    e = a/b
    assert e == a*b**(-1)
    e = a*2*2
    assert e == 4*a
    e = 2 + a*2/2
    assert e == 2 + a
    e = 2 - a - 2
    assert e == -a
    e = 2*a*2
    assert e == 4*a
    e = 2/a/2
    assert e == a**(-1)
    e = 2**a**2
    assert e == 2**(a**2)
    e = -(1 + a)
    assert e == -1 - a
    e = S.Half*(1 + a)
    assert e == S.Half + a/2


def test_div():
    e = a/b
    assert e == a*b**(-1)
    e = a/b + c/2
    assert e == a*b**(-1) + Rational(1)/2*c
    e = (1 - b)/(b - 1)
    assert e == (1 + -b)*((-1) + b)**(-1)


def test_pow_arit():
    n1 = Rational(1)
    n2 = Rational(2)
    n5 = Rational(5)
    e = a*a
    assert e == a**2
    e = a*a*a
    assert e == a**3
    e = a*a*a*a**Rational(6)
    assert e == a**9
    e = a*a*a*a**Rational(6) - a**Rational(9)
    assert e == Rational(0)
    e = a**(b - b)
    assert e == Rational(1)
    e = (a + Rational(1) - a)**b
    assert e == Rational(1)

    e = (a + b + c)**n2
    assert e == (a + b + c)**2
    assert e.expand() == 2*b*c + 2*a*c + 2*a*b + a**2 + c**2 + b**2

    e = (a + b)**n2
    assert e == (a + b)**2
    assert e.expand() == 2*a*b + a**2 + b**2

    e = (a + b)**(n1/n2)
    assert e == sqrt(a + b)
    assert e.expand() == sqrt(a + b)

    n = n5**(n1/n2)
    assert n == sqrt(5)
    e = n*a*b - n*b*a
    assert e == Rational(0)
    e = n*a*b + n*b*a
    assert e == 2*a*b*sqrt(5)
    assert e.diff(a) == 2*b*sqrt(5)
    assert e.diff(a) == 2*b*sqrt(5)
    e = a/b**2
    assert e == a*b**(-2)

    assert sqrt(2*(1 + sqrt(2))) == (2*(1 + 2**S.Half))**S.Half

    x = Symbol('x')
    y = Symbol('y')

    assert ((x*y)**3).expand() == y**3 * x**3
    assert ((x*y)**-3).expand() == y**-3 * x**-3

    assert (x**5*(3*x)**(3)).expand() == 27 * x**8
    assert (x**5*(-3*x)**(3)).expand() == -27 * x**8
    assert (x**5*(3*x)**(-3)).expand() == x**2 * Rational(1, 27)
    assert (x**5*(-3*x)**(-3)).expand() == x**2 * Rational(-1, 27)

    # expand_power_exp
    _x = Symbol('x', zero=False)
    _y = Symbol('y', zero=False)
    assert (_x**(y**(x + exp(x + y)) + z)).expand(deep=False) == \
        _x**z*_x**(y**(x + exp(x + y)))
    assert (_x**(_y**(x + exp(x + y)) + z)).expand() == \
        _x**z*_x**(_y**x*_y**(exp(x)*exp(y)))

    n = Symbol('n', even=False)
    k = Symbol('k', even=True)
    o = Symbol('o', odd=True)

    assert unchanged(Pow, -1, x)
    assert unchanged(Pow, -1, n)
    assert (-2)**k == 2**k
    assert (-1)**k == 1
    assert (-1)**o == -1


def test_pow2():
    # x**(2*y) is always (x**y)**2 but is only (x**2)**y if
    #                                  x.is_positive or y.is_integer
    # let x = 1 to see why the following are not true.
    assert (-x)**Rational(2, 3) != x**Rational(2, 3)
    assert (-x)**Rational(5, 7) != -x**Rational(5, 7)
    assert ((-x)**2)**Rational(1, 3) != ((-x)**Rational(1, 3))**2
    assert sqrt(x**2) != x


def test_pow3():
    assert sqrt(2)**3 == 2 * sqrt(2)
    assert sqrt(2)**3 == sqrt(8)


def test_mod_pow():
    for s, t, u, v in [(4, 13, 497, 445), (4, -3, 497, 365),
            (3.2, 2.1, 1.9, 0.1031015682350942), (S(3)/2, 5, S(5)/6, S(3)/32)]:
        assert pow(S(s), t, u) == v
        assert pow(S(s), S(t), u) == v
        assert pow(S(s), t, S(u)) == v
        assert pow(S(s), S(t), S(u)) == v
    assert pow(S(2), S(10000000000), S(3)) == 1
    assert pow(x, y, z) == x**y%z
    raises(TypeError, lambda: pow(S(4), "13", 497))
    raises(TypeError, lambda: pow(S(4), 13, "497"))


def test_pow_E():
    assert 2**(y/log(2)) == S.Exp1**y
    assert 2**(y/log(2)/3) == S.Exp1**(y/3)
    assert 3**(1/log(-3)) != S.Exp1
    assert (3 + 2*I)**(1/(log(-3 - 2*I) + I*pi)) == S.Exp1
    assert (4 + 2*I)**(1/(log(-4 - 2*I) + I*pi)) == S.Exp1
    assert (3 + 2*I)**(1/(log(-3 - 2*I, 3)/2 + I*pi/log(3)/2)) == 9
    assert (3 + 2*I)**(1/(log(3 + 2*I, 3)/2)) == 9
    # every time tests are run they will affirm with a different random
    # value that this identity holds
    while 1:
        b = x._random()
        r, i = b.as_real_imag()
        if i:
            break
    assert verify_numerically(b**(1/(log(-b) + sign(i)*I*pi).n()), S.Exp1)


def test_pow_issue_3516():
    assert 4**Rational(1, 4) == sqrt(2)


def test_pow_im():
    for m in (-2, -1, 2):
        for d in (3, 4, 5):
            b = m*I
            for i in range(1, 4*d + 1):
                e = Rational(i, d)
                assert (b**e - b.n()**e.n()).n(2, chop=1e-10) == 0

    e = Rational(7, 3)
    assert (2*x*I)**e == 4*2**Rational(1, 3)*(I*x)**e  # same as Wolfram Alpha
    im = symbols('im', imaginary=True)
    assert (2*im*I)**e == 4*2**Rational(1, 3)*(I*im)**e

    args = [I, I, I, I, 2]
    e = Rational(1, 3)
    ans = 2**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args = [I, I, I, 2]
    e = Rational(1, 3)
    ans = 2**e*(-I)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-3)
    ans = (6*I)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-1)
    ans = (-6*I)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans

    args = [I, I, 2]
    e = Rational(1, 3)
    ans = (-2)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-3)
    ans = (6)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-1)
    ans = (-6)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    assert Mul(Pow(-1, Rational(3, 2), evaluate=False), I, I) == I
    assert Mul(I*Pow(I, S.Half, evaluate=False)) == sqrt(I)*I


def test_real_mul():
    assert Float(0) * pi * x == 0
    assert set((Float(1) * pi * x).args) == {Float(1), pi, x}


def test_ncmul():
    A = Symbol("A", commutative=False)
    B = Symbol("B", commutative=False)
    C = Symbol("C", commutative=False)
    assert A*B != B*A
    assert A*B*C != C*B*A
    assert A*b*B*3*C == 3*b*A*B*C
    assert A*b*B*3*C != 3*b*B*A*C
    assert A*b*B*3*C == 3*A*B*C*b

    assert A + B == B + A
    assert (A + B)*C != C*(A + B)

    assert C*(A + B)*C != C*C*(A + B)

    assert A*A == A**2
    assert (A + B)*(A + B) == (A + B)**2

    assert A**-1 * A == 1
    assert A/A == 1
    assert A/(A**2) == 1/A

    assert A/(1 + A) == A/(1 + A)

    assert set((A + B + 2*(A + B)).args) == \
        {A, B, 2*(A + B)}


def test_mul_add_identity():
    m = Mul(1, 2)
    assert isinstance(m, Rational) and m.p == 2 and m.q == 1
    m = Mul(1, 2, evaluate=False)
    assert isinstance(m, Mul) and m.args == (1, 2)
    m = Mul(0, 1)
    assert m is S.Zero
    m = Mul(0, 1, evaluate=False)
    assert isinstance(m, Mul) and m.args == (0, 1)
    m = Add(0, 1)
    assert m is S.One
    m = Add(0, 1, evaluate=False)
    assert isinstance(m, Add) and m.args == (0, 1)


def test_ncpow():
    x = Symbol('x', commutative=False)
    y = Symbol('y', commutative=False)
    z = Symbol('z', commutative=False)
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    assert (x**2)*(y**2) != (y**2)*(x**2)
    assert (x**-2)*y != y*(x**2)
    assert 2**x*2**y != 2**(x + y)
    assert 2**x*2**y*2**z != 2**(x + y + z)
    assert 2**x*2**(2*x) == 2**(3*x)
    assert 2**x*2**(2*x)*2**x == 2**(4*x)
    assert exp(x)*exp(y) != exp(y)*exp(x)
    assert exp(x)*exp(y)*exp(z) != exp(y)*exp(x)*exp(z)
    assert exp(x)*exp(y)*exp(z) != exp(x + y + z)
    assert x**a*x**b != x**(a + b)
    assert x**a*x**b*x**c != x**(a + b + c)
    assert x**3*x**4 == x**7
    assert x**3*x**4*x**2 == x**9
    assert x**a*x**(4*a) == x**(5*a)
    assert x**a*x**(4*a)*x**a == x**(6*a)


def test_powerbug():
    x = Symbol("x")
    assert x**1 != (-x)**1
    assert x**2 == (-x)**2
    assert x**3 != (-x)**3
    assert x**4 == (-x)**4
    assert x**5 != (-x)**5
    assert x**6 == (-x)**6

    assert x**128 == (-x)**128
    assert x**129 != (-x)**129

    assert (2*x)**2 == (-2*x)**2


def test_Mul_doesnt_expand_exp():
    x = Symbol('x')
    y = Symbol('y')
    assert unchanged(Mul, exp(x), exp(y))
    assert unchanged(Mul, 2**x, 2**y)
    assert x**2*x**3 == x**5
    assert 2**x*3**x == 6**x
    assert x**(y)*x**(2*y) == x**(3*y)
    assert sqrt(2)*sqrt(2) == 2
    assert 2**x*2**(2*x) == 2**(3*x)
    assert sqrt(2)*2**Rational(1, 4)*5**Rational(3, 4) == 10**Rational(3, 4)
    assert (x**(-log(5)/log(3))*x)/(x*x**( - log(5)/log(3))) == sympify(1)


def test_Mul_is_integer():
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nr = Symbol('nr', rational=False)
    ir = Symbol('ir', irrational=True)
    nz = Symbol('nz', integer=True, zero=False)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    i2 = Symbol('2', prime=True, even=True)

    assert (k/3).is_integer is None
    assert (nz/3).is_integer is None
    assert (nr/3).is_integer is False
    assert (ir/3).is_integer is False
    assert (x*k*n).is_integer is None
    assert (e/2).is_integer is True
    assert (e**2/2).is_integer is True
    assert (2/k).is_integer is None
    assert (2/k**2).is_integer is None
    assert ((-1)**k*n).is_integer is True
    assert (3*k*e/2).is_integer is True
    assert (2*k*e/3).is_integer is None
    assert (e/o).is_integer is None
    assert (o/e).is_integer is False
    assert (o/i2).is_integer is False
    assert Mul(k, 1/k, evaluate=False).is_integer is None
    assert Mul(2., S.Half, evaluate=False).is_integer is None
    assert (2*sqrt(k)).is_integer is None
    assert (2*k**n).is_integer is None

    s = 2**2**2**Pow(2, 1000, evaluate=False)
    m = Mul(s, s, evaluate=False)
    assert m.is_integer

    # broken in 1.6 and before, see #20161
    xq = Symbol('xq', rational=True)
    yq = Symbol('yq', rational=True)
    assert (xq*yq).is_integer is None
    e_20161 = Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False)
    assert e_20161.is_integer is not True # expand(e_20161) -> -1/2, but no need to see that in the assumption without evaluation


def test_Add_Mul_is_integer():
    x = Symbol('x')

    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nk = Symbol('nk', integer=False)
    nr = Symbol('nr', rational=False)
    nz = Symbol('nz', integer=True, zero=False)

    assert (-nk).is_integer is None
    assert (-nr).is_integer is False
    assert (2*k).is_integer is True
    assert (-k).is_integer is True

    assert (k + nk).is_integer is False
    assert (k + n).is_integer is True
    assert (k + x).is_integer is None
    assert (k + n*x).is_integer is None
    assert (k + n/3).is_integer is None
    assert (k + nz/3).is_integer is None
    assert (k + nr/3).is_integer is False

    assert ((1 + sqrt(3))*(-sqrt(3) + 1)).is_integer is not False
    assert (1 + (1 + sqrt(3))*(-sqrt(3) + 1)).is_integer is not False


def test_Add_Mul_is_finite():
    x = Symbol('x', extended_real=True, finite=False)

    assert sin(x).is_finite is True
    assert (x*sin(x)).is_finite is None
    assert (x*atan(x)).is_finite is False
    assert (1024*sin(x)).is_finite is True
    assert (sin(x)*exp(x)).is_finite is None
    assert (sin(x)*cos(x)).is_finite is True
    assert (x*sin(x)*exp(x)).is_finite is None

    assert (sin(x) - 67).is_finite is True
    assert (sin(x) + exp(x)).is_finite is not True
    assert (1 + x).is_finite is False
    assert (1 + x**2 + (1 + x)*(1 - x)).is_finite is None
    assert (sqrt(2)*(1 + x)).is_finite is False
    assert (sqrt(2)*(1 + x)*(1 - x)).is_finite is False


def test_Mul_is_even_odd():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)

    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)

    assert (2*x).is_even is True
    assert (2*x).is_odd is False

    assert (3*x).is_even is None
    assert (3*x).is_odd is None

    assert (k/3).is_integer is None
    assert (k/3).is_even is None
    assert (k/3).is_odd is None

    assert (2*n).is_even is True
    assert (2*n).is_odd is False

    assert (2*m).is_even is True
    assert (2*m).is_odd is False

    assert (-n).is_even is False
    assert (-n).is_odd is True

    assert (k*n).is_even is False
    assert (k*n).is_odd is True

    assert (k*m).is_even is True
    assert (k*m).is_odd is False

    assert (k*n*m).is_even is True
    assert (k*n*m).is_odd is False

    assert (k*m*x).is_even is True
    assert (k*m*x).is_odd is False

    # issue 6791:
    assert (x/2).is_integer is None
    assert (k/2).is_integer is False
    assert (m/2).is_integer is True

    assert (x*y).is_even is None
    assert (x*x).is_even is None
    assert (x*(x + k)).is_even is True
    assert (x*(x + m)).is_even is None

    assert (x*y).is_odd is None
    assert (x*x).is_odd is None
    assert (x*(x + k)).is_odd is False
    assert (x*(x + m)).is_odd is None

    # issue 8648
    assert (m**2/2).is_even
    assert (m**2/3).is_even is False
    assert (2/m**2).is_odd is False
    assert (2/m).is_odd is None


@XFAIL
def test_evenness_in_ternary_integer_product_with_odd():
    # Tests that oddness inference is independent of term ordering.
    # Term ordering at the point of testing depends on SymPy's symbol order, so
    # we try to force a different order by modifying symbol names.
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    assert (x*y*(y + k)).is_even is True
    assert (y*x*(x + k)).is_even is True


def test_evenness_in_ternary_integer_product_with_even():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    m = Symbol('m', even=True)
    assert (x*y*(y + m)).is_even is None


@XFAIL
def test_oddness_in_ternary_integer_product_with_odd():
    # Tests that oddness inference is independent of term ordering.
    # Term ordering at the point of testing depends on SymPy's symbol order, so
    # we try to force a different order by modifying symbol names.
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    assert (x*y*(y + k)).is_odd is False
    assert (y*x*(x + k)).is_odd is False


def test_oddness_in_ternary_integer_product_with_even():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    m = Symbol('m', even=True)
    assert (x*y*(y + m)).is_odd is None


def test_Mul_is_rational():
    x = Symbol('x')
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True, nonzero=True)

    assert (n/m).is_rational is True
    assert (x/pi).is_rational is None
    assert (x/n).is_rational is None
    assert (m/pi).is_rational is False

    r = Symbol('r', rational=True)
    assert (pi*r).is_rational is None

    # issue 8008
    z = Symbol('z', zero=True)
    i = Symbol('i', imaginary=True)
    assert (z*i).is_rational is True
    bi = Symbol('i', imaginary=True, finite=True)
    assert (z*bi).is_zero is True


def test_Add_is_rational():
    x = Symbol('x')
    n = Symbol('n', rational=True)
    m = Symbol('m', rational=True)

    assert (n + m).is_rational is True
    assert (x + pi).is_rational is None
    assert (x + n).is_rational is None
    assert (n + pi).is_rational is False


def test_Add_is_even_odd():
    x = Symbol('x', integer=True)

    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)

    assert (k + 7).is_even is True
    assert (k + 7).is_odd is False

    assert (-k + 7).is_even is True
    assert (-k + 7).is_odd is False

    assert (k - 12).is_even is False
    assert (k - 12).is_odd is True

    assert (-k - 12).is_even is False
    assert (-k - 12).is_odd is True

    assert (k + n).is_even is True
    assert (k + n).is_odd is False

    assert (k + m).is_even is False
    assert (k + m).is_odd is True

    assert (k + n + m).is_even is True
    assert (k + n + m).is_odd is False

    assert (k + n + x + m).is_even is None
    assert (k + n + x + m).is_odd is None


def test_Mul_is_negative_positive():
    x = Symbol('x', real=True)
    y = Symbol('y', extended_real=False, complex=True)
    z = Symbol('z', zero=True)

    e = 2*z
    assert e.is_Mul and e.is_positive is False and e.is_negative is False

    neg = Symbol('neg', negative=True)
    pos = Symbol('pos', positive=True)
    nneg = Symbol('nneg', nonnegative=True)
    npos = Symbol('npos', nonpositive=True)

    assert neg.is_negative is True
    assert (-neg).is_negative is False
    assert (2*neg).is_negative is True

    assert (2*pos)._eval_is_extended_negative() is False
    assert (2*pos).is_negative is False

    assert pos.is_negative is False
    assert (-pos).is_negative is True
    assert (2*pos).is_negative is False

    assert (pos*neg).is_negative is True
    assert (2*pos*neg).is_negative is True
    assert (-pos*neg).is_negative is False
    assert (pos*neg*y).is_negative is False     # y.is_real=F;  !real -> !neg

    assert nneg.is_negative is False
    assert (-nneg).is_negative is None
    assert (2*nneg).is_negative is False

    assert npos.is_negative is None
    assert (-npos).is_negative is False
    assert (2*npos).is_negative is None

    assert (nneg*npos).is_negative is None

    assert (neg*nneg).is_negative is None
    assert (neg*npos).is_negative is False

    assert (pos*nneg).is_negative is False
    assert (pos*npos).is_negative is None

    assert (npos*neg*nneg).is_negative is False
    assert (npos*pos*nneg).is_negative is None

    assert (-npos*neg*nneg).is_negative is None
    assert (-npos*pos*nneg).is_negative is False

    assert (17*npos*neg*nneg).is_negative is False
    assert (17*npos*pos*nneg).is_negative is None

    assert (neg*npos*pos*nneg).is_negative is False

    assert (x*neg).is_negative is None
    assert (nneg*npos*pos*x*neg).is_negative is None

    assert neg.is_positive is False
    assert (-neg).is_positive is True
    assert (2*neg).is_positive is False

    assert pos.is_positive is True
    assert (-pos).is_positive is False
    assert (2*pos).is_positive is True

    assert (pos*neg).is_positive is False
    assert (2*pos*neg).is_positive is False
    assert (-pos*neg).is_positive is True
    assert (-pos*neg*y).is_positive is False    # y.is_real=F;  !real -> !neg

    assert nneg.is_positive is None
    assert (-nneg).is_positive is False
    assert (2*nneg).is_positive is None

    assert npos.is_positive is False
    assert (-npos).is_positive is None
    assert (2*npos).is_positive is False

    assert (nneg*npos).is_positive is False

    assert (neg*nneg).is_positive is False
    assert (neg*npos).is_positive is None

    assert (pos*nneg).is_positive is None
    assert (pos*npos).is_positive is False

    assert (npos*neg*nneg).is_positive is None
    assert (npos*pos*nneg).is_positive is False

    assert (-npos*neg*nneg).is_positive is False
    assert (-npos*pos*nneg).is_positive is None

    assert (17*npos*neg*nneg).is_positive is None
    assert (17*npos*pos*nneg).is_positive is False

    assert (neg*npos*pos*nneg).is_positive is None

    assert (x*neg).is_positive is None
    assert (nneg*npos*pos*x*neg).is_positive is None


def test_Mul_is_negative_positive_2():
    a = Symbol('a', nonnegative=True)
    b = Symbol('b', nonnegative=True)
    c = Symbol('c', nonpositive=True)
    d = Symbol('d', nonpositive=True)

    assert (a*b).is_nonnegative is True
    assert (a*b).is_negative is False
    assert (a*b).is_zero is None
    assert (a*b).is_positive is None

    assert (c*d).is_nonnegative is True
    assert (c*d).is_negative is False
    assert (c*d).is_zero is None
    assert (c*d).is_positive is None

    assert (a*c).is_nonpositive is True
    assert (a*c).is_positive is False
    assert (a*c).is_zero is None
    assert (a*c).is_negative is None


def test_Mul_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)

    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    assert k.is_nonpositive is True
    assert (-k).is_nonpositive is False
    assert (2*k).is_nonpositive is True

    assert n.is_nonpositive is False
    assert (-n).is_nonpositive is True
    assert (2*n).is_nonpositive is False

    assert (n*k).is_nonpositive is True
    assert (2*n*k).is_nonpositive is True
    assert (-n*k).is_nonpositive is False

    assert u.is_nonpositive is None
    assert (-u).is_nonpositive is True
    assert (2*u).is_nonpositive is None

    assert v.is_nonpositive is True
    assert (-v).is_nonpositive is None
    assert (2*v).is_nonpositive is True

    assert (u*v).is_nonpositive is True

    assert (k*u).is_nonpositive is True
    assert (k*v).is_nonpositive is None

    assert (n*u).is_nonpositive is None
    assert (n*v).is_nonpositive is True

    assert (v*k*u).is_nonpositive is None
    assert (v*n*u).is_nonpositive is True

    assert (-v*k*u).is_nonpositive is True
    assert (-v*n*u).is_nonpositive is None

    assert (17*v*k*u).is_nonpositive is None
    assert (17*v*n*u).is_nonpositive is True

    assert (k*v*n*u).is_nonpositive is None

    assert (x*k).is_nonpositive is None
    assert (u*v*n*x*k).is_nonpositive is None

    assert k.is_nonnegative is False
    assert (-k).is_nonnegative is True
    assert (2*k).is_nonnegative is False

    assert n.is_nonnegative is True
    assert (-n).is_nonnegative is False
    assert (2*n).is_nonnegative is True

    assert (n*k).is_nonnegative is False
    assert (2*n*k).is_nonnegative is False
    assert (-n*k).is_nonnegative is True

    assert u.is_nonnegative is True
    assert (-u).is_nonnegative is None
    assert (2*u).is_nonnegative is True

    assert v.is_nonnegative is None
    assert (-v).is_nonnegative is True
    assert (2*v).is_nonnegative is None

    assert (u*v).is_nonnegative is None

    assert (k*u).is_nonnegative is None
    assert (k*v).is_nonnegative is True

    assert (n*u).is_nonnegative is True
    assert (n*v).is_nonnegative is None

    assert (v*k*u).is_nonnegative is True
    assert (v*n*u).is_nonnegative is None

    assert (-v*k*u).is_nonnegative is None
    assert (-v*n*u).is_nonnegative is True

    assert (17*v*k*u).is_nonnegative is True
    assert (17*v*n*u).is_nonnegative is None

    assert (k*v*n*u).is_nonnegative is True

    assert (x*k).is_nonnegative is None
    assert (u*v*n*x*k).is_nonnegative is None


def test_Add_is_negative_positive():
    x = Symbol('x', real=True)

    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    assert (k - 2).is_negative is True
    assert (k + 17).is_negative is None
    assert (-k - 5).is_negative is None
    assert (-k + 123).is_negative is False

    assert (k - n).is_negative is True
    assert (k + n).is_negative is None
    assert (-k - n).is_negative is None
    assert (-k + n).is_negative is False

    assert (k - n - 2).is_negative is True
    assert (k + n + 17).is_negative is None
    assert (-k - n - 5).is_negative is None
    assert (-k + n + 123).is_negative is False

    assert (-2*k + 123*n + 17).is_negative is False

    assert (k + u).is_negative is None
    assert (k + v).is_negative is True
    assert (n + u).is_negative is False
    assert (n + v).is_negative is None

    assert (u - v).is_negative is False
    assert (u + v).is_negative is None
    assert (-u - v).is_negative is None
    assert (-u + v).is_negative is None

    assert (u - v + n + 2).is_negative is False
    assert (u + v + n + 2).is_negative is None
    assert (-u - v + n + 2).is_negative is None
    assert (-u + v + n + 2).is_negative is None

    assert (k + x).is_negative is None
    assert (k + x - n).is_negative is None

    assert (k - 2).is_positive is False
    assert (k + 17).is_positive is None
    assert (-k - 5).is_positive is None
    assert (-k + 123).is_positive is True

    assert (k - n).is_positive is False
    assert (k + n).is_positive is None
    assert (-k - n).is_positive is None
    assert (-k + n).is_positive is True

    assert (k - n - 2).is_positive is False
    assert (k + n + 17).is_positive is None
    assert (-k - n - 5).is_positive is None
    assert (-k + n + 123).is_positive is True

    assert (-2*k + 123*n + 17).is_positive is True

    assert (k + u).is_positive is None
    assert (k + v).is_positive is False
    assert (n + u).is_positive is True
    assert (n + v).is_positive is None

    assert (u - v).is_positive is None
    assert (u + v).is_positive is None
    assert (-u - v).is_positive is None
    assert (-u + v).is_positive is False

    assert (u - v - n - 2).is_positive is None
    assert (u + v - n - 2).is_positive is None
    assert (-u - v - n - 2).is_positive is None
    assert (-u + v - n - 2).is_positive is False

    assert (n + x).is_positive is None
    assert (n + x - k).is_positive is None

    z = (-3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2)
    assert z.is_zero
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert z.is_zero


def test_Add_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)

    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    assert (u - 2).is_nonpositive is None
    assert (u + 17).is_nonpositive is False
    assert (-u - 5).is_nonpositive is True
    assert (-u + 123).is_nonpositive is None

    assert (u - v).is_nonpositive is None
    assert (u + v).is_nonpositive is None
    assert (-u - v).is_nonpositive is None
    assert (-u + v).is_nonpositive is True

    assert (u - v - 2).is_nonpositive is None
    assert (u + v + 17).is_nonpositive is None
    assert (-u - v - 5).is_nonpositive is None
    assert (-u + v - 123).is_nonpositive is True

    assert (-2*u + 123*v - 17).is_nonpositive is True

    assert (k + u).is_nonpositive is None
    assert (k + v).is_nonpositive is True
    assert (n + u).is_nonpositive is False
    assert (n + v).is_nonpositive is None

    assert (k - n).is_nonpositive is True
    assert (k + n).is_nonpositive is None
    assert (-k - n).is_nonpositive is None
    assert (-k + n).is_nonpositive is False

    assert (k - n + u + 2).is_nonpositive is None
    assert (k + n + u + 2).is_nonpositive is None
    assert (-k - n + u + 2).is_nonpositive is None
    assert (-k + n + u + 2).is_nonpositive is False

    assert (u + x).is_nonpositive is None
    assert (v - x - n).is_nonpositive is None

    assert (u - 2).is_nonnegative is None
    assert (u + 17).is_nonnegative is True
    assert (-u - 5).is_nonnegative is False
    assert (-u + 123).is_nonnegative is None

    assert (u - v).is_nonnegative is True
    assert (u + v).is_nonnegative is None
    assert (-u - v).is_nonnegative is None
    assert (-u + v).is_nonnegative is None

    assert (u - v + 2).is_nonnegative is True
    assert (u + v + 17).is_nonnegative is None
    assert (-u - v - 5).is_nonnegative is None
    assert (-u + v - 123).is_nonnegative is False

    assert (2*u - 123*v + 17).is_nonnegative is True

    assert (k + u).is_nonnegative is None
    assert (k + v).is_nonnegative is False
    assert (n + u).is_nonnegative is True
    assert (n + v).is_nonnegative is None

    assert (k - n).is_nonnegative is False
    assert (k + n).is_nonnegative is None
    assert (-k - n).is_nonnegative is None
    assert (-k + n).is_nonnegative is True

    assert (k - n - u - 2).is_nonnegative is False
    assert (k + n - u - 2).is_nonnegative is None
    assert (-k - n - u - 2).is_nonnegative is None
    assert (-k + n - u - 2).is_nonnegative is None

    assert (u - x).is_nonnegative is None
    assert (v + x + n).is_nonnegative is None


def test_Pow_is_integer():
    x = Symbol('x')

    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True, nonnegative=True)
    m = Symbol('m', integer=True, positive=True)

    assert (k**2).is_integer is True
    assert (k**(-2)).is_integer is None
    assert ((m + 1)**(-2)).is_integer is False
    assert (m**(-1)).is_integer is None  # issue 8580

    assert (2**k).is_integer is None
    assert (2**(-k)).is_integer is None

    assert (2**n).is_integer is True
    assert (2**(-n)).is_integer is None

    assert (2**m).is_integer is True
    assert (2**(-m)).is_integer is False

    assert (x**2).is_integer is None
    assert (2**x).is_integer is None

    assert (k**n).is_integer is True
    assert (k**(-n)).is_integer is None

    assert (k**x).is_integer is None
    assert (x**k).is_integer is None

    assert (k**(n*m)).is_integer is True
    assert (k**(-n*m)).is_integer is None

    assert sqrt(3).is_integer is False
    assert sqrt(.3).is_integer is False
    assert Pow(3, 2, evaluate=False).is_integer is True
    assert Pow(3, 0, evaluate=False).is_integer is True
    assert Pow(3, -2, evaluate=False).is_integer is False
    assert Pow(S.Half, 3, evaluate=False).is_integer is False
    # decided by re-evaluating
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    assert Pow(4, S.Half, evaluate=False).is_integer is True
    assert Pow(S.Half, -2, evaluate=False).is_integer is True

    assert ((-1)**k).is_integer

    # issue 8641
    x = Symbol('x', real=True, integer=False)
    assert (x**2).is_integer is None

    # issue 10458
    x = Symbol('x', positive=True)
    assert (1/(x + 1)).is_integer is False
    assert (1/(-x - 1)).is_integer is False
    assert (-1/(x + 1)).is_integer is False
    # issue 23287
    assert (x**2/2).is_integer is None

    # issue 8648-like
    k = Symbol('k', even=True)
    assert (k**3/2).is_integer
    assert (k**3/8).is_integer
    assert (k**3/16).is_integer is None
    assert (2/k).is_integer is None
    assert (2/k**2).is_integer is False
    o = Symbol('o', odd=True)
    assert (k/o).is_integer is None
    o = Symbol('o', odd=True, prime=True)
    assert (k/o).is_integer is False


def test_Pow_is_real():
    x = Symbol('x', real=True)
    y = Symbol('y', positive=True)

    assert (x**2).is_real is True
    assert (x**3).is_real is True
    assert (x**x).is_real is None
    assert (y**x).is_real is True

    assert (x**Rational(1, 3)).is_real is None
    assert (y**Rational(1, 3)).is_real is True

    assert sqrt(-1 - sqrt(2)).is_real is False

    i = Symbol('i', imaginary=True)
    assert (i**i).is_real is None
    assert (I**i).is_extended_real is True
    assert ((-I)**i).is_extended_real is True
    assert (2**i).is_real is None  # (2**(pi/log(2) * I)) is real, 2**I is not
    assert (2**I).is_real is False
    assert (2**-I).is_real is False
    assert (i**2).is_extended_real is True
    assert (i**3).is_extended_real is False
    assert (i**x).is_real is None  # could be (-I)**(2/3)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    k = Symbol('k', integer=True)
    assert (i**e).is_extended_real is True
    assert (i**o).is_extended_real is False
    assert (i**k).is_real is None
    assert (i**(4*k)).is_extended_real is True

    x = Symbol("x", nonnegative=True)
    y = Symbol("y", nonnegative=True)
    assert im(x**y).expand(complex=True) is S.Zero
    assert (x**y).is_real is True
    i = Symbol('i', imaginary=True)
    assert (exp(i)**I).is_extended_real is True
    assert log(exp(i)).is_imaginary is None  # i could be 2*pi*I
    c = Symbol('c', complex=True)
    assert log(c).is_real is None  # c could be 0 or 2, too
    assert log(exp(c)).is_real is None  # log(0), log(E), ...
    n = Symbol('n', negative=False)
    assert log(n).is_real is None
    n = Symbol('n', nonnegative=True)
    assert log(n).is_real is None

    assert sqrt(-I).is_real is False  # issue 7843

    i = Symbol('i', integer=True)
    assert (1/(i-1)).is_real is None
    assert (1/(i-1)).is_extended_real is None

    # test issue 20715
    from sympy.core.parameters import evaluate
    x = S(-1)
    with evaluate(False):
        assert x.is_negative is True

    f = Pow(x, -1)
    with evaluate(False):
        assert f.is_imaginary is False


def test_real_Pow():
    k = Symbol('k', integer=True, nonzero=True)
    assert (k**(I*pi/log(k))).is_real


def test_Pow_is_finite():
    xe = Symbol('xe', extended_real=True)
    xr = Symbol('xr', real=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    i = Symbol('i', integer=True)

    assert (xe**2).is_finite is None  # xe could be oo
    assert (xr**2).is_finite is True

    assert (xe**xe).is_finite is None
    assert (xr**xe).is_finite is None
    assert (xe**xr).is_finite is None
    # FIXME: The line below should be True rather than None
    # assert (xr**xr).is_finite is True
    assert (xr**xr).is_finite is None

    assert (p**xe).is_finite is None
    assert (p**xr).is_finite is True

    assert (n**xe).is_finite is None
    assert (n**xr).is_finite is True

    assert (sin(xe)**2).is_finite is True
    assert (sin(xr)**2).is_finite is True

    assert (sin(xe)**xe).is_finite is None  # xe, xr could be -pi
    assert (sin(xr)**xr).is_finite is None

    # FIXME: Should the line below be True rather than None?
    assert (sin(xe)**exp(xe)).is_finite is None
    assert (sin(xr)**exp(xr)).is_finite is True

    assert (1/sin(xe)).is_finite is None  # if zero, no, otherwise yes
    assert (1/sin(xr)).is_finite is None

    assert (1/exp(xe)).is_finite is None  # xe could be -oo
    assert (1/exp(xr)).is_finite is True

    assert (1/S.Pi).is_finite is True

    assert (1/(i-1)).is_finite is None


def test_Pow_is_even_odd():
    x = Symbol('x')

    k = Symbol('k', even=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', integer=True, nonnegative=True)
    p = Symbol('p', integer=True, positive=True)

    assert ((-1)**n).is_odd
    assert ((-1)**k).is_odd
    assert ((-1)**(m - p)).is_odd

    assert (k**2).is_even is True
    assert (n**2).is_even is False
    assert (2**k).is_even is None
    assert (x**2).is_even is None

    assert (k**m).is_even is None
    assert (n**m).is_even is False

    assert (k**p).is_even is True
    assert (n**p).is_even is False

    assert (m**k).is_even is None
    assert (p**k).is_even is None

    assert (m**n).is_even is None
    assert (p**n).is_even is None

    assert (k**x).is_even is None
    assert (n**x).is_even is None

    assert (k**2).is_odd is False
    assert (n**2).is_odd is True
    assert (3**k).is_odd is None

    assert (k**m).is_odd is None
    assert (n**m).is_odd is True

    assert (k**p).is_odd is False
    assert (n**p).is_odd is True

    assert (m**k).is_odd is None
    assert (p**k).is_odd is None

    assert (m**n).is_odd is None
    assert (p**n).is_odd is None

    assert (k**x).is_odd is None
    assert (n**x).is_odd is None


def test_Pow_is_negative_positive():
    r = Symbol('r', real=True)

    k = Symbol('k', integer=True, positive=True)
    n = Symbol('n', even=True)
    m = Symbol('m', odd=True)

    x = Symbol('x')

    assert (2**r).is_positive is True
    assert ((-2)**r).is_positive is None
    assert ((-2)**n).is_positive is True
    assert ((-2)**m).is_positive is False

    assert (k**2).is_positive is True
    assert (k**(-2)).is_positive is True

    assert (k**r).is_positive is True
    assert ((-k)**r).is_positive is None
    assert ((-k)**n).is_positive is True
    assert ((-k)**m).is_positive is False

    assert (2**r).is_negative is False
    assert ((-2)**r).is_negative is None
    assert ((-2)**n).is_negative is False
    assert ((-2)**m).is_negative is True

    assert (k**2).is_negative is False
    assert (k**(-2)).is_negative is False

    assert (k**r).is_negative is False
    assert ((-k)**r).is_negative is None
    assert ((-k)**n).is_negative is False
    assert ((-k)**m).is_negative is True

    assert (2**x).is_positive is None
    assert (2**x).is_negative is None


def test_Pow_is_zero():
    z = Symbol('z', zero=True)
    e = z**2
    assert e.is_zero
    assert e.is_positive is False
    assert e.is_negative is False

    assert Pow(0, 0, evaluate=False).is_zero is False
    assert Pow(0, 3, evaluate=False).is_zero
    assert Pow(0, oo, evaluate=False).is_zero
    assert Pow(0, -3, evaluate=False).is_zero is False
    assert Pow(0, -oo, evaluate=False).is_zero is False
    assert Pow(2, 2, evaluate=False).is_zero is False

    a = Symbol('a', zero=False)
    assert Pow(a, 3).is_zero is False  # issue 7965

    assert Pow(2, oo, evaluate=False).is_zero is False
    assert Pow(2, -oo, evaluate=False).is_zero
    assert Pow(S.Half, oo, evaluate=False).is_zero
    assert Pow(S.Half, -oo, evaluate=False).is_zero is False

    # All combinations of real/complex base/exponent
    h = S.Half
    T = True
    F = False
    N = None

    pow_iszero = [
        ['**',  0,  h,  1,  2, -h, -1,-2,-2*I,-I/2,I/2,1+I,oo,-oo,zoo],
        [   0,  F,  T,  T,  T,  F,  F,  F,  F,  F,  F,  N,  T,  F,  N],
        [   h,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [   1,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  N],
        [   2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [  -h,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [  -1,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  N],
        [  -2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [-2*I,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [-I/2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [ I/2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [ 1+I,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [  oo,  F,  F,  F,  F,  T,  T,  T,  F,  F,  F,  F,  F,  T,  N],
        [ -oo,  F,  F,  F,  F,  T,  T,  T,  F,  F,  F,  F,  F,  T,  N],
        [ zoo,  F,  F,  F,  F,  T,  T,  T,  N,  N,  N,  N,  F,  T,  N]
    ]

    def test_table(table):
        n = len(table[0])
        for row in range(1, n):
            base = table[row][0]
            for col in range(1, n):
                exp = table[0][col]
                is_zero = table[row][col]
                # The actual test here:
                assert Pow(base, exp, evaluate=False).is_zero is is_zero

    test_table(pow_iszero)

    # A zero symbol...
    zo, zo2 = symbols('zo, zo2', zero=True)

    # All combinations of finite symbols
    zf, zf2 = symbols('zf, zf2', finite=True)
    wf, wf2 = symbols('wf, wf2', nonzero=True)
    xf, xf2 = symbols('xf, xf2', real=True)
    yf, yf2 = symbols('yf, yf2', nonzero=True)
    af, af2 = symbols('af, af2', positive=True)
    bf, bf2 = symbols('bf, bf2', nonnegative=True)
    cf, cf2 = symbols('cf, cf2', negative=True)
    df, df2 = symbols('df, df2', nonpositive=True)

    # Without finiteness:
    zi, zi2 = symbols('zi, zi2')
    wi, wi2 = symbols('wi, wi2', zero=False)
    xi, xi2 = symbols('xi, xi2', extended_real=True)
    yi, yi2 = symbols('yi, yi2', zero=False, extended_real=True)
    ai, ai2 = symbols('ai, ai2', extended_positive=True)
    bi, bi2 = symbols('bi, bi2', extended_nonnegative=True)
    ci, ci2 = symbols('ci, ci2', extended_negative=True)
    di, di2 = symbols('di, di2', extended_nonpositive=True)

    pow_iszero_sym = [
        ['**',zo,wf,yf,af,cf,zf,xf,bf,df,zi,wi,xi,yi,ai,bi,ci,di],
        [ zo2, F, N, N, T, F, N, N, N, F, N, N, N, N, T, N, F, F],
        [ wf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ yf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ af2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ cf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ zf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ xf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ bf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ df2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ zi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ wi2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ xi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ yi2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ ai2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ bi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ ci2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ di2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N]
    ]

    test_table(pow_iszero_sym)

    # In some cases (x**x).is_zero is different from (x**y).is_zero even if y
    # has the same assumptions as x.
    assert (zo ** zo).is_zero is False
    assert (wf ** wf).is_zero is False
    assert (yf ** yf).is_zero is False
    assert (af ** af).is_zero is False
    assert (cf ** cf).is_zero is False
    assert (zf ** zf).is_zero is None
    assert (xf ** xf).is_zero is None
    assert (bf ** bf).is_zero is False # None in table
    assert (df ** df).is_zero is None
    assert (zi ** zi).is_zero is None
    assert (wi ** wi).is_zero is None
    assert (xi ** xi).is_zero is None
    assert (yi ** yi).is_zero is None
    assert (ai ** ai).is_zero is False # None in table
    assert (bi ** bi).is_zero is False # None in table
    assert (ci ** ci).is_zero is None
    assert (di ** di).is_zero is None


def test_Pow_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)

    k = Symbol('k', integer=True, nonnegative=True)
    l = Symbol('l', integer=True, positive=True)
    n = Symbol('n', even=True)
    m = Symbol('m', odd=True)

    assert (x**(4*k)).is_nonnegative is True
    assert (2**x).is_nonnegative is True
    assert ((-2)**x).is_nonnegative is None
    assert ((-2)**n).is_nonnegative is True
    assert ((-2)**m).is_nonnegative is False

    assert (k**2).is_nonnegative is True
    assert (k**(-2)).is_nonnegative is None
    assert (k**k).is_nonnegative is True

    assert (k**x).is_nonnegative is None    # NOTE (0**x).is_real = U
    assert (l**x).is_nonnegative is True
    assert (l**x).is_positive is True
    assert ((-k)**x).is_nonnegative is None

    assert ((-k)**m).is_nonnegative is None

    assert (2**x).is_nonpositive is False
    assert ((-2)**x).is_nonpositive is None
    assert ((-2)**n).is_nonpositive is False
    assert ((-2)**m).is_nonpositive is True

    assert (k**2).is_nonpositive is None
    assert (k**(-2)).is_nonpositive is None

    assert (k**x).is_nonpositive is None
    assert ((-k)**x).is_nonpositive is None
    assert ((-k)**n).is_nonpositive is None


    assert (x**2).is_nonnegative is True
    i = symbols('i', imaginary=True)
    assert (i**2).is_nonpositive is True
    assert (i**4).is_nonpositive is False
    assert (i**3).is_nonpositive is False
    assert (I**i).is_nonnegative is True
    assert (exp(I)**i).is_nonnegative is True

    assert ((-l)**n).is_nonnegative is True
    assert ((-l)**m).is_nonpositive is True
    assert ((-k)**n).is_nonnegative is None
    assert ((-k)**m).is_nonpositive is None


def test_Mul_is_imaginary_real():
    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    i1 = Symbol('i1', imaginary=True)
    i2 = Symbol('i2', imaginary=True)
    x = Symbol('x')

    assert I.is_imaginary is True
    assert I.is_real is False
    assert (-I).is_imaginary is True
    assert (-I).is_real is False
    assert (3*I).is_imaginary is True
    assert (3*I).is_real is False
    assert (I*I).is_imaginary is False
    assert (I*I).is_real is True

    e = (p + p*I)
    j = Symbol('j', integer=True, zero=False)
    assert (e**j).is_real is None
    assert (e**(2*j)).is_real is None
    assert (e**j).is_imaginary is None
    assert (e**(2*j)).is_imaginary is None

    assert (e**-1).is_imaginary is False
    assert (e**2).is_imaginary
    assert (e**3).is_imaginary is False
    assert (e**4).is_imaginary is False
    assert (e**5).is_imaginary is False
    assert (e**-1).is_real is False
    assert (e**2).is_real is False
    assert (e**3).is_real is False
    assert (e**4).is_real is True
    assert (e**5).is_real is False
    assert (e**3).is_complex

    assert (r*i1).is_imaginary is None
    assert (r*i1).is_real is None

    assert (x*i1).is_imaginary is None
    assert (x*i1).is_real is None

    assert (i1*i2).is_imaginary is False
    assert (i1*i2).is_real is True

    assert (r*i1*i2).is_imaginary is False
    assert (r*i1*i2).is_real is True

    # Github's issue 5874:
    nr = Symbol('nr', real=False, complex=True)  # e.g. I or 1 + I
    a = Symbol('a', real=True, nonzero=True)
    b = Symbol('b', real=True)
    assert (i1*nr).is_real is None
    assert (a*nr).is_real is False
    assert (b*nr).is_real is None

    ni = Symbol('ni', imaginary=False, complex=True)  # e.g. 2 or 1 + I
    a = Symbol('a', real=True, nonzero=True)
    b = Symbol('b', real=True)
    assert (i1*ni).is_real is False
    assert (a*ni).is_real is None
    assert (b*ni).is_real is None


def test_Mul_hermitian_antihermitian():
    xz, yz = symbols('xz, yz', zero=True, antihermitian=True)
    xf, yf = symbols('xf, yf', hermitian=False, antihermitian=False, finite=True)
    xh, yh = symbols('xh, yh', hermitian=True, antihermitian=False, nonzero=True)
    xa, ya = symbols('xa, ya', hermitian=False, antihermitian=True, zero=False, finite=True)
    assert (xz*xh).is_hermitian is True
    assert (xz*xh).is_antihermitian is True
    assert (xz*xa).is_hermitian is True
    assert (xz*xa).is_antihermitian is True
    assert (xf*yf).is_hermitian is None
    assert (xf*yf).is_antihermitian is None
    assert (xh*yh).is_hermitian is True
    assert (xh*yh).is_antihermitian is False
    assert (xh*ya).is_hermitian is False
    assert (xh*ya).is_antihermitian is True
    assert (xa*ya).is_hermitian is True
    assert (xa*ya).is_antihermitian is False

    a = Symbol('a', hermitian=True, zero=False)
    b = Symbol('b', hermitian=True)
    c = Symbol('c', hermitian=False)
    d = Symbol('d', antihermitian=True)
    e1 = Mul(a, b, c, evaluate=False)
    e2 = Mul(b, a, c, evaluate=False)
    e3 = Mul(a, b, c, d, evaluate=False)
    e4 = Mul(b, a, c, d, evaluate=False)
    e5 = Mul(a, c, evaluate=False)
    e6 = Mul(a, c, d, evaluate=False)
    assert e1.is_hermitian is None
    assert e2.is_hermitian is None
    assert e1.is_antihermitian is None
    assert e2.is_antihermitian is None
    assert e3.is_antihermitian is None
    assert e4.is_antihermitian is None
    assert e5.is_antihermitian is None
    assert e6.is_antihermitian is None


def test_Add_is_comparable():
    assert (x + y).is_comparable is False
    assert (x + 1).is_comparable is False
    assert (Rational(1, 3) - sqrt(8)).is_comparable is True


def test_Mul_is_comparable():
    assert (x*y).is_comparable is False
    assert (x*2).is_comparable is False
    assert (sqrt(2)*Rational(1, 3)).is_comparable is True


def test_Pow_is_comparable():
    assert (x**y).is_comparable is False
    assert (x**2).is_comparable is False
    assert (sqrt(Rational(1, 3))).is_comparable is True


def test_Add_is_positive_2():
    e = Rational(1, 3) - sqrt(8)
    assert e.is_positive is False
    assert e.is_negative is True

    e = pi - 1
    assert e.is_positive is True
    assert e.is_negative is False


def test_Add_is_irrational():
    i = Symbol('i', irrational=True)

    assert i.is_irrational is True
    assert i.is_rational is False

    assert (i + 1).is_irrational is True
    assert (i + 1).is_rational is False


def test_Mul_is_irrational():
    expr = Mul(1, 2, 3, evaluate=False)
    assert expr.is_irrational is False
    expr = Mul(1, I, I, evaluate=False)
    assert expr.is_rational is None # I * I = -1 but *no evaluation allowed*
    # sqrt(2) * I * I = -sqrt(2) is irrational but
    # this can't be determined without evaluating the
    # expression and the eval_is routines shouldn't do that
    expr = Mul(sqrt(2), I, I, evaluate=False)
    assert expr.is_irrational is None


def test_issue_3531():
    # https://github.com/sympy/sympy/issues/3531
    # https://github.com/sympy/sympy/pull/18116
    class MightyNumeric(tuple):
        __slots__ = ()

        def __rtruediv__(self, other):
            return "something"

    assert sympify(1)/MightyNumeric((1, 2)) == "something"


def test_issue_3531b():
    class Foo:
        def __init__(self):
            self.field = 1.0

        def __mul__(self, other):
            self.field = self.field * other

        def __rmul__(self, other):
            self.field = other * self.field
    f = Foo()
    x = Symbol("x")
    assert f*x == x*f


def test_bug3():
    a = Symbol("a")
    b = Symbol("b", positive=True)
    e = 2*a + b
    f = b + 2*a
    assert e == f


def test_suppressed_evaluation():
    a = Add(0, 3, 2, evaluate=False)
    b = Mul(1, 3, 2, evaluate=False)
    c = Pow(3, 2, evaluate=False)
    assert a != 6
    assert a.func is Add
    assert a.args == (0, 3, 2)
    assert b != 6
    assert b.func is Mul
    assert b.args == (1, 3, 2)
    assert c != 9
    assert c.func is Pow
    assert c.args == (3, 2)


def test_AssocOp_doit():
    a = Add(x,x, evaluate=False)
    b = Mul(y,y, evaluate=False)
    c = Add(b,b, evaluate=False)
    d = Mul(a,a, evaluate=False)
    assert c.doit(deep=False).func == Mul
    assert c.doit(deep=False).args == (2,y,y)
    assert c.doit().func == Mul
    assert c.doit().args == (2, Pow(y,2))
    assert d.doit(deep=False).func == Pow
    assert d.doit(deep=False).args == (a, 2*S.One)
    assert d.doit().func == Mul
    assert d.doit().args == (4*S.One, Pow(x,2))


def test_Add_Mul_Expr_args():
    nonexpr = [Basic(), Poly(x, x), FiniteSet(x)]
    for typ in [Add, Mul]:
        for obj in nonexpr:
            # The cache can mess with the stacklevel check
            with warns(SymPyDeprecationWarning, test_stacklevel=False):
                typ(obj, 1)


def test_Add_as_coeff_mul():
    # issue 5524.  These should all be (1, self)
    assert (x + 1).as_coeff_mul() == (1, (x + 1,))
    assert (x + 2).as_coeff_mul() == (1, (x + 2,))
    assert (x + 3).as_coeff_mul() == (1, (x + 3,))

    assert (x - 1).as_coeff_mul() == (1, (x - 1,))
    assert (x - 2).as_coeff_mul() == (1, (x - 2,))
    assert (x - 3).as_coeff_mul() == (1, (x - 3,))

    n = Symbol('n', integer=True)
    assert (n + 1).as_coeff_mul() == (1, (n + 1,))
    assert (n + 2).as_coeff_mul() == (1, (n + 2,))
    assert (n + 3).as_coeff_mul() == (1, (n + 3,))

    assert (n - 1).as_coeff_mul() == (1, (n - 1,))
    assert (n - 2).as_coeff_mul() == (1, (n - 2,))
    assert (n - 3).as_coeff_mul() == (1, (n - 3,))


def test_Pow_as_coeff_mul_doesnt_expand():
    assert exp(x + y).as_coeff_mul() == (1, (exp(x + y),))
    assert exp(x + exp(x + y)) != exp(x + exp(x)*exp(y))

def test_issue_24751():
    expr = Add(-2, -3, evaluate=False)
    expr1 = Add(-1, expr, evaluate=False)
    assert int(expr1) == int((-3 - 2) - 1)


def test_issue_3514_18626():
    assert sqrt(S.Half) * sqrt(6) == 2 * sqrt(3)/2
    assert S.Half*sqrt(6)*sqrt(2) == sqrt(3)
    assert sqrt(6)/2*sqrt(2) == sqrt(3)
    assert sqrt(6)*sqrt(2)/2 == sqrt(3)
    assert sqrt(8)**Rational(2, 3) == 2


def test_make_args():
    assert Add.make_args(x) == (x,)
    assert Mul.make_args(x) == (x,)

    assert Add.make_args(x*y*z) == (x*y*z,)
    assert Mul.make_args(x*y*z) == (x*y*z).args

    assert Add.make_args(x + y + z) == (x + y + z).args
    assert Mul.make_args(x + y + z) == (x + y + z,)

    assert Add.make_args((x + y)**z) == ((x + y)**z,)
    assert Mul.make_args((x + y)**z) == ((x + y)**z,)


def test_issue_5126():
    assert (-2)**x*(-3)**x != 6**x
    i = Symbol('i', integer=1)
    assert (-2)**i*(-3)**i == 6**i


def test_Rational_as_content_primitive():
    c, p = S.One, S.Zero
    assert (c*p).as_content_primitive() == (c, p)
    c, p = S.Half, S.One
    assert (c*p).as_content_primitive() == (c, p)


def test_Add_as_content_primitive():
    assert (x + 2).as_content_primitive() == (1, x + 2)

    assert (3*x + 2).as_content_primitive() == (1, 3*x + 2)
    assert (3*x + 3).as_content_primitive() == (3, x + 1)
    assert (3*x + 6).as_content_primitive() == (3, x + 2)

    assert (3*x + 2*y).as_content_primitive() == (1, 3*x + 2*y)
    assert (3*x + 3*y).as_content_primitive() == (3, x + y)
    assert (3*x + 6*y).as_content_primitive() == (3, x + 2*y)

    assert (3/x + 2*x*y*z**2).as_content_primitive() == (1, 3/x + 2*x*y*z**2)
    assert (3/x + 3*x*y*z**2).as_content_primitive() == (3, 1/x + x*y*z**2)
    assert (3/x + 6*x*y*z**2).as_content_primitive() == (3, 1/x + 2*x*y*z**2)

    assert (2*x/3 + 4*y/9).as_content_primitive() == \
        (Rational(2, 9), 3*x + 2*y)
    assert (2*x/3 + 2.5*y).as_content_primitive() == \
        (Rational(1, 3), 2*x + 7.5*y)

    # the coefficient may sort to a position other than 0
    p = 3 + x + y
    assert (2*p).expand().as_content_primitive() == (2, p)
    assert (2.0*p).expand().as_content_primitive() == (1, 2.*p)
    p *= -1
    assert (2*p).expand().as_content_primitive() == (2, p)


def test_Mul_as_content_primitive():
    assert (2*x).as_content_primitive() == (2, x)
    assert (x*(2 + 2*x)).as_content_primitive() == (2, x*(1 + x))
    assert (x*(2 + 2*y)*(3*x + 3)**2).as_content_primitive() == \
        (18, x*(1 + y)*(x + 1)**2)
    assert ((2 + 2*x)**2*(3 + 6*x) + S.Half).as_content_primitive() == \
        (S.Half, 24*(x + 1)**2*(2*x + 1) + 1)


def test_Pow_as_content_primitive():
    assert (x**y).as_content_primitive() == (1, x**y)
    assert ((2*x + 2)**y).as_content_primitive() == \
        (1, (Mul(2, (x + 1), evaluate=False))**y)
    assert ((2*x + 2)**3).as_content_primitive() == (8, (x + 1)**3)


def test_issue_5460():
    u = Mul(2, (1 + x), evaluate=False)
    assert (2 + u).args == (2, u)


def test_product_irrational():
    assert (I*pi).is_irrational is False
    # The following used to be deduced from the above bug:
    assert (I*pi).is_positive is False


def test_issue_5919():
    assert (x/(y*(1 + y))).expand() == x/(y**2 + y)


def test_Mod():
    assert Mod(x, 1).func is Mod
    assert pi % pi is S.Zero
    assert Mod(5, 3) == 2
    assert Mod(-5, 3) == 1
    assert Mod(5, -3) == -1
    assert Mod(-5, -3) == -2
    assert type(Mod(3.2, 2, evaluate=False)) == Mod
    assert 5 % x == Mod(5, x)
    assert x % 5 == Mod(x, 5)
    assert x % y == Mod(x, y)
    assert (x % y).subs({x: 5, y: 3}) == 2
    assert Mod(nan, 1) is nan
    assert Mod(1, nan) is nan
    assert Mod(nan, nan) is nan

    assert Mod(0, x) == 0
    with raises(ZeroDivisionError):
        Mod(x, 0)

    k = Symbol('k', integer=True)
    m = Symbol('m', integer=True, positive=True)
    assert (x**m % x).func is Mod
    assert (k**(-m) % k).func is Mod
    assert k**m % k == 0
    assert (-2*k)**m % k == 0

    # Float handling
    point3 = Float(3.3) % 1
    assert (x - 3.3) % 1 == Mod(1.*x + 1 - point3, 1)
    assert Mod(-3.3, 1) == 1 - point3
    assert Mod(0.7, 1) == Float(0.7)
    e = Mod(1.3, 1)
    assert comp(e, .3) and e.is_Float
    e = Mod(1.3, .7)
    assert comp(e, .6) and e.is_Float
    e = Mod(1.3, Rational(7, 10))
    assert comp(e, .6) and e.is_Float
    e = Mod(Rational(13, 10), 0.7)
    assert comp(e, .6) and e.is_Float
    e = Mod(Rational(13, 10), Rational(7, 10))
    assert comp(e, .6) and e.is_Rational

    # check that sign is right
    r2 = sqrt(2)
    r3 = sqrt(3)
    for i in [-r3, -r2, r2, r3]:
        for j in [-r3, -r2, r2, r3]:
            assert verify_numerically(i % j, i.n() % j.n())
    for _x in range(4):
        for _y in range(9):
            reps = [(x, _x), (y, _y)]
            assert Mod(3*x + y, 9).subs(reps) == (3*_x + _y) % 9

    # denesting
    t = Symbol('t', real=True)
    assert Mod(Mod(x, t), t) == Mod(x, t)
    assert Mod(-Mod(x, t), t) == Mod(-x, t)
    assert Mod(Mod(x, 2*t), t) == Mod(x, t)
    assert Mod(-Mod(x, 2*t), t) == Mod(-x, t)
    assert Mod(Mod(x, t), 2*t) == Mod(x, t)
    assert Mod(-Mod(x, t), -2*t) == -Mod(x, t)
    for i in [-4, -2, 2, 4]:
        for j in [-4, -2, 2, 4]:
            for k in range(4):
                assert Mod(Mod(x, i), j).subs({x: k}) == (k % i) % j
                assert Mod(-Mod(x, i), j).subs({x: k}) == -(k % i) % j

    # known difference
    assert Mod(5*sqrt(2), sqrt(5)) == 5*sqrt(2) - 3*sqrt(5)
    p = symbols('p', positive=True)
    assert Mod(2, p + 3) == 2
    assert Mod(-2, p + 3) == p + 1
    assert Mod(2, -p - 3) == -p - 1
    assert Mod(-2, -p - 3) == -2
    assert Mod(p + 5, p + 3) == 2
    assert Mod(-p - 5, p + 3) == p + 1
    assert Mod(p + 5, -p - 3) == -p - 1
    assert Mod(-p - 5, -p - 3) == -2
    assert Mod(p + 1, p - 1).func is Mod

    # issue 27749
    n = symbols('n', integer=True, positive=True)
    assert unchanged(Mod, 1, n)
    n = symbols('n', prime=True)
    assert Mod(1, n) == 1

    # handling sums
    assert (x + 3) % 1 == Mod(x, 1)
    assert (x + 3.0) % 1 == Mod(1.*x, 1)
    assert (x - S(33)/10) % 1 == Mod(x + S(7)/10, 1)

    a = Mod(.6*x + y, .3*y)
    b = Mod(0.1*y + 0.6*x, 0.3*y)
    # Test that a, b are equal, with 1e-14 accuracy in coefficients
    eps = 1e-14
    assert abs((a.args[0] - b.args[0]).subs({x: 1, y: 1})) < eps
    assert abs((a.args[1] - b.args[1]).subs({x: 1, y: 1})) < eps

    assert (x + 1) % x == 1 % x
    assert (x + y) % x == y % x
    assert (x + y + 2) % x == (y + 2) % x
    assert (a + 3*x + 1) % (2*x) == Mod(a + x + 1, 2*x)
    assert (12*x + 18*y) % (3*x) == 3*Mod(6*y, x)

    # gcd extraction
    assert (-3*x) % (-2*y) == -Mod(3*x, 2*y)
    assert (.6*pi) % (.3*x*pi) == 0.3*pi*Mod(2, x)
    assert (.6*pi) % (.31*x*pi) == pi*Mod(0.6, 0.31*x)
    assert (6*pi) % (.3*x*pi) == 0.3*pi*Mod(20, x)
    assert (6*pi) % (.31*x*pi) == pi*Mod(6, 0.31*x)
    assert (6*pi) % (.42*x*pi) == pi*Mod(6, 0.42*x)
    assert (12*x) % (2*y) == 2*Mod(6*x, y)
    assert (12*x) % (3*5*y) == 3*Mod(4*x, 5*y)
    assert (12*x) % (15*x*y) == 3*x*Mod(4, 5*y)
    assert (-2*pi) % (3*pi) == pi
    assert (2*x + 2) % (x + 1) == 0
    assert (x*(x + 1)) % (x + 1) == (x + 1)*Mod(x, 1)
    assert Mod(5.0*x, 0.1*y) == 0.1*Mod(50*x, y)
    i = Symbol('i', integer=True)
    assert (3*i*x) % (2*i*y) == i*Mod(3*x, 2*y)
    assert Mod(4*i, 4) == 0

    # issue 8677
    n = Symbol('n', integer=True, positive=True)
    assert factorial(n) % n == 0
    assert factorial(n + 2) % n == 0
    assert (factorial(n + 4) % (n + 5)).func is Mod

    # Wilson's theorem
    assert factorial(18042, evaluate=False) % 18043 == 18042
    p = Symbol('n', prime=True)
    assert factorial(p - 1) % p == p - 1
    assert factorial(p - 1) % -p == -1
    assert (factorial(3, evaluate=False) % 4).doit() == 2
    n = Symbol('n', composite=True, odd=True)
    assert factorial(n - 1) % n == 0

    # symbolic with known parity
    n = Symbol('n', even=True)
    assert Mod(n, 2) == 0
    n = Symbol('n', odd=True)
    assert Mod(n, 2) == 1

    # issue 10963
    assert (x**6000%400).args[1] == 400

    #issue 13543
    assert Mod(Mod(x + 1, 2) + 1, 2) == Mod(x, 2)

    x1 = Symbol('x1', integer=True)
    assert Mod(Mod(x1 + 2, 4)*(x1 + 4), 4) == Mod(x1*(x1 + 2), 4)
    assert Mod(Mod(x1 + 2, 4)*4, 4) == 0

    # issue 15493
    i, j = symbols('i j', integer=True, positive=True)
    assert Mod(3*i, 2) == Mod(i, 2)
    assert Mod(8*i/j, 4) == 4*Mod(2*i/j, 1)
    assert Mod(8*i, 4) == 0

    # rewrite
    assert Mod(x, y).rewrite(floor) == x - y*floor(x/y)
    assert ((x - Mod(x, y))/y).rewrite(floor) == floor(x/y)

    # issue 21373
    from sympy.functions.elementary.hyperbolic import sinh
    from sympy.functions.elementary.piecewise import Piecewise

    x_r, y_r = symbols('x_r y_r', real=True)
    assert (Piecewise((x_r, y_r > x_r), (y_r, True)) / z) % 1
    expr = exp(sinh(Piecewise((x_r, y_r > x_r), (y_r, True)) / z))
    expr.subs({1: 1.0})
    sinh(Piecewise((x_r, y_r > x_r), (y_r, True)) * z ** -1.0).is_zero

    # issue 24215
    from sympy.abc import phi
    assert Mod(4.0*Mod(phi, 1) , 2) == 2.0*(Mod(2*(Mod(phi, 1)), 1))

    xi = symbols('x', integer=True)
    assert unchanged(Mod, xi, 2)
    assert Mod(3*xi, 2) == Mod(xi, 2)
    assert unchanged(Mod, 3*x, 2)


def test_Mod_Pow():
    # modular exponentiation
    assert isinstance(Mod(Pow(2, 2, evaluate=False), 3), Integer)

    assert Mod(Pow(4, 13, evaluate=False), 497) == Mod(Pow(4, 13), 497)
    assert Mod(Pow(2, 10000000000, evaluate=False), 3) == 1
    assert Mod(Pow(32131231232, 9**10**6, evaluate=False),10**12) == \
        pow(32131231232,9**10**6,10**12)
    assert Mod(Pow(33284959323, 123**999, evaluate=False),11**13) == \
        pow(33284959323,123**999,11**13)
    assert Mod(Pow(78789849597, 333**555, evaluate=False),12**9) == \
        pow(78789849597,333**555,12**9)

    # modular nested exponentiation
    expr = Pow(2, 2, evaluate=False)
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 16
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 6487
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 32191
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 18016
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 5137

    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 16
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 256
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 6487
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 38281
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 15928

    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 256
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 9229
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 25708
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 26608
    expr = Pow(expr, expr, evaluate=False)
    # XXX This used to fail in a nondeterministic way because of overflow
    # error.
    assert Mod(expr, 3**10) == 1966


def test_Mod_is_integer():
    p = Symbol('p', integer=True)
    q1 = Symbol('q1', integer=True)
    q2 = Symbol('q2', integer=True, nonzero=True)
    assert Mod(x, y).is_integer is None
    assert Mod(p, q1).is_integer is None
    assert Mod(x, q2).is_integer is None
    assert Mod(p, q2).is_integer


def test_Mod_is_nonposneg():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True, positive=True)
    assert (n%3).is_nonnegative
    assert Mod(n, -3).is_nonpositive
    assert Mod(n, k).is_nonnegative
    assert Mod(n, -k).is_nonpositive
    assert Mod(k, n).is_nonnegative is None


def test_issue_6001():
    A = Symbol("A", commutative=False)
    eq = A + A**2
    # it doesn't matter whether it's True or False; they should
    # just all be the same
    assert (
        eq.is_commutative ==
        (eq + 1).is_commutative ==
        (A + 1).is_commutative)

    B = Symbol("B", commutative=False)
    # Although commutative terms could cancel we return True
    # meaning "there are non-commutative symbols; aftersubstitution
    # that definition can change, e.g. (A*B).subs(B,A**-1) -> 1
    assert (sqrt(2)*A).is_commutative is False
    assert (sqrt(2)*A*B).is_commutative is False


def test_polar():
    from sympy.functions.elementary.complexes import polar_lift
    p = Symbol('p', polar=True)
    x = Symbol('x')
    assert p.is_polar
    assert x.is_polar is None
    assert S.One.is_polar is None
    assert (p**x).is_polar is True
    assert (x**p).is_polar is None
    assert ((2*p)**x).is_polar is True
    assert (2*p).is_polar is True
    assert (-2*p).is_polar is not True
    assert (polar_lift(-2)*p).is_polar is True

    q = Symbol('q', polar=True)
    assert (p*q)**2 == p**2 * q**2
    assert (2*q)**2 == 4 * q**2
    assert ((p*q)**x).expand() == p**x * q**x


def test_issue_6040():
    a, b = Pow(1, 2, evaluate=False), S.One
    assert a != b
    assert b != a
    assert not (a == b)
    assert not (b == a)


def test_issue_6082():
    # Comparison is symmetric
    assert Basic.compare(Max(x, 1), Max(x, 2)) == \
      - Basic.compare(Max(x, 2), Max(x, 1))
    # Equal expressions compare equal
    assert Basic.compare(Max(x, 1), Max(x, 1)) == 0
    # Basic subtypes (such as Max) compare different than standard types
    assert Basic.compare(Max(1, x), frozenset((1, x))) != 0


def test_issue_6077():
    assert x**2.0/x == x**1.0
    assert x/x**2.0 == x**-1.0
    assert x*x**2.0 == x**3.0
    assert x**1.5*x**2.5 == x**4.0

    assert 2**(2.0*x)/2**x == 2**(1.0*x)
    assert 2**x/2**(2.0*x) == 2**(-1.0*x)
    assert 2**x*2**(2.0*x) == 2**(3.0*x)
    assert 2**(1.5*x)*2**(2.5*x) == 2**(4.0*x)


def test_mul_flatten_oo():
    p = symbols('p', positive=True)
    n, m = symbols('n,m', negative=True)
    x_im = symbols('x_im', imaginary=True)
    assert n*oo is -oo
    assert n*m*oo is oo
    assert p*oo is oo
    assert x_im*oo != I*oo  # i could be +/- 3*I -> +/-oo


def test_add_flatten():
    # see https://github.com/sympy/sympy/issues/2633#issuecomment-29545524
    a = oo + I*oo
    b = oo - I*oo
    assert a + b is nan
    assert a - b is nan
    # FIXME: This evaluates as:
    #   >>> 1/a
    #   0*(oo + oo*I)
    # which should not simplify to 0. Should be fixed in Pow.eval
    #assert (1/a).simplify() == (1/b).simplify() == 0

    a = Pow(2, 3, evaluate=False)
    assert a + a == 16


def test_issue_5160_6087_6089_6090():
    # issue 6087
    assert ((-2*x*y**y)**3.2).n(2) == (2**3.2*(-x*y**y)**3.2).n(2)
    # issue 6089
    A, B, C = symbols('A,B,C', commutative=False)
    assert (2.*B*C)**3 == 8.0*(B*C)**3
    assert (-2.*B*C)**3 == -8.0*(B*C)**3
    assert (-2*B*C)**2 == 4*(B*C)**2
    # issue 5160
    assert sqrt(-1.0*x) == 1.0*sqrt(-x)
    assert sqrt(1.0*x) == 1.0*sqrt(x)
    # issue 6090
    assert (-2*x*y*A*B)**2 == 4*x**2*y**2*(A*B)**2


def test_float_int_round():
    assert int(float(sqrt(10))) == int(sqrt(10))
    assert int(pi**1000) % 10 == 2
    assert int(Float('1.123456789012345678901234567890e20', '')) == \
        int(112345678901234567890)
    assert int(Float('1.123456789012345678901234567890e25', '')) == \
        int(11234567890123456789012345)
    # decimal forces float so it's not an exact integer ending in 000000
    assert int(Float('1.123456789012345678901234567890e35', '')) == \
        112345678901234567890123456789000192
    assert int(Float('123456789012345678901234567890e5', '')) == \
        12345678901234567890123456789000000
    assert Integer(Float('1.123456789012345678901234567890e20', '')) == \
        112345678901234567890
    assert Integer(Float('1.123456789012345678901234567890e25', '')) == \
        11234567890123456789012345
    # decimal forces float so it's not an exact integer ending in 000000
    assert Integer(Float('1.123456789012345678901234567890e35', '')) == \
        112345678901234567890123456789000192
    assert Integer(Float('123456789012345678901234567890e5', '')) == \
        12345678901234567890123456789000000
    assert same_and_same_prec(Float('123000e-2',''), Float('1230.00', ''))
    assert same_and_same_prec(Float('123000e2',''), Float('12300000', ''))

    assert int(1 + Rational('.9999999999999999999999999')) == 1
    assert int(pi/1e20) == 0
    assert int(1 + pi/1e20) == 1
    assert int(Add(1.2, -2, evaluate=False)) == int(1.2 - 2)
    assert int(Add(1.2, +2, evaluate=False)) == int(1.2 + 2)
    assert int(Add(1 + Float('.99999999999999999', ''), evaluate=False)) == 1
    raises(TypeError, lambda: float(x))
    raises(TypeError, lambda: float(sqrt(-1)))

    assert int(12345678901234567890 + cos(1)**2 + sin(1)**2) == \
        12345678901234567891


def test_issue_6611a():
    assert Mul.flatten([3**Rational(1, 3),
        Pow(-Rational(1, 9), Rational(2, 3), evaluate=False)]) == \
        ([Rational(1, 3), (-1)**Rational(2, 3)], [], None)


def test_denest_add_mul():
    # when working with evaluated expressions make sure they denest
    eq = x + 1
    eq = Add(eq, 2, evaluate=False)
    eq = Add(eq, 2, evaluate=False)
    assert Add(*eq.args) == x + 5
    eq = x*2
    eq = Mul(eq, 2, evaluate=False)
    eq = Mul(eq, 2, evaluate=False)
    assert Mul(*eq.args) == 8*x
    # but don't let them denest unnecessarily
    eq = Mul(-2, x - 2, evaluate=False)
    assert 2*eq == Mul(-4, x - 2, evaluate=False)
    assert -eq == Mul(2, x - 2, evaluate=False)


def test_mul_coeff():
    # It is important that all Numbers be removed from the seq;
    # This can be tricky when powers combine to produce those numbers
    p = exp(I*pi/3)
    assert p**2*x*p*y*p*x*p**2 == x**2*y


def test_mul_zero_detection():
    nz = Dummy(real=True, zero=False)
    r = Dummy(extended_real=True)
    c = Dummy(real=False, complex=True)
    c2 = Dummy(real=False, complex=True)
    i = Dummy(imaginary=True)
    e = nz*r*c
    assert e.is_imaginary is None
    assert e.is_extended_real is None
    e = nz*c
    assert e.is_imaginary is None
    assert e.is_extended_real is False
    e = nz*i*c
    assert e.is_imaginary is False
    assert e.is_extended_real is None
    # check for more than one complex; it is important to use
    # uniquely named Symbols to ensure that two factors appear
    # e.g. if the symbols have the same name they just become
    # a single factor, a power.
    e = nz*i*c*c2
    assert e.is_imaginary is None
    assert e.is_extended_real is None

    # _eval_is_extended_real and _eval_is_zero both employ trapping of the
    # zero value so args should be tested in both directions and
    # TO AVOID GETTING THE CACHED RESULT, Dummy MUST BE USED

    # real is unknown
    def test(z, b, e):
        if z.is_zero and b.is_finite:
            assert e.is_extended_real and e.is_zero
        else:
            assert e.is_extended_real is None
            if b.is_finite:
                if z.is_zero:
                    assert e.is_zero
                else:
                    assert e.is_zero is None
            elif b.is_finite is False:
                if z.is_zero is None:
                    assert e.is_zero is None
                else:
                    assert e.is_zero is False


    for iz, ib in product(*[[True, False, None]]*2):
        z = Dummy('z', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('nz', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)

    # real is True
    def test(z, b, e):
        if z.is_zero and not b.is_finite:
            assert e.is_extended_real is None
        else:
            assert e.is_extended_real is True

    for iz, ib in product(*[[True, False, None]]*2):
        z = Dummy('z', nonzero=iz, extended_real=True)
        b = Dummy('b', finite=ib, extended_real=True)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('z', nonzero=iz, extended_real=True)
        b = Dummy('b', finite=ib, extended_real=True)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)


def test_Mul_with_zero_infinite():
    zer = Dummy(zero=True)
    inf = Dummy(finite=False)

    e = Mul(zer, inf, evaluate=False)
    assert e.is_extended_positive is None
    assert e.is_hermitian is None

    e = Mul(inf, zer, evaluate=False)
    assert e.is_extended_positive is None
    assert e.is_hermitian is None


def test_Mul_does_not_cancel_infinities():
    a, b = symbols('a b')
    assert ((zoo + 3*a)/(3*a + zoo)) is nan
    assert ((b - oo)/(b - oo)) is nan
    # issue 13904
    expr = (1/(a+b) + 1/(a-b))/(1/(a+b) - 1/(a-b))
    assert expr.subs(b, a) is nan


def test_Mul_does_not_distribute_infinity():
    a, b = symbols('a b')
    assert ((1 + I)*oo).is_Mul
    assert ((a + b)*(-oo)).is_Mul
    assert ((a + 1)*zoo).is_Mul
    assert ((1 + I)*oo).is_finite is False
    z = (1 + I)*oo
    assert ((1 - I)*z).expand() is oo


def test_Mul_does_not_let_0_trump_inf():
    assert Mul(*[0, a + zoo]) is S.NaN
    assert Mul(*[0, a + oo]) is S.NaN
    assert Mul(*[0, a + Integral(1/x**2, (x, 1, oo))]) is S.Zero
    # Integral is treated like an unknown like 0*x -> 0
    assert Mul(*[0, a + Integral(x, (x, 1, oo))]) is S.Zero


def test_issue_8247_8354():
    from sympy.functions.elementary.trigonometric import tan
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert z.is_positive is False  # it's 0
    z = S('''-2**(1/3)*(3*sqrt(93) + 29)**2 - 4*(3*sqrt(93) + 29)**(4/3) +
        12*sqrt(93)*(3*sqrt(93) + 29)**(1/3) + 116*(3*sqrt(93) + 29)**(1/3) +
        174*2**(1/3)*sqrt(93) + 1678*2**(1/3)''')
    assert z.is_positive is False  # it's 0
    z = 2*(-3*tan(19*pi/90) + sqrt(3))*cos(11*pi/90)*cos(19*pi/90) - \
        sqrt(3)*(-3 + 4*cos(19*pi/90)**2)
    assert z.is_positive is not True  # it's zero and it shouldn't hang
    z = S('''9*(3*sqrt(93) + 29)**(2/3)*((3*sqrt(93) +
        29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) - 2) - 2*2**(1/3))**3 +
        72*(3*sqrt(93) + 29)**(2/3)*(81*sqrt(93) + 783) + (162*sqrt(93) +
        1566)*((3*sqrt(93) + 29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) -
        2) - 2*2**(1/3))**2''')
    assert z.is_positive is False  # it's 0 (and a single _mexpand isn't enough)


def test_Add_is_zero():
    x, y = symbols('x y', zero=True)
    assert (x + y).is_zero

    # Issue 15873
    e = -2*I + (1 + I)**2
    assert e.is_zero is None


def test_issue_14392():
    assert (sin(zoo)**2).as_real_imag() == (nan, nan)


def test_divmod():
    assert divmod(x, y) == (x//y, x % y)
    assert divmod(x, 3) == (x//3, x % 3)
    assert divmod(3, x) == (3//x, 3 % x)


def test__neg__():
    assert -(x*y) == -x*y
    assert -(-x*y) == x*y
    assert -(1.*x) == -1.*x
    assert -(-1.*x) == 1.*x
    assert -(2.*x) == -2.*x
    assert -(-2.*x) == 2.*x
    with distribute(False):
        eq = -(x + y)
        assert eq.is_Mul and eq.args == (-1, x + y)
    with evaluate(False):
        eq = -(x + y)
        assert eq.is_Mul and eq.args == (-1, x + y)


def test_issue_18507():
    assert Mul(zoo, zoo, 0) is nan


def test_issue_17130():
    e = Add(b, -b, I, -I, evaluate=False)
    assert e.is_zero is None # ideally this would be True


def test_issue_21034():
    e = -I*log((re(asin(5)) + I*im(asin(5)))/sqrt(re(asin(5))**2 + im(asin(5))**2))/pi
    assert e.round(2)


def test_issue_22021():
    from sympy.calculus.accumulationbounds import AccumBounds
    # these objects are special cases in Mul
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
    L = TensorIndexType("L")
    i = tensor_indices("i", L)
    A, B = tensor_heads("A B", [L])
    e = A(i) + B(i)
    assert -e == -1*e
    e = zoo + x
    assert -e == -1*e
    a = AccumBounds(1, 2)
    e = a + x
    assert -e == -1*e
    for args in permutations((zoo, a, x)):
        e = Add(*args, evaluate=False)
        assert -e == -1*e
    assert 2*Add(1, x, x, evaluate=False) == 4*x + 2


def test_issue_22244():
    assert -(zoo*x) == zoo*x


def test_issue_22453():
    from sympy.utilities.iterables import cartes
    e = Symbol('e', extended_positive=True)
    for a, b in cartes(*[[oo, -oo, 3]]*2):
        if a == b == 3:
            continue
        i = a + I*b
        assert i**(1 + e) is S.ComplexInfinity
        assert i**-e is S.Zero
        assert unchanged(Pow, i, e)
    assert 1/(oo + I*oo) is S.Zero
    r, i = [Dummy(infinite=True, extended_real=True) for _ in range(2)]
    assert 1/(r + I*i) is S.Zero
    assert 1/(3 + I*i) is S.Zero
    assert 1/(r + I*3) is S.Zero


def test_issue_22613():
    assert (0**(x - 2)).as_content_primitive() == (1, 0**(x - 2))
    assert (0**(x + 2)).as_content_primitive() == (1, 0**(x + 2))


def test_issue_25176():
    assert sqrt(-4*3**(S(3)/4)*I/3) == 2*3**(S(7)/8)*sqrt(-I)/3